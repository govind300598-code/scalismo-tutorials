//> using scala 3.3.3
//> using dep ch.unibas.cs.gravis::scalismo:0.92.1
//> using dep ch.unibas.cs.gravis::scalismo-ui:0.92.0
//> using javaOpt -Xmx8G

// ──────────────────────────────────────────────────────────────────────────
//  OBJECTIVE 1 — FFDM MODEL DEVELOPMENT
//
//  Standalone script: run with
//      scala-cli run BuildFFDMModel.scala
//
//  What it does:
//   1. Lists all left-side scapula STL files
//   2. Selects the median-volume specimen as the registration reference
//   3. Decimates it to ~MODEL_RESOLUTION vertices
//   4. Estimates sigma from the bounding box (following Dennis Madsen's
//      advice: σ ∈ {L/2, L/5, L/10} for coarse/mid/fine Gaussians)
//   5. Runs a parameterisation study over 7 kernel configurations and
//      prints rank + cumulative variance explained @10/50/100 components
//   6. Auto-selects the best kernel and saves ffdm.h5 + ffdm_reference.stl
//
//  Kernel API note (scalismo 0.92.x)
//   • GaussianKernel[_3D](σ) * (s*s)     — variance = s²  (s = std-dev in mm)
//   • DiagonalKernel3D(scalarKernel, outputDim = 3)   ← correct class name
//   • NearestNeighborInterpolator3D()                  ← confirmed by D.Madsen
//   • LowRankGaussianProcess.approximateGPCholesky(mesh, gp, tol, interp)
//     pass the MESH, not mesh.pointSet                ← confirmed by D.Madsen
// ──────────────────────────────────────────────────────────────────────────

import scalismo.common.Field
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.geometry.*
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.kernels.{DiagonalKernel3D, GaussianKernel, MatrixValuedPDKernel, PDKernel}
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel3D}
import scalismo.utils.Random

import java.io.File

// ============================================================
//  PATHS – override via environment variables if preferred
// ============================================================
val DATA_DIR = new File(
  sys.env.getOrElse("SCAPULA_DATA_DIR",
    s"${System.getProperty("user.home")}/Documents/database_v1.11/paired_scapulae_STLs"))

val OUT_DIR = new File(
  sys.env.getOrElse("SCAPULA_OUT_DIR",
    s"${System.getProperty("user.home")}/Documents/database_v1.11/scapula_ssm_out"))

// ── tunable knobs ────────────────────────────────────────────────────────────
val MODEL_RESOLUTION: Int    = sys.env.get("SCAPULA_MODEL_RES").map(_.toInt).getOrElse(5000)
val GP_TOLERANCE    : Double = sys.env.get("SCAPULA_GP_TOL").map(_.toDouble).getOrElse(0.01)
val GP_MAX_RANK     : Int    = sys.env.get("SCAPULA_GP_MAX_RANK").map(_.toInt).getOrElse(300)
// Set SCAPULA_FFDM_REF=<id>  to skip auto-selection and use a known good reference
val FORCE_REF_ID    : String = sys.env.getOrElse("SCAPULA_FFDM_REF", "")

// ============================================================

scalismo.initialize()
implicit val rng: Random = Random(42L)
OUT_DIR.mkdirs()

println("=" * 70)
println("OBJECTIVE 1 — FFDM MODEL DEVELOPMENT")
println("=" * 70)
println(s"  STL data dir  : ${DATA_DIR.getAbsolutePath}")
println(s"  Output dir    : ${OUT_DIR.getAbsolutePath}")
println(s"  Model res     : $MODEL_RESOLUTION vertices")
println(s"  GP tolerance  : $GP_TOLERANCE  (pivoted-Cholesky relative)")
println(s"  GP max rank   : $GP_MAX_RANK")
println()

// ── helpers ──────────────────────────────────────────────────────────────────

def loadMesh(f: File): TriangleMesh[_3D] =
  MeshIO.readMesh(f).getOrElse(throw new RuntimeException(s"Cannot read ${f.getPath}"))

/** Signed volume via the divergence theorem (positive = outward-oriented). */
def signedVolume(m: TriangleMesh[_3D]): Double = {
  var acc = 0.0
  m.triangulation.triangles.foreach { t =>
    val a = m.pointSet.point(t.ptId1).toVector
    val b = m.pointSet.point(t.ptId2).toVector
    val c = m.pointSet.point(t.ptId3).toVector
    acc += a.dot(b.crossproduct(c)) / 6.0
  }
  acc
}

/** Longest axis of the mesh bounding box (mm). */
def longestDimension(m: TriangleMesh[_3D]): Double = {
  val bb = m.boundingBox
  Seq(bb.extent.x, bb.extent.y, bb.extent.z).max
}

// Zero-mean deformation field
val zeroMean: Field[_3D, EuclideanVector[_3D]] =
  Field(EuclideanSpace3D, (_: Point[_3D]) => EuclideanVector3D(0, 0, 0))

/**
 * Build a multi-scale Gaussian GP.
 *
 * components = Seq( (sigma_mm, scale_mm), ... )
 *   sigma  — Gaussian kernel bandwidth: how far one basis deformation extends
 *   scale  — std-dev of deformations in mm; stored as variance = scale² in kernel weight
 *
 * DiagonalKernel3D wraps a scalar PDKernel into an isotropic 3-D vector kernel.
 * This is the correct API for scalismo 0.92.x (confirmed in Scalismo forum by D.Madsen).
 */
def buildGP(components: Seq[(Double, Double)]): GaussianProcess[_3D, EuclideanVector[_3D]] = {
  val scalar: PDKernel[_3D] =
    components
      .map { case (sigma, scale) => GaussianKernel[_3D](sigma) * (scale * scale) }
      .reduce[PDKernel[_3D]](_ + _)
  // DiagonalKernel3D  ← correct class name in scalismo 0.92.x
  val vector: MatrixValuedPDKernel[_3D] = DiagonalKernel3D(scalar, outputDim = 3)
  GaussianProcess[_3D, EuclideanVector[_3D]](zeroMean, vector)
}

/**
 * Pivoted-Cholesky low-rank approximation.
 *
 * Pass the MESH (not mesh.pointSet) as the first argument — this is required
 * for NearestNeighborInterpolator3D to work correctly (confirmed by D.Madsen).
 */
def approximateGP(refMesh: TriangleMesh[_3D],
                  gp: GaussianProcess[_3D, EuclideanVector[_3D]]
): LowRankGaussianProcess[_3D, EuclideanVector[_3D]] =
  LowRankGaussianProcess.approximateGPCholesky(
    refMesh,                                     // ← mesh, not refMesh.pointSet
    gp,
    relativeTolerance = GP_TOLERANCE,
    interpolator      = NearestNeighborInterpolator3D[EuclideanVector[_3D]]()
  )

/** Cumulative variance fraction explained by components 1..rank. */
def cumulativeVariance(lrGP: LowRankGaussianProcess[_3D, EuclideanVector[_3D]]): IndexedSeq[Double] = {
  val evals = lrGP.eigenPairs.map(_._1)
  val total = evals.sum
  if (total <= 0.0) return IndexedSeq.fill(evals.length)(0.0)
  evals.scanLeft(0.0)(_ + _).tail.map(_ / total)
}

// ── STEP 1  Load STL files ─────────────────────────────────────────────────
println("─" * 70)
println("[STEP 1] Loading STL files")

require(DATA_DIR.exists(), s"STL directory not found: ${DATA_DIR.getAbsolutePath}")

val allFiles = Option(DATA_DIR.listFiles())
  .getOrElse(Array.empty[File])
  .filter(_.getName.toLowerCase.endsWith(".stl"))
  .sortBy(_.getName)
  .toIndexedSeq

require(allFiles.nonEmpty, s"No STL files found in ${DATA_DIR.getAbsolutePath}")

val leftFiles = allFiles.filter(_.getName.stripSuffix(".stl").endsWith("_L"))
println(s"  Total STL files : ${allFiles.size}")
println(s"  Left-side (_L)  : ${leftFiles.size}  (used for reference selection)")

// ── STEP 2  Reference selection ────────────────────────────────────────────
println()
println("─" * 70)
println("[STEP 2] Reference selection")

val refId: String =
  if (FORCE_REF_ID.nonEmpty) {
    println(s"  Forced reference : $FORCE_REF_ID  (SCAPULA_FFDM_REF)")
    FORCE_REF_ID
  } else {
    println("  Strategy: median signed-volume among left-side specimens")
    println("  (The median minimises expected shape distance to every other bone.)")
    println()

    val withVols = leftFiles.map { f =>
      val id  = f.getName.stripSuffix(".stl")
      val vol = math.abs(signedVolume(loadMesh(f)))
      (id, vol)
    }.sortBy(_._2)

    val medIdx = withVols.length / 2

    println(f"  ${"#"}%3s  ${"Specimen ID"}%-38s  Volume (cm³)")
    println("  " + "─" * 58)
    withVols.zipWithIndex.foreach { case ((id, vol), i) =>
      val tag = if (i == medIdx) "  <── SELECTED (median)" else ""
      println(f"  ${i + 1}%3d  $id%-38s  ${vol / 1000.0}%8.2f$tag")
    }
    println()

    withVols(medIdx)._1
  }

// ── STEP 3  Load + decimate reference ─────────────────────────────────────
println("─" * 70)
println("[STEP 3] Load and decimate the reference")

val refRawFile = new File(DATA_DIR, s"$refId.stl")
require(refRawFile.exists(), s"Reference STL not found: ${refRawFile.getAbsolutePath}")

val refRaw = loadMesh(refRawFile)
println(f"  Reference      : $refId")
println(f"  Original size  : ${refRaw.pointSet.numberOfPoints} vertices  /  ${refRaw.triangulation.triangles.size} triangles")

val refMesh: TriangleMesh[_3D] =
  if (refRaw.pointSet.numberOfPoints > MODEL_RESOLUTION) {
    println(s"  Decimating to ~$MODEL_RESOLUTION vertices ...")
    val dec = refRaw.operations.decimate(MODEL_RESOLUTION)
    println(f"  After decimate : ${dec.pointSet.numberOfPoints} vertices  /  ${dec.triangulation.triangles.size} triangles")
    dec
  } else {
    println(s"  Mesh is already ≤ $MODEL_RESOLUTION vertices – no decimation.")
    refRaw
  }

val refOutFile = new File(OUT_DIR, "ffdm_reference.stl")
MeshIO.writeMesh(refMesh, refOutFile).get
println(s"  Saved          : ${refOutFile.getAbsolutePath}")

// ── Estimate sigma from bounding box (D.Madsen's formula) ─────────────────
val L         = longestDimension(refMesh)
val sigCoarse = L / 2.0
val sigMid    = L / 5.0
val sigFine   = L / 10.0

println()
println(f"  Bounding-box longest side L = $L%.1f mm")
println(f"  => recommended sigmas  (D.Madsen: L/2, L/5, L/10):")
println(f"     coarse σ = $sigCoarse%.1f mm   mid σ = $sigMid%.1f mm   fine σ = $sigFine%.1f mm")

// ── STEP 4  Parameterisation study ────────────────────────────────────────
println()
println("─" * 70)
println("[STEP 4] Parameterisation study")
println(s"  Pivoted-Cholesky tolerance = $GP_TOLERANCE,  rank cap = $GP_MAX_RANK")
println(s"  DiagonalKernel3D  +  NearestNeighborInterpolator3D  (both confirmed by D.Madsen)")
println()
println(f"  ${"Kernel configuration"}%-46s  ${"Rank"}%5s  ${"Var@10"}%7s  ${"Var@50"}%7s  ${"Var@100"}%8s")
println("  " + "─" * 80)

// ┌──────────────────────────────────────────────────────────────────────────┐
// │  Test grid                                                                │
// │  σ (sigma) = spatial reach  (how far a basis deformation extends, mm)    │
// │  s (scale) = deformation std-dev (mm);  kernel weight = s²               │
// │                                                                           │
// │  Rows 1-4 : single-scale (diagnostic — real models use multi-scale)       │
// │  Rows 5-6 : two-scale mixture                                             │
// │  Row  7   : three-scale (D.Madsen's L/2, L/5, L/10 formula)              │
// └──────────────────────────────────────────────────────────────────────────┘
case class KernelCfg(label: String, components: Seq[(Double, Double)])

val testGrid: Seq[KernelCfg] = Seq(
  // single-scale (increasingly fine)
  KernelCfg(f"σ=${sigCoarse}%5.0f mm  s=15 mm  [coarse single]",  Seq((sigCoarse, 15.0))),
  KernelCfg(f"σ=${sigMid}%5.0f mm  s=10 mm  [mid single]",        Seq((sigMid,    10.0))),
  KernelCfg(f"σ=${sigFine}%5.0f mm  s= 5 mm  [fine single]",      Seq((sigFine,    5.0))),
  KernelCfg( "σ= 10 mm  s= 5 mm  [local single]",                 Seq((10.0,       5.0))),
  // two-scale
  KernelCfg(f"σ=[${sigCoarse}%4.0f+${sigFine}%3.0f] s=[15+5]  [2-scale]",
    Seq((sigCoarse, 15.0), (sigFine, 5.0))),
  // D.Madsen's three-scale formula
  KernelCfg(f"σ=[${sigCoarse}%4.0f+${sigMid}%3.0f+${sigFine}%3.0f] s=[15+10+5]  [3-scale ★]",
    Seq((sigCoarse, 15.0), (sigMid, 10.0), (sigFine, 5.0))),
  // three-scale with extra local detail
  KernelCfg(f"σ=[${sigCoarse}%4.0f+${sigMid}%3.0f+${sigFine}%3.0f+10] s=[15+10+5+5]  [4-scale]",
    Seq((sigCoarse, 15.0), (sigMid, 10.0), (sigFine, 5.0), (10.0, 5.0)))
)

case class StudyRow(cfg: KernelCfg, rank: Int, cumVar: IndexedSeq[Double])

val studyRows: Seq[StudyRow] = testGrid.map { cfg =>
  val lrGP   = approximateGP(refMesh, buildGP(cfg.components))
  val rank   = lrGP.rank
  val cumVar = cumulativeVariance(lrGP)

  val v10  = cumVar.lift(9).getOrElse(cumVar.lastOption.getOrElse(0.0))
  val v50  = cumVar.lift(49).getOrElse(cumVar.lastOption.getOrElse(0.0))
  val v100 = cumVar.lift(99).getOrElse(cumVar.lastOption.getOrElse(0.0))

  val passRank = rank >= 40 && rank <= GP_MAX_RANK
  val passVar  = v50 >= 0.90
  val tag      = if (passRank && passVar) " ✓" else "  "

  println(f"$tag ${""}  ${cfg.label}%-46s  $rank%5d  ${v10 * 100}%6.1f%%  ${v50 * 100}%6.1f%%  ${v100 * 100}%7.1f%%")
  StudyRow(cfg, rank, cumVar)
}

println()
println(s"  ✓ = rank in [40, $GP_MAX_RANK]  AND  cumul. variance at 50 components ≥ 90 %")

// ── STEP 5  Select best kernel ────────────────────────────────────────────
println()
println("─" * 70)
println("[STEP 5] Kernel selection")

val (chosenLabel, chosenComponents): (String, Seq[(Double, Double)]) =
  sys.env.get("SCAPULA_FFDM_KERNEL") match {
    case Some(spec) =>
      // Parse "sigma1:scale1,sigma2:scale2,..." from env override
      val parts = spec.split(",").toSeq.map { p =>
        val Array(s, sc) = p.trim.split(":")
        (s.toDouble, sc.toDouble)
      }
      println(s"  Using SCAPULA_FFDM_KERNEL override: \"$spec\"")
      (s"custom ($spec)", parts)

    case None =>
      // Auto-select: prefer the last (richest) kernel that satisfies both criteria.
      // D.Madsen: "hundreds or even thousands of basis functions are needed for
      // fine anatomical details." So don't be too restrictive with the rank cap.
      val passing = studyRows.filter { r =>
        r.rank >= 40 &&
        r.rank <= GP_MAX_RANK &&
        r.cumVar.lift(49).getOrElse(0.0) >= 0.90
      }
      val chosen = passing.lastOption.getOrElse(studyRows.last)
      if (passing.isEmpty)
        println("  WARNING: no kernel fully passed the criterion – using the last as fallback.")
      println(s"  Auto-selected  : ${chosen.cfg.label}")
      (chosen.cfg.label, chosen.cfg.components)
  }

// ── STEP 6  Build and save the FFDM ──────────────────────────────────────
println()
println("─" * 70)
println("[STEP 6] Building and saving the FFDM")
println(s"  Kernel : $chosenLabel")

val finalLRGP = approximateGP(refMesh, buildGP(chosenComponents))
val ffdm      = PointDistributionModel3D(refMesh, finalLRGP)

println(s"  Model rank : ${ffdm.rank}")
val cv = cumulativeVariance(finalLRGP)
Seq(10, 20, 50, 100).foreach { n =>
  val v = cv.lift(n - 1).getOrElse(cv.lastOption.getOrElse(0.0))
  println(f"  Cumul. var @ $n%-3d components : ${v * 100}%5.1f %%")
}

val modelFile = new File(OUT_DIR, "ffdm.h5")
StatisticalModelIO.writeStatisticalMeshModel(ffdm, modelFile).get
println(s"  FFDM saved : ${modelFile.getAbsolutePath}")

// ── DONE ──────────────────────────────────────────────────────────────────
println()
println("=" * 70)
println("OBJECTIVE 1 COMPLETE")
println(s"  Output directory : ${OUT_DIR.getAbsolutePath}")
println(s"  ffdm_reference.stl  — decimated reference (~$MODEL_RESOLUTION vertices)")
println(s"  ffdm.h5             — FFDM prior (${ffdm.rank} GP components)")
println()
println("  Interpretation guide:")
println("    Rank ≈ 100-300  →  good balance of flexibility vs. ICP cost")
println("    Var@50  ≥ 90 %  →  50 components capture most prior variance")
println("    If rank is too small  → increase sigma or decrease GP_TOLERANCE")
println("    If rank is too large  → decrease GP_MAX_RANK or GP_TOLERANCE,")
println("                            or use a coarser kernel")
println()
println("  Next: Stage 3 — non-rigid GP-ICP registration using ffdm.h5")
println("        as the deformation prior over all specimens.")
println("=" * 70)
