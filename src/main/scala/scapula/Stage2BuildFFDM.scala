package scapula

import scalismo.common.Field
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.geometry.*
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.kernels.{DiagonalKernel, GaussianKernel, MatrixValuedPDKernel, PDKernel}
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel3D}
import scalismo.utils.Random

import java.io.File

/**
 * OBJECTIVE 1 — FFDM MODEL DEVELOPMENT
 *
 * Builds the Free-Form Deformation Model (FFDM): a Gaussian Process prior over the reference
 * scapula surface.  The FFDM encodes our belief about plausible deformations BEFORE any training
 * data are seen, parametrised by the kernel's sigma (spatial reach, mm) and scale (deformation
 * amplitude, mm).
 *
 * Pipeline
 * --------
 *   Step 1 – Reference selection
 *     Load all left-side specimens, compute each mesh's volume and pick the median-volume
 *     specimen.  The median shape minimises the expected distortion to any other shape and
 *     therefore makes a good registration template.  Override via SCAPULA_FFDM_REF.
 *
 *   Step 2 – Decimate the reference
 *     Reduce vertex count to Config.modelResolution (default 5000).  The GP lives on the
 *     decimated mesh; every registered correspondent must share this topology.
 *
 *   Step 3 – Parameterisation study
 *     Systematically sweep single- and multi-scale Gaussian kernels and report the rank of each
 *     low-rank approximation together with cumulative variance explained at 10 / 50 / 100
 *     components.  This answers: "which kernel gives a rich-enough basis without blowing up
 *     memory or ICP cost?"
 *
 *   Step 4 – Build and save the chosen FFDM
 *     Build the chosen (or default) kernel's low-rank GP and write it as a statistical mesh
 *     model (.h5).  The reference mesh (.stl) is written alongside.
 *
 * Run
 * ---
 *   sbt "runMain scapula.Stage2BuildFFDM"
 *
 * Environment knobs (all optional)
 *   SCAPULA_DATA_DIR      raw STL directory
 *   SCAPULA_OUT_DIR       output root
 *   SCAPULA_MODEL_RES     target vertex count for reference decimation (default 5000)
 *   SCAPULA_FFDM_REF      explicit reference ID; empty = auto-select (default)
 *   SCAPULA_FFDM_KERNEL   kernel spec "sigma1:scale1,sigma2:scale2,..." (default "10:5,30:10,80:30")
 *   SCAPULA_GP_TOL        relative Cholesky tolerance (default 0.01)
 *   SCAPULA_GP_MAX_RANK   hard rank cap (default 250)
 */
object Stage2BuildFFDM {

  // ── zero-mean field over 3-D Euclidean space ──────────────────────────────
  private val zeroMean: Field[_3D, EuclideanVector[_3D]] =
    Field(EuclideanSpace3D, (_: Point[_3D]) => EuclideanVector3D(0, 0, 0))

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dir    = Config.dataDir
    val outDir = Config.outDir
    outDir.mkdirs()

    println("=" * 70)
    println("OBJECTIVE 1 — FFDM MODEL DEVELOPMENT")
    println("=" * 70)
    println(s"  Data dir  : ${dir.getAbsolutePath}")
    println(s"  Output dir: ${outDir.getAbsolutePath}")
    println(s"  Model res : ${Config.modelResolution} vertices")
    println(s"  GP tol    : ${Config.gpRelativeTolerance}")
    println(s"  GP max rank: ${Config.gpMaxRank}")
    println()

    // ── Step 1: Load specimens ────────────────────────────────────────────
    println("─" * 70)
    println("[Step 1] Loading specimens")
    val allSpecimens  = ScapulaData.specimens(dir)
    require(allSpecimens.nonEmpty,
      s"No STL files found in ${dir.getAbsolutePath}. Check SCAPULA_DATA_DIR.")

    // Use only left-side specimens for reference selection so we compare
    // anatomically oriented surfaces without artificial mirroring bias.
    val leftSpecimens = allSpecimens.filter(!_.isRight)
    println(s"  Total STL files : ${allSpecimens.size}")
    println(s"  Left-side only  : ${leftSpecimens.size}  (used for reference selection)")

    // ── Step 2: Select reference ──────────────────────────────────────────
    println()
    println("─" * 70)
    println("[Step 2] Reference selection")

    val refId: String =
      if (Config.ffdmReferenceId.nonEmpty) {
        println(s"  Using explicitly specified reference: ${Config.ffdmReferenceId}")
        Config.ffdmReferenceId
      } else {
        selectByVolumeMedian(leftSpecimens)
      }


    val refRawFile = new File(dir, s"$refId.stl")
    require(refRawFile.exists(), s"Reference STL not found: ${refRawFile.getAbsolutePath}")
    val refRaw = ScapulaData.loadMesh(refRawFile)
    println(s"  Reference       : $refId")
    println(f"  Original size   : ${refRaw.pointSet.numberOfPoints} vertices, ${refRaw.triangulation.triangles.size} triangles")

    // ── Step 3: Decimate to model resolution ──────────────────────────────
    val refMesh: TriangleMesh[_3D] =
      if (refRaw.pointSet.numberOfPoints > Config.modelResolution) {
        println(s"  Decimating to ~${Config.modelResolution} vertices...")
        val dec = refRaw.operations.decimate(Config.modelResolution)
        println(f"  Decimated size  : ${dec.pointSet.numberOfPoints} vertices, ${dec.triangulation.triangles.size} triangles")
        dec
      } else {
        println(s"  Mesh already at or below target resolution; no decimation needed.")
        refRaw
      }

    val refOutFile = new File(outDir, "ffdm_reference.stl")
    MeshIO.writeMesh(refMesh, refOutFile).get
    println(s"  Saved reference : ${refOutFile.getAbsolutePath}")

    // ── Step 4: Parameterisation study ────────────────────────────────────
    println()
    println("─" * 70)
    println("[Step 3] Parameterisation study")
    println("  Each kernel is approximated via pivoted Cholesky at tolerance = " +
      Config.gpRelativeTolerance)
    println()
    println(f"  ${"Kernel configuration"}%-42s  ${"Rank"}%5s  ${"Var@10"}%7s  ${"Var@50"}%7s  ${"Var@100"}%8s")
    println("  " + "-" * 76)

    // The table covers a systematic grid:
    //   • four single-scale entries with increasing sigma/scale
    //   • three multi-scale entries (sum of two or three Gaussians)
    val testCandidates: Seq[(String, Seq[(Double, Double)])] = Seq(
      // ── single-scale ──────────────────────────────────────────────────────
      ("σ= 10 mm  s=  5 mm [single]",   Seq((10.0,  5.0))),
      ("σ= 20 mm  s= 10 mm [single]",   Seq((20.0, 10.0))),
      ("σ= 50 mm  s= 20 mm [single]",   Seq((50.0, 20.0))),
      ("σ=100 mm  s= 50 mm [single]",   Seq((100.0, 50.0))),
      // ── two-scale ─────────────────────────────────────────────────────────
      ("σ=[10+50]  s=[5+15] [2-scale]",  Seq((10.0,  5.0), (50.0,  15.0))),
      ("σ=[20+80]  s=[10+30] [2-scale]", Seq((20.0, 10.0), (80.0,  30.0))),
      // ── three-scale (default kernel spec) ─────────────────────────────────
      ("σ=[10+30+80] s=[5+10+30] [3-scale]", Seq((10.0, 5.0), (30.0, 10.0), (80.0, 30.0)))
    )

    case class StudyResult(label: String, components: Seq[(Double, Double)],
                           rank: Int, cumVar: IndexedSeq[Double])

    val results = testCandidates.map { case (label, components) =>
      val gp     = buildGP(components)
      val lrGP   = approximateGP(refMesh, gp)
      val rank   = lrGP.rank
      val cumVar = cumulativeVariance(lrGP)

      val v10  = cumVar.lift(9).getOrElse(cumVar.lastOption.getOrElse(0.0))
      val v50  = cumVar.lift(49).getOrElse(cumVar.lastOption.getOrElse(0.0))
      val v100 = cumVar.lift(99).getOrElse(cumVar.lastOption.getOrElse(0.0))

      println(f"  $label%-42s  $rank%5d  ${v10 * 100}%6.1f%%  ${v50 * 100}%6.1f%%  ${v100 * 100}%7.1f%%")
      StudyResult(label, components, rank, cumVar)
    }

    // ── Step 5: Decide on the best kernel ────────────────────────────────
    // Recommendation criterion:
    //   • rank in [40, Config.gpMaxRank]  — not trivially small or too large
    //   • cumulative variance at 50 components ≥ 90 %
    //   • prefer the richest kernel that still satisfies the rank cap
    val recommended: StudyResult = {
      val passing = results.filter { r =>
        r.rank >= 40 &&
        r.rank <= Config.gpMaxRank &&
        r.cumVar.lift(49).getOrElse(0.0) >= 0.90
      }
      passing.lastOption.getOrElse(results.last)
    }

    // ── Step 6: Build and save the chosen FFDM ───────────────────────────
    println()
    println("─" * 70)
    println("[Step 4] Building and saving the FFDM")
    println(s"  Chosen kernel : ${recommended.label}")

    // Env-var override takes priority over auto-selection; detect by checking
    // whether SCAPULA_FFDM_KERNEL is present in the environment.
    val finalComponents: Seq[(Double, Double)] =
      sys.env.get("SCAPULA_FFDM_KERNEL") match {
        case Some(spec) =>
          val parsed = parseKernelSpec(spec)
          println(s"  (overridden by SCAPULA_FFDM_KERNEL=\"$spec\")")
          parsed
        case None =>
          recommended.components
      }

    val finalGP   = buildGP(finalComponents)
    val finalLRGP = approximateGP(refMesh, finalGP)
    val ffdm      = PointDistributionModel3D(refMesh, finalLRGP)

    println(s"  Model rank    : ${ffdm.rank}")
    val cv = cumulativeVariance(finalLRGP)
    Seq(10, 20, 50, 100).foreach { n =>
      val v = cv.lift(n - 1).getOrElse(cv.lastOption.getOrElse(0.0))
      println(f"  Cumul. var @ $n%-3d components: ${v * 100}%5.1f %%")
    }

    val modelFile = new File(outDir, "ffdm.h5")
    StatisticalModelIO.writeStatisticalMeshModel(ffdm, modelFile).get
    println(s"  FFDM saved    : ${modelFile.getAbsolutePath}")

    println()
    println("─" * 70)
    println("OBJECTIVE 1 COMPLETE")
    println("  Outputs:")
    println(s"    Reference mesh : ${refOutFile.getAbsolutePath}")
    println(s"    FFDM model     : ${modelFile.getAbsolutePath}")
    println()
    println("  Next step: Stage 3 — use the FFDM as the GP prior for non-rigid")
    println("  ICP registration to establish point-to-point correspondences across")
    println("  all specimens.")
    println("=" * 70)
  }

  // ── Helpers ──────────────────────────────────────────────────────────────

  /**
   * Select the specimen whose mesh volume is closest to the population median.
   * Volume is a fast, landmark-free proxy for "typical size", and the median
   * specimen minimises expected shape distance to any other in the population.
   */
  def selectByVolumeMedian(specimens: IndexedSeq[ScapulaData.Specimen]): String = {
    println("  Computing mesh volumes for reference selection...")
    val withVols: IndexedSeq[(String, Double)] = specimens.map { s =>
      val vol = math.abs(ScapulaData.signedVolume(ScapulaData.loadMesh(s.file)))
      (s.modelId, vol)
    }.sortBy(_._2)

    val medIdx  = withVols.length / 2
    val chosen  = withVols(medIdx)._1

    println(f"  ${"Rank"}%4s  ${"Specimen"}%-34s  Volume (cm³)")
    withVols.zipWithIndex.foreach { case ((id, vol), i) =>
      val tag = if (i == medIdx) "  <── selected (median)" else ""
      println(f"  ${i + 1}%4d  $id%-34s  ${vol / 1000.0}%8.1f$tag")
    }
    chosen
  }

  /** Parse "sigma1:scale1,sigma2:scale2,..." into a sequence of (sigma, scale) pairs. */
  def parseKernelSpec(spec: String): Seq[(Double, Double)] =
    spec.split(",").toSeq.map { part =>
      val Array(s, sc) = part.trim.split(":")
      (s.toDouble, sc.toDouble)
    }

  /** Multi-scale Gaussian GP (zero mean). */
  def buildGP(components: Seq[(Double, Double)]): GaussianProcess[_3D, EuclideanVector[_3D]] = {
    val scalarKernel: PDKernel[_3D] =
      components
        .map { case (sigma, scale) => GaussianKernel[_3D](sigma) * (scale * scale) }
        .reduce[PDKernel[_3D]](_ + _)
    val vectorKernel: MatrixValuedPDKernel[_3D] = DiagonalKernel(scalarKernel, 3)
    GaussianProcess[_3D, EuclideanVector[_3D]](zeroMean, vectorKernel)
  }

  /** Pivoted-Cholesky low-rank approximation of the GP on the reference mesh. */
  def approximateGP(ref: TriangleMesh[_3D],
                    gp: GaussianProcess[_3D, EuclideanVector[_3D]]
  ): LowRankGaussianProcess[_3D, EuclideanVector[_3D]] =
    LowRankGaussianProcess.approximateGPCholesky(
      ref.pointSet,
      gp,
      relativeTolerance = Config.gpRelativeTolerance,
      interpolator       = NearestNeighborInterpolator3D[EuclideanVector[_3D]]()
    )

  /** Cumulative fraction of total prior variance explained by the first 1..rank components. */
  def cumulativeVariance(gp: LowRankGaussianProcess[_3D, EuclideanVector[_3D]]): IndexedSeq[Double] = {
    val eigenvalues = gp.eigenPairs.map(_._1)
    val total       = eigenvalues.sum
    if (total <= 0.0) return IndexedSeq.fill(eigenvalues.length)(0.0)
    eigenvalues.scanLeft(0.0)(_ + _).tail.map(_ / total)
  }
}
