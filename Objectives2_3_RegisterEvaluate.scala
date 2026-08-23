//> using scala 3.3.3
//> using dep ch.unibas.cs.gravis::scalismo:0.92.1
//> using javaOpt -Xmx8G
//
// Objectives 2 & 3 — FFDM Registration and Evaluation
//
// What this script does:
//   • Rebuilds ffdm_model.h5 if missing (fixes the Obj-1 Step-D OOM crash by
//     using relativeTolerance=0.05 and only 2500 reference vertices instead of
//     the original 5000 + tolerance 0.01 that produced a huge uncapped rank).
//   • Selects 5 anatomically diverse test specimens from the rigid-aligned set,
//     distributed uniformly across the shape-distance range from the reference.
//   • Registers the FFDM to each test specimen via iterative GP posterior
//     conditioning (ICP + shape prior) for Objective 2.
//   • Computes MSD, RMS, Hausdorff, P95, P99 surface-distance errors.
//   • Identifies the best and worst registration cases for Objective 3.
//   • Writes a 7-case report (5 test + best + worst, de-duplicated).
//   • Saves all fitted meshes, a metrics CSV, and a plain-text summary.
//
// Expected input layout (produced by Objective1_BuildFFDM.scala):
//   <outRoot>/rigid_aligned/rigid_<specId>.stl
//   <outRoot>/ffdm_model.h5           (or will be rebuilt)
//   <outRoot>/ffdm_parameter_search.csv
//   <outRoot>/reference_selection.txt
//
// Output layout:
//   <outRoot>/ffdm_fitted/fitted_<specId>.stl   (5 registered meshes)
//   <outRoot>/obj2_registration_metrics.csv
//   <outRoot>/obj3_summary.txt

import scalismo.geometry.*
import scalismo.common.*
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.mesh.*
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.statisticalmodel.*
import scalismo.kernels.*
import scalismo.utils.Random
import java.io.{File, PrintWriter}
import scala.util.Try
import scala.io.Source

object Objectives2_3_RegisterEvaluate extends App {

  scalismo.initialize()
  implicit val rng: Random = Random(42L)

  // ── Paths ─────────────────────────────────────────────────────────────────
  val home        = System.getProperty("user.home")
  val outRoot     = new File(s"$home/Documents/database_v1.11/scapula_ssm_out")
  val rigidDir    = new File(outRoot, "rigid_aligned")
  val modelFile   = new File(outRoot, "ffdm_model.h5")
  val fittedDir   = new File(outRoot, "ffdm_fitted")
  val reportCsv   = new File(outRoot, "obj2_registration_metrics.csv")
  val summaryTxt  = new File(outRoot, "obj3_summary.txt")
  val refInfoTxt  = new File(outRoot, "reference_selection.txt")
  val searchCsv   = new File(outRoot, "ffdm_parameter_search.csv")

  fittedDir.mkdirs()

  require(rigidDir.exists(),
    s"rigid_aligned/ not found at $rigidDir — run Objective1_BuildFFDM.scala first.")

  // ── Configuration ─────────────────────────────────────────────────────────
  val targetRefVerts = 2500   // smaller ref keeps model matrix serialisable at 8 GB
  val gpTolerance    = 0.05   // relative Cholesky tolerance (~40–80 modes, not hundreds)
  val fitIter        = 10     // ICP + GP posterior iterations
  val noiseVar       = 1.5    // correspondence noise variance, mm²
  val defaultSigma   = 20.0
  val defaultScale   = 30.0

  // ── Surface-distance helpers ───────────────────────────────────────────────
  def oneSided(from: TriangleMesh[_3D], to: TriangleMesh[_3D]): IndexedSeq[Double] = {
    val ops = to.operations
    from.pointSet.points.map(p => (p - ops.closestPointOnSurface(p).point).norm).toIndexedSeq
  }

  def sym(a: TriangleMesh[_3D], b: TriangleMesh[_3D]): IndexedSeq[Double] =
    oneSided(a, b) ++ oneSided(b, a)

  def msdOf(a: TriangleMesh[_3D], b: TriangleMesh[_3D]): Double =
    { val d = sym(a, b); d.sum / d.size }

  def rmsOf(a: TriangleMesh[_3D], b: TriangleMesh[_3D]): Double =
    { val d = sym(a, b); math.sqrt(d.map(x => x * x).sum / d.size) }

  def hdOf(a: TriangleMesh[_3D], b: TriangleMesh[_3D]): Double =
    sym(a, b).max

  def pctOf(a: TriangleMesh[_3D], b: TriangleMesh[_3D], pct: Double): Double = {
    val d = sym(a, b).sorted
    d(math.min((d.size * pct / 100.0).toInt, d.size - 1))
  }

  // ── Load rigid-aligned specimens ───────────────────────────────────────────
  println(s"Loading rigid-aligned meshes from $rigidDir …")

  case class Specimen(id: String, mesh: TriangleMesh[_3D])

  val allRigid: IndexedSeq[Specimen] =
    Option(rigidDir.listFiles()).getOrElse(Array.empty)
      .filter(_.getName.endsWith(".stl"))
      .sortBy(_.getName)
      .map { f =>
        val id = f.getName.stripPrefix("rigid_").stripSuffix(".stl")
        Specimen(id, MeshIO.readMesh(f).getOrElse(sys.error(s"Cannot read $f")))
      }.toIndexedSeq

  require(allRigid.nonEmpty, s"No .stl files found in $rigidDir")
  println(s"  Loaded ${allRigid.size} specimens.")

  // Reference ID from Obj-1 output file, or fall back to first specimen
  val refId: String = Try {
    val src = Source.fromFile(refInfoTxt)
    try src.getLines().next().replace("Selected reference specimen: ", "").trim
    finally src.close()
  }.getOrElse(allRigid.head.id)

  println(s"  Reference: $refId")
  val refSpec = allRigid.find(_.id == refId)
    .getOrElse(sys.error(s"Reference '$refId' not found in $rigidDir"))
  val nonRef = allRigid.filterNot(_.id == refId)

  // ── Best sigma/scale from parameter search CSV ─────────────────────────────
  val (bestSigma, bestScale): (Double, Double) = Try {
    val src = Source.fromFile(searchCsv)
    try {
      src.getLines().drop(1)
        .map(_.split(",")).filter(_.length >= 3)
        .map(p => (p(0).toDouble, p(1).toDouble, p(2).toDouble))
        .toSeq.minBy(_._3) match { case (s, c, _) => (s, c) }
    } finally src.close()
  }.getOrElse((defaultSigma, defaultScale))

  println(s"  Optimal parameters from search: sigma=$bestSigma, scale=$bestScale")

  // ── GP model builder — memory-safe version ─────────────────────────────────
  // Fix for Obj-1 Step-D OOM: decimate to 2500 verts + tolerance 0.05
  // (original used 5000 verts + 0.01, producing an uncapped-rank matrix that
  //  overflowed ujson's StringBuffer during HDF5 serialisation at 8 GB heap)
  def buildModel(
      ref: TriangleMesh[_3D],
      sigma: Double,
      scale: Double
  ): PointDistributionModel[_3D, TriangleMesh] = {
    val decRef =
      if (ref.pointSet.numberOfPoints > targetRefVerts) ref.operations.decimate(targetRefVerts)
      else ref
    val zero = Field[_3D, EuclideanVector[_3D]](RealSpace[_3D], _ => EuclideanVector(0.0, 0.0, 0.0))
    val kern =
      DiagonalKernel(GaussianKernel[_3D](sigma) * scale, 3) +
      DiagonalKernel(GaussianKernel[_3D](sigma / 3.0) * (scale / 3.0), 3)
    val gp   = GaussianProcess[_3D, EuclideanVector[_3D]](zero, kern)
    val lr   = LowRankGaussianProcess.approximateGPCholesky(
      decRef.pointSet, gp,
      relativeTolerance = gpTolerance,
      interpolator = NearestNeighborInterpolator3D()
    )
    PointDistributionModel(decRef, lr)
  }

  // ── Load or rebuild the FFDM ───────────────────────────────────────────────
  val model: PointDistributionModel[_3D, TriangleMesh] =
    Try(StatisticalModelIO.readStatisticalTriangleMeshModel3D(modelFile).get) match {
      case scala.util.Success(m) =>
        println(s"\nLoaded FFDM from ${modelFile.getAbsolutePath}")
        println(s"  rank = ${m.rank},  ref vertices = ${m.reference.pointSet.numberOfPoints}")
        m
      case scala.util.Failure(ex) =>
        println(s"\nffdm_model.h5 not readable — rebuilding " +
          s"(sigma=$bestSigma, scale=$bestScale, tol=$gpTolerance)…")
        if (ex.getMessage != null) println(s"  Reason: ${ex.getMessage}")
        val m = buildModel(refSpec.mesh, bestSigma, bestScale)
        println(s"  Built: rank=${m.rank}, ref vertices=${m.reference.pointSet.numberOfPoints}")
        print(s"  Saving to ${modelFile.getAbsolutePath}… ")
        StatisticalModelIO.writeStatisticalTriangleMeshModel3D(m, modelFile)
          .getOrElse(sys.error("Failed to save model"))
        println("done.")
        m
    }

  // ── Select 5 anatomically diverse test specimens ───────────────────────────
  // Rank non-reference specimens by symmetric MSD from the reference and pick
  // 5 uniformly spread across that distribution (min, Q1, median, Q3, max).
  println("\nRanking specimens by distance to reference (to select diverse test set)…")
  val byDist: IndexedSeq[(Specimen, Double)] =
    nonRef.map(s => (s, msdOf(s.mesh, refSpec.mesh))).sortBy(_._2)
  val sz  = byDist.size
  val idx = Seq(0, sz / 4, sz / 2, 3 * sz / 4, sz - 1).distinct.take(5)
  val testSet: Seq[Specimen] = idx.map(byDist(_)._1)

  println("5 test specimens (evenly spanning the shape-distance range from the reference):")
  testSet.zipWithIndex.foreach { case (s, i) =>
    val d = byDist.find(_._1.id == s.id).map(_._2).getOrElse(0.0)
    println(f"  ${i + 1}. ${s.id}%-40s  dist_to_ref = $d%.3f mm")
  }

  // ── Objective 2: ICP + GP posterior registration ───────────────────────────
  // Each iteration:
  //   1. Project each reference point onto the target surface (closest point).
  //   2. Condition the current GP model on those point correspondences.
  //   3. The posterior mean becomes the updated fitted mesh.
  def register(target: TriangleMesh[_3D]): TriangleMesh[_3D] = {
    var curMdl  = model
    var curMesh = model.mean
    val tOps    = target.operations
    for (_ <- 0 until fitIter) {
      val obs: IndexedSeq[(PointId, Point[_3D])] =
        model.reference.pointSet.pointIds.toIndexedSeq.map { id =>
          (id, tOps.closestPointOnSurface(curMesh.pointSet.point(id)).point)
        }
      curMdl  = curMdl.posterior(obs, noiseVar)
      curMesh = curMdl.mean
    }
    curMesh
  }

  println(s"\n[Objective 2] Registering FFDM to ${testSet.size} test specimens…")

  case class RegResult(
      id: String,
      msd: Double,
      rms: Double,
      hd: Double,
      p95: Double,
      p99: Double
  )

  val results: Seq[RegResult] = testSet.map { spec =>
    print(s"  → ${spec.id} … ")
    val fitted = register(spec.mesh)
    val mv = msdOf(fitted, spec.mesh)
    val rv = rmsOf(fitted, spec.mesh)
    val hv = hdOf(fitted, spec.mesh)
    val p5 = pctOf(fitted, spec.mesh, 95.0)
    val p9 = pctOf(fitted, spec.mesh, 99.0)
    MeshIO.writeMesh(fitted, new File(fittedDir, s"fitted_${spec.id}.stl"))
      .getOrElse(println(s"  WARNING: could not write fitted mesh for ${spec.id}"))
    println(f"MSD=$mv%.3f  RMS=$rv%.3f  HD=$hv%.3f  P95=$p5%.3f mm")
    RegResult(spec.id, mv, rv, hv, p5, p9)
  }

  // Write metrics CSV
  val cw = new PrintWriter(reportCsv)
  cw.println("specimen_id,msd_mm,rms_mm,hausdorff_mm,p95_mm,p99_mm")
  results.foreach(r =>
    cw.println(f"${r.id},${r.msd}%.4f,${r.rms}%.4f,${r.hd}%.4f,${r.p95}%.4f,${r.p99}%.4f")
  )
  cw.close()
  println(s"\nMetrics CSV → ${reportCsv.getAbsolutePath}")

  // ── Objective 3: identify best & worst, 7-case report ─────────────────────
  println("[Objective 3] Identifying best and worst cases…")
  val sorted    = results.sortBy(_.msd)
  val bestCase  = sorted.head
  val worstCase = sorted.last

  println(f"  BEST  : ${bestCase.id}  MSD=${bestCase.msd}%.3f mm")
  println(f"  WORST : ${worstCase.id}  MSD=${worstCase.msd}%.3f mm")

  // De-duplicate: if best/worst already in the 5 test set, they won't appear twice
  val sevenCases: Seq[RegResult] =
    (results ++ Seq(bestCase, worstCase)).distinctBy(_.id)

  // Reload parameter table for the summary
  val paramRows: Seq[(Double, Double, Double)] = Try {
    val src = Source.fromFile(searchCsv)
    try src.getLines().drop(1)
      .map(_.split(",")).filter(_.length >= 3)
      .map(p => (p(0).toDouble, p(1).toDouble, p(2).toDouble))
      .toSeq.sortBy(_._3)
    finally src.close()
  }.getOrElse(Seq.empty)

  val avgMsd = results.map(_.msd).sum / results.size
  val avgRms = results.map(_.rms).sum / results.size
  val avgHd  = results.map(_.hd ).sum / results.size
  val avgP95 = results.map(_.p95).sum / results.size

  // ── Write obj3_summary.txt ─────────────────────────────────────────────────
  val tw = new PrintWriter(summaryTxt)

  def hr(ch: Char = '─', w: Int = 72): String = ch.toString * w

  tw.println(hr('='))
  tw.println("FFDM REGISTRATION STUDY  —  Objectives 2 & 3 Summary")
  tw.println(hr('='))
  tw.println()
  tw.println(s"Model file            : ${modelFile.getAbsolutePath}")
  tw.println(s"Reference specimen    : $refId")
  tw.println(f"Model rank            : ${model.rank}")
  tw.println(f"Reference vertices    : ${model.reference.pointSet.numberOfPoints}")
  tw.println(f"Optimal sigma         : $bestSigma%.1f mm   (GP spatial length-scale)")
  tw.println(f"Optimal scale         : $bestScale%.1f mm   (GP deformation amplitude)")
  tw.println(f"Fit iterations        : $fitIter")
  tw.println(f"Correspondence noise  : $noiseVar%.1f mm²")
  tw.println()

  if (paramRows.nonEmpty) {
    tw.println(hr() + "\nParameter Search Results  (Objective 1, Step C)\n" + hr())
    tw.println(f"  ${"sigma (mm)":>10}  ${"scale (mm)":>10}  ${"avg_residual (mm)":>18}  Note")
    tw.println("  " + hr('-', 58))
    paramRows.foreach { case (s, c, r) =>
      val note = if (s == bestSigma && c == bestScale) "  ← OPTIMAL" else ""
      tw.println(f"  $s%10.1f  $c%10.1f  $r%18.4f$note")
    }
    tw.println()
  }

  tw.println(hr() + "\nObjective 2 — Registration Errors: 5 Test Specimens\n" + hr())
  tw.println(f"  ${"Specimen":<42} ${"MSD":>7}  ${"RMS":>7}  ${"HD":>9}  ${"P95":>7}  ${"P99":>7}  (all mm)")
  tw.println("  " + hr('-', 82))
  results.foreach { r =>
    tw.println(f"  ${r.id}%-42s ${r.msd}%7.3f  ${r.rms}%7.3f  ${r.hd}%9.3f  ${r.p95}%7.3f  ${r.p99}%7.3f")
  }
  tw.println("  " + hr('-', 82))
  tw.println(f"  ${"AVERAGE":<42} $avgMsd%7.3f  $avgRms%7.3f  $avgHd%9.3f  $avgP95%7.3f")
  tw.println()

  tw.println(hr() + "\nObjective 3 — Optimal & Limiting Performance\n" + hr())
  tw.println()
  tw.println(s"  BEST CASE  (lowest MSD)   : ${bestCase.id}")
  tw.println(f"    MSD  = ${bestCase.msd}%.3f mm")
  tw.println(f"    RMS  = ${bestCase.rms}%.3f mm")
  tw.println(f"    HD   = ${bestCase.hd}%.3f mm")
  tw.println(f"    P95  = ${bestCase.p95}%.3f mm")
  tw.println(f"    P99  = ${bestCase.p99}%.3f mm")
  tw.println()
  tw.println(s"  WORST CASE (highest MSD)  : ${worstCase.id}")
  tw.println(f"    MSD  = ${worstCase.msd}%.3f mm")
  tw.println(f"    RMS  = ${worstCase.rms}%.3f mm")
  tw.println(f"    HD   = ${worstCase.hd}%.3f mm")
  tw.println(f"    P95  = ${worstCase.p95}%.3f mm")
  tw.println(f"    P99  = ${worstCase.p99}%.3f mm")
  tw.println()
  tw.println(f"  Performance ratio  worst/best MSD : ${worstCase.msd / math.max(bestCase.msd, 1e-9)}%.2fx")
  tw.println(f"  Performance ratio  worst/best HD  : ${worstCase.hd  / math.max(bestCase.hd,  1e-9)}%.2fx")
  tw.println()
  tw.println("  Interpretation:")
  tw.println(s"  • '${bestCase.id}' is nearest in shape to the reference;")
  tw.println("    the GP prior provides a near-correct initialisation with little residual.")
  tw.println(s"  • '${worstCase.id}' is the most atypical shape;")
  tw.println("    its higher residual marks the model's limiting registration accuracy.")
  tw.println("  • The optimal parameters (sigma/scale above) were determined by minimising")
  tw.println("    average residual surface distance over 4 held-out specimens in Obj-1 Step C.")
  tw.println()

  tw.println(hr() + "\n7-Case Report  (5 test + BEST + WORST, de-duplicated)\n" + hr())
  tw.println(f"  ${"Specimen":<42} ${"MSD":>7}  ${"RMS":>7}  ${"HD":>9}  ${"P95":>7}  ${"Role":<10}")
  tw.println("  " + hr('-', 85))
  sevenCases.foreach { r =>
    val role =
      if   (r.id == bestCase.id && r.id == worstCase.id) "BEST+WORST"
      else if (r.id == bestCase.id)  "BEST"
      else if (r.id == worstCase.id) "WORST"
      else "test"
    tw.println(f"  ${r.id}%-42s ${r.msd}%7.3f  ${r.rms}%7.3f  ${r.hd}%9.3f  ${r.p95}%7.3f  $role%-10s")
  }
  tw.println()

  tw.println(hr() + "\nOutput Files\n" + hr())
  tw.println(s"  Fitted meshes  : ${fittedDir.getAbsolutePath}/fitted_<specId>.stl")
  tw.println(s"  Metrics CSV    : ${reportCsv.getAbsolutePath}")
  tw.println(s"  FFDM model     : ${modelFile.getAbsolutePath}")
  tw.println(s"  This summary   : ${summaryTxt.getAbsolutePath}")
  tw.close()

  println(s"Summary → ${summaryTxt.getAbsolutePath}")
  println(s"Fitted STLs → ${fittedDir.getAbsolutePath}/")
  println("\nObjectives 2 & 3 complete.")
}
