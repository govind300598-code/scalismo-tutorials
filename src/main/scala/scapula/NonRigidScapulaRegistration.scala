package scapula

import scalismo.geometry.*
import scalismo.common.*
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.mesh.*
import scalismo.io.MeshIO
import scalismo.kernels.*
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, MultivariateNormalDistribution}
import scalismo.numerics.UniformMeshSampler3D
import breeze.linalg.{DenseMatrix, DenseVector}
import scalismo.ui.api.*
import scalismo.utils.Random

import java.io.File

object NonRigidScapulaRegistration extends App {

  scalismo.initialize()
  implicit val rng: Random = Random(42L)

  val dataDir = new File(sys.env.getOrElse("SCAPULA_DATA_DIR",
    "/home/g25upadh/Documents/database_v1.11/paired_scapulae_STLs"))
  val outDir  = new File(sys.env.getOrElse("SCAPULA_OUT_DIR",
    "/home/g25upadh/Documents/database_v1.11/scapula_ssm_out"), "non_rigid_output")
  val showUi  = sys.env.getOrElse("SCAPULA_UI", "true").toBoolean

  require(dataDir.exists(), s"Data directory not found: ${dataDir.getAbsolutePath}")
  outDir.mkdirs()

  val csvFile                    = ScapulaData.csvFile(dataDir)
  val (allLandmarks, fromHdr, _) = ScapulaData.readLandmarkCsv(csvFile)
  val allSpecimens               = ScapulaData.specimens(dataDir)

  val leftSpecimens = allSpecimens.filter(s => !s.isRight && allLandmarks.contains(s.modelId))
  println(s"Left specimens with landmarks: ${leftSpecimens.size}  (header-parsed CSV = $fromHdr)")
  require(leftSpecimens.size >= 7, "Need at least 7 left-side specimens with landmarks")

  val TARGET_PTS = 5000

  def loadAndDecimate(f: File): TriangleMesh[_3D] = {
    val m = ScapulaData.loadMesh(f)
    if (m.pointSet.numberOfPoints > TARGET_PTS) m.operations.decimate(TARGET_PTS) else m
  }

  // ── TASK 1 — Reference selection ─────────────────────────────────────────────
  println("\n════════════════════════════════════════")
  println("  TASK 1 — Reference Scapula Selection")
  println("════════════════════════════════════════")

  def lmCentroid(lms: IndexedSeq[Landmark[_3D]]): EuclideanVector[_3D] =
    lms.map(_.point.toVector).reduce(_ + _) * (1.0 / lms.size)

  val withCentroids = leftSpecimens.map(s => s -> lmCentroid(allLandmarks(s.modelId)))
  val datasetMean   = withCentroids.map(_._2).reduce(_ + _) * (1.0 / withCentroids.size)
  val refSpec       = withCentroids.minBy { case (_, c) => (c - datasetMean).norm2 }._1

  val referenceMesh = loadAndDecimate(refSpec.file)
  val refLms        = allLandmarks(refSpec.modelId)

  println(s"  Selected reference : ${refSpec.modelId}")
  println(s"  Vertices after decimate : ${referenceMesh.pointSet.numberOfPoints}")

  // ── GP-ICP helpers ────────────────────────────────────────────────────────────

  def buildGP(sigma: Double, scale: Double): LowRankGaussianProcess[_3D, EuclideanVector[_3D]] = {
    val zeroMean: Field[_3D, EuclideanVector[_3D]] =
      Field(EuclideanSpace3D, (_: Point[_3D]) => EuclideanVector3D(0, 0, 0))
    val scalarK = GaussianKernel[_3D](sigma) * scale
    val matrixK = DiagonalKernel(scalarK, outputDim = 3)
    val gp      = GaussianProcess[_3D, EuclideanVector[_3D]](zeroMean, matrixK)
    LowRankGaussianProcess.approximateGPCholesky(
      referenceMesh.pointSet,
      gp,
      relativeTolerance = 0.01,
      interpolator = NearestNeighborInterpolator3D()
    )
  }

  def rigidPreAlign(mesh: TriangleMesh[_3D], lms: IndexedSeq[Landmark[_3D]]): TriangleMesh[_3D] =
    RigidAlign.landmarkThenIcp(mesh, lms, referenceMesh, refLms, icpIterations = 20)._1

  def gpIcp(
    prior:      LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
    target:     TriangleMesh[_3D],
    numIter:    Int,
    numSamples: Int    = 500,
    noiseVar:   Double = 1.0
  ): TriangleMesh[_3D] = {
    val noise = MultivariateNormalDistribution(
      DenseVector.zeros[Double](3),
      DenseMatrix.eye[Double](3) * noiseVar
    )
    val tgtOps    = target.operations
    val sampleIds = UniformMeshSampler3D(referenceMesh, numSamples)
      .sample()
      .map { case (pt, _) => referenceMesh.pointSet.findClosestPoint(pt).id }
      .distinct

    var currentMesh = referenceMesh.transform(pt => pt + prior.mean(pt))

    for (i <- 1 to numIter) {
      val observations: IndexedSeq[(PointId, EuclideanVector[_3D], MultivariateNormalDistribution)] =
        sampleIds.map { pid =>
          val movedPt   = currentMesh.pointSet.point(pid)
          val closestPt = tgtOps.closestPointOnSurface(movedPt).point
          val refPt     = referenceMesh.pointSet.point(pid)
          (pid, closestPt - refPt, noise)
        }
      val posterior = prior.posterior(observations)
      currentMesh   = referenceMesh.transform(pt => pt + posterior.mean(pt))
      if (i % 10 == 0) println(s"      iter $i / $numIter")
    }
    currentMesh
  }

  // ── TASK 2 — Hyperparameter grid search ──────────────────────────────────────
  println("\n══════════════════════════════════════════════")
  println("  TASK 2 — Hyperparameter Grid Search")
  println("══════════════════════════════════════════════")

  val candidates  = leftSpecimens.filterNot(_.modelId == refSpec.modelId)
  val step        = math.max(1, candidates.size / 5)
  val fiveTargets = (0 until 5).map(i => candidates(math.min(i * step, candidates.size - 1)))
  println(s"  Five diverse targets: ${fiveTargets.map(_.modelId).mkString("\n    ", "\n    ", "")}")

  println("\n  Pre-aligning five targets ...")
  val preAligned: IndexedSeq[(ScapulaData.Specimen, TriangleMesh[_3D])] =
    fiveTargets.map { s =>
      val mesh    = loadAndDecimate(s.file)
      val aligned = rigidPreAlign(mesh, allLandmarks(s.modelId))
      println(s"    Aligned: ${s.modelId}")
      s -> aligned
    }

  case class GridResult(sigma: Double, scale: Double, avgMean: Double)

  val sigmas = Seq(30.0, 65.0, 100.0)
  val scales = Seq(50.0, 150.0)

  println("\n  Running grid (15 GP-ICP iters per target)...")
  val gridResults: Seq[GridResult] = for {
    sigma <- sigmas
    scale <- scales
  } yield {
    print(f"    sigma=$sigma%5.0f  scale=$scale%5.0f  ->  ")
    val gp   = buildGP(sigma, scale)
    val errs = preAligned.map { case (_, tgt) =>
      Metrics.symmetric(gpIcp(gp, tgt, numIter = 15, numSamples = 300), tgt).mean
    }
    val avg = errs.sum / errs.size
    println(f"avg mean dist = $avg%.3f mm   (per target: ${errs.map(e => f"$e%.2f").mkString(", ")})")
    GridResult(sigma, scale, avg)
  }

  println("\n  Grid results sorted by average mean distance:")
  println(f"  ${"sigma"}%8s  ${"scale"}%7s  ${"avgMean(mm)"}%12s")
  gridResults.sortBy(_.avgMean).foreach { r =>
    println(f"  ${r.sigma}%8.1f  ${r.scale}%7.1f  ${r.avgMean}%12.3f")
  }

  val best = gridResults.minBy(_.avgMean)
  println(f"\n  Best params: sigma=${best.sigma}  scale=${best.scale}  avgMean=${best.avgMean}%.3f mm")

  // ── TASK 3 — Visual assessment ────────────────────────────────────────────────
  println("\n════════════════════════════════════════════════════")
  println("  TASK 3 — Visual Assessment (40 iters, best params)")
  println("════════════════════════════════════════════════════")

  val bestGP = buildGP(best.sigma, best.scale)

  val fittedMeshes: IndexedSeq[(ScapulaData.Specimen, TriangleMesh[_3D], TriangleMesh[_3D])] =
    preAligned.map { case (s, tgt) =>
      println(s"  Fitting ${s.modelId} ...")
      val f       = gpIcp(bestGP, tgt, numIter = 40, numSamples = 500)
      val outFile = new File(outDir, s"non_rigid_fitted_${s.modelId}.stl")
      MeshIO.writeMesh(f, outFile).get
      println(s"    Saved -> ${outFile.getName}")
      (s, tgt, f)
    }

  // ── TASK 4 — Numerical error analysis ────────────────────────────────────────
  println("\n══════════════════════════════════════════════════════")
  println("  TASK 4 — Numerical Error Analysis (7 cases)")
  println("══════════════════════════════════════════════════════")

  val selfSt = Metrics.symmetric(referenceMesh, referenceMesh)
  println(f"\n  Reference self-error:")
  println(f"    ${refSpec.modelId}%-50s  ${selfSt.render}")

  println(f"\n  Five registered targets:")
  val fiveStats: IndexedSeq[(ScapulaData.Specimen, Metrics.SurfaceStats)] =
    fittedMeshes.map { case (s, tgt, f) =>
      val st = Metrics.symmetric(f, tgt)
      println(f"    ${s.modelId}%-50s  ${st.render}")
      s -> st
    }

  println(f"\n  Scanning up to 30 additional specimens for worst case...")
  val remaining = leftSpecimens
    .filterNot(s => s.modelId == refSpec.modelId || fiveTargets.exists(_.modelId == s.modelId))
    .take(30)

  val remainingStats: IndexedSeq[(ScapulaData.Specimen, Metrics.SurfaceStats)] =
    remaining.map { s =>
      print(s"    ${s.modelId} ... ")
      val mesh    = loadAndDecimate(s.file)
      val aligned = rigidPreAlign(mesh, allLandmarks(s.modelId))
      val f       = gpIcp(bestGP, aligned, numIter = 30, numSamples = 400)
      val st      = Metrics.symmetric(f, aligned)
      println(st.render)
      s -> st
    }

  val allStats  = fiveStats ++ remainingStats
  val worstCase = allStats.maxBy(_._2.mean)

  println(f"\n  Worst-performing specimen: ${worstCase._1.modelId}")
  println(f"    ${worstCase._1.modelId}%-50s  ${worstCase._2.render}")

  println("\n")
  println("  ┌─────────────────────────────────────────────────────────────────────────┐")
  println("  │                     SUMMARY — Surface Error (mm)                       │")
  println("  ├────────────────────────────────────────────┬──────────┬────────────────┤")
  println(f"  │ ${"Case"}%-42s │ ${"mean"}%8s │ ${"HD95"}%14s │")
  println("  ├────────────────────────────────────────────┼──────────┼────────────────┤")
  println(f"  │ ${refSpec.modelId + " (self)"}%-42s │ ${selfSt.mean}%8.3f │ ${selfSt.hd95}%14.3f │")
  (fiveStats ++ Seq(worstCase)).distinct.foreach { case (s, st) =>
    val tag = if (worstCase._1.modelId == s.modelId) " <- worst" else ""
    println(f"  │ ${(s.modelId + tag).take(42)}%-42s │ ${st.mean}%8.3f │ ${st.hd95}%14.3f │")
  }
  println("  └────────────────────────────────────────────┴──────────┴────────────────┘")

  if (showUi) {
    println("\n  Opening Scalismo viewer — press ENTER to exit.")
    val ui     = ScalismoUI()
    val refGrp = ui.createGroup("Reference")
    ui.show(refGrp, referenceMesh, "Reference")
    fittedMeshes.foreach { case (s, tgt, f) =>
      val g = ui.createGroup(s.modelId)
      ui.show(g, tgt, "Rigid-Aligned Target")
      ui.show(g, f,   "Non-Rigid Fitted")
    }
    scala.io.StdIn.readLine()
  }

  println("\nAll four tasks complete.")
}
