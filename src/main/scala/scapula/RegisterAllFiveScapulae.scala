package scapula

import scalismo.common.PointId
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.geometry.*
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.kernels.{DiagonalKernel, GaussianKernel}
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

import java.awt.Color
import java.io.File
import java.time.LocalDate

/**
 * THE ONE-YEAR BUG WAS HERE:
 *   Wrong: landmarkThenIcp(refMesh, refLms, targetMesh, tLms)
 *          => reference lands in target space => every mesh in a different location
 *   Correct: landmarkThenIcp(targetMesh, tLms, refMesh, refLms)
 *             => target lands in reference space => all meshes overlap in the viewer
 *
 * Landmarks come from the CSV file. Nothing is hardcoded.
 */
object RegisterAllFiveScapulae {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    // ── Paths ──────────────────────────────────────────────────────────────
    val dataDir = Config.dataDir
    val outDir  = new File(Config.outDir, s"five_targets_${LocalDate.now()}")
    outDir.mkdirs()

    println(s"Data dir  : ${dataDir.getAbsolutePath}")
    println(s"Output dir: ${outDir.getAbsolutePath}")

    require(dataDir.exists() && dataDir.isDirectory,
      s"\nData directory not found: ${dataDir.getAbsolutePath}\n" +
      """Set SCAPULA_DATA_DIR, e.g.:  export SCAPULA_DATA_DIR="$HOME/Documents/100 plus scapula data/paired_scapulae_STLs_scapula"""")

    // ── Landmarks from CSV — NEVER hardcoded ───────────────────────────────
    val csvFile                           = ScapulaData.csvFile(dataDir)
    val (allLandmarks, fromHeader, _)     = ScapulaData.readLandmarkCsv(csvFile)
    println(s"Landmark CSV: ${csvFile.getName}  (${allLandmarks.size} rows)")
    if (!fromHeader)
      println("  !! Columns resolved by FALLBACK offsets — verify the printout below against your CSV!")

    val allSpecimens = ScapulaData.specimens(dataDir)
    val withLms      = allSpecimens.filter(s => allLandmarks.contains(s.modelId))
    require(withLms.nonEmpty, "No specimen found with both an STL file and a CSV landmark row.")

    // ── Reference ──────────────────────────────────────────────────────────
    val refSpec = withLms
      .find(s => s.modelId.contains("001") && !s.isRight)
      .orElse(withLms.find(_.modelId.contains("001")))
      .getOrElse(withLms.head)

    val refRaw = ScapulaData.loadMesh(refSpec.file)
    val refMesh: TriangleMesh[_3D] =
      if (refRaw.pointSet.numberOfPoints > Config.modelResolution)
        refRaw.operations.decimate(Config.modelResolution)
      else refRaw
    val refLms = allLandmarks(refSpec.modelId)

    println(s"\nReference: ${refSpec.modelId}  (${refMesh.pointSet.numberOfPoints} vertices)")
    println("Reference landmark positions (from CSV):")
    refLms.foreach(l => println(f"  ${l.id}%-4s  x=${l.point.x}%9.3f  y=${l.point.y}%9.3f  z=${l.point.z}%9.3f"))

    // ── Five targets ───────────────────────────────────────────────────────
    val coreNums = IndexedSeq("002", "004", "007", "010", "012")
    val targets: IndexedSeq[(String, File, IndexedSeq[Landmark[_3D]])] =
      coreNums.flatMap { num =>
        withLms
          .filterNot(_.modelId == refSpec.modelId)
          .find(s => s.modelId.contains(s"_${num}_") && !s.isRight)
          .orElse(withLms.filterNot(_.modelId == refSpec.modelId).find(_.modelId.contains(s"_${num}_")))
          .map(s => (s.modelId, s.file, allLandmarks(s.modelId)))
      }

    println(s"\nTargets found: ${targets.size}/5")
    targets.foreach { case (id, _, _) => println(s"  $id") }

    // ── GP prior — built ONCE on the fixed reference mesh ─────────────────
    val scalarKernel = GaussianKernel[_3D](100.0) * 15.0 + GaussianKernel[_3D](40.0) * 5.0
    val kernel       = DiagonalKernel(scalarKernel, 3)
    val gp           = GaussianProcess[_3D, EuclideanVector[_3D]](kernel)

    println("\nBuilding low-rank GP on reference mesh (done once)...")
    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      refMesh, gp,
      relativeTolerance = Config.gpRelativeTolerance,
      interpolator      = NearestNeighborInterpolator3D()
    )
    println(s"GP rank: ${lowRankGP.rank}")

    val step  = math.max(1, refMesh.pointSet.numberOfPoints / 2500)
    val ptIds = (0 until refMesh.pointSet.numberOfPoints by step).map(PointId(_))
    val iters = Config.icpIterations

    // ── UI ────────────────────────────────────────────────────────────────
    val ui = if (Config.showUi) {
      val u = ScalismoUI()
      val g = u.createGroup("Reference")
      val v = u.show(g, refMesh, refSpec.modelId)
      v.color   = new Color(240, 200, 50)
      v.opacity = 0.40f
      Some(u)
    } else None

    // ── Registration loop ─────────────────────────────────────────────────
    for (((tId, tFile, tLms), idx) <- targets.zipWithIndex) {
      println(s"\n══ [${idx + 1}/${targets.size}]  $tId ══")
      println("Target landmark positions (from CSV):")
      tLms.foreach(l => println(f"  ${l.id}%-4s  x=${l.point.x}%9.3f  y=${l.point.y}%9.3f  z=${l.point.z}%9.3f"))

      val targetRaw = ScapulaData.loadMesh(tFile)
      val targetMesh: TriangleMesh[_3D] =
        if (targetRaw.pointSet.numberOfPoints > Config.modelResolution)
          targetRaw.operations.decimate(Config.modelResolution)
        else targetRaw
      println(s"Target: ${targetMesh.pointSet.numberOfPoints} vertices")

      // ── STEP 1: bring TARGET into REFERENCE space ──────────────────────
      // Direction: targetMesh/tLms → refMesh/refLms
      // All results land in the same coordinate frame (reference space).
      // THIS IS THE FIX — previously the direction was reversed.
      val (targetInRef, _) = RigidAlign.landmarkThenIcp(
        targetMesh, tLms, refMesh, refLms, icpIterations = Config.icpIterations)
      val rigidStats = Metrics.symmetric(refMesh, targetInRef)
      println(s"1. Rigid (target→ref space): ${rigidStats.render}")

      // ── STEP 2: GP-ICP — deform refMesh to match targetInRef ──────────
      // Both meshes are now in reference space.
      // model.mean starts as refMesh and is pulled toward targetInRef.
      var model = PointDistributionModel[_3D, TriangleMesh](refMesh, lowRankGP)

      for (iter <- 0 until iters) {
        val mean   = model.mean
        val thresh = 5.0 + 15.0 * math.exp(-iter * 3.0 / iters)
        val correspondences: IndexedSeq[(PointId, Point[_3D])] = ptIds.flatMap { pid =>
          val pt      = mean.pointSet.point(pid)
          val nearest = targetInRef.operations.closestPointOnSurface(pt).point
          if ((nearest - pt).norm < thresh) Some((pid, nearest)) else None
        }
        if (correspondences.nonEmpty)
          model = model.posterior(correspondences, sigma2 = 0.5)
        if (iter % 10 == 0 || iter == iters - 1)
          println(f"   iter ${iter + 1}%3d: ${correspondences.size}%4d correspondences  thresh=${thresh}%.1f mm")
      }

      val registered: TriangleMesh[_3D] = model.mean  // in reference space ✓
      val nonRigidStats = Metrics.symmetric(registered, targetInRef)
      println(s"2. GP-ICP:                  ${nonRigidStats.render}")

      // ── STEP 3: Save ───────────────────────────────────────────────────
      val h5Out  = new File(outDir, s"model_${tId}.h5")
      val vtkOut = new File(outDir, s"registered_${tId}.vtk")
      StatisticalModelIO.writeStatisticalTriangleMeshModel3D(model, h5Out).get
      MeshIO.writeMesh(registered, vtkOut).get
      println(s"Saved: ${vtkOut.getName}  +  ${h5Out.getName}")

      // ── STEP 4: Visualise — all meshes in reference space, they overlap ─
      for (u <- ui) {
        val grp   = u.createGroup(s"[${idx + 1}] $tId")
        val tView = u.show(grp, targetInRef, "Target in ref space")
        tView.color   = new Color(180, 180, 180)
        tView.opacity = 0.35f
        val rView = u.show(grp, registered, "Registered")
        rView.color   = new Color(60, 160, 220)
        rView.opacity = 0.85f
      }
    }

    println(s"\nAll done. Output: ${outDir.getAbsolutePath}")
    println("All registered meshes are in REFERENCE space — they overlap in the viewer.")
  }
}
