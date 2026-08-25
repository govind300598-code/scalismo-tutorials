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
 * Register five target scapulae to a common reference using:
 *  1. Landmark Procrustes (rigid, from CSV — NEVER hardcoded)
 *  2. Trimmed rigid ICP refinement
 *  3. Non-rigid GP-ICP with a two-scale Gaussian kernel
 *
 * Landmark coordinates are read from the CSV file found in the data directory.
 * Override paths via env vars: SCAPULA_DATA_DIR, SCAPULA_OUT_DIR.
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
      s"\n\nData directory not found: ${dataDir.getAbsolutePath}\n" +
      "Set SCAPULA_DATA_DIR, for example:\n" +
      """  export SCAPULA_DATA_DIR="$HOME/Documents/100 plus scapula data/paired_scapulae_STLs_scapula"""")

    // ── Landmark CSV ── coordinates come from here, NEVER hardcoded ────────
    val csvFile                                  = ScapulaData.csvFile(dataDir)
    val (allLandmarks, fromHeader, headerRow)    = ScapulaData.readLandmarkCsv(csvFile)
    println(s"Landmark CSV: ${csvFile.getName}")
    println(s"  ${allLandmarks.size} rows; columns resolved ${
      if (fromHeader) "by header name (safe)" else "by FALLBACK OFFSETS — verify columns!"}")

    val allSpecimens = ScapulaData.specimens(dataDir)
    val withLms      = allSpecimens.filter(s => allLandmarks.contains(s.modelId))
    require(withLms.nonEmpty,
      "No specimen found that has both an STL file and a CSV landmark row.\n" +
      "Check that the model-id column in the CSV matches the STL filename stem (without .stl).")

    // ── Reference specimen ─────────────────────────────────────────────────
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
    println("Reference landmark positions from CSV:")
    refLms.foreach { l =>
      println(f"  ${l.id}%-4s  x=${l.point.x}%9.3f  y=${l.point.y}%9.3f  z=${l.point.z}%9.3f")
    }

    // ── Five target specimens ──────────────────────────────────────────────
    val coreNums = IndexedSeq("002", "004", "007", "010", "012")
    val targets: IndexedSeq[(String, File, IndexedSeq[Landmark[_3D]])] =
      coreNums.flatMap { num =>
        withLms
          .filterNot(_.modelId == refSpec.modelId)
          .find(s => s.modelId.contains(s"_${num}_") && !s.isRight)
          .orElse(
            withLms
              .filterNot(_.modelId == refSpec.modelId)
              .find(_.modelId.contains(s"_${num}_"))
          )
          .map(s => (s.modelId, s.file, allLandmarks(s.modelId)))
      }

    println(s"\nTargets found: ${targets.size}/5")
    targets.foreach { case (id, _, _) => println(s"  $id") }
    if (targets.size < 5)
      println("  !! Missing targets — check that STL files exist and CSV has a row for each")

    // ── Gaussian-process prior ─────────────────────────────────────────────
    // Two-scale: 100 mm global (large shape modes) + 40 mm local (surface detail).
    // Amplitudes in mm — 15 mm global variance, 5 mm local variance.
    val scalarKernel = GaussianKernel[_3D](100.0) * 15.0 + GaussianKernel[_3D](40.0) * 5.0
    val kernel       = DiagonalKernel(scalarKernel, 3)
    val gp           = GaussianProcess[_3D, EuclideanVector[_3D]](kernel)

    // ── UI ────────────────────────────────────────────────────────────────
    val ui = if (Config.showUi) {
      val u = ScalismoUI()
      val g = u.createGroup("Reference")
      val v = u.show(g, refMesh, refSpec.modelId)
      v.color   = new Color(240, 200, 50)
      v.opacity = 0.25f
      Some(u)
    } else None

    // ── Registration loop ─────────────────────────────────────────────────
    for (((tId, tFile, tLms), idx) <- targets.zipWithIndex) {
      println(s"\n══ [${idx + 1}/${targets.size}]  $tId ══")
      println("Target landmark positions from CSV:")
      tLms.foreach { l =>
        println(f"  ${l.id}%-4s  x=${l.point.x}%9.3f  y=${l.point.y}%9.3f  z=${l.point.z}%9.3f")
      }

      val targetRaw = ScapulaData.loadMesh(tFile)
      val targetMesh: TriangleMesh[_3D] =
        if (targetRaw.pointSet.numberOfPoints > Config.modelResolution)
          targetRaw.operations.decimate(Config.modelResolution)
        else targetRaw
      println(s"Target mesh: ${targetMesh.pointSet.numberOfPoints} vertices")

      // 1. Landmark Procrustes → trimmed rigid ICP
      val (alignedRef, _) = RigidAlign.landmarkThenIcp(
        refMesh, refLms, targetMesh, tLms, icpIterations = Config.icpIterations)
      val rigidStats = Metrics.symmetric(alignedRef, targetMesh)
      println(s"1. After rigid:   ${rigidStats.render}")

      // 2. Low-rank GP built on the rigidly-aligned reference domain
      val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
        alignedRef,
        gp,
        relativeTolerance = Config.gpRelativeTolerance,
        interpolator      = NearestNeighborInterpolator3D()
      )
      println(s"2. GP rank: ${lowRankGP.rank}")

      // 3. GP-ICP — distance threshold shrinks from 20 mm → 5 mm
      var model = PointDistributionModel[_3D, TriangleMesh](alignedRef, lowRankGP)
      val step  = math.max(1, alignedRef.pointSet.numberOfPoints / 2500)
      val ptIds = (0 until alignedRef.pointSet.numberOfPoints by step).map(PointId(_))
      val iters = Config.icpIterations

      for (iter <- 0 until iters) {
        val mean   = model.mean
        val thresh = 5.0 + 15.0 * math.exp(-iter * 3.0 / iters)
        val correspondences: IndexedSeq[(PointId, Point[_3D])] = ptIds.flatMap { pid =>
          val pt      = mean.pointSet.point(pid)
          val nearest = targetMesh.operations.closestPointOnSurface(pt).point
          if ((nearest - pt).norm < thresh) Some((pid, nearest)) else None
        }
        if (correspondences.nonEmpty)
          model = model.posterior(correspondences, sigma2 = 0.5)
        if (iter % 10 == 0 || iter == iters - 1)
          println(f"   iter ${iter + 1}%3d: ${correspondences.size}%4d correspondences  thresh=${thresh}%.1f mm")
      }

      val registered: TriangleMesh[_3D] = model.mean
      val nonRigidStats                 = Metrics.symmetric(registered, targetMesh)
      println(s"3. After GP-ICP:  ${nonRigidStats.render}")

      // 4. Save
      val h5Out  = new File(outDir, s"model_${tId}.h5")
      val vtkOut = new File(outDir, s"registered_${tId}.vtk")
      StatisticalModelIO.writeStatisticalTriangleMeshModel3D(model, h5Out).get
      MeshIO.writeMesh(registered, vtkOut).get
      println(s"Saved: ${vtkOut.getName}  +  ${h5Out.getName}")

      // 5. Visualise
      for (u <- ui) {
        val grp   = u.createGroup(s"[${idx + 1}] $tId")
        val tView = u.show(grp, targetMesh, "Target")
        tView.color   = new Color(180, 180, 180)
        tView.opacity = 0.45f
        val rView = u.show(grp, registered, "Registered")
        rView.color   = new Color(60, 160, 220)
        rView.opacity = 0.90f
      }
    }

    println(s"\nAll done. Output: ${outDir.getAbsolutePath}")
  }
}
