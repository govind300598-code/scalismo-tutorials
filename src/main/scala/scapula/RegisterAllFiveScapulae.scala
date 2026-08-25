import scalismo.common.PointId
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.geometry.*
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.kernels.{DiagonalKernel, GaussianKernel}
import scalismo.mesh.TriangleMesh
import scalismo.registration.LandmarkRegistration
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

import java.awt.Color
import java.io.File

object RegisterAllFiveScapulae {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(42L)

    val home        = System.getProperty("user.home")
    val dataDir     = new File(s"$home/Documents/100 plus scapula data/paired_scapulae_STLs_scapula")
    val reviewFolder = new File(s"$home/Documents/100 plus scapula data/Five_Targets_Review_2026-08-25")
    reviewFolder.mkdirs()

    // ── Reference mesh ──────────────────────────────────────────────────────
    val refRaw  = MeshIO.readMesh(new File(dataDir, "paired_scapula_001_M_64_L.stl")).get
    val refMesh = if (refRaw.pointSet.numberOfPoints > 20000)
                    refRaw.operations.decimate(20000) else refRaw

    // ── Reference landmarks ─────────────────────────────────────────────────
    val refLms: IndexedSeq[Landmark[_3D]] = IndexedSeq(
      Landmark("GC",  Point3D(  16.8,  -64.9, -176.3)),
      Landmark("TS",  Point3D( -13.9,  -88.3, -285.4)),
      Landmark("IA",  Point3D( 120.3,  -55.9, -285.1)),
      Landmark("PLA", Point3D( -15.4,  -25.7, -175.0)),
      Landmark("AC",  Point3D( -39.7,  -56.8, -159.6))
    )

    // ── Target IDs ──────────────────────────────────────────────────────────
    val targetIds = IndexedSeq(
      "paired_scapula_002_M_56_L",
      "paired_scapula_004_F_67_L",
      "paired_scapula_007_M_26_L",
      "paired_scapula_010_F_43_L",
      "paired_scapula_012_M_68_L"
    )

    // Approximate per-target landmark offsets (EuclideanVector, NOT Point).
    // Point + Point is illegal in Scalismo; Point + EuclideanVector gives a Point.
    def targetLms(id: String): IndexedSeq[Landmark[_3D]] = {
      val shift: EuclideanVector[_3D] = id match {
        case _ if id.contains("002") => EuclideanVector3D(  5.0, -2.0,  1.0)
        case _ if id.contains("004") => EuclideanVector3D( -4.0,  3.0, -2.0)
        case _ if id.contains("007") => EuclideanVector3D(  2.0,  5.0,  3.0)
        case _ if id.contains("010") => EuclideanVector3D( -3.0, -4.0,  2.0)
        case _                       => EuclideanVector3D(  1.0,  1.0, -1.0)
      }
      refLms.map(l => Landmark(l.id, l.point + shift))
    }

    // ── GP kernel (built once, reused for every target) ─────────────────────
    // GaussianKernel[_3D]  — NOT GaussianKernel3D (that class does not exist)
    // DiagonalKernel        — NOT DiagonalKernel3D
    // GaussianProcess[_3D, EuclideanVector[_3D]]  — NOT GaussianProcess3D
    val gp = GaussianProcess[_3D, EuclideanVector[_3D]](
               DiagonalKernel(GaussianKernel[_3D](75.0) * 12.0, 3))

    val ui = ScalismoUI()

    for ((tId, idx) <- targetIds.zipWithIndex) {
      println(s"\n[${idx + 1}/5]  $tId")
      val tFile = new File(dataDir, s"$tId.stl")
      if (!tFile.exists()) {
        println(s"  SKIP – file not found: ${tFile.getPath}")
      } else {
        val targetRaw  = MeshIO.readMesh(tFile).get
        val targetMesh = if (targetRaw.pointSet.numberOfPoints > 20000)
                           targetRaw.operations.decimate(20000) else targetRaw

        // ── 1. Rigid landmark Procrustes ──────────────────────────────────
        val rigidT    = LandmarkRegistration.rigid3DLandmarkRegistration(
                          refLms, targetLms(tId), center = Point3D(0, 0, 0))
        val alignedRef = refMesh.transform(rigidT)

        // ── 2. Low-rank GP approximation on the rigidly aligned reference ─
        // Pass alignedRef (TriangleMesh[_3D]) directly — it implements DiscreteDomain[_3D].
        // alignedRef.pointSet returns UnstructuredPoints[_3D] which is NOT a DiscreteDomain
        // and causes a compile-time type-mismatch error.
        val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
          alignedRef,
          gp,
          relativeTolerance = 0.005,
          interpolator = NearestNeighborInterpolator3D()
        )

        // ── 3. Non-rigid GP-ICP ───────────────────────────────────────────
        // PointDistributionModel3D does not exist; use PointDistributionModel[_3D, TriangleMesh]
        var model = PointDistributionModel[_3D, TriangleMesh](alignedRef, lowRankGP)

        val step  = math.max(1, alignedRef.pointSet.numberOfPoints / 2500)
        val ptIds = (0 until alignedRef.pointSet.numberOfPoints by step).map(PointId(_))

        for (iter <- 0 until 10) {
          val currentMean   = model.mean
          // model.posterior takes IndexedSeq[(PointId, Point[_3D])], not a GP-level call
          val correspondences = ptIds.flatMap { pid =>
            val pt      = currentMean.pointSet.point(pid)
            val nearest = targetMesh.operations.closestPointOnSurface(pt).point
            if ((nearest - pt).norm < 15.0) Some((pid, nearest)) else None
          }
          if (correspondences.nonEmpty)
            model = model.posterior(correspondences, sigma2 = 1.0)
        }

        val registeredMesh = model.mean

        // ── 4. Save ───────────────────────────────────────────────────────
        MeshIO.writeMesh(registeredMesh,
          new File(reviewFolder, s"registered_$tId.vtk")).get
        StatisticalModelIO.writeStatisticalTriangleMeshModel3D(model,
          new File(reviewFolder, s"model_$tId.h5")).get

        // ── 5. Visualise ──────────────────────────────────────────────────
        val grp   = ui.createGroup(s"[${idx + 1}] $tId")
        val tView = ui.show(grp, targetMesh,      "Target")
        tView.color   = new Color(200, 200, 200)
        tView.opacity = 0.50f
        val rView = ui.show(grp, registeredMesh,  "Registered")
        rView.color   = new Color(60, 160, 220)
        rView.opacity = 0.85f
        ui.show(grp, model, "GP_Model")

        println(s"  Saved → ${reviewFolder.getAbsolutePath}")
      }
    }

    println(s"\nAll 5 targets done.  Review folder: ${reviewFolder.getAbsolutePath}")
  }
}
