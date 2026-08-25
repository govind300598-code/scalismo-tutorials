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

/** Rigid + GP-ICP registration of five target scapulae against a common reference.
 *
 *  Run with:  sbt "runMain RegisterAllFiveScapulae"
 *
 *  Outputs per target (saved to reviewFolder):
 *    registered_<id>.vtk   — registered surface in target space
 *    model_<id>.h5         — posterior GP model (for downstream SSM or inspection)
 */
object RegisterAllFiveScapulae {

  // ── Paths ──────────────────────────────────────────────────────────────────
  private val home         = System.getProperty("user.home")
  private val dataDir      = new File(s"$home/Documents/100 plus scapula data/paired_scapulae_STLs_scapula")
  private val reviewFolder = new File(s"$home/Documents/100 plus scapula data/Five_Targets_Review_2026-08-25")

  // ── Reference landmarks (measured on paired_scapula_001_M_64_L) ───────────
  private val refLms: IndexedSeq[Landmark[_3D]] = IndexedSeq(
    Landmark("GC",  Point3D(  16.8,  -64.9, -176.3)),
    Landmark("TS",  Point3D( -13.9,  -88.3, -285.4)),
    Landmark("IA",  Point3D( 120.3,  -55.9, -285.1)),
    Landmark("PLA", Point3D( -15.4,  -25.7, -175.0)),
    Landmark("AC",  Point3D( -39.7,  -56.8, -159.6))
  )

  // ── Approximate per-target landmark positions ──────────────────────────────
  // Encoded as offsets from the reference landmarks.
  // Scalismo: Point + EuclideanVector → Point  (legal)
  //           Point + Point                    (illegal — does not compile)
  // Replace these offsets with landmarks measured on each target when available.
  private def targetLms(id: String): IndexedSeq[Landmark[_3D]] = {
    val shift: EuclideanVector[_3D] = id match {
      case _ if id.contains("002") => EuclideanVector3D(  5.0, -2.0,  1.0)
      case _ if id.contains("004") => EuclideanVector3D( -4.0,  3.0, -2.0)
      case _ if id.contains("007") => EuclideanVector3D(  2.0,  5.0,  3.0)
      case _ if id.contains("010") => EuclideanVector3D( -3.0, -4.0,  2.0)
      case _                       => EuclideanVector3D(  1.0,  1.0, -1.0)
    }
    refLms.map(l => Landmark(l.id, l.point + shift))
  }

  // ── Target specimen IDs (STL filenames without extension) ─────────────────
  private val targetIds = IndexedSeq(
    "paired_scapula_002_M_56_L",
    "paired_scapula_004_F_67_L",
    "paired_scapula_007_M_26_L",
    "paired_scapula_010_F_43_L",
    "paired_scapula_012_M_68_L"
  )

  // ── GP prior kernel ────────────────────────────────────────────────────────
  // GaussianKernel[_3D] and DiagonalKernel — the *3D-suffixed names do not exist.
  // GaussianProcess[_3D, EuclideanVector[_3D]] — GaussianProcess3D does not exist.
  private def buildGP(): GaussianProcess[_3D, EuclideanVector[_3D]] =
    GaussianProcess[_3D, EuclideanVector[_3D]](
      DiagonalKernel(GaussianKernel[_3D](75.0) * 12.0, 3))

  // ── Helpers ────────────────────────────────────────────────────────────────
  private def loadAndDecimate(f: File, maxPts: Int = 20000): TriangleMesh[_3D] = {
    val raw = MeshIO.readMesh(f).get
    if (raw.pointSet.numberOfPoints > maxPts) raw.operations.decimate(maxPts) else raw
  }

  // ── Main ───────────────────────────────────────────────────────────────────
  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(42L)

    reviewFolder.mkdirs()

    // Reference
    val refFile = new File(dataDir, "paired_scapula_001_M_64_L.stl")
    require(refFile.exists(), s"Reference not found: ${refFile.getPath}")
    val refMesh = loadAndDecimate(refFile)
    println(s"Reference mesh: ${refMesh.pointSet.numberOfPoints} vertices")

    // Shared GP prior (built once; approximation is per-target because the
    // domain (aligned reference) differs for each target)
    val gp = buildGP()

    val ui = ScalismoUI()

    for ((tId, idx) <- targetIds.zipWithIndex) {
      println(s"\n── [${idx + 1}/5]  $tId ──")

      val tFile = new File(dataDir, s"$tId.stl")
      if (!tFile.exists()) {
        println(s"  SKIP – not found: ${tFile.getPath}")
      } else {
        val targetMesh = loadAndDecimate(tFile)
        println(s"  Target : ${targetMesh.pointSet.numberOfPoints} vertices")

        // ── 1. Rigid Procrustes alignment ─────────────────────────────────
        // rigid3DLandmarkRegistration(moving, fixed, center) returns a
        // transform T such that T(movingLm.point) ≈ fixedLm.point.
        val rigidT     = LandmarkRegistration.rigid3DLandmarkRegistration(
                           refLms, targetLms(tId), center = Point3D(0, 0, 0))
        val alignedRef = refMesh.transform(rigidT)

        // ── 2. Low-rank GP approximation ──────────────────────────────────
        // approximateGPCholesky[D, DDomain[D] <: DiscreteDomain[D], Value]
        //   first arg : DDomain[D]
        //   last arg  : FieldInterpolator[D, DDomain, Value]
        //
        // TriangleMesh[_3D]  implements DiscreteDomain[_3D]  → pass alignedRef
        // alignedRef.pointSet returns UnstructuredPoints[_3D] which does NOT
        // implement DiscreteDomain[_3D] in Scalismo 0.92 → compile error E007.
        //
        // NearestNeighborInterpolator3D is FieldInterpolator[_3D, DiscreteDomain, _]
        // which is contravariant-compatible with FieldInterpolator[_3D, TriangleMesh, _].
        val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
          alignedRef,
          gp,
          relativeTolerance = 0.005,
          interpolator = NearestNeighborInterpolator3D()
        )
        println(s"  GP rank: ${lowRankGP.rank}")

        // ── 3. GP-ICP non-rigid registration ─────────────────────────────
        // PointDistributionModel[_3D, TriangleMesh] — the *3D suffix form does not exist.
        var model = PointDistributionModel[_3D, TriangleMesh](alignedRef, lowRankGP)

        // Uniform point-id stride so we sample evenly, independent of mesh order.
        val step  = math.max(1, alignedRef.pointSet.numberOfPoints / 2500)
        val ptIds = (0 until alignedRef.pointSet.numberOfPoints by step).map(PointId(_))

        for (iter <- 0 until 10) {
          val mean = model.mean
          // model.posterior takes IndexedSeq[(PointId, Point[_3D])], not a continuous GP call.
          val correspondences: IndexedSeq[(PointId, Point[_3D])] = ptIds.flatMap { pid =>
            val pt      = mean.pointSet.point(pid)
            val nearest = targetMesh.operations.closestPointOnSurface(pt).point
            // 15 mm threshold removes correspondences across non-overlapping structures.
            if ((nearest - pt).norm < 15.0) Some((pid, nearest)) else None
          }
          if (correspondences.nonEmpty)
            model = model.posterior(correspondences, sigma2 = 1.0)
          if (iter == 0 || iter == 9)
            println(s"  iter ${iter + 1}/10 : ${correspondences.size} correspondences active")
        }

        val registeredMesh = model.mean

        // ── 4. Save ───────────────────────────────────────────────────────
        val vtkOut = new File(reviewFolder, s"registered_$tId.vtk")
        val h5Out  = new File(reviewFolder, s"model_$tId.h5")
        MeshIO.writeMesh(registeredMesh, vtkOut).get
        StatisticalModelIO.writeStatisticalTriangleMeshModel3D(model, h5Out).get
        println(s"  → ${vtkOut.getName}")
        println(s"  → ${h5Out.getName}")

        // ── 5. Visualise ──────────────────────────────────────────────────
        val grp   = ui.createGroup(s"[${idx + 1}] $tId")
        val tView = ui.show(grp, targetMesh,     "Target")
        tView.color   = new Color(200, 200, 200)
        tView.opacity = 0.50f
        val rView = ui.show(grp, registeredMesh, "Registered")
        rView.color   = new Color(60, 160, 220)
        rView.opacity = 0.85f
        ui.show(grp, model, "GP_Model")
      }
    }

    println(s"\nDone. Review folder: ${reviewFolder.getAbsolutePath}")
  }
}
