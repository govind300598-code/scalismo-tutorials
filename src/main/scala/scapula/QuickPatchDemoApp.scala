package scapula

import scalismo.geometry.{Landmark, _3D}
import scalismo.mesh.TriangleMesh
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

/**
 * Reproduces the small scattered patch pattern from the original S05 screenshot.
 * Uses original parameters: modelResolution=8000, gpBasis=100, gpIcpIter=10
 * Reference = 002_M_56_L, Target = 002_M_56_R_mirrored (same person, mirrored)
 */
object QuickPatchDemoApp {

  // Original parameters that produced the old S05 screenshot
  private val MESH_RES  = 8000   // dense reference — key for fine patches
  private val GP_BASIS  = 100    // original gpBasis
  private val GP_SIGMA  = 13.0   // original gpSigma (mm)
  private val GP_SCALE  = 30.0   // original gpScale
  private val GP_NOISE  = 1.0    // original gpNoise
  private val NR_ITER   = 10     // original gpIcpIter

  private def hideReflect(v: Any): Unit = {
    if (v == null) return
    v match {
      case seq: Seq[_] => seq.foreach(hideReflect)
      case _ =>
        try { v.getClass.getMethod("visible_$eq", classOf[Boolean]).invoke(v, java.lang.Boolean.FALSE) }
        catch { case _: Exception => () }
    }
  }

  private def viz[A](ui: ScalismoUI, grp: scalismo.ui.api.Group, obj: A, name: String)(
    implicit ev: scalismo.ui.api.ShowInScene[A]
  ): Unit = try {
    val v = ui.show(grp, obj, name)
    Thread.sleep(10)
    hideReflect(v)
  } catch { case e: Exception => println(s"  [warn] '$name': ${e.getMessage}") }

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dir = Config.dataDir
    val csv = ScapulaData.csvFile(dir)
    val (lmMap, _, _) = ScapulaData.readLandmarkCsv(csv)

    case class Spec(id: String, mesh: TriangleMesh[_3D], lms: IndexedSeq[Landmark[_3D]])

    val all: IndexedSeq[Spec] = ScapulaData.specimens(dir)
      .filter(s => lmMap.contains(s.modelId))
      .map { s =>
        val raw = ScapulaData.loadMesh(s.file)
        val lms = lmMap(s.modelId)
        if (s.isRight) Spec(s.modelId + "_mir", ScapulaData.mirrorMesh(raw), ScapulaData.mirrorLandmarks(lms))
        else           Spec(s.modelId, raw, lms)
      }

    // Reference = 002_M_56_L — same as original S05 screenshot
    val ref = all.find(s => s.id.contains("002") && !s.id.endsWith("_mir")).getOrElse(all.head)
    println(s"[info] Reference: ${ref.id}")

    // Target = 002_M_56_R_mirrored — same person, mirrored (shown selected in old screenshot)
    val target = all.find(s => s.id.contains("002") && s.id.endsWith("_mir")).toSeq.toIndexedSeq
    println(s"[info] Target: ${target.map(_.id).mkString(", ")}")

    // Rigid alignment
    println("[step] Rigid alignment...")
    val rigidAligned = target.map { s =>
      val t  = ScapulaData.rigidFromLandmarks(s.lms, ref.lms)
      val m1 = s.mesh.transform(t)
      val lms1 = s.lms.map(lm => lm.copy(point = t(lm.point)))
      val m2 = RigidAlign.rigidIcp(m1, ref.mesh, 40)   // original icpIterations=40
      s.copy(mesh = m2, lms = lms1)
    }
    println("[step] Rigid alignment done.")

    // Decimate reference to original 8000 pts
    val decRef = ref.mesh.operations.decimate(MESH_RES)
    println(s"[info] Decimated reference: ${decRef.pointSet.numberOfPoints} pts (target $MESH_RES)")

    // NR with original parameters
    println(s"[step] Non-rigid registration ($NR_ITER iterations, gpBasis=$GP_BASIS, sigma=${GP_SIGMA}mm)...")
    val registered = rigidAligned.zipWithIndex.map { case (s, i) =>
      print(s"  NR ${i+1}/${rigidAligned.length}  ${s.id}\r")
      val result = registerOriginal(decRef, s.mesh)
      result
    }
    println(s"\n[step] Done.")

    // Open viewer
    println("[UI] Opening viewer...")
    val ui = ScalismoUI(s"Patch Demo — 002 ref + 002_R_mir (original params)")
    Thread.sleep(1500)

    // G0: Reference alone
    val g0 = ui.createGroup("G0 Reference (002_M_56_L)")
    viz(ui, g0, decRef, "reference")

    // G1: The patch pattern — reference (white) + registered (red) overlaid
    val g1 = ui.createGroup(s"G1 PATCHES: ref(002_L white) + registered(${rigidAligned.head.id} red)")
    viz(ui, g1, decRef, "reference_white")
    viz(ui, g1, registered.head, rigidAligned.head.id + "_red")

    // G2: Before vs after NR
    val g2 = ui.createGroup(s"G2 Before/After NR: ${rigidAligned.head.id}")
    viz(ui, g2, rigidAligned.head.mesh, "before_NR")
    viz(ui, g2, registered.head,        "after_NR")
    viz(ui, g2, decRef,                 "reference")

    // G3: Target rigid-aligned alone
    val g3 = ui.createGroup(s"G3 Target rigid-aligned: ${rigidAligned.head.id}")
    viz(ui, g3, rigidAligned.head.mesh, rigidAligned.head.id)

    println("\n[info] Parameters used (original S05 values):")
    println(s"[info]   gpSigma=$GP_SIGMA mm  gpScale=$GP_SCALE  gpBasis=$GP_BASIS")
    println(s"[info]   gpNoise=$GP_NOISE  NR iterations=$NR_ITER  meshRes=$MESH_RES")
    println(s"[info]   Reference: ${ref.id}  Target: ${rigidAligned.head.id}")
    println("[info] Groups:")
    println("[info]  G0  Reference alone")
    println(s"[info]  G1  PATCHES — turn this on to see the scattered patch pattern")
    println("[info]  G2  Before/After NR")
    println("[info]  G3  Target rigid-aligned")
    println("[info] Close window to exit.")
  }

  // Registration using exact original S05 parameters
  private def registerOriginal(
    reference: TriangleMesh[_3D],
    target:    TriangleMesh[_3D]
  )(implicit rng: Random): TriangleMesh[_3D] = {
    import scalismo.common.{Field, RealSpace}
    import scalismo.geometry.{EuclideanVector, Point}
    import scalismo.kernels.{DiagonalKernel, GaussianKernel}
    import scalismo.numerics.UniformMeshSampler3D
    import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel}

    val scalarKernel = GaussianKernel[_3D](GP_SIGMA) * GP_SCALE
    val kernel       = DiagonalKernel(scalarKernel, 3)
    val zeroMean     = Field(RealSpace[_3D], (_: Point[_3D]) => EuclideanVector.zeros[_3D])
    val gp           = GaussianProcess(zeroMean, kernel)
    val sampler      = UniformMeshSampler3D(reference, GP_BASIS * 10)
    val lowRankGP    = LowRankGaussianProcess.approximateGPNystrom(gp, sampler, GP_BASIS)
    val model        = PointDistributionModel[_3D, TriangleMesh](reference, lowRankGP)

    val targetOps = target.operations
    var current   = reference
    for (_ <- 0 until NR_ITER) {
      val correspondences = current.pointSet.pointsWithId.map { case (pt, id) =>
        (id, targetOps.closestPointOnSurface(pt).point)
      }.toIndexedSeq
      current = model.posterior(correspondences, GP_NOISE).mean
    }
    current
  }
}
