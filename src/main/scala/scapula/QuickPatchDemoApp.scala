package scapula

import scalismo.geometry.{Landmark, _3D}
import scalismo.mesh.TriangleMesh
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

/**
 * Quick 5-specimen patch demo — runs in ~3-5 minutes.
 * Registers 5 specimens with reduced iterations so you can see
 * the patch / Z-fighting pattern in the viewer quickly.
 */
object QuickPatchDemoApp {

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

    // Reference = 001_M_64_L  (matches old S05 screenshot exactly)
    val ref = all.find(s => s.id.contains("001") && !s.id.endsWith("_mir")).getOrElse(all.head)
    println(s"[info] Reference: ${ref.id}")

    // Target = 002_M_56_R mirrored — large cross-subject shape difference → small patch artifacts
    val N = 1
    val target002Rmir = all.find(s => s.id.contains("002") && s.id.endsWith("_mir"))
    val targets = target002Rmir.toSeq.take(1).toIndexedSeq
    println(s"[info] Targets: ${targets.map(_.id).mkString(", ")}")

    // Rigid alignment
    println("[step] Rigid alignment...")
    val rigidAligned = targets.map { s =>
      val t = ScapulaData.rigidFromLandmarks(s.lms, ref.lms)
      val m1 = s.mesh.transform(t)
      val lms1 = s.lms.map(lm => lm.copy(point = t(lm.point)))
      val m2 = RigidAlign.rigidIcp(m1, ref.mesh, Config.icpIterations)
      s.copy(mesh = m2, lms = lms1)
    }
    println("[step] Rigid alignment done.")

    // Decimate reference
    val decRef = ref.mesh.operations.decimate(Config.modelResolution)
    println(s"[info] Decimated reference: ${decRef.pointSet.numberOfPoints} pts")

    // NR with 40 iterations — tight convergence for small patch artifacts
    println(s"[step] Non-rigid registration (${N} specimen, 40 iterations)...")
    val registered = rigidAligned.zipWithIndex.map { case (s, i) =>
      print(s"  NR ${i+1}/$N  ${s.id}\r")
      val result = registerFast(decRef, s.mesh)
      result
    }
    println(s"\n[step] Done.")

    // Open viewer
    println("[UI] Opening viewer...")
    val ui = ScalismoUI("Patch Demo — 001 ref + 002_R_mir registered")
    Thread.sleep(1500)

    // G0: Reference alone
    val g0 = ui.createGroup("G0 Reference")
    viz(ui, g0, decRef, "reference")

    // G1: Registered overlaid on reference → scattered patch artifacts
    val g1 = ui.createGroup(s"G1 Patches: ref(001_L white) + registered(${rigidAligned.head.id} red)")
    viz(ui, g1, decRef, "reference_001_L_white")
    viz(ui, g1, registered.head, rigidAligned.head.id + "_registered_red")

    // G2: Before vs after NR
    val g2 = ui.createGroup(s"G2 Before/After NR: ${rigidAligned.head.id}")
    viz(ui, g2, rigidAligned.head.mesh, "before_NR_rigid_aligned")
    viz(ui, g2, registered.head,        "after_NR_registered")
    viz(ui, g2, decRef,                 "reference_001_L")

    // G3: Target alone (rigid aligned, before NR)
    val g3 = ui.createGroup(s"G3 Target rigid-aligned (before NR): ${rigidAligned.head.id}")
    viz(ui, g3, rigidAligned.head.mesh, rigidAligned.head.id)

    println("\n[info] Groups:")
    println("[info]  G0  Reference (001_M_64_L) alone")
    println(s"[info]  G1  PATCHES: ref(001_L white) + registered(${rigidAligned.head.id} red)")
    println("[info]       → This is the small scattered patch pattern from old S05 screenshot")
    println("[info]  G2  Before/After NR for same specimen")
    println("[info]  G3  Target rigid-aligned alone")
    println("[info] Close window to exit.")
  }

  // NR with 3 iterations (fast demo)
  private def registerFast(
    reference: scalismo.mesh.TriangleMesh[_3D],
    target:    scalismo.mesh.TriangleMesh[_3D]
  )(implicit rng: Random): scalismo.mesh.TriangleMesh[_3D] = {
    import scalismo.common.{Field, RealSpace}
    import scalismo.geometry.{EuclideanVector, Point}
    import scalismo.kernels.{DiagonalKernel, GaussianKernel}
    import scalismo.numerics.UniformMeshSampler3D
    import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel}

    val scalarKernel = GaussianKernel[_3D](Config.gpSigma) * Config.gpScale
    val kernel       = DiagonalKernel(scalarKernel, 3)
    val zeroMean     = Field(RealSpace[_3D], (_: Point[_3D]) => EuclideanVector.zeros[_3D])
    val gp           = GaussianProcess(zeroMean, kernel)
    val sampler      = UniformMeshSampler3D(reference, Config.gpBasis * 10)
    val lowRankGP    = LowRankGaussianProcess.approximateGPNystrom(gp, sampler, Config.gpBasis)
    val model        = PointDistributionModel[_3D, scalismo.mesh.TriangleMesh](reference, lowRankGP)

    val targetOps = target.operations
    var current   = reference
    for (_ <- 0 until 40) {   // 40 iterations — tight convergence for dense Z-fighting patches
      val correspondences = current.pointSet.pointsWithId.map { case (pt, id) =>
        (id, targetOps.closestPointOnSurface(pt).point)
      }.toIndexedSeq
      current = model.posterior(correspondences, Config.gpNoise).mean
    }
    current
  }
}
