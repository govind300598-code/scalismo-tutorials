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

    // Reference = 002_M_56_L
    val ref = all.find(s => s.id.contains("002") && !s.id.endsWith("_mir")).getOrElse(all.head)
    println(s"[info] Reference: ${ref.id}")

    // Pick 2 specimens: same-person mirror first (best match → most Z-fighting patches)
    val N = 2
    val samePersonMirror = all.find(s => s.id.contains("002") && s.id.endsWith("_mir"))
    val others = all.filterNot(s => s.id == ref.id || s.id == samePersonMirror.map(_.id).getOrElse(""))
    val targets = (samePersonMirror.toSeq ++ others).take(N).toIndexedSeq
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

    // NR with 40 iterations — tight convergence for dense Z-fighting patches
    println(s"[step] Non-rigid registration (${N} specimens, 40 iterations each)...")
    val registered = rigidAligned.zipWithIndex.map { case (s, i) =>
      print(s"  NR ${i+1}/$N  ${s.id}\r")
      // Override iterations to 3 for speed
      val result = registerFast(decRef, s.mesh)
      result
    }
    println(s"\n[step] Done.")

    // Open viewer
    println("[UI] Opening viewer...")
    val ui = ScalismoUI("Patch Demo — 5 specimens")
    Thread.sleep(1500)

    // G0: Reference alone
    val g0 = ui.createGroup("G0 Reference")
    viz(ui, g0, decRef, "reference")

    // G1: All 5 registered overlaid on reference → Z-fighting / patches
    val g1 = ui.createGroup("G1 Reference + 5 registered OVERLAID (patches / Z-fighting)")
    viz(ui, g1, decRef, "reference_white")
    registered.zip(rigidAligned).foreach { case (m, s) =>
      viz(ui, g1, m, s.id)
    }

    // G2: Best match overlaid (maximum Z-fighting)
    val bestIdx = registered.indices.minBy { i =>
      val d = Metrics.surfaceDistances(registered(i), decRef)
      d.sum / d.length
    }
    val g2 = ui.createGroup(s"G2 Best match overlaid: ${rigidAligned(bestIdx).id}")
    viz(ui, g2, decRef, "reference")
    viz(ui, g2, registered(bestIdx), "registered_best")

    // G3: Before vs after NR for first specimen
    val g3 = ui.createGroup(s"G3 Before/After NR: ${rigidAligned.head.id}")
    viz(ui, g3, rigidAligned.head.mesh, "before_NR")
    viz(ui, g3, registered.head, "after_NR")
    viz(ui, g3, decRef, "reference")

    println("\n[info] Groups:")
    println("[info]  G0  Reference alone")
    println("[info]  G1  All 5 registered + reference OVERLAID → turn this on for patches")
    println("[info]  G2  Best-match specimen overlaid → maximum Z-fighting patches")
    println("[info]  G3  One specimen before/after NR")
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
