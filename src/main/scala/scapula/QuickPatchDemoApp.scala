package scapula

import scalismo.geometry.{Landmark, _3D}
import scalismo.mesh.TriangleMesh
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

/**
 * Reproduces the small scattered patch pattern from the original S05 screenshot.
 * KEY INSIGHT: fine patches come from MULTIPLE meshes (5+) all overlaid simultaneously.
 * With only 2 meshes you always get large blobs — you need many surfaces competing
 * for the z-buffer at once to produce the fine mosaic pattern.
 *
 * Original parameters: modelResolution=8000, gpBasis=100, gpIcpIter=10, icpIter=40
 * Reference = 002_M_56_L, Targets = first 5 available specimens
 */
object QuickPatchDemoApp {

  // Original S05 parameters
  private val MESH_RES = 8000
  private val GP_BASIS = 100
  private val GP_SIGMA = 13.0
  private val GP_SCALE = 30.0
  private val GP_NOISE = 1.0
  private val NR_ITER  = 10
  private val N        = 5    // need 5+ specimens to get fine scattered patches

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

    // Reference = 002_M_56_L  (same as original S05)
    val ref = all.find(s => s.id.contains("002") && !s.id.endsWith("_mir")).getOrElse(all.head)
    println(s"[info] Reference: ${ref.id}")

    // Take 5 specimens (excluding reference) — need multiple to get fine patches
    val targets = all.filterNot(_.id == ref.id).take(N)
    println(s"[info] Targets (${targets.length}): ${targets.map(_.id).mkString(", ")}")

    // Rigid alignment
    println("[step] Rigid alignment (Procrustes + ICP)...")
    val rigidAligned = targets.zipWithIndex.map { case (s, i) =>
      print(s"  ICP ${i+1}/$N  ${s.id}\r")
      val t    = ScapulaData.rigidFromLandmarks(s.lms, ref.lms)
      val m1   = s.mesh.transform(t)
      val lms1 = s.lms.map(lm => lm.copy(point = t(lm.point)))
      val m2   = RigidAlign.rigidIcp(m1, ref.mesh, 40)
      s.copy(mesh = m2, lms = lms1)
    }
    println(s"\n[step] Rigid alignment done.")

    // Dense reference (8000 pts — original value)
    val decRef = ref.mesh.operations.decimate(MESH_RES)
    println(s"[info] Decimated reference: ${decRef.pointSet.numberOfPoints} pts")

    // Non-rigid registration with original parameters
    println(s"[step] Non-rigid registration ($N specimens, $NR_ITER iterations each)...")
    val registered: IndexedSeq[TriangleMesh[_3D]] = rigidAligned.zipWithIndex.map { case (s, i) =>
      print(s"  NR ${i+1}/$N  ${s.id}\r")
      val result = registerOriginal(decRef, s.mesh)
      result
    }
    println(s"\n[step] Done.")

    // Open viewer
    println("[UI] Opening viewer...")
    val ui = ScalismoUI("Patch Demo — 5 specimens overlaid (original params)")
    Thread.sleep(1500)

    // G0: Reference alone
    val g0 = ui.createGroup("G0 Reference (002_M_56_L)")
    viz(ui, g0, decRef, "reference")

    // G1: ALL 5 registered + reference overlaid → fine scattered patches
    // Click any mesh in the left panel to highlight it red → see fine patches against white
    val g1 = ui.createGroup("G1 PATCHES: reference + all 5 registered overlaid — click a mesh to see patches")
    viz(ui, g1, decRef, "reference_white")
    registered.zip(rigidAligned).foreach { case (m, s) =>
      viz(ui, g1, m, s.id)
    }

    // G2: Before vs After NR for one specimen (002_R_mir or first available)
    val demoIdx = rigidAligned.indexWhere(s => s.id.contains("002") && s.id.endsWith("_mir"))
                              .max(0)
    val g2 = ui.createGroup(s"G2 Before/After NR: ${rigidAligned(demoIdx).id}")
    viz(ui, g2, rigidAligned(demoIdx).mesh, "before_NR")
    viz(ui, g2, registered(demoIdx),        "after_NR")
    viz(ui, g2, decRef,                     "reference")

    // G3: All rigid-aligned (before NR)
    val g3 = ui.createGroup("G3 All rigid-aligned (before NR)")
    rigidAligned.foreach(s => viz(ui, g3, s.mesh, s.id))

    println("\n[info] Parameters (original S05):")
    println(s"[info]   meshRes=$MESH_RES  gpBasis=$GP_BASIS  sigma=${GP_SIGMA}mm  scale=$GP_SCALE")
    println(s"[info]   gpNoise=$GP_NOISE  NR iter=$NR_ITER  rigid ICP=40")
    println(s"[info]   Reference: ${ref.id}")
    println(s"[info]   Targets: ${rigidAligned.map(_.id).mkString(", ")}")
    println("[info] ─────────────────────────────────────────────────────")
    println("[info] HOW TO SEE THE PATCHES:")
    println("[info]   1. Click G1 to expand it")
    println("[info]   2. Turn ON all meshes (eye icon next to each)")
    println("[info]   3. Click on any specimen name in the left panel")
    println("[info]      → it turns RED against the white meshes")
    println("[info]      → fine scattered patch pattern appears")
    println("[info] ─────────────────────────────────────────────────────")
    println("[info] Close window to exit.")
  }

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
