package scapula

import breeze.linalg.DenseVector
import scalismo.geometry._3D
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

/**
 * ResearchVizApp — reproduces the September 7 fine-patch Z-fighting result.
 *
 * Key parameters (hardcoded to match the original run):
 *   N_SELECT = 5    — only 5 randomly chosen specimens (seed=42)
 *   MESH_RES = 8000
 *   GP_SIGMA = 13.0, GP_SCALE = 30.0, GP_NOISE = 1.0
 *   NR_ITER  = 10, GP_BASIS = 100
 *
 * Group GD shows: reference_white + all 5 registered meshes overlaid.
 * Z-fighting (fine scattered red/white patches) appears there when
 * registration residual is <0.5 mm.
 *
 * Run with:
 *   sbt "runMain scapula.ResearchVizApp"
 */
object ResearchVizApp {

  // ── Hardcoded to match September 7 original run ────────────────────────────
  private val MESH_RES  = 8000
  private val GP_BASIS  = 100
  private val GP_SIGMA  = 13.0
  private val GP_SCALE  = 30.0
  private val GP_NOISE  = 1.0
  private val NR_ITER   = 10
  private val N_SELECT  = 5

  private def show[A](ui: ScalismoUI, grp: scalismo.ui.api.Group, obj: A, name: String)(
    implicit ev: scalismo.ui.api.ShowInScene[A]
  ): Unit = try {
    ui.show(grp, obj, name)
    Thread.sleep(20)
  } catch { case e: Exception => println(s"  [warn] display '$name': ${e.getMessage}") }

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(42L)   // fixed seed — same as September 7

    val dir = Config.dataDir
    val csv = ScapulaData.csvFile(dir)
    val (lmMap, fromHeader, _) = ScapulaData.readLandmarkCsv(csv)
    if (!fromHeader) println("[warn] landmark columns resolved by fallback offsets")

    case class Spec(id: String, mesh: TriangleMesh[_3D], lms: IndexedSeq[scalismo.geometry.Landmark[_3D]])

    // ── 1. Load ALL specimens ─────────────────────────────────────────────────
    val allSpecimens: IndexedSeq[Spec] = ScapulaData.specimens(dir)
      .filter(s => lmMap.contains(s.modelId))
      .map { s =>
        val raw = ScapulaData.loadMesh(s.file)
        val lms = lmMap(s.modelId)
        if (s.isRight) Spec(s.modelId + "_mir", ScapulaData.mirrorMesh(raw), ScapulaData.mirrorLandmarks(lms))
        else           Spec(s.modelId, raw, lms)
      }
    println(s"[info] ${allSpecimens.length} specimens loaded")

    // ── 2. Reference = specimen 002 left ──────────────────────────────────────
    val ref = allSpecimens.find(s => s.id.contains("002") && !s.id.endsWith("_mir"))
                          .getOrElse(allSpecimens.head)
    println(s"[info] Reference: ${ref.id}")

    // ── 3. Rigid alignment (Procrustes + ICP) ─────────────────────────────────
    println("[S03] Landmark Procrustes alignment")
    val lmAligned = allSpecimens.map { s =>
      val t = ScapulaData.rigidFromLandmarks(s.lms, ref.lms)
      s.copy(mesh = s.mesh.transform(t), lms = s.lms.map(lm => lm.copy(point = t(lm.point))))
    }

    println("[S04] Rigid ICP refinement")
    val rigidAligned = lmAligned.zipWithIndex.map { case (s, i) =>
      println(s"  ICP ${i + 1}/${lmAligned.length}  ${s.id}")
      s.copy(mesh = RigidAlign.rigidIcp(s.mesh, ref.mesh, Config.icpIterations))
    }
    println(s"[info] ${rigidAligned.length} rigid-aligned")

    // ── 4. Decimate reference ─────────────────────────────────────────────────
    val decRef = ref.mesh.operations.decimate(MESH_RES)
    println(s"[info] Decimated reference: ${ref.mesh.pointSet.numberOfPoints} → ${decRef.pointSet.numberOfPoints} pts")

    // ── 5. Select N_SELECT=5 targets (random, seed=42) ────────────────────────
    val others  = rigidAligned.filterNot(_.id == ref.id)
    val targets = scala.util.Random.javaRandomToRandom(
        new java.util.Random(42L)
      ).shuffle(others.toList).take(N_SELECT).toIndexedSeq
    println(s"[info] Selected ${targets.length} targets: ${targets.map(_.id).mkString(", ")}")

    // ── 6. Non-rigid GP-ICP registration of the 5 targets ────────────────────
    println(s"[S05] Non-rigid GP-ICP registration of $N_SELECT specimens")

    // Override Config values with hardcoded Sept-7 parameters via inline kernel
    import scalismo.common.{Field, RealSpace}
    import scalismo.common.interpolation.NearestNeighborInterpolator3D
    import scalismo.geometry.{EuclideanVector, Point}
    import scalismo.kernels.{DiagonalKernel, GaussianKernel}
    import scalismo.numerics.UniformMeshSampler3D
    import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess}

    def registerFixed(reference: TriangleMesh[_3D], target: TriangleMesh[_3D]): TriangleMesh[_3D] = {
      val scalarKernel = GaussianKernel[_3D](GP_SIGMA) * GP_SCALE
      val kernel       = DiagonalKernel(scalarKernel, 3)
      val zeroMean     = Field(RealSpace[_3D], (_: Point[_3D]) => EuclideanVector.zeros[_3D])
      val gp           = GaussianProcess(zeroMean, kernel)
      val sampler      = UniformMeshSampler3D(reference, GP_BASIS * 10)
      val lowRankGP    = LowRankGaussianProcess.approximateGPNystrom(gp, sampler, GP_BASIS)
      val model        = PointDistributionModel[_3D, TriangleMesh](reference, lowRankGP)
      val targetOps    = target.operations
      var current      = reference
      for (_ <- 0 until NR_ITER) {
        val correspondences = current.pointSet.pointsWithId.map { case (pt, id) =>
          (id, targetOps.closestPointOnSurface(pt).point)
        }.toIndexedSeq
        current = model.posterior(correspondences, GP_NOISE).mean
      }
      current
    }

    val cacheTag = s"research_n${N_SELECT}_s42"
    val registered: IndexedSeq[TriangleMesh[_3D]] =
      SSMBuilder.loadMeshes(cacheTag).getOrElse {
        val r = targets.zipWithIndex.map { case (s, i) =>
          println(s"  NR ${i + 1}/$N_SELECT  ${s.id}")
          registerFixed(decRef, s.mesh)
        }
        SSMBuilder.saveMeshes(r, cacheTag)
        r
      }
    if (SSMBuilder.loadMeshes(cacheTag).isDefined && registered.nonEmpty)
      println(s"  [cache] loaded from $cacheTag")
    println(s"[info] ${registered.length} non-rigid registered")

    // ── 7. Open viewer ────────────────────────────────────────────────────────
    println("[UI] Opening Scalismo viewer...")
    val ui = ScalismoUI("ResearchVizApp — Sept-7 Fine Patch Reproduction")
    Thread.sleep(2000)

    // G01 — all landmarks
    val g01 = ui.createGroup("G01_All_Landmarks")
    allSpecimens.foreach(s => show(ui, g01, s.lms.toList, s.id))

    // G02 — reference
    val g02 = ui.createGroup("G02_Reference")
    show(ui, g02, ref.lms.toList, "landmarks")
    show(ui, g02, ref.mesh, ref.id)

    // G03 — landmark-aligned (all)
    val g03 = ui.createGroup("G03_LandmarkAligned")
    lmAligned.foreach(s => show(ui, g03, s.mesh, s.id))

    // G04 — rigid-aligned (all)
    val g04 = ui.createGroup("G04_RigidAligned (all bones should overlap)")
    rigidAligned.foreach(s => show(ui, g04, s.mesh, s.id))

    // G05 — decimated reference
    val g05 = ui.createGroup("G05_DecimatedReference")
    show(ui, g05, decRef, "decRef")

    // G06 — selected 5 targets (rigid-aligned)
    val g06 = ui.createGroup(s"G06_Selected_${N_SELECT}_Targets (rigid-aligned)")
    targets.foreach(s => show(ui, g06, s.mesh, s.id))

    // G07 — selected 5 registered meshes only
    val g07 = ui.createGroup(s"G07_Registered_${N_SELECT}_only")
    registered.zipWithIndex.foreach { case (m, i) => show(ui, g07, m, s"registered_$i") }

    // G08 — reference alone (for overlay comparison)
    val g08 = ui.createGroup("G08_Reference_for_overlay")
    show(ui, g08, decRef, "reference_white")

    // G09 — registered_0 vs target_0 (individual pair check)
    val g09 = ui.createGroup("G09_Pair0: target vs registered")
    if (targets.nonEmpty) show(ui, g09, targets(0).mesh, "target_0")
    if (registered.nonEmpty) show(ui, g09, registered(0), "registered_0")

    // G10 — registered_1 vs target_1
    val g10 = ui.createGroup("G10_Pair1: target vs registered")
    if (targets.length > 1) show(ui, g10, targets(1).mesh, "target_1")
    if (registered.length > 1) show(ui, g10, registered(1), "registered_1")

    // G11 — reference alone in its own group
    val g11 = ui.createGroup("G11_Reference_alone")
    show(ui, g11, decRef, "ref")

    // G12 — registered_0 alone
    val g12 = ui.createGroup("G12_Registered_0_alone")
    if (registered.nonEmpty) show(ui, g12, registered(0), "reg0")

    // G13 — registered_1 alone
    val g13 = ui.createGroup("G13_Registered_1_alone")
    if (registered.length > 1) show(ui, g13, registered(1), "reg1")

    // G14 — registered_2 alone
    val g14 = ui.createGroup("G14_Registered_2_alone")
    if (registered.length > 2) show(ui, g14, registered(2), "reg2")

    // G15 — registered_3 alone
    val g15 = ui.createGroup("G15_Registered_3_alone")
    if (registered.length > 3) show(ui, g15, registered(3), "reg3")

    // G16 — registered_4 alone
    val g16 = ui.createGroup("G16_Registered_4_alone")
    if (registered.length > 4) show(ui, g16, registered(4), "reg4")

    // G17 — reference + registered_0 pair (fine patches appear here per pair)
    val g17 = ui.createGroup("G17_Ref+Reg0_overlap (Z-fighting pair)")
    show(ui, g17, decRef, "reference_white")
    if (registered.nonEmpty) show(ui, g17, registered(0), "registered_0")

    // G18 — Z-fighting Demo: reference + registered_1
    val g18 = ui.createGroup("G18_Ref+Reg1_overlap (Z-fighting pair)")
    show(ui, g18, decRef, "reference_white")
    if (registered.length > 1) show(ui, g18, registered(1), "registered_1")

    // GD — THE FINE PATCH GROUP: reference + ALL 5 registered overlaid
    val gD = ui.createGroup("GD PATCH PATTERN: reference + ALL 5 registered overlaid (turn all ON)")
    show(ui, gD, decRef, "reference_white")
    targets.zip(registered).foreach { case (s, reg) => show(ui, gD, reg, s.id) }

    println("")
    println("=" * 70)
    println("  LOOK AT GROUP: GD PATCH PATTERN")
    println("  Turn ALL meshes ON in that group.")
    println("  Fine scattered red/white patches = Z-fighting (<0.5mm residual)")
    println("  These are the exact September 7 patches.")
    println("=" * 70)
    println("[info] Done. Close the viewer window to exit.")
  }
}
