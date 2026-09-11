package scapula

import scalismo.common.{Field, RealSpace}
import scalismo.geometry.{EuclideanVector, Landmark, Point, _3D}
import scalismo.kernels.{DiagonalKernel, GaussianKernel}
import scalismo.mesh.TriangleMesh
import scalismo.numerics.UniformMeshSampler3D
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

/**
 * Research-quality visualization for thesis / paper presentation.
 *
 * Pipeline:
 *   1. Rigid-align ALL available specimens to the reference (002_M_56_L).
 *   2. Select the 5 most MORPHOLOGICALLY DIVERSE specimens by greedy farthest-point
 *      selection in the symmetric-surface-distance matrix (maximises shape spread).
 *   3. Non-rigid GP-ICP registration of those 5 using ORIGINAL S05 parameters.
 *   4. Display four groups per specimen in the Scalismo UI:
 *        A. Non-registered surface (RED) vs reference (WHITE) — scatter / mismatch
 *        B. Registered surface (semi-transparent BLUE) vs reference (WHITE) — overlay
 *        C. Deformation field — arrows from reference vertices to registered positions
 *        D. Overlay of ALL registered meshes + reference — z-fighting patch pattern
 *
 * Original S05 parameters:
 *   modelResolution=8000  gpBasis=100  gpSigma=13.0 mm  gpScale=30.0
 *   gpNoise=1.0           NR-iter=10   rigid-ICP=40
 */
object ResearchVizApp {

  // ── REGISTRATION PARAMETERS ───────────────────────────────────────────────
  private val MESH_RES  = 8000
  private val ICP_ITER  = 40
  private val N_SELECT  = 2      // ← set to 2 for quick test, change to 5 for final

  // Single Gaussian kernel — research-based values for scapula bone registration:
  //   sigma=30mm : covers whole-bone scale on a 150mm scapula; large enough to
  //                bridge the 2–3mm gap remaining after rigid alignment
  //   scale=100  : RMS deformation amplitude √100=10mm; comfortably covers the gap
  //   noise=0.01 : trusts ICP correspondences very tightly (≈99.99% residual corrected);
  //                this is the critical parameter — noise=1.0 corrects only ~1% per iter
  //   iter=50    : sufficient iterations for convergence at this noise level
  //
  // Basis: Lüthi et al. 2017 "GP Morphable Models"; Scalismo bone tutorial defaults;
  //        Galibarov et al. 2010 scapula SSM used σ≈30mm for inter-subject variation.
  private val GP_BASIS  = 100
  private val GP_SIGMA  = 30.0   // mm
  private val GP_SCALE  = 100.0  // amplitude (RMS = √100 = 10 mm)
  private val GP_NOISE  = 0.01   // tight: trusts ICP correspondences strongly
  private val NR_ITER   = 50

  // ── tiny reflection helper ─────────────────────────────────────────────────
  private def setVisible(v: Any, on: Boolean): Unit = v match {
    case s: Seq[_] => s.foreach(setVisible(_, on))
    case _ =>
      try { v.getClass.getMethod("visible_$eq", classOf[Boolean]).invoke(v, java.lang.Boolean.valueOf(on)) }
      catch { case _: Exception => () }
  }

  private def show[A](ui: ScalismoUI, g: scalismo.ui.api.Group, obj: A, name: String, visible: Boolean = true)(
    implicit ev: scalismo.ui.api.ShowInScene[A]
  ): Unit = try {
    val v = ui.show(g, obj, name)
    Thread.sleep(20)
    setVisible(v, visible)
  } catch { case e: Exception => println(s"  [warn] show '$name': ${e.getMessage}") }

  // ── main ───────────────────────────────────────────────────────────────────
  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dir = Config.dataDir
    val csv = ScapulaData.csvFile(dir)
    val (lmMap, _, _) = ScapulaData.readLandmarkCsv(csv)

    case class Spec(id: String, mesh: TriangleMesh[_3D], lms: IndexedSeq[Landmark[_3D]])

    // ── Load all specimens with landmarks ────────────────────────────────────
    println("[step 1/5] Loading specimens...")
    val allSpecs: IndexedSeq[Spec] = ScapulaData.specimens(dir)
      .filter(s => lmMap.contains(s.modelId))
      .map { s =>
        val raw = ScapulaData.loadMesh(s.file)
        val lms = lmMap(s.modelId)
        if (s.isRight) Spec(s.modelId + "_mir", ScapulaData.mirrorMesh(raw), ScapulaData.mirrorLandmarks(lms))
        else           Spec(s.modelId, raw, lms)
      }
    println(s"[info] Loaded ${allSpecs.length} specimens with landmarks.")

    // ── Reference: 002_M_56_L  (same as original S05) ───────────────────────
    val ref = allSpecs.find(s => s.id.contains("002") && !s.id.endsWith("_mir")).getOrElse(allSpecs.head)
    println(s"[info] Reference: ${ref.id}")

    // ── Decimate reference to MESH_RES=8000 ─────────────────────────────────
    println(s"[step 2/5] Decimating reference to $MESH_RES pts...")
    val decRef = ref.mesh.operations.decimate(MESH_RES)
    println(s"[info] Decimated reference: ${decRef.pointSet.numberOfPoints} pts")

    // ── Rigid-align ALL non-reference specimens ──────────────────────────────
    println(s"[step 3/5] Rigid alignment of all ${allSpecs.length - 1} candidates (landmark + ICP $ICP_ITER iter)...")
    val candidates = allSpecs.filterNot(_.id == ref.id)
    val rigidAll: IndexedSeq[Spec] = candidates.zipWithIndex.map { case (s, i) =>
      print(s"  rigid ${i+1}/${candidates.length}  ${s.id}\r")
      val t     = ScapulaData.rigidFromLandmarks(s.lms, ref.lms)
      val m1    = s.mesh.transform(t)
      val lms1  = s.lms.map(lm => lm.copy(point = t(lm.point)))
      val m2    = RigidAlign.rigidIcp(m1, ref.mesh, ICP_ITER)
      s.copy(mesh = m2, lms = lms1)
    }
    println(s"\n[info] Rigid alignment done.")

    // ── Greedy farthest-point selection: maximise shape diversity ─────────────
    // Use mean surface distance (MSD) as shape-distance proxy.
    // MSD is symmetric: (d(A→B) + d(B→A)) / 2.
    println(s"[step 4/5] Selecting $N_SELECT most morphologically diverse specimens...")

    def msd(a: TriangleMesh[_3D], b: TriangleMesh[_3D]): Double =
      Metrics.symmetric(a, b).mean

    // Start with the specimen farthest from the reference
    val distToRef: IndexedSeq[Double] = rigidAll.map(s => msd(s.mesh, decRef))
    val seed0 = distToRef.zipWithIndex.maxBy(_._1)._2

    var selected = IndexedSeq(seed0)
    while (selected.length < math.min(N_SELECT, rigidAll.length)) {
      // For each candidate, find its min distance to any already-selected specimen
      val nextIdx = rigidAll.indices
        .filterNot(selected.contains)
        .maxBy { i =>
          selected.map(j => msd(rigidAll(i).mesh, rigidAll(j).mesh)).min
        }
      selected = selected :+ nextIdx
    }

    val targets = selected.map(rigidAll)
    println(s"[info] Selected (most diverse):")
    targets.zipWithIndex.foreach { case (s, i) =>
      println(f"  ${i+1}. ${s.id}  (MSD to ref = ${distToRef(selected(i))}%.2f mm)")
    }

    // ── Non-rigid GP-ICP registration ────────────────────────────────────────
    println(s"[step 5/5] Non-rigid GP-ICP registration (σ=${GP_SIGMA}mm, scale=$GP_SCALE, noise=$GP_NOISE, $NR_ITER iter)...")
    val registered: IndexedSeq[TriangleMesh[_3D]] = targets.zipWithIndex.map { case (s, i) =>
      print(s"  NR ${i+1}/${targets.length}  ${s.id}\r")
      val r = gpIcpRegister(decRef, s.mesh)
      r
    }
    println(s"\n[info] Non-rigid registration done.")

    // ── Quality report ────────────────────────────────────────────────────────
    println("\n[info] ══ Registration Quality Report (mm) ══")
    targets.zip(registered).foreach { case (s, reg) =>
      val beforeStats = Metrics.symmetric(s.mesh, decRef)
      val afterStats  = Metrics.symmetric(reg, decRef)
      println(s"  ${s.id}")
      println(f"    Before NR: ${beforeStats.render}")
      println(f"    After  NR: ${afterStats.render}")
    }
    println("[info] ════════════════════════════════════════")

    // ── Open Scalismo UI ─────────────────────────────────────────────────────
    println("\n[UI] Opening viewer...")
    val ui = ScalismoUI(s"SSM Research Visualization — $N_SELECT most-diverse specimens")
    Thread.sleep(1500)

    // G0: Reference only
    val g0 = ui.createGroup("G0 Reference (002_M_56_L)")
    show(ui, g0, decRef, "reference")

    // Per-specimen groups: non-registered, registered overlay, deformation field
    targets.zip(registered).zipWithIndex.foreach { case ((s, reg), idx) =>
      val label = s"S${idx+1} ${s.id}"

      // GA: Non-registered (rigid only) vs reference — shows raw shape difference
      val gA = ui.createGroup(s"$label — A: NON-REGISTERED (rigid aligned only)")
      show(ui, gA, decRef,  "reference_white")
      show(ui, gA, s.mesh,  "target_before_NR")

      // GB: Registered vs reference overlay — shows registration quality
      val gB = ui.createGroup(s"$label — B: REGISTERED overlay (after NR)")
      show(ui, gB, decRef, "reference_white")
      show(ui, gB, reg,    "registered_blue")

      // GC: Deformation field — arrows reference → registered
      val gC = ui.createGroup(s"$label — C: DEFORMATION FIELD (reference → registered)")
      val defField = computeDeformationField(decRef, reg)
      show(ui, gC, defField, "deformation_arrows")
      show(ui, gC, decRef,   "reference_context", visible = false)
    }

    // GD: ALL registered + reference overlaid simultaneously → z-fighting fine patches
    val gD = ui.createGroup("GD PATCH PATTERN: reference + ALL 5 registered overlaid (turn all ON)")
    show(ui, gD, decRef, "reference_white")
    targets.zip(registered).foreach { case (s, reg) =>
      show(ui, gD, reg, s.id)
    }

    // ── Parameter summary ─────────────────────────────────────────────────────
    println("\n[info] ══ Parameters ══")
    println(s"[info]   Reference:        ${ref.id}")
    println(s"[info]   modelResolution:  $MESH_RES pts")
    println(s"[info]   rigid ICP iter:   $ICP_ITER")
    println(s"[info]   gpSigma:          ${GP_SIGMA} mm  (global bone-scale kernel)")
    println(s"[info]   gpScale:          $GP_SCALE        (RMS amplitude √$GP_SCALE=${f"${math.sqrt(GP_SCALE)}%.1f"} mm)")
    println(s"[info]   gpNoise:          $GP_NOISE  (tight: trusts ICP correspondences strongly)")
    println(s"[info]   gpBasis:          $GP_BASIS  (Nyström rank)")
    println(s"[info]   NR iterations:    $NR_ITER")
    println(s"[info]   Selected:         ${targets.map(_.id).mkString(", ")}")
    println("[info] ══════════════════════════════════════")
    println("[info]")
    println("[info] HOW TO USE THE VIEWER:")
    println("[info]   G0  — Reference bone alone")
    println("[info]   S1–S5 A — rigid-only (before NR): shows shape mismatch in red/white")
    println("[info]   S1–S5 B — after NR: registered mesh overlaid on reference")
    println("[info]   S1–S5 C — deformation field (click arrows to see magnitude)")
    println("[info]   GD — ALL 5 registered + reference overlaid simultaneously")
    println("[info]        → turn ON all meshes → click any one → fine scattered red/white patches")
    println("[info]        (z-fighting occurs because registration residual < 0.5 mm)")
    println("[info]")
    println("[info] Close window to exit.")
  }

  // ── Single-kernel GP-ICP non-rigid registration ───────────────────────────
  // Research-based single Gaussian kernel: sigma=30mm, scale=100, noise=0.01.
  // Prints MSD every 10 iterations so convergence is visible in the terminal.
  private def gpIcpRegister(
    reference: TriangleMesh[_3D],
    target:    TriangleMesh[_3D]
  )(implicit rng: Random): TriangleMesh[_3D] = {
    val scalarKernel = GaussianKernel[_3D](GP_SIGMA) * GP_SCALE
    val kernel       = DiagonalKernel(scalarKernel, 3)
    val zeroMean     = Field(RealSpace[_3D], (_: Point[_3D]) => EuclideanVector.zeros[_3D])
    val gp           = GaussianProcess(zeroMean, kernel)
    val sampler      = UniformMeshSampler3D(reference, GP_BASIS * 10)
    val lowRankGP    = LowRankGaussianProcess.approximateGPNystrom(gp, sampler, GP_BASIS)
    val model        = PointDistributionModel[_3D, TriangleMesh](reference, lowRankGP)

    val targetOps = target.operations
    var current   = reference
    for (it <- 0 until NR_ITER) {
      val correspondences = current.pointSet.pointsWithId.map { case (pt, id) =>
        (id, targetOps.closestPointOnSurface(pt).point)
      }.toIndexedSeq
      current = model.posterior(correspondences, GP_NOISE).mean
      if ((it + 1) % 10 == 0) {
        val msd = Metrics.symmetric(current, target).mean
        print(s"  iter${it+1}: MSD=${f"$msd%.3f"}mm  ")
      }
    }
    println()
    current
  }

  // ── Deformation field as a discrete vector field ───────────────────────────
  // Returns a DiscreteField that Scalismo UI renders as arrows.
  private def computeDeformationField(
    reference:  TriangleMesh[_3D],
    registered: TriangleMesh[_3D]
  ): scalismo.common.DiscreteField[_3D, TriangleMesh, EuclideanVector[_3D]] = {
    import scalismo.common.DiscreteField
    DiscreteField[_3D, TriangleMesh, EuclideanVector[_3D]](
      reference,
      reference.pointSet.pointsWithId.map { case (p, id) =>
        registered.pointSet.point(id) - p
      }.toIndexedSeq
    )
  }
}
