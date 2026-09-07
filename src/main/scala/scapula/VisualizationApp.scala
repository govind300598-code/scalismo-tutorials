package scapula

import breeze.linalg.DenseVector
import scalismo.common.{DiscreteField, PointId}
import scalismo.geometry.{EuclideanVector, Landmark, _3D}
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

object VisualizationApp {

  // Every ui.show wrapped — VTK errors are non-fatal
  private def viz[A](ui: ScalismoUI, grp: scalismo.ui.api.Group, obj: A, name: String)(
    implicit ev: scalismo.ui.api.ShowInScene[A]
  ): Unit = try { ui.show(grp, obj, name); Thread.sleep(15) }
  catch { case e: Exception => println(s"  [warn] '$name': ${e.getMessage}") }

  // Deformation field: pointwise displacement from reference to warped mesh
  private def defField(ref: TriangleMesh[_3D], warped: TriangleMesh[_3D])
  : DiscreteField[_3D, TriangleMesh, EuclideanVector[_3D]] =
    DiscreteField[_3D, TriangleMesh, EuclideanVector[_3D]](
      ref,
      ref.pointSet.pointsWithId.map { case (p, id) =>
        warped.pointSet.point(id) - p
      }.toIndexedSeq
    )

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    // ── 1. Load ───────────────────────────────────────────────────────────────
    val dir = Config.dataDir
    val csv = ScapulaData.csvFile(dir)
    val (lmMap, fromHeader, _) = ScapulaData.readLandmarkCsv(csv)
    if (!fromHeader) println("[warn] landmark columns resolved by fallback offsets")

    case class Spec(id: String, mesh: TriangleMesh[_3D], lms: IndexedSeq[Landmark[_3D]])

    val specimens: IndexedSeq[Spec] = ScapulaData.specimens(dir)
      .filter(s => lmMap.contains(s.modelId))
      .map { s =>
        val raw = ScapulaData.loadMesh(s.file)
        val lms = lmMap(s.modelId)
        if (s.isRight) Spec(s.modelId + "_mir", ScapulaData.mirrorMesh(raw), ScapulaData.mirrorLandmarks(lms))
        else           Spec(s.modelId, raw, lms)
      }
    println(s"[info] ${specimens.length} specimens loaded")

    // ── 2. Reference ──────────────────────────────────────────────────────────
    val ref = specimens.find(s => s.id.contains("002") && !s.id.endsWith("_mir"))
                       .getOrElse(specimens.head)
    println(s"[info] Reference: ${ref.id}")

    // ── 3. Rigid alignment ────────────────────────────────────────────────────
    println("[S03] Landmark Procrustes")
    val lmAligned = specimens.map { s =>
      val t = ScapulaData.rigidFromLandmarks(s.lms, ref.lms)
      s.copy(mesh = s.mesh.transform(t), lms = s.lms.map(lm => lm.copy(point = t(lm.point))))
    }
    println("[S04] Rigid ICP")
    val rigidAligned = lmAligned.zipWithIndex.map { case (s, i) =>
      print(s"  ICP ${i+1}/${lmAligned.length}\r")
      s.copy(mesh = RigidAlign.rigidIcp(s.mesh, ref.mesh, Config.icpIterations))
    }
    println(s"\n[info] ${rigidAligned.length} rigid-aligned")

    // ── 4. Decimate reference ─────────────────────────────────────────────────
    val decRef = ref.mesh.operations.decimate(Config.modelResolution)
    println(s"[info] Decimated: ${ref.mesh.pointSet.numberOfPoints} → ${decRef.pointSet.numberOfPoints} pts")

    // ── 5. GP model (prior) — for visualization ───────────────────────────────
    println("[GP] Building GP prior model for visualization...")
    val gpModel = NonRigidReg.buildGPModel(decRef)

    // ── 6. Non-rigid registration ──────────────────────────────────────────────
    println("[S05] Non-rigid GP-ICP")
    val registered: IndexedSeq[TriangleMesh[_3D]] =
      SSMBuilder.loadMeshes("pass_1").getOrElse {
        val r = rigidAligned.zipWithIndex.map { case (s, i) =>
          print(s"  NR ${i+1}/${rigidAligned.length}\r")
          NonRigidReg.register(decRef, s.mesh)
        }
        SSMBuilder.saveMeshes(r, "pass_1")
        r
      }
    println(s"\n[info] ${registered.length} non-rigid registered")

    // Mean shape from registered
    val meanShape = NonRigidReg.meanMesh(registered)

    // ── 7. SSM ────────────────────────────────────────────────────────────────
    println("[S06] Building SSM...")
    val ssm: PointDistributionModel[_3D, TriangleMesh] =
      SSMBuilder.loadSSM("ssm").getOrElse {
        val m = SSMBuilder.buildSSM(decRef, registered)
        SSMBuilder.saveSSM(m, "ssm")
        m
      }
    val evs: IndexedSeq[Double] = ssm.gp.klBasis.map(_.eigenvalue).toIndexedSeq
    val totalVar = evs.sum
    println(s"[info] SSM rank=${ssm.rank}")
    println("[info]  Mode   Var%   Cumul%    σ mm")
    var cumul = 0.0
    evs.take(math.min(10, ssm.rank)).zipWithIndex.foreach { case (ev, i) =>
      val pct = ev / totalVar * 100.0; cumul += pct
      println(f"[info]   ${i+1}%2d   $pct%5.2f  $cumul%6.2f  ${math.sqrt(ev)}%7.3f")
    }

    // ── 8. SSM Validation ─────────────────────────────────────────────────────
    println("[Validation] Computing metrics...")
    SSMValidation.printReport(ssm, decRef, registered)

    // ── 9. Open Scalismo Viewer ───────────────────────────────────────────────
    println("[UI] Opening viewer...")
    val ui = ScalismoUI("Scapula SSM — Full Pipeline")
    Thread.sleep(2000)

    // Pick one specimen for all per-specimen intermediate visualisations
    val eg  = rigidAligned.head
    val egR = registered.head   // same index, same topology as decRef

    // ── G00: Input Mesh (raw, unaligned) ─────────────────────────────────────
    val g00 = ui.createGroup("G00 Input Meshes (raw, unaligned)")
    specimens.foreach(s => viz(ui, g00, s.mesh, s.id))

    // ── G01: Landmarks ────────────────────────────────────────────────────────
    val g01 = ui.createGroup("G01 Landmarks (GC/TS/IA/PLA/AC on each specimen)")
    specimens.foreach(s => viz(ui, g01, s.lms.toList, s.id))

    // ── G02: Reference Mesh ───────────────────────────────────────────────────
    val g02 = ui.createGroup("G02 Reference Mesh + Landmarks")
    viz(ui, g02, ref.mesh, "reference")
    viz(ui, g02, ref.lms.toList, "ref_landmarks")

    // ── G03: Rigid Alignment (Procrustes + ICP) ───────────────────────────────
    val g03 = ui.createGroup("G03 Rigid Aligned — all specimens should overlap here")
    rigidAligned.foreach(s => viz(ui, g03, s.mesh, s.id))

    // ── G04: Kernel / GP Prior ────────────────────────────────────────────────
    // Show 3 samples from the GP prior to visualise what the kernel allows
    val g04 = ui.createGroup(s"G04 GP Prior kernel σ=${Config.gpSigma}mm scale=${Config.gpScale} — 3 samples")
    viz(ui, g04, decRef, "reference")
    (1 to 3).foreach { i =>
      viz(ui, g04, gpModel.sample(), s"GP_prior_sample_$i")
    }

    // ── G05: GP Prior Deformation Field (one sample) ──────────────────────────
    val g05 = ui.createGroup("G05 GP Prior Deformation Field (arrows = allowed deformations)")
    val priorSample = gpModel.sample()
    viz(ui, g05, decRef, "reference")
    viz(ui, g05, defField(decRef, priorSample), "deformation_field_prior")

    // ── G06: Non-rigid Registration — one specimen before/after ───────────────
    val g06 = ui.createGroup(s"G06 Non-Rigid Registration — ${eg.id} before/after")
    viz(ui, g06, eg.mesh, "rigid_aligned_before_NR")
    viz(ui, g06, egR, "non_rigid_registered")
    viz(ui, g06, decRef, "reference")

    // ── G07: Displacement Field (reference → registered, one specimen) ─────────
    val g07 = ui.createGroup(s"G07 Displacement Field — ref → ${eg.id} registered")
    viz(ui, g07, decRef, "reference")
    viz(ui, g07, defField(decRef, egR), "displacement_field")

    // ── G08: Correspondences (GP-ICP: ref pts → closest on target, one specimen) ──
    val g08 = ui.createGroup(s"G08 Correspondences — ref pts ↔ target (${eg.id})")
    viz(ui, g08, decRef, "reference")
    viz(ui, g08, eg.mesh, "target")
    // Sample 200 correspondence pairs shown as landmarks on the target surface
    val targetOps = eg.mesh.operations
    val corrLms: List[Landmark[_3D]] = decRef.pointSet.pointsWithId
      .filter { case (_, id) => id.id % (decRef.pointSet.numberOfPoints / 200 + 1) == 0 }
      .map { case (pt, id) =>
        Landmark(s"c${id.id}", targetOps.closestPointOnSurface(pt).point)
      }.toList
    viz(ui, g08, corrLms, "correspondences_on_target")

    // ── G09: All Registered Meshes ────────────────────────────────────────────
    val g09 = ui.createGroup("G09 Non-Rigid Registered (all specimens, same topology)")
    registered.zip(rigidAligned).foreach { case (m, s) => viz(ui, g09, m, s.id) }

    // ── G10: Mean Shape ───────────────────────────────────────────────────────
    val g10 = ui.createGroup("G10 Mean Shape (pointwise average of all registered)")
    viz(ui, g10, meanShape, "mean_shape")
    viz(ui, g10, ssm.mean, "ssm_mean")

    // ── G11: Covariance / PCA Eigenvalues info ────────────────────────────────
    // Scalismo has no built-in eigenvalue plot; we show modes as displacement fields
    val g11 = ui.createGroup("G11 PCA Mode 1 Deformation Fields (±1σ displacement arrows)")
    val c1p = DenseVector.zeros[Double](ssm.rank); c1p(0) = 1.0
    val c1n = DenseVector.zeros[Double](ssm.rank); c1n(0) = -1.0
    viz(ui, g11, ssm.mean, "mean")
    viz(ui, g11, defField(ssm.mean, ssm.instance(c1p)), "mode1_+1sigma_defField")
    viz(ui, g11, defField(ssm.mean, ssm.instance(c1n)), "mode1_-1sigma_defField")

    // ── G12–G14: PCA Modes ±1/2/3σ ───────────────────────────────────────────
    for (modeIdx <- 0 until math.min(3, ssm.rank)) {
      val ev    = evs(modeIdx)
      val sigma = math.sqrt(ev)
      val vPct  = (ev / totalVar * 100.0).toInt
      val gm    = ui.createGroup(f"G${12+modeIdx} Mode${modeIdx+1} σ=${sigma}%.2fmm var=$vPct%%")
      viz(ui, gm, ssm.mean, "mean")
      for (k <- Seq(-3, -2, -1, 1, 2, 3)) {
        val c = DenseVector.zeros[Double](ssm.rank); c(modeIdx) = k.toDouble
        viz(ui, gm, ssm.instance(c), s"${k}sigma")
      }
    }

    // ── G15: Eigenvectors as displacement fields for Mode 1 ───────────────────
    val g15 = ui.createGroup("G15 Eigenvector Mode1 — displacement field at ±1/2/3σ")
    viz(ui, g15, ssm.mean, "mean")
    for (k <- Seq(-3, -2, -1, 1, 2, 3)) {
      val c = DenseVector.zeros[Double](ssm.rank); c(0) = k.toDouble
      viz(ui, g15, defField(ssm.mean, ssm.instance(c)), s"eigvec_mode1_${k}sigma")
    }

    // ── G16: SSM Interactive ─────────────────────────────────────────────────
    val g16 = ui.createGroup("G16 SSM Interactive — drag Mode sliders in right panel")
    viz(ui, g16, ssm, "SSM")

    // ── G17: Random Model Samples ─────────────────────────────────────────────
    val g17 = ui.createGroup("G17 Model Samples (5 random instances)")
    (1 to 5).foreach(i => viz(ui, g17, ssm.sample(), s"sample_$i"))

    println("\n[info] All groups loaded. Groups in left panel:")
    println("[info]  G00 Input Meshes         — raw unaligned bones")
    println("[info]  G01 Landmarks             — 5 anatomical landmarks per specimen")
    println("[info]  G02 Reference             — reference mesh + landmarks")
    println("[info]  G03 Rigid Aligned         — ALL bones should overlap here")
    println("[info]  G04 GP Prior Samples      — what the kernel allows")
    println("[info]  G05 GP Deformation Field  — prior deformation arrows")
    println("[info]  G06 NR Before/After       — one specimen before vs after NR")
    println("[info]  G07 Displacement Field    — ref → registered displacement")
    println("[info]  G08 Correspondences       — ICP closest-point pairs")
    println("[info]  G09 Registered Meshes     — all NR registered (same topology)")
    println("[info]  G10 Mean Shape            — average bone shape")
    println("[info]  G11 Mode1 DefFields       — PCA mode 1 as displacement arrows")
    println("[info]  G12-G14 Modes 1-3         — ±1σ/2σ/3σ shape variants")
    println("[info]  G15 Eigenvector Field     — mode 1 as displacement field")
    println("[info]  G16 SSM Interactive       — drag sliders to explore model")
    println("[info]  G17 Model Samples         — 5 random SSM instances")
    println("[info] Close viewer window to exit.")
  }
}
