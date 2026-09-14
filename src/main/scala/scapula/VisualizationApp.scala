package scapula

import breeze.linalg.DenseVector
import scalismo.geometry._3D
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

object VisualizationApp {

  private def show[A](ui: ScalismoUI, grp: scalismo.ui.api.Group, obj: A, name: String)(
    implicit ev: scalismo.ui.api.ShowInScene[A]
  ): Unit = try {
    ui.show(grp, obj, name)
    Thread.sleep(20)
  } catch { case e: Exception => println(s"  [warn] display '$name': ${e.getMessage}") }

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    // ── 1. Load specimens ─────────────────────────────────────────────────────
    val dir = Config.dataDir
    val csv = ScapulaData.csvFile(dir)
    val (lmMap, fromHeader, _) = ScapulaData.readLandmarkCsv(csv)
    if (!fromHeader) println("[warn] landmark columns resolved by fallback offsets")

    case class Spec(id: String, mesh: TriangleMesh[_3D], lms: IndexedSeq[scalismo.geometry.Landmark[_3D]])

    val specimens: IndexedSeq[Spec] = ScapulaData.specimens(dir)
      .filter(s => lmMap.contains(s.modelId))
      .map { s =>
        val raw = ScapulaData.loadMesh(s.file)
        val lms = lmMap(s.modelId)
        if (s.isRight) Spec(s.modelId + "_mir", ScapulaData.mirrorMesh(raw), ScapulaData.mirrorLandmarks(lms))
        else           Spec(s.modelId, raw, lms)
      }
    println(s"[info] ${specimens.length} specimens loaded")

    // ── 2. Reference = specimen 002 left ─────────────────────────────────────
    val ref = specimens.find(s => s.id.contains("002") && !s.id.endsWith("_mir"))
                       .getOrElse(specimens.head)
    println(s"[info] Reference: ${ref.id}")

    // ── 3. Rigid alignment (Procrustes + ICP) ─────────────────────────────────
    println("[S03] Landmark Procrustes alignment")
    val lmAligned = specimens.map { s =>
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
    val decRef = ref.mesh.operations.decimate(Config.modelResolution)
    println(s"[info] Decimated reference: ${ref.mesh.pointSet.numberOfPoints} → ${decRef.pointSet.numberOfPoints} pts")

    // ── 5. Non-rigid GP registration — iterative multi-pass ───────────────────
    println(s"[S05] Non-rigid GP-ICP registration (${Config.refinePasses} pass(es))")

    // Each pass: register all specimens to current reference, then update
    // reference to mean of registered meshes for the next pass.
    // This mirrors the Sept-7 pipeline (pass_1 → pass_4) that produced fine patches.
    var passRef   = decRef
    var lastMeshes: IndexedSeq[TriangleMesh[_3D]] = IndexedSeq.empty
    var lastPass  = 0

    for (pass <- 1 to Config.refinePasses) {
      val tag = s"pass_$pass"
      val cached = SSMBuilder.loadMeshes(tag)
      lastMeshes = cached.getOrElse {
        println(s"  [pass $pass] registering ${rigidAligned.length} specimens to ${passRef.pointSet.numberOfPoints}-pt reference")
        val r = rigidAligned.zipWithIndex.map { case (s, i) =>
          println(s"    NR $pass.${ i + 1}/${rigidAligned.length}  ${s.id}")
          NonRigidReg.register(passRef, s.mesh)
        }
        SSMBuilder.saveMeshes(r, tag)
        r
      }
      if (cached.isDefined) println(s"  [pass $pass] loaded from cache ($tag)")
      lastPass = pass
      // Update reference to mean of this pass for next iteration
      if (pass < Config.refinePasses)
        passRef = NonRigidReg.meanMesh(lastMeshes)
    }

    val registered     = lastMeshes
    val registeredPass = s"pass_$lastPass"
    println(s"[info] ${registered.length} non-rigid registered (final: $registeredPass)")

    // ── 6. Build SSM (optional — viewer opens even if this fails) ─────────────
    println("[S06] Building SSM")
    val ssmOpt: Option[PointDistributionModel[_3D, TriangleMesh]] =
      SSMBuilder.loadSSM("ssm_pass4")
        .orElse(SSMBuilder.loadSSM("ssm_pass3"))
        .orElse(SSMBuilder.loadSSM("ssm_pass2"))
        .orElse(SSMBuilder.loadSSM("ssm_pass1"))
        .orElse(SSMBuilder.loadSSM("ssm"))
        .orElse(scala.util.Try(SSMBuilder.buildSSM(passRef, registered)).toOption)
    ssmOpt match {
      case Some(m) => println(s"[info] SSM rank=${m.rank}")
      case None    => println("[warn] SSM unavailable — S06/S07/S08/S09 will be skipped")
    }

    // ── 7. Viewer ─────────────────────────────────────────────────────────────
    println("[UI] Opening Scalismo viewer...")
    val ui = ScalismoUI("Scapula SSM — Full Pipeline Viewer")
    Thread.sleep(2000)

    val g00 = ui.createGroup("S01_Landmarks (GC/TS/IA/PLA/AC)")
    specimens.foreach(s => show(ui, g00, s.lms.toList, s.id))

    val g02 = ui.createGroup("S02_Reference")
    show(ui, g02, ref.lms.toList, "landmarks")
    show(ui, g02, ref.mesh, ref.id)

    val g03 = ui.createGroup("S03_LandmarkAligned (if still scattered — landmarks are WRONG)")
    lmAligned.foreach(s => show(ui, g03, s.mesh, s.id))

    val g04 = ui.createGroup("S04_RigidAligned (ALL bones should overlap — verify alignment)")
    rigidAligned.foreach(s => show(ui, g04, s.mesh, s.id))

    val g05 = ui.createGroup(s"S05_NonRigid_$registeredPass (all registered + reference overlap here)")
    show(ui, g05, passRef, "REFERENCE")
    registered.zipWithIndex.foreach { case (m, i) => show(ui, g05, m, s"registered_$i") }

    ssmOpt.foreach { ssm =>
      val evs: IndexedSeq[Double] = ssm.gp.klBasis.map(_.eigenvalue).toIndexedSeq
      val totalVar = evs.sum
      val g06 = ui.createGroup("S06_SSM — drag Mode sliders →")
      show(ui, g06, ssm, "SSM")
      for (modeIdx <- 0 until math.min(3, ssm.rank)) {
        val ev = evs(modeIdx); val sigma = math.sqrt(ev)
        val varPct = (ev / totalVar * 100.0).toInt
        val gm = ui.createGroup(f"S0${7+modeIdx}_Mode${modeIdx+1} σ=${sigma}%.1fmm $varPct%%")
        show(ui, gm, ssm.mean, "mean")
        for (k <- Seq(-2, -1, 1, 2)) {
          val c = DenseVector.zeros[Double](ssm.rank); c(modeIdx) = k.toDouble
          show(ui, gm, ssm.instance(c), s"${k}σ")
        }
      }
    }

    println("[info] Done. Close the viewer window to exit.")
  }
}
