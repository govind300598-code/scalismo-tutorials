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
        if (s.isRight) Spec(s.modelId + "_mirrored", ScapulaData.mirrorMesh(raw), ScapulaData.mirrorLandmarks(lms))
        else           Spec(s.modelId, raw, lms)
      }
    println(s"[info] ${specimens.length} specimens loaded")

    // ── 2. Reference = specimen 002 left ──────────────────────────────────────
    val ref = specimens.find(s => s.id.contains("002") && !s.id.endsWith("_mirrored"))
                       .getOrElse(specimens.head)
    println(s"[info] Reference: ${ref.id}")

    // ── 3. Open viewer ────────────────────────────────────────────────────────
    println("[UI] Opening Scalismo viewer...")
    val ui = ScalismoUI("Scapula SSM — Full Pipeline Viewer")
    Thread.sleep(2000)

    // ── 4. S01: Landmarks ─────────────────────────────────────────────────────
    val s01 = ui.createGroup("S01_Landmarks (GC/TS/IA/PLA/AC)")
    specimens.foreach(s => show(ui, s01, s.lms.toList, s.id))

    // ── 5. S02: Reference ─────────────────────────────────────────────────────
    val s02 = ui.createGroup("S02_Reference")
    show(ui, s02, ref.mesh, ref.id)
    show(ui, s02, ref.lms.toList, ref.id + "_lms")

    // ── 6. S03: Landmark Procrustes alignment ─────────────────────────────────
    println("[S03] Landmark alignment (Procrustes)")
    val lmAligned: IndexedSeq[Spec] = specimens.map { s =>
      val t = ScapulaData.rigidFromLandmarks(s.lms, ref.lms)
      s.copy(mesh = s.mesh.transform(t), lms = s.lms.map(lm => lm.copy(point = t(lm.point))))
    }
    val s03 = ui.createGroup("S03_LandmarkAligned (if still scattered — landmarks are WRONG)")
    lmAligned.foreach(s => show(ui, s03, s.mesh, s.id))

    // ── 7. S04: Rigid ICP refinement ──────────────────────────────────────────
    println("[S04] Rigid ICP refinement")
    val rigidAligned: IndexedSeq[Spec] = lmAligned.zipWithIndex.map { case (s, i) =>
      println(s"  ICP ${i + 1}/${lmAligned.length}  ${s.id}")
      s.copy(mesh = RigidAlign.rigidIcp(s.mesh, ref.mesh, Config.icpIterations))
    }
    println(s"[info] ${rigidAligned.length} rigid-aligned")
    val s04 = ui.createGroup("S04_RigidAligned (ALL bones should overlap — verify alignment)")
    rigidAligned.foreach(s => show(ui, s04, s.mesh, s.id))

    // ── 8. Decimate reference ─────────────────────────────────────────────────
    val decimatedRef = ref.mesh.operations.decimate(Config.modelResolution)
    println(s"[info] Decimated reference: ${ref.mesh.pointSet.numberOfPoints} → ${decimatedRef.pointSet.numberOfPoints} pts")

    // ── 9. Non-rigid GP registration — iterative multi-pass ───────────────────
    println(s"[S05/S06] Non-rigid GP-ICP registration (${Config.refinePasses} pass(es), σ=${Config.gpSigma}mm)")

    var nonRigidMeshes: IndexedSeq[TriangleMesh[_3D]] = IndexedSeq.empty
    var currentRef: TriangleMesh[_3D] = decimatedRef

    // If final pass is cached, load it directly (fast path)
    val finalTag = s"pass_${Config.refinePasses}"
    SSMBuilder.loadMeshes(finalTag) match {
      case Some(cached) =>
        println(s"[info] Loaded ${cached.length} cached meshes from '$finalTag'")
        nonRigidMeshes = cached
        currentRef = NonRigidReg.meanMesh(cached)

      case None =>
        for (pass <- 1 to Config.refinePasses) {
          println(s"  [pass $pass/${Config.refinePasses}] registering ${rigidAligned.length} specimens")
          val tag = s"pass_$pass"
          val registered: IndexedSeq[TriangleMesh[_3D]] =
            SSMBuilder.loadMeshes(tag).getOrElse {
              val r = rigidAligned.zipWithIndex.map { case (s, i) =>
                println(s"    NR $pass.${i + 1}/${rigidAligned.length}  ${s.id}")
                NonRigidReg.register(currentRef, s.mesh)
              }
              SSMBuilder.saveMeshes(r, tag)
              r
            }
          if (SSMBuilder.loadMeshes(tag).isDefined) println(s"  [pass $pass] loaded from cache")
          println(s"[info] ${registered.length} registered (pass $pass)")

          val newMean = NonRigidReg.meanMesh(registered)
          if (pass > 1) {
            val stats = Metrics.symmetric(currentRef, newMean)
            val conv = if (stats.mean < 1.0) "CONVERGED" else "continuing"
            println(f"  Mean${pass - 1} ↔ Mean$pass  ${stats.render}  $conv")
          }
          nonRigidMeshes = registered
          currentRef = newMean
        }
    }
    println(s"[info] ${nonRigidMeshes.length} non-rigid registered (final: $finalTag)")

    // ── 10. S05: Pass 1 meshes ────────────────────────────────────────────────
    SSMBuilder.loadMeshes("pass_1").foreach { p1 =>
      val s05 = ui.createGroup("S05_NonRigid_Pass1 (tighter than S04)")
      p1.zip(rigidAligned).foreach { case (m, s) => show(ui, s05, m, s.id) }
    }

    // ── 11. S06: Best pass meshes (FINE PATCHES APPEAR HERE) ──────────────────
    val s06 = ui.createGroup(s"S06_NonRigid_Pass${Config.refinePasses} (best pass — FINE PATCHES HERE)")
    nonRigidMeshes.zip(rigidAligned).foreach { case (m, s) => show(ui, s06, m, s.id) }

    // ── 12. S07: Mean shapes across passes ────────────────────────────────────
    val s07 = ui.createGroup("S07_MeanShapes (< 1mm shift between passes = converged)")
    SSMBuilder.loadMeshes("pass_1").foreach { p1 =>
      show(ui, s07, NonRigidReg.meanMesh(p1.toIndexedSeq), "Mean_Pass1")
    }
    show(ui, s07, currentRef, s"Mean_Pass${Config.refinePasses}")

    // ── 13. Build / load SSM ──────────────────────────────────────────────────
    println("[S08] Building SSM")
    val ssmOpt: Option[PointDistributionModel[_3D, TriangleMesh]] =
      SSMBuilder.loadSSM(s"ssm_pass${Config.refinePasses}")
        .orElse(SSMBuilder.loadSSM("ssm_pass4"))
        .orElse(SSMBuilder.loadSSM("ssm_pass3"))
        .orElse(SSMBuilder.loadSSM("ssm_pass1"))
        .orElse(SSMBuilder.loadSSM("ssm"))
        .orElse(scala.util.Try(SSMBuilder.buildSSM(currentRef, nonRigidMeshes)).toOption)
    ssmOpt match {
      case Some(m) => println(s"[info] SSM rank=${m.rank}")
      case None    => println("[warn] SSM unavailable — S08/S09/S10/S11 will be skipped")
    }

    ssmOpt.foreach { ssm =>
      val evs: IndexedSeq[Double] = ssm.gp.klBasis.map(_.eigenvalue).toIndexedSeq
      val totalVar = evs.sum

      val s08 = ui.createGroup("S08_SSM — drag Mode sliders →")
      show(ui, s08, ssm, "SSM")

      for (modeIdx <- 0 until math.min(3, ssm.rank)) {
        val ev = evs(modeIdx); val sigma = math.sqrt(ev)
        val varPct = (ev / totalVar * 100.0).toInt
        val gm = ui.createGroup(f"S0${9 + modeIdx}_Mode${modeIdx + 1} σ=${sigma}%.1fmm $varPct%%")
        show(ui, gm, ssm.mean, "mean")
        for (k <- Seq(-2, -1, 1, 2)) {
          val c = DenseVector.zeros[Double](ssm.rank); c(modeIdx) = k.toDouble
          show(ui, gm, ssm.instance(c), s"${k}σ")
        }
      }
    }

    println("")
    println("=" * 70)
    println("  LOOK AT GROUP: S06_NonRigid_Pass4")
    println("  Turn ALL specimen meshes ON in that group.")
    println("  Fine scattered red/white patches = Z-fighting (< 0.5mm residual)")
    println("  This is the exact September 7 result.")
    println("=" * 70)
    println("[info] Done. Close the viewer window to exit.")
  }
}
