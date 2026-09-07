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

    // ── 5. Non-rigid GP registration (1 pass, cached) ─────────────────────────
    println("[S05] Non-rigid GP-ICP registration")
    val registered: IndexedSeq[TriangleMesh[_3D]] =
      SSMBuilder.loadMeshes("pass_1").getOrElse {
        val r = rigidAligned.zipWithIndex.map { case (s, i) =>
          println(s"  NR ${i + 1}/${rigidAligned.length}  ${s.id}")
          NonRigidReg.register(decRef, s.mesh)
        }
        SSMBuilder.saveMeshes(r, "pass_1")
        r
      }
    println(s"[info] ${registered.length} non-rigid registered")

    // ── 6. Build SSM ──────────────────────────────────────────────────────────
    println("[S06] Building SSM")
    val ssm: PointDistributionModel[_3D, TriangleMesh] =
      SSMBuilder.loadSSM("ssm").getOrElse {
        val m = SSMBuilder.buildSSM(decRef, registered)
        SSMBuilder.saveSSM(m, "ssm")
        m
      }
    println(s"[info] SSM rank=${ssm.rank}")

    val evs: IndexedSeq[Double] = ssm.gp.klBasis.map(_.eigenvalue).toIndexedSeq
    val totalVar = evs.sum
    println("[info]  Mode   Var%   Cumul%    σ mm")
    var cumul = 0.0
    evs.take(math.min(10, ssm.rank)).zipWithIndex.foreach { case (ev, i) =>
      val pct = ev / totalVar * 100.0; cumul += pct
      println(f"[info]   ${i+1}%2d   ${pct}%5.2f  ${cumul}%6.2f  ${math.sqrt(ev)}%7.3f")
    }

    // ── SSM Validation ───────────────────────────────────────────────────────
    println("[Validation] Computing compactness, generalization, specificity...")
    SSMValidation.printReport(ssm, decRef, registered)

    // ── 7. Viewer ─────────────────────────────────────────────────────────────
    println("[UI] Opening Scalismo viewer...")
    val ui = ScalismoUI("Scapula SSM")
    Thread.sleep(2000)

    val g00 = ui.createGroup("S00 Raw (unaligned)")
    specimens.foreach(s => show(ui, g00, s.mesh, s.id))

    val g02 = ui.createGroup("S02 Reference")
    show(ui, g02, ref.mesh, ref.id)
    show(ui, g02, ref.lms.toList, "landmarks")

    val g03 = ui.createGroup("S03 Landmark-Aligned")
    lmAligned.foreach(s => show(ui, g03, s.mesh, s.id))

    val g04 = ui.createGroup("S04 Rigid-Aligned (ICP)")
    rigidAligned.foreach(s => show(ui, g04, s.mesh, s.id))

    val g05 = ui.createGroup("S05 Non-Rigid Registered")
    registered.zip(rigidAligned).foreach { case (m, s) => show(ui, g05, m, s.id) }

    val g06 = ui.createGroup("S06 SSM — drag Mode sliders →")
    show(ui, g06, ssm, "SSM")

    for (modeIdx <- 0 until math.min(3, ssm.rank)) {
      val ev = evs(modeIdx); val sigma = math.sqrt(ev)
      val varPct = (ev / totalVar * 100.0).toInt
      val gm = ui.createGroup(f"S0${7+modeIdx} Mode${modeIdx+1} σ=${sigma}%.1fmm $varPct%%")
      show(ui, gm, ssm.mean, "mean")
      for (k <- Seq(-2, -1, 1, 2)) {
        val c = DenseVector.zeros[Double](ssm.rank); c(modeIdx) = k.toDouble
        show(ui, gm, ssm.instance(c), s"${k}σ")
      }
    }

    println("[info] Done. Close the viewer window to exit.")
  }
}
