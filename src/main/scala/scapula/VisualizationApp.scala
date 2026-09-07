package scapula

import breeze.linalg.DenseVector
import scalismo.geometry._3D
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

object VisualizationApp {

  private def safeShow[A](ui: ScalismoUI, group: scalismo.ui.api.Group, obj: A, name: String)(
    implicit ev: scalismo.ui.api.ShowInScene[A]
  ): Unit =
    try { ui.show(group, obj, name); Thread.sleep(20) }
    catch { case e: Exception => println(s"  [warn] could not display '$name': ${e.getMessage}") }

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    // ── Load data ─────────────────────────────────────────────────────────────
    val dir = Config.dataDir
    val csv = ScapulaData.csvFile(dir)
    val (lmMap, fromHeader, _) = ScapulaData.readLandmarkCsv(csv)
    if (!fromHeader)
      println("[warn] Landmark columns resolved by FALLBACK offsets — verify CSV header.")

    val allSpecimens = ScapulaData.specimens(dir).filter(s => lmMap.contains(s.modelId))
    println(s"[info] ${allSpecimens.length} specimens with landmarks found")

    case class Spec(id: String, mesh: TriangleMesh[_3D], lms: IndexedSeq[scalismo.geometry.Landmark[_3D]])

    // Mirror right-side scapulae so all are anatomically left
    val specimens: IndexedSeq[Spec] = allSpecimens.map { s =>
      val raw = ScapulaData.loadMesh(s.file)
      val lms = lmMap(s.modelId)
      if (s.isRight)
        Spec(s.modelId + "_mirrored", ScapulaData.mirrorMesh(raw), ScapulaData.mirrorLandmarks(lms))
      else
        Spec(s.modelId, raw, lms)
    }

    // ── Open UI ───────────────────────────────────────────────────────────────
    val ui = ScalismoUI("Scapula SSM — Full Pipeline Viewer")
    Thread.sleep(2000) // VTK must finish initialising its renderer before any show()

    // ── S00: Raw input ────────────────────────────────────────────────────────
    val s00 = ui.createGroup("S00_RawInput (unaligned — deliberately scattered)")
    specimens.foreach(s => safeShow(ui, s00, s.mesh, s.id))

    // ── S01: Landmarks ────────────────────────────────────────────────────────
    val s01 = ui.createGroup("S01_Landmarks (GC/TS/IA/PLA/AC on every specimen)")
    specimens.foreach(s => safeShow(ui, s01, s.lms.toList, s.id))

    // ── S02: Reference — prefer specimen 002, female, left side ──────────────
    val ref = specimens
      .find(s => s.id.contains("002") && s.id.toLowerCase.contains("_f_") && !s.id.endsWith("_mirrored"))
      .orElse(specimens.find(s => s.id.contains("002") && !s.id.endsWith("_mirrored")))
      .getOrElse(specimens.head)
    println(s"[info] Reference specimen: ${ref.id}")
    val s02 = ui.createGroup("S02_Reference")
    safeShow(ui, s02, ref.mesh, ref.id)
    safeShow(ui, s02, ref.lms.toList, ref.id + "_lms")

    // ── S03: Landmark-aligned (Procrustes only) ───────────────────────────────
    println("[S03] Landmark alignment (Procrustes)")
    val lmAligned: IndexedSeq[Spec] = specimens.map { s =>
      val t = ScapulaData.rigidFromLandmarks(s.lms, ref.lms)
      s.copy(mesh = s.mesh.transform(t), lms = s.lms.map(lm => lm.copy(point = t(lm.point))))
    }
    val s03 = ui.createGroup("S03_LandmarkAligned (all bones should now roughly overlap)")
    lmAligned.foreach(s => safeShow(ui, s03, s.mesh, s.id))

    // ── S04: Rigid-aligned (Procrustes + ICP) ─────────────────────────────────
    println("[S04] Rigid alignment (landmark Procrustes + ICP)")
    val rigidAligned: IndexedSeq[Spec] = lmAligned.zipWithIndex.map { case (s, i) =>
      println(s"  [S04] ICP specimen ${i + 1}/${lmAligned.length}  (${s.id})")
      val refined = RigidAlign.rigidIcp(s.mesh, ref.mesh, Config.icpIterations)
      s.copy(mesh = refined)
    }
    println(s"[info]  ${rigidAligned.length} rigid-aligned meshes")
    val s04 = ui.createGroup("S04_RigidAligned (ALL bones should overlap — verify this)")
    rigidAligned.foreach(s => safeShow(ui, s04, s.mesh, s.id))

    // ── Decimate reference for non-rigid registration ─────────────────────────
    val decimatedRef = ref.mesh.operations.decimate(Config.modelResolution)
    println(s"[info] Reference decimated: ${ref.mesh.pointSet.numberOfPoints} → ${decimatedRef.pointSet.numberOfPoints} vertices")

    // ── S05/S06: Non-rigid GP registration (multi-pass, cached) ──────────────
    var nonRigidMeshes: IndexedSeq[TriangleMesh[_3D]] = IndexedSeq.empty
    var currentRef: TriangleMesh[_3D] = decimatedRef

    val finalTag = s"pass_${Config.refinePasses}"
    SSMBuilder.loadMeshes(finalTag) match {
      case Some(cached) =>
        println(s"[info] Loaded ${cached.length} cached registered meshes from '${SSMBuilder.cacheDir(finalTag)}'")
        nonRigidMeshes = cached
        currentRef = NonRigidReg.meanMesh(cached)

      case None =>
        for (pass <- 1 to Config.refinePasses) {
          println(s"[pass $pass/${Config.refinePasses}] Non-rigid GP-ICP registration (σ=${Config.gpSigma}mm)")
          val tag = s"pass_$pass"
          val registered: IndexedSeq[TriangleMesh[_3D]] =
            SSMBuilder.loadMeshes(tag).getOrElse {
              val r = rigidAligned.zipWithIndex.map { case (s, i) =>
                println(s"  specimen ${i + 1}/${rigidAligned.length}  (${s.id})")
                NonRigidReg.register(currentRef, s.mesh)
              }
              SSMBuilder.saveMeshes(r, tag)
              r
            }
          println(s"[info]  ${registered.length} registered (pass $pass)")

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

    SSMBuilder.loadMeshes("pass_1").foreach { p1 =>
      val s05 = ui.createGroup("S05_NonRigid_Pass1 (tighter than S04)")
      p1.zip(rigidAligned).foreach { case (m, s) => safeShow(ui, s05, m, s.id) }
    }

    val s06 = ui.createGroup(s"S06_NonRigid_Pass${Config.refinePasses} (best pass)")
    nonRigidMeshes.zip(rigidAligned).foreach { case (m, s) => safeShow(ui, s06, m, s.id) }

    val s07 = ui.createGroup("S07_MeanShapes (< 1 mm shift between passes = converged)")
    SSMBuilder.loadMeshes("pass_1").foreach { p1 =>
      safeShow(ui, s07, NonRigidReg.meanMesh(p1.toIndexedSeq), "Mean_Pass1")
    }
    safeShow(ui, s07, currentRef, s"Mean_Pass${Config.refinePasses}")

    // ── Build / load SSM ──────────────────────────────────────────────────────
    val ssmName = s"ssm_pass${Config.refinePasses}"
    val ssm: PointDistributionModel[_3D, TriangleMesh] = SSMBuilder.loadSSM(ssmName).getOrElse {
      println(s"[info] Building SSM from ${nonRigidMeshes.length} meshes...")
      val m = SSMBuilder.buildSSM(currentRef, nonRigidMeshes)
      SSMBuilder.saveSSM(m, ssmName)
      m
    }
    println(s"[info] SSM rank=${ssm.rank}")

    val eigenValues: IndexedSeq[Double] = ssm.gp.klBasis.map(_.eigenvalue).toIndexedSeq
    val totalVar = eigenValues.sum
    val nPrint = math.min(10, ssm.rank)
    println("[info]  Mode   Var%   Cumul%    σ mm")
    var cumul = 0.0
    eigenValues.take(nPrint).zipWithIndex.foreach { case (ev, i) =>
      val pct = ev / totalVar * 100.0; cumul += pct
      println(f"[info]   ${i + 1}%2d   ${pct}%5.2f  ${cumul}%6.2f  ${math.sqrt(ev)}%7.3f")
    }

    // ── S08: Interactive SSM ──────────────────────────────────────────────────
    val s08 = ui.createGroup("S08_SSM_Interactive (drag Mode sliders in the right panel)")
    safeShow(ui, s08, ssm, "SSM")

    // ── S09–S11: First 3 PCA modes ±1/2/3σ ───────────────────────────────────
    for (modeIdx <- 0 until math.min(3, ssm.rank)) {
      val ev = eigenValues(modeIdx)
      val sigma = math.sqrt(ev)
      val varPct = (ev / totalVar * 100.0).toInt
      val grp = ui.createGroup(s"S${9 + modeIdx}_Mode${modeIdx + 1} (σ=${f"$sigma%.1f"}mm var=$varPct%%)")
      safeShow(ui, grp, ssm.mean, "mean")
      for (k <- Seq(-3, -2, -1, 1, 2, 3)) {
        val coeffs = DenseVector.zeros[Double](ssm.rank)
        coeffs(modeIdx) = k.toDouble
        safeShow(ui, grp, ssm.instance(coeffs), s"${k}σ")
      }
    }

    // ── S12: Random samples ───────────────────────────────────────────────────
    val s12 = ui.createGroup("S12_ModelSamples (5 random instances)")
    (1 to 5).foreach(i => safeShow(ui, s12, ssm.sample(), s"sample_$i"))

    println("\n[info] Pipeline complete. Close the viewer window to exit.")
    println("[info]  S03 LandmarkAligned → bones should overlap after Procrustes")
    println("[info]  S04 RigidAligned    → tighter overlap after ICP")
    println("[info]  S08 SSM Interactive → select group, drag Mode sliders on right")
  }
}
