package scapula

import breeze.linalg.DenseVector
import scalismo.geometry._3D
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

object VisualizationApp {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    // ── Load ──────────────────────────────────────────────────────────────────
    val dir = Config.dataDir
    val csv = ScapulaData.csvFile(dir)
    val (lmMap, fromHeader, _) = ScapulaData.readLandmarkCsv(csv)
    if (!fromHeader)
      println("[warn] Landmark columns resolved by FALLBACK offsets — verify CSV header.")

    val allSpecimens = ScapulaData.specimens(dir).filter(s => lmMap.contains(s.modelId))
    println(s"[info] ${allSpecimens.length} specimens landmark-aligned and loaded with their landmarks")

    // Normalise: mirror right side so every specimen is anatomically left.
    case class Spec(id: String, mesh: TriangleMesh[_3D], lms: IndexedSeq[scalismo.geometry.Landmark[_3D]])

    val specimens: IndexedSeq[Spec] = allSpecimens.map { s =>
      val raw = ScapulaData.loadMesh(s.file)
      val lms = lmMap(s.modelId)
      if (s.isRight)
        Spec(s.modelId + "_mirrored", ScapulaData.mirrorMesh(raw), ScapulaData.mirrorLandmarks(lms))
      else
        Spec(s.modelId, raw, lms)
    }

    val ui = ScalismoUI("Scapula SSM — Full Pipeline Viewer")

    // ── S00: Raw input ────────────────────────────────────────────────────────
    val s00 = ui.createGroup("S00_RawInput (unaligned — deliberately scattered)")
    specimens.foreach(s => ui.show(s00, s.mesh, s.id))

    // ── S01: Landmark positions on raw meshes ─────────────────────────────────
    val s01 = ui.createGroup("S01_Landmarks (GC/TS/IA/PLA/AC on every specimen)")
    specimens.foreach(s => ui.show(s01, s.lms.toList, s.id))

    // ── S02: Reference specimen ───────────────────────────────────────────────
    val ref = specimens.head
    val s02 = ui.createGroup("S02_Reference")
    ui.show(s02, ref.mesh, ref.id)
    ui.show(s02, ref.lms.toList, ref.id + "_lms")

    // ── S03: Landmark-aligned (Procrustes only, no ICP) ───────────────────────
    println("[S03] Landmark alignment (Procrustes)")
    val lmAligned: IndexedSeq[Spec] = specimens.map { s =>
      val t   = ScapulaData.rigidFromLandmarks(s.lms, ref.lms)
      val m   = s.mesh.transform(t)
      val lms = s.lms.map(lm => lm.copy(point = t(lm.point)))
      s.copy(mesh = m, lms = lms)
    }
    val s03 = ui.createGroup("S03_LandmarkAligned (scattered here = landmarks WRONG)")
    lmAligned.foreach(s => ui.show(s03, s.mesh, s.id))

    // ── S04: Rigid-aligned (Procrustes + ICP) ─────────────────────────────────
    println("[S04] Rigid alignment (landmark + ICP)")
    val rigidAligned: IndexedSeq[Spec] = lmAligned.map { s =>
      val refined = RigidAlign.rigidIcp(s.mesh, ref.mesh, Config.icpIterations)
      // Recover rigid motion so landmarks travel with the mesh.
      val motion = scalismo.registration.LandmarkRegistration.rigid3DLandmarkRegistration(
        s.mesh.pointSet.points.zip(refined.pointSet.points).toIndexedSeq,
        center = scalismo.geometry.Point3D(0, 0, 0)
      )
      val lms = s.lms.map(lm => lm.copy(point = motion(lm.point)))
      s.copy(mesh = refined, lms = lms)
    }
    println(s"[info]  ${rigidAligned.length} rigid-aligned meshes loaded")
    val s04 = ui.createGroup("S04_RigidAligned (ALL bones should overlap — verify alignment)")
    rigidAligned.foreach(s => ui.show(s04, s.mesh, s.id))

    // ── Decimate reference (NN interpolation bridges original→decimated GP) ───
    val originalRef  = ref.mesh
    val decimatedRef = originalRef.operations.decimate(Config.modelResolution)
    println(s"[info] Reference: ${originalRef.pointSet.numberOfPoints} → ${decimatedRef.pointSet.numberOfPoints} vertices (decimated)")

    // ── S05 / S06: Non-rigid GP registration (multi-pass) ─────────────────────
    // Loads cached .vtk meshes from outDir if available; runs pipeline otherwise.
    var nonRigidMeshes: IndexedSeq[TriangleMesh[_3D]] = IndexedSeq.empty
    var bestRef: TriangleMesh[_3D]                    = decimatedRef

    val finalTag = s"pass_${Config.refinePasses}"
    SSMBuilder.loadMeshes(finalTag) match {
      case Some(cached) =>
        println(s"[info] Loaded ${cached.length} cached registered meshes (${SSMBuilder.cacheDir(finalTag)})")
        nonRigidMeshes = cached
        bestRef        = NonRigidReg.meanMesh(cached)

      case None =>
        var currentRef = decimatedRef
        for (pass <- 1 to Config.refinePasses) {
          println(s"[S0${4 + pass}] Non-rigid registration – Pass $pass (GP-ICP)")
          val tag = s"pass_$pass"
          val registered: IndexedSeq[TriangleMesh[_3D]] =
            SSMBuilder.loadMeshes(tag).getOrElse {
              val r = rigidAligned.zipWithIndex.map { case (s, i) =>
                println(s"  specimen ${i + 1}/${rigidAligned.length}  (${s.id})")
                NonRigidReg.register(originalRef, currentRef, s.mesh)
              }
              SSMBuilder.saveMeshes(r, tag)
              r
            }
          println(s"[info]  ${registered.length} non-rigid registered meshes (pass $pass)")

          nonRigidMeshes = registered
          val newMean = NonRigidReg.meanMesh(registered)

          if (pass > 1) {
            val stats = Metrics.symmetric(currentRef, newMean)
            val conv  = if (stats.mean < 1.0) "CONVERGED" else "continuing"
            println(f"  Mean${pass - 1} ↔ Mean$pass  ${stats.render}  $conv")
          }
          bestRef    = newMean
          currentRef = newMean
        }
    }

    // Show pass-1 group if available.
    SSMBuilder.loadMeshes("pass_1").foreach { p1 =>
      val s05 = ui.createGroup("S05_NonRigid_Pass1 (tighter overlap than S04)")
      p1.zip(rigidAligned).foreach { case (m, s) => ui.show(s05, m, s.id) }
    }

    val s06 = ui.createGroup(s"S06_NonRigid_Pass${Config.refinePasses} (last/best pass)")
    nonRigidMeshes.zip(rigidAligned).foreach { case (m, s) => ui.show(s06, m, s.id) }

    // ── S07: Mean shapes – convergence across passes ───────────────────────────
    val s07 = ui.createGroup("S07_MeanShapes (< 1 mm shift between passes = converged)")
    SSMBuilder.loadMeshes("pass_1").foreach { p1 =>
      ui.show(s07, NonRigidReg.meanMesh(p1.toIndexedSeq), "Mean_Pass1")
    }
    ui.show(s07, bestRef, s"Mean_Pass${Config.refinePasses}")

    // ── Build (or load) SSM ────────────────────────────────────────────────────
    val ssmName = s"ssm_pass${Config.refinePasses}"
    val ssm: PointDistributionModel[_3D, TriangleMesh] = SSMBuilder.loadSSM(ssmName).getOrElse {
      println(s"[info] Building SSM from ${nonRigidMeshes.length} pass-${Config.refinePasses} meshes (no .h5 found)...")
      val m = SSMBuilder.buildSSM(nonRigidMeshes)
      SSMBuilder.saveSSM(m, ssmName)
      m
    }
    println(s"[info] SSM: rank=${ssm.rank}")

    // Print variance table.
    val eigenValues = ssm.gp.klBasis.map(_.eigenValue)
    val totalVar    = eigenValues.sum
    val nPrint      = math.min(10, ssm.rank)
    println("[info]  Mode  Var%   Cumul%   σ mm")
    var cumul = 0.0
    eigenValues.take(nPrint).zipWithIndex.foreach { case (ev, i) =>
      val pct = ev / totalVar * 100; cumul += pct
      println(f"[info]   ${i + 1}%2d   ${pct}%5.2f  ${cumul}%6.2f  ${math.sqrt(ev)}%6.3f")
    }

    // ── S08: Interactive SSM ──────────────────────────────────────────────────
    println("[S08] Interactive SSM — drag Mode sliders in the right panel")
    val s08 = ui.createGroup("S08_SSM_Interactive (select group → drag 'Mode' sliders right panel)")
    ui.show(s08, ssm, "SSM")

    // ── S09–S11: First 3 PCA modes ±1σ / ±2σ / ±3σ ───────────────────────────
    for (modeIdx <- 0 until math.min(3, ssm.rank)) {
      val sigma  = math.sqrt(eigenValues(modeIdx))
      val varPct = (eigenValues(modeIdx) / totalVar * 100).toInt
      println(s"[S${9 + modeIdx}] Mode ${modeIdx + 1}: σ=${sigma.toInt}mm  var=$varPct%")

      val grp = ui.createGroup(s"S${9 + modeIdx}_Mode${modeIdx + 1} (σ=${sigma.toInt}mm var=$varPct%%)")
      ui.show(grp, ssm.mean, "mean")

      for (k <- Seq(-3, -2, -1, 1, 2, 3)) {
        val coeffs = DenseVector.zeros[Double](ssm.rank)
        coeffs(modeIdx) = k.toDouble      // k standard-deviation steps in this mode
        ui.show(grp, ssm.instance(coeffs), s"${k}σ")
      }
    }

    // ── S12: 5 random model instances ─────────────────────────────────────────
    println("[S12] 5 random model instances")
    val s12 = ui.createGroup("S12_ModelSamples (5 random instances from the final SSM)")
    (1 to 5).foreach(i => ui.show(s12, ssm.sample(), s"sample_$i"))

    // Viewer guide.
    val sep = "=" * 80
    println(s"\n[info] $sep")
    println("[info]   HOW TO USE THIS VIEWER")
    println(s"[info] $sep")
    println("[info]  S00 RawInput         → scattered unaligned bones (expected)")
    println("[info]  S01 Landmarks        → verify GC/TS/IA/PLA/AC are on the right spots")
    println("[info]  S02 Reference        → reference mesh + landmarks")
    println("[info]  S03 LandmarkAligned  → if still scattered here, landmarks are WRONG")
    println("[info]  S04 RigidAligned     → ALL bones should overlap — verify alignment")
    println("[info]  S05/S06 NonRigid     → tighter overlap than S04 — verify registration")
    println("[info]  S07 MeanShapes       → < 1 mm shift between passes = converged")
    println("[info]  S08 SSM Interactive  → select group → drag 'Mode' sliders right panel")
    println("[info]  S09-S11 Model-3      → toggle ±1σ/±2σ/±3σ meshes for each PCA mode")
    println("[info]  S12 ModelSamples     → 5 random instances from the final SSM")
    println(s"[info] $sep")
    println("[info] UI open. Close the window to exit.")
  }
}
