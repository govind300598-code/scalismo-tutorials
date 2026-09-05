package scapula

import breeze.linalg.DenseVector
import scalismo.geometry._3D
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

import java.io.File

/**
 * Visual inspection of the scapula SSM pipeline.
 *
 * Groups are loaded in the order most useful for verification:
 *
 *   1_RigidAligned      ALL specimens after landmark+ICP rigid alignment.
 *                       THIS IS THE FIRST THING YOU SEE. All bones should overlap.
 *
 *   2_Pass4_Registered  ALL specimens after GP-ICP non-rigid registration.
 *                       Tighter overlap than 1_RigidAligned.
 *
 *   3_Means             Mean1–Mean4 overlaid. Should nearly coincide.
 *
 *   4_SSM_Interactive   Drag Mode 0 slider to morph the mean shape live.
 *
 *   5_ModeK (K=1,2,3)   ±1σ / ±2σ / ±3σ shapes along PCA mode K.
 *
 *   6_Outliers          3 worst-fitting specimens vs SSM mean.
 *
 *   7_ModelSamples      5 random instances drawn from the SSM.
 *
 *   8_Reference         Reference mesh + its landmarks.
 *
 *   9_Landmarks         Landmark points on first 6 specimens.
 *
 *   Z_RawInput          Raw unaligned input (deliberately scattered).
 *
 * Usage:
 *   sbt "runMain scapula.ViewSSM"
 *   SCAPULA_OUT_DIR=/your/path sbt "runMain scapula.ViewSSM"
 */
object ViewSSM {

  private def filesIn(dir: File, prefix: String): IndexedSeq[File] =
    if (!dir.isDirectory) IndexedSeq.empty
    else Option(dir.listFiles()).getOrElse(Array.empty[File])
      .filter(f => f.getName.endsWith(".stl") && f.getName.startsWith(prefix))
      .sortBy(_.getName).toIndexedSeq

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dataDir = Config.dataDir
    val baseDir = if (args.nonEmpty) new File(args(0)) else Config.outDir

    println("\n" + "=" * 72)
    println("  Scapula SSM — Pipeline Viewer")
    println("=" * 72)

    val ui = ScalismoUI("Scapula SSM — Pipeline Viewer")

    val (lmMap, _, _) = ScapulaData.readLandmarkCsv(ScapulaData.csvFile(dataDir))
    val specimens     = ScapulaData.specimens(dataDir)
    val passDirs      = (1 to 8).map(n => n -> new File(baseDir, s"pass$n")).filter(_._2.isDirectory)

    // ═══════════════════════════════════════════════════════════════════════════
    // GROUP 1 — Rigid-aligned (landmark + ICP). FIRST VISIBLE GROUP.
    // ═══════════════════════════════════════════════════════════════════════════
    val pass1Dir   = new File(baseDir, "pass1")
    val rigidFiles = filesIn(pass1Dir, "rigid_")
    if (rigidFiles.nonEmpty) {
      println(s"\n[1] Rigid-aligned meshes — ${rigidFiles.length} specimens, landmark+ICP ONLY")
      val grp = ui.createGroup(
        s"1_RigidAligned (${rigidFiles.length} specimens — landmark+ICP, BEFORE non-rigid)")
      rigidFiles.foreach { f =>
        ui.show(grp, ScapulaData.loadMesh(f),
          f.getName.stripSuffix(".stl").stripPrefix("rigid_"))
      }
      println("  All bones should overlap in the same anatomical frame.")
      println("  If still scattered: landmarks are wrong — check group 9_Landmarks.")
    } else {
      println("\n[1] WARNING: no rigid_*.stl found in pass1/")
      println("  Run: SCAPULA_OUT_DIR=... sbt \"runMain scapula.RebuildSSM\" first.")
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // GROUP 2 — Best non-rigidly registered result (last pass).
    // ═══════════════════════════════════════════════════════════════════════════
    passDirs.lastOption.foreach { case (n, passDir) =>
      val files = filesIn(passDir, "reg_")
      if (files.nonEmpty) {
        println(s"\n[2] Pass $n registered — ${files.length} specimens, GP-ICP non-rigid")
        val grp = ui.createGroup(
          s"2_Pass${n}_Registered (${files.length} specimens, σ=${Config.kernelSigma.toInt}mm)")
        files.foreach { f =>
          ui.show(grp, ScapulaData.loadMesh(f),
            f.getName.stripSuffix(".stl").stripPrefix("reg_"))
        }
        println("  Tighter overlap than 1_RigidAligned.")
      }
    }

    // Also show pass 1 for comparison
    passDirs.headOption.foreach { case (n, passDir) =>
      val files = filesIn(passDir, "reg_")
      if (files.nonEmpty && passDirs.length > 1) {
        println(s"\n  [2b] Pass 1 registered — for comparison with pass ${passDirs.last._1}")
        val grp = ui.createGroup(
          s"2b_Pass1_Registered (${files.length} specimens — compare with 2_PassN)")
        files.foreach { f =>
          ui.show(grp, ScapulaData.loadMesh(f),
            f.getName.stripSuffix(".stl").stripPrefix("reg_"))
        }
      }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // GROUP 3 — Mean shapes.
    // ═══════════════════════════════════════════════════════════════════════════
    println("\n[3] Mean shapes — convergence check")
    val meansGrp    = ui.createGroup("3_Means (Mean1–Mean4 overlaid — < 1 mm shift = converged)")
    val loadedMeans = scala.collection.mutable.ArrayBuffer.empty[(Int, TriangleMesh[_3D])]
    (1 to 8).foreach { n =>
      val f = new File(baseDir, s"mean_pass$n.stl")
      if (f.exists()) {
        val m = ScapulaData.loadMesh(f)
        ui.show(meansGrp, m, s"Mean$n")
        loadedMeans += (n -> m)
      }
    }
    if (loadedMeans.length >= 2) {
      println(f"  ${"Comparison"}%-14s  ${"Mean mm"}%9s  ${"RMS mm"}%9s  ${"HD95 mm"}%10s  Status")
      loadedMeans.sliding(2).foreach { w =>
        val (n1, m1) = w(0); val (n2, m2) = w(1)
        val st = Metrics.symmetric(m1, m2)
        val ok = if (st.mean < 1.0) "✓ converged" else "not yet"
        println(f"  Mean$n1↔Mean$n2        ${st.mean}%9.3f  ${st.rms}%9.3f  ${st.hd95}%10.3f  $ok")
      }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // GROUPS 4, 5, 6, 7 — SSM built from last pass.
    // ═══════════════════════════════════════════════════════════════════════════
    passDirs.lastOption.foreach { case (n, passDir) =>
      val files = filesIn(passDir, "reg_")
      if (files.nonEmpty) {
        println(s"\n[4–7] Building SSM from pass $n (${files.length} meshes)…")
        val meshes = files.map(ScapulaData.loadMesh)
        val dc     = DataCollection.fromTriangleMesh3DSequence(meshes.head, meshes)
        val model  = PointDistributionModel.createUsingPCA(dc)
        val evs    = model.gp.klBasis.map(_.eigenvalue).toArray
        val total  = evs.sum
        val cumV   = evs.scanLeft(0.0)(_ + _).tail.map(_ / total * 100.0)
        def mFor(p: Double) = { val i = cumV.indexWhere(_ >= p); if (i < 0) model.rank else i + 1 }

        println(f"  rank=${model.rank}  90%%: ${mFor(90)} modes  95%%: ${mFor(95)}  99%%: ${mFor(99)}")
        println(f"  ${"Mode"}%5s  ${"Var %%"}%8s  ${"Cumul %%"}%9s  ${"σ mm"}%8s")
        evs.take(math.min(10, model.rank)).zipWithIndex.foreach { case (ev, i) =>
          println(f"  ${i+1}%5d  ${ev/total*100}%8.2f  ${cumV(i)}%9.2f  ${math.sqrt(ev)}%8.3f")
        }

        val mean = model.mean

        // 4. Interactive SSM
        val ssmGrp = ui.createGroup(s"4_SSM_Interactive (pass $n — click, drag Mode sliders)")
        ui.show(ssmGrp, model, s"SSM_pass$n")
        println(s"\n  → Click 4_SSM_Interactive → drag Mode 0 slider to morph shape")

        // 5. PCA modes ±1σ / ±2σ / ±3σ
        val nModes = math.min(3, model.rank)
        (0 until nModes).foreach { modeIdx =>
          val ev     = evs(modeIdx)
          val sigma  = math.sqrt(ev)
          val varPct = ev / total * 100.0
          def c(s: Double) = DenseVector(Array.tabulate(model.rank)(j =>
            if (j == modeIdx) s * sigma else 0.0))
          val modeGrp = ui.createGroup(
            s"5_Mode${modeIdx+1} (${varPct.toInt}%% var, 1σ=${sigma.toInt}mm)")
          ui.show(modeGrp, mean,                   s"Mode${modeIdx+1}_mean")
          ui.show(modeGrp, model.instance(c( 1.0)), s"Mode${modeIdx+1}_+1sigma")
          ui.show(modeGrp, model.instance(c(-1.0)), s"Mode${modeIdx+1}_-1sigma")
          ui.show(modeGrp, model.instance(c( 2.0)), s"Mode${modeIdx+1}_+2sigma")
          ui.show(modeGrp, model.instance(c(-2.0)), s"Mode${modeIdx+1}_-2sigma")
          ui.show(modeGrp, model.instance(c( 3.0)), s"Mode${modeIdx+1}_+3sigma")
          ui.show(modeGrp, model.instance(c(-3.0)), s"Mode${modeIdx+1}_-3sigma")
          println(f"  Mode ${modeIdx+1}: σ=${sigma}%.2f mm  var=${varPct}%.2f%%  shown at ±1σ/±2σ/±3σ")
        }

        // 6. Outliers
        val ranked = meshes.zip(files).map { case (m, f) =>
          val d = Metrics.surfaceDistances(m, mean)
          (d.sum / d.length, m, f.getName.stripSuffix(".stl").stripPrefix("reg_"))
        }.sortBy(-_._1)
        val all = ranked.map(_._1)
        println(f"\n[6] Outliers — population mean=${all.sum/all.length}%.3f  max=${all.max}%.3f  min=${all.min}%.3f  mm")
        val outlGrp = ui.createGroup("6_Outliers (3 worst vs SSM mean)")
        ui.show(outlGrp, mean, "SSM_mean")
        ranked.take(3).foreach { case (d, m, name) =>
          println(f"    [!]  $name%-42s  $d%.3f mm")
          ui.show(outlGrp, m, f"OUTLIER_${d}%.2fmm_$name")
        }
        ranked.takeRight(3).reverse.foreach { case (d, _, name) =>
          println(f"    [✓]  $name%-42s  $d%.3f mm")
        }

        // 7. Random model samples
        println("\n[7] Random model samples — 5 instances from SSM")
        val sampGrp = ui.createGroup("7_ModelSamples (5 random SSM instances)")
        (1 to 5).foreach { i => ui.show(sampGrp, model.sample(), s"Sample_$i") }
      }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // GROUP 8 — Reference mesh (loaded last so it doesn't dominate the view).
    // ═══════════════════════════════════════════════════════════════════════════
    val specsWithLm = specimens.filter(s => lmMap.contains(s.modelId))
    val refSpec     = specsWithLm(Config.refIdx.min(specsWithLm.length - 1))
    val refRaw      = ScapulaData.loadMesh(refSpec.file)
    val refFull     = if (refSpec.isRight) ScapulaData.mirrorMesh(refRaw) else refRaw
    val refMesh8k   = {
      val n = refFull.pointSet.numberOfPoints
      if (n > Config.modelResolution)
        ScapulaData.decimateInCorrespondence(refFull, IndexedSeq(refFull), Config.modelResolution).head
      else refFull
    }
    val refLms = if (refSpec.isRight) ScapulaData.mirrorLandmarks(lmMap(refSpec.modelId))
                 else lmMap(refSpec.modelId)
    val refGrp = ui.createGroup(s"8_Reference (${refSpec.modelId})")
    ui.show(refGrp, refMesh8k, s"REF_mesh")
    ui.show(refGrp, refLms,    s"REF_landmarks")
    println(s"\n[8] Reference: ${refSpec.modelId} (${refMesh8k.pointSet.numberOfPoints} vertices)")

    // ═══════════════════════════════════════════════════════════════════════════
    // GROUP 9 — Landmarks on specimens.
    // ═══════════════════════════════════════════════════════════════════════════
    println("\n[9] Landmarks — GC, TS, IA, PLA, AC on first 6 specimens")
    val lmGrp = ui.createGroup("9_Landmarks (first 6 specimens — verify placement)")
    specimens.filter(s => lmMap.contains(s.modelId)).take(6).foreach { spec =>
      val lms = if (spec.isRight) ScapulaData.mirrorLandmarks(lmMap(spec.modelId))
                else lmMap(spec.modelId)
      ui.show(lmGrp, lms, s"LM_${spec.modelId}")
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // GROUP Z — Raw unaligned input (deliberately last so it never confuses).
    // ═══════════════════════════════════════════════════════════════════════════
    println(s"\n[Z] Raw input — ${specimens.length} unaligned original-resolution bones (deliberately scattered)")
    val rawGrp = ui.createGroup(s"Z_RawInput (${specimens.length} raw — UNALIGNED — for reference only)")
    specimens.foreach { spec =>
      val raw  = ScapulaData.loadMesh(spec.file)
      val mesh = if (spec.isRight) ScapulaData.mirrorMesh(raw) else raw
      ui.show(rawGrp, mesh, spec.modelId + (if (spec.isRight) "_R→L" else ""))
    }

    println("\n" + "=" * 72)
    println("  1_RigidAligned   ← verify THIS first: all bones should overlap")
    println("  2_Pass4          ← tighter than 1_RigidAligned")
    println("  4_SSM_Interactive← click → drag Mode 0 slider")
    println("  5_Mode1/2/3      ← toggle to see ±1σ/±2σ/±3σ shape deformation")
    println("  Z_RawInput       ← always scattered — that is CORRECT")
    println("=" * 72)
    println("\nUI open. Close the window to exit.")
  }
}
