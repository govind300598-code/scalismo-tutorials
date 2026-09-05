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
 * Full visual inspection — ALL intermediate pipeline steps shown as separate groups.
 *
 * Groups (eye icon to show/hide each):
 *
 *   A_InputMeshes       Raw scapulae at ORIGINAL resolution, left+right as loaded.
 *                       EXPECT: shapes scattered, unaligned — that is correct.
 *
 *   A1_Landmarks        Landmark points on each specimen (first 6 shown).
 *                       Verify: landmarks should sit at correct anatomical sites.
 *
 *   A2_Reference        The decimated 8k-vertex reference mesh used for pass 1.
 *
 *   A3_RigidAligned     Specimens after LANDMARK+ICP rigid alignment (BEFORE non-rigid).
 *                       EXPECT: all bones roughly overlapping in a common frame.
 *                       If still spread out here, landmark quality is the problem.
 *
 *   B_PassN_Registered  After GP-ICP non-rigid registration, pass N.
 *                       EXPECT: tighter overlap than A3_RigidAligned.
 *
 *   C_Means             Mean1–Mean4 overlaid. Nearly coincident = GPA converged.
 *
 *   D_DistMap_PassN     Registered meshes from pass N alongside the mean (for visual diff).
 *
 *   E_SSM_Interactive   Drag Mode 0 slider to morph the scapula live.
 *
 *   F_Outliers          3 worst-fitting specimens vs SSM mean.
 *
 *   G_ModeK (K=1,2,3)   Mean ±1σ, ±2σ, ±3σ along PCA mode K.
 *
 *   H_ModelSamples      5 random instances drawn from the SSM.
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
    println("  Scapula SSM — Full Pipeline Visual Inspection")
    println("=" * 72)

    val ui = ScalismoUI("Scapula SSM — All Intermediate Steps")

    // ── A. Raw input meshes ────────────────────────────────────────────────────
    println("\n[A] Input meshes — original resolution, unaligned")
    val (lmMap, _, _) = ScapulaData.readLandmarkCsv(ScapulaData.csvFile(dataDir))
    val specimens     = ScapulaData.specimens(dataDir)
    val inputGrp      = ui.createGroup(s"A_InputMeshes (${specimens.length} raw — UNALIGNED)")
    specimens.foreach { spec =>
      val raw  = ScapulaData.loadMesh(spec.file)
      val mesh = if (spec.isRight) ScapulaData.mirrorMesh(raw) else raw
      ui.show(inputGrp, mesh, spec.modelId + (if (spec.isRight) "_R→L" else ""))
    }
    println(s"  ${specimens.length} meshes at original resolution. EXPECT: scattered/unaligned.")

    // ── A1. Landmarks ──────────────────────────────────────────────────────────
    println("\n[A1] Landmarks — anatomical control points on each specimen")
    val lmGrp = ui.createGroup("A1_Landmarks (first 6 specimens shown)")
    specimens.filter(s => lmMap.contains(s.modelId)).take(6).foreach { spec =>
      val lms = if (spec.isRight) ScapulaData.mirrorLandmarks(lmMap(spec.modelId))
                else lmMap(spec.modelId)
      ui.show(lmGrp, lms, s"LM_${spec.modelId}")
    }
    println("  Verify: each landmark should sit at its correct anatomical site.")

    // ── A2. Reference mesh ─────────────────────────────────────────────────────
    println("\n[A2] Reference mesh — 8k-vertex decimated specimen used as Pass 1 reference")
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
    val refGrp = ui.createGroup(s"A2_Reference (${refSpec.modelId}, ${refMesh8k.pointSet.numberOfPoints} pts)")
    ui.show(refGrp, refMesh8k, s"REF_${refSpec.modelId}")
    ui.show(refGrp, refLms,    s"REF_LM_${refSpec.modelId}")
    println(s"  Reference: ${refSpec.modelId}  (${refMesh8k.pointSet.numberOfPoints} vertices after decimation)")

    // ── A3. Rigid-aligned meshes (landmark + ICP, BEFORE non-rigid) ───────────
    println("\n[A3] Rigid alignment — after landmark Procrustes + trimmed ICP (BEFORE GP-ICP)")
    val pass1Dir     = new File(baseDir, "pass1")
    val rigidFiles   = filesIn(pass1Dir, "rigid_")
    if (rigidFiles.nonEmpty) {
      val rigidGrp = ui.createGroup(
        s"A3_RigidAligned (${rigidFiles.length} meshes, landmark+ICP only — BEFORE non-rigid)")
      rigidFiles.take(12).foreach { f =>
        ui.show(rigidGrp, ScapulaData.loadMesh(f),
          f.getName.stripSuffix(".stl").stripPrefix("rigid_"))
      }
      println(s"  ${rigidFiles.length} rigid-aligned meshes loaded (showing first 12).")
      println("  EXPECT: all bones roughly overlapping in the same anatomical frame.")
      println("  If still scattered → landmark quality needs checking (see A1_Landmarks).")
    } else {
      println("  WARNING: no rigid_*.stl files found in pass1/")
      println("  Re-run RebuildSSM to generate them, then re-open ViewSSM.")
    }

    // ── B. Non-rigidly registered meshes per pass ──────────────────────────────
    val passDirs = (1 to 8).map(n => n -> new File(baseDir, s"pass$n")).filter(_._2.isDirectory)
    passDirs.foreach { case (n, passDir) =>
      val files = filesIn(passDir, "reg_")
      if (files.nonEmpty) {
        println(s"\n[B] Pass $n — non-rigidly registered (${files.length} meshes, showing first 8)")
        val grp = ui.createGroup(
          s"B_Pass${n}_Registered (${files.length} meshes, σ=${Config.kernelSigma.toInt}mm)")
        files.take(8).foreach { f =>
          ui.show(grp, ScapulaData.loadMesh(f),
            f.getName.stripSuffix(".stl").stripPrefix("reg_"))
        }
        println(s"  EXPECT: tighter overlap than A3_RigidAligned.")
      }
    }

    // ── C. Mean shapes ─────────────────────────────────────────────────────────
    println("\n[C] Mean shapes — GPA convergence")
    val meansGrp    = ui.createGroup("C_Means (Mean1–Mean4 overlaid, < 1 mm shift = converged)")
    val loadedMeans = scala.collection.mutable.ArrayBuffer.empty[(Int, TriangleMesh[_3D])]
    (1 to 8).foreach { n =>
      val f = new File(baseDir, s"mean_pass$n.stl")
      if (f.exists()) {
        val m = ScapulaData.loadMesh(f)
        ui.show(meansGrp, m, s"Mean$n")
        loadedMeans += (n -> m)
        println(s"  Mean$n: ${m.pointSet.numberOfPoints} vertices")
      }
    }
    if (loadedMeans.length >= 2) {
      println(f"\n  ${"Comparison"}%-14s  ${"Mean (mm)"}%10s  ${"RMS (mm)"}%9s  ${"HD95 (mm)"}%10s  Status")
      println("  " + "─" * 60)
      loadedMeans.sliding(2).foreach { w =>
        val (n1, m1) = w(0); val (n2, m2) = w(1)
        val st = Metrics.symmetric(m1, m2)
        val ok = if (st.mean < 1.0) "✓ converged" else "  not yet "
        println(f"  Mean$n1 ↔ Mean$n2          ${st.mean}%10.3f  ${st.rms}%9.3f  ${st.hd95}%10.3f  $ok")
      }
    }

    // ── D. Distance maps (registered vs mean) ─────────────────────────────────
    passDirs.lastOption.foreach { case (n, passDir) =>
      val files = filesIn(passDir, "reg_")
      if (files.nonEmpty) {
        val meanMeshOpt: Option[TriangleMesh[_3D]] =
          loadedMeans.lastOption.map(_._2).orElse {
            val meshes = files.map(ScapulaData.loadMesh)
            val dc     = DataCollection.fromTriangleMesh3DSequence(meshes.head, meshes)
            Some(PointDistributionModel.createUsingPCA(dc).mean)
          }
        meanMeshOpt.foreach { mean =>
          println(s"\n[D] Distance check: pass $n registered vs mean (first 4)")
          val distGrp = ui.createGroup(s"D_DistMap_Pass${n} (registered meshes vs mean)")
          ui.show(distGrp, mean, s"REFERENCE_mean_pass$n")
          files.take(4).foreach { f =>
            val reg  = ScapulaData.loadMesh(f)
            val name = f.getName.stripSuffix(".stl").stripPrefix("reg_")
            val d    = Metrics.surfaceDistances(reg, mean)
            val avg  = d.sum / d.length
            println(f"    $name : avg dist = ${avg}%.3f mm")
            ui.show(distGrp, reg, s"${name}_vs_mean")
          }
        }
      }
    }

    // ── E + G + F + H: build SSM from last pass ────────────────────────────────
    passDirs.lastOption.foreach { case (n, passDir) =>
      val files = filesIn(passDir, "reg_")
      if (files.nonEmpty) {
        println(s"\n[E] Building SSM from pass $n (${files.length} meshes)…")
        val meshes = files.map(ScapulaData.loadMesh)
        val dc     = DataCollection.fromTriangleMesh3DSequence(meshes.head, meshes)
        val model  = PointDistributionModel.createUsingPCA(dc)
        val evs    = model.gp.klBasis.map(_.eigenvalue).toArray
        val total  = evs.sum
        val cumV   = evs.scanLeft(0.0)(_ + _).tail.map(_ / total * 100.0)
        def mFor(p: Double) = { val i = cumV.indexWhere(_ >= p); if (i < 0) model.rank else i + 1 }

        // Eigenvalues / variance table
        println(f"  rank=${model.rank}  90%%: ${mFor(90)} modes  95%%: ${mFor(95)}  99%%: ${mFor(99)}")
        println(f"\n  ${"Mode"}%5s  ${"Eigenvalue"}%12s  ${"Var %%"}%8s  ${"Cumul %%"}%9s  ${"σ (mm)"}%8s")
        println("  " + "─" * 50)
        evs.take(math.min(10, model.rank)).zipWithIndex.foreach { case (ev, i) =>
          println(f"  ${i+1}%5d  ${ev}%12.3f  ${ev/total*100}%8.2f  ${cumV(i)}%9.2f  ${math.sqrt(ev)}%8.3f")
        }

        // ── E. Interactive SSM
        val ssmGrp = ui.createGroup(s"E_SSM_Interactive (pass $n — drag Mode sliders!)")
        ui.show(ssmGrp, model, s"SSM_pass$n")
        println(s"\n  → In group E_SSM_Interactive: click the mesh → drag Mode 0 slider")
        println(s"     Mode 0 = ${evs(0)/total*100.0}%.1f%% of shape variance")

        val mean = model.mean

        // ── G. Modes of variation at ±1σ / ±2σ / ±3σ
        println(s"\n[G] PCA modes of variation — top 3 modes at ±1σ, ±2σ, ±3σ")
        val nModes = math.min(3, model.rank)
        (0 until nModes).foreach { modeIdx =>
          val ev     = evs(modeIdx)
          val sigma  = math.sqrt(ev)
          val varPct = ev / total * 100.0

          def c(s: Double) = DenseVector(Array.tabulate(model.rank)(j =>
            if (j == modeIdx) s * sigma else 0.0))

          val modeGrp = ui.createGroup(
            s"G_Mode${modeIdx+1} (${varPct.toInt}%% var, 1σ=${sigma.toInt}mm — toggle to compare)")
          ui.show(modeGrp, mean,                 s"Mode${modeIdx+1}_mean")
          ui.show(modeGrp, model.instance(c( 1.0)), s"Mode${modeIdx+1}_+1sigma")
          ui.show(modeGrp, model.instance(c(-1.0)), s"Mode${modeIdx+1}_-1sigma")
          ui.show(modeGrp, model.instance(c( 2.0)), s"Mode${modeIdx+1}_+2sigma")
          ui.show(modeGrp, model.instance(c(-2.0)), s"Mode${modeIdx+1}_-2sigma")
          ui.show(modeGrp, model.instance(c( 3.0)), s"Mode${modeIdx+1}_+3sigma")
          ui.show(modeGrp, model.instance(c(-3.0)), s"Mode${modeIdx+1}_-3sigma")
          println(f"  Mode ${modeIdx+1}: σ=${sigma}%.2f mm, var=${varPct}%.2f%% — shown at ±1σ ±2σ ±3σ")
        }

        // ── F. Outliers
        println(s"\n[F] Outlier check — worst-fitting specimens to SSM mean")
        val ranked = meshes.zip(files).map { case (m, f) =>
          val d = Metrics.surfaceDistances(m, mean)
          (d.sum / d.length, m, f.getName.stripSuffix(".stl").stripPrefix("reg_"))
        }.sortBy(-_._1)
        val all = ranked.map(_._1)
        println(f"  Population: mean=${all.sum/all.length}%.3f  max=${all.max}%.3f  min=${all.min}%.3f  mm")
        val outlGrp = ui.createGroup("F_Outliers (3 worst vs SSM mean — investigate these)")
        ui.show(outlGrp, mean, "SSM_mean")
        ranked.take(3).foreach { case (d, m, name) =>
          println(f"    [!]  $name%-42s  $d%.3f mm")
          ui.show(outlGrp, m, f"OUTLIER_${d}%.2fmm_$name")
        }
        println("  Best 3 (well-registered):")
        ranked.takeRight(3).reverse.foreach { case (d, _, name) =>
          println(f"    [✓]  $name%-42s  $d%.3f mm")
        }

        // ── H. Random model samples (model instances from SSM)
        println(s"\n[H] Random model samples — 5 random instances drawn from SSM")
        val sampGrp = ui.createGroup("H_ModelSamples (5 random SSM instances)")
        (1 to 5).foreach { i =>
          ui.show(sampGrp, model.sample(), s"Sample_$i")
        }
        println("  These are random scapula shapes the SSM considers plausible.")
        println("  Compare with G_ModeN to see structured vs random variation.")
      }
    }

    println("\n" + "=" * 72)
    println("  HOW TO READ EACH GROUP")
    println("=" * 72)
    println("  A_InputMeshes   : raw bones scattered — unaligned is CORRECT here")
    println("  A1_Landmarks    : check each landmark sits at the right anatomical point")
    println("  A2_Reference    : the 8k-vertex starting reference for pass 1")
    println("  A3_RigidAligned : after landmark+ICP — should ALL overlap tightly")
    println("                    if not → fix landmark placement and re-run")
    println("  B_PassN         : after GP-ICP non-rigid — tighter than A3")
    println("  C_Means         : Mean1–Mean4 overlaid — < 1 mm shift = converged")
    println("  D_DistMap       : registered meshes next to mean for visual comparison")
    println("  E_SSM           : click → drag Mode 0 slider → live shape morphing")
    println("  F_Outliers      : worst 3 — re-register or exclude if > 3× population avg")
    println("  G_ModeK         : toggle ±1σ/±2σ/±3σ to see physical deformation per mode")
    println("  H_ModelSamples  : 5 random SSM instances — test model plausibility")
    println("=" * 72)
    println("\nUI open. Close the window to exit.")
  }
}
