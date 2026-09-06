package scapula

import breeze.linalg.DenseVector
import scalismo.geometry._3D
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.StatisticalMeshModel
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

import java.io.File

/**
 * Comprehensive step-by-step pipeline viewer.
 *
 * Groups loaded in pipeline order — use the eye icon in the Scene panel to
 * show or hide individual groups:
 *
 *   S00_RawInput          – original unaligned meshes (R ones mirrored to L)
 *   S01_Landmarks         – GC / TS / IA / PLA / AC on every specimen
 *   S02_Reference         – reference mesh + its landmarks
 *   S03_LandmarkAligned   – Procrustes (landmark-only, NO ICP) — shows
 *                           whether landmarks are correct before ICP runs
 *   S04_RigidAligned      – landmark Procrustes + trimmed ICP (ALL specimens)
 *                           ALL bones should overlap here
 *   S05_NonRigid_Pass1    – GP-ICP registered pass 1 (ALL specimens)
 *   S06_NonRigid_LastPass – GP-ICP registered last pass (ALL specimens)
 *   S07_MeanShapes        – Mean1…Mean4 overlaid (convergence check)
 *   S08_SSM_Interactive   – drag Mode sliders on the right panel
 *   S09_Mode1 / S10_Mode2 / S11_Mode3
 *                         – static ±1σ / ±2σ / ±3σ meshes for modes 1-3
 *   S12_ModelSamples      – 5 random SSM instances
 *
 * Run with:
 *   SCAPULA_OUT_DIR=... SCAPULA_DATA_DIR=... sbt "runMain scapula.VisualizationApp"
 */
object VisualizationApp {

  private def stlsIn(dir: File, prefix: String = ""): IndexedSeq[File] =
    if (!dir.isDirectory) IndexedSeq.empty
    else Option(dir.listFiles()).getOrElse(Array.empty[File])
      .filter(f => f.getName.endsWith(".stl") && f.getName.startsWith(prefix))
      .sortBy(_.getName).toIndexedSeq

  private def loadOpt(f: File): Option[TriangleMesh[_3D]] =
    if (f.exists()) Some(ScapulaData.loadMesh(f)) else None

  private def loadModelOpt(f: File): Option[StatisticalMeshModel] =
    if (!f.exists()) None
    else scalismo.io.StatisticalModelIO.readStatisticalMeshModel(f).toOption

  private def buildSSM(meshes: IndexedSeq[TriangleMesh[_3D]]): Option[StatisticalMeshModel] = {
    if (meshes.length < 3) return None
    val dc = DataCollection.fromTriangleMesh3DSequence(meshes.head, meshes)
    StatisticalMeshModel.createUsingPCA(dc).toOption
  }

  // Show a group of meshes; returns number loaded.
  private def showMeshes(
    ui: ScalismoUI,
    groupName: String,
    files: IndexedSeq[File],
    stripPrefix: String = ""
  ): Int = {
    if (files.isEmpty) { println(s"  $groupName: no files found — skipping"); return 0 }
    val grp = ui.createGroup(groupName)
    files.foreach { f =>
      val name = f.getName.stripSuffix(".stl").stripPrefix(stripPrefix)
      ui.show(grp, ScapulaData.loadMesh(f), name)
    }
    println(s"  $groupName: ${files.length} meshes")
    files.length
  }

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = scalismo.utils.Random(Config.seed)

    val dataDir    = Config.dataDir
    val outDir     = Config.outDir
    val resultsDir = new File(outDir, "results")

    // Detect old (pass{n}/) vs new (results/SSM{n}/) layout
    val useNewLayout = new File(resultsDir, "SSM1").isDirectory
    println(s"Layout: ${if (useNewLayout) "new (results/SSM{n}/)" else "old (pass{n}/)"}")

    val ui = ScalismoUI("Scapula SSM — Full Pipeline Viewer")

    // ── Load landmarks + specimens ────────────────────────────────────────────
    val csvFile   = ScapulaData.csvFile(dataDir)
    val (lmMap, fromHeader, _) = ScapulaData.readLandmarkCsv(csvFile)
    if (!fromHeader) println("WARNING: landmark columns resolved by fallback offsets, not header names")
    val allSpecs  = ScapulaData.specimens(dataDir)
    val validSpecs = allSpecs.filter(s => lmMap.contains(s.modelId))
    println(s"\nSpecimens: ${allSpecs.length} total, ${validSpecs.length} with landmarks")

    // Helper: mirror landmarks for right scapulae
    def specimenLms(s: ScapulaData.Specimen): IndexedSeq[scalismo.geometry.Landmark[_3D]] =
      if (s.isRight) ScapulaData.mirrorLandmarks(lmMap(s.modelId)) else lmMap(s.modelId)

    def specimenMesh(s: ScapulaData.Specimen): TriangleMesh[_3D] = {
      val raw = ScapulaData.loadMesh(s.file)
      if (s.isRight) ScapulaData.mirrorMesh(raw) else raw
    }

    // Pick reference specimen
    val refSpec  = validSpecs(Config.refIdx.min(validSpecs.length - 1))
    val refMesh  = specimenMesh(refSpec)
    val refLms   = specimenLms(refSpec)
    println(s"Reference: ${refSpec.modelId} (index ${Config.refIdx})")

    // ════════════════════════════════════════════════════════════════
    // S00 — Raw unaligned input
    // ════════════════════════════════════════════════════════════════
    println("\n[S00] Raw unaligned input")
    val s00 = ui.createGroup("S00_RawInput (unaligned — deliberately scattered)")
    allSpecs.foreach { s =>
      val mesh = specimenMesh(s)
      val tag  = s.modelId + (if (s.isRight) "_mirrored" else "")
      ui.show(s00, mesh, tag)
    }
    println(s"  ${allSpecs.length} meshes loaded (R ones reflected to L frame)")

    // ════════════════════════════════════════════════════════════════
    // S01 — Landmarks on each specimen (in their own space)
    // ════════════════════════════════════════════════════════════════
    println("\n[S01] Landmarks on specimens")
    val s01 = ui.createGroup("S01_Landmarks (GC/TS/IA/PLA/AC on every specimen)")
    validSpecs.foreach { s =>
      val lms = specimenLms(s)
      ui.show(s01, lms, s"LM_${s.modelId}")
    }
    println(s"  ${validSpecs.length} landmark sets loaded")
    println(s"  Landmark names: ${ScapulaData.landmarkNames.mkString(", ")}")
    println(s"  CSV: ${csvFile.getName}  (columns from ${if (fromHeader) "header" else "fallback offsets"})")

    // ════════════════════════════════════════════════════════════════
    // S02 — Reference mesh + landmarks
    // ════════════════════════════════════════════════════════════════
    println(s"\n[S02] Reference: ${refSpec.modelId}")
    val s02 = ui.createGroup(s"S02_Reference (${refSpec.modelId})")
    ui.show(s02, refMesh, "REF_mesh")
    ui.show(s02, refLms,  "REF_landmarks")
    println(f"  ${refMesh.pointSet.numberOfPoints} vertices")
    refLms.foreach { lm => println(f"  ${lm.id}%6s  (${lm.point.x}%.1f, ${lm.point.y}%.1f, ${lm.point.z}%.1f)") }

    // ════════════════════════════════════════════════════════════════
    // S03 — Landmark-only alignment (Procrustes, NO ICP)
    //        Verify landmark accuracy before ICP runs.
    //        If these are still scattered → landmarks are wrong.
    // ════════════════════════════════════════════════════════════════
    println("\n[S03] Landmark-only Procrustes alignment (no ICP)")
    println("  If still scattered here → check landmark placement in S01.")
    val s03 = ui.createGroup("S03_LandmarkAligned (Procrustes only, NO ICP — check landmarks)")
    validSpecs.foreach { s =>
      val mesh     = specimenMesh(s)
      val lms      = specimenLms(s)
      val trans    = ScapulaData.rigidFromLandmarks(lms, refLms)
      val aligned  = mesh.transform(trans)
      val alignedL = lms.map(lm => lm.copy(point = trans(lm.point)))
      ui.show(s03, aligned,  s"LMonly_${s.modelId}")
      ui.show(s03, alignedL, s"LMonly_${s.modelId}_lm")
    }
    println(s"  ${validSpecs.length} specimens landmark-aligned and loaded with their landmarks")

    // ════════════════════════════════════════════════════════════════
    // S04 — Rigid alignment: landmark Procrustes + trimmed ICP
    //        ALL specimens must overlap here. If not → pipeline problem.
    // ════════════════════════════════════════════════════════════════
    println("\n[S04] Rigid alignment (landmark + ICP)")

    // Old layout: pass1/rigid_*.stl
    val rigidFiles: IndexedSeq[File] = if (useNewLayout)
      stlsIn(new File(resultsDir, "SSM1/rigid_registered"))
    else
      stlsIn(new File(outDir, "pass1"), prefix = "rigid_")

    if (rigidFiles.isEmpty) {
      println("  WARNING: no rigid-aligned STLs found.")
      println(s"  Expected: ${if (useNewLayout) "results/SSM1/rigid_registered/" else "pass1/rigid_*.stl"}")
      println("  Run the pipeline first to generate them.")
    } else {
      val s04 = ui.createGroup(s"S04_RigidAligned (ALL ${rigidFiles.length} specimens — should ALL overlap)")
      rigidFiles.foreach { f =>
        val name = f.getName.stripSuffix(".stl").stripPrefix("rigid_")
        val mesh = ScapulaData.loadMesh(f)
        ui.show(s04, mesh, name)
      }
      println(s"  ${rigidFiles.length} rigid-aligned meshes loaded")
      println("  → If bones are NOT overlapping here, landmarks in S01 are wrong.")
    }

    // ════════════════════════════════════════════════════════════════
    // S05 — Non-rigid registered, Pass 1
    // ════════════════════════════════════════════════════════════════
    println("\n[S05] Non-rigid registration — Pass 1 (GP-ICP)")

    val nrPass1Files: IndexedSeq[File] = if (useNewLayout)
      stlsIn(new File(resultsDir, "SSM1/nonrigid_registered"))
    else
      stlsIn(new File(outDir, "pass1"), prefix = "reg_")

    if (nrPass1Files.nonEmpty) {
      val s05 = ui.createGroup(s"S05_NonRigid_Pass1 (${nrPass1Files.length} specimens, GP-ICP)")
      nrPass1Files.foreach { f =>
        val name = f.getName.stripSuffix(".stl").stripPrefix("reg_")
        ui.show(s05, ScapulaData.loadMesh(f), name)
      }
      println(s"  ${nrPass1Files.length} non-rigid registered meshes (pass 1)")
    }

    // ════════════════════════════════════════════════════════════════
    // S06 — Non-rigid registered, last pass (tightest correspondences)
    // ════════════════════════════════════════════════════════════════
    val lastPassN = (1 to 8).reverse.find { n =>
      val dir = if (useNewLayout) new File(resultsDir, s"SSM$n/nonrigid_registered")
                else new File(outDir, s"pass$n")
      dir.isDirectory && stlsIn(dir, prefix = if (useNewLayout) "" else "reg_").nonEmpty
    }

    lastPassN.foreach { n =>
      println(s"\n[S06] Non-rigid registration — Pass $n (last/best pass)")
      val dir   = if (useNewLayout) new File(resultsDir, s"SSM$n/nonrigid_registered")
                  else new File(outDir, s"pass$n")
      val files = stlsIn(dir, prefix = if (useNewLayout) "" else "reg_")
      val s06   = ui.createGroup(s"S06_NonRigid_Pass$n (${files.length} specimens — FINAL correspondences)")
      files.foreach { f =>
        val name = f.getName.stripSuffix(".stl").stripPrefix("reg_")
        ui.show(s06, ScapulaData.loadMesh(f), name)
      }
      println(s"  ${files.length} non-rigid registered meshes (pass $n)")

      // Print surface distance vs rigid for first specimen (shows how much non-rigid helped)
      if (rigidFiles.nonEmpty && files.nonEmpty) {
        val rgd = ScapulaData.loadMesh(rigidFiles.head)
        val nrd = ScapulaData.loadMesh(files.head)
        val st  = Metrics.symmetric(rgd, nrd)
        println(f"  Rigid vs NonRigid (specimen 1): ${st.render}")
      }
    }

    // ════════════════════════════════════════════════════════════════
    // S07 — Mean shapes pass 1..4 overlaid
    // ════════════════════════════════════════════════════════════════
    println("\n[S07] Mean shapes — convergence check")
    val s07 = ui.createGroup("S07_MeanShapes (Mean1-4 overlaid — < 1 mm shift = converged)")
    val loadedMeans = (1 to 8).flatMap { n =>
      val f = if (useNewLayout) new File(resultsDir, s"SSM$n/mean/SSM${n}_mean.stl")
              else new File(outDir, s"mean_pass$n.stl")
      loadOpt(f).map { m =>
        ui.show(s07, m, s"Mean$n")
        println(s"  Mean$n: ${m.pointSet.numberOfPoints} vertices — ${f.getName}")
        n -> m
      }
    }
    if (loadedMeans.length >= 2) {
      println(f"  ${"Pair"}%-14s  ${"Mean mm"}%9s  ${"RMS mm"}%9s  ${"HD95 mm"}%10s  Status")
      loadedMeans.sliding(2).foreach { w =>
        val (n1, m1) = w(0); val (n2, m2) = w(1)
        val st = Metrics.symmetric(m1, m2)
        val ok = if (st.mean < 1.0) "CONVERGED" else "not yet"
        println(f"  Mean$n1 ↔ Mean$n2     ${st.mean}%9.3f  ${st.rms}%9.3f  ${st.hd95}%10.3f  $ok")
      }
    }

    // ════════════════════════════════════════════════════════════════
    // S08-S12 — Build SSM from last pass for interactive + modes + samples
    // ════════════════════════════════════════════════════════════════
    val ssmOpt: Option[StatisticalMeshModel] = lastPassN.flatMap { n =>
      // Try loading saved model first
      val h5 = if (useNewLayout) new File(resultsDir, s"SSM$n/model/SSM$n.h5")
               else new File("")
      loadModelOpt(h5).orElse {
        // Build on-the-fly from registered meshes
        val dir   = if (useNewLayout) new File(resultsDir, s"SSM$n/nonrigid_registered")
                    else new File(outDir, s"pass$n")
        val files = stlsIn(dir, prefix = if (useNewLayout) "" else "reg_")
        if (files.isEmpty) None
        else {
          println(s"\nBuilding SSM from ${files.length} pass-$n meshes (no .h5 found)...")
          buildSSM(files.map(ScapulaData.loadMesh))
        }
      }
    }

    ssmOpt match {
      case None =>
        println("\nNo SSM could be built — pipeline output not found.")
        println(s"Expected in: ${if (useNewLayout) resultsDir.getPath else outDir.getPath}")

      case Some(ssm) =>
        val evs    = ssm.gp.klBasis.map(_.eigenvalue)
        val total  = evs.sum
        val nModes = math.min(3, ssm.rank)

        // Variance table
        println(s"\nSSM: rank=${ssm.rank}")
        println(f"  ${"Mode"}%5s  ${"Var%%"}%8s  ${"Cumul%%"}%9s  ${"σ mm"}%8s")
        evs.take(math.min(10, ssm.rank)).zipWithIndex.foreach { case (ev, i) =>
          println(f"  ${i+1}%5d  ${ev/total*100}%8.2f  ${evs.take(i+1).sum/total*100}%9.2f  ${math.sqrt(ev)}%8.3f")
        }
        val mean = ssm.mean

        // ── S08: Interactive SSM (drag mode sliders) ────────────────────────
        println(s"\n[S08] Interactive SSM — drag Mode sliders in the right panel")
        val s08 = ui.createGroup("S08_SSM_Interactive (select → drag 'Mode N' sliders on right)")
        ui.show(s08, ssm, "SSM_final")

        // ── S09/S10/S11: Static ±1σ / ±2σ / ±3σ for modes 1-3 ─────────────
        (0 until nModes).foreach { modeIdx =>
          val ev      = evs(modeIdx)
          val sigma   = math.sqrt(ev)
          val varPct  = ev / total * 100.0
          val grpName = s"S${9 + modeIdx}_Mode${modeIdx+1} (${varPct.toInt}%% var, σ=${sigma.toInt}mm)"
          println(s"\n[S${9+modeIdx}] Mode ${modeIdx+1}: σ=${sigma.toInt}mm  var=${varPct.toInt}%%")
          val mGrp = ui.createGroup(grpName)

          def c(s: Double): DenseVector[Double] =
            DenseVector(Array.tabulate(ssm.rank)(j => if (j == modeIdx) s * sigma else 0.0))

          ui.show(mGrp, mean,                    s"Mode${modeIdx+1}_mean")
          ui.show(mGrp, ssm.instance(c( 1.0)),   s"Mode${modeIdx+1}_+1sigma")
          ui.show(mGrp, ssm.instance(c(-1.0)),   s"Mode${modeIdx+1}_-1sigma")
          ui.show(mGrp, ssm.instance(c( 2.0)),   s"Mode${modeIdx+1}_+2sigma")
          ui.show(mGrp, ssm.instance(c(-2.0)),   s"Mode${modeIdx+1}_-2sigma")
          ui.show(mGrp, ssm.instance(c( 3.0)),   s"Mode${modeIdx+1}_+3sigma")
          ui.show(mGrp, ssm.instance(c(-3.0)),   s"Mode${modeIdx+1}_-3sigma")
          println(s"  Loaded: mean + ±1σ + ±2σ + ±3σ (7 meshes)")
        }

        // ── S12: Random model samples ───────────────────────────────────────
        println("\n[S12] 5 random model instances")
        val s12 = ui.createGroup("S12_ModelSamples (5 random SSM instances)")
        (1 to 5).foreach { i => ui.show(s12, ssm.sample(), s"Sample_$i") }
    }

    // ════════════════════════════════════════════════════════════════
    // Summary guide
    // ════════════════════════════════════════════════════════════════
    println("\n" + "═" * 72)
    println("  HOW TO USE THIS VIEWER")
    println("═" * 72)
    println("  S00_RawInput        → scattered unaligned bones (expected)")
    println("  S01_Landmarks       → verify GC/TS/IA/PLA/AC are on the right spots")
    println("  S02_Reference       → reference mesh and its landmarks")
    println("  S03_LandmarkAligned → if still scattered here, landmarks are WRONG")
    println("  S04_RigidAligned    → ALL bones should overlap — verify alignment")
    println("  S05/S06_NonRigid    → tighter overlap than S04 — verify registration")
    println("  S07_MeanShapes      → < 1 mm shift between passes = converged")
    println("  S08_SSM_Interactive → select group → drag 'Mode' sliders right panel")
    println("  S09-S11_Mode1-3     → toggle ±1σ/±2σ/±3σ meshes for each PCA mode")
    println("  S12_ModelSamples    → 5 random instances from the final SSM")
    println("═" * 72)
    println("\nUI open. Close the window to exit.")
  }
}
