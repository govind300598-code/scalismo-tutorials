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
 * Scapula SSM — Full Pipeline Viewer
 *
 * Groups are loaded in the order that makes sense for inspection:
 * aligned/registered results FIRST, raw input LAST.
 *
 * Scene panel groups (hide/show with the eye icon):
 *
 *   G01_LandmarkAligned  – Procrustes (landmark-only, live) — ALWAYS available
 *   G02_RigidAligned     – landmark + ICP (from pipeline output)
 *   G03_NonRigid_Pass1   – GP-ICP pass 1
 *   G04_NonRigid_Final   – GP-ICP final pass (best correspondences)
 *   G05_MeanShapes       – Mean1–4 overlaid, convergence check
 *   G06_SSM_Interactive  – drag Mode sliders to explore shape space
 *   G07_Mode1 / G08_Mode2 / G09_Mode3
 *                        – ±1σ / ±2σ / ±3σ for first 3 modes
 *   G10_ModelSamples     – 5 random SSM instances
 *   G11_Landmarks        – GC/TS/IA/PLA/AC on every specimen (raw space)
 *   G12_Reference        – reference mesh + landmarks
 *   G13_RawInput         – original unaligned meshes (always scattered — LAST)
 *
 * Run with:
 *   SCAPULA_DATA_DIR=... SCAPULA_OUT_DIR=... sbt "runMain scapula.VisualizationApp"
 */
object VisualizationApp {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dataDir    = Config.dataDir
    val outDir     = Config.outDir
    val resultsDir = new File(outDir, "results")
    val preDir     = new File(outDir, "data/preprocessing/8k")

    val useNewLayout = new File(resultsDir, "SSM1").isDirectory
    println(s"Layout: ${if (useNewLayout) "new (results/SSM{n}/)" else "old (pass{n}/)"}")

    // ── Load landmarks + specimens ──────────────────────────────────────────
    val csvFile    = ScapulaData.csvFile(dataDir)
    val (lmMap, fromHeader, _) = ScapulaData.readLandmarkCsv(csvFile)
    if (!fromHeader)
      println("WARNING: landmark columns resolved by fallback offsets, not header names")

    val allSpecs   = ScapulaData.specimens(dataDir)
    val validSpecs = allSpecs.filter(s => lmMap.contains(s.modelId))
    println(s"Specimens: ${allSpecs.length} total, ${validSpecs.length} with landmarks")

    // Reference
    val refSpec = validSpecs(Config.refIdx.min(validSpecs.length - 1))
    val refLms  = if (refSpec.isRight) ScapulaData.mirrorLandmarks(lmMap(refSpec.modelId))
                  else lmMap(refSpec.modelId)
    println(s"Reference: ${refSpec.modelId}")

    // ── Helpers ─────────────────────────────────────────────────────────────
    def stlsIn(dir: File, prefix: String = ""): IndexedSeq[File] =
      if (!dir.isDirectory) IndexedSeq.empty
      else Option(dir.listFiles()).getOrElse(Array.empty[File])
        .filter(f => f.getName.endsWith(".stl") && f.getName.startsWith(prefix))
        .sortBy(_.getName).toIndexedSeq

    def loadOpt(f: File): Option[TriangleMesh[_3D]] =
      if (f.exists()) scala.util.Try(ScapulaData.loadMesh(f)).toOption else None

    def loadModelOpt(f: File): Option[StatisticalMeshModel] =
      if (!f.exists()) None
      else scalismo.io.StatisticalModelIO.readStatisticalMeshModel(f).toOption

    // Use decimated 8k mesh if available, otherwise full-res
    def loadWorkingMesh(s: ScapulaData.Specimen): TriangleMesh[_3D] = {
      val dec8k = new File(preDir, s.modelId + ".stl")
      val mesh  = if (dec8k.exists()) ScapulaData.loadMesh(dec8k)
                  else ScapulaData.loadMesh(s.file)
      if (s.isRight) ScapulaData.mirrorMesh(mesh) else mesh
    }

    def specimenLms(s: ScapulaData.Specimen): IndexedSeq[scalismo.geometry.Landmark[_3D]] =
      if (s.isRight) ScapulaData.mirrorLandmarks(lmMap(s.modelId)) else lmMap(s.modelId)

    val ui = ScalismoUI("Scapula SSM — Full Pipeline Viewer")

    // ════════════════════════════════════════════════════════════════════════
    // G01 — Landmark-only Procrustes (computed live, NO pipeline files needed)
    //        ALL bones should roughly overlap here.
    //        If scattered → landmark placement is wrong.
    // ════════════════════════════════════════════════════════════════════════
    println("\n[G01] Landmark-only Procrustes alignment (computed live)")
    val g01 = ui.createGroup("G01_LandmarkAligned (Procrustes only — should overlap roughly)")
    var g01Count = 0
    validSpecs.foreach { s =>
      val mesh  = loadWorkingMesh(s)
      val lms   = specimenLms(s)
      val trans = ScapulaData.rigidFromLandmarks(lms, refLms)
      ui.show(g01, mesh.transform(trans), s.modelId)
      g01Count += 1
    }
    println(s"  $g01Count specimens loaded  (ref: ${refSpec.modelId})")
    println("  Bones should be ROUGHLY overlapping — if scattered, landmarks are wrong")

    // ════════════════════════════════════════════════════════════════════════
    // G02 — Rigid-aligned (landmark + ICP) from pipeline output
    //        ALL bones must tightly overlap here.
    // ════════════════════════════════════════════════════════════════════════
    println("\n[G02] Rigid-aligned (landmark + ICP) — from pipeline")
    val rigidFiles: IndexedSeq[File] =
      if (useNewLayout) stlsIn(new File(resultsDir, "SSM1/rigid_registered"))
      else stlsIn(new File(outDir, "pass1"), prefix = "rigid_")

    if (rigidFiles.isEmpty) {
      println("  NOT FOUND — run 'sbt runMain scapula.Main' first")
    } else {
      val g02 = ui.createGroup(s"G02_RigidAligned (${rigidFiles.length} specimens — ALL must overlap)")
      rigidFiles.foreach { f =>
        val name = f.getName.stripSuffix(".stl").stripPrefix("rigid_")
        ui.show(g02, ScapulaData.loadMesh(f), name)
      }
      println(s"  ${rigidFiles.length} rigid-aligned meshes")
      println("  Bones MUST overlap here — if not, ICP failed or landmarks are wrong")
    }

    // ════════════════════════════════════════════════════════════════════════
    // G03 — Non-rigid, Pass 1 (GP-ICP)
    // ════════════════════════════════════════════════════════════════════════
    val nrPass1: IndexedSeq[File] =
      if (useNewLayout) stlsIn(new File(resultsDir, "SSM1/nonrigid_registered"))
      else stlsIn(new File(outDir, "pass1"), prefix = "reg_")

    if (nrPass1.nonEmpty) {
      println(s"\n[G03] Non-rigid registered — Pass 1 (${nrPass1.length} specimens)")
      val g03 = ui.createGroup(s"G03_NonRigid_Pass1 (${nrPass1.length} specimens, GP-ICP reg)")
      nrPass1.foreach { f =>
        val name = f.getName.stripSuffix(".stl").stripPrefix("reg_")
        ui.show(g03, ScapulaData.loadMesh(f), name)
      }
    }

    // ════════════════════════════════════════════════════════════════════════
    // G04 — Non-rigid, final pass (best correspondences)
    // ════════════════════════════════════════════════════════════════════════
    val lastPassN = (1 to 8).reverse.find { n =>
      val d = if (useNewLayout) new File(resultsDir, s"SSM$n/nonrigid_registered")
              else new File(outDir, s"pass$n")
      d.isDirectory && stlsIn(d, if (useNewLayout) "" else "reg_").nonEmpty
    }

    val lastPassFiles: IndexedSeq[File] = lastPassN.map { n =>
      val d = if (useNewLayout) new File(resultsDir, s"SSM$n/nonrigid_registered")
              else new File(outDir, s"pass$n")
      stlsIn(d, if (useNewLayout) "" else "reg_")
    }.getOrElse(IndexedSeq.empty)

    if (lastPassFiles.nonEmpty) {
      val n = lastPassN.get
      println(s"\n[G04] Non-rigid registered — Pass $n (FINAL — ${lastPassFiles.length} specimens)")
      val g04 = ui.createGroup(s"G04_NonRigid_Pass$n (${lastPassFiles.length} specimens — FINAL GP-ICP)")
      lastPassFiles.foreach { f =>
        val name = f.getName.stripSuffix(".stl").stripPrefix("reg_")
        ui.show(g04, ScapulaData.loadMesh(f), name)
      }
      if (rigidFiles.nonEmpty) {
        val d = Metrics.symmetric(ScapulaData.loadMesh(rigidFiles.head), ScapulaData.loadMesh(lastPassFiles.head))
        println(f"  Rigid vs NonRigid (specimen 1): ${d.render}")
      }
    }

    // ════════════════════════════════════════════════════════════════════════
    // G05 — Mean shapes (convergence check)
    // ════════════════════════════════════════════════════════════════════════
    println("\n[G05] Mean shapes — convergence")
    val g05 = ui.createGroup("G05_MeanShapes (Mean1–4 overlaid — <1 mm shift = converged)")
    val loadedMeans = (1 to 8).flatMap { n =>
      val f = if (useNewLayout) new File(resultsDir, s"SSM$n/mean/SSM${n}_mean.stl")
              else new File(outDir, s"mean_pass$n.stl")
      loadOpt(f).map { m => ui.show(g05, m, s"Mean$n"); n -> m }
    }
    if (loadedMeans.length >= 2) {
      println(f"  ${"Pair"}%-16s  ${"Mean mm"}%9s  Status")
      loadedMeans.sliding(2).foreach { w =>
        val (n1, m1) = w(0); val (n2, m2) = w(1)
        val st = Metrics.symmetric(m1, m2)
        println(f"  Mean$n1 ↔ Mean$n2       ${st.mean}%9.3f  ${if (st.mean < 1.0) "CONVERGED" else "not yet"}")
      }
    } else println("  No mean shapes found — run pipeline first")

    // ════════════════════════════════════════════════════════════════════════
    // G06–G10 — SSM (interactive + modes + samples)
    // ════════════════════════════════════════════════════════════════════════
    val ssmOpt: Option[StatisticalMeshModel] = lastPassN.flatMap { n =>
      val h5 = if (useNewLayout) new File(resultsDir, s"SSM$n/model/SSM$n.h5") else new File("")
      loadModelOpt(h5).orElse {
        if (lastPassFiles.isEmpty) None
        else {
          println(s"\nBuilding SSM from ${lastPassFiles.length} meshes (no .h5 found)...")
          val meshes = lastPassFiles.map(ScapulaData.loadMesh)
          if (meshes.length < 3) None
          else {
            val dc = DataCollection.fromTriangleMesh3DSequence(meshes.head, meshes)
            StatisticalMeshModel.createUsingPCA(dc).toOption
          }
        }
      }
    }

    ssmOpt match {
      case None =>
        println("\nNo SSM available — run pipeline first")
      case Some(ssm) =>
        val evs   = ssm.gp.klBasis.map(_.eigenvalue)
        val total = evs.sum
        println(s"\nSSM rank=${ssm.rank}  (from pass ${lastPassN.getOrElse("?")})")
        println(f"  ${"Mode"}%5s  ${"Var%%"}%8s  ${"Cumul%%"}%8s  ${"σ mm"}%8s")
        evs.take(math.min(10, ssm.rank)).zipWithIndex.foreach { case (ev, i) =>
          println(f"  ${i+1}%5d  ${ev/total*100}%8.2f  ${evs.take(i+1).sum/total*100}%8.2f  ${math.sqrt(ev)}%8.3f")
        }
        val meanMesh = ssm.mean
        val nModes   = math.min(3, ssm.rank)

        // G06 — Interactive
        println("\n[G06] Interactive SSM — select the group, then drag Mode sliders (right panel)")
        val g06 = ui.createGroup("G06_SSM_Interactive (select → drag Mode sliders)")
        ui.show(g06, ssm, "SSM_final")

        // G07/G08/G09 — Static mode shapes
        (0 until nModes).foreach { modeIdx =>
          val sigma  = math.sqrt(evs(modeIdx))
          val varPct = evs(modeIdx) / total * 100.0
          val gNum   = 7 + modeIdx
          println(s"\n[G0$gNum] Mode ${modeIdx+1}: σ=${sigma.toInt}mm  var=${varPct.toInt}%%")
          val mGrp = ui.createGroup(
            s"G0${gNum}_Mode${modeIdx+1} (σ=${sigma.toInt}mm, ${varPct.toInt}%% var) ±1σ/±2σ/±3σ")

          def coeff(alpha: Double): DenseVector[Double] =
            DenseVector(Array.tabulate(ssm.rank)(j => if (j == modeIdx) alpha * sigma else 0.0))

          ui.show(mGrp, meanMesh,                 s"Mode${modeIdx+1}_mean")
          ui.show(mGrp, ssm.instance(coeff( 1.0)), s"Mode${modeIdx+1}_+1sigma")
          ui.show(mGrp, ssm.instance(coeff(-1.0)), s"Mode${modeIdx+1}_-1sigma")
          ui.show(mGrp, ssm.instance(coeff( 2.0)), s"Mode${modeIdx+1}_+2sigma")
          ui.show(mGrp, ssm.instance(coeff(-2.0)), s"Mode${modeIdx+1}_-2sigma")
          ui.show(mGrp, ssm.instance(coeff( 3.0)), s"Mode${modeIdx+1}_+3sigma")
          ui.show(mGrp, ssm.instance(coeff(-3.0)), s"Mode${modeIdx+1}_-3sigma")
        }

        // G10 — Random samples
        println("\n[G10] 5 random SSM instances")
        val g10 = ui.createGroup("G10_ModelSamples (5 random instances)")
        (1 to 5).foreach { i => ui.show(g10, ssm.sample(), s"Sample_$i") }
    }

    // ════════════════════════════════════════════════════════════════════════
    // G11 — Landmarks (raw space) — loaded late so it doesn't appear first
    // ════════════════════════════════════════════════════════════════════════
    println("\n[G11] Landmarks on all specimens")
    val g11 = ui.createGroup("G11_Landmarks (GC/TS/IA/PLA/AC on every specimen, raw space)")
    validSpecs.foreach { s => ui.show(g11, specimenLms(s), s"LM_${s.modelId}") }
    println(s"  ${validSpecs.length} landmark sets (raw world space, not aligned)")

    // ════════════════════════════════════════════════════════════════════════
    // G12 — Reference mesh + landmarks
    // ════════════════════════════════════════════════════════════════════════
    println(s"\n[G12] Reference: ${refSpec.modelId}")
    val g12     = ui.createGroup(s"G12_Reference (${refSpec.modelId})")
    val refMesh = loadWorkingMesh(refSpec)
    ui.show(g12, refMesh, "REF_mesh")
    ui.show(g12, refLms,  "REF_landmarks")
    println(f"  ${refMesh.pointSet.numberOfPoints} vertices")
    refLms.foreach { lm =>
      println(f"  ${lm.id}%6s  (${lm.point.x}%.1f, ${lm.point.y}%.1f, ${lm.point.z}%.1f)")
    }

    // ════════════════════════════════════════════════════════════════════════
    // G13 — Raw input (LAST — always scattered, use decimated meshes)
    // ════════════════════════════════════════════════════════════════════════
    println(s"\n[G13] Raw unaligned input (LAST group — always scattered)")
    val g13 = ui.createGroup("G13_RawInput (UNALIGNED — always scattered, scroll past this)")
    allSpecs.foreach { s =>
      // Use 8k decimated mesh if available to keep rendering clean
      val dec = new File(preDir, s.modelId + ".stl")
      val mesh = if (dec.exists()) ScapulaData.loadMesh(dec)
                 else ScapulaData.loadMesh(s.file)
      val finalMesh = if (s.isRight) ScapulaData.mirrorMesh(mesh) else mesh
      ui.show(g13, finalMesh, s.modelId + (if (s.isRight) "_mirrored" else ""))
    }
    println(s"  ${allSpecs.length} raw meshes (${if (preDir.isDirectory) "8k decimated" else "full-res"})")

    // ════════════════════════════════════════════════════════════════════════
    // Guide
    // ════════════════════════════════════════════════════════════════════════
    println("\n" + "═" * 72)
    println("  VIEWER GUIDE — groups listed in inspection order")
    println("═" * 72)
    println("  G01_LandmarkAligned  → roughly overlapping (landmark Procrustes)")
    println("  G02_RigidAligned     → ALL bones must overlap tightly (ICP result)")
    println("  G03_NonRigid_Pass1   → GP-ICP pass 1 (tighter than G02)")
    println("  G04_NonRigid_Final   → BEST registration (use for publication)")
    println("  G05_MeanShapes       → convergence: <1 mm shift between passes")
    println("  G06_SSM_Interactive  → select group → drag Mode sliders on right")
    println("  G07/G08/G09_Mode1-3  → ±1σ/±2σ/±3σ static shapes")
    println("  G10_ModelSamples     → 5 random instances")
    println("  G11_Landmarks        → raw space landmark positions")
    println("  G12_Reference        → reference mesh + landmarks")
    println("  G13_RawInput         → RAW UNALIGNED — always scattered (expected)")
    println("═" * 72)
  }
}
