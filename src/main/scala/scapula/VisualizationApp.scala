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
 * Shows every stage of the pipeline in a single Scalismo UI window.
 * All groups are in the REFERENCE coordinate space so the camera stays
 * at bone scale (~150 mm) throughout.
 *
 *   G01  Decimated_Aligned      8k meshes, live landmark-Procrustes
 *   G02  RigidAligned           landmark + ICP (from pipeline output)
 *   G03  NonRigidRegistered     GP-ICP, dense correspondence (from pipeline)
 *   G04  MeanShape              SSM mean mesh
 *   G05  SSM_Interactive        select group → drag Mode sliders (right panel)
 *   G06  Mode1 (σ, var%)        ±1σ / ±2σ / ±3σ static shapes for mode 1
 *   G07  Mode2 (σ, var%)        ±1σ / ±2σ / ±3σ static shapes for mode 2
 *   G08  Mode3 (σ, var%)        ±1σ / ±2σ / ±3σ static shapes for mode 3
 *   G09  ModelSamples           5 random SSM instances
 *   G10  Reference              reference mesh + 5 landmarks
 *
 * NOTE: Raw scanner-space meshes are intentionally omitted.
 *       They span ~1200 mm; showing them forces ScalismoUI's camera to
 *       zoom out so far that aligned bones appear as tiny fragments.
 *       View raw STLs in ParaView / MeshLab if needed.
 *
 * Run independently:
 *   SCAPULA_DATA_DIR=... SCAPULA_OUT_DIR=... sbt "runMain scapula.VisualizationApp"
 * Or set SCAPULA_UI=true before running Main.
 */
object VisualizationApp {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dataDir  = Config.dataDir
    val outDir   = Config.outDir
    val preDir   = new File(outDir, "data/8k")
    val rigidDir = new File(outDir, "rigid_registered")
    val nrDir    = new File(outDir, "nonrigid_registered")
    val modelDir = new File(outDir, "model")
    val meanDir  = new File(outDir, "mean")

    // ── Load landmarks and specimens ─────────────────────────────────────────
    val csvFile = ScapulaData.csvFile(dataDir)
    val (lmMap, fromHeader, _) = ScapulaData.readLandmarkCsv(csvFile)
    if (!fromHeader)
      println("WARNING: landmark columns resolved by fallback offsets — verify CSV header")

    val allSpecs   = ScapulaData.specimens(dataDir)
    val validSpecs = allSpecs.filter(s => lmMap.contains(s.modelId))
    println(s"Specimens: ${allSpecs.length} total, ${validSpecs.length} with landmarks")

    val refSpec = validSpecs(Config.refIdx.min(validSpecs.length - 1))
    val refLms  = if (refSpec.isRight) ScapulaData.mirrorLandmarks(lmMap(refSpec.modelId))
                  else lmMap(refSpec.modelId)
    println(s"Reference: ${refSpec.modelId}")

    // ── Helpers ──────────────────────────────────────────────────────────────
    def stlsIn(dir: File): IndexedSeq[File] =
      if (!dir.isDirectory) IndexedSeq.empty
      else Option(dir.listFiles()).getOrElse(Array.empty)
        .filter(_.getName.endsWith(".stl"))
        .sortBy(_.getName).toIndexedSeq

    def loadOpt(f: File): Option[TriangleMesh[_3D]] =
      if (f.exists()) scala.util.Try(ScapulaData.loadMesh(f)).toOption else None

    def loadModelOpt(f: File): Option[StatisticalMeshModel] =
      if (!f.exists()) None
      else scala.util.Try(
        scalismo.io.StatisticalModelIO.readStatisticalMeshModel(f)
          .getOrElse(throw new RuntimeException("model read failed"))
      ).toOption

    // 8k decimated mesh with mirroring for right scapulae
    def workingMesh(spec: ScapulaData.Specimen): TriangleMesh[_3D] = {
      val f = new File(preDir, spec.modelId + ".stl")
      val raw = if (f.exists()) ScapulaData.loadMesh(f)
                else ScapulaData.loadMesh(spec.file)
      val mesh = Decimation.keepLargestComponent(raw)
      if (spec.isRight) ScapulaData.mirrorMesh(mesh) else mesh
    }

    def specimenLms(spec: ScapulaData.Specimen) =
      if (spec.isRight) ScapulaData.mirrorLandmarks(lmMap(spec.modelId))
      else lmMap(spec.modelId)

    val ui = ScalismoUI("Scapula SSM — Full Pipeline Viewer")

    // ════════════════════════════════════════════════════════════════════════
    // G01 — Decimated + landmark-Procrustes aligned (live computation)
    //
    //   Stage : DECIMATION + RIGID (landmark-only, no ICP)
    //   Source: 8k working meshes from data/8k/; alignment computed live.
    //   Space : reference coordinate frame (Procrustes-aligned)
    //   What to look for: all bones should ROUGHLY overlap.
    //     If they are scattered → landmark CSV columns are wrong.
    //     Expect ±30–50 mm spread at this stage (no ICP yet).
    // ════════════════════════════════════════════════════════════════════════
    println("\n[G01] Decimated + landmark-Procrustes (live, no pipeline needed)")
    val g01 = ui.createGroup("G01_Decimated_Aligned (8k, landmark-Procrustes, ~30-50mm spread)")
    var g01n = 0
    var bbMinX = Double.MaxValue; var bbMaxX = Double.MinValue
    var bbMinY = Double.MaxValue; var bbMaxY = Double.MinValue
    var bbMinZ = Double.MaxValue; var bbMaxZ = Double.MinValue

    validSpecs.foreach { spec =>
      val mesh    = workingMesh(spec)
      val lms     = specimenLms(spec)
      val trans   = ScapulaData.rigidFromLandmarks(lms, refLms)
      val aligned = mesh.transform(trans)
      aligned.pointSet.points.foreach { p =>
        if (p.x < bbMinX) bbMinX = p.x; if (p.x > bbMaxX) bbMaxX = p.x
        if (p.y < bbMinY) bbMinY = p.y; if (p.y > bbMaxY) bbMaxY = p.y
        if (p.z < bbMinZ) bbMinZ = p.z; if (p.z > bbMaxZ) bbMaxZ = p.z
      }
      ui.show(g01, aligned, spec.modelId).opacity = 0.4
      g01n += 1
    }
    println(f"  $g01n specimens in reference frame")
    println(f"  Bounding box: X=[${bbMinX}%.0f, ${bbMaxX}%.0f]  " +
            f"Y=[${bbMinY}%.0f, ${bbMaxY}%.0f]  Z=[${bbMinZ}%.0f, ${bbMaxZ}%.0f]")
    val extX = bbMaxX - bbMinX
    if (extX < 300) println("  ✓ Bone-scale (≤300 mm) — alignment OK")
    else            println("  ✗ HUGE bounding box — landmark CSV columns may be wrong!")

    // ════════════════════════════════════════════════════════════════════════
    // G02 — Rigid-aligned (landmark Procrustes + trimmed ICP) — pipeline output
    //
    //   Stage : RIGID REGISTRATION  (landmark Procrustes + 40-iter trimmed ICP)
    //   Source: rigid_registered/*.stl  (written by Main during pipeline run)
    //   Space : reference coordinate frame
    //   What to look for: all bones must TIGHTLY overlap (< 5 mm spread).
    //     If scattered → ICP did not converge; check Config.icpIterations.
    // ════════════════════════════════════════════════════════════════════════
    println("\n[G02] Rigid-aligned (landmark + ICP) — from pipeline")
    val rigidFiles = stlsIn(rigidDir)
    if (rigidFiles.isEmpty) {
      println("  NOT FOUND — run 'sbt runMain scapula.Main' first")
    } else {
      val g02 = ui.createGroup(s"G02_RigidAligned (${rigidFiles.length} specimens, landmark+ICP)")
      rigidFiles.foreach { f =>
        ui.show(g02, ScapulaData.loadMesh(f), f.getName.stripSuffix(".stl")).opacity = 0.4
      }
      println(s"  ${rigidFiles.length} rigid-aligned meshes")
    }

    // ════════════════════════════════════════════════════════════════════════
    // G03 — Non-rigid registered (GP-ICP, in dense correspondence)
    //
    //   Stage : NON-RIGID REGISTRATION  (GP-ICP, single Gaussian kernel)
    //           σ=30mm, scale=10mm, NearestNeighborInterpolator3D
    //   Source: nonrigid_registered/*.stl
    //   Space : reference coordinate frame (all meshes have SAME topology)
    //   What to look for: better overlap than G02; shape variation now reflects
    //     true biological shape differences, not pose/size differences.
    // ════════════════════════════════════════════════════════════════════════
    println("\n[G03] Non-rigid registered (GP-ICP, dense correspondence) — from pipeline")
    val nrFiles = stlsIn(nrDir)
    if (nrFiles.isEmpty) {
      println("  NOT FOUND — run pipeline first")
    } else {
      val g03 = ui.createGroup(s"G03_NonRigidRegistered (${nrFiles.length} specimens, GP-ICP, same topology)")
      nrFiles.foreach { f =>
        ui.show(g03, ScapulaData.loadMesh(f), f.getName.stripSuffix(".stl")).opacity = 0.4
      }
      println(s"  ${nrFiles.length} non-rigid registered meshes")
    }

    // ── Load or build SSM ────────────────────────────────────────────────────
    val ssmOpt: Option[StatisticalMeshModel] = {
      val h5 = new File(modelDir, "scapula_ssm.h5")
      loadModelOpt(h5).map { m => println(s"\nSSM loaded from ${h5.getPath}  rank=${m.rank}"); m }
        .orElse {
          val meshes = nrFiles.flatMap(f => loadOpt(f))
          if (meshes.length >= 3) {
            println(s"\nBuilding SSM from ${meshes.length} non-rigid meshes (no .h5 found)…")
            val dc = DataCollection.fromTriangleMesh3DSequence(meshes.head, meshes)
            StatisticalMeshModel.createUsingPCA(dc).toOption.map { m =>
              println(s"  Built SSM rank=${m.rank}"); m
            }
          } else None
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // G04 — Mean shape
    //
    //   Stage : SSM MEAN  (arithmetic mean of all non-rigidly registered shapes)
    //   Source: mean/mean.stl (or computed from SSM.mean when file absent)
    //   Space : reference coordinate frame
    //   What to look for: compact, representative scapula shape.
    // ════════════════════════════════════════════════════════════════════════
    println("\n[G04] Mean shape")
    val g04 = ui.createGroup("G04_MeanShape (arithmetic mean of registered population)")
    loadOpt(new File(meanDir, "mean.stl")) match {
      case Some(m) =>
        ui.show(g04, m, "mean")
        println(s"  Mean loaded from file  (${m.pointSet.numberOfPoints} vertices)")
      case None    =>
        ssmOpt.foreach { ssm =>
          ui.show(g04, ssm.mean, "mean_from_ssm")
          println(s"  Mean from SSM  (${ssm.mean.pointSet.numberOfPoints} vertices)")
        }
        if (ssmOpt.isEmpty) println("  Not available — run pipeline first")
    }

    // ════════════════════════════════════════════════════════════════════════
    // G05–G09 — SSM groups (only when SSM is available)
    // ════════════════════════════════════════════════════════════════════════
    ssmOpt match {
      case None =>
        println("\nNo SSM available — G05-G09 skipped. Run 'sbt runMain scapula.Main'.")

      case Some(ssm) =>
        val evs   = ssm.gp.klBasis.map(_.eigenvalue)
        val total = evs.sum
        println(s"\nSSM  rank=${ssm.rank}")
        println(f"  ${"Mode"}%5s  ${"Var%%"}%8s  ${"Cumul%%"}%8s  ${"σ (mm)"}%10s")
        evs.take(math.min(10, ssm.rank)).zipWithIndex.foreach { case (ev, i) =>
          println(f"  ${i+1}%5d  ${ev/total*100}%8.2f  ${evs.take(i+1).sum/total*100}%8.2f  ${math.sqrt(ev)}%10.3f")
        }

        // ── G05 — Interactive SSM ─────────────────────────────────────────────
        // Select this group in the scene panel, then drag the Mode sliders
        // that appear in the right-hand panel to explore the shape space.
        println("\n[G05] Interactive SSM")
        val g05 = ui.createGroup("G05_SSM_Interactive (select → drag Mode sliders in right panel)")
        ui.show(g05, ssm, "SSM")

        // ── G06 / G07 / G08 — Static mode shapes (±1σ / ±2σ / ±3σ) ──────────
        val nModes = math.min(3, ssm.rank)
        (0 until nModes).foreach { modeIdx =>
          val sigma  = math.sqrt(evs(modeIdx))
          val varPct = evs(modeIdx) / total * 100.0
          val gNum   = 6 + modeIdx
          println(s"\n[G0$gNum] Mode ${modeIdx + 1}: σ=${f"$sigma%.1f"}mm  ${f"$varPct%.1f"}% variance")
          val mGrp = ui.createGroup(
            f"G0${gNum}_Mode${modeIdx+1} (σ=${sigma.toInt}mm, ${varPct.toInt}%% var)  ±1σ/±2σ/±3σ overlaid")

          def coeff(a: Double) =
            DenseVector(Array.tabulate(ssm.rank)(j => if (j == modeIdx) a * sigma else 0.0))

          // Graduated opacity: mean is most opaque, ±3σ is most transparent.
          // This reduces Z-fighting between the 7 overlapping surfaces.
          ui.show(mGrp, ssm.mean,                  s"mode${modeIdx+1}_mean"   ).opacity = 0.9
          ui.show(mGrp, ssm.instance(coeff( 1.0)),  s"mode${modeIdx+1}_+1sigma").opacity = 0.55
          ui.show(mGrp, ssm.instance(coeff(-1.0)),  s"mode${modeIdx+1}_-1sigma").opacity = 0.55
          ui.show(mGrp, ssm.instance(coeff( 2.0)),  s"mode${modeIdx+1}_+2sigma").opacity = 0.4
          ui.show(mGrp, ssm.instance(coeff(-2.0)),  s"mode${modeIdx+1}_-2sigma").opacity = 0.4
          ui.show(mGrp, ssm.instance(coeff( 3.0)),  s"mode${modeIdx+1}_+3sigma").opacity = 0.3
          ui.show(mGrp, ssm.instance(coeff(-3.0)),  s"mode${modeIdx+1}_-3sigma").opacity = 0.3
        }

        // ── G09 — Random samples ──────────────────────────────────────────────
        println("\n[G09] 5 random SSM instances (specificity check)")
        val g09 = ui.createGroup("G09_ModelSamples (5 random instances — should look like scapulae)")
        (1 to 5).foreach { i => ui.show(g09, ssm.sample(), s"sample_$i").opacity = 0.4 }
    }

    // ════════════════════════════════════════════════════════════════════════
    // G10 — Reference mesh + landmarks
    //
    //   The reference mesh is shown in the reference coordinate frame —
    //   the same frame as all other groups — so the camera is not affected.
    //   Source: model/reference.stl (saved by pipeline) or live from 8k folder.
    // ════════════════════════════════════════════════════════════════════════
    println(s"\n[G10] Reference: ${refSpec.modelId}")
    val g10 = ui.createGroup(s"G10_Reference (${refSpec.modelId})  + landmarks")
    val refMeshDisplay = loadOpt(new File(modelDir, "reference.stl"))
      .getOrElse(workingMesh(refSpec))
    ui.show(g10, refMeshDisplay, "reference_mesh")
    ui.show(g10, refLms,        "landmarks_GC_TS_IA_PLA_AC")
    println(f"  ${refMeshDisplay.pointSet.numberOfPoints} vertices")
    refLms.foreach { lm =>
      println(f"  ${lm.id}%5s  (${lm.point.x}%7.1f, ${lm.point.y}%7.1f, ${lm.point.z}%7.1f)")
    }

    // ── Viewer guide ─────────────────────────────────────────────────────────
    println("\n" + "═" * 72)
    println("  VIEWER — all groups in reference coordinate space")
    println("  Eye icon in scene panel = hide / show group")
    println("═" * 72)
    println("  G01  Decimated_Aligned       8k meshes, landmark-Procrustes (live)")
    println("       → bones ROUGHLY overlap; spread ~30-50 mm at this stage")
    println("  G02  RigidAligned            landmark + ICP  (from pipeline)")
    println("       → bones TIGHTLY overlap; spread < 5 mm expected")
    println("  G03  NonRigidRegistered      GP-ICP, dense correspondence")
    println("       → all meshes share the same vertex topology as the reference")
    println("  G04  MeanShape               SSM mean (average scapula shape)")
    println("  G05  SSM_Interactive         select → drag Mode sliders (right panel)")
    println("  G06  Mode1  ±1σ/±2σ/±3σ     shape variation along PC1")
    println("  G07  Mode2  ±1σ/±2σ/±3σ     shape variation along PC2")
    println("  G08  Mode3  ±1σ/±2σ/±3σ     shape variation along PC3")
    println("  G09  ModelSamples            5 random instances (should look like scapulae)")
    println("  G10  Reference               reference mesh + 5 landmarks")
    println("─" * 72)
    println("  NOTE: Raw scanner-space meshes are NOT shown — they span ~1200 mm")
    println("        and force the camera so far out that all bones look like specks.")
    println("        View raw STLs in ParaView or MeshLab if needed.")
    println("═" * 72)
  }
}
