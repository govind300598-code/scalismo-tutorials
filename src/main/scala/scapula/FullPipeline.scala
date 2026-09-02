package scapula

import scalismo.geometry.*
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.mesh.*
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.utils.Random

import java.io.{File, PrintWriter}

/**
 * Full 2-pass pipeline: rigid alignment → FFDM build → GP-ICP (pass 1) → SSM 1 →
 *                        mean-reference update → GP-ICP (pass 2) → SSM 2 → evaluation.
 *
 * Two passes remove reference-shape bias: pass 1 uses a single specimen as reference;
 * pass 2 re-registers to the unbiased mean shape of pass 1.
 *
 * Run:  sbt "runMain scapula.FullPipeline"
 *
 * Override any path or parameter without editing code:
 *   SCAPULA_DATA_DIR=/path/to/stls \
 *   SCAPULA_OUT_DIR=/path/to/output \
 *   SCAPULA_FFDM_REF=paired_scapula_002_M_56_L \
 *   sbt "runMain scapula.FullPipeline"
 */
object FullPipeline {

  // ── type alias for brevity
  type PDM = PointDistributionModel[_3D, TriangleMesh]
  case class SpecMesh(spec: ScapulaData.Specimen, mesh: TriangleMesh[_3D], lms: IndexedSeq[scalismo.geometry.Landmark[_3D]])

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val outDir = Config.outDir
    outDir.mkdirs()
    val pass1Dir = new File(outDir, "pass1"); pass1Dir.mkdirs()
    val pass2Dir = new File(outDir, "pass2"); pass2Dir.mkdirs()

    println("=" * 72)
    println("FULL SSM PIPELINE  (2-pass rigid → GP-ICP → PCA)")
    println("=" * 72)
    println(s"  Data    : ${Config.dataDir.getAbsolutePath}")
    println(s"  Output  : ${outDir.getAbsolutePath}")
    println(s"  Ref ID  : ${Config.ffdmRefId}")
    println(s"  ICP iter: ${Config.icpIterations}  GP rank cap: ${Config.gpMaxRank}")
    println()

    // ── 0. Load all specimens + landmarks ──────────────────────────────────────
    val csv = ScapulaData.csvFile(Config.dataDir)
    val (landmarks, _, _) = ScapulaData.readLandmarkCsv(csv)
    val allSpecs = ScapulaData.specimens(Config.dataDir)
    val withLms  = allSpecs.filter(s => landmarks.contains(s.modelId))
    println(s"Specimens with landmarks : ${withLms.length} / ${allSpecs.length}")

    // Mirror right scapulae so every shape is left-oriented
    val normalized: IndexedSeq[SpecMesh] = withLms.map { s =>
      val rawMesh = ScapulaData.loadMesh(s.file)
      val rawLms  = landmarks(s.modelId)
      if (s.isRight)
        SpecMesh(s, ScapulaData.mirrorMesh(rawMesh), ScapulaData.mirrorLandmarks(rawLms))
      else
        SpecMesh(s, rawMesh, rawLms)
    }
    val nRight = normalized.count(_.spec.isRight)
    println(s"  $nRight right-side meshes mirrored to left orientation")
    println(s"  Total specimens entering pipeline: ${normalized.length}")

    // ── 1. Locate / verify seed reference ─────────────────────────────────────
    val seedSpecOpt = normalized.find(_.spec.modelId == Config.ffdmRefId)
    val seedRef: SpecMesh = seedSpecOpt.getOrElse {
      println(s"  [warn] Seed reference '${Config.ffdmRefId}' not in landmark CSV; using first left specimen.")
      normalized.find(!_.spec.isRight).getOrElse(normalized.head)
    }
    println(s"\nSeed reference : ${seedRef.spec.modelId}")
    MeshIO.writeMesh(seedRef.mesh, new File(outDir, "seed_reference.stl")).get

    // ── 2. Rigid alignment (landmark Procrustes + trimmed ICP) ────────────────
    println("\n[STEP 1] Rigid alignment — all to seed reference")
    val rigidAligned: IndexedSeq[(ScapulaData.Specimen, TriangleMesh[_3D])] =
      normalized.map { sm =>
        val (aligned, _) = RigidAlign.landmarkThenIcp(
          sm.mesh, sm.lms, seedRef.mesh, seedRef.lms, Config.icpIterations
        )
        val d = Metrics.symmetric(aligned, seedRef.mesh)
        println(f"  ${sm.spec.modelId}%-32s  mean=${d.mean}%5.2f  HD95=${d.hd95}%5.2f  HD=${d.hd}%6.2f mm")
        MeshIO.writeMesh(aligned, new File(outDir, s"rigid_${sm.spec.modelId}.stl")).get
        (sm.spec, aligned)
      }

    // ── 3. Build FFDM on seed reference ───────────────────────────────────────
    println("\n[STEP 2] Building FFDM prior — seed reference")
    val ffdm1 = buildFfdm(seedRef.mesh)
    println(s"  FFDM rank: ${ffdm1.rank}")
    StatisticalModelIO.writeStatisticalMeshModel(ffdm1, new File(outDir, "ffdm_pass1.h5")).get

    // ── 4. GP-ICP pass 1 ──────────────────────────────────────────────────────
    println(s"\n[STEP 3] GP-ICP registration — Pass 1 (${rigidAligned.length} specimens)")
    val pass1: IndexedSeq[(ScapulaData.Specimen, TriangleMesh[_3D])] =
      rigidAligned.zipWithIndex.map { case ((spec, rigMesh), i) =>
        println(s"  [${i + 1}/${rigidAligned.length}] ${spec.modelId}")
        val fitted = NonRigidReg.gpIcp(ffdm1, rigMesh, Config.icpIterations)
        val d = Metrics.symmetric(fitted, rigMesh)
        println(f"    surf: mean=${d.mean}%.2f  HD95=${d.hd95}%.2f  HD=${d.hd}%.2f mm")
        MeshIO.writeMesh(fitted, new File(pass1Dir, s"reg_${spec.modelId}.stl")).get
        (spec, fitted)
      }

    // ── 5. SSM pass 1 ─────────────────────────────────────────────────────────
    println("\n[STEP 4] Building SSM — Pass 1")
    val ssm1 = buildSsm(seedRef.mesh, pass1.map(_._2))
    StatisticalModelIO.writeStatisticalMeshModel(ssm1, new File(outDir, "ssm_pass1.h5")).get
    println(s"  SSM rank: ${ssm1.rank}")
    val mean1 = ssm1.mean
    MeshIO.writeMesh(mean1, new File(outDir, "mean_pass1.stl")).get
    println(s"  Mean shape saved → mean_pass1.stl")

    // ── 6. Build FFDM on pass-1 mean (new reference) ─────────────────────────
    println("\n[STEP 5] Building FFDM prior — pass-1 mean (new reference)")
    val ffdm2 = buildFfdm(mean1)
    println(s"  FFDM rank: ${ffdm2.rank}")
    StatisticalModelIO.writeStatisticalMeshModel(ffdm2, new File(outDir, "ffdm_pass2.h5")).get

    // ── 7. GP-ICP pass 2  (target = rigid-aligned originals, ref = mean1) ────
    //  Using the same rigidly-aligned meshes as targets keeps the number of mesh-loading
    //  steps minimal; mean1 is close to all of them so GP-ICP converges quickly.
    println(s"\n[STEP 6] GP-ICP registration — Pass 2 (${rigidAligned.length} specimens)")
    val pass2: IndexedSeq[(ScapulaData.Specimen, TriangleMesh[_3D])] =
      rigidAligned.zipWithIndex.map { case ((spec, rigMesh), i) =>
        println(s"  [${i + 1}/${rigidAligned.length}] ${spec.modelId}")
        val fitted = NonRigidReg.gpIcp(ffdm2, rigMesh, Config.icpIterations)
        val d = Metrics.symmetric(fitted, rigMesh)
        println(f"    surf: mean=${d.mean}%.2f  HD95=${d.hd95}%.2f  HD=${d.hd}%.2f mm")
        MeshIO.writeMesh(fitted, new File(pass2Dir, s"reg_${spec.modelId}.stl")).get
        (spec, fitted)
      }

    // ── 8. SSM pass 2 (final) ─────────────────────────────────────────────────
    println("\n[STEP 7] Building SSM — Pass 2 (final)")
    val ssm2 = buildSsm(mean1, pass2.map(_._2))
    StatisticalModelIO.writeStatisticalMeshModel(ssm2, new File(outDir, "ssm_final.h5")).get
    println(s"  Final SSM rank: ${ssm2.rank}")
    val mean2 = ssm2.mean
    MeshIO.writeMesh(mean2, new File(outDir, "mean_pass2.stl")).get

    // ── 9. Stability check ────────────────────────────────────────────────────
    val stab = Metrics.symmetric(mean1, mean2)
    println()
    println(f"[STABILITY] pass-1 mean vs pass-2 mean — mean=${stab.mean}%.2f  HD=${stab.hd}%.2f mm")
    println(f"  (< 0.5 mm mean = model is stable; you can stop iterating)")

    // ── 10. Full evaluation ───────────────────────────────────────────────────
    println("\n[STEP 8] Computing all metrics")
    SSMEval.evaluate(
      finalSsm       = ssm2,
      meanRef        = mean2,
      registered     = pass2,
      rigidOriginals = rigidAligned.toMap,
      stability      = stab,
      outDir         = outDir
    )

    println("\n" + "=" * 72)
    println("PIPELINE COMPLETE")
    println(s"  All outputs in: ${outDir.getAbsolutePath}")
    println()
    println("  Key files:")
    println(s"    ssm_final.h5         — final SSM model (HDF5)")
    println(s"    mean_pass2.stl       — unbiased mean scapula shape")
    println(s"    pass2/reg_*.stl      — all registered shapes (in correspondence)")
    println(s"    metrics_per_spec.csv — per-specimen Hausdorff / RMSE / Chamfer etc.")
    println(s"    ssm_quality.txt      — compactness / generalization / specificity")
    println("=" * 72)
  }

  // ── Helpers ─────────────────────────────────────────────────────────────────

  /** Build a multi-scale GP prior (FFDM) on the given reference mesh. */
  def buildFfdm(reference: TriangleMesh[_3D])(implicit rng: Random): PDM = {
    val L    = BuildFFDMModel.longestDimension(reference)
    val comps = Seq(
      (L / 2.0,  L * L / 4.0),
      (L / 5.0,  L * L / 25.0),
      (L / 10.0, L * L / 100.0)
    )
    val gp = BuildFFDMModel.buildLowRankGP(reference, comps, Config.gpMaxRank)
    PointDistributionModel[_3D, TriangleMesh](reference, gp)
  }

  /** PCA-based SSM from a collection of registered (in-correspondence) meshes. */
  def buildSsm(
    reference:        TriangleMesh[_3D],
    registeredMeshes: IndexedSeq[TriangleMesh[_3D]]
  ): PDM = {
    require(registeredMeshes.nonEmpty, "Need at least 1 shape to build SSM")
    val dc = DataCollection.fromTriangleMeshSequence(reference, registeredMeshes)
    PointDistributionModel.createUsingPCA(dc)
  }
}
