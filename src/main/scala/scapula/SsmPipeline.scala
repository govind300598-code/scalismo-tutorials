package scapula

import scalismo.geometry.*
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.{LowRankGaussianProcess, PointDistributionModel}
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.utils.Random

import java.io.File

/**
 * Complete two-iteration SSM pipeline:
 *
 *   Stage 2  – Register all meshes to an initial reference  →  SSM1 registered meshes
 *   Stage 3  – PCA on those meshes                          →  SSM1
 *   Stage 4  – Register all meshes to the SSM1 mean         →  SSM2 registered meshes
 *   Stage 4b – PCA on those meshes                          →  SSM2
 *   Stage 5  – Evaluate: compactness / generalization / specificity
 *   Stage 6  – Compare SSM1 vs SSM2 mean shapes
 *
 * All configurable via environment variables (see Config in ScapulaData.scala).
 * Set SCAPULA_REF_ID to a mesh name (without .stl) to choose a specific reference.
 *
 * Run with:
 *   sbt "runMain scapula.SsmPipeline"
 */
object SsmPipeline {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dataDir = Config.dataDir
    val outDir  = Config.outDir
    outDir.mkdirs()

    banner("SCAPULA SSM PIPELINE")
    println(s"  data   : ${dataDir.getAbsolutePath}")
    println(s"  output : ${outDir.getAbsolutePath}")
    println(s"  seed   : ${Config.seed}")

    // ---------------------------------------------------------------- load
    val csv = ScapulaData.csvFile(dataDir)
    val (landmarks, fromHeader, _) = ScapulaData.readLandmarkCsv(csv)
    if (!fromHeader)
      println("WARNING: landmark columns resolved by fallback offsets — verify against your CSV")

    val allSpecimens = ScapulaData.specimens(dataDir)
    val specimens    = allSpecimens.filter(s => landmarks.contains(s.modelId))
    println(s"Specimens with landmarks: ${specimens.length} / ${allSpecimens.length}")
    require(specimens.length >= 3, "Need at least 3 specimens to build an SSM")

    // ---------------------------------------------------------------- pick reference
    val refSpec = sys.env.get("SCAPULA_REF_ID")
      .flatMap(id => specimens.find(_.modelId == id))
      .getOrElse(specimens.head)
    println(s"Reference: ${refSpec.modelId}  (override with SCAPULA_REF_ID env var)")

    val rawRef    = ScapulaData.loadMesh(refSpec.file)
    val reference = rawRef.operations.decimate(Config.modelResolution)
    val refLms    = if (refSpec.isRight) ScapulaData.mirrorLandmarks(landmarks(refSpec.modelId))
                    else landmarks(refSpec.modelId)

    println(s"Reference after decimation: ${reference.pointSet.numberOfPoints} vertices")

    // Cache point-ids for each landmark on the decimated reference
    val refLmPtIds = refLms.map(lm => lm.id -> reference.pointSet.findClosestPoint(lm.point).id).toMap

    // ---------------------------------------------------------------- GP prior (SSM1)
    println("\nBuilding GP prior on reference mesh...")
    val gpPrior1 = NonRigidReg.buildGpPrior(reference)
    println(s"GP prior rank: ${gpPrior1.rank}")

    // ================================================================ SSM1
    banner("STAGE 2 – REGISTRATION FOR SSM1")

    val ssm1RegDir = new File(outDir, "ssm1_registered")
    ssm1RegDir.mkdirs()

    val ssm1Meshes = registerAll(
      specimens = specimens,
      landmarks = landmarks,
      reference = reference,
      refLms    = refLms,
      gpPrior   = gpPrior1,
      outDir    = ssm1RegDir
    )

    banner("STAGE 3 – BUILD SSM1")

    val ssm1     = buildSsm(reference, ssm1Meshes, "SSM1")
    val ssm1File = new File(outDir, "ssm1.h5")
    StatisticalModelIO.writeStatisticalTriangleMeshModel3D(ssm1, ssm1File).get
    println(s"SSM1 saved: ${ssm1File.getName}")

    val ssm1Mean     = ssm1.mean
    val ssm1MeanFile = new File(outDir, "ssm1_mean.stl")
    MeshIO.writeMesh(ssm1Mean, ssm1MeanFile).get
    println(s"SSM1 mean saved: ${ssm1MeanFile.getName}")

    // Transfer landmarks to SSM1 mean via the reference point-id correspondence
    val ssm1MeanLms = refLms.map { lm =>
      lm.copy(point = ssm1Mean.pointSet.point(refLmPtIds(lm.id)))
    }

    // ================================================================ SSM2
    banner("STAGE 4 – REGISTRATION FOR SSM2 (SSM1 mean as new reference)")

    println("Building GP prior on SSM1 mean shape...")
    val gpPrior2 = NonRigidReg.buildGpPrior(ssm1Mean)
    println(s"GP prior 2 rank: ${gpPrior2.rank}")

    val ssm2RegDir = new File(outDir, "ssm2_registered")
    ssm2RegDir.mkdirs()

    val ssm2Meshes = registerAll(
      specimens = specimens,
      landmarks = landmarks,
      reference = ssm1Mean,
      refLms    = ssm1MeanLms,
      gpPrior   = gpPrior2,
      outDir    = ssm2RegDir
    )

    banner("STAGE 4b – BUILD SSM2")

    val ssm2     = buildSsm(ssm1Mean, ssm2Meshes, "SSM2")
    val ssm2File = new File(outDir, "ssm2.h5")
    StatisticalModelIO.writeStatisticalTriangleMeshModel3D(ssm2, ssm2File).get
    println(s"SSM2 saved: ${ssm2File.getName}")

    val ssm2Mean     = ssm2.mean
    val ssm2MeanFile = new File(outDir, "ssm2_mean.stl")
    MeshIO.writeMesh(ssm2Mean, ssm2MeanFile).get
    println(s"SSM2 mean saved: ${ssm2MeanFile.getName}")

    // ================================================================ Stage 5: Evaluation
    banner("STAGE 5 – EVALUATION")

    println("Compactness SSM1...")
    val comp1 = Evaluate.compactness(ssm1)
    Evaluate.saveCsv(
      Seq(Seq("mode", "cumulative_variance")) ++
        comp1.map(r => Seq(r._1.toString, f"${r._2}%.6f")),
      new File(outDir, "ssm1_compactness.csv")
    )

    println("Compactness SSM2...")
    val comp2 = Evaluate.compactness(ssm2)
    Evaluate.saveCsv(
      Seq(Seq("mode", "cumulative_variance")) ++
        comp2.map(r => Seq(r._1.toString, f"${r._2}%.6f")),
      new File(outDir, "ssm2_compactness.csv")
    )

    println("Generalization SSM1 (leave-one-out CV)...")
    val (gen1Mean, gen1Std) = Evaluate.generalization(reference, ssm1Meshes)
    println(f"  SSM1: mean=${gen1Mean}%.3f mm  std=${gen1Std}%.3f mm")

    println("Generalization SSM2 (leave-one-out CV)...")
    val (gen2Mean, gen2Std) = Evaluate.generalization(ssm1Mean, ssm2Meshes)
    println(f"  SSM2: mean=${gen2Mean}%.3f mm  std=${gen2Std}%.3f mm")

    println("Specificity SSM1 (100 random samples)...")
    val (spec1Mean, spec1Std) = Evaluate.specificity(ssm1, ssm1Meshes)
    println(f"  SSM1: mean=${spec1Mean}%.3f mm  std=${spec1Std}%.3f mm")

    println("Specificity SSM2 (100 random samples)...")
    val (spec2Mean, spec2Std) = Evaluate.specificity(ssm2, ssm2Meshes)
    println(f"  SSM2: mean=${spec2Mean}%.3f mm  std=${spec2Std}%.3f mm")

    Evaluate.saveCsv(
      Seq(
        Seq("metric", "ssm1", "ssm2"),
        Seq("generalization_mean_mm", f"${gen1Mean}%.4f", f"${gen2Mean}%.4f"),
        Seq("generalization_std_mm",  f"${gen1Std}%.4f",  f"${gen2Std}%.4f"),
        Seq("specificity_mean_mm",    f"${spec1Mean}%.4f", f"${spec2Mean}%.4f"),
        Seq("specificity_std_mm",     f"${spec1Std}%.4f",  f"${spec2Std}%.4f")
      ),
      new File(outDir, "evaluation.csv")
    )

    // ================================================================ Stage 6: Mean shape comparison
    banner("STAGE 6 – MEAN SHAPE COMPARISON (SSM1 vs SSM2)")

    val symStats = Metrics.symmetric(ssm1Mean, ssm2Mean)
    println(s"Symmetric surface distance: ${symStats.render}")

    // Point-to-point RMSE is valid because both means have the same reference topology
    val ptDists = Metrics.correspondingDistances(ssm1Mean, ssm2Mean)
    val rmse    = math.sqrt(ptDists.map(d => d * d).sum / ptDists.length)
    println(f"Point-to-point RMSE: $rmse%.3f mm")

    Evaluate.saveCsv(
      Seq(
        Seq("metric", "value_mm"),
        Seq("mean_surface_distance",  f"${symStats.mean}%.4f"),
        Seq("rms_surface_distance",   f"${symStats.rms}%.4f"),
        Seq("hd95",                   f"${symStats.hd95}%.4f"),
        Seq("max_distance",           f"${symStats.hd}%.4f"),
        Seq("pointwise_rmse",         f"$rmse%.4f")
      ),
      new File(outDir, "mean_shape_comparison.csv")
    )

    // ================================================================ Final table
    banner("FINAL COMPARISON TABLE")

    val c1at5 = comp1.find(_._1 == math.min(5, ssm1.rank)).map(r => f"${r._2 * 100}%.1f%%").getOrElse("N/A")
    val c2at5 = comp2.find(_._1 == math.min(5, ssm2.rank)).map(r => f"${r._2 * 100}%.1f%%").getOrElse("N/A")

    printRow("Metric",                      "SSM1",            "SSM2",            "Interpretation")
    println("-" * 96)
    printRow("Number of training meshes",   ssm1Meshes.length.toString, ssm2Meshes.length.toString, "")
    printRow("Number of PCA modes",         ssm1.rank.toString, ssm2.rank.toString, "")
    printRow("Compactness (var@5 modes)",   c1at5,             c2at5,             "higher = more compact")
    printRow("Generalization error (mm)",   f"${gen1Mean}%.3f ± ${gen1Std}%.3f",
                                            f"${gen2Mean}%.3f ± ${gen2Std}%.3f", "lower = better")
    printRow("Specificity error (mm)",      f"${spec1Mean}%.3f ± ${spec1Std}%.3f",
                                            f"${spec2Mean}%.3f ± ${spec2Std}%.3f", "lower = better")
    printRow("Mean shape distance (mm)",    "N/A",             f"${symStats.mean}%.3f", "SSM1↔SSM2 stability")
    printRow("RMSE between means (mm)",     "N/A",             f"$rmse%.3f",      "SSM1↔SSM2 stability")

    println(s"\nAll outputs written to: ${outDir.getAbsolutePath}")
    println("Pipeline complete.")
  }

  // ---------------------------------------------------------------------------
  // Registration helper
  // ---------------------------------------------------------------------------

  private def registerAll(
    specimens : IndexedSeq[ScapulaData.Specimen],
    landmarks : Map[String, IndexedSeq[Landmark[_3D]]],
    reference : TriangleMesh[_3D],
    refLms    : IndexedSeq[Landmark[_3D]],
    gpPrior   : LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
    outDir    : File
  )(implicit rng: Random): IndexedSeq[TriangleMesh[_3D]] = {

    specimens.zipWithIndex.map { case (spec, idx) =>
      val outFile = new File(outDir, s"${spec.modelId}_registered.stl")

      if (outFile.exists()) {
        println(s"  [${idx + 1}/${specimens.length}] ${spec.modelId} — cached, loading")
        ScapulaData.loadMesh(outFile)
      } else {
        println(s"  [${idx + 1}/${specimens.length}] ${spec.modelId} — registering...")

        val rawMesh = ScapulaData.loadMesh(spec.file)
        val rawLms  = landmarks(spec.modelId)

        // Mirror right scapulae into the left-side coordinate frame
        val (mesh, lms) =
          if (spec.isRight) (ScapulaData.mirrorMesh(rawMesh), ScapulaData.mirrorLandmarks(rawLms))
          else (rawMesh, rawLms)

        // Step 1: landmark-based Procrustes + trimmed rigid ICP
        println("    Rigid alignment (landmarks → ICP)...")
        val (rigidAligned, _) =
          RigidAlign.landmarkThenIcp(mesh, lms, reference, refLms, icpIterations = 30)

        // Step 2: GP-ICP non-rigid registration
        println(s"    GP-ICP (${Config.icpIterations} iterations)...")
        val registered = NonRigidReg.gpIcp(
          reference  = reference,
          target     = rigidAligned,
          lowRankGP  = gpPrior,
          iterations = Config.icpIterations
        )

        MeshIO.writeMesh(registered, outFile).get
        println(s"    Saved: ${outFile.getName}")
        registered
      }
    }
  }

  // ---------------------------------------------------------------------------
  // SSM builder
  // ---------------------------------------------------------------------------

  private def buildSsm(
    reference : TriangleMesh[_3D],
    meshes    : IndexedSeq[TriangleMesh[_3D]],
    label     : String
  ): PointDistributionModel[_3D, TriangleMesh] = {
    println(s"\nBuilding $label from ${meshes.length} meshes...")
    val dc    = DataCollection.fromTriangleMesh3DSequence(reference, meshes)
    val dcGpa = DataCollection.gpa(dc)
    val model = PointDistributionModel.createUsingPCA(dcGpa)
      .getOrElse(throw new RuntimeException(s"Failed to build $label"))
    println(s"$label built: rank=${model.rank}  vertices=${model.mean.pointSet.numberOfPoints}")
    model
  }

  // ---------------------------------------------------------------------------
  private def banner(msg: String): Unit = {
    println()
    println("=" * 70)
    println(msg)
    println("=" * 70)
  }

  private def printRow(col1: String, col2: String, col3: String, col4: String): Unit =
    println(f"$col1%-36s  $col2%-24s  $col3%-24s  $col4")
}
