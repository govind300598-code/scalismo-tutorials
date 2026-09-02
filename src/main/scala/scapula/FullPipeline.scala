package scapula

import scalismo.geometry.*
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.mesh.*
import scalismo.statisticalmodel.StatisticalMeshModel
import scalismo.utils.Random

import java.io.{File, PrintWriter}
import scala.util.{Try, Using}

/**
 * Complete two-pass SSM pipeline.
 *
 * Pass 1 – register every specimen to the user-chosen seed reference
 *           (paired_scapula_002_M_56_L, or first available).
 * Mean-1  – compute the mean shape of all Pass-1 registered meshes.
 * Pass 2  – re-register every specimen to Mean-1 (removes initial reference bias).
 * Mean-2  – compute mean shape of Pass-2 meshes (the SSM reference).
 * SSM     – build PCA model from Pass-2 meshes.
 * Eval    – run all standard metrics via SSMEval.
 *
 * Every stage is written to disk so the run can be inspected and resumed.
 * Environment variables (see Config) override all defaults without code changes.
 */
object FullPipeline {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val outDir = Config.outDir
    outDir.mkdirs()

    banner("SCAPULA SSM FULL PIPELINE")

    // -----------------------------------------------------------------------
    // 1.  Load data
    // -----------------------------------------------------------------------
    section("1", "Loading specimens and landmarks")
    val allSpecimens = ScapulaData.specimens(Config.dataDir)
    val csv          = ScapulaData.csvFile(Config.dataDir)
    val (landmarks, fromHeader, _) = ScapulaData.readLandmarkCsv(csv)

    if (!fromHeader)
      println("  WARNING: landmark columns resolved by fallback offsets, not header names.")

    val specsWithLms = allSpecimens.filter(s => landmarks.contains(s.modelId))
    println(s"  ${allSpecimens.length} STL files found, ${specsWithLms.length} have landmark rows")

    require(specsWithLms.nonEmpty, "No specimens with landmarks — cannot continue")

    // -----------------------------------------------------------------------
    // 2.  Prepare seed reference
    // -----------------------------------------------------------------------
    section("2", "Preparing seed reference")
    val seedId = "paired_scapula_002_M_56_L"
    val seedSpec = specsWithLms.find(_.modelId == seedId).getOrElse {
      println(s"  '$seedId' not found – falling back to first specimen: ${specsWithLms.head.modelId}")
      specsWithLms.head
    }
    println(s"  Seed reference: ${seedSpec.modelId}")

    val rawSeed  = ScapulaData.loadMesh(seedSpec.file)
    val refMesh  = rawSeed.operations.decimate(Config.modelResolution)
    val refLms   = landmarks(seedSpec.modelId)
    println(s"  Decimated to ${refMesh.pointSet.numberOfPoints} vertices (target=${Config.modelResolution})")
    MeshIO.writeMesh(refMesh, new File(outDir, "ref_seed.stl")).get

    // -----------------------------------------------------------------------
    // 3.  GP prior on seed reference
    // -----------------------------------------------------------------------
    section("3", "Building GP prior on seed reference")
    val prior1 = NonRigidReg.buildPrior(refMesh)
    println(s"  GP rank: ${prior1.rank}")

    // -----------------------------------------------------------------------
    // 4.  PASS 1 – register all specimens → seed reference
    // -----------------------------------------------------------------------
    section("4", s"Pass 1 – registering ${specsWithLms.length} specimens to seed reference")
    val pass1Dir = new File(outDir, "pass1"); pass1Dir.mkdirs()

    val pass1Pairs: IndexedSeq[(String, TriangleMesh[_3D])] =
      specsWithLms.zipWithIndex.map { case (spec, idx) =>
        println(s"  [${idx + 1}/${specsWithLms.length}] ${spec.modelId}")

        val raw = ScapulaData.loadMesh(spec.file)
        val lms = landmarks(spec.modelId)
        val (mesh, meshLms) =
          if (spec.isRight) (ScapulaData.mirrorMesh(raw), ScapulaData.mirrorLandmarks(lms))
          else              (raw, lms)

        val (rigidAligned, _) =
          RigidAlign.landmarkThenIcp(mesh, meshLms, refMesh, refLms, icpIterations = 30)
        val registered = NonRigidReg.registerOne(refMesh, rigidAligned, prior1)

        MeshIO.writeMesh(registered, new File(pass1Dir, s"reg1_${spec.modelId}.stl")).get
        val st = Metrics.symmetric(registered, rigidAligned)
        println(f"    ${st.render}")

        spec.modelId -> registered
      }

    // -----------------------------------------------------------------------
    // 5.  Compute Mean-1 (new reference for pass 2)
    // -----------------------------------------------------------------------
    section("5", "Computing Mean-1 reference from Pass 1")
    val meanRef = NonRigidReg.meanMesh(refMesh, pass1Pairs.map(_._2))
    MeshIO.writeMesh(meanRef, new File(outDir, "mean_ref_pass1.stl")).get

    val stab1 = Metrics.symmetric(refMesh, meanRef)
    println(s"  Seed ref → Mean-1 surface distance (stability): ${stab1.render}")

    // Approximate Mean-1 landmarks: propagate seed landmarks via nearest point on mean mesh
    val meanRefLms: IndexedSeq[Landmark[_3D]] = refLms.map { lm =>
      val nearestOnMean = meanRef.operations.closestPointOnSurface(lm.point).point
      lm.copy(point = nearestOnMean)
    }

    // -----------------------------------------------------------------------
    // 6.  GP prior on Mean-1
    // -----------------------------------------------------------------------
    section("6", "Building GP prior on Mean-1 reference")
    val prior2 = NonRigidReg.buildPrior(meanRef)
    println(s"  GP rank: ${prior2.rank}")

    // -----------------------------------------------------------------------
    // 7.  PASS 2 – register all specimens → Mean-1 reference
    // -----------------------------------------------------------------------
    section("7", s"Pass 2 – registering ${specsWithLms.length} specimens to Mean-1 reference")
    val pass2Dir = new File(outDir, "pass2"); pass2Dir.mkdirs()

    val pass2Pairs: IndexedSeq[(String, TriangleMesh[_3D])] =
      specsWithLms.zipWithIndex.map { case (spec, idx) =>
        println(s"  [${idx + 1}/${specsWithLms.length}] ${spec.modelId}")

        val raw = ScapulaData.loadMesh(spec.file)
        val lms = landmarks(spec.modelId)
        val (mesh, meshLms) =
          if (spec.isRight) (ScapulaData.mirrorMesh(raw), ScapulaData.mirrorLandmarks(lms))
          else              (raw, lms)

        val (rigidAligned, _) =
          RigidAlign.landmarkThenIcp(mesh, meshLms, meanRef, meanRefLms, icpIterations = 30)
        val registered = NonRigidReg.registerOne(meanRef, rigidAligned, prior2)

        // Save final registered mesh with clean name for downstream use
        MeshIO.writeMesh(registered, new File(pass2Dir, s"reg_${spec.modelId}.stl")).get
        val st = Metrics.symmetric(registered, rigidAligned)
        println(f"    ${st.render}")

        spec.modelId -> registered
      }

    // -----------------------------------------------------------------------
    // 8.  Compute Mean-2 (the SSM mean shape; used as SSM reference)
    // -----------------------------------------------------------------------
    section("8", "Computing Mean-2 from Pass 2")
    val mean2 = NonRigidReg.meanMesh(meanRef, pass2Pairs.map(_._2))
    MeshIO.writeMesh(mean2, new File(outDir, "mean_ref_pass2.stl")).get

    val stab2 = Metrics.symmetric(meanRef, mean2)
    println(s"  Mean-1 → Mean-2 surface distance (stability): ${stab2.render}")

    // -----------------------------------------------------------------------
    // 9.  Build SSM
    // -----------------------------------------------------------------------
    section("9", "Building SSM")

    val pass2Meshes = pass2Pairs.map(_._2)

    // Full SSM (all specimens including both sides)
    val fullModel = buildSSM(meanRef, pass2Meshes)
    StatisticalModelIO.writeStatisticalTriangleMeshModel3D(
      fullModel, new File(outDir, "ssm_full.h5")
    ).get
    println(s"  Full SSM: ${pass2Meshes.length} shapes, ${fullModel.rank} modes")

    // Independent SSM (one side per subject – avoids inflating n with bilateral pairs)
    val indepMeshes: IndexedSeq[TriangleMesh[_3D]] = if (Config.buildIndependentModel) {
      val grouped = pass2Pairs.groupBy { case (id, _) => ScapulaData.subjectKey(id) }
      val oneSideEach = grouped.values.toIndexedSeq.map { group =>
        // Prefer the left side; fall back to right if no left found
        group.find { case (id, _) => !id.endsWith("_R") }
          .orElse(group.headOption)
          .get._2
      }
      val indepModel = buildSSM(meanRef, oneSideEach)
      StatisticalModelIO.writeStatisticalTriangleMeshModel3D(
        indepModel, new File(outDir, "ssm_independent.h5")
      ).get
      println(s"  Independent SSM: ${oneSideEach.length} subjects, ${indepModel.rank} modes")
      oneSideEach
    } else pass2Meshes

    // -----------------------------------------------------------------------
    // 10.  Comprehensive evaluation
    // -----------------------------------------------------------------------
    section("10", "Comprehensive evaluation")
    SSMEval.evaluate(
      model           = fullModel,
      ssmReference    = meanRef,
      pass1Pairs      = pass1Pairs.toMap,
      pass2Pairs      = pass2Pairs.toMap,
      specimens       = specsWithLms,
      stability1      = stab1,
      stability2      = stab2,
      outDir          = outDir
    )

    banner(s"Pipeline complete.  Output: ${outDir.getAbsolutePath}")
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  def buildSSM(reference: TriangleMesh[_3D], meshes: IndexedSeq[TriangleMesh[_3D]]): StatisticalMeshModel = {
    require(meshes.nonEmpty, "No meshes provided for SSM")
    require(
      meshes.forall(_.pointSet.numberOfPoints == reference.pointSet.numberOfPoints),
      "All registered meshes must have the same vertex count as the reference"
    )
    StatisticalMeshModel.createUsingPCA(reference, meshes).get
  }

  private def banner(msg: String): Unit = {
    println()
    println("=" * 80)
    println(msg)
    println("=" * 80)
  }

  private def section(n: String, title: String): Unit = {
    println()
    println(s"[$n] $title")
  }
}
