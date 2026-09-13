package scapula

import scalismo.geometry._3D
import scalismo.io.MeshIO
import scalismo.mesh.TriangleMesh
import scalismo.utils.Random

import java.io.File

/**
 * Entry point for the scapula SSM pipeline.
 *
 * Pipeline:
 *   1. Load specimens + landmark CSV
 *   2. Decimate originals to ~8k working meshes (idempotent)
 *   3. Pick reference specimen
 *   4. For each specimen:
 *        a. Load 8k mesh; mirror right scapulae to left frame
 *        b. Landmark Procrustes + trimmed ICP → rigid_registered/
 *        c. GP-ICP non-rigid registration (Nystrom, σ=13mm, basis=70) → pass_1/
 *   5. PCA → PointDistributionModel → scapula_ssm.h5
 *
 * Configure via environment variables:
 *   SCAPULA_DATA_DIR      folder with STL files and landmark CSV
 *   SCAPULA_OUT_DIR       root output folder (default: scapula_output)
 *   SCAPULA_MODEL_RES     target vertex count after decimation (default 8000)
 *   SCAPULA_ICP_ITERS     rigid ICP iterations (default 40)
 *   SCAPULA_GP_SIGMA      Gaussian kernel σ in mm (default 13.0)
 *   SCAPULA_GP_SCALE      Gaussian kernel amplitude (default 30.0)
 *   SCAPULA_GP_BASIS      Nystrom basis functions (default 70)
 *   SCAPULA_GP_ICP_ITER   GP-ICP iterations (default 8)
 *   SCAPULA_GP_NOISE      GP posterior noise σ² (default 1.0)
 *
 * Run with:
 *   sbt "runMain scapula.Main"
 */
object Main {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dataDir  = Config.dataDir
    val outDir   = Config.outDir
    val preDir   = new File(outDir, "data/8k")
    val rigidDir = new File(outDir, "rigid_registered")
    val modelDir = new File(outDir, "model")

    Seq(preDir, rigidDir, modelDir).foreach(_.mkdirs())

    println("=" * 80)
    println("  Scapula SSM Pipeline")
    println("=" * 80)
    println(s"  Data dir : ${dataDir.getAbsolutePath}")
    println(s"  Out dir  : ${outDir.getAbsolutePath}")
    println(s"  GP kernel: σ=${Config.gpSigma}mm  scale=${Config.gpScale}  basis=${Config.gpBasis}  noise=${Config.gpNoise}")
    println(s"  GP-ICP iters: ${Config.gpIcpIter}")

    // ── 1. Load specimens and landmarks ──────────────────────────────────────
    val csvFile = ScapulaData.csvFile(dataDir)
    println(s"\nLandmark CSV: ${csvFile.getName}")
    val (lmMap, fromHeader, _) = ScapulaData.readLandmarkCsv(csvFile)
    if (!fromHeader)
      println("  WARNING: landmark columns resolved by fallback offsets — verify CSV header")

    val allSpecs   = ScapulaData.specimens(dataDir)
    val validSpecs = allSpecs.filter(s => lmMap.contains(s.modelId))
    println(s"  ${validSpecs.length} / ${allSpecs.length} specimens have landmark rows")
    require(validSpecs.nonEmpty, "No specimens with landmarks found. Check SCAPULA_DATA_DIR.")

    // ── 2. Decimate to 8k (idempotent) ───────────────────────────────────────
    println(s"\n[DECIMATION] ~${Config.modelResolution}-vertex working meshes → ${preDir.getPath}")
    Decimation.generateAll(validSpecs, preDir, Config.modelResolution)

    // ── 3. Pick reference specimen ────────────────────────────────────────────
    val refIdx  = 0.max(validSpecs.length - 1).min(2)
    val refSpec = validSpecs(refIdx)
    val refLms  = if (refSpec.isRight) ScapulaData.mirrorLandmarks(lmMap(refSpec.modelId))
                  else lmMap(refSpec.modelId)
    println(s"\n[REFERENCE] index=$refIdx  id=${refSpec.modelId}")

    val refRaw  = ScapulaData.loadMesh(new File(preDir, refSpec.modelId + ".stl"))
    val refMesh = Decimation.keepLargestComponent(refRaw)
    println(f"  ${refMesh.pointSet.numberOfPoints} vertices (after keepLargestComponent)")

    MeshIO.writeMesh(refMesh, new File(modelDir, "reference.stl"))
      .fold(e => println(s"  WARNING: could not save reference: $e"), _ => ())

    // ── 4. Register all specimens ─────────────────────────────────────────────
    println(s"\n[REGISTRATION] ${validSpecs.length} specimens  " +
            s"(rigid ICP iters=${Config.icpIterations}, GP-ICP iters=${Config.gpIcpIter})")

    val registered = scala.collection.mutable.ArrayBuffer.empty[TriangleMesh[_3D]]

    validSpecs.zipWithIndex.foreach { case (spec, specIdx) =>
      val specLms = lmMap(spec.modelId)
      try {
        print(s"  [${specIdx + 1}/${validSpecs.length}] ${spec.modelId} ")

        val raw8k  = ScapulaData.loadMesh(new File(preDir, spec.modelId + ".stl"))
        val mesh8k = Decimation.keepLargestComponent(raw8k)
        val (workMesh, workLms) =
          if (spec.isRight) (ScapulaData.mirrorMesh(mesh8k), ScapulaData.mirrorLandmarks(specLms))
          else (mesh8k, specLms)

        // a. Landmark Procrustes + trimmed rigid ICP
        val (rigidMesh, _) = RigidAlign.landmarkThenIcp(
          workMesh, workLms, refMesh, refLms,
          icpIterations = Config.icpIterations)
        MeshIO.writeMesh(rigidMesh, new File(rigidDir, spec.modelId + ".stl"))
          .getOrElse(throw new RuntimeException("rigid write failed"))
        print("[rigid] ")

        // b. GP-ICP non-rigid registration — Nystrom, σ=13mm, all correspondences
        val nrMesh = SSMBuilder.loadOneMesh("pass_1", specIdx).getOrElse {
          val m = NonRigidReg.register(reference = refMesh, target = rigidMesh)
          SSMBuilder.saveOneMesh(m, "pass_1", specIdx)
          m
        }
        print("[nreg] ")

        val sd = Metrics.symmetric(nrMesh, rigidMesh)
        println(f"done  reg-dist mean=${sd.mean}%5.2f mm  hd95=${sd.hd95}%5.2f mm")

        registered += nrMesh
      } catch {
        case e: Exception => println(s"\n  ERROR: ${e.getMessage}")
      }
    }

    println(s"\n  ${registered.length} / ${validSpecs.length} successful registrations")
    require(registered.nonEmpty, "No successful registrations — cannot build SSM.")

    // ── 5. PCA → PointDistributionModel ──────────────────────────────────────
    println(s"\n[SSM]  PCA on ${registered.length} registered meshes")
    val ssm = SSMBuilder.buildSSM(refMesh, registered.toIndexedSeq)
    println(s"  SSM rank = ${ssm.rank}")
    SSMBuilder.saveSSM(ssm, "scapula_ssm")

    val meanFile = new File(outDir, "mean.stl")
    MeshIO.writeMesh(ssm.mean, meanFile)
      .fold(e => println(s"  WARNING: mean save failed: $e"), _ => println(s"  Mean → ${meanFile.getPath}"))

    println("\n" + "=" * 80)
    println(s"  Pipeline complete.")
    println(s"  Model → ${new File(outDir, "scapula_ssm.h5").getPath}")
    println("=" * 80)
  }
}
