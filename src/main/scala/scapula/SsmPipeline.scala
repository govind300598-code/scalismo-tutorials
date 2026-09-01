package scapula

import scalismo.common.{PointId, UnstructuredPoints3D}
import scalismo.geometry.*
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.mesh.{TriangleMesh, TriangleMesh3D}
import scalismo.registration.LandmarkRegistration
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.utils.Random

import java.io.{File, PrintWriter}
import scala.util.Try

object SsmPipeline {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dataDir = Config.dataDir
    val outDir  = Config.outDir
    outDir.mkdirs()
    new File(outDir, "ssm1_registered").mkdirs()
    new File(outDir, "ssm2_registered").mkdirs()

    println("=" * 70)
    println("  Scapula SSM Pipeline  (SSM1 → SSM2 → Evaluation → Comparison)")
    println("=" * 70)
    println(s"  Data   : ${dataDir.getAbsolutePath}")
    println(s"  Output : ${outDir.getAbsolutePath}")

    // =========================================================================
    // 1. Load data
    // =========================================================================
    println("\n[1] Loading data...")
    val csv = ScapulaData.csvFile(dataDir)
    val (allLandmarks, _, _) = ScapulaData.readLandmarkCsv(csv)
    val allSpecimens = ScapulaData.specimens(dataDir)
    val specimens    = allSpecimens.filter(s => allLandmarks.contains(s.modelId))
    println(s"  ${specimens.size} specimens with landmarks")

    // =========================================================================
    // 2. Reference mesh (first left specimen, alphabetical)
    // =========================================================================
    val refSpec = specimens.find(!_.isRight).getOrElse(specimens.head)
    val refRaw  = ScapulaData.loadMesh(refSpec.file)
    val refMesh = if (refRaw.pointSet.numberOfPoints > Config.modelResolution)
                    refRaw.operations.decimate(Config.modelResolution) else refRaw
    val refLms  = allLandmarks(refSpec.modelId)
    println(s"  Reference: ${refSpec.modelId}  (${refMesh.pointSet.numberOfPoints} pts)")
    MeshIO.writeMesh(refMesh, new File(outDir, "ssm1_reference.stl"))

    // =========================================================================
    // 3. Rigid alignment of all specimens to the reference
    // =========================================================================
    println("\n[2] Rigid alignment (landmark Procrustes + trimmed ICP)...")
    val rigidAligned: IndexedSeq[(String, TriangleMesh[_3D])] = specimens
      .filterNot(_.modelId == refSpec.modelId)
      .zipWithIndex
      .flatMap { case (spec, i) =>
        Try {
          val raw = ScapulaData.loadMesh(spec.file)
          val dec = if (raw.pointSet.numberOfPoints > Config.modelResolution * 3)
                      raw.operations.decimate(Config.modelResolution * 2) else raw
          val (mesh, lms) =
            if (spec.isRight)
              (ScapulaData.mirrorMesh(dec), ScapulaData.mirrorLandmarks(allLandmarks(spec.modelId)))
            else
              (dec, allLandmarks(spec.modelId))
          val (aligned, _) = RigidAlign.landmarkThenIcp(mesh, lms, refMesh, refLms)
          print(s"  rigid [${i + 1}/${specimens.size - 1}] ${spec.modelId}\r")
          (spec.modelId, aligned)
        }.toOption
      }
    println(s"\n  -> ${rigidAligned.size} shapes rigidly aligned")

    // =========================================================================
    // 4. GP-ICP → SSM1 registered shapes
    // =========================================================================
    println("\n[3] Non-rigid GP-ICP registration → SSM1...")
    val ssm1RegDir = new File(outDir, "ssm1_registered")
    // Only register shapes that don't already have a file (restart-safe).
    val ssm1Targets = rigidAligned.filterNot { case (id, _) =>
      new File(ssm1RegDir, s"reg_$id.stl").exists()
    }
    val ssm1Cached = rigidAligned.filter { case (id, _) =>
      new File(ssm1RegDir, s"reg_$id.stl").exists()
    }.map { case (id, _) =>
      (id, ScapulaData.loadMesh(new File(ssm1RegDir, s"reg_$id.stl")))
    }
    val ssm1NewReg = if (ssm1Targets.nonEmpty)
      NonRigidReg.registerAll(refMesh, ssm1Targets, Config.icpIterations, ssm1RegDir)
    else IndexedSeq.empty
    val ssm1RegShapes = (ssm1Cached ++ ssm1NewReg).sortBy(_._1)
    println(s"  -> ${ssm1RegShapes.size} shapes in correspondence (SSM1)")

    // =========================================================================
    // 5. GPA on SSM1 shapes, then build SSM1
    // =========================================================================
    println("\n[4] GPA + PCA → SSM1...")
    val (ssm1GpaShapes, ssm1GpaMean) = runGpa(ssm1RegShapes.map(_._2), refMesh)
    MeshIO.writeMesh(ssm1GpaMean, new File(outDir, "ssm1_gpa_mean.stl"))
    val ssm1Dc    = DataCollection.fromTriangleMesh3DSequence(refMesh, ssm1GpaShapes)
    val ssm1      = PointDistributionModel.createUsingPCA(ssm1Dc)
    val ssm1Mean  = ssm1.mean
    println(s"  SSM1: ${ssm1.rank} modes from ${ssm1GpaShapes.size} shapes")
    MeshIO.writeMesh(ssm1Mean, new File(outDir, "ssm1_mean.stl"))
    Try(StatisticalModelIO.writeStatisticalTriangleMeshModel3D(ssm1, new File(outDir, "ssm1.h5")))
      .fold(e => println(s"  WARNING SSM1 save failed: $e"), _ => println("  Saved -> ssm1.h5"))

    // =========================================================================
    // 6. Transfer landmarks to SSM1 mean via mesh correspondence
    // =========================================================================
    val ssm1MeanLms: IndexedSeq[Landmark[_3D]] = refLms.map { lm =>
      val closestId = refMesh.pointSet.findClosestPoint(lm.point).id
      lm.copy(point = ssm1Mean.pointSet.point(closestId))
    }

    // =========================================================================
    // 7. Rigid alignment to SSM1 mean reference
    // =========================================================================
    println("\n[5] Rigid alignment to SSM1 mean reference...")
    val ssm2RigidAligned: IndexedSeq[(String, TriangleMesh[_3D])] = specimens
      .filterNot(_.modelId == refSpec.modelId)
      .zipWithIndex
      .flatMap { case (spec, i) =>
        Try {
          val raw = ScapulaData.loadMesh(spec.file)
          val dec = if (raw.pointSet.numberOfPoints > Config.modelResolution * 3)
                      raw.operations.decimate(Config.modelResolution * 2) else raw
          val (mesh, lms) =
            if (spec.isRight)
              (ScapulaData.mirrorMesh(dec), ScapulaData.mirrorLandmarks(allLandmarks(spec.modelId)))
            else
              (dec, allLandmarks(spec.modelId))
          val (aligned, _) = RigidAlign.landmarkThenIcp(mesh, lms, ssm1Mean, ssm1MeanLms)
          print(s"  rigid [${i + 1}/${specimens.size - 1}] ${spec.modelId}\r")
          (spec.modelId, aligned)
        }.toOption
      }
    println(s"\n  -> ${ssm2RigidAligned.size} shapes rigidly aligned to SSM1 mean")

    // =========================================================================
    // 8. GP-ICP → SSM2 registered shapes
    // =========================================================================
    println("\n[6] Non-rigid GP-ICP registration → SSM2...")
    val ssm2RegDir = new File(outDir, "ssm2_registered")
    val ssm2Targets = ssm2RigidAligned.filterNot { case (id, _) =>
      new File(ssm2RegDir, s"reg_$id.stl").exists()
    }
    val ssm2Cached = ssm2RigidAligned.filter { case (id, _) =>
      new File(ssm2RegDir, s"reg_$id.stl").exists()
    }.map { case (id, _) =>
      (id, ScapulaData.loadMesh(new File(ssm2RegDir, s"reg_$id.stl")))
    }
    val ssm2NewReg = if (ssm2Targets.nonEmpty)
      NonRigidReg.registerAll(ssm1Mean, ssm2Targets, Config.icpIterations, ssm2RegDir)
    else IndexedSeq.empty
    val ssm2RegShapes = (ssm2Cached ++ ssm2NewReg).sortBy(_._1)
    println(s"  -> ${ssm2RegShapes.size} shapes in correspondence (SSM2)")

    // =========================================================================
    // 9. GPA on SSM2 shapes, then build SSM2
    // =========================================================================
    println("\n[7] GPA + PCA → SSM2...")
    val (ssm2GpaShapes, ssm2GpaMean) = runGpa(ssm2RegShapes.map(_._2), ssm1Mean)
    MeshIO.writeMesh(ssm2GpaMean, new File(outDir, "ssm2_gpa_mean.stl"))
    val ssm2Dc   = DataCollection.fromTriangleMesh3DSequence(ssm1Mean, ssm2GpaShapes)
    val ssm2     = PointDistributionModel.createUsingPCA(ssm2Dc)
    val ssm2Mean = ssm2.mean
    println(s"  SSM2: ${ssm2.rank} modes from ${ssm2GpaShapes.size} shapes")
    MeshIO.writeMesh(ssm2Mean, new File(outDir, "ssm2_mean.stl"))
    Try(StatisticalModelIO.writeStatisticalTriangleMeshModel3D(ssm2, new File(outDir, "ssm2.h5")))
      .fold(e => println(s"  WARNING SSM2 save failed: $e"), _ => println("  Saved -> ssm2.h5"))

    // =========================================================================
    // 10. Evaluate SSM1 and SSM2
    // =========================================================================
    println("\n" + "=" * 70)
    println("  EVALUATION")
    println("=" * 70)
    println("\n--- SSM1 ---")
    val ssm1Metrics = Evaluate.evaluateModel("ssm1", ssm1, refMesh, ssm1GpaShapes, outDir)
    println("\n--- SSM2 ---")
    val ssm2Metrics = Evaluate.evaluateModel("ssm2", ssm2, ssm1Mean, ssm2GpaShapes, outDir)

    // =========================================================================
    // 11. Compare SSM1 mean vs SSM2 mean shape
    // =========================================================================
    println("\n--- Mean Shape Comparison (SSM1 mean vs SSM2 mean) ---")
    // SSM2 mean is in SSM1 mean coordinate frame (same topology).
    val meanComparison = Metrics.correspondingDistances(ssm1Mean, ssm2Mean)
    val compMean = meanComparison.sum / meanComparison.size
    val compRms  = math.sqrt(meanComparison.map(d => d * d).sum / meanComparison.size)
    val compMax  = meanComparison.max
    val compHd95 = Metrics.percentile(meanComparison, 0.95)
    println(f"  Mean vertex distance : $compMean%.4f mm")
    println(f"  RMSE                 : $compRms%.4f mm")
    println(f"  Max                  : $compMax%.4f mm")
    println(f"  HD95                 : $compHd95%.4f mm")

    val cmpPw = new PrintWriter(new File(outDir, "mean_shape_comparison.csv"))
    cmpPw.println("Metric,Value_mm")
    cmpPw.println(s"MeanDistance,$compMean")
    cmpPw.println(s"RMSE,$compRms")
    cmpPw.println(s"HD95,$compHd95")
    cmpPw.println(s"MaxDistance,$compMax")
    cmpPw.close()

    // =========================================================================
    // 12. Summary table
    // =========================================================================
    println("\n" + "=" * 70)
    println("  FINAL COMPARISON TABLE")
    println("=" * 70)
    println(f"  ${"Metric"}%-32s ${"SSM1"}%10s ${"SSM2"}%10s")
    println("  " + "-" * 54)
    println(f"  ${"Training meshes"}%-32s ${ssm1GpaShapes.size}%10d ${ssm2GpaShapes.size}%10d")
    println(f"  ${"PCA modes"}%-32s ${ssm1.rank}%10d ${ssm2.rank}%10d")
    println(f"  ${"Modes for 90% variance"}%-32s ${ssm1Metrics.compactness90pct}%10d ${ssm2Metrics.compactness90pct}%10d")
    println(f"  ${"Modes for 95% variance"}%-32s ${ssm1Metrics.compactness95pct}%10d ${ssm2Metrics.compactness95pct}%10d")
    println(f"  ${"Generalization (mm, all modes)"}%-32s ${ssm1Metrics.generalizationMean}%10.4f ${ssm2Metrics.generalizationMean}%10.4f")
    println(f"  ${"Specificity (mm, all modes)"}%-32s ${ssm1Metrics.specificityMean}%10.4f ${ssm2Metrics.specificityMean}%10.4f")
    println(f"  ${"Mean shape distance (mm)"}%-32s ${compMean}%10.4f ${"(SSM1 vs SSM2)"}%10s")
    println(f"  ${"RMSE between mean shapes (mm)"}%-32s ${compRms}%10.4f ${"(SSM1 vs SSM2)"}%10s")

    println(s"\n  All outputs in: ${outDir.getAbsolutePath}")
    println("  Done.")
  }

  // ── GPA utilities (same as ScapulaFullSSMPipeline) ──────────────────────────

  private val origin = Point3D(0.0, 0.0, 0.0)

  private def vtxMean(meshes: IndexedSeq[TriangleMesh[_3D]]): TriangleMesh[_3D] = {
    val ref    = meshes.head
    val invN   = 1.0 / meshes.size
    val meanPts = (0 until ref.pointSet.numberOfPoints).map { i =>
      val ptId = PointId(i)
      val vsum = meshes.foldLeft(EuclideanVector.zeros[_3D]) { (acc, m) =>
        acc + (m.pointSet.point(ptId) - origin)
      }
      origin + vsum * invN
    }
    TriangleMesh3D(UnstructuredPoints3D(meanPts), ref.triangulation)
  }

  private def rigidToMean(mesh: TriangleMesh[_3D], mean: TriangleMesh[_3D]): TriangleMesh[_3D] = {
    val pairs = mesh.pointSet.points.zip(mean.pointSet.points).toIndexedSeq
    val trans = LandmarkRegistration.rigid3DLandmarkRegistration(pairs, origin)
    mesh.transform(trans)
  }

  private def runGpa(
    meshes:  IndexedSeq[TriangleMesh[_3D]],
    refMesh: TriangleMesh[_3D],
    maxIter: Int = 8,
    tol:     Double = 0.05
  ): (IndexedSeq[TriangleMesh[_3D]], TriangleMesh[_3D]) = {
    var cur       = meshes
    var prevLoss  = Double.MaxValue
    var finalMean = refMesh
    for (iter <- 0 until maxIter) {
      finalMean = vtxMean(cur)
      cur       = cur.map(m => rigidToMean(m, finalMean))
      val loss: Double = cur.map { m =>
        val d = Metrics.correspondingDistances(m, finalMean)
        d.sum / d.size
      }.sum / cur.size
      println(f"  GPA iter ${iter + 1}: mean vertex dist = $loss%.4f mm")
      if (math.abs(prevLoss - loss) < tol) { println("  GPA converged"); return (cur, finalMean) }
      prevLoss = loss
    }
    (cur, finalMean)
  }
}
