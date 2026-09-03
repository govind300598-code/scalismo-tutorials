package ssmpipeline

import scalismo.common.PointId
import scalismo.geometry.*
import scalismo.io.StatisticalModelIO
import scalismo.mesh.*
import scalismo.registration.LandmarkRegistration
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.utils.Random

import java.io.File

/**
 * Main pipeline: rigid alignment → GP-ICP non-rigid registration → GPA → PCA → SSM evaluation.
 *
 * SSM1: built from all registered specimens (both left and mirrored-right brought to left-side frame).
 * SSM2: built from one specimen per subject (left-side preferred), removing bilateral pseudo-replication.
 *       Bilateral scapulae from the same person are not statistically independent; including both inflates
 *       apparent sample size and narrows the model's apparent variability.
 *
 * Paths are controlled via environment variables — see Config in ScapulaData.scala.
 */
object SsmPipeline {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val outDir = Config.outDir
    outDir.mkdirs()

    // ---------------------------------------------------------------------- [1] load
    val dir       = Config.dataDir
    val specimens = ScapulaData.specimens(dir)
    val csv       = ScapulaData.csvFile(dir)
    val (landmarks, _, _) = ScapulaData.readLandmarkCsv(csv)
    println(s"[1] ${specimens.length} specimens, landmark CSV: ${csv.getName}")

    // ---------------------------------------------------------------------- [2] reference
    val leftWithLms = specimens.filter(s => !s.isRight && landmarks.contains(s.modelId))
    require(leftWithLms.nonEmpty, "No left-side specimens with landmarks found.")
    val refSpec   = leftWithLms.head
    val rawRef    = ScapulaData.loadMesh(refSpec.file)
    val reference = rawRef.operations.decimate(Config.modelResolution)
    val refLms    = landmarks(refSpec.modelId)
    println(s"[2] Reference: ${refSpec.modelId} (${reference.pointSet.numberOfPoints} vertices after decimation)")

    // ---------------------------------------------------------------------- [3] GP model
    println(s"[3] Building GP model (tol=${Config.gpRelativeTolerance}, maxRank=${Config.gpMaxRank})...")
    val gpModel = NonRigidReg.buildGpModel(reference)
    println(s"    Rank: ${gpModel.gp.klBasis.length}")

    // ---------------------------------------------------------------------- [4] rigid + GP-ICP
    println(s"[4] Rigid alignment + GP-ICP (${Config.icpIterations} iterations per specimen)...")
    val registered: IndexedSeq[(ScapulaData.Specimen, TriangleMesh[_3D])] =
      specimens.zipWithIndex.flatMap { case (s, idx) =>
        landmarks.get(s.modelId).map { lms =>
          // Mirror right scapulae so they look like left before registering to the left-side reference.
          val (rawMesh, rawLms) =
            if (s.isRight) (ScapulaData.mirrorMesh(ScapulaData.loadMesh(s.file)), ScapulaData.mirrorLandmarks(lms))
            else (ScapulaData.loadMesh(s.file), lms)

          val (rigidMesh, _) = RigidAlign.landmarkThenIcp(rawMesh, rawLms, reference, refLms)
          val reg             = NonRigidReg.register(gpModel, rigidMesh, Config.icpIterations)
          println(s"    GP-ICP [${idx + 1}/${specimens.length}] ${s.modelId}")
          (s, reg)
        }
      }
    println(s"    -> ${registered.length} shapes in correspondence")

    // ---------------------------------------------------------------------- [5] helper
    def buildSsm(shapes: IndexedSeq[TriangleMesh[_3D]], label: String): PointDistributionModel[_3D, TriangleMesh] = {
      val aligned = gpa(shapes, label)
      val model   = PointDistributionModel.createUsingPCA(aligned)
      println(s"    $label: ${model.gp.klBasis.length} modes from ${shapes.length} shapes")
      model
    }

    // ---------------------------------------------------------------------- [6] SSM1
    val allShapes = registered.map(_._2)
    println(s"\n[6] GPA + PCA → SSM1...")
    val ssm1 = buildSsm(allShapes, "ssm1")
    val ssm1File = new File(outDir, "ssm1.h5")
    StatisticalModelIO.writeStatisticalMeshModel(ssm1, ssm1File).get
    println(s"    Saved -> ${ssm1File.getName}")

    // ---------------------------------------------------------------------- [7] SSM2 (independent)
    val indepShapes: IndexedSeq[TriangleMesh[_3D]] = registered
      .groupBy { case (s, _) => s.subject }
      .toIndexedSeq
      .sortBy(_._1)
      .map { case (_, group) =>
        group.find(!_._1.isRight).getOrElse(group.head)._2
      }
    println(s"\n[7] GPA + PCA → SSM2...")
    val ssm2 = buildSsm(indepShapes, "ssm2")
    val ssm2File = new File(outDir, "ssm2.h5")
    StatisticalModelIO.writeStatisticalMeshModel(ssm2, ssm2File).get
    println(s"    Saved -> ${ssm2File.getName}")

    // ---------------------------------------------------------------------- [8] evaluate
    println(s"\n${"=" * 65}")
    println("EVALUATION")
    println("=" * 65)

    val c1_90 = Evaluate.compactnessAt(ssm1, 0.90)
    val c1_95 = Evaluate.compactnessAt(ssm1, 0.95)
    val c2_90 = Evaluate.compactnessAt(ssm2, 0.90)
    val c2_95 = Evaluate.compactnessAt(ssm2, 0.95)
    val (g1m, g1s) = Evaluate.generalization(allShapes)
    val (g2m, g2s) = Evaluate.generalization(indepShapes)
    val (sp1m, sp1s) = Evaluate.specificity(ssm1, allShapes)
    val (sp2m, sp2s) = Evaluate.specificity(ssm2, indepShapes)

    Seq(("SSM1", ssm1, c1_90, c1_95, g1m, g1s, sp1m, sp1s, allShapes),
        ("SSM2", ssm2, c2_90, c2_95, g2m, g2s, sp2m, sp2s, indepShapes))
      .foreach { case (label, _, c90, c95, gm, gs, sm, ss, shapes) =>
        println(s"\n--- $label ---")
        println(s"   --- Compactness ($label) ---")
        println(s"     90% variance: $c90 modes   95% variance: $c95 modes")
        println(s"   --- Generalization ($label, LOO on ${shapes.length} shapes) ---")
        println(f"     Best (all modes): mean=$gm%.4f  std=$gs%.4f mm")
        println(s"   --- Specificity ($label) ---")
        println(f"     Best (all modes): mean=$sm%.4f  std=$ss%.4f mm")
      }

    // ---------------------------------------------------------------------- [9] mean shape comparison
    val meanDists = Metrics.correspondingDistances(ssm1.mean, ssm2.mean)
    val mMean = meanDists.sum / meanDists.length
    val mRms  = math.sqrt(meanDists.map(d => d * d).sum / meanDists.length)
    val mMax  = meanDists.max
    val mHd95 = Metrics.percentile(meanDists, 0.95)
    println(s"\n--- Mean Shape Comparison (SSM1 mean vs SSM2 mean) ---")
    println(f"  Mean vertex distance : $mMean%.4f mm")
    println(f"  RMSE                  : $mRms%.4f mm")
    println(f"  Max                   : $mMax%.4f mm")
    println(f"  HD95                  : $mHd95%.4f mm")

    // ---------------------------------------------------------------------- final table
    println(s"\n${"=" * 65}")
    println("FINAL COMPARISON TABLE")
    println("=" * 65)
    println(f"  ${"Metric"}%-40s ${"SSM1"}%8s  ${"SSM2"}%8s")
    println(f"  ${"-" * 58}")
    println(f"  ${"Training meshes"}%-40s ${allShapes.length}%8d  ${indepShapes.length}%8d")
    println(f"  ${"PCA modes"}%-40s ${ssm1.gp.klBasis.length}%8d  ${ssm2.gp.klBasis.length}%8d")
    println(f"  ${"Modes for 90% variance"}%-40s $c1_90%8d  $c2_90%8d")
    println(f"  ${"Modes for 95% variance"}%-40s $c1_95%8d  $c2_95%8d")
    println(f"  ${"Generalization (mm, all modes)"}%-40s $g1m%8.4f  $g2m%8.4f")
    println(f"  ${"Specificity (mm, all modes)"}%-40s $sp1m%8.4f  $sp2m%8.4f")
    println(f"  ${"Mean shape distance (mm)"}%-40s $mMean%8.4f  ${" " * 8}")
    println(f"  ${"RMSE between mean shapes (mm)"}%-40s $mRms%8.4f  ${" " * 8}")

    println(s"\nAll outputs in: ${outDir.getAbsolutePath}")
    println("Done.")
  }

  // ---------------------------------------------------------------------------
  // Generalized Procrustes Analysis
  //
  // All shapes are already in point-to-point correspondence (same topology as the reference). GPA rigidly
  // aligns them to their common mean, iterating until convergence. This removes residual pose differences that
  // survive rigid ICP, so PCA sees only shape variability.
  // ---------------------------------------------------------------------------
  private def gpa(shapes: IndexedSeq[TriangleMesh[_3D]],
                  label: String,
                  maxIter: Int = 10,
                  tol: Double = 0.005
  )(implicit rng: Random): IndexedSeq[TriangleMesh[_3D]] = {
    var aligned     = shapes
    var prevMeanPts = correspondenceMean(shapes).pointSet.points.toIndexedSeq

    for (iter <- 1 to maxIter) {
      val newMean    = correspondenceMean(aligned)
      val nextAligned = aligned.map(m => rigidAlignTo(m, newMean))
      val nextMeanPts = correspondenceMean(nextAligned).pointSet.points.toIndexedSeq
      val change      = prevMeanPts.zip(nextMeanPts).map { case (a, b) => (a - b).norm }.sum / prevMeanPts.length
      println(f"  GPA iter $iter: mean vertex dist = $change%.4f mm")
      aligned     = nextAligned
      prevMeanPts = nextMeanPts
      if (change < tol) {
        println("  GPA converged")
        return aligned
      }
    }
    aligned
  }

  private def correspondenceMean(meshes: IndexedSeq[TriangleMesh[_3D]]): TriangleMesh[_3D] = {
    val n    = meshes.length
    val ref  = meshes.head
    val pts  = (0 until ref.pointSet.numberOfPoints).map { i =>
      val id             = PointId(i)
      var sx = 0.0; var sy = 0.0; var sz = 0.0
      meshes.foreach { m => val p = m.pointSet.point(id); sx += p.x; sy += p.y; sz += p.z }
      Point3D(sx / n, sy / n, sz / n)
    }
    TriangleMesh3D(pts, ref.triangulation)
  }

  private def rigidAlignTo(mesh: TriangleMesh[_3D], target: TriangleMesh[_3D]): TriangleMesh[_3D] = {
    val pairs = mesh.pointSet.points.zip(target.pointSet.points).toIndexedSeq
    val trans = LandmarkRegistration.rigid3DLandmarkRegistration(pairs, center = Point3D(0, 0, 0))
    mesh.transform(trans)
  }
}
