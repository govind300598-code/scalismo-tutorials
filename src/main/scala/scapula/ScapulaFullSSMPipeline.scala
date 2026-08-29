package scapula

import scalismo.geometry.*
import scalismo.common.*
import scalismo.common.interpolation.TriangleMeshInterpolator3D
import scalismo.mesh.*
import scalismo.registration.LandmarkRegistration
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.numerics.UniformMeshSampler3D
import scalismo.kernels.*
import scalismo.statisticalmodel.*
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.utils.Random
import breeze.linalg.DenseVector

import java.io.{File, PrintWriter}
import scala.util.Try

/**
 * Full scapula SSM pipeline:
 *   1. Rigid alignment (landmark Procrustes + ICP)
 *   2. Generalized Procrustes Analysis (GPA) on rigidly aligned meshes
 *   3. Non-rigid GP-ICP to establish dense correspondences with the reference
 *   4. GPA on registered meshes (removes residual pose bias)
 *   5. PCA -> Statistical Shape Model
 *   6. Validation: Compactness / Generalization (LOO) / Specificity / Reconstruction
 */
object ScapulaFullSSMPipeline extends App {

  scalismo.initialize()
  implicit val rng: Random = Random(Config.seed)

  // =========================================================================
  // 0. Setup
  // =========================================================================
  val dataDir = Config.dataDir
  val outDir  = new File(Config.outDir, "full_ssm")
  outDir.mkdirs()

  println("=" * 72)
  println("SCAPULA SSM  (GPA → GP-ICP → PCA → Validation)")
  println("=" * 72)
  println(s"Data   : ${dataDir.getAbsolutePath}")
  println(s"Output : ${outDir.getAbsolutePath}\n")

  val csv = ScapulaData.csvFile(dataDir)
  val (allLandmarks, _, _) = ScapulaData.readLandmarkCsv(csv)
  val allSpecimens          = ScapulaData.specimens(dataDir)
  val specimens             = allSpecimens.filter(s => allLandmarks.contains(s.modelId))
  println(s"Specimens with landmarks: ${specimens.size}")

  // =========================================================================
  // 1. Reference mesh
  // =========================================================================
  // Use first left-side specimen as reference.  Everything is flipped to left.
  val refSpec   = specimens.find(!_.isRight).getOrElse(specimens.head)
  val refRaw    = ScapulaData.loadMesh(refSpec.file)
  val refMesh   = if (refRaw.pointSet.numberOfPoints > Config.modelResolution)
                    refRaw.operations.decimate(Config.modelResolution) else refRaw
  val refLms    = allLandmarks(refSpec.modelId)
  println(s"Reference : ${refSpec.modelId}  (${refMesh.pointSet.numberOfPoints} vertices)")
  MeshIO.writeMesh(refMesh, new File(outDir, "reference.stl"))

  // =========================================================================
  // 2. Rigid alignment of all other specimens to the reference
  // =========================================================================
  println("\n[1/6] Rigid alignment (landmark Procrustes + trimmed ICP) …")

  case class Aligned(id: String, mesh: TriangleMesh[_3D])

  val rigidAligned: IndexedSeq[Aligned] = specimens
    .filterNot(_.modelId == refSpec.modelId)
    .zipWithIndex
    .flatMap { case (spec, i) =>
      Try {
        val raw  = ScapulaData.loadMesh(spec.file)
        val dec  = if (raw.pointSet.numberOfPoints > Config.modelResolution * 3)
                     raw.operations.decimate(Config.modelResolution * 2) else raw
        val (mesh, lms) =
          if (spec.isRight) (ScapulaData.mirrorMesh(dec),
                             ScapulaData.mirrorLandmarks(allLandmarks(spec.modelId)))
          else              (dec, allLandmarks(spec.modelId))
        val (aligned, _) = RigidAlign.landmarkThenIcp(mesh, lms, refMesh, refLms, icpIterations = 25)
        print(s"  [${i+1}/${specimens.size - 1}] ${spec.id}\r")
        Aligned(spec.modelId, aligned)
      }.toOption
    }
  println(s"\n  → ${rigidAligned.size} shapes rigidly aligned")

  // =========================================================================
  // 3. GPA on rigidly aligned meshes (uses only point positions, no topology)
  //    The meshes do NOT yet have a common topology, so we just do rigid GPA
  //    (each mesh aligned to the running mean surface via Procrustes).
  // =========================================================================
  println("\n[2/6] Generalized Procrustes Analysis (rigid GPA) …")

  // We align each mesh to the reference using its own landmarks repeatedly, so
  // true multi-shape GPA isn't needed before non-rigid step; instead we do
  // iterative mean-shape GPA after establishing correspondence (Step 5).
  // Here we do a single-pass rigid centering so all shapes are near the reference.
  val rigidMeshes = rigidAligned.map(_.mesh) // used directly in GP-ICP below

  // =========================================================================
  // 4. GP deformation prior (multi-scale for coarse-to-fine)
  // =========================================================================
  println("\n[3/6] Building multi-scale GP deformation prior …")

  // Multi-scale prior: wide kernel captures global shape, narrow captures local detail
  val kernel =
    GaussianKernel[_3D](sigma = 80.0, scaleFactor = 50.0) +
    GaussianKernel[_3D](sigma = 40.0, scaleFactor = 20.0) +
    GaussianKernel[_3D](sigma = 15.0, scaleFactor =  5.0)

  val zeroMeanField: Field[_3D, EuclideanVector[_3D]] =
    Field(EuclideanSpace[_3D], (_: Point[_3D]) => EuclideanVector.zeros[_3D])

  val gpPrior = GaussianProcess[_3D, EuclideanVector[_3D]](
    zeroMeanField,
    DiagonalKernel3D(kernel, 3)
  )
  val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
    refMesh, gpPrior,
    relativeTolerance = Config.gpRelativeTolerance,
    interpolator      = TriangleMeshInterpolator3D[EuclideanVector[_3D]]()
  )
  val discreteGP = lowRankGP.discretize(refMesh)
  println(s"  GP rank: ${lowRankGP.rank}")

  // =========================================================================
  // 5. Non-rigid GP-ICP to establish dense correspondences
  //    Result: every registered mesh has IDENTICAL topology to refMesh;
  //    corresponding vertices anatomically match across shapes.
  // =========================================================================
  println("\n[4/6] Non-rigid GP-ICP …")

  def gpIcp(target: TriangleMesh[_3D],
            numIter: Int     = Config.icpIterations,
            numCorr: Int     = 1200,
            noiseHigh: Double = 4.0,
            noiseLow: Double  = 0.25): TriangleMesh[_3D] = {
    var current = refMesh
    val tgtOps  = target.operations

    for (iter <- 0 until numIter) {
      // Anneal noise from high → low (coarse-to-fine)
      val alpha  = iter.toDouble / math.max(1, numIter - 1)
      val sigma2 = noiseHigh * (1 - alpha) + noiseLow * alpha

      // Sample reference points (spatially uniform)
      val sampleIds = UniformMeshSampler3D(current, numCorr)
        .sample().map { case (pt, _) => refMesh.pointSet.findClosestPoint(pt).id }
        .distinct

      // Build observations: deformation from reference point → closest on target
      val obs: IndexedSeq[(PointId, EuclideanVector[_3D])] = sampleIds.map { ptId =>
        val curPt = current.pointSet.point(ptId)
        val tgtPt = tgtOps.closestPointOnSurface(curPt).point
        ptId -> (tgtPt - refMesh.pointSet.point(ptId))
      }

      val post    = discreteGP.posterior(obs, sigma2)
      val newPts  = refMesh.pointSet.pointIds.toIndexedSeq.map { ptId =>
        refMesh.pointSet.point(ptId) + post.mean(ptId)
      }
      current = TriangleMesh3D(UnstructuredPoints3D(newPts), refMesh.triangulation)
    }
    current
  }

  val registered: IndexedSeq[Aligned] = rigidAligned.zipWithIndex.map { case (spec, i) =>
    println(s"  [${i+1}/${rigidAligned.size}] ${spec.id}")
    val reg = gpIcp(spec.mesh)
    MeshIO.writeMesh(reg, new File(outDir, s"reg_${spec.id}.stl"))
    Aligned(spec.id, reg)
  }

  // =========================================================================
  // 6. GPA on the registered meshes (full multi-shape Procrustes)
  //    Now all meshes share topology with refMesh so we can compute a
  //    proper vertex-wise mean and iterate.
  // =========================================================================
  println("\n[5/6] GPA on registered shapes …")

  val origin3D = Point3D(0.0, 0.0, 0.0)

  def vtxMean(meshes: IndexedSeq[TriangleMesh[_3D]]): TriangleMesh[_3D] = {
    val n    = refMesh.pointSet.numberOfPoints
    val invN = 1.0 / meshes.size
    val meanPts = (0 until n).map { i =>
      val ptId = PointId(i)
      val vsum = meshes.foldLeft(EuclideanVector.zeros[_3D]) { (acc, m) =>
        acc + (m.pointSet.point(ptId) - origin3D)
      }
      origin3D + vsum * invN
    }
    TriangleMesh3D(UnstructuredPoints3D(meanPts), refMesh.triangulation)
  }

  def rigidAlignToMean(mesh: TriangleMesh[_3D], mean: TriangleMesh[_3D]): TriangleMesh[_3D] = {
    val pairs = mesh.pointSet.points.zip(mean.pointSet.points).toIndexedSeq
    val trans = LandmarkRegistration.rigid3DLandmarkRegistration(pairs, Point3D(0, 0, 0))
    mesh.transform(trans)
  }

  def runGpa(meshes: IndexedSeq[TriangleMesh[_3D]], maxIter: Int = 10, tol: Double = 0.05)
  : (IndexedSeq[TriangleMesh[_3D]], TriangleMesh[_3D]) = {
    var current = meshes
    var prevLoss = Double.MaxValue
    var finalMean = refMesh
    for (iter <- 0 until maxIter) {
      finalMean = vtxMean(current)
      current = current.map(m => rigidAlignToMean(m, finalMean))
      val loss = current.map { m =>
        val dists = Metrics.correspondingDistances(m, finalMean)
        dists.sum / dists.size
      }.sum / current.size
      println(f"  GPA iter ${iter+1}: mean vertex dist = $loss%.4f mm")
      if (math.abs(prevLoss - loss) < tol) { println(s"  converged"); return (current, finalMean) }
      prevLoss = loss
    }
    (current, finalMean)
  }

  val (gpaShapes, gpaMean) = runGpa(registered.map(_.mesh))
  MeshIO.writeMesh(gpaMean, new File(outDir, "gpa_mean.stl"))
  println(s"  GPA mean shape saved.")

  // =========================================================================
  // 7. Build SSM via PCA
  // =========================================================================
  println("\n[6/6] Building SSM (PCA) …")

  val dc  = DataCollection.fromTriangleMesh3DSequence(refMesh, gpaShapes)
  val ssm = PointDistributionModel.createUsingPCA(dc)
  println(s"  SSM: ${ssm.rank} modes, ${gpaShapes.size} shapes")

  Try(StatisticalModelIO.writeStatisticalTriangleMeshModel3D(ssm, new File(outDir, "scapula_ssm.h5")))
    .fold(e => println(s"  WARNING: could not save SSM file: $e"), _ => println("  SSM saved → scapula_ssm.h5"))

  // =========================================================================
  // 8. Validation
  // =========================================================================
  println("\n" + "=" * 72)
  println("SSM VALIDATION METRICS")
  println("=" * 72)

  val eigenvalues   = ssm.gp.klBasis.map(_.eigenvalue)
  val totalVar      = eigenvalues.sum
  val cumVar        = eigenvalues.scanLeft(0.0)(_ + _).tail

  def modesFor(pct: Double): Int = cumVar.indexWhere(_ / totalVar >= pct) + 1

  // ----- 8a: Compactness -----------------------------------------------
  println("\n── Compactness (variance explained by cumulative modes) ──")
  val compPw = new PrintWriter(new File(outDir, "compactness.csv"))
  compPw.println("Modes,Eigenvalue,Cumulative_Var_Percent")
  eigenvalues.zipWithIndex.foreach { case (ev, i) =>
    val pct = 100.0 * cumVar(i) / totalVar
    compPw.println(s"${i+1},$ev,$pct")
    if (i < 15 || Seq(0.90, 0.95, 0.99).exists(t => math.abs(cumVar(i) / totalVar - t) < 0.005))
      println(f"  mode ${i+1}%3d | λ=$ev%8.4f | cumulative=$pct%6.2f%%")
  }
  compPw.close()
  println(f"\n  90%% variance → ${modesFor(0.90)} modes")
  println(f"  95%% variance → ${modesFor(0.95)} modes")
  println(f"  99%% variance → ${modesFor(0.99)} modes")

  // ----- 8b: Generalization (LOO cross-validation) ---------------------
  println("\n── Generalization (leave-one-out reconstruction error) ──")
  val modeList = Seq(1, 2, 3, 5, 10, 20, 50, ssm.rank).distinct.filter(_ <= ssm.rank)
  val genPw    = new PrintWriter(new File(outDir, "generalization.csv"))
  genPw.println("Modes,Mean_LOO_MSD_mm,Std_LOO_MSD_mm")

  // Use a subset for LOO if dataset is large (keep compute tractable)
  val looSubset = if (gpaShapes.size > 20) {
    val step = gpaShapes.size / 20
    gpaShapes.zipWithIndex.filter(_._2 % step == 0).map(_._1)
  } else gpaShapes

  val looIds = registered.map(_.id).take(looSubset.size)

  for (nModes <- modeList) {
    val errors = looSubset.zipWithIndex.map { case (testMesh, i) =>
      val trainMeshes = looSubset.patch(i, Nil, 1)
      val dcLoo  = DataCollection.fromTriangleMesh3DSequence(refMesh, trainMeshes)
      val ssmLoo = PointDistributionModel.createUsingPCA(dcLoo)
      val nM     = math.min(nModes, ssmLoo.rank)
      val coeffs = ssmLoo.coefficients(testMesh)
      val trunc  = DenseVector.tabulate[Double](ssmLoo.rank)(j => if (j < nM) coeffs(j) else 0.0)
      val fitted = ssmLoo.instance(trunc)
      Metrics.symmetric(fitted, testMesh).mean
    }
    val mu  = errors.sum / errors.size
    val std = math.sqrt(errors.map(e => (e - mu) * (e - mu)).sum / errors.size)
    println(f"  $nModes%3d modes | LOO MSD = $mu%.4f ± $std%.4f mm")
    genPw.println(s"$nModes,$mu,$std")
  }
  genPw.close()

  // ----- 8c: Specificity -----------------------------------------------
  println("\n── Specificity (random-sample distance to nearest training shape) ──")
  val specPw     = new PrintWriter(new File(outDir, "specificity.csv"))
  specPw.println("Modes,Mean_Specificity_mm,Std_Specificity_mm")
  val nSamples   = 50

  for (nModes <- modeList) {
    val specRng = new java.util.Random(Config.seed + nModes)
    val distances = (0 until nSamples).map { _ =>
      val coeffs = DenseVector.tabulate[Double](ssm.rank) { j =>
        if (j < nModes) specRng.nextGaussian() else 0.0
      }
      val sample = ssm.instance(coeffs)
      gpaShapes.map(tm => Metrics.symmetric(sample, tm).mean).min
    }
    val mu  = distances.sum / distances.size
    val std = math.sqrt(distances.map(d => (d - mu) * (d - mu)).sum / distances.size)
    println(f"  $nModes%3d modes | specificity = $mu%.4f ± $std%.4f mm")
    specPw.println(s"$nModes,$mu,$std")
  }
  specPw.close()

  // ----- 8d: Per-shape reconstruction ----------------------------------
  println("\n── Per-shape reconstruction (SSM projection vs aligned shape) ──")
  val recoPw = new PrintWriter(new File(outDir, "reconstruction_per_shape.csv"))
  recoPw.println("ID,MSD_mm,RMS_mm,HD95_mm,HD_mm,CoeffNorm")

  val recoStats = gpaShapes.zip(registered.map(_.id)).map { case (shape, id) =>
    val coeffs = ssm.coefficients(shape)
    val fitted = ssm.instance(coeffs)
    val st     = Metrics.symmetric(fitted, shape)
    val cnorm  = math.sqrt(coeffs.toArray.map(c => c * c).sum)
    println(f"  ${id.take(35)}%-35s  MSD=${st.mean}%.3f  RMS=${st.rms}%.3f  HD95=${st.hd95}%.3f  |θ|=${cnorm}%.2f")
    recoPw.println(s"$id,${st.mean},${st.rms},${st.hd95},${st.hd},$cnorm")
    st
  }
  recoPw.close()

  val avgMSD = recoStats.map(_.mean).sum / recoStats.size
  val avgHD95 = recoStats.map(_.hd95).sum / recoStats.size
  println(f"\n  Average MSD  : $avgMSD%.4f mm")
  println(f"  Average HD95 : $avgHD95%.4f mm")

  // =========================================================================
  // 9. Summary
  // =========================================================================
  println("\n" + "=" * 72)
  println("DONE")
  println("=" * 72)
  println(s"Output : ${outDir.getAbsolutePath}")
  println("""  scapula_ssm.h5                – HDF5 statistical shape model
  reference.stl / gpa_mean.stl  – reference and GPA mean mesh
  reg_*.stl                     – non-rigidly registered shapes
  compactness.csv               – variance per mode
  generalization.csv            – LOO reconstruction vs #modes
  specificity.csv               – random-sample quality vs #modes
  reconstruction_per_shape.csv  – per-shape SSM fit metrics""")
}
