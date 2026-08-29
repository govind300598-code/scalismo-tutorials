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

object ScapulaFullSSMPipeline extends App {

  scalismo.initialize()
  implicit val rng: Random = Random(42L)

  // ── CONFIGURE THESE TWO PATHS ────────────────────────────────
  val dataDir = new File("/home/g25upadh/Documents/100 plus scapula data/paired_scapulae_STLs_scapula")
  val outDir  = new File(dataDir.getParentFile, "scapula_full_ssm_output")
  // ─────────────────────────────────────────────────────────────

  val MODEL_RES = 5000
  val ICP_ITERS = 40
  val GP_TOL    = 0.01

  outDir.mkdirs()
  println("=" * 70)
  println("SCAPULA FULL SSM  (Rigid -> GPA -> GP-ICP -> PCA -> Validation)")
  println("=" * 70)
  println(s"Data   : ${dataDir.getAbsolutePath}")
  println(s"Output : ${outDir.getAbsolutePath}")

  // =========================================================================
  // Step 1  Load data
  // =========================================================================
  val csv = ScapulaData.csvFile(dataDir)
  val (allLandmarks, _, _) = ScapulaData.readLandmarkCsv(csv)
  val allSpecimens = ScapulaData.specimens(dataDir)
  val specimens    = allSpecimens.filter(s => allLandmarks.contains(s.modelId))
  println(s"\nSpecimens with landmarks: ${specimens.size}")

  // =========================================================================
  // Step 2  Reference mesh  (first left specimen, alphabetically)
  // =========================================================================
  val refSpec = specimens.find(!_.isRight).getOrElse(specimens.head)
  val refRaw  = ScapulaData.loadMesh(refSpec.file)
  val refMesh = if (refRaw.pointSet.numberOfPoints > MODEL_RES)
                  refRaw.operations.decimate(MODEL_RES) else refRaw
  val refLms  = allLandmarks(refSpec.modelId)
  println(s"Reference : ${refSpec.modelId}  (${refMesh.pointSet.numberOfPoints} vertices)")
  MeshIO.writeMesh(refMesh, new File(outDir, "reference.stl"))

  // =========================================================================
  // Step 3  Rigid alignment – landmark Procrustes + trimmed ICP
  // =========================================================================
  println("\n[1/5] Rigid alignment ...")

  case class Aligned(id: String, mesh: TriangleMesh[_3D])

  val rigidAligned: IndexedSeq[Aligned] = specimens
    .filterNot(_.modelId == refSpec.modelId)
    .zipWithIndex
    .flatMap { case (spec, i) =>
      Try {
        val raw = ScapulaData.loadMesh(spec.file)
        val dec = if (raw.pointSet.numberOfPoints > MODEL_RES * 3)
                    raw.operations.decimate(MODEL_RES * 2) else raw
        val (mesh, lms) =
          if (spec.isRight)
            (ScapulaData.mirrorMesh(dec), ScapulaData.mirrorLandmarks(allLandmarks(spec.modelId)))
          else
            (dec, allLandmarks(spec.modelId))
        val (aligned, _) = RigidAlign.landmarkThenIcp(mesh, lms, refMesh, refLms, icpIterations = 25)
        print(s"  rigid [${i+1}/${specimens.size - 1}] ${spec.id}\r")
        Aligned(spec.modelId, aligned)
      }.toOption
    }
  println(s"\n  -> ${rigidAligned.size} shapes rigidly aligned")

  // =========================================================================
  // Step 4  Multi-scale GP deformation prior
  // =========================================================================
  println("\n[2/5] Building GP prior ...")

  val kernel =
    GaussianKernel[_3D](sigma = 80.0, scaleFactor = 50.0) +
    GaussianKernel[_3D](sigma = 40.0, scaleFactor = 20.0) +
    GaussianKernel[_3D](sigma = 15.0, scaleFactor =  5.0)

  val gpPrior = GaussianProcess[_3D, EuclideanVector[_3D]](
    Field(EuclideanSpace[_3D], (_: Point[_3D]) => EuclideanVector.zeros[_3D]),
    DiagonalKernel3D(kernel, 3)
  )
  val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
    refMesh, gpPrior,
    relativeTolerance = GP_TOL,
    interpolator      = TriangleMeshInterpolator3D[EuclideanVector[_3D]]()
  )
  val discreteGP = lowRankGP.discretize(refMesh)
  println(s"  GP rank: ${lowRankGP.rank}")

  // =========================================================================
  // Step 5  Non-rigid GP-ICP  (establishes dense vertex correspondences)
  //         After this step every registered mesh has the SAME topology as
  //         refMesh and corresponding vertices are anatomically matched.
  // =========================================================================
  println("\n[3/5] Non-rigid GP-ICP ...")

  def gpIcp(target: TriangleMesh[_3D]): TriangleMesh[_3D] = {
    var current = refMesh
    val tgtOps  = target.operations
    for (iter <- 0 until ICP_ITERS) {
      val alpha  = iter.toDouble / math.max(1, ICP_ITERS - 1)
      val sigma2 = 4.0 * (1 - alpha) + 0.25 * alpha     // annealed noise
      val sampleIds = UniformMeshSampler3D(current, 1200)
        .sample().map { case (pt, _) => refMesh.pointSet.findClosestPoint(pt).id }.distinct
      val obs: IndexedSeq[(PointId, EuclideanVector[_3D])] = sampleIds.map { ptId =>
        val curPt = current.pointSet.point(ptId)
        val tgtPt = tgtOps.closestPointOnSurface(curPt).point
        ptId -> (tgtPt - refMesh.pointSet.point(ptId))
      }
      val post   = discreteGP.posterior(obs, sigma2)
      val newPts = refMesh.pointSet.pointIds.toIndexedSeq.map { ptId =>
        refMesh.pointSet.point(ptId) + post.mean(ptId)
      }
      current = TriangleMesh3D(UnstructuredPoints3D(newPts), refMesh.triangulation)
    }
    current
  }

  val registered: IndexedSeq[Aligned] = rigidAligned.zipWithIndex.map { case (spec, i) =>
    println(s"  GP-ICP [${i+1}/${rigidAligned.size}] ${spec.id}")
    val reg = gpIcp(spec.mesh)
    MeshIO.writeMesh(reg, new File(outDir, s"reg_${spec.id}.stl"))
    Aligned(spec.id, reg)
  }

  // =========================================================================
  // Step 6  Generalized Procrustes Analysis on the registered shapes
  //         (all meshes now share refMesh topology -> vertex-wise mean is valid)
  // =========================================================================
  println("\n[4/5] GPA on registered shapes ...")

  val origin = Point3D(0.0, 0.0, 0.0)

  def vtxMean(meshes: IndexedSeq[TriangleMesh[_3D]]): TriangleMesh[_3D] = {
    val n    = refMesh.pointSet.numberOfPoints
    val invN = 1.0 / meshes.size
    val meanPts = (0 until n).map { i =>
      val ptId = PointId(i)
      val vsum = meshes.foldLeft(EuclideanVector.zeros[_3D]) { (acc, m) =>
        acc + (m.pointSet.point(ptId) - origin)
      }
      origin + vsum * invN
    }
    TriangleMesh3D(UnstructuredPoints3D(meanPts), refMesh.triangulation)
  }

  def rigidToMean(mesh: TriangleMesh[_3D], mean: TriangleMesh[_3D]): TriangleMesh[_3D] = {
    val pairs = mesh.pointSet.points.zip(mean.pointSet.points).toIndexedSeq
    val trans = LandmarkRegistration.rigid3DLandmarkRegistration(pairs, origin)
    mesh.transform(trans)
  }

  def runGpa(meshes: IndexedSeq[TriangleMesh[_3D]], maxIter: Int = 10, tol: Double = 0.05)
  : (IndexedSeq[TriangleMesh[_3D]], TriangleMesh[_3D]) = {
    var cur      = meshes
    var prevLoss = Double.MaxValue
    var finalMean = refMesh
    for (iter <- 0 until maxIter) {
      finalMean = vtxMean(cur)
      cur = cur.map(m => rigidToMean(m, finalMean))
      val loss: Double = cur.map { m =>
        val d = Metrics.correspondingDistances(m, finalMean)
        d.sum / d.size
      }.sum / cur.size
      println(f"  GPA iter ${iter+1}: mean vertex dist = $loss%.4f mm")
      if (math.abs(prevLoss - loss) < tol) { println("  GPA converged"); return (cur, finalMean) }
      prevLoss = loss
    }
    (cur, finalMean)
  }

  val (gpaShapes, gpaMean) = runGpa(registered.map(_.mesh))
  MeshIO.writeMesh(gpaMean, new File(outDir, "gpa_mean.stl"))
  println("  GPA mean saved.")

  // =========================================================================
  // Step 7  PCA -> Statistical Shape Model
  // =========================================================================
  println("\n[5/5] Building SSM via PCA ...")
  val dc  = DataCollection.fromTriangleMesh3DSequence(refMesh, gpaShapes)
  val ssm = PointDistributionModel.createUsingPCA(dc)
  println(s"  SSM: ${ssm.rank} modes from ${gpaShapes.size} shapes")
  Try(StatisticalModelIO.writeStatisticalTriangleMeshModel3D(ssm, new File(outDir, "scapula_ssm.h5")))
    .fold(e => println(s"  WARNING SSM save failed: $e"), _ => println("  Saved -> scapula_ssm.h5"))

  // =========================================================================
  // Validation
  // =========================================================================
  println("\n" + "=" * 70)
  println("SSM VALIDATION")
  println("=" * 70)

  val eigenvalues = ssm.gp.klBasis.map(_.eigenvalue)
  val totalVar    = eigenvalues.sum
  val cumVar      = eigenvalues.scanLeft(0.0)(_ + _).tail

  def modesFor(pct: Double): Int = cumVar.indexWhere(_ / totalVar >= pct) + 1

  // ── A. Compactness ─────────────────────────────────────────────────────
  println("\n--- Compactness ---")
  val compPw = new PrintWriter(new File(outDir, "compactness.csv"))
  compPw.println("Modes,Eigenvalue,CumulativeVariancePct")
  eigenvalues.zipWithIndex.foreach { case (ev, i) =>
    val pct = 100.0 * cumVar(i) / totalVar
    compPw.println(s"${i+1},$ev,$pct")
    if (i < 10 || Seq(90,95,99).exists(t => math.abs(pct - t) < 0.6))
      println(f"  mode ${i+1}%3d  lambda=${ev}%9.3f  cumulative=${pct}%6.2f%%")
  }
  compPw.close()
  println(f"  90%% variance: ${modesFor(0.90)} modes")
  println(f"  95%% variance: ${modesFor(0.95)} modes")
  println(f"  99%% variance: ${modesFor(0.99)} modes")

  // ── B. Generalization  (LOO on up to 15 shapes) ────────────────────────
  println("\n--- Generalization (LOO reconstruction error) ---")
  val modeList = Seq(1,2,3,5,10,20,50,ssm.rank).distinct.filter(_ <= ssm.rank)
  val genPw    = new PrintWriter(new File(outDir, "generalization.csv"))
  genPw.println("Modes,MeanLOO_mm,StdLOO_mm")

  val looMax = 15
  val step   = math.max(1, gpaShapes.size / looMax)
  val looShapes = gpaShapes.zipWithIndex.filter(_._2 % step == 0).map(_._1)
  println(s"  Using ${looShapes.size} shapes for LOO")

  for (nModes <- modeList) {
    val errors: IndexedSeq[Double] = looShapes.zipWithIndex.map { case (testMesh, i) =>
      val train  = looShapes.patch(i, Nil, 1)
      val dcLoo  = DataCollection.fromTriangleMesh3DSequence(refMesh, train)
      val ssmLoo = PointDistributionModel.createUsingPCA(dcLoo)
      val nM     = math.min(nModes, ssmLoo.rank)
      val coeffs = ssmLoo.coefficients(testMesh)
      val trunc  = DenseVector.tabulate[Double](ssmLoo.rank)(j => if (j < nM) coeffs(j) else 0.0)
      Metrics.symmetric(ssmLoo.instance(trunc), testMesh).mean
    }
    val mu  = errors.sum / errors.size
    val std = math.sqrt(errors.map(e => (e - mu) * (e - mu)).sum / errors.size)
    println(f"  $nModes%3d modes  LOO MSD = $mu%.4f +/- $std%.4f mm")
    genPw.println(s"$nModes,$mu,$std")
  }
  genPw.close()

  // ── C. Specificity ─────────────────────────────────────────────────────
  println("\n--- Specificity (random samples vs training shapes) ---")
  val specPw  = new PrintWriter(new File(outDir, "specificity.csv"))
  specPw.println("Modes,MeanSpec_mm,StdSpec_mm")
  val nSpec   = 30

  for (nModes <- modeList) {
    val specRng = new java.util.Random(42L + nModes)
    val dists: IndexedSeq[Double] = (0 until nSpec).toIndexedSeq.map { _ =>
      val coeffs = DenseVector.tabulate[Double](ssm.rank)(j =>
        if (j < nModes) specRng.nextGaussian() else 0.0)
      val sample = ssm.instance(coeffs)
      gpaShapes.map(tm => Metrics.symmetric(sample, tm).mean).min
    }
    val mu  = dists.sum / dists.size
    val std = math.sqrt(dists.map(d => (d - mu) * (d - mu)).sum / dists.size)
    println(f"  $nModes%3d modes  specificity = $mu%.4f +/- $std%.4f mm")
    specPw.println(s"$nModes,$mu,$std")
  }
  specPw.close()

  // ── D. Per-shape reconstruction ────────────────────────────────────────
  println("\n--- Per-shape reconstruction (SSM fitted to each GPA shape) ---")
  val recoPw = new PrintWriter(new File(outDir, "reconstruction_per_shape.csv"))
  recoPw.println("ID,MSD_mm,RMS_mm,HD95_mm,HD_mm,CoeffNorm")

  val recoStats: IndexedSeq[Metrics.SurfaceStats] = gpaShapes.zip(registered.map(_.id)).map { case (shape, id) =>
    val coeffs = ssm.coefficients(shape)
    val fitted = ssm.instance(coeffs)
    val st     = Metrics.symmetric(fitted, shape)
    val cnorm  = math.sqrt(coeffs.toArray.map(c => c * c).sum)
    println(f"  ${id.take(38)}%-38s  MSD=${st.mean}%.3f  RMS=${st.rms}%.3f  HD95=${st.hd95}%.3f  |coeff|=${cnorm}%.2f")
    recoPw.println(s"$id,${st.mean},${st.rms},${st.hd95},${st.hd},$cnorm")
    st
  }
  recoPw.close()

  val avgMSD  = recoStats.map(_.mean).sum / recoStats.size
  val avgHD95 = recoStats.map(_.hd95).sum / recoStats.size
  println(f"\n  Average MSD  : $avgMSD%.4f mm")
  println(f"  Average HD95 : $avgHD95%.4f mm")

  // =========================================================================
  // Summary
  // =========================================================================
  println("\n" + "=" * 70)
  println("COMPLETE")
  println("=" * 70)
  println(s"Output directory : ${outDir.getAbsolutePath}")
  println("""Files written:
  scapula_ssm.h5               - HDF5 statistical shape model
  reference.stl / gpa_mean.stl - reference and GPA consensus mean
  reg_*.stl                    - non-rigidly registered shapes (all in correspondence)
  compactness.csv              - eigenvalue spectrum
  generalization.csv           - LOO reconstruction error vs modes
  specificity.csv              - random-sample quality vs modes
  reconstruction_per_shape.csv - per-shape MSD / RMS / HD95 / HD""")
}
