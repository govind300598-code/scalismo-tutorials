package scapula

import breeze.linalg.{DenseMatrix, DenseVector}
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.common.{EuclideanVector, Field}
import scalismo.geometry.{EuclideanSpace3D, Landmark, _3D}
import scalismo.io.{MeshIO, StatismoIO}
import scalismo.kernels.{DiagonalKernel3D, GaussianKernel}
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, MultivariateNormalDistribution, StatisticalMeshModel}
import scalismo.utils.Random

import java.io.File

/**
 * Two-pass SSM build.
 *   Pass 1 – register all specimens to an arbitrary first specimen.
 *   Pass 2 – re-register to the mean of SSM1, removing reference bias.
 *
 * Outputs:
 *   $SCAPULA_OUT_DIR/pass1/  *.vtk  + ssm.h5
 *   $SCAPULA_OUT_DIR/pass2/  *.vtk  + ssm.h5
 */
object RebuildSSM {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val outDir  = if (args.nonEmpty)    new File(args(0)) else Config.outDir
    val dataDir = if (args.length > 1) new File(args(1)) else Config.dataDir

    require(dataDir.exists() && dataDir.isDirectory,
      s"Data directory not found: ${dataDir.getPath}\n  Set SCAPULA_DATA_DIR or pass it as the second argument.")

    val pass1Dir = new File(outDir, "pass1")
    val pass2Dir = new File(outDir, "pass2")
    pass1Dir.mkdirs()
    pass2Dir.mkdirs()

    val csv = ScapulaData.csvFile(dataDir)
    val (landmarks, fromHeader, _) = ScapulaData.readLandmarkCsv(csv)
    if (!fromHeader)
      println("WARNING: landmark columns resolved by fallback — verify the CSV header")

    val specimens = ScapulaData.specimens(dataDir).filter(s => landmarks.contains(s.modelId))
    require(specimens.nonEmpty, s"No STL specimens with landmark data found in ${dataDir.getPath}")
    println(s"${specimens.size} specimens with landmarks found")

    // Reference: first specimen normalised to left side, decimated to model resolution
    val refSpec  = specimens.head
    val refRaw   = ScapulaData.loadMesh(refSpec.file)
    val refMesh0 = if (refSpec.isRight) ScapulaData.mirrorMesh(refRaw) else refRaw
    val refLms0  = if (refSpec.isRight) ScapulaData.mirrorLandmarks(landmarks(refSpec.modelId))
                   else landmarks(refSpec.modelId)
    val ref1 = refMesh0.operations.decimate(Config.modelResolution)
    val ref1Lms = refLms0.map(lm => lm.copy(point = ref1.pointSet.findClosestPoint(lm.point).point))

    // ── Pass 1 ──────────────────────────────────────────────────────────────
    println("\n=== PASS 1: register all specimens to initial reference ===")
    val reg1 = registerAll(specimens, landmarks, ref1, ref1Lms, pass1Dir, "P1")
    println(s"Building SSM1 from ${reg1.size} meshes …")
    val ssm1 = buildSSM(ref1, reg1)
    StatismoIO.writeStatismoMeshModel(ssm1, new File(pass1Dir, "ssm.h5")).get
    println(s"SSM1: ${ssm1.rank} components → ${pass1Dir.getPath}/ssm.h5")

    // ── Pass 2 ──────────────────────────────────────────────────────────────
    println("\n=== PASS 2: register all specimens to mean of SSM1 ===")
    val ref2 = ssm1.mean
    // ref2 has same topology as ref1, so we can reuse point IDs for landmarks
    val ref2Lms = ref1Lms.map { lm =>
      val id = ref1.pointSet.findClosestPoint(lm.point).id
      lm.copy(point = ref2.pointSet.point(id))
    }
    val reg2 = registerAll(specimens, landmarks, ref2, ref2Lms, pass2Dir, "P2")
    println(s"Building SSM2 from ${reg2.size} meshes …")
    val ssm2 = buildSSM(ref2, reg2)
    StatismoIO.writeStatismoMeshModel(ssm2, new File(pass2Dir, "ssm.h5")).get
    println(s"SSM2: ${ssm2.rank} components → ${pass2Dir.getPath}/ssm.h5")

    println("\nDone.  Next: sbt \"runMain scapula.SSMValidation\"")
  }

  // ── helpers ──────────────────────────────────────────────────────────────

  def loadSSM(dir: File): StatisticalMeshModel =
    StatismoIO.readStatismoMeshModel(new File(dir, "ssm.h5")).get

  private def registerAll(
    specimens: IndexedSeq[ScapulaData.Specimen],
    landmarks: Map[String, IndexedSeq[Landmark[_3D]]],
    reference: TriangleMesh[_3D],
    refLms:    IndexedSeq[Landmark[_3D]],
    outDir:    File,
    tag:       String
  )(implicit rng: Random): IndexedSeq[TriangleMesh[_3D]] = {
    val priorModel = buildPriorModel(reference)
    specimens.zipWithIndex.map { case (spec, i) =>
      val cached = new File(outDir, s"${spec.modelId}.vtk")
      if (cached.exists()) {
        println(s"  [$tag ${i + 1}/${specimens.size}] ${spec.modelId} (cached)")
        MeshIO.readMesh(cached).get
      } else {
        println(s"  [$tag ${i + 1}/${specimens.size}] ${spec.modelId}")
        val rawMesh = ScapulaData.loadMesh(spec.file)
        val mesh    = if (spec.isRight) ScapulaData.mirrorMesh(rawMesh)                    else rawMesh
        val lms     = if (spec.isRight) ScapulaData.mirrorLandmarks(landmarks(spec.modelId)) else landmarks(spec.modelId)
        val (rigid, _) = RigidAlign.landmarkThenIcp(mesh, lms, reference, refLms)
        val registered = gpIcp(priorModel, rigid, Config.icpIterations)
        MeshIO.writeMesh(registered, cached).get
        registered
      }
    }
  }

  private def buildPriorModel(reference: TriangleMesh[_3D])(implicit rng: Random): StatisticalMeshModel = {
    val zeroMean = Field[_3D, EuclideanVector[_3D]](EuclideanSpace3D, _ => EuclideanVector.zeros[_3D])
    val kernel   = DiagonalKernel3D(GaussianKernel[_3D](50.0) * 25.0, 3)
    val gp       = GaussianProcess(zeroMean, kernel)
    val lr = LowRankGaussianProcess.approximateGPCholesky(
      reference.pointSet, gp,
      relativeTolerance = Config.gpRelativeTolerance,
      interpolator      = NearestNeighborInterpolator3D()
    )
    StatisticalMeshModel(reference, lr)
  }

  private def gpIcp(
    initModel: StatisticalMeshModel,
    target:    TriangleMesh[_3D],
    iters:     Int
  )(implicit rng: Random): TriangleMesh[_3D] = {
    val nSamples = 2000
    val noise = MultivariateNormalDistribution(
      DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3) * 1.0
    )
    var model = initModel
    for (_ <- 0 until iters) {
      val ids = RigidAlign.uniformIds(model.reference, nSamples)
      val obs = ids.map { id =>
        val currentPt = model.mean.pointSet.point(id)
        val targetPt  = target.operations.closestPointOnSurface(currentPt).point
        (id, targetPt, noise)
      }
      model = model.posterior(obs)
    }
    model.mean
  }

  def buildSSM(reference: TriangleMesh[_3D], meshes: IndexedSeq[TriangleMesh[_3D]]): StatisticalMeshModel = {
    val dc = DataCollection.fromTriangleMesh3DSequence(reference, meshes)
    StatisticalMeshModel.createUsingPCA(dc).get
  }
}
