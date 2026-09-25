package scapula

import scalismo.io.MeshIO
import scalismo.ui.api.ScalismoUI
import java.io.File
import scalismo.kernels.GaussianKernel3D
import scalismo.kernels.DiagonalKernel3D
import scalismo.statisticalmodel.GaussianProcess3D
import scalismo.geometry.EuclideanVector
import scalismo.geometry._3D
import scalismo.statisticalmodel.LowRankGaussianProcess
import scalismo.statisticalmodel.PointDistributionModel3D
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.mesh.TriangleMesh
import breeze.linalg.DenseVector
import scalismo.registration.GaussianProcessTransformationSpace
import scalismo.numerics.FixedPointsUniformMeshSampler3D
import scalismo.registration.MeanSquaresMetric
import scalismo.numerics.LBFGSOptimizer
import scalismo.registration.L2Regularizer
import scalismo.registration.Registration
import scalismo.utils.Random as ScalismoRandom

import scala.util.Random

/**
 * Direct port of the non-rigid GPMM fitting example Dennis Madsen posted to the scalismo mailing list on
 * 2024-06-27, kept as close to the original as possible.
 *
 * The email's code snippet was missing the body of `doRegistration`; it is restored here following the same
 * pattern as the official "Tutorial 12 - Parametric, non-rigid registration"
 * (https://scalismo.org/docs/tutorials/tutorial12), which is what the snippet itself was adapted from.
 *
 * As in the mailing list post, the target is a random sample drawn from the GPMM itself (not a real second
 * scan), so this only exercises the shape-fitting machinery: it does not include a pose/rigid pre-alignment
 * step (Dennis's suggestion there was to run CPD from GiNGR first for a real target).
 */
object NonRigidFittingExample {

  case class RegistrationParameters(regularizationWeight: Double, numberOfIterations: Int, numberOfSampledPoints: Int)

  private def env(key: String, default: String): String = sys.env.getOrElse(key, default)

  /** Defaults to where the STL from https://commons.wikimedia.org/wiki/File:Human_scapula_1.stl was found on disk. */
  val referenceFile: File =
    new File(env("SCAPULA_REFERENCE_STL", "/home/g25upadh/Documents/20210626175424!Human_scapula_1.stl"))

  def doRegistration(lowRankGP: LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
                     referenceMesh: TriangleMesh[_3D],
                     targetMesh: TriangleMesh[_3D],
                     initialCoefficients: DenseVector[Double],
                     registrationParameters: RegistrationParameters,
                     onIteration: DenseVector[Double] => Unit
  )(implicit rng: ScalismoRandom): DenseVector[Double] = {
    val transformationSpace = GaussianProcessTransformationSpace(lowRankGP)
    val fixedImage = referenceMesh.operations.toDistanceImage
    val movingImage = targetMesh.operations.toDistanceImage
    val sampler = FixedPointsUniformMeshSampler3D(referenceMesh, registrationParameters.numberOfSampledPoints)
    val metric = MeanSquaresMetric(fixedImage, movingImage, transformationSpace, sampler)
    val optimizer = LBFGSOptimizer(registrationParameters.numberOfIterations)
    val regularizer = L2Regularizer(transformationSpace)
    val registration = Registration(metric, regularizer, registrationParameters.regularizationWeight, optimizer)
    val registrationIterator = registration.iterator(initialCoefficients)
    val visualizingRegistrationIterator = for (it <- registrationIterator) yield {
      onIteration(it.parameters)
      it
    }
    val registrationResult = visualizingRegistrationIterator.toSeq.last
    registrationResult.parameters
  }

  def createRandomDenseVector(vecSize: Int, minValue: Double = 0.0, maxValue: Double = 0.5): DenseVector[Double] =
    DenseVector.fill(vecSize) {
      minValue + (maxValue - minValue) * Random.nextDouble()
    }

  def main(args: Array[String]): Unit = {
    // Required before touching any VTK-backed class (MeshIO's STL reader, ScalismoUI, ...): it extracts and
    // loads the native VTK libraries. Without it, VTK objects fail to initialize with an UnsatisfiedLinkError
    // on VTKInit because the native libraries were never loaded in the first place.
    scalismo.initialize()
    implicit val rng: ScalismoRandom = ScalismoRandom(Config.seed)

    println("Doing stuff")
    val raw = MeshIO.readMesh(referenceFile).get
    val reference = raw.operations.decimate(10000)

    val kernelCoarse = GaussianKernel3D(150 / 2, 15) // 150mm on longest side
    val kernelMid = GaussianKernel3D(150 / 5, 10)
    val kernelFine = GaussianKernel3D(150 / 10, 5)
    val kernelfinal = kernelCoarse + kernelMid + kernelFine
    val kernel = DiagonalKernel3D(kernelfinal, outputDim = 3)
    val gp = GaussianProcess3D[EuclideanVector[_3D]](kernel)
    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      reference,
      gp,
      relativeTolerance = 0.01,
      interpolator = NearestNeighborInterpolator3D()
    )
    val gpmm = PointDistributionModel3D(reference, lowRankGP)
    println(s"Model with ${gpmm.rank} basis functions and the reference with ${reference.pointSet.numberOfPoints} vertices")

    val ui = ScalismoUI()
    val modelGroup = ui.createGroup("model")
    val modelView = ui.show(modelGroup, gpmm, "model")

    val initialCoefficients = DenseVector.zeros[Double](gpmm.rank)

    val instancePars = createRandomDenseVector(gpmm.rank)
    val target = gpmm.instance(instancePars) // alternative: gpmm.sample()
    val meshGroup = ui.createGroup("target")
    ui.show(meshGroup, target, "target")
    ui.show(meshGroup, reference, "reference")

    val registrationParameters = Seq(
      RegistrationParameters(regularizationWeight = 1e-1, numberOfIterations = 50, numberOfSampledPoints = 100),
      RegistrationParameters(regularizationWeight = 1e-2, numberOfIterations = 50, numberOfSampledPoints = 500),
      RegistrationParameters(regularizationWeight = 1e-4, numberOfIterations = 50, numberOfSampledPoints = 100),
      RegistrationParameters(regularizationWeight = 1e-6, numberOfIterations = 50, numberOfSampledPoints = 5000)
    )

    val finalCoefficients = registrationParameters.foldLeft(initialCoefficients) { (coefficients, params) =>
      println(params)
      doRegistration(
        lowRankGP,
        reference,
        target,
        coefficients,
        params,
        onIteration = pars => modelView.shapeModelTransformationView.shapeTransformationView.coefficients = pars
      )
    }

    val finalFit = gpmm.instance(finalCoefficients)
    ui.show(meshGroup, finalFit, "fit")
  }
}
