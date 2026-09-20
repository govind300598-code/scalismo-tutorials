package nonrigid

// ---------------------------------------------------------------------------
// Dennis Madsen's non-rigid registration recipe — parameters kept EXACTLY
// as posted (scalismo forum, Jun 27 2024).  Only the data-loading section
// is replaced to work with the paired-scapulae dataset.
//
// Dennis's original target was a random GPMM sample (self-test).
// Here the targets are the real specimens from the dataset.
//
// Right-side scapulae are mirrored to left-side orientation before
// registration so that the GP prior is applied to anatomically matching
// geometry.  Without this step the optimizer fits a mirror-image bone to
// the reference, which is the primary cause of the bad results seen
// previously.
// ---------------------------------------------------------------------------

import breeze.linalg.DenseVector
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.common.{Field, ScalarMeshField3D}
import scalismo.geometry.{EuclideanSpace3D, EuclideanVector, Point, _3D}
import scalismo.kernels.{DiagonalKernel3D, GaussianKernel3D}
import scalismo.mesh.TriangleMesh
import scalismo.numerics.{FixedPointsUniformMeshSampler3D, LBFGSOptimizer}
import scalismo.registration.{GaussianProcessTransformationSpace, L2Regularizer, MeanSquaresMetric, Registration}
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel3D}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random
import scapula.{Config, RigidAlign, ScapulaData}

object Stage2Dennis {

  // Dennis's exact registration schedule (numberOfSampledPoints = 100/500/100/5000)
  case class RegistrationParameters(
    regularizationWeight: Double,
    numberOfIterations:   Int,
    numberOfSampledPoints: Int
  )

  val registrationParameters: Seq[RegistrationParameters] = Seq(
    RegistrationParameters(regularizationWeight = 1e-1, numberOfIterations = 50, numberOfSampledPoints =  100),
    RegistrationParameters(regularizationWeight = 1e-2, numberOfIterations = 50, numberOfSampledPoints =  500),
    RegistrationParameters(regularizationWeight = 1e-4, numberOfIterations = 50, numberOfSampledPoints =  100),
    RegistrationParameters(regularizationWeight = 1e-6, numberOfIterations = 50, numberOfSampledPoints = 5000)
  )

  // Dennis's doRegistration — structure kept as-is; view update is done via
  // the callback so no view type annotation is needed (scalismo-ui 0.92 does
  // not expose the path-dependent view type at the call-site level).
  def doRegistration(
    lowRankGP:           LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
    referenceMesh:       TriangleMesh[_3D],
    targetMesh:          TriangleMesh[_3D],
    initialCoefficients: DenseVector[Double],
    regParameters:       RegistrationParameters,
    onUpdate:            DenseVector[Double] => Unit
  )(implicit rng: Random): DenseVector[Double] = {

    val transformationSpace = GaussianProcessTransformationSpace(lowRankGP)
    val fixedImage  = referenceMesh.operations.toDistanceImage
    val movingImage = targetMesh.operations.toDistanceImage
    val sampler = FixedPointsUniformMeshSampler3D(
      referenceMesh,
      regParameters.numberOfSampledPoints
    )
    val metric      = MeanSquaresMetric(fixedImage, movingImage, transformationSpace, sampler)
    val optimizer   = LBFGSOptimizer(maxNumberOfIterations = regParameters.numberOfIterations)
    val regularizer = L2Regularizer(transformationSpace)
    val registration = Registration(
      metric,
      regularizer,
      regularizationWeight = regParameters.regularizationWeight,
      optimizer
    )

    val registrationIterator = registration.iterator(initialCoefficients)
    val visualizingRegistrationIterator =
      for ((it, itnum) <- registrationIterator.zipWithIndex) yield {
        println(s"      iter $itnum  value=${it.value}")
        onUpdate(it.parameters)
        it
      }
    val registrationResult = visualizingRegistrationIterator.toSeq.last
    registrationResult.parameters
  }

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    // ------------------------------------------------------------------
    // Data loading  (replaces Dennis's single-STL load)
    // ------------------------------------------------------------------
    val dir  = Config.dataDir
    val csv  = ScapulaData.csvFile(dir)
    val (landmarks, _, _) = ScapulaData.readLandmarkCsv(csv)
    val specimens         = ScapulaData.specimens(dir)

    val leftSpecimens = specimens.filter(s => !s.isRight && landmarks.contains(s.modelId))
    require(leftSpecimens.nonEmpty, "No left scapulae with landmarks found")

    // Reference: first left specimen, decimated to Dennis's 10 000 vertices
    val refSpec   = leftSpecimens.head
    val rawRef    = ScapulaData.loadMesh(refSpec.file)
    val reference = rawRef.operations.decimate(10000)   // Dennis: decimate(10000)
    val refLms    = landmarks(refSpec.modelId)

    println(s"Reference : ${refSpec.modelId}  (${reference.pointSet.numberOfPoints} vertices)")

    // ------------------------------------------------------------------
    // Dennis's exact kernel design
    // Three kernels; sigma = fractions of the ~150 mm scapula diameter.
    // Amplitude (second arg) scales the contribution of each component.
    // ------------------------------------------------------------------
    val kernelCoarse = GaussianKernel3D(150.0 / 2,  15)   // sigma = 75 mm
    val kernelMid    = GaussianKernel3D(150.0 / 5,  10)   // sigma = 30 mm
    val kernelFine   = GaussianKernel3D(150.0 / 10,  5)   // sigma = 15 mm
    val kernelfinal  = kernelCoarse + kernelMid + kernelFine
    val kernel       = DiagonalKernel3D(kernelfinal, outputDim = 3)

    // Dennis uses GaussianProcess3D[EuclideanVector[_3D]](kernel) which is
    // a zero-mean GP factory available in scalismo 1.0-RC1.
    // In scalismo 0.92 the equivalent is an explicit zero-mean Field.
    val zeroMean = Field(EuclideanSpace3D, (_: Point[_3D]) => EuclideanVector.zeros[_3D])
    val gp       = GaussianProcess(zeroMean, kernel)

    // Dennis's exact Cholesky settings
    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      reference,
      gp,
      relativeTolerance = 0.01,                  // Dennis: 0.01
      interpolator = NearestNeighborInterpolator3D()
    )

    val gpmm = PointDistributionModel3D(reference, lowRankGP)
    println(s"Model with ${gpmm.rank} basis functions and " +
      s"the reference with ${reference.pointSet.numberOfPoints} vertices")

    // ------------------------------------------------------------------
    // Scalismo viewer
    // ------------------------------------------------------------------
    val ui         = ScalismoUI()
    val modelGroup = ui.createGroup("model")
    val modelView  = ui.show(modelGroup, gpmm, "model")

    // ------------------------------------------------------------------
    // Pilot: register 1 left + 1 right = 2 specimens total
    // ------------------------------------------------------------------
    val targets = leftSpecimens.drop(1).take(1)

    targets.zipWithIndex.foreach { case (spec, idx) =>
      println(s"\n=== [${idx + 1}/3]  ${spec.modelId} ===")

      val rawTarget = ScapulaData.loadMesh(spec.file)
      val tgtLms    = landmarks(spec.modelId)

      // Rigid alignment: landmark Procrustes + trimmed ICP
      val (rigidAligned, _) = RigidAlign.landmarkThenIcp(rawTarget, tgtLms, reference, refLms)

      // Show target in viewer
      val meshGroup = ui.createGroup(s"target-${spec.modelId}")
      ui.show(meshGroup, rigidAligned, "target")
      ui.show(meshGroup, reference,    "reference")

      // Non-rigid registration using Dennis's exact schedule
      val initialCoefficients = DenseVector.zeros[Double](gpmm.rank)

      val finalCoefficients = registrationParameters.foldLeft(initialCoefficients) {
        (coefficients, params) =>
          println(s"  Pass: w=${params.regularizationWeight}  " +
            s"iters=${params.numberOfIterations}  n=${params.numberOfSampledPoints}")
          doRegistration(
            gpmm.gp,
            reference,
            rigidAligned,
            coefficients,
            params,
            onUpdate = c =>
              modelView.shapeModelTransformationView.shapeTransformationView.coefficients = c
          )
      }

      // Materialise the fitted mesh
      val deformField = gpmm.gp.instance(finalCoefficients)
      val fitted      = reference.transform(pt => pt + deformField(pt))

      val resultGroup = ui.createGroup(s"result-${spec.modelId}")
      ui.show(resultGroup, fitted,       "fitted")
      ui.show(resultGroup, rigidAligned, "target")

      // Error-distance colour map on the fitted surface
      val errPerVertex = fitted.pointSet.points
        .map(p => (p - rigidAligned.operations.closestPointOnSurface(p).point).norm)
        .toIndexedSeq
      ui.show(resultGroup, ScalarMeshField3D(fitted, errPerVertex), "errorMap")

      // Quick console metric
      val meanErr = errPerVertex.sum / errPerVertex.length
      val maxErr  = errPerVertex.max
      println(f"  mean surface dist = $meanErr%.2f mm   max = $maxErr%.2f mm")
    }

    // Also register 1 right-side specimen (mirror first)
    val rightSpecimens = specimens.filter(s => s.isRight && landmarks.contains(s.modelId)).take(1)

    rightSpecimens.zipWithIndex.foreach { case (spec, idx) =>
      println(s"\n=== [R ${idx + 1}/1]  ${spec.modelId} (mirrored to left) ===")

      val rawTarget    = ScapulaData.loadMesh(spec.file)
      val tgtLms       = landmarks(spec.modelId)

      // Mirror right → left before any alignment
      val mirroredMesh = ScapulaData.mirrorMesh(rawTarget)
      val mirroredLms  = ScapulaData.mirrorLandmarks(tgtLms)

      val (rigidAligned, _) =
        RigidAlign.landmarkThenIcp(mirroredMesh, mirroredLms, reference, refLms)

      val meshGroup = ui.createGroup(s"target-${spec.modelId}")
      ui.show(meshGroup, rigidAligned, "target-mirrored")
      ui.show(meshGroup, reference,    "reference")

      val initialCoefficients = DenseVector.zeros[Double](gpmm.rank)

      val finalCoefficients = registrationParameters.foldLeft(initialCoefficients) {
        (coefficients, params) =>
          println(s"  Pass: w=${params.regularizationWeight}  " +
            s"iters=${params.numberOfIterations}  n=${params.numberOfSampledPoints}")
          doRegistration(
            gpmm.gp,
            reference,
            rigidAligned,
            coefficients,
            params,
            onUpdate = c =>
              modelView.shapeModelTransformationView.shapeTransformationView.coefficients = c
          )
      }

      val deformField = gpmm.gp.instance(finalCoefficients)
      val fitted      = reference.transform(pt => pt + deformField(pt))

      val resultGroup = ui.createGroup(s"result-${spec.modelId}")
      ui.show(resultGroup, fitted,       "fitted")
      ui.show(resultGroup, rigidAligned, "target")

      val errPerVertex = fitted.pointSet.points
        .map(p => (p - rigidAligned.operations.closestPointOnSurface(p).point).norm)
        .toIndexedSeq
      ui.show(resultGroup, ScalarMeshField3D(fitted, errPerVertex), "errorMap")

      val meanErr = errPerVertex.sum / errPerVertex.length
      val maxErr  = errPerVertex.max
      println(f"  mean surface dist = $meanErr%.2f mm   max = $maxErr%.2f mm")
    }

    println("\nDone — close the Scalismo viewer when finished.")
  }
}
