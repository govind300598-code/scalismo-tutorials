package scapula

import scalismo.geometry.*
import scalismo.common.*
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.mesh.*
import scalismo.registration.*
import scalismo.io.MeshIO
import scalismo.numerics.*
import scalismo.kernels.*
import scalismo.statisticalmodel.*
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random
import breeze.linalg.{DenseMatrix, DenseVector}

import java.io.File

/**
 * STAGE 2 — NON-RIGID REGISTRATION (shared core).
 *
 * Pipeline per specimen:
 *   1. Mirror right scapulae → left orientation.
 *   2. Landmark-based rigid Procrustes.
 *   3. Automatic trimmed rigid ICP.
 *   4. Zero-mean GP (single Gaussian kernel) conditioned on landmark
 *      correspondences (posterior GP) to anchor the deformation anatomically.
 *   5. Multi-resolution LBFGS optimisation (decreasing regularisation weight).
 *   6. Nearest-neighbour projection onto the rigidly-aligned target surface.
 *   7. Save in reference-mesh topology for Stage 3 SSM building.
 *
 * Reference selection: left-side specimen whose 5 landmarks lie closest to
 * the cross-subject mean — minimises systematic reference-bias in the SSM.
 *
 * Call runWithGpParams(...) from the per-set launcher objects, or run this
 * object directly (SCAPULA_GP_SET env var selects the set, default = 2).
 *
 * GP parameter sets (σ = Gaussian bandwidth, s = amplitude / scale factor):
 *   Set 1: σ=100 s=100  — GPMM paper baseline
 *   Set 2: σ=100 s=150  — more flexibility, same reach
 *   Set 3: σ=100 s=200  — Alemneh's scapula-tuned value  ← default
 *   Set 4: σ= 75 s=150  — Madsen's tighter-reach fix
 *   Set 5: σ= 75 s=200  — tighter reach, maximum flexibility
 */
object Stage2NonRigidReg {

  // -------------------------------------------------------------------------
  // Parameter types
  // -------------------------------------------------------------------------

  case class GpParams(sigma: Double, scaleFactor: Double, label: String)

  val gpParamSets: IndexedSeq[GpParams] = IndexedSeq(
    GpParams(100, 100, "set1-baseline"),
    GpParams(100, 150, "set2-flexible"),
    GpParams(100, 200, "set3-alemneh"),   // default
    GpParams( 75, 150, "set4-madsen"),
    GpParams( 75, 200, "set5-tight-flex")
  )

  case class RegistrationParameters(
      regularizationWeight: Double,
      numberOfIterations: Int,
      numberOfSampledPoints: Int
  )

  /** Coarse-to-fine schedule. */
  val regSchedule: IndexedSeq[RegistrationParameters] = IndexedSeq(
    RegistrationParameters(1e-1, 20,  500),
    RegistrationParameters(1e-2, 30, 1000),
    RegistrationParameters(1e-4, 40, 2000),
    RegistrationParameters(1e-6, 50, 4000)
  )

  // -------------------------------------------------------------------------
  // Reference selection
  // -------------------------------------------------------------------------

  def chooseReference(
      specimens: IndexedSeq[ScapulaData.Specimen],
      landmarks: Map[String, IndexedSeq[Landmark[_3D]]]
  ): ScapulaData.Specimen = {
    val candidates = specimens.filter(s => !s.isRight && landmarks.contains(s.modelId))
    require(candidates.nonEmpty, "No left-side specimens with landmarks found")

    val allLmData = candidates.map(s => landmarks(s.modelId))
    val meanPositions: Map[String, Point[_3D]] = ScapulaData.landmarkNames.map { nm =>
      val pts = allLmData.flatMap(_.find(_.id == nm).map(_.point))
      require(pts.nonEmpty, s"Landmark '$nm' missing from every candidate")
      val sum = pts.foldLeft(EuclideanVector3D(0, 0, 0))((acc, p) => acc + p.toVector)
      val mean = sum * (1.0 / pts.length)
      nm -> Point3D(mean.x, mean.y, mean.z)
    }.toMap

    candidates.minBy { s =>
      ScapulaData.landmarkNames.map { nm =>
        val pt = landmarks(s.modelId).find(_.id == nm).map(_.point).getOrElse(Point3D(0, 0, 0))
        (pt - meanPositions(nm)).norm
      }.sum
    }
  }

  // -------------------------------------------------------------------------
  // GP construction — single Gaussian kernel + posterior conditioning
  // -------------------------------------------------------------------------

  /**
   * Builds a zero-mean GP over the reference mesh using a SINGLE Gaussian
   * kernel (DiagonalKernel3D broadcasting one GaussianKernel3D to all three
   * spatial dimensions independently).  The GP is then conditioned on the
   * landmark correspondences so that the posterior mean already maps each
   * reference landmark towards the corresponding (rigidly-aligned) target
   * landmark before the LBFGS optimisation starts.
   *
   * NearestNeighborInterpolator3D is used so every off-vertex query is
   * resolved to the closest mesh vertex — consistent with the NN projection
   * step at the end of the pipeline.
   *
   * @param lmNoiseStdMm  observation noise (mm) — set to ~digitisation error
   */
  def buildPosteriorGP(
      refMesh: TriangleMesh[_3D],
      refLms: IndexedSeq[Landmark[_3D]],
      targetLmsAligned: IndexedSeq[Landmark[_3D]],
      gpParams: GpParams,
      lmNoiseStdMm: Double = 2.0
  ): LowRankGaussianProcess[_3D, EuclideanVector[_3D]] = {

    // Single Gaussian kernel — same bandwidth and amplitude in x, y, z.
    val zeroMean = Field(EuclideanSpace3D, (_: Point[_3D]) => EuclideanVector.zeros[_3D])
    val kernel   = DiagonalKernel3D(
      GaussianKernel3D(sigma = gpParams.sigma, scaleFactor = gpParams.scaleFactor),
      outputDim = 3
    )
    val gp = GaussianProcess(zeroMean, kernel)

    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      referenceMesh     = refMesh,
      gp                = gp,
      relativeTolerance = Config.gpRelativeTolerance,
      interpolator      = NearestNeighborInterpolator3D[EuclideanVector[_3D]]()
    )
    println(s"    GP rank=${lowRankGP.rank}  σ=${gpParams.sigma}  s=${gpParams.scaleFactor}")

    // Posterior conditioned on landmark observations.
    val lmNoise = MultivariateNormalDistribution(
      DenseVector.zeros[Double](3),
      DenseMatrix.eye[Double](3) * (lmNoiseStdMm * lmNoiseStdMm)
    )
    val observations: IndexedSeq[(Point[_3D], EuclideanVector[_3D], MultivariateNormalDistribution)] =
      refLms.zip(targetLmsAligned).map { case (src, tgt) =>
        (src.point, tgt.point - src.point, lmNoise)
      }

    lowRankGP.posterior(observations)
  }

  // -------------------------------------------------------------------------
  // Single registration pass at one regularisation weight
  // -------------------------------------------------------------------------

  def doRegistration(
      posteriorGP: LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
      referenceMesh: TriangleMesh[_3D],
      targetMesh: TriangleMesh[_3D],
      regParams: RegistrationParameters,
      initialCoefficients: DenseVector[Double]
  ): DenseVector[Double] = {
    val transformationSpace = GaussianProcessTransformationSpace(posteriorGP)
    val fixedImage          = referenceMesh.operations.toDistanceImage
    val movingImage         = targetMesh.operations.toDistanceImage
    val sampler             = FixedPointsUniformMeshSampler3D(referenceMesh, regParams.numberOfSampledPoints)
    val metric              = MeanSquaresMetric(fixedImage, movingImage, transformationSpace, sampler)
    val optimizer           = LBFGSOptimizer(regParams.numberOfIterations)
    val regularizer         = L2Regularizer(transformationSpace)
    val registration        = Registration(metric, regularizer, regParams.regularizationWeight, optimizer)

    val regIterator = registration.iterator(initialCoefficients)
    val loggingIterator = for ((state, itnum) <- regIterator.zipWithIndex) yield {
      if (itnum % 10 == 0) println(f"      iter $itnum%3d  obj=${state.value}%.4e")
      state
    }
    loggingIterator.toSeq.last.parameters
  }

  // -------------------------------------------------------------------------
  // Full per-specimen pipeline
  // -------------------------------------------------------------------------

  /**
   * Returns (rigidlyAligned, registeredInRefTopology).  The caller shows both
   * in the UI so the user can compare before/after non-rigid registration.
   */
  def registerOne(
      refMesh: TriangleMesh[_3D],
      refLms: IndexedSeq[Landmark[_3D]],
      specimen: ScapulaData.Specimen,
      specimenLms: IndexedSeq[Landmark[_3D]],
      gpParams: GpParams
  )(implicit rng: Random): (TriangleMesh[_3D], TriangleMesh[_3D]) = {

    val (rawMesh, rawLms) =
      if (specimen.isRight)
        (ScapulaData.mirrorMesh(ScapulaData.loadMesh(specimen.file)),
         ScapulaData.mirrorLandmarks(specimenLms))
      else
        (ScapulaData.loadMesh(specimen.file), specimenLms)

    val (rigidMesh, rigidLms) = RigidAlign.landmarkThenIcp(
      mesh          = rawMesh,
      lms           = rawLms,
      targetMesh    = refMesh,
      targetLms     = refLms,
      icpIterations = Config.icpIterations
    )

    val posteriorGP = buildPosteriorGP(refMesh, refLms, rigidLms, gpParams)

    val finalCoeffs = regSchedule.foldLeft(DenseVector.zeros[Double](posteriorGP.rank)) {
      (coeffs, regParams) =>
        println(s"    λ=${regParams.regularizationWeight}  iters=${regParams.numberOfIterations}  pts=${regParams.numberOfSampledPoints}")
        doRegistration(posteriorGP, refMesh, rigidMesh, regParams, coeffs)
    }

    val transformationSpace   = GaussianProcessTransformationSpace(posteriorGP)
    val registrationTransform = transformationSpace.transformationForParameters(finalCoeffs)
    val targetOps             = rigidMesh.operations
    // Nearest-neighbour projection onto the rigidly-aligned target surface.
    val projection            = (pt: Point[_3D]) => targetOps.closestPointOnSurface(pt).point
    val registeredMesh        = refMesh.transform(registrationTransform.andThen(projection))

    (rigidMesh, registeredMesh)
  }

  // -------------------------------------------------------------------------
  // Shared runner — called by per-set launchers and by main()
  // -------------------------------------------------------------------------

  def runWithGpParams(gpParams: GpParams): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dir    = Config.dataDir
    val outDir = Config.outDir
    outDir.mkdirs()
    println(s"Data dir : ${dir.getAbsolutePath}")
    println(s"Out  dir : ${outDir.getAbsolutePath}")
    println(s"GP set   : σ=${gpParams.sigma}  s=${gpParams.scaleFactor}  (${gpParams.label})")

    val csv = ScapulaData.csvFile(dir)
    val (landmarks, fromHeader, _) = ScapulaData.readLandmarkCsv(csv)
    if (!fromHeader)
      println("WARNING: landmark columns resolved by hardcoded offsets — verify against your CSV")

    val specimens = ScapulaData.specimens(dir)
    val reference = chooseReference(specimens, landmarks)
    println(s"Reference: ${reference.modelId}")
    val refMesh = ScapulaData.loadMesh(reference.file)
    val refLms  = landmarks(reference.modelId)

    val gpOutDir = new File(outDir, gpParams.label)
    gpOutDir.mkdirs()

    val ui         = if (Config.showUi) Some(ScalismoUI()) else None
    val refGroup   = ui.map(_.createGroup("reference"))
    val origGroup  = ui.map(_.createGroup("original-non-registered"))
    val rigidGroup = ui.map(_.createGroup("rigid-aligned"))
    val regGroup   = ui.map(_.createGroup("registered"))

    ui.foreach { u =>
      val rv = u.show(refGroup.get, refMesh, "reference")
      rv.color = java.awt.Color.RED
    }

    val toRegister = specimens.filter(s =>
      s.modelId != reference.modelId && landmarks.contains(s.modelId)
    )

    toRegister.zipWithIndex.foreach { case (spec, idx) =>
      println(s"\n[${idx + 1}/${toRegister.length}] ${spec.modelId}")

      // Show original (non-registered) mesh in gray — user can compare before/after.
      val rawForDisplay =
        if (spec.isRight) ScapulaData.mirrorMesh(ScapulaData.loadMesh(spec.file))
        else              ScapulaData.loadMesh(spec.file)
      ui.foreach { u =>
        val ov = u.show(origGroup.get, rawForDisplay, s"orig_${spec.modelId}")
        ov.color   = java.awt.Color.LIGHT_GRAY
        ov.opacity = 0.4f
      }

      val (rigidMesh, registeredMesh) = registerOne(refMesh, refLms, spec, landmarks(spec.modelId), gpParams)

      ui.foreach { u =>
        val rv = u.show(rigidGroup.get, rigidMesh, s"rigid_${spec.modelId}")
        rv.color   = java.awt.Color.BLUE
        rv.opacity = 0.4f
      }
      ui.foreach { u =>
        val rv = u.show(regGroup.get, registeredMesh, s"reg_${spec.modelId}")
        rv.color   = java.awt.Color.GREEN
        rv.opacity = 0.7f
      }

      val stats = Metrics.symmetric(registeredMesh, rigidMesh)
      println(s"  surface dist: ${stats.render}")

      MeshIO.writeMesh(registeredMesh, new File(gpOutDir, s"${spec.modelId}_registered.stl")).getOrElse(
        throw new RuntimeException(s"Could not write mesh for ${spec.modelId}")
      )
    }

    MeshIO.writeMesh(refMesh, new File(gpOutDir, s"${reference.modelId}_registered.stl")).getOrElse(
      throw new RuntimeException("Could not save reference mesh")
    )

    println(s"\nDone. ${toRegister.length + 1} meshes → ${gpOutDir.getAbsolutePath}")
    println("UI: gray=original | blue=rigid-only | green=registered | red=reference")
  }

  def main(args: Array[String]): Unit = {
    val gpSetIdx = sys.env.getOrElse("SCAPULA_GP_SET", "2").toInt
      .max(0).min(gpParamSets.length - 1)
    runWithGpParams(gpParamSets(gpSetIdx))
  }
}
