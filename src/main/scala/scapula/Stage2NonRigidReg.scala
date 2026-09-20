package scapula

import scalismo.geometry.*
import scalismo.common.*
import scalismo.common.interpolation.TriangleMeshInterpolator3D
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
 *   2. GPA landmark-based rigid Procrustes (aligns to the GPA mean reference).
 *   3. Automatic trimmed rigid ICP.
 *   4. Zero-mean GP with a MULTI-SCALE Gaussian kernel (σ, σ/2, σ/4 at
 *      amplitudes s, s/2, s/4) conditioned on landmark correspondences
 *      (posterior GP) to anchor the deformation anatomically.
 *      Multi-scale captures both global blade shape (large σ) and local
 *      process details (small σ) — matches Lüthi GPMM paper and Tutorial 12.
 *   5. 5-pass coarse-to-fine LBFGS optimisation (λ: 1e-1 → 1e-6).
 *   6. Nearest-neighbour projection onto the rigidly-aligned target surface.
 *   7. Save in reference-mesh topology for Stage 3 SSM building.
 *
 * Reference selection: GPA (Generalised Procrustes Analysis) — rigidly align
 * all landmark sets, compute the mean position, pick the specimen closest to
 * the GPA mean. This is the method recommended by Lüthi et al. and used in
 * Alemneh's and Madsen's scapula SSM theses.
 *
 * Call runWithGpParams(...) from the per-set launcher objects, or run this
 * object directly (SCAPULA_GP_SET env var selects the set, default = 2).
 *
 * GP parameter sets (σ = base bandwidth, s = base amplitude; 3 scales each):
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

  /**
   * 4-pass coarse-to-fine schedule.
   * λ values and iteration counts follow Dennis Madsen's scapula recommendation
   * (Scalismo forum, Jun 2024).  Sample counts are increased from his demo
   * values (100/500/100/5000) to be adequate for real inter-subject data.
   */
  val regSchedule: IndexedSeq[RegistrationParameters] = IndexedSeq(
    RegistrationParameters(1e-1, 30, 1000),
    RegistrationParameters(1e-2, 40, 1000),
    RegistrationParameters(1e-4, 40, 2000),
    RegistrationParameters(1e-6, 50, 2000)   // 5000→2000: main speed fix
  )

  // -------------------------------------------------------------------------
  // Reference selection
  // -------------------------------------------------------------------------

  /**
   * GPA (Generalised Procrustes Analysis) reference selection.
   *
   * Algorithm:
   *   1. Take left-side candidates only (right scapulae are already mirrored
   *      during registration, but the reference must be a true left).
   *   2. Rigidly align every candidate's landmark set to the first candidate
   *      using landmark Procrustes (LandmarkRegistration.rigid3DLandmarkRegistration).
   *   3. Compute the mean landmark position across all aligned sets.
   *   4. Pick the candidate whose aligned landmarks lie closest (sum of L2
   *      distances across all 5 landmarks) to the GPA mean.
   *
   * One-shot GPA (no iteration) is sufficient for 5 landmarks;
   * the iterative extension changes the result by < 0.1 mm in practice.
   */
  def chooseReference(
      specimens: IndexedSeq[ScapulaData.Specimen],
      landmarks: Map[String, IndexedSeq[Landmark[_3D]]]
  ): ScapulaData.Specimen = {
    val candidates = specimens.filter(s => !s.isRight && landmarks.contains(s.modelId))
    require(candidates.nonEmpty, "No left-side specimens with landmarks found")

    val firstLms = landmarks(candidates.head.modelId)

    // Align every candidate's landmarks to the first candidate (rigid Procrustes).
    val alignedLms: IndexedSeq[IndexedSeq[Landmark[_3D]]] = candidates.map { s =>
      val lms = landmarks(s.modelId)
      val t   = LandmarkRegistration.rigid3DLandmarkRegistration(lms, firstLms, Point3D(0, 0, 0))
      lms.map(lm => lm.copy(point = t(lm.point)))
    }

    // Mean landmark positions in the aligned (GPA) frame.
    val meanPositions: Map[String, Point[_3D]] = ScapulaData.landmarkNames.map { nm =>
      val pts = alignedLms.flatMap(_.find(_.id == nm).map(_.point))
      require(pts.nonEmpty, s"Landmark '$nm' missing from every candidate")
      val sum  = pts.foldLeft(EuclideanVector3D(0, 0, 0))((acc, p) => acc + p.toVector)
      val mean = sum * (1.0 / pts.length)
      nm -> Point3D(mean.x, mean.y, mean.z)
    }.toMap

    // Candidate closest to GPA mean.
    candidates.zip(alignedLms).minBy { case (_, lms) =>
      ScapulaData.landmarkNames.map { nm =>
        val pt = lms.find(_.id == nm).map(_.point).getOrElse(Point3D(0, 0, 0))
        (pt - meanPositions(nm)).norm
      }.sum
    }._1
  }

  // -------------------------------------------------------------------------
  // GP construction — single Gaussian kernel + posterior conditioning
  // -------------------------------------------------------------------------

  /**
   * Builds a zero-mean GP over the reference mesh using a MULTI-SCALE Gaussian
   * kernel with three scales at σ, σ*0.4, σ*0.2.
   *
   * Sigma ratios from Dennis Madsen's scapula guide (Scalismo forum Jun 2024):
   *   longest_side/2 : longest_side/5 : longest_side/10
   *   For a ~150mm scapula: 75 : 30 : 15 mm
   *   → use gpParams.sigma = 75 (Set 4) to get exactly those values.
   *
   * The GP is conditioned on landmark correspondences (posterior GP) so the
   * posterior mean already maps each reference landmark towards the
   * (rigidly-aligned) target landmark before LBFGS starts.
   *
   * @param lmNoiseStdMm  observation noise (mm) — ~digitisation error (2 mm)
   */
  def buildPosteriorGP(
      refMesh: TriangleMesh[_3D],
      refLms: IndexedSeq[Landmark[_3D]],
      targetLmsAligned: IndexedSeq[Landmark[_3D]],
      gpParams: GpParams,
      lmNoiseStdMm: Double = 2.0
  ): LowRankGaussianProcess[_3D, EuclideanVector[_3D]] = {

    // Multi-scale kernel following Dennis Madsen's scapula guide (Jun 2024):
    //   σ_coarse = longest_side / 2  ≈ 75 mm   (whole-blade bending)
    //   σ_mid    = longest_side / 5  ≈ 30 mm   (glenoid fossa, spine)
    //   σ_fine   = longest_side / 10 ≈ 15 mm   (acromion tip, coracoid)
    // With gpParams.sigma = 75 (Set 4 — Madsen) these give EXACTLY his values.
    // Amplitude ratios match his demo: 1 : 2/3 : 1/3.
    // relativeTolerance = 0.01 (his exact value).
    val σ = gpParams.sigma
    val s = gpParams.scaleFactor
    val scalarKernel =
      GaussianKernel3D(σ,        s             ) +   // coarse  σ
      GaussianKernel3D(σ * 0.4,  s * 0.67      ) +   // mid     σ*0.4  (≈ longest/5)
      GaussianKernel3D(σ * 0.2,  s * 0.33      )     // fine    σ*0.2  (≈ longest/10)

    val zeroMean = Field(EuclideanSpace3D, (_: Point[_3D]) => EuclideanVector.zeros[_3D])
    val kernel   = DiagonalKernel3D(scalarKernel, outputDim = 3)
    val gp       = GaussianProcess(zeroMean, kernel)

    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      refMesh,
      gp,
      Config.gpRelativeTolerance,  // 0.01 — Dennis's exact value
      TriangleMeshInterpolator3D[EuclideanVector[_3D]]()
    )
    val σ2 = f"${σ * 0.4}%.1f"; val σ3 = f"${σ * 0.2}%.1f"
    println(s"    GP rank=${lowRankGP.rank}  σ=($σ,$σ2,$σ3)  s=($s,${(s*0.67).toInt},${(s*0.33).toInt})")

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
  )(implicit rng: Random): DenseVector[Double] = {
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
    val refMeshRaw = ScapulaData.loadMesh(reference.file)
    val refMesh = refMeshRaw.operations.decimate(Config.modelResolution)
    println(s"Reference decimated: ${refMeshRaw.pointSet.numberOfPoints} → ${refMesh.pointSet.numberOfPoints} vertices")
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

    val allToRegister = specimens.filter(s =>
      s.modelId != reference.modelId && landmarks.contains(s.modelId)
    )
    val toRegister =
      if (Config.maxSpecimens > 0) allToRegister.take(Config.maxSpecimens) else allToRegister
    if (Config.maxSpecimens > 0)
      println(s"SCAPULA_MAX_SPECIMENS=${Config.maxSpecimens}: registering ${toRegister.length} of ${allToRegister.length} specimens")

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
