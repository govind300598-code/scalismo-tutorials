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
 * STAGE 2 — NON-RIGID REGISTRATION.
 *
 * Pipeline per specimen:
 *   1. Mirror right scapulae → left orientation.
 *   2. Landmark-based rigid Procrustes.
 *   3. Automatic trimmed rigid ICP.
 *   4. Build a zero-mean GP with Gaussian kernel; condition it on landmark
 *      correspondences (posterior GP) so the non-rigid deformation is anchored
 *      at anatomical points from the start.
 *   5. Multi-resolution LBFGS optimisation (decreasing regularisation weight).
 *   6. Project each deformed reference point onto the closest surface point of
 *      the (rigidly aligned) target to obtain exact point-to-surface
 *      correspondence.
 *   7. Save the result in reference-mesh topology for Stage 3 SSM building.
 *
 * Reference selection: choose the left-side specimen whose 5 landmarks lie
 * closest (sum of Euclidean distances) to the cross-subject mean landmark
 * positions. This minimises systematic reference-bias in the SSM.
 *
 * GP parameter sets (σ = Gaussian bandwidth, s = scale factor / amplitude):
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
    GpParams(100, 200, "set3-alemneh"),  // default
    GpParams( 75, 150, "set4-madsen"),
    GpParams( 75, 200, "set5-tight-flex")
  )

  case class RegistrationParameters(
      regularizationWeight: Double,
      numberOfIterations: Int,
      numberOfSampledPoints: Int
  )

  /** Coarse-to-fine schedule — same pattern as Tutorial 12 but tuned for bone. */
  val regSchedule: IndexedSeq[RegistrationParameters] = IndexedSeq(
    RegistrationParameters(1e-1, 20,  500),
    RegistrationParameters(1e-2, 30, 1000),
    RegistrationParameters(1e-4, 40, 2000),
    RegistrationParameters(1e-6, 50, 4000)
  )

  // -------------------------------------------------------------------------
  // Reference selection
  // -------------------------------------------------------------------------

  /**
   * Choose the reference specimen as the left-side scapula whose 5 landmark
   * positions are closest to the cross-subject mean. Using the mean position
   * rather than a minimum-pairwise-distance criterion keeps the cost O(N)
   * instead of O(N²) while giving a closely equivalent result.
   */
  def chooseReference(
      specimens: IndexedSeq[ScapulaData.Specimen],
      landmarks: Map[String, IndexedSeq[Landmark[_3D]]]
  ): ScapulaData.Specimen = {
    // Only left-side specimens that have landmarks can be reference candidates.
    val candidates = specimens.filter(s => !s.isRight && landmarks.contains(s.modelId))
    require(candidates.nonEmpty, "No left-side specimens with landmarks found — check data directory and CSV")

    val allLmData = candidates.map(s => landmarks(s.modelId))

    // Cross-subject mean for each landmark.
    val meanPositions: Map[String, Point[_3D]] = ScapulaData.landmarkNames.map { nm =>
      val pts = allLmData.flatMap(_.find(_.id == nm).map(_.point))
      require(pts.nonEmpty, s"Landmark '$nm' missing from every candidate")
      val sum = pts.foldLeft(EuclideanVector3D(0, 0, 0))((acc, p) => acc + p.toVector)
      val mean = sum * (1.0 / pts.length)
      nm -> Point3D(mean.x, mean.y, mean.z)
    }.toMap

    // Specimen whose landmarks have the smallest total distance to the mean.
    candidates.minBy { s =>
      ScapulaData.landmarkNames.map { nm =>
        val pt = landmarks(s.modelId).find(_.id == nm).map(_.point).getOrElse(Point3D(0, 0, 0))
        (pt - meanPositions(nm)).norm
      }.sum
    }
  }

  // -------------------------------------------------------------------------
  // GP construction and posterior conditioning on landmarks
  // -------------------------------------------------------------------------

  /**
   * Build a zero-mean GP over the reference mesh using a Gaussian kernel, then
   * condition it on the landmark observations so that the posterior mean already
   * maps each reference landmark towards the corresponding target landmark.
   * Using NearestNeighborInterpolator3D ensures that every off-vertex query is
   * resolved to the closest mesh vertex — consistent with nearest-neighbour
   * correspondence throughout the pipeline.
   *
   * @param refMesh          reference mesh (topology that all registered meshes will share)
   * @param refLms           landmarks on the reference mesh (in reference frame)
   * @param targetLmsAligned landmarks on the target mesh AFTER rigid alignment to reference
   * @param gpParams         kernel bandwidth and amplitude
   * @param lmNoiseStdMm     observation noise (mm) — set to ~digitisation error, e.g. 2 mm
   */
  def buildPosteriorGP(
      refMesh: TriangleMesh[_3D],
      refLms: IndexedSeq[Landmark[_3D]],
      targetLmsAligned: IndexedSeq[Landmark[_3D]],
      gpParams: GpParams,
      lmNoiseStdMm: Double = 2.0
  ): LowRankGaussianProcess[_3D, EuclideanVector[_3D]] = {

    val zeroMean = Field(EuclideanSpace3D, (_: Point[_3D]) => EuclideanVector.zeros[_3D])
    val kernel = DiagonalKernel3D(
      GaussianKernel3D(sigma = gpParams.sigma, scaleFactor = gpParams.scaleFactor),
      outputDim = 3
    )
    val gp = GaussianProcess(zeroMean, kernel)

    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      referenceMesh = refMesh,
      gp            = gp,
      relativeTolerance = Config.gpRelativeTolerance,
      interpolator  = NearestNeighborInterpolator3D[EuclideanVector[_3D]]()
    )

    println(s"    GP rank = ${lowRankGP.rank}  (σ=${gpParams.sigma} s=${gpParams.scaleFactor})")

    // Landmark observations: for each reference landmark point, the deformation
    // should move it towards the corresponding target landmark.
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
  // Single-pass registration at one regularisation weight
  // -------------------------------------------------------------------------

  def doRegistration(
      posteriorGP: LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
      referenceMesh: TriangleMesh[_3D],
      targetMesh: TriangleMesh[_3D],
      regParams: RegistrationParameters,
      initialCoefficients: DenseVector[Double]
  ): DenseVector[Double] = {
    val transformationSpace = GaussianProcessTransformationSpace(posteriorGP)
    val fixedImage           = referenceMesh.operations.toDistanceImage
    val movingImage          = targetMesh.operations.toDistanceImage
    val sampler              = FixedPointsUniformMeshSampler3D(referenceMesh, regParams.numberOfSampledPoints)
    val metric               = MeanSquaresMetric(fixedImage, movingImage, transformationSpace, sampler)
    val optimizer            = LBFGSOptimizer(regParams.numberOfIterations)
    val regularizer          = L2Regularizer(transformationSpace)
    val registration         = Registration(metric, regularizer, regParams.regularizationWeight, optimizer)

    val regIterator = registration.iterator(initialCoefficients)
    val loggingIterator = for ((state, itnum) <- regIterator.zipWithIndex) yield {
      if (itnum % 10 == 0) println(f"      iter $itnum%3d  obj=${state.value}%.4e")
      state
    }
    loggingIterator.toSeq.last.parameters
  }

  // -------------------------------------------------------------------------
  // Full registration pipeline for a single specimen
  // -------------------------------------------------------------------------

  /**
   * Registers one specimen to the reference mesh.
   *
   * Steps:
   *   1. Mirror right → left.
   *   2. Landmark rigid Procrustes + trimmed ICP.
   *   3. Posterior GP conditioned on landmark correspondences.
   *   4. Multi-resolution LBFGS (regSchedule).
   *   5. Nearest-neighbour projection onto target surface.
   *
   * Returns (rigidlyAligned, registeredInRefTopology) so the caller can
   * display both stages in the UI.
   */
  def registerOne(
      refMesh: TriangleMesh[_3D],
      refLms: IndexedSeq[Landmark[_3D]],
      specimen: ScapulaData.Specimen,
      specimenLms: IndexedSeq[Landmark[_3D]],
      gpParams: GpParams
  )(implicit rng: Random): (TriangleMesh[_3D], TriangleMesh[_3D]) = {

    // Step 1 — mirror right to left orientation
    val (rawMesh, rawLms) =
      if (specimen.isRight)
        (ScapulaData.mirrorMesh(ScapulaData.loadMesh(specimen.file)),
         ScapulaData.mirrorLandmarks(specimenLms))
      else
        (ScapulaData.loadMesh(specimen.file), specimenLms)

    // Step 2 — landmark rigid Procrustes + automatic trimmed rigid ICP
    val (rigidMesh, rigidLms) = RigidAlign.landmarkThenIcp(
      mesh          = rawMesh,
      lms           = rawLms,
      targetMesh    = refMesh,
      targetLms     = refLms,
      icpIterations = Config.icpIterations
    )

    // Step 3 — posterior GP anchored on landmark observations
    val posteriorGP = buildPosteriorGP(refMesh, refLms, rigidLms, gpParams)

    // Step 4 — multi-resolution non-rigid LBFGS
    val finalCoeffs = regSchedule.foldLeft(DenseVector.zeros[Double](posteriorGP.rank)) {
      (coeffs, regParams) =>
        println(s"    regWeight=${regParams.regularizationWeight}  iters=${regParams.numberOfIterations}  pts=${regParams.numberOfSampledPoints}")
        doRegistration(posteriorGP, refMesh, rigidMesh, regParams, coeffs)
    }

    // Step 5 — transform reference mesh and project onto target surface
    val transformationSpace      = GaussianProcessTransformationSpace(posteriorGP)
    val registrationTransform    = transformationSpace.transformationForParameters(finalCoeffs)
    val targetOps                = rigidMesh.operations

    // Nearest-neighbour projection: each reference vertex is mapped to its
    // closest point on the (rigidly aligned) target surface.
    val projection               = (pt: Point[_3D]) => targetOps.closestPointOnSurface(pt).point
    val finalTransform           = registrationTransform.andThen(projection)
    val registeredMesh           = refMesh.transform(finalTransform)

    (rigidMesh, registeredMesh)
  }

  // -------------------------------------------------------------------------
  // main
  // -------------------------------------------------------------------------

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dir    = Config.dataDir
    val outDir = Config.outDir
    outDir.mkdirs()
    println(s"Data dir : ${dir.getAbsolutePath}")
    println(s"Out  dir : ${outDir.getAbsolutePath}")

    val csv = ScapulaData.csvFile(dir)
    val (landmarks, fromHeader, _) = ScapulaData.readLandmarkCsv(csv)
    if (!fromHeader)
      println("WARNING: landmark columns resolved by hardcoded offsets — verify against your CSV")
    println(s"Landmarks: ${landmarks.size} rows | ${ScapulaData.landmarkNames.mkString(", ")}")

    val specimens = ScapulaData.specimens(dir)
    println(s"STL files: ${specimens.length}")

    // ---- select reference ---------------------------------------------------
    val reference = chooseReference(specimens, landmarks)
    println(s"\nSelected reference: ${reference.modelId}")
    val refMesh = ScapulaData.loadMesh(reference.file)
    val refLms  = landmarks(reference.modelId)

    // ---- GP parameter set ---------------------------------------------------
    // Default: Set 3 (Alemneh, σ=100 s=200). Override via SCAPULA_GP_SET (0-indexed).
    val gpSetIdx = sys.env.getOrElse("SCAPULA_GP_SET", "2").toInt
      .max(0).min(gpParamSets.length - 1)
    val gpParams = gpParamSets(gpSetIdx)
    println(s"GP set $gpSetIdx : σ=${gpParams.sigma}  scaleFactor=${gpParams.scaleFactor}  (${gpParams.label})")

    val gpOutDir = new File(outDir, gpParams.label)
    gpOutDir.mkdirs()

    // ---- UI -----------------------------------------------------------------
    val ui           = if (Config.showUi) Some(ScalismoUI()) else None
    val refGroup     = ui.map(_.createGroup("reference"))
    val origGroup    = ui.map(_.createGroup("original-non-registered"))
    val rigidGroup   = ui.map(_.createGroup("rigid-aligned"))
    val regGroup     = ui.map(_.createGroup("registered"))

    // Show reference mesh in red.
    ui.foreach { u =>
      val rv = u.show(refGroup.get, refMesh, "reference")
      rv.color = java.awt.Color.RED
    }

    // ---- register -----------------------------------------------------------
    val toRegister = specimens.filter(s =>
      s.modelId != reference.modelId && landmarks.contains(s.modelId)
    )

    toRegister.zipWithIndex.foreach { case (spec, idx) =>
      println(s"\n[${idx + 1}/${toRegister.length}] ${spec.modelId}")

      // Show the original (non-registered) mesh in the UI so the user can
      // compare before and after non-rigid registration.
      val rawForDisplay =
        if (spec.isRight) ScapulaData.mirrorMesh(ScapulaData.loadMesh(spec.file))
        else              ScapulaData.loadMesh(spec.file)
      ui.foreach { u =>
        val ov = u.show(origGroup.get, rawForDisplay, s"orig_${spec.modelId}")
        ov.color = java.awt.Color.LIGHT_GRAY
        ov.opacity = 0.4f
      }

      val (rigidMesh, registeredMesh) = registerOne(refMesh, refLms, spec, landmarks(spec.modelId), gpParams)

      // Show rigid-aligned mesh in blue.
      ui.foreach { u =>
        val rv = u.show(rigidGroup.get, rigidMesh, s"rigid_${spec.modelId}")
        rv.color = java.awt.Color.BLUE
        rv.opacity = 0.4f
      }

      // Show registered mesh in green.
      ui.foreach { u =>
        val rv = u.show(regGroup.get, registeredMesh, s"reg_${spec.modelId}")
        rv.color = java.awt.Color.GREEN
        rv.opacity = 0.7f
      }

      // Surface distance of the final registered mesh vs. the (aligned) target.
      val stats = Metrics.symmetric(registeredMesh, rigidMesh)
      println(s"  surface dist: ${stats.render}")

      val outFile = new File(gpOutDir, s"${spec.modelId}_registered.stl")
      MeshIO.writeMesh(registeredMesh, outFile).getOrElse(
        throw new RuntimeException(s"Could not write ${outFile.getPath}")
      )
    }

    // Save the reference itself (already in reference topology).
    MeshIO.writeMesh(refMesh, new File(gpOutDir, s"${reference.modelId}_registered.stl")).getOrElse(
      throw new RuntimeException("Could not save reference mesh")
    )

    println(s"\nDone. ${toRegister.length + 1} meshes written to ${gpOutDir.getAbsolutePath}")
    println("UI groups:  'original-non-registered' (gray) | 'rigid-aligned' (blue) | 'registered' (green) | 'reference' (red)")
    println("Proceed to Stage 3 to build the statistical shape model from the registered meshes.")
  }
}
