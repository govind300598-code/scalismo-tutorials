package scapula

import scalismo.geometry.*
import scalismo.mesh.*
import scalismo.io.MeshIO
import scalismo.registration.LandmarkRegistration
import scalismo.transformations.TranslationAfterRotation

import java.io.File
import scala.io.Source
import scala.util.Using

/**
 * Configuration. Everything is overridable through environment variables so that the same jar can be run against a
 * different dataset without editing code.
 */
object Config {
  private def env(key: String, default: String): String = sys.env.getOrElse(key, default)
  private def envDoubles(key: String, default: Seq[Double]): Seq[Double] =
    sys.env.get(key).map(_.split(",").map(_.trim.toDouble).toIndexedSeq).getOrElse(default)
  private def envInts(key: String, default: Seq[Int]): Seq[Int] =
    sys.env.get(key).map(_.split(",").map(_.trim.toInt).toIndexedSeq).getOrElse(default)

  val dataDir: File =
    new File(env("SCAPULA_DATA_DIR", "/home/g25upadh/Documents/100 plus scapula data/paired_scapulae_STLs_scapula"))
  // Deliberately distinct from any pre-existing output folder (e.g. ssm_output, ssm_final, aligned_scapulae_output)
  // so this pipeline's results never land in, or get mixed up with, results from a different script/run.
  val outDir: File =
    new File(env("SCAPULA_OUT_DIR", "/home/g25upadh/Documents/100 plus scapula data/scapula_gp_registration_ssm_out"))

  /**
   * Force a specific specimen (by model id, e.g. paired_scapula_005_F_67_R) as the reference, instead of Stage 2's
   * own GPA-based selection (closest to the population's iteratively-estimated mean landmark configuration). Useful
   * when a "most average" specimen has already been identified by an external analysis. Leave unset ("") to keep
   * the automatic GPA-based selection.
   */
  val referenceSpecimenId: Option[String] =
    sys.env.get("SCAPULA_REFERENCE_SPECIMEN").map(_.trim).filter(_.nonEmpty)

  /**
   * Comma-separated specimen ids (e.g. flagged outliers from an external variation analysis) to call out separately
   * in Stage 2's registration-accuracy report: printed as their own highlighted section plus written to
   * registration_accuracy_watchlist.csv, in addition to appearing in the full registration_accuracy.csv like every
   * other specimen. Leave unset ("") to skip this extra section entirely.
   */
  val watchSpecimenIds: Seq[String] =
    sys.env.get("SCAPULA_WATCH_SPECIMENS").map(_.split(",").map(_.trim).filter(_.nonEmpty).toIndexedSeq).getOrElse(Seq.empty)

  /**
   * Restrict Stage 2 to registering ONLY this comma-separated set of specimen ids (e.g. a handful of flagged
   * outliers), instead of every specimen in SCAPULA_DATA_DIR. The reference specimen (SCAPULA_REFERENCE_SPECIMEN, if
   * set) is automatically included even if not listed here, since it has to be part of the run regardless. Leave
   * unset ("") to process every specimen, as normal.
   */
  val onlySpecimenIds: Option[Set[String]] =
    sys.env.get("SCAPULA_ONLY_SPECIMENS").map(_.split(",").map(_.trim).filter(_.nonEmpty).toSet).filter(_.nonEmpty)

  /**
   * Optional external template mesh to use as the SSM reference topology instead of an in-population specimen.
   * Disabled by default -- Stage2GPNonRigidRegistration instead picks the specimen from your own 24 whose landmarks
   * are closest to the population's mean landmark configuration (see referenceSpecimen in that file), which needs
   * no chirality guessing since it's landmark-aligned like every other specimen. Set SCAPULA_REFERENCE_MESH to a
   * file path to opt back into an external template if you want one later.
   */
  val referenceMeshPath: String = env("SCAPULA_REFERENCE_MESH", "")
  val referenceMeshFile: Option[File] =
    if (referenceMeshPath.trim.isEmpty) None else Some(new File(referenceMeshPath))

  /**
   * Which orientation(s) of the external reference to try: "auto" (default) tries both as-is and mirrored, and
   * keeps whichever aligns better (lower HD95) onto the population's pivot specimen -- a mismatched chirality
   * (template is the opposite side from the population) cannot be fixed by any rigid rotation, only a reflection,
   * so this is a real ambiguity that has to be resolved by trying both, not guessed. "asis" / "mirrored" force one
   * orientation only (skips the other candidate, saving a little time once you already know which is right).
   */
  val referenceOrientation: String = env("SCAPULA_REFERENCE_ORIENTATION", "auto").toLowerCase

  /** Above this mean residual (mm) for the BEST orientation, warn loudly (but still proceed). */
  val referenceAlignWarnMeanMm: Double = env("SCAPULA_REFERENCE_ALIGN_WARN_MM", "10.0").toDouble

  /**
   * Above this HD95 (mm) for the BEST orientation, ABORT before the (~1-2 hour) non-rigid registration loop rather
   * than spend that time on a reference that's already known to be a bad fit. HD95 catches a chirality mismatch that
   * mean residual alone can miss: a wrong-side template can still look deceptively OK on average (similar overall
   * bounding envelope) while its actual anatomical features are systematically misplaced, which mean blurs out
   * across thousands of points but HD95 does not. Override with SCAPULA_REFERENCE_FORCE=true to proceed anyway.
   */
  val referenceAlignAbortHD95Mm: Double = env("SCAPULA_REFERENCE_ALIGN_ABORT_HD95_MM", "15.0").toDouble
  val referenceAlignForce: Boolean = env("SCAPULA_REFERENCE_FORCE", "false").toBoolean

  /** Number of vertices of the model reference. All registered shapes and the SSM live at this resolution. */
  val modelResolution: Int = env("SCAPULA_MODEL_RES", "5000").toInt

  /**
   * A registered mesh's per-point displacement from the reference is a real anatomical deformation, so it is
   * bounded by how different a scapula can plausibly be from the reference (tens of mm at most). Above this bound
   * it is not a "big deformation" -- it means the file is not actually in correspondence with the current
   * reference.vtk (e.g. a stale registered/<id>.vtk left over from an earlier run against a different reference
   * topology). Used by Stage3SSMModelValidation.loadRegistered as a loud, actionable failure instead of silently
   * feeding garbage correspondence into PCA.
   */
  val maxPlausibleDisplacementMm: Double = env("SCAPULA_MAX_PLAUSIBLE_DISPLACEMENT_MM", "60.0").toDouble

  /** Non-rigid (GP) ICP iterations per pass. */
  val icpIterations: Int = env("SCAPULA_ICP_ITERS", "40").toInt

  /**
   * Number of registration passes. Pass 1 registers to an arbitrary specimen; each further pass rebuilds the reference
   * as the mean of the previous pass and re-registers. This removes reference bias.
   */
  val refinePasses: Int = env("SCAPULA_REFINE_PASSES", "2").toInt

  /** Relative tolerance for the pivoted-Cholesky low-rank approximation of the GP prior. Smaller => higher rank. */
  val gpRelativeTolerance: Double = env("SCAPULA_GP_TOL", "0.01").toDouble

  /** Hard cap on the rank of the GP prior (keeps memory and posterior cost bounded). */
  val gpMaxRank: Int = env("SCAPULA_GP_MAX_RANK", "250").toInt

  /**
   * If true, build a second SSM using only one side per subject. Left and mirrored-right scapulae from the same person
   * are NOT statistically independent samples; including both inflates apparent sample size.
   */
  val buildIndependentModel: Boolean = env("SCAPULA_INDEPENDENT_MODEL", "true").toBoolean

  val showUi: Boolean = env("SCAPULA_UI", "true").toBoolean

  val seed: Long = env("SCAPULA_SEED", "42").toLong

  // ---------------------------------------------------------------------------------------------------------------
  // Stage 2: non-rigid (GP) registration. Kernel length scales are DIVISORS of the reference's own bounding-box
  // diagonal rather than fixed millimetre constants, so the same three-scale design (coarse blade / mid body /
  // fine glenoid detail) that Dennis Madsen tuned for his example mesh (150 mm bbox: 150/2, 150/5, 150/10) rescales
  // automatically to whatever size scapula is used as the reference.
  // ---------------------------------------------------------------------------------------------------------------
  val kernelCoarseDivisor: Double = env("SCAPULA_KERNEL_COARSE_DIVISOR", "2").toDouble
  val kernelMidDivisor: Double = env("SCAPULA_KERNEL_MID_DIVISOR", "5").toDouble
  val kernelFineDivisor: Double = env("SCAPULA_KERNEL_FINE_DIVISOR", "10").toDouble
  val kernelCoarseScale: Double = env("SCAPULA_KERNEL_COARSE_SCALE", "15").toDouble
  val kernelMidScale: Double = env("SCAPULA_KERNEL_MID_SCALE", "10").toDouble
  val kernelFineScale: Double = env("SCAPULA_KERNEL_FINE_SCALE", "5").toDouble

  /**
   * Fourth, ultra-fine kernel scale (divisor ~20 -> length scale of a couple cm on a ~200mm scapula, narrowing
   * further toward sub-cm as SCAPULA_MODEL_RES increases and can actually resolve it) targeting small structures the
   * original three scales are too coarse for -- specifically the glenoid rim and coracoid process, the exact detail
   * Loane Le Gall's original mailing-list post wanted more precision on. Small scale factor (2, vs. 5/10/15 for the
   * other three) since fine-detail deformation should be a small correction on top of the coarser scales, not a
   * dominant one -- a large scale factor here would let the optimizer chase per-vertex noise instead of anatomy.
   */
  val kernelUltraFineDivisor: Double = env("SCAPULA_KERNEL_ULTRAFINE_DIVISOR", "20").toDouble
  val kernelUltraFineScale: Double = env("SCAPULA_KERNEL_ULTRAFINE_SCALE", "2").toDouble

  /**
   * Multi-resolution registration cascade (same structure as the mailing-list code: decreasing regularization weight,
   * increasing iteration budget, denser point sampling). Sample-point fractions are of the reference's own vertex
   * count instead of hardcoded absolute counts, so they scale with `modelResolution`.
   */
  val registrationSampleFractions: Seq[Double] = envDoubles("SCAPULA_REG_SAMPLE_FRACTIONS", Seq(0.2, 0.4, 0.8, 1.0))
  val registrationRegWeights: Seq[Double] =
    envDoubles("SCAPULA_REG_WEIGHTS", Seq(1e-1, 1e-2, 1e-4, 1e-6))
  val registrationIterations: Seq[Int] = envInts("SCAPULA_REG_ITERS", Seq(50, 50, 100, 100))

  // ---------------------------------------------------------------------------------------------------------------
  // Stage 3: SSM validation (compactness / generalization / specificity).
  // ---------------------------------------------------------------------------------------------------------------
  /** Upper bound on how many modes are swept when reporting the three validation curves. */
  val maxValidationModes: Int = env("SCAPULA_MAX_VALIDATION_MODES", "15").toInt

  /** Random samples drawn per mode count when estimating specificity. */
  val specificitySamples: Int = env("SCAPULA_SPECIFICITY_SAMPLES", "100").toInt

  /** Compactness thresholds (%) at which generalization/specificity are additionally reported as single numbers. */
  val compactnessThresholds: Seq[Double] = envDoubles("SCAPULA_COMPACTNESS_THRESHOLDS", Seq(90.0, 95.0, 99.0))

  // ---------------------------------------------------------------------------------------------------------------
  // Stage 2 registration METHOD. "lbfgs" (default) is the original mailing-list doRegistration approach --
  // GaussianProcessTransformationSpace + MeanSquaresMetric + L2Regularizer + LBFGSOptimizer, one continuous
  // gradient-based fit. It works well when a specimen differs from the reference by many SMALL deformations
  // spread across the surface, but structurally underfits a specimen with one LARGE, spatially localized
  // deformation (confirmed via SyntheticParamExperiment: six different cascade/kernel settings left the
  // worst-case error on such a case essentially unchanged, ~43mm). "icp-gpr" instead does what ICP does for rigid
  // alignment, but non-rigidly, via GPRegistrationCore.icpGprRegister: iterative closest-point correspondence +
  // closed-form Gaussian process regression (LowRankGaussianProcess.posterior), annealing observation noise from
  // loose to tight -- the "ICP-GPR" instance of GiNGR (Madsen et al. 2022), the same mailing-list author's own
  // successor method. On the same synthetic hard case it cut mean correspondence error by ~87% and worst-case by
  // ~20%, so it is the better default for a population with genuine outlier anatomy -- but "lbfgs" is kept as the
  // default here so nothing about already-validated results changes without an explicit opt-in.
  // ---------------------------------------------------------------------------------------------------------------
  val registrationMethod: String = env("SCAPULA_REGISTRATION_METHOD", "lbfgs").toLowerCase
  val icpGprIterations: Int = env("SCAPULA_ICPGPR_ITERS", "30").toInt
  val icpGprNumPoints: Int = env("SCAPULA_ICPGPR_NUM_POINTS", "2000").toInt

  /** Unlike rigid ICP, trimming the worst correspondences here is counterproductive on exactly the case this
    * method targets: a real, large, localized deformation IS the point with the largest residual, so trimming it
    * away discards the one training signal needed to fit it. Validated on the synthetic hard-deformation case:
    * trim=0.1 gave a better mean (0.50mm) but a worse worst-case (34.0mm) than trim=0.0 (1.14mm mean, 29.9mm
    * worst-case) -- since worst-case (HD95/HD) is what actually matters here, default to no trimming. */
  val icpGprTrimFraction: Double = env("SCAPULA_ICPGPR_TRIM", "0.0").toDouble

  /** Observation-noise variance (mm^2) per iteration, loose -> tight. Same schedule validated on the synthetic
    * hard-deformation test case; transfers reasonably since a scapula's bounding-box scale (~150-250mm) is the
    * same order of magnitude as that test's. */
  val icpGprSigma2Schedule: Seq[Double] =
    envDoubles("SCAPULA_ICPGPR_SIGMA2_SCHEDULE", Seq(100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05))
}

/** Loading, landmark parsing, mirroring and the small geometric helpers shared by all stages. */
object ScapulaData {

  val landmarkNames: IndexedSeq[String] = IndexedSeq("GC", "TS", "IA", "PLA", "AC")

  /** Fallback column offsets, used only if the header cannot be parsed by name. */
  private val fallbackStartIdx: Map[String, Int] =
    Map("GC" -> 11, "TS" -> 14, "IA" -> 17, "PLA" -> 20, "AC" -> 23)

  final case class Specimen(modelId: String, file: File, isRight: Boolean, subject: String)

  def csvFile(dir: File): File = {
    val files = Option(dir.listFiles()).getOrElse(Array.empty[File])
    files
      .filter(_.getName.toLowerCase.endsWith(".csv"))
      .filter(f => f.getName.toLowerCase.contains("scapula") && f.getName.toLowerCase.contains("model_data"))
      .filterNot(_.getName.toLowerCase.startsWith("single"))
      .sortBy(_.getName)
      .headOption
      .getOrElse(
        throw new RuntimeException(
          s"No landmark CSV found in ${dir.getPath}. Present: ${files.map(_.getName).mkString(", ")}"
        )
      )
  }

  private def normaliseHeader(h: String): String = h.trim.toLowerCase.replaceAll("[^a-z0-9]", "")

  /**
   * Resolve the (x, y, z) column indices for each landmark by matching the header, falling back to the hardcoded
   * offsets only if that fails. Returns the resolution together with a flag saying whether it came from the header, so
   * the caller can warn.
   */
  def resolveColumns(header: IndexedSeq[String]): (Map[String, (Int, Int, Int)], Boolean) = {
    val norm = header.map(normaliseHeader)

    def findAxis(lm: String, axis: Char): Option[Int] = {
      val l = lm.toLowerCase
      norm.indexWhere { h =>
        h == s"$l$axis" || h == s"${l}_$axis".replace("_", "") || (h.startsWith(l) && h.endsWith(axis.toString) &&
          h.length <= l.length + 2)
      } match {
        case -1 => None
        case i  => Some(i)
      }
    }

    val byName = landmarkNames.flatMap { lm =>
      for {
        xi <- findAxis(lm, 'x')
        yi <- findAxis(lm, 'y')
        zi <- findAxis(lm, 'z')
      } yield lm -> (xi, yi, zi)
    }.toMap

    if (byName.size == landmarkNames.size) (byName, true)
    else (fallbackStartIdx.map { case (lm, i) => lm -> (i, i + 1, i + 2) }, false)
  }

  def readLandmarkCsv(file: File): (Map[String, IndexedSeq[Landmark[_3D]]], Boolean, IndexedSeq[String]) = {
    val lines = Using.resource(Source.fromFile(file))(_.getLines().toIndexedSeq)
    require(lines.nonEmpty, s"Landmark CSV ${file.getName} is empty")

    val header = lines.head.split(",", -1).toIndexedSeq.map(_.trim)
    val (cols, fromHeader) = resolveColumns(header)

    val entries = lines.tail.filter(_.trim.nonEmpty).map { line =>
      val c = line.split(",", -1)
      val modelId = c(0).trim
      val lms = landmarkNames.map { nm =>
        val (xi, yi, zi) = cols(nm)
        require(
          c.length > math.max(xi, math.max(yi, zi)),
          s"Row '$modelId' has ${c.length} columns but landmark $nm needs column ${math.max(xi, math.max(yi, zi))}"
        )
        Landmark(nm, Point3D(c(xi).trim.toDouble, c(yi).trim.toDouble, c(zi).trim.toDouble))
      }
      modelId -> lms
    }.toMap

    (entries, fromHeader, header)
  }

  def specimens(dir: File): IndexedSeq[Specimen] = {
    Option(dir.listFiles())
      .getOrElse(Array.empty[File])
      .filter(_.getName.toLowerCase.endsWith(".stl"))
      .sortBy(_.getName)
      .toIndexedSeq
      .map { f =>
        val id = f.getName.stripSuffix(".stl")
        Specimen(id, f, id.endsWith("_R"), subjectKey(id))
      }
  }

  def subjectKey(modelId: String): String = modelId.stripSuffix("_L").stripSuffix("_R")

  // ---------------------------------------------------------------------------
  // Mirroring.
  //
  // NOTE ON METHOD: reflecting about the world plane x = 0 is not a special
  // choice.  A reflection about ANY plane differs from a reflection about x = 0
  // by a rigid motion, and a rigid alignment is applied immediately afterwards,
  // which absorbs exactly that difference.  What DOES matter is (a) that the map
  // is a genuine reflection (determinant -1) and (b) that the triangle winding is
  // flipped so surface normals keep pointing outwards.  Both are asserted below.
  // ---------------------------------------------------------------------------
  def mirrorPoint(p: Point[_3D]): Point[_3D] = Point3D(-p.x, p.y, p.z)

  def mirrorMesh(mesh: TriangleMesh[_3D]): TriangleMesh[_3D] = {
    val pts = mesh.pointSet.points.map(mirrorPoint).toIndexedSeq
    val flipped = mesh.triangulation.triangles.map(t => TriangleCell(t.ptId1, t.ptId3, t.ptId2))
    TriangleMesh3D(pts, TriangleList(flipped))
  }

  def mirrorLandmarks(lms: IndexedSeq[Landmark[_3D]]): IndexedSeq[Landmark[_3D]] =
    lms.map(lm => lm.copy(point = mirrorPoint(lm.point)))

  /**
   * Signed volume of a closed mesh (divergence theorem). Positive for a watertight, outward-oriented surface. Used to
   * confirm that mirroring + winding flip preserved orientation instead of turning the bone inside out.
   */
  def signedVolume(mesh: TriangleMesh[_3D]): Double = {
    var acc = 0.0
    mesh.triangulation.triangles.foreach { t =>
      val a = mesh.pointSet.point(t.ptId1).toVector
      val b = mesh.pointSet.point(t.ptId2).toVector
      val c = mesh.pointSet.point(t.ptId3).toVector
      acc += a.dot(b.crossproduct(c)) / 6.0
    }
    acc
  }

  def rigidFromLandmarks(from: IndexedSeq[Landmark[_3D]],
                         to: IndexedSeq[Landmark[_3D]]
  ): TranslationAfterRotation[_3D] =
    LandmarkRegistration.rigid3DLandmarkRegistration(from, to, center = Point3D(0, 0, 0))

  def perLandmarkDistances(a: IndexedSeq[Landmark[_3D]], b: IndexedSeq[Landmark[_3D]]): IndexedSeq[(String, Double)] = {
    val bById = b.map(l => l.id -> l.point).toMap
    a.flatMap(la => bById.get(la.id).map(pb => la.id -> (la.point - pb).norm))
  }

  def loadMesh(f: File): TriangleMesh[_3D] =
    MeshIO.readMesh(f).getOrElse(throw new RuntimeException(s"Could not read mesh ${f.getPath}"))
}

/** Surface-distance measures. Kept separate from MeshMetrics so the directionality is explicit. */
object Metrics {

  /** Distance from every vertex of `from` to the closest point on the SURFACE of `to`. */
  def surfaceDistances(from: TriangleMesh[_3D], to: TriangleMesh[_3D]): IndexedSeq[Double] = {
    val ops = to.operations
    from.pointSet.points.map(p => (p - ops.closestPointOnSurface(p).point).norm).toIndexedSeq
  }

  def percentile(values: IndexedSeq[Double], p: Double): Double = {
    require(values.nonEmpty)
    val sorted = values.sorted
    val idx = math.min(sorted.length - 1, math.max(0, math.ceil(p * sorted.length).toInt - 1))
    sorted(idx)
  }

  final case class SurfaceStats(mean: Double, rms: Double, hd95: Double, hd: Double, chamfer: Double) {
    def render: String = f"mean=$mean%5.2f  rms=$rms%5.2f  HD95=$hd95%5.2f  HD=$hd%6.2f  Chamfer=$chamfer%6.2f"
  }

  /**
   * Chamfer distance (mm^2): mean squared nearest-point distance in EACH direction, summed -- the standard
   * point-cloud/mesh-comparison definition. Deliberately NOT the same computation as `mean`/`rms` below, which pool
   * both directions' distances into one set before averaging; Chamfer keeps the two directions separate (so a
   * denser point set on one side doesn't implicitly get more weight in the combined average) and sums their
   * mean-squared values rather than pooling them.
   */
  def chamferDistance(a: TriangleMesh[_3D], b: TriangleMesh[_3D]): Double = {
    def meanSquared(from: TriangleMesh[_3D], to: TriangleMesh[_3D]): Double = {
      val d = surfaceDistances(from, to)
      d.map(x => x * x).sum / d.length
    }
    meanSquared(a, b) + meanSquared(b, a)
  }

  /** Symmetric statistics: both directions pooled, which is what "distance between two surfaces" should mean. */
  def symmetric(a: TriangleMesh[_3D], b: TriangleMesh[_3D]): SurfaceStats = {
    val d = surfaceDistances(a, b) ++ surfaceDistances(b, a)
    SurfaceStats(
      mean = d.sum / d.length,
      rms = math.sqrt(d.map(x => x * x).sum / d.length),
      hd95 = percentile(d, 0.95),
      hd = d.max,
      chamfer = chamferDistance(a, b)
    )
  }

  /** Point-to-point statistics for meshes that are already in correspondence (same point ids). */
  def correspondingDistances(a: TriangleMesh[_3D], b: TriangleMesh[_3D]): IndexedSeq[Double] = {
    require(a.pointSet.numberOfPoints == b.pointSet.numberOfPoints, "meshes are not in correspondence")
    a.pointSet.points.zip(b.pointSet.points).map { case (p, q) => (p - q).norm }.toIndexedSeq
  }
}
