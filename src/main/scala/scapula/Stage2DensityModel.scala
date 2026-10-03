package scapula

import scalismo.geometry.*
import scalismo.mesh.*
import scalismo.image.{DiscreteImage, DiscreteImageDomain}
import scalismo.io.{ImageIO, MeshIO, StatismoIO}
import scalismo.statisticalmodel.{StatisticalMeshModel, PointDistributionModel}
import scalismo.utils.Random
import java.io.File

/**
 * Stage 2 — Statistical Density Model.
 *
 * PREREQUISITE: Stage 1 (rigid alignment) and a GP non-rigid registration
 * have already produced:
 *   - registeredMeshes(i)  : mesh of specimen i mapped to reference space,
 *                            same vertex count and topology as reference
 *   - rawMeshes(i)         : original (pre-alignment) mesh of specimen i,
 *                            in the SAME coordinate system as ct(i)
 *   - ct(i)                : CT image of specimen i (NIfTI, read by Scalismo),
 *                            in LPS or RAS — must match rawMeshes(i)
 *
 * WHAT THIS STAGE DOES:
 *   1. Create interior reference sample points (inside the reference mesh).
 *   2. For each specimen: use nearest-neighbour surface displacement to map
 *      those points from reference space → original CT space.
 *   3. Sample the CT at those positions → density (HU) vector per specimen.
 *   4. Apply calibration: HU → BMD (g/cm³).
 *   5. PCA on density vectors → Statistical Density Model (SDM).
 *   6. Combine shape vectors (from registered mesh) + density vectors
 *      → Joint PCA → SSDM (baseline, follows Sharif-Ahmadian [4]).
 *
 * NOTE ON COORDINATE SYSTEMS (read this or you will get garbage):
 *   rawMesh(i) and ct(i) MUST be in the same physical space.
 *   If you segmented the STL from the CT in ITK-SNAP / 3D Slicer, they are.
 *   If you exported from a different tool, verify by loading both in
 *   3D Slicer and checking that the mesh sits inside the bone in the CT.
 *   registeredMesh(i) is in reference space — a different frame entirely.
 *   The mapping between them is the NN displacement computed below.
 */
object Stage2DensityModel {

  // ── tuneable parameters ──────────────────────────────────────────────────
  // Interior sample grid density. Lower = coarser, faster; higher = finer model.
  val GridN: Int = 8            // points per axis ⟹ up to 8^3 = 512 candidates
  val MinDistFromSurface: Double = 1.5  // mm — exclude points too close to cortex
  val HuAir: Float = -1000f    // threshold: below this = air, not bone
  val HuBone: Float = 200f     // threshold: above this = bone (include only bone)

  // Calibration coefficients  BMD (g/cm³) = a * HU + b
  // Default: Keyak 1994 (for cancellous + cortical combined)
  val CalibA: Double = 0.000186
  val CalibB: Double = 0.1160

  // ── Step 1: create interior reference points ──────────────────────────────
  /**
   * Sample a grid inside the bounding box of ref, keep only points that are
   * geometrically inside the mesh (nearest-surface-normal test) and at least
   * MinDistFromSurface away from the cortex.
   *
   * Returns an IndexedSeq of interior Point[_3D] in reference space.
   */
  def interiorReferencePoints(ref: TriangleMesh[_3D]): IndexedSeq[Point[_3D]] = {
    val bb = ref.boundingBox
    val ops = ref.operations

    val candidates = for {
      i <- 0 until GridN
      j <- 0 until GridN
      k <- 0 until GridN
    } yield Point3D(
      bb.origin.x + i.toDouble / (GridN - 1) * bb.extent.x,
      bb.origin.y + j.toDouble / (GridN - 1) * bb.extent.y,
      bb.origin.z + k.toDouble / (GridN - 1) * bb.extent.z
    )

    candidates.filter { pt =>
      val closest = ops.closestPointOnSurface(pt)
      val dist    = (pt - closest.point).norm
      // A point is inside the mesh when the vector FROM the point TO the
      // surface points in the SAME direction as the outward surface normal.
      // (Outward normal dotted with toSurface > 0  ⟹  inside)
      val toSurface = closest.point - pt
      // Approximate normal at closest point via its cell normal.
      val cellId    = closest.id
      val cellNorm  = ref.cellNormals.atPoint(cellId)
      val inside    = toSurface.dot(cellNorm) > 0
      inside && dist >= MinDistFromSurface
    }.toIndexedSeq
  }

  // ── Step 2: map ref interior points to CT space using NN displacement ─────
  /**
   * For each interior reference point p:
   *   1. Find the nearest surface vertex v on the reference mesh.
   *   2. Get the displacement of v to the corresponding vertex on rawMesh
   *      (rawMesh has the same vertex topology as registeredMesh, just in the
   *       original CT coordinate frame).
   *   3. Apply that displacement to p → approximate CT-space coordinate.
   *
   * This is the nearest-neighbour method used by Fouefack et al.
   */
  def mapToCtSpace(
    interiorPts: IndexedSeq[Point[_3D]],   // in reference space
    referenceMesh: TriangleMesh[_3D],      // reference mesh, in reference space
    rawMesh: TriangleMesh[_3D]            // SAME topology, in CT space
  ): IndexedSeq[Point[_3D]] = {
    require(
      referenceMesh.pointSet.numberOfPoints == rawMesh.pointSet.numberOfPoints,
      "referenceMesh and rawMesh must have the same vertex count"
    )
    interiorPts.map { p =>
      // Nearest vertex on reference surface
      val nearestId = referenceMesh.pointSet.findClosestPoint(p).id
      val vRef      = referenceMesh.pointSet.point(nearestId)
      val vRaw      = rawMesh.pointSet.point(nearestId)
      // Displacement in reference space (approximate — valid for small offsets)
      val disp      = vRaw - vRef
      p + disp
    }
  }

  // ── Step 3: sample CT at a list of physical points ────────────────────────
  /**
   * Returns the CT intensity (HU) at each point, or None if the point is
   * outside the image domain.
   *
   * Uses nearest-neighbour lookup (fast, sufficient for density sampling).
   */
  def sampleCt(ct: DiscreteImage[_3D, Float],
               points: IndexedSeq[Point[_3D]]): IndexedSeq[Option[Float]] = {
    val dom = ct.domain
    points.map { pt =>
      if (dom.boundingBox.contains(pt)) {
        val idx = dom.pointToContinuousIndex(pt)
        // Round to nearest voxel
        val i = math.round(idx.i).toInt.max(0).min(dom.size.i - 1)
        val j = math.round(idx.j).toInt.max(0).min(dom.size.j - 1)
        val k = math.round(idx.k).toInt.max(0).min(dom.size.k - 1)
        Some(ct(scalismo.image.IntVector(i, j, k)))
      } else None
    }
  }

  // ── Step 4: calibrate HU → BMD ────────────────────────────────────────────
  def calibrate(hu: Double): Double = CalibA * hu + CalibB

  // ── Step 5: build density matrix ──────────────────────────────────────────
  /**
   * For each specimen: sample CT at NN-displaced interior reference points,
   * calibrate HU → BMD, return a vector of BMD values (one per interior point).
   *
   * Interior points for which CT lookup fails (outside domain) are filled with
   * the population mean across specimens at that point.  After PCA, these
   * fill-values add no variance, so they do not bias the model — but inspect
   * the fraction of missing values: if > 5 % of a specimen is missing, the
   * CT–mesh registration for that specimen is probably wrong.
   */
  def buildDensityMatrix(
    interiorRefPts: IndexedSeq[Point[_3D]],
    referenceMesh: TriangleMesh[_3D],
    rawMeshes: IndexedSeq[TriangleMesh[_3D]],
    ctImages: IndexedSeq[DiscreteImage[_3D, Float]]
  ): IndexedSeq[IndexedSeq[Double]] = {
    require(rawMeshes.length == ctImages.length)

    val N = rawMeshes.length
    val D = interiorRefPts.length

    // raw samples, None = outside domain
    val raw: IndexedSeq[IndexedSeq[Option[Double]]] = rawMeshes.zip(ctImages).map { case (raw, ct) =>
      val ctPts  = mapToCtSpace(interiorRefPts, referenceMesh, raw)
      val huVals = sampleCt(ct, ctPts)
      huVals.map(_.map(hu => calibrate(hu.toDouble)))
    }

    // per-point mean for fill
    val means: IndexedSeq[Double] = (0 until D).map { d =>
      val vals = raw.flatMap(_(d))
      if (vals.isEmpty) 0.0 else vals.sum / vals.length
    }

    // fill missing
    raw.map { specimen =>
      specimen.zipWithIndex.map { case (opt, d) => opt.getOrElse(means(d)) }
    }
  }

  // ── Step 6: PCA on density vectors ────────────────────────────────────────
  /**
   * Returns (mean density vector, modes matrix, eigenvalues).
   * Uses the economy (thin) SVD via breeze, which Scalismo already depends on.
   */
  def densityPca(
    densityMatrix: IndexedSeq[IndexedSeq[Double]]
  ): (IndexedSeq[Double], IndexedSeq[IndexedSeq[Double]], IndexedSeq[Double]) = {
    import breeze.linalg.*
    val N = densityMatrix.length
    val D = densityMatrix.head.length

    val mat = DenseMatrix.tabulate(N, D) { (i, j) => densityMatrix(i)(j) }

    // Mean-center
    val mean = DenseVector(mat.t(*, ::).map(col => col.sum / N).toArray)
    val centered = mat(*, ::).map(row => row - mean)

    // Economy SVD: centered = U * S * Vt, shape N × D
    val breeze.linalg.svd.SVD(u, s, vt) = svd.reduced(centered)

    // Principal components (rows of vt) and variances (s^2 / (N-1))
    val nModes = math.min(N - 1, D)
    val modes  = (0 until nModes).map(k => vt(k, ::).t.toArray.toIndexedSeq)
    val vars   = (0 until nModes).map(k => s(k) * s(k) / (N - 1))

    (mean.toArray.toIndexedSeq, modes, vars)
  }

  // ── Step 7: choose number of modes at variance threshold ──────────────────
  def modesForVariance(variances: IndexedSeq[Double], threshold: Double = 0.90): Int = {
    val total = variances.sum
    var acc   = 0.0
    var k     = 0
    while (acc < threshold * total && k < variances.length) {
      acc += variances(k)
      k   += 1
    }
    k
  }

  // ── Step 8: build joint shape+density model (baseline joint PCA) ──────────
  /**
   * Concatenates [shape_vector | w * density_vector] for each specimen then
   * runs PCA.  w is the shape-density weight (default = 1.0).
   *
   * shape_vector   = registered mesh point positions (3N_surface floats)
   * density_vector = BMD values at interior points  (D floats)
   *
   * This is the approach of Sharif-Ahmadian [4] and Beagley [12].
   * The returned model is an SSIM (statistical shape-intensity model).
   */
  def buildJointPca(
    registeredMeshes: IndexedSeq[TriangleMesh[_3D]],
    densityMatrix: IndexedSeq[IndexedSeq[Double]],
    densityWeight: Double = 1.0
  ): (IndexedSeq[Double], IndexedSeq[IndexedSeq[Double]], IndexedSeq[Double]) = {
    import breeze.linalg.*

    val N  = registeredMeshes.length
    val Ns = registeredMeshes.head.pointSet.numberOfPoints * 3  // shape DOF
    val Nd = densityMatrix.head.length                          // density DOF

    val mat = DenseMatrix.tabulate(N, Ns + Nd) { (i, j) =>
      if (j < Ns) {
        val pid  = j / 3
        val axis = j % 3
        registeredMeshes(i).pointSet.point(scalismo.common.PointId(pid)).toArray(axis)
      } else {
        densityWeight * densityMatrix(i)(j - Ns)
      }
    }

    val mean = DenseVector(mat.t(*, ::).map(col => col.sum / N).toArray)
    val centered = mat(*, ::).map(row => row - mean)

    val breeze.linalg.svd.SVD(_, s, vt) = svd.reduced(centered)
    val nModes = math.min(N - 1, Ns + Nd)
    val modes  = (0 until nModes).map(k => vt(k, ::).t.toArray.toIndexedSeq)
    val vars   = (0 until nModes).map(k => s(k) * s(k) / (N - 1))

    (mean.toArray.toIndexedSeq, modes, vars)
  }

  // ── main: wire everything together ────────────────────────────────────────
  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    // ── LOAD DATA ──
    // Adjust these paths to match your actual file organisation.
    // Each specimen needs three files:
    //   (a) rawMesh     : STL exported from CT segmentation, in CT/scanner space
    //   (b) regMesh     : STL after full GP registration, in reference space
    //   (c) ctImage     : NIfTI (.nii or .nii.gz) in the SAME space as rawMesh
    //
    // Naming convention assumed here:
    //   <dataDir>/raw/<id>.stl
    //   <dataDir>/registered/<id>.stl
    //   <dataDir>/ct/<id>.nii.gz

    val dataDir  = Config.dataDir
    val rawDir   = new File(dataDir, "raw")
    val regDir   = new File(dataDir, "registered")
    val ctDir    = new File(dataDir, "ct")

    val ids = rawDir.listFiles().filter(_.getName.endsWith(".stl")).map(_.getName.stripSuffix(".stl")).sorted.toIndexedSeq

    println(s"Found ${ids.length} specimens")

    val rawMeshes = ids.map(id => MeshIO.readMesh(new File(rawDir, s"$id.stl")).get)
    val regMeshes = ids.map(id => MeshIO.readMesh(new File(regDir, s"$id.stl")).get)
    val ctImages  = ids.map { id =>
      ImageIO.read3DScalarImage[Float](new File(ctDir, s"$id.nii.gz"))
        .getOrElse(throw new RuntimeException(s"Cannot read CT for $id"))
    }

    // ── LOAD REFERENCE MESH ──
    // This is the mesh every registered specimen was fitted to.
    val referenceMesh = MeshIO.readMesh(new File(dataDir, "reference.stl")).get

    // ── STEP 1: interior reference points ──
    println("Creating interior reference points …")
    val interiorPts = interiorReferencePoints(referenceMesh)
    println(s"  ${interiorPts.length} interior points created")

    // ── STEP 2–5: density matrix ──
    println("Sampling CT for each specimen …")
    val densityMatrix = buildDensityMatrix(interiorPts, referenceMesh, rawMeshes, ctImages)
    println(s"  Density matrix: ${densityMatrix.length} specimens × ${densityMatrix.head.length} points")

    // ── STEP 6: density PCA ──
    println("Running density PCA …")
    val (densityMean, densityModes, densityVars) = densityPca(densityMatrix)
    val k90 = modesForVariance(densityVars, 0.90)
    val k95 = modesForVariance(densityVars, 0.95)
    println(f"  Modes for 90%% variance: $k90   Modes for 95%%: $k95")

    // ── STEP 7: shape PCA (baseline comparison) ──
    println("Building shape model (SSM) …")
    val ssm = StatisticalMeshModel.createUsingPCA(regMeshes).get
    println(s"  SSM: ${ssm.rank} modes")

    // ── STEP 8: joint PCA (SSIM baseline) ──
    println("Building joint shape+density model (SSIM) …")
    val (_, _, jointVars) = buildJointPca(regMeshes, densityMatrix)
    val kJoint90 = modesForVariance(jointVars, 0.90)
    println(s"  Joint model: $kJoint90 modes for 90%% variance")

    // ── REPORT ──
    println()
    println("=" * 70)
    println("SUMMARY")
    println("=" * 70)
    println(f"  Shape model (SSM)           : ${ssm.rank} modes (all PCs)")
    println(f"  Density model (SDM)         : $k90 modes for 90%%  |  $k95 for 95%%")
    println(f"  Joint model (SSIM)          : $kJoint90 modes for 90%%")
    println()
    println("  If SDM needs far more modes than SSM, check:")
    println("  1. CT–mesh alignment (most common cause)")
    println("  2. Inconsistent HU calibration across scanners")
    println("  3. Small cohort inflating noise modes")
  }
}
