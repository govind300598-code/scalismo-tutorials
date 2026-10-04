package scapula

import breeze.linalg.*
import scalismo.common.PointId
import scalismo.geometry.*
import scalismo.image.{DiscreteImage, IntVector}
import scalismo.io.{ImageIO, MeshIO, StatismoIO}
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.{MultivariateNormalDistribution, StatisticalMeshModel}
import scalismo.utils.Random

import java.io.File

/**
 * Stage 2 — Statistical Shape and Density Model (SSDM).
 *
 * ════════════════════════════════════════════════════════════════════════════
 *  HOW THESE FILES CONNECT (read this first)
 * ════════════════════════════════════════════════════════════════════════════
 *
 *  You uploaded five files from Scalismo's Active Shape Model (ASM) package:
 *
 *    SearchPointSampler  — samples points along the surface normal direction
 *    FeatureExtractor    — reads CT intensity/gradient along those points
 *    Profiles            — stores the distribution of features at each vertex
 *    ActiveShapeModel    — the full ASM training + fitting pipeline
 *    IOHandler           — HDF5 serialisation framework
 *
 *  An ASM = SSM + intensity profiles along the surface normal, used to
 *  SEGMENT a new CT scan (find the bone boundary in an unseen image).
 *
 *  An SSDM = SSM + DENSITY profiles along the inward surface normal, used to
 *  PREDICT bone density for biomechanical simulation.
 *
 *  The two pipelines are IDENTICAL structurally.  Only the meaning of the
 *  "feature" changes:
 *
 *    ASM feature  = CT gradient or intensity along the normal
 *                   → used to find the bone surface in a new scan
 *    SSDM feature = calibrated bone mineral density (BMD) along the inward normal
 *                   → used to build the density model and predict density
 *
 *  Concretely:
 *    SearchPointSampler.apply(mesh, ptId)       ←→  Step 2 below
 *    FeatureExtractor.apply(image, pt, mesh, id) ←→  Step 3 below
 *    MultivariateNormalDistribution.estimateFromData ←→  Step 4 below
 *    Profiles(...)                               ←→  the SDM itself
 *
 *  The `IOHandler` files (ncsa.hdf) are for the old HDF5 serialisation format;
 *  in Scalismo 0.92 you save/load models with `StatismoIO` instead.
 *
 * ════════════════════════════════════════════════════════════════════════════
 *  PREREQUISITES
 * ════════════════════════════════════════════════════════════════════════════
 *
 *  For each specimen you need THREE files in matching coordinate spaces:
 *    rawMesh   — STL exported from CT segmentation software, in LPS/RAS
 *    ct        — NIfTI CT image, in the EXACT same physical space as rawMesh
 *    regMesh   — result of GP registration, in reference space
 *
 *  rawMesh and ct MUST be in the same space.  If you segmented the STL from
 *  the CT in ITK-SNAP or 3D Slicer they are.  Verify in 3D Slicer: load both
 *  and the mesh must sit exactly inside the bone.
 *
 *  regMesh and rawMesh MUST have the same vertex count and vertex IDs; only
 *  the positions differ (different coordinate frames).  This is guaranteed if
 *  you built the SSM by deforming the reference mesh to each specimen.
 *
 * ════════════════════════════════════════════════════════════════════════════
 *  THE NORMAL-DIRECTION SAMPLING APPROACH
 * ════════════════════════════════════════════════════════════════════════════
 *
 *  This is EXACTLY what NormalDirectionSearchPointSampler does:
 *
 *    for each surface vertex v:
 *      compute the inward unit normal  n = -normalize(vertexNormal(v))
 *      for k = 1 to ProfileDepth:
 *        samplePt = v + n * (k * ProfileSpacing_mm)
 *        hu       = CT value at samplePt
 *        bmd      = calibrate(hu)
 *
 *  This gives ProfileDepth density values per vertex, forming a "density
 *  profile" — the same concept as the ASM intensity profile.
 *
 *  Interior correspondence is AUTOMATIC: corresponding vertex IDs on rawMesh
 *  across specimens give corresponding profiles.  No GridN, no NN-displacement
 *  mapping needed.
 */
object Stage2DensityModel {

  // ── tuneable parameters ─────────────────────────────────────────────────
  val ProfileDepth: Int = 8        // sample points along inward normal per vertex
  val ProfileSpacing: Double = 1.5 // mm between consecutive profile points

  // Calibration: BMD (g/cm³) = CalibA × HU + CalibB  (Keyak 1994)
  val CalibA: Double = 0.000186
  val CalibB: Double = 0.1160

  def calibrate(hu: Double): Double = CalibA * hu + CalibB

  // ── Step 1: check that rawMesh and regMesh are topology-compatible ──────
  def requireSameTopology(raw: TriangleMesh[_3D], reg: TriangleMesh[_3D]): Unit =
    require(
      raw.pointSet.numberOfPoints == reg.pointSet.numberOfPoints,
      s"rawMesh (${raw.pointSet.numberOfPoints} pts) and regMesh " +
        s"(${reg.pointSet.numberOfPoints} pts) must have the same vertex count. " +
        "Did you export the rawMesh from the same reference topology?"
    )

  // ── Step 2 + 3: normal-direction density profile at every vertex ────────
  //
  // This is the SSDM equivalent of NormalDirectionFeatureExtractor.apply().
  // rawMesh is in CT space so we sample the CT directly at the profile points.
  //
  // Returns IndexedSeq[IndexedSeq[Double]]:
  //   outer index = vertex id  (same as rawMesh.pointSet.pointId)
  //   inner index = profile depth k  (k=0 → 1*spacing from surface, k=D-1 → D*spacing)
  def densityProfiles(rawMesh: TriangleMesh[_3D],
                      ct: DiscreteImage[_3D, Float]): IndexedSeq[IndexedSeq[Double]] = {
    val nPts = rawMesh.pointSet.numberOfPoints
    val dom  = ct.domain
    val bb   = dom.boundingBox

    (0 until nPts).map { idx =>
      val ptId   = PointId(idx)
      val pt     = rawMesh.pointSet.point(ptId)
      val nRaw   = rawMesh.vertexNormals(ptId)
      val nNorm  = nRaw.norm
      if (nNorm < 1e-10) return IndexedSeq.fill(ProfileDepth)(0.0) // degenerate normal
      val inward = nRaw * (-1.0 / nNorm) // inward = pointing INTO bone

      (1 to ProfileDepth).map { k =>
        val samplePt = pt + inward * (k * ProfileSpacing)
        if (!bb.contains(samplePt)) 0.0
        else {
          // Nearest-voxel lookup (fast; sufficient for density at 1–12 mm scale)
          val ci = dom.pointToContinuousIndex(samplePt)
          val i  = ci.i.round.toInt.max(0).min(dom.size.i - 1)
          val j  = ci.j.round.toInt.max(0).min(dom.size.j - 1)
          val kk = ci.k.round.toInt.max(0).min(dom.size.k - 1)
          calibrate(ct(IntVector(i, j, kk)).toDouble)
        }
      }.toIndexedSeq
    }
  }

  // ── Step 4: flatten profiles → density vector per specimen ─────────────
  //
  // densityVector(i) has length  N_vertices × ProfileDepth.
  // Layout: [v0_d1, v0_d2, …, v0_dD, v1_d1, …, vN_dD]
  // This is the vector the PCA runs on — analogous to the shape vector.
  def densityVector(profiles: IndexedSeq[IndexedSeq[Double]]): IndexedSeq[Double] =
    profiles.flatten

  // ── Step 5: build the density matrix [N_specimens × D_total] ───────────
  def buildDensityMatrix(rawMeshes: IndexedSeq[TriangleMesh[_3D]],
                         ctImages: IndexedSeq[DiscreteImage[_3D, Float]]): IndexedSeq[IndexedSeq[Double]] = {
    require(rawMeshes.length == ctImages.length)
    rawMeshes.zip(ctImages).map { case (mesh, ct) =>
      densityVector(densityProfiles(mesh, ct))
    }
  }

  // ── Step 6: validate — check fraction of zero/out-of-domain values ─────
  def validateDensityMatrix(matrix: IndexedSeq[IndexedSeq[Double]], ids: IndexedSeq[String]): Unit = {
    matrix.zip(ids).foreach { case (vec, id) =>
      val zeroes = vec.count(_ == 0.0)
      val pct    = 100.0 * zeroes / vec.length
      if (pct > 5.0)
        println(f"  !! $id: $pct%.1f%% of profile points outside CT domain. " +
          "Check CT–mesh coordinate alignment for this specimen.")
    }
  }

  // ── Step 7: PCA on density vectors ─────────────────────────────────────
  //
  // This is the same operation that MultivariateNormalDistribution.estimateFromData
  // does in ActiveShapeModel.trainModel, but run globally across ALL vertices
  // at once to build the population density model.
  case class PcaResult(mean: DenseVector[Double],
                       modes: DenseMatrix[Double],  // rows = modes
                       variances: DenseVector[Double])

  def runPca(matrix: IndexedSeq[IndexedSeq[Double]]): PcaResult = {
    val N = matrix.length
    val D = matrix.head.length
    val mat = DenseMatrix.tabulate(N, D) { (i, j) => matrix(i)(j) }

    val mean     = DenseVector(mat(*, ::).map(_.sum / N).toArray) // per-column mean
    val centered = mat(*, ::).map { row => row - mean }

    val svd.SVD(_, s, vt) = svd.reduced(centered)
    val nModes   = math.min(N - 1, D)
    val modes    = vt(0 until nModes, ::)  // nModes × D
    val variances = DenseVector(s.toArray.take(nModes).map(v => v * v / (N - 1)))

    PcaResult(mean, modes, variances)
  }

  def modesForVariance(variances: DenseVector[Double], threshold: Double = 0.90): Int = {
    val total = variances.sum
    var acc   = 0.0
    var k     = 0
    while (acc < threshold * total && k < variances.length) {
      acc += variances(k)
      k   += 1
    }
    k
  }

  // ── Step 8: per-vertex density distribution (Profiles equivalent) ───────
  //
  // This is what Profiles/MultivariateNormalDistribution stores in the ASM.
  // For each surface vertex, fit a multivariate normal to the ProfileDepth-dim
  // density profile across all specimens.
  // Useful for: visualising per-region density variability, partial-model fitting.
  case class VertexDensityProfile(pointId: PointId,
                                  distribution: MultivariateNormalDistribution)

  def buildVertexProfiles(rawMeshes: IndexedSeq[TriangleMesh[_3D]],
                          ctImages: IndexedSeq[DiscreteImage[_3D, Float]]): IndexedSeq[VertexDensityProfile] = {
    val allProfiles: IndexedSeq[IndexedSeq[IndexedSeq[Double]]] =
      rawMeshes.zip(ctImages).map { case (m, ct) => densityProfiles(m, ct) }

    val nVertices = rawMeshes.head.pointSet.numberOfPoints
    (0 until nVertices).map { vIdx =>
      // profilesForVertex(i) = ProfileDepth-dim density profile of specimen i at vertex vIdx
      val profilesForVertex: IndexedSeq[DenseVector[Double]] =
        allProfiles.map { specProfiles =>
          DenseVector(specProfiles(vIdx).toArray)
        }
      val dist = MultivariateNormalDistribution.estimateFromData(profilesForVertex)
      VertexDensityProfile(PointId(vIdx), dist)
    }
  }

  // ── Step 9: joint shape + density PCA (SSIM baseline) ──────────────────
  //
  // Concatenates [shape_vector | density_vector] for each specimen, runs PCA.
  // This is the approach of Sharif-Ahmadian [4] and Beagley [12].
  def buildJointPca(regMeshes: IndexedSeq[TriangleMesh[_3D]],
                    densityMatrix: IndexedSeq[IndexedSeq[Double]],
                    densityWeight: Double = 1.0): PcaResult = {
    val N  = regMeshes.length
    val Ns = regMeshes.head.pointSet.numberOfPoints * 3
    val Nd = densityMatrix.head.length

    val mat = DenseMatrix.tabulate(N, Ns + Nd) { (i, j) =>
      if (j < Ns) {
        val pid  = PointId(j / 3)
        val axis = j % 3
        regMeshes(i).pointSet.point(pid).toArray(axis)
      } else {
        densityWeight * densityMatrix(i)(j - Ns)
      }
    }

    val mean     = DenseVector(mat(*, ::).map(_.sum / N).toArray)
    val centered = mat(*, ::).map { row => row - mean }
    val svd.SVD(_, s, vt) = svd.reduced(centered)
    val nModes   = math.min(N - 1, Ns + Nd)
    val variances = DenseVector(s.toArray.take(nModes).map(v => v * v / (N - 1)))

    PcaResult(mean, vt(0 until nModes, ::), variances)
  }

  // ── main: wire everything together ─────────────────────────────────────
  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dataDir = Config.dataDir
    val rawDir  = new File(dataDir, "raw")        // STLs in CT space
    val regDir  = new File(dataDir, "registered") // STLs after GP registration
    val ctDir   = new File(dataDir, "ct")         // NIfTI CT images

    val ids = rawDir.listFiles()
      .filter(_.getName.endsWith(".stl"))
      .map(_.getName.stripSuffix(".stl"))
      .sorted.toIndexedSeq

    println(s"Found ${ids.length} specimens")

    val rawMeshes = ids.map(id => MeshIO.readMesh(new File(rawDir, s"$id.stl")).get)
    val regMeshes = ids.map(id => MeshIO.readMesh(new File(regDir, s"$id.stl")).get)
    val ctImages  = ids.map { id =>
      ImageIO.read3DScalarImage[Float](new File(ctDir, s"$id.nii.gz"))
        .getOrElse(throw new RuntimeException(s"Cannot read CT for $id"))
    }

    // Sanity-check topology
    rawMeshes.zip(regMeshes).foreach { case (r, g) => requireSameTopology(r, g) }

    // Step 4–5: build density matrix using normal-direction profiles
    println("Sampling density profiles along inward surface normals …")
    val densityMatrix = buildDensityMatrix(rawMeshes, ctImages)
    val D = densityMatrix.head.length
    println(s"  Matrix: ${ids.length} specimens × $D density values per specimen")
    println(s"  ($D = ${rawMeshes.head.pointSet.numberOfPoints} vertices × $ProfileDepth profile depths)")

    // Step 6: validate CT–mesh alignment
    validateDensityMatrix(densityMatrix, ids)

    // Step 7: density PCA
    println("Running density PCA …")
    val sdm = runPca(densityMatrix)
    val k90 = modesForVariance(sdm.variances, 0.90)
    val k95 = modesForVariance(sdm.variances, 0.95)
    println(f"  SDM: $k90 modes for 90%%  |  $k95 for 95%% variance")

    // SSM for comparison
    println("Building SSM …")
    val ssm   = StatisticalMeshModel.createUsingPCA(regMeshes).get
    val ssmK90 = {
      val v = ssm.gp.klBasis.map(_.eigenvalue)
      val total = v.sum
      var acc = 0.0; var k = 0
      while (acc < 0.90 * total && k < v.length) { acc += v(k); k += 1 }
      k
    }
    println(s"  SSM: $ssmK90 modes for 90%% variance")

    // Step 9: joint SSIM
    println("Building joint SSIM …")
    val ssim = buildJointPca(regMeshes, densityMatrix)
    val ssimK90 = modesForVariance(ssim.variances, 0.90)
    println(s"  SSIM: $ssimK90 modes for 90%% variance")

    // Report
    println()
    println("=" * 70)
    println("SUMMARY")
    println("=" * 70)
    println(f"  SSM  (shape only)        : $ssmK90 modes for 90%% variance")
    println(f"  SDM  (density only)      : $k90 modes for 90%%  |  $k95 for 95%%")
    println(f"  SSIM (shape + density)   : $ssimK90 modes for 90%% variance")
    println()
    if (k90 > 3 * ssmK90)
      println("  WARNING: SDM needs many more modes than SSM. Most likely causes:")
      println("   1. CT images from different scanners with different HU scales")
      println("      → you need a calibration phantom or scanner-specific correction")
      println("   2. CT and mesh are not in the same coordinate space for some specimens")
      println("      → check the validation warnings above")
      println("   3. Very small cohort — noise dominates the covariance")
    else
      println("  SDM mode count looks reasonable relative to SSM. Proceed to GPMM.")
  }
}
