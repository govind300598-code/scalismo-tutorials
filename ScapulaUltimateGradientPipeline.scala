//> using scala 3.3.3
//> using dep ch.unibas.cs.gravis::scalismo:0.92.1
//> using dep ch.unibas.cs.gravis::scalismo-ui:0.92.1
//> using javaOpt -Xmx10G

import scalismo.geometry.*
import scalismo.common.*
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.mesh.*
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.statisticalmodel.*
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.numerics.{FixedPointsUniformMeshSampler3D, UniformMeshSampler3D}
import scalismo.registration.*
import scalismo.kernels.*
import breeze.linalg.DenseVector
import java.io.File
import scala.io.{Source, StdIn}
import scala.util.Using
import scalismo.utils.Random
import scalismo.ui.api.*

// =============================================================================
// ScapulaUltimateGradientPipeline
//
// Full 3-stage pipeline:
//   Stage 1 – Landmark Procrustes + trimmed rigid ICP
//   Stage 2 – GPMM non-rigid registration via distance-image gradient optimisation
//   Stage 3 – SSM construction via PCA + modes-of-variation export
//
// Run:  scala-cli run ScapulaUltimateGradientPipeline.scala
//
// Paths are resolved from environment variables so the binary runs unchanged
// against a different dataset:
//   SCAPULA_DATA_DIR  – directory containing *.stl files and the landmark CSV
//   SCAPULA_OUT_DIR   – root for all outputs (created if absent)
// =============================================================================
object ScapulaUltimateGradientPipeline extends App {

  scalismo.initialize()
  implicit val rng: Random = Random(42L)

  // ---------------------------------------------------------------------------
  // PATHS
  // ---------------------------------------------------------------------------
  val home    = System.getProperty("user.home")
  val dataDir = new File(
    sys.env.getOrElse("SCAPULA_DATA_DIR",
      s"$home/Documents/database_v1.11/paired_scapulae_STLs"))
  val outRoot = new File(
    sys.env.getOrElse("SCAPULA_OUT_DIR",
      s"$home/Documents/database_v1.11/scapula_ssm_out"))

  val rigidDir   = new File(outRoot, "rigid_aligned")
  val correspDir = new File(outRoot, "correspondences")
  val modesDir   = new File(outRoot, "modes_of_variation")
  Seq(outRoot, rigidDir, correspDir, modesDir).foreach(_.mkdirs())
  val ssmFile = new File(outRoot, "scapula_ssm.h5")

  require(dataDir.exists(),
    s"Data directory not found: ${dataDir.getAbsolutePath}\n" +
    s"Set SCAPULA_DATA_DIR to the folder containing the STL files and landmark CSV.")

  // ---------------------------------------------------------------------------
  // LANDMARK CSV
  // The CSV must be in dataDir; it is identified by containing both "scapula"
  // and "model_data" in its name (ignoring case) and not starting with "single".
  // Column positions are resolved by matching header names so the code is
  // robust to column reordering; it falls back to the hardcoded offsets below
  // only when the header cannot be matched by name.
  // ---------------------------------------------------------------------------
  val landmarkNames  = IndexedSeq("GC", "TS", "IA", "PLA", "AC")
  val fallbackColIdx = Map("GC" -> 11, "TS" -> 14, "IA" -> 17, "PLA" -> 20, "AC" -> 23)

  val csvFile: File = Option(dataDir.listFiles()).getOrElse(Array.empty[File])
    .filter { f =>
      val n = f.getName.toLowerCase
      n.endsWith(".csv") && n.contains("scapula") && n.contains("model_data") && !n.startsWith("single")
    }
    .sortBy(_.getName)
    .headOption
    .getOrElse(throw new RuntimeException(
      s"No landmark CSV found in ${dataDir.getPath}.\n" +
      s"Files present: ${Option(dataDir.listFiles()).getOrElse(Array.empty).map(_.getName).mkString(", ")}"))

  println(s"Landmark CSV : ${csvFile.getAbsolutePath}")

  def readLandmarkCsv(file: File): Map[String, IndexedSeq[Landmark[_3D]]] = {
    val lines  = Using.resource(Source.fromFile(file))(_.getLines().toIndexedSeq)
    require(lines.nonEmpty, s"CSV ${file.getName} is empty")
    val header = lines.head.split(",", -1).map(_.trim.toLowerCase.replaceAll("[^a-z0-9]", "")).toIndexedSeq

    def colOf(lm: String, axis: Char): Int = {
      val l = lm.toLowerCase
      val i = header.indexWhere { h =>
        h == s"$l$axis" ||
        (h.startsWith(l) && h.endsWith(axis.toString) && h.length <= l.length + 2)
      }
      if (i >= 0) i
      else fallbackColIdx(lm) + "xyz".indexOf(axis)
    }

    val colMap = landmarkNames.map(nm =>
      nm -> (colOf(nm, 'x'), colOf(nm, 'y'), colOf(nm, 'z'))
    ).toMap

    lines.tail.filter(_.trim.nonEmpty).map { line =>
      val c  = line.split(",", -1)
      val id = c(0).trim
      val lms = landmarkNames.map { nm =>
        val (xi, yi, zi) = colMap(nm)
        require(c.length > Seq(xi, yi, zi).max,
          s"Row '$id' has ${c.length} columns but landmark $nm needs column ${Seq(xi, yi, zi).max}")
        Landmark(nm, Point3D(c(xi).trim.toDouble, c(yi).trim.toDouble, c(zi).trim.toDouble))
      }
      id -> lms
    }.toMap
  }

  val allLandmarks = readLandmarkCsv(csvFile)
  println(s"Loaded ${allLandmarks.size} entries, ${landmarkNames.size} landmarks each (${landmarkNames.mkString(", ")})")

  // ---------------------------------------------------------------------------
  // REFERENCE MESH
  // The reference is decimated to modelResolution vertices so that the GP
  // approximation and the SSM both live at a manageable resolution.
  // Writing and re-reading canonicalises topology (removes duplicate vertices)
  // which can otherwise cause NaN in the Cholesky factorisation.
  // ---------------------------------------------------------------------------
  val modelResolution = 5000
  val refId   = "paired_scapula_001_M_64_L"
  val refFile = new File(dataDir, s"$refId.stl")
  require(refFile.exists(), s"Reference mesh not found: ${refFile.getAbsolutePath}")
  require(allLandmarks.contains(refId),
    s"Reference '$refId' has no row in the landmark CSV.")

  val rawRef      = MeshIO.readMesh(refFile).getOrElse(throw new RuntimeException(s"Cannot read $refFile"))
  val decimated   = rawRef.operations.decimate(modelResolution)
  val referenceMesh: TriangleMesh[_3D] = {
    val tmp = File.createTempFile("ref_clean", ".stl")
    try { MeshIO.writeMesh(decimated, tmp); MeshIO.readMesh(tmp).get }
    finally { tmp.delete() }
  }
  val refLandmarks = allLandmarks(refId)
  println(s"Reference    : $refId | ${referenceMesh.pointSet.numberOfPoints} vertices (decimated from ${rawRef.pointSet.numberOfPoints})")

  // ---------------------------------------------------------------------------
  // MIRROR UTILITIES
  // Right-side scapulae are the mirror image of left-side ones. Reflecting
  // about x = 0 is equivalent to any other reflection up to a rigid motion,
  // which the subsequent Procrustes step absorbs.  What matters is that the
  // reflection has determinant -1 AND that triangle winding is flipped so
  // surface normals remain outward.
  // ---------------------------------------------------------------------------
  def mirrorPoint(p: Point[_3D]): Point[_3D] = Point3D(-p.x, p.y, p.z)

  def mirrorMesh(m: TriangleMesh[_3D]): TriangleMesh[_3D] = {
    val pts     = m.pointSet.points.map(mirrorPoint).toIndexedSeq
    val flipped = m.triangulation.triangles.map(t => TriangleCell(t.ptId1, t.ptId3, t.ptId2))
    TriangleMesh3D(pts, TriangleList(flipped))
  }

  def mirrorLandmarks(lms: IndexedSeq[Landmark[_3D]]): IndexedSeq[Landmark[_3D]] =
    lms.map(lm => lm.copy(point = mirrorPoint(lm.point)))

  // ---------------------------------------------------------------------------
  // STAGE 1 — RIGID ALIGNMENT
  // Step A: landmark Procrustes (closed-form, no tuning required)
  // Step B: trimmed rigid ICP (the worst trimFraction of correspondences is
  //         dropped each iteration, preventing thin structures from dragging
  //         the pose estimate)
  // ---------------------------------------------------------------------------
  def uniformSampleIds(mesh: TriangleMesh[_3D], n: Int): IndexedSeq[PointId] = {
    val k = math.min(n, mesh.pointSet.numberOfPoints)
    UniformMeshSampler3D(mesh, k).sample()
      .map { case (pt, _) => mesh.pointSet.findClosestPoint(pt).id }
      .distinct
  }

  def trimmedRigidIcp(
    moving:       TriangleMesh[_3D],
    target:       TriangleMesh[_3D],
    iterations:   Int    = 30,
    nPoints:      Int    = 2000,
    trimFraction: Double = 0.15
  ): TriangleMesh[_3D] = {
    val ops = target.operations
    val ids = uniformSampleIds(moving, nPoints)
    var cur = moving
    for (_ <- 0 until iterations) {
      val pairs = ids.map { id =>
        val p = cur.pointSet.point(id)
        (p, ops.closestPointOnSurface(p).point)
      }
      val keep    = math.max(10, (pairs.length * (1.0 - trimFraction)).toInt)
      val trimmed = pairs.sortBy { case (m, t) => (m - t).norm }.take(keep)
      val trans   = LandmarkRegistration.rigid3DLandmarkRegistration(trimmed, center = Point3D(0, 0, 0))
      cur = cur.transform(trans)
    }
    cur
  }

  case class RigidResult(
    name:        String,
    wasMirrored: Boolean,
    lmErrorMm:   Double,
    mesh:        TriangleMesh[_3D]
  )

  def rigidAlignOne(file: File): RigidResult = {
    val name    = file.getName.stripSuffix(".stl")
    val isRight = name.endsWith("_R")
    val rawLms  = allLandmarks.getOrElse(name,
      throw new RuntimeException(s"No CSV row for model '$name'. Check that IDs match exactly."))
    val rawMesh = MeshIO.readMesh(file).getOrElse(
      throw new RuntimeException(s"Cannot read mesh ${file.getPath}"))

    // Mirror R-side to L-handedness before rigid alignment.
    val (wMesh, wLms) =
      if (isRight) (mirrorMesh(rawMesh), mirrorLandmarks(rawLms)) else (rawMesh, rawLms)

    // Step A: landmark Procrustes.
    val lmTrans    = LandmarkRegistration.rigid3DLandmarkRegistration(wLms, refLandmarks, center = Point3D(0, 0, 0))
    val lmAligned  = wMesh.transform(lmTrans)
    val alignedLms = wLms.map(lm => lm.copy(point = lmTrans(lm.point)))
    val lmError    = refLandmarks.zip(alignedLms).map { case (r, a) => (r.point - a.point).norm }.sum /
                       refLandmarks.size.toDouble

    // Step B: trimmed rigid ICP.
    val finalMesh = trimmedRigidIcp(lmAligned, referenceMesh)

    MeshIO.writeMesh(finalMesh, new File(rigidDir, s"rigid_$name.stl"))
    println(f"  [RIGID] $name%-32s | mirror=$isRight%-5s | LM err: $lmError%5.2f mm")
    RigidResult(name, isRight, lmError, finalMesh)
  }

  val allStlFiles = Option(dataDir.listFiles()).getOrElse(Array.empty[File])
    .filter(f => f.getName.endsWith(".stl") && f.getName != refFile.getName)
    .sortBy(_.getName)
    .toIndexedSeq

  println(s"\n[STAGE 1] Rigid alignment (landmark Procrustes + trimmed ICP) for ${allStlFiles.length} meshes\n")
  val rigidResults = allStlFiles.map(rigidAlignOne)

  val lSide = rigidResults.filterNot(_.wasMirrored)
  val rSide = rigidResults.filter(_.wasMirrored)
  if (lSide.nonEmpty) println(f"  L-side LM mean: ${lSide.map(_.lmErrorMm).sum / lSide.size}%.2f mm (n=${lSide.size})")
  if (rSide.nonEmpty) println(f"  R-side LM mean: ${rSide.map(_.lmErrorMm).sum / rSide.size}%.2f mm (n=${rSide.size})")
  println(s"  Rigid STLs saved to: ${rigidDir.getAbsolutePath}")

  // ---------------------------------------------------------------------------
  // STAGE 2 — GPMM NON-RIGID REGISTRATION
  //
  // Kernel design follows Dennis Madsen's recommendation for a scapula of
  // ~150 mm on its longest side:
  //   coarse kernel : sigma = 75 mm, scale = 15  – large-scale bending
  //   mid    kernel : sigma = 30 mm, scale = 10  – medium-scale shape
  //   fine   kernel : sigma = 15 mm, scale = 5   – local detail
  //
  // NearestNeighborInterpolator is used instead of TriangleMeshInterpolator to
  // avoid tearing artefacts at the boundary of the decimated reference.
  //
  // The fitted mesh is taken as gpmm.instance(finalCoeffs) – a pure GPMM
  // shape – not the projection of the instance onto the target surface.
  // Projecting to the target causes triangle tearing and connectivity breaks.
  // ---------------------------------------------------------------------------
  val kernelCoarse = GaussianKernel3D(150.0 / 2.0, 15.0)
  val kernelMid    = GaussianKernel3D(150.0 / 5.0, 10.0)
  val kernelFine   = GaussianKernel3D(150.0 / 10.0, 5.0)
  val kernel       = DiagonalKernel3D(kernelCoarse + kernelMid + kernelFine, outputDim = 3)

  val gp = GaussianProcess3D[EuclideanVector[_3D]](kernel)
  val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
    referenceMesh, gp,
    relativeTolerance = 0.01,
    interpolator = NearestNeighborInterpolator3D()
  )
  val gpmm = PointDistributionModel3D(referenceMesh, lowRankGP)
  println(s"\n[STAGE 2] GPMM prior: ${gpmm.rank} basis functions on ${referenceMesh.pointSet.numberOfPoints} vertices")

  // Annealing schedule: start with high regularisation + few sample points;
  // progressively tighten both to fit finer-scale details.
  case class RegParams(regularizationWeight: Double, iterations: Int, sampledPoints: Int)
  val schedule = Seq(
    RegParams(1e-1, 50,  100),
    RegParams(1e-2, 50,  500),
    RegParams(1e-4, 50, 1000),
    RegParams(1e-6, 50, 5000)
  )

  def doRegistration(
    targetMesh:    TriangleMesh[_3D],
    initialCoeffs: DenseVector[Double],
    params:        RegParams
  ): DenseVector[Double] = {
    val tSpace   = GaussianProcessTransformationSpace(lowRankGP)
    val fixed    = referenceMesh.operations.toDistanceImage
    val moving   = targetMesh.operations.toDistanceImage
    val sampler  = FixedPointsUniformMeshSampler3D(referenceMesh, params.sampledPoints)
    val metric   = MeanSquaresMetric(fixed, moving, tSpace, sampler)
    val optim    = LBFGSOptimizer(params.iterations)
    val regul    = L2Regularizer(tSpace)
    val reg      = Registration(metric, regul, params.regularizationWeight, optim)
    reg.iterator(initialCoeffs).toSeq.last.parameters
  }

  println(s"\n[STAGE 2] Non-rigid GPMM registration for ${rigidResults.length} meshes\n")

  val zero = DenseVector.zeros[Double](gpmm.rank)

  // The reference itself is its own correspondent (identity = zero coefficients).
  // Including it in the SSM prevents the mean from drifting off the reference.
  val selfCorresp: TriangleMesh[_3D] = referenceMesh

  val nonRigidCorresps: IndexedSeq[TriangleMesh[_3D]] = rigidResults.zipWithIndex.map { case (r, idx) =>
    val finalCoeffs = schedule.foldLeft(zero)((c, p) => doRegistration(r.mesh, c, p))
    val fitted      = gpmm.instance(finalCoeffs)
    MeshIO.writeMesh(fitted, new File(correspDir, s"corresp_${r.name}.stl"))
    println(f"  [GPMM] [${idx + 1}%3d/${rigidResults.size}] ${r.name}")
    fitted
  }

  val allCorresps: IndexedSeq[TriangleMesh[_3D]] = selfCorresp +: nonRigidCorresps
  println(s"  Correspondences saved to: ${correspDir.getAbsolutePath}")

  // ---------------------------------------------------------------------------
  // STAGE 3 — STATISTICAL SHAPE MODEL VIA PCA
  // ---------------------------------------------------------------------------
  println(s"\n[STAGE 3] Building SSM from ${allCorresps.size} correspondences...")
  val dc  = DataCollection.fromTriangleMesh3DSequence(referenceMesh, allCorresps)
  val ssm = PointDistributionModel.createUsingPCA(dc)
  StatisticalModelIO.writeStatisticalTriangleMeshModel3D(ssm, ssmFile)
  println(s"  SSM saved : ${ssmFile.getAbsolutePath}")
  println(s"  SSM rank  : ${ssm.rank}")

  // Eigenvalue summary (variance explained per mode).
  val eigenvalues = ssm.gp.klBasis.map(_.eigenvalue)
  val totalVar    = eigenvalues.sum
  println("  Variance explained by first modes:")
  eigenvalues.take(math.min(5, ssm.rank)).zipWithIndex.foreach { case (v, i) =>
    val cumulative = eigenvalues.take(i + 1).sum / totalVar * 100.0
    println(f"    Mode ${i + 1}: variance=${v}%.4f  cumulative=${cumulative}%.1f%%")
  }

  // Export mean + ±2SD / ±3SD for first 3 modes.
  MeshIO.writeMesh(ssm.mean, new File(modesDir, "Mode_Mean.stl"))
  for (m <- 0 until math.min(3, ssm.rank); sd <- Seq(-3.0, -2.0, 2.0, 3.0)) {
    val coeff = DenseVector.zeros[Double](ssm.rank)
    coeff(m) = sd
    val sdStr = if (sd > 0) f"+${sd.toInt}SD" else f"${sd.toInt}SD"
    MeshIO.writeMesh(ssm.instance(coeff), new File(modesDir, f"Mode_${m + 1}_$sdStr.stl"))
  }
  println(s"  Modes of variation saved to: ${modesDir.getAbsolutePath}")

  println(s"\n[DONE] All outputs are in: ${outRoot.getAbsolutePath}")
  println(  "       scapula_ssm.h5        – the statistical shape model")
  println(  "       rigid_aligned/         – STLs after landmark + ICP alignment")
  println(  "       correspondences/       – STLs in point-to-point correspondence")
  println(  "       modes_of_variation/    – mean ± 2SD / ± 3SD for first 3 modes")

  // ---------------------------------------------------------------------------
  // OPTIONAL VIEWER
  // ---------------------------------------------------------------------------
  println("\nLaunching Scalismo-UI viewer...")
  val ui    = ScalismoUI()
  val grp   = ui.createGroup("ScapulaSSM_Pipeline")

  ui.show(grp, referenceMesh,  s"REFERENCE_$refId")
  ui.show(grp, refLandmarks,   "REFERENCE_landmarks")
  ui.show(grp, ssm,            "SSM")

  println("Viewer open. Press ENTER in this terminal to exit.")
  StdIn.readLine()
}
