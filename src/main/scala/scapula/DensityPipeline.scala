package scapula

import breeze.linalg.{DenseMatrix, DenseVector}
import scalismo.common.{Field, RealSpace}
import scalismo.geometry.*
import scalismo.image.DiscreteImage
import scalismo.kernels.{DiagonalKernel, GaussianKernel}
import scalismo.mesh.TriangleMesh
import scalismo.numerics.UniformMeshSampler3D
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, MultivariateNormalDistribution, PointDistributionModel}
import scalismo.utils.Random

import java.io.File
import java.time.LocalDate

/**
 * Full Statistical Density Modeling pipeline for the scapula.
 *
 * Based on Cootes et al. "A combined active shape and mean appearance model" — the
 * density field (Hounsfield Units at correspondence vertices) plays the role of the
 * texture/appearance, and the pipeline produces both a shape model and a density model,
 * plus an optional joint AAM-style combined model.
 *
 * ┌───────────────────────────────────────────────────────────────────────────┐
 * │  Stage 1 — DISCOVER & PREPROCESS                                         │
 * │    Load paired *_volume.nrrd / *_scapula_0.seg.nrrd                     │
 * │    Surface extraction (STL if present, else Marching Cubes)             │
 * │    Quick HU sanity check (bone surface should average 300–1500 HU)      │
 * ├───────────────────────────────────────────────────────────────────────────┤
 * │  Stage 2 — RIGID ALIGNMENT                                               │
 * │    Trimmed ICP (uses existing RigidAlign object)                         │
 * ├───────────────────────────────────────────────────────────────────────────┤
 * │  Stage 3 — GP NON-RIGID REGISTRATION                                     │
 * │    Gaussian-process deformation prior → posterior ICP correspondence      │
 * │    Produces reference-topology meshes for all specimens                  │
 * ├───────────────────────────────────────────────────────────────────────────┤
 * │  Stage 4 — HU FIELD SAMPLING                                             │
 * │    Sample HU at registered reference vertices from each specimen's CT    │
 * │    Saves one <id>_HU.csv per specimen in raw_hu_per_specimen/            │
 * ├───────────────────────────────────────────────────────────────────────────┤
 * │  Stage 5 — STATISTICAL DENSITY MODEL                                     │
 * │    PCA on N × P HU matrix → density model                               │
 * │    Optional joint shape+density model (AAM-style)                        │
 * │    Exports CSV files for downstream analysis / visualisation             │
 * └───────────────────────────────────────────────────────────────────────────┘
 */
object DensityPipeline {

  // ---------------------------------------------------------------------------
  // Config
  // ---------------------------------------------------------------------------

  val nrrdDir: File = new File(
    sys.env.getOrElse("SCAPULA_NRRD_DIR", "/home/user/Documents/armcortnet_output_B3")
  )

  // Default output folder is date-stamped so each run is distinct.
  private val runDate: String = LocalDate.now().toString
  val densityOutDir: File = new File(
    sys.env.getOrElse(
      "SCAPULA_DENSITY_OUT",
      s"/home/user/Documents/scapula_density_sdm_$runDate"
    )
  )

  // ---------------------------------------------------------------------------
  // Stage 1 – Discover & preprocess
  // ---------------------------------------------------------------------------

  def stage1_preprocess(specimens: IndexedSeq[NrrdData.CtSpecimen])
  : IndexedSeq[(NrrdData.CtSpecimen, TriangleMesh[_3D], DiscreteImage[_3D, Short])] = {

    println(s"\n[Stage 1] Preprocessing ${specimens.length} CT specimens")
    specimens.zipWithIndex.map { case (spec, idx) =>
      println(s"  [${idx + 1}/${specimens.length}] ${spec.id}")

      val volume = NrrdData.loadVolume(spec.volumeFile)
      val mesh   = NrrdData.extractSurface(spec)
      println(f"    surface: ${mesh.pointSet.numberOfPoints} vertices, " +
              f"${mesh.triangulation.triangles.length} triangles")

      val hu = NrrdData.sampleHU(mesh, volume)
      val (minHU, maxHU, meanHU) = NrrdData.huStats(hu)
      val cFrac = NrrdData.corticalMask(hu).count(identity).toDouble / hu.length
      println(f"    HU: min=$minHU%.0f  max=$maxHU%.0f  mean=$meanHU%.0f  cortical(>300HU)=${cFrac * 100}%.1f%%")
      if (meanHU < 50f || meanHU > 2000f)
        println(s"    !! Unusual mean HU for ${spec.id} — check segmentation / CT units")

      (spec, mesh, volume)
    }
  }

  // ---------------------------------------------------------------------------
  // Stage 2 – Rigid alignment
  // ---------------------------------------------------------------------------

  def stage2_rigidAlign(
    data: IndexedSeq[(NrrdData.CtSpecimen, TriangleMesh[_3D], DiscreteImage[_3D, Short])]
  )(implicit rng: Random)
  : (TriangleMesh[_3D], IndexedSeq[(NrrdData.CtSpecimen, TriangleMesh[_3D], DiscreteImage[_3D, Short])]) = {

    println(s"\n[Stage 2] Rigid alignment")
    val (refSpec, refMesh, refVol) = data.head
    println(s"  Reference: ${refSpec.id}")

    val aligned = data.tail.map { case (spec, mesh, vol) =>
      val alignedMesh = RigidAlign.rigidIcp(mesh, refMesh, iterations = Config.icpIterations)
      val d = Metrics.symmetric(alignedMesh, refMesh)
      println(f"  ${spec.id} -> ${d.render}")
      (spec, alignedMesh, vol)
    }
    (refMesh, (refSpec, refMesh, refVol) +: aligned)
  }

  // ---------------------------------------------------------------------------
  // Stage 3 – GP non-rigid registration
  //
  // Builds a low-rank Gaussian-process deformation prior on the reference mesh,
  // then fits each target via ICP in deformation space.
  // ---------------------------------------------------------------------------

  def stage3_nonrigidRegister(
    refMesh: TriangleMesh[_3D],
    aligned: IndexedSeq[(NrrdData.CtSpecimen, TriangleMesh[_3D], DiscreteImage[_3D, Short])]
  )(implicit rng: Random)
  : IndexedSeq[(NrrdData.CtSpecimen, TriangleMesh[_3D], DiscreteImage[_3D, Short])] = {

    println(s"\n[Stage 3] GP non-rigid registration")

    // Two-scale Gaussian kernel: large scale captures global shape variation,
    // small scale captures local surface detail.
    val kernel =
      DiagonalKernel[_3D](GaussianKernel[_3D](sigma = 50.0) * 50.0, outputDim = 3) +
      DiagonalKernel[_3D](GaussianKernel[_3D](sigma = 10.0) * 10.0, outputDim = 3)

    // Zero-mean GP deformation field over R³
    val zeroMean = Field[_3D, EuclideanVector[_3D]](RealSpace[_3D])(_ => EuclideanVector(0.0, 0.0, 0.0))
    val gp       = GaussianProcess(zeroMean, kernel)

    // Nystrom approximation sampled uniformly on the reference mesh
    val sampler   = UniformMeshSampler3D(refMesh, numberOfPoints = 500)
    val lowRankGP = LowRankGaussianProcess.approximateGPNystrom(gp, sampler, numBasisFunctions = Config.gpMaxRank)
    val model     = PointDistributionModel[_3D, TriangleMesh](refMesh, lowRankGP)
    println(s"  GP rank: ${lowRankGP.rank}")

    aligned.zipWithIndex.map { case ((spec, targetMesh, vol), idx) =>
      println(s"  [${idx + 1}/${aligned.length}] ${spec.id}")
      val fittedMesh = gpIcp(model, targetMesh, iterations = Config.icpIterations)
      val d = Metrics.symmetric(fittedMesh, targetMesh)
      println(f"    registration residual: ${d.render}")
      (spec, fittedMesh, vol)
    }
  }

  /**
   * GP-ICP: iterative closest-point in GP deformation space.
   *
   * Each iteration samples points on the current best-fit mesh, finds their
   * closest counterparts on the target surface, then computes the posterior
   * mean of the shape model given those correspondences.
   */
  private def gpIcp(
    model: PointDistributionModel[_3D, TriangleMesh],
    target: TriangleMesh[_3D],
    iterations: Int,
    numPoints: Int = 500,
    noiseStdDev: Double = 1.0
  )(implicit rng: Random): TriangleMesh[_3D] = {

    // Isotropic Gaussian noise on correspondences (in mm)
    val noiseDist = MultivariateNormalDistribution(
      DenseVector.zeros[Double](3),
      DenseMatrix.eye[Double](3) * (noiseStdDev * noiseStdDev)
    )

    var current   = model.mean
    val targetOps = target.operations

    for (_ <- 0 until iterations) {
      val sampled = UniformMeshSampler3D(current, numPoints).sample().map(_._1)
      val obs = sampled.map { pt =>
        val closest = targetOps.closestPointOnSurface(pt).point
        val id      = current.pointSet.findClosestPoint(pt).id
        (id, closest, noiseDist)
      }
      if (obs.nonEmpty) {
        val posterior = model.posterior(obs)
        current = posterior.mean
      }
    }
    current
  }

  // ---------------------------------------------------------------------------
  // Stage 4 – HU field sampling at reference vertices
  // ---------------------------------------------------------------------------

  def stage4_sampleHU(
    refMesh: TriangleMesh[_3D],
    registered: IndexedSeq[(NrrdData.CtSpecimen, TriangleMesh[_3D], DiscreteImage[_3D, Short])]
  ): IndexedSeq[(String, IndexedSeq[Float])] = {

    println(s"\n[Stage 4] Sampling HU at ${refMesh.pointSet.numberOfPoints} reference vertices")

    // Raw per-specimen HU folder: one CSV per specimen
    val rawHuDir = new File(densityOutDir, "raw_hu_per_specimen")
    rawHuDir.mkdirs()

    registered.zipWithIndex.map { case ((spec, regMesh, volume), idx) =>
      val hu = NrrdData.sampleHU(regMesh, volume)
      val (mn, mx, avg) = NrrdData.huStats(hu)
      println(f"  [${idx + 1}/${registered.length}] ${spec.id}  HU: min=$mn%.0f max=$mx%.0f mean=$avg%.0f")

      // ── Per-specimen raw HU CSV (vertex_idx, x, y, z, HU) ──────────────────
      val outFile = new File(rawHuDir, s"${spec.id}_HU.csv")
      val pw = new java.io.PrintWriter(outFile)
      try {
        pw.println("vertex_idx,x,y,z,HU")
        regMesh.pointSet.points.toIndexedSeq.zipWithIndex.foreach { case (pt, j) =>
          pw.println(f"$j,${pt.x}%.3f,${pt.y}%.3f,${pt.z}%.3f,${hu(j)}%.1f")
        }
      } finally pw.close()

      (spec.id, hu)
    }
  }

  // ---------------------------------------------------------------------------
  // Stage 5 – Build and export the Statistical Density Model
  // ---------------------------------------------------------------------------

  def stage5_buildModel(
    refMesh: TriangleMesh[_3D],
    huFields: IndexedSeq[(String, IndexedSeq[Float])],
    registered: IndexedSeq[(NrrdData.CtSpecimen, TriangleMesh[_3D], DiscreteImage[_3D, Short])]
  ): StatisticalDensityModel.DensityModel = {

    println(s"\n[Stage 5] Building Statistical Density Model  (AAM-style)")

    val ids    = huFields.map(_._1)
    val fields = huFields.map(_._2)

    val model = StatisticalDensityModel.build(ids, fields, nComp = math.min(ids.length - 1, 50))
    StatisticalDensityModel.printSummary(model)

    densityOutDir.mkdirs()
    StatisticalDensityModel.exportCsv(model, new File(densityOutDir, "density_model.csv"))

    // Per-specimen latent scores CSV
    val scoresFile = new File(densityOutDir, "density_scores.csv")
    val pw = new java.io.PrintWriter(scoresFile)
    try {
      val nComp = math.min(10, model.k)
      pw.println("id," + (1 to nComp).map(c => s"PC$c").mkString(","))
      ids.zipWithIndex.foreach { case (id, i) =>
        val s = model.scores(i, nComp)
        pw.println(id + "," + s.toArray.map(v => f"$v%.4f").mkString(","))
      }
    } finally pw.close()
    println(s"  Per-specimen scores: ${scoresFile.getAbsolutePath}")

    // Mean HU mapped to reference mesh vertices (open in ParaView / 3D Slicer)
    val meanCsv = new File(densityOutDir, "reference_mesh_meanHU.csv")
    val mw = new java.io.PrintWriter(meanCsv)
    try {
      mw.println("vertex_idx,x,y,z,mean_HU")
      refMesh.pointSet.points.toIndexedSeq.zipWithIndex.foreach { case (pt, j) =>
        mw.println(f"$j,${pt.x}%.3f,${pt.y}%.3f,${pt.z}%.3f,${model.meanHU(j)}%.2f")
      }
    } finally mw.close()
    println(s"  Mean HU mesh:        ${meanCsv.getAbsolutePath}")

    model
  }

  // ---------------------------------------------------------------------------
  // Main
  // ---------------------------------------------------------------------------

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    println("=" * 80)
    println("Statistical Density Modeling Pipeline — Scapula (AAM-style)")
    println("=" * 80)
    println(s"NRRD input  : ${nrrdDir.getAbsolutePath}")
    println(s"Output      : ${densityOutDir.getAbsolutePath}")

    val specimens = NrrdData.discoverSpecimens(nrrdDir)
    println(s"\nFound ${specimens.length} paired specimens:")
    specimens.foreach(s => println(s"  ${s.id}"))

    if (specimens.isEmpty) {
      println("\nERROR: No specimens found. Check SCAPULA_NRRD_DIR.")
      println("Expected: *_volume.nrrd + *_scapula_0.seg.nrrd")
      sys.exit(1)
    }
    if (specimens.length < 3)
      println("\nWARNING: Fewer than 3 specimens — density model will have trivial statistics.")

    val preprocessed                 = stage1_preprocess(specimens)
    val (refMesh, rigidAligned)      = stage2_rigidAlign(preprocessed)
    val registered                   = stage3_nonrigidRegister(refMesh, rigidAligned)
    val huFields                     = stage4_sampleHU(refMesh, registered)
    val densityModel                 = stage5_buildModel(refMesh, huFields, registered)

    println()
    println("=" * 80)
    println("COMPLETE")
    println("=" * 80)
    println(s"  Specimens : ${specimens.length}")
    println(s"  Model rank: ${densityModel.k}")
    println(f"  PC1-5 variance explained: ${densityModel.explainedVarianceRatio(5) * 100}%.1f%%")
    println(s"  Output dir: ${densityOutDir.getAbsolutePath}")
    println()
    println("  Output files:")
    println("    raw_hu_per_specimen/<id>_HU.csv  ← raw Hounsfield Units at each vertex")
    println("    reference_mesh_meanHU.csv         ← mean HU mapped to reference mesh")
    println("    density_model.csv                 ← mean + PC1-5 at every vertex")
    println("    density_scores.csv                ← per-specimen latent scores")
    println()
    println("  Visualise: open reference_mesh_meanHU.csv in ParaView or 3D Slicer")
    println("  Joint shape+density model: StatisticalDensityModel.buildJoint(...)")
  }
}
