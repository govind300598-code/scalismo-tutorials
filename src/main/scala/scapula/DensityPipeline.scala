package scapula

import scalismo.geometry.*
import scalismo.image.DiscreteImage
import scalismo.kernels.{DiagonalKernel, GaussianKernel}
import scalismo.mesh.TriangleMesh
import scalismo.numerics.UniformMeshSampler3D
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel}
import scalismo.utils.Random

import java.io.File

/**
 * Full Statistical Density Modeling pipeline for the scapula.
 *
 * ┌───────────────────────────────────────────────────────────────────────────┐
 * │  Stage 1 — DISCOVER & PREPROCESS                                         │
 * │    Load paired *_volume.nrrd / *_scapula_0.seg.nrrd                     │
 * │    Marching-Cubes surface extraction → TriangleMesh per specimen          │
 * │    Quick HU sanity check (bone HU should be 300–2000)                   │
 * ├───────────────────────────────────────────────────────────────────────────┤
 * │  Stage 2 — RIGID ALIGNMENT                                               │
 * │    Trimmed landmark + ICP alignment (uses existing RigidAlign object)    │
 * │    Mirror right → left so all bones are in the same handedness           │
 * ├───────────────────────────────────────────────────────────────────────────┤
 * │  Stage 3 — GP NON-RIGID REGISTRATION                                     │
 * │    Build a low-rank Gaussian-process deformation model on the reference  │
 * │    ICP in deformation space → point-to-point correspondence              │
 * │    Warp each specimen onto the reference topology                        │
 * ├───────────────────────────────────────────────────────────────────────────┤
 * │  Stage 4 — HU FIELD SAMPLING                                             │
 * │    For each specimen: sample HU at *reference* vertex positions using    │
 * │    the inverse warp + the specimen's own CT volume                       │
 * ├───────────────────────────────────────────────────────────────────────────┤
 * │  Stage 5 — STATISTICAL DENSITY MODEL                                     │
 * │    PCA on N × P matrix of HU values                                      │
 * │    Optional: joint shape + density model                                 │
 * │    Export: CSV of mean + components, per-specimen HU scores              │
 * └───────────────────────────────────────────────────────────────────────────┘
 */
object DensityPipeline {

  // ---------------------------------------------------------------------------
  // Extended config for NRRD data
  // ---------------------------------------------------------------------------

  /** Directory that contains the *_volume.nrrd and *_scapula_0.seg.nrrd files. */
  val nrrdDir: File = new File(
    sys.env.getOrElse("SCAPULA_NRRD_DIR", "/home/user/Documents/armcortnet_output_B3")
  )

  val densityOutDir: File = new File(
    sys.env.getOrElse("SCAPULA_DENSITY_OUT", "/home/user/Documents/scapula_density_out")
  )

  // ---------------------------------------------------------------------------
  // Stage 1 – Discover & preprocess
  // ---------------------------------------------------------------------------

  def stage1_preprocess(specimens: IndexedSeq[NrrdData.CtSpecimen])
  : IndexedSeq[(NrrdData.CtSpecimen, TriangleMesh[_3D], DiscreteImage[_3D, Short])] = {

    println(s"\n[Stage 1] Preprocessing ${specimens.length} CT specimens")
    specimens.zipWithIndex.map { case (spec, idx) =>
      println(s"  [${idx + 1}/${specimens.length}] ${spec.id}")

      val seg    = NrrdData.loadSegmentation(spec.segFile)
      val volume = NrrdData.loadVolume(spec.volumeFile)

      val mesh = NrrdData.extractSurface(seg)
      println(f"    surface: ${mesh.pointSet.numberOfPoints} vertices, ${mesh.triangulation.triangles.length} triangles")

      // Sanity-check HU distribution at the bone surface
      val hu = NrrdData.sampleHU(mesh, volume)
      val (minHU, maxHU, meanHU) = NrrdData.huStats(hu)
      val corticalFrac = NrrdData.corticalMask(hu).count(identity).toDouble / hu.length
      println(f"    HU at surface: min=$minHU%.0f  max=$maxHU%.0f  mean=$meanHU%.0f  cortical(>300HU)=${corticalFrac * 100}%.1f%%")
      if (meanHU < 100f || meanHU > 1800f)
        println(s"    !! Unusual mean HU – check segmentation orientation / units for ${spec.id}")

      (spec, mesh, volume)
    }
  }

  // ---------------------------------------------------------------------------
  // Stage 2 – Rigid alignment (reference = first specimen)
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
  // We build a low-rank Gaussian-process deformation prior on the reference mesh
  // and fit it to each target by iterative closest point in deformation space.
  // The result is a warp: reference → target that establishes dense correspondence.
  // ---------------------------------------------------------------------------

  def stage3_nonrigidRegister(
    refMesh: TriangleMesh[_3D],
    aligned: IndexedSeq[(NrrdData.CtSpecimen, TriangleMesh[_3D], DiscreteImage[_3D, Short])]
  )(implicit rng: Random)
  : IndexedSeq[(NrrdData.CtSpecimen, TriangleMesh[_3D], DiscreteImage[_3D, Short])] = {

    println(s"\n[Stage 3] GP non-rigid registration")

    // Build a Gaussian deformation prior: sum of two length scales for
    // large (coarse) and small (fine) deformations typical of bone surfaces.
    val kernel = DiagonalKernel[_3D](GaussianKernel[_3D](sigma = 50.0) * 50.0, outputDim = 3) +
                 DiagonalKernel[_3D](GaussianKernel[_3D](sigma = 10.0) * 10.0, outputDim = 3)

    val gp = GaussianProcess[_3D, EuclideanVector[_3D]](kernel)
    val lowRankGP = LowRankGaussianProcess.approximateGPNystrom(
      gp, refMesh.pointSet, numBasisFunctions = Config.gpMaxRank
    )
    val model = PointDistributionModel[_3D, TriangleMesh](refMesh, lowRankGP)

    println(s"  GP rank: ${lowRankGP.rank}")

    aligned.zipWithIndex.map { case ((spec, targetMesh, vol), idx) =>
      println(s"  [${idx + 1}/${aligned.length}] ${spec.id}")

      // ICP in deformation-model space: fit the model to the target
      val fittedMesh = gpIcp(model, targetMesh, iterations = Config.icpIterations)
      val d = Metrics.symmetric(fittedMesh, targetMesh)
      println(f"    registration residual: ${d.render}")
      (spec, fittedMesh, vol)
    }
  }

  /**
   * GP-ICP: iteratively find the posterior of the shape model that best explains
   * the target surface, producing a warp of the reference onto the target topology.
   *
   * Each iteration:
   *   1. Sample points on the current best-fit mesh
   *   2. Find closest points on the target surface
   *   3. Compute the GP posterior constrained to those correspondences
   *   4. Take the posterior mean as the new candidate
   */
  private def gpIcp(
    model: PointDistributionModel[_3D, TriangleMesh],
    target: TriangleMesh[_3D],
    iterations: Int,
    numPoints: Int = 500,
    noise: Double = 1.0
  )(implicit rng: Random): TriangleMesh[_3D] = {
    var current = model.mean
    val targetOps = target.operations

    for (_ <- 0 until iterations) {
      val sampled = UniformMeshSampler3D(current, numPoints).sample().map(_._1)
      val obs = sampled.flatMap { pt =>
        val closest = targetOps.closestPointOnSurface(pt).point
        val id = current.pointSet.findClosestPoint(pt).id
        Some((id, closest))
      }
      if (obs.nonEmpty) {
        // PointDistributionModel.posterior expects Seq[(PointId, Point[_3D], Double)]
        // where the Double is the isotropic noise standard deviation.
        val trainingData = obs.map { case (id, tgt) => (id, tgt, noise) }
        val posterior = model.posterior(trainingData)
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
    println("  Strategy: map each reference vertex through the specimen's registered warp,")
    println("  then query the specimen's own CT volume at that warped position.")

    registered.zipWithIndex.map { case ((spec, regMesh, volume), idx) =>
      // regMesh is the reference topology deformed to match the specimen.
      // Its vertices are in the specimen's (rigidly aligned) coordinate frame.
      // We query the CT volume at those positions directly.
      val hu = NrrdData.sampleHU(regMesh, volume)
      val (mn, mx, avg) = NrrdData.huStats(hu)
      println(f"  [${idx + 1}/${registered.length}] ${spec.id}  HU: min=$mn%.0f max=$mx%.0f mean=$avg%.0f")
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

    println(s"\n[Stage 5] Building Statistical Density Model")

    val ids    = huFields.map(_._1)
    val fields = huFields.map(_._2)

    val model = StatisticalDensityModel.build(ids, fields, nComp = math.min(ids.length - 1, 50))
    StatisticalDensityModel.printSummary(model)

    densityOutDir.mkdirs()
    StatisticalDensityModel.exportCsv(model, new File(densityOutDir, "density_model.csv"))

    // Per-specimen scores
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
    println(s"  Per-specimen scores written to: ${scoresFile.getAbsolutePath}")

    // Also export mean HU field as a coloured mesh (values stored as scalar field)
    // Requires writing a VTK or PLY; here we write a simple per-vertex CSV.
    val meanHuMeshCsv = new File(densityOutDir, "reference_mesh_meanHU.csv")
    val mw = new java.io.PrintWriter(meanHuMeshCsv)
    try {
      mw.println("vertex_idx,x,y,z,mean_HU")
      refMesh.pointSet.points.toIndexedSeq.zipWithIndex.foreach { case (pt, j) =>
        mw.println(f"$j,${pt.x}%.3f,${pt.y}%.3f,${pt.z}%.3f,${model.meanHU(j)}%.2f")
      }
    } finally mw.close()
    println(s"  Mean HU mesh written to: ${meanHuMeshCsv.getAbsolutePath}")

    model
  }

  // ---------------------------------------------------------------------------
  // Main entry point
  // ---------------------------------------------------------------------------

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    println("=" * 80)
    println("Statistical Density Modeling Pipeline for Scapula")
    println("=" * 80)
    println(s"NRRD data directory : ${nrrdDir.getAbsolutePath}")
    println(s"Output directory    : ${densityOutDir.getAbsolutePath}")

    // --- Discover paired NRRD files ---
    val specimens = NrrdData.discoverSpecimens(nrrdDir)
    println(s"\nFound ${specimens.length} paired specimens:")
    specimens.foreach(s => println(s"  ${s.id}"))

    if (specimens.isEmpty) {
      println("\nERROR: No specimens found. Check SCAPULA_NRRD_DIR points to the correct folder.")
      println(s"Expected files: *_volume.nrrd + *_scapula_0.seg.nrrd")
      sys.exit(1)
    }

    if (specimens.length < 3) {
      println("\nWARNING: Fewer than 3 specimens — the density model will have trivial statistics.")
    }

    // --- Run stages ---
    val preprocessed = stage1_preprocess(specimens)

    val (refMesh, rigidAligned) = stage2_rigidAlign(preprocessed)

    val registered = stage3_nonrigidRegister(refMesh, rigidAligned)

    val huFields = stage4_sampleHU(refMesh, registered)

    val densityModel = stage5_buildModel(refMesh, huFields, registered)

    // --- Summary ---
    println()
    println("=" * 80)
    println("PIPELINE COMPLETE")
    println("=" * 80)
    println(s"  Specimens processed : ${specimens.length}")
    println(f"  Model rank          : ${densityModel.k}")
    println(f"  Variance (PC1-5)    : ${densityModel.explainedVarianceRatio(5) * 100}%.1f%%")
    println(s"  Results in          : ${densityOutDir.getAbsolutePath}")
    println()
    println("  Outputs:")
    println("    density_model.csv          – mean HU + principal components at each vertex")
    println("    density_scores.csv         – per-specimen scores on each PC")
    println("    reference_mesh_meanHU.csv  – reference mesh vertices with mean HU values")
    println()
    println("  Next steps:")
    println("    • Open reference_mesh_meanHU.csv in ParaView or 3D Slicer to visualise bone density")
    println("    • Use density_scores.csv for downstream statistics (age/sex regression, pathology detection)")
    println("    • Build a joint shape+density model with StatisticalDensityModel.buildJoint()")
  }
}
