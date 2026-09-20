package scapula

import scalismo.geometry.*
import scalismo.image.DiscreteImage
import scalismo.mesh.TriangleMesh
import scalismo.utils.Random

import java.io.File
import java.time.LocalDate

/**
 * Full Statistical Density Modeling pipeline for the scapula.
 *
 * Based on Cootes et al. "A combined active shape and mean appearance model" — the
 * density field (Hounsfield Units at correspondence vertices) plays the role of the
 * texture/appearance, and the pipeline produces a density model plus an optional
 * AAM-style combined shape+density model.
 *
 * ┌───────────────────────────────────────────────────────────────────────────┐
 * │  Stage 1 — DISCOVER & PREPROCESS                                         │
 * │    Load paired *_volume.nrrd / *.stl                                     │
 * │    Sample HU at each surface vertex (nearest-neighbour voxel lookup)     │
 * │    Quick HU sanity check (bone surface should average 300–1500 HU)       │
 * ├───────────────────────────────────────────────────────────────────────────┤
 * │  Stage 2 — RIGID ALIGNMENT                                               │
 * │    Trimmed ICP (uses existing RigidAlign object)                         │
 * │    HU values travel with their vertices through the transform            │
 * ├───────────────────────────────────────────────────────────────────────────┤
 * │  Stage 3 — TOPOLOGY TRANSFER                                             │
 * │    For every reference vertex, find the closest aligned-specimen vertex  │
 * │    and borrow its HU value — dense correspondence without GP             │
 * ├───────────────────────────────────────────────────────────────────────────┤
 * │  Stage 4 — STATISTICAL DENSITY MODEL                                     │
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
  // Stage 1 – Discover, load surface + volume, sample HU at mesh vertices
  // ---------------------------------------------------------------------------

  def stage1_preprocess(specimens: IndexedSeq[NrrdData.CtSpecimen])
  : IndexedSeq[(NrrdData.CtSpecimen, TriangleMesh[_3D], IndexedSeq[Float], DiscreteImage[_3D, Short])] = {

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

      (spec, mesh, hu, volume)
    }
  }

  // ---------------------------------------------------------------------------
  // Stage 2 – Rigid alignment
  //
  // Mesh vertex positions change; HU values are indexed by vertex id so they
  // travel with the mesh unchanged.
  // ---------------------------------------------------------------------------

  def stage2_rigidAlign(
    data: IndexedSeq[(NrrdData.CtSpecimen, TriangleMesh[_3D], IndexedSeq[Float], DiscreteImage[_3D, Short])]
  )(implicit rng: Random)
  : (TriangleMesh[_3D], IndexedSeq[(NrrdData.CtSpecimen, TriangleMesh[_3D], IndexedSeq[Float])]) = {

    println(s"\n[Stage 2] Rigid alignment")
    val (refSpec, refMesh, refHU, _) = data.head
    println(s"  Reference: ${refSpec.id}")

    val aligned = data.tail.map { case (spec, mesh, hu, _) =>
      val alignedMesh = RigidAlign.rigidIcp(mesh, refMesh, iterations = Config.icpIterations)
      val d = Metrics.symmetric(alignedMesh, refMesh)
      println(f"  ${spec.id} -> ${d.render}")
      (spec, alignedMesh, hu)
    }
    (refMesh, (refSpec, refMesh, refHU) +: aligned)
  }

  // ---------------------------------------------------------------------------
  // Stage 3 – Topology transfer to reference mesh
  //
  // For every vertex on the reference mesh, find the closest vertex on the
  // rigidly-aligned specimen mesh and borrow its HU value.  This gives a
  // common P-dimensional HU vector for every specimen without requiring GP
  // non-rigid registration.
  // ---------------------------------------------------------------------------

  def stage3_topologyTransfer(
    refMesh: TriangleMesh[_3D],
    aligned: IndexedSeq[(NrrdData.CtSpecimen, TriangleMesh[_3D], IndexedSeq[Float])]
  ): IndexedSeq[(String, IndexedSeq[Float])] = {

    println(s"\n[Stage 3] Topology transfer " +
            s"(${refMesh.pointSet.numberOfPoints} reference vertices)")

    // Raw per-specimen HU folder: one CSV per specimen in reference topology
    val rawHuDir = new File(densityOutDir, "raw_hu_per_specimen")
    rawHuDir.mkdirs()

    aligned.zipWithIndex.map { case ((spec, specMesh, specHU), idx) =>
      println(s"  [${idx + 1}/${aligned.length}] ${spec.id}")

      val mappedHU: IndexedSeq[Float] = refMesh.pointSet.points.map { refPt =>
        val closestId = specMesh.pointSet.findClosestPoint(refPt).id
        specHU(closestId.id)
      }.toIndexedSeq

      val (mn, mx, avg) = NrrdData.huStats(mappedHU)
      println(f"    mapped HU: min=$mn%.0f  max=$mx%.0f  mean=$avg%.0f")

      // Per-specimen CSV: (vertex_idx, x, y, z, HU) in reference space
      val outFile = new File(rawHuDir, s"${spec.id}_HU.csv")
      val pw = new java.io.PrintWriter(outFile)
      try {
        pw.println("vertex_idx,x,y,z,HU")
        refMesh.pointSet.points.toIndexedSeq.zipWithIndex.foreach { case (pt, j) =>
          pw.println(f"$j,${pt.x}%.3f,${pt.y}%.3f,${pt.z}%.3f,${mappedHU(j)}%.1f")
        }
      } finally pw.close()

      (spec.id, mappedHU)
    }
  }

  // ---------------------------------------------------------------------------
  // Stage 4 – Build and export the Statistical Density Model
  // ---------------------------------------------------------------------------

  def stage4_buildModel(
    refMesh: TriangleMesh[_3D],
    huFields: IndexedSeq[(String, IndexedSeq[Float])]
  ): StatisticalDensityModel.DensityModel = {

    println(s"\n[Stage 4] Building Statistical Density Model (AAM-style)")

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
      println("Expected: *_volume.nrrd + *.stl  (in the same directory)")
      sys.exit(1)
    }
    if (specimens.length < 3)
      println("\nWARNING: Fewer than 3 specimens — density model will have trivial statistics.")

    val preprocessed                = stage1_preprocess(specimens)
    val (refMesh, rigidAligned)     = stage2_rigidAlign(preprocessed)
    val huFields                    = stage3_topologyTransfer(refMesh, rigidAligned)
    val densityModel                = stage4_buildModel(refMesh, huFields)

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
    println("    raw_hu_per_specimen/<id>_HU.csv  ← Hounsfield Units at each reference vertex")
    println("    reference_mesh_meanHU.csv         ← mean HU mapped to reference mesh")
    println("    density_model.csv                 ← mean + PC1-5 at every vertex")
    println("    density_scores.csv                ← per-specimen latent scores")
    println()
    println("  Visualise: open reference_mesh_meanHU.csv in ParaView or 3D Slicer")
    println("  Joint shape+density model: StatisticalDensityModel.buildJoint(...)")
  }
}
