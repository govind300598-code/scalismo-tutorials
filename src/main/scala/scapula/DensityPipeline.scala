package scapula

import scalismo.geometry.*
import scalismo.image.DiscreteImage
import scalismo.mesh.TriangleMesh
import scalismo.common.ScalarMeshField3D
import scalismo.io.MeshIO
import scalismo.utils.Random

import java.io.File
import java.time.LocalDate

/**
 * Full Statistical Density Modeling pipeline for the scapula (AAM-style).
 *
 * Input layout
 * ────────────
 *   SCAPULA_NRRD_DIR  (default: ~/Documents/all ct from hoel 3d)
 *     SH_01256_volume.nrrd
 *     SH_01256_scapula_0.seg.nrrd
 *     …
 *
 *   SCAPULA_STL_DIR   (default: NRRD_DIR/stl/both stl/combined 1/Combined)
 *     SH_01256_scapula_0_Segment_2.stl   ← left scapula
 *     SH_01256_scapula_0_Segment_3.stl   ← right scapula
 *     …
 *
 * Each STL file is one specimen.  Both segments of a subject share the
 * same CT volume, so HU is sampled from that volume for each surface.
 *
 * Pipeline stages
 * ───────────────
 *   Stage 1 – Load every (volume, STL) pair; sample HU at surface vertices
 *   Stage 2 – Rigid ICP alignment of all meshes to the first specimen
 *   Stage 3 – Topology transfer: map each reference vertex to the nearest
 *              aligned-specimen vertex and borrow its HU value
 *   Stage 4 – PCA on the N × P HU matrix; export CSVs
 *
 * Output goes to a date-stamped folder (SCAPULA_DENSITY_OUT) so it never
 * collides with other running pipelines.
 */
object DensityPipeline {

  // ---------------------------------------------------------------------------
  // Config — all paths overridable via environment variables
  // ---------------------------------------------------------------------------

  val nrrdDir: File = new File(
    sys.env.getOrElse("SCAPULA_NRRD_DIR",
      s"${sys.env.getOrElse("HOME", "/home/user")}/Documents/all ct from hoel 3d")
  )

  // STL files live in a subfolder of the NRRD directory
  val stlDir: File = new File(
    sys.env.getOrElse("SCAPULA_STL_DIR",
      new File(nrrdDir, "stl/both stl/combined 1/Combined").getAbsolutePath)
  )

  private val runDate: String = LocalDate.now().toString
  val densityOutDir: File = new File(
    sys.env.getOrElse("SCAPULA_DENSITY_OUT",
      s"${sys.env.getOrElse("HOME", "/home/user")}/Documents/scapula_density_sdm_$runDate")
  )

  // ---------------------------------------------------------------------------
  // Stage 1 – Load meshes + volumes, sample HU at surface vertices
  // ---------------------------------------------------------------------------

  def stage1_preprocess(specimens: IndexedSeq[NrrdData.CtSpecimen])
  : IndexedSeq[(NrrdData.CtSpecimen, TriangleMesh[_3D], IndexedSeq[Float])] = {

    println(s"\n[Stage 1] Preprocessing ${specimens.length} specimens")

    // Group by CT volume so each volume is loaded once then discarded before the next is loaded.
    // Keeping all volumes in memory simultaneously would exhaust the heap (each ~150 MB as Short).
    val grouped = specimens.groupBy(_.volumeFile.getAbsolutePath)
      .toIndexedSeq.sortBy(_._1)

    var globalIdx = 0
    grouped.flatMap { case (_, specsForVol) =>
      val volume: NrrdData.HuVolume = NrrdData.loadVolume(specsForVol.head.volumeFile)
      val results = specsForVol.map { spec =>
        globalIdx += 1
        println(s"  [$globalIdx/${specimens.length}] ${spec.id}")
        val mesh = NrrdData.extractSurface(spec)
        println(f"    surface: ${mesh.pointSet.numberOfPoints} vertices")
        val hu = NrrdData.sampleHU(mesh, volume)
        val (minHU, maxHU, meanHU) = NrrdData.huStats(hu)
        val cFrac = NrrdData.corticalMask(hu).count(identity).toDouble / hu.length
        println(f"    HU: min=$minHU%.0f  max=$maxHU%.0f  mean=$meanHU%.0f  cortical(>300HU)=${cFrac * 100}%.1f%%")
        if (meanHU < 50f || meanHU > 2000f)
          println(s"    !! Unusual mean HU — check segmentation / CT units")
        (spec, mesh, hu)
      }
      results  // volume goes out of scope here; GC reclaims it before next CT loads
    }
  }

  // ---------------------------------------------------------------------------
  // Stage 2 – Rigid alignment
  // HU values are indexed by vertex id and travel unchanged through the transform.
  // ---------------------------------------------------------------------------

  def stage2_rigidAlign(
    data: IndexedSeq[(NrrdData.CtSpecimen, TriangleMesh[_3D], IndexedSeq[Float])]
  )(implicit rng: Random)
  : (TriangleMesh[_3D], IndexedSeq[(NrrdData.CtSpecimen, TriangleMesh[_3D], IndexedSeq[Float])]) = {

    println(s"\n[Stage 2] Rigid alignment")
    val (refSpec, refMesh, refHU) = data.head
    println(s"  Reference: ${refSpec.id}")

    val buf = scala.collection.mutable.ArrayBuffer[(NrrdData.CtSpecimen, TriangleMesh[_3D], IndexedSeq[Float])]()
    buf += ((refSpec, refMesh, refHU))

    for ((spec, mesh, hu) <- data.tail) {
      val alignedMesh = RigidAlign.rigidIcp(mesh, refMesh, iterations = Config.icpIterations)
      val d = Metrics.symmetric(alignedMesh, refMesh)
      println(f"  ${spec.id} -> ${d.render}")
      buf += ((spec, alignedMesh, hu))
      System.gc()
    }
    (refMesh, buf.toIndexedSeq)
  }

  // ---------------------------------------------------------------------------
  // Stage 3 – Topology transfer to reference mesh
  // For each reference vertex, borrow the HU of the nearest aligned-specimen vertex.
  // ---------------------------------------------------------------------------

  def stage3_topologyTransfer(
    refMesh: TriangleMesh[_3D],
    aligned: IndexedSeq[(NrrdData.CtSpecimen, TriangleMesh[_3D], IndexedSeq[Float])]
  ): IndexedSeq[(String, IndexedSeq[Float])] = {

    println(s"\n[Stage 3] Topology transfer " +
            s"(${refMesh.pointSet.numberOfPoints} reference vertices × ${aligned.length} specimens)")

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

      // Per-specimen CSV in reference topology
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
  // Stage 4 – Statistical Density Model
  // ---------------------------------------------------------------------------

  def stage4_buildModel(
    refMesh: TriangleMesh[_3D],
    huFields: IndexedSeq[(String, IndexedSeq[Float])]
  ): StatisticalDensityModel.DensityModel = {

    println(s"\n[Stage 4] Building Statistical Density Model (${huFields.length} specimens)")

    val ids    = huFields.map(_._1)
    val fields = huFields.map(_._2)
    val model  = StatisticalDensityModel.build(ids, fields, nComp = math.min(ids.length - 1, 50))
    StatisticalDensityModel.printSummary(model)

    densityOutDir.mkdirs()
    StatisticalDensityModel.exportCsv(model, new File(densityOutDir, "density_model.csv"))

    // Per-specimen latent scores
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

    // Mean HU on reference mesh
    val meanCsv = new File(densityOutDir, "reference_mesh_meanHU.csv")
    val mw = new java.io.PrintWriter(meanCsv)
    try {
      mw.println("vertex_idx,x,y,z,mean_HU")
      refMesh.pointSet.points.toIndexedSeq.zipWithIndex.foreach { case (pt, j) =>
        mw.println(f"$j,${pt.x}%.3f,${pt.y}%.3f,${pt.z}%.3f,${model.meanHU(j)}%.2f")
      }
    } finally mw.close()
    println(s"  Mean HU mesh:        ${meanCsv.getAbsolutePath}")

    // Same data as a real VTK scalar mesh field (geometry + per-vertex mean HU in one
    // file) so it opens directly in ParaView or ViewResults with correct topology and
    // coloring -- the CSV above has no triangle connectivity, so tools have to rebuild
    // the surface themselves (TableToPoints + Delaunay3D) instead of just opening it.
    val meanVtk = new File(densityOutDir, "reference_mesh_meanHU.vtk")
    MeshIO.writeScalarMeshField(ScalarMeshField3D(refMesh, model.meanHU.toArray.toIndexedSeq), meanVtk)
    println(s"  Mean HU VTK mesh:    ${meanVtk.getAbsolutePath}")

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
    println(s"NRRD dir    : ${nrrdDir.getAbsolutePath}")
    println(s"STL dir     : ${stlDir.getAbsolutePath}")
    println(s"Output      : ${densityOutDir.getAbsolutePath}")

    if (!nrrdDir.exists()) {
      println(s"\nERROR: NRRD directory not found: ${nrrdDir.getAbsolutePath}")
      println("Set environment variable SCAPULA_NRRD_DIR to the correct path.")
      sys.exit(1)
    }
    if (!stlDir.exists()) {
      println(s"\nERROR: STL directory not found: ${stlDir.getAbsolutePath}")
      println("Set environment variable SCAPULA_STL_DIR to the correct path.")
      sys.exit(1)
    }

    val specimens = NrrdData.discoverSpecimens(nrrdDir, stlDir)
    println(s"\nFound ${specimens.length} specimens (volume + STL pairs):")
    specimens.foreach(s => println(s"  ${s.id}  <-  ${s.volumeFile.getName}"))

    if (specimens.isEmpty) {
      println("\nERROR: No paired specimens found.")
      println("  Each CT volume needs at least one matching STL in the STL directory.")
      println("  Volume pattern : <id>_volume.nrrd")
      println("  STL pattern    : <id>_scapula_0_Segment_<N>.stl")
      sys.exit(1)
    }
    if (specimens.length < 3)
      println("\nWARNING: Fewer than 3 specimens — model statistics will be trivial.")

    val preprocessed = stage1_preprocess(specimens)

    // ---------------------------------------------------------------------------
    // Filter: keep only specimens with plausible bone HU (mean >= 50 HU).
    // Specimens showing mean ≈ −1000 HU have their STL mesh entirely outside
    // the CT volume (coordinate mismatch).  Those named "marche pas" or
    // "scaling" are also typically invalid — the HU filter catches them too.
    // ---------------------------------------------------------------------------
    val huMin = 50f
    val (valid, dropped) = preprocessed.partition { case (_, _, hu) =>
      NrrdData.huStats(hu)._3 >= huMin
    }

    println(s"\n[Quality Filter] threshold: mean HU >= $huMin")
    if (dropped.nonEmpty) {
      println(s"  Dropping ${dropped.length} specimen(s) with invalid HU:")
      dropped.foreach { case (spec, _, hu) =>
        val (mn, mx, avg) = NrrdData.huStats(hu)
        println(f"    DROPPED  ${spec.id}  (mean=$avg%.0f  min=$mn%.0f  max=$mx%.0f HU)")
      }
    }
    println(s"  Retained: ${valid.length} / ${preprocessed.length} specimens")

    if (valid.length < 3) {
      println("\nERROR: Fewer than 3 valid specimens after quality filter — cannot build a model.")
      sys.exit(1)
    }

    val (refMesh, rigidAligned) = stage2_rigidAlign(valid)
    val huFields                = stage3_topologyTransfer(refMesh, rigidAligned)
    val densityModel            = stage4_buildModel(refMesh, huFields)

    println()
    println("=" * 80)
    println("COMPLETE")
    println("=" * 80)
    println(s"  Specimens total     : ${specimens.length}")
    println(s"  Dropped (bad HU)    : ${dropped.length}")
    println(s"  Modelled            : ${valid.length}")
    println(s"  Model rank          : ${densityModel.k}")
    println(f"  PC1-5 var explained : ${densityModel.explainedVarianceRatio(5) * 100}%.1f%%")
    println(s"  Output dir          : ${densityOutDir.getAbsolutePath}")
    println()
    println("  Output files:")
    println("    raw_hu_per_specimen/<id>_HU.csv  ← HU at each reference vertex per specimen")
    println("    reference_mesh_meanHU.csv         ← mean HU on reference mesh (open in ParaView)")
    println("    density_model.csv                 ← mean + PC1-5 at every vertex")
    println("    density_scores.csv                ← per-specimen latent PC scores")
    println()
    println("  NOTE: Segment_2 and Segment_3 (left/right scapulae) are modelled together.")
    println("  To model only one side, set SCAPULA_SEGMENT=2 or SCAPULA_SEGMENT=3")
    println("  and re-run — the code will filter to that segment number automatically.")
  }
}
