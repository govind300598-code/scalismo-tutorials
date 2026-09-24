package scapula

import scalismo.io.MeshIO
import scalismo.utils.Random

import java.io.File

/**
 * STAGE 2 -- CORTICAL-ANCHOR ALIGNMENT
 *
 * Implements the strategy described by the collaborator:
 *
 *   "Since you segmented both the cortical and trabecular bone from the exact same
 *    image, they already share the same spatial positioning.  Align the cortical bone
 *    (which is much easier).  Then use that aligned cortical structure as an 'anchor'
 *    to automatically position the corresponding trabecular bone."
 *
 * HOW TO RUN
 * ----------
 * Environment variables (all optional; sensible defaults shown):
 *
 *   SCAPULA_BONE_LAYOUT  sibling | suffix   (default: sibling)
 *   SCAPULA_DATA_DIR     path to data root  (default: from Config)
 *   SCAPULA_OUT_DIR      path for output    (default: from Config)
 *   SCAPULA_ICP_ITERS    ICP iterations     (default: from Config)
 *   SCAPULA_UI           true/false         show meshes in viewer
 *
 * EXPECTED DIRECTORY LAYOUT (sibling mode, the default)
 * -----------------------------------------------------
 *   <data-dir>/
 *     cortical/
 *       subject001_L.stl
 *       subject001_R.stl
 *       …
 *     trabecular/
 *       subject001_L.stl
 *       subject001_R.stl
 *       …
 *     scapula_model_data.csv   (same landmark CSV as Stage 1)
 *
 * EXPECTED DIRECTORY LAYOUT (suffix mode)
 * ----------------------------------------
 *   <data-dir>/
 *     subject001_L_cortical.stl
 *     subject001_L_trabecular.stl
 *     …
 *     scapula_model_data.csv
 */
object Stage2CorticalAnchor {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val layout  = sys.env.getOrElse("SCAPULA_BONE_LAYOUT", "sibling")
    val dataDir = Config.dataDir
    val outDir  = Config.outDir

    println(s"Data directory : ${dataDir.getAbsolutePath}")
    println(s"Output directory : ${outDir.getAbsolutePath}")
    println(s"Layout mode : $layout")

    // ----------------------------------------------------------------
    // Load landmark CSV (same file used in Stage 1 — cortical landmarks only)
    val csv = ScapulaData.csvFile(dataDir)
    val (landmarks, fromHeader, _) = ScapulaData.readLandmarkCsv(csv)
    if (!fromHeader) println("!! landmark columns resolved via fallback offsets — check CSV header")

    // ----------------------------------------------------------------
    // Discover bone pairs
    val pairs = layout match {
      case "suffix"  => BonePairLoader.fromSuffixedDir(dataDir)
      case _         => BonePairLoader.fromSiblingDirs(dataDir)
    }
    println(s"Found ${pairs.length} bone pairs")
    require(pairs.nonEmpty, "No bone pairs found — check directory layout and SCAPULA_BONE_LAYOUT")

    // ----------------------------------------------------------------
    // Choose a reference: the first specimen that has a landmark row.
    val refPair = pairs.find(p => landmarks.contains(p.modelId))
      .getOrElse(throw new RuntimeException("No bone pair has a matching landmark row"))
    val refCortical    = ScapulaData.loadMesh(refPair.corticalFile)
    val refCorticalLms = landmarks(refPair.modelId)
    println(s"Reference specimen : ${refPair.modelId}")

    // ----------------------------------------------------------------
    // Align every other pair, save outputs.
    val corticalOutDir    = new File(outDir, "aligned_cortical")
    val trabecularOutDir  = new File(outDir, "aligned_trabecular")
    corticalOutDir.mkdirs()
    trabecularOutDir.mkdirs()

    val results = pairs.flatMap { pair =>
      landmarks.get(pair.modelId).map { lms =>

        val cortical    = ScapulaData.loadMesh(pair.corticalFile)
        val trabecular  = ScapulaData.loadMesh(pair.trabecularFile)

        val aligned = CorticalAnchorAlign.alignPair(
          cortical       = cortical,
          trabecular     = trabecular,
          corticalLms    = lms,
          refCortical    = refCortical,
          refCorticalLms = refCorticalLms,
          icpIterations  = Config.icpIterations
        )

        val dCortical   = Metrics.symmetric(aligned.cortical,   refCortical)
        val dTrabecular = Metrics.symmetric(aligned.trabecular, aligned.cortical)

        MeshIO.writeMesh(aligned.cortical,
          new File(corticalOutDir,   s"${pair.modelId}.stl")).get
        MeshIO.writeMesh(aligned.trabecular,
          new File(trabecularOutDir, s"${pair.modelId}.stl")).get

        println(
          f"  ${pair.modelId}%-30s " +
          f"cortical→ref: ${dCortical.render}   " +
          f"trabecular↔cortical: ${dTrabecular.render}"
        )
        (pair.modelId, dCortical, dTrabecular)
      }
    }

    // ----------------------------------------------------------------
    // Summary
    println()
    println("=" * 110)
    println("SUMMARY")
    println("=" * 110)
    println(s"  Aligned ${results.length} pairs")
    if (results.nonEmpty) {
      val meanCortRef  = results.map(_._2.mean).sum / results.length
      val meanTrabCort = results.map(_._3.mean).sum / results.length
      println(f"  mean cortical→reference surface distance : $meanCortRef%.2f mm")
      println(f"  mean trabecular↔cortical surface distance: $meanTrabCort%.2f mm")
      println()
      println("  The trabecular↔cortical distance measures anatomical layer separation, not")
      println("  registration error — it should be non-zero and anatomically meaningful.")
    }
    println()
    println(s"  Aligned meshes written to:")
    println(s"    cortical    → ${corticalOutDir.getAbsolutePath}")
    println(s"    trabecular  → ${trabecularOutDir.getAbsolutePath}")
  }
}
