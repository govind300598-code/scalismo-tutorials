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
 *
 * EXPECTED DIRECTORY LAYOUT (segment23 mode)
 * -------------------------------------------
 *   Combined/
 *     SH_00894_scapula_0_Segment_2.stl   SH_00894_scapula_0_Segment_3.stl  …
 *   Set SCAPULA_COMBINED_DIR to this folder.
 *   Set SCAPULA_CSV_PATH to the single_scapulae CSV (or put it in SCAPULA_DATA_DIR).
 *   Optionally set SCAPULA_GOOD_SPECIMENS to the good_specimens.txt written by Stage 0.
 */
object Stage2CorticalAnchor {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val layout = sys.env.getOrElse("SCAPULA_BONE_LAYOUT", "sibling")
    val outDir = Config.outDir

    println(s"Output directory : ${outDir.getAbsolutePath}")
    println(s"Layout mode : $layout")

    // ----------------------------------------------------------------
    // Resolve data directory and landmark CSV for the selected layout.
    val dataDir = layout match {
      case "segment23" => Config.combinedDataDir
      case _           => Config.dataDir
    }
    println(s"Data directory : ${dataDir.getAbsolutePath}")

    val landmarks: Map[String, IndexedSeq[scalismo.geometry.Landmark[scalismo.geometry._3D]]] =
      scala.util.Try {
        val csv = layout match {
          case "segment23" => ScapulaData.resolvedCsv(Config.csvOverride, Config.dataDir, preferSingle = true)
          case _           => ScapulaData.csvFile(Config.dataDir)
        }
        println(s"Landmark CSV   : ${csv.getAbsolutePath}")
        val (lmMap, fromHeader, _) = ScapulaData.readLandmarkCsv(csv)
        if (!fromHeader) println("!! landmark columns resolved via fallback offsets — check CSV header")
        lmMap
      }.getOrElse {
        println("No landmark CSV found — all specimens will use centroid+ICP alignment")
        Map.empty
      }

    // ----------------------------------------------------------------
    // Discover bone pairs, then optionally filter to the Stage-0 approved list.
    val allPairs = layout match {
      case "segment23" => Segment23Loader.fromDir(dataDir)
      case "suffix"    => BonePairLoader.fromSuffixedDir(dataDir)
      case _           => BonePairLoader.fromSiblingDirs(dataDir)
    }

    val goodFilter: Option[Set[String]] = sys.env.get("SCAPULA_GOOD_SPECIMENS").map { path =>
      import scala.io.Source
      val ids = Source.fromFile(path).getLines().map(_.trim).filter(_.nonEmpty).toSet
      println(s"Quality filter loaded: ${ids.size} good specimens from $path")
      ids
    }
    val filteredPairs = goodFilter match {
      case Some(allowed) =>
        val filtered = allPairs.filter(p => allowed.contains(p.modelId))
        println(s"Pairs after quality filter: ${filtered.length} / ${allPairs.length}")
        filtered
      case None => allPairs
    }
    val maxPairs = sys.env.get("SCAPULA_MAX_PAIRS").map(_.toInt)
    val pairs = maxPairs match {
      case Some(n) =>
        val capped = filteredPairs.take(n)
        println(s"SCAPULA_MAX_PAIRS=$n — running on first ${capped.length} specimen(s) only")
        capped
      case None => filteredPairs
    }

    println(s"Found ${pairs.length} bone pairs")
    require(pairs.nonEmpty, "No bone pairs found — check directory layout and SCAPULA_BONE_LAYOUT")

    // ----------------------------------------------------------------
    // Landmark lookup: tries full modelId then the shorter subject prefix.
    // Returns None when no CSV row exists (Hoel specimens have no landmarks —
    // centroid+ICP is used instead).
    def lookupLandmarks(pair: CorticalAnchorAlign.BonePair): Option[IndexedSeq[scalismo.geometry.Landmark[scalismo.geometry._3D]]] =
      landmarks.get(pair.modelId).orElse(landmarks.get(pair.subject))

    // Use the first available pair as reference; landmarks are not required.
    val refPair = pairs.headOption
      .getOrElse(throw new RuntimeException("No bone pairs found — check directory layout"))
    val refCortical    = ScapulaData.loadMesh(refPair.corticalFile)
    val refCorticalLms = lookupLandmarks(refPair)
    println(
      s"Reference specimen : ${refPair.modelId}" +
      (if (refCorticalLms.isDefined) " (with landmarks)" else " (no landmarks — centroid+ICP mode)")
    )

    // ----------------------------------------------------------------
    // Align every pair, save outputs.
    val corticalOutDir    = new File(outDir, "aligned_cortical")
    val trabecularOutDir  = new File(outDir, "aligned_trabecular")
    corticalOutDir.mkdirs()
    trabecularOutDir.mkdirs()

    val results = pairs.map { pair =>
      val cortical    = ScapulaData.loadMesh(pair.corticalFile)
      val trabecular  = ScapulaData.loadMesh(pair.trabecularFile)

      val aligned = CorticalAnchorAlign.alignPair(
        cortical       = cortical,
        trabecular     = trabecular,
        corticalLms    = lookupLandmarks(pair),
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
