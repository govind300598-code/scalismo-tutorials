package scapula

import java.io.{File, PrintWriter}
import scala.util.{Try, Using}

/**
 * STAGE 0 -- DATA QUALITY AUDIT
 *
 * Scans the Combined folder (_Segment_2 / _Segment_3 naming), runs geometric
 * quality checks on every bone pair, and writes:
 *
 *   <outDir>/good_specimens.txt        one baseId per line  (input for Stage 2)
 *   <outDir>/bad_specimens_report.txt  baseId + reasons for every failure
 *
 * HOW TO RUN
 * ----------
 *   SCAPULA_COMBINED_DIR=/path/to/Combined         [required — the folder with _Segment_2 files]
 *   SCAPULA_OUT_DIR=/path/to/output                [required]
 *   SCAPULA_CSV_PATH=/path/to/single_scap.csv      [optional — enables landmark presence check]
 *   SCAPULA_MIN_VOLUME_MM3=5000                    [optional, default 5000  mm³ = 5 cm³]
 *   SCAPULA_MAX_VOLUME_MM3=800000                  [optional, default 800000 mm³ = 800 cm³]
 *   SCAPULA_MIN_VERTICES=100                       [optional, default 100]
 *
 *   sbt "runMain scapula.Stage0DataAudit"
 *
 * QUALITY CHECKS (each is independently reported)
 * ------------------------------------------------
 *   NAME_FLAG    filename contains a known bad-quality marker ("marche pas", "scaling", …)
 *   LOAD_FAIL    mesh file could not be opened / parsed
 *   NEG_VOLUME   signed volume ≤ 0 → inverted normals / open mesh
 *   VOL_RANGE    |volume| outside [minVol, maxVol] mm³ → wrong scale or degenerate geometry
 *   FEW_VERTS    vertex count below threshold → mesh has been collapsed or is a stub
 *   TRAB_LARGER  trabecular volume ≥ cortical volume → segmentation labels may be swapped
 *   NO_CSV_ROW   no matching landmark row in the CSV (warning only, not a geometry failure)
 */
object Stage0DataAudit {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()

    val combinedDir = Config.combinedDataDir
    val outDir      = Config.outDir
    outDir.mkdirs()

    println("=" * 100)
    println("STAGE 0  --  DATA QUALITY AUDIT")
    println("=" * 100)
    println(s"Combined directory : ${combinedDir.getAbsolutePath}")
    println(s"Output directory   : ${outDir.getAbsolutePath}")
    println(s"Volume range       : [${Config.minVolumeMm3 / 1000.0} cm³, ${Config.maxVolumeMm3 / 1000.0} cm³]")
    println(s"Min vertices       : ${Config.minVertices}")
    println()

    // ---------------------------------------------------------------- discover pairs
    val pairs = Segment23Loader.fromDir(combinedDir)
    println(s"Discovered ${pairs.length} paired specimens (_Segment_2 + _Segment_3)")
    if (pairs.isEmpty) {
      println(s"!! No pairs found. Check that SCAPULA_COMBINED_DIR points at the folder with _Segment_2.stl files.")
      return
    }
    println()

    // ---------------------------------------------------------------- optional CSV
    val landmarkMap: Map[String, Boolean] = loadLandmarkIds(Config.csvOverride, Config.dataDir)
    val hasLandmarkCsv = landmarkMap.nonEmpty
    if (hasLandmarkCsv)
      println(s"Landmark CSV loaded — ${landmarkMap.size} model IDs available for cross-check")
    else
      println("No landmark CSV loaded (set SCAPULA_CSV_PATH to enable landmark presence check)")
    println()

    // ---------------------------------------------------------------- check each pair
    case class AuditRow(
      base: String, subject: String,
      issues: List[String],       // hard failures
      warnings: List[String],     // soft warnings
      cortVol: Option[Double],    // mm³
      trabVol: Option[Double],
      cortVerts: Option[Int],
      trabVerts: Option[Int]
    ) {
      def isGood: Boolean = issues.isEmpty
      def tag: String     = if (isGood && warnings.isEmpty) "PASS"
                            else if (isGood) "WARN"
                            else "FAIL"
    }

    val rows: IndexedSeq[AuditRow] = pairs.map { pair =>
      val base    = pair.modelId
      val subject = pair.subject

      val issues   = List.newBuilder[String]
      val warnings = List.newBuilder[String]

      // Check 0: name flag
      if (Segment23Loader.isFlaggedBad(base))
        issues += s"NAME_FLAG: filename contains a bad-quality marker ('${flagInName(base)}')"

      // Load cortical
      val cortMeshTry   = Try(ScapulaData.loadMesh(pair.corticalFile))
      val trabMeshTry   = Try(ScapulaData.loadMesh(pair.trabecularFile))

      if (cortMeshTry.isFailure) issues += s"LOAD_FAIL[cortical]: ${cortMeshTry.failed.get.getMessage}"
      if (trabMeshTry.isFailure) issues += s"LOAD_FAIL[trabecular]: ${trabMeshTry.failed.get.getMessage}"

      val cortVol   = cortMeshTry.toOption.map(m => ScapulaData.signedVolume(m))
      val trabVol   = trabMeshTry.toOption.map(m => ScapulaData.signedVolume(m))
      val cortVerts = cortMeshTry.toOption.map(_.pointSet.numberOfPoints)
      val trabVerts = trabMeshTry.toOption.map(_.pointSet.numberOfPoints)

      // Check 1: negative signed volume (inverted mesh)
      cortVol.foreach { v =>
        if (v <= 0) issues += f"NEG_VOLUME[cortical]: signed volume = ${v / 1000.0}%.1f cm³ (must be > 0)"
      }
      trabVol.foreach { v =>
        if (v <= 0) issues += f"NEG_VOLUME[trabecular]: signed volume = ${v / 1000.0}%.1f cm³ (must be > 0)"
      }

      // Check 2: volume range
      cortVol.foreach { v =>
        val absV = math.abs(v)
        if (absV < Config.minVolumeMm3 || absV > Config.maxVolumeMm3)
          issues += f"VOL_RANGE[cortical]: |volume| = ${absV / 1000.0}%.1f cm³, expected [${Config.minVolumeMm3 / 1000.0}%.0f, ${Config.maxVolumeMm3 / 1000.0}%.0f] cm³"
      }
      trabVol.foreach { v =>
        val absV = math.abs(v)
        if (absV < Config.minVolumeMm3 || absV > Config.maxVolumeMm3)
          issues += f"VOL_RANGE[trabecular]: |volume| = ${absV / 1000.0}%.1f cm³, expected [${Config.minVolumeMm3 / 1000.0}%.0f, ${Config.maxVolumeMm3 / 1000.0}%.0f] cm³"
      }

      // Check 3: vertex count
      cortVerts.foreach { n =>
        if (n < Config.minVertices)
          issues += s"FEW_VERTS[cortical]: $n vertices < ${Config.minVertices}"
      }
      trabVerts.foreach { n =>
        if (n < Config.minVertices)
          issues += s"FEW_VERTS[trabecular]: $n vertices < ${Config.minVertices}"
      }

      // Check 4: trabecular should be smaller than cortical
      for {
        cv <- cortVol if cv > 0
        tv <- trabVol if tv > 0
      } if (tv >= cv)
        warnings += f"TRAB_LARGER: trabecular volume (${tv / 1000.0}%.1f cm³) ≥ cortical (${cv / 1000.0}%.1f cm³) — check segmentation labels"

      // Check 5: CSV landmark presence (warning only)
      if (hasLandmarkCsv) {
        val found = landmarkMap.contains(base) || landmarkMap.contains(subject)
        if (!found)
          warnings += s"NO_CSV_ROW: neither '$base' nor '$subject' found in landmark CSV — this pair cannot be used in Stage 2 without landmarks"
      }

      AuditRow(base, subject,
        issues.result(), warnings.result(),
        cortVol.map(math.abs), trabVol.map(math.abs),
        cortVerts, trabVerts
      )
    }

    // ---------------------------------------------------------------- print table
    val passCount = rows.count(_.isGood)
    val warnCount = rows.count(r => r.isGood && r.warnings.nonEmpty)
    val failCount = rows.count(!_.isGood)

    println(f"${"BASE ID"}%-40s  ${"CORT cm³"}%9s  ${"TRAB cm³"}%9s  ${"CVERTS"}%7s  ${"TVERTS"}%7s  STATUS")
    println("-" * 100)
    rows.sortBy(_.base).foreach { r =>
      val cv = r.cortVol.map(v => f"${v / 1000.0}%8.1f").getOrElse("    ERR ")
      val tv = r.trabVol.map(v => f"${v / 1000.0}%8.1f").getOrElse("    ERR ")
      val cn = r.cortVerts.map(n => f"$n%7d").getOrElse("    ERR")
      val tn = r.trabVerts.map(n => f"$n%7d").getOrElse("    ERR")
      println(f"${r.base}%-40s  $cv  $tv  $cn  $tn  ${r.tag}")
      r.issues.foreach  (msg => println(s"    !! $msg"))
      r.warnings.foreach(msg => println(s"    ~~ $msg"))
    }
    println("-" * 100)
    println(s"TOTAL: ${rows.length}   PASS: $passCount   WARN (pass with warnings): $warnCount   FAIL: $failCount")

    // ---------------------------------------------------------------- write outputs
    val goodSpecimens = rows.filter(_.isGood).map(_.base)
    val goodFile = new File(outDir, "good_specimens.txt")
    Using.resource(new PrintWriter(goodFile)) { w =>
      goodSpecimens.sorted.foreach(w.println)
    }
    println()
    println(s"Good specimens written to : ${goodFile.getAbsolutePath}")

    val badRows = rows.filterNot(_.isGood)
    if (badRows.nonEmpty) {
      val badFile = new File(outDir, "bad_specimens_report.txt")
      Using.resource(new PrintWriter(badFile)) { w =>
        w.println("BAD SPECIMENS REPORT")
        w.println("=" * 80)
        badRows.sortBy(_.base).foreach { r =>
          w.println()
          w.println(s"SPECIMEN: ${r.base}  (subject: ${r.subject})")
          r.issues.foreach(msg => w.println(s"  FAIL  $msg"))
          r.warnings.foreach(msg => w.println(s"  WARN  $msg"))
        }
      }
      println(s"Bad specimens report      : ${badFile.getAbsolutePath}")
    }

    println()
    println("=" * 100)
    println(s"AUDIT COMPLETE: $passCount / ${rows.length} specimens pass quality checks.")
    println("Pass these IDs to Stage 2 with:")
    println(s"  SCAPULA_GOOD_SPECIMENS=${goodFile.getAbsolutePath}")
    println("  SCAPULA_BONE_LAYOUT=segment23")
    println("  SCAPULA_COMBINED_DIR=<same dir as above>")
    println("  sbt \"runMain scapula.Stage2CorticalAnchor\"")
    println("=" * 100)
  }

  // Load all CSV model IDs into a set (true = present). Returns empty map on failure.
  private def loadLandmarkIds(csvOverride: Option[File], fallbackDir: File): Map[String, Boolean] = {
    val csvTry = Try {
      val csv = csvOverride.filter(_.exists()).getOrElse(ScapulaData.csvFileSingle(fallbackDir))
      val (lmMap, _, _) = ScapulaData.readLandmarkCsv(csv)
      lmMap.map { case (k, _) => k -> true }
    }
    csvTry.getOrElse(Map.empty)
  }

  private def flagInName(base: String): String = {
    val lo = base.toLowerCase
    Seq("marche pas", "scaling", "bad stl", "exclude", "corrupt")
      .find(lo.contains)
      .getOrElse("unknown flag")
  }
}
