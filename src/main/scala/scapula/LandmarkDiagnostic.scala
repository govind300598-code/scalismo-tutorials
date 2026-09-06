package scapula

import scalismo.geometry._3D

import java.io.File
import scala.io.Source
import scala.util.Using

/**
 * Prints every landmark coordinate read from the CSV so you can
 * verify the values are anatomically correct before running the pipeline.
 *
 * Run with:
 *   SCAPULA_DATA_DIR=... sbt "runMain scapula.LandmarkDiagnostic"
 *
 * What to check in the output:
 *   - All five landmarks (GC, TS, IA, PLA, AC) should have reasonable 3-D
 *     coordinates (not 0, not obviously wrong like 1.0/2.0/3.0/...)
 *   - "Spread" (std-dev) of each landmark across specimens should be ~10-80 mm
 *     (bone-scale variation). Near-zero spread = reading the same wrong column.
 *   - After rigid alignment the "residual LM error" per specimen should be
 *     < 5 mm. Large errors → wrong landmark coordinates.
 */
object LandmarkDiagnostic {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()

    val dataDir = Config.dataDir
    println("=" * 72)
    println("  LANDMARK DIAGNOSTIC")
    println("=" * 72)
    println(s"Data dir : ${dataDir.getAbsolutePath}")

    // ── 1. Locate CSV ──────────────────────────────────────────────────────
    val csvFile = ScapulaData.csvFile(dataDir)
    println(s"CSV file : ${csvFile.getName}")

    // ── 2. Print raw header ────────────────────────────────────────────────
    val lines  = Using.resource(Source.fromFile(csvFile))(_.getLines().toIndexedSeq)
    val header = lines.head.split(",", -1).toIndexedSeq.map(_.trim)
    println(s"\nCSV header has ${header.length} columns:")
    header.zipWithIndex.foreach { case (h, i) =>
      println(f"  [$i%3d]  $h")
    }

    // ── 3. Resolve landmark columns (show which method was used) ───────────
    val (cols, fromHeader) = ScapulaData.resolveColumns(header)
    println(s"\nColumn resolution: ${if (fromHeader) "FROM HEADER (accurate)" else "FALLBACK OFFSETS (may be wrong!)"}")
    ScapulaData.landmarkNames.foreach { lm =>
      val (xi, yi, zi) = cols(lm)
      println(f"  $lm%-6s  x=col[$xi%d] '${header.lift(xi).getOrElse("???")}%s'  " +
              f"y=col[$yi%d] '${header.lift(yi).getOrElse("???")}%s'  " +
              f"z=col[$zi%d] '${header.lift(zi).getOrElse("???")}%s'")
    }

    // ── 4. Read landmarks ──────────────────────────────────────────────────
    val (lmMap, _, _) = ScapulaData.readLandmarkCsv(csvFile)
    println(s"\nLandmarks read for ${lmMap.size} specimens")

    // ── 5. Print per-specimen landmark coordinates ─────────────────────────
    println("\n" + "─" * 72)
    println("PER-SPECIMEN LANDMARK COORDINATES")
    println("─" * 72)
    lmMap.toSeq.sortBy(_._1).foreach { case (modelId, lms) =>
      println(s"\n  $modelId")
      lms.foreach { lm =>
        println(f"    ${lm.id}%-6s  (${lm.point.x}%9.2f, ${lm.point.y}%9.2f, ${lm.point.z}%9.2f)")
      }
    }

    // ── 6. Per-landmark spread (std-dev) across all specimens ──────────────
    println("\n" + "─" * 72)
    println("LANDMARK SPREAD ACROSS ALL SPECIMENS  (should be ~10–80 mm)")
    println("─" * 72)
    ScapulaData.landmarkNames.foreach { nm =>
      val pts = lmMap.values.flatMap(_.find(_.id == nm).map(_.point)).toIndexedSeq
      if (pts.nonEmpty) {
        val xs = pts.map(_.x); val ys = pts.map(_.y); val zs = pts.map(_.z)
        def std(v: IndexedSeq[Double]): Double = {
          val m = v.sum / v.length
          math.sqrt(v.map(x => (x - m) * (x - m)).sum / v.length)
        }
        val mx = xs.sum / xs.length; val my = ys.sum / ys.length; val mz = zs.sum / zs.length
        println(f"  $nm%-6s  mean=(${mx}%8.1f, ${my}%8.1f, ${mz}%8.1f)  " +
                f"std=(${std(xs)}%6.1f, ${std(ys)}%6.1f, ${std(zs)}%6.1f)")
      }
    }

    // ── 7. Rigid alignment residuals for reference specimen ────────────────
    val allSpecs  = ScapulaData.specimens(dataDir)
    val validSpecs = allSpecs.filter(s => lmMap.contains(s.modelId))

    val refSpec = validSpecs(Config.refIdx.min(validSpecs.length - 1))
    val refLms  = if (refSpec.isRight) ScapulaData.mirrorLandmarks(lmMap(refSpec.modelId))
                  else lmMap(refSpec.modelId)
    println(s"\n" + "─" * 72)
    println(s"RIGID ALIGNMENT RESIDUALS  (ref = ${refSpec.modelId})")
    println("─" * 72)
    println(f"  ${"Specimen"}%-30s  ${"LM error mm"}%12s  ${"Verdict"}%10s")
    validSpecs.foreach { s =>
      val rawLms = lmMap(s.modelId)
      val lms    = if (s.isRight) ScapulaData.mirrorLandmarks(rawLms) else rawLms
      val trans  = ScapulaData.rigidFromLandmarks(lms, refLms)
      val aligned = lms.map(lm => lm.copy(point = trans(lm.point)))
      val dists  = ScapulaData.perLandmarkDistances(refLms, aligned)
      val avg    = dists.map(_._2).sum / dists.size
      val verdict = if (avg < 5.0) "OK" else if (avg < 15.0) "POOR" else "BAD ← wrong LMs?"
      println(f"  ${s.modelId}%-30s  ${avg}%12.2f  $verdict")
    }

    println("\n" + "═" * 72)
    println("  INTERPRETATION GUIDE")
    println("═" * 72)
    println("  Column resolution = FROM HEADER    → columns are correct")
    println("  Column resolution = FALLBACK OFFSETS → might be wrong")
    println("  Landmark spread < 1 mm             → reading a constant/wrong column!")
    println("  Residual LM error < 5 mm           → good alignment")
    println("  Residual LM error > 15 mm          → landmarks are wrong or mismatched")
    println("  If FALLBACK OFFSETS is shown, check col[] names above for each landmark")
    println("  and set SCAPULA_LM_COL_GC etc. or fix the CSV header.")
    println("=" * 72)
  }
}
