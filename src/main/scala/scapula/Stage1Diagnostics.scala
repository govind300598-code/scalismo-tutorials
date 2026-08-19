package scapula

import scalismo.geometry.*
import scalismo.mesh.*
import scalismo.utils.Random

/**
 * STAGE 1 -- DIAGNOSTICS. Run this before anything else.
 *
 * It answers one question: is the large surface distance between two scapulae caused by a defect in the pipeline
 * (mirror axis, landmark correspondence, coordinate frame, rigid alignment), or is it the genuine anatomical
 * difference between two different people's bones?
 *
 * The design uses the fact that the dataset is PAIRED: each subject contributes a left and a right scapula. Bilateral
 * scapulae from one person are near mirror-symmetric, so the within-subject comparison is a built-in ground truth that
 * the between-subject comparison does not have.
 */
object Stage1Diagnostics {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dir = Config.dataDir
    val csv = ScapulaData.csvFile(dir)
    println(s"Data directory : ${dir.getAbsolutePath}")
    println(s"Landmark CSV   : ${csv.getAbsolutePath}")

    // ---------------------------------------------------------------- A1
    val (landmarks, fromHeader, header) = ScapulaData.readLandmarkCsv(csv)
    println()
    println("[A1] LANDMARK COLUMN RESOLUTION")
    val (cols, _) = ScapulaData.resolveColumns(header)
    if (fromHeader) println("  columns resolved BY HEADER NAME (safe against column shifts)")
    else
      println("  !! header names could not be matched -- FALLING BACK to hardcoded column offsets.\n" +
        "     Verify the printout below against your CSV before trusting any downstream number.")
    ScapulaData.landmarkNames.foreach { nm =>
      val (xi, yi, zi) = cols(nm)
      val names = Seq(xi, yi, zi).map(i => header.lift(i).getOrElse("<missing>"))
      println(f"    $nm%-4s -> cols ($xi, $yi, $zi) = ${names.mkString(", ")}")
    }
    println(s"  ${landmarks.size} rows parsed, ${ScapulaData.landmarkNames.size} landmarks each")

    val specimens = ScapulaData.specimens(dir)
    val missing = specimens.map(_.modelId).filterNot(landmarks.contains)
    if (missing.nonEmpty) println(s"  !! STL files with no CSV row: ${missing.mkString(", ")}")
    println(s"  ${specimens.length} STL files found")

    // ---------------------------------------------------------------- A2
    println()
    println("[A2] MIRROR VALIDITY")
    println("  Reflection about x=0 has determinant -1; triangle winding is flipped to keep normals outward.")
    println("  Note: the CHOICE of mirror plane cannot itself cause misregistration -- reflecting about any")
    println("  other plane differs by a rigid motion, which the rigid alignment that follows absorbs exactly.")
    println("  What must hold is that orientation is preserved. Signed volume (positive = watertight, outward):")
    specimens.take(4).foreach { s =>
      val m = ScapulaData.loadMesh(s.file)
      val vBefore = ScapulaData.signedVolume(m)
      val vAfter = ScapulaData.signedVolume(ScapulaData.mirrorMesh(m))
      val ok = if (math.signum(vBefore) == math.signum(vAfter) && vBefore > 0) "OK" else "!! ORIENTATION PROBLEM"
      println(f"    ${s.modelId}%-28s before=${vBefore / 1000.0}%10.1f cm^3  after=${vAfter / 1000.0}%10.1f cm^3  $ok")
    }

    // ---------------------------------------------------------------- A3
    println()
    println("[A3] WITHIN-SUBJECT LANDMARK RESIDUAL  (left vs mirrored right, same person)")
    println("  Tests landmark labelling and digitisation consistency between sides.")
    println("  Expect a few mm. A single landmark standing out means that landmark's placement protocol is the fault.")

    val bySubject = specimens.groupBy(_.subject).toIndexedSeq.sortBy(_._1)
    val lmResiduals = bySubject.flatMap { case (subject, group) =>
      for {
        l <- group.find(!_.isRight)
        r <- group.find(_.isRight)
        lLms <- landmarks.get(l.modelId)
        rLms <- landmarks.get(r.modelId)
      } yield {
        val rMirrored = ScapulaData.mirrorLandmarks(rLms)
        val t = ScapulaData.rigidFromLandmarks(rMirrored, lLms)
        val moved = rMirrored.map(lm => lm.copy(point = t(lm.point)))
        val per = ScapulaData.perLandmarkDistances(lLms, moved)
        val mean = per.map(_._2).sum / per.length
        val flag = if (mean > 5.0) "  <-- inspect" else ""
        println(f"    $subject%-26s mean=$mean%5.2f mm | " +
          per.sortBy(_._1).map { case (n, d) => f"$n=$d%.1f" }.mkString(" ") + flag)
        (subject, mean, per)
      }
    }
    if (lmResiduals.nonEmpty) {
      val overall = lmResiduals.map(_._2).sum / lmResiduals.length
      println(f"    ${"OVERALL"}%-26s mean=$overall%5.2f mm")
      val byLandmark = lmResiduals.flatMap(_._3).groupBy(_._1).view.mapValues(v => v.map(_._2).sum / v.length).toMap
      println("    per-landmark mean across subjects: " +
        byLandmark.toIndexedSeq.sortBy(-_._2).map { case (n, d) => f"$n=$d%.2f" }.mkString("  "))
      val worst = byLandmark.maxBy(_._2)
      if (worst._2 > 1.6 * (byLandmark.values.sum / byLandmark.size))
        println(f"    !! '${worst._1}' is a clear outlier (${worst._2}%.2f mm) -- re-check its placement protocol.")
    }

    // ---------------------------------------------------------------- A4 / A5
    println()
    println("[A4] WITHIN-SUBJECT SURFACE DISTANCE  (left vs mirrored right, same person, after rigid alignment)")
    println("  Landmark-independent. This is the noise floor of the whole rigid pipeline.")

    def rigidSurfaceStats(aFile: java.io.File,
                          aLms: IndexedSeq[Landmark[_3D]],
                          aMirror: Boolean,
                          bFile: java.io.File,
                          bLms: IndexedSeq[Landmark[_3D]],
                          bMirror: Boolean
    ): Metrics.SurfaceStats = {
      def prep(f: java.io.File, lms: IndexedSeq[Landmark[_3D]], mirror: Boolean) = {
        val m = ScapulaData.loadMesh(f)
        if (mirror) (ScapulaData.mirrorMesh(m), ScapulaData.mirrorLandmarks(lms)) else (m, lms)
      }
      val (bm, bl) = prep(bFile, bLms, bMirror)
      val (am, al) = prep(aFile, aLms, aMirror)
      val (aligned, _) = RigidAlign.landmarkThenIcp(am, al, bm, bl)
      Metrics.symmetric(aligned, bm)
    }

    val withinPairs = bySubject.flatMap { case (subject, group) =>
      for {
        l <- group.find(!_.isRight)
        r <- group.find(_.isRight)
        lLms <- landmarks.get(l.modelId)
        rLms <- landmarks.get(r.modelId)
      } yield {
        val st = rigidSurfaceStats(r.file, rLms, true, l.file, lLms, false)
        println(f"    $subject%-26s ${st.render}")
        st
      }
    }

    println()
    println("[A5] BETWEEN-SUBJECT SURFACE DISTANCE  (different people, after the same rigid alignment)")
    println("  This is the anatomical variability floor. No rigid method can go below it.")

    val leftSpecimens = specimens.filter(s => !s.isRight && landmarks.contains(s.modelId))
    val betweenPairs = leftSpecimens.sliding(2).toIndexedSeq.take(8).flatMap {
      case IndexedSeq(a, b) =>
        val st = rigidSurfaceStats(a.file, landmarks(a.modelId), false, b.file, landmarks(b.modelId), false)
        println(f"    ${a.subject}%-12s vs ${b.subject}%-12s ${st.render}")
        Some(st)
      case _ => None
    }

    // ---------------------------------------------------------------- verdict
    println()
    println("=" * 100)
    println("VERDICT")
    println("=" * 100)
    if (withinPairs.nonEmpty && betweenPairs.nonEmpty) {
      val wMean = withinPairs.map(_.mean).sum / withinPairs.length
      val wHd = withinPairs.map(_.hd).sum / withinPairs.length
      val bMean = betweenPairs.map(_.mean).sum / betweenPairs.length
      val bHd = betweenPairs.map(_.hd).sum / betweenPairs.length
      println(f"  within-subject  (same person, L vs mirrored R) : mean=$wMean%5.2f mm   Hausdorff=$wHd%6.2f mm")
      println(f"  between-subject (different people)             : mean=$bMean%5.2f mm   Hausdorff=$bHd%6.2f mm")
      println()
      if (wHd < bHd * 0.7) {
        println("  The within-subject distance is clearly smaller than the between-subject distance.")
        println("  => The mirror, the coordinate frame, the landmarks and the rigid alignment are WORKING.")
        println("     The large Hausdorff distance you are seeing is the real shape difference between")
        println("     different people's scapulae. No amount of rigid-ICP tuning can remove it, because a")
        println("     rigid transform has 6 degrees of freedom and cannot change shape.")
        println()
        println("  => Correct next step: NON-RIGID registration (Stage 2), which establishes point-to-point")
        println("     correspondence and reduces the residual surface distance to sub-millimetre, and then")
        println("     a statistical shape model (Stage 3) that MODELS the remaining variability instead of")
        println("     trying to make it disappear.")
      } else {
        println("  The within-subject distance is NOT clearly smaller than the between-subject distance.")
        println("  => Something upstream really is wrong. Work through, in order: the [A2] orientation check,")
        println("     then the per-landmark breakdown in [A3] to find which landmark is inconsistent.")
      }
    }
    println()
    println("  Reminder on metrics: MeshMetrics.hausdorffDistance is the maximum over ALL vertices in BOTH")
    println("  directions -- a single worst point on the acromion tip or the coracoid sets it. Every ICP")
    println("  variant optimises a MEAN criterion, so ICP is entitled to increase the Hausdorff distance while")
    println("  genuinely improving the fit. Report HD95 alongside it, never the raw maximum alone.")
    println("  Dice is also unreliable here: it is estimated from 10000 samples in the union bounding box, and a")
    println("  scapula is a thin blade occupying a small fraction of its box, so very few samples land inside.")
  }
}
