package scapula

import scalismo.geometry._3D
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.utils.Random

import java.io.{File, PrintWriter}
import scala.util.Using

/**
 * SSM validation: compactness, generalization, specificity, pairwise distances,
 * distance-to-mean — run for BOTH SSM1 and SSM2.
 *
 * SSM1  — registered meshes from pass1/ (initial arbitrary reference).
 * SSM2  — registered meshes from pass2/ (reference = SSM1 mean shape; GPA-style bias removal).
 *
 * The comparison quantifies the benefit of the mean-shape reference update:
 * a more compact SSM2 with a smaller mean-shape shift confirms reference bias was removed.
 *
 * All results are written as CSV files to <outDir>/plots/, with suffixes _ssm1 / _ssm2.
 * After running, plot everything with:
 *
 *   pip install matplotlib seaborn pandas numpy
 *   python3 plot_ssm_metrics.py --plots-dir <outDir>/plots
 *
 * Usage:
 *   sbt "runMain scapula.SSMValidation [/path/to/output/dir]"
 *   SCAPULA_OUT_DIR=/your/path sbt "runMain scapula.SSMValidation"
 *
 * Output CSV files (per model, _ssm1 / _ssm2 suffix on each)
 * ──────────────────────────────────────────────────────────────
 *  compactness_<m>.csv                  mode, variance_pct, cumulative_variance_pct
 *  generalization_<m>.csv               num_modes, mean_error_mm
 *  reconstruction_error_matrix_<m>.csv  specimen_id, num_modes, error_mm (long)
 *  specificity_<m>.csv                  num_modes, mean_specificity_mm, std_specificity_mm
 *  pairwise_distances_<m>.csv           specimen_i, specimen_j, mean_mm, rms_mm, hd95_mm
 *  distance_to_mean_<m>.csv             specimen_id, mean_mm, rms_mm, hd95_mm, hd_mm
 *  stability.csv                        metric, value_mm  (SSM1 mean vs SSM2 mean convergence)
 */
object SSMValidation {

  private def writeRows(file: File, header: String, rows: Seq[String]): Unit = {
    file.getParentFile.mkdirs()
    Using.resource(new PrintWriter(file)) { pw =>
      pw.println(header)
      rows.foreach(pw.println)
    }
    println(f"  -> ${file.getName}%-55s ${rows.length}%4d rows")
  }

  private def loadAndDecimate(passDir: File, label: String)(implicit rng: Random)
      : (IndexedSeq[TriangleMesh[_3D]], IndexedSeq[String]) = {

    val regFiles = Option(passDir.listFiles()).getOrElse(Array.empty[File])
      .filter(f => f.getName.endsWith(".stl") && f.getName.startsWith("reg_"))
      .sortBy(_.getName)
    require(regFiles.nonEmpty, s"No reg_*.stl files in ${passDir.getAbsolutePath}")

    val names     = regFiles.map(_.getName.stripSuffix(".stl").stripPrefix("reg_")).toIndexedSeq
    val rawMeshes = regFiles.map(ScapulaData.loadMesh).toIndexedSeq
    val nPts      = rawMeshes.head.pointSet.numberOfPoints
    println(s"  $label: ${rawMeshes.length} meshes, $nPts vertices each")

    val target = Config.modelResolution
    val meshes = if (nPts > target) {
      println(s"  Decimating $nPts → ~$target vertices (preserving correspondence)...")
      val dec = ScapulaData.decimateInCorrespondence(rawMeshes.head, rawMeshes, target)
      println(s"  Actual: ${dec.head.pointSet.numberOfPoints} vertices")
      dec
    } else {
      println(s"  Already ≤ $target vertices — skipping decimation")
      rawMeshes
    }
    (meshes, names)
  }

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val baseDir  = if (args.nonEmpty) new File(args(0)) else Config.outDir
    val pass1Dir = new File(baseDir, "pass1")
    val pass2Dir = new File(baseDir, "pass2")
    val plotsDir = new File(baseDir, "plots")
    plotsDir.mkdirs()

    require(pass1Dir.isDirectory,
      s"pass1/ not found: ${pass1Dir.getAbsolutePath}\n" +
      s"  Run 'sbt runMain scapula.RebuildSSM' first, or set SCAPULA_OUT_DIR.")
    require(pass2Dir.isDirectory,
      s"pass2/ not found: ${pass2Dir.getAbsolutePath}\n" +
      s"  Run 'sbt runMain scapula.RebuildSSM' first, or set SCAPULA_OUT_DIR.")

    println(s"\nLoading registered meshes from output directory: ${baseDir.getAbsolutePath}")
    val (ssm1Meshes, ssm1Names) = loadAndDecimate(pass1Dir, "SSM1 (pass1, initial ref)")
    val (ssm2Meshes, ssm2Names) = loadAndDecimate(pass2Dir, "SSM2 (pass2, mean-shape ref)")

    // ── Validation helper: 5 metrics for one model ────────────────────────────
    def runValidation(
        label:  String,
        suffix: String,
        meshes: IndexedSeq[TriangleMesh[_3D]],
        names:  IndexedSeq[String]
    ): PointDistributionModel[_3D, TriangleMesh] = {

      println(s"\n\n╔══════════════════════════════════════════════════════════╗")
      println(s"║  $label — ${meshes.length} specimens")
      println(s"╚══════════════════════════════════════════════════════════╝")

      println("Building SSM via PCA...")
      val t0          = System.currentTimeMillis()
      val dc          = DataCollection.fromTriangleMesh3DSequence(meshes.head, meshes)
      val model       = PointDistributionModel.createUsingPCA(dc)
      val nModes      = model.rank
      val eigenvalues = model.gp.klBasis.map(_.eigenvalue).toArray
      val totalVar    = eigenvalues.sum
      val meanMesh    = model.mean
      println(f"  rank = $nModes  |  elapsed = ${(System.currentTimeMillis() - t0) / 1000.0}%.1f s")

      // 1. Compactness
      println(s"\n[1/5] Compactness")
      val cumVar = eigenvalues.scanLeft(0.0)(_ + _).tail.map(_ / totalVar * 100.0)
      writeRows(
        new File(plotsDir, s"compactness$suffix.csv"),
        "mode,variance_pct,cumulative_variance_pct",
        (1 to nModes).map { k =>
          f"$k,${eigenvalues(k - 1) / totalVar * 100.0}%.4f,${cumVar(k - 1)}%.4f"
        })
      def modesFor(pct: Double): String = {
        val i = cumVar.indexWhere(_ >= pct); if (i < 0) s">$nModes" else s"${i + 1}"
      }
      println(f"  90%% variance: ${modesFor(90)} modes  |  95%%: ${modesFor(95)}  |  99%%: ${modesFor(99)}")

      // 2. Generalization
      println(s"\n[2/5] Generalization (projection reconstruction error)...")
      val genData: IndexedSeq[(Int, String, Double)] = (1 to nModes).flatMap { k =>
        if (k == 1 || k % 5 == 0 || k == nModes) print(s"  k=$k")
        val kModel = model.truncate(k)
        meshes.zip(names).map { case (m, name) =>
          val proj  = kModel.project(m)
          val dists = Metrics.correspondingDistances(proj, m)
          (k, name, dists.sum / dists.length)
        }
      }
      println()
      writeRows(new File(plotsDir, s"generalization$suffix.csv"), "num_modes,mean_error_mm",
        genData.groupBy(_._1).toSeq.sortBy(_._1).map { case (k, entries) =>
          f"$k,${entries.map(_._3).sum / entries.length}%.4f"
        })
      writeRows(new File(plotsDir, s"reconstruction_error_matrix$suffix.csv"),
        "specimen_id,num_modes,error_mm",
        genData.map { case (k, name, err) => f"$name,$k,$err%.4f" })

      // 3. Specificity
      println(s"\n[3/5] Specificity (200 samples per mode count)...")
      val nSamples = 200
      val specRows = (1 to nModes).map { k =>
        if (k == 1 || k % 5 == 0 || k == nModes) print(s"  k=$k")
        val kModel   = model.truncate(k)
        val minDists = (0 until nSamples).map { _ =>
          val s = kModel.sample()
          meshes.map { m =>
            val d = Metrics.correspondingDistances(s, m); d.sum / d.length
          }.min
        }
        val mean = minDists.sum / minDists.length
        val std  = math.sqrt(minDists.map(d => (d - mean) * (d - mean)).sum / minDists.length)
        f"$k,$mean%.4f,$std%.4f"
      }
      println()
      writeRows(new File(plotsDir, s"specificity$suffix.csv"),
        "num_modes,mean_specificity_mm,std_specificity_mm", specRows)

      // 4. Pairwise distances
      println(s"\n[4/5] Pairwise point-to-point distances...")
      val n = meshes.length
      writeRows(new File(plotsDir, s"pairwise_distances$suffix.csv"),
        "specimen_i,specimen_j,mean_mm,rms_mm,hd95_mm",
        (for {
          i <- 0 until n
          j <- i + 1 until n
        } yield {
          val d    = Metrics.correspondingDistances(meshes(i), meshes(j))
          val mean = d.sum / d.length
          val rms  = math.sqrt(d.map(x => x * x).sum / d.length)
          val hd95 = Metrics.percentile(d, 0.95)
          f"${names(i)},${names(j)},$mean%.4f,$rms%.4f,$hd95%.4f"
        }).toSeq)

      // 5. Distance to mean
      println(s"\n[5/5] Distance to mean shape...")
      val meanRows = meshes.zip(names).map { case (m, name) =>
        val d    = Metrics.correspondingDistances(m, meanMesh)
        val mean = d.sum / d.length
        val rms  = math.sqrt(d.map(x => x * x).sum / d.length)
        val hd95 = Metrics.percentile(d, 0.95)
        val hd   = d.max
        f"$name,$mean%.4f,$rms%.4f,$hd95%.4f,$hd%.4f"
      }
      writeRows(new File(plotsDir, s"distance_to_mean$suffix.csv"),
        "specimen_id,mean_mm,rms_mm,hd95_mm,hd_mm", meanRows)

      // Summary
      val avg90 = cumVar.indexWhere(_ >= 90.0)
      println(s"\n$label summary:")
      println(f"  specimens = ${meshes.length}  |  rank = $nModes  |  modes for 90%% var = ${if (avg90 < 0) nModes else avg90 + 1}")
      val distMean = meanRows.map(r => r.split(",").last.toDouble)
      println(f"  mean Hausdorff dist to SSM mean = ${distMean.sum / distMean.length}%.3f mm")

      model
    }

    // ── SSM1: registered to arbitrary initial reference (pass 1) ──────────────
    val model1 = runValidation("SSM1  (pass 1 — initial reference)", "_ssm1", ssm1Meshes, ssm1Names)

    // ── SSM2: registered to SSM1 mean shape (pass 2) ─────────────────────────
    val model2 = runValidation("SSM2  (pass 2 — mean-shape reference)", "_ssm2", ssm2Meshes, ssm2Names)

    // ── Stability check: SSM1 mean vs SSM2 mean convergence ──────────────────
    println(s"\n\n─── Stability check: SSM1 mean ↔ SSM2 mean ───")
    println("  Measures whether the GPA reference-update converged.")
    println("  A small shift (< 1 mm mean) confirms reference bias has been removed.\n")

    val stability = Metrics.symmetric(model1.mean, model2.mean)
    println(f"  Surface distance SSM1_mean ↔ SSM2_mean:")
    println(f"    ${stability.render}")
    if (stability.mean < 1.0)
      println(f"  ✓ CONVERGED  (mean shift = ${stability.mean}%.3f mm < 1 mm threshold)")
    else
      println(f"  ✗ NOT CONVERGED  (mean shift = ${stability.mean}%.3f mm). " +
              f"Consider SCAPULA_REFINE_PASSES=3.")

    writeRows(new File(plotsDir, "stability.csv"), "metric,value_mm",
      Seq(
        f"mean,${stability.mean}%.4f",
        f"rms,${stability.rms}%.4f",
        f"hd95,${stability.hd95}%.4f",
        f"hd,${stability.hd}%.4f"
      ))

    // ── Comparison summary table ──────────────────────────────────────────────
    val ev1 = model1.gp.klBasis.map(_.eigenvalue).toArray
    val ev2 = model2.gp.klBasis.map(_.eigenvalue).toArray
    val cv1 = ev1.scanLeft(0.0)(_ + _).tail.map(_ / ev1.sum * 100.0)
    val cv2 = ev2.scanLeft(0.0)(_ + _).tail.map(_ / ev2.sum * 100.0)
    def modesForArr(cv: Array[Double], n: Int, pct: Double): Int = {
      val i = cv.indexWhere(_ >= pct); if (i < 0) n else i + 1
    }

    val sep = "═" * 62
    println(s"\n\n$sep")
    println(f"  ${"Comparison: SSM1 vs SSM2"}%-36s  ${"SSM1"}%8s  ${"SSM2"}%8s")
    println(sep)
    println(f"  ${"Specimens"}%-36s  ${ssm1Names.length}%8d  ${ssm2Names.length}%8d")
    println(f"  ${"PCA rank"}%-36s  ${model1.rank}%8d  ${model2.rank}%8d")
    println(f"  ${"Modes for 90%% variance"}%-36s  ${modesForArr(cv1, model1.rank, 90)}%8d  ${modesForArr(cv2, model2.rank, 90)}%8d")
    println(f"  ${"Modes for 95%% variance"}%-36s  ${modesForArr(cv1, model1.rank, 95)}%8d  ${modesForArr(cv2, model2.rank, 95)}%8d")
    println(f"  ${"Modes for 99%% variance"}%-36s  ${modesForArr(cv1, model1.rank, 99)}%8d  ${modesForArr(cv2, model2.rank, 99)}%8d")
    println(f"  ${"Mean shape shift (mm)"}%-36s  ${"—"}%8s  ${stability.mean}%8.3f")
    println(f"  ${"HD95 shape shift (mm)"}%-36s  ${"—"}%8s  ${stability.hd95}%8.3f")
    println(sep)
    println(s"  SSM2 mean-shape reference = SSM1 mean  (GPA-style reference update)")
    println(s"  A lower mode count for the same variance % in SSM2 confirms bias removal.")
    println(sep)

    // ── Final summary ─────────────────────────────────────────────────────────
    println(s"\nAll CSVs written to: ${plotsDir.getAbsolutePath}")
    println("\nTo plot:")
    println("  pip install matplotlib seaborn pandas numpy")
    println(s"  python3 plot_ssm_metrics.py --plots-dir ${plotsDir.getAbsolutePath}")
  }
}
