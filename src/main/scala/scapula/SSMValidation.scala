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
 * distance to mean — run for BOTH SSM1 (all specimens) and SSM2 (one per subject).
 *
 * SSM1  — all registered specimens (includes paired L+R from the same subject).
 * SSM2  — one specimen per subject; left side is preferred when both are available.
 *         Removing the within-subject pairing gives an independent sample.
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
 *  compactness_<m>.csv               mode, variance_pct, cumulative_variance_pct
 *  generalization_<m>.csv            num_modes, mean_error_mm
 *  reconstruction_error_matrix_<m>.csv  specimen_id, num_modes, error_mm (long)
 *  specificity_<m>.csv               num_modes, mean_specificity_mm, std_specificity_mm
 *  pairwise_distances_<m>.csv        specimen_i, specimen_j, mean_mm, rms_mm, hd95_mm
 *  distance_to_mean_<m>.csv          specimen_id, mean_mm, rms_mm, hd95_mm, hd_mm
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

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val baseDir  = if (args.nonEmpty) new File(args(0)) else Config.outDir
    val pass2Dir = new File(baseDir, "pass2")
    val plotsDir = new File(baseDir, "plots")
    plotsDir.mkdirs()

    require(
      pass2Dir.isDirectory,
      s"pass2/ directory not found: ${pass2Dir.getAbsolutePath}\n" +
      s"  Set SCAPULA_OUT_DIR or pass the output directory as the first argument.")

    val regFiles = Option(pass2Dir.listFiles())
      .getOrElse(Array.empty[File])
      .filter(f => f.getName.endsWith(".stl") && f.getName.startsWith("reg_"))
      .sortBy(_.getName)
    require(regFiles.nonEmpty, s"No reg_*.stl files in ${pass2Dir.getAbsolutePath}")

    // ── Load and optionally decimate all registered meshes ────────────────────
    println(s"\nLoading ${regFiles.length} registered meshes from ${pass2Dir.getAbsolutePath}")
    val allNames  = regFiles.map(_.getName.stripSuffix(".stl").stripPrefix("reg_")).toIndexedSeq
    val rawMeshes = regFiles.map(ScapulaData.loadMesh).toIndexedSeq
    val nPts      = rawMeshes.head.pointSet.numberOfPoints
    println(s"  ${rawMeshes.length} meshes, $nPts vertices each")

    val target = Config.modelResolution
    val allMeshes = if (nPts > target) {
      println(s"Decimating $nPts → ~$target vertices (preserving correspondence)...")
      val dec = ScapulaData.decimateInCorrespondence(rawMeshes.head, rawMeshes, target)
      println(s"  Actual: ${dec.head.pointSet.numberOfPoints} vertices")
      dec
    } else {
      println(s"  Already ≤ $target vertices — skipping decimation")
      rawMeshes
    }

    // ── Local helper: run all 5 metrics for one model ─────────────────────────
    def runValidation(
        label: String,
        suffix: String,
        meshes: IndexedSeq[TriangleMesh[_3D]],
        names:  IndexedSeq[String]
    ): Unit = {

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
      println(s"\n[2/5] Generalization (full-model projection)...")
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
      val distMean = meanRows.map(_.split(",").last.toDouble)
      println(f"  mean-shape distance (avg HD) = ${distMean.sum / distMean.length}%.3f mm")
    }

    // ── SSM1: ALL specimens ───────────────────────────────────────────────────
    runValidation("SSM1 (all specimens)", "_ssm1", allMeshes, allNames)

    // ── SSM2: ONE specimen per subject (left preferred) ───────────────────────
    // groupBy subject key → take the alphabetically first (= _L when both L+R exist)
    val subjectGroups = allNames.zipWithIndex
      .groupBy { case (name, _) => ScapulaData.subjectKey(name) }
    val ssm2Indices = subjectGroups.values.map { pairs =>
      pairs.minBy(_._1)._2   // alphabetical min → _L before _R
    }.toIndexedSeq.sorted

    val ssm2Meshes = ssm2Indices.map(allMeshes(_))
    val ssm2Names  = ssm2Indices.map(allNames(_))

    println(s"\nSSM2 specimens (${ssm2Meshes.length}):")
    ssm2Names.foreach(n => println(s"  $n"))

    if (ssm2Meshes.length >= 3)
      runValidation("SSM2 (one per subject)", "_ssm2", ssm2Meshes, ssm2Names)
    else
      println("\nSSM2 skipped: fewer than 3 unique subjects.")

    // ── Final summary ─────────────────────────────────────────────────────────
    println(s"\n\nAll CSVs written to: ${plotsDir.getAbsolutePath}")
    println("\nTo plot:")
    println("  pip install matplotlib seaborn pandas numpy")
    println(s"  python3 plot_ssm_metrics.py --plots-dir ${plotsDir.getAbsolutePath}")
  }
}
