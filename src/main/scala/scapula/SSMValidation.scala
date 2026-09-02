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
 * distance to mean, and per-specimen reconstruction-error matrix.
 *
 * All results are written as CSV files to <outDir>/plots/.
 * After running this, plot everything with:
 *
 *   pip install matplotlib seaborn pandas numpy
 *   python3 plot_ssm_metrics.py --plots-dir <outDir>/plots
 *
 * Usage:
 *   sbt "runMain scapula.SSMValidation [/path/to/output/dir]"
 *   SCAPULA_OUT_DIR=/your/path sbt "runMain scapula.SSMValidation"
 *
 * Output CSV files
 * ────────────────
 *  compactness.csv                  — mode, variance_pct, cumulative_variance_pct
 *  generalization.csv               — num_modes, mean_error_mm  (full-model projection)
 *  reconstruction_error_matrix.csv  — specimen_id, num_modes, error_mm  (long format)
 *  specificity.csv                  — num_modes, mean_specificity_mm, std_specificity_mm
 *  pairwise_distances.csv           — specimen_i, specimen_j, mean_mm, rms_mm, hd95_mm
 *  distance_to_mean.csv             — specimen_id, mean_mm, rms_mm, hd95_mm, hd_mm
 */
object SSMValidation {

  private def writeRows(file: File, header: String, rows: Seq[String]): Unit = {
    file.getParentFile.mkdirs()
    Using.resource(new PrintWriter(file)) { pw =>
      pw.println(header)
      rows.foreach(pw.println)
    }
    println(f"  -> ${file.getName}%-50s ${rows.length}%4d rows")
  }

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val baseDir  = if (args.nonEmpty) new File(args(0)) else Config.outDir
    val pass2Dir = new File(baseDir, "pass2")
    val plotsDir = new File(baseDir, "plots")
    plotsDir.mkdirs()

    // ── Load registered meshes ───────────────────────────────────────────────
    require(
      pass2Dir.isDirectory,
      s"pass2/ directory not found: ${pass2Dir.getAbsolutePath}\n" +
      s"  Set SCAPULA_OUT_DIR or pass the output directory as the first argument."
    )
    val regFiles = Option(pass2Dir.listFiles())
      .getOrElse(Array.empty[File])
      .filter(f => f.getName.endsWith(".stl") && f.getName.startsWith("reg_"))
      .sortBy(_.getName)

    require(regFiles.nonEmpty, s"No reg_*.stl files found in ${pass2Dir.getAbsolutePath}")

    println(s"\nLoading ${regFiles.length} registered meshes from ${pass2Dir.getAbsolutePath}")
    val names  = regFiles.map(_.getName.stripSuffix(".stl")).toIndexedSeq
    val meshes = regFiles.map(ScapulaData.loadMesh).toIndexedSeq
    val nPts   = meshes.head.pointSet.numberOfPoints
    println(s"  ${meshes.length} meshes, $nPts vertices each")

    // ── Build full SSM ───────────────────────────────────────────────────────
    println("\nBuilding SSM via PCA...")
    val t0    = System.currentTimeMillis()
    val dc    = DataCollection.fromTriangleMesh3DSequence(meshes.head, meshes)
    val model = PointDistributionModel.createUsingPCA(dc)
    println(f"  rank = ${model.rank}  |  elapsed = ${(System.currentTimeMillis() - t0) / 1000.0}%.1f s")

    val nModes      = model.rank
    val eigenvalues = model.gp.eigenvalues.toArray
    val totalVar    = eigenvalues.sum
    val meanMesh    = model.mean

    // ── 1. Compactness ───────────────────────────────────────────────────────
    println("\n[1/5] Compactness")
    val cumVar = eigenvalues.scanLeft(0.0)(_ + _).tail.map(_ / totalVar * 100.0)
    writeRows(
      new File(plotsDir, "compactness.csv"),
      "mode,variance_pct,cumulative_variance_pct",
      (1 to nModes).map { k =>
        f"$k,${eigenvalues(k - 1) / totalVar * 100.0}%.4f,${cumVar(k - 1)}%.4f"
      }
    )
    def modesFor(pct: Double): String = {
      val i = cumVar.indexWhere(_ >= pct); if (i < 0) s">$nModes" else s"${i + 1}"
    }
    println(f"  90%% variance: ${modesFor(90)} modes  |  95%%: ${modesFor(95)}  |  99%%: ${modesFor(99)}")

    // ── 2. Generalization (full-model projection, bias-aware note below) ─────
    // Full-model generalization is optimistic (training data is projected onto
    // its own model).  For a fully unbiased estimate use leave-one-out; for
    // practical SSM work with N<=30, this in-sample version is the standard
    // first look.
    println("\n[2/5] Generalization (full-model projection)...")
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

    val genSummary = genData.groupBy(_._1).toSeq.sortBy(_._1).map { case (k, entries) =>
      f"$k,${entries.map(_._3).sum / entries.length}%.4f"
    }
    writeRows(new File(plotsDir, "generalization.csv"), "num_modes,mean_error_mm", genSummary)

    writeRows(
      new File(plotsDir, "reconstruction_error_matrix.csv"),
      "specimen_id,num_modes,error_mm",
      genData.map { case (k, name, err) => f"$name,$k,$err%.4f" }
    )

    // ── 3. Specificity ───────────────────────────────────────────────────────
    // For each mode count k, sample 200 random shapes from the k-mode model
    // and report the mean minimum distance to the nearest training specimen.
    // Lower is better (sampled shapes stay close to real anatomy).
    println("\n[3/5] Specificity (200 samples per mode count)...")
    val nSamples = 200
    val specRows = (1 to nModes).map { k =>
      if (k == 1 || k % 5 == 0 || k == nModes) print(s"  k=$k")
      val kModel   = model.truncate(k)
      val minDists = (0 until nSamples).map { _ =>
        val s = kModel.sample()
        meshes.map { m =>
          val d = Metrics.correspondingDistances(s, m)
          d.sum / d.length
        }.min
      }
      val mean = minDists.sum / minDists.length
      val std  = math.sqrt(minDists.map(d => (d - mean) * (d - mean)).sum / minDists.length)
      f"$k,$mean%.4f,$std%.4f"
    }
    println()
    writeRows(new File(plotsDir, "specificity.csv"),
              "num_modes,mean_specificity_mm,std_specificity_mm", specRows)

    // ── 4. Pairwise distances ────────────────────────────────────────────────
    println("\n[4/5] Pairwise point-to-point distances...")
    val n        = meshes.length
    val pairRows = (for {
      i <- 0 until n
      j <- i + 1 until n
    } yield {
      val dists = Metrics.correspondingDistances(meshes(i), meshes(j))
      val mean  = dists.sum / dists.length
      val rms   = math.sqrt(dists.map(d => d * d).sum / dists.length)
      val hd95  = Metrics.percentile(dists, 0.95)
      f"${names(i)},${names(j)},$mean%.4f,$rms%.4f,$hd95%.4f"
    }).toSeq
    writeRows(new File(plotsDir, "pairwise_distances.csv"),
              "specimen_i,specimen_j,mean_mm,rms_mm,hd95_mm", pairRows)

    // ── 5. Distance to mean shape ────────────────────────────────────────────
    println("\n[5/5] Distance to mean shape...")
    val meanRows = meshes.zip(names).map { case (m, name) =>
      val dists = Metrics.correspondingDistances(m, meanMesh)
      val mean  = dists.sum / dists.length
      val rms   = math.sqrt(dists.map(d => d * d).sum / dists.length)
      val hd95  = Metrics.percentile(dists, 0.95)
      val hd    = dists.max
      f"$name,$mean%.4f,$rms%.4f,$hd95%.4f,$hd%.4f"
    }
    writeRows(new File(plotsDir, "distance_to_mean.csv"),
              "specimen_id,mean_mm,rms_mm,hd95_mm,hd_mm", meanRows)

    // ── Summary ──────────────────────────────────────────────────────────────
    println(s"\nAll metrics written to: ${plotsDir.getAbsolutePath}")
    println("\nNext steps:")
    println("  pip install matplotlib seaborn pandas numpy")
    println(s"  python3 plot_ssm_metrics.py --plots-dir ${plotsDir.getAbsolutePath}")
  }
}
