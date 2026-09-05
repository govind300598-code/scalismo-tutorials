package scapula

import scalismo.geometry._3D
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.utils.Random

import java.io.{File, PrintWriter}
import scala.util.Using

/**
 * SSM validation across ALL passes produced by RebuildSSM.
 *
 * Discovers every pass<N>/ directory under <outDir> and, for each one,
 * builds SSM<N> and computes the full validation suite:
 *
 *  1. Compactness       — cumulative % variance vs mode count
 *  2. Generalization    — projection reconstruction error (all modes)
 *  3. Specificity       — min-distance of random samples to training set
 *  4. Pairwise distances — point-to-point between all specimen pairs
 *  5. Distance to mean  — point-to-point each specimen vs SSM mean
 *
 * Then prints a side-by-side comparison table (SSM1 vs SSM2 vs SSM3 vs SSM4)
 * and a consecutive mean-shape stability table (confirms GPA convergence).
 *
 * Typical interpretation:
 *   SSM1  — high reference bias, less compact
 *   SSM2  — reference = SSM1 mean; bias largely removed
 *   SSM3+ — fine-tuning; diminishing returns when mean shift < 1 mm
 *
 * Usage:
 *   sbt "runMain scapula.SSMValidation [/path/to/output/dir]"
 *   SCAPULA_OUT_DIR=/your/path sbt "runMain scapula.SSMValidation"
 *
 * Output CSV files (all in <outDir>/plots/, suffix _pass<N>)
 * ─────────────────────────────────────────────────────────────
 *   compactness_pass<N>.csv
 *   generalization_pass<N>.csv
 *   reconstruction_error_matrix_pass<N>.csv
 *   specificity_pass<N>.csv
 *   pairwise_distances_pass<N>.csv
 *   distance_to_mean_pass<N>.csv
 *   stability.csv                        consecutive mean-shape distances
 */
object SSMValidation {

  private def writeRows(file: File, header: String, rows: Seq[String]): Unit = {
    file.getParentFile.mkdirs()
    Using.resource(new PrintWriter(file)) { pw =>
      pw.println(header)
      rows.foreach(pw.println)
    }
    println(f"  -> ${file.getName}%-60s ${rows.length}%4d rows")
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
    if (nPts > target) {
      println(s"  Decimating $nPts → ~$target vertices (preserving correspondence)...")
      val dec = ScapulaData.decimateInCorrespondence(rawMeshes.head, rawMeshes, target)
      println(s"  Actual: ${dec.head.pointSet.numberOfPoints} vertices")
      (dec, names)
    } else {
      println(s"  Already ≤ $target vertices — skipping decimation")
      (rawMeshes, names)
    }
  }

  // Run all 5 validation metrics for one SSM; returns the built model.
  private def runValidation(
      label:    String,
      suffix:   String,
      meshes:   IndexedSeq[TriangleMesh[_3D]],
      names:    IndexedSeq[String],
      plotsDir: File
  )(implicit rng: Random): PointDistributionModel[_3D, TriangleMesh] = {

    println(s"\n╔══════════════════════════════════════════════════════════╗")
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
        f"$k,${eigenvalues(k-1) / totalVar * 100.0}%.4f,${cumVar(k-1)}%.4f"
      })
    def modesFor(pct: Double): String = {
      val i = cumVar.indexWhere(_ >= pct); if (i < 0) s">$nModes" else s"${i+1}"
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
    println(s"\n[3/5] Specificity (200 random samples per mode count)...")
    val nSamples = 200
    val specRows = (1 to nModes).map { k =>
      if (k == 1 || k % 5 == 0 || k == nModes) print(s"  k=$k")
      val kModel   = model.truncate(k)
      val minDists = (0 until nSamples).map { _ =>
        val s = kModel.sample()
        meshes.map { m => val d = Metrics.correspondingDistances(s, m); d.sum / d.length }.min
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

    // Summary line
    val avg90 = cumVar.indexWhere(_ >= 90.0)
    println(s"\n$label summary:")
    println(f"  specimens=${meshes.length}  rank=$nModes  modes@90%%=${if (avg90 < 0) nModes else avg90+1}")
    val hdVals = meanRows.map(_.split(",").last.toDouble)
    println(f"  avg Hausdorff dist to mean = ${hdVals.sum / hdVals.length}%.3f mm")

    model
  }

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val baseDir  = if (args.nonEmpty) new File(args(0)) else Config.outDir
    val plotsDir = new File(baseDir, "plots")
    plotsDir.mkdirs()

    // ── Discover all pass<N>/ directories (sorted by N) ──────────────────────
    val passDirs: IndexedSeq[(Int, File)] = {
      val all = Option(baseDir.listFiles()).getOrElse(Array.empty[File])
        .filter(f => f.isDirectory && f.getName.matches("pass\\d+"))
        .sortBy(f => f.getName.drop(4).toInt)
        .toIndexedSeq
      require(all.nonEmpty,
        s"No pass<N>/ directories found in ${baseDir.getAbsolutePath}\n" +
        s"  Run 'sbt runMain scapula.RebuildSSM' first.")
      all.map(f => f.getName.drop(4).toInt -> f)
    }
    println(s"\nFound ${passDirs.length} pass director(ies) under ${baseDir.getAbsolutePath}:")
    passDirs.foreach { case (n, d) => println(s"  pass$n/ → ${d.getAbsolutePath}") }

    // ── Load meshes for every pass ────────────────────────────────────────────
    println(s"\nLoading registered meshes...")
    val passData: IndexedSeq[(Int, IndexedSeq[TriangleMesh[_3D]], IndexedSeq[String])] =
      passDirs.map { case (n, dir) =>
        val (meshes, names) = loadAndDecimate(dir, s"SSM$n (pass$n)")
        (n, meshes, names)
      }

    // ── Run validation for each pass ──────────────────────────────────────────
    val models: IndexedSeq[(Int, PointDistributionModel[_3D, TriangleMesh])] =
      passData.map { case (n, meshes, names) =>
        val model = runValidation(
          s"SSM$n  (pass $n — ${if (n == 1) "initial reference" else s"SSM${n-1} mean reference"})",
          s"_pass$n",
          meshes, names, plotsDir)
        (n, model)
      }

    // ── Mean-shape stability across consecutive passes ────────────────────────
    if (models.length >= 2) {
      println(s"\n\n─── Mean-shape convergence (consecutive passes) ───")
      println("  Surface distance between each pair of consecutive SSM means.")
      println("  < 1 mm mean shift = reference bias has been removed.\n")

      val stabilityRows = scala.collection.mutable.ArrayBuffer.empty[String]
      for (k <- 1 until models.length) {
        val (n1, m1) = models(k - 1)
        val (n2, m2) = models(k)
        val st = Metrics.symmetric(m1.mean, m2.mean)
        val tag = if (st.mean < 1.0) "✓ CONVERGED" else "  not yet  "
        println(f"  SSM$n1 mean ↔ SSM$n2 mean:  ${st.render}  $tag")
        stabilityRows += f"$n1,$n2,${st.mean}%.4f,${st.rms}%.4f,${st.hd95}%.4f,${st.hd}%.4f"
      }
      writeRows(new File(plotsDir, "stability.csv"),
        "pass_from,pass_to,mean_mm,rms_mm,hd95_mm,hd_mm", stabilityRows.toSeq)
    }

    // ── Side-by-side comparison table ─────────────────────────────────────────
    println(s"\n\n══════════════════════════════════════════════════════════════════")
    print(f"  ${"Metric"}%-34s")
    models.foreach { case (n, _) => print(f"  ${"SSM"+n}%8s") }
    println()
    println(s"══════════════════════════════════════════════════════════════════")

    // Rank
    print(f"  ${"PCA rank"}%-34s")
    models.foreach { case (_, m) => print(f"  ${m.rank}%8d") }
    println()

    // Modes for 90 / 95 / 99 % variance
    for (pct <- Seq(90, 95, 99)) {
      print(f"  ${s"Modes for $pct%% variance"}%-34s")
      models.foreach { case (_, m) =>
        val ev = m.gp.klBasis.map(_.eigenvalue).toArray
        val cv = ev.scanLeft(0.0)(_ + _).tail.map(_ / ev.sum * 100.0)
        val k  = { val i = cv.indexWhere(_ >= pct); if (i < 0) m.rank else i + 1 }
        print(f"  $k%8d")
      }
      println()
    }

    // Mean-shift vs first model
    if (models.length >= 2) {
      val (_, mRef) = models.head
      print(f"  ${"Mean shift vs SSM1 (mm)"}%-34s")
      models.foreach { case (_, m) =>
        if (m eq mRef) print(f"  ${"—"}%8s")
        else {
          val st = Metrics.symmetric(mRef.mean, m.mean)
          print(f"  ${st.mean}%8.3f")
        }
      }
      println()
    }

    println(s"══════════════════════════════════════════════════════════════════")
    println(s"  SSM1 = pass1 (arbitrary initial reference)")
    for (k <- 2 to models.length)
      println(s"  SSM$k = pass$k (reference = SSM${k-1} mean shape)")
    println(s"══════════════════════════════════════════════════════════════════")

    // ── Final summary ─────────────────────────────────────────────────────────
    println(s"\nAll CSVs written to: ${plotsDir.getAbsolutePath}")
    println("\nTo plot:")
    println("  pip install matplotlib seaborn pandas numpy")
    println(s"  python3 plot_ssm_metrics.py --plots-dir ${plotsDir.getAbsolutePath}")
  }
}
