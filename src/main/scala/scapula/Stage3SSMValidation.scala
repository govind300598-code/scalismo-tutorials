package scapula

import scalismo.geometry.*
import scalismo.mesh.*
import scalismo.io.MeshIO
import scalismo.utils.Random
import breeze.linalg.*

import java.io.{File, PrintWriter}

/**
 * Stage 3 – SSM validation: compactness, generalization, specificity.
 *
 * Loads all *_fit.stl meshes from SCAPULA_OUT_DIR (excluding the three
 * known outliers), splits into hillsachs / paired groups, builds one PCA
 * model per group and computes the three standard SSM quality curves.
 *
 * Specificity uses 100,000 random samples evaluated with the exact RMS
 * formula in latent space:
 *   dist_rms(α, β_j)² = (1/N) · Σ_k λ_k (α_k − β_{j,k})²
 * where α_k are the whitened sample coefficients (~N(0,1)) and β_{j,k}
 * are the whitened training-shape coefficients.  This is O(samples × rank
 * × N_train) – no per-sample mesh I/O – so 100 K samples finish in seconds.
 *
 * Writes one CSV per (group × metric) to SCAPULA_OUT_DIR, then prints
 * a summary table.  Run scripts/plot_ssm_validation.py afterwards to
 * produce publication-ready validation figures.
 */
object Stage3SSMValidation {

  private val outlierIds: Set[String] = Set(
    "hill_sachs_032_M_50_R",
    "hill_sachs_033_F_53_R",
    "hill_sachs_042_M_29_L"
  )

  // ── entry point ────────────────────────────────────────────────────────────

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val outDir = Config.outDir
    println(s"Output directory : ${outDir.getAbsolutePath}")

    val fitFiles = Option(outDir.listFiles())
      .getOrElse(Array.empty[File])
      .filter(_.getName.endsWith("_fit.stl"))
      .filterNot(f => outlierIds.exists(id => f.getName.startsWith(id)))
      .sortBy(_.getName)

    println(s"Fit meshes found : ${fitFiles.length}  (outliers excluded: ${outlierIds.size})")
    if (fitFiles.isEmpty) sys.error("No _fit.stl files found – run Stage3PCAModel first.")

    println("Loading meshes …")
    val allShapes = fitFiles.map { f =>
      val id = f.getName.stripSuffix("_fit.stl")
      id -> loadMesh(f)
    }.toIndexedSeq

    println(f"Points per mesh  : ${allShapes.head._2.pointSet.numberOfPoints}")

    val hillsachs = allShapes.filter(_._1.startsWith("hill_sachs")).map(_._2)
    val paired    = allShapes.filter { case (id, _) =>
      id.startsWith("paired_scapula") || id.startsWith("paired_shoulder")
    }.map(_._2)

    println(f"hillsachs group  : N=${hillsachs.length}")
    println(f"paired group     : N=${paired.length}")
    println()

    val line = "=" * 72
    println(line)
    val results = Seq("hillsachs" -> hillsachs, "paired" -> paired).map {
      case (name, shapes) =>
        val row = validateGroup(name, shapes, outDir, nSpecSamples = 100_000)
        println()
        row
    }

    println(line)
    println(f"${"GROUP"}%-12s  ${"N"}%4s  ${"RANK"}%5s  ${"90%"}%6s  ${"95%"}%6s  ${"Gen@rank(mm)"}%14s  ${"Spec@rank(mm)"}%14s")
    println("-" * 72)
    results.foreach { r =>
      println(f"${r.name}%-12s  ${r.n}%4d  ${r.rank}%5d  ${r.modes90}%6d  ${r.modes95}%6d  ${r.genAtRank}%14.3f  ${r.specAtRank}%14.3f")
    }
    println(line)
    println()
    println("All CSV files written. Run:  python3 scripts/plot_ssm_validation.py")
  }

  // ── result record ──────────────────────────────────────────────────────────

  case class GroupResult(
    name: String, n: Int, rank: Int,
    modes90: Int, modes95: Int,
    genAtRank: Double, specAtRank: Double
  )

  // ── per-group validation ───────────────────────────────────────────────────

  def validateGroup(
    name: String,
    shapes: IndexedSeq[TriangleMesh[_3D]],
    outDir: File,
    nSpecSamples: Int
  )(implicit rng: Random): GroupResult = {

    val n    = shapes.length
    val nPts = shapes.head.pointSet.numberOfPoints
    val nDim = nPts * 3

    println(s"[$name] Building PCA from $n shapes ($nDim-dim space) …")

    // ── flatten each mesh to a row of the data matrix ─────────────────────────
    val X = DenseMatrix.zeros[Double](n, nDim)
    shapes.zipWithIndex.foreach { case (mesh, i) =>
      var col = 0
      mesh.pointSet.points.foreach { p =>
        X(i, col)   = p.x
        X(i, col+1) = p.y
        X(i, col+2) = p.z
        col += 3
      }
    }

    // ── column-wise mean and centring ──────────────────────────────────────────
    val meanVec = DenseVector.zeros[Double](nDim)
    for (i <- 0 until n) meanVec += X(i, ::).t
    meanVec /= n.toDouble
    for (i <- 0 until n) X(i, ::) -= meanVec.t

    // ── economy SVD: X/√(n-1) of shape (n × nDim), n << nDim ─────────────────
    // U: n×n,  s: n,  Vt: n×nDim
    val Xscaled = X / math.sqrt(n - 1)
    val decomp  = svd.reduced(Xscaled)
    val U = decomp.U   // n × n
    val s = decomp.S   // length n

    // Eigenvalues λ_k = s_k²  (mm² per point equivalent)
    val rank  = n - 1  // last singular value ≈ 0 due to centering
    val lambdas: Array[Double] = s.toArray.take(rank).map(sv => sv * sv)
    val totalVar = lambdas.sum

    val pc1 = lambdas(0) / totalVar * 100
    val pc2 = lambdas(1) / totalVar * 100
    val pc3 = lambdas(2) / totalVar * 100
    println(f"[$name] rank=$rank  PC1-3: $pc1%.1f%%  $pc2%.1f%%  $pc3%.1f%%  (total var=${totalVar / 1e6}%.2f ×10⁶ mm²)")

    // Whitened training coefficients β_{i,k} = U(i,k).
    // Physical coef = U(i,k) * s(k).
    // RMS distance²: Σ_k λ_k (α_k − β_{j,k})²  /  nPts
    val beta: Array[Array[Double]] =
      (0 until n).map(i => (0 until rank).map(k => U(i, k)).toArray).toArray

    // ── compactness ────────────────────────────────────────────────────────────
    val compData: IndexedSeq[(Int, Double)] = (1 to rank).map { k =>
      k -> lambdas.take(k).sum / totalVar * 100.0
    }
    val modes90 = compData.find(_._2 >= 90.0).map(_._1).getOrElse(rank)
    val modes95 = compData.find(_._2 >= 95.0).map(_._1).getOrElse(rank)
    writeCsv(outDir, s"ssm_validation_${name}_compactness.csv",
      "modes,cumulative_variance_pct",
      compData.map { case (k, v) => f"$k,$v%.6f" })
    println(f"[$name] Compactness  90%%→$modes90 modes   95%%→$modes95 modes")

    // ── generalization (projection error vs k modes) ───────────────────────────
    // For training shape i projected onto first k modes:
    //   err_rms_i(k) = sqrt( Σ_{j≥k} λ_j β_{i,j}² / nPts )
    val genData: IndexedSeq[(Int, Double)] = (1 to rank).map { k =>
      val meanErr = beta.map { b =>
        val unexplained = (k until rank).map(j => lambdas(j) * b(j) * b(j)).sum
        math.sqrt(unexplained / nPts)
      }.sum / n
      k -> meanErr
    }
    val genAtRank = genData.last._2
    writeCsv(outDir, s"ssm_validation_${name}_generalization.csv",
      "modes,mean_rms_error_mm",
      genData.map { case (k, v) => f"$k,$v%.6f" })
    println(f"[$name] Generalization at full rank: $genAtRank%.4f mm")

    // ── specificity (100K samples) ────────────────────────────────────────────
    // Curve sampled at a handful of key mode counts plus full rank.
    val modeCurve: IndexedSeq[Int] = {
      val pts = (1 to math.min(10, rank)).toSeq ++
                (15 to rank by 5).toSeq :+ rank
      pts.filter(_ <= rank).distinct.sorted.toIndexedSeq
    }

    println(f"[$name] Specificity: $nSpecSamples samples × ${modeCurve.length} mode-counts …")

    val specData: IndexedSeq[(Int, Double)] = modeCurve.map { k =>
      // Pre-compute tail contribution for each training shape:
      //   tailSq_j = Σ_{m=k}^{rank-1} λ_m β_{j,m}²  (α_m = 0 for m ≥ k)
      val tailSq: Array[Double] = beta.map { b =>
        var s = 0.0; var m = k; while (m < rank) { s += lambdas(m) * b(m) * b(m); m += 1 }; s
      }

      val javaRng = new java.util.Random(Config.seed + 7L * name.hashCode.toLong + k.toLong)

      var sumMinRms = 0.0
      var sampleIdx = 0
      while (sampleIdx < nSpecSamples) {
        // Draw one sample α (only first k components matter)
        val alpha = Array.fill(k)(javaRng.nextGaussian())

        // Find nearest training shape
        var minDist2 = Double.MaxValue
        var j = 0
        while (j < n) {
          // head: Σ_{m=0}^{k-1} λ_m (α_m − β_{j,m})²
          var d2 = tailSq(j)
          var m = 0
          while (m < k) {
            val diff = alpha(m) - beta(j)(m)
            d2 += lambdas(m) * diff * diff
            m += 1
          }
          if (d2 < minDist2) minDist2 = d2
          j += 1
        }
        sumMinRms += math.sqrt(minDist2 / nPts)
        sampleIdx += 1
      }
      k -> (sumMinRms / nSpecSamples)
    }

    val specAtRank = specData.last._2
    writeCsv(outDir, s"ssm_validation_${name}_specificity.csv",
      "modes,mean_min_rms_mm",
      specData.map { case (k, v) => f"$k,$v%.6f" })
    println(f"[$name] Specificity at full rank ($nSpecSamples samples): $specAtRank%.4f mm")

    // ── per-mode variance table ───────────────────────────────────────────────
    writeCsv(outDir, s"ssm_validation_${name}_variance.csv",
      "mode,eigenvalue_mm2,variance_pct,cumulative_pct",
      lambdas.zipWithIndex.map { case (lk, ki) =>
        val pct = lk / totalVar * 100.0
        val cum = lambdas.take(ki + 1).sum / totalVar * 100.0
        f"${ki+1},$lk%.4f,$pct%.4f,$cum%.4f"
      }.toIndexedSeq)

    GroupResult(name, n, rank, modes90, modes95, genAtRank, specAtRank)
  }

  // ── helpers ──────────────────────────────────────────────────────────────────

  private def loadMesh(f: File): TriangleMesh[_3D] =
    MeshIO.readMesh(f).getOrElse(throw new RuntimeException(s"Cannot read ${f.getName}"))

  private def writeCsv(dir: File, filename: String, header: String, rows: Seq[String]): Unit = {
    val f  = new File(dir, filename)
    val pw = new PrintWriter(f)
    try { pw.println(header); rows.foreach(pw.println) }
    finally pw.close()
    println(s"  wrote: $filename")
  }
}
