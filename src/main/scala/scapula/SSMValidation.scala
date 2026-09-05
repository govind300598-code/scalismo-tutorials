package scapula

import breeze.linalg.DenseVector
import breeze.stats.distributions.Gaussian
import scalismo.geometry._3D
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.mesh.TriangleMesh
import scalismo.numerics.PivotedCholesky
import scalismo.statisticalmodel.StatisticalMeshModel
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.utils.Random

import java.io.{File, FileNotFoundException, PrintWriter}

/**
 * Compute per-mode SSM validation metrics for SSM1–SSM4 and write CSV files.
 *
 * Run with:
 *   sbt "runMain scapula.SSMValidation"
 *
 * Prerequisites: the main pipeline must have been run first.
 * Reads:  <SCAPULA_OUT_DIR>/results/SSM{1..4}/model/SSM{n}.h5
 *         <SCAPULA_OUT_DIR>/results/SSM{1..4}/nonrigid_registered/*.stl
 * Writes: <SCAPULA_OUT_DIR>/validation/compactness.csv
 *         <SCAPULA_OUT_DIR>/validation/generalization.csv
 *         <SCAPULA_OUT_DIR>/validation/specificity.csv
 *
 * Then run ssm_validation.py to produce the publication figure.
 */
object SSMValidation {

  private val N_SPEC_SAMPLES = 50  // random shapes per mode for specificity

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val outDir     = Config.outDir
    val resultsDir = new File(outDir, "results")
    val valDir     = new File(outDir, "validation")
    valDir.mkdirs()

    val ssmLabels = Seq("SSM1", "SSM2", "SSM3", "SSM4")

    // ── Detect folder layout (new results/SSMn/ vs old pass{n}/) ──────────────
    // New layout: outDir/results/SSM{n}/model/SSM{n}.h5
    // Old layout: outDir/pass{n}/reg_*.stl  (RebuildSSM output)
    def stlsIn(dir: File, prefix: String = ""): IndexedSeq[File] =
      if (!dir.isDirectory) IndexedSeq.empty
      else Option(dir.listFiles()).getOrElse(Array.empty)
        .filter(f => f.getName.endsWith(".stl") && f.getName.startsWith(prefix))
        .sorted.toIndexedSeq

    val useNewLayout = new File(resultsDir, "SSM1").isDirectory

    // ── Load saved models and registered meshes ────────────────────────────────
    val ssmData = ssmLabels.zipWithIndex.map { case (label, idx) =>
      val passNum = idx + 1
      println(s"Loading $label ...")

      val meshes: IndexedSeq[TriangleMesh[_3D]] = if (useNewLayout) {
        val regDir   = new File(resultsDir, s"$label/nonrigid_registered")
        if (!regDir.isDirectory) throw new FileNotFoundException(
          s"Directory not found: $regDir\n  → Run 'sbt runMain scapula.Main' first.")
        val stlFiles = stlsIn(regDir)
        require(stlFiles.nonEmpty, s"No registered STLs in $regDir")
        stlFiles.map(f => MeshIO.readMesh(f).getOrElse(throw new RuntimeException(s"Cannot load: $f")))
      } else {
        // Old layout: outDir/pass{n}/reg_*.stl
        val passDir  = new File(outDir, s"pass$passNum")
        if (!passDir.isDirectory) throw new FileNotFoundException(
          s"Pass directory not found: $passDir")
        val stlFiles = stlsIn(passDir, prefix = "reg_")
        require(stlFiles.nonEmpty, s"No reg_*.stl found in $passDir")
        stlFiles.map(f => MeshIO.readMesh(f).getOrElse(throw new RuntimeException(s"Cannot load: $f")))
      }
      println(s"  ${meshes.length} registered meshes loaded")

      // Load or build the SSM
      val model: StatisticalMeshModel = if (useNewLayout) {
        val modelFile = new File(resultsDir, s"$label/model/$label.h5")
        if (!modelFile.exists()) throw new FileNotFoundException(s"Model not found: $modelFile")
        StatisticalModelIO.readStatisticalMeshModel(modelFile)
          .getOrElse(throw new RuntimeException(s"Cannot load: $modelFile"))
      } else {
        // Build SSM on-the-fly from the registered meshes via PCA
        println(s"  Building $label SSM from ${meshes.length} meshes via PCA...")
        val dc = DataCollection.fromTriangleMesh3DSequence(meshes.head, meshes)
        StatisticalMeshModel.createUsingPCA(dc)
          .getOrElse(throw new RuntimeException(s"PCA failed for $label"))
      }
      println(s"  rank = ${model.rank}")

      (label, model, meshes)
    }

    // Use min rank minus 1 so all SSMs have the same x-axis
    val maxModes = ssmData.map(_._2.rank).min - 1
    val modeIdxs = (1 to maxModes).toIndexedSeq
    println(s"\nComputing metrics for modes 1 .. $maxModes\n")

    // ── A. Compactness ─────────────────────────────────────────────────────────
    println("=" * 60)
    println("[A] COMPACTNESS")
    println("=" * 60)
    val compactness: Map[String, IndexedSeq[Double]] = ssmData.map { case (label, model, _) =>
      val evs    = model.gp.klBasis.map(_.eigenvalue)
      val total  = evs.sum
      val cumPct = evs.scanLeft(0.0)(_ + _).tail.map(_ / total * 100.0).take(maxModes).toIndexedSeq
      println(s"  $label: mode1=${f"${cumPct(0)}%.1f"}%  top5=${f"${cumPct.lift(4).getOrElse(cumPct.last)}%.1f"}%")
      label -> cumPct
    }.toMap
    writeCsv(new File(valDir, "compactness.csv"), modeIdxs, ssmLabels, compactness)
    println("  → compactness.csv")

    // ── B. Generalization ──────────────────────────────────────────────────────
    println("\n" + "=" * 60)
    println("[B] GENERALIZATION (leave-one-out)")
    println("=" * 60)
    val generalization: Map[String, IndexedSeq[Double]] = ssmData.map { case (label, _, meshes) =>
      println(s"  $label: ${meshes.length} folds × $maxModes modes")
      val perMode = computeLooGeneralization(meshes, maxModes)
      modeIdxs.zip(perMode).foreach { case (m, e) => println(f"    mode $m%2d: $e%.3f mm") }
      label -> perMode
    }.toMap
    writeCsv(new File(valDir, "generalization.csv"), modeIdxs, ssmLabels, generalization)
    println("  → generalization.csv")

    // ── C. Specificity ─────────────────────────────────────────────────────────
    println("\n" + "=" * 60)
    println("[C] SPECIFICITY")
    println("=" * 60)
    val specificity: Map[String, IndexedSeq[Double]] = ssmData.map { case (label, model, meshes) =>
      println(s"  $label: $maxModes modes × $N_SPEC_SAMPLES samples each")
      val perMode = modeIdxs.map { k =>
        val e = specificityAtK(model, meshes, k)
        println(f"    mode $k%2d: $e%.3f mm")
        e
      }
      label -> perMode
    }.toMap
    writeCsv(new File(valDir, "specificity.csv"), modeIdxs, ssmLabels, specificity)
    println("  → specificity.csv")

    println(s"\n${"=" * 60}")
    println(s"Done.  Results in: ${valDir.getAbsolutePath}")
    println("Next: python ssm_validation.py")
    println("=" * 60)
  }

  // ---------------------------------------------------------------------------
  // B. LOO generalization – per mode
  // Build one DataCollection per fold, then call createUsingPCA with k modes for each k.
  // ---------------------------------------------------------------------------

  def computeLooGeneralization(
    meshes: IndexedSeq[TriangleMesh[_3D]],
    maxModes: Int
  )(implicit rng: Random): IndexedSeq[Double] = {
    val n         = meshes.length
    val reference = meshes.head
    val errSum    = Array.fill(maxModes)(0.0)

    meshes.indices.foreach { i =>
      val testMesh = meshes(i)
      val trainSet = meshes.patch(i, Nil, 1)
      val dc       = DataCollection.fromTriangleMesh3DSequence(reference, trainSet)

      val observations = testMesh.pointSet.pointIds.toIndexedSeq.map { id =>
        (id, testMesh.pointSet.point(id))
      }

      print(s"    fold ${i + 1}/$n: ")
      (1 to maxModes).foreach { k =>
        val kModel = StatisticalMeshModel
          .createUsingPCA(dc, PivotedCholesky.NumberOfEigenfunctions(k))
          .getOrElse(throw new RuntimeException(s"PCA failed at k=$k fold=$i"))

        val recon = kModel.posterior(observations, sigma2 = 0.5).mean
        val d     = Metrics.surfaceDistances(testMesh, recon)
        errSum(k - 1) += d.sum / d.length
        print(k)
        if (k < maxModes) print(" ")
      }
      println()
    }

    errSum.toIndexedSeq.map(_ / n)
  }

  // ---------------------------------------------------------------------------
  // C. Specificity – per mode
  // Sample from first-k modes of the full model; measure mean distance to nearest
  // training shape.
  // ---------------------------------------------------------------------------

  def specificityAtK(
    model: StatisticalMeshModel,
    trainMeshes: IndexedSeq[TriangleMesh[_3D]],
    k: Int
  )(implicit rng: Random): Double = {
    val kClamp = k.min(model.rank)
    val normal = Gaussian(0, 1)(rng.breezeRandBasis)

    val errors = (0 until N_SPEC_SAMPLES).map { _ =>
      val sample = sampleKModes(model, kClamp, normal)
      trainMeshes.map { ref =>
        val d = Metrics.surfaceDistances(sample, ref)
        d.sum / d.length
      }.min
    }
    errors.sum / errors.length
  }

  /**
   * Sample a random instance using only the first k modes of the model.
   * Coefficient for mode j ~ N(0, eigenvalue_j) so its contribution has the correct variance.
   */
  private def sampleKModes(
    model: StatisticalMeshModel,
    k: Int,
    normal: Gaussian
  ): TriangleMesh[_3D] = {
    val coeffs = DenseVector(
      Array.tabulate(model.rank) { j =>
        if (j < k) normal.draw() * math.sqrt(model.gp.klBasis(j).eigenvalue)
        else 0.0
      }
    )
    model.instance(coeffs)
  }

  // ---------------------------------------------------------------------------
  // CSV writer
  // ---------------------------------------------------------------------------

  private def writeCsv(
    file: File,
    modes: IndexedSeq[Int],
    labels: Seq[String],
    data: Map[String, IndexedSeq[Double]]
  ): Unit = {
    val pw = new PrintWriter(file)
    pw.println("mode," + labels.mkString(","))
    modes.zipWithIndex.foreach { case (m, idx) =>
      val row = labels.map(l => f"${data(l)(idx)}%.6f").mkString(",")
      pw.println(s"$m,$row")
    }
    pw.close()
  }
}
