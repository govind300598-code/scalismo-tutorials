package scapula

import scalismo.geometry.*
import scalismo.mesh.*
import scalismo.statisticalmodel.*
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.utils.Random

import java.io.{File, PrintWriter}

/**
 * STAGE 3 -- BUILD THE STATISTICAL SHAPE MODEL.
 *
 * scalismo tutorial "Building a shape model from data", unchanged:
 *
 *     val dc    = DataCollection.fromTriangleMesh3DSequence(reference, correspondingMeshes)
 *     val model = PointDistributionModel.createUsingPCA(dc)
 *
 * Everything after that (compactness / generalization / specificity) is the standard SSM quality triad
 * (Styner et al. 2003 / Davies et al.) -- not part of the basic tutorial API, but the standard way to report
 * whether a model is any good, and worth having in the same run.
 *
 * IMPORTANT: generalization (LOOCV) is computed per SUBJECT, not per scapula. A left scapula and its own
 * mirrored right scapula are near-duplicates, not independent samples (see ScapulaData.Config docstring on
 * buildIndependentModel). If you leave out one side of a subject while its mirror twin stays in the training
 * set, the model has effectively already seen the test shape -- the LOOCV score would be optimistic. Removing
 * both sides of a subject together avoids that leakage regardless of whether Stage 2 trained on one side or both.
 */
object Stage3BuildModel {

  def pointToPointRMSE(a: TriangleMesh[_3D], b: TriangleMesh[_3D]): Double = {
    require(a.pointSet.numberOfPoints == b.pointSet.numberOfPoints, "meshes are not in correspondence")
    val sq = a.pointSet.points.zip(b.pointSet.points).map { case (p, q) => (p - q).norm2 }.toIndexedSeq
    math.sqrt(sq.sum / sq.length)
  }

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val outDir = Config.outDir
    val corrDir = new File(outDir, "corresponded")
    val referenceFile = new File(outDir, "reference.stl")
    require(referenceFile.exists(), s"${referenceFile.getPath} not found -- run Stage2Registration first")
    val referenceMesh = MeshIO.readMesh(referenceFile).get

    val corresponded = Option(corrDir.listFiles())
      .getOrElse(Array.empty[File])
      .filter(_.getName.endsWith(".stl"))
      .sortBy(_.getName)
      .map(f => f.getName.stripSuffix(".stl") -> MeshIO.readMesh(f).get)
      .toIndexedSeq
    require(corresponded.nonEmpty, s"no corresponded meshes in ${corrDir.getPath} -- run Stage2Registration first")
    println(s"[Stage3] ${corresponded.length} corresponded specimens loaded from ${corrDir.getPath}")

    // ---- scalismo tutorial "Building a shape model from data" ---------------------------------------------------
    val dc = DataCollection.fromTriangleMesh3DSequence(referenceMesh, corresponded.map(_._2))
    val ssm = PointDistributionModel.createUsingPCA(dc)

    val modelFile = new File(outDir, "scapula_ssm.h5")
    StatisticalModelIO.writeStatisticalTriangleMeshModel3D(ssm, modelFile).get
    println(s"[Stage3] Model written to ${modelFile.getAbsolutePath} (rank ${ssm.rank})")

    // ---- Compactness: how much variance is captured by the first k modes -----------------------------------------
    val eigenvalues = ssm.gp.klBasis.map(_.eigenvalue).toIndexedSeq
    val totalVar = eigenvalues.sum
    val cumVar = eigenvalues.scanLeft(0.0)(_ + _).tail.map(_ / totalVar)

    println("\n[Stage3] Compactness")
    (0 until math.min(10, eigenvalues.length)).foreach { i =>
      println(
        f"  mode ${i + 1}%2d  var=${eigenvalues(i)}%9.2f mm^2 (${eigenvalues(i) / totalVar * 100}%5.1f%%)  cum=${cumVar(i) * 100}%5.1f%%"
      )
    }

    val maxModes = math.min(6, ssm.rank)

    // ---- Generalization: leave-one-SUBJECT-out reconstruction error -----------------------------------------------
    val bySubject = corresponded.groupBy { case (name, _) => ScapulaData.subjectKey(name) }.toIndexedSeq.sortBy(_._1)
    println(s"\n[Stage3] Evaluating LOOCV generalization over ${bySubject.length} subjects (modes 1..$maxModes)...")
    val genRmse = (1 to maxModes).map { numModes =>
      val perSubjectErrors = bySubject.flatMap { case (subject, group) =>
        val trainMeshes = corresponded.filterNot { case (n, _) => ScapulaData.subjectKey(n) == subject }.map(_._2)
        if (trainMeshes.length < 3) None
        else {
          val subDc = DataCollection.fromTriangleMesh3DSequence(referenceMesh, trainMeshes)
          val subSsmFull = PointDistributionModel.createUsingPCA(subDc)
          val subSsm = subSsmFull.truncate(math.min(numModes, subSsmFull.rank))
          Some(group.map { case (_, mesh) => pointToPointRMSE(subSsm.project(mesh), mesh) }.sum / group.length)
        }
      }
      val avg = perSubjectErrors.sum / perSubjectErrors.length
      println(f"  modes=$numModes  LOOCV RMSE=$avg%.3f mm  (${perSubjectErrors.length} held-out subjects)")
      numModes -> avg
    }

    // ---- Specificity: how close are random model samples to a real specimen ---------------------------------------
    println(s"\n[Stage3] Evaluating specificity (modes 1..$maxModes)...")
    val specRmse = (1 to maxModes).map { numModes =>
      val trunc = ssm.truncate(numModes)
      val errors = (0 until 30).map { _ =>
        val sample = trunc.sample()
        corresponded.map { case (_, m) => pointToPointRMSE(sample, m) }.min
      }
      val avg = errors.sum / errors.length
      println(f"  modes=$numModes  specificity RMSE=$avg%.3f mm")
      numModes -> avg
    }

    // ---- CSV --------------------------------------------------------------------------------------------------
    val csv = new File(outDir, "ssm_validation_metrics.csv")
    val pw = new PrintWriter(csv)
    try {
      pw.println("mode,variance_mm2,cum_variance_ratio,generalization_rmse_mm,specificity_rmse_mm")
      (0 until maxModes).foreach { i =>
        val g = genRmse.find(_._1 == i + 1).map(_._2).getOrElse(Double.NaN)
        val s = specRmse.find(_._1 == i + 1).map(_._2).getOrElse(Double.NaN)
        pw.println(s"${i + 1},${eigenvalues(i)},${cumVar(i)},$g,$s")
      }
    } finally pw.close()
    println(s"\n[Stage3] Validation metrics written to ${csv.getAbsolutePath}")
  }
}
