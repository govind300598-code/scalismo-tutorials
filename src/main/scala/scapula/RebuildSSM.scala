package scapula

import breeze.linalg.DenseVector
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.common.{PointId, Vectorizer}
import scalismo.geometry.{EuclideanVector, EuclideanVector3D, Landmark, _3D}
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.kernels.{DiagonalKernel, GaussianKernel}
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel, StatisticalMeshModel}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

import java.io.File

/**
 * Utility to rebuild or refine an SSM from existing registered meshes.
 *
 * Fixed for scalismo 0.92 API:
 *  - ScalismoUI is in scalismo.ui.api (not scalismo.ui)
 *  - EuclideanSpace3D is not needed; use the kernel-only GaussianProcess constructor
 *  - GaussianProcess vectorizer must be explicit to avoid ambiguous given instances
 *  - LowRankGaussianProcess.approximateGPCholesky takes TriangleMesh (DiscreteDomain),
 *    NOT mesh.pointSet (UnstructuredPoints is not DiscreteDomain in 0.92)
 *  - StatisticalMeshModel uses .referenceMesh (not .reference)
 *
 * Run with: sbt "runMain scapula.RebuildSSM"
 */
object RebuildSSM {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val outDir     = Config.outDir
    val resultsDir = new File(outDir, "results")

    // ── Load registered meshes from an existing SSM pass ─────────────────────
    val ssmPass = args.headOption.getOrElse("SSM1")
    val nrDir   = new File(resultsDir, s"$ssmPass/nonrigid_registered")

    if (!nrDir.exists()) {
      println(s"No registered meshes found at ${nrDir.getPath}")
      println("Run the main pipeline first: sbt \"runMain scapula.Main\"")
      System.exit(1)
    }

    val meshFiles = Option(nrDir.listFiles()).getOrElse(Array.empty)
      .filter(_.getName.endsWith(".stl"))
      .sortBy(_.getName)
      .toIndexedSeq

    println(s"Loading ${meshFiles.length} registered meshes from $ssmPass …")
    val meshes: IndexedSeq[TriangleMesh[_3D]] = meshFiles.map(ScapulaData.loadMesh)

    require(meshes.nonEmpty, "No meshes loaded")

    // ── Rebuild SSM via PCA ───────────────────────────────────────────────────
    println("Building SSM via PCA …")
    val ssm = PointDistributionModel.createUsingPCA(meshes)
    println(s"  rank = ${ssm.rank},  n = ${meshes.length}")

    // ── Variance report ───────────────────────────────────────────────────────
    SSMBuilder.varianceReport(ssm, ssmPass)

    // ── Save rebuilt model ────────────────────────────────────────────────────
    val modelFile = new File(new File(resultsDir, s"$ssmPass/model"), s"${ssmPass}_rebuilt.h5")
    modelFile.getParentFile.mkdirs()
    StatisticalModelIO.writeStatisticalMeshModel(ssm, modelFile)
      .getOrElse(println(s"WARNING: could not save model to ${modelFile.getPath}"))
    println(s"Saved rebuilt model → ${modelFile.getPath}")

    // ── Optionally compute GPMM prior and show generalization ─────────────────
    println("\nBuilding GP prior on reference mesh …")
    val reference = ssm.mean

    // Explicit Vectorizer to resolve the ambiguous given instances error.
    // (scalismo exposes ShortVectorizer and IntVectorizer for internal use;
    //  specifying the vectorizer here tells the compiler exactly which to use.)
    given Vectorizer[EuclideanVector[_3D]] = EuclideanVector3D.vectorizer

    val scalarKernel = GaussianKernel[_3D](NonRigidReg.gpSigma) * (NonRigidReg.gpScaleFactor * NonRigidReg.gpScaleFactor)
    val matKernel    = DiagonalKernel(scalarKernel, 3)

    // NOTE: pass `reference` (TriangleMesh, which IS a DiscreteDomain) — NOT reference.pointSet
    val gp: GaussianProcess[_3D, EuclideanVector[_3D]] = GaussianProcess(matKernel)
    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      reference,
      gp,
      relativeTolerance = Config.gpRelativeTolerance,
      interpolator = NearestNeighborInterpolator3D()
    )
    println(s"  GP prior rank = ${lowRankGP.rank}")

    // ── Generalization (LOO) ──────────────────────────────────────────────────
    if (meshes.length >= 3) {
      SSMBuilder.generalizationError(meshes, ssmPass)
    }

    // ── Specificity ───────────────────────────────────────────────────────────
    SSMBuilder.specificityError(ssm, meshes, label = ssmPass)

    // ── Mode deformation meshes ───────────────────────────────────────────────
    val modesDir = new File(resultsDir, s"$ssmPass/PCA_modes")
    SSMBuilder.saveModeDeformations(ssm, modesDir, ssmPass)

    // ── Visualise in Scalismo UI ──────────────────────────────────────────────
    if (Config.showUi) {
      val ui    = ScalismoUI(s"RebuildSSM – $ssmPass")
      val group = ui.createGroup(ssmPass)
      ui.show(group, ssm.mean, s"${ssmPass}_mean")

      // Show three mode deformations
      val nModes = math.min(3, ssm.rank)
      for (modeIdx <- 0 until nModes) {
        val stdDev = math.sqrt(ssm.variance(modeIdx))
        for (alpha <- Seq(-3.0, 0.0, 3.0)) {
          val coeffs = DenseVector.zeros[Double](ssm.rank)
          coeffs(modeIdx) = alpha * stdDev
          val shape = ssm.instance(coeffs)
          val tag   = if (alpha < 0) s"m${(-alpha).toInt}sd" else if (alpha == 0.0) "mean" else s"p${alpha.toInt}sd"
          ui.show(group, shape, s"mode${modeIdx + 1}_$tag")
        }
      }

      println("Scalismo UI ready. Close the window to exit.")
    }
  }
}
