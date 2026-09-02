package scapula

import scalismo.common.{Field, RealSpace}
import scalismo.geometry.*
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.kernels.{DiagonalKernel, GaussianKernel, MatrixValuedPDKernel}
import scalismo.mesh.*
import scalismo.numerics.UniformMeshSampler3D
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel}
import scalismo.utils.Random

import java.io.File

/**
 * Builds the Free-Form Deformation Model (FFDM) — a GP prior on the reference mesh.
 *
 * ROOT CAUSE of the Scala 2.12 syntax errors the user saw:
 *   /home/g25upadh/Downloads/BuildFFDMModel.scala had bare top-level `val` and `def`
 *   statements outside any class or object.  Scala 2.x requires every statement to be
 *   enclosed in a class or object (Scala 3 allows top-level defs but NOT bare vals).
 *   The fix — demonstrated here — is simply to wrap everything inside `object BuildFFDMModel`.
 *
 * The FFDM encodes shape prior knowledge (plausible deformations of the reference)
 * using a multi-scale Gaussian kernel following Madsen et al. (sigma = L/2, L/5, L/10
 * where L is the longest bounding-box dimension).  It is the morphable-model prior for
 * GP-ICP registration in NonRigidReg.
 */
object BuildFFDMModel {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val outDir = Config.outDir
    outDir.mkdirs()

    println("=" * 70)
    println("FFDM BUILD — GP prior on reference mesh")
    println("=" * 70)
    println(s"  Data dir     : ${Config.dataDir.getAbsolutePath}")
    println(s"  Output dir   : ${outDir.getAbsolutePath}")
    println(s"  Seed ref     : ${Config.ffdmRefId}")
    println(s"  GP max rank  : ${Config.gpMaxRank}")
    println(s"  GP tolerance : ${Config.gpRelativeTolerance}")
    println()

    val refFile = new File(Config.dataDir, s"${Config.ffdmRefId}.stl")
    require(refFile.exists(), s"Reference STL not found: ${refFile.getAbsolutePath}")

    val refMesh = ScapulaData.loadMesh(refFile)
    println(s"Loaded reference: ${refMesh.pointSet.numberOfPoints} vertices, " +
      s"${refMesh.triangulation.triangles.size} triangles")

    val L = longestDimension(refMesh)
    println(f"  Bounding-box longest side L = $L%.1f mm")

    val components = Seq(
      (L / 2.0,  L * L / 4.0),
      (L / 5.0,  L * L / 25.0),
      (L / 10.0, L * L / 100.0)
    )
    println(f"  Kernel sigmas: coarse=${L/2.0}%.1f  mid=${L/5.0}%.1f  fine=${L/10.0}%.1f mm")

    val lowRankGP = buildLowRankGP(refMesh, components, Config.gpMaxRank)
    val ffdm      = PointDistributionModel[_3D, TriangleMesh](refMesh, lowRankGP)

    println(s"\nFFDM rank: ${ffdm.rank}")
    printCompactness(lowRankGP)

    val modelFile = new File(outDir, "ffdm.h5")
    StatisticalModelIO.writeStatisticalTriangleMeshModel3D(ffdm, modelFile).get
    println(s"\nSaved FFDM model : ${modelFile.getAbsolutePath}")
    MeshIO.writeMesh(refMesh, new File(outDir, "ffdm_reference.stl")).get
    println(s"Saved reference  : ${new File(outDir, "ffdm_reference.stl").getAbsolutePath}")
    println()
    println("Next step: sbt \"runMain scapula.FullPipeline\"")
  }

  /** Longest bounding-box dimension of a mesh, in the mesh's length units. */
  def longestDimension(mesh: TriangleMesh[_3D]): Double = {
    val pts = mesh.pointSet.points.toIndexedSeq
    val xs = pts.map(_.x); val ys = pts.map(_.y); val zs = pts.map(_.z)
    Seq(xs.max - xs.min, ys.max - ys.min, zs.max - zs.min).max
  }

  /**
   * Build a low-rank GP approximation on the reference mesh.
   *
   * @param components  Seq of (sigma, scale) pairs.  Each pair adds a Gaussian kernel component:
   *                    k_i(x,y) = scale * exp(-||x-y||^2 / (2*sigma^2)).
   * @param numBasisFns Number of basis functions (Nystrom approximation); caps the rank.
   */
  def buildLowRankGP(
    reference:    TriangleMesh[_3D],
    components:   Seq[(Double, Double)],
    numBasisFns:  Int = 300
  )(implicit rng: Random): LowRankGaussianProcess[_3D, EuclideanVector[_3D]] = {
    val zeroMean = Field[_3D, EuclideanVector[_3D]](RealSpace[_3D], _ => EuclideanVector.zeros[_3D])
    val kernelList = components.map { case (sigma, scale) =>
      DiagonalKernel[_3D](GaussianKernel[_3D](sigma) * scale, 3)
    }
    val kernel: MatrixValuedPDKernel[_3D] =
      kernelList.tail.foldLeft[MatrixValuedPDKernel[_3D]](kernelList.head)(_ + _)
    val gp      = GaussianProcess[_3D, EuclideanVector[_3D]](zeroMean, kernel)
    val sampler = UniformMeshSampler3D(reference, numBasisFns)
    LowRankGaussianProcess.approximateGPNystrom(gp, sampler, numBasisFns)
  }

  private def printCompactness(gp: LowRankGaussianProcess[_3D, EuclideanVector[_3D]]): Unit = {
    val evs   = gp.klBasis.map(_.eigenvalue)
    val total = evs.sum
    println("  Cumulative variance explained:")
    for (n <- Seq(10, 20, 50, 100, 200)) {
      val pct = evs.take(n).sum / total * 100
      val bar = "█" * (pct.toInt / 2)
      println(f"    modes 1-$n%3d: $pct%5.1f%%  $bar")
    }
  }
}
