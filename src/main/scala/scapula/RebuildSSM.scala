package scapula

import scalismo.common.{Domain, Field, PointId}
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.geometry.*
import scalismo.io.MeshIO
import scalismo.kernels.{DiagonalKernel, GaussianKernel}
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel}
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.utils.Random

import java.io.File
import scala.util.{Failure, Success}

/**
 * Iterative multi-pass SSM construction pipeline (GPA-style).
 *
 * Algorithm:
 *   Pass 1  : register all specimens → arbitrary initial reference (first specimen).
 *             Build SSM₁ via PCA.  Mean shape = new reference.
 *   Pass 2  : register all specimens → SSM₁ mean.
 *             Build SSM₂.  Mean shape = new reference.
 *   Pass k  : register all specimens → SSM_{k-1} mean.
 *             Build SSMₖ.
 *   ...
 *   Pass N  : final model SSMₙ.
 *
 * At each step a STABILITY metric is printed:
 *   surface distance between consecutive mean shapes.
 *   Convergence (< 1 mm mean shift) confirms reference bias has been removed.
 *
 * Number of passes is controlled by SCAPULA_REFINE_PASSES (default 4).
 *
 * Methods used
 * ────────────
 *  • Landmark-based Procrustes rigid registration + trimmed rigid ICP
 *  • Single isotropic Gaussian kernel GP prior
 *  • Pivoted-Cholesky low-rank approximation with NearestNeighborInterpolator3D
 *  • GP-ICP: iterated nearest-neighbour posterior regression (non-rigid)
 *  • GPA-style iterative mean-shape reference update
 *  • PCA-based SSM (PointDistributionModel.createUsingPCA)
 *
 * Usage:
 *   sbt "runMain scapula.RebuildSSM"
 *   SCAPULA_DATA_DIR=/x SCAPULA_OUT_DIR=/y SCAPULA_REFINE_PASSES=4 sbt "runMain scapula.RebuildSSM"
 */
object RebuildSSM {

  // ── GP model construction ──────────────────────────────────────────────────

  def buildGpModel(
    reference: TriangleMesh[_3D]
  )(implicit rng: Random): PointDistributionModel[_3D, TriangleMesh] = {

    // Whole-space domain (defined everywhere); avoids depending on any
    // specific scalismo object name for the Euclidean space.
    val wholeDomain = new Domain[_3D] { def isDefinedAt(pt: Point[_3D]) = true }
    val zeroMean = Field[_3D, EuclideanVector[_3D]](
      wholeDomain, _ => EuclideanVector.zeros[_3D])

    val scalarKernel = GaussianKernel[_3D](Config.kernelSigma) * Config.kernelScale
    val kernel       = DiagonalKernel(scalarKernel, outputDim = 3)
    val gp           = GaussianProcess[_3D, EuclideanVector[_3D]](zeroMean, kernel)

    // Pass the TriangleMesh (not reference.pointSet) so the type matches
    // NearestNeighborInterpolator3D which is typed for TriangleMesh domain.
    // NearestNeighborInterpolator3D — exact role:
    //   The pivoted-Cholesky approximation produces eigenfunctions defined only
    //   at the discrete set of mesh vertices.  NearestNeighborInterpolator3D
    //   extends these piecewise-constant to *any* 3D query point by mapping it
    //   to the nearest vertex.  This is used:
    //     (a) inside gpIcp to evaluate the posterior at the current-mean vertex
    //         positions after each ICP step, and
    //     (b) when model.instance() is called to reconstruct a shape instance.
    //   A smooth (barycentric) interpolator would give the same SSM; the
    //   nearest-neighbour choice is faster and avoids triangle-quality issues in
    //   the decimated mesh while introducing no bias in the correspondence step.
    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      domain            = reference,
      gp                = gp,
      relativeTolerance = Config.gpRelativeTolerance,
      interpolator      = NearestNeighborInterpolator3D()
    )
    PointDistributionModel[_3D, TriangleMesh](reference, lowRankGP)
  }

  // ── GP-ICP non-rigid registration ─────────────────────────────────────────

  def gpIcp(
    model:      PointDistributionModel[_3D, TriangleMesh],
    target:     TriangleMesh[_3D],
    iterations: Int    = Config.icpIterations,
    numPoints:  Int    = 1000,
    sigma2:     Double = Config.icpSigma2
  )(implicit rng: Random): TriangleMesh[_3D] = {

    val targetOps = target.operations
    var current   = model

    for (_ <- 0 until iterations) {
      val fitted = current.mean
      val sampleIds = RigidAlign.uniformIds(fitted, numPoints)
      val observations: IndexedSeq[(PointId, Point[_3D])] = sampleIds.map { id =>
        val fittedPt  = fitted.pointSet.point(id)
        val closestPt = targetOps.closestPointOnSurface(fittedPt).point
        (id, closestPt)
      }
      current = current.posterior(observations, sigma2)
    }
    current.mean
  }

  // ── One registration pass ──────────────────────────────────────────────────

  private case class PreparedSpecimen(
    modelId: String,
    mesh:    TriangleMesh[_3D],
    lms:     IndexedSeq[Landmark[_3D]]
  )

  private def doPass(
    label:   String,
    passDir: File,
    preps:   IndexedSeq[PreparedSpecimen],
    refMesh: TriangleMesh[_3D],
    refLms:  IndexedSeq[Landmark[_3D]]
  )(implicit rng: Random): IndexedSeq[(TriangleMesh[_3D], IndexedSeq[Landmark[_3D]])] = {

    passDir.mkdirs()
    println(s"\n╔══ $label ══╗")
    println(s"  Reference: ${refMesh.pointSet.numberOfPoints} vertices")
    println(s"  Building GP model  (σ=${Config.kernelSigma} mm, scale=${Config.kernelScale} mm, " +
            s"tol=${Config.gpRelativeTolerance})...")

    val gpModel = buildGpModel(refMesh)
    println(s"  GP rank = ${gpModel.rank}   GP-ICP iterations = ${Config.icpIterations}")

    preps.zipWithIndex.map { case (p, i) =>
      val tag = s"[${i+1}/${preps.length}]"
      print(s"  $tag  ${p.modelId}  rigid-align ...")

      val (rigidMesh, rigidLms) = RigidAlign.landmarkThenIcp(p.mesh, p.lms, refMesh, refLms)

      print(s"  gp-icp (${Config.icpIterations} iter) ...")

      val registered = gpIcp(gpModel, rigidMesh)
      val st = Metrics.symmetric(registered, rigidMesh)
      println(f"  done | resid ${st.render}")

      val outF = new File(passDir, s"reg_${p.modelId}.stl")
      MeshIO.writeMesh(registered, outF) match {
        case Failure(e) => println(s"  WARNING: could not save ${outF.getName}: ${e.getMessage}")
        case Success(_) => ()
      }

      (registered, rigidLms)
    }
  }

  // ── Main ──────────────────────────────────────────────────────────────────

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dataDir    = Config.dataDir
    val outDir     = Config.outDir
    val nPasses    = Config.refinePasses
    outDir.mkdirs()

    println(s"Data dir : ${dataDir.getAbsolutePath}")
    println(s"Out dir  : ${outDir.getAbsolutePath}")
    println(s"Passes   : $nPasses")
    println(s"Kernel   : σ=${Config.kernelSigma} mm, scale=${Config.kernelScale} mm")
    println(s"GP-ICP   : ${Config.icpIterations} iterations, σ²=${Config.icpSigma2}")

    // ── Load data ────────────────────────────────────────────────────────────
    val (lmMap, _, _) = ScapulaData.readLandmarkCsv(ScapulaData.csvFile(dataDir))
    val specimens     = ScapulaData.specimens(dataDir).filter(s => lmMap.contains(s.modelId))
    println(s"Specimens with landmarks: ${specimens.length}")

    val preps: IndexedSeq[PreparedSpecimen] = specimens.map { s =>
      val (m, l) =
        if (s.isRight) (ScapulaData.mirrorMesh(ScapulaData.loadMesh(s.file)),
                        ScapulaData.mirrorLandmarks(lmMap(s.modelId)))
        else           (ScapulaData.loadMesh(s.file), lmMap(s.modelId))
      PreparedSpecimen(s.modelId, m, l)
    }

    // ── Decimate initial reference to modelResolution ─────────────────────────
    // All passes work at the target resolution so the GP and SSM are never
    // built at the raw STL vertex count (which can be 50k+).
    val initRef = preps(Config.refIdx.min(preps.length - 1))
    val initRefMesh: TriangleMesh[_3D] = {
      val n = initRef.mesh.pointSet.numberOfPoints
      if (n > Config.modelResolution) {
        println(s"Decimating initial reference [${initRef.modelId}] $n → ~${Config.modelResolution} vertices " +
                s"(Voronoi coarsening)...")
        val dec = ScapulaData.decimateInCorrespondence(
          initRef.mesh, IndexedSeq(initRef.mesh), Config.modelResolution)
        println(s"  Actual: ${dec.head.pointSet.numberOfPoints} vertices")
        dec.head
      } else {
        println(s"Initial reference [${initRef.modelId}]: $n vertices (≤ ${Config.modelResolution}, no decimation needed)")
        initRef.mesh
      }
    }

    // ── Iterative passes ──────────────────────────────────────────────────────
    var currentRefMesh: TriangleMesh[_3D]         = initRefMesh
    var currentRefLms:  IndexedSeq[Landmark[_3D]] = initRef.lms

    val ssmMeans  = scala.collection.mutable.ArrayBuffer.empty[TriangleMesh[_3D]]
    val ssmRanks  = scala.collection.mutable.ArrayBuffer.empty[Int]

    for (pass <- 1 to nPasses) {
      val passDir = new File(outDir, s"pass$pass")

      val pOut    = doPass(s"PASS $pass / $nPasses", passDir, preps, currentRefMesh, currentRefLms)
      val pMeshes = pOut.map(_._1)
      val pLms    = pOut.map(_._2)

      // Build SSMₖ
      println(s"\nBuilding SSM$pass from pass $pass...")
      val dc  = DataCollection.fromTriangleMesh3DSequence(pMeshes.head, pMeshes)
      val ssm = PointDistributionModel.createUsingPCA(dc)
      println(f"  SSM$pass rank = ${ssm.rank}  |  ${pMeshes.length} specimens")

      val mean = ssm.mean
      MeshIO.writeMesh(mean, new File(outDir, s"mean_pass$pass.stl")).toOption
      println(s"  Mean shape → mean_pass$pass.stl")

      // Stability vs previous mean
      if (ssmMeans.nonEmpty) {
        val prevMean  = ssmMeans.last
        val stability = Metrics.symmetric(prevMean, mean)
        println(f"\n  ── Stability check (pass ${pass-1} mean ↔ pass $pass mean) ──")
        println(f"    ${stability.render}")
        if (stability.mean < 1.0)
          println(f"  ✓ CONVERGED  (mean shift = ${stability.mean}%.3f mm < 1 mm)")
        else
          println(f"  ✗ not yet converged  (${stability.mean}%.3f mm)")
      }

      ssmMeans  += mean
      ssmRanks  += ssm.rank

      // Prepare reference for next pass
      if (pass < nPasses) {
        // Reference landmarks = mean of rigidly-aligned landmark positions from this pass
        val nextRefLms: IndexedSeq[Landmark[_3D]] = ScapulaData.landmarkNames.map { nm =>
          val pts = pLms.flatMap(_.find(_.id == nm).map(_.point))
          Landmark(nm, Point3D(pts.map(_.x).sum / pts.length,
                               pts.map(_.y).sum / pts.length,
                               pts.map(_.z).sum / pts.length))
        }

        // Decimate mean if needed for faster GP computation
        val nextRefMesh = {
          val n = mean.pointSet.numberOfPoints
          if (n > Config.modelResolution) {
            println(s"  Decimating mean reference $n → ~${Config.modelResolution} vertices...")
            ScapulaData.decimateInCorrespondence(mean, IndexedSeq(mean), Config.modelResolution).head
          } else mean
        }

        currentRefMesh = nextRefMesh
        currentRefLms  = nextRefLms
      }
    }

    // ── Final stability summary ───────────────────────────────────────────────
    println(s"\n══════════════════════════════════════════════════════════════════════")
    println(s"  Pipeline complete.  $nPasses passes.  σ=${Config.kernelSigma} mm  " +
            s"scale=${Config.kernelScale} mm  res=${Config.modelResolution} pts")
    println(s"══════════════════════════════════════════════════════════════════════")
    println(f"  ${"Pass"}%-6s  ${"Specimens"}%9s  ${"SSM rank"}%8s  ${"Mean file"}")
    println("  " + "─" * 50)
    for (pass <- 1 to nPasses) {
      println(f"  pass$pass%-2s    ${preps.length}%9d  ${ssmRanks(pass-1)}%8d  mean_pass$pass.stl")
    }
    if (ssmMeans.length >= 2) {
      println(s"\n  Surface-to-surface distances between consecutive mean shapes:")
      println(f"  ${"Comparison"}%-18s  ${"Mean (mm)"}%10s  ${"RMS (mm)"}%9s  ${"HD95 (mm)"}%10s  Status")
      println("  " + "─" * 65)
      for (k <- 1 until ssmMeans.length) {
        val st  = Metrics.symmetric(ssmMeans(k-1), ssmMeans(k))
        val tag = if (st.mean < 1.0) "✓ converged" else "  not yet "
        println(f"  Mean$k ↔ Mean${k+1}             ${st.mean}%10.3f  ${st.rms}%9.3f  ${st.hd95}%10.3f  $tag")
      }
    }
    println()
    println("Next steps:")
    println("  sbt \"runMain scapula.SSMValidation\"    # compactness / generalization / specificity")
    println("  sbt \"runMain scapula.ViewSSM\"          # interactive SSM with PCA sliders")
    println("  sbt \"runMain scapula.ViewRegistration\" # target vs fitted comparison + metrics table")
  }
}
