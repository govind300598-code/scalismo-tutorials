package scapula

import breeze.linalg.DenseVector
import scalismo.common.PointId
import scalismo.geometry.*
import scalismo.io.MeshIO
import scalismo.mesh.{TriangleList, TriangleMesh, TriangleMesh3D}
import scalismo.utils.Random

import java.io.{File, PrintWriter}

/**
 * STAGE 2 -- build point-to-point correspondence across the whole population by non-rigidly registering a common
 * reference to every specimen (see NonRigidRegistration for the fitting algorithm itself, ported from the mailing
 * list). Run Stage1Diagnostics first: this stage assumes the rigid pipeline (mirroring, landmark frame, orientation)
 * is already known to be sound.
 *
 * Output layout (under Config.outDir), all in VTK (never STL) because STL is an unindexed triangle soup -- reloading
 * one does NOT reproduce the original vertex order, which would silently destroy the correspondence Stage 3 depends
 * on:
 *   reference.vtk              the final (post-refinement) reference topology, in the rigid landmark frame
 *   registered/<modelId>.vtk   every specimen warped into that reference's correspondence
 *   registration_accuracy.csv  per-specimen surface-distance residual between the registered mesh and the real target
 */
object Stage2Registration {

  private def meanMesh(meshes: IndexedSeq[TriangleMesh[_3D]], triangles: TriangleList): TriangleMesh[_3D] = {
    val n = meshes.head.pointSet.numberOfPoints
    val m = meshes.length.toDouble
    val meanPoints = (0 until n).map { i =>
      val pts = meshes.map(_.pointSet.point(PointId(i)))
      Point3D(pts.map(_.x).sum / m, pts.map(_.y).sum / m, pts.map(_.z).sum / m)
    }
    TriangleMesh3D(meanPoints, triangles)
  }

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dir = Config.dataDir
    val csv = ScapulaData.csvFile(dir)
    println(s"Data directory : ${dir.getAbsolutePath}")
    println(s"Landmark CSV   : ${csv.getAbsolutePath}")
    println(s"Output dir     : ${Config.outDir.getAbsolutePath}")

    val (landmarks, _, _) = ScapulaData.readLandmarkCsv(csv)
    val specimens = ScapulaData.specimens(dir).filter(s => landmarks.contains(s.modelId))
    require(specimens.nonEmpty, s"No specimens with both a mesh and a landmark row found in ${dir.getPath}")
    println(s"${specimens.length} specimens with mesh + landmarks (expect 24 = 12 subjects x L/R)")

    // -------------------------------------------------------------------------------------------------- reference
    // Pick, deterministically, the specimen whose landmarks are closest to the population's mean landmark
    // configuration -- a more representative starting point than an arbitrary first file, and fully reproducible.
    val meanLandmarks = ScapulaData.landmarkNames.map { name =>
      val pts = specimens.map(s => landmarks(s.modelId).find(_.id == name).get.point)
      name -> Point3D(pts.map(_.x).sum / pts.length, pts.map(_.y).sum / pts.length, pts.map(_.z).sum / pts.length)
    }.toMap
    val referenceSpecimen = specimens.minBy { s =>
      landmarks(s.modelId).map { lm =>
        val d = (lm.point - meanLandmarks(lm.id)).norm
        d * d
      }.sum
    }
    println(s"Reference specimen (closest to mean landmark configuration): ${referenceSpecimen.modelId}")

    // ------------------------------------------------------------------------------------------------ rigid align
    val referenceLms = landmarks(referenceSpecimen.modelId)
    val referenceRawMesh = ScapulaData.loadMesh(referenceSpecimen.file)

    val alignedTargets: IndexedSeq[(String, TriangleMesh[_3D])] = specimens.map { s =>
      val mesh = ScapulaData.loadMesh(s.file)
      val lms = landmarks(s.modelId)
      if (s.modelId == referenceSpecimen.modelId) s.modelId -> mesh
      else {
        val (aligned, _) = RigidAlign.landmarkThenIcp(mesh, lms, referenceRawMesh, referenceLms, Config.icpIterations)
        s.modelId -> aligned
      }
    }
    println(s"Rigidly aligned all ${alignedTargets.length} specimens into the reference's landmark frame " +
      s"(${Config.icpIterations} ICP iterations after landmark Procrustes).")

    var reference = referenceRawMesh.operations.decimate(Config.modelResolution)
    println(s"Reference decimated to ${reference.pointSet.numberOfPoints} vertices " +
      s"(requested ${Config.modelResolution}).")

    // -------------------------------------------------------------------------------------------- refinement loop
    var registered: IndexedSeq[(String, TriangleMesh[_3D])] = IndexedSeq.empty
    for (pass <- 1 to Config.refinePasses) {
      println(s"\n=== Registration pass $pass / ${Config.refinePasses} " +
        s"(reference has ${reference.pointSet.numberOfPoints} vertices) ===")
      val lowRankGP = NonRigidRegistration.buildMultiscaleGP(reference)
      println(f"  GP prior rank = ${lowRankGP.rank} (relativeTolerance=${Config.gpRelativeTolerance}, " +
        f"maxRank=${Config.gpMaxRank})")
      val cascade = NonRigidRegistration.defaultCascade(reference)
      val initialCoefficients = DenseVector.zeros[Double](lowRankGP.rank)

      registered = alignedTargets.zipWithIndex.map { case ((modelId, target), i) =>
        val t0 = System.nanoTime()
        val coeffs =
          NonRigidRegistration.registerToTarget(lowRankGP, reference, target, initialCoefficients, cascade)
        val mesh = NonRigidRegistration.warpedMesh(lowRankGP, reference, coeffs)
        val seconds = (System.nanoTime() - t0) / 1e9
        println(f"  [${i + 1}%2d/${alignedTargets.length}] $modelId%-20s registered in $seconds%6.1fs")
        modelId -> mesh
      }

      if (pass < Config.refinePasses) {
        reference = meanMesh(registered.map(_._2), reference.triangulation)
        println(s"  Rebuilt reference as the mean of pass $pass's registered meshes (removes reference bias).")
      }
    }

    // ------------------------------------------------------------------------------------------------------- save
    Config.outDir.mkdirs()
    val registeredDir = new File(Config.outDir, "registered")
    registeredDir.mkdirs()
    MeshIO.writeMesh(reference, new File(Config.outDir, "reference.vtk")).get
    registered.foreach { case (modelId, mesh) =>
      MeshIO.writeMesh(mesh, new File(registeredDir, s"$modelId.vtk")).get
    }
    println(s"\nWrote reference.vtk and ${registered.length} registered meshes to ${registeredDir.getPath}")

    // --------------------------------------------------------------------------------------- registration accuracy
    // Surface distance between each registered (reference-topology) mesh and its ACTUAL target mesh. This is the
    // standard "how good is the correspondence" number reported in SSM papers -- not to be confused with the SSM's
    // own generalization/specificity (Stage 3), which measure the *model*, not the per-specimen registration fit.
    println("\n[registration accuracy] registered mesh vs. real target surface, symmetric, after rigid alignment:")
    val targetById = alignedTargets.toMap
    val accuracy = registered.map { case (modelId, mesh) =>
      val stats = Metrics.symmetric(mesh, targetById(modelId))
      println(f"  $modelId%-20s ${stats.render}")
      modelId -> stats
    }

    val accCsv = new File(Config.outDir, "registration_accuracy.csv")
    val pw = new PrintWriter(accCsv)
    try {
      pw.println("model_id,mean_mm,rms_mm,hd95_mm,hd_mm")
      accuracy.foreach { case (id, s) => pw.println(f"$id,${s.mean}%.4f,${s.rms}%.4f,${s.hd95}%.4f,${s.hd}%.4f") }
    } finally pw.close()

    val n = accuracy.length
    def avg(f: Metrics.SurfaceStats => Double): Double = accuracy.map { case (_, s) => f(s) }.sum / n
    def sd(f: Metrics.SurfaceStats => Double): Double = {
      val m = avg(f)
      math.sqrt(accuracy.map { case (_, s) => math.pow(f(s) - m, 2) }.sum / n)
    }
    println(f"\n  POPULATION (n=$n): mean=${avg(_.mean)}%.3f+-${sd(_.mean)}%.3f mm  " +
      f"rms=${avg(_.rms)}%.3f+-${sd(_.rms)}%.3f mm  hd95=${avg(_.hd95)}%.3f+-${sd(_.hd95)}%.3f mm  " +
      f"hd=${avg(_.hd)}%.3f+-${sd(_.hd)}%.3f mm")
    println(s"  wrote ${accCsv.getPath}")
    println("\nNext: run Stage3SSMValidation.")
  }
}
