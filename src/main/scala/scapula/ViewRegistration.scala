package scapula

import scalismo.geometry._3D
import scalismo.geometry.{Landmark, Point3D}
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

import java.io.File

/**
 * Registration quality viewer + SSM explorer.
 *
 * Groups shown in the Scalismo UI:
 *   - "SSM (sliders)"              — interactive statistical model with coefficient sliders
 *   - "Mean shape"                 — PCA mean mesh
 *   - "Best / Registered"          — top-3 registered (fitted) meshes
 *   - "Best / Original (aligned)"  — corresponding original STLs rigidly aligned to reference frame
 *   - "Worst / Registered"         — bottom-3 registered meshes
 *   - "Worst / Original (aligned)" — corresponding original STLs rigidly aligned to reference frame
 *
 * Prints to console: per-specimen error metrics table vs. SSM mean shape:
 *   Mean distance, RMSE, HD95, Hausdorff, Chamfer distance (all in mm).
 *
 * Usage:
 *   sbt "runMain scapula.ViewRegistration [outDir [dataDir]]"
 *   SCAPULA_OUT_DIR=/x SCAPULA_DATA_DIR=/y sbt "runMain scapula.ViewRegistration"
 */
object ViewRegistration {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val outDir   = if (args.nonEmpty)   new File(args(0)) else Config.outDir
    val dataDir  = if (args.length > 1) new File(args(1)) else Config.dataDir
    val pass2Dir = new File(outDir, "pass2")

    require(pass2Dir.isDirectory,
      s"pass2/ not found: ${pass2Dir.getAbsolutePath}\n" +
      s"  Set SCAPULA_OUT_DIR or pass the output directory as the first argument.")

    // ── 1. Load registered (fitted) meshes ────────────────────────────────────
    val regFiles = Option(pass2Dir.listFiles()).getOrElse(Array.empty[File])
      .filter(f => f.getName.endsWith(".stl") && f.getName.startsWith("reg_"))
      .sortBy(_.getName)
    require(regFiles.nonEmpty, s"No reg_*.stl in ${pass2Dir.getAbsolutePath}")

    println(s"\nLoading ${regFiles.length} registered meshes from ${pass2Dir.getAbsolutePath}")
    val regIds    = regFiles.map(_.getName.stripSuffix(".stl").stripPrefix("reg_")).toIndexedSeq
    val rawReg    = regFiles.map(ScapulaData.loadMesh).toIndexedSeq
    val nOrig     = rawReg.head.pointSet.numberOfPoints
    println(s"  ${rawReg.length} meshes, $nOrig vertices each")

    val target    = Config.modelResolution
    val regMeshes = if (nOrig > target) {
      println(s"  Decimating $nOrig → ~$target vertices (Voronoi coarsening)...")
      val dec = ScapulaData.decimateInCorrespondence(rawReg.head, rawReg, target)
      println(s"  Actual: ${dec.head.pointSet.numberOfPoints} vertices")
      dec
    } else { println(s"  Already ≤ $target vertices — skipping decimation"); rawReg }

    // ── 2. Build SSM in memory ────────────────────────────────────────────────
    println("Building SSM via PCA...")
    val t0    = System.currentTimeMillis()
    val dc    = DataCollection.fromTriangleMesh3DSequence(regMeshes.head, regMeshes)
    val model = PointDistributionModel.createUsingPCA(dc)
    val mean  = model.mean
    println(f"  rank = ${model.rank} | ${(System.currentTimeMillis() - t0) / 1000.0}%.1f s")

    val eigenvalues = model.gp.klBasis.map(_.eigenvalue).toArray
    val totalVar    = eigenvalues.sum
    val cumVar      = eigenvalues.scanLeft(0.0)(_ + _).tail.map(_ / totalVar * 100.0)
    def modesFor(pct: Double): Int = { val i = cumVar.indexWhere(_ >= pct); if (i < 0) model.rank else i + 1 }
    println(f"  90%% variance: ${modesFor(90)} modes  |  95%%: ${modesFor(95)}  |  99%%: ${modesFor(99)}")

    // ── 3. Per-specimen error metrics vs. mean shape ──────────────────────────
    println("\nComputing per-specimen error metrics vs. mean shape...")
    final case class SM(name: String, meanD: Double, rms: Double, hd95: Double, hd: Double, chamfer: Double)

    val metrics: IndexedSeq[SM] = (regMeshes zip regIds).map { case (m, id) =>
      val corr = Metrics.correspondingDistances(m, mean)
      val sym  = Metrics.symmetric(m, mean)
      SM(id,
        corr.sum / corr.length,
        math.sqrt(corr.map(x => x * x).sum / corr.length),
        Metrics.percentile(corr, 0.95),
        corr.max,
        sym.mean)
    }
    val sorted = metrics.sortBy(_.meanD)

    val colW = math.max(45, sorted.map(_.name.length).max + 2)
    val sep  = "─" * (colW + 58)
    println(s"\n$sep")
    println(f"  ${"Specimen"}%-${colW}s  ${"Mean(mm)":>9}  ${"RMSE":>9}  ${"HD95":>9}  ${"HD":>9}  ${"Chamfer":>9}")
    println(sep)
    sorted.foreach { m =>
      println(f"  ${m.name}%-${colW}s  ${m.meanD}%9.3f  ${m.rms}%9.3f  ${m.hd95}%9.3f  ${m.hd}%9.3f  ${m.chamfer}%9.3f")
    }
    val k = metrics.length.toDouble
    println(sep)
    println(f"  ${"AVERAGE"}%-${colW}s  ${metrics.map(_.meanD).sum/k}%9.3f  ${metrics.map(_.rms).sum/k}%9.3f  ${metrics.map(_.hd95).sum/k}%9.3f  ${metrics.map(_.hd).sum/k}%9.3f  ${metrics.map(_.chamfer).sum/k}%9.3f")
    println(s"$sep\n  Note: distances are point-to-point vs. SSM mean shape (all specimens in correspondence)")

    // ── 4. Load & rigidly align original STLs for registration comparison ─────
    println("\nLoading landmark CSV and original meshes for registration comparison...")
    val origAligned: Map[String, Option[TriangleMesh[_3D]]] =
      try {
        val csvFile           = ScapulaData.csvFile(dataDir)
        val (lmMap, _, _)     = ScapulaData.readLandmarkCsv(csvFile)

        // Mean of all specimens' landmark coordinates ≈ registration reference frame
        val refLms: IndexedSeq[Landmark[_3D]] = ScapulaData.landmarkNames.map { lmName =>
          val pts = lmMap.values.flatMap(_.find(_.id == lmName).map(_.point)).toIndexedSeq
          require(pts.nonEmpty, s"Landmark '$lmName' absent from CSV")
          Landmark(lmName, Point3D(
            pts.map(_.x).sum / pts.length,
            pts.map(_.y).sum / pts.length,
            pts.map(_.z).sum / pts.length))
        }

        val specimens = ScapulaData.specimens(dataDir)
        val specByModelId = specimens.map(s => s.modelId -> s).toMap

        regIds.map { id =>
          val result = specByModelId.get(id).flatMap { spec =>
            lmMap.get(spec.modelId).map { lms =>
              val (mesh0, lms0) =
                if (spec.isRight) (ScapulaData.mirrorMesh(ScapulaData.loadMesh(spec.file)),
                                   ScapulaData.mirrorLandmarks(lms))
                else              (ScapulaData.loadMesh(spec.file), lms)
              val rigid = ScapulaData.rigidFromLandmarks(lms0, refLms)
              mesh0.transform(rigid)
            }
          }
          id -> result
        }.toMap
      } catch { case e: Exception =>
        println(s"  Warning: could not load/align originals: ${e.getMessage}")
        println(  "  Registration comparison groups will be empty.")
        regIds.map(_ -> None).toMap
      }

    val nFound = origAligned.values.count(_.isDefined)
    println(s"  Aligned $nFound / ${regIds.length} original meshes")

    // ── 5. Open Scalismo UI ───────────────────────────────────────────────────
    println("\nOpening Scalismo UI...")
    val ui = ScalismoUI()

    ui.show(ui.createGroup("SSM (sliders)"), model, "SSM")
    ui.show(ui.createGroup("Mean shape"),    mean,  "mean")

    val nShow = math.min(3, sorted.length)
    val meshByName = (regMeshes zip regIds).toMap

    // Best cases — registered vs original
    val bReg  = ui.createGroup("Best / Registered")
    val bOrig = ui.createGroup("Best / Original (aligned)")
    sorted.take(nShow).foreach { m =>
      ui.show(bReg, meshByName(m.name), m.name)
      origAligned.get(m.name).flatten.foreach(om => ui.show(bOrig, om, m.name + "_orig"))
    }

    // Worst cases — registered vs original
    val wReg  = ui.createGroup("Worst / Registered")
    val wOrig = ui.createGroup("Worst / Original (aligned)")
    sorted.takeRight(nShow).foreach { m =>
      ui.show(wReg, meshByName(m.name), m.name)
      origAligned.get(m.name).flatten.foreach(om => ui.show(wOrig, om, m.name + "_orig"))
    }

    println("UI open. Close the window to exit.")
  }
}
