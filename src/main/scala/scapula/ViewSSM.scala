package scapula

import breeze.linalg.DenseVector
import scalismo.geometry._3D
import scalismo.mesh.{ScalarMeshField, TriangleMesh}
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

import java.io.File

/**
 * Full visual inspection for the scapula SSM pipeline.
 *
 * Scene groups (eye icon to show/hide):
 *
 *   A_NonRegistered         ALL raw scapulae BEFORE any registration.
 *                           Right-side bones mirrored to the left anatomical frame.
 *                           These are at ORIGINAL resolution — NOT the 8k working mesh.
 *                           EXPECT: shapes spread in space, unaligned. That is correct.
 *
 *   B_PassN_Registered      Up to 8 non-rigidly registered shapes from pass N.
 *                           (σ=30 mm kernel, scaleFactor=10 mm, 40 GP-ICP iterations)
 *                           All should tightly overlap → good dense correspondence.
 *                           Toggle Pass1 vs Pass4 to confirm registration improves.
 *
 *   C_Means                 Mean1, Mean2, Mean3, Mean4 overlaid.
 *                           Should nearly coincide — console prints Mean1↔2, 2↔3, 3↔4 mm.
 *
 *   D_DistMap_PassN         Colour-coded surface-to-surface distance: registered → mean.
 *                           Blue = small residual, Red = large residual.
 *                           First 4 specimens shown per pass.
 *
 *   E_SSM_Interactive       Interactive SSM (last available pass, FULL resolution).
 *                           Click it → drag Mode 0 slider in right panel.
 *                           Mode 0 = main axis of shape variation.
 *
 *   F_Outliers              3 worst-fitting specimens shown alongside SSM mean.
 *
 *   G_ModeK (K=1,2,3)       Static shapes at Mean ± 1σ and ± 2σ along PCA mode K.
 *                           Shows the physical deformation each mode produces.
 *                           Toggling all 5 meshes in one group = Mode K atlas.
 *
 * Usage:
 *   sbt "runMain scapula.ViewSSM"
 *   SCAPULA_OUT_DIR=/your/path sbt "runMain scapula.ViewSSM"
 */
object ViewSSM {

  private def regFilesIn(dir: File): IndexedSeq[File] =
    if (!dir.isDirectory) IndexedSeq.empty
    else Option(dir.listFiles()).getOrElse(Array.empty[File])
      .filter(f => f.getName.endsWith(".stl") && f.getName.startsWith("reg_"))
      .sortBy(_.getName).toIndexedSeq

  /** Surface-distance colour map: reg mesh → target, shown as ScalarMeshField. */
  private def showDistMap(
    ui: ScalismoUI,
    group: scalismo.ui.api.Group,
    regMesh: TriangleMesh[_3D],
    target:  TriangleMesh[_3D],
    name: String
  ): Unit = {
    val dists = Metrics.surfaceDistances(regMesh, target).map(_.toFloat)
    val field = ScalarMeshField(regMesh, dists)
    ui.show(group, field, name)
  }

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dataDir = Config.dataDir
    val baseDir = if (args.nonEmpty) new File(args(0)) else Config.outDir

    println("\n" + "=" * 72)
    println("  Scapula SSM — Full Visual Inspection")
    println("=" * 72)

    val ui = ScalismoUI("Scapula SSM — Visual Inspection")

    // ── A. Non-registered scapulae at ORIGINAL resolution ─────────────────
    println("\n[A] Non-registered scapulae (ORIGINAL full-resolution STLs)")
    val specimens = ScapulaData.specimens(dataDir)
    val origGrp   = ui.createGroup(
      s"A_NonRegistered (${specimens.length} raw bones — BEFORE any registration)")
    specimens.foreach { spec =>
      val raw  = ScapulaData.loadMesh(spec.file)
      val mesh = if (spec.isRight) ScapulaData.mirrorMesh(raw) else raw
      val tag  = if (spec.isRight) "_R→L" else ""
      ui.show(origGrp, mesh, spec.modelId + tag)
    }
    println(s"  ${specimens.length} meshes loaded at original resolution.")
    println("  Right-side bones mirrored to left anatomical frame.")
    println("  EXPECT: shapes spread in space — unaligned is correct here.")
    println("  NOTE: these are the PRESERVED original meshes, NOT the 8k working copies.")

    // ── B. Registered meshes per pass ──────────────────────────────────────
    val passDirs = (1 to 8).map(n => n -> new File(baseDir, s"pass$n"))
      .filter(_._2.isDirectory)

    passDirs.foreach { case (n, passDir) =>
      val files = regFilesIn(passDir)
      if (files.nonEmpty) {
        println(s"\n[B] Pass $n registered meshes (${files.length} total, showing first 8)")
        println(s"  σ=${Config.kernelSigma} mm, scaleFactor=${Config.kernelScale} mm, " +
                s"res=${files.head.getName}")
        val grp = ui.createGroup(
          s"B_Pass${n}_Registered (${files.length} meshes, σ=${Config.kernelSigma.toInt}mm)")
        files.take(8).foreach { f =>
          ui.show(grp, ScapulaData.loadMesh(f),
            f.getName.stripSuffix(".stl").stripPrefix("reg_"))
        }
        println(s"  EXPECT: all shapes overlap tightly → good dense correspondence.")
        println(s"  Compare B_Pass1 vs B_Pass4 to confirm registration improvement.")
      }
    }

    // ── C. Mean1–Mean4 (convergence check) ────────────────────────────────
    println("\n[C] Mean shapes (Mean1–Mean4) — GPA convergence check")
    val meansGrp   = ui.createGroup("C_Means (Mean1–Mean4 overlaid — < 1 mm = converged)")
    val loadedMeans = scala.collection.mutable.ArrayBuffer.empty[(Int, TriangleMesh[_3D])]
    (1 to 8).foreach { n =>
      val f = new File(baseDir, s"mean_pass$n.stl")
      if (f.exists()) {
        val m = ScapulaData.loadMesh(f)
        ui.show(meansGrp, m, s"Mean$n")
        loadedMeans += (n -> m)
        println(s"  Mean$n: ${m.pointSet.numberOfPoints} vertices (mean_pass$n.stl)")
      }
    }
    if (loadedMeans.length >= 2) {
      println(f"\n  Surface-to-surface distances between consecutive means:")
      println(f"  ${"Comparison"}%-14s  ${"Mean (mm)"}%10s  ${"RMS (mm)"}%9s  ${"HD95 (mm)"}%10s  Status")
      println("  " + "─" * 60)
      loadedMeans.sliding(2).foreach { w =>
        val (n1, m1) = w(0); val (n2, m2) = w(1)
        val st = Metrics.symmetric(m1, m2)
        val ok = if (st.mean < 1.0) "✓ converged" else "  not yet "
        println(f"  Mean$n1 ↔ Mean$n2          ${st.mean}%10.3f  ${st.rms}%9.3f  ${st.hd95}%10.3f  $ok")
      }
    }

    // ── D. Colour-coded surface distance maps ──────────────────────────────
    // Build mean from last pass for reference
    passDirs.lastOption.foreach { case (n, passDir) =>
      val files = regFilesIn(passDir)
      if (files.nonEmpty) {
        // Use the saved mean if available, else compute from meshes
        val meanMeshOpt: Option[TriangleMesh[_3D]] =
          loadedMeans.lastOption.map(_._2)
            .orElse {
              println(s"  Computing mean for colour maps from pass $n…")
              val meshes = files.map(ScapulaData.loadMesh)
              val dc     = DataCollection.fromTriangleMesh3DSequence(meshes.head, meshes)
              Some(PointDistributionModel.createUsingPCA(dc).mean)
            }

        meanMeshOpt.foreach { mean =>
          println(s"\n[D] Surface distance colour maps: pass $n → mean")
          println("    Blue = small residual  |  Red = large residual")
          val distGrp = ui.createGroup(s"D_DistMap_Pass${n} (blue=good red=bad)")
          ui.show(distGrp, mean, s"REFERENCE_mean_pass$n")

          files.take(4).foreach { f =>
            val reg  = ScapulaData.loadMesh(f)
            val name = f.getName.stripSuffix(".stl").stripPrefix("reg_")
            val d    = Metrics.surfaceDistances(reg, mean)
            val avg  = d.sum / d.length
            println(f"    $name : avg=${avg}%.3f mm")
            showDistMap(ui, distGrp, reg, mean, s"${name}_dist")
          }
        }
      }
    }

    // ── E. Interactive SSM (no decimation — full resolution) ───────────────
    passDirs.lastOption.foreach { case (n, passDir) =>
      val files = regFilesIn(passDir)
      if (files.nonEmpty) {
        println(s"\n[E] Building interactive SSM from pass $n (${files.length} meshes, FULL res)…")
        val meshes = files.map(ScapulaData.loadMesh)
        val nPts   = meshes.head.pointSet.numberOfPoints
        println(s"  $nPts vertices per mesh — skipping decimation for best quality")

        val t0    = System.currentTimeMillis()
        val dc    = DataCollection.fromTriangleMesh3DSequence(meshes.head, meshes)
        val model = PointDistributionModel.createUsingPCA(dc)
        val evs   = model.gp.klBasis.map(_.eigenvalue).toArray
        val total = evs.sum
        val cumV  = evs.scanLeft(0.0)(_ + _).tail.map(_ / total * 100.0)
        def mFor(p: Double) = { val i = cumV.indexWhere(_ >= p); if (i < 0) model.rank else i + 1 }

        println(f"  Done in ${(System.currentTimeMillis()-t0)/1000.0}%.1f s")
        println(f"  rank=${model.rank}  |  90%%: ${mFor(90)} modes  |  95%%: ${mFor(95)}  |  99%%: ${mFor(99)}")
        println("\n  SSM MODE VARIANCE TABLE (SSM1=pass1, final=last pass):")
        println(f"  ${"Mode"}%5s  ${"Variance %"}%10s  ${"Cumul %"}%10s  ${"σ (mm)"}%8s")
        println("  " + "-" * 40)
        evs.take(10).zipWithIndex.foreach { case (ev, i) =>
          println(f"  ${i+1}%5d  ${ev/total*100}%10.2f  ${cumV(i)}%10.2f  ${math.sqrt(ev)}%8.3f")
        }

        val ssmGrp = ui.createGroup(s"E_SSM_Interactive (pass $n — drag Mode sliders!)")
        ui.show(ssmGrp, model, s"SSM_pass$n")
        println(s"\n  → Click 'E_SSM_Interactive' → drag Mode 0 slider on right panel")
        println(s"     Mode 0 accounts for ${evs(0)/total*100.0f}%.1f%% of shape variance")

        // ── G. Main modes of variance at ±1σ / ±2σ ────────────────────────
        println(s"\n[G] Main modes of variance — showing top 3 modes at ±1σ and ±2σ")
        println(f"  ${"Mode"}%5s  ${"σ (mm)"}%8s  ${"Var %"}%8s  ${"Shown as"}")
        println("  " + "-" * 48)
        val nModesToShow = math.min(3, model.rank)
        val mean         = model.mean
        (0 until nModesToShow).foreach { modeIdx =>
          val ev     = evs(modeIdx)
          val sigma  = math.sqrt(ev)
          val varPct = ev / total * 100.0
          println(f"  ${modeIdx+1}%5d  ${sigma}%8.3f  ${varPct}%8.2f  ±1σ + ±2σ meshes")

          val coeffs1pos = Array.tabulate(model.rank)(j => if (j == modeIdx)  1.0 * sigma else 0.0)
          val coeffs1neg = Array.tabulate(model.rank)(j => if (j == modeIdx) -1.0 * sigma else 0.0)
          val coeffs2pos = Array.tabulate(model.rank)(j => if (j == modeIdx)  2.0 * sigma else 0.0)
          val coeffs2neg = Array.tabulate(model.rank)(j => if (j == modeIdx) -2.0 * sigma else 0.0)

          val modeGrp = ui.createGroup(s"G_Mode${modeIdx+1} (${varPct.toInt}%% var, σ=${sigma.toInt}mm)")
          ui.show(modeGrp, mean,                                  s"Mode${modeIdx+1}_mean")
          ui.show(modeGrp, model.instance(DenseVector(coeffs1pos)), s"Mode${modeIdx+1}_+1sigma")
          ui.show(modeGrp, model.instance(DenseVector(coeffs1neg)), s"Mode${modeIdx+1}_-1sigma")
          ui.show(modeGrp, model.instance(DenseVector(coeffs2pos)), s"Mode${modeIdx+1}_+2sigma")
          ui.show(modeGrp, model.instance(DenseVector(coeffs2neg)), s"Mode${modeIdx+1}_-2sigma")
        }
        println("  EXPECT: toggle each G_ModeN group to see how each axis deforms the scapula")

        // ── F. Outlier specimens ────────────────────────────────────────────
        println(s"\n[F] Registration outlier check (specimen → SSM mean distance)…")
        val ranked = meshes.zip(files).map { case (m, f) =>
          val d = Metrics.surfaceDistances(m, mean)
          (d.sum / d.length, m, f.getName.stripSuffix(".stl").stripPrefix("reg_"))
        }.sortBy(-_._1)

        val all = ranked.map(_._1)
        println(f"  Population: mean=${all.sum/all.length}%.3f  max=${all.max}%.3f  min=${all.min}%.3f  mm")
        println(f"\n  Worst 3 (INVESTIGATE THESE):")
        val outlGrp = ui.createGroup("F_Outliers (worst fit to SSM mean)")
        ui.show(outlGrp, mean, "SSM_mean")
        ranked.take(3).foreach { case (d, m, name) =>
          println(f"    [!]  $name%-40s  $d%.3f mm")
          ui.show(outlGrp, m, f"OUTLIER_${d}%.2fmm_$name")
        }
        println(f"\n  Best 3 (well-registered):")
        ranked.takeRight(3).reverse.foreach { case (d, _, name) =>
          println(f"    [✓]  $name%-40s  $d%.3f mm")
        }
      }
    }

    println("\n" + "=" * 72)
    println("  HOW TO USE")
    println("=" * 72)
    println("  Eye icon       : show / hide any group")
    println("  A_Original     : raw — should look unaligned (non-registered surfaces)")
    println("  B_PassN        : toggle pass 1 vs pass 4 to see registration improvement")
    println("  C_Means        : overlaid means — nearly identical = converged GPA")
    println("  D_DistMap      : colour map — blue=good, red=large residual error")
    println("  E_SSM          : click it → drag Mode 0 slider to explore variation")
    println("  F_Outliers     : worst specimens — re-register or exclude if > 3× avg")
    println("  G_ModeN        : Mode N shapes at ±1σ and ±2σ — main axes of variation")
    println()
    println("  TO COMPUTE SSM VALIDATION METRICS (compactness / generalization / specificity):")
    println("    sbt \"runMain scapula.SSMValidation\"")
    println("=" * 72)
    println("\nUI open. Close the window to exit.")
  }
}
