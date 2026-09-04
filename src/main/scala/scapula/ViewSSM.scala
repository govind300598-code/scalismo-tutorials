package scapula

import scalismo.geometry._3D
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

import java.io.File

/**
 * Comprehensive visual inspection for the scapula SSM pipeline.
 *
 * Groups (click the eye icon in the Scene panel to show/hide each):
 *
 *   A_Original              Raw, non-registered scapulae (first 8).
 *                           R ones mirrored to L frame.
 *                           These should look spread/unaligned — that is correct.
 *
 *   B_PassN_Registered      Non-rigidly registered meshes from pass N (first 8).
 *                           All shapes should tightly overlap.
 *                           Compare pass 1 vs pass 4 to see registration improvement.
 *
 *   C_Means                 Mean shapes from every pass overlaid.
 *                           They should converge (< 1 mm shift between passes).
 *
 *   D_SSM_Interactive       Interactive SSM from the last available pass.
 *                           Click it → drag "Mode 0" slider on the right panel
 *                           to explore the main axis of shape variation.
 *
 *   E_Outlier_specimens     The 3 worst-fitting specimens (highest mean surface
 *                           distance to SSM mean) shown alongside the mean.
 *                           Use these to spot failed registrations.
 *
 * Usage:
 *   sbt "runMain scapula.ViewSSM"
 *   SCAPULA_OUT_DIR=/your/path sbt "runMain scapula.ViewSSM [/output/dir]"
 */
object ViewSSM {

  private def regFilesIn(dir: File): IndexedSeq[File] =
    if (!dir.isDirectory) IndexedSeq.empty
    else Option(dir.listFiles()).getOrElse(Array.empty[File])
      .filter(f => f.getName.endsWith(".stl") && f.getName.startsWith("reg_"))
      .sortBy(_.getName).toIndexedSeq

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dataDir = Config.dataDir
    val baseDir = if (args.nonEmpty) new File(args(0)) else Config.outDir

    println("\n" + "=" * 70)
    println("  Scapula SSM — Visual Inspection")
    println("=" * 70)

    val ui = ScalismoUI("Scapula SSM — Visual Inspection")

    // ── A. Original (non-registered) scapulae ─────────────────────────────
    println("\n[A] Original (non-registered) scapulae")
    val specimens = ScapulaData.specimens(dataDir)
    val origGroup = ui.createGroup("A_Original (raw, before registration)")
    specimens.take(8).foreach { spec =>
      val raw  = ScapulaData.loadMesh(spec.file)
      val mesh = if (spec.isRight) ScapulaData.mirrorMesh(raw) else raw
      ui.show(origGroup, mesh, spec.modelId + (if (spec.isRight) "_mirrored" else ""))
    }
    println(s"  ${math.min(8, specimens.length)} specimens loaded (R ones mirrored to L frame)")
    println("  EXPECT: shapes spread across space — registration will align these")

    // ── B. Registered meshes per pass ─────────────────────────────────────
    val passDirs = (1 to 4).map(n => n -> new File(baseDir, s"pass$n"))
      .filter(_._2.isDirectory)

    passDirs.foreach { case (n, passDir) =>
      val allReg = regFilesIn(passDir)
      if (allReg.nonEmpty) {
        println(s"\n[B] Pass $n registered meshes (${allReg.length} total, showing first 8)")
        val grp = ui.createGroup(s"B_Pass${n}_Registered")
        allReg.take(8).foreach { f =>
          ui.show(grp, ScapulaData.loadMesh(f),
            f.getName.stripSuffix(".stl").stripPrefix("reg_"))
        }
        println("  EXPECT: tight overlap of all shapes → good dense correspondence")
      }
    }

    // ── C. Mean shapes (convergence check) ────────────────────────────────
    println("\n[C] Mean shapes from each pass (convergence check)")
    val meansGroup = ui.createGroup("C_Means (should converge < 1 mm)")
    val loadedMeans = scala.collection.mutable.ArrayBuffer.empty[(Int, TriangleMesh[_3D])]
    (1 to 4).foreach { n =>
      val f = new File(baseDir, s"mean_pass$n.stl")
      if (f.exists()) {
        val m = ScapulaData.loadMesh(f)
        ui.show(meansGroup, m, s"mean_pass$n")
        loadedMeans += (n -> m)
        println(s"  mean_pass$n: ${m.pointSet.numberOfPoints} vertices")
      }
    }
    if (loadedMeans.length >= 2) {
      println("  Consecutive mean distances:")
      loadedMeans.sliding(2).foreach { w =>
        val (n1, m1) = w(0); val (n2, m2) = w(1)
        val st = Metrics.symmetric(m1, m2)
        val ok = if (st.mean < 1.0) " ✓" else " !"
        println(f"    pass$n1→pass$n2 : mean=${st.mean}%.3f  rms=${st.rms}%.3f  HD95=${st.hd95}%.3f  mm$ok")
      }
      println("  EXPECT: mean shift < 1 mm = reference bias removed")
    }

    // ── D. Interactive SSM from the last available pass ───────────────────
    passDirs.lastOption.foreach { case (n, passDir) =>
      val allReg = regFilesIn(passDir)
      if (allReg.nonEmpty) {
        println(s"\n[D] Building interactive SSM from pass $n (${allReg.length} meshes)…")

        val meshes = allReg.map(ScapulaData.loadMesh)
        val nPts   = meshes.head.pointSet.numberOfPoints
        val target = Config.modelResolution

        val workMeshes =
          if (nPts > target) {
            println(s"  Decimating $nPts → ~$target vertices…")
            val dec = ScapulaData.decimateInCorrespondence(meshes.head, meshes, target)
            println(s"  Actual: ${dec.head.pointSet.numberOfPoints} vertices")
            dec
          } else {
            println(s"  $nPts vertices ≤ $target — no decimation needed")
            meshes
          }

        println("  Running PCA…")
        val t0    = System.currentTimeMillis()
        val dc    = DataCollection.fromTriangleMesh3DSequence(workMeshes.head, workMeshes)
        val model = PointDistributionModel.createUsingPCA(dc)
        val evs   = model.gp.klBasis.map(_.eigenvalue).toArray
        val total = evs.sum
        val cumVar = evs.scanLeft(0.0)(_ + _).tail.map(_ / total * 100.0)
        def modesFor(p: Double) = { val i = cumVar.indexWhere(_ >= p); if (i < 0) model.rank else i + 1 }

        println(f"  Done in ${(System.currentTimeMillis()-t0)/1000.0}%.1f s  rank=${model.rank}")
        println(f"  90%% variance: ${modesFor(90)} modes  |  95%%: ${modesFor(95)}  |  99%%: ${modesFor(99)}")
        println("  Top modes:")
        evs.take(5).zipWithIndex.foreach { case (ev, i) =>
          println(f"    Mode ${i+1}: ${ev/total*100}%.1f%%  (σ=${math.sqrt(ev)}%.2f mm)")
        }

        val ssmGroup = ui.createGroup(s"D_SSM_Interactive (pass $n)")
        ui.show(ssmGroup, model, s"SSM_pass$n")
        println(s"  → Click 'D_SSM_Interactive' in Scene panel, then drag Mode sliders on right")

        // ── E. Outlier specimens ───────────────────────────────────────────
        println(s"\n[E] Registration quality check (distance to SSM mean)…")
        val meanMesh = model.mean
        val specDists = workMeshes.zip(allReg).map { case (m, f) =>
          val d = Metrics.surfaceDistances(m, meanMesh)
          (d.sum / d.length, m, f.getName.stripSuffix(".stl").stripPrefix("reg_"))
        }.sortBy(-_._1)

        val outlierGroup = ui.createGroup("E_Outlier_specimens (worst fit to mean)")
        ui.show(outlierGroup, meanMesh, "SSM_mean")

        println("  Worst 3 (flag for manual inspection):")
        specDists.take(3).foreach { case (dist, m, name) =>
          println(f"    [!] $name : $dist%.3f mm to mean")
          ui.show(outlierGroup, m, f"OUTLIER_${name}_${dist}%.2fmm")
        }
        println("  Best 3:")
        specDists.takeRight(3).reverse.foreach { case (dist, _, name) =>
          println(f"    [✓] $name : $dist%.3f mm to mean")
        }
        val allDists = specDists.map(_._1)
        println(f"  Overall: mean=${allDists.sum/allDists.length}%.3f  " +
                f"max=${allDists.max}%.3f  min=${allDists.min}%.3f  mm")
      }
    }

    println("\n" + "=" * 70)
    println("  HOW TO USE THE VIEWER")
    println("=" * 70)
    println("  Eye icon  : show / hide a group")
    println("  A_Original: raw input — should look unaligned")
    println("  B_PassN   : registered — should overlap tightly; compare pass 1 vs 4")
    println("  C_Means   : overlaid means — should be nearly identical if converged")
    println("  D_SSM     : click it → drag Mode 0 slider to explore shape variation")
    println("  E_Outlier : worst-fitting shapes — investigate if dist >> mean dist")
    println("=" * 70)
    println("\nUI open. Close the window to exit.")
  }
}
