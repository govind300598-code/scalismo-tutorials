package scapula

import scalismo.geometry.*
import scalismo.io.MeshIO
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.ui.api.*
import scalismo.utils.Random

import java.awt.Color
import java.io.File

/**
 * Registration Comparison + PDM Smoothness Viewer
 * ==================================================
 *
 * Pipeline per specimen:
 *   1. Landmark-based rigid registration (Procrustes) + rigid ICP refinement
 *   2. Reference decimated to <= 8000 points
 *   3. Multi-scale Gaussian Process prior built on the decimated reference
 *   4. GP-ICP non-rigid registration -> fitted mesh (same topology as reference)
 *
 * After registering ALL specimens, this ranks them by RMS registration error and
 * opens ScalismoUI showing:
 *
 *   Registration_Comparison_Results
 *     Reference_Template                            [yellow]
 *     Target_<id> (Best Case)      [red]   Fitted_<id> (Best Case)      [white]
 *     Target_<id> (Standard Target)[red] x3 Fitted_<id> (Standard Target)[white]
 *     Target_<id> (Worst Case)     [red]   Fitted_<id> (Worst Case)     [white]
 *
 *   Statistical_Shape_Model_Modes
 *     Scapula_PDM_Modes   <- drag "shape" sliders
 *
 * Run with:
 *   sbt "runMain scapula.RegistrationComparisonViewer"
 *
 * Configurable via environment variables (falls back to Config.* / sensible defaults):
 *   SCAPULA_REF_ID        - reference specimen id (default: first specimen found)
 *   SCAPULA_DECIMATE_PTS  - max points after decimation (default: 8000)
 */
object RegistrationComparisonViewer {

  val recommendedScales: Seq[(Double, Double)] = Seq(
    (100.0, 15.0),
    (50.0,  10.0),
    (20.0,  10.0),
    (10.0,   3.0)
  )

  case class RegisteredCase(
    specimenId: String,
    target:     TriangleMesh[_3D],
    fitted:     TriangleMesh[_3D],
    rmsMm:      Double,
    hd95Mm:     Double
  )

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dataDir    = Config.dataDir
    val decimateTo = sys.env.get("SCAPULA_DECIMATE_PTS").map(_.toInt).getOrElse(8000)

    println("=" * 70)
    println("  Registration Comparison + PDM Smoothness Viewer")
    println("=" * 70)
    println(s"  Data       : ${dataDir.getAbsolutePath}")
    println(s"  Decimation : <= $decimateTo points")
    println(s"  Kernel     : sigma=20.0, scaleFactor=5.0 (single Gaussian)")

    // ---------------------------------------------------------------- load
    val csv = ScapulaData.csvFile(dataDir)
    val (landmarks, fromHeader, _) = ScapulaData.readLandmarkCsv(csv)
    if (!fromHeader)
      println("WARNING: landmark columns resolved by fallback offsets -- verify against your CSV")

    val allSpecimens = ScapulaData.specimens(dataDir)
    val specimens    = allSpecimens.filter(s => landmarks.contains(s.modelId))
    println(s"Specimens with landmarks: ${specimens.length} / ${allSpecimens.length}")
    require(specimens.length >= 5, "Need at least 5 specimens for a Best/Standard/Worst comparison")

    // ---------------------------------------------------------------- reference
    val refSpec = sys.env.get("SCAPULA_REF_ID")
      .flatMap(id => specimens.find(_.modelId == id))
      .getOrElse(specimens.head)
    println(s"Reference: ${refSpec.modelId}  (override with SCAPULA_REF_ID env var)")

    val rawRef     = ScapulaData.loadMesh(refSpec.file)
    val decimatePts = math.min(decimateTo, rawRef.pointSet.numberOfPoints)
    val reference   = rawRef.operations.decimate(decimatePts)
    println(s"Reference decimated: ${rawRef.pointSet.numberOfPoints} -> ${reference.pointSet.numberOfPoints} points")

    val refLms = if (refSpec.isRight) ScapulaData.mirrorLandmarks(landmarks(refSpec.modelId))
                 else landmarks(refSpec.modelId)

    // ---------------------------------------------------------------- GP prior
    println("\nBuilding multi-scale GP prior on decimated reference...")
    val gpPrior = NonRigidReg.buildGpPrior(reference, recommendedScales)
    println(s"GP prior rank: ${gpPrior.rank}")

    // ---------------------------------------------------------------- register all (excluding reference)
    val targets = specimens.filterNot(_.modelId == refSpec.modelId)
    println(s"\nRegistering ${targets.length} target specimens...\n")

    val results: IndexedSeq[RegisteredCase] = targets.zipWithIndex.map { case (spec, idx) =>
      print(f"  [${idx + 1}%3d/${targets.length}%3d] ${spec.modelId}%-30s ... ")

      val rawMesh = ScapulaData.loadMesh(spec.file)
      val rawLms  = landmarks(spec.modelId)

      val (mesh, lms) =
        if (spec.isRight) (ScapulaData.mirrorMesh(rawMesh), ScapulaData.mirrorLandmarks(rawLms))
        else (rawMesh, rawLms)

      val (rigidAligned, _) =
        RigidAlign.landmarkThenIcp(mesh, lms, reference, refLms, icpIterations = 30)

      val fitted = NonRigidReg.gpIcp(
        reference  = reference,
        target     = rigidAligned,
        lowRankGP  = gpPrior,
        iterations = Config.icpIterations
      )

      val sym = Metrics.symmetric(fitted, rigidAligned)
      println(f"RMS=${sym.rms}%.3f mm  HD95=${sym.hd95}%.3f mm")

      RegisteredCase(spec.modelId, rigidAligned, fitted, sym.rms, sym.hd95)
    }

    // ---------------------------------------------------------------- rank & classify
    val ranked    = results.sortBy(_.rmsMm)
    val best      = ranked.head
    val worst     = ranked.last
    val nStandard = math.min(3, math.max(0, ranked.length - 2))
    val standardCases = {
      val middle = ranked.slice(1, ranked.length - 1)
      if (middle.isEmpty) IndexedSeq.empty
      else {
        val step = math.max(1, middle.length / (nStandard + 1))
        (1 to nStandard).map(i => middle(math.min(i * step, middle.length - 1)))
      }
    }

    println("\n" + "=" * 70)
    println("  RANKED RESULTS (by RMS, mm)")
    println("=" * 70)
    ranked.foreach(r => println(f"  ${r.specimenId}%-30s RMS=${r.rmsMm}%.3f  HD95=${r.hd95Mm}%.3f"))
    println(s"\n  Best Case      : ${best.specimenId}  (RMS=${best.rmsMm}%.3f mm)")
    standardCases.foreach(c => println(f"  Standard Target: ${c.specimenId}%-25s (RMS=${c.rmsMm}%.3f mm)"))
    println(f"  Worst Case     : ${worst.specimenId}  (RMS=${worst.rmsMm}%.3f mm)")

    // ---------------------------------------------------------------- save CSV
    val outDir = Config.outDir
    outDir.mkdirs()
    Evaluate.saveCsv(
      Seq(Seq("specimen", "rms_mm", "hd95_mm")) ++
        ranked.map(r => Seq(r.specimenId, f"${r.rmsMm}%.4f", f"${r.hd95Mm}%.4f")),
      new File(outDir, "registration_comparison_ranked.csv")
    )
    println(s"\nSaved: ${new File(outDir, "registration_comparison_ranked.csv").getName}")

    // ---------------------------------------------------------------- build a PDM from ALL fitted meshes
    println("\nBuilding PDM from all fitted (in-correspondence) meshes for smoothness inspection...")
    val dc    = DataCollection.fromTriangleMesh3DSequence(reference, results.map(_.fitted))
    val dcGpa = DataCollection.gpa(dc)
    val pdm   = PointDistributionModel.createUsingPCA(dcGpa)
    println(s"PDM built: rank=${pdm.rank}  vertices=${pdm.mean.pointSet.numberOfPoints}")

    // ---------------------------------------------------------------- ScalismoUI
    println("\nLaunching ScalismoUI...")
    val ui = ScalismoUI()

    val resultsGroup = ui.createGroup("Registration_Comparison_Results")

    val refView = ui.show(resultsGroup, reference, "Reference_Template")
    refView.color = Color.YELLOW

    def showCase(c: RegisteredCase, label: String): Unit = {
      val targetView = ui.show(resultsGroup, c.target, s"Target_${c.specimenId} ($label)")
      targetView.color = Color.RED
      val fittedView = ui.show(resultsGroup, c.fitted, s"Fitted_${c.specimenId} ($label)")
      fittedView.color = Color.WHITE
    }

    showCase(best, "Best Case")
    standardCases.foreach(c => showCase(c, "Standard Target"))
    showCase(worst, "Worst Case")

    val ssmGroup = ui.createGroup("Statistical_Shape_Model_Modes")
    // explicit type ascription avoids the ambiguous ShowInScene given
    val pdmTyped: PointDistributionModel[_3D, TriangleMesh] = pdm
    ui.show(ssmGroup, pdmTyped, "Scapula_PDM_Modes")

    println("\n" + "=" * 70)
    println("  UI READY")
    println("=" * 70)
    println("  - Rotate/inspect Target (red) vs Fitted (white) overlap per case.")
    println("    Tight, near-total white-under-red coverage = good fit.")
    println("    Large exposed red patches = poor local fit at that scale/region.")
    println("  - Open 'Shape model transformations' -> drag 'shape' coefficient")
    println("    sliders on Scapula_PDM_Modes. Deformation should stay smooth --")
    println("    watch for spikes, self-intersection, or creasing as red flags.")
    println("=" * 70)
  }
}
