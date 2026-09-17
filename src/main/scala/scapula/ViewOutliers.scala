package scapula

import scalismo.geometry.*
import scalismo.mesh.TriangleMesh
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

/**
 * Rigidly aligns a chosen "most average" reference specimen and a chosen list of "outlier" (highest shape-variation)
 * specimens onto a common frame, and shows them together in scalismo-ui, so the outliers can be visually inspected
 * against the reference to see WHAT is different about each one -- size, glenoid angle, acromion/coracoid shape,
 * etc. -- rather than just trusting a summary statistic that flagged them as outliers.
 *
 * Reference/outlier IDs default to the ones supplied externally; override with SCAPULA_REFERENCE_SPECIMEN and
 * SCAPULA_OUTLIER_SPECIMENS (comma-separated model IDs, e.g. paired_scapula_001_M_64_L).
 */
object ViewOutliers {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val referenceId = sys.env.getOrElse("SCAPULA_REFERENCE_SPECIMEN", "paired_scapula_005_F_67_R")
    val outlierIds = sys.env.getOrElse(
      "SCAPULA_OUTLIER_SPECIMENS",
      "paired_scapula_002_M_56_L,paired_scapula_006_F_60_R,paired_scapula_001_M_64_L," +
        "paired_scapula_008_F_73_L,paired_scapula_007_M_26_L"
    ).split(",").map(_.trim).filter(_.nonEmpty).toIndexedSeq

    val dir = Config.dataDir
    val csv = ScapulaData.csvFile(dir)
    val (landmarks, _, _) = ScapulaData.readLandmarkCsv(csv)
    val specimens = ScapulaData.specimens(dir).filter(s => landmarks.contains(s.modelId))
    val byId = specimens.map(s => s.modelId -> s).toMap

    // Same left-side canonicalization as the rest of the pipeline -- a right-side outlier must be mirrored before
    // it can be meaningfully compared against a left-side (or already-mirrored) reference.
    def canonical(id: String): (TriangleMesh[_3D], IndexedSeq[Landmark[_3D]]) = {
      val s = byId.getOrElse(
        id,
        throw new RuntimeException(s"Unknown specimen id '$id' -- must be one of: ${specimens.map(_.modelId).mkString(", ")}")
      )
      val rawMesh = ScapulaData.loadMesh(s.file)
      val rawLms = landmarks(s.modelId)
      if (s.isRight) (ScapulaData.mirrorMesh(rawMesh), ScapulaData.mirrorLandmarks(rawLms)) else (rawMesh, rawLms)
    }

    val (refMesh, refLms) = canonical(referenceId)
    println(s"Reference (most average): $referenceId")

    val alignedOutliers = outlierIds.map { id =>
      val (mesh, lms) = canonical(id)
      val (aligned, _) = RigidAlign.landmarkThenIcp(mesh, lms, refMesh, refLms, Config.icpIterations)
      val stats = Metrics.symmetric(aligned, refMesh)
      println(f"  $id%-24s vs. reference: ${stats.render}")
      id -> aligned
    }

    if (Config.showUi) {
      val ui = ScalismoUI()
      val refGroup = ui.createGroup("reference (most average)")
      ui.show(refGroup, refMesh, referenceId)
      val outlierGroup = ui.createGroup("outliers (variation)")
      alignedOutliers.foreach { case (id, mesh) => ui.show(outlierGroup, mesh, id) }
      println(s"\nScalismo-UI window opened with 'reference (most average)' (${referenceId}) and 'outliers " +
        s"(variation)' (${alignedOutliers.length} specimens) groups. Toggle each outlier on/off against the " +
        "reference in the scene tree to see exactly what shape difference sets it apart -- overall size, glenoid " +
        "angle, acromion/coracoid shape, etc. Close the window when done.")
    } else {
      println("SCAPULA_UI=false -- skipping the interactive viewer; the residuals above are the only output.")
    }
  }
}
