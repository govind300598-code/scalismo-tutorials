package scapula

import scalismo.io.MeshIO
import scalismo.ui.api.ScalismoUI

import java.awt.Color
import java.io.File

/**
 * Loads ALL final registered (fitted) meshes from Config.outDir and shows them overlaid in one ScalismoUI window.
 * Useful for visually checking population spread and correspondence quality at a glance.
 *
 * Run on your LOCAL machine (where SCAPULA_OUT_DIR points to your results):
 *   sbt "runMain scapula.ViewAllRegistered"
 *
 * What you will see:
 *   - "reference" group : the final mean-shape reference (solid, visible)
 *   - "all registered fits" group : all 22 warped meshes overlaid, each a different color at 40% opacity
 *   - "all targets" group : the 22 original (rigidly aligned) target scans, hidden by default
 *
 * Use the Appearance panel's Opacity slider to toggle individual meshes on/off.
 */
object ViewAllRegistered {

  // 22 visually distinct colors cycling through hue
  private val palette: Array[Color] = Array(
    new Color(231,  76,  60),  // red
    new Color( 52, 152, 219),  // blue
    new Color( 46, 204, 113),  // green
    new Color(241, 196,  15),  // yellow
    new Color(155,  89, 182),  // purple
    new Color( 26, 188, 156),  // teal
    new Color(230, 126,  34),  // orange
    new Color( 52,  73,  94),  // dark blue-grey
    new Color(149, 165, 166),  // silver
    new Color(211,  84,   0),  // dark orange
    new Color( 39, 174,  96),  // dark green
    new Color( 41, 128, 185),  // dark blue
    new Color(142,  68, 173),  // dark purple
    new Color( 22, 160, 133),  // dark teal
    new Color(243, 156,  18),  // amber
    new Color(192,  57,  43),  // dark red
    new Color( 44, 62,  80),   // navy
    new Color(127, 140, 141),  // grey
    new Color(210, 180, 140),  // tan
    new Color( 72, 201, 176),  // aquamarine
    new Color(250, 128, 114),  // salmon
    new Color(144, 238, 144),  // light green
  )

  def main(args: Array[String]): Unit = {
    scalismo.initialize()

    val dir = Config.outDir
    require(dir.exists(),
      s"""Output directory not found: ${dir.getAbsolutePath}
         |Run Stage2ReferenceRefinement first, or set SCAPULA_OUT_DIR to your results directory.""".stripMargin)

    val ui = ScalismoUI()

    // ── Reference ──────────────────────────────────────────────────────────────
    val refFile = new File(dir, "reference_mean_shape.stl")
    if (refFile.exists()) {
      val refGroup = ui.createGroup("reference (mean shape)")
      ui.show(refGroup, MeshIO.readMesh(refFile).get, "reference_mean_shape")
      println(s"Reference loaded: ${refFile.getName}")
    } else {
      println(s"!! reference_mean_shape.stl not found in ${dir.getAbsolutePath}")
    }

    // ── All fitted meshes overlaid ─────────────────────────────────────────────
    val fitFiles = Option(dir.listFiles((_, name) => name.startsWith("final_") && name.endsWith("_fit.stl")))
      .getOrElse(Array.empty[File])
      .sortBy(_.getName)

    if (fitFiles.isEmpty) {
      println(s"!! No final_*_fit.stl files found in ${dir.getAbsolutePath}")
      println( "   Make sure Stage2ReferenceRefinement has finished.")
    } else {
      val fitsGroup = ui.createGroup("all registered fits (overlaid)")
      fitFiles.zipWithIndex.foreach { case (f, i) =>
        val subjectId = f.getName.stripPrefix("final_").stripSuffix("_fit.stl")
        val view = ui.show(fitsGroup, MeshIO.readMesh(f).get, subjectId)
        view.color   = palette(i % palette.length)
        view.opacity = 0.40
      }
      println(s"Loaded ${fitFiles.length} registered fits — all overlaid, each a different color at 40% opacity.")
      println( "  -> Use the Appearance panel to adjust opacity or isolate individual subjects.")
    }

    // ── All target scans (hidden by default) ───────────────────────────────────
    val targetFiles = Option(dir.listFiles((_, name) => name.startsWith("final_") && name.endsWith("_target.stl")))
      .getOrElse(Array.empty[File])
      .sortBy(_.getName)

    if (targetFiles.nonEmpty) {
      val targetsGroup = ui.createGroup("all targets (hidden — enable per subject)")
      targetFiles.foreach { f =>
        val subjectId = f.getName.stripPrefix("final_").stripSuffix("_target.stl")
        val view = ui.show(targetsGroup, MeshIO.readMesh(f).get, subjectId)
        view.opacity = 0.0  // hidden by default — too crowded to show all at once
      }
      println(s"Loaded ${targetFiles.length} target scans (all hidden — set opacity > 0 in Appearance panel to show one).")
    }

    println(s"\nDone. Files read from: ${dir.getAbsolutePath}")
    println( "Groups in the scene tree:")
    println( "  1. reference (mean shape)         — the unbiased population template")
    println( "  2. all registered fits (overlaid)  — all 22 warped to correspondence, shows population spread")
    println( "  3. all targets (hidden)            — original rigidly-aligned scans, enable one at a time")
  }
}
