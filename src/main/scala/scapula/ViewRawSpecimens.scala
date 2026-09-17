package scapula

import scalismo.ui.api.ScalismoUI

/**
 * Shows every RAW specimen STL exactly as it is in Config.dataDir -- no rigid alignment, no mirroring, no
 * registration, nothing computed or changed. Purely so you can manually, visually inspect your own real data:
 * orientation, obvious defects, which side is which, etc., with zero trust required in any of the pipeline code.
 *
 * Left and right specimens are put in separate scalismo-ui groups (by filename, via ScapulaData.specimens'
 * isRight flag) purely for organisation in the scene tree; nothing about the meshes themselves is touched.
 */
object ViewRawSpecimens {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()

    val dir = Config.dataDir
    val specimens = ScapulaData.specimens(dir)
    require(specimens.nonEmpty, s"No STL files found in ${dir.getPath}")
    println(s"${specimens.length} raw specimens found in ${dir.getPath}:")
    specimens.foreach(s => println(s"  ${s.modelId}  (${if (s.isRight) "right" else "left"})"))

    if (Config.showUi) {
      val ui = ScalismoUI()
      val leftGroup = ui.createGroup("left (raw)")
      val rightGroup = ui.createGroup("right (raw)")
      specimens.foreach { s =>
        val mesh = ScapulaData.loadMesh(s.file)
        val group = if (s.isRight) rightGroup else leftGroup
        ui.show(group, mesh, s.modelId)
      }
      println(s"\nScalismo-UI window opened with all ${specimens.length} RAW specimens, grouped into 'left (raw)' " +
        "and 'right (raw)' in the scene tree -- exactly as stored in your STL files, no processing applied. Each " +
        "specimen sits at whatever position/orientation it was originally scanned at, so they will NOT overlap " +
        "each other (that's expected here -- overlap only happens after rigid alignment, which this viewer " +
        "deliberately skips). Toggle each one on/off individually to inspect its shape and orientation by eye. " +
        "Close the window when done.")
    } else {
      println("SCAPULA_UI=false -- skipping the interactive viewer; the specimen list above is the only output.")
    }
  }
}
