package scapula

import scalismo.geometry.*
import scalismo.image.DiscreteImage
import scalismo.image.interpolation.LinearImageInterpolator3D
import scalismo.io.ImageIO
import scalismo.mesh.*

import java.io.File

/**
 * Discovers, loads and preprocesses paired CT volume / segmentation NRRD files.
 *
 * File naming convention expected (from 3D Slicer exports):
 *   <id>_volume.nrrd          -- raw CT intensities in Hounsfield Units (stored as Short)
 *   <id>_scapula_0.seg.nrrd   -- binary label map: 1 = scapula, 0 = background
 *
 * If a pre-extracted <id>.stl surface mesh also sits in the same directory it is
 * preferred over running Marching Cubes (faster, and avoids the dependency).
 */
object NrrdData {

  final case class CtSpecimen(id: String, volumeFile: File, segFile: File, stlFile: Option[File])

  def discoverSpecimens(dir: File): IndexedSeq[CtSpecimen] = {
    val all    = Option(dir.listFiles()).getOrElse(Array.empty[File])
    val byName = all.map(f => f.getName -> f).toMap

    all
      .filter(_.getName.endsWith("_volume.nrrd"))
      .sortBy(_.getName)
      .flatMap { vol =>
        val id  = vol.getName.stripSuffix("_volume.nrrd")
        byName.get(s"${id}_scapula_0.seg.nrrd").map { seg =>
          val stl = byName.get(s"$id.stl")
          CtSpecimen(id, vol, seg, stl)
        }
      }
      .toIndexedSeq
  }

  def loadVolume(file: File): DiscreteImage[_3D, Short] =
    ImageIO
      .read3DScalarImage[Short](file)
      .getOrElse(throw new RuntimeException(s"Cannot read CT volume: ${file.getName}"))

  def loadSegmentation(file: File): DiscreteImage[_3D, Short] =
    ImageIO
      .read3DScalarImage[Short](file)
      .getOrElse(throw new RuntimeException(s"Cannot read segmentation: ${file.getName}"))

  /**
   * Extract a surface mesh from a binary segmentation.
   *
   * Prefers a pre-exported STL (faster).  Falls back to Marching Cubes from
   * the segmentation.  If neither works the exception message tells the user to
   * export an STL from 3D Slicer instead.
   */
  def extractSurface(spec: CtSpecimen): TriangleMesh[_3D] = {
    spec.stlFile match {
      case Some(f) =>
        MeshIO.readMesh(f).getOrElse(throw new RuntimeException(s"Cannot read STL: ${f.getName}"))

      case None =>
        val seg      = loadSegmentation(spec.segFile)
        val floatSeg = seg.map(_.toFloat)
        // MarchingCubes lives in scalismo.mesh
        try scalismo.mesh.MarchingCubes.marchingCubes(floatSeg, isoValue = 0.5)
        catch {
          case e: Exception =>
            throw new RuntimeException(
              s"Marching Cubes failed for ${spec.id}: ${e.getMessage}\n" +
              "Tip: export the segmentation as a surface model (STL) from 3D Slicer and place it\n" +
              s"next to the NRRD files as ${spec.id}.stl — it will be loaded automatically.",
              e
            )
        }
    }
  }

  /**
   * Sample Hounsfield Units at mesh vertices using linear interpolation of the CT volume.
   * Points outside the CT volume domain fall back to −1000 HU (air).
   */
  def sampleHU(mesh: TriangleMesh[_3D], volume: DiscreteImage[_3D, Short]): IndexedSeq[Float] = {
    val interp = volume.interpolate(LinearImageInterpolator3D[Short]())
    mesh.pointSet.points.map { pt =>
      try interp(pt).toFloat
      catch { case _: Exception => -1000f }
    }.toIndexedSeq
  }

  def huStats(hu: IndexedSeq[Float]): (Float, Float, Float) =
    (hu.min, hu.max, hu.sum / hu.length)

  def corticalMask(hu: IndexedSeq[Float], threshold: Float = 300f): IndexedSeq[Boolean] =
    hu.map(_ >= threshold)
}
