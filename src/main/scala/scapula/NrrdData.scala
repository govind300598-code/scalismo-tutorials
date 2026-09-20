package scapula

import scalismo.common.PointId
import scalismo.geometry.*
import scalismo.image.DiscreteImage
import scalismo.io.{ImageIO, MeshIO}
import scalismo.mesh.*

import java.io.File

/**
 * Discovers, loads and preprocesses paired CT volume / segmentation NRRD files.
 *
 * File naming convention expected (from 3D Slicer exports):
 *   <id>_volume.nrrd          -- raw CT intensities in Hounsfield Units (stored as Short)
 *   <id>_scapula_0.seg.nrrd   -- binary label map: 1 = scapula, 0 = background
 *
 * A pre-extracted <id>.stl surface mesh must sit in the same directory.
 * Export the surface model from 3D Slicer (Models module → Export as STL).
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
   * Load a pre-exported STL surface mesh.  Marching Cubes is not available in Scalismo 0.92.x,
   * so the STL must be exported from 3D Slicer (Segment Editor → Export to Files → STL).
   */
  def extractSurface(spec: CtSpecimen): TriangleMesh[_3D] =
    spec.stlFile match {
      case Some(f) =>
        MeshIO.readMesh(f).getOrElse(throw new RuntimeException(s"Cannot read STL: ${f.getName}"))
      case None =>
        throw new RuntimeException(
          s"No STL surface mesh found for ${spec.id}.\n" +
          "Export the scapula segmentation as a surface model (STL) from 3D Slicer\n" +
          s"and place it next to the NRRD files as ${spec.id}.stl"
        )
    }

  /**
   * Sample Hounsfield Units at mesh vertices using nearest-neighbour voxel lookup.
   * Points outside the CT volume domain fall back to −1000 HU (air).
   */
  def sampleHU(mesh: TriangleMesh[_3D], volume: DiscreteImage[_3D, Short]): IndexedSeq[Float] = {
    val domain = volume.domain
    val o  = domain.origin
    val sp = domain.spacing
    val sz = domain.size
    val nx = sz(0); val ny = sz(1); val nz = sz(2)
    mesh.pointSet.points.map { pt =>
      val ci = math.round((pt.x - o.x) / sp.x).toInt
      val cj = math.round((pt.y - o.y) / sp.y).toInt
      val ck = math.round((pt.z - o.z) / sp.z).toInt
      if (ci >= 0 && ci < nx && cj >= 0 && cj < ny && ck >= 0 && ck < nz)
        volume(PointId(ci + nx * (cj + ny * ck))).toFloat
      else -1000f
    }.toIndexedSeq
  }

  def huStats(hu: IndexedSeq[Float]): (Float, Float, Float) =
    (hu.min, hu.max, hu.sum / hu.length)

  def corticalMask(hu: IndexedSeq[Float], threshold: Float = 300f): IndexedSeq[Boolean] =
    hu.map(_ >= threshold)
}
