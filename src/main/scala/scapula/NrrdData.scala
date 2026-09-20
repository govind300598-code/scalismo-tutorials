package scapula

import scalismo.geometry.*
import scalismo.image.*
import scalismo.io.ImageIO
import scalismo.mesh.*

import java.io.File

/**
 * Discovers, loads and preprocesses paired CT volume / segmentation NRRD files.
 *
 * File naming convention expected (from 3D Slicer exports):
 *   <id>_volume.nrrd          -- raw CT intensities in Hounsfield Units (stored as Short)
 *   <id>_scapula_0.seg.nrrd   -- binary label map: 1 = scapula, 0 = background
 */
object NrrdData {

  /** One CT specimen: matched volume + segmentation paths. */
  final case class CtSpecimen(id: String, volumeFile: File, segFile: File)

  /**
   * Scans a directory for pairs of *_volume.nrrd and *_scapula_0.seg.nrrd files.
   * Specimens without both files are silently skipped.
   */
  def discoverSpecimens(dir: File): IndexedSeq[CtSpecimen] = {
    val all = Option(dir.listFiles()).getOrElse(Array.empty[File])
    val byName = all.map(f => f.getName -> f).toMap

    all
      .filter(_.getName.endsWith("_volume.nrrd"))
      .sortBy(_.getName)
      .flatMap { vol =>
        val id  = vol.getName.stripSuffix("_volume.nrrd")
        val seg = byName.get(s"${id}_scapula_0.seg.nrrd")
        seg.map(s => CtSpecimen(id, vol, s))
      }
      .toIndexedSeq
  }

  /** Read the raw CT volume. Values are in Hounsfield Units (Short range). */
  def loadVolume(file: File): DiscreteImage[_3D, Short] =
    ImageIO
      .read3DScalarImage[Short](file)
      .getOrElse(throw new RuntimeException(s"Cannot read CT volume: ${file.getName}"))

  /** Read the binary segmentation label map (values 0 or 1, stored as Short). */
  def loadSegmentation(file: File): DiscreteImage[_3D, Short] =
    ImageIO
      .read3DScalarImage[Short](file)
      .getOrElse(throw new RuntimeException(s"Cannot read segmentation: ${file.getName}"))

  /**
   * Extract a surface mesh from the binary segmentation using Marching Cubes.
   *
   * The segmentation values are 0/1; isoValue 0.5 places the surface exactly halfway.
   * The output is in the same world-coordinate frame as the CT volume (millimetres).
   */
  def extractSurface(seg: DiscreteImage[_3D, Short]): TriangleMesh[_3D] = {
    val floatImg: DiscreteImage[_3D, Float] = seg.map(_.toFloat)
    MarchingCubes.marchingCubes(floatImg, isoValue = 0.5)
  }

  /**
   * Sample Hounsfield Units at the given mesh vertices using B-spline interpolation.
   *
   * Points that fall outside the volume domain (e.g. after registration) are assigned
   * the HU value for air (−1000), which is the physically correct outside value for CT.
   */
  def sampleHU(mesh: TriangleMesh[_3D], volume: DiscreteImage[_3D, Short]): IndexedSeq[Float] = {
    // degree-3 B-spline gives continuous interpolation across voxels
    val continuous = volume.interpolate(3)
    mesh.pointSet.points.map { pt =>
      if (volume.isDefinedAt(pt)) continuous(pt).toFloat
      else -1000f
    }.toIndexedSeq
  }

  /**
   * Compute simple per-vertex HU statistics useful for a sanity check:
   * returns (minHU, maxHU, meanHU) over the provided vertex HU array.
   */
  def huStats(hu: IndexedSeq[Float]): (Float, Float, Float) = {
    val mn  = hu.min
    val mx  = hu.max
    val avg = hu.sum / hu.length
    (mn, mx, avg)
  }

  /**
   * Cortical-bone mask: returns true for vertices whose HU is above a threshold
   * commonly used to distinguish cortical bone (>300 HU) from trabecular bone / soft tissue.
   */
  def corticalMask(hu: IndexedSeq[Float], threshold: Float = 300f): IndexedSeq[Boolean] =
    hu.map(_ >= threshold)
}
