package scapula

import scalismo.common.PointId
import scalismo.geometry.*
import scalismo.image.DiscreteImage
import scalismo.io.{ImageIO, MeshIO}
import scalismo.mesh.*

import java.io.File

/**
 * Discovers, loads and preprocesses paired CT volume + STL surface files.
 *
 * NRRD directory layout (set SCAPULA_NRRD_DIR):
 *   <id>_volume.nrrd          — raw CT intensities in Hounsfield Units (Short)
 *   <id>_scapula_0.seg.nrrd   — binary segmentation (not needed if STL present)
 *
 * STL directory layout (set SCAPULA_STL_DIR):
 *   <id>_scapula_0_Segment_2.stl   — one scapula (e.g. left)
 *   <id>_scapula_0_Segment_3.stl   — other scapula (e.g. right)
 *
 * Each STL file becomes one CtSpecimen entry.  Both segments of the same
 * subject share the same CT volume, so HU is sampled from that volume for
 * whichever surface is being processed.
 */
object NrrdData {

  /**
   * @param id         unique label for this specimen (derived from STL filename)
   * @param volumeFile paired CT volume NRRD
   * @param stlFile    scapula surface STL
   */
  final case class CtSpecimen(id: String, volumeFile: File, stlFile: File)

  /**
   * Discover all paired (volume.nrrd, STL) specimens.
   *
   * For every `*_volume.nrrd` in nrrdDir, all STL files in stlDir whose
   * base name starts with the volume's id are returned as separate entries.
   * Names with spaces or "marche pas" / "scaling" suffixes are handled
   * automatically because the match is prefix-based.
   */
  def discoverSpecimens(nrrdDir: File, stlDir: File): IndexedSeq[CtSpecimen] = {
    val volumeFiles = Option(nrrdDir.listFiles()).getOrElse(Array.empty[File])
      .filter(_.getName.endsWith("_volume.nrrd"))
      .sortBy(_.getName)

    val stlFiles = Option(stlDir.listFiles()).getOrElse(Array.empty[File])
      .filter(_.getName.endsWith(".stl"))

    volumeFiles.flatMap { vol =>
      val volId = vol.getName.stripSuffix("_volume.nrrd")
      // find STL files whose name starts with this volume's id
      val matching = stlFiles.filter(_.getName.startsWith(volId))
      matching.map { stl =>
        val specId = stl.getName.stripSuffix(".stl")
        CtSpecimen(specId, vol, stl)
      }
    }.toIndexedSeq
  }

  /**
   * Load a CT volume as a Float image.
   *
   * Scalismo 0.92.x ImageIO supports .nii but NOT .nii.gz or .nrrd directly.
   * When given a .nrrd path we look for a sibling .nii first, then .nii.gz as fallback
   * (both produced by convert_nrrd_to_nii.py).  Run that script once before the pipeline:
   *
   *   python3 convert_nrrd_to_nii.py "/path/to/all ct from hoel 3d"
   */
  def loadVolume(file: File): DiscreteImage[_3D, Float] = {
    // For .nrrd: prefer .nii sibling (Scalismo 0.92.x reads this), then .nii.gz, then original
    val candidate: File =
      if (file.getName.endsWith(".nrrd")) {
        val base   = file.getName.stripSuffix(".nrrd")
        val nii    = new File(file.getParent, base + ".nii")
        val niiGz  = new File(file.getParent, base + ".nii.gz")
        if (nii.exists()) nii else if (niiGz.exists()) niiGz else file
      } else file

    if (!candidate.exists())
      throw new RuntimeException(
        s"CT volume not found: ${candidate.getAbsolutePath}\n" +
        (if (file.getName.endsWith(".nrrd"))
           s"Convert NRRD to NIfTI first:\n  python3 convert_nrrd_to_nii.py \"${file.getParent}\""
         else ""))

    ImageIO.read3DScalarImage[Float](candidate) match {
      case scala.util.Success(img) => img
      case scala.util.Failure(ex) =>
        throw new RuntimeException(
          s"Cannot read CT volume: ${candidate.getAbsolutePath}\n" +
          s"Cause: ${ex.getClass.getSimpleName}: ${ex.getMessage}", ex)
    }
  }

  def extractSurface(spec: CtSpecimen): TriangleMesh[_3D] =
    MeshIO.readMesh(spec.stlFile)
      .getOrElse(throw new RuntimeException(s"Cannot read STL: ${spec.stlFile.getName}"))

  /**
   * Sample Hounsfield Units at mesh vertices using nearest-neighbour voxel lookup.
   * Points outside the CT volume domain fall back to −1000 HU (air).
   */
  def sampleHU(mesh: TriangleMesh[_3D], volume: DiscreteImage[_3D, Float]): IndexedSeq[Float] = {
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
        volume(PointId(ci + nx * (cj + ny * ck)))
      else -1000f
    }.toIndexedSeq
  }

  def huStats(hu: IndexedSeq[Float]): (Float, Float, Float) =
    (hu.min, hu.max, hu.sum / hu.length)

  def corticalMask(hu: IndexedSeq[Float], threshold: Float = 300f): IndexedSeq[Boolean] =
    hu.map(_ >= threshold)
}
