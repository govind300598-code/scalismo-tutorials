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

  def loadVolume(file: File): DiscreteImage[_3D, Float] = {
    if (!file.exists())
      throw new RuntimeException(s"CT volume file not found: ${file.getAbsolutePath}")
    if (!file.canRead)
      throw new RuntimeException(s"CT volume file not readable (permissions?): ${file.getAbsolutePath}")
    ImageIO.read3DScalarImage[Float](file) match {
      case scala.util.Success(img) => img
      case scala.util.Failure(ex) =>
        // Print full stack trace to help diagnose (type mismatch, bad NRRD header, etc.)
        ex.printStackTrace()
        throw new RuntimeException(
          s"Cannot read CT volume as Float: ${file.getAbsolutePath}\n" +
          s"Root cause — ${ex.getClass.getSimpleName}: ${ex.getMessage}\n" +
          s"(Stack trace printed above)", ex)
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
