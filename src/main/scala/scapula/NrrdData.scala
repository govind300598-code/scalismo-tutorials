package scapula

import scalismo.common.PointId
import scalismo.geometry.*
import scalismo.image.{DiscreteImage, DiscreteImageDomain}
import scalismo.io.{ImageIO, MeshIO}
import scalismo.mesh.*

import java.io.File

object NrrdData {

  final case class CtSpecimen(id: String, volumeFile: File, stlFile: File)

  /**
   * Thin wrapper around a loaded CT volume.
   * Holds only the voxel domain and a per-voxel Float lookup so no
   * type-conversion copy is ever allocated (avoids OOM on large Short/Int images).
   */
  final case class HuVolume(
    domain: DiscreteImageDomain[_3D],
    at: PointId => Float
  )

  def discoverSpecimens(nrrdDir: File, stlDir: File): IndexedSeq[CtSpecimen] = {
    val volumeFiles = Option(nrrdDir.listFiles()).getOrElse(Array.empty[File])
      .filter(_.getName.endsWith("_volume.nrrd"))
      .sortBy(_.getName)

    val stlFiles = Option(stlDir.listFiles()).getOrElse(Array.empty[File])
      .filter(_.getName.endsWith(".stl"))

    volumeFiles.flatMap { vol =>
      val volId = vol.getName.stripSuffix("_volume.nrrd")
      val matching = stlFiles.filter(_.getName.startsWith(volId))
      matching.map { stl =>
        CtSpecimen(stl.getName.stripSuffix(".stl"), vol, stl)
      }
    }.toIndexedSeq
  }

  /**
   * Load a CT volume.  Tries Short, Int, then Float in sequence so the
   * pipeline handles any NIfTI scalar type without a full-array conversion.
   *
   * Requires a .nii sibling of the .nrrd file (produced by convert_nrrd_to_nii.py).
   */
  def loadVolume(file: File): HuVolume = {
    val candidate: File =
      if (file.getName.endsWith(".nrrd")) {
        val base  = file.getName.stripSuffix(".nrrd")
        val nii   = new File(file.getParent, base + ".nii")
        val niiGz = new File(file.getParent, base + ".nii.gz")
        if (nii.exists()) nii else if (niiGz.exists()) niiGz else file
      } else file

    if (!candidate.exists())
      throw new RuntimeException(
        s"CT volume not found: ${candidate.getAbsolutePath}\n" +
        (if (file.getName.endsWith(".nrrd"))
           s"Convert NRRD to NIfTI first:\n  python3 convert_nrrd_to_nii.py \"${file.getParent}\""
         else ""))

    // Try Short (Int16) first — all .nii files must be converted with convert_nrrd_to_nii.py
    // which casts to Int16 and zeroes scl_slope so VTK does not upcast to Float.
    // Fall back to Float for edge cases (e.g. already-float source).
    // tryInt is deliberately omitted: a 32-bit array for a large CT volume can easily
    // exhaust the 12 GB heap once many meshes and HU vectors have accumulated.
    def tryShort = ImageIO.read3DScalarImage[Short](candidate).map(img => HuVolume(img.domain, id => img(id).toFloat))
    def tryFloat = ImageIO.read3DScalarImage[Float](candidate).map(img => HuVolume(img.domain, id => img(id)))

    (tryShort orElse tryFloat) match {
      case scala.util.Success(v) => v
      case scala.util.Failure(ex) =>
        throw new RuntimeException(
          s"Cannot read CT volume as Short or Float: ${candidate.getAbsolutePath}\n" +
          s"The file may still be stored as Int32. Delete it and reconvert:\n" +
          s"  rm \"${candidate.getAbsolutePath}\"\n" +
          s"  python3 convert_nrrd_to_nii.py \"${file.getParent}\"\n" +
          s"Cause: ${ex.getClass.getSimpleName}: ${ex.getMessage}", ex)
    }
  }

  def extractSurface(spec: CtSpecimen): TriangleMesh[_3D] =
    MeshIO.readMesh(spec.stlFile)
      .getOrElse(throw new RuntimeException(s"Cannot read STL: ${spec.stlFile.getName}"))

  def sampleHU(mesh: TriangleMesh[_3D], volume: HuVolume): IndexedSeq[Float] = {
    val o  = volume.domain.origin
    val sp = volume.domain.spacing
    val sz = volume.domain.size
    val nx = sz(0); val ny = sz(1); val nz = sz(2)
    mesh.pointSet.points.map { pt =>
      val ci = math.round((pt.x - o.x) / sp.x).toInt
      val cj = math.round((pt.y - o.y) / sp.y).toInt
      val ck = math.round((pt.z - o.z) / sp.z).toInt
      if (ci >= 0 && ci < nx && cj >= 0 && cj < ny && ck >= 0 && ck < nz)
        volume.at(PointId(ci + nx * (cj + ny * ck)))
      else -1000f
    }.toIndexedSeq
  }

  def huStats(hu: IndexedSeq[Float]): (Float, Float, Float) =
    (hu.min, hu.max, hu.sum / hu.length)

  def corticalMask(hu: IndexedSeq[Float], threshold: Float = 300f): IndexedSeq[Boolean] =
    hu.map(_ >= threshold)
}
