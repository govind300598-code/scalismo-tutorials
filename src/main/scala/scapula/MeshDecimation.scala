package scapula

import scalismo.geometry._3D
import scalismo.mesh.TriangleMesh
import scalismo.utils.MeshConversion
import vtk.vtkDecimatePro

/**
 * Caps mesh resolution via VTK's decimation (vtkDecimatePro) so the handful of
 * oversized specimens (500K+ vertices) don't dominate memory and runtime through
 * every later stage. HU is a smooth field and the model itself caps at 50 PCA
 * components, so tens of thousands of surface vertices already far exceeds what's
 * needed to capture meaningful density variation -- the extra resolution was pure
 * memory cost, not signal.
 */
object MeshDecimation {

  /** Leaves the mesh untouched if it's already at or under maxVertices. */
  def capVertices(mesh: TriangleMesh[_3D], maxVertices: Int): TriangleMesh[_3D] = {
    val n = mesh.pointSet.numberOfPoints
    if (n <= maxVertices) mesh
    else {
      val targetReduction = 1.0 - (maxVertices.toDouble / n)
      val poly = MeshConversion.meshToVtkPolyData(mesh, None)
      val dec = new vtkDecimatePro()
      dec.SetInputData(poly)
      dec.SetTargetReduction(targetReduction)
      dec.PreserveTopologyOn()       // never tear the surface into disconnected pieces
      dec.SplittingOff()             // no new vertices at sharp features -- pure reduction
      dec.BoundaryVertexDeletionOff() // keep the outline intact
      dec.Update()
      MeshConversion
        .vtkPolyDataToTriangleMesh(dec.GetOutput())
        .getOrElse(throw new RuntimeException(s"Decimation failed for a $n-vertex mesh"))
    }
  }
}
