package scapula

import scalismo.common.PointId
import scalismo.geometry._3D
import scalismo.mesh.{TriangleCell, TriangleList, TriangleMesh, TriangleMesh3D}
import scalismo.io.MeshIO
import scalismo.numerics.UniformMeshSampler3D
import scalismo.utils.Random

import java.io.File

/**
 * Mesh decimation to produce ~8,000-vertex working meshes.
 *
 * Strategy: stride-based vertex subsampling. Every `stride`-th vertex is kept;
 * a triangle is retained only if all three of its vertices are in the kept set.
 * This guarantees a topologically valid sub-mesh derived from the original without
 * any external Delaunay step, and the output vertex count is reliably close to
 * `targetN`.
 *
 * Working meshes are generated ONCE from the originals and reused across all SSM
 * iterations (SSM1→SSM4). We never re-decimate an already-decimated mesh.
 */
object Decimation {

  def ensureDir(d: File): File = { d.mkdirs(); d }

  /**
   * Decimate `mesh` so that it contains approximately `targetN` vertices.
   *
   * If the mesh already has ≤ `targetN * 1.1` vertices it is returned unchanged.
   */
  def decimateByStride(mesh: TriangleMesh[_3D], targetN: Int = 8000): TriangleMesh[_3D] = {
    val nPts = mesh.pointSet.numberOfPoints
    if (nPts <= (targetN * 1.1).toInt) return mesh

    val stride = math.max(1, nPts / targetN)
    val keptIds: Set[PointId] = (0 until nPts by stride).map(PointId).toSet

    // Keep triangles whose three vertices are all retained
    val keptTriangles = mesh.triangulation.triangles.filter { t =>
      keptIds.contains(t.ptId1) && keptIds.contains(t.ptId2) && keptIds.contains(t.ptId3)
    }

    // Re-index: consecutive IDs in sorted order
    val keptIdSeq = keptIds.toIndexedSeq.sortBy(_.id)
    val remap      = keptIdSeq.zipWithIndex.map { case (old, i) => old -> PointId(i) }.toMap

    val newPts = keptIdSeq.map(id => mesh.pointSet.point(id))
    val newTris = keptTriangles.map { t =>
      TriangleCell(remap(t.ptId1), remap(t.ptId2), remap(t.ptId3))
    }

    TriangleMesh3D(newPts, TriangleList(newTris))
  }

  /**
   * Keep only the largest connected component of `mesh`.
   *
   * Stride-based decimation can produce isolated triangles disconnected from the
   * main bone body.  This removes them by running union-find over triangle edges
   * and discarding all components smaller than the largest one.
   */
  def keepLargestComponent(mesh: TriangleMesh[_3D]): TriangleMesh[_3D] = {
    val n = mesh.pointSet.numberOfPoints
    val parent = Array.tabulate(n)(identity)

    def find(x: Int): Int = {
      if (parent(x) != x) parent(x) = find(parent(x))
      parent(x)
    }
    def union(a: Int, b: Int): Unit = parent(find(a)) = find(b)

    mesh.triangulation.triangles.foreach { t =>
      union(t.ptId1.id, t.ptId2.id)
      union(t.ptId2.id, t.ptId3.id)
    }

    val compSize = Array.fill(n)(0)
    (0 until n).foreach(i => compSize(find(i)) += 1)
    val largestRoot = (0 until n).maxBy(i => compSize(find(i)))
    val largestComp = find(largestRoot)

    if (compSize(largestComp) == n) return mesh

    val remap = Array.fill(n)(-1)
    var idx = 0
    (0 until n).foreach { i =>
      if (find(i) == largestComp) { remap(i) = idx; idx += 1 }
    }

    val newPts  = (0 until n).filter(i => remap(i) >= 0).map(i => mesh.pointSet.point(PointId(i))).toIndexedSeq
    val newTris = mesh.triangulation.triangles
      .filter(t => remap(t.ptId1.id) >= 0 && remap(t.ptId2.id) >= 0 && remap(t.ptId3.id) >= 0)
      .map(t => TriangleCell(PointId(remap(t.ptId1.id)), PointId(remap(t.ptId2.id)), PointId(remap(t.ptId3.id))))
    TriangleMesh3D(newPts, TriangleList(newTris))
  }

  /**
   * Generate 8k working meshes for all specimens into `outDir`.
   *
   * Idempotent: if a file already exists it is loaded rather than regenerated.
   * Returns pairs of (specimen, decimated-mesh-file).
   */
  def generateAll(
    specimens: IndexedSeq[ScapulaData.Specimen],
    outDir: File,
    targetN: Int = 8000
  ): IndexedSeq[(ScapulaData.Specimen, File)] = {
    ensureDir(outDir)
    specimens.map { spec =>
      val outFile = new File(outDir, spec.modelId + ".stl")
      if (!outFile.exists()) {
        println(s"  Decimating ${spec.modelId}")
        val original  = ScapulaData.loadMesh(spec.file)
        val decimated = keepLargestComponent(decimateByStride(original, targetN))
        MeshIO.writeMesh(decimated, outFile)
          .getOrElse(throw new RuntimeException(s"Failed to write ${outFile.getPath}"))
        println(f"    ${spec.modelId}: ${original.pointSet.numberOfPoints} → ${decimated.pointSet.numberOfPoints} vertices")
      }
      (spec, outFile)
    }
  }
}
