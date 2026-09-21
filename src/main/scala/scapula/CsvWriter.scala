package scapula

import java.io.{File, PrintWriter}

/** Minimal CSV writing, used by Stage2ReferenceRefinement to save its distance-error reports. */
object CsvWriter {
  def write(file: File, header: Seq[String], rows: Seq[Seq[Any]]): Unit = {
    val pw = new PrintWriter(file)
    try {
      pw.println(header.mkString(","))
      rows.foreach(row => pw.println(row.mkString(",")))
    } finally pw.close()
  }
}
