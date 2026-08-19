name := "scapula-ssm"
version := "1.0.0"
scalaVersion := "3.3.1"

scalacOptions ++= Seq("-encoding", "UTF-8", "-feature", "-deprecation", "-language:implicitConversions")

resolvers += Resolver.jcenterRepo

libraryDependencies ++= Seq(
  "ch.unibas.cs.gravis" %% "scalismo-ui" % "0.92.0",
  "ch.unibas.cs.gravis" %% "scalismo"    % "0.92.1"
)

// The pipeline holds several meshes plus a low-rank GP basis in memory.
run / javaOptions ++= Seq("-Xmx8g")
run / fork := true
run / connectInput := true
