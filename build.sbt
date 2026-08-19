name         := "scapula-alignment"
version      := "1.0.0"
scalaVersion := "3.3.6"

resolvers += Resolver.jcenterRepo

libraryDependencies ++= Seq(
  "ch.unibas.cs.gravis" %% "scalismo-ui" % "0.92.0"
)

Compile / run / fork    := true
Compile / run / javaOptions ++= Seq("-Xmx8g")
