name := "scapula-ssm"
version := "1.0.0"
scalaVersion := "3.3.1"

scalacOptions ++= Seq("-encoding", "UTF-8", "-feature", "-deprecation", "-language:implicitConversions")

resolvers += Resolver.jcenterRepo

libraryDependencies ++= Seq(
  "ch.unibas.cs.gravis" %% "scalismo-ui" % "0.92.0",
  "ch.unibas.cs.gravis" %% "scalismo"    % "0.92.1",
  // Scalismo already pulls in Breeze; we make it explicit for SVD in DensityModel.
  "org.scalanlp" %% "breeze" % "2.1.0"
)

// 20g assumed more free RAM than the dev machine actually has once the desktop, VS
// Code, and other apps are running -- it drove the whole system into swap (8/8 GiB
// used) instead of failing cleanly. 12g fits a 30 GiB machine with real headroom.
// ExitOnOutOfMemoryError means a genuine heap exhaustion kills the JVM immediately
// with a clear error instead of the process (and the OS) thrashing indefinitely.
run / javaOptions ++= Seq("-Xmx12g", "-XX:+ExitOnOutOfMemoryError")
run / fork := true
run / connectInput := true
