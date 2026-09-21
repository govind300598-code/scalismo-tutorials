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

// 20g drove the whole system into swap when other apps (VS Code, Metals, Bloop)
// were competing for RAM. 12g then wasn't enough for Stage 1, which holds every
// specimen's full mesh in memory at once -- including the largest single specimen,
// a 577,928-vertex mesh, that specifically needs the extra headroom. 16g is the
// middle ground, assuming no other heavy apps are running alongside it.
// ExitOnOutOfMemoryError means a genuine heap exhaustion kills the JVM immediately
// with a clear error instead of the process (and the OS) thrashing indefinitely.
run / javaOptions ++= Seq("-Xmx16g", "-XX:+ExitOnOutOfMemoryError")
run / fork := true
run / connectInput := true
