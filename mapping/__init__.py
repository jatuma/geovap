"""Bidirectional mapping between Ladybug equirectangular panoramas and the Drazkov MLS point cloud.

Conventions (verified against TerraScan RGB and JVF overlays, see 02_obarveni_pointcloudu.md SS2):
  - yaw is a mathematical azimuth, CCW from +Easting
  - roll and pitch are applied with NEGATIVE sign
  - v = 0 is the zenith
  - seam u = 0 lies at azimuth == yaw
"""
