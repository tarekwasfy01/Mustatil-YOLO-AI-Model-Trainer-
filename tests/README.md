# Test-suite starting point

The included metadata test is intentionally minimal and is **not sufficient for JOSS review**. It verifies only that the repository can be installed as the `mustatil` Python distribution.

Before submission, add objective tests for the research-critical transformations:

1. **Tile coverage:** every pixel in a synthetic raster is covered with the documented overlap and no unintended gaps.
2. **Coordinate restoration:** a known pixel bounding box in a synthetic GeoTIFF becomes the expected map-coordinate geometry.
3. **Boundary de-duplication:** overlapping detections across adjacent tiles are merged according to the documented rule.
4. **Export integrity:** GeoPackage and GeoJSON outputs retain CRS, class, confidence, and geometry.
5. **Project round-trip:** saving and reopening a project preserves classes and essential parameters.
6. **Headless smoke test:** the application package imports and starts its core services without a display server.

Use small synthetic fixtures that can be redistributed. Do not place restricted satellite imagery or sensitive coordinates in the test suite.
