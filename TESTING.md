# Testing and reproducibility plan

## Automated checks

The proposed CI matrix covers Python 3.10–3.12 on Windows and Linux. The workflow becomes usable after the unpacked source and valid Python packaging metadata are present at the repository root.

The current scaffold is not a claim of comprehensive testing. JOSS reviewers require an objective way to determine whether the research-critical functionality works. The priority is to test coordinate transformations, tiling, de-duplication, and geospatial export independently of the graphical interface.

## Manual reference workflow

Create a redistributable sample project containing:

- a small synthetic GeoTIFF with a known CRS and affine transform;
- several visible geometric targets, including one crossing a tile boundary;
- a tiny test model or deterministic mock detector;
- expected GeoJSON and GeoPackage outputs;
- a short guide with screenshots and exact expected feature counts and coordinates.

A reviewer should be able to install Mustatil, run this workflow, and compare the output against the expected files in less than 15 minutes.

## Platform evidence

For each release, record tested operating systems, Python versions, installation methods, and whether CPU and CUDA execution were exercised. Keep test claims limited to environments that were actually used.
