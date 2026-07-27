---
title: 'Mustatil: An offline geospatial AI workspace for large-scale archaeological remote-sensing surveys'
tags:
  - Python
  - archaeological remote sensing
  - geospatial artificial intelligence
  - object detection
  - satellite imagery
  - GIS
  - cultural heritage
authors:
  - name: Tarek Wasfy
    affiliation: 1
    corresponding: true
affiliations:
  - name: Independent Researcher and Software Developer, Germany
    index: 1
date: 27 July 2026
bibliography: paper.bib
---

# Summary

Mustatil is an open-source desktop workspace for detecting, reviewing, and mapping objects in satellite, aerial, and other large raster images. It combines image annotation, model training, tiled inference, visual inspection, and export to geographic information system (GIS) formats in one local workflow. The software was originally developed to survey *mustatils*—monumental rectangular stone structures of Neolithic Arabia—but it is not limited to a particular monument type or region. Researchers can train models for their own classes and apply them to large GeoTIFF or BigTIFF datasets without uploading source imagery or coordinates to a cloud service.

The name has an intentional double meaning: *mustatil* (مستطيل) means “rectangle” in Arabic and is also the archaeological term used for the elongated monuments that motivated the project; object-detection systems likewise represent candidate features using rectangular bounding boxes. Mustatil is archived on Zenodo [@wasfy_mustatil_software_2026], distributed as a Python package, and developed openly on GitHub.

# Statement of need

Systematic archaeological survey in satellite imagery is difficult to scale. A researcher may need to inspect thousands of square kilometres while distinguishing subtle stone features from geology, tracks, modern structures, vegetation, shadows, and image artefacts. General-purpose GIS applications can display and edit geospatial rasters, while computer-vision libraries can train and run detection models, but the complete workflow often remains fragmented across scripts, notebooks, annotation tools, command-line programs, and GIS applications.

Mustatil addresses this gap by providing a single desktop environment in which a researcher can prepare annotations, train a detector, analyse large georeferenced rasters in overlapping tiles, review detections against the source imagery, and export accepted results to GIS-compatible datasets. Its local-first design is particularly relevant where source imagery, prospective site coordinates, or unpublished cultural-heritage data should not be transferred to third-party cloud platforms.

The original research context concerns mustatils. Excavated examples in north-western Arabia have been interpreted as late-sixth-millennium BC ritual installations and are among the earliest monumental stone-building traditions known in Arabia [@thomas2021mustatils]. Their known distribution and landscape positioning remain active research topics [@hatton2024landscape]. Satellite imagery offers a practical means of locating candidate structures over large and remote areas, but detections from imagery are hypotheses rather than archaeological confirmations. Mustatil therefore treats model output as a prioritisation layer for subsequent visual review, expert assessment, higher-resolution examination, or field verification.

# State of the field

Common workflows combine a GIS package with a separate annotation platform and a machine-learning framework. This is flexible for experienced developers, but it increases setup complexity and makes it difficult to preserve a consistent chain from training data to georeferenced survey output. Web-based platforms simplify some steps but may require imagery and coordinates to be uploaded and may introduce recurring service costs or dependence on remote infrastructure.

Mustatil differs by integrating the principal stages of a remote-sensing object-detection survey into an offline desktop application. It is designed around large geospatial images rather than isolated photographs. In addition to YOLO-based workflows, later releases include experimental support for alternative detection and segmentation model families. The software is intended to complement, rather than replace, established GIS tools and archaeological interpretation: exported GeoPackage, GeoJSON, and tabular results can be inspected and analysed in external GIS software.

# Software design

The workflow is organised around five linked operations:

1. **Data preparation and annotation.** Researchers load imagery, define object classes, draw positive examples, and prepare project data for model training.
2. **Model training.** Training controls expose commonly used parameters while retaining the generated datasets and model files for reuse.
3. **Large-raster processing.** GeoTIFF and BigTIFF inputs are analysed as overlapping tiles so that rasters larger than available system memory can be processed incrementally.
4. **Review and validation.** Candidate detections are displayed over the source imagery. Confidence and class filters help prioritise manual inspection, and detections can be edited or rejected before export.
5. **Geospatial output.** Pixel-space detections are transformed back into the coordinate reference system of the input raster and exported to GIS-compatible formats.

This architecture separates automated candidate generation from interpretation. A model can reduce the area requiring close inspection, but it cannot establish chronology, cultural attribution, monument function, or archaeological significance. The software therefore preserves confidence values and spatial geometry while leaving final classification to the researcher.

# Research impact statement

Mustatil has already supported several open remote-sensing datasets. An initial manually assembled dataset records more than one hundred candidate stone structures near Al Badi and Al Khashabi in central Saudi Arabia and distributes the mapped locations in KML, GPX, CSV, GeoJSON, OSM, and GeoTIFF formats [@wasfy_albadi_2026]. This material provided a practical motivation for developing a repeatable detection and GIS-export workflow.

A subsequent survey near Al Faw used Mustatil for image inspection, YOLO-based detection, review, annotation, and GIS export. The archived research object includes georeferenced imagery, GeoPackage detections, a YOLO model, and a Faster R-CNN model, while explicitly describing the mapped cairns, burial cairns, and pendant-like features as candidates requiring independent archaeological verification [@wasfy_alfaw_2026].

Mustatil was also used in a systematic analysis of an approximately 200 × 200 km georeferenced satellite dataset in central Saudi Arabia. The resulting archive contains geospatial detections associated with a candidate necropolis and includes a GeoPackage and a Faster R-CNN model. The record emphasises possible false positives, false negatives, classification errors, and spatial inaccuracies, and recommends expert and field validation [@wasfy_necropolis_2026].

These applications demonstrate uses beyond the software’s namesake monument type. The same workflow can be adapted to burial mounds, cairns, pendants, enclosures, buildings, vegetation patterns, geological features, or other visually distinguishable targets for which suitable training data can be created. The contribution is therefore not a claim that automated detection confirms archaeological sites, but a reproducible way to search large imagery collections, rank locations for review, and preserve candidate results as open geospatial research data.

# AI usage disclosure

Generative AI tools, including OpenAI ChatGPT, were used during the development of portions of the software for code drafting, debugging assistance, documentation, and language editing. They were also used to assist with the structure and wording of this paper. The software concept, research questions, workflow design, selection and interpretation of survey material, and final decisions were made by the author. AI-generated code and prose were reviewed, tested, corrected, and accepted by the author, who takes responsibility for the software and manuscript.

# Acknowledgements

No external funding was received for the development of Mustatil or for the preparation of this paper. The author thanks the developers and maintainers of the open-source geospatial and machine-learning libraries on which the project depends.

# References
