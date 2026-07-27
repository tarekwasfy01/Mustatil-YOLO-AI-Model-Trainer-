---
title: 'Mustatil: A local-first geospatial AI workspace for large-raster research surveys'
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

Mustatil is an open-source desktop workspace for finding, reviewing, and mapping objects in satellite, aerial, and other large raster images. It connects image annotation, detector training, tiled analysis, human review, and export to geographic information system (GIS) formats in one local workflow. The software was originally developed to survey *mustatils*—monumental rectangular stone structures of Neolithic Arabia—but researchers can define their own classes and apply the same workflow to other visually distinguishable targets. GeoTIFF and BigTIFF data can be processed without uploading source imagery or prospective site coordinates to a third-party service.

The name has an intentional double meaning: *mustatil* (مستطيل) means “rectangle” in Arabic and is the archaeological term used for the elongated monuments that motivated the project; object-detection systems also describe candidate features with rectangular bounding boxes. Version 5.6 is archived on Zenodo [@wasfy_mustatil_software_2026], distributed through package registries, and developed on GitHub.

# Statement of need

Systematic visual survey does not scale easily. Archaeological researchers may need to inspect thousands of square kilometres while separating subtle stone features from geology, tracks, modern construction, vegetation, shadows, and image artefacts. A conventional workflow can require a GIS application, an annotation program, a machine-learning library, scripts for splitting large rasters, and additional code for restoring map coordinates. Moving data between these components increases setup effort and creates opportunities for losing class definitions, model settings, coordinate-reference information, or review decisions.

Mustatil provides a continuous workflow for researchers who need computer vision but do not want to assemble and maintain that toolchain themselves. Users can annotate examples, train or load a detector, scan large georeferenced rasters as overlapping tiles, review candidates against the original image, and export accepted geometry to GeoPackage, GeoJSON, or tabular formats. The target audience includes archaeologists, remote-sensing researchers, GIS users, and investigators in adjacent fields who have domain knowledge and imagery but limited interest in writing custom inference and georeferencing code.

The motivating archaeological problem concerns mustatils. Excavated examples in north-western Arabia have been interpreted as late-sixth-millennium BCE ritual installations and form an early monumental stone-building tradition [@thomas2021mustatils]. Their distribution and landscape positioning remain active research topics [@hatton2024landscape]. Satellite detections are nevertheless hypotheses, not confirmations: imagery alone cannot establish chronology, cultural attribution, function, or archaeological significance. Mustatil therefore treats model output as a prioritisation layer for visual inspection, expert assessment, higher-resolution examination, and, where possible, field verification.

# State of the field

Several established projects cover parts of this workflow. QGIS is a mature general-purpose GIS for visualisation, editing, and spatial analysis [@graser2025qgis], but it does not by itself provide an integrated annotation-to-training-to-large-raster detection workflow. CVAT provides extensive image and video annotation capabilities [@cvat2026], while Ultralytics YOLO supplies command-line and Python interfaces for model training and prediction [@ultralytics2026]. Orfeo ToolBox offers high-performance remote-sensing image processing through applications, Python, QGIS, and C++ interfaces [@grizonnet2017otb]. These projects are more mature and broader within their respective domains than Mustatil.

The build-versus-contribute justification is therefore not that these alternatives are inadequate individually. The gap is at their intersection: a locally operated desktop workflow that keeps annotation, training, tiled geospatial detection, candidate review, and GIS export connected for a researcher working with very large rasters. Implementing this as an application rather than as extensions to several independent projects makes it possible to preserve one project state and one user-facing sequence across model and GIS operations. Mustatil remains interoperable with the wider ecosystem: model files and training data can be reused outside the application, and exported geospatial results can be analysed in QGIS or other GIS software.

# Software design

Mustatil uses a local-first desktop architecture. This trades the convenience and elastic computing capacity of hosted services for control over imagery, coordinates, models, and computational costs. It is especially relevant for unpublished surveys or culturally sensitive locations, although local execution also places responsibility for hardware, drivers, and dependency compatibility on the user.

Large rasters are processed incrementally as tiles rather than loaded into memory as one array. Tile overlap reduces missed objects at tile boundaries, but increases computation and can create duplicate detections. Mustatil exposes overlap and confidence controls and maps retained detections from tile coordinates back into the coordinate reference system of the input raster. The raster transformation is therefore part of the detection workflow rather than a separate post-processing script.

The software separates candidate generation from interpretation. Automated models rank possible locations; a human reviewer can inspect, filter, edit, or reject detections before export. This design reduces the risk of presenting every prediction as an archaeological site and retains confidence and class information for later audit. Standard GIS outputs are used instead of a proprietary result database so that researchers can perform spatial statistics, cartography, field planning, or quality control with established tools.

Model support is modular. YOLO-based training and detection remain a central path, while later releases expose helpers for additional detection and segmentation families. Supporting several model families increases experimentation options but also raises maintenance and dependency complexity. The stable conceptual core is consequently model-independent: annotations define target classes, models generate candidate geometry, georeferencing restores spatial coordinates, and review determines which candidates enter the research dataset.

# Research impact statement

Mustatil has been used by the author to produce several open geospatial research objects. An initial dataset records more than one hundred candidate stone structures near Al Badi and Al Khashabi in central Saudi Arabia and distributes mapped locations in multiple GIS formats [@wasfy_albadi_2026]. A later survey near Al Faw used Mustatil for image inspection, YOLO-based detection, review, annotation, and GIS export; its archive includes georeferenced data and trained YOLO and Faster R-CNN models, while explicitly treating cairns and pendant-like features as candidates requiring independent verification [@wasfy_alfaw_2026].

Mustatil was also used to analyse an approximately 200 × 200 km georeferenced satellite dataset in central Saudi Arabia. The resulting archive documents a candidate necropolis and includes geospatial detections and model material, together with warnings about false positives, false negatives, classification errors, and spatial uncertainty [@wasfy_necropolis_2026]. A separate application in Mongolia mapped possible khirigsuur burial and ceremonial features, demonstrating that the workflow can be transferred beyond Arabia and beyond the monument type that gave the software its name [@wasfy_mongolia_2026].

These examples are author-led uses rather than evidence of broad independent adoption. Their value for scope assessment is that they show repeated application to distinct landscapes, monument classes, model families, and open outputs. The archives preserve candidate geometry, models, and descriptive metadata as traceable research objects. They also demonstrate the intended epistemic boundary: Mustatil accelerates search and documentation, but archaeological identification remains a domain-expert decision.

The same architecture can support other research questions involving spatially referenced visual targets, including environmental features, buildings, vegetation patterns, or geological forms, provided that suitable imagery, labels, and validation procedures exist. The scholarly contribution is the integration of large-raster computer vision with human-in-the-loop geospatial review, not a claim that automated prediction replaces field or disciplinary expertise.

# AI usage disclosure

OpenAI ChatGPT was used during development sessions for code drafting, debugging assistance, documentation, and interface-related problem solving. Model routing varied during earlier sessions; OpenAI GPT-5.6 Thinking was used during the final preparation and revision of this manuscript in July 2026. Generative AI was also used to organise source material, propose manuscript wording, and prepare repository-support files. The author defined the research problem, software concept, architecture, workflows, survey applications, and final decisions. AI-generated code and prose were reviewed, tested where applicable, corrected, and accepted by the author. Bibliographic claims and software metadata were checked against the cited primary or official sources. The author takes responsibility for the software and manuscript.

# Conflict of interest

The author declares no competing interests.

# Acknowledgements

No external funding was received for the development of Mustatil or the preparation of this paper. The author thanks the maintainers of the open-source geospatial and machine-learning projects on which Mustatil depends.

# References
