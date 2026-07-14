# Update: 14.07.2026 

The Mustatil 6 release might be in a few days, I am working on getting it published in the Microsoft Store again, as an Update. Right now it is in certification. But honestly I do think that the Python only 5.6 Version might be the better one. Version 5.6 will stay free and is still the recommended stable version. Version 6 will be under development and hopefully will become the better version over time.
I have chosen to publish in the Microsoft Store, because there it gets licenced for free if its packaged as a MSIX. This gives you a licenced and signed version, wich is proofen to be unharmfull and virus free.

## My Tip for you is to aquire Mustatil from the Microsoft Store as long as it is free, then you will get the update for free aswell. 
    Update Notes:  - New C++ Gui with dark look (there is a setting for bright mode)
                   - Unfortuanatly ADAF did not make it into the new version
                   - Standard DINO Model is included
                   - Multi GPU support
                   - Faster startup time
                   - It is packaged as a All in One, so no need to download runtimes for each model
                   
    <img width="1280" height="720" alt="Screenshot 2026-07-14 191226" src="https://github.com/user-attachments/assets/25778ab8-2e0e-479d-8313-53a9c1845ebd"/>

                   
My Honest view is that it does have some improvements, but overall there has not been added a lot of new features. Many features, like the GIS Review Tab do perform better in version 5.6. Overall I do still recommend the free version Mustatil 5.6, maybe Mustatil 6 will get more improvements in the future, but right now it might just be a way to donate. I am planning on adding more Lidar features in the future.

# I am currently working on a new C++ GUI, this will make it a Windows only project. I will still recomend the 5.6 version, because it has all functions and is available on all platforms. Unfortunately this will also be the end for new pip and conda versions. It will stay the 5.6 there. 
# I am planing to change to a paid format where it will be available in the Microsoft Store exclusivly for 14,99 $. The 5.6 will stay opensource and free forever!




                    


<div align="center">

# Mustatil

### GIS AI Vision Workspace for Annotation, YOLO Training, Large-Scale Detection, Satellite-Map Analysis, and Visual AI Pipelines

<p>
  <a href="https://github.com/tarekwasfy01/Mustatil-YOLO-AI-Model-Trainer-">
    <img alt="GitHub Repository" src="https://img.shields.io/badge/GitHub-Repository-111111?style=for-the-badge&logo=github&logoColor=white">
  </a>
  <a href="https://pypi.org/project/mustatil/">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/mustatil?label=PyPI&style=for-the-badge&color=111111&logo=pypi&logoColor=white">
  </a>
  <a href="https://snapcraft.io/mustatil">
    <img alt="Snap Store" src="https://img.shields.io/badge/Snap%20Store-mustatil-111111?style=for-the-badge&logo=snapcraft&logoColor=white">
  </a>
  <a href="https://doi.org/10.5281/zenodo.20481110">
    <img alt="Zenodo DOI" src="https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.20481110-111111?style=for-the-badge&logo=zenodo&logoColor=white">
  </a>
  <a href="https://github.com/tarekwasfy01/Mustatil-YOLO-AI-Model-Trainer-/blob/main/LICENSE">
    <img alt="License LGPL v3" src="https://img.shields.io/badge/License-LGPL%20v3-111111?style=for-the-badge">
  </a>
</p>

<p>
  <a href="https://snapcraft.io/mustatil">
    <img alt="Get it from the Snap Store" src="https://snapcraft.io/en/dark/install.svg" width="220">
  </a>
</p>
<a href="https://get.microsoft.com/installer/download/9PN11ZK9QL42?referrer=appbadge" target="_self" > <img src="https://get.microsoft.com/images/en-us%20dark.svg" width="200"/> </a>

<img alt="Mustatil workspace screenshot" src="https://github.com/user-attachments/assets/e869cd6a-3c36-42e1-b587-65dfc3edcbbb" width="900">

</div>

---

## Overview

**Mustatil** is a GIS-level AI vision workspace for **annotation**, **YOLO training**, **large-scale object detection**, **satellite-map analysis**, **GeoPackage/GIS export**, and **graphical AI pipeline building**.

It is designed for images, GeoTIFFs, BigTIFFs, satellite map areas, archaeological datasets, and remote-sensing workflows that are too large or too geospatially specific for many conventional computer-vision tools.

The name **Mustatil** means **rectangle** — a reference to both archaeological mustatils and the rectangular detection boxes used in AI object detection.

---

## Main Links

| Resource | Address |
|---|---|
| Official website | https://mustatil.de |
| Additional website | https://mustatil-ai.com |
| GitHub repository | https://github.com/tarekwasfy01/Mustatil-YOLO-AI-Model-Trainer- |
| GitHub releases | https://github.com/tarekwasfy01/Mustatil-YOLO-AI-Model-Trainer-/releases |
| PyPI package | https://pypi.org/project/mustatil/ |
| Snap Store | https://snapcraft.io/mustatil |
| itch.io page | https://tarekwasfy01.itch.io/mustatil-qt-workspace |
| Zenodo DOI | https://doi.org/10.5281/zenodo.20481110 |
| License | https://github.com/tarekwasfy01/Mustatil-YOLO-AI-Model-Trainer-/blob/main/LICENSE |
| Python | https://www.python.org/downloads/ |

---

## Downloads

| Platform | Download |
|---|---|
| Windows installer | https://github.com/tarekwasfy01/Mustatil-YOLO-AI-Model-Trainer-/releases/download/Mustatil-5.6/Mustatil_5.6_Setup.exe |
| macOS installer | https://github.com/tarekwasfy01/Mustatil-YOLO-AI-Model-Trainer-/releases/download/Mustatil-5.6/Mustatil_5.6_macOS.pkg |
| Linux DEB installer | https://github.com/tarekwasfy01/Mustatil-YOLO-AI-Model-Trainer-/releases/download/Mustatil-5.6/Mustatil_5.6_Linux.deb |
| Snap Store | https://snapcraft.io/mustatil |
| Alternative download page | https://tarekwasfy01.itch.io/mustatil-qt-workspace |
| Release archive | https://github.com/tarekwasfy01/Mustatil-YOLO-AI-Model-Trainer-/releases |

---

## Quick Install

### Windows / Python

```powershell
py -m pip install mustatil
mustatil
```

### Linux / macOS / Python

```bash
python3 -m pip install mustatil
mustatil
```

### Conda

```bash
conda install mustatil::mustatil
```

### Snap

```bash
sudo snap install mustatil
```

The Windows installer is intended for users who do not normally work with Python. It downloads Python and the required dependencies, then starts the desktop GUI.

---

## Core Features

- GIS-level object detection workspace
- Large-image and satellite-map detection
- Annotation workflow with classes and training data
- YOLO training and detection workflows
- Support for RF-DETR, R-CNN, Mask R-CNN, U-Net, SAM2, ADAF, Grounding DINO, LAE-DINO, OWL-ViT / OWLv2, and related AI vision workflows
- GeoPackage and GIS export workflows
- Map-based visual review and detection correction
- Graphical AI pipeline builder
- Desktop installer workflow for non-Python users

---

## AI Model and Workflow Support

### YOLO / Ultralytics

Train and run YOLO models for object detection, large images, satellite imagery, tiled inference, annotation review, and GIS export workflows. Suitable for YOLOv8, YOLOv11, and other Ultralytics-compatible models.

### RF-DETR

DETR-style object detection workflow for modern transformer-based detection. Useful as an alternative to YOLO for project-based datasets and advanced object detection experiments.

### Faster R-CNN / R-CNN

Classical region-based object detection support for workflows where proposal-based detection is useful, especially as a comparison or alternative to YOLO-style detection.

### Mask R-CNN

Instance-segmentation workflow for object masks, not only rectangular bounding boxes. Useful where object outlines or separated instances are more important than simple boxes.

### U-Net

Semantic-segmentation workflow for pixel-level masks, raster-style classification, and image-to-mask tasks. Useful for remote sensing and map/image segmentation workflows.

### SAM2

Segment Anything 2 workflow for interactive and AI-assisted segmentation. Useful for mask creation, object separation, annotation assistance, and segmentation-based review.

### ADAF

Additional AI detection and filtering workflow inside the Mustatil ecosystem for advanced detection experiments, filtering, and model-assisted review.

### Google OWL-ViT / OWLv2

Experimental open-vocabulary object detection from text prompts. Useful for searching objects without training a fixed class-specific model first.

### Grounding DINO

Text-guided detection for flexible object search and prompt-based localization. Extends Mustatil beyond fixed-label YOLO workflows.

### LAE-DINO

DINO-based detection and training workflow with project-based dataset creation, model testing, and training support.

### Detect Anything / LocateAnything

Promptable detection workflow for flexible object localization. Designed as an experimental AI vision extension beside the regular YOLO/GIS workflow.

---

## GIS and Scientific Use

Mustatil is useful for:

- Archaeological remote sensing
- Satellite-image object detection
- Large raster and GeoTIFF analysis
- Training custom object-detection datasets
- Reviewing detections on maps
- Exporting detections for GIS software
- Producing GeoPackage / GIS-compatible outputs

A GeoPackage converter for QGIS is also available for situations where files need additional compatibility handling. Normally, the EPSG code for a layer can be changed directly in QGIS.

---

## Screenshots

<p align="center">
  <img alt="Mustatil screenshot 1" src="https://github.com/user-attachments/assets/e869cd6a-3c36-42e1-b587-65dfc3edcbbb" width="48%">
  <img alt="Mustatil screenshot 2" src="https://github.com/user-attachments/assets/fe9d1958-78fb-498c-96bd-636aae9ed342" width="48%">
</p>

<p align="center">
  <img alt="Mustatil screenshot 3" src="https://github.com/user-attachments/assets/7e9099d2-20c1-43e7-b5d9-fb0f05e61ff9" width="48%">
  <img alt="Mustatil screenshot 4" src="https://github.com/user-attachments/assets/03291052-c523-4aee-8007-d5c964e9115b" width="48%">
</p>

---

## Citation

If you use Mustatil in research, datasets, training workflows, or publications, please cite the Zenodo archive:

```text
Mustatil: GIS AI Vision Workspace
DOI: https://doi.org/10.5281/zenodo.20481110
```

---

## License

Mustatil is licensed under **LGPL v3**.

License file:

```text
https://github.com/tarekwasfy01/Mustatil-YOLO-AI-Model-Trainer-/blob/main/LICENSE
```

---

## Author

**Mustatil by Tarek Wasfy and AI.**

