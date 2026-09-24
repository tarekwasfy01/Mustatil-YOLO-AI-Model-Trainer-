#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mustatil Remote Sensing tab plugin - local raster preview with in-preview settings overlay.

Install:
    Copy this file into the folder: mustatil_plugins/

What it does:
    Adds an English "Remote Sensing" tab directly after "Satellite Detection".
    This version contains NO web preview and NO QtWebEngine dependency.

Features:
    - Local raster preview for GeoTIFF/TIFF/IMG and common image files
    - RGB/false-colour band selection
    - None / min-max / percentile stretch
    - Spectral index preview and export: NDVI, NDWI, MNDWI, NDBI, SAVI, EVI, BSI
    - Desert-safe index rendering: auto contrast boost, robust grayscale or fixed -1..1 grayscale
    - Generic band calculator with B1, B2, B3 ... variables
    - Save preview as PNG
    - Export every mode as georeferenced GeoTIFF with chunked direct TIFF streaming when rasterio is available
    - Chunked YOLO detection on huge rasters with CPU/CUDA device selection and GeoPackage export
    - Current RGB/index/formula/stretch settings are displayed inside the preview; reset buttons are available on every page
    - Archaeology tools for Mustatil/cairn search: Hillshade, Local Relief Model, PCA, Edge Enhancement, Rectangular Feature Enhancement
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_PLUGIN_INSTALLED = False
_ORIGINAL_ADDTAB = None


def mustatil_plugin_init():
    install_remote_sensing_tab_hook()


def install_remote_sensing_tab_hook():
    """Patch QTabWidget.addTab so the tab can be inserted during UI construction."""
    global _PLUGIN_INSTALLED, _ORIGINAL_ADDTAB
    if _PLUGIN_INSTALLED:
        return
    try:
        from PySide6.QtWidgets import QTabWidget
    except Exception as exc:
        print("[Remote Sensing Plugin] PySide6 not available:", exc)
        return

    _ORIGINAL_ADDTAB = QTabWidget.addTab

    def _patched_add_tab(self, widget, label_or_icon, label_text=None):
        if label_text is None:
            idx = _ORIGINAL_ADDTAB(self, widget, label_or_icon)
            tab_label = str(label_or_icon)
        else:
            idx = _ORIGINAL_ADDTAB(self, widget, label_or_icon, label_text)
            tab_label = str(label_text)
        try:
            already = any(self.tabText(i) == "Remote Sensing" for i in range(self.count()))
            if (not already) and tab_label == "Satellite Detection":
                page = RemoteSensingTab(_find_workspace(self))
                self.insertTab(idx + 1, page, "Remote Sensing")
                print("[Remote Sensing Plugin] Added local raster Remote Sensing tab after Satellite Detection.")
        except Exception as exc:
            print("[Remote Sensing Plugin] Could not insert tab:", exc)
        return idx

    QTabWidget.addTab = _patched_add_tab
    _PLUGIN_INSTALLED = True
    print("[Remote Sensing Plugin] QTabWidget hook installed.")


def _find_workspace(widget):
    p = widget
    while p is not None:
        try:
            if p.__class__.__name__ == "MustatilQtWorkspace":
                return p
            p = p.parent()
        except Exception:
            break
    return None


def _safe_log(workspace, text: str):
    try:
        if workspace is not None and hasattr(workspace, "log"):
            workspace.log(str(text))
            return
    except Exception:
        pass
    print("[Remote Sensing Plugin]", text)


def _have(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


class RemoteSensingTab:  # replaced after Qt imports
    pass


def _build_remote_sensing_class():
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap, QImage, QPainter, QBrush, QColor, QPen
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
        QPushButton, QFileDialog, QMessageBox, QComboBox, QSpinBox,
        QDoubleSpinBox, QTextEdit, QGroupBox, QStackedWidget,
        QGraphicsView, QGraphicsScene, QSplitter
    )

    class RasterCanvas(QGraphicsView):
        """Detection-like local raster preview canvas.

        Wheel zoom and drag pan behave like Mustatil's Detection canvas. The
        preview is intentionally local-only: no WebEngine and no web tile calls.
        A small overlay in scene coordinates shows the currently active raster,
        RGB/index/formula and stretch settings directly inside the preview.
        """
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setScene(QGraphicsScene(self))
            self.setRenderHint(QPainter.Antialiasing, False)
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            self.setBackgroundBrush(QBrush(QColor("#f4f4f4")))
            self.pixmap_item = None
            self.overlay_group = []
            self.overlay_text = "No raster loaded."

        def wheelEvent(self, event):
            try:
                factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
                self.scale(factor, factor)
            except Exception:
                super().wheelEvent(event)

        def set_overlay_text(self, text: str):
            self.overlay_text = str(text or "")
            self._draw_overlay()

        def set_pil_image(self, pil_img, overlay_text: str = ""):
            if pil_img is None:
                return
            im = pil_img.convert("RGBA")
            data = im.tobytes("raw", "RGBA")
            qimg = QImage(data, im.width, im.height, QImage.Format_RGBA8888)
            pix = QPixmap.fromImage(qimg.copy())
            self.scene().clear()
            self.overlay_group = []
            self.pixmap_item = self.scene().addPixmap(pix)
            self.scene().setSceneRect(0, 0, pix.width(), pix.height())
            self.overlay_text = str(overlay_text or self.overlay_text or "")
            self._draw_overlay()
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

        def _draw_overlay(self):
            try:
                for item in list(self.overlay_group):
                    try:
                        self.scene().removeItem(item)
                    except Exception:
                        pass
                self.overlay_group = []
                text = (self.overlay_text or "").strip()
                if not text:
                    return
                lines = text.splitlines()
                fm_width = max(260, min(900, max(len(line) for line in lines) * 7 + 22))
                height = max(34, len(lines) * 18 + 16)
                bg = self.scene().addRect(10, 10, fm_width, height, QPen(QColor(40, 40, 40)), QBrush(QColor(255, 255, 255, 225)))
                bg.setZValue(10000)
                t = self.scene().addText(text)
                t.setDefaultTextColor(QColor(20, 20, 20))
                t.setPos(20, 16)
                t.setZValue(10001)
                self.overlay_group = [bg, t]
            except Exception:
                pass

    class _RemoteSensingTab(QWidget):
        INDEX_DEFS: Dict[str, Tuple[str, str]] = {
            "NDVI": ("(NIR - Red) / (NIR + Red)", "Vegetation and crop/plant contrast"),
            "NDWI": ("(Green - NIR) / (Green + NIR)", "Water and moisture contrast"),
            "MNDWI": ("(Green - SWIR1) / (Green + SWIR1)", "Open water / built-up suppression"),
            "NDBI": ("(SWIR1 - NIR) / (SWIR1 + NIR)", "Built-up / bare surface contrast"),
            "SAVI": ("1.5 * (NIR - Red) / (NIR + Red + 0.5)", "Vegetation with soil adjustment"),
            "EVI": ("2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)", "Enhanced vegetation contrast"),
            "BSI": ("((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))", "Bare soil / archaeological surface contrast"),
        }

        def __init__(self, workspace=None):
            super().__init__()
            self.workspace = workspace
            self.raster_path = ""
            self.src = None
            self.profile = None
            self.raster_width = 0
            self.raster_height = 0
            self.raster_count = 0
            self.preview_image = None
            self.last_result = None
            self.last_result_profile = None
            self._build_ui()
            self._update_status()

        def _build_ui(self):
            root = QVBoxLayout(self)
            root.setContentsMargins(8, 8, 8, 8)
            root.setSpacing(6)

            header = QGroupBox("Input raster")
            hgrid = QGridLayout(header)
            self.path_edit = QLineEdit()
            self.path_edit.setPlaceholderText("Load a local GeoTIFF/TIFF/IMG or image file")
            browse = QPushButton("Browse raster")
            browse.clicked.connect(self.browse_raster)
            load = QPushButton("Load / Refresh preview")
            load.clicked.connect(self.load_raster)
            save_preview = QPushButton("Save preview PNG")
            save_preview.clicked.connect(self.save_preview_png)
            export_current = QPushButton("Export current mode GeoTIFF")
            export_current.clicked.connect(self.export_result_geotiff)
            hgrid.addWidget(QLabel("Raster"), 0, 0)
            hgrid.addWidget(self.path_edit, 0, 1)
            hgrid.addWidget(browse, 0, 2)
            hgrid.addWidget(load, 0, 3)
            hgrid.addWidget(save_preview, 0, 4)
            hgrid.addWidget(export_current, 0, 5)
            root.addWidget(header)

            yolo_box = QGroupBox("Chunked YOLO detection on loaded raster")
            ygrid = QGridLayout(yolo_box)
            self.yolo_model_edit = QLineEdit()
            self.yolo_model_edit.setPlaceholderText("Select a YOLO .pt model for chunked raster detection")
            yolo_browse = QPushButton("Browse model")
            yolo_browse.clicked.connect(self.browse_yolo_model)
            self.yolo_device_combo = QComboBox()
            self.yolo_device_combo.setEditable(True)
            self.yolo_device_combo.addItems(["Auto CUDA", "cuda", "0", "cpu"])
            self.yolo_conf_spin = QDoubleSpinBox(); self.yolo_conf_spin.setRange(0.001, 1.0); self.yolo_conf_spin.setDecimals(3); self.yolo_conf_spin.setSingleStep(0.01); self.yolo_conf_spin.setValue(0.05)
            self.yolo_imgsz_spin = QSpinBox(); self.yolo_imgsz_spin.setRange(64, 8192); self.yolo_imgsz_spin.setSingleStep(32); self.yolo_imgsz_spin.setValue(640)
            self.yolo_tile_spin = QSpinBox(); self.yolo_tile_spin.setRange(256, 8192); self.yolo_tile_spin.setSingleStep(256); self.yolo_tile_spin.setValue(1024)
            self.yolo_overlap_spin = QSpinBox(); self.yolo_overlap_spin.setRange(0, 4096); self.yolo_overlap_spin.setSingleStep(32); self.yolo_overlap_spin.setValue(160)
            self.yolo_shifted_tiles = QComboBox(); self.yolo_shifted_tiles.addItems(["Normal tiles", "Shifted tiles also"])
            self.run_yolo_btn = QPushButton("Run chunked YOLO")
            self.run_yolo_btn.clicked.connect(self.run_chunked_yolo)
            self.export_yolo_btn = QPushButton("Export YOLO detections GPKG")
            self.export_yolo_btn.clicked.connect(self.export_yolo_detections_gpkg)
            ygrid.addWidget(QLabel("YOLO model"), 0, 0); ygrid.addWidget(self.yolo_model_edit, 0, 1, 1, 5); ygrid.addWidget(yolo_browse, 0, 6)
            ygrid.addWidget(QLabel("Device"), 1, 0); ygrid.addWidget(self.yolo_device_combo, 1, 1)
            ygrid.addWidget(QLabel("Confidence"), 1, 2); ygrid.addWidget(self.yolo_conf_spin, 1, 3)
            ygrid.addWidget(QLabel("imgsz"), 1, 4); ygrid.addWidget(self.yolo_imgsz_spin, 1, 5)
            ygrid.addWidget(QLabel("Tile"), 2, 0); ygrid.addWidget(self.yolo_tile_spin, 2, 1)
            ygrid.addWidget(QLabel("Overlap"), 2, 2); ygrid.addWidget(self.yolo_overlap_spin, 2, 3)
            ygrid.addWidget(self.yolo_shifted_tiles, 2, 4)
            ygrid.addWidget(self.run_yolo_btn, 2, 5)
            ygrid.addWidget(self.export_yolo_btn, 2, 6)
            root.addWidget(yolo_box)
            self.yolo_detections = []

            split = QSplitter(Qt.Horizontal)
            root.addWidget(split, 1)

            left = QWidget()
            left_lay = QVBoxLayout(left)
            left_lay.setContentsMargins(0, 0, 6, 0)
            left_lay.setSpacing(6)

            mode_box = QGroupBox("Analysis mode")
            mode_lay = QGridLayout(mode_box)
            self.mode_combo = QComboBox()
            self.mode_combo.addItems(["Raster Viewer", "Spectral Indices", "Band Calculator", "Archaeological Features"])
            mode_lay.addWidget(QLabel("Mode"), 0, 0)
            mode_lay.addWidget(self.mode_combo, 0, 1)
            left_lay.addWidget(mode_box)

            controls = QStackedWidget()
            left_lay.addWidget(controls, 1)
            self.mode_combo.currentIndexChanged.connect(controls.setCurrentIndex)
            self.mode_combo.currentTextChanged.connect(self._on_mode_changed)

            viewer_tab = QWidget()
            viewer_lay = QGridLayout(viewer_tab)
            self.max_preview_spin = QSpinBox(); self.max_preview_spin.setRange(512, 8192); self.max_preview_spin.setValue(2400)
            self.stretch_combo = QComboBox(); self.stretch_combo.addItems(["Percentile 2-98", "Min-Max", "None"])
            self.r_band = QSpinBox(); self.g_band = QSpinBox(); self.b_band = QSpinBox()
            for s in (self.r_band, self.g_band, self.b_band):
                s.setRange(1, 999); s.setValue(1)
            self.g_band.setValue(2); self.b_band.setValue(3)
            apply_rgb = QPushButton("Apply RGB preview")
            apply_rgb.clicked.connect(self.refresh_preview)
            reset_viewer = QPushButton("Set back to standard values")
            reset_viewer.clicked.connect(self.reset_viewer_defaults)
            viewer_lay.addWidget(QLabel("Max preview size"), 0, 0); viewer_lay.addWidget(self.max_preview_spin, 0, 1)
            viewer_lay.addWidget(QLabel("Stretch"), 1, 0); viewer_lay.addWidget(self.stretch_combo, 1, 1)
            viewer_lay.addWidget(QLabel("Red band"), 2, 0); viewer_lay.addWidget(self.r_band, 2, 1)
            viewer_lay.addWidget(QLabel("Green band"), 3, 0); viewer_lay.addWidget(self.g_band, 3, 1)
            viewer_lay.addWidget(QLabel("Blue band"), 4, 0); viewer_lay.addWidget(self.b_band, 4, 1)
            viewer_lay.addWidget(apply_rgb, 5, 0, 1, 2)
            viewer_lay.addWidget(reset_viewer, 6, 0, 1, 2)
            viewer_lay.setRowStretch(7, 1)
            controls.addWidget(viewer_tab)

            index_tab = QWidget()
            igrid = QGridLayout(index_tab)
            self.index_combo = QComboBox(); self.index_combo.addItems(list(self.INDEX_DEFS.keys()))
            self.index_render_combo = QComboBox(); self.index_render_combo.addItems(["Auto desert contrast", "Robust 1-99 grayscale", "Fixed -1..1 grayscale"])
            self.blue_band = QSpinBox(); self.green_band = QSpinBox(); self.red_band = QSpinBox(); self.nir_band = QSpinBox(); self.swir1_band = QSpinBox()
            defaults = [1, 2, 3, 4, 5]
            for spin, val in zip((self.blue_band, self.green_band, self.red_band, self.nir_band, self.swir1_band), defaults):
                spin.setRange(1, 999); spin.setValue(val)
            self.savi_l = QDoubleSpinBox(); self.savi_l.setRange(0, 2); self.savi_l.setSingleStep(0.1); self.savi_l.setValue(0.5)
            run_index = QPushButton("Preview index")
            run_index.clicked.connect(self.preview_index)
            export_index = QPushButton("Export index GeoTIFF")
            export_index.clicked.connect(self.export_result_geotiff)
            reset_index = QPushButton("Set back to standard values")
            reset_index.clicked.connect(self.reset_index_defaults)
            row = 0
            for label, widget in [
                ("Index", self.index_combo), ("Preview contrast", self.index_render_combo), ("Blue", self.blue_band), ("Green", self.green_band), ("Red", self.red_band),
                ("NIR", self.nir_band), ("SWIR1", self.swir1_band), ("SAVI L", self.savi_l)
            ]:
                igrid.addWidget(QLabel(label), row, 0); igrid.addWidget(widget, row, 1); row += 1
            igrid.addWidget(run_index, row, 0, 1, 2); row += 1
            igrid.addWidget(export_index, row, 0, 1, 2); row += 1
            igrid.addWidget(reset_index, row, 0, 1, 2)
            igrid.setRowStretch(row + 1, 1)
            controls.addWidget(index_tab)

            calc_tab = QWidget()
            cgrid = QGridLayout(calc_tab)
            self.formula_edit = QLineEdit("(NIR - Red) / (NIR + Red)")
            self.formula_edit.setPlaceholderText("Example: (NIR - Red) / (NIR + Red)")
            run_formula = QPushButton("Preview formula")
            run_formula.clicked.connect(self.preview_formula)
            export_formula = QPushButton("Export formula GeoTIFF")
            export_formula.clicked.connect(self.export_result_geotiff)
            reset_formula = QPushButton("Set back to standard values")
            reset_formula.clicked.connect(self.reset_formula_defaults)
            help_text = QLabel(
                "Use readable names or band numbers:\n"
                "Blue, Green, Red, NIR, SWIR1 are taken from the Spectral Indices band settings.\n"
                "B1, B2, B3 ... also work directly for raster band numbers.\n"
                "Example NDVI: (NIR - Red) / (NIR + Red)\n"
                "Example BSI: ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))"
            )
            help_text.setWordWrap(True)
            cgrid.addWidget(QLabel("Formula"), 0, 0); cgrid.addWidget(self.formula_edit, 0, 1)
            cgrid.addWidget(QLabel("Variables"), 1, 0); cgrid.addWidget(help_text, 1, 1)
            cgrid.addWidget(run_formula, 2, 0, 1, 2)
            cgrid.addWidget(export_formula, 3, 0, 1, 2)
            cgrid.addWidget(reset_formula, 4, 0, 1, 2)
            cgrid.setRowStretch(5, 1)
            controls.addWidget(calc_tab)

            arch_tab = QWidget()
            agrid = QGridLayout(arch_tab)
            self.arch_method_combo = QComboBox(); self.arch_method_combo.addItems([
                "Hillshade", "Local Relief Model", "PCA false colour", "Edge Enhancement", "Rectangular Feature Enhancement"
            ])
            self.arch_band = QSpinBox(); self.arch_band.setRange(1, 999); self.arch_band.setValue(1)
            self.hill_azimuth = QDoubleSpinBox(); self.hill_azimuth.setRange(0, 360); self.hill_azimuth.setSingleStep(15); self.hill_azimuth.setValue(315)
            self.hill_altitude = QDoubleSpinBox(); self.hill_altitude.setRange(1, 89); self.hill_altitude.setSingleStep(5); self.hill_altitude.setValue(40)
            self.lrm_radius = QSpinBox(); self.lrm_radius.setRange(3, 401); self.lrm_radius.setSingleStep(2); self.lrm_radius.setValue(41)
            self.edge_strength = QDoubleSpinBox(); self.edge_strength.setRange(0.5, 8.0); self.edge_strength.setSingleStep(0.25); self.edge_strength.setValue(2.0)
            run_arch = QPushButton("Preview archaeology filter")
            run_arch.clicked.connect(self.preview_archaeology)
            export_arch = QPushButton("Export archaeology GeoTIFF")
            export_arch.clicked.connect(self.export_result_geotiff)
            reset_arch = QPushButton("Set back to standard values")
            reset_arch.clicked.connect(self.reset_archaeology_defaults)
            arch_help = QLabel(
                "Use these filters before YOLO to make desert structures more visible.\n"
                "Hillshade: best for DEMs. LRM: removes broad terrain and highlights small walls/cairns.\n"
                "PCA: compresses multispectral contrast. Edge/Rectangular Enhancement: emphasizes straight walls and corners."
            )
            arch_help.setWordWrap(True)
            row = 0
            for label, widget in [
                ("Method", self.arch_method_combo), ("Main/DEM band", self.arch_band),
                ("Hillshade azimuth", self.hill_azimuth), ("Hillshade altitude", self.hill_altitude),
                ("LRM radius px", self.lrm_radius), ("Edge strength", self.edge_strength)
            ]:
                agrid.addWidget(QLabel(label), row, 0); agrid.addWidget(widget, row, 1); row += 1
            agrid.addWidget(run_arch, row, 0, 1, 2); row += 1
            agrid.addWidget(export_arch, row, 0, 1, 2); row += 1
            agrid.addWidget(reset_arch, row, 0, 1, 2); row += 1
            agrid.addWidget(QLabel("Use for Mustatils"), row, 0); agrid.addWidget(arch_help, row, 1); row += 1
            agrid.setRowStretch(row, 1)
            controls.addWidget(arch_tab)


            info_box = QGroupBox("Raster information / log")
            il = QVBoxLayout(info_box)
            self.info = QTextEdit(); self.info.setReadOnly(True); self.info.setMinimumHeight(160)
            il.addWidget(self.info)
            left_lay.addWidget(info_box)

            split.addWidget(left)

            right = QWidget()
            right_lay = QVBoxLayout(right)
            right_lay.setContentsMargins(6, 0, 0, 0)
            self.raster_canvas = RasterCanvas()
            right_lay.addWidget(self.raster_canvas, 1)
            self.status_label = QLabel("No raster loaded.")
            right_lay.addWidget(self.status_label)
            split.addWidget(right)
            split.setSizes([360, 900])

        def _on_mode_changed(self, mode: str):
            """Keep preview and log in sync when the dropdown mode changes.

            This deliberately uses one QStackedWidget page per mode instead of
            internal tabs, so the Remote Sensing tab cannot lose the
            Archaeological Features page when the UI is rebuilt.
            """
            try:
                if str(mode) == "Archaeological Features":
                    self.status_label.setText("Mode: Archaeological Features. Choose a filter and click Preview archaeology filter.")
                else:
                    self.status_label.setText("Mode: " + str(mode))
                self._log("Remote Sensing mode selected: " + str(mode))
            except Exception:
                pass

        def reset_viewer_defaults(self):
            """Reset Raster Viewer controls to safe default values."""
            try:
                self.max_preview_spin.setValue(2400)
                self.stretch_combo.setCurrentText("Percentile 2-98")
                self.r_band.setValue(1)
                self.g_band.setValue(2 if self.raster_count >= 2 else 1)
                self.b_band.setValue(3 if self.raster_count >= 3 else 1)
                if self.src is not None:
                    self.refresh_preview()
                self._log("Raster Viewer settings reset to standard values.")
            except Exception as exc:
                self._log("Reset Raster Viewer failed: " + str(exc))

        def reset_index_defaults(self):
            """Reset Spectral Indices controls to common satellite defaults."""
            try:
                self.index_combo.setCurrentText("NDVI")
                self.index_render_combo.setCurrentText("Auto desert contrast")
                self.blue_band.setValue(1)
                self.green_band.setValue(2 if self.raster_count >= 2 else 1)
                self.red_band.setValue(3 if self.raster_count >= 3 else 1)
                self.nir_band.setValue(4 if self.raster_count >= 4 else max(1, self.raster_count))
                self.swir1_band.setValue(5 if self.raster_count >= 5 else max(1, self.raster_count))
                self.savi_l.setValue(0.5)
                if self.src is not None:
                    self.preview_index()
                self._log("Spectral Indices settings reset to standard values.")
            except Exception as exc:
                self._log("Reset Spectral Indices failed: " + str(exc))

        def reset_formula_defaults(self):
            """Reset Band Calculator to a readable NDVI formula."""
            try:
                self.formula_edit.setText("(NIR - Red) / (NIR + Red)")
                if self.src is not None:
                    self.preview_formula()
                self._log("Band Calculator settings reset to standard values.")
            except Exception as exc:
                self._log("Reset Band Calculator failed: " + str(exc))

        def reset_archaeology_defaults(self):
            """Reset Archaeology controls to useful desert/DEM defaults."""
            try:
                self.arch_method_combo.setCurrentText("Hillshade")
                self.arch_band.setValue(1)
                self.hill_azimuth.setValue(315)
                self.hill_altitude.setValue(40)
                self.lrm_radius.setValue(41)
                self.edge_strength.setValue(2.0)
                if self.src is not None:
                    self.preview_archaeology()
                self._log("Archaeology settings reset to standard values.")
            except Exception as exc:
                self._log("Reset Archaeology failed: " + str(exc))

        def _log(self, text: str):
            try:
                self.info.append(str(text))
            except Exception:
                pass
            _safe_log(self.workspace, text)

        def _raster_name(self):
            try:
                return Path(self.raster_path).name if self.raster_path else "No raster"
            except Exception:
                return "Raster"

        def _preview_settings_text(self, mode: str, extra: str = "") -> str:
            try:
                base = [
                    f"Remote Sensing Preview | {mode}",
                    f"Raster: {self._raster_name()} | {self.raster_width} x {self.raster_height} px | bands: {self.raster_count}",
                    f"Stretch: {self.stretch_combo.currentText()} | max preview: {self.max_preview_spin.value()} px",
                ]
                if mode == "RGB":
                    base.append(f"RGB bands: R={self.r_band.value()}  G={self.g_band.value()}  B={self.b_band.value()}")
                elif mode == "Spectral index":
                    base.append(
                        f"Index: {self.index_combo.currentText()} | Contrast={self.index_render_combo.currentText()} | Blue={self.blue_band.value()} Green={self.green_band.value()} Red={self.red_band.value()} NIR={self.nir_band.value()} SWIR1={self.swir1_band.value()} SAVI L={self.savi_l.value():.2f}"
                    )
                elif mode == "Formula":
                    base.append(f"Formula: {self.formula_edit.text().strip()}")
                elif mode in ("Archaeology", "Archaeological Features"):
                    base.append(f"Method: {self.arch_method_combo.currentText()} | Band={self.arch_band.value()} | Hillshade az={self.hill_azimuth.value():.0f} alt={self.hill_altitude.value():.0f} | LRM radius={self.lrm_radius.value()} | Edge strength={self.edge_strength.value():.2f}")
                if extra:
                    base.append(str(extra))
                return "\n".join(base)
            except Exception:
                return str(mode or "Remote Sensing Preview")

        def _update_status(self):
            deps = [
                "Local raster preview only: Web preview is removed.",
                "rasterio: " + ("OK - GeoTIFF metadata/export enabled" if _have("rasterio") else "missing - install rasterio for GeoTIFF metadata/export"),
                "Pillow: " + ("OK" if _have("PIL") else "missing"),
                "numpy: " + ("OK" if _have("numpy") else "missing"),
                "ultralytics/torch: " + ("OK - YOLO detection available" if _have("ultralytics") and _have("torch") else "missing - install ultralytics and torch for YOLO"),
                "Use this tab to inspect raster bands, create false-colour previews, calculate indices, export derived rasters, and run chunked YOLO on huge rasters.",
            ]
            self.info.setPlainText("\n".join(deps))

        def browse_raster(self):
            path, _ = QFileDialog.getOpenFileName(
                self, "Open raster", "", "Raster files (*.tif *.tiff *.img *.vrt *.jp2 *.png *.jpg *.jpeg *.bmp *.webp);;All files (*.*)"
            )
            if path:
                self.path_edit.setText(path)
                self.load_raster()

        def load_raster(self):
            path = self.path_edit.text().strip().strip('"')
            if not path:
                QMessageBox.warning(self, "Remote Sensing", "Please select a raster file first.")
                return
            if not Path(path).exists():
                QMessageBox.warning(self, "Remote Sensing", "Raster file does not exist.")
                return
            self.raster_path = path
            self.last_result = None
            self.last_result_profile = None
            try:
                if _have("rasterio") and Path(path).suffix.lower() in {".tif", ".tiff", ".img", ".vrt", ".jp2"}:
                    import rasterio
                    if self.src is not None:
                        try: self.src.close()
                        except Exception: pass
                    self.src = rasterio.open(path)
                    self.profile = dict(self.src.profile)
                    self.raster_width = int(self.src.width); self.raster_height = int(self.src.height); self.raster_count = int(self.src.count)
                    for spin in (self.r_band, self.g_band, self.b_band, self.blue_band, self.green_band, self.red_band, self.nir_band, self.swir1_band, self.arch_band):
                        spin.setRange(1, max(1, self.raster_count))
                    if self.raster_count < 3:
                        self.g_band.setValue(1); self.b_band.setValue(1)
                    self._log(f"Loaded rasterio raster: {self.raster_width} x {self.raster_height}, bands={self.raster_count}, crs={self.src.crs}")
                else:
                    from PIL import Image
                    im = Image.open(path)
                    self.src = im
                    self.profile = None
                    self.raster_width, self.raster_height = im.size
                    self.raster_count = len(im.getbands()) if hasattr(im, "getbands") else 3
                    for spin in (self.r_band, self.g_band, self.b_band, self.blue_band, self.green_band, self.red_band, self.nir_band, self.swir1_band, self.arch_band):
                        spin.setRange(1, max(1, self.raster_count))
                    self._log(f"Loaded image raster: {self.raster_width} x {self.raster_height}, bands={self.raster_count}")
                self.refresh_preview()
            except Exception as exc:
                QMessageBox.critical(self, "Remote Sensing", "Could not load raster:\n" + str(exc))
                self._log("Load failed: " + str(exc))

        def _read_bands_preview(self, bands):
            import numpy as np
            maxs = int(self.max_preview_spin.value())
            if self.src is None:
                raise RuntimeError("No raster loaded.")
            if _have("rasterio") and hasattr(self.src, "read"):
                scale = min(1.0, maxs / max(1, self.raster_width), maxs / max(1, self.raster_height))
                ow = max(1, int(self.raster_width * scale)); oh = max(1, int(self.raster_height * scale))
                bands = [max(1, min(int(b), self.raster_count)) for b in bands]
                arr = self.src.read(bands, out_shape=(len(bands), oh, ow), masked=True).astype("float32")
                if hasattr(arr, "filled"):
                    arr = arr.filled(float("nan"))
                return arr
            from PIL import Image
            im = self.src.convert("RGB") if hasattr(self.src, "convert") else Image.open(self.raster_path).convert("RGB")
            im.thumbnail((maxs, maxs))
            arr = np.asarray(im).astype("float32")
            arr = np.transpose(arr[:, :, :3], (2, 0, 1))
            return arr

        def _read_band_full(self, band: int):
            import numpy as np
            if self.src is None:
                raise RuntimeError("No raster loaded.")
            band = int(band)
            if _have("rasterio") and hasattr(self.src, "read"):
                b = max(1, min(band, self.raster_count))
                arr = self.src.read(b, masked=True).astype("float32")
                if hasattr(arr, "filled"):
                    arr = arr.filled(float("nan"))
                return arr
            arr = self._read_bands_preview([band])
            return arr[0]

        def _stretch_to_uint8(self, arr):
            import numpy as np
            arr = np.asarray(arr, dtype="float32")
            out = []
            mode = self.stretch_combo.currentText()
            for band in arr:
                finite = np.isfinite(band)
                if not finite.any():
                    out.append(np.zeros(band.shape, dtype="uint8")); continue
                vals = band[finite]
                if mode == "None":
                    lo, hi = 0.0, 255.0
                elif mode == "Min-Max":
                    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
                else:
                    lo, hi = float(np.nanpercentile(vals, 2)), float(np.nanpercentile(vals, 98))
                if hi <= lo: hi = lo + 1.0
                scaled = (np.clip((band - lo) / (hi - lo), 0, 1) * 255).astype("uint8")
                out.append(scaled)
            return np.stack(out, axis=0)

        def _array_to_preview(self, arr):
            import numpy as np
            from PIL import Image
            arr8 = self._stretch_to_uint8(arr)
            if arr8.shape[0] == 1:
                img = Image.fromarray(arr8[0], mode="L").convert("RGB")
            else:
                while arr8.shape[0] < 3:
                    arr8 = np.concatenate([arr8, arr8[-1:]], axis=0)
                img = Image.fromarray(np.transpose(arr8[:3], (1, 2, 0)), mode="RGB")
            return img

        def refresh_preview(self):
            try:
                bands = [self.r_band.value(), self.g_band.value(), self.b_band.value()]
                arr = self._read_bands_preview(bands)
                img = self._array_to_preview(arr)
                self.preview_image = img
                overlay = self._preview_settings_text("RGB", f"Preview image: {img.width} x {img.height} px")
                self.raster_canvas.set_pil_image(img, overlay)
                self.status_label.setText(f"Preview: RGB bands {bands} | {img.width} x {img.height}")
            except Exception as exc:
                QMessageBox.warning(self, "Remote Sensing", "Preview failed:\n" + str(exc))
                self._log("Preview failed: " + str(exc))

        def _safe_div(self, a, b):
            import numpy as np
            with np.errstate(divide="ignore", invalid="ignore"):
                out = a / b
            out[~np.isfinite(out)] = np.nan
            return out

        def _calc_index(self, full=True):
            import numpy as np
            index = self.index_combo.currentText()
            reader = self._read_band_full if full else (lambda b: self._read_bands_preview([b])[0])
            Blue = reader(self.blue_band.value())
            Green = reader(self.green_band.value())
            Red = reader(self.red_band.value())
            NIR = reader(self.nir_band.value())
            SWIR1 = reader(self.swir1_band.value())
            L = float(self.savi_l.value())
            if index == "NDVI":
                return self._safe_div(NIR - Red, NIR + Red)
            if index == "NDWI":
                return self._safe_div(Green - NIR, Green + NIR)
            if index == "MNDWI":
                return self._safe_div(Green - SWIR1, Green + SWIR1)
            if index == "NDBI":
                return self._safe_div(SWIR1 - NIR, SWIR1 + NIR)
            if index == "SAVI":
                return (1.0 + L) * self._safe_div(NIR - Red, NIR + Red + L)
            if index == "EVI":
                return 2.5 * self._safe_div(NIR - Red, NIR + 6.0 * Red - 7.5 * Blue + 1.0)
            if index == "BSI":
                return self._safe_div((SWIR1 + Red) - (NIR + Blue), (SWIR1 + Red) + (NIR + Blue))
            raise RuntimeError("Unknown index: " + str(index))

        def _index_stats_text(self, arr):
            """Return compact statistics so flat desert indices are understandable."""
            import numpy as np
            arr = np.asarray(arr, dtype="float32")
            finite = np.isfinite(arr)
            if not finite.any():
                return "Index stats: no finite values"
            vals = arr[finite]
            try:
                p1, p50, p99 = [float(x) for x in np.nanpercentile(vals, [1, 50, 99])]
                mean = float(np.nanmean(vals))
                std = float(np.nanstd(vals))
                note = ""
                if (p99 - p1) < 0.03:
                    note = " | Low contrast: desert/bare soil can look flat. Auto contrast is boosted."
                return f"Index stats: p1={p1:.3f} median={p50:.3f} p99={p99:.3f} mean={mean:.3f} std={std:.3f}{note}"
            except Exception:
                return "Index stats: available"

        def _index_array_to_preview(self, arr):
            """Render index arrays visibly, including low-variance desert scenes.

            In deserts, NDVI/NDWI/NDBI often varies only in a very small numeric
            range. A mathematically correct -1..1 render therefore looks flat
            gray. The default 'Auto desert contrast' intentionally stretches the
            local preview distribution so subtle surface differences become visible.
            Exported GeoTIFF values remain the real float index values.
            """
            import numpy as np
            from PIL import Image
            arr = np.asarray(arr, dtype="float32")
            finite = np.isfinite(arr)
            if not finite.any():
                return Image.fromarray(np.zeros(arr.shape, dtype="uint8"), mode="L").convert("RGB")
            vals = arr[finite]
            mode = "Auto desert contrast"
            try:
                mode = self.index_render_combo.currentText()
            except Exception:
                pass

            if mode == "Fixed -1..1 grayscale":
                lo, hi = -1.0, 1.0
            elif mode == "Robust 1-99 grayscale":
                lo, hi = float(np.nanpercentile(vals, 1)), float(np.nanpercentile(vals, 99))
            else:
                # Desert-safe auto contrast. Use a robust percentile range, but
                # when the range is tiny, expand around the median/std so the
                # preview does not collapse into one gray tone.
                p1, p5, p50, p95, p99 = [float(x) for x in np.nanpercentile(vals, [1, 5, 50, 95, 99])]
                std = float(np.nanstd(vals))
                lo, hi = p1, p99
                if (hi - lo) < 0.08:
                    width = max(0.02, min(0.35, max((p95 - p5) * 1.8, std * 5.0, (hi - lo) * 3.0)))
                    lo, hi = p50 - width / 2.0, p50 + width / 2.0

            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    lo, hi = 0.0, 1.0

            scaled = (np.clip((arr - lo) / (hi - lo), 0, 1) * 255).astype("uint8")
            scaled[~finite] = 0
            return Image.fromarray(scaled, mode="L").convert("RGB")

        def preview_index(self):
            try:
                arr = self._calc_index(full=False)
                self.last_result = arr.astype("float32")
                self.last_result_profile = None
                img = self._index_array_to_preview(arr)
                self.preview_image = img
                desc = self.INDEX_DEFS.get(self.index_combo.currentText(), ("", ""))[1]
                stats = self._index_stats_text(arr)
                overlay = self._preview_settings_text("Spectral index", desc + "\n" + stats)
                self.raster_canvas.set_pil_image(img, overlay)
                self.status_label.setText(f"Index preview: {self.index_combo.currentText()} | {desc}")
                self._log(f"Index preview created: {self.index_combo.currentText()} - {desc} | {stats}")
            except Exception as exc:
                QMessageBox.warning(self, "Remote Sensing", "Index preview failed:\n" + str(exc))
                self._log("Index preview failed: " + str(exc))

        def _build_formula_env(self, full=True):
            import numpy as np
            env = {
                "np": np, "numpy": np,
                "sqrt": np.sqrt, "abs": np.abs, "minimum": np.minimum, "maximum": np.maximum,
                "clip": np.clip, "log": np.log, "log10": np.log10,
            }
            reader = self._read_band_full if full else (lambda b: self._read_bands_preview([b])[0])
            bands = {}
            for i in range(1, int(self.raster_count or 0) + 1):
                arr = reader(i)
                bands[f"B{i}"] = arr
                bands[f"b{i}"] = arr
            env.update(bands)
            # Readable aliases. These follow the Spectral Indices band selectors, so users do not need to guess what B4 means.
            alias_map = {
                "Blue": self.blue_band.value(), "Green": self.green_band.value(), "Red": self.red_band.value(),
                "NIR": self.nir_band.value(), "SWIR1": self.swir1_band.value(),
            }
            for name, band in alias_map.items():
                try:
                    env[name] = reader(int(band))
                    env[name.lower()] = env[name]
                except Exception:
                    pass
            return env

        def preview_formula(self):
            try:
                import numpy as np
                formula = self.formula_edit.text().strip()
                if not formula:
                    raise RuntimeError("Formula is empty.")
                env = self._build_formula_env(full=False)
                allowed = {"__builtins__": {}}
                arr = eval(formula, allowed, env)  # local scientific calculator; builtins disabled
                arr = np.asarray(arr, dtype="float32")
                self.last_result = arr
                self.last_result_profile = None
                img = self._index_array_to_preview(arr)
                self.preview_image = img
                stats = self._index_stats_text(arr)
                overlay = self._preview_settings_text("Formula", "Band calculator preview\n" + stats)
                self.raster_canvas.set_pil_image(img, overlay)
                self.status_label.setText("Formula preview created.")
                self._log("Formula preview created: " + formula + " | " + stats)
            except Exception as exc:
                QMessageBox.warning(self, "Remote Sensing", "Formula preview failed:\n" + str(exc))
                self._log("Formula preview failed: " + str(exc))


        def _normalize_single_to_uint8(self, arr, robust=True):
            import numpy as np
            arr = np.asarray(arr, dtype="float32")
            finite = np.isfinite(arr)
            if not finite.any():
                return np.zeros(arr.shape, dtype="uint8")
            vals = arr[finite]
            if robust:
                lo, hi = float(np.nanpercentile(vals, 1)), float(np.nanpercentile(vals, 99))
            else:
                lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
                if hi <= lo:
                    hi = lo + 1.0
            out = (np.clip((arr - lo) / (hi - lo), 0, 1) * 255).astype("uint8")
            out[~finite] = 0
            return out

        def _gray_preview_image(self, arr, robust=True):
            from PIL import Image
            return Image.fromarray(self._normalize_single_to_uint8(arr, robust=robust), mode="L").convert("RGB")

        def _preview_gray_band(self):
            return self._read_bands_preview([self.arch_band.value()])[0]

        def _calc_hillshade_preview(self):
            import numpy as np
            z = np.asarray(self._preview_gray_band(), dtype="float32")
            z = np.nan_to_num(z, nan=float(np.nanmedian(z[np.isfinite(z)])) if np.isfinite(z).any() else 0.0)
            # Remove strong radiometric offsets; hillshade only needs local gradient.
            zy, zx = np.gradient(z)
            az = np.deg2rad(360.0 - float(self.hill_azimuth.value()) + 90.0)
            alt = np.deg2rad(float(self.hill_altitude.value()))
            slope = np.pi / 2.0 - np.arctan(np.sqrt(zx * zx + zy * zy))
            aspect = np.arctan2(-zx, zy)
            hs = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
            return np.clip(hs, 0, 1)

        def _calc_lrm_preview(self):
            import numpy as np
            from PIL import Image, ImageFilter
            z = np.asarray(self._preview_gray_band(), dtype="float32")
            z8 = self._normalize_single_to_uint8(z, robust=True)
            radius = max(3, int(self.lrm_radius.value()))
            if radius % 2 == 0:
                radius += 1
            smooth = Image.fromarray(z8, mode="L").filter(ImageFilter.GaussianBlur(radius=max(1, radius / 3.0)))
            local = z8.astype("float32") - np.asarray(smooth).astype("float32")
            return local

        def _calc_edge_preview(self):
            import numpy as np
            z = np.asarray(self._preview_gray_band(), dtype="float32")
            z = self._normalize_single_to_uint8(z, robust=True).astype("float32") / 255.0
            gy, gx = np.gradient(z)
            mag = np.sqrt(gx * gx + gy * gy) * float(self.edge_strength.value())
            return mag

        def _calc_rectangular_preview(self):
            import numpy as np
            edge = self._calc_edge_preview()
            # Emphasize straight horizontal/vertical wall-like gradients.
            z = self._normalize_single_to_uint8(self._preview_gray_band(), robust=True).astype("float32") / 255.0
            gy, gx = np.gradient(z)
            hv = (np.abs(gx) + np.abs(gy))
            diag_penalty = np.abs(np.abs(gx) - np.abs(gy))
            rect = (edge * 0.65 + hv * 1.35 + diag_penalty * 0.35) * float(self.edge_strength.value())
            return rect

        def _calc_pca_preview_image(self):
            import numpy as np
            from PIL import Image
            bands = list(range(1, min(6, int(self.raster_count or 1)) + 1))
            arr = self._read_bands_preview(bands).astype("float32")
            c, h, w = arr.shape
            flat = arr.reshape(c, -1).T
            finite = np.isfinite(flat).all(axis=1)
            if finite.sum() < 10 or c < 2:
                return self._gray_preview_image(arr[0], robust=True), arr[0]
            X = flat[finite]
            # robust scale each band before PCA
            med = np.nanmedian(X, axis=0)
            p1 = np.nanpercentile(X, 1, axis=0)
            p99 = np.nanpercentile(X, 99, axis=0)
            scale = np.where((p99 - p1) <= 0, 1.0, (p99 - p1))
            Xs = (X - med) / scale
            U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
            scores = np.zeros((flat.shape[0], min(3, Vt.shape[0])), dtype="float32")
            scores[finite, :] = (Xs @ Vt[:scores.shape[1]].T).astype("float32")
            pcs = scores.T.reshape(scores.shape[1], h, w)
            while pcs.shape[0] < 3:
                pcs = np.concatenate([pcs, pcs[-1:]], axis=0)
            rgb = []
            for pc in pcs[:3]:
                rgb.append(self._normalize_single_to_uint8(pc, robust=True))
            img = Image.fromarray(np.transpose(np.stack(rgb[:3], axis=0), (1, 2, 0)), mode="RGB")
            return img, pcs[0]

        def preview_archaeology(self):
            try:
                import numpy as np
                from PIL import Image
                method = self.arch_method_combo.currentText()
                if method == "Hillshade":
                    arr = self._calc_hillshade_preview(); img = self._gray_preview_image(arr, robust=False)
                    note = "DEM shading: walls and low relief become visible from artificial light."
                elif method == "Local Relief Model":
                    arr = self._calc_lrm_preview(); img = self._gray_preview_image(arr, robust=True)
                    note = "LRM: broad terrain is removed; small walls, cairns and platforms stand out."
                elif method == "PCA false colour":
                    img, arr = self._calc_pca_preview_image()
                    note = "PCA: combines several bands into high-contrast false colour for subtle surface differences."
                elif method == "Rectangular Feature Enhancement":
                    arr = self._calc_rectangular_preview(); img = self._gray_preview_image(arr, robust=True)
                    note = "Rectangular enhancement: emphasizes straight lines, wall traces and corners; use as a Mustatil candidate pre-filter."
                else:
                    arr = self._calc_edge_preview(); img = self._gray_preview_image(arr, robust=True)
                    note = "Edge enhancement: emphasizes linear structures before YOLO or manual inspection."
                self.last_result = np.asarray(arr, dtype="float32") if 'arr' in locals() else None
                self.preview_image = img
                stats = self._index_stats_text(np.asarray(arr, dtype="float32")) if 'arr' in locals() else ""
                overlay = self._preview_settings_text("Archaeological Features", note + ("\n" + stats if stats else ""))
                self.raster_canvas.set_pil_image(img, overlay)
                self.status_label.setText("Archaeological Features preview: " + method)
                self._log("Archaeological Features preview created: " + method + " | " + note)
            except Exception as exc:
                QMessageBox.warning(self, "Remote Sensing", "Archaeology preview failed:\n" + str(exc))
                self._log("Archaeology preview failed: " + str(exc))

        def _read_window_band(self, band: int, window, boundless: bool = False):
            """Read one raster band for a rasterio window as float32 with NaN nodata."""
            import numpy as np
            band = max(1, min(int(band), int(self.raster_count or 1)))
            arr = self.src.read(band, window=window, boundless=bool(boundless), masked=True).astype("float32")
            if hasattr(arr, "filled"):
                arr = arr.filled(float("nan"))
            return np.asarray(arr, dtype="float32")

        def _read_window_bands_uint8(self, bands, window):
            """Read RGB bands for chunked RGB GeoTIFF export and apply local stretch."""
            import numpy as np
            bands = [max(1, min(int(b), int(self.raster_count or 1))) for b in bands]
            arr = self.src.read(bands, window=window, masked=True).astype("float32")
            if hasattr(arr, "filled"):
                arr = arr.filled(float("nan"))
            return self._stretch_to_uint8(arr)

        def _calc_index_window(self, window):
            index = self.index_combo.currentText()
            Blue = self._read_window_band(self.blue_band.value(), window)
            Green = self._read_window_band(self.green_band.value(), window)
            Red = self._read_window_band(self.red_band.value(), window)
            NIR = self._read_window_band(self.nir_band.value(), window)
            SWIR1 = self._read_window_band(self.swir1_band.value(), window)
            L = float(self.savi_l.value())
            if index == "NDVI":
                return self._safe_div(NIR - Red, NIR + Red)
            if index == "NDWI":
                return self._safe_div(Green - NIR, Green + NIR)
            if index == "MNDWI":
                return self._safe_div(Green - SWIR1, Green + SWIR1)
            if index == "NDBI":
                return self._safe_div(SWIR1 - NIR, SWIR1 + NIR)
            if index == "SAVI":
                return (1.0 + L) * self._safe_div(NIR - Red, NIR + Red + L)
            if index == "EVI":
                return 2.5 * self._safe_div(NIR - Red, NIR + 6.0 * Red - 7.5 * Blue + 1.0)
            if index == "BSI":
                return self._safe_div((SWIR1 + Red) - (NIR + Blue), (SWIR1 + Red) + (NIR + Blue))
            raise RuntimeError("Unknown index: " + str(index))

        def _build_formula_env_window(self, window):
            import numpy as np
            env = {
                "np": np, "numpy": np,
                "sqrt": np.sqrt, "abs": np.abs, "minimum": np.minimum, "maximum": np.maximum,
                "clip": np.clip, "log": np.log, "log10": np.log10,
            }
            for i in range(1, int(self.raster_count or 0) + 1):
                arr = self._read_window_band(i, window)
                env[f"B{i}"] = arr
                env[f"b{i}"] = arr
            alias_map = {
                "Blue": self.blue_band.value(), "Green": self.green_band.value(), "Red": self.red_band.value(),
                "NIR": self.nir_band.value(), "SWIR1": self.swir1_band.value(),
            }
            for name, band in alias_map.items():
                try:
                    env[name] = self._read_window_band(int(band), window)
                    env[name.lower()] = env[name]
                except Exception:
                    pass
            return env

        def _expand_window(self, window, margin: int):
            from rasterio.windows import Window
            margin = max(0, int(margin))
            col0 = max(0, int(window.col_off) - margin)
            row0 = max(0, int(window.row_off) - margin)
            col1 = min(int(self.raster_width), int(window.col_off + window.width) + margin)
            row1 = min(int(self.raster_height), int(window.row_off + window.height) + margin)
            return Window(col0, row0, max(1, col1-col0), max(1, row1-row0)), int(window.col_off) - col0, int(window.row_off) - row0

        def _crop_core(self, arr, xoff: int, yoff: int, window):
            return arr[int(yoff):int(yoff + window.height), int(xoff):int(xoff + window.width)]

        def _calc_hillshade_array(self, z):
            import numpy as np
            z = np.asarray(z, dtype="float32")
            z = np.nan_to_num(z, nan=float(np.nanmedian(z[np.isfinite(z)])) if np.isfinite(z).any() else 0.0)
            zy, zx = np.gradient(z)
            az = np.deg2rad(360.0 - float(self.hill_azimuth.value()) + 90.0)
            alt = np.deg2rad(float(self.hill_altitude.value()))
            slope = np.pi / 2.0 - np.arctan(np.sqrt(zx * zx + zy * zy))
            aspect = np.arctan2(-zx, zy)
            hs = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
            return np.clip(hs, 0, 1).astype("float32")

        def _calc_lrm_array(self, z):
            from PIL import Image, ImageFilter
            import numpy as np
            z8 = self._normalize_single_to_uint8(z, robust=True)
            radius = max(3, int(self.lrm_radius.value()))
            if radius % 2 == 0:
                radius += 1
            smooth = Image.fromarray(z8, mode="L").filter(ImageFilter.GaussianBlur(radius=max(1, radius / 3.0)))
            return (z8.astype("float32") - np.asarray(smooth).astype("float32")).astype("float32")

        def _calc_edge_array(self, z):
            import numpy as np
            z = self._normalize_single_to_uint8(z, robust=True).astype("float32") / 255.0
            gy, gx = np.gradient(z)
            return (np.sqrt(gx * gx + gy * gy) * float(self.edge_strength.value())).astype("float32")

        def _calc_rectangular_array(self, z):
            import numpy as np
            z_norm = self._normalize_single_to_uint8(z, robust=True).astype("float32") / 255.0
            gy, gx = np.gradient(z_norm)
            edge = np.sqrt(gx * gx + gy * gy) * float(self.edge_strength.value())
            hv = (np.abs(gx) + np.abs(gy))
            diag_penalty = np.abs(np.abs(gx) - np.abs(gy))
            return ((edge * 0.65 + hv * 1.35 + diag_penalty * 0.35) * float(self.edge_strength.value())).astype("float32")

        def _calc_arch_window(self, window):
            method = self.arch_method_combo.currentText()
            if method == "PCA false colour":
                # True global PCA needs a full-image model. For safe direct streaming,
                # export a chunked false-colour band composite using the first three
                # available bands. The preview still shows PCA false colour.
                bands = list(range(1, min(3, int(self.raster_count or 1)) + 1))
                while len(bands) < 3:
                    bands.append(bands[-1])
                return self._read_window_bands_uint8(bands, window), "uint8", 3
            margin = 0
            if method in ("Hillshade", "Edge Enhancement", "Rectangular Feature Enhancement"):
                margin = 4
            elif method == "Local Relief Model":
                margin = max(8, int(self.lrm_radius.value()) * 2)
            win2, xoff, yoff = self._expand_window(window, margin)
            z = self._read_window_band(self.arch_band.value(), win2)
            if method == "Hillshade":
                arr = self._calc_hillshade_array(z)
            elif method == "Local Relief Model":
                arr = self._calc_lrm_array(z)
            elif method == "Rectangular Feature Enhancement":
                arr = self._calc_rectangular_array(z)
            else:
                arr = self._calc_edge_array(z)
            return self._crop_core(arr, xoff, yoff, window).astype("float32"), "float32", 1

        def _current_export_kind(self):
            mode = self.mode_combo.currentText()
            if mode == "Raster Viewer":
                return "rgb"
            if mode == "Spectral Indices":
                return "index"
            if mode == "Band Calculator":
                return "formula"
            return "archaeology"

        def browse_yolo_model(self):
            path, _ = QFileDialog.getOpenFileName(self, "Choose YOLO model", "", "YOLO models (*.pt *.onnx *.engine);;All files (*.*)")
            if path:
                self.yolo_model_edit.setText(path)

        def _resolve_yolo_device(self) -> str:
            raw = str(self.yolo_device_combo.currentText() or "cpu").strip()
            if raw.lower() == "auto cuda":
                try:
                    import torch
                    return "0" if torch.cuda.is_available() else "cpu"
                except Exception:
                    return "cpu"
            if raw.lower() == "cuda":
                return "0"
            return raw or "cpu"

        def _iter_tile_windows(self, tile_size: int, overlap: int, shifted: bool = False):
            from rasterio.windows import Window
            tile_size = max(64, int(tile_size))
            overlap = max(0, min(int(overlap), tile_size - 1))
            stride = max(1, tile_size - overlap)
            offsets = [0]
            if shifted and stride // 2 > 0:
                offsets.append(stride // 2)
            seen = set()
            for off in offsets:
                y = off
                while y < int(self.raster_height):
                    x = off
                    while x < int(self.raster_width):
                        col = int(x); row = int(y)
                        if (col, row) not in seen:
                            seen.add((col, row))
                            w = min(tile_size, int(self.raster_width) - col)
                            h = min(tile_size, int(self.raster_height) - row)
                            if w > 8 and h > 8:
                                yield Window(col, row, w, h)
                        if x + tile_size >= int(self.raster_width):
                            break
                        x += stride
                    if y + tile_size >= int(self.raster_height):
                        break
                    y += stride

        def _read_window_rgb_image(self, window):
            import numpy as np
            from PIL import Image
            if self.src is None:
                raise RuntimeError("No raster loaded.")
            bands = [self.r_band.value(), self.g_band.value(), self.b_band.value()]
            if hasattr(self.src, "read"):
                data = self._read_window_bands_uint8(bands, window)
                return Image.fromarray(np.transpose(data[:3], (1, 2, 0)), mode="RGB")
            # Fallback for normal image files.
            left, top = int(window.col_off), int(window.row_off)
            right, bottom = left + int(window.width), top + int(window.height)
            return self.src.convert("RGB").crop((left, top, right, bottom))

        def _pixel_bbox_to_geometry(self, x1: float, y1: float, x2: float, y2: float):
            # Return a GeoJSON polygon in the source raster CRS when georeferencing is available.
            try:
                transform = self.src.transform
                pts_px = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
                pts = []
                for px, py in pts_px:
                    X, Y = transform * (float(px), float(py))
                    pts.append((float(X), float(Y)))
                return {"type": "Polygon", "coordinates": [pts]}
            except Exception:
                pts = [(float(x1), float(y1)), (float(x2), float(y1)), (float(x2), float(y2)), (float(x1), float(y2)), (float(x1), float(y1))]
                return {"type": "Polygon", "coordinates": [pts]}

        def _nms_detections(self, detections, iou_threshold: float = 0.50):
            # Lightweight NMS across overlapped tiles to remove duplicate boxes.
            def iou(a, b):
                ax1, ay1, ax2, ay2 = a["px_bbox"]; bx1, by1, bx2, by2 = b["px_bbox"]
                ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                if inter <= 0:
                    return 0.0
                aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
                bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
                return inter / max(1e-9, aa + bb - inter)
            out = []
            for det in sorted(detections, key=lambda d: float(d.get("confidence", 0.0)), reverse=True):
                same = [d for d in out if int(d.get("class_id", -1)) == int(det.get("class_id", -2))]
                if any(iou(det, d) >= iou_threshold for d in same):
                    continue
                out.append(det)
            return out

        def run_chunked_yolo(self):
            """Run YOLO on the loaded raster tile-by-tile, using CPU/CUDA like Detection.

            Huge GeoTIFFs are never loaded completely into RAM. Each tile is read,
            converted to RGB, sent to YOLO, then discarded. Overlap and optional
            shifted tiles are supported to catch objects crossing tile borders.
            """
            model_path = str(self.yolo_model_edit.text() or "").strip().strip('"')
            if not model_path or not Path(model_path).exists():
                QMessageBox.warning(self, "Remote Sensing YOLO", "Please select an existing YOLO .pt model first.")
                return
            if self.src is None:
                QMessageBox.warning(self, "Remote Sensing YOLO", "Load a raster before running YOLO.")
                return
            try:
                import numpy as np
                from ultralytics import YOLO
                model = YOLO(model_path)
                device = self._resolve_yolo_device()
                conf = float(self.yolo_conf_spin.value())
                imgsz = int(self.yolo_imgsz_spin.value())
                tile = int(self.yolo_tile_spin.value())
                overlap = int(self.yolo_overlap_spin.value())
                shifted = self.yolo_shifted_tiles.currentText().startswith("Shifted")
                windows = list(self._iter_tile_windows(tile, overlap, shifted=shifted))
                total = len(windows)
                detections = []
                self._log(f"Chunked YOLO started: model={Path(model_path).name}, device={device}, conf={conf}, imgsz={imgsz}, tile={tile}, overlap={overlap}, chunks={total}")
                self.status_label.setText(f"YOLO running: 0/{total} chunks | device={device}")
                for idx, win in enumerate(windows, 1):
                    img = self._read_window_rgb_image(win)
                    arr = np.asarray(img.convert("RGB"))
                    results = model.predict(source=arr, conf=conf, imgsz=imgsz, device=device, verbose=False)
                    for res in results:
                        boxes = getattr(res, "boxes", None)
                        if boxes is None:
                            continue
                        try:
                            xyxy = boxes.xyxy.cpu().numpy()
                            cls = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros((len(xyxy),), dtype=float)
                            cf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros((len(xyxy),), dtype=float)
                        except Exception:
                            continue
                        names = getattr(res, "names", {}) or getattr(model, "names", {}) or {}
                        for bb, c, score in zip(xyxy, cls, cf):
                            x1, y1, x2, y2 = [float(v) for v in bb]
                            ax1 = float(win.col_off) + x1; ay1 = float(win.row_off) + y1
                            ax2 = float(win.col_off) + x2; ay2 = float(win.row_off) + y2
                            cid = int(c)
                            detections.append({
                                "class_id": cid,
                                "class_name": str(names.get(cid, cid)) if isinstance(names, dict) else str(cid),
                                "confidence": float(score),
                                "px_bbox": (ax1, ay1, ax2, ay2),
                                "tile_col": int(win.col_off),
                                "tile_row": int(win.row_off),
                                "tile_w": int(win.width),
                                "tile_h": int(win.height),
                            })
                    if idx == 1 or idx % 10 == 0 or idx == total:
                        self.status_label.setText(f"YOLO running: {idx}/{total} chunks | detections={len(detections)} | device={device}")
                self.yolo_detections = self._nms_detections(detections, iou_threshold=0.50)
                self.status_label.setText(f"YOLO finished: {len(self.yolo_detections)} detections after NMS")
                self._log(f"Chunked YOLO finished: raw={len(detections)}, after_nms={len(self.yolo_detections)}")
                QMessageBox.information(self, "Remote Sensing YOLO", f"YOLO finished. Detections after overlap cleanup: {len(self.yolo_detections)}")
            except Exception as exc:
                QMessageBox.critical(self, "Remote Sensing YOLO", "Chunked YOLO failed:\n" + str(exc))
                self._log("Chunked YOLO failed: " + str(exc))

        def export_yolo_detections_gpkg(self):
            if not getattr(self, "yolo_detections", None):
                QMessageBox.warning(self, "Remote Sensing YOLO", "No YOLO detections available. Run chunked YOLO first.")
                return
            if not _have("fiona"):
                QMessageBox.warning(self, "Remote Sensing YOLO", "fiona is required for GPKG export. Install fiona or export from the Detection tab.")
                return
            path, _ = QFileDialog.getSaveFileName(self, "Export YOLO detections GeoPackage", "remote_sensing_yolo_detections.gpkg", "GeoPackage (*.gpkg)")
            if not path:
                return
            try:
                import fiona
                from fiona.crs import CRS
                schema = {
                    "geometry": "Polygon",
                    "properties": {
                        "class_id": "int",
                        "class_name": "str",
                        "confidence": "float",
                        "x1_px": "float", "y1_px": "float", "x2_px": "float", "y2_px": "float",
                        "tile_col": "int", "tile_row": "int",
                    },
                }
                crs = None
                try:
                    if hasattr(self.src, "crs") and self.src.crs:
                        crs = CRS.from_wkt(self.src.crs.to_wkt())
                except Exception:
                    crs = None
                kwargs = {"driver": "GPKG", "schema": schema, "layer": "yolo_detections"}
                if crs is not None:
                    kwargs["crs"] = crs
                with fiona.open(path, "w", **kwargs) as dst:
                    for det in self.yolo_detections:
                        x1, y1, x2, y2 = det["px_bbox"]
                        dst.write({
                            "geometry": self._pixel_bbox_to_geometry(x1, y1, x2, y2),
                            "properties": {
                                "class_id": int(det.get("class_id", -1)),
                                "class_name": str(det.get("class_name", ""))[:250],
                                "confidence": float(det.get("confidence", 0.0)),
                                "x1_px": float(x1), "y1_px": float(y1), "x2_px": float(x2), "y2_px": float(y2),
                                "tile_col": int(det.get("tile_col", 0)), "tile_row": int(det.get("tile_row", 0)),
                            },
                        })
                self._log("Exported YOLO detections GeoPackage: " + path)
                QMessageBox.information(self, "Remote Sensing YOLO", "Exported YOLO detections GeoPackage:\n" + path)
            except Exception as exc:
                QMessageBox.critical(self, "Remote Sensing YOLO", "GPKG export failed:\n" + str(exc))
                self._log("YOLO GPKG export failed: " + str(exc))

        def export_result_geotiff(self):
            """Export current mode as a georeferenced GeoTIFF using chunked direct streaming.

            The output is written window-by-window through rasterio. This avoids
            loading a huge raster or derived result completely into RAM and keeps
            CRS/transform/extent from the original raster.
            """
            if not _have("rasterio"):
                QMessageBox.warning(self, "Remote Sensing", "rasterio is required for GeoTIFF export.")
                return
            if self.src is None or not hasattr(self.src, "read"):
                QMessageBox.warning(self, "Remote Sensing", "Load a GeoTIFF/rasterio raster before exporting GeoTIFF.")
                return
            kind = self._current_export_kind()
            default_name = "remote_sensing_" + kind + ".tif"
            path, _ = QFileDialog.getSaveFileName(self, "Export georeferenced GeoTIFF", default_name, "GeoTIFF (*.tif)")
            if not path:
                return
            try:
                import rasterio
                import numpy as np
                from rasterio.windows import Window
                block = 1024
                try:
                    bx, by = self.src.block_shapes[0]
                    # Use reasonable tiled chunks even if source blocks are strips.
                    block = max(256, min(2048, int(max(bx, by))))
                except Exception:
                    pass
                profile = dict(self.src.profile)
                profile.pop("blockxsize", None)
                profile.pop("blockysize", None)
                profile.pop("interleave", None)
                profile.update(driver="GTiff", compress="deflate", tiled=True, blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
                if kind == "rgb":
                    count, dtype, nodata = 3, "uint8", None
                elif kind == "archaeology" and self.arch_method_combo.currentText() == "PCA false colour":
                    count, dtype, nodata = 3, "uint8", None
                else:
                    count, dtype, nodata = 1, "float32", np.nan
                profile.update(count=count, dtype=dtype, nodata=nodata, width=self.raster_width, height=self.raster_height)

                total = ((self.raster_width + block - 1) // block) * ((self.raster_height + block - 1) // block)
                done = 0
                self._log(f"Chunked GeoTIFF export started: mode={kind}, chunks={total}, output={path}")
                with rasterio.open(path, "w", **profile) as dst:
                    for row in range(0, int(self.raster_height), block):
                        h = min(block, int(self.raster_height) - row)
                        for col in range(0, int(self.raster_width), block):
                            w = min(block, int(self.raster_width) - col)
                            win = Window(col, row, w, h)
                            if kind == "rgb":
                                data = self._read_window_bands_uint8([self.r_band.value(), self.g_band.value(), self.b_band.value()], win)
                                dst.write(data.astype("uint8"), window=win)
                            elif kind == "index":
                                arr = self._calc_index_window(win).astype("float32")
                                dst.write(arr, 1, window=win)
                            elif kind == "formula":
                                formula = self.formula_edit.text().strip()
                                if not formula:
                                    raise RuntimeError("Formula is empty.")
                                env = self._build_formula_env_window(win)
                                arr = np.asarray(eval(formula, {"__builtins__": {}}, env), dtype="float32")
                                if arr.shape == ():
                                    arr = np.full((h, w), float(arr), dtype="float32")
                                dst.write(arr.astype("float32"), 1, window=win)
                            else:
                                data, out_dtype, out_count = self._calc_arch_window(win)
                                if out_count == 3:
                                    dst.write(data.astype("uint8"), window=win)
                                else:
                                    dst.write(data.astype("float32"), 1, window=win)
                            done += 1
                            if done == 1 or done % 25 == 0 or done == total:
                                self.status_label.setText(f"Exporting GeoTIFF: {done}/{total} chunks")
                self.status_label.setText("GeoTIFF export finished.")
                self._log("Exported georeferenced chunked GeoTIFF: " + path)
                QMessageBox.information(self, "Remote Sensing", "Exported georeferenced GeoTIFF:\n" + path)
            except Exception as exc:
                QMessageBox.critical(self, "Remote Sensing", "GeoTIFF export failed:\n" + str(exc))
                self._log("GeoTIFF export failed: " + str(exc))

        def save_preview_png(self):
            if self.preview_image is None:
                QMessageBox.warning(self, "Remote Sensing", "No preview image is available yet.")
                return
            path, _ = QFileDialog.getSaveFileName(self, "Save preview PNG", "remote_sensing_preview.png", "PNG (*.png)")
            if not path:
                return
            try:
                self.preview_image.save(path)
                self._log("Saved preview PNG: " + path)
            except Exception as exc:
                QMessageBox.critical(self, "Remote Sensing", "PNG save failed:\n" + str(exc))

    return _RemoteSensingTab


try:
    RemoteSensingTab = _build_remote_sensing_class()
except Exception as exc:
    print("[Remote Sensing Plugin] Qt class build failed:", exc)
