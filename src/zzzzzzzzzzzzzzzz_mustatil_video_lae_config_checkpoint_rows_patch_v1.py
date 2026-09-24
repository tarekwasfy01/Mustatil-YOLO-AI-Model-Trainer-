#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mustatil AI Pipeline clean basic block UI patch v9

Drop this file into mustatil_plugins.

Adds/keeps the existing visual AI/YOLO Pipeline tab functionality while cleaning the UI and adding scientific filters:
- one single AI model block picker; no duplicate model buttons/dropdowns
- per-block model/checkpoint/config/FormLearner selection buttons in the visual blocks and right editor
- scientific filter blocks: morphometry/shape, size, local contrast/texture, Geo-NMS, cluster/context, linear alignment, and Scientific FormLearner
- functional Faster R-CNN, Mask R-CNN and U-Net pipeline block dispatch
- a combined SAM2 + FormLearner Filter block
- left-button background drag/pan for the canvas
- large central block canvas, compact surrounding controls
- right side rebuilt as one scrollable, grouped editor with process actions and log at the bottom
- code view can switch between editable pipeline data and executable Python dispatch code
- bottom Python console input for commands in the current pipeline context
- v9 layout fix: colored block frame and inline area include the Classes field fully

It patches the existing tab in-place; it does not create a second AI Pipeline tab.
"""
from __future__ import annotations

import importlib
import json
import math
import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

_PATCHED_CLASSES = set()
_PATCHED_TABS = set()
_PATCHED_VIEWS = set()
_CLEAN_RIGHT_TABS = set()
_CLEAN_TOOLBAR_TABS = set()
_SCAN_TIMER = None
_QTAB_PATCHED = False
_ORIG_QTAB_ADD = None
_ORIG_QTAB_INSERT = None

SCIENTIFIC_FILTER_TYPES = {
    "scientific_shape_filter",
    "scientific_size_filter",
    "scientific_texture_filter",
    "geo_nms_filter",
    "cluster_context_filter",
    "linear_alignment_filter",
    "scientific_formlearner_filter",
}
ENHANCED_TYPES = {"faster_rcnn", "mask_rcnn", "unet", "sam2_formlearner", "sam2_form_filter"} | SCIENTIFIC_FILTER_TYPES
TORCHVISION_TYPES = {"faster_rcnn", "faster", "faster_r_cnn", "mask_rcnn", "mask", "mask_r_cnn", "unet", "u_net", "unet_seg", "unet_segmentation"}
MODEL_BLOCK_TYPES = {"yolo", "owlv2", "grounding_dino", "lae_dino", "sam2", "sam2_formlearner", "faster_rcnn", "mask_rcnn", "unet", "formlearner"} | SCIENTIFIC_FILTER_TYPES
TYPE_ALIASES = {
    "faster": "faster_rcnn",
    "faster_rcnn": "faster_rcnn",
    "faster_r_cnn": "faster_rcnn",
    "faster-r-cnn": "faster_rcnn",
    "faster r-cnn": "faster_rcnn",
    "mask": "mask_rcnn",
    "mask_rcnn": "mask_rcnn",
    "mask_r_cnn": "mask_rcnn",
    "mask-r-cnn": "mask_rcnn",
    "mask r-cnn": "mask_rcnn",
    "unet": "unet",
    "u_net": "unet",
    "u-net": "unet",
    "unet_seg": "unet",
    "unet_segmentation": "unet",
    "sam2_formlearner": "sam2_formlearner",
    "sam2_form_filter": "sam2_formlearner",
    "sam2+formlearner": "sam2_formlearner",
    "sam2_form": "sam2_formlearner",
    "scientific_shape": "scientific_shape_filter",
    "shape_filter": "scientific_shape_filter",
    "morphometry": "scientific_shape_filter",
    "morphometric_filter": "scientific_shape_filter",
    "scientific_size": "scientific_size_filter",
    "size_filter": "scientific_size_filter",
    "area_filter": "scientific_size_filter",
    "scientific_texture": "scientific_texture_filter",
    "texture_filter": "scientific_texture_filter",
    "contrast_filter": "scientific_texture_filter",
    "geo_nms": "geo_nms_filter",
    "nms_filter": "geo_nms_filter",
    "duplicate_filter": "geo_nms_filter",
    "cluster_filter": "cluster_context_filter",
    "context_filter": "cluster_context_filter",
    "density_filter": "cluster_context_filter",
    "linear_filter": "linear_alignment_filter",
    "alignment_filter": "linear_alignment_filter",
    "linearity_filter": "linear_alignment_filter",
    "scientific_formlearner": "scientific_formlearner_filter",
    "formlearner_science": "scientific_formlearner_filter",
}
MODE_FOR_TYPE = {"faster_rcnn": "faster", "mask_rcnn": "mask", "unet": "unet"}

AI_CHOICES: List[Tuple[str, str]] = [
    ("YOLO", "yolo"),
    ("Google OWLv2", "owlv2"),
    ("Grounding DINO", "grounding_dino"),
    ("LAE-DINO", "lae_dino"),
    ("SAM2 from boxes", "sam2"),
    ("SAM2 + FormLearner Filter", "sam2_formlearner"),
    ("Scientific shape / morphometry filter", "scientific_shape_filter"),
    ("Scientific size / scale filter", "scientific_size_filter"),
    ("Scientific texture / contrast filter", "scientific_texture_filter"),
    ("Geo-NMS duplicate filter", "geo_nms_filter"),
    ("Cluster / context density filter", "cluster_context_filter"),
    ("Linear alignment filter", "linear_alignment_filter"),
    ("Scientific FormLearner filter", "scientific_formlearner_filter"),
    ("Faster R-CNN", "faster_rcnn"),
    ("Mask R-CNN", "mask_rcnn"),
    ("U-Net Segmentation", "unet"),
]

DEFAULT_NAMES = {
    "yolo": "YOLO block",
    "owlv2": "OWLv2 block",
    "grounding_dino": "Grounding DINO block",
    "lae_dino": "LAE-DINO block",
    "sam2": "SAM2 segment block",
    "sam2_formlearner": "SAM2 + FormLearner filter",
    "faster_rcnn": "Faster R-CNN block",
    "mask_rcnn": "Mask R-CNN block",
    "unet": "U-Net segmentation block",
    "scientific_shape_filter": "Scientific shape filter",
    "scientific_size_filter": "Scientific size filter",
    "scientific_texture_filter": "Scientific texture/contrast filter",
    "geo_nms_filter": "Geo-NMS duplicate filter",
    "cluster_context_filter": "Cluster/context density filter",
    "linear_alignment_filter": "Linear alignment filter",
    "scientific_formlearner_filter": "Scientific FormLearner filter",
    "formlearner": "FormLearner block",
    "rule": "IF / Logic block",
}



BASIC_FILTER_PARAM_SPECS: Dict[str, List[Tuple[str, float, float, float, int]]] = {
    # key, default, min, max, decimals -- deliberately basic so the block UI stays readable.
    "scientific_shape_filter": [("area_min", 16.0, 0.0, 1e9, 0), ("aspect_max", 12.0, 1.0, 100.0, 2), ("fill_min", 0.05, 0.0, 1.0, 3)],
    "scientific_size_filter": [("area_min", 16.0, 0.0, 1e12, 0), ("area_max", 999999999.0, 1.0, 1e12, 0)],
    "scientific_texture_filter": [("contrast_min", 0.03, 0.0, 1.0, 3), ("std_min", 2.0, 0.0, 255.0, 2)],
    "geo_nms_filter": [("iou", 0.60, 0.01, 0.99, 2)],
    "cluster_context_filter": [("radius", 250.0, 1.0, 1000000.0, 0), ("neighbors_min", 1.0, 0.0, 1000000.0, 0)],
    "linear_alignment_filter": [("radius", 450.0, 1.0, 1000000.0, 0), ("line_score_min", 0.55, 0.0, 1.0, 3)],
    "scientific_formlearner_filter": [("mix_score_min", 0.50, 0.0, 1.0, 3), ("weight_form", 0.55, 0.0, 1.0, 2)],
}

ALGORITHM_ONLY_FILTER_TYPES = {
    "scientific_shape_filter",
    "scientific_size_filter",
    "scientific_texture_filter",
    "geo_nms_filter",
    "cluster_context_filter",
    "linear_alignment_filter",
}

RIGHT_MODELS_WITH_COMPANION = {"sam2", "sam2_formlearner", "lae_dino"}
RIGHT_MODEL_TYPES = {"yolo", "owlv2", "grounding_dino", "lae_dino", "sam2", "sam2_formlearner", "faster_rcnn", "mask_rcnn", "unet"}
AI_BUTTON_KEYWORDS = (
    "+ yolo", "+ owl", "+ google owl", "+ grounding", "+ lae", "+ sam", "+ faster", "+ mask", "+ u-net", "+ unet",
)
KEEP_BUTTON_KEYWORDS = ("formlearner", "form learner", "rule", "logic", "if")


def _log(msg: str) -> None:
    try:
        print("[Mustatil AI Pipeline Clean Basic v8] " + str(msg))
    except Exception:
        pass


def _ws_log(ws: Any, msg: str) -> None:
    try:
        ws.log("[AI Pipeline Patch] " + str(msg))
    except Exception:
        _log(msg)


def _norm_type(t: Any) -> str:
    raw = str(t or "").strip().lower()
    norm = raw.replace(" ", "_").replace("-", "_").replace("+", "_")
    return TYPE_ALIASES.get(raw) or TYPE_ALIASES.get(norm) or norm


def _get_var(v: Any, default: Any = "") -> Any:
    try:
        if hasattr(v, "get"):
            return v.get()
    except Exception:
        pass
    return default


def _workspace_from_widget(widget: Any) -> Optional[Any]:
    cur = widget
    for _ in range(140):
        if cur is None:
            break
        try:
            if hasattr(cur, "tabs") and (hasattr(cur, "dets") or hasattr(cur, "satellite_detections") or hasattr(cur, "image")):
                return cur
        except Exception:
            pass
        try:
            cur = cur.parentWidget()
        except Exception:
            try:
                cur = cur.parent()
            except Exception:
                break
    return None


def _all_workspaces() -> List[Any]:
    out: List[Any] = []
    seen = set()
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            return out
        for w in app.topLevelWidgets():
            ws = _workspace_from_widget(w)
            if ws is not None and id(ws) not in seen:
                out.append(ws)
                seen.add(id(ws))
    except Exception:
        pass
    return out


def _schedule_scan(delay_ms: int = 150) -> None:
    global _SCAN_TIMER
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
        if QApplication.instance() is None:
            return
        if _SCAN_TIMER is None:
            _SCAN_TIMER = QTimer()
            _SCAN_TIMER.setSingleShot(True)
            _SCAN_TIMER.timeout.connect(_scan_and_patch)
        _SCAN_TIMER.start(max(20, int(delay_ms)))
    except Exception:
        try:
            _scan_and_patch()
        except Exception:
            pass


def _patch_qtabwidget() -> None:
    global _QTAB_PATCHED, _ORIG_QTAB_ADD, _ORIG_QTAB_INSERT
    if _QTAB_PATCHED:
        return
    try:
        from PySide6.QtWidgets import QTabWidget
    except Exception:
        return
    _ORIG_QTAB_ADD = QTabWidget.addTab
    _ORIG_QTAB_INSERT = QTabWidget.insertTab

    def add_tab_patched(self, *args, **kwargs):
        ret = _ORIG_QTAB_ADD(self, *args, **kwargs)
        _schedule_scan(80)
        return ret

    def insert_tab_patched(self, *args, **kwargs):
        ret = _ORIG_QTAB_INSERT(self, *args, **kwargs)
        _schedule_scan(80)
        return ret

    QTabWidget.addTab = add_tab_patched
    QTabWidget.insertTab = insert_tab_patched
    _QTAB_PATCHED = True


# -----------------------------------------------------------------------------
# TorchVision / U-Net bridge, reusing the existing model-tabs plugin
# -----------------------------------------------------------------------------

def _import_torchvision_model_plugin():
    candidates = [
        "zzzz_mustatil_torchvision_frcnn_maskrcnn_unet_sam2_tabs",
        "mustatil_plugin_zzzz_mustatil_torchvision_frcnn_maskrcnn_unet_sam2_tabs",
        "zzzz_mustatil_torchvision_frcnn_maskrcnn_unet_tabs",
        "mustatil_plugin_zzzz_mustatil_torchvision_frcnn_maskrcnn_unet_tabs",
    ]
    for name in candidates:
        try:
            mod = sys.modules.get(name) or importlib.import_module(name)
            if hasattr(mod, "_model_infer_pil"):
                return mod
        except Exception:
            pass
    raise RuntimeError(
        "TorchVision model helper not found. Keep zzzz_mustatil_torchvision_frcnn_maskrcnn_unet_sam2_tabs.py active in mustatil_plugins."
    )


def _parse_class_filter(text: Any) -> set:
    return {p.strip().lower() for p in str(text or "").replace(";", ",").split(",") if p.strip()}


def _class_count_from_workspace(ws: Any) -> int:
    # torchvision detection models include background class.
    try:
        classes = list(getattr(getattr(ws, "project_state", None), "classes", []) or [])
        if classes:
            return max(2, len(classes) + 1)
    except Exception:
        pass
    try:
        classes = list(getattr(ws, "classes", []) or [])
        if classes:
            return max(2, len(classes) + 1)
    except Exception:
        pass
    return 2


# -----------------------------------------------------------------------------
# Model/config selection helpers used by the clean right panel and inline blocks
# -----------------------------------------------------------------------------

def _widget_text(widget: Any) -> str:
    try:
        if hasattr(widget, "_edit"):
            return str(widget._edit.text() or "")
        if hasattr(widget, "text"):
            return str(widget.text() or "")
        if hasattr(widget, "currentText"):
            return str(widget.currentText() or "")
    except Exception:
        pass
    return ""


def _set_widget_text(widget: Any, text: Any) -> None:
    try:
        val = str(text or "")
        if hasattr(widget, "_edit"):
            widget._edit.setText(val)
        elif hasattr(widget, "setText"):
            widget.setText(val)
        elif hasattr(widget, "setCurrentText"):
            widget.setCurrentText(val)
    except Exception:
        pass


def _current_block_for_tab(tab: Any) -> Optional[Any]:
    try:
        if hasattr(tab, "_current_block"):
            return tab._current_block()
    except Exception:
        pass
    try:
        bid = getattr(tab, "selected_block_id", "")
        for b in list(getattr(tab, "blocks", []) or []):
            if getattr(b, "id", None) == bid:
                return b
    except Exception:
        pass
    return None


def _save_and_refresh_block_editor(tab: Any, b: Optional[Any] = None) -> None:
    try:
        if hasattr(tab, "_save_editor_to_block"):
            tab._save_editor_to_block()
    except Exception:
        pass
    try:
        if b is None:
            b = _current_block_for_tab(tab)
        if b is not None and hasattr(tab, "_load_block_to_editor"):
            tab._load_block_to_editor(b)
    except Exception:
        pass
    try:
        if hasattr(tab, "refresh_code_from_blocks"):
            tab.refresh_code_from_blocks()
    except Exception:
        pass
    try:
        if b is not None and getattr(b, "id", None) in getattr(tab, "block_items", {}):
            item = tab.block_items[b.id]
            item.refresh_embedded_controls()
            item.update()
    except Exception:
        pass


def _block_type(tab: Any, block: Optional[Any] = None) -> str:
    if block is None:
        block = _current_block_for_tab(tab)
    return _norm_type(getattr(block, "type", "") if block is not None else "")


def _file_filter_for_block(typ: str, slot: str) -> str:
    typ = _norm_type(typ)
    if typ in SCIENTIFIC_FILTER_TYPES:
        if slot in {"formlearner", "companion"}:
            return "FormLearner JSON (*.json);;Scientific preset JSON (*.json);;All files (*)"
        if slot == "sam2_config":
            return "Scientific filter preset JSON (*.json);;YAML preset (*.yaml *.yml);;All files (*)"
        return "Scientific filter preset JSON (*.json);;All files (*)"
    if slot == "formlearner":
        return "FormLearner JSON (*.json);;All files (*)"
    if slot == "sam2_config":
        return "SAM2 config YAML (*.yaml *.yml);;Python config (*.py);;All files (*)"
    if slot == "companion":
        if typ == "lae_dino":
            return "LAE-DINO companion (*.py *.yaml *.yml *.pth *.pt *.json);;All files (*)"
        if typ == "sam2":
            return "SAM2 config YAML (*.yaml *.yml);;Python config (*.py);;All files (*)"
        if typ == "sam2_formlearner":
            return "FormLearner JSON (*.json);;All files (*)"
        if typ in {"faster_rcnn", "mask_rcnn", "unet"}:
            return "Checkpoint / weights (*.pt *.pth *.ckpt *.safetensors);;All files (*)"
        return "Companion/config (*.json *.yaml *.yml *.py *.pt *.pth);;All files (*)"
    if typ == "yolo":
        return "YOLO / exported model (*.pt *.onnx *.engine *.xml *.tflite);;All files (*)"
    if typ in {"faster_rcnn", "mask_rcnn", "unet"}:
        return "PyTorch checkpoint (*.pt *.pth *.ckpt *.safetensors);;All files (*)"
    if typ == "sam2" or typ == "sam2_formlearner":
        return "SAM2 checkpoint (*.pt *.pth *.ckpt *.safetensors);;All files (*)"
    if typ == "lae_dino":
        return "LAE-DINO config/checkpoint (*.py *.yaml *.yml *.pth *.pt *.json);;All files (*)"
    if typ in {"owlv2", "grounding_dino"}:
        return "Local HF model/config/checkpoint (*.json *.bin *.safetensors *.pt *.pth *.yaml *.yml);;All files (*)"
    return "Model/checkpoint/config (*.pt *.pth *.onnx *.engine *.xml *.json *.yaml *.yml *.py *.safetensors);;All files (*)"


def _start_dir_from_widget(widget: Any) -> str:
    try:
        raw = _widget_text(widget).strip().strip('"')
        if raw:
            pp = Path(raw).expanduser()
            if pp.exists():
                return str(pp if pp.is_dir() else pp.parent)
            if pp.parent.exists():
                return str(pp.parent)
    except Exception:
        pass
    return ""


def _select_file_for_current_block(tab: Any, slot: str = "primary") -> None:
    try:
        from PySide6.QtWidgets import QFileDialog
    except Exception:
        return
    b = _current_block_for_tab(tab)
    if b is None:
        try:
            tab._log("Select a block first, then choose its model/config file.")
        except Exception:
            pass
        return
    typ = _block_type(tab, b)
    title_map = {
        "primary": "Choose model/checkpoint for selected block",
        "companion": "Choose companion/config for selected block",
        "sam2_config": "Choose SAM2 config YAML for selected block",
        "formlearner": "Choose FormLearner JSON for selected block",
    }
    target_widget = getattr(tab, "model_path", None)
    if slot in {"companion", "formlearner"}:
        target_widget = getattr(tab, "form_model", None)
    elif slot == "sam2_config":
        target_widget = getattr(tab, "classes_filter", None)
    start = _start_dir_from_widget(target_widget)
    path, _ = QFileDialog.getOpenFileName(tab, title_map.get(slot, "Choose file"), start, _file_filter_for_block(typ, slot))
    if not path:
        return
    if slot == "primary":
        _set_widget_text(getattr(tab, "model_path", None), path)
        try:
            b.model_path = path
        except Exception:
            pass
    elif slot == "sam2_config":
        _set_sam2_config_path(tab, path, b)
    else:
        _set_widget_text(getattr(tab, "form_model", None), path)
        try:
            b.formlearner_model = path
        except Exception:
            pass
    _save_and_refresh_block_editor(tab, b)
    try:
        tab._log(f"{getattr(b, 'name', 'Block')}: selected {slot} file: {path}")
    except Exception:
        pass


def _select_folder_for_current_block(tab: Any) -> None:
    try:
        from PySide6.QtWidgets import QFileDialog
    except Exception:
        return
    b = _current_block_for_tab(tab)
    if b is None:
        return
    start = _start_dir_from_widget(getattr(tab, "model_path", None))
    path = QFileDialog.getExistingDirectory(tab, "Choose local model folder", start)
    if not path:
        return
    _set_widget_text(getattr(tab, "model_path", None), path)
    try:
        b.model_path = path
    except Exception:
        pass
    _save_and_refresh_block_editor(tab, b)


def _set_current_block_default_model(tab: Any) -> None:
    b = _current_block_for_tab(tab)
    if b is None:
        return
    typ = _block_type(tab, b)
    default = ""
    if typ == "owlv2":
        default = "google/owlv2-base-patch16-ensemble"
    elif typ == "grounding_dino":
        default = "IDEA-Research/grounding-dino-base"
    elif typ == "lae_dino":
        for attr in ("mustatil_lae_existing_v9_config", "mustatil_lae_existing_v8_config", "mustatil_lae_config"):
            default = str(getattr(getattr(tab, "ws", None), attr, "") or "").strip()
            if default:
                break
    elif typ == "yolo":
        try:
            obj = getattr(getattr(tab, "ws", None), "trainmodel", None)
            if hasattr(obj, "get"):
                default = str(obj.get() or "").strip()
        except Exception:
            default = ""
    if not default:
        try:
            tab._log(f"No automatic/default model was found for block type {typ}. Use the file/folder button or paste a model id/path.")
        except Exception:
            pass
        return
    _set_widget_text(getattr(tab, "model_path", None), default)
    try:
        b.model_path = default
    except Exception:
        pass
    _save_and_refresh_block_editor(tab, b)


def _parse_key_path(text: Any, key: str) -> str:
    key = str(key).strip().lower()
    for raw in str(text or "").replace(";", ",").split(","):
        part = raw.strip()
        if not part:
            continue
        if part.lower().startswith(key + "="):
            return part.split("=", 1)[1].strip().strip('"')
    return ""


def _replace_key_path(text: Any, key: str, value: str) -> str:
    key = str(key).strip()
    parts = []
    for raw in str(text or "").replace(";", ",").split(","):
        part = raw.strip()
        if not part:
            continue
        if part.lower().startswith(key.lower() + "="):
            continue
        parts.append(part)
    if value:
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def _set_sam2_config_path(tab: Any, path: str, b: Optional[Any] = None) -> None:
    if b is None:
        b = _current_block_for_tab(tab)
    typ = _block_type(tab, b)
    if typ == "sam2":
        # Existing SAM2 runner expects the config in formlearner_model/FormLearner field.
        _set_widget_text(getattr(tab, "form_model", None), path)
        try:
            b.formlearner_model = path
        except Exception:
            pass
    else:
        # Combined SAM2+FormLearner needs FormLearner JSON separately, so store the SAM2 config in the options field.
        cur = _widget_text(getattr(tab, "classes_filter", None))
        new = _replace_key_path(cur, "config", path)
        _set_widget_text(getattr(tab, "classes_filter", None), new)
        try:
            b.classes_filter = new
            b.sam2_config = path
        except Exception:
            pass


def _extract_sam2_config_for_block(b: Any) -> str:
    for attr in ("sam2_config", "sam_config", "config_path"):
        try:
            val = str(getattr(b, attr, "") or "").strip().strip('"')
            if val:
                return val
        except Exception:
            pass
    val = _parse_key_path(getattr(b, "classes_filter", ""), "config")
    if val:
        return val
    return ""


def _add_model_button_grid(parent_layout: Any, tab: Any) -> None:
    try:
        from PySide6.QtWidgets import QWidget, QGridLayout, QPushButton, QSizePolicy
    except Exception:
        return
    try:
        box = QWidget()
        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        grid = QGridLayout(box)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        def btn(text: str, cb: Any, tip: str = ""):
            b = QPushButton(text)
            b.setMinimumHeight(30)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            if tip:
                b.setToolTip(tip)
            try:
                b.clicked.connect(cb)
            except Exception:
                pass
            return b

        grid.addWidget(btn("Select model/checkpoint…", lambda: _select_file_for_current_block(tab, "primary"), "Sets the main model path for the selected block."), 0, 0)
        grid.addWidget(btn("Select model folder…", lambda: _select_folder_for_current_block(tab), "Use for local HuggingFace folders or exported model folders."), 0, 1)
        grid.addWidget(btn("Select companion/config…", lambda: _select_file_for_current_block(tab, "companion"), "LAE checkpoint/config, pure SAM2 config, or generic companion file."), 1, 0)
        grid.addWidget(btn("Select FormLearner JSON…", lambda: _select_file_for_current_block(tab, "formlearner"), "Sets the FormLearner model JSON for FormLearner and SAM2+FormLearner blocks."), 1, 1)
        grid.addWidget(btn("Select SAM2 config YAML…", lambda: _select_file_for_current_block(tab, "sam2_config"), "For SAM2+FormLearner this is stored as config=... in the options field."), 2, 0)
        grid.addWidget(btn("Use default/workspace model", lambda: _set_current_block_default_model(tab), "OWLv2/Grounding DINO defaults or current workspace model when available."), 2, 1)
        parent_layout.addWidget(box)
    except Exception:
        pass


def _collect_model_choices_for_block(tab: Any, block: Any, slot: str = "primary") -> List[Tuple[str, str]]:
    choices: List[Tuple[str, str]] = [("choose / paste custom value", "")]
    seen: Set[str] = set()
    typ = _norm_type(getattr(block, "type", "") if block is not None else "")
    ws = getattr(tab, "ws", None)

    def add(raw: Any, label: Optional[str] = None) -> None:
        val = str(raw or "").strip().strip('"')
        if not val or val in seen:
            return
        seen.add(val)
        shown = label or (Path(val).name if ("/" in val or "\\" in val) else val)
        choices.append((shown, val))

    if slot == "primary":
        if typ == "owlv2":
            add("google/owlv2-base-patch16-ensemble", "default: OWLv2 base")
            add("google/owlv2-large-patch14-ensemble", "OWLv2 large")
        elif typ == "grounding_dino":
            add("IDEA-Research/grounding-dino-base", "default: Grounding DINO base")
            add("IDEA-Research/grounding-dino-tiny", "Grounding DINO tiny")
        elif typ == "yolo":
            for v in getattr(ws, "models", []) or []:
                try:
                    add(v.get())
                except Exception:
                    pass
            for attr in ("trainmodel", "auto_annotate_yolo_model"):
                try:
                    obj = getattr(ws, attr, None)
                    if hasattr(obj, "get"):
                        add(obj.get(), attr)
                except Exception:
                    pass
        elif typ == "lae_dino":
            for attr in ("mustatil_lae_existing_v9_config", "mustatil_lae_existing_v8_config", "mustatil_lae_config"):
                add(getattr(ws, attr, ""), "LAE config")
        elif typ in {"sam2", "sam2_formlearner"}:
            for attr in ("sam2_checkpoint", "mustatil_sam2_checkpoint", "sam_checkpoint"):
                add(getattr(ws, attr, ""), "SAM2 checkpoint")
        elif typ in {"faster_rcnn", "mask_rcnn", "unet"}:
            for attr in ("torchvision_model_path", "rcnn_model_path", "maskrcnn_model_path", "unet_model_path", "trainmodel"):
                try:
                    obj = getattr(ws, attr, "")
                    add(obj.get() if hasattr(obj, "get") else obj, attr)
                except Exception:
                    pass
        elif typ in SCIENTIFIC_FILTER_TYPES:
            for attr in ("scientific_filter_preset", "scientific_filter_json", "formtrainer_scientific_preset"):
                try:
                    obj = getattr(ws, attr, "")
                    add(obj.get() if hasattr(obj, "get") else obj, "scientific preset")
                except Exception:
                    pass
    else:
        if typ == "lae_dino":
            for attr in ("mustatil_lae_existing_v9_weights", "mustatil_lae_existing_v8_weights", "mustatil_lae_weights", "mustatil_lae_checkpoint"):
                add(getattr(ws, attr, ""), "LAE checkpoint")
        elif typ == "sam2":
            for attr in ("sam2_config", "mustatil_sam2_config", "sam_config"):
                add(getattr(ws, attr, ""), "SAM2 config")
        elif typ in {"sam2_formlearner", "formlearner"} or typ in SCIENTIFIC_FILTER_TYPES:
            for attr in ("fl_model_path", "form_model_path", "formlearner_model", "formtrainer_model", "scientific_formlearner_json"):
                try:
                    obj = getattr(ws, attr, "")
                    add(obj.get() if hasattr(obj, "get") else obj, "FormLearner JSON")
                except Exception:
                    pass

    search_dirs: List[Path] = []
    try:
        app_dir = Path(getattr(ws, "MUSTATIL_APP_DIR", Path.cwd()))
        search_dirs.extend([app_dir, app_dir / "weights", app_dir / "models", app_dir / "mustatil_model_runtimes"])
        here = Path(__file__).resolve().parent
        search_dirs.extend([here, here / "weights", here / "models", here / "mustatil_model_runtimes"])
    except Exception:
        pass
    try:
        project = str(getattr(ws, "project", None).get() or "").strip()
        if project:
            pp = Path(project)
            search_dirs.extend([pp, pp / "weights", pp / "models", pp / "lae_dino_dataset" / "work_dirs" / "lae_dino_mustatil"])
    except Exception:
        pass

    if slot == "primary":
        if typ == "yolo":
            patterns = ("*.pt", "*.onnx", "*.engine", "*.xml")
        elif typ in {"faster_rcnn", "mask_rcnn", "unet", "sam2", "sam2_formlearner"}:
            patterns = ("*.pt", "*.pth", "*.ckpt", "*.safetensors")
        elif typ in SCIENTIFIC_FILTER_TYPES:
            patterns = ("*.json", "*.yaml", "*.yml")
        elif typ == "lae_dino":
            patterns = ("*.py", "*.yaml", "*.yml", "*.pth", "*.pt", "*.safetensors")
        else:
            patterns = ("*.pt", "*.pth", "*.bin", "*.safetensors", "*.json", "*.yaml", "*.yml")
    else:
        if typ in {"sam2", "lae_dino"}:
            patterns = ("*.yaml", "*.yml", "*.py", "*.json", "*.pth", "*.pt")
        else:
            patterns = ("*.json", "*.yaml", "*.yml", "*.py", "*.pth", "*.pt")
    for folder in search_dirs:
        try:
            if not folder.exists() or not folder.is_dir():
                continue
            for pat in patterns:
                for file in sorted(folder.glob(pat))[:80]:
                    add(str(file))
        except Exception:
            continue
    return choices


def _tile_positions(w: int, h: int, tile: int, overlap: int) -> List[Tuple[int, int]]:
    tile = max(64, int(tile or 1024))
    overlap = max(0, min(int(overlap or 0), tile - 1))
    step = max(1, tile - overlap)
    out: List[Tuple[int, int]] = []
    y = 0
    while y < h:
        x = 0
        while x < w:
            out.append((x, y))
            if x + tile >= w:
                break
            x += step
        if y + tile >= h:
            break
        y += step
    return out


def _box_iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ax1, ay1, ax2, ay2 = float(a["x1"]), float(a["y1"]), float(a["x2"]), float(a["y2"])
    bx1, by1, bx2, by2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = aa + bb - inter
    return 0.0 if den <= 0 else inter / den


def _nms_records(records: List[Dict[str, Any]], thr: float = 0.80) -> List[Dict[str, Any]]:
    def score(r: Dict[str, Any]) -> float:
        try:
            return float(r.get("conf", r.get("confidence", r.get("score", 0.0))) or 0.0)
        except Exception:
            return 0.0

    def cls(r: Dict[str, Any]) -> int:
        try:
            return int(r.get("class_id", r.get("cls", 0)) or 0)
        except Exception:
            return 0

    kept: List[Dict[str, Any]] = []
    for r in sorted(records, key=score, reverse=True):
        try:
            if any(cls(k) == cls(r) and str(k.get("parent_id", "")) == str(r.get("parent_id", "")) and _box_iou(k, r) >= thr for k in kept):
                continue
        except Exception:
            pass
        kept.append(r)
    return kept


def _make_pipeline_record(tab: Any, block: Any, parent: Dict[str, Any], r: Dict[str, Any], off_x: float, off_y: float, image_path: Path) -> Optional[Dict[str, Any]]:
    try:
        full_w = float(getattr(tab, "_mustatil_current_full_width", 10**9) or 10**9)
        full_h = float(getattr(tab, "_mustatil_current_full_height", 10**9) or 10**9)
        x1 = max(0.0, min(full_w, off_x + float(r["x1"])))
        y1 = max(0.0, min(full_h, off_y + float(r["y1"])))
        x2 = max(0.0, min(full_w, off_x + float(r["x2"])))
        y2 = max(0.0, min(full_h, off_y + float(r["y2"])))
        if x2 <= x1 or y2 <= y1:
            return None
        cid = int(r.get("class_id", r.get("cls", 0)) or 0)
        label = str(r.get("label", r.get("class_name", f"class_{cid}")))
        conf = float(r.get("conf", r.get("confidence", r.get("score", 0.0))) or 0.0)
        rec = {
            "id": "r_" + uuid.uuid4().hex[:10],
            "block_id": block.id,
            "block_name": block.name,
            "block_type": block.type,
            "parent_id": parent.get("id"),
            "parent_block_id": parent.get("block_id"),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "conf": conf,
            "class_id": cid,
            "label": label,
            "model": str(r.get("model", getattr(block, "model_path", "")) or block.type),
            "source_image": str(image_path),
        }
        for k in ("area_px", "mask_area_px", "mask_fill_ratio", "shape_ratio", "extent"):
            if k in r:
                rec[k] = r[k]
        return rec
    except Exception:
        return None


def _run_torchvision_ai_block(tab: Any, b: Any, full: Any, image_path: Path) -> List[Dict[str, Any]]:
    mod = _import_torchvision_model_plugin()
    typ = _norm_type(getattr(b, "type", ""))
    mode = MODE_FOR_TYPE.get(typ)
    if not mode:
        raise RuntimeError(f"Unsupported TorchVision/U-Net block type: {getattr(b, 'type', '')}")
    parents = tab._input_records(b, full) if hasattr(tab, "_input_records") else []
    class_filter = _parse_class_filter(getattr(b, "classes_filter", ""))
    out: List[Dict[str, Any]] = []
    device = str(getattr(b, "device", "cuda") or "cuda")
    conf = float(getattr(b, "confidence", 0.25) or 0.25)
    tile_size = int(getattr(b, "tile_size", getattr(b, "imgsz", 1024)) or 1024)
    overlap = int(getattr(b, "overlap", 128) or 128)
    crop_padding = int(getattr(b, "crop_padding", 0) or 0)
    model_path = str(getattr(b, "model_path", "") or "").strip().strip('"')
    num_classes = _class_count_from_workspace(getattr(tab, "ws", None))
    settings: Dict[str, Any] = {
        "title": str(getattr(b, "name", mode)),
        "device": device,
        "model_path": model_path,
        "num_classes": num_classes if mode in {"faster", "mask"} else 1,
        "conf": conf,
        "threshold": conf,
        "mask_threshold": conf,
        "min_area": 64,
        "base": 32,
        "nms_iou": 0.80,
        "prompt_mode": "full_tile_only",
        "max_prompts": 256,
        "sam_class_id": 0,
    }
    if mode == "unet" and not model_path:
        try:
            tab._log(f"{b.name}: U-Net block needs a trained checkpoint in Model path.")
        except Exception:
            pass
        return []
    try:
        tab._log(f"{b.name}: {mode} started; parents={len(parents)}, conf={conf}, device={device}, tile={tile_size}, overlap={overlap}")
    except Exception:
        pass
    for parent in parents:
        try:
            crop, offset = tab._crop_parent(full, parent, crop_padding) if hasattr(tab, "_crop_parent") else (full, (0.0, 0.0))
            crop = crop.convert("RGB")
            cw, ch = int(crop.width), int(crop.height)
            positions = _tile_positions(cw, ch, tile_size, overlap) if (cw > tile_size or ch > tile_size) else [(0, 0)]
            for tx, ty in positions:
                tw = min(tile_size, cw - tx)
                th = min(tile_size, ch - ty)
                if tw <= 0 or th <= 0:
                    continue
                tile_img = crop.crop((tx, ty, tx + tw, ty + th)).convert("RGB") if positions != [(0, 0)] else crop
                local = mod._model_infer_pil(getattr(tab, "ws", None), mode, tile_img, dict(settings))
                for r in list(local or []):
                    try:
                        cid = int(r.get("class_id", r.get("cls", 0)) or 0)
                        label = str(r.get("label", r.get("class_name", cid))).lower()
                        if class_filter and str(cid).lower() not in class_filter and label not in class_filter:
                            continue
                    except Exception:
                        pass
                    rec = _make_pipeline_record(tab, b, parent, r, float(offset[0]) + tx, float(offset[1]) + ty, image_path)
                    if rec is not None:
                        out.append(rec)
        except Exception as exc:
            try:
                tab._log(f"{b.name}: parent failed: {exc}")
            except Exception:
                pass
            traceback.print_exc()
    out = _nms_records(out, 0.80)
    try:
        tab._log(f"{b.name}: {mode} finished: {len(out)} result(s)")
    except Exception:
        pass
    return out


# -----------------------------------------------------------------------------
# SAM2 + FormLearner filter
# -----------------------------------------------------------------------------

def _shape_metrics_from_record(r: Dict[str, Any]) -> Dict[str, float]:
    w = max(1.0, abs(float(r.get("x2", 0)) - float(r.get("x1", 0))))
    h = max(1.0, abs(float(r.get("y2", 0)) - float(r.get("y1", 0))))
    bbox_area = max(1.0, w * h)
    mask_area = float(r.get("mask_area_px", r.get("area_px", bbox_area)) or bbox_area)
    fill = max(0.0, min(1.0, mask_area / bbox_area))
    ratio = max(w / h, h / w)
    # Conservative generic regularity score. It is metadata, not a replacement for a trained FormLearner JSON.
    ratio_score = math.exp(-abs(math.log(max(1.0, ratio) / 2.2))) if ratio > 0 else 0.0
    fill_score = 1.0 - min(1.0, abs(fill - 0.55) / 0.55)
    shape_score = max(0.0, min(1.0, 0.55 * ratio_score + 0.45 * fill_score))
    return {"mask_fill_ratio": fill, "shape_ratio": ratio, "extent": fill, "shape_score": shape_score}


def _load_formlearner_model(model_path: str):
    path = Path(str(model_path or "").strip().strip('"')).expanduser()
    if not path.is_file():
        return None, str(path) if str(path) else ""
    try:
        import mustatil_legacy_backend as backend
        return backend.SimpleFormLearner.load(str(path)), str(path)
    except Exception as exc:
        raise RuntimeError(f"Could not load FormLearner model: {exc}")


def _run_sam2_formlearner_block(tab: Any, b: Any, full: Any, image_path: Path) -> List[Dict[str, Any]]:
    parents = tab._input_records(b, full) if hasattr(tab, "_input_records") else []
    if not parents or str(getattr(b, "input_ref", "original")) == "original":
        try:
            tab._log(f"{b.name}: connect an upstream detector block first. This block is a filter for boxes, not a whole-image detector.")
        except Exception:
            pass
        return []

    # Step 1: SAM2 refines every upstream parent box into a tighter mask bbox.
    # For the combined block, model_path is the SAM2 checkpoint, formlearner_model is the
    # FormLearner JSON, and the SAM2 config is stored as config=... in classes_filter.
    # The original SAM2 runner expects the config in formlearner_model, so use a shallow
    # proxy object for the SAM2 stage instead of overwriting the user's FormLearner JSON.
    try:
        import copy as _copy
        sam_proxy = _copy.copy(b)
        sam_proxy.type = "sam2"
        sam_cfg = _extract_sam2_config_for_block(b)
        if sam_cfg:
            sam_proxy.formlearner_model = sam_cfg
        sam_records = tab._run_sam2_block(sam_proxy, full, image_path) if hasattr(tab, "_run_sam2_block") else []
    except Exception as exc:
        try:
            tab._log(f"{b.name}: SAM2 step failed; using upstream boxes for FormLearner filter. Reason: {exc}")
        except Exception:
            pass
        sam_records = []
    if not sam_records:
        sam_records = []
        for parent in parents:
            rec = dict(parent)
            rec.update({
                "id": "r_" + uuid.uuid4().hex[:10],
                "block_id": getattr(b, "id", ""),
                "block_name": getattr(b, "name", "SAM2 + FormLearner filter"),
                "block_type": getattr(b, "type", "sam2_formlearner"),
                "parent_id": parent.get("id"),
                "sam2_status": "fallback_parent_box",
                "source_image": str(image_path),
            })
            sam_records.append(rec)

    # Step 2: FormLearner scores the SAM2-refined boxes. If no JSON is supplied, keep SAM2 output and expose shape metrics.
    # FormLearner JSON must not be confused with the SAM2 checkpoint.
    model_path = str(getattr(b, "formlearner_model", "") or getattr(b, "form_model", "") or "")
    try:
        form_model, resolved_form_path = _load_formlearner_model(model_path)
    except Exception as exc:
        try:
            tab._log(f"{b.name}: FormLearner model could not be loaded; keeping SAM2 objects with shape metrics only. Reason: {exc}")
        except Exception:
            pass
        form_model, resolved_form_path = None, str(model_path or "")
    threshold = float(getattr(b, "form_threshold", 0.50) or 0.50)
    out: List[Dict[str, Any]] = []
    rejected = 0
    try:
        import mustatil_legacy_backend as backend
    except Exception:
        backend = None

    for rec in sam_records:
        r = dict(rec)
        metrics = _shape_metrics_from_record(r)
        r.update(metrics)
        score = None
        if form_model is not None and backend is not None:
            try:
                bbox = (float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]))
                score = float(form_model.predict(backend.crop_features(full, bbox)))
            except Exception as exc:
                try:
                    tab._log(f"{b.name}: FormLearner scoring failed for one SAM2 object: {exc}")
                except Exception:
                    pass
                score = None
        if score is None:
            # Metadata-only mode: useful SAM2 refinement without hiding results unexpectedly.
            score = float(metrics.get("shape_score", 0.0))
            keep = form_model is None or score >= threshold
            r["form_status"] = "sam2_only_no_form_model" if form_model is None else ("positive" if keep else "false_positive")
        else:
            keep = score >= threshold
            r["form_status"] = "positive" if keep else "false_positive"
        r.update({
            "id": "r_" + uuid.uuid4().hex[:10],
            "block_id": getattr(b, "id", ""),
            "block_name": getattr(b, "name", "SAM2 + FormLearner filter"),
            "block_type": getattr(b, "type", "sam2_formlearner"),
            "parent_id": rec.get("parent_id") or rec.get("id"),
            "form_score": float(score),
            "form_threshold": threshold,
            "form_model": resolved_form_path,
            "sam2_form_filter": "keep" if keep else "reject",
            "label": str(r.get("label", "object")) if keep else "false_positive",
            "class_id": int(r.get("class_id", 0) or 0) if keep else 1,
        })
        if keep:
            out.append(r)
        else:
            rejected += 1
    try:
        if form_model is None:
            tab._log(f"{b.name}: SAM2 refined {len(sam_records)} object(s). No FormLearner JSON was set, so objects were kept with shape metrics only.")
        else:
            tab._log(f"{b.name}: kept {len(out)} / {len(sam_records)} after SAM2 + FormLearner filter; rejected={rejected}, threshold={threshold:.2f}")
    except Exception:
        pass
    return out



# -----------------------------------------------------------------------------
# Scientific filter blocks and FormLearner science feature bridge
# -----------------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _parse_scientific_options(text: Any, preset_path: Any = "") -> Dict[str, Any]:
    opts: Dict[str, Any] = {}
    # Optional JSON preset selected through the model/config button.
    for raw_path in (preset_path, _parse_key_path(text, "preset"), _parse_key_path(text, "config")):
        try:
            pp = Path(str(raw_path or "").strip().strip('"')).expanduser()
            if pp.is_file() and pp.suffix.lower() == ".json":
                data = json.loads(pp.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    opts.update(data)
                break
        except Exception:
            pass
    for raw in str(text or "").replace(";", ",").split(","):
        part = raw.strip()
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            k = k.strip().lower().replace("-", "_")
            v = v.strip().strip('"')
            if v.lower() in {"true", "yes", "on"}:
                opts[k] = True
            elif v.lower() in {"false", "no", "off"}:
                opts[k] = False
            else:
                try:
                    opts[k] = float(v) if any(ch in v for ch in ".eE") else int(v)
                except Exception:
                    opts[k] = v
        else:
            # Keep bare words as enabled flags, e.g. "rectangular" or "strict".
            opts[part.lower().replace("-", "_")] = True
    return opts



def _scientific_options_text_from_dict(opts: Dict[str, Any]) -> str:
    parts = []
    for k, v in opts.items():
        if v in (None, ""):
            continue
        if isinstance(v, float):
            if abs(v - round(v)) < 1e-9 and abs(v) > 9:
                shown = str(int(round(v)))
            else:
                shown = f"{v:.6g}"
        else:
            shown = str(v)
        parts.append(f"{k}={shown}")
    return ", ".join(parts)

def _basic_filter_opts_for_block(block: Any) -> Dict[str, Any]:
    typ = _norm_type(getattr(block, "type", ""))
    opts = _parse_scientific_options(getattr(block, "classes_filter", ""), getattr(block, "model_path", ""))
    for key, default, _minv, _maxv, _dec in BASIC_FILTER_PARAM_SPECS.get(typ, []):
        opts.setdefault(key, default)
    return opts

def _set_basic_filter_param(block: Any, key: str, value: Any) -> None:
    opts = _parse_scientific_options(getattr(block, "classes_filter", ""), getattr(block, "model_path", ""))
    if isinstance(value, float) and abs(value - round(value)) < 1e-9:
        value = int(round(value))
    opts[str(key)] = value
    # Keep only simple scalar options in the visible block field so the UI stays basic.
    scalar = {}
    for k, v in opts.items():
        if isinstance(v, (str, int, float, bool)):
            scalar[k] = v
    block.classes_filter = _scientific_options_text_from_dict(scalar)

def _apply_basic_filter_defaults(block: Any) -> None:
    typ = _norm_type(getattr(block, "type", ""))
    if typ not in SCIENTIFIC_FILTER_TYPES:
        return
    opts = _basic_filter_opts_for_block(block)
    keep = {}
    for key, _default, _minv, _maxv, _dec in BASIC_FILTER_PARAM_SPECS.get(typ, []):
        keep[key] = opts.get(key)
    # Preserve selected preset/config/FormLearner fields, but don't flood the block UI.
    for k in ("preset", "config"):
        if k in opts:
            keep[k] = opts[k]
    block.classes_filter = _scientific_options_text_from_dict(keep)

def _bbox_from_record(r: Dict[str, Any]) -> Tuple[float, float, float, float]:
    return (
        _safe_float(r.get("x1"), 0.0), _safe_float(r.get("y1"), 0.0),
        _safe_float(r.get("x2"), 0.0), _safe_float(r.get("y2"), 0.0),
    )


def _crop_gray_array(full: Any, bbox: Tuple[float, float, float, float], pad: int = 0):
    x1, y1, x2, y2 = bbox
    try:
        w, h = int(full.width), int(full.height)
        ix1 = max(0, int(math.floor(min(x1, x2))) - int(pad))
        iy1 = max(0, int(math.floor(min(y1, y2))) - int(pad))
        ix2 = min(w, int(math.ceil(max(x1, x2))) + int(pad))
        iy2 = min(h, int(math.ceil(max(y1, y2))) + int(pad))
        if ix2 <= ix1 or iy2 <= iy1:
            return None
        crop = full.crop((ix1, iy1, ix2, iy2)).convert("L")
        try:
            import numpy as np
            return np.asarray(crop, dtype="float32")
        except Exception:
            # Pillow-only fallback: return list-like data through ImageStat users.
            return crop
    except Exception:
        return None


def _local_image_metrics(full: Any, bbox: Tuple[float, float, float, float], pad: int = 8) -> Dict[str, float]:
    arr = _crop_gray_array(full, bbox, pad)
    if arr is None:
        return {"local_mean": 0.0, "local_std": 0.0, "local_contrast": 0.0, "edge_density": 0.0}
    try:
        import numpy as np
        if not hasattr(arr, "shape"):
            raise RuntimeError("no numpy array")
        if arr.size <= 0:
            return {"local_mean": 0.0, "local_std": 0.0, "local_contrast": 0.0, "edge_density": 0.0}
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        p5 = float(np.percentile(arr, 5))
        p95 = float(np.percentile(arr, 95))
        contrast = max(0.0, min(1.0, (p95 - p5) / 255.0))
        if arr.shape[0] > 2 and arr.shape[1] > 2:
            gx = np.abs(np.diff(arr, axis=1))
            gy = np.abs(np.diff(arr, axis=0))
            edge = float((np.mean(gx) + np.mean(gy)) / 510.0)
        else:
            edge = 0.0
        return {
            "local_mean": mean,
            "local_std": std,
            "local_contrast": contrast,
            "edge_density": max(0.0, min(1.0, edge)),
        }
    except Exception:
        try:
            from PIL import ImageStat
            stat = ImageStat.Stat(arr)
            mean = float(stat.mean[0])
            std = float(stat.stddev[0])
            # Conservative fallback: Pillow stat has no percentile; std/64 is a useful normalized texture proxy.
            return {"local_mean": mean, "local_std": std, "local_contrast": max(0.0, min(1.0, std / 64.0)), "edge_density": 0.0}
        except Exception:
            return {"local_mean": 0.0, "local_std": 0.0, "local_contrast": 0.0, "edge_density": 0.0}


def _scientific_features_for_record(full: Any, r: Dict[str, Any], opts: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    opts = opts or {}
    x1, y1, x2, y2 = _bbox_from_record(r)
    w = max(1.0, abs(x2 - x1))
    h = max(1.0, abs(y2 - y1))
    area = max(1.0, w * h)
    cx = min(x1, x2) + w / 2.0
    cy = min(y1, y2) + h / 2.0
    aspect = max(w / h, h / w)
    short_side = min(w, h)
    long_side = max(w, h)
    fill = _safe_float(r.get("mask_fill_ratio", r.get("extent", r.get("fill_ratio", 1.0))), 1.0)
    if "mask_area_px" in r:
        fill = max(0.0, min(1.0, _safe_float(r.get("mask_area_px"), area) / area))
    elongation = max(0.0, min(1.0, 1.0 - (short_side / long_side)))
    # Rectangularity high for box-like or mask-filled objects; compactness high for not-extreme aspect.
    rectangularity = max(0.0, min(1.0, fill))
    compactness = max(0.0, min(1.0, 1.0 / aspect))
    ratio_target = _safe_float(opts.get("aspect_target"), 2.2)
    aspect_score = math.exp(-abs(math.log(max(1.0, aspect) / max(1.001, ratio_target))))
    fill_target = _safe_float(opts.get("fill_target"), 0.55)
    fill_score = 1.0 - min(1.0, abs(fill - fill_target) / max(0.05, fill_target))
    shape_score = max(0.0, min(1.0, 0.45 * aspect_score + 0.25 * fill_score + 0.30 * compactness))
    pad = _safe_int(opts.get("texture_pad", opts.get("pad", 8)), 8)
    img = _local_image_metrics(full, (x1, y1, x2, y2), pad=pad)
    return {
        "bbox_width_px": float(w),
        "bbox_height_px": float(h),
        "area_px": float(area),
        "center_x": float(cx),
        "center_y": float(cy),
        "aspect_ratio": float(aspect),
        "elongation": float(elongation),
        "rectangularity": float(rectangularity),
        "compactness": float(compactness),
        "mask_fill_ratio": float(max(0.0, min(1.0, fill))),
        "shape_score": float(shape_score),
        **img,
    }


def _add_context_features(records: List[Dict[str, Any]], opts: Dict[str, Any]) -> None:
    radius = _safe_float(opts.get("radius", opts.get("context_radius", 250.0)), 250.0)
    if radius <= 0:
        radius = 250.0
    centers = []
    for r in records:
        f = r.get("scientific_features") or {}
        centers.append((_safe_float(f.get("center_x"), 0.0), _safe_float(f.get("center_y"), 0.0)))
    for i, r in enumerate(records):
        cx, cy = centers[i]
        dists = []
        for j, (ox, oy) in enumerate(centers):
            if i == j:
                continue
            d = math.hypot(cx - ox, cy - oy)
            dists.append(d)
        near = [d for d in dists if d <= radius]
        f = dict(r.get("scientific_features") or {})
        f["neighbor_count"] = float(len(near))
        f["nearest_distance_px"] = float(min(dists) if dists else 10**9)
        f["density_score"] = max(0.0, min(1.0, len(near) / max(1.0, _safe_float(opts.get("neighbors_target"), 4.0))))
        r["scientific_features"] = f
        r.update({k: v for k, v in f.items() if k not in r})


def _add_alignment_features(records: List[Dict[str, Any]], opts: Dict[str, Any]) -> None:
    radius = _safe_float(opts.get("radius", opts.get("line_radius", 450.0)), 450.0)
    min_neighbors = _safe_int(opts.get("min_neighbors", opts.get("line_min_neighbors", 2)), 2)
    centers = []
    for r in records:
        f = r.get("scientific_features") or {}
        centers.append((_safe_float(f.get("center_x"), 0.0), _safe_float(f.get("center_y"), 0.0)))
    for i, r in enumerate(records):
        cx, cy = centers[i]
        pts = [(cx, cy)]
        for j, (ox, oy) in enumerate(centers):
            if i != j and math.hypot(cx - ox, cy - oy) <= radius:
                pts.append((ox, oy))
        line_score = 0.0
        orientation = 0.0
        if len(pts) >= max(3, min_neighbors + 1):
            try:
                import numpy as np
                arr = np.asarray(pts, dtype="float64")
                arr = arr - arr.mean(axis=0, keepdims=True)
                cov = np.cov(arr.T)
                vals, vecs = np.linalg.eigh(cov)
                vals = sorted([float(v) for v in vals], reverse=True)
                den = max(1e-9, vals[0] + vals[1])
                line_score = max(0.0, min(1.0, (vals[0] - vals[1]) / den))
                # principal vector orientation for GIS/archaeology interpretation.
                vals2, vecs2 = np.linalg.eigh(cov)
                v = vecs2[:, int(np.argmax(vals2))]
                orientation = float((math.degrees(math.atan2(v[1], v[0])) + 360.0) % 180.0)
            except Exception:
                # Pure-Python fallback: compare bounding-box elongation of local centers.
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                dx = max(xs) - min(xs); dy = max(ys) - min(ys)
                long = max(dx, dy, 1.0); short = max(min(dx, dy), 1.0)
                line_score = max(0.0, min(1.0, 1.0 - short / long))
                orientation = 0.0 if dx >= dy else 90.0
        f = dict(r.get("scientific_features") or {})
        f["line_score"] = float(line_score)
        f["line_orientation_deg"] = float(orientation)
        r["scientific_features"] = f
        r.update({k: v for k, v in f.items() if k not in r})


def _scientific_keep_reason(typ: str, f: Dict[str, float], opts: Dict[str, Any], score: float, form_score: Optional[float] = None) -> Tuple[bool, str]:
    thr = _safe_float(opts.get("threshold", opts.get("score_min", opts.get("conf", 0.0))), 0.0)
    if typ == "scientific_shape_filter":
        area_min = _safe_float(opts.get("area_min", opts.get("min_area", 1.0)), 1.0)
        area_max = _safe_float(opts.get("area_max", opts.get("max_area", 1e18)), 1e18)
        aspect_min = _safe_float(opts.get("aspect_min", 1.0), 1.0)
        aspect_max = _safe_float(opts.get("aspect_max", opts.get("ratio_max", 12.0)), 12.0)
        fill_min = _safe_float(opts.get("fill_min", 0.0), 0.0)
        fill_max = _safe_float(opts.get("fill_max", 1.0), 1.0)
        keep = area_min <= f["area_px"] <= area_max and aspect_min <= f["aspect_ratio"] <= aspect_max and fill_min <= f["mask_fill_ratio"] <= fill_max and f["shape_score"] >= thr
        return keep, "shape_score/area/aspect/fill"
    if typ == "scientific_size_filter":
        area_min = _safe_float(opts.get("area_min", opts.get("min_area", 1.0)), 1.0)
        area_max = _safe_float(opts.get("area_max", opts.get("max_area", 1e18)), 1e18)
        width_min = _safe_float(opts.get("width_min", 1.0), 1.0)
        width_max = _safe_float(opts.get("width_max", 1e18), 1e18)
        height_min = _safe_float(opts.get("height_min", 1.0), 1.0)
        height_max = _safe_float(opts.get("height_max", 1e18), 1e18)
        keep = area_min <= f["area_px"] <= area_max and width_min <= f["bbox_width_px"] <= width_max and height_min <= f["bbox_height_px"] <= height_max
        return keep, "area/width/height"
    if typ == "scientific_texture_filter":
        contrast_min = _safe_float(opts.get("contrast_min", opts.get("local_contrast_min", thr)), thr)
        std_min = _safe_float(opts.get("std_min", opts.get("texture_std_min", 0.0)), 0.0)
        edge_min = _safe_float(opts.get("edge_min", 0.0), 0.0)
        keep = f["local_contrast"] >= contrast_min and f["local_std"] >= std_min and f["edge_density"] >= edge_min
        return keep, "local_contrast/std/edge_density"
    if typ == "cluster_context_filter":
        min_n = _safe_float(opts.get("neighbors_min", opts.get("min_neighbors", 1.0)), 1.0)
        max_n = _safe_float(opts.get("neighbors_max", opts.get("max_neighbors", 1e18)), 1e18)
        keep = min_n <= f.get("neighbor_count", 0.0) <= max_n
        return keep, "neighbor_count/context_density"
    if typ == "linear_alignment_filter":
        min_line = _safe_float(opts.get("line_score_min", opts.get("threshold", 0.55)), 0.55)
        min_n = _safe_float(opts.get("neighbors_min", opts.get("min_neighbors", 2.0)), 2.0)
        keep = f.get("line_score", 0.0) >= min_line and f.get("neighbor_count", 0.0) >= min_n
        return keep, "line_score/local_alignment"
    if typ == "scientific_formlearner_filter":
        mix_min = _safe_float(opts.get("mix_score_min", opts.get("threshold", thr if thr > 0 else 0.50)), 0.50)
        keep = score >= mix_min
        return keep, "form_score+scientific_feature_mix"
    return True, "pass"


def _run_scientific_filter_block(tab: Any, b: Any, full: Any, image_path: Path) -> List[Dict[str, Any]]:
    typ = _norm_type(getattr(b, "type", ""))
    parents = tab._input_records(b, full) if hasattr(tab, "_input_records") else []
    if not parents or str(getattr(b, "input_ref", "original")) == "original":
        try:
            tab._log(f"{getattr(b, 'name', 'Scientific filter')}: connect an upstream detection/segmentation block first.")
        except Exception:
            pass
        return []
    opts = _parse_scientific_options(getattr(b, "classes_filter", ""), getattr(b, "model_path", ""))
    # The normal confidence spin is reused as a general scientific threshold if the options text does not override it.
    try:
        opts.setdefault("threshold", float(getattr(b, "confidence", 0.0) or 0.0))
    except Exception:
        pass
    scored: List[Dict[str, Any]] = []
    for parent in parents:
        rec = dict(parent)
        f = _scientific_features_for_record(full, rec, opts)
        rec["scientific_features"] = f
        rec.update({k: v for k, v in f.items() if k not in rec})
        scored.append(rec)
    _add_context_features(scored, opts)
    _add_alignment_features(scored, opts)

    if typ == "geo_nms_filter":
        iou = _safe_float(opts.get("iou", opts.get("nms_iou", getattr(b, "confidence", 0.60))), 0.60)
        out = _nms_records(scored, iou)
        for rec in out:
            rec.update({
                "id": "r_" + uuid.uuid4().hex[:10],
                "block_id": getattr(b, "id", ""),
                "block_name": getattr(b, "name", "Geo-NMS duplicate filter"),
                "block_type": typ,
                "scientific_filter": "geo_nms_keep",
                "scientific_reason": f"iou<{iou:.3f}",
                "parent_id": rec.get("id") or rec.get("parent_id"),
            })
        try:
            tab._log(f"{getattr(b, 'name', 'Geo-NMS')}: kept {len(out)} / {len(scored)} after duplicate removal, IoU={iou:.2f}")
        except Exception:
            pass
        return out

    form_model = None
    backend = None
    form_path = str(getattr(b, "formlearner_model", "") or "").strip().strip('"')
    if typ == "scientific_formlearner_filter" and form_path:
        try:
            import mustatil_legacy_backend as backend  # type: ignore[no-redef]
            form_model = backend.SimpleFormLearner.load(form_path)
        except Exception as exc:
            try:
                tab._log(f"{getattr(b, 'name', 'Scientific FormLearner')}: FormLearner JSON could not be loaded; falling back to scientific features only. Reason: {exc}")
            except Exception:
                pass
            form_model = None
            backend = None

    out: List[Dict[str, Any]] = []
    rejected = 0
    for rec in scored:
        f = dict(rec.get("scientific_features") or {})
        form_score: Optional[float] = None
        if typ == "scientific_formlearner_filter" and form_model is not None and backend is not None:
            try:
                bbox = _bbox_from_record(rec)
                form_score = float(form_model.predict(backend.crop_features(full, bbox)))
            except Exception:
                form_score = None
        # Scientific composite score. Conservative and transparent: no model is hidden behind it.
        texture_score = max(0.0, min(1.0, f.get("local_contrast", 0.0) * 0.70 + f.get("edge_density", 0.0) * 0.30))
        context_score = max(0.0, min(1.0, f.get("density_score", 0.0) * 0.50 + f.get("line_score", 0.0) * 0.50))
        if typ == "scientific_texture_filter":
            score = texture_score
        elif typ == "cluster_context_filter":
            score = f.get("density_score", 0.0)
        elif typ == "linear_alignment_filter":
            score = f.get("line_score", 0.0)
        elif typ == "scientific_formlearner_filter":
            base_form = f.get("shape_score", 0.0) if form_score is None else form_score
            w_form = _safe_float(opts.get("weight_form", 0.55), 0.55)
            w_shape = _safe_float(opts.get("weight_shape", 0.25), 0.25)
            w_texture = _safe_float(opts.get("weight_texture", 0.10), 0.10)
            w_context = _safe_float(opts.get("weight_context", 0.10), 0.10)
            den = max(1e-9, w_form + w_shape + w_texture + w_context)
            score = (w_form * base_form + w_shape * f.get("shape_score", 0.0) + w_texture * texture_score + w_context * context_score) / den
        else:
            score = f.get("shape_score", 0.0)
        keep, reason = _scientific_keep_reason(typ, f, opts, score, form_score)
        new = dict(rec)
        new.update({
            "id": "r_" + uuid.uuid4().hex[:10],
            "block_id": getattr(b, "id", ""),
            "block_name": getattr(b, "name", "Scientific filter"),
            "block_type": typ,
            "parent_id": rec.get("id") or rec.get("parent_id"),
            "scientific_score": float(score),
            "scientific_filter": "keep" if keep else "reject",
            "scientific_reason": reason,
            "form_score": float(form_score) if form_score is not None else rec.get("form_score"),
            "model": str(getattr(b, "model_path", "") or typ),
            "source_image": str(image_path),
        })
        if keep:
            out.append(new)
        else:
            rejected += 1
    try:
        tab._log(f"{getattr(b, 'name', 'Scientific filter')}: kept {len(out)} / {len(scored)}; rejected={rejected}; type={typ}")
    except Exception:
        pass
    return out


def _run_formlearner_block_scientific(tab: Any, b: Any, full: Any) -> List[Dict[str, Any]]:
    """FormLearner block with the same scientific feature metadata used by the filter blocks.

    This keeps the existing FormLearner behavior, but records morphometry/texture/context
    features so a later scientific filter block can use them. If classes/options contain
    scientific_filter=true, the FormLearner block also applies the same composite threshold.
    """
    parents = tab._input_records(b, full) if hasattr(tab, "_input_records") else []
    model_path = Path(str((getattr(b, "formlearner_model", "") or getattr(b, "model_path", "") or "")).strip().strip('"')).expanduser()
    if not model_path.is_file():
        raise RuntimeError(f"FormLearner block '{getattr(b, 'name', '')}' needs a valid .json model.")
    import mustatil_legacy_backend as backend
    form_model = backend.SimpleFormLearner.load(str(model_path))
    opts = _parse_scientific_options(getattr(b, "classes_filter", ""), "")
    opts.setdefault("threshold", float(getattr(b, "form_threshold", 0.5) or 0.5))
    provisional: List[Dict[str, Any]] = []
    for parent in parents:
        bbox = _bbox_from_record(parent)
        score = float(form_model.predict(backend.crop_features(full, bbox)))
        rec = dict(parent)
        f = _scientific_features_for_record(full, rec, opts)
        rec["scientific_features"] = f
        rec.update({k: v for k, v in f.items() if k not in rec})
        rec.update({
            "id": "r_" + uuid.uuid4().hex[:10],
            "block_id": getattr(b, "id", ""),
            "block_name": getattr(b, "name", "FormLearner block"),
            "block_type": getattr(b, "type", "formlearner"),
            "parent_id": parent.get("id"),
            "form_score": score,
            "form_threshold": float(getattr(b, "form_threshold", 0.5) or 0.5),
            "form_status": "positive" if score >= float(getattr(b, "form_threshold", 0.5) or 0.5) else "false_positive",
            "label": parent.get("label", "object") if score >= float(getattr(b, "form_threshold", 0.5) or 0.5) else "false_positive",
            "class_id": parent.get("class_id", 0) if score >= float(getattr(b, "form_threshold", 0.5) or 0.5) else 1,
            "scientific_feature_set": "morphometry_texture_context_v1",
        })
        provisional.append(rec)
    _add_context_features(provisional, opts)
    _add_alignment_features(provisional, opts)
    if bool(opts.get("scientific_filter", False)) or bool(opts.get("science_filter", False)):
        out: List[Dict[str, Any]] = []
        for rec in provisional:
            f = dict(rec.get("scientific_features") or {})
            texture_score = max(0.0, min(1.0, f.get("local_contrast", 0.0) * 0.70 + f.get("edge_density", 0.0) * 0.30))
            context_score = max(0.0, min(1.0, f.get("density_score", 0.0) * 0.50 + f.get("line_score", 0.0) * 0.50))
            score = (0.55 * _safe_float(rec.get("form_score"), 0.0) + 0.25 * f.get("shape_score", 0.0) + 0.10 * texture_score + 0.10 * context_score)
            keep = score >= _safe_float(opts.get("mix_score_min", opts.get("threshold", getattr(b, "form_threshold", 0.5))), 0.5)
            rec["scientific_score"] = float(score)
            rec["scientific_filter"] = "keep" if keep else "reject"
            if keep:
                out.append(rec)
        return out
    return provisional


# -----------------------------------------------------------------------------
# Canvas pan patch and UI layout polish
# -----------------------------------------------------------------------------

def _patch_view_class(view_cls: Any) -> None:
    if view_cls is None or id(view_cls) in _PATCHED_VIEWS:
        return
    _PATCHED_VIEWS.add(id(view_cls))
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QCursor
    except Exception:
        return
    orig_press = getattr(view_cls, "mousePressEvent", None)
    orig_move = getattr(view_cls, "mouseMoveEvent", None)
    orig_release = getattr(view_cls, "mouseReleaseEvent", None)

    def _event_pos(event):
        try:
            return event.position().toPoint()
        except Exception:
            return event.pos()

    def mouse_press_patched(self, event):
        try:
            if event.button() == Qt.LeftButton:
                pos = _event_pos(event)
                scene_pos = self.mapToScene(pos)
                # Keep socket-to-socket connection behaviour exactly as before.
                try:
                    if self._socket_item_at(scene_pos, "output") is not None:
                        return orig_press(self, event)
                except Exception:
                    pass
                clicked_item = None
                try:
                    clicked_item = self.itemAt(pos)
                except Exception:
                    clicked_item = None
                block_items = []
                try:
                    block_items = list(getattr(getattr(self, "tab", None), "block_items", {}).values())
                except Exception:
                    block_items = []
                # Pan only on empty canvas/arrow/background. Clicking blocks or embedded controls still moves/edits blocks.
                on_block = False
                cur = clicked_item
                for _ in range(6):
                    if cur in block_items:
                        on_block = True
                        break
                    try:
                        cur = cur.parentItem()
                    except Exception:
                        break
                if not on_block:
                    self._mustatil_left_pan_active = True
                    self._mustatil_left_pan_last = pos
                    try:
                        self.setCursor(QCursor(Qt.ClosedHandCursor))
                    except Exception:
                        pass
                    event.accept()
                    return
        except Exception:
            pass
        return orig_press(self, event) if callable(orig_press) else None

    def mouse_move_patched(self, event):
        try:
            if getattr(self, "_mustatil_left_pan_active", False):
                pos = _event_pos(event)
                last = getattr(self, "_mustatil_left_pan_last", pos)
                dx = pos.x() - last.x()
                dy = pos.y() - last.y()
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - dx)
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() - dy)
                self._mustatil_left_pan_last = pos
                event.accept()
                return
        except Exception:
            pass
        return orig_move(self, event) if callable(orig_move) else None

    def mouse_release_patched(self, event):
        try:
            if getattr(self, "_mustatil_left_pan_active", False):
                self._mustatil_left_pan_active = False
                try:
                    self.unsetCursor()
                except Exception:
                    pass
                event.accept()
                return
        except Exception:
            pass
        return orig_release(self, event) if callable(orig_release) else None

    view_cls.mousePressEvent = mouse_press_patched
    view_cls.mouseMoveEvent = mouse_move_patched
    view_cls.mouseReleaseEvent = mouse_release_patched


def _safe_set_min_width(widget: Any, width: int) -> None:
    try:
        widget.setMinimumWidth(int(width))
    except Exception:
        pass


def _safe_set_max_height(widget: Any, height: int) -> None:
    try:
        widget.setMaximumHeight(int(height))
    except Exception:
        pass


def _hide_old_top_controls(tab: Any) -> None:
    """Remove/hide every older AI Pipeline top-control row.

    v8 must leave exactly one visible block chooser.  Older patches created
    their own groups, buttons and combo boxes; hiding only individual buttons
    leaves empty rows behind.  This function therefore hides complete old
    containers first, then the orphan controls.
    """
    try:
        from PySide6.QtWidgets import QPushButton, QComboBox, QLabel, QGroupBox, QWidget
    except Exception:
        return

    def is_v8_widget(w: Any) -> bool:
        try:
            return bool(w.property("mustatil_v8_toolbar") or w.property("mustatil_v8_toolbar_child"))
        except Exception:
            return False

    def hide_widget(w: Any) -> None:
        if w is None or is_v8_widget(w):
            return
        try:
            w.setVisible(False)
        except Exception:
            pass
        try:
            w.setMaximumHeight(0)
        except Exception:
            pass
        try:
            w.setMinimumHeight(0)
        except Exception:
            pass
        try:
            w.setSizePolicy(w.sizePolicy().horizontalPolicy(), w.sizePolicy().Fixed)
        except Exception:
            pass

    try:
        # Hide complete old toolbar containers.  This is the important part for
        # the "top UI not double" problem.
        for gb in tab.findChildren(QGroupBox):
            obj = str(gb.objectName() or "")
            title = str(gb.title() or "").strip().lower()
            if is_v8_widget(gb):
                continue
            if (
                obj in {
                    "MustatilCleanAiPipelineBlockChooser",
                    "MustatilAiPipelineSingleToolbarV8",
                    "MustatilAiPipelineModelDropdownBar",
                    "MustatilAiPipelineModelChooser",
                }
                or obj.startswith("MustatilCleanAiPipelineBlockChooser")
                or obj.startswith("MustatilAiPipelineModelDropdown")
                or title in {"block-auswahl", "block auswahl", "ai model", "ai model block", "model block"}
                or "pipeline model" in title
                or "block-auswahl" in title
            ):
                hide_widget(gb)

        # Hide row widgets that only contain an old model dropdown/buttons.
        for w in tab.findChildren(QWidget):
            if is_v8_widget(w):
                continue
            obj = str(w.objectName() or "")
            if obj.startswith("MustatilCleanAiPipeline") or obj.startswith("MustatilAiPipelineModelDropdown") or obj.startswith("MustatilAiPipelineAdd"):
                hide_widget(w)

        # Hide orphan buttons created by older patches or by the original row.
        for btn in tab.findChildren(QPushButton):
            if is_v8_widget(btn):
                continue
            text = str(btn.text() or "").strip().lower()
            obj = str(btn.objectName() or "")
            if (
                text in {
                    "+ yolo block", "+ owlv2 block", "+ grounding dino block", "+ lae-dino block",
                    "+ sam2 block", "+ formlearner block", "+ if / logic block",
                    "duplicate selected", "delete selected", "auto layout", "load pipeline", "save pipeline",
                    "view: code", "view: blocks", "+ ai block", "+ add ai block", "+ add ai model block",
                    "add ai model block", "duplicate", "delete", "load", "save"
                }
                or any(k in text for k in AI_BUTTON_KEYWORDS)
                or obj.startswith("MustatilAiPipelineAdd")
                or obj.startswith("MustatilCleanAiPipeline")
            ):
                hide_widget(btn)

        for combo in tab.findChildren(QComboBox):
            if is_v8_widget(combo):
                continue
            obj = str(combo.objectName() or "")
            if obj.startswith("MustatilAiPipelineModelDropdown") or obj.startswith("MustatilCleanAiPipeline"):
                hide_widget(combo)

        for lab in tab.findChildren(QLabel):
            if is_v8_widget(lab):
                continue
            txt = str(lab.text() or "").strip().lower()
            obj = str(lab.objectName() or "")
            if txt in {"ai model", "ai-model", "model", "ai model block"} or obj.startswith("MustatilCleanAiPipeline"):
                hide_widget(lab)
    except Exception:
        pass

def _install_single_clean_toolbar(tab: Any) -> None:
    """Create one clean top row and aggressively remove old duplicates."""
    global _CLEAN_TOOLBAR_TABS
    try:
        from PySide6.QtWidgets import (
            QGroupBox, QHBoxLayout, QLabel, QPushButton, QComboBox, QSizePolicy
        )
    except Exception:
        return

    _hide_old_top_controls(tab)
    try:
        root = tab.layout()
        if root is None:
            return

        # If a v8 toolbar already exists, only refresh its choices and hide old bars.
        old_bar = getattr(tab, "_mustatil_clean_toolbar_v8", None)
        old_combo = getattr(tab, "_mustatil_clean_ai_combo_v8", None)
        if old_bar is not None and old_combo is not None:
            try:
                existing_data = {str(old_combo.itemData(i)) for i in range(old_combo.count())}
                for label, typ in AI_CHOICES:
                    if typ not in existing_data:
                        old_combo.addItem(label, typ)
                old_bar.setVisible(True)
                old_bar.setMaximumHeight(16777215)
                _hide_old_top_controls(tab)
                return
            except Exception:
                pass

        bar = QGroupBox("Block-Auswahl")
        bar.setObjectName("MustatilAiPipelineSingleToolbarV8")
        bar.setProperty("mustatil_v8_toolbar", True)
        bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(8)

        lab = QLabel("AI model block")
        lab.setObjectName("MustatilAiPipelineSingleToolbarLabelV8")
        lab.setProperty("mustatil_v8_toolbar_child", True)
        lay.addWidget(lab)

        combo = QComboBox()
        combo.setObjectName("MustatilAiPipelineSingleToolbarComboV8")
        combo.setProperty("mustatil_v8_toolbar_child", True)
        combo.setProperty("mustatil_clean_ai_combo", True)
        for label, typ in AI_CHOICES:
            combo.addItem(label, typ)
        combo.setMinimumWidth(270)
        combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lay.addWidget(combo, 2)

        def _mk_button(text: str, slot: Any, minw: int = 0):
            b = QPushButton(text)
            b.setProperty("mustatil_v8_toolbar_child", True)
            b.setProperty("mustatil_clean_toolbar_button", True)
            if minw:
                b.setMinimumWidth(minw)
            try:
                b.clicked.connect(slot)
            except Exception:
                pass
            lay.addWidget(b)
            return b

        def add_ai():
            try:
                typ = str(combo.currentData() or combo.currentText()).strip()
                tab._add_block(typ)
            except Exception as exc:
                try:
                    tab._log("Add AI model block failed: " + str(exc))
                except Exception:
                    _log("Add AI model block failed: " + str(exc))

        _mk_button("+ Add", add_ai, 72)
        _mk_button("+ FormLearner", lambda: tab._add_block("formlearner"), 112)
        _mk_button("+ IF / Logic", lambda: tab._add_block("rule"), 100)
        _mk_button("Duplicate", getattr(tab, "_duplicate_current", lambda: None), 86)
        _mk_button("Delete", getattr(tab, "_delete_current", lambda: None), 72)
        _mk_button("Auto layout", getattr(tab, "auto_layout", lambda: None), 92)

        def toggle_view():
            try:
                tab.toggle_block_code_view()
                old = getattr(tab, "view_mode_btn", None)
                if old is not None and hasattr(old, "text"):
                    view_btn.setText(str(old.text() or "View"))
            except Exception as exc:
                try:
                    tab._log("Toggle view failed: " + str(exc))
                except Exception:
                    pass

        view_btn = _mk_button("View: Code", toggle_view, 92)
        try:
            old = getattr(tab, "view_mode_btn", None)
            if old is not None and hasattr(old, "text"):
                view_btn.setText(str(old.text() or "View: Code"))
        except Exception:
            pass
        lay.addStretch(1)
        _mk_button("Load", getattr(tab, "load_pipeline", lambda: None), 64)
        _mk_button("Save", getattr(tab, "save_pipeline", lambda: None), 64)

        try:
            root.insertWidget(0, bar)
        except Exception:
            root.addWidget(bar)
        tab._mustatil_clean_ai_combo = combo
        tab._mustatil_clean_ai_combo_v8 = combo
        tab._mustatil_clean_toolbar = bar
        tab._mustatil_clean_toolbar_v8 = bar
        _CLEAN_TOOLBAR_TABS.add(id(tab))
        _hide_old_top_controls(tab)
        try:
            bar.setVisible(True)
            bar.setMaximumHeight(16777215)
        except Exception:
            pass
    except Exception:
        traceback.print_exc()

def _add_vertical_field(parent_layout: Any, label_text: str, widget: Any, hint: str = "") -> None:
    """Add a non-squeezed label-above-field row."""
    try:
        from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
    except Exception:
        return
    try:
        row = QWidget()
        row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        lay = QVBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lab = QLabel(label_text)
        lab.setStyleSheet("font-weight: bold;")
        lab.setWordWrap(True)
        lay.addWidget(lab)
        if hint:
            h = QLabel(hint)
            h.setWordWrap(True)
            h.setStyleSheet("color: #666; font-size: 8pt;")
            lay.addWidget(h)
        try:
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        except Exception:
            pass
        lay.addWidget(widget)
        parent_layout.addWidget(row)
    except Exception:
        pass


def _add_horizontal_pair(parent_layout: Any, left_label: str, left_widget: Any, right_label: str, right_widget: Any) -> None:
    try:
        from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSizePolicy
    except Exception:
        return
    try:
        row = QWidget()
        row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        outer = QHBoxLayout(row)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)
        for text, widget in ((left_label, left_widget), (right_label, right_widget)):
            box = QWidget()
            box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
            lay = QVBoxLayout(box)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(2)
            lab = QLabel(text); lab.setStyleSheet("font-weight: bold;")
            lay.addWidget(lab)
            try:
                widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            except Exception:
                pass
            lay.addWidget(widget)
            outer.addWidget(box, 1)
        parent_layout.addWidget(row)
    except Exception:
        pass


def _make_clean_group(title: str) -> Tuple[Any, Any]:
    from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QSizePolicy
    gb = QGroupBox(title)
    gb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
    lay = QVBoxLayout(gb)
    lay.setContentsMargins(10, 8, 10, 8)
    lay.setSpacing(8)
    return gb, lay


def _find_main_property_scroll(tab: Any) -> Optional[Any]:
    try:
        from PySide6.QtWidgets import QScrollArea, QGroupBox
        for scroll in tab.findChildren(QScrollArea):
            try:
                w = scroll.widget()
                if isinstance(w, QGroupBox) and "Selected block" in str(w.title() or ""):
                    return scroll
                if w is not None and str(w.objectName() or "") == "MustatilCleanRightPanel":
                    return scroll
            except Exception:
                pass
        areas = tab.findChildren(QScrollArea)
        return areas[0] if areas else None
    except Exception:
        return None


def _clean_field_widths(tab: Any) -> None:
    try:
        fields = [
            getattr(tab, "name", None), getattr(tab, "input_ref", None), getattr(tab, "classes_filter", None),
            getattr(tab, "typ", None), getattr(tab, "device", None), getattr(tab, "model_path", None),
            getattr(tab, "form_model", None), getattr(tab, "pipeline_image_path", None),
        ]
        for w in fields:
            if w is None:
                continue
            _safe_set_min_width(w, 250)
            try:
                if hasattr(w, "_edit"):
                    _safe_set_min_width(w._edit, 215)
            except Exception:
                pass
        try:
            getattr(tab, "rule_json").setMinimumHeight(170)
        except Exception:
            pass
        try:
            getattr(tab, "summary").setMinimumHeight(150)
            getattr(tab, "summary").setMaximumHeight(260)
        except Exception:
            pass
    except Exception:
        pass


def _rebuild_right_panel_clean(tab: Any) -> None:
    """Right side: one scroll area, dynamically simplified for the selected block type."""
    global _CLEAN_RIGHT_TABS
    try:
        from PySide6.QtWidgets import (
            QWidget, QVBoxLayout, QScrollArea, QGroupBox, QLabel, QPushButton,
            QDoubleSpinBox, QGridLayout, QSizePolicy
        )
        from PySide6.QtCore import Qt
    except Exception:
        return
    try:
        _clean_field_widths(tab)
        if getattr(tab, "_mustatil_clean_right_panel_v8", None) is not None:
            _update_clean_right_panel_for_block(tab, _current_block_for_tab(tab))
            _polish_splitters_and_text(tab)
            return

        scroll = _find_main_property_scroll(tab)
        if scroll is None:
            return

        action_box = None
        for gb in tab.findChildren(QGroupBox):
            try:
                title = str(gb.title() or "")
                if "Block actions" in title or "Process" in title or "functions" in title.lower():
                    action_box = gb
                    break
            except Exception:
                pass
        summary = getattr(tab, "summary", None)

        panel = QWidget()
        panel.setObjectName("MustatilCleanRightPanel")
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        panel_lay = QVBoxLayout(panel)
        panel_lay.setContentsMargins(8, 8, 8, 8)
        panel_lay.setSpacing(10)
        tab._mustatil_right_groups = {}
        tab._mustatil_right_rows = {}

        def field_row(parent_lay, key: str, label_text: str, widget: Any, hint: str = ""):
            row = QWidget()
            row.setObjectName("MustatilRightRow_" + key)
            row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
            row_lay = QVBoxLayout(row)
            row_lay.setContentsMargins(0, 0, 0, 0)
            row_lay.setSpacing(2)
            lab = QLabel(label_text); lab.setStyleSheet("font-weight: bold;"); lab.setWordWrap(True)
            row_lay.addWidget(lab)
            if hint:
                h = QLabel(hint); h.setWordWrap(True); h.setStyleSheet("color: #666; font-size: 8pt;")
                row_lay.addWidget(h)
            if widget is not None:
                try:
                    widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                except Exception:
                    pass
                row_lay.addWidget(widget)
            parent_lay.addWidget(row)
            tab._mustatil_right_rows[key] = row
            return row

        def pair_row(parent_lay, key: str, left_label: str, left_widget: Any, right_label: str, right_widget: Any):
            from PySide6.QtWidgets import QHBoxLayout
            row = QWidget(); row.setObjectName("MustatilRightRow_" + key)
            row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
            outer = QHBoxLayout(row); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(10)
            for subkey, text, widget in ((key + "_left", left_label, left_widget), (key + "_right", right_label, right_widget)):
                box = QWidget(); box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
                lay = QVBoxLayout(box); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(2)
                lab = QLabel(text); lab.setStyleSheet("font-weight: bold;")
                lay.addWidget(lab)
                try:
                    widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                except Exception:
                    pass
                lay.addWidget(widget)
                outer.addWidget(box, 1)
                tab._mustatil_right_rows[subkey] = box
            parent_lay.addWidget(row)
            tab._mustatil_right_rows[key] = row
            return row

        g, lay = _make_clean_group("Selected block")
        tab._mustatil_right_groups["selected"] = g
        try:
            lay.addWidget(getattr(tab, "enabled"))
        except Exception:
            pass
        field_row(lay, "name", "Name", getattr(tab, "name", None))
        pair_row(lay, "type_input", "Block type", getattr(tab, "typ", None), "Input from", getattr(tab, "input_ref", None))
        panel_lay.addWidget(g)

        g, lay = _make_clean_group("Model / runtime")
        tab._mustatil_right_groups["model"] = g
        field_row(lay, "model_path", "Model / checkpoint", getattr(tab, "model_path", None), "Only visible for blocks that actually need a model.")
        field_row(lay, "form_model", "Companion / config / FormLearner JSON", getattr(tab, "form_model", None), "Visible only when this block needs a second file.")
        _add_model_button_grid(lay, tab)
        pair_row(lay, "device_conf", "Device", getattr(tab, "device", None), "Confidence", getattr(tab, "conf", None))
        pair_row(lay, "imgsz_form", "Image size", getattr(tab, "imgsz", None), "Form threshold", getattr(tab, "form_threshold", None))
        panel_lay.addWidget(g)

        g, lay = _make_clean_group("Basic filter parameters")
        tab._mustatil_right_groups["filter"] = g
        info = QLabel("These algorithmic filters use only a few parameters. Presets can be trained/calibrated in the FormLearner/FormTrainer tab and then used here.")
        info.setWordWrap(True); info.setStyleSheet("color: #555;")
        lay.addWidget(info)
        param_box = QWidget(); param_lay = QGridLayout(param_box); param_lay.setContentsMargins(0,0,0,0); param_lay.setHorizontalSpacing(8); param_lay.setVerticalSpacing(6)
        lay.addWidget(param_box)
        tab._mustatil_v8_filter_param_box = param_box
        tab._mustatil_v8_filter_param_layout = param_lay
        field_row(lay, "filter_options", "Raw options / preset", getattr(tab, "classes_filter", None), "Optional. Usually the basic fields above are enough.")
        panel_lay.addWidget(g)

        g, lay = _make_clean_group("Image / crop processing")
        tab._mustatil_right_groups["processing"] = g
        try: lay.addWidget(getattr(tab, "use_chunked_yolo"))
        except Exception: pass
        try: lay.addWidget(getattr(tab, "shifted_tiles"))
        except Exception: pass
        pair_row(lay, "tile", "Tile size", getattr(tab, "tile_size", None), "Overlap", getattr(tab, "tile_overlap", None))
        field_row(lay, "crop_padding", "Parent crop padding", getattr(tab, "crop_padding", None))
        field_row(lay, "classes_filter", "Class / prompt / options", getattr(tab, "classes_filter", None), "For detector prompts/classes or special options such as config=... .")
        panel_lay.addWidget(g)

        g, lay = _make_clean_group("IF / Logic")
        tab._mustatil_right_groups["rule"] = g
        field_row(lay, "rule_json", "Rule JSON", getattr(tab, "rule_json", None))
        panel_lay.addWidget(g)

        if action_box is not None:
            try:
                action_box.setTitle("Process / functions")
                action_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
                tab._mustatil_right_groups["actions"] = action_box
                panel_lay.addWidget(action_box)
            except Exception:
                pass

        if summary is not None:
            try:
                log_group, log_lay = _make_clean_group("Log / result summary")
                summary.setMinimumHeight(130); summary.setMaximumHeight(240)
                summary.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
                log_lay.addWidget(summary)
                tab._mustatil_right_groups["log"] = log_group
                panel_lay.addWidget(log_group)
            except Exception:
                pass

        panel_lay.addStretch(1)
        try:
            old = scroll.takeWidget()
            if old is not None:
                old.setParent(None)
        except Exception:
            pass
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(400)
        scroll.setWidget(panel)
        tab._mustatil_clean_right_panel = panel
        tab._mustatil_clean_right_panel_v8 = panel
        _CLEAN_RIGHT_TABS.add(id(tab))
        _update_clean_right_panel_for_block(tab, _current_block_for_tab(tab))
        _polish_splitters_and_text(tab)
    except Exception:
        traceback.print_exc()


def _set_row_visible(tab: Any, key: str, visible: bool) -> None:
    try:
        row = getattr(tab, "_mustatil_right_rows", {}).get(key)
        if row is not None:
            row.setVisible(bool(visible))
    except Exception:
        pass


def _set_group_visible(tab: Any, key: str, visible: bool) -> None:
    try:
        g = getattr(tab, "_mustatil_right_groups", {}).get(key)
        if g is not None:
            g.setVisible(bool(visible))
    except Exception:
        pass


def _clear_layout(layout: Any) -> None:
    try:
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
    except Exception:
        pass


def _update_filter_param_grid(tab: Any, b: Optional[Any]) -> None:
    try:
        from PySide6.QtWidgets import QLabel, QDoubleSpinBox, QPushButton
    except Exception:
        return
    try:
        lay = getattr(tab, "_mustatil_v8_filter_param_layout", None)
        if lay is None:
            return
        _clear_layout(lay)
        if b is None:
            return
        typ = _norm_type(getattr(b, "type", ""))
        specs = BASIC_FILTER_PARAM_SPECS.get(typ, [])
        opts = _basic_filter_opts_for_block(b)
        tab._mustatil_v8_filter_spins_right = {}
        row = col = 0
        for key, default, minv, maxv, decimals in specs:
            lab = QLabel(key.replace("_", " "))
            spin = QDoubleSpinBox(); spin.setRange(float(minv), float(maxv)); spin.setDecimals(int(decimals))
            spin.setSingleStep(0.01 if decimals >= 2 else (1.0 if maxv > 10 else 0.1))
            spin.setValue(float(opts.get(key, default)))
            tab._mustatil_v8_filter_spins_right[key] = spin
            lay.addWidget(lab, row, col); lay.addWidget(spin, row, col + 1)
            def changed(_value, k=key, s=spin):
                try:
                    _set_basic_filter_param(b, k, float(s.value()))
                    _set_widget_text(getattr(tab, "classes_filter", None), getattr(b, "classes_filter", ""))
                    if hasattr(tab, "refresh_code_from_blocks"):
                        tab.refresh_code_from_blocks()
                    if getattr(b, "id", None) in getattr(tab, "block_items", {}):
                        tab.block_items[b.id].refresh_embedded_controls()
                except Exception:
                    pass
            spin.valueChanged.connect(changed)
            col += 2
            if col >= 4:
                row += 1; col = 0
        train_btn = QPushButton("Train/calibrate this filter in FormLearner tab")
        train_btn.setToolTip("Adds/opens the FormLearner scientific preset tools. They can save a JSON preset from current positive examples.")
        def _train_click():
            try:
                ws = getattr(tab, "ws", None)
                if ws is not None:
                    _install_formlearner_scientific_training_ui(ws)
                tab._log("Use the FormLearner/FormTrainer scientific preset box to calibrate this algorithmic filter, then select the saved JSON preset here.")
            except Exception:
                pass
        train_btn.clicked.connect(_train_click)
        lay.addWidget(train_btn, row + 1, 0, 1, 4)
    except Exception:
        traceback.print_exc()


def _update_clean_right_panel_for_block(tab: Any, b: Optional[Any] = None) -> None:
    try:
        if b is None:
            b = _current_block_for_tab(tab)
        typ = _norm_type(getattr(b, "type", "") if b is not None else "")
        is_filter = typ in SCIENTIFIC_FILTER_TYPES
        is_algorithm_filter = typ in ALGORITHM_ONLY_FILTER_TYPES
        is_model = typ in RIGHT_MODEL_TYPES
        is_form = typ == "formlearner"
        is_rule = typ == "rule"

        _set_group_visible(tab, "model", is_model or is_form or typ == "scientific_formlearner_filter")
        _set_group_visible(tab, "filter", is_filter)
        _set_group_visible(tab, "processing", (is_model or is_form) and not is_filter and not is_rule)
        _set_group_visible(tab, "rule", is_rule)

        _set_row_visible(tab, "model_path", is_model and typ not in ALGORITHM_ONLY_FILTER_TYPES)
        _set_row_visible(tab, "form_model", typ in RIGHT_MODELS_WITH_COMPANION or is_form or typ == "scientific_formlearner_filter")
        _set_row_visible(tab, "device_conf", is_model)
        _set_row_visible(tab, "imgsz_form", typ in {"yolo", "faster_rcnn", "mask_rcnn", "unet", "sam2_formlearner", "formlearner"})
        _set_row_visible(tab, "classes_filter", is_model or is_form)
        _set_row_visible(tab, "filter_options", is_filter)
        _update_filter_param_grid(tab, b if is_filter else None)

        groups = getattr(tab, "_mustatil_right_groups", {})
        if groups.get("model") is not None:
            if typ == "scientific_formlearner_filter":
                groups["model"].setTitle("FormLearner model for scientific filter")
            elif is_form:
                groups["model"].setTitle("FormLearner model")
            elif is_model:
                groups["model"].setTitle("Model / runtime")
        if groups.get("filter") is not None:
            title = DEFAULT_NAMES.get(typ, "Basic filter parameters")
            groups["filter"].setTitle(title)
    except Exception:
        pass

def _polish_splitters_and_text(tab: Any) -> None:
    try:
        from PySide6.QtWidgets import QLabel, QTextEdit, QSplitter, QScrollArea, QGroupBox
    except Exception:
        return
    try:
        for lab in tab.findChildren(QLabel):
            txt = str(lab.text() or "")
            low = txt.lower()
            # Hide verbose/deprecated help and duplicate log labels. The grouped right panel replaces them.
            if (
                "how to connect:" in low or "lego mindstorms" in low or "code view:" in low
                or txt.strip() in {"Pipeline log / result summary", "Log"}
            ):
                if not str(lab.objectName() or "").startswith("MustatilClean"):
                    lab.setVisible(False)
                    try:
                        lab.setMaximumHeight(0)
                    except Exception:
                        pass
        for scroll in tab.findChildren(QScrollArea):
            try:
                # Let scrolling solve height pressure; keep width reasonable so the canvas remains dominant.
                scroll.setMinimumWidth(max(420, min(520, scroll.minimumWidth() or 420)))
            except Exception:
                pass
        for gb in tab.findChildren(QGroupBox):
            try:
                title = str(gb.title() or "")
                if "Selected block" in title or "Process" in title or "Log" in title:
                    gb.setMinimumWidth(0)
            except Exception:
                pass
        for te in tab.findChildren(QTextEdit):
            try:
                if te.isReadOnly():
                    te.setMaximumHeight(260)
                    te.setMinimumHeight(130)
            except Exception:
                pass
        for split in tab.findChildren(QSplitter):
            try:
                split.setSizes([1250, 470])
                split.setStretchFactor(0, 1)
                split.setStretchFactor(1, 0)
                split.setCollapsible(1, False)
            except Exception:
                pass
    except Exception:
        pass


def _polish_right_side(tab: Any) -> None:
    try:
        _hide_old_top_controls(tab)
        _rebuild_right_panel_clean(tab)
        _polish_splitters_and_text(tab)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Patching the existing visual pipeline class/instance
# -----------------------------------------------------------------------------


def _patch_visual_block_inline_controls(vbi: Any) -> None:
    """Compact inline editors: model blocks get only essential selectors; filter blocks get only basic parameters."""
    if vbi is None or getattr(vbi, "_mustatil_v9_basic_inline_controls_patched", False):
        return
    try:
        from PySide6.QtWidgets import (
            QWidget, QGridLayout, QLabel, QComboBox, QDoubleSpinBox, QSpinBox,
            QLineEdit, QPushButton, QGraphicsProxyWidget, QFileDialog
        )
        from PySide6.QtGui import QColor, QPen
        from PySide6.QtCore import QRectF, Qt
    except Exception:
        return

    orig_refresh = getattr(vbi, "refresh_embedded_controls", None)
    orig_width = getattr(vbi, "block_width", None)
    orig_height = getattr(vbi, "block_height", None)
    orig_bounding_rect = getattr(vbi, "boundingRect", None)

    # v9: make the actual QGraphicsItem bounding rectangle follow the patched
    # block_height().  Some original builds use a fixed boundingRect, which made
    # the colored outline stop above the final CLASS/classes_filter row even when
    # the proxy widget itself was taller.
    try:
        if not getattr(vbi, "_mustatil_v9_bounding_rect_patched", False):
            def boundingRect_v9(self):
                try:
                    typ = _norm_type(getattr(getattr(self, "block", None), "type", ""))
                    if typ in MODEL_BLOCK_TYPES:
                        return QRectF(0, 0, float(self.block_width()), float(self.block_height()))
                except Exception:
                    pass
                try:
                    return orig_bounding_rect(self) if callable(orig_bounding_rect) else QRectF(0, 0, 320, 190)
                except Exception:
                    return QRectF(0, 0, 320, 190)
            vbi.boundingRect = boundingRect_v9
            vbi._mustatil_v9_bounding_rect_patched = True
    except Exception:
        pass

    # Draw one continuous colored outline around the whole block and a second
    # outline around the embedded input area.  This fixes the visual issue where
    # the colored frame stopped above the inline input fields.
    try:
        orig_paint = getattr(vbi, "paint", None)
        if callable(orig_paint) and not getattr(vbi, "_mustatil_v9_colored_input_border_patched", False):
            def paint_with_v8_input_border(self, painter, option, widget=None):
                orig_paint(self, painter, option, widget)
                try:
                    typ = _norm_type(getattr(getattr(self, "block", None), "type", ""))
                    color = getattr(type(self), "TYPE_COLORS", {}).get(typ, QColor(90, 130, 210))
                    if not getattr(getattr(self, "block", None), "enabled", True):
                        color = QColor(145, 145, 145)
                    painter.save()
                    painter.setRenderHint(painter.Antialiasing, True)
                    painter.setBrush(Qt.NoBrush)
                    painter.setPen(QPen(color, 3.0 if not self.isSelected() else 4.2))
                    painter.drawRoundedRect(self.boundingRect().adjusted(2, 2, -2, -2), 12, 12)
                    if getattr(self, "rule_proxy", None) is not None and typ in MODEL_BLOCK_TYPES:
                        lower = QRectF(7, 70, max(20.0, self.block_width() - 14), max(20.0, self.block_height() - 76))
                        painter.setPen(QPen(color, 2.2))
                        painter.drawRoundedRect(lower, 8, 8)
                    painter.restore()
                except Exception:
                    pass
            vbi.paint = paint_with_v8_input_border
            vbi._mustatil_v9_colored_input_border_patched = True
    except Exception:
        pass

    def _item_type(self) -> str:
        return _norm_type(getattr(getattr(self, "block", None), "type", ""))

    def _is_patched_block_item(self) -> bool:
        return _item_type(self) in MODEL_BLOCK_TYPES

    def block_width_v8(self):
        typ = _item_type(self)
        if typ in SCIENTIFIC_FILTER_TYPES:
            return 370.0
        if typ in MODEL_BLOCK_TYPES:
            b = getattr(self, "block", None)
            longest = max([len(str(getattr(b, a, "") or "")) for a in ("model_path", "formlearner_model", "classes_filter")] + [0])
            return float(max(430, min(610, 430 + longest * 2.0)))
        return orig_width(self) if callable(orig_width) else 300.0

    def block_height_v8(self):
        typ = _item_type(self)
        # v9: reserve enough vertical space for the complete inline editor.
        # In v8 the visual frame could end above the CLASS / classes_filter row.
        if typ in SCIENTIFIC_FILTER_TYPES:
            return 194.0 if typ != "scientific_formlearner_filter" else 226.0
        if typ in MODEL_BLOCK_TYPES:
            if typ == "sam2_formlearner":
                return 286.0
            if typ in {"sam2", "lae_dino"}:
                return 266.0
            if typ in {"formlearner"}:
                return 252.0
            return 242.0
        return orig_height(self) if callable(orig_height) else 190.0

    def _combo_set(combo: Any, value: str) -> None:
        try:
            ix = combo.findData(value)
            if ix >= 0:
                combo.setCurrentIndex(ix)
            else:
                combo.setEditText(value)
        except Exception:
            pass

    def _combo_value(combo: Any) -> str:
        try:
            data = combo.currentData()
            return str(data if data not in (None, "") else combo.currentText()).strip()
        except Exception:
            return ""

    def _clear_proxy(self) -> None:
        try:
            if getattr(self, "rule_proxy", None) is not None:
                self.scene().removeItem(self.rule_proxy)
        except Exception:
            pass
        self.rule_proxy = None
        self._proxy_type = ""

    def _update_editor(self):
        try:
            if hasattr(self, "tab") and hasattr(self.tab, "_update_editor_from_inline"):
                self.tab._update_editor_from_inline(self.block)
            if hasattr(self, "tab") and hasattr(self.tab, "refresh_code_from_blocks"):
                self.tab.refresh_code_from_blocks()
            if hasattr(self, "tab"):
                _update_clean_right_panel_for_block(self.tab, self.block)
        except Exception:
            pass
        try:
            self.update()
        except Exception:
            pass

    def _fill_model_combo(self, combo: Any, block: Any, slot: str) -> None:
        try:
            old = combo.blockSignals(True)
            combo.clear()
            for label, value in _collect_model_choices_for_block(getattr(self, "tab", None), block, slot):
                combo.addItem(label, value)
            cur = str(getattr(block, "model_path", "") if slot == "primary" else getattr(block, "formlearner_model", "") or "")
            _combo_set(combo, cur)
            combo.blockSignals(old)
        except Exception:
            pass

    def _browse_primary(self):
        try:
            typ = _item_type(self)
            path, _ = QFileDialog.getOpenFileName(getattr(self, "tab", None), "Choose model/checkpoint", "", _file_filter_for_block(typ, "primary"))
            if path:
                self._v8_model_combo.setEditText(path)
                _model_changed(self)
        except Exception:
            pass

    def _browse_companion(self):
        try:
            typ = _item_type(self)
            slot = "formlearner" if typ in {"sam2_formlearner", "scientific_formlearner_filter", "formlearner"} else "companion"
            path, _ = QFileDialog.getOpenFileName(getattr(self, "tab", None), "Choose companion/config", "", _file_filter_for_block(typ, slot))
            if path:
                self._v8_companion_combo.setEditText(path)
                _model_changed(self)
        except Exception:
            pass

    def _browse_sam2_cfg(self):
        try:
            typ = _item_type(self)
            path, _ = QFileDialog.getOpenFileName(getattr(self, "tab", None), "Choose SAM2 config YAML", "", _file_filter_for_block(typ, "sam2_config"))
            if not path:
                return
            if typ == "sam2":
                self._v8_companion_combo.setEditText(path)
            else:
                self._v8_options_edit.setText(_replace_key_path(str(self._v8_options_edit.text() or ""), "config", path))
            _model_changed(self)
        except Exception:
            pass

    def _model_changed(self):
        if getattr(self, "_updating_inline_controls", False):
            return
        try:
            b = self.block
            typ = _item_type(self)
            if hasattr(self, "_v8_model_combo"):
                b.model_path = _combo_value(self._v8_model_combo)
            if hasattr(self, "_v8_companion_combo"):
                b.formlearner_model = _combo_value(self._v8_companion_combo)
            if hasattr(self, "_v8_conf_spin"):
                b.confidence = float(self._v8_conf_spin.value())
            if hasattr(self, "_v8_imgsz_spin"):
                b.imgsz = int(self._v8_imgsz_spin.value())
            if hasattr(self, "_v8_device_combo"):
                b.device = str(self._v8_device_combo.currentText() or "cpu")
            if hasattr(self, "_v8_options_edit"):
                b.classes_filter = str(self._v8_options_edit.text() or "")
            if hasattr(self, "_v8_form_threshold_spin"):
                b.form_threshold = float(self._v8_form_threshold_spin.value())
            if typ in SCIENTIFIC_FILTER_TYPES:
                _apply_basic_filter_defaults(b)
            _update_editor(self)
        except Exception:
            pass

    def _filter_param_changed(self, key: str, spin: Any):
        if getattr(self, "_updating_inline_controls", False):
            return
        try:
            _set_basic_filter_param(self.block, key, float(spin.value()))
            _update_editor(self)
        except Exception:
            pass

    def _refresh_filter_controls(self):
        self._updating_inline_controls = True
        try:
            opts = _basic_filter_opts_for_block(self.block)
            for key, spin in getattr(self, "_v8_filter_spins", {}).items():
                try:
                    spin.setValue(float(opts.get(key, spin.value())))
                except Exception:
                    pass
            if hasattr(self, "_v8_companion_combo"):
                _fill_model_combo(self, self.block, "companion")
            if hasattr(self, "_v8_form_threshold_spin"):
                self._v8_form_threshold_spin.setValue(float(getattr(self.block, "form_threshold", 0.5) or 0.5))
        finally:
            self._updating_inline_controls = False

    def _refresh_model_controls(self):
        self._updating_inline_controls = True
        try:
            _fill_model_combo(self, self.block, "primary")
            if hasattr(self, "_v8_companion_combo"):
                _fill_model_combo(self, self.block, "companion")
            if hasattr(self, "_v8_conf_spin"):
                self._v8_conf_spin.setValue(float(getattr(self.block, "confidence", 0.05) or 0.05))
            if hasattr(self, "_v8_imgsz_spin"):
                self._v8_imgsz_spin.setValue(int(getattr(self.block, "imgsz", 640) or 640))
            if hasattr(self, "_v8_device_combo"):
                self._v8_device_combo.setCurrentText(str(getattr(self.block, "device", "cpu") or "cpu"))
            if hasattr(self, "_v8_options_edit"):
                self._v8_options_edit.setText(str(getattr(self.block, "classes_filter", "") or ""))
            if hasattr(self, "_v8_form_threshold_spin"):
                self._v8_form_threshold_spin.setValue(float(getattr(self.block, "form_threshold", 0.5) or 0.5))
        finally:
            self._updating_inline_controls = False

    def refresh_embedded_controls_v8(self):
        typ = _item_type(self)
        if typ not in MODEL_BLOCK_TYPES:
            return orig_refresh(self) if callable(orig_refresh) else None

        wanted = "filter_v9" if typ in SCIENTIFIC_FILTER_TYPES else "model_v9"
        if getattr(self, "_proxy_type", "") != wanted:
            _clear_proxy(self)

        if getattr(self, "rule_proxy", None) is None:
            w = QWidget()
            w.setObjectName("ModelBlockInlineEditorV8")
            try:
                _c = getattr(type(self), "TYPE_COLORS", {}).get(typ, QColor(90, 130, 210))
                _border = f"rgb({_c.red()}, {_c.green()}, {_c.blue()})"
            except Exception:
                _border = "rgb(90, 130, 210)"
            w.setStyleSheet(
                f"QWidget#ModelBlockInlineEditorV8 {{ background: rgba(255,255,255,230); border: 2px solid {_border}; border-radius: 8px; }}"
                "QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit { font-size: 7.5pt; min-height: 20px; max-height: 22px; }"
                "QPushButton { font-size: 7.5pt; min-height: 20px; max-height: 22px; padding-left: 3px; padding-right: 3px; }"
                "QLabel { font-size: 7.5pt; font-weight: bold; }"
            )
            lay = QGridLayout(w)
            lay.setContentsMargins(6, 4, 6, 4)
            lay.setHorizontalSpacing(5)
            lay.setVerticalSpacing(4)

            if typ in SCIENTIFIC_FILTER_TYPES:
                self._v8_filter_spins = {}
                lay.addWidget(QLabel("BASIC FILTER PARAMETERS"), 0, 0, 1, 4)
                row = 1
                col = 0
                for key, default, minv, maxv, decimals in BASIC_FILTER_PARAM_SPECS.get(typ, [])[:3]:
                    lab = QLabel(key.replace("_", " ").upper())
                    spin = QDoubleSpinBox()
                    spin.setRange(float(minv), float(maxv))
                    spin.setDecimals(int(decimals))
                    step = 0.01 if decimals >= 2 else (1.0 if maxv > 10 else 0.1)
                    spin.setSingleStep(step)
                    spin.setValue(float(default))
                    self._v8_filter_spins[key] = spin
                    lay.addWidget(lab, row, col)
                    lay.addWidget(spin, row, col + 1)
                    spin.valueChanged.connect(lambda _v, k=key, s=spin: _filter_param_changed(self, k, s))
                    col += 2
                    if col >= 4:
                        row += 1
                        col = 0
                if typ == "scientific_formlearner_filter":
                    self._v8_companion_combo = QComboBox(); self._v8_companion_combo.setEditable(True); self._v8_companion_combo.setMaxVisibleItems(10)
                    self._v8_companion_btn = QPushButton("Form JSON…")
                    lay.addWidget(QLabel("FORM"), row + 1, 0)
                    lay.addWidget(self._v8_companion_combo, row + 1, 1, 1, 2)
                    lay.addWidget(self._v8_companion_btn, row + 1, 3)
                    self._v8_companion_combo.currentTextChanged.connect(lambda *_: _model_changed(self))
                    self._v8_companion_btn.clicked.connect(lambda *_: _browse_companion(self))
                self._proxy_type = "filter_v9"
            else:
                self._v8_model_combo = QComboBox(); self._v8_model_combo.setEditable(True); self._v8_model_combo.setMinimumWidth(220); self._v8_model_combo.setMaxVisibleItems(12)
                self._v8_model_btn = QPushButton("Model…")
                self._v8_model_btn.setToolTip("Choose model/checkpoint")
                lay.addWidget(QLabel("MODEL"), 0, 0)
                lay.addWidget(self._v8_model_combo, 0, 1, 1, 3)
                lay.addWidget(self._v8_model_btn, 0, 4)
                next_row = 1
                if typ in RIGHT_MODELS_WITH_COMPANION:
                    self._v8_companion_combo = QComboBox(); self._v8_companion_combo.setEditable(True); self._v8_companion_combo.setMinimumWidth(220); self._v8_companion_combo.setMaxVisibleItems(12)
                    self._v8_companion_btn = QPushButton("Config…" if typ != "sam2_formlearner" else "Form JSON…")
                    lay.addWidget(QLabel("CONFIG" if typ != "sam2_formlearner" else "FORM"), next_row, 0)
                    lay.addWidget(self._v8_companion_combo, next_row, 1, 1, 3)
                    lay.addWidget(self._v8_companion_btn, next_row, 4)
                    self._v8_companion_combo.currentTextChanged.connect(lambda *_: _model_changed(self))
                    self._v8_companion_btn.clicked.connect(lambda *_: _browse_companion(self))
                    next_row += 1
                    if typ == "sam2_formlearner":
                        self._v8_sam_cfg_btn = QPushButton("SAM2 cfg…")
                        lay.addWidget(self._v8_sam_cfg_btn, next_row, 4)
                        self._v8_sam_cfg_btn.clicked.connect(lambda *_: _browse_sam2_cfg(self))
                self._v8_conf_spin = QDoubleSpinBox(); self._v8_conf_spin.setRange(0.001, 1.0); self._v8_conf_spin.setDecimals(3); self._v8_conf_spin.setSingleStep(0.01)
                self._v8_device_combo = QComboBox(); self._v8_device_combo.setEditable(True); self._v8_device_combo.addItems(["auto cuda", "cpu", "cuda", "0", "rocm", "directml", "openvino"])
                self._v8_options_edit = QLineEdit(); self._v8_options_edit.setPlaceholderText("class / prompt / options")
                lay.addWidget(QLabel("CONF"), next_row, 0); lay.addWidget(self._v8_conf_spin, next_row, 1)
                lay.addWidget(QLabel("DEVICE"), next_row, 2); lay.addWidget(self._v8_device_combo, next_row, 3, 1, 2)
                next_row += 1
                if typ in {"yolo", "faster_rcnn", "mask_rcnn", "unet"}:
                    self._v8_imgsz_spin = QSpinBox(); self._v8_imgsz_spin.setRange(64, 8192); self._v8_imgsz_spin.setSingleStep(32)
                    lay.addWidget(QLabel("SIZE"), next_row, 0); lay.addWidget(self._v8_imgsz_spin, next_row, 1)
                if typ in {"sam2_formlearner", "formlearner"}:
                    self._v8_form_threshold_spin = QDoubleSpinBox(); self._v8_form_threshold_spin.setRange(0.0, 1.0); self._v8_form_threshold_spin.setDecimals(3); self._v8_form_threshold_spin.setSingleStep(0.01)
                    lay.addWidget(QLabel("FORM ≥"), next_row, 2); lay.addWidget(self._v8_form_threshold_spin, next_row, 3, 1, 2)
                    next_row += 1
                lay.addWidget(QLabel("CLASS"), next_row, 0); lay.addWidget(self._v8_options_edit, next_row, 1, 1, 4)
                self._v8_model_combo.currentTextChanged.connect(lambda *_: _model_changed(self))
                self._v8_model_btn.clicked.connect(lambda *_: _browse_primary(self))
                self._v8_conf_spin.valueChanged.connect(lambda *_: _model_changed(self))
                self._v8_device_combo.currentTextChanged.connect(lambda *_: _model_changed(self))
                self._v8_options_edit.textChanged.connect(lambda *_: _model_changed(self))
                if hasattr(self, "_v8_imgsz_spin"):
                    self._v8_imgsz_spin.valueChanged.connect(lambda *_: _model_changed(self))
                if hasattr(self, "_v8_form_threshold_spin"):
                    self._v8_form_threshold_spin.valueChanged.connect(lambda *_: _model_changed(self))
                self._proxy_type = "model_v9"
            proxy_h = max(98, int(self.block_height() - 92))
            try:
                w.setMinimumHeight(proxy_h); w.setMaximumHeight(proxy_h); w.setFixedHeight(proxy_h)
            except Exception:
                pass
            self.rule_proxy = QGraphicsProxyWidget(self)
            self.rule_proxy.setWidget(w)
            self.rule_proxy.setPos(9, 74)
            self.rule_proxy.resize(self.block_width() - 18, proxy_h)
        if typ in SCIENTIFIC_FILTER_TYPES:
            _refresh_filter_controls(self)
        else:
            _refresh_model_controls(self)
        return None

    vbi.block_width = block_width_v8
    vbi.block_height = block_height_v8
    vbi.refresh_embedded_controls = refresh_embedded_controls_v8
    vbi._mustatil_v9_basic_inline_controls_patched = True

def _patch_visual_block_class(cls: Any) -> None:
    try:
        mod = sys.modules.get(cls.__module__)
        vbi = getattr(mod, "VisualBlockItem", None)
        if vbi is None:
            return
        try:
            from PySide6.QtGui import QColor
            if hasattr(vbi, "TYPE_COLORS"):
                vbi.TYPE_COLORS.update({
                    "sam2_formlearner": QColor(25, 150, 120),
                    "sam2_form_filter": QColor(25, 150, 120),
                    "faster_rcnn": QColor(120, 95, 210),
                    "mask_rcnn": QColor(165, 75, 190),
                    "unet": QColor(40, 165, 125),
                    "scientific_shape_filter": QColor(90, 140, 70),
                    "scientific_size_filter": QColor(110, 150, 80),
                    "scientific_texture_filter": QColor(135, 125, 55),
                    "geo_nms_filter": QColor(170, 110, 50),
                    "cluster_context_filter": QColor(70, 130, 150),
                    "linear_alignment_filter": QColor(60, 110, 170),
                    "scientific_formlearner_filter": QColor(145, 90, 150),
                })
        except Exception:
            pass
        try:
            _patch_visual_block_inline_controls(vbi)
        except Exception:
            pass
        orig_height = getattr(vbi, "block_height", None)
        if callable(orig_height) and not getattr(vbi, "_mustatil_v9_height_patched", False):
            def block_height_patched(self):
                typ = _norm_type(getattr(getattr(self, "block", None), "type", ""))
                # Basic algorithmic filters show only a few parameters and should
                # not waste space.  Model blocks get enough room so no inline
                # combo/button can protrude below the colored block frame.
                if typ in SCIENTIFIC_FILTER_TYPES:
                    return 198.0 if typ != "scientific_formlearner_filter" else 232.0
                if typ == "sam2_formlearner":
                    return 292.0
                if typ in {"sam2", "lae_dino"}:
                    return 272.0
                if typ == "formlearner":
                    return 258.0
                if typ in {"yolo", "owlv2", "grounding_dino", "faster_rcnn", "mask_rcnn", "unet"}:
                    return 248.0
                return orig_height(self)
            vbi.block_height = block_height_patched
            vbi._mustatil_v9_height_patched = True
        try:
            view_cls = getattr(mod, "PipelineGraphicsView", None)
            _patch_view_class(view_cls)
        except Exception:
            pass
    except Exception:
        pass



def _run_passthrough_input_block(tab: Any, b: Any, full: Any, image_path: Path, status: str) -> List[Dict[str, Any]]:
    """Safe fallback so a block never becomes dead when an optional backend is missing."""
    try:
        parents = tab._input_records(b, full) if hasattr(tab, "_input_records") else []
    except Exception:
        parents = []
    if not parents or str(getattr(b, "input_ref", "original")) == "original":
        try:
            tab._log(f"{getattr(b, 'name', 'Block')}: {status}; no upstream records to pass through.")
        except Exception:
            pass
        return []
    out: List[Dict[str, Any]] = []
    for parent in parents:
        rec = dict(parent)
        rec.update({
            "id": _new_id("fallback"),
            "block_id": getattr(b, "id", ""),
            "block_name": getattr(b, "name", ""),
            "block_type": getattr(b, "type", ""),
            "parent_id": parent.get("id"),
            "pipeline_status": status,
            "fallback_passthrough": True,
        })
        out.append(rec)
    try:
        tab._log(f"{getattr(b, 'name', 'Block')}: {status}; passed through {len(out)} upstream record(s).")
    except Exception:
        pass
    return out


def _run_sam2_block_basic_fallback(tab: Any, b: Any, full: Any, image_path: Path) -> List[Dict[str, Any]]:
    return _run_passthrough_input_block(tab, b, full, image_path, "SAM2 backend not available; kept upstream boxes as mask candidates")


def _run_detector_block_missing_backend(tab: Any, b: Any, full: Any, image_path: Path) -> List[Dict[str, Any]]:
    try:
        tab._log(f"{getattr(b, 'name', 'Detector')}: backend method for {getattr(b, 'type', '')} is not loaded. Select/install the matching model plugin, then run again.")
    except Exception:
        pass
    return []

def _patch_pipeline_class(cls: Any) -> bool:
    if cls is None or id(cls) in _PATCHED_CLASSES:
        return False
    _PATCHED_CLASSES.add(id(cls))
    _patch_visual_block_class(cls)

    # Add helper methods directly to the tab class.
    cls._run_torchvision_ai_block = _run_torchvision_ai_block
    cls._run_sam2_formlearner_block = _run_sam2_formlearner_block
    cls._run_scientific_filter_block = _run_scientific_filter_block
    cls._run_formlearner_block = _run_formlearner_block_scientific
    # Minimal safe fallbacks: every block can be executed without crashing.
    # Real model backends are still used whenever the original plugin provides them.
    if not hasattr(cls, "_run_sam2_block"):
        cls._run_sam2_block = _run_sam2_block_basic_fallback
    for _missing_name in ("_run_yolo_block", "_run_owlv2_block", "_run_grounding_dino_block", "_run_lae_dino_block"):
        if not hasattr(cls, _missing_name):
            setattr(cls, _missing_name, _run_detector_block_missing_backend)

    orig_update_from_inline = getattr(cls, "_update_editor_from_inline", None)
    if callable(orig_update_from_inline) and not getattr(cls, "_mustatil_v8_update_inline_patched", False):
        def _update_editor_from_inline_patched(self, block):
            if getattr(self, "selected_block_id", "") != getattr(block, "id", None) or getattr(self, "_updating", False):
                return
            self._updating = True
            try:
                _set_widget_text(getattr(self, "model_path", None), getattr(block, "model_path", ""))
                try:
                    self.conf.setValue(float(getattr(block, "confidence", 0.05) or 0.05))
                except Exception:
                    pass
                try:
                    self.imgsz.setValue(int(getattr(block, "imgsz", 640) or 640))
                except Exception:
                    pass
                try:
                    self.device.setCurrentText(str(getattr(block, "device", "cpu") or "cpu"))
                except Exception:
                    pass
                try:
                    self.classes_filter.setText(str(getattr(block, "classes_filter", "") or ""))
                except Exception:
                    pass
                # Important: for SAM2+FormLearner the model_path is the SAM2 checkpoint;
                # the companion field must stay the FormLearner JSON, not fall back to model_path.
                _set_widget_text(getattr(self, "form_model", None), getattr(block, "formlearner_model", "") or "")
                try:
                    self.form_threshold.setValue(float(getattr(block, "form_threshold", 0.5) or 0.5))
                except Exception:
                    pass
            finally:
                self._updating = False
            try:
                _update_clean_right_panel_for_block(self, block)
            except Exception:
                pass
        cls._update_editor_from_inline = _update_editor_from_inline_patched
        cls._mustatil_v8_update_inline_patched = True

    orig_load_editor = getattr(cls, "_load_block_to_editor", None)
    if callable(orig_load_editor) and not getattr(cls, "_mustatil_v8_load_editor_dynamic_patched", False):
        def _load_block_to_editor_v8(self, block, *args, **kwargs):
            ret = orig_load_editor(self, block, *args, **kwargs)
            try:
                _update_clean_right_panel_for_block(self, block)
            except Exception:
                pass
            return ret
        cls._load_block_to_editor = _load_block_to_editor_v8
        cls._mustatil_v8_load_editor_dynamic_patched = True

    orig_add_block = getattr(cls, "_add_block", None)
    if callable(orig_add_block) and not getattr(cls, "_mustatil_v8_fullblocks_add_patched", False):
        def _add_block_patched(self, typ="yolo", name=None, x=None, y=None, input_ref=None):
            alias = _norm_type(typ)
            default_name = name or DEFAULT_NAMES.get(alias, DEFAULT_NAMES.get(str(typ), "AI model block"))
            b = orig_add_block(self, alias, name=default_name, x=x, y=y, input_ref=input_ref)
            try:
                b.type = alias
                if alias in {"faster_rcnn", "mask_rcnn"}:
                    b.confidence = float(getattr(b, "confidence", 0.25) or 0.25)
                    b.device = str(getattr(b, "device", "cuda") or "cuda")
                    b.tile_size = max(768, int(getattr(b, "tile_size", getattr(b, "imgsz", 1024)) or 1024))
                    b.overlap = max(96, int(getattr(b, "overlap", 128) or 128))
                    b.use_chunked_yolo = True
                elif alias == "unet":
                    b.confidence = 0.50
                    b.device = str(getattr(b, "device", "cuda") or "cuda")
                    b.tile_size = max(768, int(getattr(b, "tile_size", getattr(b, "imgsz", 1024)) or 1024))
                    b.overlap = max(96, int(getattr(b, "overlap", 128) or 128))
                    b.use_chunked_yolo = True
                    b.classes_filter = "segmentation"
                elif alias == "sam2_formlearner":
                    b.confidence = max(0.05, float(getattr(b, "confidence", 0.50) or 0.50))
                    b.crop_padding = max(16, int(getattr(b, "crop_padding", 16) or 16))
                    b.use_chunked_yolo = False
                    b.classes_filter = str(getattr(b, "classes_filter", "") or "sam2_form_filter")
                    b.form_threshold = float(getattr(b, "form_threshold", 0.50) or 0.50)
                    if not getattr(b, "input_ref", "original") or getattr(b, "input_ref", "original") == "original":
                        try:
                            if len(getattr(self, "blocks", []) or []) > 1:
                                # The block was appended already; connect to the previous block by default.
                                prev = [x for x in self.blocks if x.id != b.id][-1]
                                b.input_ref = prev.id
                        except Exception:
                            pass
                elif alias in SCIENTIFIC_FILTER_TYPES:
                    b.confidence = float(getattr(b, "confidence", 0.0) or 0.0)
                    b.device = str(getattr(b, "device", "cpu") or "cpu")
                    b.use_chunked_yolo = False
                    b.form_threshold = float(getattr(b, "form_threshold", 0.50) or 0.50)
                    if not str(getattr(b, "classes_filter", "") or "").strip():
                        b.classes_filter = ""
                    _apply_basic_filter_defaults(b)
                    if not getattr(b, "input_ref", "original") or getattr(b, "input_ref", "original") == "original":
                        try:
                            if len(getattr(self, "blocks", []) or []) > 1:
                                prev = [x for x in self.blocks if x.id != b.id][-1]
                                b.input_ref = prev.id
                        except Exception:
                            pass
                try:
                    if hasattr(self, "refresh_code_from_blocks"):
                        self.refresh_code_from_blocks()
                    if hasattr(self, "_load_block_to_editor"):
                        self._load_block_to_editor(b)
                except Exception:
                    pass
            except Exception:
                pass
            return b
        cls._add_block = _add_block_patched
        cls._mustatil_v8_fullblocks_add_patched = True

    # Patch execution. When no enhanced block exists, preserve the currently installed original run method.
    orig_run = getattr(cls, "run_pipeline", None)
    if callable(orig_run) and not getattr(cls, "_mustatil_v8_fullblocks_run_patched", False):
        def run_pipeline_patched(self):
            try:
                has_enhanced = any(_norm_type(getattr(b, "type", "")) in ENHANCED_TYPES for b in list(getattr(self, "blocks", []) or []))
                if not has_enhanced:
                    return orig_run(self)
            except Exception:
                pass
            image_path = self._current_pipeline_image_path() if hasattr(self, "_current_pipeline_image_path") else Path(str(_get_var(getattr(self.ws, "image", None), "") or "").strip().strip('"')).expanduser()
            if not image_path.is_file():
                raise RuntimeError("Choose a valid image at the top of the AI Pipeline tab first. This image is used as the Original image block.")
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            self.results = []
            self.results_by_block = {}
            self._log(f"Visual pipeline started: {image_path}")
            full = Image.open(image_path)
            try:
                self._mustatil_current_full_width = int(full.width)
                self._mustatil_current_full_height = int(full.height)
            except Exception:
                pass
            order = self._execution_order()
            self._log("Execution order: " + " → ".join(b.name for b in order))
            for b in order:
                typ = _norm_type(getattr(b, "type", ""))
                try:
                    if typ == "yolo" and hasattr(self, "_run_yolo_block"):
                        out = self._run_yolo_block(b, full, image_path)
                    elif typ in {"owlv2", "owl", "owl_v2"} and hasattr(self, "_run_owlv2_block"):
                        out = self._run_owlv2_block(b, full, image_path)
                    elif typ in {"grounding_dino", "grounding", "gdino"} and hasattr(self, "_run_grounding_dino_block"):
                        out = self._run_grounding_dino_block(b, full, image_path)
                    elif typ in {"lae_dino", "laedino"} and hasattr(self, "_run_lae_dino_block"):
                        out = self._run_lae_dino_block(b, full, image_path)
                    elif typ in {"sam2", "sam", "sam_2"} and hasattr(self, "_run_sam2_block"):
                        out = self._run_sam2_block(b, full, image_path)
                    elif typ == "sam2_formlearner":
                        out = self._run_sam2_formlearner_block(b, full, image_path)
                    elif typ in {"faster_rcnn", "mask_rcnn", "unet"}:
                        out = self._run_torchvision_ai_block(b, full, image_path)
                    elif typ in SCIENTIFIC_FILTER_TYPES:
                        out = self._run_scientific_filter_block(b, full, image_path)
                    elif typ == "formlearner" and hasattr(self, "_run_formlearner_block"):
                        out = self._run_formlearner_block(b, full)
                    elif typ == "rule" and hasattr(self, "_run_rule_block"):
                        out = self._run_rule_block(b)
                    else:
                        self._log(f"Unknown block type skipped: {getattr(b, 'name', '')} / {getattr(b, 'type', '')}")
                        out = []
                except Exception as exc:
                    self._log(f"{getattr(b, 'name', typ)} failed but pipeline continues: {exc}")
                    try:
                        traceback.print_exc()
                    except Exception:
                        pass
                    out = []
                self.results_by_block[b.id] = out
                self.results.extend(out)
                self._log(f"{b.name}: {len(out)} result(s)")
            self._log(f"Visual pipeline finished. Total records: {len(self.results)}")
            try:
                self._write_summary()
            except Exception:
                pass
            return self.results
        cls.run_pipeline = run_pipeline_patched
        cls._mustatil_v8_fullblocks_run_patched = True

    # v8 compatibility fix: older patch versions installed select_block(block_id) only.
    # Qt scene items call select_block(block_id, from_scene=True), so this wrapper accepts
    # both positional and keyword forms and safely falls back when wrapping an older wrapper.
    orig_select = getattr(cls, "select_block", None)
    if callable(orig_select) and not getattr(cls, "_mustatil_v8_fullblocks_select_patched", False):
        def select_block_patched(self, block_id: str, *args, **kwargs):
            ret = None
            called = False
            try:
                ret = orig_select(self, block_id, *args, **kwargs)
                called = True
            except TypeError as exc:
                msg = str(exc)
                if "from_scene" in msg or "unexpected keyword" in msg or "positional" in msg:
                    try:
                        ret = orig_select(self, block_id)
                        called = True
                    except Exception:
                        called = False
                else:
                    raise
            except Exception:
                called = False
            if not called:
                try:
                    b = None
                    if hasattr(self, "_block_by_id"):
                        b = self._block_by_id(block_id)
                    else:
                        for cand in list(getattr(self, "blocks", []) or []):
                            if getattr(cand, "id", None) == block_id:
                                b = cand
                                break
                    if b is not None:
                        self.selected_block_id = getattr(b, "id", block_id)
                        if not bool(kwargs.get("from_scene", False)):
                            for bid, item in getattr(self, "block_items", {}).items():
                                try:
                                    item.setSelected(bid == getattr(b, "id", block_id))
                                except Exception:
                                    pass
                        if hasattr(self, "_load_block_to_editor"):
                            self._load_block_to_editor(b)
                except Exception:
                    pass
            try:
                _patch_pipeline_instance(self)
                _extend_type_combo(self)
                _ensure_code_console_ui(self)
                b = self._current_block() if hasattr(self, "_current_block") else None
                if b is not None and hasattr(self, "typ"):
                    ix = self.typ.findText(str(getattr(b, "type", "")))
                    if ix >= 0:
                        self.typ.blockSignals(True)
                        self.typ.setCurrentIndex(ix)
                        self.typ.blockSignals(False)
            except Exception:
                pass
            return ret
        cls.select_block = select_block_patched
        cls._mustatil_v8_fullblocks_select_patched = True

    return True



# -----------------------------------------------------------------------------
# v8 code-view and Python console utilities
# -----------------------------------------------------------------------------


def _pipeline_data_for_code(tab: Any) -> Dict[str, Any]:
    try:
        if hasattr(tab, "pipeline_to_dict"):
            return tab.pipeline_to_dict()
    except Exception:
        pass
    blocks = []
    try:
        for b in list(getattr(tab, "blocks", []) or []):
            d = {}
            for k in (
                "id", "name", "type", "input_ref", "model_path", "formlearner_model",
                "confidence", "imgsz", "device", "classes_filter", "form_threshold",
                "rule_json", "x", "y",
            ):
                if hasattr(b, k):
                    d[k] = getattr(b, k)
            blocks.append(d)
    except Exception:
        pass
    return {"version": 6, "blocks": blocks}


def _pipeline_executable_python_code(tab: Any) -> str:
    # Generate real Python dispatch code matching the patched visual run loop.
    import pprint
    data = _pipeline_data_for_code(tab)
    body = pprint.pformat(data, width=120, sort_dicts=False)
    return f'''# Mustatil AI Pipeline - executable Python dispatch code
# Generated from the current visual block graph.
# In the Mustatil AI Pipeline console, variables are available:
#   tab, ws, blocks, results, pipeline, run_pipeline()
# Outside that console, set `tab` to the current YoloPipelineTab before running.

from pathlib import Path

pipeline = {body}


def _norm_type(t):
    raw = str(t or "").strip().lower()
    norm = raw.replace(" ", "_").replace("-", "_").replace("+", "_")
    aliases = {{
        "faster": "faster_rcnn", "faster_r_cnn": "faster_rcnn", "faster-r-cnn": "faster_rcnn", "faster r-cnn": "faster_rcnn",
        "mask": "mask_rcnn", "mask_r_cnn": "mask_rcnn", "mask-r-cnn": "mask_rcnn", "mask r-cnn": "mask_rcnn",
        "u_net": "unet", "u-net": "unet", "sam2_form_filter": "sam2_formlearner", "sam2+formlearner": "sam2_formlearner",
    }}
    return aliases.get(raw) or aliases.get(norm) or norm


def execute_visual_pipeline(tab=None):
    # Execute the visual pipeline using the same Mustatil block methods as the tab.
    tab = tab or globals().get("tab") or globals().get("pipeline_tab")
    if tab is None:
        raise RuntimeError("No pipeline tab object was supplied. Run this in the Mustatil Pipeline Console or call execute_visual_pipeline(tab).")

    image_path = tab._current_pipeline_image_path() if hasattr(tab, "_current_pipeline_image_path") else Path("")
    if not Path(image_path).is_file():
        raise RuntimeError("Choose a valid image at the top of the AI Pipeline tab first.")

    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    full = Image.open(image_path)
    tab.results = []
    tab.results_by_block = {{}}

    order = tab._execution_order() if hasattr(tab, "_execution_order") else list(getattr(tab, "blocks", []) or [])
    for b in order:
        typ = _norm_type(getattr(b, "type", ""))
        if typ == "yolo" and hasattr(tab, "_run_yolo_block"):
            out = tab._run_yolo_block(b, full, image_path)
        elif typ in {{"owlv2", "owl", "owl_v2"}} and hasattr(tab, "_run_owlv2_block"):
            out = tab._run_owlv2_block(b, full, image_path)
        elif typ in {{"grounding_dino", "grounding", "gdino"}} and hasattr(tab, "_run_grounding_dino_block"):
            out = tab._run_grounding_dino_block(b, full, image_path)
        elif typ in {{"lae_dino", "laedino"}} and hasattr(tab, "_run_lae_dino_block"):
            out = tab._run_lae_dino_block(b, full, image_path)
        elif typ in {{"sam2", "sam", "sam_2"}} and hasattr(tab, "_run_sam2_block"):
            out = tab._run_sam2_block(b, full, image_path)
        elif typ == "sam2_formlearner" and hasattr(tab, "_run_sam2_formlearner_block"):
            out = tab._run_sam2_formlearner_block(b, full, image_path)
        elif typ in {{"faster_rcnn", "mask_rcnn", "unet"}} and hasattr(tab, "_run_torchvision_ai_block"):
            out = tab._run_torchvision_ai_block(b, full, image_path)
        elif typ in {{"scientific_shape_filter", "scientific_size_filter", "scientific_texture_filter", "geo_nms_filter", "cluster_context_filter", "linear_alignment_filter", "scientific_formlearner_filter"}} and hasattr(tab, "_run_scientific_filter_block"):
            out = tab._run_scientific_filter_block(b, full, image_path)
        elif typ == "formlearner" and hasattr(tab, "_run_formlearner_block"):
            out = tab._run_formlearner_block(b, full)
        elif typ == "rule" and hasattr(tab, "_run_rule_block"):
            out = tab._run_rule_block(b)
        else:
            out = []
            if hasattr(tab, "_log"):
                tab._log(f"Skipped unknown/unavailable block: {{getattr(b, 'name', '')}} / {{getattr(b, 'type', '')}}")

        tab.results_by_block[getattr(b, "id", str(len(tab.results_by_block)))] = out
        tab.results.extend(out)
        if hasattr(tab, "_log"):
            tab._log(f"{{getattr(b, 'name', typ)}}: {{len(out)}} result(s)")

    if hasattr(tab, "_write_summary"):
        tab._write_summary()
    return tab.results


# Run manually when wanted:
# results = execute_visual_pipeline(tab)
'''


def _console_env(tab: Any) -> Dict[str, Any]:
    env = dict(getattr(tab, "_mustatil_v8_console_env", {}) or {})
    env.update({
        "tab": tab,
        "pipeline_tab": tab,
        "ws": getattr(tab, "ws", None),
        "blocks": getattr(tab, "blocks", []),
        "results": getattr(tab, "results", []),
        "pipeline": _pipeline_data_for_code(tab),
        "run_pipeline": getattr(tab, "run_pipeline", lambda: None),
        "Path": Path,
        "json": json,
        "math": math,
    })
    tab._mustatil_v8_console_env = env
    return env


def _append_console_output(tab: Any, text: str) -> None:
    text = str(text or "").rstrip()
    if not text:
        return
    try:
        if hasattr(tab, "summary") and tab.summary is not None:
            tab.summary.append("[Python Console] " + text)
            return
    except Exception:
        pass
    try:
        tab._log("[Python Console] " + text)
    except Exception:
        _log("[Python Console] " + text)


def _run_console_command(tab: Any) -> None:
    try:
        edit = getattr(tab, "_mustatil_v8_console_input", None)
        if edit is None:
            return
        code = str(edit.text() if hasattr(edit, "text") else edit.toPlainText()).strip()
        if not code:
            return
        if hasattr(edit, "clear"):
            edit.clear()
        env = _console_env(tab)
        import io
        import contextlib
        buf = io.StringIO()
        result = None
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                compiled = compile(code, "<Mustatil AI Pipeline Console>", "eval")
                result = eval(compiled, env, env)
            except SyntaxError:
                exec(compile(code, "<Mustatil AI Pipeline Console>", "exec"), env, env)
        out = buf.getvalue().rstrip()
        if result is not None:
            out = (out + "\n" if out else "") + repr(result)
        _append_console_output(tab, out or "ok")
    except Exception as exc:
        _append_console_output(tab, "ERROR: " + str(exc))
        try:
            traceback.print_exc()
        except Exception:
            pass


def _refresh_code_view_v8(tab: Any) -> None:
    try:
        mode = "data"
        combo = getattr(tab, "_mustatil_v8_code_mode", None)
        if combo is not None:
            mode = str(combo.currentData() or combo.currentText() or "data").lower()
        if not hasattr(tab, "pipeline_code"):
            return
        if mode.startswith("exec") or "python" in mode:
            tab.pipeline_code.setPlainText(_pipeline_executable_python_code(tab))
        else:
            orig = getattr(tab, "_mustatil_v8_orig_refresh_code", None)
            if callable(orig):
                return orig()
            if hasattr(tab, "_pipeline_to_python_code"):
                tab.pipeline_code.setPlainText(tab._pipeline_to_python_code())
            else:
                import pprint
                tab.pipeline_code.setPlainText("pipeline = " + pprint.pformat(_pipeline_data_for_code(tab), width=120, sort_dicts=False))
    except Exception as exc:
        try:
            tab._log(f"Could not refresh code view: {exc}")
        except Exception:
            pass


def _run_code_editor_python(tab: Any) -> None:
    try:
        if not hasattr(tab, "pipeline_code"):
            return
        code = str(tab.pipeline_code.toPlainText() or "")
        env = _console_env(tab)
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(compile(code, "<Mustatil AI Pipeline Code View>", "exec"), env, env)
        _append_console_output(tab, buf.getvalue().rstrip() or "Python code loaded. Run: execute_visual_pipeline(tab)")
    except Exception as exc:
        _append_console_output(tab, "ERROR in code view: " + str(exc))


def _patch_code_methods(tab: Any) -> None:
    try:
        cls = tab.__class__
        if not getattr(cls, "_mustatil_v8_code_console_methods_patched", False):
            orig_refresh = getattr(cls, "refresh_code_from_blocks", None)
            orig_apply = getattr(cls, "apply_code_to_blocks", None)
            orig_toggle = getattr(cls, "toggle_block_code_view", None)

            def refresh_code_from_blocks_v8(self):
                return _refresh_code_view_v8(self)

            def apply_code_to_blocks_v8(self, silent: bool = False):
                mode = "data"
                try:
                    combo = getattr(self, "_mustatil_v8_code_mode", None)
                    if combo is not None:
                        mode = str(combo.currentData() or combo.currentText() or "data").lower()
                except Exception:
                    mode = "data"
                if mode.startswith("exec") or "python" in mode:
                    if not silent:
                        _append_console_output(self, "Executable Python mode is not converted back to blocks. Use Run Python code or switch to Pipeline data to edit/apply blocks.")
                    return True
                if callable(orig_apply):
                    return orig_apply(self, silent=silent)
                return True

            def toggle_block_code_view_v8(self):
                try:
                    if not hasattr(self, "mode_stack"):
                        return None
                    if self.mode_stack.currentIndex() == 0:
                        self.refresh_code_from_blocks()
                        self.mode_stack.setCurrentIndex(1)
                        if hasattr(self, "view_mode_btn"):
                            self.view_mode_btn.setText("View: Blocks")
                    else:
                        mode = "data"
                        combo = getattr(self, "_mustatil_v8_code_mode", None)
                        if combo is not None:
                            mode = str(combo.currentData() or combo.currentText() or "data").lower()
                        if mode.startswith("exec") or "python" in mode or self.apply_code_to_blocks(silent=False):
                            self.mode_stack.setCurrentIndex(0)
                            if hasattr(self, "view_mode_btn"):
                                self.view_mode_btn.setText("View: Code")
                except Exception:
                    if callable(orig_toggle):
                        return orig_toggle(self)
                return None

            cls._mustatil_v8_orig_refresh_code = orig_refresh
            cls.refresh_code_from_blocks = refresh_code_from_blocks_v8
            cls.apply_code_to_blocks = apply_code_to_blocks_v8
            cls.toggle_block_code_view = toggle_block_code_view_v8
            cls._mustatil_v8_code_console_methods_patched = True
        if getattr(tab, "_mustatil_v8_orig_refresh_code", None) is None:
            orig = getattr(tab.__class__, "_mustatil_v8_orig_refresh_code", None)
            if callable(orig):
                tab._mustatil_v8_orig_refresh_code = lambda: orig(tab)
    except Exception:
        pass


def _ensure_code_console_ui(tab: Any) -> None:
    try:
        _patch_code_methods(tab)
    except Exception:
        pass
    if tab is None or not hasattr(tab, "pipeline_code"):
        return
    try:
        from PySide6.QtWidgets import (
            QHBoxLayout, QLabel, QPushButton, QComboBox,
            QLineEdit, QGroupBox, QSizePolicy
        )
    except Exception:
        return
    try:
        if getattr(tab, "_mustatil_v8_code_console_ui_installed", False):
            return
        code_edit = getattr(tab, "pipeline_code")
        page = code_edit.parentWidget()
        lay = page.layout() if page is not None else None
        if lay is None:
            return

        top = QGroupBox("Code mode")
        top.setObjectName("MustatilPipelineCodeModeV8")
        top_lay = QHBoxLayout(top)
        top_lay.setContentsMargins(8, 5, 8, 5)
        top_lay.setSpacing(6)
        top_lay.addWidget(QLabel("Mode"))
        mode = QComboBox()
        mode.addItem("Pipeline data editor", "data")
        mode.addItem("Executable Python dispatch code", "executable_python")
        mode.setMinimumWidth(230)
        top_lay.addWidget(mode)
        refresh_btn = QPushButton("Refresh")
        apply_btn = QPushButton("Apply data to blocks")
        run_btn = QPushButton("Run Python code")
        copy_btn = QPushButton("Copy")
        top_lay.addWidget(refresh_btn)
        top_lay.addWidget(apply_btn)
        top_lay.addWidget(run_btn)
        top_lay.addWidget(copy_btn)
        top_lay.addStretch(1)
        try:
            lay.insertWidget(0, top)
        except Exception:
            lay.addWidget(top)

        console = QGroupBox("Python console input")
        console.setObjectName("MustatilPipelineConsoleV8")
        console_lay = QHBoxLayout(console)
        console_lay.setContentsMargins(8, 5, 8, 5)
        console_lay.setSpacing(6)
        console_lay.addWidget(QLabel(">>>"))
        line = QLineEdit()
        line.setPlaceholderText("Python in pipeline context, e.g. len(blocks), blocks[0].type, run_pipeline()")
        line.setMinimumWidth(520)
        line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        console_lay.addWidget(line, 1)
        run_console = QPushButton("Run")
        console_lay.addWidget(run_console)
        lay.addWidget(console)

        tab._mustatil_v8_code_mode = mode
        tab._mustatil_v8_console_input = line
        tab._mustatil_v8_code_console_ui_installed = True

        mode.currentIndexChanged.connect(lambda *_: tab.refresh_code_from_blocks())
        refresh_btn.clicked.connect(lambda *_: tab.refresh_code_from_blocks())
        apply_btn.clicked.connect(lambda *_: tab.apply_code_to_blocks(silent=False))
        run_btn.clicked.connect(lambda *_: _run_code_editor_python(tab))
        copy_btn.clicked.connect(lambda *_: code_edit.selectAll() or code_edit.copy())
        run_console.clicked.connect(lambda *_: _run_console_command(tab))
        line.returnPressed.connect(lambda: _run_console_command(tab))

        try:
            code_edit.setMinimumHeight(460)
        except Exception:
            pass
        tab.refresh_code_from_blocks()
    except Exception:
        traceback.print_exc()


def _extend_type_combo(tab: Any) -> None:
    try:
        combo = getattr(tab, "typ", None)
        if combo is None or not hasattr(combo, "addItem"):
            return
        existing = {str(combo.itemText(i)) for i in range(combo.count())}
        for _label, typ in AI_CHOICES:
            if typ not in existing:
                combo.addItem(typ)
                existing.add(typ)
    except Exception:
        pass


def _patch_model_dropdown(tab: Any) -> None:
    # v2 deliberately uses only one clean AI model picker and removes older duplicates.
    try:
        _install_single_clean_toolbar(tab)
    except Exception:
        traceback.print_exc()


def _patch_pipeline_instance(tab: Any) -> bool:
    if tab is None:
        return False
    try:
        _patch_pipeline_class(tab.__class__)
    except Exception:
        pass
    try:
        _extend_type_combo(tab)
        _patch_model_dropdown(tab)
        _polish_right_side(tab)
        _ensure_code_console_ui(tab)
        try:
            ws = getattr(tab, "ws", None)
            if ws is not None:
                _install_formlearner_scientific_training_ui(ws)
        except Exception:
            pass
        try:
            view = getattr(tab, "view", None)
            if view is not None:
                _patch_view_class(view.__class__)
        except Exception:
            pass
    except Exception:
        traceback.print_exc()
    if id(tab) not in _PATCHED_TABS:
        _PATCHED_TABS.add(id(tab))
        try:
            tab._log("AI Pipeline scientific filters v8 installed: one AI model picker, full block dispatch, scientific filters, robust model buttons, no-overflow blocks and scrollable grouped right editor.")
        except Exception:
            pass
        try:
            ws = getattr(tab, "ws", None)
            if ws is not None:
                _ws_log(ws, "Existing AI Pipeline cleaned in-place; no separate tab and no duplicate AI model controls were added.")
        except Exception:
            pass
        return True
    return False



def _record_from_any(obj: Any) -> Optional[Dict[str, Any]]:
    if isinstance(obj, dict):
        d = dict(obj)
    else:
        d = {}
        for k in ("x1", "y1", "x2", "y2", "xmin", "ymin", "xmax", "ymax", "bbox", "conf", "class_id", "label"):
            try:
                if hasattr(obj, k):
                    d[k] = getattr(obj, k)
            except Exception:
                pass
    if "bbox" in d and isinstance(d["bbox"], (list, tuple)) and len(d["bbox"]) >= 4:
        d.setdefault("x1", d["bbox"][0]); d.setdefault("y1", d["bbox"][1]); d.setdefault("x2", d["bbox"][2]); d.setdefault("y2", d["bbox"][3])
    if "xmin" in d: d.setdefault("x1", d.get("xmin"))
    if "ymin" in d: d.setdefault("y1", d.get("ymin"))
    if "xmax" in d: d.setdefault("x2", d.get("xmax"))
    if "ymax" in d: d.setdefault("y2", d.get("ymax"))
    if all(k in d for k in ("x1", "y1", "x2", "y2")):
        try:
            if abs(float(d["x2"]) - float(d["x1"])) > 0 and abs(float(d["y2"]) - float(d["y1"])) > 0:
                return d
        except Exception:
            pass
    return None

def _records_for_scientific_training(ws: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    sources = []
    try:
        tab = getattr(ws, "yolo_pipeline_tab", None)
        if tab is not None:
            sources.append(getattr(tab, "results", []))
    except Exception:
        pass
    for attr in ("detections", "dets", "satellite_detections", "annotation_boxes", "boxes", "visible_detections"):
        try:
            val = getattr(ws, attr, None)
            if hasattr(val, "get"):
                val = val.get()
            if val:
                sources.append(val)
        except Exception:
            pass
    for seq in sources:
        try:
            for obj in list(seq or []):
                r = _record_from_any(obj)
                if r is not None:
                    out.append(r)
        except Exception:
            continue
    # Deduplicate identical bboxes
    seen = set(); clean = []
    for r in out:
        key = tuple(round(_safe_float(r.get(k), 0.0), 2) for k in ("x1", "y1", "x2", "y2"))
        if key not in seen:
            seen.add(key); clean.append(r)
    return clean

def _current_training_image(ws: Any):
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        for attr in ("image", "satellite_image", "current_image", "pipeline_image"):
            try:
                obj = getattr(ws, attr, None)
                val = obj.get() if hasattr(obj, "get") else obj
                pp = Path(str(val or "").strip().strip('"')).expanduser()
                if pp.is_file():
                    return Image.open(pp)
            except Exception:
                pass
    except Exception:
        pass
    return None

def _percentile(values: List[float], q: float, default: float = 0.0) -> float:
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not vals:
        return default
    if len(vals) == 1:
        return vals[0]
    pos = max(0.0, min(1.0, q)) * (len(vals) - 1)
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)

def _build_scientific_preset_from_workspace(ws: Any, filter_type: str) -> Dict[str, Any]:
    typ = _norm_type(filter_type)
    records = _records_for_scientific_training(ws)
    img = _current_training_image(ws)
    if img is None:
        # Synthetic one-pixel image fallback still lets size/shape/context presets work.
        try:
            from PIL import Image
            img = Image.new("RGB", (8, 8), "white")
        except Exception:
            img = None
    if not records:
        # Use default basic params rather than failing; this keeps the button functional.
        params = {k: default for k, default, _a, _b, _d in BASIC_FILTER_PARAM_SPECS.get(typ, [])}
        return {"filter_type": typ, "params": params, "trained": False, "reason": "No current detections/boxes found; defaults were saved."}
    scored = []
    for r in records:
        rec = dict(r)
        try:
            f = _scientific_features_for_record(img, rec, {}) if img is not None else {}
        except Exception:
            f = {}
        rec["scientific_features"] = f
        rec.update(f)
        scored.append(rec)
    try:
        _add_context_features(scored, {"radius": 250})
        _add_alignment_features(scored, {"radius": 450, "min_neighbors": 2})
    except Exception:
        pass
    fs = [r.get("scientific_features", {}) for r in scored]
    params: Dict[str, Any] = {}
    if typ == "scientific_shape_filter":
        params = {
            "area_min": max(1.0, _percentile([f.get("area_px", 1) for f in fs], 0.05, 16)),
            "aspect_max": max(1.0, _percentile([f.get("aspect_ratio", 1) for f in fs], 0.95, 12)),
            "fill_min": max(0.0, min(1.0, _percentile([f.get("mask_fill_ratio", 0.05) for f in fs], 0.05, 0.05))),
            "threshold": max(0.0, min(1.0, _percentile([f.get("shape_score", 0.2) for f in fs], 0.15, 0.2))),
        }
    elif typ == "scientific_size_filter":
        params = {
            "area_min": max(1.0, _percentile([f.get("area_px", 1) for f in fs], 0.05, 16)),
            "area_max": max(1.0, _percentile([f.get("area_px", 1) for f in fs], 0.95, 999999999)),
        }
    elif typ == "scientific_texture_filter":
        params = {
            "contrast_min": max(0.0, min(1.0, _percentile([f.get("local_contrast", 0.03) for f in fs], 0.20, 0.03))),
            "std_min": max(0.0, _percentile([f.get("local_std", 2.0) for f in fs], 0.20, 2.0)),
        }
    elif typ == "geo_nms_filter":
        params = {"iou": 0.60}
    elif typ == "cluster_context_filter":
        nearest = [f.get("nearest_distance_px", 250) for f in fs if f.get("nearest_distance_px", 1e9) < 1e8]
        params = {"radius": max(50.0, min(5000.0, _percentile(nearest, 0.50, 250) * 1.5)), "neighbors_min": 1}
    elif typ == "linear_alignment_filter":
        nearest = [f.get("nearest_distance_px", 300) for f in fs if f.get("nearest_distance_px", 1e9) < 1e8]
        params = {"radius": max(75.0, min(8000.0, _percentile(nearest, 0.50, 300) * 2.0)), "line_score_min": max(0.1, min(0.95, _percentile([f.get("line_score", 0.55) for f in fs], 0.40, 0.55)))}
    else:
        params = {k: default for k, default, _a, _b, _d in BASIC_FILTER_PARAM_SPECS.get(typ, [])}
    # Round for clean GUI and readable presets.
    for k, v in list(params.items()):
        if isinstance(v, float):
            params[k] = round(v, 4 if abs(v) < 10 else 1)
    return {
        "filter_type": typ,
        "scientific_feature_set": "morphometry_texture_context_v1",
        "trained": True,
        "sample_count": len(scored),
        **params,
    }

def _install_formlearner_scientific_training_ui(ws: Any) -> None:
    """Add a small preset trainer/calibrator to the existing FormLearner/FormTrainer tab.

    The algorithmic filters are not black-box ML models. Training here means: derive
    practical thresholds from the user's current positive examples/detections and save
    them as JSON presets that the front pipeline filters can load.
    """
    if ws is None or getattr(ws, "_mustatil_v8_formtrainer_science_ui_installed", False):
        return
    try:
        from PySide6.QtWidgets import QApplication, QTabWidget, QWidget, QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QFileDialog
    except Exception:
        return
    try:
        app = QApplication.instance()
        candidates: List[Any] = []
        tabs = getattr(ws, "tabs", None)
        if isinstance(tabs, QTabWidget):
            for i in range(tabs.count()):
                title = str(tabs.tabText(i) or "").lower()
                if "form" in title and ("learner" in title or "trainer" in title):
                    candidates.append(tabs.widget(i))
        if app is not None:
            for w in app.allWidgets():
                try:
                    if isinstance(w, QWidget):
                        name = (str(w.objectName() or "") + " " + str(getattr(w, "windowTitle", lambda: "")())).lower()
                        if "form" in name and ("learner" in name or "trainer" in name):
                            candidates.append(w)
                except Exception:
                    pass
        target = None
        for c in candidates:
            try:
                if c is not None and c.layout() is not None:
                    target = c; break
            except Exception:
                pass
        if target is None:
            return
        lay = target.layout()
        box = QGroupBox("Scientific filter training presets")
        box.setObjectName("MustatilFormTrainerScientificPresetsV8")
        bl = QVBoxLayout(box); bl.setContentsMargins(10,8,10,8); bl.setSpacing(6)
        txt = QLabel("Calibrate the basic algorithmic pipeline filters from the current positive boxes/results. The saved JSON can be selected in a filter block at the front of the AI Pipeline.")
        txt.setWordWrap(True); bl.addWidget(txt)
        row = QHBoxLayout(); row.addWidget(QLabel("Filter"))
        combo = QComboBox()
        for label, typ in AI_CHOICES:
            if typ in ALGORITHM_ONLY_FILTER_TYPES:
                combo.addItem(label.replace("Scientific ", ""), typ)
        row.addWidget(combo, 1)
        save_btn = QPushButton("Train / save JSON presetâ¦")
        apply_btn = QPushButton("Use preset in selected pipeline filter")
        row.addWidget(save_btn); row.addWidget(apply_btn)
        bl.addLayout(row)
        status = QLabel(""); status.setWordWrap(True); bl.addWidget(status)
        ws._mustatil_v8_last_scientific_preset = ""

        def save_preset():
            typ = str(combo.currentData() or combo.currentText())
            data = _build_scientific_preset_from_workspace(ws, typ)
            default_name = f"mustatil_{typ}_preset.json"
            path, _ = QFileDialog.getSaveFileName(target, "Save scientific filter preset", default_name, "JSON preset (*.json);;All files (*)")
            if not path:
                return
            Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            ws._mustatil_v8_last_scientific_preset = path
            status.setText(f"Saved preset: {path} | samples={data.get('sample_count', 0)}")
            try:
                _ws_log(ws, f"Scientific filter preset saved: {path}")
            except Exception:
                pass

        def apply_preset():
            path = str(getattr(ws, "_mustatil_v8_last_scientific_preset", "") or "").strip()
            if not path:
                path, _ = QFileDialog.getOpenFileName(target, "Choose scientific filter preset", "", "JSON preset (*.json);;All files (*)")
            if not path:
                return
            tab = getattr(ws, "yolo_pipeline_tab", None)
            b = _current_block_for_tab(tab) if tab is not None else None
            if b is None or _norm_type(getattr(b, "type", "")) not in ALGORITHM_ONLY_FILTER_TYPES:
                status.setText("Select an algorithmic scientific filter block in the AI Pipeline first.")
                return
            b.model_path = path
            b.classes_filter = _replace_key_path(getattr(b, "classes_filter", ""), "preset", path)
            try:
                tab._load_block_to_editor(b); _update_clean_right_panel_for_block(tab, b); tab.refresh_code_from_blocks()
                if getattr(b, "id", None) in getattr(tab, "block_items", {}):
                    tab.block_items[b.id].refresh_embedded_controls()
                status.setText("Preset applied to selected pipeline filter.")
            except Exception:
                status.setText("Preset saved/selected, but the AI Pipeline tab was not reachable.")

        save_btn.clicked.connect(save_preset)
        apply_btn.clicked.connect(apply_preset)
        try:
            lay.addWidget(box)
        except Exception:
            lay.insertWidget(max(0, lay.count() - 1), box)
        ws._mustatil_v8_formtrainer_science_ui_installed = True
    except Exception:
        traceback.print_exc()


def _find_pipeline_instances() -> List[Any]:
    out: List[Any] = []
    seen = set()
    try:
        from PySide6.QtWidgets import QApplication, QWidget
        app = QApplication.instance()
        if app is not None:
            for w in app.allWidgets():
                try:
                    if isinstance(w, QWidget) and hasattr(w, "blocks") and hasattr(w, "_add_block") and hasattr(w, "run_pipeline"):
                        if id(w) not in seen:
                            out.append(w)
                            seen.add(id(w))
                except Exception:
                    pass
    except Exception:
        pass
    for ws in _all_workspaces():
        try:
            tab = getattr(ws, "yolo_pipeline_tab", None)
            if tab is not None and id(tab) not in seen:
                out.append(tab)
                seen.add(id(tab))
        except Exception:
            pass
    return out


def _remove_ai_pipeline_plus_tabs(ws: Any) -> None:
    try:
        tabs = getattr(ws, "tabs", None)
        if tabs is None:
            return
        for i in reversed(range(tabs.count())):
            title = str(tabs.tabText(i) or "")
            if "AI Pipeline+" in title:
                tabs.removeTab(i)
                _ws_log(ws, "Removed separate AI Pipeline+ tab; patched the existing AI Pipeline instead.")
    except Exception:
        pass


def _patch_modules() -> None:
    for name, mod in list(sys.modules.items()):
        try:
            cls = getattr(mod, "YoloPipelineTab", None)
            if cls is not None:
                _patch_pipeline_class(cls)
        except Exception:
            pass
    for name in ("yolo_pipeline_plugin", "zzzz_mustatil_ai_pipeline_next_to_lae_trainer_hook"):
        try:
            mod = importlib.import_module(name)
            cls = getattr(mod, "YoloPipelineTab", None)
            if cls is not None:
                _patch_pipeline_class(cls)
        except Exception:
            pass


def _scan_and_patch() -> None:
    try:
        _patch_modules()
        for ws in _all_workspaces():
            _remove_ai_pipeline_plus_tabs(ws)
            try:
                _install_formlearner_scientific_training_ui(ws)
            except Exception:
                pass
        for tab in _find_pipeline_instances():
            _patch_pipeline_instance(tab)
    except Exception:
        traceback.print_exc()
    try:
        now = time.time()
        if not hasattr(_scan_and_patch, "_until"):
            _scan_and_patch._until = now + 25.0  # type: ignore[attr-defined]
        if now < getattr(_scan_and_patch, "_until", now):
            _schedule_scan(700)
    except Exception:
        pass


def mustatil_plugin_init() -> None:
    _patch_qtabwidget()
    _schedule_scan(100)
    _schedule_scan(800)
    _schedule_scan(2000)
    _log("initialized")


def register_plugin(*args, **kwargs):
    mustatil_plugin_init()


def init_plugin(*args, **kwargs):
    mustatil_plugin_init()


def load_plugin(*args, **kwargs):
    mustatil_plugin_init()


try:
    mustatil_plugin_init()
except Exception:
    pass
