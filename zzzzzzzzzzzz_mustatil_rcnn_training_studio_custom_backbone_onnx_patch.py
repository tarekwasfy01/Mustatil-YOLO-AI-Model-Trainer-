#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mustatil AI Pipeline clean single-block-picker patch v3

Drop this file into mustatil_plugins.

Adds/keeps the existing visual AI/YOLO Pipeline tab functionality while cleaning the UI:
- one single AI model block picker; no duplicate model buttons/dropdowns
- per-block model/checkpoint/config/FormLearner selection buttons in the visual blocks and right editor
- functional Faster R-CNN, Mask R-CNN and U-Net pipeline block dispatch
- a combined SAM2 + FormLearner Filter block
- left-button background drag/pan for the canvas
- large central block canvas, compact surrounding controls
- right side rebuilt as one scrollable, grouped editor with process actions and log at the bottom

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

ENHANCED_TYPES = {"faster_rcnn", "mask_rcnn", "unet", "sam2_formlearner", "sam2_form_filter"}
TORCHVISION_TYPES = {"faster_rcnn", "faster", "faster_r_cnn", "mask_rcnn", "mask", "mask_r_cnn", "unet", "u_net", "unet_seg", "unet_segmentation"}
MODEL_BLOCK_TYPES = {"yolo", "owlv2", "grounding_dino", "lae_dino", "sam2", "sam2_formlearner", "faster_rcnn", "mask_rcnn", "unet"}
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
}
MODE_FOR_TYPE = {"faster_rcnn": "faster", "mask_rcnn": "mask", "unet": "unet"}

AI_CHOICES: List[Tuple[str, str]] = [
    ("YOLO", "yolo"),
    ("Google OWLv2", "owlv2"),
    ("Grounding DINO", "grounding_dino"),
    ("LAE-DINO", "lae_dino"),
    ("SAM2 from boxes", "sam2"),
    ("SAM2 + FormLearner Filter", "sam2_formlearner"),
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
    "formlearner": "FormLearner block",
    "rule": "IF / Logic block",
}

AI_BUTTON_KEYWORDS = (
    "+ yolo", "+ owl", "+ google owl", "+ grounding", "+ lae", "+ sam", "+ faster", "+ mask", "+ u-net", "+ unet",
)
KEEP_BUTTON_KEYWORDS = ("formlearner", "form learner", "rule", "logic", "if")


def _log(msg: str) -> None:
    try:
        print("[Mustatil AI Pipeline Clean Single Picker v3] " + str(msg))
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
    else:
        if typ == "lae_dino":
            for attr in ("mustatil_lae_existing_v9_weights", "mustatil_lae_existing_v8_weights", "mustatil_lae_weights", "mustatil_lae_checkpoint"):
                add(getattr(ws, attr, ""), "LAE checkpoint")
        elif typ == "sam2":
            for attr in ("sam2_config", "mustatil_sam2_config", "sam_config"):
                add(getattr(ws, attr, ""), "SAM2 config")
        elif typ in {"sam2_formlearner", "formlearner"}:
            for attr in ("fl_model_path", "form_model_path", "formlearner_model"):
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
    """Hide duplicate add buttons/dropdowns created by older pipeline patches.

    The actual pipeline methods stay untouched; this only removes visual clutter.
    """
    try:
        from PySide6.QtWidgets import QPushButton, QComboBox, QLabel
    except Exception:
        return
    try:
        for btn in tab.findChildren(QPushButton):
            try:
                if btn.property("mustatil_clean_toolbar_button"):
                    continue
            except Exception:
                pass
            text = str(btn.text() or "").strip().lower()
            obj = str(btn.objectName() or "")
            # Remove all previous model/action rows from the old toolbar; v2 recreates one clean row.
            if (
                text in {
                    "+ yolo block", "+ owlv2 block", "+ grounding dino block", "+ lae-dino block",
                    "+ sam2 block", "+ formlearner block", "+ if / logic block",
                    "duplicate selected", "delete selected", "auto layout", "load pipeline", "save pipeline",
                    "view: code", "view: blocks", "+ ai block", "+ add ai block", "+ add ai model block",
                }
                or any(k in text for k in AI_BUTTON_KEYWORDS)
                or obj.startswith("MustatilAiPipelineAdd")
            ):
                btn.setVisible(False)
                try:
                    btn.setMaximumHeight(0)
                except Exception:
                    pass
        for combo in tab.findChildren(QComboBox):
            try:
                if combo.property("mustatil_clean_ai_combo"):
                    continue
            except Exception:
                pass
            obj = str(combo.objectName() or "")
            if obj.startswith("MustatilAiPipelineModelDropdown"):
                combo.setVisible(False)
                try:
                    combo.setMaximumHeight(0)
                except Exception:
                    pass
        for lab in tab.findChildren(QLabel):
            try:
                if lab.property("mustatil_clean_toolbar_label"):
                    continue
            except Exception:
                pass
            txt = str(lab.text() or "").strip().lower()
            if txt in {"ai model", "ai-model", "model"} and "MustatilClean" not in str(lab.objectName() or ""):
                lab.setVisible(False)
                try:
                    lab.setMaximumHeight(0)
                except Exception:
                    pass
    except Exception:
        pass


def _install_single_clean_toolbar(tab: Any) -> None:
    """Create one ergonomic top row: one AI model block chooser plus workflow buttons."""
    global _CLEAN_TOOLBAR_TABS
    try:
        from PySide6.QtWidgets import (
            QWidget, QGroupBox, QHBoxLayout, QLabel, QPushButton, QComboBox, QSizePolicy
        )
    except Exception:
        return
    _hide_old_top_controls(tab)
    try:
        if id(tab) in _CLEAN_TOOLBAR_TABS and getattr(tab, "_mustatil_clean_ai_combo", None) is not None:
            combo = getattr(tab, "_mustatil_clean_ai_combo")
            existing_data = {str(combo.itemData(i)) for i in range(combo.count())}
            for label, typ in AI_CHOICES:
                if typ not in existing_data:
                    combo.addItem(label, typ)
            return
    except Exception:
        pass

    try:
        root = tab.layout()
        if root is None:
            return
        bar = QGroupBox("Block-Auswahl")
        bar.setObjectName("MustatilCleanAiPipelineBlockChooser")
        bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(8)

        lab = QLabel("AI model block")
        lab.setObjectName("MustatilCleanAiPipelineModelLabel")
        lab.setProperty("mustatil_clean_toolbar_label", True)
        lay.addWidget(lab)

        combo = QComboBox()
        combo.setObjectName("MustatilCleanAiPipelineModelCombo")
        combo.setProperty("mustatil_clean_ai_combo", True)
        for label, typ in AI_CHOICES:
            combo.addItem(label, typ)
        combo.setMinimumWidth(270)
        combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lay.addWidget(combo, 2)

        def _mk_button(text: str, slot: Any, minw: int = 0):
            b = QPushButton(text)
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

        _mk_button("+ Add AI model block", add_ai, 150)
        _mk_button("+ FormLearner", lambda: tab._add_block("formlearner"), 112)
        _mk_button("+ IF / Logic", lambda: tab._add_block("rule"), 100)
        _mk_button("Duplicate", getattr(tab, "_duplicate_current", lambda: None), 90)
        _mk_button("Delete", getattr(tab, "_delete_current", lambda: None), 76)
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
        tab._mustatil_clean_toolbar = bar
        _CLEAN_TOOLBAR_TABS.add(id(tab))
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
    """Turn the right side into one scrollable grouped editor.

    The original widgets are reused, so all signal connections and functions remain active.
    """
    global _CLEAN_RIGHT_TABS
    try:
        from PySide6.QtWidgets import (
            QWidget, QVBoxLayout, QScrollArea, QGroupBox, QTextEdit, QLabel, QSizePolicy
        )
        from PySide6.QtCore import Qt
    except Exception:
        return
    try:
        _clean_field_widths(tab)
        if id(tab) in _CLEAN_RIGHT_TABS:
            # Still enforce splitter/canvas sizing on later scans.
            _polish_splitters_and_text(tab)
            return

        scroll = _find_main_property_scroll(tab)
        if scroll is None:
            return

        # Find reusable action box and summary log before reparenting.
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

        g, lay = _make_clean_group("1. Selected block")
        try:
            lay.addWidget(getattr(tab, "enabled"))
        except Exception:
            pass
        _add_vertical_field(lay, "Name", getattr(tab, "name", None))
        _add_horizontal_pair(lay, "Block type", getattr(tab, "typ", None), "Input from", getattr(tab, "input_ref", None))
        panel_lay.addWidget(g)

        g, lay = _make_clean_group("2. Model, config and runtime")
        _add_vertical_field(lay, "Model / checkpoint", getattr(tab, "model_path", None), "YOLO/ONNX/Engine, HuggingFace model id/folder, Faster/Mask/U-Net checkpoint or SAM2 checkpoint.")
        _add_vertical_field(lay, "Companion file", getattr(tab, "form_model", None), "Pure SAM2: config YAML. LAE-DINO: checkpoint/config companion. SAM2+FormLearner: FormLearner JSON.")
        _add_model_button_grid(lay, tab)
        _add_horizontal_pair(lay, "Device", getattr(tab, "device", None), "Confidence", getattr(tab, "conf", None))
        _add_horizontal_pair(lay, "Image size", getattr(tab, "imgsz", None), "Form threshold", getattr(tab, "form_threshold", None))
        panel_lay.addWidget(g)

        g, lay = _make_clean_group("3. Large image / parent-crop processing")
        try:
            lay.addWidget(getattr(tab, "use_chunked_yolo"))
        except Exception:
            pass
        try:
            lay.addWidget(getattr(tab, "shifted_tiles"))
        except Exception:
            pass
        _add_horizontal_pair(lay, "Tile size", getattr(tab, "tile_size", None), "Tile overlap", getattr(tab, "tile_overlap", None))
        _add_vertical_field(lay, "Parent crop padding", getattr(tab, "crop_padding", None))
        _add_vertical_field(lay, "Class / prompt / special options", getattr(tab, "classes_filter", None), "Comma separated classes/prompts. For SAM2+FormLearner you can also pass config=... when needed.")
        panel_lay.addWidget(g)

        g, lay = _make_clean_group("4. Rule / logic JSON")
        _add_vertical_field(lay, "Rule JSON", getattr(tab, "rule_json", None), "Used by IF / Logic blocks; kept scrollable and large enough for manual editing.")
        panel_lay.addWidget(g)

        if action_box is not None:
            try:
                action_box.setTitle("5. Process / functions")
                action_box.setMinimumWidth(0)
                action_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
                panel_lay.addWidget(action_box)
            except Exception:
                pass

        if summary is not None:
            try:
                log_group, log_lay = _make_clean_group("6. Log / result summary")
                summary.setMinimumHeight(150)
                summary.setMaximumHeight(260)
                summary.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
                log_lay.addWidget(summary)
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
        scroll.setMinimumWidth(430)
        scroll.setWidget(panel)
        tab._mustatil_clean_right_panel = panel
        _CLEAN_RIGHT_TABS.add(id(tab))
        _polish_splitters_and_text(tab)
    except Exception:
        traceback.print_exc()


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
    """Give every AI model block an inline model/config chooser with real buttons."""
    if vbi is None or getattr(vbi, "_mustatil_v3_inline_model_buttons_patched", False):
        return
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QWidget, QGridLayout, QLabel, QComboBox, QDoubleSpinBox, QSpinBox,
            QLineEdit, QPushButton, QGraphicsProxyWidget, QFileDialog
        )
    except Exception:
        return

    orig_refresh = getattr(vbi, "refresh_embedded_controls", None)
    orig_width = getattr(vbi, "block_width", None)
    orig_height = getattr(vbi, "block_height", None)

    def _is_model_block_item(self) -> bool:
        return _norm_type(getattr(getattr(self, "block", None), "type", "")) in MODEL_BLOCK_TYPES

    def block_width_v3(self):
        if _is_model_block_item(self):
            b = getattr(self, "block", None)
            content = [getattr(b, "name", ""), getattr(b, "model_path", ""), getattr(b, "formlearner_model", ""), getattr(b, "classes_filter", "")]
            longest = max([len(str(x)) for x in content] + [0])
            return float(max(getattr(type(self), "WIDTH", 270), min(620, 300 + longest * 5)))
        return orig_width(self) if callable(orig_width) else 300.0

    def block_height_v3(self):
        if _is_model_block_item(self):
            return 226.0
        return orig_height(self) if callable(orig_height) else 178.0

    def _set_combo_value(combo: Any, value: str) -> None:
        try:
            ix = combo.findData(value)
            if ix >= 0:
                combo.setCurrentIndex(ix)
            else:
                combo.setEditText(value)
        except Exception:
            pass

    def _inline_current_value(combo: Any) -> str:
        try:
            data = combo.currentData()
            return str(data if data not in (None, "") else combo.currentText()).strip()
        except Exception:
            return ""

    def update_model_controls_values_v3(self):
        if getattr(self, "rule_proxy", None) is None or not _is_model_block_item(self):
            return
        self._updating_inline_controls = True
        try:
            b = self.block
            old = self._v3_model_combo.blockSignals(True)
            self._v3_model_combo.clear()
            for label, value in _collect_model_choices_for_block(getattr(self, "tab", None), b, "primary"):
                self._v3_model_combo.addItem(label, value)
            _set_combo_value(self._v3_model_combo, str(getattr(b, "model_path", "") or ""))
            self._v3_model_combo.blockSignals(old)

            old = self._v3_companion_combo.blockSignals(True)
            self._v3_companion_combo.clear()
            for label, value in _collect_model_choices_for_block(getattr(self, "tab", None), b, "companion"):
                self._v3_companion_combo.addItem(label, value)
            _set_combo_value(self._v3_companion_combo, str(getattr(b, "formlearner_model", "") or ""))
            self._v3_companion_combo.blockSignals(old)

            self._v3_conf_spin.setValue(float(getattr(b, "confidence", 0.05) or 0.05))
            self._v3_imgsz_spin.setValue(int(getattr(b, "imgsz", 640) or 640))
            self._v3_device_combo.setCurrentText(str(getattr(b, "device", "cpu") or "cpu"))
            self._v3_class_edit.setText(str(getattr(b, "classes_filter", "") or ""))
            try:
                self._v3_form_threshold_spin.setValue(float(getattr(b, "form_threshold", 0.5) or 0.5))
            except Exception:
                pass
        finally:
            self._updating_inline_controls = False

    def inline_model_changed_v3(self):
        if getattr(self, "_updating_inline_controls", False):
            return
        try:
            b = self.block
            b.model_path = _inline_current_value(self._v3_model_combo)
            b.formlearner_model = _inline_current_value(self._v3_companion_combo)
            b.confidence = float(self._v3_conf_spin.value())
            b.imgsz = int(self._v3_imgsz_spin.value())
            b.device = str(self._v3_device_combo.currentText() or "cpu")
            b.classes_filter = str(self._v3_class_edit.text() or "")
            b.form_threshold = float(self._v3_form_threshold_spin.value())
            try:
                self.tab._update_editor_from_inline(b)
            except Exception:
                pass
            try:
                self.tab.refresh_code_from_blocks()
            except Exception:
                pass
            self.update()
        except Exception:
            pass

    def _browse_inline_primary(self):
        try:
            typ = _norm_type(getattr(self.block, "type", ""))
            path, _ = QFileDialog.getOpenFileName(getattr(self, "tab", None), "Choose model/checkpoint", "", _file_filter_for_block(typ, "primary"))
            if path:
                self._v3_model_combo.setEditText(path)
                inline_model_changed_v3(self)
        except Exception:
            pass

    def _browse_inline_companion(self):
        try:
            typ = _norm_type(getattr(self.block, "type", ""))
            path, _ = QFileDialog.getOpenFileName(getattr(self, "tab", None), "Choose companion/config", "", _file_filter_for_block(typ, "companion"))
            if path:
                self._v3_companion_combo.setEditText(path)
                inline_model_changed_v3(self)
        except Exception:
            pass

    def _browse_inline_sam2_config(self):
        try:
            typ = _norm_type(getattr(self.block, "type", ""))
            path, _ = QFileDialog.getOpenFileName(getattr(self, "tab", None), "Choose SAM2 config YAML", "", _file_filter_for_block(typ, "sam2_config"))
            if not path:
                return
            if typ == "sam2":
                self._v3_companion_combo.setEditText(path)
            else:
                self._v3_class_edit.setText(_replace_key_path(str(self._v3_class_edit.text() or ""), "config", path))
            inline_model_changed_v3(self)
        except Exception:
            pass

    def refresh_embedded_controls_v3(self):
        typ = _norm_type(getattr(getattr(self, "block", None), "type", ""))
        if typ not in MODEL_BLOCK_TYPES:
            return orig_refresh(self) if callable(orig_refresh) else None
        if getattr(self, "rule_proxy", None) is not None and getattr(self, "_proxy_type", "") != "model_v3":
            try:
                self.scene().removeItem(self.rule_proxy)
            except Exception:
                pass
            self.rule_proxy = None
            self._proxy_type = ""
        if getattr(self, "rule_proxy", None) is None:
            w = QWidget()
            w.setObjectName("ModelBlockInlineEditorV3")
            w.setStyleSheet(
                "QWidget#ModelBlockInlineEditorV3 { background: rgba(255,255,255,222); border-radius: 7px; }"
                "QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit { font-size: 8pt; min-height: 22px; max-height: 25px; }"
                "QPushButton { font-size: 8pt; min-height: 22px; max-height: 25px; padding-left: 4px; padding-right: 4px; }"
                "QLabel { font-size: 8pt; font-weight: bold; }"
            )
            lay = QGridLayout(w)
            lay.setContentsMargins(6, 4, 6, 4)
            lay.setHorizontalSpacing(4)
            lay.setVerticalSpacing(3)
            self._v3_model_combo = QComboBox(); self._v3_model_combo.setEditable(True); self._v3_model_combo.setMinimumWidth(190)
            self._v3_model_btn = QPushButton("Model…"); self._v3_model_btn.setToolTip("Choose model/checkpoint for this block")
            self._v3_companion_combo = QComboBox(); self._v3_companion_combo.setEditable(True); self._v3_companion_combo.setMinimumWidth(190)
            self._v3_companion_btn = QPushButton("Comp…"); self._v3_companion_btn.setToolTip("Choose companion/config/FormLearner file")
            self._v3_sam_cfg_btn = QPushButton("SAM2 cfg…"); self._v3_sam_cfg_btn.setToolTip("Choose SAM2 config YAML; useful for SAM2 + FormLearner")
            self._v3_conf_spin = QDoubleSpinBox(); self._v3_conf_spin.setRange(0.001, 1.0); self._v3_conf_spin.setDecimals(3); self._v3_conf_spin.setSingleStep(0.01)
            self._v3_imgsz_spin = QSpinBox(); self._v3_imgsz_spin.setRange(64, 8192); self._v3_imgsz_spin.setSingleStep(32)
            self._v3_device_combo = QComboBox(); self._v3_device_combo.setEditable(True); self._v3_device_combo.addItems(["auto cuda", "cpu", "cuda", "0", "rocm", "directml", "openml", "openvino"])
            self._v3_class_edit = QLineEdit(); self._v3_class_edit.setPlaceholderText("class/prompt/options, e.g. mustatil, config=C:/sam2.yaml")
            self._v3_form_threshold_spin = QDoubleSpinBox(); self._v3_form_threshold_spin.setRange(0.0, 1.0); self._v3_form_threshold_spin.setDecimals(3); self._v3_form_threshold_spin.setSingleStep(0.01)

            lay.addWidget(QLabel("MODEL"), 0, 0); lay.addWidget(self._v3_model_combo, 0, 1, 1, 3); lay.addWidget(self._v3_model_btn, 0, 4)
            lay.addWidget(QLabel("COMP"), 1, 0); lay.addWidget(self._v3_companion_combo, 1, 1, 1, 3); lay.addWidget(self._v3_companion_btn, 1, 4)
            lay.addWidget(QLabel("CONF"), 2, 0); lay.addWidget(self._v3_conf_spin, 2, 1); lay.addWidget(QLabel("SIZE"), 2, 2); lay.addWidget(self._v3_imgsz_spin, 2, 3); lay.addWidget(self._v3_sam_cfg_btn, 2, 4)
            lay.addWidget(QLabel("DEVICE"), 3, 0); lay.addWidget(self._v3_device_combo, 3, 1); lay.addWidget(QLabel("FORM ≥"), 3, 2); lay.addWidget(self._v3_form_threshold_spin, 3, 3, 1, 2)
            lay.addWidget(QLabel("CLASS"), 4, 0); lay.addWidget(self._v3_class_edit, 4, 1, 1, 4)

            self._v3_model_combo.currentTextChanged.connect(lambda *_: inline_model_changed_v3(self))
            self._v3_companion_combo.currentTextChanged.connect(lambda *_: inline_model_changed_v3(self))
            self._v3_conf_spin.valueChanged.connect(lambda *_: inline_model_changed_v3(self))
            self._v3_imgsz_spin.valueChanged.connect(lambda *_: inline_model_changed_v3(self))
            self._v3_device_combo.currentTextChanged.connect(lambda *_: inline_model_changed_v3(self))
            self._v3_class_edit.textChanged.connect(lambda *_: inline_model_changed_v3(self))
            self._v3_form_threshold_spin.valueChanged.connect(lambda *_: inline_model_changed_v3(self))
            self._v3_model_btn.clicked.connect(lambda *_: _browse_inline_primary(self))
            self._v3_companion_btn.clicked.connect(lambda *_: _browse_inline_companion(self))
            self._v3_sam_cfg_btn.clicked.connect(lambda *_: _browse_inline_sam2_config(self))
            self.rule_proxy = QGraphicsProxyWidget(self)
            self.rule_proxy.setWidget(w)
            self.rule_proxy.setPos(9, 82)
            self.rule_proxy.resize(self.block_width() - 18, 134)
            self._proxy_type = "model_v3"
        update_model_controls_values_v3(self)
        return

    vbi.block_width = block_width_v3
    vbi.block_height = block_height_v3
    vbi.refresh_embedded_controls = refresh_embedded_controls_v3
    vbi._mustatil_v3_inline_model_buttons_patched = True

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
                })
        except Exception:
            pass
        try:
            _patch_visual_block_inline_controls(vbi)
        except Exception:
            pass
        orig_height = getattr(vbi, "block_height", None)
        if callable(orig_height) and not getattr(vbi, "_mustatil_sam2_form_height_patched", False):
            def block_height_patched(self):
                typ = _norm_type(getattr(getattr(self, "block", None), "type", ""))
                if typ in MODEL_BLOCK_TYPES:
                    return 226.0
                return orig_height(self)
            vbi.block_height = block_height_patched
            vbi._mustatil_sam2_form_height_patched = True
        try:
            view_cls = getattr(mod, "PipelineGraphicsView", None)
            _patch_view_class(view_cls)
        except Exception:
            pass
    except Exception:
        pass


def _patch_pipeline_class(cls: Any) -> bool:
    if cls is None or id(cls) in _PATCHED_CLASSES:
        return False
    _PATCHED_CLASSES.add(id(cls))
    _patch_visual_block_class(cls)

    # Add helper methods directly to the tab class.
    cls._run_torchvision_ai_block = _run_torchvision_ai_block
    cls._run_sam2_formlearner_block = _run_sam2_formlearner_block

    orig_update_from_inline = getattr(cls, "_update_editor_from_inline", None)
    if callable(orig_update_from_inline) and not getattr(cls, "_mustatil_v3_update_inline_patched", False):
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
        cls._update_editor_from_inline = _update_editor_from_inline_patched
        cls._mustatil_v3_update_inline_patched = True

    orig_add_block = getattr(cls, "_add_block", None)
    if callable(orig_add_block) and not getattr(cls, "_mustatil_fullblocks_add_patched", False):
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
        cls._mustatil_fullblocks_add_patched = True

    # Patch execution. When no enhanced block exists, preserve the currently installed original run method.
    orig_run = getattr(cls, "run_pipeline", None)
    if callable(orig_run) and not getattr(cls, "_mustatil_fullblocks_run_patched", False):
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
                elif typ == "formlearner" and hasattr(self, "_run_formlearner_block"):
                    out = self._run_formlearner_block(b, full)
                elif typ == "rule" and hasattr(self, "_run_rule_block"):
                    out = self._run_rule_block(b)
                else:
                    self._log(f"Unknown block type skipped: {getattr(b, 'name', '')} / {getattr(b, 'type', '')}")
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
        cls._mustatil_fullblocks_run_patched = True

    orig_select = getattr(cls, "select_block", None)
    if callable(orig_select) and not getattr(cls, "_mustatil_fullblocks_select_patched", False):
        def select_block_patched(self, block_id: str, *args, **kwargs):
            ret = orig_select(self, block_id, *args, **kwargs)
            try:
                _patch_pipeline_instance(self)
                _extend_type_combo(self)
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
        cls._mustatil_fullblocks_select_patched = True

    return True


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
            tab._log("AI Pipeline clean v2 installed: one AI model picker, full block dispatch, SAM2 + FormLearner filter, large canvas and scrollable grouped right editor.")
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
