#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mustatil plugin: patch the existing visual YOLO/AI Pipeline tab with an AI model dropdown.

Drop this single file into `mustatil_plugins`.

Purpose
-------
This does NOT create a separate "AI Pipeline+" tab. It patches the existing visual
YOLO/AI Pipeline tab so the AI models are added through one dropdown instead of
separate model buttons.

Kept unchanged:
  - FormLearner block
  - IF / Logic block
  - existing visual block canvas
  - existing connections / arrows / run / export buttons

Added through one dropdown:
  - YOLO
  - Google OWLv2, Grounding DINO, LAE-DINO, SAM2 if the current pipeline supports them
  - Faster R-CNN, Mask R-CNN, U-Net through the TorchVision/SAM2 model plugin

Expected companion plugin for Faster/Mask/U-Net/SAM2 inference:
  zzzz_mustatil_torchvision_frcnn_maskrcnn_unet_sam2_tabs.py

The patch is defensive: it waits until yolo_pipeline_plugin and the pipeline tab
exist, then patches the class/instance. It also removes the previously generated
separate "AI Pipeline+" tab if it is present.
"""
from __future__ import annotations

import importlib
import math
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_PATCHED_MODULE_IDS = set()
_PATCHED_TAB_IDS = set()
_SCAN_TIMER = None
_QTAB_PATCHED = False
_ORIG_ADD_TAB = None
_ORIG_INSERT_TAB = None

AI_CHOICES: List[Tuple[str, str]] = [
    ("YOLO", "yolo"),
    ("Google OWLv2", "owlv2"),
    ("Grounding DINO", "grounding_dino"),
    ("LAE-DINO", "lae_dino"),
    ("SAM2 from boxes", "sam2"),
    ("Faster R-CNN", "faster_rcnn"),
    ("Mask R-CNN", "mask_rcnn"),
    ("U-Net Segmentation", "unet"),
]

TORCHVISION_BLOCK_TYPES = {"faster_rcnn", "faster", "mask_rcnn", "mask", "unet", "unet_seg", "unet_segmentation"}
MODE_ALIASES = {
    "faster_rcnn": "faster",
    "faster": "faster",
    "faster-r-cnn": "faster",
    "faster r-cnn": "faster",
    "mask_rcnn": "mask",
    "mask": "mask",
    "mask-r-cnn": "mask",
    "mask r-cnn": "mask",
    "unet": "unet",
    "u-net": "unet",
    "unet_seg": "unet",
    "unet_segmentation": "unet",
}

DEFAULT_NAMES = {
    "yolo": "YOLO block",
    "owlv2": "OWLv2 block",
    "grounding_dino": "Grounding DINO block",
    "lae_dino": "LAE-DINO block",
    "sam2": "SAM2 segment block",
    "faster_rcnn": "Faster R-CNN block",
    "mask_rcnn": "Mask R-CNN block",
    "unet": "U-Net Segmentation block",
    "formlearner": "FormLearner block",
    "rule": "IF / Logic block",
}

AI_BUTTON_KEYWORDS = (
    "+ yolo", "+ owl", "+ google owl", "+ grounding", "+ lae", "+ sam", "+ faster", "+ mask", "+ u-net", "+ unet",
)
KEEP_BUTTON_KEYWORDS = ("formlearner", "rule", "logic", "if")


def _log(msg: str) -> None:
    try:
        print("[Mustatil AI Pipeline Dropdown Patch] " + str(msg))
    except Exception:
        pass


def _ws_log(ws: Any, msg: str) -> None:
    try:
        ws.log("[AI Pipeline Dropdown] " + str(msg))
    except Exception:
        _log(msg)


def _get_var(v: Any, default: Any = "") -> Any:
    try:
        if hasattr(v, "get"):
            return v.get()
    except Exception:
        pass
    return default


def _set_var(v: Any, value: Any) -> bool:
    try:
        if hasattr(v, "set"):
            v.set(value)
            return True
    except Exception:
        pass
    return False


def _workspace_from_widget(widget: Any) -> Optional[Any]:
    cur = widget
    for _ in range(140):
        if cur is None:
            break
        try:
            if hasattr(cur, "tabs") and (hasattr(cur, "dets") or hasattr(cur, "satellite_detections")):
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
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            return out
        seen = set()
        for w in app.topLevelWidgets():
            ws = _workspace_from_widget(w)
            if ws is not None and id(ws) not in seen:
                out.append(ws); seen.add(id(ws))
    except Exception:
        pass
    return out


def _schedule_scan(delay_ms: int = 150) -> None:
    global _SCAN_TIMER
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
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
    global _QTAB_PATCHED, _ORIG_ADD_TAB, _ORIG_INSERT_TAB
    if _QTAB_PATCHED:
        return
    try:
        from PySide6.QtWidgets import QTabWidget
    except Exception:
        return
    _ORIG_ADD_TAB = QTabWidget.addTab
    _ORIG_INSERT_TAB = QTabWidget.insertTab

    def add_tab_patched(self, *args, **kwargs):
        ret = _ORIG_ADD_TAB(self, *args, **kwargs)
        _schedule_scan(60)
        return ret

    def insert_tab_patched(self, *args, **kwargs):
        ret = _ORIG_INSERT_TAB(self, *args, **kwargs)
        _schedule_scan(60)
        return ret

    QTabWidget.addTab = add_tab_patched
    QTabWidget.insertTab = insert_tab_patched
    _QTAB_PATCHED = True


# -----------------------------------------------------------------------------
# Generic model bridge
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
            if name in sys.modules:
                mod = sys.modules[name]
            else:
                mod = importlib.import_module(name)
            if hasattr(mod, "_model_infer_pil"):
                return mod
        except Exception:
            pass
    raise RuntimeError(
        "TorchVision/SAM2 model plugin not found. Install "
        "zzzz_mustatil_torchvision_frcnn_maskrcnn_unet_sam2_tabs.py in mustatil_plugins."
    )


def _norm_type(t: Any) -> str:
    return str(t or "").strip().lower().replace(" ", "_").replace("-", "_")


def _type_to_mode(t: Any) -> Optional[str]:
    raw = str(t or "").strip().lower()
    norm = _norm_type(raw)
    return MODE_ALIASES.get(raw) or MODE_ALIASES.get(norm)


def _parse_class_filter(text: str) -> set:
    return {p.strip().lower() for p in str(text or "").replace(";", ",").split(",") if p.strip()}


def _class_count_from_workspace(ws: Any) -> int:
    # torchvision detection num_classes includes background.
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


def _nms_records(records: List[Dict[str, Any]], thr: float = 0.80) -> List[Dict[str, Any]]:
    def bbox(r):
        return float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])
    def score(r):
        try:
            return float(r.get("conf", r.get("confidence", r.get("score", 0.0))) or 0.0)
        except Exception:
            return 0.0
    def cls(r):
        try:
            return int(r.get("class_id", r.get("cls", 0)) or 0)
        except Exception:
            return 0
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        den = aa + bb - inter
        return 0.0 if den <= 0 else inter / den
    kept: List[Dict[str, Any]] = []
    for r in sorted(records, key=score, reverse=True):
        try:
            b = bbox(r)
        except Exception:
            kept.append(r); continue
        c = cls(r)
        if any(cls(k) == c and iou(b, bbox(k)) >= float(thr) for k in kept if all(x in k for x in ("x1", "y1", "x2", "y2"))):
            continue
        kept.append(r)
    kept.sort(key=lambda r: str(r.get("id", "")))
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
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "conf": conf,
            "class_id": cid,
            "label": label,
            "model": str(r.get("model", getattr(block, "model_path", "")) or block.type),
            "source_image": str(image_path),
        }
        for k in ("area_px", "mask_area_px"):
            if k in r:
                rec[k] = r[k]
        return rec
    except Exception:
        return None


def _run_torchvision_ai_block(tab: Any, b: Any, full, image_path: Path) -> List[Dict[str, Any]]:
    mod = _import_torchvision_model_plugin()
    mode = _type_to_mode(getattr(b, "type", ""))
    if mode is None:
        raise RuntimeError(f"Unsupported AI model block type: {getattr(b, 'type', '')}")

    class_filter = _parse_class_filter(getattr(b, "classes_filter", ""))
    parents = tab._input_records(b, full) if hasattr(tab, "_input_records") else []
    out: List[Dict[str, Any]] = []

    # Existing pipeline fields reused deliberately:
    # model_path = custom .pt/.pth; empty for COCO pretrained Faster/Mask R-CNN.
    # confidence = score threshold; imgsz/tile_size = tile size for large parent crops.
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
            tab._log(f"{b.name}: U-Net block skipped: model_path is empty. Choose a trained U-Net checkpoint.")
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
                local_settings = dict(settings)
                if mode == "sam2":
                    # In the visual pipeline SAM2 is segmentation-only. The upstream block's records are
                    # used as parent crops; the current crop is prompted by its full extent.
                    local_settings["prompt_boxes"] = [[0.0, 0.0, float(max(1, tile_img.width - 1)), float(max(1, tile_img.height - 1))]]
                local = mod._model_infer_pil(getattr(tab, "ws", None), mode, tile_img, local_settings)
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
# Patching the existing visual pipeline
# -----------------------------------------------------------------------------

def _patch_pipeline_class(cls: Any) -> bool:
    if cls is None or id(cls) in _PATCHED_MODULE_IDS:
        return False
    _PATCHED_MODULE_IDS.add(id(cls))

    # Extend visual colors if the existing VisualBlockItem class has a color map.
    try:
        mod = sys.modules.get(cls.__module__)
        vbi = getattr(mod, "VisualBlockItem", None)
        if vbi is not None and hasattr(vbi, "TYPE_COLORS"):
            from PySide6.QtGui import QColor
            vbi.TYPE_COLORS.update({
                "faster_rcnn": QColor(120, 95, 210),
                "mask_rcnn": QColor(165, 75, 190),
                "unet": QColor(40, 165, 125),
                "unet_segmentation": QColor(40, 165, 125),
            })
    except Exception:
        pass

    orig_add_block = getattr(cls, "_add_block", None)
    if callable(orig_add_block) and not getattr(cls, "_mustatil_ai_dropdown_add_patched", False):
        def _add_block_patched(self, typ="yolo", name=None, x=None, y=None, input_ref=None):
            typ_norm = _norm_type(typ)
            # Normalize GUI aliases to stable block types.
            alias = {
                "faster": "faster_rcnn",
                "faster_r_cnn": "faster_rcnn",
                "mask": "mask_rcnn",
                "mask_r_cnn": "mask_rcnn",
                "u_net": "unet",
                "unet_seg": "unet",
                "unet_segmentation": "unet",
            }.get(typ_norm, typ_norm)
            default_name = name or DEFAULT_NAMES.get(alias, DEFAULT_NAMES.get(str(typ), "AI model block"))
            b = orig_add_block(self, alias, name=default_name, x=x, y=y, input_ref=input_ref)
            try:
                if alias in {"faster_rcnn", "mask_rcnn"}:
                    b.confidence = 0.25
                    b.device = "cuda"
                    b.imgsz = max(768, int(getattr(b, "imgsz", 1024) or 1024))
                    b.crop_padding = 0
                    b.classes_filter = str(getattr(b, "classes_filter", "") or "")
                elif alias == "unet":
                    b.confidence = 0.50
                    b.device = "cuda"
                    b.imgsz = max(768, int(getattr(b, "imgsz", 1024) or 1024))
                    b.crop_padding = 0
                    b.classes_filter = "segmentation"
                elif alias == "sam2":
                    b.confidence = 0.50
                    b.device = "cuda"
                    b.crop_padding = max(16, int(getattr(b, "crop_padding", 16) or 16))
                    b.classes_filter = "sam2_segment"
                if hasattr(self, "refresh_code_from_blocks"):
                    self.refresh_code_from_blocks()
            except Exception:
                pass
            return b
        cls._add_block = _add_block_patched
        cls._mustatil_ai_dropdown_add_patched = True

    # Patch execution: preserve existing YOLO/FormLearner/Rule and existing OWL/GDINO/LAE/SAM2 methods;
    # add only Faster/Mask/U-Net through the TorchVision/SAM2 tab plugin.
    orig_run_pipeline = getattr(cls, "run_pipeline", None)
    if callable(orig_run_pipeline) and not getattr(cls, "_mustatil_ai_dropdown_run_patched", False):
        def run_pipeline_patched(self):
            try:
                has_new = any(_norm_type(getattr(b, "type", "")) in TORCHVISION_BLOCK_TYPES for b in list(getattr(self, "blocks", []) or []))
                if not has_new:
                    return orig_run_pipeline(self)
            except Exception:
                pass

            image_path = self._current_pipeline_image_path() if hasattr(self, "_current_pipeline_image_path") else Path(str(_get_var(getattr(self.ws, "image", None), "") or "").strip().strip('"')).expanduser()
            if not image_path.is_file():
                raise RuntimeError("Choose a valid image at the top of the AI Pipeline tab first. This image is used as the Original image block.")
            self.results = []
            self.results_by_block = {}
            self._log(f"Visual pipeline started: {image_path}")
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
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
                if typ == "yolo":
                    out = self._run_yolo_block(b, full, image_path)
                elif typ in {"owlv2", "owl", "owl_v2"} and hasattr(self, "_run_owlv2_block"):
                    out = self._run_owlv2_block(b, full, image_path)
                elif typ in {"grounding_dino", "grounding", "gdino"} and hasattr(self, "_run_grounding_dino_block"):
                    out = self._run_grounding_dino_block(b, full, image_path)
                elif typ in {"lae_dino", "laedino"} and hasattr(self, "_run_lae_dino_block"):
                    out = self._run_lae_dino_block(b, full, image_path)
                elif typ in {"sam2", "sam", "sam_2"} and hasattr(self, "_run_sam2_block") and typ not in TORCHVISION_BLOCK_TYPES:
                    out = self._run_sam2_block(b, full, image_path)
                elif typ in TORCHVISION_BLOCK_TYPES:
                    out = _run_torchvision_ai_block(self, b, full, image_path)
                elif typ == "formlearner":
                    out = self._run_formlearner_block(b, full)
                elif typ == "rule":
                    out = self._run_rule_block(b)
                else:
                    self._log(f"Unknown block type skipped: {b.name} / {b.type}")
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
        cls._mustatil_ai_dropdown_run_patched = True

    # Extend the selected block type combo after UI build.
    orig_select_block = getattr(cls, "select_block", None)
    if callable(orig_select_block) and not getattr(cls, "_mustatil_ai_dropdown_select_patched", False):
        def select_block_patched(self, block_id: str):
            ret = orig_select_block(self, block_id)
            try:
                _patch_pipeline_instance(self)
                b = self._current_block() if hasattr(self, "_current_block") else None
                if b is not None and hasattr(self, "typ"):
                    ix = self.typ.findText(str(getattr(b, "type", "")))
                    if ix >= 0:
                        self.typ.blockSignals(True); self.typ.setCurrentIndex(ix); self.typ.blockSignals(False)
            except Exception:
                pass
            return ret
        cls.select_block = select_block_patched
        cls._mustatil_ai_dropdown_select_patched = True

    return True


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
                            out.append(w); seen.add(id(w))
                except Exception:
                    pass
    except Exception:
        pass
    for ws in _all_workspaces():
        try:
            tab = getattr(ws, "yolo_pipeline_tab", None)
            if tab is not None and id(tab) not in seen:
                out.append(tab); seen.add(id(tab))
        except Exception:
            pass
    return out


def _remove_ai_pipeline_plus_tabs(ws: Any) -> None:
    try:
        setattr(ws, "_mustatil_ai_pipeline_plus_installed", True)
    except Exception:
        pass
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


def _patch_toolbar_dropdown(tab: Any) -> None:
    if getattr(tab, "_mustatil_ai_dropdown_toolbar_done", False):
        return
    try:
        from PySide6.QtWidgets import QPushButton, QComboBox, QLabel
    except Exception:
        return

    # Hide separate AI model buttons, but keep FormLearner and IF/Logic buttons.
    first_ai_button = None
    try:
        for btn in tab.findChildren(QPushButton):
            text = str(btn.text() or "").strip()
            low = text.lower()
            if any(k in low for k in KEEP_BUTTON_KEYWORDS):
                continue
            if any(k in low for k in AI_BUTTON_KEYWORDS):
                if first_ai_button is None:
                    first_ai_button = btn
                btn.setVisible(False)
    except Exception:
        pass

    combo = QComboBox()
    combo.setObjectName("MustatilAiPipelineModelDropdown")
    for label, typ in AI_CHOICES:
        combo.addItem(label, typ)
    combo.setMinimumWidth(190)
    add_btn = QPushButton("+ AI model block")
    add_btn.setObjectName("MustatilAiPipelineAddModelBlock")

    def add_selected():
        try:
            typ = str(combo.currentData() or combo.currentText()).strip()
            tab._add_block(typ)
        except Exception as exc:
            try:
                tab._log("Add AI model block failed: " + str(exc))
            except Exception:
                _log("Add AI model block failed: " + str(exc))

    add_btn.clicked.connect(add_selected)

    # Insert into the toolbar layout where the old +YOLO/+model buttons lived.
    inserted = False
    try:
        parent = first_ai_button.parentWidget() if first_ai_button is not None else None
        lay = parent.layout() if parent is not None else None
        if lay is not None:
            idx = 0
            if first_ai_button is not None:
                for i in range(lay.count()):
                    item = lay.itemAt(i)
                    if item and item.widget() is first_ai_button:
                        idx = i
                        break
            lay.insertWidget(idx, QLabel("AI model")); idx += 1
            lay.insertWidget(idx, combo); idx += 1
            lay.insertWidget(idx, add_btn)
            inserted = True
    except Exception:
        inserted = False

    if not inserted:
        try:
            lay = tab.layout()
            if lay is not None:
                lay.insertWidget(0, add_btn)
                lay.insertWidget(0, combo)
                lay.insertWidget(0, QLabel("AI model"))
        except Exception:
            pass

    tab._mustatil_ai_model_combo = combo
    tab._mustatil_ai_dropdown_toolbar_done = True


def _patch_selected_type_combo(tab: Any) -> None:
    try:
        combo = getattr(tab, "typ", None)
        if combo is not None and hasattr(combo, "addItem"):
            existing = {str(combo.itemText(i)) for i in range(combo.count())}
            for _label, typ in AI_CHOICES:
                if typ not in existing:
                    combo.addItem(typ)
                    existing.add(typ)
    except Exception:
        pass


def _patch_pipeline_instance(tab: Any) -> bool:
    if tab is None:
        return False
    try:
        _patch_pipeline_class(tab.__class__)
    except Exception:
        pass
    if id(tab) in _PATCHED_TAB_IDS:
        # Still refresh combos because the tab may have been rebuilt.
        _patch_selected_type_combo(tab)
        return False
    _PATCHED_TAB_IDS.add(id(tab))
    try:
        _patch_toolbar_dropdown(tab)
        _patch_selected_type_combo(tab)
        if hasattr(tab, "_log"):
            tab._log("AI model dropdown patch installed. FormLearner and IF / Logic blocks are unchanged.")
        try:
            ws = getattr(tab, "ws", None)
            if ws is not None:
                _ws_log(ws, "Existing AI Pipeline patched with model dropdown; no separate AI Pipeline+ tab was added.")
        except Exception:
            pass
        return True
    except Exception as exc:
        _log("patch pipeline instance failed: " + str(exc))
        traceback.print_exc()
        return False


def _patch_modules() -> None:
    names = list(sys.modules.keys())
    for name in names:
        if "yolo_pipeline_plugin" not in name and "ai_pipeline_next_to_lae" not in name:
            continue
        try:
            mod = sys.modules.get(name)
            cls = getattr(mod, "YoloPipelineTab", None)
            if cls is not None:
                _patch_pipeline_class(cls)
        except Exception:
            pass
    # Try canonical imports. These may fail before Mustatil has inserted the plugin dir into sys.path.
    for name in ("yolo_pipeline_plugin",):
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
    # Repeat a few times because the workspace imports/builds the pipeline after generic plugin loading.
    try:
        now = time.time()
        if not hasattr(_scan_and_patch, "_until"):
            _scan_and_patch._until = now + 20.0  # type: ignore[attr-defined]
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


try:
    mustatil_plugin_init()
except Exception:
    pass
