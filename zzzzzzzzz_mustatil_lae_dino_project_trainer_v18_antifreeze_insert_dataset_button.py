#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mustatil AI Pipeline patch v1

Drop this file into mustatil_plugins.

Adds to the existing visual AI/YOLO Pipeline tab:
- functional Faster R-CNN, Mask R-CNN and U-Net pipeline block dispatch
- a combined SAM2 + FormLearner Filter block
- one AI-model dropdown including the new combined filter block
- left-button background drag/pan for the canvas
- more usable right-side space by hiding the long help text and widening the property panel

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
from typing import Any, Dict, List, Optional, Tuple

_PATCHED_CLASSES = set()
_PATCHED_TABS = set()
_PATCHED_VIEWS = set()
_SCAN_TIMER = None
_QTAB_PATCHED = False
_ORIG_QTAB_ADD = None
_ORIG_QTAB_INSERT = None

ENHANCED_TYPES = {"faster_rcnn", "mask_rcnn", "unet", "sam2_formlearner", "sam2_form_filter"}
TORCHVISION_TYPES = {"faster_rcnn", "faster", "faster_r_cnn", "mask_rcnn", "mask", "mask_r_cnn", "unet", "u_net", "unet_seg", "unet_segmentation"}
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
        print("[Mustatil AI Pipeline FullBlocks SAM2/Form Patch] " + str(msg))
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


def _parse_combined_block_paths(b: Any) -> Dict[str, str]:
    """Resolve paths for the combined SAM2 + FormLearner block.

    Recommended UI convention:
      - Model path: SAM2 checkpoint (.pt/.pth)
      - FormLearner model: FormLearner JSON (.json)
      - optional SAM2 config: put config=C:/path/to/sam2_config.yaml in Class filter,
        or add {"sam2_config": "..."} in Rule JSON.

    If FormLearner model is a YAML/PY file, it is treated as the SAM2 config for
    compatibility with the older plain SAM2 block.
    """
    out = {"checkpoint": "", "sam2_config": "", "form_json": ""}
    model_path = str(getattr(b, "model_path", "") or "").strip().strip('"')
    second = str(getattr(b, "formlearner_model", "") or "").strip().strip('"')
    out["checkpoint"] = model_path

    def set_path(key: str, value: Any):
        val = str(value or "").strip().strip('"')
        if val:
            out[key] = val

    if second:
        suf = Path(second).suffix.lower()
        if suf == ".json":
            out["form_json"] = second
        elif suf in {".yaml", ".yml", ".py"}:
            out["sam2_config"] = second
        else:
            # Unknown companion file: keep it as form path only if it exists as JSON-like model.
            out["form_json"] = second

    # Rule JSON can carry hidden advanced settings without requiring more right-side UI rows.
    try:
        rule = json.loads(str(getattr(b, "rule_json", "") or "{}"))
        if isinstance(rule, dict):
            for k in ("sam2_config", "config", "sam_config"):
                if rule.get(k):
                    set_path("sam2_config", rule.get(k))
            for k in ("formlearner_model", "form_model", "form_json"):
                if rule.get(k):
                    set_path("form_json", rule.get(k))
    except Exception:
        pass

    # Class filter supports lightweight key=value entries, e.g. config=C:/x.yaml, form=C:/form.json.
    raw = str(getattr(b, "classes_filter", "") or "")
    for part in raw.replace(";", ",").split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip().lower()
        v = v.strip().strip('"')
        if k in {"config", "sam2_config", "sam_config"}:
            set_path("sam2_config", v)
        elif k in {"form", "form_json", "formlearner", "formlearner_model"}:
            set_path("form_json", v)

    # Try to auto-find a SAM2 config near the checkpoint or in common plugin runtime dirs.
    if not out["sam2_config"]:
        candidates: List[Path] = []
        cp = Path(model_path).expanduser() if model_path else Path("")
        dirs: List[Path] = []
        try:
            if cp and cp.parent.exists():
                dirs.extend([cp.parent, cp.parent.parent])
        except Exception:
            pass
        try:
            plugin_dir = Path(__file__).resolve().parent
            dirs.extend([
                plugin_dir,
                plugin_dir / "models",
                plugin_dir / "weights",
                plugin_dir / "mustatil_model_runtimes",
                plugin_dir / "mustatil_model_runtimes" / "SAM2",
                plugin_dir / "mustatil_model_runtimes" / "sam2",
            ])
        except Exception:
            pass
        seen_dirs = set()
        for d in dirs:
            try:
                d = d.expanduser()
                if not d.exists() or str(d) in seen_dirs:
                    continue
                seen_dirs.add(str(d))
                for pat in ("*sam2*.yaml", "*sam2*.yml", "*hiera*.yaml", "*hiera*.yml", "*.yaml", "*.yml", "*.py"):
                    candidates.extend(list(d.glob(pat))[:40])
            except Exception:
                pass
        if candidates:
            # Prefer obvious SAM2/Hiera configs.
            candidates = sorted(candidates, key=lambda x: (0 if ("sam2" in x.name.lower() or "hiera" in x.name.lower()) else 1, len(str(x))))
            out["sam2_config"] = str(candidates[0])
    return out


def _run_sam2_formlearner_block(tab: Any, b: Any, full: Any, image_path: Path) -> List[Dict[str, Any]]:
    parents = tab._input_records(b, full) if hasattr(tab, "_input_records") else []
    if not parents or str(getattr(b, "input_ref", "original")) == "original":
        try:
            tab._log(f"{b.name}: connect an upstream detector block first. This block is a filter for boxes, not a whole-image detector.")
        except Exception:
            pass
        return []

    paths = _parse_combined_block_paths(b)

    # Step 1: SAM2 refines every upstream parent box into a tighter mask bbox.
    # The plain SAM2 block historically uses model_path=checkpoint and formlearner_model=config.
    # The combined filter frees formlearner_model for the FormLearner JSON, so the config is
    # resolved from class-filter key=value, rule_json, or auto-detection and injected temporarily.
    old_form_field = getattr(b, "formlearner_model", "")
    try:
        if paths.get("sam2_config"):
            setattr(b, "formlearner_model", paths["sam2_config"])
        sam_records = tab._run_sam2_block(b, full, image_path) if hasattr(tab, "_run_sam2_block") else []
    except Exception as exc:
        try:
            tab._log(f"{b.name}: SAM2 step failed; using upstream boxes for FormLearner filter. Reason: {exc}")
        except Exception:
            pass
        sam_records = []
    finally:
        try:
            setattr(b, "formlearner_model", old_form_field)
        except Exception:
            pass
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
    model_path = paths.get("form_json") or str(getattr(b, "formlearner_model", "") or "")
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


def _polish_right_side(tab: Any) -> None:
    try:
        from PySide6.QtWidgets import QLabel, QTextEdit, QSplitter, QScrollArea, QGroupBox
    except Exception:
        return
    try:
        for lab in tab.findChildren(QLabel):
            txt = str(lab.text() or "")
            low = txt.lower()
            if "how to connect:" in low or "lego mindstorms" in low or "code view:" in low:
                lab.setVisible(False)
            elif txt.strip() == "Pipeline log / result summary":
                lab.setText("Log")
        for scroll in tab.findChildren(QScrollArea):
            try:
                if scroll.minimumWidth() < 640:
                    scroll.setMinimumWidth(640)
                if scroll.minimumHeight() < 470:
                    scroll.setMinimumHeight(470)
            except Exception:
                pass
        for gb in tab.findChildren(QGroupBox):
            try:
                title = str(gb.title() or "")
                if "Selected block" in title or "Block actions" in title:
                    gb.setMinimumWidth(620)
            except Exception:
                pass
        for te in tab.findChildren(QTextEdit):
            try:
                if te.isReadOnly():
                    te.setMaximumHeight(150)
                    te.setMinimumHeight(90)
            except Exception:
                pass
        for split in tab.findChildren(QSplitter):
            try:
                split.setSizes([820, 720])
                split.setStretchFactor(0, 1)
                split.setStretchFactor(1, 0)
                split.setCollapsible(1, False)
            except Exception:
                pass
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Patching the existing visual pipeline class/instance
# -----------------------------------------------------------------------------

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
        orig_height = getattr(vbi, "block_height", None)
        if callable(orig_height) and not getattr(vbi, "_mustatil_sam2_form_height_patched", False):
            def block_height_patched(self):
                if _norm_type(getattr(getattr(self, "block", None), "type", "")) == "sam2_formlearner":
                    return float(max(getattr(self, "FORM_HEIGHT", 154), 178))
                if _norm_type(getattr(getattr(self, "block", None), "type", "")) in {"faster_rcnn", "mask_rcnn", "unet"}:
                    return float(getattr(self, "YOLO_HEIGHT", 178))
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
                    b.classes_filter = str(getattr(b, "classes_filter", "") or "sam2_form_filter, optional: config=C:/sam2.yaml")
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
    try:
        from PySide6.QtWidgets import QPushButton, QComboBox, QLabel
    except Exception:
        return

    # Reuse an older dropdown if another AI-pipeline dropdown patch already created one.
    existing_combo = getattr(tab, "_mustatil_ai_model_combo", None)
    if existing_combo is None:
        try:
            for c in tab.findChildren(QComboBox):
                if str(c.objectName() or "").startswith("MustatilAiPipelineModelDropdown"):
                    existing_combo = c
                    tab._mustatil_ai_model_combo = c
                    break
        except Exception:
            existing_combo = None
    if existing_combo is not None:
        try:
            existing_data = {str(existing_combo.itemData(i)) for i in range(existing_combo.count())}
            for label, typ in AI_CHOICES:
                if typ not in existing_data:
                    existing_combo.addItem(label, typ)
        except Exception:
            pass
        tab._mustatil_fullblocks_toolbar_done = True
        return

    if getattr(tab, "_mustatil_fullblocks_toolbar_done", False):
        return

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
    combo.setObjectName("MustatilAiPipelineModelDropdownFullBlocks")
    for label, typ in AI_CHOICES:
        combo.addItem(label, typ)
    combo.setMinimumWidth(230)
    add_btn = QPushButton("+ AI block")
    add_btn.setObjectName("MustatilAiPipelineAddFullBlock")

    def add_selected():
        try:
            typ = str(combo.currentData() or combo.currentText()).strip()
            tab._add_block(typ)
        except Exception as exc:
            try:
                tab._log("Add AI block failed: " + str(exc))
            except Exception:
                _log("Add AI block failed: " + str(exc))

    add_btn.clicked.connect(add_selected)
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
    tab._mustatil_fullblocks_toolbar_done = True


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
            tab._log("AI Pipeline patch installed: all model blocks dispatch, SAM2 + FormLearner filter, left-drag canvas pan, wider right panel.")
        except Exception:
            pass
        try:
            ws = getattr(tab, "ws", None)
            if ws is not None:
                _ws_log(ws, "Existing AI Pipeline enhanced in-place; no separate tab was added.")
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
