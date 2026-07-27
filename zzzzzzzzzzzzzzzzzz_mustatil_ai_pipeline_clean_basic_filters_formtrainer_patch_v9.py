#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mustatil patch plugin: FormTrainer -> Pipeline Filter Model Trainer

Purpose
-------
Adds one clean, non-duplicated training/calibration panel to the existing
FormLearner/FormTrainer tab. The panel trains/saves JSON presets for the same
scientific/basic filter blocks that are used in the AI Pipeline tab:

- scientific_shape_filter
- scientific_size_filter
- scientific_texture_filter
- geo_nms_filter
- cluster_context_filter
- linear_alignment_filter
- scientific_formlearner_filter
- sam2_formlearner threshold helper

The AI Pipeline patches v7/v8/v9 read scientific presets from a filter block's
model_path or from classes_filter containing preset=<json>. This plugin writes
that compatible JSON format and can apply the preset directly to the currently
selected AI Pipeline block.

The algorithmic filters are not black-box neural networks. "Training" here means
calibrating transparent parameters from positive examples/current detections and
saving them as reusable filter-model JSON presets.
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import math
import os
import sqlite3
import struct
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from PIL import Image, ImageStat
    Image.MAX_IMAGE_PIXELS = None
except Exception:  # pragma: no cover - Mustatil usually has Pillow
    Image = None
    ImageStat = None

try:
    import numpy as _np
except Exception:  # pragma: no cover - pure Python fallback exists
    _np = None

try:
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtWidgets import (
        QApplication, QWidget, QTabWidget, QGroupBox, QVBoxLayout, QHBoxLayout,
        QGridLayout, QLabel, QComboBox, QPushButton, QFileDialog, QLineEdit,
        QPlainTextEdit, QCheckBox, QDoubleSpinBox, QMessageBox
    )
except Exception:  # pragma: no cover - import happens inside Mustatil
    QApplication = None  # type: ignore

PLUGIN_NAME = "FormTrainerPipelineFilterModelsPatchV1"
NEW_BOX_OBJECT_NAME = "MustatilFormTrainerPipelineFilterModelsV1"
OLD_DUPLICATE_OBJECT_NAMES = {
    "MustatilFormTrainerScientificPresetsV8",
    "MustatilFormTrainerScientificPresetsV7",
    "MustatilFormTrainerScientificPresetsV6",
}

FILTER_SPECS: List[Tuple[str, str, List[Tuple[str, float]]]] = [
    ("Shape / morphometry filter", "scientific_shape_filter", [("area_min", 16.0), ("aspect_max", 12.0), ("fill_min", 0.05), ("threshold", 0.20)]),
    ("Size / scale filter", "scientific_size_filter", [("area_min", 16.0), ("area_max", 999999999.0), ("width_min", 1.0), ("height_min", 1.0)]),
    ("Texture / contrast filter", "scientific_texture_filter", [("contrast_min", 0.03), ("std_min", 2.0), ("edge_min", 0.0)]),
    ("Geo-NMS duplicate filter", "geo_nms_filter", [("iou", 0.60)]),
    ("Cluster / context filter", "cluster_context_filter", [("radius", 250.0), ("neighbors_min", 1.0), ("neighbors_max", 999999999.0)]),
    ("Linear alignment filter", "linear_alignment_filter", [("radius", 450.0), ("line_score_min", 0.55), ("neighbors_min", 2.0)]),
    ("Scientific FormLearner filter", "scientific_formlearner_filter", [("mix_score_min", 0.50), ("weight_form", 0.55), ("weight_shape", 0.25), ("weight_texture", 0.10), ("weight_context", 0.10)]),
    ("SAM2 + FormLearner threshold helper", "sam2_formlearner", [("form_threshold", 0.50), ("shape_threshold", 0.20)]),
]
FILTER_LABEL_BY_TYPE = {t: label for label, t, _ in FILTER_SPECS}
DEFAULTS_BY_TYPE = {t: dict(defaults) for _label, t, defaults in FILTER_SPECS}
ALGORITHM_ONLY_FILTER_TYPES = {
    "scientific_shape_filter",
    "scientific_size_filter",
    "scientific_texture_filter",
    "geo_nms_filter",
    "cluster_context_filter",
    "linear_alignment_filter",
}
PIPELINE_FILTER_TYPES = set(DEFAULTS_BY_TYPE)


def _log(msg: str) -> None:
    print(f"[Mustatil FormTrainer Pipeline Filter Models] {msg}")


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


def _percentile(values: Sequence[Any], q: float, default: float = 0.0) -> float:
    vals: List[float] = []
    for v in values:
        try:
            fv = float(v)
            if math.isfinite(fv):
                vals.append(fv)
        except Exception:
            pass
    vals.sort()
    if not vals:
        return float(default)
    if len(vals) == 1:
        return float(vals[0])
    pos = max(0.0, min(1.0, float(q))) * (len(vals) - 1)
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals[lo])
    return float(vals[lo] * (hi - pos) + vals[hi] * (pos - lo))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _round_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, float):
            if abs(v) >= 100:
                out[k] = round(v, 1)
            elif abs(v) >= 10:
                out[k] = round(v, 2)
            else:
                out[k] = round(v, 4)
        else:
            out[k] = v
    return out


def _norm_type(value: Any) -> str:
    s = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "shape_filter": "scientific_shape_filter",
        "morphometry": "scientific_shape_filter",
        "morphometry_filter": "scientific_shape_filter",
        "scientific_shape": "scientific_shape_filter",
        "size_filter": "scientific_size_filter",
        "scale_filter": "scientific_size_filter",
        "area_filter": "scientific_size_filter",
        "texture_filter": "scientific_texture_filter",
        "contrast_filter": "scientific_texture_filter",
        "scientific_texture": "scientific_texture_filter",
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
        "sam2_form_filter": "sam2_formlearner",
        "sam2_formlearner_filter": "sam2_formlearner",
    }
    return aliases.get(s, s)


def _bbox_from_record(r: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    if "bbox" in r and isinstance(r.get("bbox"), (list, tuple)) and len(r.get("bbox") or []) >= 4:
        vals = list(r.get("bbox") or [])[:4]
        # Accept both xyxy and xywh. If third/fourth appear to be width/height, convert.
        x1, y1, a, b = [_safe_float(x, 0.0) for x in vals]
        if a > x1 and b > y1:
            return (x1, y1, a, b)
        return (x1, y1, x1 + max(1.0, a), y1 + max(1.0, b))
    aliases = {
        "x1": ("x1", "xmin", "minx", "left"),
        "y1": ("y1", "ymin", "miny", "top"),
        "x2": ("x2", "xmax", "maxx", "right"),
        "y2": ("y2", "ymax", "maxy", "bottom"),
    }
    vals: Dict[str, float] = {}
    for key, names in aliases.items():
        for n in names:
            if n in r:
                vals[key] = _safe_float(r.get(n), 0.0)
                break
    if all(k in vals for k in ("x1", "y1", "x2", "y2")):
        x1, y1, x2, y2 = vals["x1"], vals["y1"], vals["x2"], vals["y2"]
        if abs(x2 - x1) > 0 and abs(y2 - y1) > 0:
            return (x1, y1, x2, y2)
    return None


def _record_from_any(obj: Any) -> Optional[Dict[str, Any]]:
    if isinstance(obj, dict):
        d = dict(obj)
    else:
        d = {}
        for k in (
            "x1", "y1", "x2", "y2", "xmin", "ymin", "xmax", "ymax", "bbox",
            "conf", "confidence", "score", "class_id", "class", "label", "name",
            "form_score", "shape_score", "mask_area_px", "mask_fill_ratio"
        ):
            try:
                if hasattr(obj, k):
                    d[k] = getattr(obj, k)
            except Exception:
                pass
    bbox = _bbox_from_record(d)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    d["x1"], d["y1"], d["x2"], d["y2"] = float(x1), float(y1), float(x2), float(y2)
    if "label" not in d and "class" in d:
        d["label"] = d.get("class")
    if "conf" not in d:
        d["conf"] = d.get("confidence", d.get("score", d.get("form_score", 1.0)))
    return d


def _allowed_labels(text: str) -> List[str]:
    return [p.strip().lower() for p in str(text or "").replace(";", ",").split(",") if p.strip()]


def _is_positive_record(r: Dict[str, Any], positive_text: str) -> bool:
    allowed = _allowed_labels(positive_text)
    label = str(r.get("label", r.get("class", r.get("name", ""))) or "").strip().lower()
    cid = str(r.get("class_id", r.get("class", "")) or "").strip().lower()
    status = str(r.get("form_status", r.get("status", r.get("scientific_filter", ""))) or "").strip().lower()
    negative_words = {"false_positive", "false positive", "negative", "reject", "rejected", "background", "fp"}
    if label in negative_words or status in negative_words or status == "false_positive":
        return False
    if allowed:
        return label in allowed or cid in allowed
    # By Mustatil convention class_id 1 is often false_positive. Do not force this if label says otherwise.
    if cid == "1" and (not label or "false" in label or "negative" in label):
        return False
    return True


def _dedupe_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set(); out: List[Dict[str, Any]] = []
    for r in records:
        bbox = _bbox_from_record(r)
        if bbox is None:
            continue
        key = tuple(round(float(v), 2) for v in bbox)
        label = str(r.get("label", r.get("class_id", "")))
        full_key = key + (label,)
        if full_key in seen:
            continue
        seen.add(full_key)
        d = dict(r)
        d["x1"], d["y1"], d["x2"], d["y2"] = bbox
        out.append(d)
    return out


def _crop_gray_array(img: Any, bbox: Tuple[float, float, float, float], pad: int = 8):
    if img is None or Image is None:
        return None
    try:
        w, h = int(img.width), int(img.height)
        x1, y1, x2, y2 = bbox
        ix1 = max(0, int(math.floor(min(x1, x2))) - int(pad))
        iy1 = max(0, int(math.floor(min(y1, y2))) - int(pad))
        ix2 = min(w, int(math.ceil(max(x1, x2))) + int(pad))
        iy2 = min(h, int(math.ceil(max(y1, y2))) + int(pad))
        if ix2 <= ix1 or iy2 <= iy1:
            return None
        crop = img.crop((ix1, iy1, ix2, iy2)).convert("L")
        if _np is not None:
            return _np.asarray(crop, dtype="float32")
        return crop
    except Exception:
        return None


def _local_image_metrics(img: Any, bbox: Tuple[float, float, float, float]) -> Dict[str, float]:
    arr = _crop_gray_array(img, bbox, pad=8)
    if arr is None:
        return {"local_mean": 0.0, "local_std": 0.0, "local_contrast": 0.0, "edge_density": 0.0}
    if _np is not None and hasattr(arr, "shape"):
        try:
            if arr.size <= 0:
                raise RuntimeError("empty crop")
            mean = float(_np.mean(arr)); std = float(_np.std(arr))
            p5 = float(_np.percentile(arr, 5)); p95 = float(_np.percentile(arr, 95))
            contrast = _clamp((p95 - p5) / 255.0, 0.0, 1.0)
            if arr.shape[0] > 2 and arr.shape[1] > 2:
                gx = _np.abs(_np.diff(arr, axis=1))
                gy = _np.abs(_np.diff(arr, axis=0))
                edge = _clamp((float(_np.mean(gx)) + float(_np.mean(gy))) / 510.0, 0.0, 1.0)
            else:
                edge = 0.0
            return {"local_mean": mean, "local_std": std, "local_contrast": contrast, "edge_density": edge}
        except Exception:
            pass
    if ImageStat is not None:
        try:
            stat = ImageStat.Stat(arr)
            mean = float(stat.mean[0]); std = float(stat.stddev[0])
            return {"local_mean": mean, "local_std": std, "local_contrast": _clamp(std / 64.0, 0.0, 1.0), "edge_density": 0.0}
        except Exception:
            pass
    return {"local_mean": 0.0, "local_std": 0.0, "local_contrast": 0.0, "edge_density": 0.0}


def _features_for_record(r: Dict[str, Any], img: Any = None) -> Dict[str, float]:
    bbox = _bbox_from_record(r)
    if bbox is None:
        bbox = (0.0, 0.0, 1.0, 1.0)
    x1, y1, x2, y2 = bbox
    w = max(1.0, abs(x2 - x1)); h = max(1.0, abs(y2 - y1))
    area = max(1.0, w * h)
    cx = min(x1, x2) + w / 2.0; cy = min(y1, y2) + h / 2.0
    aspect = max(w / h, h / w)
    short_side = min(w, h); long_side = max(w, h)
    fill = _safe_float(r.get("mask_fill_ratio", r.get("extent", r.get("fill_ratio", 1.0))), 1.0)
    if "mask_area_px" in r:
        fill = _clamp(_safe_float(r.get("mask_area_px"), area) / max(1.0, area), 0.0, 1.0)
    elif "area_px" in r and _safe_float(r.get("area_px"), area) < area:
        fill = _clamp(_safe_float(r.get("area_px"), area) / max(1.0, area), 0.0, 1.0)
    elongation = _clamp(1.0 - short_side / long_side, 0.0, 1.0)
    compactness = _clamp(1.0 / max(1.0, aspect), 0.0, 1.0)
    ratio_target = 2.2
    aspect_score = math.exp(-abs(math.log(max(1.0, aspect) / max(1.001, ratio_target))))
    fill_score = 1.0 - min(1.0, abs(fill - 0.55) / 0.55)
    shape_score = _clamp(0.45 * aspect_score + 0.25 * fill_score + 0.30 * compactness, 0.0, 1.0)
    out = {
        "bbox_width_px": float(w),
        "bbox_height_px": float(h),
        "area_px": float(area),
        "center_x": float(cx),
        "center_y": float(cy),
        "aspect_ratio": float(aspect),
        "elongation": float(elongation),
        "rectangularity": float(fill),
        "compactness": float(compactness),
        "mask_fill_ratio": float(_clamp(fill, 0.0, 1.0)),
        "shape_score": float(shape_score),
        "form_score": _safe_float(r.get("form_score"), _safe_float(r.get("score"), _safe_float(r.get("conf"), 0.0))),
    }
    out.update(_local_image_metrics(img, bbox))
    return out


def _add_context_and_line_features(records: List[Dict[str, Any]], radius: float = 450.0, min_neighbors: int = 2) -> None:
    centers = []
    for r in records:
        f = dict(r.get("scientific_features") or {})
        centers.append((_safe_float(f.get("center_x"), 0.0), _safe_float(f.get("center_y"), 0.0)))
    for i, r in enumerate(records):
        cx, cy = centers[i]
        dists = [math.hypot(cx - ox, cy - oy) for j, (ox, oy) in enumerate(centers) if j != i]
        near = [d for d in dists if d <= radius]
        f = dict(r.get("scientific_features") or {})
        f["neighbor_count"] = float(len(near))
        f["nearest_distance_px"] = float(min(dists) if dists else 10**9)
        f["density_score"] = _clamp(len(near) / 4.0, 0.0, 1.0)
        # Alignment score from local neighborhood PCA.
        pts = [(cx, cy)] + [(ox, oy) for j, (ox, oy) in enumerate(centers) if j != i and math.hypot(cx - ox, cy - oy) <= radius]
        line_score = 0.0; orientation = 0.0
        if len(pts) >= max(3, int(min_neighbors) + 1):
            if _np is not None:
                try:
                    arr = _np.asarray(pts, dtype="float64")
                    arr = arr - arr.mean(axis=0, keepdims=True)
                    cov = _np.cov(arr.T)
                    vals, vecs = _np.linalg.eigh(cov)
                    order = _np.argsort(vals)[::-1]
                    vals = vals[order]; vecs = vecs[:, order]
                    den = max(1e-9, float(vals[0] + vals[1]))
                    line_score = _clamp(float((vals[0] - vals[1]) / den), 0.0, 1.0)
                    v = vecs[:, 0]
                    orientation = float((math.degrees(math.atan2(float(v[1]), float(v[0]))) + 360.0) % 180.0)
                except Exception:
                    line_score = 0.0
            if line_score <= 0.0:
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                dx = max(xs) - min(xs); dy = max(ys) - min(ys)
                long = max(dx, dy, 1.0); short = max(min(dx, dy), 1.0)
                line_score = _clamp(1.0 - short / long, 0.0, 1.0)
                orientation = 0.0 if dx >= dy else 90.0
        f["line_score"] = float(line_score)
        f["line_orientation_deg"] = float(orientation)
        r["scientific_features"] = f
        r.update(f)


def _iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ba = _bbox_from_record(a); bb = _bbox_from_record(b)
    if ba is None or bb is None:
        return 0.0
    ax1, ay1, ax2, ay2 = ba; bx1, by1, bx2, by2 = bb
    ax1, ax2 = min(ax1, ax2), max(ax1, ax2); ay1, ay2 = min(ay1, ay2), max(ay1, ay2)
    bx1, bx2 = min(bx1, bx2), max(bx1, bx2); by1, by2 = min(by1, by2), max(by1, by2)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1)); area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return _clamp(inter / max(1e-9, area_a + area_b - inter), 0.0, 1.0)


def _open_image(path: str):
    if not path or Image is None:
        return None
    try:
        p = Path(path).expanduser()
        if p.is_file():
            return Image.open(str(p))
    except Exception:
        pass
    return None


def _gpkg_bbox_from_blob(blob: Any) -> Optional[Tuple[float, float, float, float]]:
    if blob is None:
        return None
    try:
        b = bytes(blob)
    except Exception:
        return None
    if len(b) < 8:
        return None
    if b[:2] == b"GP":
        flags = b[3]
        endian = "<" if (flags & 1) else ">"
        env_code = (flags >> 1) & 0b111
        off = 8
        # Envelope order in GeoPackage binary: minx, maxx, miny, maxy, optional z/m.
        if env_code in (1, 2, 3, 4) and len(b) >= off + 32:
            try:
                minx, maxx, miny, maxy = struct.unpack(endian + "dddd", b[off:off + 32])
                return (float(minx), float(miny), float(maxx), float(maxy))
            except Exception:
                pass
        # Fallback to WKB after header/envelope.
        env_lens = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
        off = 8 + env_lens.get(env_code, 0)
        geom = b[off:]
    else:
        geom = b
    try:
        from shapely import wkb  # type: ignore
        g = wkb.loads(geom)
        minx, miny, maxx, maxy = g.bounds
        return (float(minx), float(miny), float(maxx), float(maxy))
    except Exception:
        return None


def _load_records_from_json(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        seq = data
    elif isinstance(data, dict):
        seq = []
        for k in ("records", "results", "pipeline_results", "detections", "features", "items", "boxes"):
            if isinstance(data.get(k), list):
                seq = data[k]
                break
        if not seq and all(k in data for k in ("x1", "y1", "x2", "y2")):
            seq = [data]
    else:
        seq = []
    return _dedupe_records(filter(None, (_record_from_any(x) for x in seq)))  # type: ignore[arg-type]


def _load_records_from_csv(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = _record_from_any(row)
            if r is not None:
                out.append(r)
    return _dedupe_records(out)


def _load_records_from_gpkg(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    con = sqlite3.connect(str(path))
    try:
        cur = con.cursor()
        tables: List[str] = []
        try:
            for (t,) in cur.execute("SELECT table_name FROM gpkg_contents WHERE data_type='features'").fetchall():
                tables.append(str(t))
        except Exception:
            pass
        if not tables:
            for (t,) in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
                if not str(t).startswith(("gpkg_", "rtree_", "sqlite_")):
                    tables.append(str(t))
        for table in tables:
            try:
                cols = [str(row[1]) for row in cur.execute(f'PRAGMA table_info("{table}")').fetchall()]
                if not cols:
                    continue
                geom_cols = [c for c in cols if c.lower() in ("geom", "geometry", "shape")]
                qcols = ", ".join([f'"{c}"' for c in cols])
                for row in cur.execute(f'SELECT {qcols} FROM "{table}" LIMIT 200000'):
                    d = dict(zip(cols, row))
                    bbox = _bbox_from_record(d)
                    if bbox is None:
                        for gc in geom_cols:
                            bbox = _gpkg_bbox_from_blob(d.get(gc))
                            if bbox is not None:
                                break
                    if bbox is not None:
                        d["x1"], d["y1"], d["x2"], d["y2"] = bbox
                        r = _record_from_any(d)
                        if r is not None:
                            out.append(r)
            except Exception:
                continue
    finally:
        con.close()
    return _dedupe_records(out)


def _load_records_from_file(path_text: str) -> List[Dict[str, Any]]:
    p = Path(str(path_text or "").strip().strip('"')).expanduser()
    if not p.is_file():
        return []
    suffix = p.suffix.lower()
    if suffix in (".json", ".geojson"):
        return _load_records_from_json(p)
    if suffix in (".csv", ".txt"):
        return _load_records_from_csv(p)
    if suffix in (".gpkg", ".sqlite", ".db"):
        return _load_records_from_gpkg(p)
    return []


def _all_app_widgets() -> List[Any]:
    try:
        app = QApplication.instance() if QApplication is not None else None
        return list(app.allWidgets()) if app is not None else []
    except Exception:
        return []


def _candidate_workspaces() -> List[Any]:
    out: List[Any] = []
    seen = set()
    for w in _all_app_widgets():
        try:
            cls_name = type(w).__name__.lower()
            obj_name = str(w.objectName() or "").lower() if hasattr(w, "objectName") else ""
            if hasattr(w, "tabs") or "mustatil" in cls_name or "mustatil" in obj_name:
                if id(w) not in seen:
                    out.append(w); seen.add(id(w))
        except Exception:
            pass
    return out


def _pipeline_tabs() -> List[Any]:
    out: List[Any] = []
    seen = set()
    for w in _all_app_widgets():
        try:
            if hasattr(w, "blocks") and hasattr(w, "run_pipeline") and (hasattr(w, "_add_block") or hasattr(w, "results")):
                if id(w) not in seen:
                    out.append(w); seen.add(id(w))
        except Exception:
            pass
    for ws in _candidate_workspaces():
        try:
            tab = getattr(ws, "yolo_pipeline_tab", None)
            if tab is not None and id(tab) not in seen:
                out.append(tab); seen.add(id(tab))
        except Exception:
            pass
    return out


def _selected_pipeline_block() -> Tuple[Optional[Any], Optional[Any]]:
    for tab in _pipeline_tabs():
        try:
            if hasattr(tab, "_current_block"):
                b = tab._current_block()
                if b is not None:
                    return tab, b
        except Exception:
            pass
        try:
            sid = getattr(tab, "selected_block_id", "")
            for b in list(getattr(tab, "blocks", []) or []):
                if getattr(b, "id", None) == sid:
                    return tab, b
        except Exception:
            pass
    return None, None


def _collect_records_from_runtime() -> List[Dict[str, Any]]:
    sources: List[Any] = []
    for tab in _pipeline_tabs():
        try:
            if getattr(tab, "results", None):
                sources.append(getattr(tab, "results"))
        except Exception:
            pass
        try:
            rb = getattr(tab, "results_by_block", None)
            if isinstance(rb, dict):
                for seq in rb.values():
                    if seq:
                        sources.append(seq)
        except Exception:
            pass
    attr_names = (
        "detections", "dets", "visible_detections", "satellite_detections", "pipeline_results",
        "annotation_boxes", "boxes", "rectangles", "items", "records", "current_detections"
    )
    for obj in _candidate_workspaces() + _all_app_widgets()[:2500]:
        for attr in attr_names:
            try:
                val = getattr(obj, attr, None)
                if hasattr(val, "get"):
                    val = val.get()
                if isinstance(val, dict):
                    for vv in val.values():
                        if isinstance(vv, (list, tuple)):
                            sources.append(vv)
                elif isinstance(val, (list, tuple)) and val:
                    sources.append(val)
            except Exception:
                pass
    out: List[Dict[str, Any]] = []
    for seq in sources:
        try:
            for obj in list(seq or []):
                r = _record_from_any(obj)
                if r is not None:
                    out.append(r)
        except Exception:
            continue
    return _dedupe_records(out)


def _trainable_records(source_file: str, positive_text: str) -> List[Dict[str, Any]]:
    records = _load_records_from_file(source_file) if str(source_file or "").strip() else _collect_records_from_runtime()
    records = [r for r in records if _is_positive_record(r, positive_text)]
    return _dedupe_records(records)


def _score_records(records: List[Dict[str, Any]], image_path: str = "") -> List[Dict[str, Any]]:
    img = _open_image(image_path)
    scored: List[Dict[str, Any]] = []
    for r in records:
        rec = dict(r)
        f = _features_for_record(rec, img)
        rec["scientific_features"] = f
        rec.update(f)
        scored.append(rec)
    _add_context_and_line_features(scored, radius=450.0, min_neighbors=2)
    return scored


def _estimate_nms_iou(records: List[Dict[str, Any]]) -> float:
    vals: List[float] = []
    n = min(len(records), 400)
    for i in range(n):
        for j in range(i + 1, n):
            v = _iou(records[i], records[j])
            if v > 0.05:
                vals.append(v)
    if not vals:
        return 0.60
    # Keep duplicate threshold safely above normal incidental overlaps.
    return _clamp(_percentile(vals, 0.85, 0.60), 0.45, 0.85)


def _build_filter_preset(filter_type: str, records: List[Dict[str, Any]], image_path: str = "", positive_text: str = "", formlearner_json: str = "") -> Dict[str, Any]:
    typ = _norm_type(filter_type)
    defaults = dict(DEFAULTS_BY_TYPE.get(typ, {}))
    scored = _score_records(records, image_path) if records else []
    fs = [r.get("scientific_features", {}) for r in scored]
    params: Dict[str, Any] = dict(defaults)
    if scored:
        if typ == "scientific_shape_filter":
            params.update({
                "area_min": max(1.0, _percentile([f.get("area_px", 1.0) for f in fs], 0.05, defaults.get("area_min", 16.0))),
                "area_max": max(1.0, _percentile([f.get("area_px", 1.0) for f in fs], 0.99, 1e12)),
                "aspect_min": max(1.0, _percentile([f.get("aspect_ratio", 1.0) for f in fs], 0.02, 1.0)),
                "aspect_max": max(1.0, _percentile([f.get("aspect_ratio", 1.0) for f in fs], 0.95, defaults.get("aspect_max", 12.0))),
                "fill_min": _clamp(_percentile([f.get("mask_fill_ratio", 0.05) for f in fs], 0.05, defaults.get("fill_min", 0.05)), 0.0, 1.0),
                "threshold": _clamp(_percentile([f.get("shape_score", 0.20) for f in fs], 0.15, defaults.get("threshold", 0.20)), 0.0, 1.0),
            })
        elif typ == "scientific_size_filter":
            params.update({
                "area_min": max(1.0, _percentile([f.get("area_px", 1.0) for f in fs], 0.05, defaults.get("area_min", 16.0))),
                "area_max": max(1.0, _percentile([f.get("area_px", 1.0) for f in fs], 0.95, defaults.get("area_max", 999999999.0))),
                "width_min": max(1.0, _percentile([f.get("bbox_width_px", 1.0) for f in fs], 0.05, 1.0)),
                "width_max": max(1.0, _percentile([f.get("bbox_width_px", 1.0) for f in fs], 0.95, 1e12)),
                "height_min": max(1.0, _percentile([f.get("bbox_height_px", 1.0) for f in fs], 0.05, 1.0)),
                "height_max": max(1.0, _percentile([f.get("bbox_height_px", 1.0) for f in fs], 0.95, 1e12)),
            })
        elif typ == "scientific_texture_filter":
            params.update({
                "contrast_min": _clamp(_percentile([f.get("local_contrast", 0.03) for f in fs], 0.20, defaults.get("contrast_min", 0.03)), 0.0, 1.0),
                "std_min": max(0.0, _percentile([f.get("local_std", 2.0) for f in fs], 0.20, defaults.get("std_min", 2.0))),
                "edge_min": _clamp(_percentile([f.get("edge_density", 0.0) for f in fs], 0.10, defaults.get("edge_min", 0.0)), 0.0, 1.0),
            })
        elif typ == "geo_nms_filter":
            params.update({"iou": _estimate_nms_iou(scored)})
        elif typ == "cluster_context_filter":
            nearest = [f.get("nearest_distance_px", 250.0) for f in fs if _safe_float(f.get("nearest_distance_px"), 1e9) < 1e8]
            radius = _percentile(nearest, 0.50, defaults.get("radius", 250.0)) * 1.5 if nearest else defaults.get("radius", 250.0)
            neigh = [f.get("neighbor_count", 0.0) for f in fs]
            params.update({
                "radius": _clamp(radius, 25.0, 1000000.0),
                "neighbors_min": max(0.0, math.floor(_percentile(neigh, 0.15, defaults.get("neighbors_min", 1.0)))),
                "neighbors_max": max(1.0, math.ceil(_percentile(neigh, 0.98, defaults.get("neighbors_max", 999999999.0)))),
            })
        elif typ == "linear_alignment_filter":
            nearest = [f.get("nearest_distance_px", 300.0) for f in fs if _safe_float(f.get("nearest_distance_px"), 1e9) < 1e8]
            radius = _percentile(nearest, 0.50, defaults.get("radius", 450.0)) * 2.0 if nearest else defaults.get("radius", 450.0)
            params.update({
                "radius": _clamp(radius, 50.0, 1000000.0),
                "line_score_min": _clamp(_percentile([f.get("line_score", 0.55) for f in fs], 0.40, defaults.get("line_score_min", 0.55)), 0.05, 0.99),
                "neighbors_min": max(1.0, math.floor(_percentile([f.get("neighbor_count", 2.0) for f in fs], 0.25, defaults.get("neighbors_min", 2.0)))),
            })
        elif typ == "scientific_formlearner_filter":
            # Keep weights transparent. If records already have form_score, use its lower-positive percentile.
            form_scores = [f.get("form_score", 0.0) for f in fs if _safe_float(f.get("form_score"), 0.0) > 0]
            mix_min = _percentile(form_scores, 0.15, defaults.get("mix_score_min", 0.50)) if form_scores else defaults.get("mix_score_min", 0.50)
            params.update({
                "mix_score_min": _clamp(mix_min, 0.0, 1.0),
                "weight_form": defaults.get("weight_form", 0.55),
                "weight_shape": defaults.get("weight_shape", 0.25),
                "weight_texture": defaults.get("weight_texture", 0.10),
                "weight_context": defaults.get("weight_context", 0.10),
            })
        elif typ == "sam2_formlearner":
            form_scores = [f.get("form_score", 0.0) for f in fs if _safe_float(f.get("form_score"), 0.0) > 0]
            params.update({
                "form_threshold": _clamp(_percentile(form_scores, 0.15, defaults.get("form_threshold", 0.50)), 0.0, 1.0) if form_scores else defaults.get("form_threshold", 0.50),
                "shape_threshold": _clamp(_percentile([f.get("shape_score", 0.20) for f in fs], 0.15, defaults.get("shape_threshold", 0.20)), 0.0, 1.0),
            })
    params = _round_params(params)
    preset = {
        "mustatil_filter_preset_version": 1,
        "filter_type": typ,
        "pipeline_block_type": typ,
        "label": FILTER_LABEL_BY_TYPE.get(typ, typ),
        "scientific_feature_set": "morphometry_texture_context_v1",
        "trained": bool(scored),
        "sample_count": int(len(scored)),
        "positive_selector": str(positive_text or ""),
        "created_utc": _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "image_source": str(image_path or ""),
        "params": params,
    }
    if formlearner_json:
        preset["formlearner_model"] = str(formlearner_json)
    # Duplicate simple params at top level because the current Pipeline parser merges JSON dicts directly.
    preset.update(params)
    if not scored:
        preset["reason"] = "No positive current examples found; saved safe default parameters."
    return preset


def _params_text(params: Dict[str, Any]) -> str:
    parts = []
    for k, v in params.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.6g}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def _save_preset(path: str, preset: Dict[str, Any]) -> str:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(preset, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(p)


def _default_output_dir() -> str:
    try:
        return str(Path.home() / "Documents" / "Mustatil" / "filter_presets")
    except Exception:
        return os.getcwd()


def _safe_filename_filter_type(typ: str) -> str:
    return _norm_type(typ).replace("/", "_").replace("\\", "_")


def _replace_or_append_option(text: str, key: str, value: str) -> str:
    key_l = key.lower()
    parts = [p.strip() for p in str(text or "").replace(";", ",").split(",") if p.strip()]
    out = []
    replaced = False
    for part in parts:
        if "=" in part and part.split("=", 1)[0].strip().lower().replace("-", "_") == key_l:
            out.append(f"{key}={value}")
            replaced = True
        else:
            out.append(part)
    if not replaced:
        out.insert(0, f"{key}={value}")
    return ", ".join(out)


def _apply_preset_to_selected_block(preset_path: str, preset: Optional[Dict[str, Any]] = None) -> str:
    if preset is None:
        try:
            preset = json.loads(Path(preset_path).read_text(encoding="utf-8"))
        except Exception:
            preset = {}
    typ = _norm_type((preset or {}).get("pipeline_block_type") or (preset or {}).get("filter_type") or "")
    tab, b = _selected_pipeline_block()
    if tab is None or b is None:
        return "No selected AI Pipeline block found. Select a filter block in the Pipeline tab first."
    btyp = _norm_type(getattr(b, "type", ""))
    if typ and btyp != typ:
        # Accept algorithm preset into scientific_formlearner only if user intentionally selected it? Keep strict to avoid mistakes.
        return f"Selected block is '{btyp}', but preset is for '{typ}'. Select the matching block first."
    try:
        if btyp in ALGORITHM_ONLY_FILTER_TYPES or btyp == "scientific_formlearner_filter":
            b.model_path = str(preset_path)
            b.classes_filter = _replace_or_append_option(str(getattr(b, "classes_filter", "") or ""), "preset", str(preset_path))
            params = dict((preset or {}).get("params") or {})
            for k, v in params.items():
                if k in {"area_min", "area_max", "aspect_max", "fill_min", "contrast_min", "std_min", "edge_min", "iou", "radius", "neighbors_min", "line_score_min", "mix_score_min", "weight_form"}:
                    b.classes_filter = _replace_or_append_option(str(getattr(b, "classes_filter", "") or ""), k, str(v))
            if btyp == "scientific_formlearner_filter" and (preset or {}).get("formlearner_model"):
                b.formlearner_model = str((preset or {}).get("formlearner_model") or "")
        elif btyp == "sam2_formlearner":
            params = dict((preset or {}).get("params") or {})
            if "form_threshold" in params:
                b.form_threshold = float(params.get("form_threshold") or 0.50)
            b.classes_filter = _replace_or_append_option(str(getattr(b, "classes_filter", "") or ""), "preset", str(preset_path))
            if (preset or {}).get("formlearner_model"):
                b.formlearner_model = str((preset or {}).get("formlearner_model") or "")
        else:
            return f"Selected block '{btyp}' is not a trainable pipeline filter block."
        # Refresh UI without assuming one exact patch version.
        try:
            if hasattr(tab, "_load_block_to_editor"):
                tab._load_block_to_editor(b)
        except Exception:
            pass
        try:
            if hasattr(tab, "refresh_code_from_blocks"):
                tab.refresh_code_from_blocks()
        except Exception:
            pass
        try:
            items = getattr(tab, "block_items", {}) or {}
            item = items.get(getattr(b, "id", ""))
            if item is not None and hasattr(item, "refresh_embedded_controls"):
                item.refresh_embedded_controls()
            if item is not None:
                item.update()
        except Exception:
            pass
        try:
            if hasattr(tab, "scene"):
                tab.scene.update()
        except Exception:
            pass
        return f"Applied preset to selected block: {getattr(b, 'name', btyp)}"
    except Exception as exc:
        return f"Could not apply preset: {exc}"


def _hide_old_duplicate_panels(page: Any) -> None:
    try:
        for box in page.findChildren(QGroupBox):
            try:
                name = str(box.objectName() or "")
                title = str(box.title() or "")
                if name in OLD_DUPLICATE_OBJECT_NAMES or title.strip().lower() == "scientific filter training presets":
                    box.setVisible(False)
            except Exception:
                pass
    except Exception:
        pass


def _append_log(edit: Any, msg: str) -> None:
    try:
        edit.appendPlainText(str(msg))
    except Exception:
        pass


def _install_panel_on_page(page: Any) -> bool:
    if page is None or QApplication is None:
        return False
    try:
        if page.findChild(QGroupBox, NEW_BOX_OBJECT_NAME) is not None:
            _hide_old_duplicate_panels(page)
            return True
    except Exception:
        pass
    try:
        _hide_old_duplicate_panels(page)
        lay = page.layout()
        if lay is None:
            lay = QVBoxLayout(page)
            page.setLayout(lay)
        box = QGroupBox("Pipeline filter model trainer")
        box.setObjectName(NEW_BOX_OBJECT_NAME)
        box.setToolTip("Trains/calibrates JSON filter-model presets for the AI Pipeline filter blocks.")
        root = QVBoxLayout(box)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        intro = QLabel(
            "Train/calibrate the filter models used by the AI Pipeline. "
            "Algorithmic filters save transparent JSON presets; those presets can be used directly in the matching Pipeline filter block."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        grid = QGridLayout(); grid.setHorizontalSpacing(6); grid.setVerticalSpacing(5)
        row = 0
        grid.addWidget(QLabel("Filter model"), row, 0)
        filter_combo = QComboBox()
        for label, typ, _defaults in FILTER_SPECS:
            filter_combo.addItem(label, typ)
        grid.addWidget(filter_combo, row, 1, 1, 3)

        row += 1
        grid.addWidget(QLabel("Positive classes"), row, 0)
        positive_edit = QLineEdit("mustatil, 0, positive, object")
        positive_edit.setPlaceholderText("Comma-separated labels/class ids; blank = all except false_positive")
        grid.addWidget(positive_edit, row, 1, 1, 3)

        row += 1
        grid.addWidget(QLabel("Boxes source"), row, 0)
        source_file = QLineEdit("")
        source_file.setPlaceholderText("Blank = current Mustatil detections / pipeline results, or choose JSON/GeoJSON/GPKG/CSV")
        browse_source = QPushButton("Choose boxes…")
        clear_source = QPushButton("Use current")
        grid.addWidget(source_file, row, 1)
        grid.addWidget(browse_source, row, 2)
        grid.addWidget(clear_source, row, 3)

        row += 1
        grid.addWidget(QLabel("Image for texture"), row, 0)
        image_file = QLineEdit("")
        image_file.setPlaceholderText("Optional image/GeoTIFF; texture filter works better with this")
        browse_image = QPushButton("Choose image…")
        clear_image = QPushButton("No image")
        grid.addWidget(image_file, row, 1)
        grid.addWidget(browse_image, row, 2)
        grid.addWidget(clear_image, row, 3)

        row += 1
        grid.addWidget(QLabel("FormLearner JSON"), row, 0)
        form_file = QLineEdit("")
        form_file.setPlaceholderText("Optional, only for Scientific FormLearner / SAM2+FormLearner helper")
        browse_form = QPushButton("Choose JSON…")
        grid.addWidget(form_file, row, 1)
        grid.addWidget(browse_form, row, 2)

        row += 1
        grid.addWidget(QLabel("Output folder"), row, 0)
        out_dir = QLineEdit(_default_output_dir())
        browse_out = QPushButton("Folder…")
        grid.addWidget(out_dir, row, 1)
        grid.addWidget(browse_out, row, 2)

        root.addLayout(grid)

        opts_row = QHBoxLayout()
        auto_apply = QCheckBox("Apply saved preset to selected Pipeline block")
        auto_apply.setChecked(False)
        show_details = QCheckBox("Show JSON preview")
        show_details.setChecked(True)
        opts_row.addWidget(auto_apply)
        opts_row.addWidget(show_details)
        opts_row.addStretch(1)
        root.addLayout(opts_row)

        btn_row = QHBoxLayout()
        scan_btn = QPushButton("Scan/count examples")
        train_btn = QPushButton("Train selected preset…")
        train_all_btn = QPushButton("Train ALL presets")
        apply_btn = QPushButton("Apply existing preset…")
        btn_row.addWidget(scan_btn)
        btn_row.addWidget(train_btn)
        btn_row.addWidget(train_all_btn)
        btn_row.addWidget(apply_btn)
        root.addLayout(btn_row)

        status = QLabel("")
        status.setWordWrap(True)
        root.addWidget(status)
        preview = QPlainTextEdit()
        preview.setObjectName("MustatilPipelineFilterTrainerPreview")
        preview.setMaximumHeight(170)
        preview.setPlaceholderText("Training log / preset preview")
        root.addWidget(preview)

        state: Dict[str, Any] = {"last_preset_path": "", "last_preset": None}

        def collect_now() -> List[Dict[str, Any]]:
            recs = _trainable_records(source_file.text(), positive_edit.text())
            return recs

        def update_count() -> None:
            try:
                recs = collect_now()
                msg = f"Positive examples found: {len(recs)}"
                if source_file.text().strip():
                    msg += f" | source: {source_file.text().strip()}"
                else:
                    msg += " | source: current Mustatil runtime"
                status.setText(msg)
                _append_log(preview, msg)
            except Exception as exc:
                status.setText(f"Scan failed: {exc}")
                _append_log(preview, traceback.format_exc())

        def choose_source() -> None:
            path, _ = QFileDialog.getOpenFileName(page, "Choose boxes/results file", "", "Boxes/results (*.json *.geojson *.gpkg *.csv);;All files (*)")
            if path:
                source_file.setText(path)
                update_count()

        def choose_image() -> None:
            path, _ = QFileDialog.getOpenFileName(page, "Choose image for texture training", "", "Images (*.tif *.tiff *.png *.jpg *.jpeg *.webp);;All files (*)")
            if path:
                image_file.setText(path)

        def choose_form() -> None:
            path, _ = QFileDialog.getOpenFileName(page, "Choose FormLearner JSON", "", "FormLearner JSON (*.json);;All files (*)")
            if path:
                form_file.setText(path)

        def choose_folder() -> None:
            path = QFileDialog.getExistingDirectory(page, "Choose output folder", out_dir.text() or _default_output_dir())
            if path:
                out_dir.setText(path)

        def train_selected(save_dialog: bool = True) -> Optional[str]:
            try:
                typ = _norm_type(filter_combo.currentData())
                recs = collect_now()
                preset = _build_filter_preset(typ, recs, image_file.text().strip(), positive_edit.text(), form_file.text().strip())
                default_name = f"mustatil_{_safe_filename_filter_type(typ)}_preset.json"
                default_path = str(Path(out_dir.text() or _default_output_dir()) / default_name)
                if save_dialog:
                    path, _ = QFileDialog.getSaveFileName(page, "Save Pipeline filter model preset", default_path, "JSON preset (*.json);;All files (*)")
                    if not path:
                        return None
                else:
                    path = default_path
                saved = _save_preset(path, preset)
                state["last_preset_path"] = saved; state["last_preset"] = preset
                msg = f"Saved {FILTER_LABEL_BY_TYPE.get(typ, typ)} preset: {saved} | samples={preset.get('sample_count', 0)}"
                status.setText(msg)
                if show_details.isChecked():
                    preview.setPlainText(json.dumps(preset, indent=2, ensure_ascii=False))
                else:
                    _append_log(preview, msg)
                if auto_apply.isChecked():
                    status.setText(msg + "\n" + _apply_preset_to_selected_block(saved, preset))
                return saved
            except Exception as exc:
                status.setText(f"Training failed: {exc}")
                preview.setPlainText(traceback.format_exc())
                return None

        def train_all() -> None:
            try:
                recs = collect_now()
                folder = Path(out_dir.text() or _default_output_dir()).expanduser()
                folder.mkdir(parents=True, exist_ok=True)
                saved_paths = []
                for _label, typ, _defs in FILTER_SPECS:
                    preset = _build_filter_preset(typ, recs, image_file.text().strip(), positive_edit.text(), form_file.text().strip())
                    path = folder / f"mustatil_{_safe_filename_filter_type(typ)}_preset.json"
                    _save_preset(str(path), preset)
                    saved_paths.append(str(path))
                state["last_preset_path"] = saved_paths[0] if saved_paths else ""
                status.setText(f"Saved {len(saved_paths)} Pipeline filter presets to: {folder}")
                preview.setPlainText("Saved presets:\n" + "\n".join(saved_paths))
            except Exception as exc:
                status.setText(f"Train ALL failed: {exc}")
                preview.setPlainText(traceback.format_exc())

        def apply_existing() -> None:
            path = str(state.get("last_preset_path") or "")
            if not path:
                path, _ = QFileDialog.getOpenFileName(page, "Choose Pipeline filter preset", out_dir.text() or _default_output_dir(), "JSON preset (*.json);;All files (*)")
            if not path:
                return
            try:
                preset = json.loads(Path(path).read_text(encoding="utf-8"))
            except Exception:
                preset = None
            msg = _apply_preset_to_selected_block(path, preset)
            status.setText(msg)
            _append_log(preview, msg)

        browse_source.clicked.connect(choose_source)
        clear_source.clicked.connect(lambda: (source_file.setText(""), update_count()))
        browse_image.clicked.connect(choose_image)
        clear_image.clicked.connect(lambda: image_file.setText(""))
        browse_form.clicked.connect(choose_form)
        browse_out.clicked.connect(choose_folder)
        scan_btn.clicked.connect(update_count)
        train_btn.clicked.connect(lambda: train_selected(True))
        train_all_btn.clicked.connect(train_all)
        apply_btn.clicked.connect(apply_existing)

        # Insert near bottom without covering existing FormTrainer tools.
        try:
            lay.addWidget(box)
        except Exception:
            try:
                lay.insertWidget(max(0, lay.count() - 1), box)
            except Exception:
                return False
        _log("Installed FormTrainer Pipeline filter model trainer panel.")
        return True
    except Exception:
        traceback.print_exc()
        return False


def _find_formtrainer_pages() -> List[Any]:
    pages: List[Any] = []
    seen = set()
    for w in _all_app_widgets():
        try:
            if isinstance(w, QTabWidget):
                for i in range(w.count()):
                    title = str(w.tabText(i) or "").lower()
                    if "form" in title and ("learner" in title or "trainer" in title):
                        page = w.widget(i)
                        if page is not None and id(page) not in seen:
                            pages.append(page); seen.add(id(page))
        except Exception:
            pass
    for w in _all_app_widgets():
        try:
            name = (str(w.objectName() or "") + " " + str(getattr(w, "windowTitle", lambda: "")())).lower()
            if "form" in name and ("learner" in name or "trainer" in name) and id(w) not in seen:
                pages.append(w); seen.add(id(w))
        except Exception:
            pass
    return pages


def _scan_and_install() -> None:
    try:
        installed = False
        for page in _find_formtrainer_pages():
            installed = _install_panel_on_page(page) or installed
        if installed:
            _log("active")
    except Exception:
        traceback.print_exc()
    # Keep scanning briefly because Mustatil creates trainer tabs from other plugins after startup.
    try:
        now = _dt.datetime.utcnow().timestamp()
        if not hasattr(_scan_and_install, "_until"):
            _scan_and_install._until = now + 30.0  # type: ignore[attr-defined]
        if now < getattr(_scan_and_install, "_until", now):
            QTimer.singleShot(900, _scan_and_install)
    except Exception:
        pass


def mustatil_plugin_init(*args, **kwargs) -> None:
    if QApplication is None:
        return
    try:
        QTimer.singleShot(100, _scan_and_install)
        QTimer.singleShot(900, _scan_and_install)
        QTimer.singleShot(2200, _scan_and_install)
        _log("initialized")
    except Exception:
        traceback.print_exc()


def register_plugin(*args, **kwargs):
    mustatil_plugin_init(*args, **kwargs)


def init_plugin(*args, **kwargs):
    mustatil_plugin_init(*args, **kwargs)


def load_plugin(*args, **kwargs):
    mustatil_plugin_init(*args, **kwargs)


try:
    mustatil_plugin_init()
except Exception:
    pass
