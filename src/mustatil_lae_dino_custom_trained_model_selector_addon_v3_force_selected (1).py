#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mustatil plugin: class filter + Geo NMS + selected-class export.

FIXED: class dropdown now uses a workspace-level selected-class state, so Detection preview updates even when Satellite tab also has its own dropdown.

Install:
    Copy this file directly into mustatil_plugins/ next to mustatil_qt_workspace.py.

What it does:
- Hooks QTabWidget.addTab, so it patches tabs when Mustatil creates them.
- Adds an in-page "Class filter / Geo NMS" box to Detection and Satellite Detection.
- Intended position: directly below existing filter sliders when the left control layout can be detected.
- Adds a dropdown to choose which class is shown/exported:
    All classes
    Class 0 / positive / mustatil
    Class 1 / false_positive
- Adds optional Geo NMS with an IoU threshold to remove duplicate YOLO boxes.
- Adds "Export selected class" button.
- Keeps the old false-positive behavior as a checkbox:
    "Hide class 1 / false_positive"

Notes:
- This plugin does not create a floating toolbar and does not create a separate tab.
- It works on Mustatil Det objects and satellite record dictionaries as far as possible.
- Geo NMS uses bbox IoU. For detection objects it uses x1/y1/x2/y2-like attributes.
- Export uses Mustatil's own export helpers when available.
"""
from __future__ import annotations

import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

_PATCHED_QTABWIDGET = False
_PATCHED_WIDGET_IDS = set()
_PATCHED_WORKSPACE_IDS = set()
_ORIGINAL_QTAB_ADD = None


def _log(msg: str) -> None:
    try:
        print("[Mustatil Class Filter / Geo NMS] " + str(msg))
    except Exception:
        pass


def _txt(obj: Any) -> str:
    try:
        if hasattr(obj, "text") and callable(obj.text):
            return str(obj.text() or "")
    except Exception:
        pass
    try:
        if hasattr(obj, "title") and callable(obj.title):
            return str(obj.title() or "")
    except Exception:
        pass
    return ""


def _workspace_from_widget(widget: Any) -> Optional[Any]:
    """Walk parents until the main Mustatil window is found."""
    cur = widget
    for _ in range(80):
        if cur is None:
            break
        if hasattr(cur, "tabs") and (hasattr(cur, "dets") or hasattr(cur, "satellite_detections")):
            return cur
        try:
            cur = cur.parentWidget()
        except Exception:
            break
    return None


def _find_left_controls_layout(page: Any):
    """Return the QVBoxLayout of the left control panel in a Detection-like tab."""
    try:
        from PySide6.QtWidgets import QSplitter, QScrollArea
    except Exception:
        return None
    try:
        splitters = page.findChildren(QSplitter)
        for splitter in splitters:
            if splitter.count() < 1:
                continue
            left = splitter.widget(0)
            # Mustatil often wraps controls in self._scroll(controls), so this is normally QScrollArea.
            if isinstance(left, QScrollArea):
                controls = left.widget()
                if controls is not None and controls.layout() is not None:
                    return controls.layout()
            if left is not None and left.layout() is not None:
                return left.layout()
    except Exception:
        pass
    try:
        return page.layout()
    except Exception:
        return None


def _widget_text_recursive(widget: Any) -> str:
    parts = []
    try:
        t = _txt(widget)
        if t:
            parts.append(t)
    except Exception:
        pass
    try:
        for child in widget.findChildren(object):
            t = _txt(child)
            if t:
                parts.append(t)
    except Exception:
        pass
    return " ".join(parts).lower()


def _insert_under_filter_sliders(layout: Any, box: Any) -> None:
    """
    Best-effort insertion under existing filter sliders.
    If we cannot detect them, insert before final stretch / near bottom.
    """
    try:
        candidate = -1
        keywords = [
            "filter", "confidence", "score", "form", "slider",
            "threshold", "iou", "visible", "minimum", "maximum"
        ]
        for i in range(layout.count()):
            item = layout.itemAt(i)
            w = item.widget() if item is not None else None
            if w is None:
                continue
            text = _widget_text_recursive(w)
            if any(k in text for k in keywords):
                candidate = i
        if candidate >= 0:
            layout.insertWidget(candidate + 1, box)
            return
    except Exception:
        pass

    try:
        count = layout.count()
        insert_at = max(0, count - 1)
        layout.insertWidget(insert_at, box)
    except Exception:
        layout.addWidget(box)


def _get_value(obj: Any, keys: Iterable[str], default: Any = None) -> Any:
    if isinstance(obj, dict):
        for key in keys:
            if key in obj:
                return obj.get(key)
        return default
    for key in keys:
        try:
            if hasattr(obj, key):
                return getattr(obj, key)
        except Exception:
            pass
    return default


def _det_class(det: Any) -> Optional[int]:
    value = _get_value(det, ("cls", "class_id", "class", "category_id", "label_id"), None)
    try:
        if value is not None and str(value).strip() != "":
            return int(float(value))
    except Exception:
        pass

    # Some pipeline exports use duplicated suffix columns.
    value = _get_value(det, ("class_id_741", "cls_741", "class_741"), None)
    try:
        if value is not None and str(value).strip() != "":
            return int(float(value))
    except Exception:
        pass

    text = str(_get_value(det, ("status", "label", "class_name", "class_label", "name"), "") or "").strip().lower()
    if text in {"mustatil", "positive", "pos", "true_positive", "true positive"}:
        return 0
    if text in {"false_positive", "false positive", "false-positive", "fp"}:
        return 1
    return None


def _det_label(det: Any) -> str:
    value = _get_value(det, ("label", "class_name", "class_label", "name", "status"), "")
    return str(value or "")


def _is_false_positive(det: Any) -> bool:
    cls = _det_class(det)
    if cls == 1:
        return True
    text = _det_label(det).strip().lower()
    return text in {"false_positive", "false positive", "false-positive", "fp"}


def _selected_class(ws: Any) -> Optional[int]:
    """Return the global selected class state.

    Important: Detection and Satellite tabs each have their own dropdown.
    Older builds stored only the last created combo on ws, so the Satellite combo
    could overwrite the Detection combo and changing the Detection dropdown did
    not affect preview. We now store the selected value directly on the workspace.
    """
    try:
        value = getattr(ws, "mustatil_selected_class_id", None)
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        pass
    try:
        combo = getattr(ws, "mustatil_class_filter_combo", None)
        if combo is None:
            return None
        data = combo.currentData()
        if data is None or data == "":
            return None
        return int(data)
    except Exception:
        return None


def _hide_fp_enabled(ws: Any) -> bool:
    try:
        return bool(getattr(ws, "mustatil_hide_false_positives", False))
    except Exception:
        return False


def _geo_nms_enabled(ws: Any) -> bool:
    try:
        return bool(getattr(ws, "mustatil_geo_nms_enabled", False))
    except Exception:
        return False


def _geo_nms_iou(ws: Any) -> float:
    try:
        spin = getattr(ws, "mustatil_geo_nms_iou_spin", None)
        if spin is not None:
            return float(spin.value())
    except Exception:
        pass
    return 0.35


def _filter_by_selected_class(items: Iterable[Any], ws: Any) -> List[Any]:
    selected = _selected_class(ws)
    out = []
    for item in list(items or []):
        if _hide_fp_enabled(ws) and _is_false_positive(item):
            continue
        if selected is not None:
            cls = _det_class(item)
            if cls != selected:
                continue
        out.append(item)
    return out


def _score(item: Any) -> float:
    value = _get_value(item, ("confidence", "conf", "score", "rule_score", "form_score", "probability"), 0.0)
    try:
        if value is None:
            return 0.0
        v = float(value)
        if math.isnan(v):
            return 0.0
        return v
    except Exception:
        return 0.0


def _bbox_from_item(item: Any) -> Optional[Tuple[float, float, float, float]]:
    """
    Return x1,y1,x2,y2 for Mustatil detection objects or dict records.
    Handles common names from detection, satellite and pipeline exports.
    """
    # Direct bbox fields
    for keys in [
        ("x1", "y1", "x2", "y2"),
        ("px_x1", "px_y1", "px_x2", "px_y2"),
        ("x1_740", "y1_740", "x2_740", "y2_740"),
        ("left", "top", "right", "bottom"),
    ]:
        vals = [_get_value(item, (k,), None) for k in keys]
        if all(v is not None for v in vals):
            try:
                x1, y1, x2, y2 = [float(v) for v in vals]
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1
                return (x1, y1, x2, y2)
            except Exception:
                pass

    # pixel_bbox stored as string/list
    raw_bbox = _get_value(item, ("pixel_bbox", "bbox", "box"), None)
    if raw_bbox is not None:
        try:
            if isinstance(raw_bbox, str):
                cleaned = raw_bbox.strip().strip("[]()")
                parts = [p.strip() for p in cleaned.replace(";", ",").split(",") if p.strip()]
                if len(parts) >= 4:
                    vals = [float(p) for p in parts[:4]]
                else:
                    vals = []
            else:
                vals = list(raw_bbox)[:4]
                vals = [float(v) for v in vals]
            if len(vals) == 4:
                x1, y1, x2, y2 = vals
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1
                return (x1, y1, x2, y2)
        except Exception:
            pass

    # Geometry bounds if present, e.g. geopandas row-like dict
    geom = _get_value(item, ("geometry", "geom"), None)
    if geom is not None:
        try:
            b = geom.bounds
            return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        except Exception:
            pass

    return None


def _bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _geo_nms_items(items: Iterable[Any], iou_threshold: float = 0.35, class_aware: bool = True) -> List[Any]:
    data = list(items or [])
    if len(data) <= 1:
        return data

    # Sort by score descending, stable enough for equal scores.
    data_sorted = sorted(enumerate(data), key=lambda p: _score(p[1]), reverse=True)
    kept_pairs = []

    for original_index, item in data_sorted:
        box = _bbox_from_item(item)
        if box is None:
            # Keep items without bbox. They cannot be safely deduplicated.
            kept_pairs.append((original_index, item, None))
            continue

        cls = _det_class(item)
        suppress = False

        for _, kept_item, kept_box in kept_pairs:
            if kept_box is None:
                continue
            if class_aware and _det_class(kept_item) != cls:
                continue
            if _bbox_iou(box, kept_box) >= float(iou_threshold):
                suppress = True
                break

        if not suppress:
            kept_pairs.append((original_index, item, box))

    # Preserve original visual/export order after NMS.
    kept_pairs.sort(key=lambda p: p[0])
    return [p[1] for p in kept_pairs]


def _apply_current_filters(items: Iterable[Any], ws: Any) -> List[Any]:
    out = _filter_by_selected_class(items, ws)
    if _geo_nms_enabled(ws):
        out = _geo_nms_items(out, iou_threshold=_geo_nms_iou(ws), class_aware=True)
    return out


def _patch_workspace_methods(ws: Any) -> None:
    """Patch instance preview filter methods after the workspace exists."""
    if ws is None or id(ws) in _PATCHED_WORKSPACE_IDS:
        return
    _PATCHED_WORKSPACE_IDS.add(id(ws))

    # Detection tab: visible() is used by redraw() and export_geo().
    try:
        old_visible = getattr(ws, "visible", None)
        if callable(old_visible) and not getattr(ws, "_class_geo_nms_visible_patched", False):
            def visible_class_geo_filtered(*args, **kwargs):
                out = old_visible(*args, **kwargs)
                return _apply_current_filters(out, ws)
            ws.visible = visible_class_geo_filtered
            ws._class_geo_nms_visible_patched = True
            _log("Patched Detection visible() class filter / Geo NMS.")
    except Exception as exc:
        _log("Detection visible() patch failed: " + str(exc))

    # Satellite tab: _satellite_visible_records() is used by overlay/export.
    try:
        old_sat_visible = getattr(ws, "_satellite_visible_records", None)
        if callable(old_sat_visible) and not getattr(ws, "_class_geo_nms_sat_visible_patched", False):
            def satellite_visible_class_geo_filtered(records=None, *args, **kwargs):
                out = old_sat_visible(records, *args, **kwargs)
                return _apply_current_filters(out, ws)
            ws._satellite_visible_records = satellite_visible_class_geo_filtered
            ws._class_geo_nms_sat_visible_patched = True
            _log("Patched Satellite visible-record class filter / Geo NMS.")
    except Exception as exc:
        _log("Satellite visible-record patch failed: " + str(exc))


def _current_output_path(ws: Any, attr_name: str, fallback_name: str) -> Path:
    raw = ""
    try:
        obj = getattr(ws, attr_name, None)
        if hasattr(obj, "get"):
            raw = str(obj.get() or "").strip().strip('"')
        elif obj:
            raw = str(obj or "").strip().strip('"')
    except Exception:
        raw = ""
    if raw:
        p = Path(raw).expanduser()
        if p.suffix.lower() not in (".gpkg", ".geojson"):
            p = p.with_suffix(".gpkg")
        return p
    try:
        project = str(getattr(ws, "project", None).get() or "").strip()
        root = Path(project).expanduser() if project else Path.home()
    except Exception:
        root = Path.home()
    return root / "exports" / fallback_name


def _suffix_selected_class(path: Path, selected: Optional[int]) -> Path:
    suffix = path.suffix or ".gpkg"
    if suffix.lower() != ".geojson":
        suffix = ".gpkg"
    cls_text = "all_classes" if selected is None else f"class_{selected}"
    if "selected_class" in path.stem.lower():
        return path.with_suffix(suffix)
    return path.with_name(path.stem + f"_selected_class_{cls_text}" + suffix)


def _choose_save_path(parent: Any, start: Path, title: str) -> Optional[Path]:
    try:
        from PySide6.QtWidgets import QFileDialog
        start.parent.mkdir(parents=True, exist_ok=True)
        fn, _ = QFileDialog.getSaveFileName(
            parent,
            title,
            str(start),
            "GeoPackage (*.gpkg);;GeoJSON (*.geojson);;All files (*)",
        )
        if not fn:
            return None
        p = Path(fn).expanduser()
        if p.suffix.lower() not in (".gpkg", ".geojson"):
            p = p.with_suffix(".gpkg")
        return p
    except Exception:
        return start


def _export_selected_detection_class(ws: Any, parent: Any = None) -> None:
    try:
        # Use unpatched source if possible, then apply current filters once.
        if hasattr(ws, "dets"):
            dets = list(getattr(ws, "dets", []) or [])
        elif callable(getattr(ws, "visible", None)):
            dets = list(ws.visible() or [])
        else:
            dets = []
        kept = _apply_current_filters(dets, ws)
        selected = _selected_class(ws)
        removed = len(dets) - len(kept)
        if not kept:
            raise RuntimeError("No detections match the selected class/filter settings.")

        base = _current_output_path(ws, "output", "detections_selected_class.gpkg")
        out_path = _choose_save_path(parent or ws, _suffix_selected_class(base, selected), "Export selected class")
        if out_path is None:
            return

        if not hasattr(ws, "_dets_to_features") or not hasattr(ws, "_write_features_auto"):
            raise RuntimeError("Mustatil export helpers are not available on this workspace instance.")

        feats = ws._dets_to_features(kept)
        final = ws._write_features_auto(out_path, feats, layer="selected_class", log_fn=getattr(ws, "log", print))
        try:
            cls_text = "all classes" if selected is None else f"class {selected}"
            ws.log(f"Export selected class ({cls_text}): kept {len(kept)}, removed/hidden {removed} -> {final}")
        except Exception:
            pass
    except Exception as exc:
        try:
            ws.show_error("Export selected class", str(exc))
        except Exception:
            _log("Detection selected-class export failed: " + str(exc))


def _export_selected_satellite_class(ws: Any, parent: Any = None) -> None:
    try:
        if hasattr(ws, "satellite_detections"):
            records = list(getattr(ws, "satellite_detections", []) or [])
        elif hasattr(ws, "sat_last_records"):
            records = list(getattr(ws, "sat_last_records", []) or [])
        elif callable(getattr(ws, "_satellite_visible_records", None)):
            records = list(ws._satellite_visible_records() or [])
        else:
            records = []

        kept = _apply_current_filters(records, ws)
        selected = _selected_class(ws)
        removed = len(records) - len(kept)
        if not kept:
            raise RuntimeError("No satellite detections match the selected class/filter settings.")

        base = _current_output_path(ws, "sat_output_gpkg", "satellite_selected_class.gpkg")
        out_path = _choose_save_path(parent or ws, _suffix_selected_class(base, selected), "Export selected class")
        if out_path is None:
            return

        if not hasattr(ws, "_satellite_features_to_file"):
            raise RuntimeError("Satellite export helper is not available on this workspace instance.")

        ws._satellite_features_to_file([dict(r) if isinstance(r, dict) else r for r in kept], out_path)
        try:
            cls_text = "all classes" if selected is None else f"class {selected}"
            ws.log(f"Satellite export selected class ({cls_text}): kept {len(kept)}, removed/hidden {removed} -> {out_path}")
        except Exception:
            pass
    except Exception as exc:
        try:
            ws.show_error("Export selected class", str(exc))
        except Exception:
            _log("Satellite selected-class export failed: " + str(exc))


def _redraw_after_filter_change(ws: Any) -> None:
    """Force both Detection and Satellite previews to repaint after filter changes."""
    try:
        if callable(getattr(ws, "redraw", None)):
            ws.redraw(fit=False)
    except TypeError:
        try:
            ws.redraw()
        except Exception:
            pass
    except Exception:
        pass
    try:
        sigs = getattr(ws, "signals", None)
        sig = getattr(sigs, "redraw", None) if sigs is not None else None
        if sig is not None:
            sig.emit()
    except Exception:
        pass
    try:
        if callable(getattr(ws, "satellite_redraw_detection_overlay", None)):
            ws.satellite_redraw_detection_overlay()
    except Exception:
        pass
    try:
        sigs = getattr(ws, "signals", None)
        sig = getattr(sigs, "sat_overlay_redraw_requested", None) if sigs is not None else None
        if sig is not None:
            sig.emit(0)
    except Exception:
        pass
    try:
        if callable(getattr(ws, "refresh_layers", None)):
            ws.refresh_layers()
    except Exception:
        pass


def _refresh_class_dropdown(ws: Any, combo: Any, tab_kind: str) -> None:
    try:
        current = combo.currentData()
    except Exception:
        current = None

    try:
        combo.blockSignals(True)
    except Exception:
        pass

    try:
        combo.clear()
        combo.addItem("All classes", None)
        known = {}

        if tab_kind == "satellite":
            records = list(getattr(ws, "satellite_detections", []) or getattr(ws, "sat_last_records", []) or [])
        else:
            records = list(getattr(ws, "dets", []) or [])

        for rec in records:
            cls = _det_class(rec)
            if cls is None:
                continue
            label = _det_label(rec).strip()
            if not label:
                if cls == 0:
                    label = "positive / mustatil"
                elif cls == 1:
                    label = "false_positive"
                else:
                    label = f"class {cls}"
            known[int(cls)] = label

        # Always show the two main Mustatil classes even before detections exist.
        if 0 not in known:
            known[0] = "positive / mustatil"
        if 1 not in known:
            known[1] = "false_positive"

        for cls in sorted(known):
            combo.addItem(f"Class {cls}: {known[cls]}", int(cls))

        # restore. Prefer the workspace-level selected class if it exists.
        try:
            state_value = getattr(ws, "mustatil_selected_class_id", current)
        except Exception:
            state_value = current
        index_to_set = 0
        for i in range(combo.count()):
            if combo.itemData(i) == state_value:
                index_to_set = i
                break
        combo.setCurrentIndex(index_to_set)
    except Exception as exc:
        _log("Class dropdown refresh failed: " + str(exc))
    finally:
        try:
            combo.blockSignals(False)
        except Exception:
            pass


def _add_filter_box(page: Any, tab_kind: str) -> bool:
    """Add in-page class filter / Geo NMS controls to a tab page."""
    try:
        if id(page) in _PATCHED_WIDGET_IDS:
            return False

        ws = _workspace_from_widget(page)
        if ws is None:
            return False

        _patch_workspace_methods(ws)

        from PySide6.QtWidgets import (
            QGroupBox, QVBoxLayout, QHBoxLayout, QCheckBox, QPushButton,
            QLabel, QComboBox, QDoubleSpinBox
        )

        layout = _find_left_controls_layout(page)
        if layout is None:
            return False

        box = QGroupBox("Class filter / Geo NMS")
        box.setObjectName("MustatilClassFilterGeoNmsBox")
        bl = QVBoxLayout(box)
        bl.setContentsMargins(8, 6, 8, 6)

        info = QLabel("Choose the class for preview/export and optionally remove duplicate YOLO boxes.")
        info.setWordWrap(True)
        try:
            info.setStyleSheet("color: #555;")
        except Exception:
            pass
        bl.addWidget(info)

        combo_row = QHBoxLayout()
        combo_label = QLabel("Selected class:")
        combo = QComboBox()
        combo.setObjectName("MustatilSelectedClassCombo")
        combo_row.addWidget(combo_label)
        combo_row.addWidget(combo, 1)
        bl.addLayout(combo_row)

        cb_hide_fp = QCheckBox("Hide class 1 / false_positive")
        cb_hide_fp.setToolTip("Useful shortcut: hides/removes class_id/class/cls = 1 or label/status false_positive.")
        cb_hide_fp.setChecked(bool(getattr(ws, "mustatil_hide_false_positives", False)))
        bl.addWidget(cb_hide_fp)

        cb_geo = QCheckBox("Apply Geo NMS to preview/export")
        cb_geo.setToolTip("Removes duplicate overlapping boxes class-wise before preview/export.")
        cb_geo.setChecked(bool(getattr(ws, "mustatil_geo_nms_enabled", False)))
        bl.addWidget(cb_geo)

        iou_row = QHBoxLayout()
        iou_label = QLabel("Geo NMS IoU:")
        iou_spin = QDoubleSpinBox()
        iou_spin.setRange(0.01, 0.99)
        iou_spin.setSingleStep(0.05)
        iou_spin.setDecimals(2)
        iou_spin.setValue(float(getattr(ws, "mustatil_geo_nms_iou", 0.35)))
        iou_spin.setToolTip("Lower value removes more duplicate boxes. Typical: 0.30 to 0.45.")
        iou_row.addWidget(iou_label)
        iou_row.addWidget(iou_spin, 1)
        bl.addLayout(iou_row)

        row = QHBoxLayout()
        btn_refresh = QPushButton("Refresh classes")
        btn_export = QPushButton("Export selected class")
        btn_export.setMinimumHeight(30)
        row.addWidget(btn_refresh)
        row.addWidget(btn_export, 1)
        bl.addLayout(row)

        # Keep all dropdowns synchronized through a workspace-level value.
        # This fixes the case where the Satellite tab's combo overwrote the
        # Detection tab's combo and the Detection preview did not change.
        try:
            if not hasattr(ws, "mustatil_class_filter_combos"):
                ws.mustatil_class_filter_combos = []
            if combo not in ws.mustatil_class_filter_combos:
                ws.mustatil_class_filter_combos.append(combo)
        except Exception:
            pass
        ws.mustatil_class_filter_combo = combo
        ws.mustatil_geo_nms_iou_spin = iou_spin

        _refresh_class_dropdown(ws, combo, tab_kind)

        def _sync_class_combos(source_combo):
            try:
                data = source_combo.currentData()
                ws.mustatil_selected_class_id = None if data is None or data == "" else int(data)
            except Exception:
                ws.mustatil_selected_class_id = None
            try:
                for other in list(getattr(ws, "mustatil_class_filter_combos", []) or []):
                    if other is source_combo:
                        continue
                    for i in range(other.count()):
                        if other.itemData(i) == getattr(ws, "mustatil_selected_class_id", None):
                            other.blockSignals(True)
                            other.setCurrentIndex(i)
                            other.blockSignals(False)
                            break
            except Exception:
                pass

        def on_class_changed(*_):
            _sync_class_combos(combo)
            try:
                if hasattr(ws, "log"):
                    sel = _selected_class(ws)
                    ws.log("Selected class filter: " + ("All classes" if sel is None else f"Class {sel}"))
            except Exception:
                pass
            _redraw_after_filter_change(ws)

        def on_hide_fp_toggled(state=False):
            try:
                ws.mustatil_hide_false_positives = bool(state)
                # Backward-compatible flag name used by older version.
                ws.mustatil_remove_false_positives = bool(state)
                if hasattr(ws, "log"):
                    ws.log("Hide class 1 / false_positive: " + ("ON" if state else "OFF"))
            except Exception:
                pass
            _redraw_after_filter_change(ws)

        def on_geo_toggled(state=False):
            try:
                ws.mustatil_geo_nms_enabled = bool(state)
                if hasattr(ws, "log"):
                    ws.log("Geo NMS preview/export: " + ("ON" if state else "OFF"))
            except Exception:
                pass
            _redraw_after_filter_change(ws)

        def on_iou_changed(*_):
            try:
                ws.mustatil_geo_nms_iou = float(iou_spin.value())
            except Exception:
                pass
            if _geo_nms_enabled(ws):
                _redraw_after_filter_change(ws)

        combo.currentIndexChanged.connect(on_class_changed)
        cb_hide_fp.toggled.connect(on_hide_fp_toggled)
        cb_geo.toggled.connect(on_geo_toggled)
        iou_spin.valueChanged.connect(on_iou_changed)
        btn_refresh.clicked.connect(lambda: _refresh_class_dropdown(ws, combo, tab_kind))

        if tab_kind == "satellite":
            btn_export.clicked.connect(lambda: _export_selected_satellite_class(ws, page))
        else:
            btn_export.clicked.connect(lambda: _export_selected_detection_class(ws, page))

        _insert_under_filter_sliders(layout, box)

        _PATCHED_WIDGET_IDS.add(id(page))
        _log(f"Added Class filter / Geo NMS controls to {tab_kind} tab.")
        return True
    except Exception as exc:
        _log("Could not add Class filter / Geo NMS box: " + str(exc))
        try:
            traceback.print_exc()
        except Exception:
            pass
        return False


def _install_qtabwidget_hook() -> None:
    global _PATCHED_QTABWIDGET, _ORIGINAL_QTAB_ADD
    if _PATCHED_QTABWIDGET:
        return
    try:
        from PySide6.QtWidgets import QTabWidget
        from PySide6.QtCore import QTimer
    except Exception as exc:
        _log("Qt not available; hook not installed: " + str(exc))
        return

    _ORIGINAL_QTAB_ADD = QTabWidget.addTab

    def addTab_patched(self, page, *args, **kwargs):
        result = _ORIGINAL_QTAB_ADD(self, page, *args, **kwargs)
        try:
            label = ""
            # addTab(QWidget, str) or addTab(QWidget, icon, str)
            if args:
                for a in reversed(args):
                    if isinstance(a, str):
                        label = a
                        break
            if not label:
                try:
                    label = self.tabText(int(result))
                except Exception:
                    label = ""
            norm = str(label or "").strip().lower()
            if norm == "detection":
                QTimer.singleShot(0, lambda p=page: _add_filter_box(p, "detection"))
                QTimer.singleShot(300, lambda p=page: _add_filter_box(p, "detection"))
                QTimer.singleShot(1200, lambda p=page: _add_filter_box(p, "detection"))
            elif norm == "satellite detection":
                QTimer.singleShot(0, lambda p=page: _add_filter_box(p, "satellite"))
                QTimer.singleShot(300, lambda p=page: _add_filter_box(p, "satellite"))
                QTimer.singleShot(1200, lambda p=page: _add_filter_box(p, "satellite"))
        except Exception as exc:
            _log("addTab hook warning: " + str(exc))
        return result

    QTabWidget.addTab = addTab_patched
    _PATCHED_QTABWIDGET = True
    _log("QTabWidget.addTab hook installed for Detection and Satellite Detection.")


def mustatil_plugin_init():
    _install_qtabwidget_hook()
    try:
        _install_console_tab_hook_patch()
    except Exception:
        pass


def register_plugin(app=None, main_window=None):
    _install_qtabwidget_hook()
    try:
        _install_console_tab_hook_patch()
    except Exception:
        pass
    return True


def init_plugin(app=None, main_window=None):
    return register_plugin(app, main_window)


def load_plugin(app=None, main_window=None):
    return register_plugin(app, main_window)


# Also install immediately in case the loader does not call mustatil_plugin_init().
try:
    _install_qtabwidget_hook()
except Exception:
    pass


# ---------------------------------------------------------------------------
# FormLearner / Mustatil real console addon
# ---------------------------------------------------------------------------

_PATCHED_CONSOLE_WIDGET_IDS = set()


def _find_console_like_widgets(root: Any) -> list:
    """Find QTextEdit/QPlainTextEdit widgets that look like log/FormLearner consoles."""
    out = []
    try:
        from PySide6.QtWidgets import QTextEdit, QPlainTextEdit
    except Exception:
        return out

    try:
        widgets = list(root.findChildren(QTextEdit)) + list(root.findChildren(QPlainTextEdit))
    except Exception:
        widgets = []

    for w in widgets:
        try:
            name = str(w.objectName() or "").lower()
        except Exception:
            name = ""
        txt = ""
        try:
            txt = str(w.toPlainText() or "")[-3000:].lower()
        except Exception:
            pass

        hay = name + " " + txt
        if any(k in hay for k in ["formlearner", "console", "log", "trainer", "pipeline", "python"]):
            out.append(w)

    return out


def _append_console_text(widget: Any, text: str) -> None:
    try:
        widget.appendPlainText(str(text))
        return
    except Exception:
        pass
    try:
        widget.append(str(text))
        return
    except Exception:
        pass


def _get_console_text(widget: Any) -> str:
    try:
        return str(widget.toPlainText() or "")
    except Exception:
        return ""


class _MustatilRealConsoleMixin:
    pass


def _make_real_console_panel(ws: Any, original_console: Any = None):
    """
    Create a compact interactive Python console.
    Variables:
        ws      = Mustatil workspace
        app     = QApplication instance
        page    = current patched tab/page if supplied later
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QLineEdit, QTextEdit, QCheckBox
    )
    from contextlib import redirect_stdout, redirect_stderr
    import io
    import traceback as _tb

    panel = QWidget()
    panel.setObjectName("MustatilRealConsolePanel")
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(6, 6, 6, 6)

    top = QHBoxLayout()
    title = QLabel("<b>Real Python Console</b> — variables: <code>ws</code>, <code>app</code>")
    title.setTextFormat(Qt.RichText)
    top.addWidget(title, 1)

    clear_btn = QPushButton("Clear")
    help_btn = QPushButton("Help")
    top.addWidget(help_btn)
    top.addWidget(clear_btn)
    layout.addLayout(top)

    output = QTextEdit()
    output.setReadOnly(True)
    output.setMinimumHeight(140)
    output.setObjectName("MustatilRealConsoleOutput")
    layout.addWidget(output, 1)

    row = QHBoxLayout()
    prompt = QLabel(">>>")
    entry = QLineEdit()
    entry.setObjectName("MustatilRealConsoleInput")
    entry.setPlaceholderText("Python command, e.g. len(ws.dets), ws.redraw(), dir(ws)")
    run_btn = QPushButton("Run")
    row.addWidget(prompt)
    row.addWidget(entry, 1)
    row.addWidget(run_btn)
    layout.addLayout(row)

    mirror_check = QCheckBox("Mirror FormLearner/log text above")
    mirror_check.setChecked(True)
    layout.addWidget(mirror_check)

    ns = {
        "ws": ws,
        "app": None,
        "Path": Path,
        "os": os,
        "sys": sys,
        "math": math,
    }
    try:
        from PySide6.QtWidgets import QApplication
        ns["app"] = QApplication.instance()
    except Exception:
        pass

    history = []
    hist_pos = {"i": 0}

    def write_line(text=""):
        try:
            output.append(str(text))
        except Exception:
            pass

    def show_help():
        write_line("")
        write_line("Examples:")
        write_line("  len(ws.dets)")
        write_line("  ws.visible()")
        write_line("  dir(ws)")
        write_line("  ws.redraw()")
        write_line("  [d for d in ws.dets if getattr(d, 'cls', None) == 0][:5]")
        write_line("  ws.log('test from console')")
        write_line("")
        write_line("Use Shift+Enter only if your Qt build supports multiline paste; otherwise paste exec(\"\"\"...\"\"\").")

    def run_command():
        cmd = entry.text()
        if not cmd.strip():
            return
        entry.clear()
        history.append(cmd)
        hist_pos["i"] = len(history)

        write_line(">>> " + cmd)

        stdout = io.StringIO()
        try:
            with redirect_stdout(stdout), redirect_stderr(stdout):
                try:
                    # Try eval first for expression-like commands.
                    result = eval(cmd, ns, ns)
                    if result is not None:
                        write_line(repr(result))
                except SyntaxError:
                    exec(cmd, ns, ns)

            captured = stdout.getvalue()
            if captured:
                write_line(captured.rstrip())
        except Exception:
            captured = stdout.getvalue()
            if captured:
                write_line(captured.rstrip())
            write_line(_tb.format_exc().rstrip())

    def clear_output():
        output.clear()

    def key_press(event):
        try:
            key = event.key()
            if key in (Qt.Key_Return, Qt.Key_Enter):
                run_command()
                return
            if key == Qt.Key_Up and history:
                hist_pos["i"] = max(0, hist_pos["i"] - 1)
                entry.setText(history[hist_pos["i"]])
                return
            if key == Qt.Key_Down and history:
                hist_pos["i"] = min(len(history), hist_pos["i"] + 1)
                if hist_pos["i"] >= len(history):
                    entry.clear()
                else:
                    entry.setText(history[hist_pos["i"]])
                return
        except Exception:
            pass
        return QLineEdit.keyPressEvent(entry, event)

    entry.keyPressEvent = key_press
    run_btn.clicked.connect(run_command)
    clear_btn.clicked.connect(clear_output)
    help_btn.clicked.connect(show_help)

    write_line("Real console ready. Workspace is available as ws.")
    show_help()

    # Optional: copy the existing console text once.
    try:
        if original_console is not None and mirror_check.isChecked():
            txt = _get_console_text(original_console).strip()
            if txt:
                write_line("")
                write_line("--- existing console/log text ---")
                write_line(txt[-8000:])
    except Exception:
        pass

    return panel, output, entry, ns


def _patch_ws_log_to_console(ws: Any, console_output: Any) -> None:
    """Mirror future ws.log messages into the real console output."""
    try:
        old_log = getattr(ws, "log", None)
        if not callable(old_log) or getattr(ws, "_real_console_log_patched", False):
            return

        def log_with_console_mirror(*args, **kwargs):
            try:
                msg = " ".join(str(a) for a in args)
                _append_console_text(console_output, msg)
            except Exception:
                pass
            return old_log(*args, **kwargs)

        ws.log = log_with_console_mirror
        ws._real_console_log_patched = True
        _log("Patched ws.log mirror into real console.")
    except Exception as exc:
        _log("Could not patch ws.log mirror: " + str(exc))


def _upgrade_console_area(page: Any, tab_kind: str = "") -> bool:
    """
    Replace/enhance a console-like QTextEdit with a real Python console panel.
    Works best on Console/FormLearner-like tabs.
    """
    try:
        if id(page) in _PATCHED_CONSOLE_WIDGET_IDS:
            return False

        ws = _workspace_from_widget(page)
        if ws is None:
            return False

        from PySide6.QtWidgets import QVBoxLayout, QSplitter, QTextEdit, QPlainTextEdit
        from PySide6.QtCore import Qt

        # Avoid adding console panel into every detection page unless explicitly console-like.
        page_text = _widget_text_recursive(page)
        page_name = ""
        try:
            page_name = str(page.objectName() or "").lower()
        except Exception:
            pass

        looks_console_tab = any(k in (tab_kind.lower() + " " + page_name + " " + page_text) for k in [
            "console", "formlearner", "log"
        ])

        console_widgets = _find_console_like_widgets(page)
        if not console_widgets and not looks_console_tab:
            return False

        original_console = console_widgets[0] if console_widgets else None
        panel, output, entry, ns = _make_real_console_panel(ws, original_console=original_console)
        ns["page"] = page

        layout = page.layout()
        if layout is None:
            layout = QVBoxLayout(page)
            page.setLayout(layout)

        # If there is an existing console QTextEdit, make a vertical splitter:
        # old log above, real console below.
        if original_console is not None:
            try:
                parent = original_console.parentWidget()
                parent_layout = parent.layout() if parent is not None else layout
                idx = -1
                for i in range(parent_layout.count()):
                    item = parent_layout.itemAt(i)
                    if item is not None and item.widget() is original_console:
                        idx = i
                        break

                splitter = QSplitter(Qt.Vertical)
                splitter.setObjectName("MustatilRealConsoleSplitter")
                original_console.setParent(None)
                splitter.addWidget(original_console)
                splitter.addWidget(panel)
                splitter.setStretchFactor(0, 1)
                splitter.setStretchFactor(1, 1)

                if idx >= 0:
                    parent_layout.insertWidget(idx, splitter)
                else:
                    layout.addWidget(splitter)
            except Exception:
                layout.addWidget(panel)
        else:
            layout.addWidget(panel)

        _patch_ws_log_to_console(ws, output)

        _PATCHED_CONSOLE_WIDGET_IDS.add(id(page))
        _log(f"Upgraded console area in {tab_kind or 'unknown'} tab.")
        return True
    except Exception as exc:
        _log("Could not upgrade console area: " + str(exc))
        try:
            traceback.print_exc()
        except Exception:
            pass
        return False


def _install_console_tab_hook_patch() -> None:
    """
    Extend the existing QTabWidget.addTab hook so console/formlearner tabs are upgraded too.
    If the class-filter hook is already installed, it wraps the current addTab safely.
    """
    global _ORIGINAL_QTAB_ADD

    try:
        from PySide6.QtWidgets import QTabWidget
        from PySide6.QtCore import QTimer
    except Exception as exc:
        _log("Qt not available for real-console hook: " + str(exc))
        return

    current_add = QTabWidget.addTab
    if getattr(current_add, "_mustatil_real_console_wrapped", False):
        return

    def addTab_console_wrapped(self, page, *args, **kwargs):
        result = current_add(self, page, *args, **kwargs)
        try:
            label = ""
            if args:
                for a in reversed(args):
                    if isinstance(a, str):
                        label = a
                        break
            if not label:
                try:
                    label = self.tabText(int(result))
                except Exception:
                    label = ""
            norm = str(label or "").strip().lower()

            if any(k in norm for k in ["console", "formlearner", "form learner", "log"]):
                QTimer.singleShot(0, lambda p=page, n=norm: _upgrade_console_area(p, n))
                QTimer.singleShot(500, lambda p=page, n=norm: _upgrade_console_area(p, n))
                QTimer.singleShot(1500, lambda p=page, n=norm: _upgrade_console_area(p, n))
        except Exception as exc:
            _log("real-console addTab hook warning: " + str(exc))
        return result

    addTab_console_wrapped._mustatil_real_console_wrapped = True
    QTabWidget.addTab = addTab_console_wrapped
    _log("QTabWidget.addTab hook installed for real console / FormLearner console.")


# Old Console-tab hook removed in v3.


# ---------------------------------------------------------------------------
# v3: Detection-tab FormLearner console only + displayed detection counter
# ---------------------------------------------------------------------------
# This overrides the previous real-console behaviour:
# - no console is added to the Console tab
# - the mini console is inserted above the FormLearner/log area in the Detection tab
# - a counter shows how many detections are currently displayed after class/Geo-NMS filters

_DET_FORM_CONSOLE_PATCHED_PAGES = set()
_DETECTION_COUNTER_LABELS = []
_DET_FORM_CONSOLE_INSTALLED = set()


def _install_console_tab_hook_patch() -> None:
    """Disabled in v3: do not modify the separate Console tab."""
    try:
        _log("Console-tab real console disabled; console will be installed only inside Detection tab.")
    except Exception:
        pass
    return


def _count_current_displayed_detections(ws: Any) -> int:
    try:
        if hasattr(ws, "dets"):
            raw = list(getattr(ws, "dets", []) or [])
        elif callable(getattr(ws, "visible", None)):
            raw = list(ws.visible() or [])
        else:
            raw = []
        return len(_apply_current_filters(raw, ws))
    except Exception:
        try:
            return len(list(getattr(ws, "dets", []) or []))
        except Exception:
            return 0


def _count_all_detections(ws: Any) -> int:
    try:
        return len(list(getattr(ws, "dets", []) or []))
    except Exception:
        return 0


def _update_detection_counter(ws: Any) -> None:
    try:
        shown = _count_current_displayed_detections(ws)
        total = _count_all_detections(ws)
        selected = None
        try:
            selected = getattr(ws, "mustatil_selected_class", None)
        except Exception:
            pass
        if selected is None:
            cls_text = "All classes"
        else:
            cls_text = f"Class {selected}"
        text = f"Displayed detections: {shown} / {total} ({cls_text})"
        if _geo_nms_enabled(ws):
            text += f" | Geo NMS IoU {_geo_nms_iou(ws):.2f}"
        for lbl in list(_DETECTION_COUNTER_LABELS):
            try:
                lbl.setText(text)
            except Exception:
                pass
    except Exception:
        pass


def _patch_counter_redraw_hooks(ws: Any) -> None:
    """Update counter after redraw/log-relevant actions."""
    try:
        if getattr(ws, "_detection_counter_hooks_patched", False):
            return
        ws._detection_counter_hooks_patched = True

        old_redraw = getattr(ws, "redraw", None)
        if callable(old_redraw):
            def redraw_with_counter(*args, **kwargs):
                result = old_redraw(*args, **kwargs)
                try:
                    _update_detection_counter(ws)
                except Exception:
                    pass
                return result
            ws.redraw = redraw_with_counter

        old_log = getattr(ws, "log", None)
        if callable(old_log):
            def log_with_counter(*args, **kwargs):
                result = old_log(*args, **kwargs)
                try:
                    # Detection counts often change after detection/export log messages.
                    _update_detection_counter(ws)
                except Exception:
                    pass
                return result
            ws.log = log_with_counter
    except Exception as exc:
        try:
            _log("Counter hook patch failed: " + str(exc))
        except Exception:
            pass


def _find_formlearner_log_widget(page: Any):
    """Find a log QTextEdit/QPlainTextEdit in the Detection page, preferably FormLearner-related."""
    try:
        from PySide6.QtWidgets import QTextEdit, QPlainTextEdit
    except Exception:
        return None

    widgets = []
    try:
        widgets += list(page.findChildren(QTextEdit))
    except Exception:
        pass
    try:
        widgets += list(page.findChildren(QPlainTextEdit))
    except Exception:
        pass

    best = None
    best_score = -999
    for w in widgets:
        score = 0
        try:
            name = str(w.objectName() or "").lower()
        except Exception:
            name = ""
        try:
            text = str(w.toPlainText() or "").lower()
        except Exception:
            text = ""
        hay = name + " " + text
        for key in ("formlearner", "form learner", "form_score", "log", "console", "detection"):
            if key in hay:
                score += 5
        # Prefer larger text areas.
        try:
            score += min(10, int(w.height() / 60))
        except Exception:
            pass
        if score > best_score:
            best = w
            best_score = score

    return best if best_score >= 0 else None


def _make_detection_form_console(ws: Any, page: Any = None):
    """Small developer console for Detection tab, inserted above FormLearner/log."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QLineEdit, QTextEdit, QCheckBox
    )
    from contextlib import redirect_stdout, redirect_stderr
    import io
    import traceback as _tb

    panel = QWidget()
    panel.setObjectName("MustatilDetectionFormLearnerConsolePanel")
    panel.setMaximumHeight(260)

    layout = QVBoxLayout(panel)
    layout.setContentsMargins(6, 6, 6, 6)
    layout.setSpacing(4)

    header = QHBoxLayout()
    title = QLabel("<b>FormLearner / Detection Console</b> — <code>ws</code> = workspace")
    title.setTextFormat(Qt.RichText)
    header.addWidget(title, 1)

    clear_btn = QPushButton("Clear")
    help_btn = QPushButton("Help")
    header.addWidget(help_btn)
    header.addWidget(clear_btn)
    layout.addLayout(header)

    out = QTextEdit()
    out.setReadOnly(True)
    out.setObjectName("MustatilDetectionFormConsoleOutput")
    out.setMinimumHeight(80)
    layout.addWidget(out, 1)

    row = QHBoxLayout()
    row.addWidget(QLabel(">>>"))
    entry = QLineEdit()
    entry.setObjectName("MustatilDetectionFormConsoleInput")
    entry.setPlaceholderText("e.g. len(ws.dets), ws.visible(), ws.redraw(), dir(ws)")
    run_btn = QPushButton("Run")
    row.addWidget(entry, 1)
    row.addWidget(run_btn)
    layout.addLayout(row)

    ns = {
        "ws": ws,
        "page": page,
        "Path": Path,
        "os": os,
        "sys": sys,
        "math": math,
    }
    try:
        from PySide6.QtWidgets import QApplication
        ns["app"] = QApplication.instance()
    except Exception:
        ns["app"] = None

    history = []
    hist = {"i": 0}

    def write(text=""):
        try:
            out.append(str(text))
        except Exception:
            pass

    def help_text():
        write("Examples:")
        write("  len(ws.dets)")
        write("  ws.visible()")
        write("  ws.redraw()")
        write("  [d for d in ws.dets if getattr(d, 'cls', None) == 0][:5]")
        write("  ws.log('hello from Detection console')")
        write("")

    def run_cmd():
        cmd = entry.text()
        if not cmd.strip():
            return
        entry.clear()
        history.append(cmd)
        hist["i"] = len(history)
        write(">>> " + cmd)
        buf = io.StringIO()
        try:
            with redirect_stdout(buf), redirect_stderr(buf):
                try:
                    result = eval(cmd, ns, ns)
                    if result is not None:
                        write(repr(result))
                except SyntaxError:
                    exec(cmd, ns, ns)
            captured = buf.getvalue()
            if captured:
                write(captured.rstrip())
        except Exception:
            captured = buf.getvalue()
            if captured:
                write(captured.rstrip())
            write(_tb.format_exc().rstrip())
        try:
            _update_detection_counter(ws)
        except Exception:
            pass

    def key_press(event):
        try:
            key = event.key()
            if key in (Qt.Key_Return, Qt.Key_Enter):
                run_cmd()
                return
            if key == Qt.Key_Up and history:
                hist["i"] = max(0, hist["i"] - 1)
                entry.setText(history[hist["i"]])
                return
            if key == Qt.Key_Down and history:
                hist["i"] = min(len(history), hist["i"] + 1)
                if hist["i"] >= len(history):
                    entry.clear()
                else:
                    entry.setText(history[hist["i"]])
                return
        except Exception:
            pass
        return QLineEdit.keyPressEvent(entry, event)

    entry.keyPressEvent = key_press
    run_btn.clicked.connect(run_cmd)
    clear_btn.clicked.connect(lambda: out.clear())
    help_btn.clicked.connect(help_text)

    write("Detection/FormLearner console ready.")
    help_text()
    return panel, out


def _insert_detection_form_console(page: Any, ws: Any) -> bool:
    if page is None or ws is None:
        return False
    if id(page) in _DET_FORM_CONSOLE_INSTALLED:
        return False

    try:
        from PySide6.QtWidgets import QVBoxLayout
    except Exception:
        return False

    log_widget = _find_formlearner_log_widget(page)
    panel, out = _make_detection_form_console(ws, page)

    inserted = False
    if log_widget is not None:
        try:
            parent = log_widget.parentWidget()
            parent_layout = parent.layout() if parent is not None else None
            if parent_layout is not None:
                idx = -1
                for i in range(parent_layout.count()):
                    item = parent_layout.itemAt(i)
                    if item is not None and item.widget() is log_widget:
                        idx = i
                        break
                if idx >= 0:
                    parent_layout.insertWidget(idx, panel)
                    inserted = True
        except Exception:
            inserted = False

    if not inserted:
        # Fallback: put it into the left control layout, below the filter box if possible.
        try:
            layout = _find_left_controls_layout(page) or page.layout()
            if layout is not None:
                layout.addWidget(panel)
                inserted = True
        except Exception:
            inserted = False

    if inserted:
        try:
            old_log = getattr(ws, "log", None)
            if callable(old_log) and not getattr(ws, "_det_form_console_log_mirror_patched", False):
                def log_with_detection_console(*args, **kwargs):
                    try:
                        msg = " ".join(str(a) for a in args)
                        out.append(msg)
                    except Exception:
                        pass
                    return old_log(*args, **kwargs)
                ws.log = log_with_detection_console
                ws._det_form_console_log_mirror_patched = True
        except Exception:
            pass

        _DET_FORM_CONSOLE_INSTALLED.add(id(page))
        try:
            _log("Inserted real console above FormLearner/log in Detection tab.")
        except Exception:
            pass
        return True

    return False


def _add_counter_to_filter_box(page: Any, ws: Any) -> None:
    """Add visible counter below the class/Geo-NMS filter box."""
    try:
        from PySide6.QtWidgets import QLabel
    except Exception:
        return

    try:
        existing = page.findChild(QLabel, "MustatilDisplayedDetectionCounterLabel")
        if existing is not None:
            return
    except Exception:
        pass

    try:
        box = page.findChild(object, "MustatilClassFilterGeoNmsBox")
        if box is None:
            return
        layout = box.layout()
        if layout is None:
            return

        lbl = QLabel("Displayed detections: 0 / 0")
        lbl.setObjectName("MustatilDisplayedDetectionCounterLabel")
        lbl.setWordWrap(True)
        try:
            lbl.setStyleSheet("font-weight: bold; color: #1f4e79; padding: 3px;")
        except Exception:
            pass
        layout.insertWidget(1, lbl)
        _DETECTION_COUNTER_LABELS.append(lbl)
        _patch_counter_redraw_hooks(ws)
        _update_detection_counter(ws)
        _log("Added displayed detection counter.")
    except Exception as exc:
        try:
            _log("Counter insert failed: " + str(exc))
        except Exception:
            pass


# Wrap the current _add_filter_box so Detection pages also receive the counter and FormLearner console.
try:
    _OLD_ADD_FILTER_BOX_V3 = _add_filter_box

    def _add_filter_box(page: Any, tab_kind: str) -> bool:
        result = _OLD_ADD_FILTER_BOX_V3(page, tab_kind)
        try:
            ws = _workspace_from_widget(page)
            if ws is not None and str(tab_kind).lower() == "detection":
                _add_counter_to_filter_box(page, ws)
                _insert_detection_form_console(page, ws)
                _update_detection_counter(ws)
        except Exception as exc:
            try:
                _log("v3 Detection extras failed: " + str(exc))
            except Exception:
                pass
        return result

except Exception:
    pass


# If Mustatil calls plugin init after this file is imported, ensure only the main tab hook is active.
def mustatil_plugin_init():
    _install_qtabwidget_hook()
    try:
        _log("v3 active: Console tab is not patched; Detection FormLearner console + counter enabled.")
    except Exception:
        pass


def register_plugin(app=None, main_window=None):
    _install_qtabwidget_hook()
    try:
        _log("v3 active: Console tab is not patched; Detection FormLearner console + counter enabled.")
    except Exception:
        pass
    return True


def init_plugin(app=None, main_window=None):
    return register_plugin(app, main_window)


def load_plugin(app=None, main_window=None):
    return register_plugin(app, main_window)

