#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mustatil plugin: Detection + Satellite Detection GeoPackage autosave
===================================================================

Install:
    Put this file into the folder "mustatil_plugins" next to mustatil_qt_workspace.py
    and restart Mustatil.

Behavior:
    - Autosaves every 10 minutes.
    - Also autosaves once immediately after Detection / Satellite Detection tasks.
    - Saves ALL Detection objects from self.dets, not only visible/filter-passing boxes.
    - Saves ALL Satellite records from self.satellite_detections / self.sat_last_records.
    - Adds class_name fields and a mustatil_classes layer with all class names.
    - Adds a mustatil_variables layer with current Mustatil Var/project settings.
    - Detection output is written next to the selected TIFF/image:
          <image_stem>_mustatil_detection_autosave_all.gpkg
    - Satellite output is written next to sat_output_tif if set, otherwise project/exports:
          <sat_tif_stem>_mustatil_satellite_autosave_all.gpkg

Notes:
    - Requires geopandas + shapely, same as Mustatil's normal GPKG export.
    - Existing autosave GPKG is replaced atomically when possible.
    - If Windows locks the existing GPKG, a timestamp fallback file is written.
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import math
import shutil
import traceback
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PLUGIN_NAME = "Mustatil Detection/Satellite Autosave GPKG"
PLUGIN_VERSION = "1.0.0"

AUTOSAVE_INTERVAL_MS = int(os.environ.get("MUSTATIL_GPKG_AUTOSAVE_MS", "600000"))  # 10 minutes
FIRST_AUTOSAVE_DELAY_MS = int(os.environ.get("MUSTATIL_GPKG_FIRST_AUTOSAVE_MS", "60000"))
DETECTION_SUFFIX = os.environ.get("MUSTATIL_DETECTION_AUTOSAVE_SUFFIX", "_mustatil_detection_autosave_all.gpkg")
SATELLITE_SUFFIX = os.environ.get("MUSTATIL_SATELLITE_AUTOSAVE_SUFFIX", "_mustatil_satellite_autosave_all.gpkg")

_PATCHED_QTAB = False
_ORIG_ADD_TAB = None
_ORIG_INSERT_TAB = None
_INSTALLED_WORKSPACES: set[int] = set()
_ACTIVE_WRITES: Dict[int, threading.Lock] = {}


def _log(msg: str) -> None:
    line = f"[{PLUGIN_NAME} v{PLUGIN_VERSION}] {msg}"
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        base = Path(__file__).resolve().parent
        with (base / "mustatil_detection_satellite_autosave_gpkg.log").open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _qt():
    try:
        from PySide6.QtWidgets import QTabWidget, QApplication
        from PySide6.QtCore import QTimer
        return QTabWidget, QApplication, QTimer
    except Exception as exc:
        _log("PySide6 import failed: " + repr(exc))
        return None, None, None


def _now_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _file_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _safe_var_get(v: Any, default: Any = "") -> Any:
    try:
        if hasattr(v, "get") and callable(v.get):
            val = v.get()
            return default if val is None else val
    except Exception:
        pass
    return default if v is None else v


def _safe_var_set(v: Any, value: Any) -> None:
    try:
        if hasattr(v, "set") and callable(v.set):
            v.set(value)
    except Exception:
        pass


def _workspace_from_widget(widget: Any) -> Optional[Any]:
    cur = widget
    for _ in range(120):
        if cur is None:
            return None
        try:
            if hasattr(cur, "tabs") and hasattr(cur, "dets") and hasattr(cur, "satellite_detections"):
                return cur
        except Exception:
            pass
        try:
            cur = cur.parentWidget()
        except Exception:
            try:
                cur = cur.parent()
            except Exception:
                return None
    return None


def _tab_title_from_args(args: Tuple[Any, ...]) -> str:
    try:
        return str(args[-1] or "")
    except Exception:
        return ""


def _is_target_tab_title(title: str) -> bool:
    t = str(title or "").strip().lower()
    return t in {"detection", "satellite detection"} or ("detection" in t and ("satellite" in t or t == "detection"))


def _status(ws: Any, msg: str) -> None:
    _log(msg)
    try:
        if hasattr(ws, "signals") and hasattr(ws.signals, "log"):
            ws.signals.log.emit("General", msg)
            return
    except Exception:
        pass
    try:
        if hasattr(ws, "log"):
            ws.log(msg)
    except Exception:
        pass
    try:
        sb = ws.statusBar()
        if sb is not None:
            sb.showMessage(msg, 9000)
    except Exception:
        pass


def _as_path(raw: Any) -> Optional[Path]:
    try:
        s = str(raw or "").strip().strip('"').strip("'")
    except Exception:
        return None
    if not s:
        return None
    try:
        return Path(os.path.expandvars(os.path.expanduser(s))).resolve()
    except Exception:
        try:
            return Path(os.path.expandvars(os.path.expanduser(s)))
        except Exception:
            return None


def _selected_detection_image_path(ws: Any) -> Optional[Path]:
    p = _as_path(_safe_var_get(getattr(ws, "image", None), ""))
    if p is not None and p.suffix.lower() in {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return p
    return None


def _detection_autosave_path(ws: Any) -> Optional[Path]:
    img = _selected_detection_image_path(ws)
    if img is not None:
        return img.with_name(f"{img.stem}{DETECTION_SUFFIX}")
    # Fallback: project exports.
    project = _as_path(_safe_var_get(getattr(ws, "project", None), ""))
    if project is not None:
        out = project / "exports" / f"{project.name or 'mustatil'}{DETECTION_SUFFIX}"
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
    return None


def _satellite_base_tif_path(ws: Any) -> Optional[Path]:
    # User-selected / generated BigTIFF path is preferred.
    for attr in ("sat_output_tif",):
        p = _as_path(_safe_var_get(getattr(ws, attr, None), ""))
        if p is not None and p.suffix.lower() in {".tif", ".tiff"}:
            return p

    # If a satellite output GPKG was already selected, use its folder/stem.
    p = _as_path(_safe_var_get(getattr(ws, "sat_output_gpkg", None), ""))
    if p is not None and p.suffix.lower() in {".gpkg", ".geojson", ".json"}:
        return p.with_suffix(".tif")

    # Fallback to last satellite output.
    p = _as_path(getattr(ws, "satellite_output_last", ""))
    if p is not None and p.suffix.lower() in {".gpkg", ".geojson", ".json"}:
        return p.with_suffix(".tif")

    return None


def _satellite_autosave_path(ws: Any) -> Optional[Path]:
    tif = _satellite_base_tif_path(ws)
    if tif is not None:
        out = tif.with_name(f"{tif.stem}{SATELLITE_SUFFIX}")
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    project = _as_path(_safe_var_get(getattr(ws, "project", None), ""))
    if project is not None:
        try:
            z = int(_safe_var_get(getattr(ws, "sat_zoom", None), 18) or 18)
        except Exception:
            z = 18
        out = project / "exports" / f"satellite_z{z}{SATELLITE_SUFFIX}"
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
    return None


def _classes(ws: Any) -> List[str]:
    # Project state first.
    try:
        ps = getattr(ws, "project_state", None)
        if ps is not None and getattr(ps, "classes", None):
            return [str(x) for x in list(ps.classes)]
    except Exception:
        pass
    try:
        c = getattr(ws, "CLASSES", None)
        if c:
            return [str(x) for x in list(c)]
    except Exception:
        pass
    try:
        g = globals().get("MUSTATIL_GLOBALS", {}) or {}
        c = g.get("CLASSES", None)
        if c:
            return [str(x) for x in list(c)]
    except Exception:
        pass
    return ["mustatil", "false_positive"]


def _class_name(ws: Any, class_id: Any) -> str:
    try:
        i = int(float(class_id))
        names = _classes(ws)
        if 0 <= i < len(names):
            return names[i]
    except Exception:
        pass
    return ""


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    try:
        # numpy scalar support
        if hasattr(value, "item"):
            return _jsonable(value.item())
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set, dict)):
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return str(value)
    try:
        return str(value)
    except Exception:
        return None


def _clean_field_name(name: str, used: set[str]) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", str(name or "field")).strip("_")
    if not s:
        s = "field"
    if s[0].isdigit():
        s = "f_" + s
    # GPKG can handle longer names, but keeping them moderate avoids older driver issues.
    s = s[:58]
    base = s
    i = 2
    while s.lower() in used:
        suffix = f"_{i}"
        s = (base[:58 - len(suffix)] + suffix)
        i += 1
    used.add(s.lower())
    return s


def _clean_records_for_gdf(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # First collect stable field-name mapping.
    keys: List[str] = []
    seen = set()
    for r in records:
        for k in r.keys():
            if k == "geometry":
                continue
            if k not in seen:
                seen.add(k)
                keys.append(k)

    used_names = set()
    mapping = {k: _clean_field_name(k, used_names) for k in keys}

    out = []
    for r in records:
        row = {}
        for k in keys:
            row[mapping[k]] = _jsonable(r.get(k))
        row["geometry"] = r.get("geometry")
        out.append(row)
    return out


def _det_to_dict(det: Any) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    try:
        if is_dataclass(det):
            data.update(asdict(det))
    except Exception:
        pass
    try:
        dct = getattr(det, "__dict__", None)
        if isinstance(dct, dict):
            for k, v in dct.items():
                if not k.startswith("_"):
                    data[k] = v
    except Exception:
        pass

    # Important common fields, even if they are properties rather than __dict__ fields.
    for k in (
        "x1", "y1", "x2", "y2", "cls", "class_id", "conf", "confidence",
        "score", "consensus", "form_score", "slot", "model", "name"
    ):
        try:
            if hasattr(det, k):
                data[k] = getattr(det, k)
        except Exception:
            pass

    try:
        if hasattr(det, "bbox") and callable(det.bbox):
            bb = det.bbox()
            if bb and len(bb) >= 4:
                data["bbox_px_x1"] = float(bb[0])
                data["bbox_px_y1"] = float(bb[1])
                data["bbox_px_x2"] = float(bb[2])
                data["bbox_px_y2"] = float(bb[3])
    except Exception:
        pass

    # Normalize bbox fields.
    try:
        data.setdefault("bbox_px_x1", float(data.get("x1")))
        data.setdefault("bbox_px_y1", float(data.get("y1")))
        data.setdefault("bbox_px_x2", float(data.get("x2")))
        data.setdefault("bbox_px_y2", float(data.get("y2")))
    except Exception:
        pass

    return data


def _pixel_box_from_det_dict(d: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    keys = ("bbox_px_x1", "bbox_px_y1", "bbox_px_x2", "bbox_px_y2")
    try:
        if all(k in d and d.get(k) is not None for k in keys):
            x1, y1, x2, y2 = [float(d[k]) for k in keys]
            return x1, y1, x2, y2
    except Exception:
        pass
    try:
        x1, y1, x2, y2 = float(d["x1"]), float(d["y1"]), float(d["x2"]), float(d["y2"])
        return x1, y1, x2, y2
    except Exception:
        return None


def _detection_crs(ws: Any) -> Any:
    try:
        if hasattr(ws, "_feature_crs_wkt"):
            wkt = ws._feature_crs_wkt()
            if wkt:
                return wkt
    except Exception:
        pass
    try:
        if hasattr(ws, "_feature_crs_name"):
            name = ws._feature_crs_name()
            if name and str(name).upper().startswith("EPSG:"):
                return str(name)
    except Exception:
        pass
    return None


def _map_polygon_for_pixel_box(ws: Any, box: Tuple[float, float, float, float]):
    from shapely.geometry import Polygon
    x1, y1, x2, y2 = box
    try:
        if hasattr(ws, "_px_to_map"):
            pts = [
                ws._px_to_map(x1, y1),
                ws._px_to_map(x2, y1),
                ws._px_to_map(x2, y2),
                ws._px_to_map(x1, y2),
            ]
            if all(p is not None for p in pts):
                pts2 = [(float(p[0]), float(p[1])) for p in pts]
                pts2.append(pts2[0])
                return Polygon(pts2)
    except Exception:
        pass
    # Pixel fallback.
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)])


def _snapshot_detection_records(ws: Any) -> Tuple[List[Dict[str, Any]], Any]:
    records: List[Dict[str, Any]] = []
    dets = list(getattr(ws, "dets", []) or [])
    source_image = str(_safe_var_get(getattr(ws, "image", None), "") or "")
    classes = _classes(ws)

    for i, det in enumerate(dets):
        row = _det_to_dict(det)
        row["autosave_tab"] = "Detection"
        row["autosave_index"] = i
        row["autosave_time"] = _now_stamp()
        row["source_image"] = source_image
        # Preserve and normalize class information.
        cls = row.get("cls", row.get("class_id", 0))
        try:
            cls_int = int(float(cls))
        except Exception:
            cls_int = 0
        row["class_id"] = cls_int
        row["class_name"] = _class_name(ws, cls_int)
        # Normalize confidence/model slot names while keeping original fields.
        if "confidence" not in row and "conf" in row:
            row["confidence"] = row.get("conf")
        if "model_slot" not in row and "slot" in row:
            try:
                row["model_slot"] = int(row.get("slot")) + 1
            except Exception:
                pass

        box = _pixel_box_from_det_dict(row)
        if box is None:
            continue
        row["geometry"] = _map_polygon_for_pixel_box(ws, box)
        records.append(row)

    return records, _detection_crs(ws)


def _satellite_polygon(record: Dict[str, Any]):
    from shapely.geometry import Polygon
    poly = record.get("polygon_lonlat")
    if isinstance(poly, str):
        try:
            poly = json.loads(poly)
        except Exception:
            poly = None
    if poly:
        try:
            pts = [(float(x), float(y)) for x, y in poly]
            if pts[0] != pts[-1]:
                pts.append(pts[0])
            return Polygon(pts)
        except Exception:
            pass
    try:
        west = float(record.get("bbox_lon_min"))
        south = float(record.get("bbox_lat_min"))
        east = float(record.get("bbox_lon_max"))
        north = float(record.get("bbox_lat_max"))
        return Polygon([(west, north), (east, north), (east, south), (west, south), (west, north)])
    except Exception:
        return None


def _snapshot_satellite_records(ws: Any) -> Tuple[List[Dict[str, Any]], Any]:
    source = list(getattr(ws, "satellite_detections", []) or getattr(ws, "sat_last_records", []) or [])
    out: List[Dict[str, Any]] = []

    for i, rec in enumerate(source):
        if not isinstance(rec, dict):
            continue
        row = dict(rec)
        geom = _satellite_polygon(row)
        if geom is None:
            continue

        # Keep original polygon as JSON attribute, not as nested Python object.
        if "polygon_lonlat" in row:
            row["polygon_lonlat_json"] = row.get("polygon_lonlat")
            row.pop("polygon_lonlat", None)

        row["autosave_tab"] = "Satellite Detection"
        row["autosave_index"] = i
        row["autosave_time"] = _now_stamp()

        cls = row.get("class_id", row.get("cls", 0))
        try:
            cls_int = int(float(cls))
        except Exception:
            cls_int = 0
        row["class_id"] = cls_int
        row["class_name"] = _class_name(ws, cls_int)

        if "confidence" not in row and "conf" in row:
            row["confidence"] = row.get("conf")

        row["geometry"] = geom
        out.append(row)

    return out, "EPSG:4326"


def _classes_records(ws: Any):
    from shapely.geometry import Point
    records = []
    for i, name in enumerate(_classes(ws)):
        records.append({
            "class_id": int(i),
            "class_name": str(name),
            "autosave_time": _now_stamp(),
            "geometry": Point(0, 0),
        })
    return records, "EPSG:4326"


def _variables_records(ws: Any):
    from shapely.geometry import Point
    rows = []
    stamp = _now_stamp()

    def add(name: str, value: Any, group: str):
        rows.append({
            "variable": str(name),
            "value": _jsonable(value),
            "group_name": str(group),
            "autosave_time": stamp,
            "geometry": Point(0, 0),
        })

    # Var-like fields on workspace.
    try:
        for name, obj in sorted(getattr(ws, "__dict__", {}).items()):
            if name.startswith("_"):
                continue
            if hasattr(obj, "get") and callable(obj.get):
                try:
                    add(name, obj.get(), "workspace_var")
                except Exception:
                    pass
    except Exception:
        pass

    # Project state dataclass.
    try:
        ps = getattr(ws, "project_state", None)
        if ps is not None:
            if is_dataclass(ps):
                for k, v in asdict(ps).items():
                    add(k, v, "project_state")
            elif hasattr(ps, "__dict__"):
                for k, v in ps.__dict__.items():
                    if not k.startswith("_"):
                        add(k, v, "project_state")
    except Exception:
        pass

    # A few runtime counters.
    try:
        add("detection_count_all", len(list(getattr(ws, "dets", []) or [])), "runtime")
        add("satellite_detection_count_all", len(list(getattr(ws, "satellite_detections", []) or getattr(ws, "sat_last_records", []) or [])), "runtime")
        add("app_version", globals().get("MUSTATIL_GLOBALS", {}).get("APP_VERSION", ""), "runtime")
    except Exception:
        pass

    if not rows:
        add("autosave", "no variables found", "runtime")

    return rows, "EPSG:4326"


def _info_records(ws: Any, output_path: Path, kind: str, feature_count: int):
    from shapely.geometry import Point
    return [{
        "autosave_kind": kind,
        "autosave_time": _now_stamp(),
        "output_path": str(output_path),
        "feature_count": int(feature_count),
        "interval_minutes": AUTOSAVE_INTERVAL_MS / 60000.0,
        "source_image": str(_safe_var_get(getattr(ws, "image", None), "") or ""),
        "satellite_tif": str(_safe_var_get(getattr(ws, "sat_output_tif", None), "") or ""),
        "project": str(_safe_var_get(getattr(ws, "project", None), "") or ""),
        "geometry": Point(0, 0),
    }], "EPSG:4326"


def _write_gpkg(path: Path, layer_payloads: List[Tuple[str, List[Dict[str, Any]], Any]]) -> Path:
    import geopandas as gpd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp_{os.getpid()}_{int(time.time())}.gpkg")
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass

    wrote_any = False
    for layer_name, records, crs in layer_payloads:
        if not records:
            continue
        clean = _clean_records_for_gdf(records)
        gdf = gpd.GeoDataFrame(clean, geometry="geometry", crs=crs)
        gdf.to_file(tmp, driver="GPKG", layer=layer_name)
        wrote_any = True

    if not wrote_any:
        raise RuntimeError("No records to write.")

    try:
        if path.exists():
            path.unlink()
        os.replace(tmp, path)
        return path
    except Exception as exc:
        # Windows may lock an already opened GeoPackage. Keep the requested name if possible;
        # otherwise write timestamp fallback.
        fallback = path.with_name(f"{path.stem}_{_file_stamp()}{path.suffix}")
        _log(f"Could not replace {path}; writing fallback {fallback}: {exc}")
        try:
            if fallback.exists():
                fallback.unlink()
            os.replace(tmp, fallback)
            return fallback
        except Exception:
            try:
                tmp.unlink()
            except Exception:
                pass
            raise


def _write_detection_file(ws: Any) -> Optional[Path]:
    det_records, det_crs = _snapshot_detection_records(ws)
    if not det_records:
        return None
    out = _detection_autosave_path(ws)
    if out is None:
        return None

    classes_records, classes_crs = _classes_records(ws)
    vars_records, vars_crs = _variables_records(ws)
    info_records, info_crs = _info_records(ws, out, "detection", len(det_records))

    return _write_gpkg(out, [
        ("detection_all", det_records, det_crs),
        ("mustatil_classes", classes_records, classes_crs),
        ("mustatil_variables", vars_records, vars_crs),
        ("autosave_info", info_records, info_crs),
    ])


def _write_satellite_file(ws: Any) -> Optional[Path]:
    sat_records, sat_crs = _snapshot_satellite_records(ws)
    if not sat_records:
        return None
    out = _satellite_autosave_path(ws)
    if out is None:
        return None

    classes_records, classes_crs = _classes_records(ws)
    vars_records, vars_crs = _variables_records(ws)
    info_records, info_crs = _info_records(ws, out, "satellite_detection", len(sat_records))

    return _write_gpkg(out, [
        ("satellite_all", sat_records, sat_crs),
        ("mustatil_classes", classes_records, classes_crs),
        ("mustatil_variables", vars_records, vars_crs),
        ("autosave_info", info_records, info_crs),
    ])


def _autosave_worker(ws: Any, reason: str) -> None:
    lock = _ACTIVE_WRITES.setdefault(id(ws), threading.Lock())
    if not lock.acquire(blocking=False):
        _status(ws, "GPKG autosave skipped: previous autosave is still writing.")
        return

    try:
        written: List[str] = []
        det_count = len(list(getattr(ws, "dets", []) or []))
        sat_count = len(list(getattr(ws, "satellite_detections", []) or getattr(ws, "sat_last_records", []) or []))

        if det_count:
            p = _write_detection_file(ws)
            if p is not None:
                written.append(str(p))

        if sat_count:
            p = _write_satellite_file(ws)
            if p is not None:
                written.append(str(p))

        if written:
            _status(ws, "GPKG autosave written (" + reason + "): " + " | ".join(written))
        else:
            _status(ws, "GPKG autosave: no Detection/Satellite records available yet.")
    except Exception:
        _status(ws, "GPKG autosave failed. See mustatil_detection_satellite_autosave_gpkg.log")
        _log(traceback.format_exc())
    finally:
        try:
            lock.release()
        except Exception:
            pass


def _start_autosave(ws: Any, reason: str = "timer") -> None:
    # Snapshot and writing use worker thread to avoid freezing the Qt UI.
    try:
        t = threading.Thread(target=_autosave_worker, args=(ws, reason), daemon=True)
        t.start()
    except Exception:
        _log("Could not start autosave thread:\n" + traceback.format_exc())


def _install_workspace(ws: Any) -> None:
    if ws is None:
        return
    if id(ws) in _INSTALLED_WORKSPACES:
        return
    _INSTALLED_WORKSPACES.add(id(ws))

    QTabWidget, QApplication, QTimer = _qt()
    if QTimer is None:
        return

    _status(ws, "Installing Detection/Satellite GPKG autosave: every 10 minutes.")

    # Timer belongs to workspace so it stays alive.
    try:
        timer = QTimer(ws)
        timer.setInterval(max(60_000, AUTOSAVE_INTERVAL_MS))
        timer.timeout.connect(lambda: _start_autosave(ws, "10-minute timer"))
        timer.start()
        ws._mustatil_gpkg_autosave_timer = timer
    except Exception:
        _log("Could not create autosave timer:\n" + traceback.format_exc())

    # First delayed save after startup, in case detections were imported from project.
    try:
        QTimer.singleShot(max(10_000, FIRST_AUTOSAVE_DELAY_MS), lambda: _start_autosave(ws, "startup delayed"))
    except Exception:
        pass

    # Patch run_task so completed detection tasks are saved immediately too.
    try:
        if hasattr(ws, "run_task") and not getattr(ws.run_task, "_mustatil_autosave_wrapped", False):
            original_run_task = ws.run_task

            def run_task_wrapper(name: str, func, allow_parallel: bool = False):
                task_name = str(name or "")

                def wrapped_func():
                    result = func()
                    try:
                        low = task_name.lower()
                        fname = str(getattr(func, "__name__", "") or "").lower()
                        if ("detection" in low or "detect" in fname or "satellite" in low or "satellite" in fname):
                            _start_autosave(ws, f"after task {task_name}")
                    except Exception:
                        _log("post-task autosave start failed:\n" + traceback.format_exc())
                    return result

                return original_run_task(task_name, wrapped_func, allow_parallel=allow_parallel)

            run_task_wrapper._mustatil_autosave_wrapped = True
            ws.run_task = run_task_wrapper
            _status(ws, "Detection/Satellite GPKG autosave hooked into run_task.")
    except Exception:
        _log("Could not patch run_task:\n" + traceback.format_exc())

    # Public manual helper for console.
    try:
        ws.autosave_detection_satellite_gpkg_now = lambda: _start_autosave(ws, "manual")
    except Exception:
        pass


def _patch_qtabwidget() -> None:
    global _PATCHED_QTAB, _ORIG_ADD_TAB, _ORIG_INSERT_TAB
    if _PATCHED_QTAB:
        return

    QTabWidget, QApplication, QTimer = _qt()
    if QTabWidget is None:
        return

    _ORIG_ADD_TAB = QTabWidget.addTab
    _ORIG_INSERT_TAB = QTabWidget.insertTab

    def addTab_patched(self, page, *args, **kwargs):
        title = _tab_title_from_args(args)
        idx = _ORIG_ADD_TAB(self, page, *args, **kwargs)
        try:
            if _is_target_tab_title(title):
                ws = _workspace_from_widget(self) or _workspace_from_widget(page)
                _install_workspace(ws)
        except Exception:
            _log("addTab hook failed:\n" + traceback.format_exc())
        return idx

    def insertTab_patched(self, index, page, *args, **kwargs):
        title = _tab_title_from_args(args)
        idx = _ORIG_INSERT_TAB(self, index, page, *args, **kwargs)
        try:
            if _is_target_tab_title(title):
                ws = _workspace_from_widget(self) or _workspace_from_widget(page)
                _install_workspace(ws)
        except Exception:
            _log("insertTab hook failed:\n" + traceback.format_exc())
        return idx

    QTabWidget.addTab = addTab_patched
    QTabWidget.insertTab = insertTab_patched
    _PATCHED_QTAB = True
    _log("QTabWidget.addTab/insertTab hooked for autosave installation.")


def _scan_existing_windows() -> None:
    # Useful if plugin is imported after the workspace already exists.
    QTabWidget, QApplication, QTimer = _qt()
    if QApplication is None:
        return
    try:
        app = QApplication.instance()
        if app is None:
            return
        for top in app.topLevelWidgets():
            ws = _workspace_from_widget(top)
            if ws is not None:
                _install_workspace(ws)
    except Exception:
        _log("existing window scan failed:\n" + traceback.format_exc())


def mustatil_plugin_init():
    _patch_qtabwidget()
    _scan_existing_windows()
    _log("Plugin init complete.")


# Auto-init because Mustatil imports plugin files before the main workspace is created.
try:
    mustatil_plugin_init()
except Exception:
    _log("Auto-init failed:\n" + traceback.format_exc())
