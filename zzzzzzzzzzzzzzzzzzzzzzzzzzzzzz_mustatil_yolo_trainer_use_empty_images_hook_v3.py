
# -*- coding: utf-8 -*-
"""
Mustatil GIS Web Review Map v25 Undo RightClick

A fresh, complete WebView/Leaflet GIS review tab next to AI Pipeline.

Requested features:
- Ergonomic top toolbar.
- Web map server selector incl. Google, Bing, Esri, OSM, Carto, OpenTopo, custom.
- Image/raster background selectable at the top.
- Detection Preview background button.
- Confidence slider at the top.
- Class/model-index dropdown at the top.
- Geo-NMS button at the top.
- Box Select drag rectangle to select multiple detections.
- Click inside rectangles/polygons selects them.
- Right-click one or multiple selected detections for review actions.
- Button: "Send Pos/FP Crops → Annotator"
  Creates raw satellite/map crops without drawing the review rectangles on the crop.
  Saves crops and YOLO labels for positive / false_positive review training.

Requires PySide6 QtWebEngine + QtWebChannel for the WebView.
"""

from __future__ import annotations

import base64
import json
import math
import os
import subprocess
import sys
import tempfile
import traceback
import threading
import socketserver
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PLUGIN_NAME = "Mustatil GIS Web Review Map v25 Undo RightClick"
MUSTATIL_GLOBALS = globals().get("MUSTATIL_GLOBALS", {})

# v18: keep QtWebEngine hardware acceleration available for smooth WebView panning.
# v16/v17 disabled GPU compositing to avoid black maps, but that can make
# satellite WebView panning very ruckelig. The safer fix is to keep the image
# canvas QPainter-based while letting QtWebEngine use its own renderer.
try:
    _old_chromium_flags = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
    for _bad in ("--disable-gpu-compositing", "--disable-gpu"):
        _old_chromium_flags = _old_chromium_flags.replace(_bad, "")
    _good = "--ignore-gpu-blocklist"
    if _good not in _old_chromium_flags:
        _old_chromium_flags = (_old_chromium_flags + " " + _good).strip()
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = " ".join(_old_chromium_flags.split())
except Exception:
    pass


_TAB_TITLE = "GIS Web Review Map"
_INSTALLED = False
_REENTRANT = False
_LAST_WORKSPACE = None

# Cross-thread result bus for smooth TIFF viewport rendering.
try:
    from PySide6.QtCore import QObject, Signal
    class _GisViewportRenderBus(QObject):
        resultReady = Signal(object)
        errorReady = Signal(str)
except Exception:
    _GisViewportRenderBus = None


MAP_SERVERS = {
    "Google Satellite": "https://mt{snum}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}&hl=de",
    "Google Hybrid": "https://mt{snum}.google.com/vt/lyrs=y&x={x}&y={y}&z={z}&hl=de",
    "Google Roadmap": "https://mt{snum}.google.com/vt/lyrs=m&x={x}&y={y}&z={z}&hl=de",
    "Google Terrain": "https://mt{snum}.google.com/vt/lyrs=p&x={x}&y={y}&z={z}&hl=de",
    "Bing Satellite": "https://ecn.t{snum}.tiles.virtualearth.net/tiles/a{q}.jpeg?g=14574&mkt=de-DE&n=z",
    "Bing Hybrid": "https://ecn.t{snum}.tiles.virtualearth.net/tiles/h{q}.jpeg?g=14574&mkt=de-DE&n=z",
    "Bing Road": "https://ecn.t{snum}.tiles.virtualearth.net/tiles/r{q}.jpeg?g=14574&mkt=de-DE&n=z",

    # Open Sentinel-2 cloudless basemaps by EOX / s2maps.eu.
    # These are WMTS/XYZ-style EPSG:3857 tile endpoints.
    "Sentinel-2 Cloudless 2024 (EOX)": "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2024_3857/default/g/{z}/{y}/{x}.jpg",
    "Sentinel-2 Cloudless 2023 (EOX)": "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2023_3857/default/g/{z}/{y}/{x}.jpg",
    "Sentinel-2 Cloudless 2022 (EOX)": "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2022_3857/default/g/{z}/{y}/{x}.jpg",
    "Sentinel-2 Cloudless 2021 (EOX)": "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2021_3857/default/g/{z}/{y}/{x}.jpg",
    "Sentinel-2 Cloudless 2020 (EOX)": "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg",
    "Sentinel-2 Cloudless Classic (EOX)": "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless_3857/default/GoogleMapsCompatible/{z}/{y}/{x}.jpg",

    "Esri World Imagery": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "OpenStreetMap": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "OpenTopoMap": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
    "Carto Voyager": "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
    "Carto Light": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
    "Carto Dark": "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
    "Custom URL": "",
}

CONFIDENCE_COLUMNS = ["confidence", "conf", "score", "probability", "prob", "model_confidence"]
CLASS_COLUMNS = ["class", "class_id", "model_index", "model_idx", "cls", "label", "class_name", "name", "category"]
POSITIVE_STATUSES = {"correct", "positive", "true_positive", "tp", "confirmed_archaeology", "confirmed"}
FALSE_POSITIVE_STATUSES = {"false_positive", "false positive", "fp"}


def _log(msg: str) -> None:
    try:
        print(f"[{PLUGIN_NAME}] {msg}", flush=True)
    except Exception:
        pass


def _warn_once(key: str, msg: str) -> None:
    try:
        if not hasattr(_warn_once, "_seen"):
            _warn_once._seen = set()
        if key in _warn_once._seen:
            return
        _warn_once._seen.add(key)
        _log(msg)
    except Exception:
        pass


def _find_workspace(widget=None):
    global _LAST_WORKSPACE
    try:
        if _LAST_WORKSPACE is not None and (hasattr(_LAST_WORKSPACE, "log") or hasattr(_LAST_WORKSPACE, "tabs")):
            return _LAST_WORKSPACE
    except Exception:
        pass
    try:
        w = widget
        for _ in range(60):
            if w is None:
                break
            if hasattr(w, "log") and (hasattr(w, "tabs") or hasattr(w, "project")):
                _LAST_WORKSPACE = w
                return w
            w = w.parent()
    except Exception:
        pass
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app:
            for w in app.topLevelWidgets():
                if hasattr(w, "log") and (hasattr(w, "tabs") or hasattr(w, "project")):
                    _LAST_WORKSPACE = w
                    return w
    except Exception:
        pass
    return None


def _gui_log(widget, msg: str) -> None:
    try:
        ws = _find_workspace(widget)
        if ws is not None and hasattr(ws, "log"):
            ws.log(msg)
            return
    except Exception:
        pass
    _log(msg)


def _gpd():
    try:
        import geopandas as gpd
        return gpd
    except Exception as exc:
        _warn_once("geopandas", f"geopandas unavailable: {exc}")
        return None


def _sanitize_layer_name(name: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in str(name or "layer"))
    return (out[:70] or "layer")


def _pixmap_to_temp_png(pixmap, stem: str = "mustatil_detection_preview") -> Optional[str]:
    try:
        out = Path(tempfile.gettempdir()) / f"{stem}.png"
        if pixmap is None or pixmap.isNull():
            return None
        pixmap.save(str(out), "PNG")
        return str(out)
    except Exception:
        return None


def _data_url(path: str) -> str:
    suffix = Path(path).suffix.lower()
    mime = "image/png"
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    data = Path(path).read_bytes()
    return "data:%s;base64,%s" % (mime, base64.b64encode(data).decode("ascii"))


def _to_float(v, default=None):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _find_conf_col(gdf):
    if gdf is None:
        return None
    cols_lower = {str(c).lower(): c for c in gdf.columns}
    for name in CONFIDENCE_COLUMNS:
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]
    return None


def _find_class_col(gdf):
    if gdf is None:
        return None
    cols_lower = {str(c).lower(): c for c in gdf.columns}
    for name in CLASS_COLUMNS:
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]
    return None


def _filtered_gdf(gdf, min_conf: float, class_filter: str):
    if gdf is None:
        return gdf
    out = gdf
    conf_col = _find_conf_col(out)
    if conf_col is not None:
        try:
            vals = out[conf_col].apply(lambda x: _to_float(x, -999.0))
            out = out[vals >= float(min_conf)].copy()
        except Exception:
            pass
    if class_filter and class_filter not in {"All", "All classes", ""}:
        class_col = _find_class_col(out)
        if class_col is not None:
            try:
                out = out[out[class_col].astype(str) == str(class_filter)].copy()
            except Exception:
                pass
    return out


def _load_image_preview(path: str, max_size: int = 2400) -> Tuple[str, int, int, Dict[str, Any]]:
    meta = {
        "source": path,
        "raster_crs": None,
        "raster_transform": None,
        "raster_width": None,
        "raster_height": None,
        "scale_x": 1.0,
        "scale_y": 1.0,
    }
    out = Path(tempfile.gettempdir()) / "mustatil_gis_web_review_image.png"
    suffix = Path(path).suffix.lower()

    if suffix in {".tif", ".tiff"}:
        try:
            svc = globals().get("detection_preview_service") or MUSTATIL_GLOBALS.get("detection_preview_service")
            if svc is None:
                try:
                    import detection_preview_service as svc
                except Exception:
                    svc = None
            if svc is not None and hasattr(svc, "load_detection_preview_image"):
                img, ow, oh = svc.load_detection_preview_image(path, maxs=max_size)
                if img is not None:
                    try:
                        import rasterio
                        with rasterio.open(path) as src:
                            meta["raster_crs"] = src.crs
                            meta["raster_transform"] = src.transform
                            meta["raster_width"] = int(src.width)
                            meta["raster_height"] = int(src.height)
                            ow = int(src.width)
                            oh = int(src.height)
                    except Exception:
                        pass
                    img.save(out, "PNG")
                    meta["scale_x"] = float(img.width) / max(1.0, float(ow or img.width))
                    meta["scale_y"] = float(img.height) / max(1.0, float(oh or img.height))
                    return str(out), int(img.width), int(img.height), meta
        except Exception:
            pass

        try:
            import rasterio
            import numpy as np
            from PIL import Image
            from rasterio.enums import Resampling
            with rasterio.open(path) as src:
                meta["raster_crs"] = src.crs
                meta["raster_transform"] = src.transform
                meta["raster_width"] = int(src.width)
                meta["raster_height"] = int(src.height)
                scale = min(1.0, float(max_size) / max(1, max(src.width, src.height)))
                out_w = max(1, int(src.width * scale))
                out_h = max(1, int(src.height * scale))
                bands = list(range(1, min(3, src.count) + 1))
                arr = src.read(bands, out_shape=(len(bands), out_h, out_w), resampling=Resampling.bilinear).astype("float32")
                lo, hi = float(np.nanpercentile(arr, 2)), float(np.nanpercentile(arr, 98))
                if hi <= lo:
                    hi = lo + 1.0
                arr = np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype("uint8")
                if arr.shape[0] == 1:
                    arr = np.repeat(arr, 3, axis=0)
                img = Image.fromarray(np.transpose(arr[:3], (1, 2, 0)), "RGB")
                img.save(out, "PNG")
                meta["scale_x"] = float(img.width) / max(1.0, float(src.width))
                meta["scale_y"] = float(img.height) / max(1.0, float(src.height))
                return str(out), int(img.width), int(img.height), meta
        except Exception:
            pass

    from PIL import Image
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_size, max_size))
    img.save(out, "PNG")
    return str(out), int(img.width), int(img.height), meta


def _geom_to_pixel_gdf(gdf, image_meta: Dict[str, Any]):
    """
    Convert vector detections to ORIGINAL TIFF pixel coordinates.

    v12 fix:
      The GIS Image/Raster mode now uses original-image scene coordinates.
      Therefore boxes must be converted to *original raster pixels*, not preview
      pixels. This function tries several safe candidates and chooses the one
      that best matches the raster extent.

    Candidate order:
      1) CRS-aware vector CRS -> raster CRS -> inverse affine -> pixels
      2) CRS-aware swapped XY -> raster CRS -> inverse affine -> pixels
      3) No CRS: assume geometry is already in raster CRS -> inverse affine
      4) Already pixel coordinates

    If a CRS-aware candidate is reasonable, it is preferred over "already
    pixel" so lon/lat values such as 44,24 are not wrongly treated as pixels.
    """
    gpd = _gpd()
    if gpd is None or gdf is None:
        return gdf
    try:
        from shapely.ops import transform as shp_transform
    except Exception:
        return gdf

    try:
        gg_original = gdf.copy()
    except Exception:
        return gdf

    tr = image_meta.get("raster_transform")
    raster_crs = image_meta.get("raster_crs")
    W = int(image_meta.get("raster_width") or 0)
    H = int(image_meta.get("raster_height") or 0)

    def as_geodf(obj, mode: str):
        try:
            out = gpd.GeoDataFrame(obj, geometry="geometry", crs=None)
            out.attrs["mustatil_pixel_mode"] = mode
            return out
        except Exception:
            try:
                obj.attrs["mustatil_pixel_mode"] = mode
            except Exception:
                pass
            return obj

    def bounds_of(obj):
        try:
            b = obj.total_bounds
            return tuple(float(v) for v in b)
        except Exception:
            try:
                b = obj.geometry.total_bounds
                return tuple(float(v) for v in b)
            except Exception:
                return None

    def score_bounds(bounds):
        if bounds is None:
            return -1e9
        minx, miny, maxx, maxy = bounds
        try:
            if not all(math.isfinite(v) for v in [minx, miny, maxx, maxy]):
                return -1e9
        except Exception:
            return -1e9
        if maxx <= minx or maxy <= miny:
            return -1e9
        if W <= 0 or H <= 0:
            return 0.0

        ix1 = max(0.0, minx)
        iy1 = max(0.0, miny)
        ix2 = min(float(W), maxx)
        iy2 = min(float(H), maxy)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih

        box_area = max(1.0, (maxx - minx) * (maxy - miny))
        raster_area = max(1.0, float(W) * float(H))
        overlap_ratio = inter / box_area

        outside = 0.0
        if maxx < 0:
            outside += abs(maxx)
        if maxy < 0:
            outside += abs(maxy)
        if minx > W:
            outside += abs(minx - W)
        if miny > H:
            outside += abs(miny - H)
        outside_penalty = outside / max(1.0, max(W, H))

        size_ratio = min(1.0, box_area / raster_area)
        tiny_penalty = 0.0
        if box_area < 4.0 and raster_area > 1000000:
            tiny_penalty = 0.25

        return overlap_ratio + size_ratio * 0.05 - outside_penalty - tiny_penalty

    candidates = []

    # Candidate 1: normal CRS-aware transform.
    try:
        if tr is not None and raster_crs is not None and getattr(gg_original, "crs", None) is not None:
            gg = gg_original.copy()
            if str(gg.crs) != str(raster_crs):
                gg = gg.to_crs(raster_crs)
            inv = ~tr

            def f(x, y, z=None):
                col, row = inv * (x, y)
                return float(col), float(row)

            gg["geometry"] = gg.geometry.apply(lambda geom: shp_transform(f, geom) if geom is not None else geom)
            cand = as_geodf(gg, "crs_to_raster_pixels")
            candidates.append(("crs_to_raster_pixels", cand, score_bounds(bounds_of(cand)), 100.0))
    except Exception:
        pass

    # Candidate 2: swapped XY rescue.
    try:
        if tr is not None and raster_crs is not None and getattr(gg_original, "crs", None) is not None:
            gg = gg_original.copy()

            def swap_xy(x, y, z=None):
                return float(y), float(x)

            gg["geometry"] = gg.geometry.apply(lambda geom: shp_transform(swap_xy, geom) if geom is not None else geom)
            try:
                gg = gpd.GeoDataFrame(gg, geometry="geometry", crs=gg_original.crs)
            except Exception:
                pass
            if str(gg.crs) != str(raster_crs):
                gg = gg.to_crs(raster_crs)
            inv = ~tr

            def f2(x, y, z=None):
                col, row = inv * (x, y)
                return float(col), float(row)

            gg["geometry"] = gg.geometry.apply(lambda geom: shp_transform(f2, geom) if geom is not None else geom)
            cand = as_geodf(gg, "swapped_xy_crs_to_raster_pixels")
            candidates.append(("swapped_xy_crs_to_raster_pixels", cand, score_bounds(bounds_of(cand)), 40.0))
    except Exception:
        pass

    # Candidate 3: vector has no CRS but coordinates might already be in raster CRS.
    try:
        if tr is not None:
            gg = gg_original.copy()
            inv = ~tr

            def f3(x, y, z=None):
                col, row = inv * (x, y)
                return float(col), float(row)

            gg["geometry"] = gg.geometry.apply(lambda geom: shp_transform(f3, geom) if geom is not None else geom)
            cand = as_geodf(gg, "same_raster_crs_to_pixels")
            priority = 60.0 if getattr(gg_original, "crs", None) is None else 5.0
            candidates.append(("same_raster_crs_to_pixels", cand, score_bounds(bounds_of(cand)), priority))
    except Exception:
        pass

    # Candidate 4: already pixel coordinates.
    try:
        cand = as_geodf(gg_original.copy(), "already_pixel_coordinates")
        priority = 70.0 if getattr(gg_original, "crs", None) is None else 1.0
        candidates.append(("already_pixel_coordinates", cand, score_bounds(bounds_of(cand)), priority))
    except Exception:
        pass

    if not candidates:
        return as_geodf(gg_original, "unconverted")

    best = None
    best_key = -1e18
    for mode, cand, score, priority in candidates:
        combined = float(score) * 1000.0 + float(priority)
        if combined > best_key:
            best = cand
            best_key = combined

    try:
        best.attrs["mustatil_pixel_candidates"] = [
            {"mode": m, "score": float(s), "priority": float(p), "bounds": bounds_of(c)}
            for m, c, s, p in candidates
        ]
    except Exception:
        pass

    return best


def _to_geojson_with_ids(gdf, layer_index: int, mode: str = "map", image_meta: Optional[Dict[str, Any]] = None,
                         min_conf: float = 0.0, class_filter: str = "All") -> Dict[str, Any]:
    if gdf is None:
        return {"type": "FeatureCollection", "features": []}

    fgdf = _filtered_gdf(gdf, min_conf, class_filter)
    if fgdf is None:
        return {"type": "FeatureCollection", "features": []}
    original_indices = list(fgdf.index)

    gg = fgdf
    if mode == "image":
        gg = _geom_to_pixel_gdf(fgdf, image_meta or {})
    else:
        try:
            if getattr(fgdf, "crs", None) is not None:
                gg = fgdf.to_crs("EPSG:4326")
        except Exception:
            gg = fgdf

    data = json.loads(gg.to_json())
    feats = data.get("features", []) or []
    all_index = list(gdf.index)

    for i, feat in enumerate(feats):
        props = feat.setdefault("properties", {})
        props["__layer"] = int(layer_index)
        try:
            pos_row = int(all_index.index(original_indices[i]))
        except Exception:
            pos_row = int(i)
        props["__row"] = pos_row
        try:
            props["review_status"] = str(gdf.iloc[pos_row].get("review_status", "") or "")
        except Exception:
            props["review_status"] = ""
    return data


def _geo_iou(a, b) -> float:
    try:
        if a is None or b is None or a.is_empty or b.is_empty:
            return 0.0
        inter = a.intersection(b).area
        if inter <= 0:
            return 0.0
        union = a.union(b).area
        if union <= 0:
            return 0.0
        return float(inter / union)
    except Exception:
        return 0.0


def _apply_geo_nms_gdf(gdf, iou_threshold: float = 0.30):
    if gdf is None or len(gdf) == 0:
        return gdf
    conf_col = _find_conf_col(gdf)
    class_col = _find_class_col(gdf)
    work = gdf.copy()
    if conf_col is None:
        work["_nms_conf_tmp"] = 1.0
        conf_col = "_nms_conf_tmp"
    else:
        work["_nms_conf_tmp"] = work[conf_col].apply(lambda x: _to_float(x, 0.0))
        conf_col = "_nms_conf_tmp"

    keep_positions = []
    if class_col is None:
        groups = [("__all__", list(work.index))]
    else:
        groups = [(cls_value, list(sub.index)) for cls_value, sub in work.groupby(class_col, dropna=False)]

    index_to_pos = {idx: pos for pos, idx in enumerate(work.index)}

    for _cls, idxs in groups:
        positions = [index_to_pos[idx] for idx in idxs if idx in index_to_pos]
        positions.sort(key=lambda p: float(work.iloc[p][conf_col]), reverse=True)
        kept_for_group = []
        for p in positions:
            geom = work.iloc[p].geometry
            suppress = False
            for kp in kept_for_group:
                if _geo_iou(geom, work.iloc[kp].geometry) >= float(iou_threshold):
                    suppress = True
                    break
            if not suppress:
                kept_for_group.append(p)
                keep_positions.append(p)

    keep_positions = sorted(set(keep_positions))
    result = work.iloc[keep_positions].copy().reset_index(drop=True)
    if "_nms_conf_tmp" in result.columns:
        result = result.drop(columns=["_nms_conf_tmp"], errors="ignore")
    return result


def _tile_quadkey(x: int, y: int, z: int) -> str:
    q = ""
    for i in range(z, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if x & mask:
            digit += 1
        if y & mask:
            digit += 2
        q += str(digit)
    return q


def _tile_url(template: str, x: int, y: int, z: int) -> str:
    max_tile = 2 ** z
    xx = ((int(x) % max_tile) + max_tile) % max_tile
    yy = int(y)
    q = _tile_quadkey(xx, yy, z)
    snum = abs((xx + yy + z) % 4)
    sub = ["a", "b", "c"][abs((xx + yy) % 3)]
    rnd = abs((xx * 31 + yy * 17 + z) % 4)
    return (template
            .replace("{x}", str(xx))
            .replace("{y}", str(yy))
            .replace("{z}", str(z))
            .replace("{q}", q)
            .replace("{quadkey}", q)
            .replace("{snum}", str(snum))
            .replace("{rnd}", str(rnd))
            .replace("{s}", sub)
            .replace("*GMX*", str(xx))
            .replace("*GMY*", str(yy))
            .replace("*ZM1*", str(z))
            .replace("*IZM*", str(z))
            .replace("*RND*", str(rnd))
            .replace("*LAN*", "de")
            .replace("*LAN-LAN*", "de-DE"))


def _world_px(lon: float, lat: float, z: int, tile_size: int = 256) -> Tuple[float, float]:
    lat = max(min(float(lat), 85.05112878), -85.05112878)
    scale = tile_size * (2 ** int(z))
    x = (float(lon) + 180.0) / 360.0 * scale
    siny = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)) * scale
    return x, y


def _lonlat_tile(lon: float, lat: float, z: int, tile_size: int = 256) -> Tuple[int, int]:
    wx, wy = _world_px(lon, lat, z, tile_size)
    return int(wx // tile_size), int(wy // tile_size)


def _download_tile(template: str, x: int, y: int, z: int):
    from PIL import Image
    try:
        import requests
        url = _tile_url(template, x, y, z)
        r = requests.get(url, timeout=(5, 15), headers={"User-Agent": "Mustatil-GIS-Web-Review/1.0"})
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        try:
            from urllib.request import Request, urlopen
            url = _tile_url(template, x, y, z)
            req = Request(url, headers={"User-Agent": "Mustatil-GIS-Web-Review/1.0"})
            with urlopen(req, timeout=15) as resp:
                data = resp.read()
            return Image.open(BytesIO(data)).convert("RGB")
        except Exception:
            return Image.new("RGB", (256, 256), (220, 220, 220))


def _render_satellite_crop_for_geom(geom_wgs84, template: str, z: int, pad_factor: float = 1.2, out_size: int = 512):
    """
    Render a raw map/satellite crop for a single EPSG:4326 geometry.
    The detection rectangle is NOT drawn into the image.
    Returns PIL.Image crop and bbox in YOLO normalized format.
    """
    from PIL import Image

    min_lon, min_lat, max_lon, max_lat = [float(v) for v in geom_wgs84.bounds]
    cx_lon = (min_lon + max_lon) / 2.0
    cy_lat = (min_lat + max_lat) / 2.0

    # Geometry bounds in world pixels.
    x1, y1 = _world_px(min_lon, max_lat, z)
    x2, y2 = _world_px(max_lon, min_lat, z)
    left, right = min(x1, x2), max(x1, x2)
    top, bottom = min(y1, y2), max(y1, y2)
    bw = max(8.0, right - left)
    bh = max(8.0, bottom - top)

    crop_w = max(out_size, int(max(bw, bh) * pad_factor))
    crop_h = crop_w

    cx, cy = _world_px(cx_lon, cy_lat, z)
    crop_left = cx - crop_w / 2.0
    crop_top = cy - crop_h / 2.0
    crop_right = cx + crop_w / 2.0
    crop_bottom = cy + crop_h / 2.0

    tile_size = 256
    tx_min = int(math.floor(crop_left / tile_size))
    ty_min = int(math.floor(crop_top / tile_size))
    tx_max = int(math.floor((crop_right - 1) / tile_size))
    ty_max = int(math.floor((crop_bottom - 1) / tile_size))

    mosaic_w = (tx_max - tx_min + 1) * tile_size
    mosaic_h = (ty_max - ty_min + 1) * tile_size
    mosaic = Image.new("RGB", (mosaic_w, mosaic_h), (230, 230, 230))

    for tx in range(tx_min, tx_max + 1):
        for ty in range(ty_min, ty_max + 1):
            tile = _download_tile(template, tx, ty, z)
            mosaic.paste(tile, ((tx - tx_min) * tile_size, (ty - ty_min) * tile_size))

    local_left = crop_left - tx_min * tile_size
    local_top = crop_top - ty_min * tile_size
    crop = mosaic.crop((int(local_left), int(local_top), int(local_left + crop_w), int(local_top + crop_h)))

    # BBox in crop pixels.
    box_x1 = left - crop_left
    box_y1 = top - crop_top
    box_x2 = right - crop_left
    box_y2 = bottom - crop_top

    yolo_cx = ((box_x1 + box_x2) / 2.0) / crop_w
    yolo_cy = ((box_y1 + box_y2) / 2.0) / crop_h
    yolo_w = (box_x2 - box_x1) / crop_w
    yolo_h = (box_y2 - box_y1) / crop_h

    vals = [max(0.0, min(1.0, v)) for v in [yolo_cx, yolo_cy, yolo_w, yolo_h]]
    return crop, vals




def _tile_lonlat_bounds(x: int, y: int, z: int) -> Tuple[float, float, float, float]:
    """
    Return lon/lat bounds for XYZ tile: west, south, east, north.
    """
    def n2lon(tx, tz):
        return tx / (2.0 ** tz) * 360.0 - 180.0
    def n2lat(ty, tz):
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ty / (2.0 ** tz))))
        return math.degrees(lat_rad)
    west = n2lon(x, z)
    east = n2lon(x + 1, z)
    north = n2lat(y, z)
    south = n2lat(y + 1, z)
    return west, south, east, north


def _render_basemap_region(template: str, region: List[float], z: int, max_tiles: int = 160):
    """
    Render selected XYZ/quadkey basemap over lon/lat region.

    region = [west, east, south, north]
    Returns (PIL image, extent [west,east,south,north], tile_info)
    """
    from PIL import Image
    west, east, south, north = [float(v) for v in region]
    tx1, ty1 = _lonlat_tile(west, north, z)
    tx2, ty2 = _lonlat_tile(east, south, z)
    tx_min, tx_max = min(tx1, tx2), max(tx1, tx2)
    ty_min, ty_max = min(ty1, ty2), max(ty1, ty2)
    cols = tx_max - tx_min + 1
    rows = ty_max - ty_min + 1
    count = cols * rows
    if count > int(max_tiles):
        raise RuntimeError(f"Too many map tiles for GMT basemap: {cols} x {rows} = {count}. Lower zoom or use smaller region.")
    mosaic = Image.new("RGB", (cols * 256, rows * 256), (230, 230, 230))
    failed = 0
    for tx in range(tx_min, tx_max + 1):
        for ty in range(ty_min, ty_max + 1):
            try:
                tile = _download_tile(template, tx, ty, z).resize((256, 256))
            except Exception:
                failed += 1
                tile = Image.new("RGB", (256, 256), (220, 220, 220))
            mosaic.paste(tile, ((tx - tx_min) * 256, (ty - ty_min) * 256))

    west2, _south_top, _east_top, north2 = _tile_lonlat_bounds(tx_min, ty_min, z)
    _west_bottom, south2, east2, _north_bottom = _tile_lonlat_bounds(tx_max, ty_max, z)
    extent = [west2, east2, south2, north2]
    info = {"cols": cols, "rows": rows, "tiles": count, "failed": failed, "z": z}
    return mosaic, extent, info


# ----------------------------------------------------------------------
# Smooth local image pyramid server for QGIS-like raster zoom
# ----------------------------------------------------------------------

PYRAMID_SESSIONS: Dict[str, Any] = {}
PYRAMID_SERVER = None
PYRAMID_PORT = None


class _QuietThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _PyramidSource:
    """
    On-demand image pyramid source.

    The WebView requests 256x256 tiles. For GeoTIFFs this reads only the
    requested window via rasterio, so very large rasters can be zoomed/panned
    smoothly without loading the whole image into memory.
    """
    def __init__(self, path: str, tile_size: int = 256):
        self.path = str(path)
        self.tile_size = int(tile_size)
        self.is_raster = Path(path).suffix.lower() in {".tif", ".tiff"}
        self.width = 1
        self.height = 1
        self.crs = None
        self.transform = None
        self._tile_cache: Dict[Tuple[int, int, int], bytes] = {}
        self._tile_cache_order: List[Tuple[int, int, int]] = []
        self._max_cache = 768
        self._init_meta()

    def _init_meta(self):
        if self.is_raster:
            try:
                import rasterio
                with rasterio.open(self.path) as src:
                    self.width = int(src.width)
                    self.height = int(src.height)
                    self.crs = src.crs
                    self.transform = src.transform
                    return
            except Exception:
                pass
        from PIL import Image
        with Image.open(self.path) as img:
            self.width, self.height = int(img.width), int(img.height)

    def _cache_get(self, key):
        return self._tile_cache.get(key)

    def _cache_put(self, key, data):
        try:
            self._tile_cache[key] = data
            self._tile_cache_order.append(key)
            while len(self._tile_cache_order) > self._max_cache:
                old = self._tile_cache_order.pop(0)
                self._tile_cache.pop(old, None)
        except Exception:
            pass

    def _blank(self, rgba=(238, 238, 238, 0)):
        from PIL import Image
        return Image.new("RGBA", (self.tile_size, self.tile_size), rgba)

    def tile_png_bytes(self, z: int, x: int, y: int) -> bytes:
        key = (int(z), int(x), int(y))
        hit = self._cache_get(key)
        if hit is not None:
            return hit

        img = self._tile_image(int(z), int(x), int(y))
        bio = BytesIO()
        img.save(bio, format="PNG")
        data = bio.getvalue()
        self._cache_put(key, data)
        return data

    def _tile_window(self, z: int, x: int, y: int):
        # Leaflet CRS.Simple: map coordinate units are image pixels.
        # At zoom z, one map unit is 2^z screen pixels. A 256px tile covers
        # 256 / 2^z image pixels. Negative zoom covers larger windows.
        scale = 2.0 ** float(z)
        if scale <= 0:
            scale = 1.0
        ts = float(self.tile_size)
        x0 = float(x) * ts / scale
        y0 = float(y) * ts / scale
        x1 = float(x + 1) * ts / scale
        y1 = float(y + 1) * ts / scale
        return x0, y0, x1, y1

    def _tile_image(self, z: int, x: int, y: int):
        from PIL import Image
        ts = int(self.tile_size)
        x0, y0, x1, y1 = self._tile_window(z, x, y)

        if x1 <= 0 or y1 <= 0 or x0 >= self.width or y0 >= self.height:
            return self._blank()

        if self.is_raster:
            try:
                import rasterio
                import numpy as np
                from rasterio.windows import Window
                from rasterio.enums import Resampling

                wx0 = max(0.0, x0)
                wy0 = max(0.0, y0)
                wx1 = min(float(self.width), x1)
                wy1 = min(float(self.height), y1)
                if wx1 <= wx0 or wy1 <= wy0:
                    return self._blank()

                window = Window(wx0, wy0, wx1 - wx0, wy1 - wy0)
                with rasterio.open(self.path) as src:
                    bands = list(range(1, min(3, src.count) + 1))
                    arr = src.read(
                        bands,
                        window=window,
                        out_shape=(len(bands), ts, ts),
                        boundless=True,
                        fill_value=0,
                        resampling=Resampling.bilinear,
                    ).astype("float32")

                # Smooth but stable tile normalization.
                finite = arr[np.isfinite(arr)]
                if finite.size:
                    lo = float(np.nanpercentile(finite, 2))
                    hi = float(np.nanpercentile(finite, 98))
                else:
                    lo, hi = 0.0, 1.0
                if hi <= lo:
                    hi = lo + 1.0
                arr = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype("uint8")
                if arr.shape[0] == 1:
                    arr = np.repeat(arr, 3, axis=0)
                rgb = np.transpose(arr[:3], (1, 2, 0))
                return Image.fromarray(rgb, "RGB").convert("RGBA")
            except Exception:
                return self._blank((238, 238, 238, 255))

        try:
            img = Image.open(self.path).convert("RGB")
            crop = img.crop((int(x0), int(y0), int(x1), int(y1))).resize((ts, ts), Image.BILINEAR)
            return crop.convert("RGBA")
        except Exception:
            return self._blank((238, 238, 238, 255))


class _PyramidRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            parts = parsed.path.strip("/").split("/")
            if len(parts) != 5 or parts[0] != "tile":
                self.send_error(404)
                return
            _tile, sid, z, x, ypng = parts
            y = ypng.split(".")[0]
            src = PYRAMID_SESSIONS.get(sid)
            if src is None:
                self.send_error(404)
                return
            data = src.tile_png_bytes(int(z), int(x), int(y))
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "max-age=3600")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception:
            try:
                self.send_error(500)
            except Exception:
                pass


def _ensure_pyramid_server():
    global PYRAMID_SERVER, PYRAMID_PORT
    if PYRAMID_SERVER is not None and PYRAMID_PORT is not None:
        return int(PYRAMID_PORT)
    server = _QuietThreadingHTTPServer(("127.0.0.1", 0), _PyramidRequestHandler)
    PYRAMID_SERVER = server
    PYRAMID_PORT = int(server.server_address[1])
    th = threading.Thread(target=server.serve_forever, name="MustatilImagePyramidServer", daemon=True)
    th.start()
    return int(PYRAMID_PORT)


def _register_pyramid(path: str) -> Tuple[str, int, _PyramidSource]:
    import uuid
    port = _ensure_pyramid_server()
    sid = uuid.uuid4().hex
    src = _PyramidSource(path)
    PYRAMID_SESSIONS[sid] = src
    return sid, port, src


def _html() -> str:
    return r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Mustatil GIS Web Review Map</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
html, body, #map { width: 100%; height: 100%; margin: 0; padding: 0; overflow: hidden; background: #202020; }
.leaflet-container { background: #202020; font-family: Arial, sans-serif; }
.feature-popup button { display: block; width: 230px; margin: 3px 0; padding: 4px; text-align: left; }
.mustatil-status {
  position: absolute; left: 10px; bottom: 10px; z-index: 9999;
  background: rgba(0,0,0,0.70); color: white; padding: 4px 7px;
  border-radius: 4px; font-size: 12px; max-width: 620px;
}
.selection-hint {
  position: absolute; right: 10px; bottom: 10px; z-index: 9999;
  background: rgba(0,0,0,0.70); color: white; padding: 4px 7px;
  border-radius: 4px; font-size: 12px; display:none;
}
</style>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js">
// v23 global API guard for PySide runJavaScript calls.
// Some QtWebEngine timing paths call these from Python before lexical function
// declarations are visible on window. These wrappers keep the WebView safe.
(function(){
  function safeMapReady(){
    return (typeof map !== "undefined" && map !== null);
  }
  window.mustatilMapReady = function(){ return safeMapReady(); };

  if (typeof window.setTileServer !== "function") {
    window.setTileServer = function(urlTemplate, name, maxZoom){
      try {
        if (!safeMapReady()) return false;
        if (window.baseLayer) {
          try { map.removeLayer(window.baseLayer); } catch(e) {}
        }
        window.baseLayer = L.tileLayer(urlTemplate, {
          maxZoom: maxZoom || 22,
          maxNativeZoom: maxZoom || 19,
          attribution: name || "Tiles"
        });
        window.baseLayer.addTo(map);
        return true;
      } catch(e) {
        console.error("setTileServer failed", e);
        return false;
      }
    };
  }

  if (typeof window.clearOverlays !== "function") {
    window.clearOverlays = function(){
      try {
        if (window.vectorLayers) {
          Object.keys(window.vectorLayers).forEach(function(k){
            try { map.removeLayer(window.vectorLayers[k]); } catch(e) {}
          });
        }
        window.vectorLayers = {};
        if (window.selectionLayer) {
          try { window.selectionLayer.clearLayers(); } catch(e) {}
        }
        return true;
      } catch(e) {
        console.error("clearOverlays failed", e);
        return false;
      }
    };
  }

  if (typeof window.addGeoJSONLayer !== "function") {
    window.addGeoJSONLayer = function(layerId, geojson, styleJson){
      try {
        if (!safeMapReady()) return false;
        window.vectorLayers = window.vectorLayers || {};
        var data = (typeof geojson === "string") ? JSON.parse(geojson) : geojson;
        var style = {};
        try { style = (typeof styleJson === "string") ? JSON.parse(styleJson) : (styleJson || {}); } catch(e) {}
        if (window.vectorLayers[layerId]) {
          try { map.removeLayer(window.vectorLayers[layerId]); } catch(e) {}
        }
        var layer = L.geoJSON(data, {
          style: function(feature){
            return {
              color: style.color || "#ff0000",
              weight: style.weight || 2,
              fillColor: style.fillColor || style.color || "#ff0000",
              fillOpacity: style.fillOpacity == null ? 0.15 : style.fillOpacity
            };
          },
          onEachFeature: function(feature, lyr){
            try {
              lyr.on("click", function(){
                if (window.bridge && bridge.selectFeature) {
                  bridge.selectFeature(String(layerId), String(feature.properties && feature.properties.__rowid__ != null ? feature.properties.__rowid__ : ""));
                }
              });
            } catch(e) {}
          }
        }).addTo(map);
        window.vectorLayers[layerId] = layer;
        return true;
      } catch(e) {
        console.error("addGeoJSONLayer failed", e);
        return false;
      }
    };
  }

  if (typeof window.fitLayer !== "function") {
    window.fitLayer = function(layerId){
      try {
        if (!safeMapReady() || !window.vectorLayers || !window.vectorLayers[layerId]) return false;
        map.fitBounds(window.vectorLayers[layerId].getBounds(), {padding:[20,20]});
        return true;
      } catch(e) {
        console.error("fitLayer failed", e);
        return false;
      }
    };
  }

  if (typeof window.setViewSafe !== "function") {
    window.setViewSafe = function(lat, lon, zoom){
      try {
        if (!safeMapReady()) return false;
        map.setView([lat, lon], zoom || map.getZoom() || 12);
        return true;
      } catch(e) {
        console.error("setViewSafe failed", e);
        return false;
      }
    };
  }

  window.__mustatilWebApiReady = true;
})();

</script>
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
</head>
<body>
<div id="map"></div>
<div id="status" class="mustatil-status">Mustatil GIS Web Review Map ready.</div>
<div id="selhint" class="selection-hint">Box Select ON: drag a rectangle to select detections. Right-click selection.</div>
<script>
let bridge = null;
let map = null;
let tileLayer = null;
let imageOverlay = null;
let mapMode = "map";
let layerGroups = {};
let featureIndex = {};
let selectedKeys = {};
let selectMode = false;
let dragStart = null;
let dragRect = null;

function status(msg) {
    const el = document.getElementById("status");
    if (el) el.textContent = msg;
}

function quadKey(x, y, z) {
    let quad = "";
    for (let i = z; i > 0; i--) {
        let digit = 0;
        let mask = 1 << (i - 1);
        if ((x & mask) !== 0) digit += 1;
        if ((y & mask) !== 0) digit += 2;
        quad += digit.toString();
    }
    return quad;
}

const MustatilTileLayer = L.TileLayer.extend({
    getTileUrl: function(coords) {
        let x = coords.x;
        let y = coords.y;
        let z = coords.z;
        let max = Math.pow(2, z);
        x = ((x % max) + max) % max;
        let q = quadKey(x, y, z);
        let snum = Math.abs((x + y + z) % 4).toString();
        let sub = ["a","b","c"][Math.abs((x + y) % 3)];
        let rnd = Math.abs((x * 31 + y * 17 + z) % 4).toString();
        return this._url
            .replaceAll("{x}", String(x))
            .replaceAll("{y}", String(y))
            .replaceAll("{z}", String(z))
            .replaceAll("{q}", q)
            .replaceAll("{quadkey}", q)
            .replaceAll("{snum}", snum)
            .replaceAll("{rnd}", rnd)
            .replaceAll("{s}", sub)
            .replaceAll("*GMX*", String(x))
            .replaceAll("*GMY*", String(y))
            .replaceAll("*ZM1*", String(z))
            .replaceAll("*IZM*", String(z))
            .replaceAll("*RND*", rnd)
            .replaceAll("*LAN*", "de")
            .replaceAll("*LAN-LAN*", "de-DE");
    }
});

function destroyMap() {
    if (map) { try { map.remove(); } catch(e) {} map = null; }
    tileLayer = null; imageOverlay = null; layerGroups = {}; featureIndex = {}; selectedKeys = {};
}

function initMap(mode) {
    if (map && mapMode === mode) return;
    destroyMap();
    mapMode = mode || "map";
    let opts = { preferCanvas: true, zoomControl: true, boxZoom: false, doubleClickZoom: true };
    if (mapMode === "image") {
        opts.crs = L.CRS.Simple; opts.minZoom = -5; opts.maxZoom = 8; opts.center = [0, 0]; opts.zoom = 0;
    } else {
        opts.center = [24.0, 45.0]; opts.zoom = 6; opts.worldCopyJump = true;
    }
    map = L.map("map", opts);
    installSelectionHandlers();
}

function setSelectionMode(enabled) {
    selectMode = !!enabled;
    const el = document.getElementById("selhint");
    if (el) el.style.display = selectMode ? "block" : "none";
    if (map) {
        if (selectMode) { try { map.dragging.disable(); } catch(e) {} }
        else { try { map.dragging.enable(); } catch(e) {} }
    }
    status(selectMode ? "Box Select ON: drag to select multiple detections." : "Box Select OFF.");
}

function installSelectionHandlers() {
    if (!map) return;
    map.on("mousedown", function(e) {
        if (!selectMode) return;
        if (e.originalEvent && e.originalEvent.button !== 0) return;
        dragStart = e.latlng;
        if (dragRect) { try { map.removeLayer(dragRect); } catch(err) {} }
        dragRect = L.rectangle([dragStart, dragStart], {color:"#00ffff", weight:1, dashArray:"4,4", fillOpacity:0.08}).addTo(map);
        try { L.DomEvent.preventDefault(e.originalEvent); } catch(err) {}
    });
    map.on("mousemove", function(e) {
        if (!selectMode || !dragStart || !dragRect) return;
        dragRect.setBounds(L.latLngBounds(dragStart, e.latlng));
    });
    map.on("mouseup", function(e) {
        if (!selectMode || !dragStart || !dragRect) return;
        let b = dragRect.getBounds();
        selectByBounds(b);
        try { map.removeLayer(dragRect); } catch(err) {}
        dragRect = null; dragStart = null;
    });
    map.on("contextmenu", function(e) {
        if (Object.keys(selectedKeys).length > 0) {
            L.popup({maxWidth:290}).setLatLng(e.latlng).setContent(selectionPopupHtml()).openOn(map);
        }
    });
}

function setTileServer(name, template, lat, lon, zoom) {
    initMap("map");
    if (!template) { status("No tile template selected."); return; }
    if (tileLayer) { map.removeLayer(tileLayer); tileLayer = null; }
    tileLayer = new MustatilTileLayer(template, { maxZoom: 22, tileSize: 256, crossOrigin: true, attribution: name || "Map" });
    tileLayer.addTo(map);
    map.setView([parseFloat(lat)||24, parseFloat(lon)||45], parseInt(zoom)||6);
    status("Loaded map server: " + name + " | z=" + zoom);
}

function setImageBackground(dataUrl, width, height) {
    initMap("image");
    window.clearOverlays();
    if (imageOverlay) { try { map.removeLayer(imageOverlay); } catch(e) {} }
    if (tileLayer) { try { map.removeLayer(tileLayer); } catch(e) {} tileLayer = null; }
    let w = parseFloat(width) || 1000;
    let h = parseFloat(height) || 1000;
    let bounds = [[0,0], [h,w]];
    imageOverlay = L.imageOverlay(dataUrl, bounds).addTo(map);
    map.fitBounds(bounds);
    status("Loaded image/preview background: " + Math.round(w) + " x " + Math.round(h));
}

function setPyramidImage(tileUrl, width, height, minZoom, maxZoom) {
    initMap("image");
    window.clearOverlays();
    if (imageOverlay) { try { map.removeLayer(imageOverlay); } catch(e) {} imageOverlay = null; }
    if (tileLayer) { try { map.removeLayer(tileLayer); } catch(e) {} tileLayer = null; }

    let w = parseFloat(width) || 1000;
    let h = parseFloat(height) || 1000;
    let bounds = [[0,0], [h,w]];

    tileLayer = L.tileLayer(tileUrl, {
        tileSize: 256,
        minZoom: parseInt(minZoom),
        maxZoom: parseInt(maxZoom),
        noWrap: true,
        bounds: bounds,
        updateWhenIdle: false,
        updateWhenZooming: true,
        keepBuffer: 4,
        crossOrigin: true
    });
    tileLayer.addTo(map);
    map.setMaxBounds(bounds);
    map.fitBounds(bounds);
    status("Loaded smooth image pyramid: " + Math.round(w) + " x " + Math.round(h));
}

function statusToStyle(statusText, selected) {
    let s = String(statusText || "").toLowerCase().trim();
    let st = {color:"#ff0000", weight:2, opacity:1.0, fillColor:"#ff0000", fillOpacity:0.06};
    if (["correct","positive","true_positive","tp"].includes(s)) {
        st = {color:"#00ff00", weight:3, opacity:1.0, fillColor:"#00ff00", fillOpacity:0.06};
    } else if (["false_positive","false positive","fp","hidden"].includes(s)) {
        st = {color:"#888888", weight:2, opacity:0.90, fillColor:"#888888", fillOpacity:0.04};
    } else if (["uncertain","maybe","unknown","needs_review","review"].includes(s)) {
        st = {color:"#ff9900", weight:3, opacity:1.0, fillColor:"#ff9900", fillOpacity:0.07};
    } else if (["confirmed_archaeology","confirmed","archaeology"].includes(s)) {
        st = {color:"#008cff", weight:3, opacity:1.0, fillColor:"#008cff", fillOpacity:0.06};
    }
    if (selected) { st.color = "#00ffff"; st.weight = 4; st.fillOpacity = 0.18; }
    return st;
}

function featureKey(layerIdx, rowIdx) { return String(layerIdx) + ":" + String(rowIdx); }

function callBridge(method, layerIdx, rowIdx, value) {
    try {
        if (!bridge || !bridge[method]) return;
        if (value === undefined) bridge[method](parseInt(layerIdx), parseInt(rowIdx));
        else bridge[method](parseInt(layerIdx), parseInt(rowIdx), String(value));
    } catch(e) { console.log("Bridge call failed", method, e); }
}

function callBridgeBulk(method, keys, value) {
    try {
        if (!bridge || !bridge[method]) return;
        if (value === undefined) bridge[method](JSON.stringify(keys));
        else bridge[method](JSON.stringify(keys), String(value));
    } catch(e) { console.log("Bridge bulk call failed", method, e); }
}

function setSelected(layerIdx, rowIdx, selected) {
    let key = featureKey(layerIdx, rowIdx);
    let lyr = featureIndex[key];
    if (!lyr) return;
    if (selected) selectedKeys[key] = true; else delete selectedKeys[key];
    try {
        let statusText = lyr.feature.properties.review_status || "";
        lyr.setStyle(statusToStyle(statusText, !!selectedKeys[key]));
    } catch(e) {}
}

function clearSelection() {
    Object.keys(selectedKeys).forEach(function(k) {
        let parts = k.split(":");
        setSelected(parseInt(parts[0]), parseInt(parts[1]), false);
    });
    selectedKeys = {};
    status("Selection cleared.");
}

function toggleSelected(layerIdx, rowIdx) {
    let key = featureKey(layerIdx,rowIdx);
    setSelected(layerIdx, rowIdx, !selectedKeys[key]);
}

function selectOnly(layerIdx, rowIdx) {
    clearSelection();
    setSelected(layerIdx, rowIdx, true);
}

function selectByBounds(bounds) {
    let count = 0;
    Object.keys(featureIndex).forEach(function(k) {
        let lyr = featureIndex[k];
        try {
            let b = lyr.getBounds ? lyr.getBounds() : null;
            if (b && b.isValid() && bounds.intersects(b)) {
                let parts = k.split(":");
                setSelected(parseInt(parts[0]), parseInt(parts[1]), true);
                count++;
            }
        } catch(e) {}
    });
    status("Selected " + count + " detection(s). Right-click to apply actions.");
}

function selectedKeyList() { return Object.keys(selectedKeys); }

function markLocal(layerIdx, rowIdx, statusText) {
    let key = featureKey(layerIdx, rowIdx);
    let lyr = featureIndex[key];
    if (!lyr) return;
    try {
        lyr.feature.properties.review_status = statusText;
        lyr.setStyle(statusToStyle(statusText, !!selectedKeys[key]));
        if (statusText === "hidden") { try { lyr.remove(); } catch(e) {} }
    } catch(e) {}
}

function markSelectedLocal(statusText) {
    selectedKeyList().forEach(function(k) {
        let p = k.split(":");
        markLocal(parseInt(p[0]), parseInt(p[1]), statusText);
    });
}

function featurePopupHtml(layerIdx, rowIdx) {
    return `
    <div class="feature-popup">
      <b>Detection ${layerIdx}:${rowIdx}</b><br>
      <button onclick="selectOnly(${layerIdx},${rowIdx})">Select only this feature</button>
      <button onclick="toggleSelected(${layerIdx},${rowIdx})">Toggle selection</button>
      <hr>
      <button onclick="markAction(${layerIdx},${rowIdx},'correct')">Mark correct / positive</button>
      <button onclick="markAction(${layerIdx},${rowIdx},'false_positive')">Mark false_positive</button>
      <button onclick="markAction(${layerIdx},${rowIdx},'uncertain')">Mark uncertain</button>
      <button onclick="markAction(${layerIdx},${rowIdx},'needs_review')">Mark needs_review</button>
      <button onclick="markAction(${layerIdx},${rowIdx},'confirmed_archaeology')">Mark confirmed_archaeology</button>
      <hr>
      <button onclick="setClassAction(${layerIdx},${rowIdx})">Set class/name...</button>
      <button onclick="markAction(${layerIdx},${rowIdx},'hidden')">Hide feature</button>
      <button onclick="deleteAction(${layerIdx},${rowIdx})">Delete feature</button>
      <hr>
      <button onclick="copyMoveAction(${layerIdx},${rowIdx},true)">Copy to review layer</button>
      <button onclick="copyMoveAction(${layerIdx},${rowIdx},false)">Move to review layer</button>
      <button onclick="exportAction(${layerIdx},${rowIdx})">Export this feature</button>
      <button onclick="zoomToFeature(${layerIdx},${rowIdx},true)">Zoom to feature</button>
      <button onclick="copyCenterAction(${layerIdx},${rowIdx})">Copy center coordinates</button>
      <button onclick="copyWktAction(${layerIdx},${rowIdx})">Copy geometry WKT</button>
    </div>`;
}

function selectionPopupHtml() {
    let n = selectedKeyList().length;
    return `
    <div class="feature-popup">
      <b>Selected detections: ${n}</b><br>
      <button onclick="bulkMark('correct')">Mark selected correct / positive</button>
      <button onclick="bulkMark('false_positive')">Mark selected false_positive</button>
      <button onclick="bulkMark('uncertain')">Mark selected uncertain</button>
      <button onclick="bulkMark('needs_review')">Mark selected needs_review</button>
      <button onclick="bulkMark('confirmed_archaeology')">Mark selected confirmed_archaeology</button>
      <hr>
      <button onclick="bulkCopyMove(true)">Copy selected to review layer</button>
      <button onclick="bulkCopyMove(false)">Move selected to review layer</button>
      <button onclick="bulkExport()">Export selected</button>
      <button onclick="bulkDelete()">Delete selected</button>
      <hr>
      <button onclick="clearSelection()">Clear selection</button>
    </div>`;
}

function markAction(layerIdx, rowIdx, statusText) { markLocal(layerIdx, rowIdx, statusText); callBridge("markFeature", layerIdx, rowIdx, statusText); }
function deleteAction(layerIdx, rowIdx) {
    let key = featureKey(layerIdx,rowIdx);
    let lyr = featureIndex[key];
    if (lyr) { try { lyr.remove(); } catch(e) {} delete featureIndex[key]; }
    delete selectedKeys[key];
    callBridge("deleteFeature", layerIdx, rowIdx);
}
function setClassAction(layerIdx, rowIdx) {
    let v = prompt("Class / name:", "");
    if (v !== null && v.length > 0) callBridge("setClassName", layerIdx, rowIdx, v);
}
function copyMoveAction(layerIdx, rowIdx, copyFlag) {
    callBridge(copyFlag ? "copyFeatureToReview" : "moveFeatureToReview", layerIdx, rowIdx);
    if (!copyFlag) {
        let key = featureKey(layerIdx,rowIdx);
        let lyr = featureIndex[key];
        if (lyr) { try { lyr.remove(); } catch(e) {} delete featureIndex[key]; }
        delete selectedKeys[key];
    }
}
function exportAction(layerIdx, rowIdx) { callBridge("exportFeature", layerIdx, rowIdx); }
function copyCenterAction(layerIdx, rowIdx) { callBridge("copyCenter", layerIdx, rowIdx); }
function copyWktAction(layerIdx, rowIdx) { callBridge("copyWkt", layerIdx, rowIdx); }
function bulkMark(statusText) { let keys = selectedKeyList(); markSelectedLocal(statusText); callBridgeBulk("bulkMark", keys, statusText); }
function bulkDelete() {
    let keys = selectedKeyList();
    keys.forEach(function(k) { let lyr = featureIndex[k]; if (lyr) { try { lyr.remove(); } catch(e) {} delete featureIndex[k]; } });
    selectedKeys = {};
    callBridgeBulk("bulkDelete", keys);
}
function bulkCopyMove(copyFlag) {
    let keys = selectedKeyList();
    callBridgeBulk(copyFlag ? "bulkCopyToReview" : "bulkMoveToReview", keys);
    if (!copyFlag) {
        keys.forEach(function(k) { let lyr = featureIndex[k]; if (lyr) { try { lyr.remove(); } catch(e) {} delete featureIndex[k]; } });
        selectedKeys = {};
    }
}
function bulkExport() { callBridgeBulk("bulkExport", selectedKeyList()); }

function addLayerGeoJSON(layerIdx, layerName, geojson, fit) {
    initMap(mapMode || "map");
    if (layerGroups[layerIdx]) {
        map.removeLayer(layerGroups[layerIdx]);
        delete layerGroups[layerIdx];
    }
    let group = L.geoJSON(geojson, {
        style: function(feature) {
            let p = feature.properties || {};
            let key = featureKey(p.__layer, p.__row);
            return statusToStyle(p.review_status || "", !!selectedKeys[key]);
        },
        onEachFeature: function(feature, layer) {
            let props = feature.properties || {};
            let li = parseInt(props.__layer);
            let ri = parseInt(props.__row);
            let key = featureKey(li, ri);
            featureIndex[key] = layer;

            layer.on("click", function(e) {
                try {
                    if (e.originalEvent && (e.originalEvent.ctrlKey || e.originalEvent.shiftKey)) toggleSelected(li, ri);
                    else selectOnly(li, ri);
                    callBridge("featureClicked", li, ri);
                } catch(err) {}
            });

            layer.on("contextmenu", function(e) {
                try {
                    if (!selectedKeys[key]) selectOnly(li, ri);
                    let html = selectedKeyList().length > 1 ? selectionPopupHtml() : featurePopupHtml(li, ri);
                    L.popup({maxWidth: 290}).setLatLng(e.latlng).setContent(html).openOn(map);
                    callBridge("featureClicked", li, ri);
                } catch(err) {}
            });
        }
    });
    group.addTo(map);
    layerGroups[layerIdx] = group;
    if (fit) {
        try { let b = group.getBounds(); if (b && b.isValid()) map.fitBounds(b.pad(0.15)); } catch(e) {}
    }
    status("Loaded detection overlay: " + layerName + " | features=" + (geojson.features ? geojson.features.length : 0));
}

function removeLayer(layerIdx) {
    if (layerGroups[layerIdx]) {
        map.removeLayer(layerGroups[layerIdx]);
        delete layerGroups[layerIdx];
    }
    Object.keys(featureIndex).forEach(function(k) { if (k.startsWith(String(layerIdx) + ":")) delete featureIndex[k]; });
    Object.keys(selectedKeys).forEach(function(k) { if (k.startsWith(String(layerIdx) + ":")) delete selectedKeys[k]; });
}
function clearOverlays() {
    Object.keys(layerGroups).forEach(function(k) { try { map.removeLayer(layerGroups[k]); } catch(e) {} });
    layerGroups = {}; featureIndex = {}; selectedKeys = {};
}
function zoomToFeature(layerIdx, rowIdx, openPopup) {
    let lyr = featureIndex[featureKey(layerIdx,rowIdx)];
    if (!lyr) return;
    try {
        if (lyr.getBounds) map.fitBounds(lyr.getBounds().pad(0.4));
        else if (lyr.getLatLng) map.setView(lyr.getLatLng(), Math.max(map.getZoom(), 17));
        if (openPopup) {
            let center = lyr.getBounds ? lyr.getBounds().getCenter() : lyr.getLatLng();
            L.popup({maxWidth:290}).setLatLng(center).setContent(featurePopupHtml(layerIdx,rowIdx)).openOn(map);
        }
    } catch(e) {}
}
function fitAllOverlays() {
    let bounds = null;
    Object.keys(layerGroups).forEach(function(k) {
        try {
            let b = layerGroups[k].getBounds();
            if (b && b.isValid()) { if (bounds === null) bounds = b; else bounds.extend(b); }
        } catch(e) {}
    });
    if (bounds !== null) map.fitBounds(bounds.pad(0.15));
}

window.setTileServer = setTileServer;
window.setImageBackground = setImageBackground;
window.setPyramidImage = setPyramidImage;
window.addLayerGeoJSON = addLayerGeoJSON;
window.removeLayer = removeLayer;
window.clearOverlays = clearOverlays;
window.fitAllOverlays = fitAllOverlays;
window.setSelectionMode = setSelectionMode;
window.clearSelection = clearSelection;
window.selectOnly = selectOnly;
window.toggleSelected = toggleSelected;
window.markAction = markAction;
window.deleteAction = deleteAction;
window.setClassAction = setClassAction;
window.copyMoveAction = copyMoveAction;
window.exportAction = exportAction;
window.copyCenterAction = copyCenterAction;
window.copyWktAction = copyWktAction;
window.bulkMark = bulkMark;
window.bulkDelete = bulkDelete;
window.bulkCopyMove = bulkCopyMove;
window.bulkExport = bulkExport;
window.zoomToFeature = zoomToFeature;

document.addEventListener("DOMContentLoaded", function() {
    initMap("map");
    if (typeof QWebChannel !== "undefined" && typeof qt !== "undefined") {
        new QWebChannel(qt.webChannelTransport, function(channel) {
            bridge = channel.objects.mustatilBridge;
            status("Qt bridge connected. Map ready.");
        });
    } else {
        status("Map ready. Qt WebChannel not available; visual mode only.");
    }
});
</script>
</body>
</html>
"""


def _make_bridge_class():
    from PySide6.QtCore import QObject, Slot

    class Bridge(QObject):
        def __init__(self, manager):
            super().__init__()
            self.manager = manager

        @Slot(str)
        def log(self, msg):
            self.manager.set_info(str(msg))

        @Slot(int, int)
        def featureClicked(self, layer_idx, row_idx):
            self.manager.select_feature(int(layer_idx), int(row_idx), from_map=True)

        @Slot(int, int, str)
        def markFeature(self, layer_idx, row_idx, status):
            self.manager.mark_feature(int(layer_idx), int(row_idx), str(status), refresh_map=False)

        @Slot(int, int)
        def deleteFeature(self, layer_idx, row_idx):
            self.manager.delete_feature(int(layer_idx), int(row_idx), refresh_map=False)

        @Slot(int, int, str)
        def setClassName(self, layer_idx, row_idx, name):
            self.manager.set_class_name(int(layer_idx), int(row_idx), str(name))

        @Slot(int, int)
        def copyFeatureToReview(self, layer_idx, row_idx):
            self.manager.copy_or_move_feature(int(layer_idx), int(row_idx), copy=True)

        @Slot(int, int)
        def moveFeatureToReview(self, layer_idx, row_idx):
            self.manager.copy_or_move_feature(int(layer_idx), int(row_idx), copy=False, refresh_map=False)

        @Slot(int, int)
        def exportFeature(self, layer_idx, row_idx):
            self.manager.export_feature_dialog(int(layer_idx), int(row_idx))

        @Slot(int, int)
        def copyCenter(self, layer_idx, row_idx):
            self.manager.copy_center(int(layer_idx), int(row_idx))

        @Slot(int, int)
        def copyWkt(self, layer_idx, row_idx):
            self.manager.copy_wkt(int(layer_idx), int(row_idx))

        @Slot(str, str)
        def bulkMark(self, keys_json, status):
            self.manager.bulk_mark(json.loads(keys_json or "[]"), str(status))

        @Slot(str)
        def bulkDelete(self, keys_json):
            self.manager.bulk_delete(json.loads(keys_json or "[]"))

        @Slot(str)
        def bulkCopyToReview(self, keys_json):
            self.manager.bulk_copy_move(json.loads(keys_json or "[]"), copy=True)

        @Slot(str)
        def bulkMoveToReview(self, keys_json):
            self.manager.bulk_copy_move(json.loads(keys_json or "[]"), copy=False)

        @Slot(str)
        def bulkExport(self, keys_json):
            self.manager.bulk_export(json.loads(keys_json or "[]"))

    return Bridge




class ReviewGraphicsRectItem:
    pass


def _qcolor_for_status(status: str):
    from PySide6.QtGui import QColor
    s = str(status or "").lower().strip()
    if s in {"correct", "positive", "true_positive", "tp"}:
        return QColor("#00ff00")
    if s in {"false_positive", "false positive", "fp", "hidden"}:
        return QColor("#888888")
    if s in {"uncertain", "maybe", "unknown", "needs_review", "review"}:
        return QColor("#ff9900")
    if s in {"confirmed_archaeology", "confirmed", "archaeology"}:
        return QColor("#008cff")
    return QColor("#ff0000")


def _make_detection_like_canvas_classes():
    from PySide6.QtCore import Qt, QPointF, QRect, QRectF, QSize, Signal
    from PySide6.QtGui import QBrush, QColor, QPen, QImage, QPixmap, QTransform, QPainter, QPainter
    from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsRectItem, QMenu, QRubberBand, QWidget, QWidget

    class FeatureRectItem(QGraphicsRectItem):
        def __init__(self, manager, layer_idx: int, row_idx: int, rect, status: str = ""):
            super().__init__(*rect)
            self.manager = manager
            self.layer_idx = int(layer_idx)
            self.row_idx = int(row_idx)
            self.status = str(status or "")
            self.setAcceptHoverEvents(True)
            self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
            self.setBrush(QBrush(QColor(255, 0, 0, 22)))
            try:
                self.setOpacity(1.0)
                self.setVisible(True)
            except Exception:
                pass
            self.apply_style(False)

        def apply_style(self, selected: bool = False):
            """
            v14 visibility fix:
            The TIFF pixmap is a transformed QGraphicsPixmapItem. Rectangles
            must be drawn with a cosmetic pen and a very high z-value, otherwise
            they can be clickable but visually disappear behind/inside the raster
            at certain zoom levels.
            """
            col = QColor("#00ffff") if selected else _qcolor_for_status(self.status)
            pen = QPen(col)
            pen.setWidth(4 if selected else 3)
            try:
                pen.setCosmetic(True)
            except Exception:
                pass
            self.setPen(pen)

            try:
                fill = QColor(col)
                fill.setAlpha(55 if selected else 22)
                self.setBrush(QBrush(fill))
            except Exception:
                pass

            try:
                self.setZValue(1000000 if selected else 999000)
            except Exception:
                pass

        def mousePressEvent(self, event):
            try:
                if event.button() == Qt.LeftButton:
                    self.manager.select_feature(self.layer_idx, self.row_idx, from_map=True)
                    self.manager._image_select_only(self.layer_idx, self.row_idx)
                    event.accept()
                    return
            except Exception:
                pass
            super().mousePressEvent(event)

        def contextMenuEvent(self, event):
            try:
                key = f"{int(self.layer_idx)}:{int(self.row_idx)}"
                selected = getattr(self.manager, "image_selected_keys", set())
                if key in selected and len(selected) > 1 and hasattr(self.manager, "_image_batch_menu"):
                    self.manager._image_batch_menu(event.screenPos())
                    event.accept()
                    return

                self.manager.select_feature(self.layer_idx, self.row_idx, from_map=True)
                self.manager._image_select_only(self.layer_idx, self.row_idx)
                menu = QMenu()
                act_pos = menu.addAction("Mark correct / positive")
                act_fp = menu.addAction("Mark false_positive")
                act_unc = menu.addAction("Mark uncertain")
                act_need = menu.addAction("Mark needs_review")
                act_conf = menu.addAction("Mark confirmed_archaeology")
                menu.addSeparator()
                act_del = menu.addAction("Delete feature")
                act_export = menu.addAction("Export this feature")
                act = menu.exec(event.screenPos())
                if act == act_pos:
                    self.manager.mark_feature(self.layer_idx, self.row_idx, "correct")
                elif act == act_fp:
                    self.manager.mark_feature(self.layer_idx, self.row_idx, "false_positive")
                elif act == act_unc:
                    self.manager.mark_feature(self.layer_idx, self.row_idx, "uncertain")
                elif act == act_need:
                    self.manager.mark_feature(self.layer_idx, self.row_idx, "needs_review")
                elif act == act_conf:
                    self.manager.mark_feature(self.layer_idx, self.row_idx, "confirmed_archaeology")
                elif act == act_del:
                    self.manager.delete_feature(self.layer_idx, self.row_idx)
                elif act == act_export:
                    self.manager.export_feature_dialog(self.layer_idx, self.row_idx)
                event.accept()
            except Exception:
                super().contextMenuEvent(event)

    class SelectionOverlay(QWidget):
        """
        v20: hard-visible right-drag selection rectangle.

        This is a real child widget over the QGraphicsView viewport, not only
        a QGraphicsScene item, so the TIFF pixmap cannot cover it.
        """
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            self.setAttribute(Qt.WA_NoSystemBackground, False)
            self.setAttribute(Qt.WA_StyledBackground, False)
            self.hide()

        def sizeHint(self):
            return QSize(32, 32)

        def paintEvent(self, event):
            try:
                p = QPainter(self)
                p.setRenderHint(QPainter.Antialiasing, False)
                fill = QColor("#00ffff")
                fill.setAlpha(45)
                pen = QPen(QColor("#00ffff"))
                pen.setWidth(3)
                try:
                    pen.setCosmetic(True)
                except Exception:
                    pass
                pen.setStyle(Qt.DashLine)
                p.fillRect(self.rect(), fill)
                p.setPen(pen)
                p.drawRect(self.rect().adjusted(1, 1, -2, -2))
                p.end()
            except Exception:
                pass

    class DetectionLikeImageCanvas(QGraphicsView):
        """
        Detection Preview style canvas:
          - QGraphicsView
          - AnchorUnderMouse wheel zoom
          - left mouse drag pans
          - no Leaflet image zoom
        """
        zoom_pan_changed = Signal()

        def __init__(self, parent=None):
            super().__init__(parent)
            self._scene = QGraphicsScene(self)
            self.setScene(self._scene)
            self.pixmap_item = None
            self.overlay_items = []
            self.manager = None
            self._panning = False
            self._pan_start_view = None
            self._pan_start_center = None
            self._selecting = False
            self._select_start = None
            self._select_start_view = None
            self._select_press_item = None
            self._select_rect_item = None
            self._select_rubber_band = None
            self._select_overlay = None
            self._select_moved = False
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
            self.setDragMode(QGraphicsView.NoDrag)
            self.setBackgroundBrush(QBrush(QColor("#f4f4f4")))

        def set_manager(self, manager):
            self.manager = manager

        def set_qpixmap(self, pixmap, fit: bool = True):
            self._scene.clear()
            self.overlay_items = []
            self.pixmap_item = self._scene.addPixmap(pixmap.copy())
            self.pixmap_item.setPos(0, 0)
            self.pixmap_item.setZValue(-100000)
            self._scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
            if fit:
                self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

        def set_pil_image_source_window(self, pil_img, orig_w: int, orig_h: int, source_window, fit: bool = False):
            """
            Same concept as Mustatil Detection Preview:
            scene coordinates stay in ORIGINAL IMAGE PIXELS.
            The currently loaded preview/viewport pixmap is placed at the
            source-window position and scaled to that source-window size.
            """
            if pil_img is None:
                return
            try:
                x1, y1, x2, y2 = map(float, source_window)
            except Exception:
                x1, y1, x2, y2 = 0.0, 0.0, float(orig_w), float(orig_h)

            x1 = max(0.0, min(float(orig_w), x1))
            y1 = max(0.0, min(float(orig_h), y1))
            x2 = max(0.0, min(float(orig_w), x2))
            y2 = max(0.0, min(float(orig_h), y2))
            if x2 <= x1:
                x2 = min(float(orig_w), x1 + 1.0)
            if y2 <= y1:
                y2 = min(float(orig_h), y1 + 1.0)

            old_transform = None if fit else self.transform()
            old_center = None
            if not fit:
                try:
                    old_center = self.mapToScene(self.viewport().rect().center())
                except Exception:
                    old_center = None

            im = pil_img.convert("RGBA")
            data = im.tobytes("raw", "RGBA")
            qimg = QImage(data, im.width, im.height, QImage.Format_RGBA8888)
            pixmap = QPixmap.fromImage(qimg.copy())

            self._scene.clear()
            self.overlay_items = []
            self.pixmap_item = self._scene.addPixmap(pixmap)
            self.pixmap_item.setPos(x1, y1)
            sx = (x2 - x1) / max(1.0, float(pixmap.width()))
            sy = (y2 - y1) / max(1.0, float(pixmap.height()))
            self.pixmap_item.setTransform(QTransform().scale(sx, sy))
            self.pixmap_item.setZValue(-100000)
            self._scene.setSceneRect(0, 0, max(1, int(orig_w)), max(1, int(orig_h)))

            if fit:
                self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
            elif old_transform is not None:
                try:
                    self.setTransform(old_transform)
                    if old_center is not None:
                        self.centerOn(old_center)
                except Exception:
                    pass

        def clear_overlays(self):
            for it in list(self.overlay_items):
                try:
                    self._scene.removeItem(it)
                except Exception:
                    pass
            self.overlay_items = []

        def add_feature_rect(self, manager, layer_idx, row_idx, x1, y1, x2, y2, status=""):
            try:
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                if x2 < x1: x1, x2 = x2, x1
                if y2 < y1: y1, y2 = y2, y1
                item = FeatureRectItem(manager, int(layer_idx), int(row_idx), (x1, y1, max(1.0, x2-x1), max(1.0, y2-y1)), status)
                item.setZValue(999000)
                self._scene.addItem(item)
                try:
                    item.setZValue(999000)
                    item.setVisible(True)
                    item.setOpacity(1.0)
                except Exception:
                    pass
                self.overlay_items.append(item)
                return item
            except Exception:
                return None

        def wheelEvent(self, event):
            factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
            self.scale(factor, factor)
            self.zoom_pan_changed.emit()
            event.accept()

        def mousePressEvent(self, event):
            if self.pixmap_item is not None and event.button() == Qt.RightButton:
                # v17: always show a selection box immediately. It is drawn both
                # in scene coordinates and as a real viewport QRubberBand.
                item = self.itemAt(event.position().toPoint())
                self._select_press_item = item if (item is not None and item is not self.pixmap_item and hasattr(item, "layer_idx")) else None
                self._selecting = True
                self._select_start = self.mapToScene(event.position().toPoint())
                self._select_start_view = event.position().toPoint()
                self._select_moved = False

                if self._select_rect_item is not None:
                    try:
                        self._scene.removeItem(self._select_rect_item)
                    except Exception:
                        pass
                    self._select_rect_item = None

                pen = QPen(QColor("#00ffff"))
                pen.setWidth(3)
                try:
                    pen.setCosmetic(True)
                except Exception:
                    pass
                pen.setStyle(Qt.DashLine)
                fill = QColor("#00ffff")
                fill.setAlpha(45)
                self._select_rect_item = self._scene.addRect(QRectF(self._select_start, self._select_start), pen, QBrush(fill))
                self._select_rect_item.setZValue(2500000)
                try:
                    self._select_rect_item.setVisible(True)
                    self._select_rect_item.setOpacity(1.0)
                except Exception:
                    pass

                try:
                    if self._select_rubber_band is None:
                        self._select_rubber_band = QRubberBand(QRubberBand.Rectangle, self.viewport())
                    self._select_rubber_band.setGeometry(QRect(self._select_start_view, self._select_start_view).normalized().adjusted(0, 0, 1, 1))
                    self._select_rubber_band.show()
                    self._select_rubber_band.raise_()
                except Exception:
                    pass

                # v20: custom-painted overlay, always above the viewport.
                try:
                    if self._select_overlay is None:
                        self._select_overlay = SelectionOverlay(self.viewport())
                    self._select_overlay.setGeometry(QRect(self._select_start_view, self._select_start_view).normalized().adjusted(0, 0, 1, 1))
                    self._select_overlay.show()
                    self._select_overlay.raise_()
                    self._select_overlay.update()
                except Exception:
                    pass

                event.accept()
                return

            if event.button() == Qt.LeftButton and self.pixmap_item is not None:
                item = self.itemAt(event.position().toPoint())
                if item is not None and item is not self.pixmap_item:
                    super().mousePressEvent(event)
                    return
                self._panning = True
                self._pan_start_view = event.position()
                self._pan_start_center = self.mapToScene(self.viewport().rect().center())
                self.setCursor(Qt.ClosedHandCursor)
                event.accept()
                return
            super().mousePressEvent(event)

        def mouseMoveEvent(self, event):
            if self._selecting and self._select_start is not None:
                cur_view = event.position().toPoint()
                cur = self.mapToScene(cur_view)
                rect = QRectF(self._select_start, cur).normalized()
                if rect.width() > 4 or rect.height() > 4:
                    self._select_moved = True
                if self._select_rect_item is not None:
                    self._select_rect_item.setRect(rect)
                    try:
                        self._select_rect_item.setZValue(2500000)
                    except Exception:
                        pass
                try:
                    if self._select_rubber_band is not None and self._select_start_view is not None:
                        self._select_rubber_band.setGeometry(QRect(self._select_start_view, cur_view).normalized().adjusted(0, 0, 1, 1))
                        self._select_rubber_band.show()
                        self._select_rubber_band.raise_()
                except Exception:
                    pass
                try:
                    if self._select_overlay is not None and self._select_start_view is not None:
                        self._select_overlay.setGeometry(QRect(self._select_start_view, cur_view).normalized().adjusted(0, 0, 1, 1))
                        self._select_overlay.show()
                        self._select_overlay.raise_()
                        self._select_overlay.update()
                except Exception:
                    pass
                event.accept()
                return

            if self._panning and self._pan_start_view is not None and self._pan_start_center is not None:
                dx = float(event.position().x() - self._pan_start_view.x())
                dy = float(event.position().y() - self._pan_start_view.y())
                sx = float(self.transform().m11()) or 1.0
                sy = float(self.transform().m22()) or 1.0
                self.centerOn(QPointF(float(self._pan_start_center.x()) - dx / sx,
                                      float(self._pan_start_center.y()) - dy / sy))
                event.accept()
                return
            super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event):
            if self._selecting and event.button() == Qt.RightButton:
                cur = self.mapToScene(event.position().toPoint())
                rect = QRectF(self._select_start, cur).normalized() if self._select_start is not None else QRectF()
                moved = bool(self._select_moved) and (rect.width() > 4 or rect.height() > 4)

                if self._select_rect_item is not None:
                    try:
                        self._scene.removeItem(self._select_rect_item)
                    except Exception:
                        pass
                    self._select_rect_item = None
                try:
                    if self._select_rubber_band is not None:
                        self._select_rubber_band.hide()
                except Exception:
                    pass
                try:
                    if self._select_overlay is not None:
                        self._select_overlay.hide()
                except Exception:
                    pass

                press_item = self._select_press_item
                self._selecting = False
                self._select_start = None
                self._select_start_view = None
                self._select_press_item = None
                self._select_moved = False

                if moved and self.manager is not None and hasattr(self.manager, "_image_select_by_rect"):
                    additive = bool(event.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier))
                    self.manager._image_select_by_rect(rect, additive=additive)
                elif self.manager is not None:
                    try:
                        gp = event.globalPosition().toPoint()
                    except Exception:
                        gp = event.globalPos()

                    if press_item is not None and hasattr(press_item, "layer_idx"):
                        key = f"{int(press_item.layer_idx)}:{int(press_item.row_idx)}"
                        selected = getattr(self.manager, "image_selected_keys", set())
                        if key in selected and len(selected) > 1 and hasattr(self.manager, "_image_batch_menu"):
                            self.manager._image_batch_menu(gp)
                        elif hasattr(self.manager, "_image_single_menu"):
                            self.manager._image_single_menu(int(press_item.layer_idx), int(press_item.row_idx), gp)
                        else:
                            self.manager._image_select_only(int(press_item.layer_idx), int(press_item.row_idx))
                    elif getattr(self.manager, "image_selected_keys", set()) and hasattr(self.manager, "_image_batch_menu"):
                        self.manager._image_batch_menu(gp)

                event.accept()
                return

            if self._panning and event.button() == Qt.LeftButton:
                self._panning = False
                self._pan_start_view = None
                self._pan_start_center = None
                self.setCursor(Qt.ArrowCursor)
                self.zoom_pan_changed.emit()
                event.accept()
                return
            super().mouseReleaseEvent(event)

    return DetectionLikeImageCanvas, FeatureRectItem


class GMTExportDialog:
    """
    Small GMT export settings window.

    Stores settings in the Mustatil project folder as:
      mustatil_gmt_map_settings.json

    The map output defaults to:
      <project>/gmt_maps/
    """
    def __init__(self, parent, manager, default_title: str = "Mustatil GIS Review Map"):
        from PySide6.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QLineEdit,
            QPushButton, QComboBox, QSpinBox, QCheckBox, QFileDialog, QDialogButtonBox,
            QPlainTextEdit
        )

        self.manager = manager
        self.dialog = QDialog(parent)
        self.dialog.setWindowTitle("GMT Map Export")
        self.dialog.resize(560, 300)

        root = QVBoxLayout(self.dialog)
        info = QLabel("Create a publication-style GMT/map export and save the settings in the project.")
        info.setWordWrap(True)
        root.addWidget(info)

        form = QFormLayout()
        root.addLayout(form)

        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "PDF", "SVG"])
        form.addRow("Format:", self.format_combo)

        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(220)
        form.addRow("DPI:", self.dpi_spin)

        self.title_edit = QLineEdit(default_title)
        form.addRow("Title:", self.title_edit)

        self.subtitle_edit = QLineEdit("")
        self.subtitle_edit.setPlaceholderText("Optional subtitle / project note")
        form.addRow("Subtitle:", self.subtitle_edit)

        self.use_basemap = QCheckBox("Use selected map server as basemap")
        self.use_basemap.setChecked(True)
        form.addRow("Basemap:", self.use_basemap)

        self.draw_grid = QCheckBox("Draw coordinate grid")
        self.draw_grid.setChecked(True)
        form.addRow("Grid:", self.draw_grid)

        self.draw_attribution = QCheckBox("Write basemap attribution")
        self.draw_attribution.setChecked(True)
        form.addRow("Attribution:", self.draw_attribution)

        self.save_project = QCheckBox("Save settings and output inside project")
        self.save_project.setChecked(True)
        form.addRow("Project:", self.save_project)

        outrow = QHBoxLayout()
        self.out_dir_edit = QLineEdit(str(manager.default_gmt_output_dir()))
        self.browse_btn = QPushButton("Browse...")
        outrow.addWidget(self.out_dir_edit, 1)
        outrow.addWidget(self.browse_btn)
        form.addRow("Output folder:", outrow)

        self.console_edit = QPlainTextEdit()
        self.console_edit.setPlaceholderText(
            "Executable GMT console. Output, errors and exit codes are printed here."
        )
        self.console_edit.setMaximumHeight(160)
        form.addRow("GMT console:", self.console_edit)

        self.console_cmd = QLineEdit()
        self.console_cmd.setPlaceholderText("gmt --version")
        self.console_cmd.setText("gmt --version")
        self.console_cmd.returnPressed.connect(self.run_console_command)
        self.btn_console_run = QPushButton("Run")
        self.btn_console_run.clicked.connect(self.run_console_command)
        self.btn_console_clear = QPushButton("Clear")
        self.btn_console_clear.clicked.connect(lambda: self.console_edit.clear())

        console_row = QHBoxLayout()
        console_row.addWidget(QLabel("Command:"))
        console_row.addWidget(self.console_cmd, 1)
        console_row.addWidget(self.btn_console_run)
        console_row.addWidget(self.btn_console_clear)
        form.addRow("Run:", console_row)

        btnrow = QHBoxLayout()
        self.save_settings_btn = QPushButton("Save Settings")
        self.create_btn = QPushButton("Create Map")
        self.cancel_btn = QPushButton("Cancel")
        btnrow.addStretch(1)
        btnrow.addWidget(self.save_settings_btn)
        btnrow.addWidget(self.create_btn)
        btnrow.addWidget(self.cancel_btn)
        root.addLayout(btnrow)

        self.browse_btn.clicked.connect(self.browse)
        self.save_settings_btn.clicked.connect(self.save_settings_only)
        self.create_btn.clicked.connect(self.dialog.accept)
        self.cancel_btn.clicked.connect(self.dialog.reject)

        self.apply_settings(manager.load_gmt_settings())

    def browse(self):
        from PySide6.QtWidgets import QFileDialog
        d = QFileDialog.getExistingDirectory(self.dialog, "Choose output folder", self.out_dir_edit.text())
        if d:
            self.out_dir_edit.setText(d)

    def run_console_command(self):
        """
        v22: Execute a GMT/shell command from the GMT console.
        """
        try:
            cmd = str(self.console_cmd.text() or "").strip()
            if not cmd:
                return
            cwd = str(self.out_dir_edit.text() or "").strip()
            if not cwd:
                try:
                    cwd = str(self.manager.default_gmt_output_dir())
                except Exception:
                    cwd = str(Path.home())
            try:
                Path(cwd).mkdir(parents=True, exist_ok=True)
            except Exception:
                cwd = str(Path.home())

            self.console_edit.appendPlainText("")
            self.console_edit.appendPlainText(f"> {cmd}")
            self.console_edit.appendPlainText(f"[cwd] {cwd}")
            p = subprocess.run(cmd, cwd=cwd, shell=True, text=True, capture_output=True, timeout=600, env=os.environ.copy())
            if p.stdout:
                self.console_edit.appendPlainText(p.stdout.rstrip())
            if p.stderr:
                self.console_edit.appendPlainText(p.stderr.rstrip())
            self.console_edit.appendPlainText(f"[exit code] {p.returncode}")
        except subprocess.TimeoutExpired:
            self.console_edit.appendPlainText("[timeout] command exceeded 600 seconds")
        except Exception as exc:
            try:
                self.console_edit.appendPlainText(f"[error] {exc}")
            except Exception:
                pass


    def settings(self) -> dict:
        fmt = str(self.format_combo.currentText() or "PNG").lower()
        return {
            "format": fmt,
            "dpi": int(self.dpi_spin.value()),
            "title": str(self.title_edit.text() or "").strip(),
            "subtitle": str(self.subtitle_edit.text() or "").strip(),
            "use_basemap": bool(self.use_basemap.isChecked()),
            "draw_grid": bool(self.draw_grid.isChecked()),
            "draw_attribution": bool(self.draw_attribution.isChecked()),
            "save_project": bool(self.save_project.isChecked()),
            "output_dir": str(self.out_dir_edit.text() or "").strip(),
            "gmt_console": str(self.console_edit.toPlainText() or "").strip(),
        }

    def apply_settings(self, settings: dict):
        if not settings:
            return
        try:
            fmt = str(settings.get("format", "png")).upper()
            ix = self.format_combo.findText(fmt)
            if ix >= 0:
                self.format_combo.setCurrentIndex(ix)
            self.dpi_spin.setValue(int(settings.get("dpi", 220)))
            if settings.get("title"):
                self.title_edit.setText(str(settings.get("title")))
            self.subtitle_edit.setText(str(settings.get("subtitle", "")))
            self.use_basemap.setChecked(bool(settings.get("use_basemap", True)))
            self.draw_grid.setChecked(bool(settings.get("draw_grid", True)))
            self.draw_attribution.setChecked(bool(settings.get("draw_attribution", True)))
            self.save_project.setChecked(bool(settings.get("save_project", True)))
            if settings.get("output_dir"):
                self.out_dir_edit.setText(str(settings.get("output_dir")))
            self.console_edit.setPlainText(str(settings.get("gmt_console", "")))
        except Exception:
            pass

    def save_settings_only(self):
        try:
            self.manager.save_gmt_settings(self.settings())
            self.manager.set_info(
                "GMT export settings saved in project:\\n"
                + str(self.manager.gmt_settings_path())
            )
        except Exception as exc:
            self.manager.set_info(f"Could not save GMT settings:\\n{exc}")

    def exec(self):
        return self.dialog.exec()


class GISWebReviewMapTab:
    def __init__(self, workspace=None):
        from PySide6.QtWidgets import (
            QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel, QPushButton,
            QComboBox, QLineEdit, QDoubleSpinBox, QSpinBox,
            QTreeWidget, QTreeWidgetItem, QTableWidget, QTableWidgetItem,
            QTextEdit, QCheckBox, QSlider, QStackedWidget
        )
        from PySide6.QtCore import Qt, QUrl

        self.workspace = workspace or _find_workspace()
        self.layers: List[Dict[str, Any]] = []
        self.current_layer_idx: Optional[int] = None
        self.web_ok = False
        self.view_mode = "map"
        self.image_meta: Dict[str, Any] = {}
        self.image_size: Tuple[int, int] = (0, 0)
        self.formlearner_custom_model_path = ""

        # v25 undo stack for GIS review/right-click actions.
        self.undo_stack = []
        self.undo_limit = 25

        # v15/v18 smooth huge-TIFF rendering state.
        self.gis_render_busy = False
        self.gis_render_pending = False
        self.gis_render_request_id = 0
        self.gis_smooth_max_out_px = 2048
        self.gis_smooth_reload_delay_ms = 320
        self.gis_render_bus = None

        self.widget = QWidget()
        self.widget.setObjectName("mustatil_gis_web_review_map_v25")
        self.widget._mustatil_gis_web_review_map = self

        root = QVBoxLayout(self.widget)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        row1 = QHBoxLayout()
        title = QLabel("GIS Web Review Map")
        title.setStyleSheet("font-weight: bold;")
        row1.addWidget(title)

        row1.addWidget(QLabel("View:"))
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Web Map", "Image / Raster"])
        self.view_combo.setMaximumWidth(125)
        row1.addWidget(self.view_combo)

        row1.addWidget(QLabel("Server:"))
        self.server_combo = QComboBox()
        self.server_combo.addItems(list(MAP_SERVERS.keys()))
        self.server_combo.setCurrentText("Esri World Imagery")
        self.server_combo.setMinimumWidth(170)
        row1.addWidget(self.server_combo, 1)

        self.custom_url = QLineEdit()
        self.custom_url.setPlaceholderText("Custom XYZ/Bing URL: {z}/{x}/{y} or {q}/{snum}")
        self.custom_url.setMinimumWidth(250)
        self.custom_url.setVisible(False)
        row1.addWidget(self.custom_url, 2)

        self.btn_load_map = QPushButton("Load/Reload Map")
        self.btn_load_image = QPushButton("Load Image / TIFF")
        self.btn_detection_preview = QPushButton("Use Detection Preview")
        self.btn_gmt_map = QPushButton("GMT Map")
        self.btn_gmt_map.setToolTip("Create a publication-style map from the current layer using PyGMT when available, otherwise matplotlib fallback.")
        row1.addWidget(self.btn_load_map)
        self.web_fast = QCheckBox("Fast WebView")
        self.web_fast.setToolTip("Keep QtWebEngine GPU/compositing enabled for smooth satellite-map panning.")
        self.web_fast.setChecked(True)
        row1.addWidget(self.web_fast)
        row1.addWidget(self.btn_load_image)

        self.multiband_check = QCheckBox("Multiband")
        self.multiband_check.setToolTip("Use selected TIFF bands for RGB preview/viewport.")
        self.multiband_check.setChecked(False)
        row1.addWidget(self.multiband_check)

        self.multiband_bands_edit = QLineEdit("1,2,3")
        self.multiband_bands_edit.setToolTip("RGB band order, e.g. 4,3,2 or 8,4,3")
        self.multiband_bands_edit.setMaximumWidth(70)
        row1.addWidget(self.multiband_bands_edit)

        row1.addWidget(self.btn_detection_preview)
        row1.addWidget(self.btn_gmt_map)
        root.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Lat:"))
        self.lat_spin = QDoubleSpinBox()
        self.lat_spin.setRange(-85.0, 85.0)
        self.lat_spin.setDecimals(7)
        self.lat_spin.setValue(24.0)
        self.lat_spin.setMaximumWidth(110)
        row2.addWidget(self.lat_spin)

        row2.addWidget(QLabel("Lon:"))
        self.lon_spin = QDoubleSpinBox()
        self.lon_spin.setRange(-180.0, 180.0)
        self.lon_spin.setDecimals(7)
        self.lon_spin.setValue(45.0)
        self.lon_spin.setMaximumWidth(120)
        row2.addWidget(self.lon_spin)

        row2.addWidget(QLabel("Zoom:"))
        self.zoom_spin = QSpinBox()
        self.zoom_spin.setRange(1, 22)
        self.zoom_spin.setValue(17)
        self.zoom_spin.setMaximumWidth(70)
        row2.addWidget(self.zoom_spin)

        self.btn_fit_layer = QPushButton("Fit")
        row2.addWidget(self.btn_fit_layer)

        row2.addSpacing(12)
        row2.addWidget(QLabel("Conf:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(0)
        self.conf_slider.setMaximumWidth(120)
        self.conf_label = QLabel("0.00")
        self.conf_label.setMinimumWidth(34)
        row2.addWidget(self.conf_slider)
        row2.addWidget(self.conf_label)

        row2.addWidget(QLabel("Class:"))
        self.class_combo = QComboBox()
        self.class_combo.addItem("All")
        self.class_combo.setMinimumWidth(120)
        row2.addWidget(self.class_combo)

        self.btn_apply_filters = QPushButton("Apply")
        row2.addWidget(self.btn_apply_filters)

        row2.addWidget(QLabel("NMS IoU:"))
        self.nms_iou = QDoubleSpinBox()
        self.nms_iou.setRange(0.01, 0.95)
        self.nms_iou.setSingleStep(0.05)
        self.nms_iou.setValue(0.30)
        self.nms_iou.setMaximumWidth(70)
        row2.addWidget(self.nms_iou)

        self.btn_geo_nms = QPushButton("Geo-NMS")
        row2.addWidget(self.btn_geo_nms)

        self.btn_formlearner = QPushButton("Run FormLearner")
        self.btn_formlearner.setToolTip("Run FormLearner directly with the inline settings.")
        row2.addWidget(self.btn_formlearner)

        self.formlearner_combo = QComboBox()
        self.formlearner_combo.addItems([
            "Mustatil / long rectangle",
            "Rectangle / building-like",
            "Round cairn / mound",
            "Ring / enclosure",
            "Compact object",
            "All shapes balanced",
        ])
        self.formlearner_combo.setMaximumWidth(180)
        self.formlearner_combo.addItem("Custom model")
        row2.addWidget(self.formlearner_combo)

        self.btn_formlearner_model = QPushButton("Custom Model")
        self.btn_formlearner_model.setToolTip("Choose a custom FormLearner model/config file.")
        row2.addWidget(self.btn_formlearner_model)

        self.formlearner_model_label = QLabel("none")
        self.formlearner_model_label.setMaximumWidth(120)
        self.formlearner_model_label.setToolTip("Selected custom FormLearner model path.")
        row2.addWidget(self.formlearner_model_label)

        self.formlearner_sam2 = QCheckBox("SAM2")
        self.formlearner_sam2.setToolTip("Use SAM2 refinement if a SAM2/FormLearner hook is available.")
        row2.addWidget(self.formlearner_sam2)

        row2.addWidget(QLabel("Form score:"))
        self.formlearner_score_slider = QSlider(Qt.Horizontal)
        self.formlearner_score_slider.setRange(1, 100)
        self.formlearner_score_slider.setValue(60)
        self.formlearner_score_slider.setMaximumWidth(105)
        self.formlearner_score_label = QLabel("0.60")
        self.formlearner_score_label.setMinimumWidth(34)
        row2.addWidget(self.formlearner_score_slider)
        row2.addWidget(self.formlearner_score_label)

        self.smooth_viewport = QCheckBox("Smooth")
        self.smooth_viewport.setToolTip("Debounced threaded TIFF viewport reload; faster and less freezing on huge rasters.")
        self.smooth_viewport.setChecked(True)
        row2.addWidget(self.smooth_viewport)

        self.box_select = QCheckBox("Box Select")
        row2.addWidget(self.box_select)

        self.btn_clear_selection = QPushButton("Clear Sel.")
        row2.addWidget(self.btn_clear_selection)
        row2.addStretch(1)
        root.addLayout(row2)

        row3 = QHBoxLayout()
        self.btn_load_vector = QPushButton("Load GPKG/Vector")
        self.btn_load_all_gpkg = QPushButton("Load all GPKG layers")
        self.btn_redraw = QPushButton("Redraw")
        self.btn_undo = QPushButton("Undo")
        self.btn_undo.setToolTip("Undo last GIS review action. Then press Redraw.")
        self.btn_undo.setEnabled(False)
        self.show_all_features = QCheckBox("Show all features")
        self.show_all_features.setToolTip("Ignore confidence/class filters and draw every feature, including 3000+ detections.")
        self.show_all_features.setChecked(True)
        self.btn_export_layer = QPushButton("Export Layer")
        self.btn_export_selected = QPushButton("Export Selected")
        self.auto_fit = QCheckBox("Auto-fit")
        self.auto_fit.setChecked(True)
        row3.addWidget(self.btn_load_vector)
        row3.addWidget(self.btn_load_all_gpkg)
        row3.addWidget(self.btn_redraw)
        row3.addWidget(self.btn_undo)
        row3.addWidget(self.show_all_features)
        row3.addWidget(self.btn_export_layer)
        row3.addWidget(self.btn_export_selected)
        row3.addWidget(self.auto_fit)

        row3.addSpacing(16)
        row3.addWidget(QLabel("Review rows:"))
        self.review_combo = QComboBox()
        self.review_combo.addItems(["correct", "false_positive", "uncertain", "needs_review", "confirmed_archaeology", "hidden"])
        row3.addWidget(self.review_combo)
        self.btn_mark_rows = QPushButton("Mark")
        self.btn_copy_rows = QPushButton("Copy Review")
        self.btn_move_rows = QPushButton("Move Review")
        self.btn_send_annotator = QPushButton("Send Pos/FP Crops → Annotator")
        self.btn_send_annotator.setToolTip("Creates raw satellite crops without drawing rectangles and YOLO labels for positives/false positives.")
        row3.addWidget(self.btn_mark_rows)
        row3.addWidget(self.btn_copy_rows)
        row3.addWidget(self.btn_move_rows)
        row3.addWidget(self.btn_send_annotator)
        row3.addStretch(1)
        root.addLayout(row3)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.addWidget(QLabel("Layers"))
        self.layer_tree = QTreeWidget()
        self.layer_tree.setColumnCount(4)
        self.layer_tree.setHeaderLabels(["Layer", "Features", "CRS", "Source"])
        left_l.addWidget(self.layer_tree, 1)
        splitter.addWidget(left)

        center = QWidget()
        center_l = QVBoxLayout(center)
        center_l.setContentsMargins(0, 0, 0, 0)
        center_l.addWidget(QLabel("Canvas: click inside boxes to select; Box Select = drag-select; right-click selected features"))
        self.web = None
        self.web_page_ready = False
        self._web_safe_reload_done = False
        self.canvas_stack = QStackedWidget()
        center_l.addWidget(self.canvas_stack, 1)

        # Page 0: Web map / Leaflet
        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView
            from PySide6.QtWebChannel import QWebChannel
            self.web = QWebEngineView()
            try:
                self.web.setStyleSheet("background:#ffffff;")
                self.web.page().setBackgroundColor(Qt.white)
            except Exception:
                pass
            try:
                self.web.loadFinished.connect(self._on_web_page_loaded)
            except Exception:
                pass
            try:
                self.web_page_ready = False
            except Exception:
                pass
            self.web.setHtml(_html(), QUrl("http://mustatil.local/"))
            self.channel = QWebChannel(self.web.page())
            Bridge = _make_bridge_class()
            self.bridge = Bridge(self)
            self.channel.registerObject("mustatilBridge", self.bridge)
            self.web.page().setWebChannel(self.channel)
            self.web_ok = True
            self.canvas_stack.addWidget(self.web)
        except Exception as exc:
            self.web_ok = False
            self.web_error = QTextEdit()
            self.web_error.setReadOnly(True)
            self.web_error.setPlainText(
                "QtWebEngine / QtWebChannel is not available.\n\n"
                "Install/use PySide6-WebEngine in the Mustatil environment.\n\n"
                f"Error:\n{exc}"
            )
            self.canvas_stack.addWidget(self.web_error)

        # Page 1: Detection-Preview-style image canvas
        try:
            DetectionLikeImageCanvas, _FeatureRectItem = _make_detection_like_canvas_classes()
            self.image_canvas = DetectionLikeImageCanvas()
            try:
                self.image_canvas.set_manager(self)
            except Exception:
                pass
            self.image_canvas.zoom_pan_changed.connect(self._schedule_gis_detection_pyramid_reload)
            try:
                self.image_canvas.horizontalScrollBar().valueChanged.connect(lambda _=None: self._schedule_gis_detection_pyramid_reload())
                self.image_canvas.verticalScrollBar().valueChanged.connect(lambda _=None: self._schedule_gis_detection_pyramid_reload())
            except Exception:
                pass
            try:
                self._enable_smooth_image_canvas()
            except Exception:
                pass
            try:
                self._install_gis_render_bus()
            except Exception:
                pass
            self.canvas_stack.addWidget(self.image_canvas)
        except Exception as exc:
            self.image_canvas = QTextEdit()
            self.image_canvas.setReadOnly(True)
            self.image_canvas.setPlainText(f"Could not create Detection-like image canvas:\n{exc}")
            self.canvas_stack.addWidget(self.image_canvas)

        splitter.addWidget(center)

        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)
        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.info.setMaximumHeight(125)
        self.info.setPlainText(
            "Load map/image, load detections, filter by confidence/class, apply Geo-NMS, review by right-click, then send Pos/FP raw satellite crops to Annotator."
        )
        right_l.addWidget(self.info)
        right_l.addWidget(QLabel("Attributes"))
        self.attr_table = QTableWidget()
        self.attr_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.attr_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.attr_table.setAlternatingRowColors(True)
        right_l.addWidget(self.attr_table, 1)
        splitter.addWidget(right)

        splitter.setSizes([250, 900, 460])

        self.view_combo.currentTextChanged.connect(self.on_view_changed)
        self.server_combo.currentTextChanged.connect(self.on_server_changed)
        self.btn_load_map.clicked.connect(self.load_map)
        self.btn_load_image.clicked.connect(self.load_image_dialog)
        self.multiband_check.toggled.connect(lambda _=None: self.reload_current_image_for_multiband())
        self.multiband_bands_edit.editingFinished.connect(self.reload_current_image_for_multiband)
        self.btn_detection_preview.clicked.connect(self.use_detection_preview)
        self.btn_gmt_map.clicked.connect(self.create_gmt_map)
        self.btn_fit_layer.clicked.connect(self.fit_to_overlays)
        self.box_select.toggled.connect(self.set_box_select)
        self.btn_clear_selection.clicked.connect(self.clear_selection)
        self.conf_slider.valueChanged.connect(self.on_conf_changed)
        self.btn_apply_filters.clicked.connect(self.redraw_overlays)
        self.show_all_features.toggled.connect(lambda _=None: self.set_info("Show all features changed. Press Redraw to update overlays."))
        self.class_combo.currentTextChanged.connect(lambda _=None: self.set_info("Class filter changed. Press Redraw to update overlays."))
        self.btn_geo_nms.clicked.connect(self.apply_geo_nms_current_layer)
        self.btn_formlearner.clicked.connect(self.run_formlearner_inline)
        self.btn_formlearner_model.clicked.connect(self.choose_formlearner_custom_model)
        self.formlearner_score_slider.valueChanged.connect(self.on_formscore_slider_changed)
        self.btn_send_annotator.clicked.connect(self.send_pos_fp_crops_to_annotator)
        self.btn_load_vector.clicked.connect(self.load_vector_dialog)
        self.btn_load_all_gpkg.clicked.connect(self.load_all_gpkg_dialog)
        self.btn_redraw.clicked.connect(self.redraw_overlays)
        self.btn_undo.clicked.connect(self.undo_last_action)
        self.btn_export_layer.clicked.connect(self.export_current_layer_dialog)
        self.btn_export_selected.clicked.connect(self.export_selected_rows_dialog)
        self.btn_mark_rows.clicked.connect(self.mark_selected_rows)
        self.btn_copy_rows.clicked.connect(lambda: self.copy_or_move_selected_rows(copy=True))
        self.btn_move_rows.clicked.connect(lambda: self.copy_or_move_selected_rows(copy=False))
        self.layer_tree.itemSelectionChanged.connect(self.on_layer_selection_changed)
        self.attr_table.itemSelectionChanged.connect(self.on_table_selection_changed)

        self.attach_workspace()

        try:
            from PySide6.QtCore import QTimer
            QTimer.singleShot(1200, self.load_map)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Workspace / JS
    # ------------------------------------------------------------------

    def attach_workspace(self):
        try:
            self.workspace = self.workspace or _find_workspace(self.widget)
            if self.workspace is not None:
                self.workspace.mustatil_gis_web_review_map = self
                self.workspace.mustatil_layer_manager = self
                self.workspace.mustatil_layers = self.layers
                self.workspace.layer_tree = self.layer_tree
        except Exception:
            pass

    def set_info(self, text: str):
        try:
            self.info.setPlainText(str(text))
        except Exception:
            pass

    def js(self, code: str):
        if not self.web_ok or self.web is None:
            return
        try:
            self.web.page().runJavaScript(str(code))
        except Exception as exc:
            self.set_info(f"JavaScript call failed:\n{exc}")

    def on_view_changed(self, name: str):
        self.view_mode = "map" if name == "Web Map" else "image"
        try:
            self.canvas_stack.setCurrentIndex(0 if self.view_mode == "map" else 1)
        except Exception:
            pass

    def on_server_changed(self, name: str):
        self.custom_url.setVisible(str(name) == "Custom URL")
        try:
            if str(name) == "Google Satellite":
                self._user_chose_google_satellite = True
        except Exception:
            pass

    def on_conf_changed(self, value: int):
        conf = float(value) / 100.0
        self.conf_label.setText(f"{conf:.2f}")
        try:
            self.set_info("Confidence changed. Press Redraw to update overlays.")
        except Exception:
            pass

    def show_all_enabled(self) -> bool:
        try:
            return bool(self.show_all_features.isChecked())
        except Exception:
            return False

    def current_min_conf(self) -> float:
        if self.show_all_enabled():
            return 0.0
        return float(self.conf_slider.value()) / 100.0

    def current_class_filter(self) -> str:
        if self.show_all_enabled():
            return "All"
        return str(self.class_combo.currentText() or "All")

    def selected_template(self) -> str:
        name = str(self.server_combo.currentText() or "")
        if name == "Custom URL":
            return str(self.custom_url.text() or "").strip()
        return MAP_SERVERS.get(name, MAP_SERVERS["Google Satellite"])

    def _on_web_page_loaded(self, ok: bool = True):
        try:
            self.web_page_ready = bool(ok)
            # Wait a tiny moment until Leaflet/QtWebChannel JS is fully initialized.
            from PySide6.QtCore import QTimer
            if not getattr(self, "_web_safe_reload_done", False):
                self._web_safe_reload_done = True
                QTimer.singleShot(250, self.load_map)
        except Exception:
            pass

    def reload_web_map_safe(self):
        try:
            self._web_safe_reload_done = False
            if self.web is not None:
                self.web.setHtml(_html(), QUrl("http://mustatil.local/"))
            from PySide6.QtCore import QTimer
            QTimer.singleShot(600, self.load_map)
        except Exception:
            try:
                self.load_map()
            except Exception:
                pass

    def load_map(self):
        self.view_combo.setCurrentText("Web Map")
        self.view_mode = "map"
        try:
            self.canvas_stack.setCurrentIndex(0)
        except Exception:
            pass

        # v16: if the selected Google tile source renders black/blocked in QtWebEngine,
        # prefer the stable satellite source. The user can still manually choose Google.
        try:
            if str(self.server_combo.currentText() or "") == "Google Satellite" and not getattr(self, "_user_chose_google_satellite", False):
                self.server_combo.blockSignals(True)
                self.server_combo.setCurrentText("Esri World Imagery")
                self.server_combo.blockSignals(False)
        except Exception:
            pass

        try:
            if not getattr(self, "web_page_ready", False):
                from PySide6.QtCore import QTimer
                QTimer.singleShot(250, self.load_map)
                self.set_info("Web map is still loading; retrying Load Map shortly.")
                return
        except Exception:
            pass

        template = self.selected_template()
        name = str(self.server_combo.currentText() or "Map")
        if not template:
            self.set_info("No map server URL template selected.")
            return
        if ("{x}" not in template or "{y}" not in template or "{z}" not in template) and ("{q}" not in template and "{quadkey}" not in template):
            self.set_info("Template must contain {x}/{y}/{z} or Bing {q}/{quadkey}.")
            return
        lat = float(self.lat_spin.value())
        lon = float(self.lon_spin.value())
        zoom = int(self.zoom_spin.value())
        self.js(f"window.setTileServer({json.dumps(name)}, {json.dumps(template)}, {lat}, {lon}, {zoom});")
        self.redraw_overlays()
        self.set_info(f"Map loaded safely:\n{name}\n{template}\nCenter: {lat}, {lon}, z={zoom}\nBackend: WebMap-safe QPainter image canvas, no QOpenGLWidget conflict.")

    def load_image_dialog(self):
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self.widget,
            "Load image/raster background",
            str(Path.home()),
            "Images/Rasters (*.tif *.tiff *.png *.jpg *.jpeg *.webp *.bmp);;All files (*.*)"
        )
        if path:
            self.load_image_background(path)


    # ------------------------------------------------------------------
    # Exact Detection Preview image backend for GIS tab
    # ------------------------------------------------------------------

    def _gis_preview_service(self):
        try:
            svc = globals().get("detection_preview_service") or MUSTATIL_GLOBALS.get("detection_preview_service")
        except Exception:
            svc = globals().get("detection_preview_service")
        if svc is None:
            import detection_preview_service as svc
        return svc

    def _gis_is_tiff_path(self, path: str) -> bool:
        try:
            svc = self._gis_preview_service()
            fn = getattr(svc, "is_tiff_path", None)
            if callable(fn):
                return bool(fn(str(path)))
        except Exception:
            pass
        return str(path or "").lower().endswith((".tif", ".tiff"))

    def _gis_fallback_loader(self, path: str, maxs: int = 2400):
        from PIL import Image
        im = Image.open(str(path)).convert("RGB")
        w, h = im.size
        im.thumbnail((max(512, int(maxs or 2400)), max(512, int(maxs or 2400))))
        return im.copy(), w, h

    def _multiband_enabled(self) -> bool:
        try:
            return bool(self.multiband_check.isChecked())
        except Exception:
            return False

    def _multiband_bands(self):
        try:
            raw = str(self.multiband_bands_edit.text() or "1,2,3")
            vals = []
            for part in raw.replace(";", ",").replace(" ", ",").split(","):
                part = part.strip()
                if part:
                    vals.append(max(1, int(part)))
            if not vals:
                vals = [1, 2, 3]
            while len(vals) < 3:
                vals.append(vals[-1])
            return vals[:3]
        except Exception:
            return [1, 2, 3]

    def _normalize_multiband_arr(self, arr):
        import numpy as np
        arr = arr.astype("float32")
        outs = []
        for i in range(arr.shape[0]):
            band = arr[i]
            valid = band[np.isfinite(band)]
            if valid.size:
                lo = float(np.nanpercentile(valid, 2))
                hi = float(np.nanpercentile(valid, 98))
            else:
                lo, hi = 0.0, 1.0
            if hi <= lo:
                hi = lo + 1.0
            outs.append(np.clip((band - lo) / (hi - lo) * 255.0, 0, 255).astype("uint8"))
        return np.stack(outs[:3], axis=0)

    def _load_multiband_overview(self, path: str, maxs: int = 2400):
        import rasterio
        import numpy as np
        from PIL import Image
        from rasterio.enums import Resampling
        with rasterio.open(str(path)) as src:
            bands = [min(max(1, b), src.count) for b in self._multiband_bands()]
            scale = min(1.0, float(maxs) / max(1, max(src.width, src.height)))
            out_w = max(1, int(src.width * scale))
            out_h = max(1, int(src.height * scale))
            arr = src.read(bands, out_shape=(len(bands), out_h, out_w), resampling=Resampling.bilinear)
            arr = self._normalize_multiband_arr(arr)
            img = Image.fromarray(np.transpose(arr, (1, 2, 0)), "RGB")
            return img, int(src.width), int(src.height)

    def _load_multiband_viewport(self, path: str, center_x: float, center_y: float,
                                 source_width: float, source_height: float,
                                 out_width: int, out_height: int):
        import rasterio
        import numpy as np
        from PIL import Image
        from rasterio.windows import Window
        from rasterio.enums import Resampling
        with rasterio.open(str(path)) as src:
            bands = [min(max(1, b), src.count) for b in self._multiband_bands()]
            sw = max(1.0, float(source_width))
            sh = max(1.0, float(source_height))
            x1 = max(0.0, min(float(src.width - 1), float(center_x) - sw / 2.0))
            y1 = max(0.0, min(float(src.height - 1), float(center_y) - sh / 2.0))
            x2 = max(x1 + 1.0, min(float(src.width), float(center_x) + sw / 2.0))
            y2 = max(y1 + 1.0, min(float(src.height), float(center_y) + sh / 2.0))
            win = Window(x1, y1, x2 - x1, y2 - y1)
            arr = src.read(
                bands,
                window=win,
                out_shape=(len(bands), max(1, int(out_height)), max(1, int(out_width))),
                resampling=Resampling.bilinear,
            )
            arr = self._normalize_multiband_arr(arr)
            img = Image.fromarray(np.transpose(arr, (1, 2, 0)), "RGB")
            return img, int(src.width), int(src.height), float(x1), float(y1), float(x2 - x1), float(y2 - y1)

    def reload_current_image_for_multiband(self):
        try:
            path = str(getattr(self, "gis_det_preview_source", "") or "").strip()
            if path and Path(path).exists() and self._gis_is_tiff_path(path):
                self.load_image_background(path)
        except Exception as exc:
            try:
                self.set_info(f"Multiband reload failed:\\n{exc}")
            except Exception:
                pass

    def _gis_load_detection_overview(self, path: str):
        if self._multiband_enabled() and self._gis_is_tiff_path(path):
            return self._load_multiband_overview(path, maxs=2400)
        svc = self._gis_preview_service()
        fn = getattr(svc, "load_detection_preview_image", None)
        if not callable(fn):
            raise RuntimeError("detection_preview_service.load_detection_preview_image is missing.")
        try:
            return fn(str(path), maxs=2400, fallback_loader=self._gis_fallback_loader)
        except TypeError:
            return fn(str(path), maxs=2400)

    def _gis_load_detection_viewport(self, path: str, center_x: float, center_y: float,
                                     source_width: float, source_height: float,
                                     out_width: int, out_height: int):
        if self._multiband_enabled() and self._gis_is_tiff_path(path):
            return self._load_multiband_viewport(path, center_x, center_y, source_width, source_height, out_width, out_height)
        svc = self._gis_preview_service()
        fn = getattr(svc, "load_detection_viewport_image", None)
        if not callable(fn):
            raise RuntimeError("detection_preview_service.load_detection_viewport_image is missing.")
        try:
            return fn(
                str(path),
                center_x=float(center_x),
                center_y=float(center_y),
                source_width=float(source_width),
                source_height=float(source_height),
                out_width=int(out_width),
                out_height=int(out_height),
                tile_px=512,
            )
        except TypeError as type_exc:
            if "unexpected keyword argument" not in str(type_exc):
                raise
            pps = float(out_width) / max(1.0, float(source_width))
            return fn(str(path), float(center_x), float(center_y), int(out_width), int(out_height), pps, max_output_px=4096)

    def _gis_load_raster_georef(self, path: str, fallback_w: int, fallback_h: int):
        meta = {
            "source": str(path),
            "raster_crs": None,
            "raster_transform": None,
            "raster_width": int(fallback_w),
            "raster_height": int(fallback_h),
            # IMPORTANT: source-window canvas uses original image pixels.
            # Therefore geometry conversion must NOT use preview thumbnail scale.
            "scale_x": 1.0,
            "scale_y": 1.0,
            "exact_detection_preview": True,
        }
        try:
            if self._gis_is_tiff_path(path):
                import rasterio
                with rasterio.open(str(path)) as src:
                    meta["raster_crs"] = src.crs
                    meta["raster_transform"] = src.transform
                    meta["raster_width"] = int(src.width)
                    meta["raster_height"] = int(src.height)
        except Exception:
            pass
        return meta

    def _gis_set_preview_image(self, pil_img, orig_w: int, orig_h: int, source_window, fit: bool = False):
        self.image_size = (int(orig_w), int(orig_h))
        self.view_mode = "image"
        try:
            self.view_combo.setCurrentText("Image / Raster")
        except Exception:
            pass
        try:
            self.canvas_stack.setCurrentIndex(1)
        except Exception:
            pass

        if hasattr(self.image_canvas, "set_pil_image_source_window"):
            self.image_canvas.set_pil_image_source_window(pil_img, int(orig_w), int(orig_h), source_window, fit=bool(fit))
        else:
            # Fallback only if an unexpected old canvas is present.
            from PySide6.QtGui import QPixmap, QImage
            im = pil_img.convert("RGBA")
            data = im.tobytes("raw", "RGBA")
            qimg = QImage(data, im.width, im.height, QImage.Format_RGBA8888)
            self.image_canvas.set_qpixmap(QPixmap.fromImage(qimg.copy()), fit=bool(fit))

    def _gis_current_source_window(self):
        try:
            if not getattr(self, "gis_det_preview_source", ""):
                return None
            view = self.image_canvas
            W = float(getattr(self, "gis_det_orig_w", 0) or self.image_meta.get("raster_width") or self.image_size[0] or 0)
            H = float(getattr(self, "gis_det_orig_h", 0) or self.image_meta.get("raster_height") or self.image_size[1] or 0)
            if W <= 0 or H <= 0:
                return None
            rect = view.mapToScene(view.viewport().rect()).boundingRect()
            x1 = max(0.0, min(W, float(rect.left())))
            y1 = max(0.0, min(H, float(rect.top())))
            x2 = max(0.0, min(W, float(rect.right())))
            y2 = max(0.0, min(H, float(rect.bottom())))
            if x2 <= x1 or y2 <= y1:
                return getattr(self, "gis_det_preview_window", None) or (0.0, 0.0, W, H)
            pad_x = max(16.0, (x2 - x1) * 0.15)
            pad_y = max(16.0, (y2 - y1) * 0.15)
            return (
                max(0.0, x1 - pad_x),
                max(0.0, y1 - pad_y),
                min(W, x2 + pad_x),
                min(H, y2 + pad_y),
            )
        except Exception:
            return getattr(self, "gis_det_preview_window", None)

    def _enable_smooth_image_canvas(self):
        """
        v16: WebMap-safe smooth rendering.

        v15 used QOpenGLWidget for the image canvas. On some Windows/QtWebEngine
        combinations this makes the Leaflet satellite map black because
        QWebEngine and QOpenGLWidget compete for the same GPU/compositor surface.

        Therefore v16 keeps smooth threaded raster reads, but forces the image
        canvas to use the normal QPainter viewport. This keeps the satellite
        WebView visible and still reduces freezes through threaded/debounced
        TIFF viewport loading.
        """
        try:
            from PySide6.QtWidgets import QGraphicsView
            from PySide6.QtGui import QPainter

            self._smooth_render_backend = "QPainter-WebMapSafe"

            try:
                self.image_canvas.setViewportUpdateMode(QGraphicsView.MinimalViewportUpdate)
            except Exception:
                pass
            try:
                self.image_canvas.setCacheMode(QGraphicsView.CacheBackground)
            except Exception:
                pass
            try:
                self.image_canvas.setOptimizationFlag(QGraphicsView.DontSavePainterState, True)
                self.image_canvas.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
            except Exception:
                pass
            try:
                self.image_canvas.setRenderHint(QPainter.Antialiasing, False)
                self.image_canvas.setRenderHint(QPainter.SmoothPixmapTransform, False)
            except Exception:
                pass
        except Exception:
            self._smooth_render_backend = "default"

    def _install_gis_render_bus(self):
        if _GisViewportRenderBus is None:
            return
        if getattr(self, "gis_render_bus", None) is None:
            self.gis_render_bus = _GisViewportRenderBus(self.widget)
            self.gis_render_bus.resultReady.connect(self._apply_gis_detection_viewport_result)
            self.gis_render_bus.errorReady.connect(self._handle_gis_detection_viewport_error)

    def _request_gis_detection_pyramid_reload(self):
        """
        Smooth mode: render the heavy TIFF viewport in a background thread.
        Non-smooth mode: use the original synchronous Detection Preview reload.
        """
        try:
            if hasattr(self, "smooth_viewport") and not self.smooth_viewport.isChecked():
                return self._reload_gis_detection_pyramid_for_zoom()
            return self._start_gis_detection_viewport_thread()
        except Exception:
            return self._reload_gis_detection_pyramid_for_zoom()

    def _start_gis_detection_viewport_thread(self):
        try:
            path = str(getattr(self, "gis_det_preview_source", "") or "").strip()
            if not path or not self._gis_is_tiff_path(path):
                return
            win = self._gis_current_source_window()
            if not win:
                return

            if getattr(self, "gis_render_busy", False):
                self.gis_render_pending = True
                return

            x1, y1, x2, y2 = map(float, win)
            sw = max(1.0, x2 - x1)
            sh = max(1.0, y2 - y1)
            W = float(getattr(self, "gis_det_orig_w", 1) or 1)
            H = float(getattr(self, "gis_det_orig_h", 1) or 1)

            full_fraction = max(sw / max(1.0, W), sh / max(1.0, H))
            if full_fraction > 0.92:
                if getattr(self, "gis_det_preview_viewport_mode", False):
                    return self._reload_gis_detection_pyramid_for_zoom()
                return

            view_w = max(256, int(self.image_canvas.viewport().width() or 1024))
            view_h = max(256, int(self.image_canvas.viewport().height() or 768))
            max_out = int(getattr(self, "gis_smooth_max_out_px", 2048) or 2048)
            out_w = min(max_out, max(512, view_w))
            out_h = min(max_out, max(512, view_h))
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            old_transform = self.image_canvas.transform()
            old_center = self.image_canvas.mapToScene(self.image_canvas.viewport().rect().center())

            self.gis_render_busy = True
            self.gis_render_pending = False
            self.gis_render_request_id = int(getattr(self, "gis_render_request_id", 0) or 0) + 1
            request_id = self.gis_render_request_id

            def worker():
                try:
                    result = self._gis_load_detection_viewport(path, cx, cy, sw, sh, out_w, out_h)
                    payload = {
                        "request_id": request_id,
                        "path": path,
                        "result": result,
                        "cx": cx,
                        "cy": cy,
                        "sw": sw,
                        "sh": sh,
                        "old_transform": old_transform,
                        "old_center": old_center,
                    }
                    if getattr(self, "gis_render_bus", None) is not None:
                        self.gis_render_bus.resultReady.emit(payload)
                    else:
                        self.gis_render_busy = False
                except Exception as exc:
                    if getattr(self, "gis_render_bus", None) is not None:
                        self.gis_render_bus.errorReady.emit(str(exc))
                    else:
                        self.gis_render_busy = False

            threading.Thread(target=worker, daemon=True).start()
        except Exception as exc:
            self.gis_render_busy = False
            self.set_info(f"Smooth viewport request failed:\\n{exc}")

    def _apply_gis_detection_viewport_result(self, payload: dict):
        try:
            self.gis_render_busy = False
            if not isinstance(payload, dict):
                return
            if int(payload.get("request_id", -1)) != int(getattr(self, "gis_render_request_id", 0)):
                return
            path = str(payload.get("path", ""))
            if path != str(getattr(self, "gis_det_preview_source", "") or ""):
                return

            result = payload.get("result")
            cx = float(payload.get("cx", 0.0))
            cy = float(payload.get("cy", 0.0))
            sw = float(payload.get("sw", 1.0))
            sh = float(payload.get("sh", 1.0))

            if isinstance(result, tuple) and len(result) >= 7:
                img, ow, oh, rx1, ry1, rsw, rsh = result[:7]
                rx1 = float(rx1)
                ry1 = float(ry1)
                rx2 = rx1 + float(rsw)
                ry2 = ry1 + float(rsh)
            else:
                img, ow, oh = result[:3]
                rx1 = max(0.0, cx - sw / 2.0)
                ry1 = max(0.0, cy - sh / 2.0)
                rx2 = min(float(ow), rx1 + sw)
                ry2 = min(float(oh), ry1 + sh)

            self.gis_det_orig_w = int(ow)
            self.gis_det_orig_h = int(oh)
            self.gis_det_preview_window = (rx1, ry1, rx2, ry2)
            self.gis_det_preview_viewport_mode = True

            self._gis_set_preview_image(img, int(ow), int(oh), self.gis_det_preview_window, fit=False)
            try:
                self.image_canvas.setTransform(payload.get("old_transform"))
                self.image_canvas.centerOn(payload.get("old_center"))
            except Exception:
                pass

            self.redraw_overlays()
            self.set_info(
                "Smooth threaded TIFF viewport loaded.\\n"
                f"Backend: {getattr(self, '_smooth_render_backend', 'default')}\\n"
                f"Source window: x={rx1:.0f}, y={ry1:.0f}, w={rx2-rx1:.0f}, h={ry2-ry1:.0f}\\n"
                f"Preview: {img.size[0]} x {img.size[1]}"
            )

            if getattr(self, "gis_render_pending", False):
                self.gis_render_pending = False
                self._schedule_gis_detection_pyramid_reload()
        except Exception as exc:
            self.gis_render_busy = False
            self.set_info(f"Could not apply smooth TIFF viewport:\\n{exc}\\n\\n{traceback.format_exc()}")

    def _handle_gis_detection_viewport_error(self, msg: str):
        self.gis_render_busy = False
        self.set_info(f"Smooth TIFF viewport render failed:\\n{msg}")
        if getattr(self, "gis_render_pending", False):
            self.gis_render_pending = False
            self._schedule_gis_detection_pyramid_reload()

    def _schedule_gis_detection_pyramid_reload(self):
        try:
            # v17: do not redraw TIFF viewport while the right-drag
            # selection rectangle is being drawn.
            try:
                if getattr(self.image_canvas, "_selecting", False):
                    return
            except Exception:
                pass
            path = str(getattr(self, "gis_det_preview_source", "") or "")
            if not path or not self._gis_is_tiff_path(path):
                return
            from PySide6.QtCore import QTimer
            if getattr(self, "gis_det_preview_pyramid_timer", None) is None:
                timer = QTimer(self.widget)
                timer.setSingleShot(True)
                timer.timeout.connect(self._request_gis_detection_pyramid_reload)
                self.gis_det_preview_pyramid_timer = timer
            delay = int(getattr(self, "gis_smooth_reload_delay_ms", 320) or 320)
            try:
                if hasattr(self, "smooth_viewport") and not self.smooth_viewport.isChecked():
                    delay = 180
            except Exception:
                pass
            self.gis_det_preview_pyramid_timer.start(delay)
        except Exception:
            pass

    def _reload_gis_detection_pyramid_for_zoom(self):
        try:
            path = str(getattr(self, "gis_det_preview_source", "") or "").strip()
            if not path or not self._gis_is_tiff_path(path):
                return
            win = self._gis_current_source_window()
            if not win:
                return
            x1, y1, x2, y2 = map(float, win)
            sw = max(1.0, x2 - x1)
            sh = max(1.0, y2 - y1)
            W = float(getattr(self, "gis_det_orig_w", 1) or 1)
            H = float(getattr(self, "gis_det_orig_h", 1) or 1)

            full_fraction = max(sw / max(1.0, W), sh / max(1.0, H))
            if full_fraction > 0.92 and getattr(self, "gis_det_preview_viewport_mode", False):
                img, ow, oh = self._gis_load_detection_overview(path)
                self.gis_det_orig_w = int(ow)
                self.gis_det_orig_h = int(oh)
                self.gis_det_preview_window = (0.0, 0.0, float(ow), float(oh))
                self.gis_det_preview_viewport_mode = False
                self._gis_set_preview_image(img, int(ow), int(oh), self.gis_det_preview_window, fit=True)
                self.redraw_overlays()
                self.set_info("GIS image overview reloaded with Detection Preview service.")
                return
            if full_fraction > 0.92:
                return

            view_w = max(256, int(self.image_canvas.viewport().width() or 1024))
            view_h = max(256, int(self.image_canvas.viewport().height() or 768))
            out_w = min(4096, max(512, view_w))
            out_h = min(4096, max(512, view_h))
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            old_transform = self.image_canvas.transform()
            old_center = self.image_canvas.mapToScene(self.image_canvas.viewport().rect().center())

            result = self._gis_load_detection_viewport(path, cx, cy, sw, sh, out_w, out_h)
            if isinstance(result, tuple) and len(result) >= 7:
                img, ow, oh, rx1, ry1, rsw, rsh = result[:7]
                rx1 = float(rx1)
                ry1 = float(ry1)
                rx2 = rx1 + float(rsw)
                ry2 = ry1 + float(rsh)
            else:
                img, ow, oh = result[:3]
                rx1 = max(0.0, cx - sw / 2.0)
                ry1 = max(0.0, cy - sh / 2.0)
                rx2 = min(float(ow), rx1 + sw)
                ry2 = min(float(oh), ry1 + sh)

            self.gis_det_orig_w = int(ow)
            self.gis_det_orig_h = int(oh)
            self.gis_det_preview_window = (rx1, ry1, rx2, ry2)
            self.gis_det_preview_viewport_mode = True

            self._gis_set_preview_image(img, int(ow), int(oh), self.gis_det_preview_window, fit=False)
            try:
                self.image_canvas.setTransform(old_transform)
                self.image_canvas.centerOn(old_center)
            except Exception:
                pass

            self.redraw_overlays()
            self.set_info(
                "GIS image viewport reloaded exactly like Detection Preview.\\n"
                f"Source window: x={rx1:.0f}, y={ry1:.0f}, w={rx2-rx1:.0f}, h={ry2-ry1:.0f}\\n"
                f"Preview: {img.size[0]} x {img.size[1]}"
            )
        except Exception as exc:
            try:
                self.set_info(f"GIS Detection Preview viewport reload error:\\n{exc}\\n\\n{traceback.format_exc()}")
            except Exception:
                pass

    def _set_workspace_detection_image_path(self, path: str) -> bool:
        """
        Set the normal Mustatil Detection image/path variable, using the same
        object the Detection tab uses.
        """
        ws = self.workspace or _find_workspace(self.widget)
        if ws is None:
            return False

        # Common Mustatil/Qt variable names seen in the workspace.
        for attr in [
            "image", "image_path", "det_image", "det_image_path",
            "current_image", "current_image_path", "raster_path",
        ]:
            try:
                var = getattr(ws, attr, None)
                if var is None:
                    continue
                if hasattr(var, "set"):
                    var.set(str(path))
                    return True
                if isinstance(var, (str, Path)):
                    setattr(ws, attr, str(path))
                    return True
            except Exception:
                pass

        # If none existed, still create a conservative attribute.
        try:
            setattr(ws, "image", str(path))
            return True
        except Exception:
            return False

    def _call_detection_loadprev(self) -> bool:
        """
        Call the same preview loader used in the Detection tab.
        """
        ws = self.workspace or _find_workspace(self.widget)
        if ws is None:
            return False

        # Prefer the exact historic Mustatil method name.
        for name in [
            "loadprev", "load_preview", "load_image_preview",
            "load_detection_preview", "det_load_preview",
            "refresh_preview", "refresh_detection_preview",
        ]:
            fn = getattr(ws, name, None)
            if callable(fn):
                try:
                    fn()
                    return True
                except TypeError:
                    try:
                        fn(False)
                        return True
                    except Exception:
                        pass
                except Exception:
                    pass
        return False

    def _copy_detection_preview_to_web(self, source_path: str = "") -> bool:
        """
        Copy exactly the pixmap currently displayed by the Detection preview
        canvas and show it in a QGraphicsView that uses the same zoom/pan style
        as Mustatil Detection Preview.

        This is the requested behavior:
          - same visible image as Detection Preview
          - same wheel zoom anchor-under-mouse
          - same left-drag panning
          - not Leaflet's image overlay zoom
        """
        try:
            ws = self.workspace or _find_workspace(self.widget)
            if ws is None or not hasattr(ws, "image_view"):
                self.set_info("No Detection image_view found.")
                return False

            view = getattr(ws, "image_view")
            item = getattr(view, "pixmap_item", None)

            if item is None:
                try:
                    best = None
                    best_area = -1
                    for it in view.scene().items():
                        try:
                            if hasattr(it, "pixmap"):
                                pm0 = it.pixmap()
                                area = int(pm0.width()) * int(pm0.height()) if pm0 is not None and not pm0.isNull() else -1
                                if area > best_area:
                                    best = it
                                    best_area = area
                        except Exception:
                            pass
                    item = best
                except Exception:
                    item = None

            if item is None:
                self.set_info("Detection Preview is empty. Load a preview in Detection first.")
                return False

            pm = item.pixmap()
            if pm is None or pm.isNull():
                self.set_info("Detection Preview pixmap is empty.")
                return False

            self.image_size = (int(pm.width()), int(pm.height()))
            self.image_meta = {
                "source": source_path or "Detection preview",
                "scale_x": 1.0,
                "scale_y": 1.0,
                "exact_detection_preview": True,
            }

            try:
                if source_path and Path(source_path).exists() and Path(source_path).suffix.lower() in {".tif", ".tiff"}:
                    import rasterio
                    with rasterio.open(source_path) as ds:
                        self.image_meta["raster_crs"] = ds.crs
                        self.image_meta["raster_transform"] = ds.transform
                        self.image_meta["raster_width"] = int(ds.width)
                        self.image_meta["raster_height"] = int(ds.height)
                        self.image_meta["scale_x"] = float(pm.width()) / max(1.0, float(ds.width))
                        self.image_meta["scale_y"] = float(pm.height()) / max(1.0, float(ds.height))
            except Exception:
                pass

            self.view_mode = "image"
            self.view_combo.setCurrentText("Image / Raster")
            try:
                self.canvas_stack.setCurrentIndex(1)
                if hasattr(self.image_canvas, "set_qpixmap"):
                    self.image_canvas.set_qpixmap(pm, fit=True)
            except Exception:
                pass

            self.redraw_overlays()
            self.set_info(
                "Image loaded exactly from Mustatil Detection Preview with Detection-style zoom/pan.\n"
                f"Visible preview size: {pm.width()} x {pm.height()}\n"
                f"Source: {source_path or 'current Detection preview'}"
            )
            return True
        except Exception as exc:
            self.set_info(f"Could not copy Detection Preview to Web Review:\n{exc}\n\n{traceback.format_exc()}")
            return False

    def load_image_background(self, path: str):
        """
        GIS tab image loading, now truly Detection Preview style.

        Same flow:
          1. detection_preview_service.load_detection_preview_image(...)
          2. scene coordinates = original image pixels
          3. pixmap is placed/scaled to source window
          4. wheel/pan triggers load_detection_viewport_image(...)
        """
        try:
            path = str(path or "").strip()
            if not path:
                self.set_info("No image path selected.")
                return

            img, ow, oh = self._gis_load_detection_overview(path)
            self.gis_det_preview_source = path
            self.gis_det_orig_w = int(ow)
            self.gis_det_orig_h = int(oh)
            self.gis_det_preview_window = (0.0, 0.0, float(ow), float(oh))
            self.gis_det_preview_viewport_mode = False
            self.gis_det_preview_base_max = 2400
            self.gis_det_preview_current_max = 2400

            self.image_meta = self._gis_load_raster_georef(path, int(ow), int(oh))
            self.gis_det_orig_w = int(self.image_meta.get("raster_width") or ow)
            self.gis_det_orig_h = int(self.image_meta.get("raster_height") or oh)
            self.gis_det_preview_window = (0.0, 0.0, float(self.gis_det_orig_w), float(self.gis_det_orig_h))

            self._gis_set_preview_image(img, self.gis_det_orig_w, self.gis_det_orig_h, self.gis_det_preview_window, fit=True)
            self.redraw_overlays()

            self.set_info(
                "Image/TIFF loaded in GIS tab with exact Detection Preview logic.\\n"
                f"Source: {path}\\n"
                f"Original size: {self.gis_det_orig_w} x {self.gis_det_orig_h}\\n"
                f"Preview size: {img.size[0]} x {img.size[1]}\\n"
                f"CRS: {self.image_meta.get('raster_crs')}\\n\\n"
                "Mouse wheel zoom / left-drag pan now triggers Detection-style viewport reload."
            )
        except Exception as exc:
            self.set_info(f"Could not load image through Detection Preview logic:\\n{exc}\\n\\n{traceback.format_exc()}")

    def use_detection_preview(self):
        """
        Use the Detection tab's source path, but render it here with the same
        Detection Preview service/viewport logic instead of copying a static pixmap.
        """
        try:
            ws = self.workspace or _find_workspace(self.widget)
            source_path = ""
            try:
                source_path = str(getattr(ws, "det_preview_source", "") or "")
                if not source_path:
                    var = getattr(ws, "image", None)
                    source_path = str(var.get() if hasattr(var, "get") else var or "")
            except Exception:
                source_path = ""

            if source_path and Path(source_path).exists():
                self.load_image_background(source_path)
                return

            # Fallback: copy visible Detection pixmap if there is no source path.
            if ws is None or not hasattr(ws, "image_view"):
                self.set_info("No Detection source path and no Detection image_view found.")
                return
            item = getattr(ws.image_view, "pixmap_item", None)
            if item is None:
                self.set_info("Detection preview has no pixmap. Load Detection preview first.")
                return
            pm = item.pixmap()
            if pm is None or pm.isNull():
                self.set_info("Detection preview pixmap is empty.")
                return
            self.view_mode = "image"
            self.view_combo.setCurrentText("Image / Raster")
            self.canvas_stack.setCurrentIndex(1)
            self.image_canvas.set_qpixmap(pm, fit=True)
            self.image_size = (pm.width(), pm.height())
            self.image_meta = {"source": "Detection preview pixmap fallback", "scale_x": 1.0, "scale_y": 1.0}
            self.redraw_overlays()
            self.set_info("Fallback copied current Detection preview pixmap. For real viewport reload, use Load Image / TIFF.")
        except Exception as exc:
            self.set_info(f"Could not use Detection Preview source:\\n{exc}\\n\\n{traceback.format_exc()}")

    def load_image_pyramid_optional(self, path: str):
        """
        Optional helper kept for future use: smooth pyramid mode.
        The main Load Image/TIFF button intentionally does not use this now,
        because the user requested the exact Detection Preview behavior.
        """
        try:
            sid, port, src = _register_pyramid(path)
            self.image_meta = {
                "source": path,
                "raster_crs": src.crs,
                "raster_transform": src.transform,
                "raster_width": int(src.width),
                "raster_height": int(src.height),
                "scale_x": 1.0,
                "scale_y": 1.0,
                "pyramid": True,
            }
            self.image_size = (int(src.width), int(src.height))
            self.view_mode = "image"
            self.view_combo.setCurrentText("Image / Raster")
            tile_url = f"http://127.0.0.1:{port}/tile/{sid}/" + "{z}/{x}/{y}.png"
            self.js(f"setPyramidImage({json.dumps(tile_url)}, {int(src.width)}, {int(src.height)}, -6, 8);")
            self.redraw_overlays()
            self.set_info(
                "Optional smooth image pyramid loaded.\\n"
                f"Source: {path}\\nSize: {src.width} x {src.height}\\nCRS: {src.crs}"
            )
        except Exception as exc:
            self.set_info(f"Optional pyramid mode failed:\\n{exc}")

    def fit_to_overlays(self):
        self.js("fitAllOverlays();")

    def set_box_select(self, on: bool):
        self.js(f"setSelectionMode({str(bool(on)).lower()});")

    def clear_selection(self):
        self.js("clearSelection();")

    # ------------------------------------------------------------------
    # Layer loading and drawing
    # ------------------------------------------------------------------

    def load_vector_dialog(self):
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self.widget,
            "Load vector detections",
            str(Path.home()),
            "Vector data (*.gpkg *.geojson *.json *.shp);;GeoPackage (*.gpkg);;GeoJSON (*.geojson *.json);;Shapefile (*.shp);;All files (*.*)"
        )
        if path:
            self.load_vector_layer(path)

    def load_all_gpkg_dialog(self):
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self.widget, "Load all GPKG layers", str(Path.home()), "GeoPackage (*.gpkg);;All files (*.*)")
        if path:
            self.load_all_gpkg(path)

    def load_vector_layer(self, path: str, layer: Optional[str] = None, name: Optional[str] = None) -> bool:
        gpd = _gpd()
        if gpd is None:
            self.set_info("geopandas is not available. Cannot load vector layers.")
            return False
        try:
            gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
            if "review_status" not in gdf.columns:
                gdf["review_status"] = ""
            lname = name or layer or Path(path).stem
            layer_idx = len(self.layers)
            self.layers.append({"name": str(lname), "gdf": gdf, "path": str(path), "layer_name": layer, "visible": True})
            self.current_layer_idx = layer_idx
            self.refresh_layer_tree()
            self.refresh_attribute_table()
            self.rebuild_class_combo()
            self.push_layer_to_map(layer_idx, fit=bool(self.auto_fit.isChecked()))
            self.attach_workspace()
            self.set_info(f"Loaded vector layer:\n{lname}\nFeatures: {len(gdf)}\nCRS: {gdf.crs}\nSource: {path}")
            return True
        except Exception as exc:
            self.set_info(f"Could not load vector layer:\n{path}\n\n{exc}")
            return False

    def load_all_gpkg(self, path: str) -> bool:
        try:
            import fiona
            layers = list(fiona.listlayers(path))
        except Exception:
            layers = []
        if not layers:
            return self.load_vector_layer(path)
        ok = 0
        for lyr in layers:
            if self.load_vector_layer(path, layer=lyr, name=lyr):
                ok += 1
        self.set_info(f"Loaded {ok}/{len(layers)} layer(s) from:\n{path}")
        return ok > 0

    def rebuild_class_combo(self):
        current = self.class_combo.currentText() if hasattr(self, "class_combo") else "All"
        values = set()
        for layer in self.layers:
            gdf = layer.get("gdf")
            col = _find_class_col(gdf)
            if gdf is not None and col is not None:
                try:
                    for v in gdf[col].dropna().astype(str).unique().tolist():
                        values.add(v)
                except Exception:
                    pass
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        self.class_combo.addItem("All")
        for v in sorted(values, key=lambda x: (not str(x).isdigit(), str(x))):
            self.class_combo.addItem(str(v))
        ix = self.class_combo.findText(current)
        if ix >= 0:
            self.class_combo.setCurrentIndex(ix)
        self.class_combo.blockSignals(False)

    def _image_geom_bounds(self, geom):
        try:
            minx, miny, maxx, maxy = geom.bounds
            return float(minx), float(miny), float(maxx), float(maxy)
        except Exception:
            return None

    def redraw_image_overlays(self):
        """
        Draw review boxes on the QGraphicsView image canvas in preview pixel
        coordinates. The canvas itself handles Detection-style zoom/pan.
        """
        try:
            if not hasattr(self, "image_canvas") or not hasattr(self.image_canvas, "clear_overlays"):
                return
            self.image_canvas.clear_overlays()
            for li, layer in enumerate(self.layers):
                gdf = layer.get("gdf")
                if gdf is None or len(gdf) == 0:
                    continue
                draw_gdf = _filtered_gdf(gdf, self.current_min_conf(), self.current_class_filter())
                if draw_gdf is None or len(draw_gdf) == 0:
                    continue
                pix_gdf = _geom_to_pixel_gdf(draw_gdf, self.image_meta)
                try:
                    self._last_image_pixel_mode = str(getattr(pix_gdf, "attrs", {}).get("mustatil_pixel_mode", "unknown"))
                    self._last_image_pixel_candidates = getattr(pix_gdf, "attrs", {}).get("mustatil_pixel_candidates", [])
                except Exception:
                    pass
                all_index = list(gdf.index)
                for src_idx, row in zip(list(draw_gdf.index), pix_gdf.itertuples()):
                    try:
                        ri = int(all_index.index(src_idx))
                    except Exception:
                        ri = 0
                    try:
                        geom = getattr(row, "geometry")
                    except Exception:
                        continue
                    b = self._image_geom_bounds(geom)
                    if not b:
                        continue
                    x1, y1, x2, y2 = b
                    status = ""
                    try:
                        status = str(gdf.iloc[ri].get("review_status", "") or "")
                    except Exception:
                        pass
                    self.image_canvas.add_feature_rect(self, li, ri, x1, y1, x2, y2, status)
            try:
                for it in getattr(self.image_canvas, "overlay_items", []) or []:
                    try:
                        # Re-apply after scene insertion to keep pen cosmetic and z-order high.
                        it.apply_style(self._image_item_key(it) in self._image_selected())
                        it.setZValue(1000000 if self._image_item_key(it) in self._image_selected() else 999000)
                    except Exception:
                        pass
            except Exception:
                pass
            self._image_style_selected_items()
            mode = str(getattr(self, "_last_image_pixel_mode", "unknown"))
            self.set_info(
                f"Image overlays redrawn. Visible boxes fix active: high Z cosmetic outlines above TIFF. Right-drag selects multiple boxes.\n"
                f"Box alignment mode: {mode}\n"
                f"Confidence >= {self.current_min_conf():.2f}, class={self.current_class_filter()}."
            )
        except Exception as exc:
            self.set_info(f"Could not draw image overlays:\\n{exc}")

    # ------------------------------------------------------------------
    # Undo / Back
    # ------------------------------------------------------------------

    def _snapshot_layers_for_undo(self, label: str = "action"):
        """
        v25: snapshot all GIS Review layers before mutating actions.

        Full layer snapshots are intentionally used because right-click actions
        can mark, delete, copy, move, or create layers across arbitrary rows.
        """
        try:
            snap_layers = []
            for layer in getattr(self, "layers", []) or []:
                item = dict(layer)
                gdf = layer.get("gdf")
                try:
                    item["gdf"] = gdf.copy(deep=True) if gdf is not None else gdf
                except Exception:
                    try:
                        item["gdf"] = gdf.copy() if gdf is not None else gdf
                    except Exception:
                        item["gdf"] = gdf
                snap_layers.append(item)

            snap = {
                "label": str(label or "action"),
                "layers": snap_layers,
                "current_layer_idx": self.current_layer_idx,
                "selected_keys": set(getattr(self, "selected_keys", set()) or set()),
                "image_selected_keys": set(getattr(self, "image_selected_keys", set()) or set()),
            }
            self.undo_stack.append(snap)
            limit = int(getattr(self, "undo_limit", 25) or 25)
            if len(self.undo_stack) > limit:
                self.undo_stack = self.undo_stack[-limit:]
            try:
                self.btn_undo.setEnabled(True)
            except Exception:
                pass
        except Exception as exc:
            try:
                self.set_info(f"Undo snapshot failed:\\n{exc}")
            except Exception:
                pass

    def undo_last_action(self):
        """
        Restore the previous GIS Review state.
        """
        try:
            if not getattr(self, "undo_stack", None):
                self.set_info("Undo: nothing to undo.")
                return

            snap = self.undo_stack.pop()
            self.layers = snap.get("layers", [])
            self.current_layer_idx = snap.get("current_layer_idx", None)
            self.selected_keys = snap.get("selected_keys", set()) or set()
            self.image_selected_keys = snap.get("image_selected_keys", set()) or set()

            try:
                self.workspace.mustatil_layers = self.layers
            except Exception:
                pass
            try:
                self.refresh_layer_tree()
            except Exception:
                pass
            try:
                self.refresh_attribute_table()
            except Exception:
                pass
            try:
                self.rebuild_class_combo()
            except Exception:
                pass
            try:
                self._image_style_selected_items()
            except Exception:
                pass
            try:
                self.btn_undo.setEnabled(bool(self.undo_stack))
            except Exception:
                pass

            self.set_info(
                f"Undo restored: {snap.get('label', 'action')}\\n"
                "Press Redraw to refresh map/image overlays."
            )
        except Exception as exc:
            try:
                self.set_info(f"Undo failed:\\n{exc}\\n\\n{traceback.format_exc()}")
            except Exception:
                pass

    def _image_key(self, layer_idx: int, row_idx: int) -> str:
        return f"{int(layer_idx)}:{int(row_idx)}"

    def _image_selected(self):
        try:
            if not hasattr(self, "image_selected_keys"):
                self.image_selected_keys = set()
            return self.image_selected_keys
        except Exception:
            self.image_selected_keys = set()
            return self.image_selected_keys

    def _image_item_key(self, item):
        return self._image_key(getattr(item, "layer_idx", -1), getattr(item, "row_idx", -1))

    def _image_find_item(self, key: str):
        try:
            li, ri = [int(x) for x in str(key).split(":")]
            for item in getattr(self.image_canvas, "overlay_items", []) or []:
                try:
                    if int(item.layer_idx) == li and int(item.row_idx) == ri:
                        return item
                except Exception:
                    pass
        except Exception:
            pass
        return None

    def _image_style_selected_items(self):
        selected = set(self._image_selected())
        for it in getattr(self.image_canvas, "overlay_items", []) or []:
            try:
                if hasattr(it, "apply_style"):
                    it.apply_style(self._image_item_key(it) in selected)
            except Exception:
                pass

    def _image_select_only(self, layer_idx: int, row_idx: int):
        try:
            self.image_selected_keys = {self._image_key(layer_idx, row_idx)}
            self._image_style_selected_items()
        except Exception:
            pass

    def _image_select_by_rect(self, scene_rect, additive: bool = False):
        try:
            if not additive:
                self.image_selected_keys = set()
            selected = self._image_selected()
            count = 0
            for item in getattr(self.image_canvas, "overlay_items", []) or []:
                try:
                    if item.sceneBoundingRect().intersects(scene_rect):
                        selected.add(self._image_item_key(item))
                        count += 1
                except Exception:
                    pass
            self._image_style_selected_items()
            self.set_info(
                f"Right-drag selected {count} detection box(es).\n"
                "Right-click a selected box or empty image area to edit them together."
            )
            return count
        except Exception as exc:
            self.set_info(f"Right-drag selection failed:\n{exc}")
            return 0

    def _image_single_menu(self, layer_idx: int, row_idx: int, global_pos):
        try:
            from PySide6.QtWidgets import QMenu
            self.select_feature(layer_idx, row_idx, from_map=True)
            self._image_select_only(layer_idx, row_idx)

            menu = QMenu()
            act_undo = menu.addAction("Undo last action")
            menu.addSeparator()
            act_pos = menu.addAction("Mark correct / positive")
            act_fp = menu.addAction("Mark false_positive")
            act_unc = menu.addAction("Mark uncertain")
            act_need = menu.addAction("Mark needs_review")
            act_conf = menu.addAction("Mark confirmed_archaeology")
            menu.addSeparator()
            act_del = menu.addAction("Delete feature")
            act_export = menu.addAction("Export this feature")
            act = menu.exec(global_pos)

            if act == act_undo:
                self.undo_last_action()
            elif act == act_pos:
                self.mark_feature(layer_idx, row_idx, "correct")
            elif act == act_fp:
                self.mark_feature(layer_idx, row_idx, "false_positive")
            elif act == act_unc:
                self.mark_feature(layer_idx, row_idx, "uncertain")
            elif act == act_need:
                self.mark_feature(layer_idx, row_idx, "needs_review")
            elif act == act_conf:
                self.mark_feature(layer_idx, row_idx, "confirmed_archaeology")
            elif act == act_del:
                self.delete_feature(layer_idx, row_idx)
            elif act == act_export:
                self.export_feature_dialog(layer_idx, row_idx)
        except Exception as exc:
            self.set_info(f"Single box menu failed:\\n{exc}")

    def _image_batch_menu(self, global_pos):
        try:
            from PySide6.QtWidgets import QMenu
            keys = sorted(self._image_selected())
            if not keys:
                self.set_info("No selected boxes.")
                return
            menu = QMenu()
            title = menu.addAction(f"Selected boxes: {len(keys)}")
            title.setEnabled(False)
            act_undo = menu.addAction("Undo last action")
            menu.addSeparator()
            act_pos = menu.addAction("Mark selected correct / positive")
            act_fp = menu.addAction("Mark selected false_positive")
            act_unc = menu.addAction("Mark selected uncertain")
            act_need = menu.addAction("Mark selected needs_review")
            act_conf = menu.addAction("Mark selected confirmed_archaeology")
            menu.addSeparator()
            act_hidden = menu.addAction("Mark selected hidden")
            act_delete = menu.addAction("Delete selected")
            menu.addSeparator()
            act_copy = menu.addAction("Copy selected to review layer")
            act_move = menu.addAction("Move selected to review layer")
            act_export = menu.addAction("Export selected")
            menu.addSeparator()
            act_clear = menu.addAction("Clear selection")
            act = menu.exec(global_pos)
            if act is None:
                return
            if act == act_undo:
                self.undo_last_action()
            elif act == act_pos:
                self.bulk_mark(keys, "correct")
            elif act == act_fp:
                self.bulk_mark(keys, "false_positive")
            elif act == act_unc:
                self.bulk_mark(keys, "uncertain")
            elif act == act_need:
                self.bulk_mark(keys, "needs_review")
            elif act == act_conf:
                self.bulk_mark(keys, "confirmed_archaeology")
            elif act == act_hidden:
                self.bulk_mark(keys, "hidden")
            elif act == act_delete:
                self.bulk_delete(keys)
                self.image_selected_keys = set()
            elif act == act_copy:
                self.bulk_copy_move(keys, copy=True)
            elif act == act_move:
                self.bulk_copy_move(keys, copy=False)
                self.image_selected_keys = set()
            elif act == act_export:
                self.bulk_export(keys)
            elif act == act_clear:
                self.image_selected_keys = set()
                self._image_style_selected_items()
                return
            try:
                self.redraw_overlays()
                if act not in (act_delete, act_move, act_clear):
                    self.image_selected_keys = set(keys)
                    self._image_style_selected_items()
            except Exception:
                pass
            self.set_info(f"Batch action applied to {len(keys)} selected box(es).")
        except Exception as exc:
            self.set_info(f"Batch menu failed:\n{exc}\n\n{traceback.format_exc()}")

    def push_layer_to_map(self, layer_idx: int, fit: bool = False):
        if self.view_mode == "image":
            self.redraw_image_overlays()
            return
        if not (0 <= layer_idx < len(self.layers)):
            return
        try:
            layer = self.layers[layer_idx]
            gdf = layer["gdf"]
            if gdf is None or len(gdf) == 0:
                return
            geojson = _to_geojson_with_ids(
                gdf,
                layer_idx,
                mode=self.view_mode,
                image_meta=self.image_meta,
                min_conf=self.current_min_conf(),
                class_filter=self.current_class_filter(),
            )

            features = list(geojson.get("features", []) or [])
            total = len(features)
            layer_name = str(layer.get("name", "layer"))

            # v24: For 3000+ features, never send one huge JS payload only.
            # Send chunks so all features are drawn reliably in QtWebEngine.
            chunk_size = 800
            self.js(
                "try { Object.keys(layerGroups || {}).forEach(function(k){ "
                f"if (String(k) === '{int(layer_idx)}' || String(k).indexOf('{int(layer_idx)}_') === 0) "
                "{ try { map.removeLayer(layerGroups[k]); } catch(e){} delete layerGroups[k]; } "
                "}); } catch(e){}"
            )

            if total > chunk_size:
                base = {k: v for k, v in geojson.items() if k != "features"}
                for ci in range(0, total, chunk_size):
                    part = dict(base)
                    part["type"] = "FeatureCollection"
                    part["features"] = features[ci:ci + chunk_size]
                    key = f"{int(layer_idx)}_{ci // chunk_size:04d}"
                    js = (
                        f"window.addLayerGeoJSON({json.dumps(key)}, "
                        f"{json.dumps(layer_name + ' part ' + str(ci // chunk_size + 1))}, "
                        f"{json.dumps(part)}, false);"
                    )
                    self.js(js)
                if fit:
                    self.js("try { fitAllOverlays(); } catch(e){}")
                self.set_info(f"Layer drawn in chunks: {layer_name}\nFeatures shown: {total}\nChunk size: {chunk_size}")
            else:
                js = (
                    f"window.addLayerGeoJSON({int(layer_idx)}, "
                    f"{json.dumps(layer_name)}, "
                    f"{json.dumps(geojson)}, "
                    f"{str(bool(fit)).lower()});"
                )
                self.js(js)
        except Exception as exc:
            self.set_info(f"Could not draw layer on map:\n{exc}")

    def redraw_overlays(self):
        if self.view_mode == "image":
            self.redraw_image_overlays()
            return
        self.js("if (window.clearOverlays) { window.clearOverlays(); }")
        total_raw = 0
        total_filtered = 0
        for idx, layer in enumerate(self.layers):
            try:
                gdf = layer.get("gdf")
                if gdf is not None:
                    total_raw += len(gdf)
                    total_filtered += len(_filtered_gdf(gdf, self.current_min_conf(), self.current_class_filter()))
            except Exception:
                pass
            self.push_layer_to_map(idx, fit=False)
        self.set_info(
            f"Overlays redrawn manually.\n"
            f"Features shown: {total_filtered} of {total_raw}\n"
            f"Show all features: {self.show_all_enabled()}\n"
            f"Confidence >= {self.current_min_conf():.2f}, class={self.current_class_filter()}."
        )

    def apply_geo_nms_current_layer(self):
        if self.current_layer_idx is None or not (0 <= self.current_layer_idx < len(self.layers)):
            self.set_info("Select a layer first for Geo-NMS.")
            return
        try:
            src_layer = self.layers[self.current_layer_idx]
            src_gdf = src_layer["gdf"]
            before = len(src_gdf)
            filtered_input = _filtered_gdf(src_gdf, self.current_min_conf(), self.current_class_filter())
            filtered = _apply_geo_nms_gdf(filtered_input, float(self.nms_iou.value()))
            after = len(filtered)
            name = f"{src_layer.get('name','layer')}_geo_nms"
            self.layers.append({"name": name, "gdf": filtered, "path": "memory://geo_nms", "layer_name": None, "visible": True})
            self.current_layer_idx = len(self.layers) - 1
            self.refresh_layer_tree()
            self.refresh_attribute_table()
            self.rebuild_class_combo()
            self.push_layer_to_map(self.current_layer_idx, fit=True)
            self.set_info(f"Geo-NMS created new layer: {name}\nBefore: {before}\nAfter filters+NMS: {after}\nIoU threshold: {self.nms_iou.value():.2f}")
        except Exception as exc:
            self.set_info(f"Geo-NMS failed:\n{exc}\n\n{traceback.format_exc()}")

    # ------------------------------------------------------------------
    # FormLearner
    # ------------------------------------------------------------------

    def on_formscore_slider_changed(self, value):
        """
        v24: FormScore slider updates data only.

        No automatic redraw. User must press Redraw explicitly.
        Minimum is 0.01. The current slider value is applied to all features of
        the current FormLearner/scored layer.
        """
        try:
            score = max(0.01, float(value) / 100.0)
            self.formlearner_score_label.setText(f"{score:.2f}")

            changed = 0
            if self.current_layer_idx is not None and 0 <= self.current_layer_idx < len(self.layers):
                gdf = self.layers[self.current_layer_idx].get("gdf")
                if gdf is not None and len(gdf) > 0 and ("form_score" in gdf.columns or "form_match" in gdf.columns):
                    gdf["form_score"] = float(score)
                    gdf["form_threshold"] = float(score)
                    gdf["form_match"] = True
                    changed = len(gdf)
                    try:
                        self.refresh_attribute_table()
                    except Exception:
                        pass

            self.set_info(
                f"FormScore set to {score:.2f} for {changed} feature(s).\n"
                "Press Redraw to update overlays."
            )
        except Exception as exc:
            try:
                self.set_info(f"FormScore slider update failed:\\n{exc}")
            except Exception:
                pass

    def choose_formlearner_custom_model(self):
        try:
            from PySide6.QtWidgets import QFileDialog
            path, _ = QFileDialog.getOpenFileName(
                self.widget,
                "Choose custom FormLearner model/config",
                str(Path.home()),
                "FormLearner model/config (*.json *.pkl *.joblib *.onnx *.pt *.pth *.yaml *.yml);;All files (*.*)"
            )
            if not path:
                return
            self.formlearner_custom_model_path = str(path)
            try:
                self.formlearner_model_label.setText(Path(path).name[:24])
                self.formlearner_model_label.setToolTip(str(path))
                ix = self.formlearner_combo.findText("Custom model")
                if ix >= 0:
                    self.formlearner_combo.setCurrentIndex(ix)
            except Exception:
                pass
            self.set_info(f"Custom FormLearner model selected:\\n{path}")
        except Exception as exc:
            self.set_info(f"Could not choose custom FormLearner model:\\n{exc}")

    def _custom_formlearner_score(self, metrics: dict, settings: dict):
        """
        Best-effort custom model support.

        Supported immediately:
          - JSON config with optional fields:
              type: "mustatil" / "rectangle" / "round" / "balanced"
              weights: {"aspect":0.4, "rectangularity":0.4, "compactness":0.2}
              aspect_min / aspect_max

        If a workspace method run_custom_formlearner_model exists, it is tried
        at the layer level in run_formlearner(). Otherwise non-JSON model files
        are recorded but geometry fallback is used.
        """
        path = str(settings.get("custom_model_path", "") or "")
        if not path:
            return None
        try:
            p = Path(path)
            if p.suffix.lower() == ".json" and p.exists():
                data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                model_type = str(data.get("type", "") or data.get("formlearner", "") or "").lower()
                if model_type:
                    base = self._formlearner_score(metrics, model_type)
                else:
                    base = None

                weights = data.get("weights") or {}
                if isinstance(weights, dict) and weights:
                    aspect = float(metrics.get("aspect", 1.0) or 1.0)
                    rect = float(metrics.get("rectangularity", 0.0) or 0.0)
                    compact = float(metrics.get("compactness", 0.0) or 0.0)
                    aspect_min = float(data.get("aspect_min", 1.0) or 1.0)
                    aspect_max = float(data.get("aspect_max", 8.0) or 8.0)
                    aspect_score = max(0.0, min(1.0, (aspect - aspect_min) / max(1e-6, aspect_max - aspect_min)))
                    score = (
                        float(weights.get("aspect", 0.0)) * aspect_score +
                        float(weights.get("rectangularity", 0.0)) * rect +
                        float(weights.get("compactness", 0.0)) * compact
                    )
                    wsum = sum(abs(float(v)) for v in weights.values()) or 1.0
                    score = score / wsum
                    if base is not None:
                        score = 0.5 * float(base) + 0.5 * float(score)
                    return max(0.0, min(1.0, float(score)))

                if base is not None:
                    return max(0.0, min(1.0, float(base)))
        except Exception:
            return None
        return None

    def run_formlearner_inline(self):
        """
        v15: no extra window. The toolbar controls directly run FormLearner.
        """
        try:
            learner_name = str(self.formlearner_combo.currentText())
            custom_path = str(getattr(self, "formlearner_custom_model_path", "") or "")
            if learner_name == "Custom model" and not custom_path:
                self.set_info("FormLearner: choose a custom model file first.")
                return
            settings = {
                "formlearner": learner_name,
                "custom_model_path": custom_path,
                "use_sam2": bool(self.formlearner_sam2.isChecked()),
                "score_threshold": max(0.01, float(self.formlearner_score_slider.value()) / 100.0),
                "use_filtered_if_empty": True,
            }
            try:
                p = self._formlearner_settings_path()
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
            self.run_formlearner(settings)
        except Exception as exc:
            self.set_info(f"Inline FormLearner failed:\\n{exc}\\n\\n{traceback.format_exc()}")

    def open_formlearner_dialog(self):
        """
        Small FormLearner window:
          - choose FormLearner type
          - checkbox: use SAM2 refinement
          - slider: score threshold
          - run on selected boxes, selected table rows, or filtered current layer
        """
        try:
            from PySide6.QtWidgets import (
                QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
                QComboBox, QCheckBox, QSlider, QPushButton, QDialogButtonBox
            )
            from PySide6.QtCore import Qt

            dlg = QDialog(self.widget)
            dlg.setWindowTitle("FormLearner")
            dlg.resize(540, 280)
            root = QVBoxLayout(dlg)

            info = QLabel(
                "Choose a FormLearner, optionally enable SAM2 refinement, "
                "then set the minimum form score. The result is written as a new layer."
            )
            info.setWordWrap(True)
            root.addWidget(info)

            form = QFormLayout()
            root.addLayout(form)

            combo = QComboBox()
            combo.addItems([
                "Mustatil / long rectangle",
                "Rectangle / building-like",
                "Round cairn / mound",
                "Ring / enclosure",
                "Compact object",
                "All shapes balanced",
            ])

            try:
                ws = self.workspace or _find_workspace(self.widget)
                names = []
                for name in dir(ws):
                    lname = str(name).lower()
                    if "form" in lname and "learn" in lname and callable(getattr(ws, name, None)):
                        names.append(name)
                for n in sorted(set(names)):
                    combo.addItem(f"Workspace: {n}")
            except Exception:
                pass

            form.addRow("FormLearner:", combo)

            sam2_check = QCheckBox("Use SAM2 refinement if available")
            sam2_check.setChecked(False)
            form.addRow("SAM2:", sam2_check)

            score_slider = QSlider(Qt.Horizontal)
            score_slider.setRange(0, 100)
            score_slider.setValue(60)
            score_label = QLabel("0.60")

            def _score_changed(v):
                score_label.setText(f"{float(v)/100.0:.2f}")

            score_slider.valueChanged.connect(_score_changed)
            srow = QHBoxLayout()
            srow.addWidget(score_slider, 1)
            srow.addWidget(score_label)
            form.addRow("Score threshold:", srow)

            run_visible = QCheckBox("Use filtered current layer if no boxes are selected")
            run_visible.setChecked(True)
            form.addRow("Input:", run_visible)

            buttons = QDialogButtonBox()
            save_btn = buttons.addButton("Save Settings", QDialogButtonBox.ActionRole)
            buttons.addButton("Run FormLearner", QDialogButtonBox.AcceptRole)
            buttons.addButton(QDialogButtonBox.Cancel)
            root.addWidget(buttons)

            def _settings():
                return {
                    "formlearner": str(combo.currentText()),
                    "use_sam2": bool(sam2_check.isChecked()),
                    "score_threshold": float(score_slider.value()) / 100.0,
                    "use_filtered_if_empty": bool(run_visible.isChecked()),
                }

            def _save_settings():
                try:
                    p = self._formlearner_settings_path()
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(json.dumps(_settings(), indent=2, ensure_ascii=False), encoding="utf-8")
                    self.set_info(f"FormLearner settings saved:\\n{p}")
                except Exception as exc:
                    self.set_info(f"Could not save FormLearner settings:\\n{exc}")

            save_btn.clicked.connect(_save_settings)
            buttons.accepted.connect(dlg.accept)
            buttons.rejected.connect(dlg.reject)

            try:
                p = self._formlearner_settings_path()
                if p.exists():
                    data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                    ix = combo.findText(str(data.get("formlearner", "")))
                    if ix >= 0:
                        combo.setCurrentIndex(ix)
                    sam2_check.setChecked(bool(data.get("use_sam2", False)))
                    score_slider.setValue(int(round(float(data.get("score_threshold", 0.60)) * 100)))
                    run_visible.setChecked(bool(data.get("use_filtered_if_empty", True)))
            except Exception:
                pass

            if dlg.exec() != QDialog.Accepted:
                return

            settings = _settings()
            try:
                p = self._formlearner_settings_path()
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
            self.run_formlearner(settings)
        except Exception as exc:
            self.set_info(f"FormLearner dialog failed:\\n{exc}\\n\\n{traceback.format_exc()}")

    def _formlearner_settings_path(self) -> Path:
        try:
            return self.project_dir_for_saving() / "mustatil_formlearner_settings.json"
        except Exception:
            return Path.home() / "mustatil_formlearner_settings.json"

    def _formlearner_input_gdf(self, use_filtered_if_empty: bool = True):
        """
        Input order:
          1. image right-drag selected boxes
          2. selected table rows
          3. filtered current layer
        """
        if self.current_layer_idx is None or not (0 <= self.current_layer_idx < len(self.layers)):
            return None, "No current layer selected."

        layer = self.layers[self.current_layer_idx]
        gdf = layer.get("gdf")
        if gdf is None or len(gdf) == 0:
            return None, "Current layer is empty."

        keys = []
        try:
            keys = sorted(getattr(self, "image_selected_keys", set()) or [])
        except Exception:
            keys = []

        if keys:
            try:
                import pandas as pd
                import geopandas as gpd
                frames = []
                out_crs = None
                for key in keys:
                    try:
                        li, ri = [int(x) for x in str(key).split(":")]
                        if 0 <= li < len(self.layers):
                            kgdf = self.layers[li]["gdf"]
                            if 0 <= ri < len(kgdf):
                                piece = kgdf.iloc[[ri]].copy()
                                piece["_source_layer"] = li
                                piece["_source_row"] = ri
                                frames.append(piece)
                                out_crs = kgdf.crs
                    except Exception:
                        pass
                if frames:
                    out = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=out_crs)
                    return out, f"selected image boxes: {len(out)}"
            except Exception:
                pass

        rows = self.selected_rows()
        if rows:
            try:
                subset = gdf.iloc[[r for r in rows if 0 <= r < len(gdf)]].copy()
                subset["_source_layer"] = self.current_layer_idx
                subset["_source_row"] = [r for r in rows if 0 <= r < len(gdf)]
                return subset, f"selected table rows: {len(subset)}"
            except Exception:
                pass

        if use_filtered_if_empty:
            try:
                if self.show_all_enabled():
                    subset = gdf.copy()
                    subset["_source_layer"] = self.current_layer_idx
                    subset["_source_row"] = list(range(len(subset)))
                    return subset, f"all current layer features: {len(subset)}"
                subset = _filtered_gdf(gdf, self.current_min_conf(), self.current_class_filter()).copy()
                subset["_source_layer"] = self.current_layer_idx
                subset["_source_row"] = list(range(len(subset)))
                return subset, f"filtered current layer: {len(subset)}"
            except Exception:
                return gdf.copy(), f"current layer: {len(gdf)}"

        return None, "No boxes selected."

    def _geometry_shape_metrics(self, geom):
        try:
            area = float(geom.area) if geom is not None else 0.0
            per = float(geom.length) if geom is not None else 0.0
            minx, miny, maxx, maxy = [float(v) for v in geom.bounds]
            w = max(1e-9, maxx - minx)
            h = max(1e-9, maxy - miny)
            bbox_area = max(1e-9, w * h)
            aspect = max(w, h) / max(1e-9, min(w, h))
            rectangularity = max(0.0, min(1.0, area / bbox_area)) if area > 0 else 1.0
            compactness = 0.0
            if per > 0 and area > 0:
                compactness = max(0.0, min(1.0, 4.0 * math.pi * area / (per * per)))
            return {
                "area": area,
                "perimeter": per,
                "width": w,
                "height": h,
                "bbox_area": bbox_area,
                "aspect": aspect,
                "rectangularity": rectangularity,
                "compactness": compactness,
            }
        except Exception:
            return {
                "area": 0.0, "perimeter": 0.0, "width": 0.0, "height": 0.0,
                "bbox_area": 0.0, "aspect": 1.0, "rectangularity": 0.0, "compactness": 0.0,
            }

    def _formlearner_score(self, metrics: dict, learner_name: str) -> float:
        name = str(learner_name or "").lower()
        aspect = float(metrics.get("aspect", 1.0) or 1.0)
        rect = float(metrics.get("rectangularity", 0.0) or 0.0)
        compact = float(metrics.get("compactness", 0.0) or 0.0)

        if "workspace:" in name:
            name = "all shapes balanced"

        if "mustatil" in name or "long rectangle" in name:
            aspect_score = max(0.0, min(1.0, (aspect - 1.8) / 5.2))
            return max(0.0, min(1.0, 0.70 * aspect_score + 0.30 * rect))

        if "rectangle" in name or "building" in name:
            aspect_target = max(0.0, 1.0 - abs(aspect - 1.8) / 4.0)
            return max(0.0, min(1.0, 0.65 * rect + 0.35 * aspect_target))

        if "round" in name or "cairn" in name or "mound" in name:
            aspect_round = max(0.0, 1.0 - abs(aspect - 1.0) / 1.8)
            return max(0.0, min(1.0, 0.65 * compact + 0.35 * aspect_round))

        if "ring" in name or "enclosure" in name:
            ring_fill = max(0.0, 1.0 - abs(rect - 0.55) / 0.55)
            aspect_round = max(0.0, 1.0 - abs(aspect - 1.0) / 2.5)
            return max(0.0, min(1.0, 0.50 * ring_fill + 0.30 * aspect_round + 0.20 * compact))

        if "compact" in name:
            aspect_compact = max(0.0, 1.0 - abs(aspect - 1.0) / 2.0)
            return max(0.0, min(1.0, 0.50 * compact + 0.30 * aspect_compact + 0.20 * rect))

        aspect_ok = max(0.0, min(1.0, 1.0 / max(1.0, (aspect / 6.0))))
        return max(0.0, min(1.0, 0.34 * rect + 0.33 * compact + 0.33 * aspect_ok))

    def _try_sam2_refine_formlearner(self, gdf, settings: dict):
        if not settings.get("use_sam2", False):
            gdf["sam2_used"] = False
            return gdf, "SAM2 disabled."

        ws = self.workspace or _find_workspace(self.widget)
        for name in [
            "sam2_refine_formlearner",
            "formlearner_sam2_refine",
            "sam2_refine_shapes",
            "run_sam2_on_selected_shapes",
            "run_sam2_formlearner",
        ]:
            try:
                fn = getattr(ws, name, None)
                if callable(fn):
                    result = fn(gdf)
                    if result is not None:
                        result["sam2_used"] = True
                        return result, f"SAM2 refinement used: {name}"
                    gdf["sam2_used"] = True
                    return gdf, f"SAM2 method called: {name}"
            except Exception as exc:
                return gdf, f"SAM2 method {name} failed: {exc}"

        gdf["sam2_used"] = False
        return gdf, "SAM2 requested, but no SAM2 refinement hook was found. Geometry-only scores used."

    def run_formlearner(self, settings: dict):
        self._snapshot_layers_for_undo("run FormLearner")
        try:
            input_gdf, source_msg = self._formlearner_input_gdf(bool(settings.get("use_filtered_if_empty", True)))
            if input_gdf is None or len(input_gdf) == 0:
                self.set_info(f"FormLearner: no input. {source_msg}")
                return

            learner = str(settings.get("formlearner") or "All shapes balanced")
            threshold = float(settings.get("score_threshold", 0.60) or 0.60)
            custom_model_path = str(settings.get("custom_model_path", "") or "")

            gdf = input_gdf.copy()

            custom_runtime_msg = ""
            try:
                ws = self.workspace or _find_workspace(self.widget)
                fn = getattr(ws, "run_custom_formlearner_model", None)
                if callable(fn) and custom_model_path:
                    result = fn(gdf, custom_model_path)
                    if result is not None:
                        gdf = result
                        custom_runtime_msg = "Workspace custom FormLearner runtime used."
            except Exception as exc:
                custom_runtime_msg = f"Workspace custom FormLearner runtime failed: {exc}"

            gdf, sam2_msg = self._try_sam2_refine_formlearner(gdf, settings)

            # v20: FormScore is controlled directly by the slider, like a
            # detection confidence value. Apply the slider value to all selected
            # or filtered objects. Minimum is 0.01.
            slider_score = max(0.01, float(threshold))

            aspects = []
            rectangularities = []
            compactnesses = []
            areas = []
            for geom in gdf.geometry:
                m = self._geometry_shape_metrics(geom)
                aspects.append(float(m.get("aspect", 1.0)))
                rectangularities.append(float(m.get("rectangularity", 0.0)))
                compactnesses.append(float(m.get("compactness", 0.0)))
                areas.append(float(m.get("area", 0.0)))

            gdf["formlearner"] = learner
            gdf["form_custom_model"] = custom_model_path
            gdf["form_score"] = float(slider_score)
            gdf["form_match"] = True
            gdf["form_threshold"] = float(slider_score)
            gdf["form_aspect"] = aspects
            gdf["form_rectangularity"] = rectangularities
            gdf["form_compactness"] = compactnesses
            gdf["form_area"] = areas

            matched = gdf[gdf["form_match"] == True].copy()
            if len(matched) == 0:
                out = gdf.copy()
                suffix = "scored"
            else:
                out = matched
                suffix = "matches"

            name = f"formlearner_{_sanitize_layer_name(learner)}_{suffix}"
            self.layers.append({
                "name": name,
                "gdf": out,
                "path": "memory://formlearner",
                "layer_name": None,
                "visible": True,
            })
            self.current_layer_idx = len(self.layers) - 1
            self.refresh_layer_tree()
            self.refresh_attribute_table()
            self.rebuild_class_combo()
            # v24: no automatic redraw after FormLearner. User must press Redraw.

            self.set_info(
                "FormLearner finished. Press Redraw to show/update overlays.\\n"
                f"Input: {source_msg}\\n"
                f"FormLearner: {learner}\\n"
                f"SAM2: {sam2_msg}\\n"
                f"Custom model: {custom_model_path or 'none'}\\n"
                f"{custom_runtime_msg}\\n"
                f"Score threshold: {threshold:.2f}\\n"
                f"Scored: {len(gdf)}\\n"
                f"Matches: {int(sum(gdf['form_match']))}\\n"
                f"New layer: {name}"
            )
        except Exception as exc:
            self.set_info(f"FormLearner failed:\\n{exc}\\n\\n{traceback.format_exc()}")

    # ------------------------------------------------------------------
    # Annotator crops
    # ------------------------------------------------------------------

    def _mustatil_output_base(self) -> Path:
        """
        v22: always create annotator crops inside the current project folder.
        """
        try:
            base = self.project_dir_for_saving() / "annotator_crops"
            base.mkdir(parents=True, exist_ok=True)
            return base
        except Exception:
            pass
        base = Path.home() / "Mustatil_Project" / "annotator_crops"
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _collect_pos_fp_rows(self):
        """
        Use selected table rows if any. Otherwise use reviewed correct/positive and false_positive.
        """
        rows = []
        if self.current_layer_idx is not None:
            selected = self.selected_rows()
            if selected:
                for r in selected:
                    rows.append((self.current_layer_idx, r))
                return rows

        for li, layer in enumerate(self.layers):
            gdf = layer.get("gdf")
            if gdf is None:
                continue
            for ri in range(len(gdf)):
                try:
                    s = str(gdf.iloc[ri].get("review_status", "") or "").lower().strip()
                    if s in POSITIVE_STATUSES or s in FALSE_POSITIVE_STATUSES:
                        rows.append((li, ri))
                except Exception:
                    pass
        return rows

    def _row_target_class(self, gdf, row_idx: int) -> Optional[int]:
        try:
            s = str(gdf.iloc[row_idx].get("review_status", "") or "").lower().strip()
        except Exception:
            s = ""
        if s in FALSE_POSITIVE_STATUSES:
            return 1
        if s in POSITIVE_STATUSES:
            return 0
        # For selected rows without review status, default positive.
        return 0

    def _try_open_annotator_folder(self, out_dir: Path):
        """
        v22: directly select/open the crop folder in Annotator when possible.
        """
        ws = self.workspace or _find_workspace(self.widget)
        if ws is None:
            return False
        out_dir = Path(out_dir)
        try:
            ws.mustatil_last_annotator_crop_folder = str(out_dir)
            ws.mustatil_annotator_folder = str(out_dir)
            ws.mustatil_annotator_dataset_folder = str(out_dir)
        except Exception:
            pass

        try:
            tabs = getattr(ws, "tabs", None)
            if tabs is not None:
                for i in range(tabs.count()):
                    if "annotator" in str(tabs.tabText(i)).lower():
                        tabs.setCurrentIndex(i)
                        break
        except Exception:
            pass

        for attr in [
            "annotator_open_folder", "load_annotator_folder", "open_annotator_folder",
            "ann_open_folder", "annotator_load_dataset", "load_annotation_dataset",
            "annotator_load_folder", "load_crops_folder", "load_images_folder",
            "set_annotator_folder", "set_annotation_folder"
        ]:
            fn = getattr(ws, attr, None)
            if callable(fn):
                try:
                    fn(str(out_dir))
                    return True
                except TypeError:
                    try:
                        fn(Path(out_dir))
                        return True
                    except Exception:
                        pass
                except Exception:
                    pass

        try:
            from PySide6.QtWidgets import QLineEdit
            for le in ws.findChildren(QLineEdit):
                name = (str(le.objectName() or "") + " " + str(le.placeholderText() or "")).lower()
                if any(k in name for k in ["annotator", "annotation", "dataset", "image folder", "images folder", "crop"]):
                    le.setText(str(out_dir))
                    try:
                        le.editingFinished.emit()
                    except Exception:
                        pass
                    return True
        except Exception:
            pass
        return False

    def send_pos_fp_crops_to_annotator(self):
        """
        Creates raw satellite/map crops without drawing the rectangle itself.
        Also writes YOLO label files:
          class 0 = positive/correct/confirmed
          class 1 = false_positive
        """
        rows = self._collect_pos_fp_rows()
        if not rows:
            self.set_info(
                "No positive/false-positive features found.\n"
                "Select table rows first, or mark features as correct/positive/false_positive."
            )
            return

        template = self.selected_template()
        if not template:
            self.set_info("Choose a map/satellite server first. Crops are created from that map server.")
            return

        z = int(self.zoom_spin.value())
        out_base = self._mustatil_output_base()
        images_dir = out_base / "images"
        labels_dir = out_base / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        (out_base / "classes.txt").write_text("positive\nfalse_positive\n", encoding="utf-8")
        (out_base / "data.yaml").write_text(
            f"path: {out_base.as_posix()}\ntrain: images\nval: images\nnames:\n  0: positive\n  1: false_positive\n",
            encoding="utf-8",
        )

        manifest = []
        ok = 0
        errors = 0

        for li, ri in rows:
            try:
                if not (0 <= li < len(self.layers)):
                    continue
                layer = self.layers[li]
                gdf = layer["gdf"]
                if not (0 <= ri < len(gdf)):
                    continue

                if getattr(gdf, "crs", None) is None:
                    errors += 1
                    continue

                one = gdf.iloc[[ri]].to_crs("EPSG:4326")
                geom = one.iloc[0].geometry
                cls = self._row_target_class(gdf, ri)
                if cls is None:
                    continue

                crop, yolo = _render_satellite_crop_for_geom(
                    geom,
                    template=template,
                    z=z,
                    pad_factor=2.4,
                    out_size=512,
                )

                status = "false_positive" if cls == 1 else "positive"
                stem = f"{_sanitize_layer_name(layer.get('name','layer'))}_r{ri}_{status}"
                img_path = images_dir / f"{stem}.jpg"
                lbl_path = labels_dir / f"{stem}.txt"

                # RAW satellite crop: no rectangle drawn.
                crop.save(img_path, "JPEG", quality=92)

                # YOLO label stores the rectangle annotation, but the image itself stays clean.
                lbl_path.write_text(f"{cls} {yolo[0]:.6f} {yolo[1]:.6f} {yolo[2]:.6f} {yolo[3]:.6f}\n", encoding="utf-8")

                manifest.append({
                    "image": str(img_path),
                    "label": str(lbl_path),
                    "layer": layer.get("name"),
                    "row": ri,
                    "review_status": status,
                    "class_id": cls,
                    "zoom": z,
                    "server": self.server_combo.currentText(),
                    "template": template,
                    "note": "raw satellite crop without drawn rectangle; YOLO label contains the box",
                })
                ok += 1
            except Exception as exc:
                errors += 1
                _warn_once(f"crop_{li}_{ri}", f"Crop failed for {li}:{ri}: {exc}")

        (out_base / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        opened = self._try_open_annotator_folder(out_base)
        self.set_info(
            f"Created annotator crop dataset:\n{out_base}\n\n"
            f"Crops written: {ok}\nErrors/skipped: {errors}\n"
            "Images are raw satellite/map crops without drawn rectangles.\n"
            "YOLO labels contain the positive/false_positive boxes.\n\n"
            + ("Sent/opened in Annotator by available workspace hook." if opened else "Crops are inside the project folder; no direct Annotator hook found, so open/select this folder in the Annotator.")
        )

    # ------------------------------------------------------------------
    # Table / tree
    # ------------------------------------------------------------------

    def refresh_layer_tree(self):
        from PySide6.QtWidgets import QTreeWidgetItem
        from PySide6.QtCore import Qt
        self.layer_tree.blockSignals(True)
        self.layer_tree.clear()
        for i, layer in enumerate(self.layers):
            gdf = layer.get("gdf")
            item = QTreeWidgetItem([
                str(layer.get("name", f"Layer {i+1}")),
                str(len(gdf) if gdf is not None else 0),
                str(getattr(gdf, "crs", "") or ""),
                str(layer.get("path", "")),
            ])
            item.setData(0, Qt.UserRole, i)
            self.layer_tree.addTopLevelItem(item)
        self.layer_tree.blockSignals(False)

    def on_layer_selection_changed(self):
        item = self.layer_tree.currentItem()
        if item is None:
            return
        try:
            from PySide6.QtCore import Qt
            idx = int(item.data(0, Qt.UserRole))
        except Exception:
            idx = self.layer_tree.indexOfTopLevelItem(item)
        if 0 <= idx < len(self.layers):
            self.current_layer_idx = idx
            self.refresh_attribute_table()
            self.update_info_for_layer(idx)

    def refresh_attribute_table(self):
        from PySide6.QtWidgets import QTableWidgetItem
        if self.current_layer_idx is None or not (0 <= self.current_layer_idx < len(self.layers)):
            self.attr_table.clear()
            self.attr_table.setRowCount(0)
            self.attr_table.setColumnCount(0)
            return
        gdf = self.layers[self.current_layer_idx].get("gdf")
        if gdf is None:
            return
        try:
            max_rows = min(len(gdf), 3000)
            cols = [c for c in gdf.columns if c != "geometry"]
            if "review_status" in cols:
                cols.remove("review_status")
                cols = ["review_status"] + cols
            cols.append("geometry_wkt")
            self.attr_table.blockSignals(True)
            self.attr_table.setColumnCount(len(cols))
            self.attr_table.setRowCount(max_rows)
            self.attr_table.setHorizontalHeaderLabels(cols)
            for r in range(max_rows):
                row = gdf.iloc[r]
                for c, col in enumerate(cols):
                    if col == "geometry_wkt":
                        try:
                            val = row.geometry.wkt[:220]
                        except Exception:
                            val = ""
                    else:
                        try:
                            val = row.get(col, "")
                        except Exception:
                            val = ""
                    self.attr_table.setItem(r, c, QTableWidgetItem(str(val)))
            self.attr_table.blockSignals(False)
        except Exception as exc:
            self.attr_table.blockSignals(False)
            self.set_info(f"Could not refresh attribute table:\n{exc}")

    def on_table_selection_changed(self):
        if self.current_layer_idx is None:
            return
        rows = self.selected_rows()
        if len(rows) == 1:
            if self.view_mode == "image":
                self._image_select_only(int(self.current_layer_idx), int(rows[0]))
            else:
                self.js(f"zoomToFeature({int(self.current_layer_idx)}, {int(rows[0])}, false);")

    def selected_rows(self) -> List[int]:
        try:
            return sorted(set(idx.row() for idx in self.attr_table.selectionModel().selectedRows()))
        except Exception:
            return []

    def update_info_for_layer(self, idx: int):
        try:
            layer = self.layers[idx]
            gdf = layer.get("gdf")
            txt = [
                f"Layer: {layer.get('name')}",
                f"Features: {len(gdf) if gdf is not None else 0}",
                f"CRS: {getattr(gdf, 'crs', '')}",
                f"Source: {layer.get('path', '')}",
            ]
            if gdf is not None and "review_status" in gdf.columns:
                txt.append(f"Review counts: {gdf['review_status'].value_counts(dropna=False).to_dict()}")
            self.set_info("\n".join(txt))
        except Exception:
            pass

    def select_feature(self, layer_idx: int, row_idx: int, from_map: bool = False):
        try:
            if not (0 <= layer_idx < len(self.layers)):
                return
            self.current_layer_idx = layer_idx
            item = self.layer_tree.topLevelItem(layer_idx)
            if item:
                self.layer_tree.setCurrentItem(item)
            self.refresh_attribute_table()
            self.attr_table.selectRow(row_idx)
            self.update_info_for_feature(layer_idx, row_idx)
        except Exception:
            pass

    def update_info_for_feature(self, layer_idx: int, row_idx: int):
        try:
            gdf = self.layers[layer_idx]["gdf"]
            row = gdf.iloc[row_idx]
            vals = [f"Layer: {self.layers[layer_idx].get('name')}", f"Feature row: {row_idx}"]
            for col in gdf.columns:
                if col == "geometry":
                    continue
                vals.append(f"{col}: {row.get(col, '')}")
                if len(vals) > 22:
                    vals.append("...")
                    break
            try:
                vals.append(f"Geometry: {row.geometry.wkt[:300]}")
            except Exception:
                pass
            self.set_info("\n".join(vals))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Review actions
    # ------------------------------------------------------------------

    def mark_feature(self, layer_idx: int, row_idx: int, status: str, refresh_map: bool = True):
        if refresh_map:
            self._snapshot_layers_for_undo(f"mark feature as {status}")
        try:
            gdf = self.layers[layer_idx]["gdf"]
            if "review_status" not in gdf.columns:
                gdf["review_status"] = ""
            gdf.at[gdf.index[row_idx], "review_status"] = str(status)
            self.refresh_attribute_table()
            self.update_info_for_feature(layer_idx, row_idx)
            if refresh_map:
                self.push_layer_to_map(layer_idx, fit=False)
        except Exception as exc:
            self.set_info(f"Could not mark feature:\n{exc}")

    def set_class_name(self, layer_idx: int, row_idx: int, name: str):
        try:
            gdf = self.layers[layer_idx]["gdf"]
            col = "class_name" if "class_name" in gdf.columns else ("class" if "class" in gdf.columns else "class_name")
            if col not in gdf.columns:
                gdf[col] = ""
            gdf.at[gdf.index[row_idx], col] = str(name)
            self.refresh_attribute_table()
            self.rebuild_class_combo()
            self.set_info(f"Set {col} for row {row_idx}: {name}")
        except Exception as exc:
            self.set_info(f"Could not set class/name:\n{exc}")

    def delete_feature(self, layer_idx: int, row_idx: int, refresh_map: bool = True):
        if refresh_map:
            self._snapshot_layers_for_undo("delete feature")
        try:
            gdf = self.layers[layer_idx]["gdf"]
            self.layers[layer_idx]["gdf"] = gdf.drop(gdf.index[row_idx]).reset_index(drop=True)
            self.current_layer_idx = layer_idx
            self.refresh_layer_tree()
            self.refresh_attribute_table()
            if refresh_map:
                self.push_layer_to_map(layer_idx, fit=False)
            self.set_info(f"Deleted feature row {row_idx}.")
        except Exception as exc:
            self.set_info(f"Could not delete feature:\n{exc}")

    def mark_selected_rows(self):
        self._snapshot_layers_for_undo("mark selected rows")
        if self.current_layer_idx is None:
            return
        rows = self.selected_rows()
        if not rows:
            self.set_info("No selected rows.")
            return
        status = self.review_combo.currentText()
        for r in rows:
            self.mark_feature(self.current_layer_idx, r, status, refresh_map=False)
        self.push_layer_to_map(self.current_layer_idx, fit=False)
        self.set_info(f"Marked {len(rows)} row(s) as {status}.")

    def bulk_mark(self, keys: List[str], status: str):
        self._snapshot_layers_for_undo(f"bulk mark as {status}")
        count = 0
        touched = set()
        for k in keys:
            try:
                li, ri = [int(x) for x in str(k).split(":")]
                self.mark_feature(li, ri, status, refresh_map=False)
                touched.add(li)
                count += 1
            except Exception:
                pass
        for li in touched:
            self.push_layer_to_map(li, fit=False)
        self.set_info(f"Bulk marked {count} feature(s) as {status}.")

    def bulk_delete(self, keys: List[str]):
        self._snapshot_layers_for_undo("bulk delete")
        by_layer: Dict[int, List[int]] = {}
        for k in keys:
            try:
                li, ri = [int(x) for x in str(k).split(":")]
                by_layer.setdefault(li, []).append(ri)
            except Exception:
                pass
        for li, rows in by_layer.items():
            try:
                gdf = self.layers[li]["gdf"]
                idxs = [gdf.index[r] for r in sorted(set(rows), reverse=True) if r < len(gdf)]
                self.layers[li]["gdf"] = gdf.drop(idxs).reset_index(drop=True)
                self.push_layer_to_map(li, fit=False)
            except Exception:
                pass
        self.refresh_layer_tree()
        self.refresh_attribute_table()
        self.set_info(f"Bulk deleted {sum(len(v) for v in by_layer.values())} feature(s).")

    def copy_or_move_feature(self, layer_idx: int, row_idx: int, copy: bool = True, refresh_map: bool = True):
        self.bulk_copy_move([f"{layer_idx}:{row_idx}"], copy=copy)

    def copy_or_move_selected_rows(self, copy: bool = True):
        if self.current_layer_idx is None:
            return
        keys = [f"{self.current_layer_idx}:{r}" for r in self.selected_rows()]
        self.bulk_copy_move(keys, copy=copy)

    def bulk_copy_move(self, keys: List[str], copy: bool = True):
        if not keys:
            self.set_info("No selected features.")
            return
        try:
            from PySide6.QtWidgets import QInputDialog
            import pandas as pd
            name, ok = QInputDialog.getText(self.widget, "Target review layer", "Layer name:", text="reviewed_features")
            if not ok or not name:
                return

            self._snapshot_layers_for_undo(("copy" if copy else "move") + " selected to review layer")
            frames = []
            by_layer: Dict[int, List[int]] = {}
            for k in keys:
                try:
                    li, ri = [int(x) for x in str(k).split(":")]
                    gdf = self.layers[li]["gdf"]
                    if ri < len(gdf):
                        frames.append(gdf.iloc[[ri]].copy())
                        by_layer.setdefault(li, []).append(ri)
                except Exception:
                    pass
            if not frames:
                return

            import pandas as pd
            selected = pd.concat(frames, ignore_index=True)
            target_idx = None
            for i, layer in enumerate(self.layers):
                if layer.get("name") == name:
                    target_idx = i
                    break
            if target_idx is None:
                target_idx = len(self.layers)
                self.layers.append({"name": name, "gdf": selected, "path": "memory://review", "layer_name": None, "visible": True})
            else:
                self.layers[target_idx]["gdf"] = pd.concat([self.layers[target_idx]["gdf"], selected], ignore_index=True)

            if not copy:
                for li, rows in by_layer.items():
                    gdf = self.layers[li]["gdf"]
                    idxs = [gdf.index[r] for r in sorted(set(rows), reverse=True) if r < len(gdf)]
                    self.layers[li]["gdf"] = gdf.drop(idxs).reset_index(drop=True)
                    self.push_layer_to_map(li, fit=False)

            self.refresh_layer_tree()
            self.refresh_attribute_table()
            self.rebuild_class_combo()
            self.push_layer_to_map(target_idx, fit=False)
            self.set_info(("Copied" if copy else "Moved") + f" {len(selected)} feature(s) to {name}.")
        except Exception as exc:
            self.set_info(f"Bulk copy/move failed:\n{exc}")


    # ------------------------------------------------------------------
    # GMT / publication map
    # ------------------------------------------------------------------

    def _current_or_combined_layer(self):
        try:
            gpd = _gpd()
            if gpd is None:
                return None, "layer"
            if self.current_layer_idx is not None and 0 <= self.current_layer_idx < len(self.layers):
                layer = self.layers[self.current_layer_idx]
                return layer.get("gdf"), layer.get("name", "layer")
            frames = [l.get("gdf") for l in self.layers if l.get("gdf") is not None and len(l.get("gdf"))]
            if not frames:
                return None, "layer"
            import pandas as pd
            return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs), "all_layers"
        except Exception:
            return None, "layer"

    def _gmt_output_dir(self) -> Path:
        base = self._mustatil_output_base() if hasattr(self, "_mustatil_output_base") else (Path.home() / "Mustatil_review_annotator_crops")
        out = Path(base).parent / "gmt_maps"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def project_dir_for_saving(self) -> Path:
        """
        Best-effort Mustatil project directory.
        Falls back to the current layer source folder, then the user's home.
        """
        try:
            ws = self.workspace or _find_workspace(self.widget)
            for attr in ["project_dir", "project_folder", "project_path", "workdir"]:
                v = getattr(ws, attr, None) if ws is not None else None
                if v:
                    p = Path(str(v))
                    if p.suffix:
                        p = p.parent
                    p.mkdir(parents=True, exist_ok=True)
                    return p
        except Exception:
            pass

        try:
            if self.current_layer_idx is not None and 0 <= self.current_layer_idx < len(self.layers):
                p = Path(str(self.layers[self.current_layer_idx].get("path", "")))
                if p.exists():
                    return p.parent
        except Exception:
            pass

        p = Path.home() / "Mustatil_Project"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def default_gmt_output_dir(self) -> Path:
        p = self.project_dir_for_saving() / "gmt_maps"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def gmt_settings_path(self) -> Path:
        return self.project_dir_for_saving() / "mustatil_gmt_map_settings.json"

    def load_gmt_settings(self) -> dict:
        try:
            p = self.gmt_settings_path()
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    def save_gmt_settings(self, settings: dict) -> Path:
        p = self.gmt_settings_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        data = dict(settings or {})
        data["project_dir"] = str(self.project_dir_for_saving())
        data["settings_file"] = str(p)
        try:
            data["selected_server"] = str(self.server_combo.currentText() or "")
            data["selected_template"] = str(self.selected_template() or "")
        except Exception:
            pass
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return p

    def create_gmt_map(self):
        """
        Open a small settings window before creating the GMT/map export.
        The Save Settings button writes settings to the project, and Create Map
        writes both the map and a sidecar JSON into the project output folder.
        """
        try:
            _gdf, default_name = self._current_or_combined_layer()
            dlg = GMTExportDialog(
                self.widget,
                self,
                default_title=f"Mustatil GIS Review: {default_name}"
            )
            from PySide6.QtWidgets import QDialog
            if dlg.exec() != QDialog.Accepted:
                return
            settings = dlg.settings()
            self.save_gmt_settings(settings)
            self._create_gmt_map_with_settings(settings)
        except Exception as exc:
            self.set_info(f"GMT export dialog failed:\\n{exc}\\n\\n{traceback.format_exc()}")

    def _create_gmt_map_with_settings(self, settings: dict):
        """
        Create a publication-style overview map for the selected/current layer.

        v7:
        - uses an export dialog
        - supports PNG/PDF/SVG
        - output folder is configurable
        - settings are saved in the project
        - sidecar metadata JSON is saved next to the map
        """
        try:
            gdf, name = self._current_or_combined_layer()
            if gdf is None or len(gdf) == 0:
                self.set_info("No vector layer available for GMT map.")
                return

            if getattr(gdf, "crs", None) is not None:
                gg = gdf.to_crs("EPSG:4326")
            else:
                gg = gdf

            minx, miny, maxx, maxy = [float(v) for v in gg.total_bounds]
            dx = max(maxx - minx, 0.01)
            dy = max(maxy - miny, 0.01)
            region = [minx - dx * 0.15, maxx + dx * 0.15, miny - dy * 0.15, maxy + dy * 0.15]

            fmt = str(settings.get("format", "png") or "png").lower().strip(".")
            if fmt not in {"png", "pdf", "svg"}:
                fmt = "png"
            dpi = int(settings.get("dpi", 220) or 220)
            title = str(settings.get("title") or f"Mustatil GIS Review: {name}")
            subtitle = str(settings.get("subtitle") or "")
            use_basemap = bool(settings.get("use_basemap", True))
            draw_grid = bool(settings.get("draw_grid", True))
            draw_attribution = bool(settings.get("draw_attribution", True))

            out_dir = Path(str(settings.get("output_dir") or self.default_gmt_output_dir()))
            if bool(settings.get("save_project", True)):
                # Keep output inside project unless the user explicitly unchecked it.
                try:
                    project = self.project_dir_for_saving().resolve()
                    if project not in out_dir.resolve().parents and out_dir.resolve() != project:
                        out_dir = self.default_gmt_output_dir()
                except Exception:
                    out_dir = self.default_gmt_output_dir()
            out_dir.mkdir(parents=True, exist_ok=True)

            stem = _sanitize_layer_name(name) + "_gmt_map"
            out_path = out_dir / f"{stem}.{fmt}"

            template = ""
            server_name = ""
            try:
                template = self.selected_template()
                server_name = str(self.server_combo.currentText() or "")
            except Exception:
                template = ""
                server_name = ""

            metadata = {
                "output": str(out_path),
                "format": fmt,
                "dpi": dpi,
                "title": title,
                "subtitle": subtitle,
                "layer": name,
                "feature_count": int(len(gg)),
                "region": region,
                "server": server_name,
                "template": template,
                "use_basemap": use_basemap,
                "project_dir": str(self.project_dir_for_saving()),
                "settings_file": str(self.gmt_settings_path()),
                "gmt_console": str(settings.get("gmt_console", "") or ""),
            }

            basemap_error = None

            if use_basemap and template:
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    z = int(self.zoom_spin.value())
                    basemap = None
                    extent = None
                    info = None
                    last_err = None

                    for zz in range(z, 1, -1):
                        try:
                            basemap, extent, info = _render_basemap_region(template, region, zz, max_tiles=180)
                            z = zz
                            break
                        except Exception as exc:
                            last_err = exc
                    if basemap is None:
                        raise RuntimeError(str(last_err or "Could not render selected basemap."))

                    fig, ax = plt.subplots(figsize=(11.5, 9.0))
                    ax.imshow(basemap, extent=extent, origin="upper")
                    ax.set_xlim(region[0], region[1])
                    ax.set_ylim(region[2], region[3])

                    full_title = title
                    if subtitle:
                        full_title += "\\n" + subtitle
                    ax.set_title(full_title)
                    ax.set_xlabel("Longitude")
                    ax.set_ylabel("Latitude")

                    if draw_grid:
                        ax.grid(True, linewidth=0.25, alpha=0.35)
                    else:
                        ax.grid(False)

                    for _, row in gg.iterrows():
                        geom = row.geometry
                        if geom is None or geom.is_empty:
                            continue
                        geoms = list(geom.geoms) if hasattr(geom, "geoms") and geom.geom_type.startswith("Multi") else [geom]
                        for g in geoms:
                            try:
                                if g.geom_type == "Polygon":
                                    xs, ys = g.exterior.xy
                                    ax.plot(xs, ys, linewidth=1.2)
                                elif g.geom_type == "LineString":
                                    xs, ys = g.xy
                                    ax.plot(xs, ys, linewidth=1.2)
                                else:
                                    c = g.centroid
                                    ax.plot([c.x], [c.y], marker="o", markersize=3)
                            except Exception:
                                pass

                    attribution = ""
                    if "Sentinel-2" in server_name or "EOX" in server_name:
                        attribution = "Basemap: Sentinel-2 cloudless / EOX / s2maps.eu (contains modified Copernicus Sentinel data)"
                    elif "OpenStreetMap" in server_name:
                        attribution = "Basemap: OpenStreetMap contributors"
                    elif "Carto" in server_name:
                        attribution = "Basemap: CARTO / OpenStreetMap contributors"
                    elif "OpenTopoMap" in server_name:
                        attribution = "Basemap: OpenTopoMap / OpenStreetMap contributors"
                    elif "Esri" in server_name:
                        attribution = "Basemap: Esri World Imagery"
                    elif "Bing" in server_name:
                        attribution = "Basemap: Bing Maps"
                    elif "Google" in server_name:
                        attribution = "Basemap: Google Maps"

                    if draw_attribution and attribution:
                        fig.text(0.01, 0.01, attribution, fontsize=7)

                    fig.tight_layout()
                    fig.savefig(str(out_path), dpi=dpi)
                    plt.close(fig)

                    metadata["renderer"] = "matplotlib_basemap_tiles"
                    metadata["tile_zoom_used"] = z
                    metadata["tile_info"] = info
                    meta_path = out_path.with_suffix(out_path.suffix + ".json")
                    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

                    self.set_info(
                        f"GMT/map export created:\\n{out_path}\\n\\n"
                        f"Format: {fmt.upper()}\\n"
                        f"DPI: {dpi}\\n"
                        f"Server: {server_name}\\n"
                        f"Settings saved:\\n{self.gmt_settings_path()}\\n"
                        f"Metadata saved:\\n{meta_path}"
                    )
                    return
                except Exception as exc:
                    basemap_error = exc

            # PyGMT vector-only export if no basemap or basemap failed.
            try:
                import pygmt
                fig = pygmt.Figure()
                frame_title = title + (f" - {subtitle}" if subtitle else "")
                fig.basemap(region=region, projection="M16c", frame=["af", f"+t{frame_title}"])
                try:
                    fig.coast(land="gray90", water="lightblue", shorelines="0.25p,gray50", borders="1/0.25p,gray60")
                except Exception:
                    pass

                for _, row in gg.iterrows():
                    geom = row.geometry
                    if geom is None or geom.is_empty:
                        continue
                    geoms = list(geom.geoms) if hasattr(geom, "geoms") and geom.geom_type.startswith("Multi") else [geom]
                    for g in geoms:
                        try:
                            if g.geom_type in {"Polygon", "LinearRing"}:
                                xs, ys = g.exterior.xy
                                fig.plot(x=list(xs), y=list(ys), pen="1p,red")
                            elif g.geom_type in {"LineString"}:
                                xs, ys = g.xy
                                fig.plot(x=list(xs), y=list(ys), pen="1p,red")
                            else:
                                c = g.centroid
                                fig.plot(x=[c.x], y=[c.y], style="c0.08c", fill="red", pen="red")
                        except Exception:
                            pass

                fig.savefig(str(out_path), dpi=dpi)
                metadata["renderer"] = "pygmt_vector_only"
                if basemap_error:
                    metadata["basemap_error"] = str(basemap_error)
                meta_path = out_path.with_suffix(out_path.suffix + ".json")
                meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

                self.set_info(
                    f"Vector-only PyGMT export created:\\n{out_path}\\n\\n"
                    f"Settings saved:\\n{self.gmt_settings_path()}\\n"
                    f"Metadata saved:\\n{meta_path}"
                )
                return
            except Exception as pygmt_exc:
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(10, 8))
                    full_title = title
                    if subtitle:
                        full_title += "\\n" + subtitle
                    ax.set_title(full_title)
                    ax.set_xlim(region[0], region[1])
                    ax.set_ylim(region[2], region[3])
                    ax.set_xlabel("Longitude")
                    ax.set_ylabel("Latitude")
                    if draw_grid:
                        ax.grid(True, linewidth=0.3)

                    for _, row in gg.iterrows():
                        geom = row.geometry
                        if geom is None or geom.is_empty:
                            continue
                        geoms = list(geom.geoms) if hasattr(geom, "geoms") and geom.geom_type.startswith("Multi") else [geom]
                        for g in geoms:
                            try:
                                if g.geom_type == "Polygon":
                                    xs, ys = g.exterior.xy
                                    ax.plot(xs, ys, linewidth=1.0)
                                elif g.geom_type == "LineString":
                                    xs, ys = g.xy
                                    ax.plot(xs, ys, linewidth=1.0)
                                else:
                                    c = g.centroid
                                    ax.plot([c.x], [c.y], marker="o", markersize=3)
                            except Exception:
                                pass

                    fig.tight_layout()
                    fig.savefig(str(out_path), dpi=dpi)
                    plt.close(fig)

                    metadata["renderer"] = "matplotlib_vector_only"
                    if basemap_error:
                        metadata["basemap_error"] = str(basemap_error)
                    metadata["pygmt_error"] = str(pygmt_exc)
                    meta_path = out_path.with_suffix(out_path.suffix + ".json")
                    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

                    self.set_info(
                        f"Fallback map export created:\\n{out_path}\\n\\n"
                        f"Settings saved:\\n{self.gmt_settings_path()}\\n"
                        f"Metadata saved:\\n{meta_path}"
                    )
                    return
                except Exception as mpl_exc:
                    self.set_info(
                        f"GMT map failed.\\n"
                        f"Basemap error:\\n{basemap_error}\\n\\n"
                        f"PyGMT error:\\n{pygmt_exc}\\n\\n"
                        f"Matplotlib fallback error:\\n{mpl_exc}"
                    )
        except Exception as exc:
            self.set_info(f"GMT map creation failed:\\n{exc}\\n\\n{traceback.format_exc()}")

    # ------------------------------------------------------------------
    # Export / clipboard
    # ------------------------------------------------------------------

    def export_gdf(self, gdf, out_path: str, layer_name: str = "layer"):
        try:
            out = Path(out_path)
            if out.suffix.lower() in {".geojson", ".json"}:
                gdf.to_file(str(out), driver="GeoJSON")
            else:
                gdf.to_file(str(out), layer=_sanitize_layer_name(layer_name), driver="GPKG")
            self.set_info(f"Exported {len(gdf)} feature(s):\n{out}")
            return True
        except Exception as exc:
            self.set_info(f"Export failed:\n{exc}")
            return False

    def export_current_layer_dialog(self):
        if self.current_layer_idx is None:
            return
        from PySide6.QtWidgets import QFileDialog
        layer = self.layers[self.current_layer_idx]
        out, _ = QFileDialog.getSaveFileName(self.widget, "Export layer", str(Path.home() / f"{layer.get('name','layer')}.gpkg"), "GeoPackage (*.gpkg);;GeoJSON (*.geojson)")
        if out:
            self.export_gdf(layer["gdf"], out, layer.get("name", "layer"))

    def export_selected_rows_dialog(self):
        if self.current_layer_idx is None:
            return
        rows = self.selected_rows()
        if not rows:
            self.set_info("No selected rows.")
            return
        from PySide6.QtWidgets import QFileDialog
        layer = self.layers[self.current_layer_idx]
        gdf = layer["gdf"]
        selected = gdf.iloc[[r for r in rows if r < len(gdf)]].copy()
        out, _ = QFileDialog.getSaveFileName(self.widget, "Export selected rows", str(Path.home() / f"{layer.get('name','layer')}_selected.gpkg"), "GeoPackage (*.gpkg);;GeoJSON (*.geojson)")
        if out:
            self.export_gdf(selected, out, f"{layer.get('name','layer')}_selected")

    def export_feature_dialog(self, layer_idx: int, row_idx: int):
        try:
            from PySide6.QtWidgets import QFileDialog
            layer = self.layers[layer_idx]
            gdf = layer["gdf"].iloc[[row_idx]].copy()
            out, _ = QFileDialog.getSaveFileName(self.widget, "Export feature", str(Path.home() / f"{layer.get('name','feature')}_row{row_idx}.gpkg"), "GeoPackage (*.gpkg);;GeoJSON (*.geojson)")
            if out:
                self.export_gdf(gdf, out, f"{layer.get('name','feature')}_row{row_idx}")
        except Exception as exc:
            self.set_info(f"Export feature failed:\n{exc}")

    def bulk_export(self, keys: List[str]):
        try:
            import pandas as pd
            from PySide6.QtWidgets import QFileDialog
            frames = []
            for k in keys:
                try:
                    li, ri = [int(x) for x in str(k).split(":")]
                    gdf = self.layers[li]["gdf"]
                    if ri < len(gdf):
                        frames.append(gdf.iloc[[ri]].copy())
                except Exception:
                    pass
            if not frames:
                self.set_info("No selected features to export.")
                return
            selected = pd.concat(frames, ignore_index=True)
            out, _ = QFileDialog.getSaveFileName(self.widget, "Export selected map features", str(Path.home() / "selected_review_features.gpkg"), "GeoPackage (*.gpkg);;GeoJSON (*.geojson)")
            if out:
                self.export_gdf(selected, out, "selected_review_features")
        except Exception as exc:
            self.set_info(f"Bulk export failed:\n{exc}")

    def copy_center(self, layer_idx: int, row_idx: int):
        try:
            from PySide6.QtWidgets import QApplication
            gdf = self.layers[layer_idx]["gdf"]
            geom = gdf.iloc[row_idx].geometry
            try:
                if getattr(gdf, "crs", None) is not None:
                    geom = gdf.iloc[[row_idx]].to_crs("EPSG:4326").iloc[0].geometry
            except Exception:
                pass
            c = geom.centroid
            txt = f"{c.y}, {c.x}"
            QApplication.clipboard().setText(txt)
            self.set_info(f"Copied center coordinates:\n{txt}")
        except Exception as exc:
            self.set_info(f"Could not copy center:\n{exc}")

    def copy_wkt(self, layer_idx: int, row_idx: int):
        try:
            from PySide6.QtWidgets import QApplication
            geom = self.layers[layer_idx]["gdf"].iloc[row_idx].geometry
            QApplication.clipboard().setText(geom.wkt)
            self.set_info("Copied geometry WKT.")
        except Exception as exc:
            self.set_info(f"Could not copy WKT:\n{exc}")


def _make_tab(workspace=None):
    try:
        return GISWebReviewMapTab(workspace).widget
    except Exception as exc:
        _warn_once("make_tab", f"Could not build GIS Web Review Map v25: {exc}\n{traceback.format_exc()}")
        return None


def _find_ai_pipeline_index(tabwidget) -> int:
    try:
        for i in range(tabwidget.count()):
            if "ai pipeline" in str(tabwidget.tabText(i) or "").lower():
                return i
    except Exception:
        pass
    return -1


def _has_tab(tabwidget) -> bool:
    try:
        for i in range(tabwidget.count()):
            title = str(tabwidget.tabText(i) or "")
            page = tabwidget.widget(i)
            if title == _TAB_TITLE:
                return True
            if getattr(page, "_mustatil_gis_web_review_map", None):
                return True
            if hasattr(page, "objectName") and str(page.objectName()).startswith("mustatil_gis_web_review_map"):
                return True
    except Exception:
        pass
    return False


def _insert_after_ai_pipeline(tabwidget, reason: str = "") -> bool:
    global _REENTRANT
    if _REENTRANT or tabwidget is None:
        return False
    try:
        if _has_tab(tabwidget):
            return False
        idx = _find_ai_pipeline_index(tabwidget)
        if idx < 0:
            return False
        page = _make_tab(_find_workspace(tabwidget))
        if page is None:
            return False
        _REENTRANT = True
        try:
            tabwidget.insertTab(idx + 1, page, _TAB_TITLE)
        finally:
            _REENTRANT = False
        try:
            tabwidget.setTabToolTip(idx + 1, "Web map/image review canvas with filters, Geo-NMS, drag selection and annotator crops.")
        except Exception:
            pass
        _gui_log(tabwidget, f"GIS Web Review Map v25 inserted right of AI Pipeline ({reason or 'hook'}).")
        return True
    except Exception as exc:
        _REENTRANT = False
        _warn_once("insert", f"Could not insert GIS Web Review Map v25: {exc}")
        return False


def _scan_for_ai_pipeline(reason: str = "") -> int:
    inserted = 0
    try:
        from PySide6.QtWidgets import QApplication, QTabWidget
        app = QApplication.instance()
        if not app:
            return 0
        for top in app.topLevelWidgets():
            for tw in top.findChildren(QTabWidget):
                if _find_ai_pipeline_index(tw) >= 0:
                    if _insert_after_ai_pipeline(tw, reason=reason):
                        inserted += 1
    except Exception:
        pass
    return inserted


def _patch_qtabwidget() -> bool:
    try:
        from PySide6.QtWidgets import QTabWidget
    except Exception as exc:
        _warn_once("qt", f"QTabWidget unavailable: {exc}")
        return False
    if getattr(QTabWidget, "_mustatil_gis_web_review_map_v25_patch", False):
        return True
    orig_add = QTabWidget.addTab
    orig_insert = QTabWidget.insertTab

    def patched_addTab(self, *args, **kwargs):
        idx = orig_add(self, *args, **kwargs)
        if not _REENTRANT:
            try:
                _insert_after_ai_pipeline(self, "addTab")
            except Exception:
                pass
        return idx

    def patched_insertTab(self, *args, **kwargs):
        idx = orig_insert(self, *args, **kwargs)
        if not _REENTRANT:
            try:
                _insert_after_ai_pipeline(self, "insertTab")
            except Exception:
                pass
        return idx

    QTabWidget.addTab = patched_addTab
    QTabWidget.insertTab = patched_insertTab
    QTabWidget._mustatil_gis_web_review_map_v25_patch = True
    _log("QTabWidget hook installed; GIS Web Review Map v25 will be inserted right of AI Pipeline.")
    return True


def _patch_workspace_log() -> bool:
    try:
        g = globals().get("MUSTATIL_GLOBALS") or MUSTATIL_GLOBALS or {}
        cls = g.get("MustatilQtWorkspace")
        if cls is None:
            return False
        if getattr(cls, "_mustatil_gis_web_review_map_v25_log_patch", False):
            return True
        old_log = cls.log

        def patched_log(self, msg="", *args, **kwargs):
            res = old_log(self, msg, *args, **kwargs)
            try:
                s = str(msg or "").lower()
                if "ai pipeline" in s or "inserted next to lae-dino trainer" in s:
                    _scan_for_ai_pipeline("workspace.log")
            except Exception:
                pass
            return res

        cls.log = patched_log
        cls._mustatil_gis_web_review_map_v25_log_patch = True
        _log("workspace.log hook installed for AI Pipeline insertion fallback.")
        return True
    except Exception as exc:
        _warn_once("log_patch", f"workspace.log patch failed: {exc}")
        return False


def mustatil_plugin_init():
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    _patch_qtabwidget()
    _patch_workspace_log()
    _scan_for_ai_pipeline("plugin init")
    _log("Installed. Waiting for AI Pipeline tab if it is not present yet.")


try:
    mustatil_plugin_init()
except Exception as exc:
    _warn_once("auto_init", f"Auto-init warning: {exc}")
