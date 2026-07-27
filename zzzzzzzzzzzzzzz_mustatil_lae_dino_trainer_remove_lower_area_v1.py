#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mustatil Universal Chunked TIFF Detection Input Patch v1
--------------------------------------------------------
Drop-in patch plugin for Mustatil Qt Workspace.

Goal
----
For local Detection images, especially huge .tif/.tiff/.geotiff files, make the
additional detection plugins feed model tiles by reading raster windows from the
source TIFF instead of loading the whole image into RAM first.

Patched when present:
  - Google OWLv2 image Detection tab
  - Grounding DINO image Detection tab
  - LAE-DINO image Detection tab
  - Faster R-CNN image Detection tab
  - Mask R-CNN image Detection tab
  - U-Net image Detection tab
  - SAM2 image Detection tab
  - AI/Yolo Pipeline crop feeding for TIFF parents/tiles

It does not replace the original YOLO Detection. Mustatil's original YOLO path
already uses the legacy tiled workflow. This plugin only redirects companion
plugins that previously did Image.open(...).convert("RGB") on the full image.

Install
-------
Copy this file into:
    mustatil_plugins/
Restart Mustatil.

Recommended dependency for true chunked TIFF reading:
    python -m pip install rasterio

Without rasterio the patch keeps Mustatil running but falls back to PIL crop
loading and logs a warning.
"""
from __future__ import annotations

import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PLUGIN_NAME = "Mustatil Universal Chunked TIFF Detection Input Patch v1"
PLUGIN_PREFIX = "[Mustatil Chunked TIFF Detection Input v1]"

_PATCHED_MODULES: set[int] = set()
_PATCHED_CLASSES: set[int] = set()
_SCAN_TIMER = None
_SCAN_COUNT = 0
_MAX_SCANS = 80

TIFF_EXTS = {".tif", ".tiff", ".geotiff", ".gtif", ".bigtif", ".bigtiiff"}


def _log(msg: Any) -> None:
    try:
        print(f"{PLUGIN_PREFIX} {msg}", flush=True)
    except Exception:
        pass


def _ws_log(ws: Any, msg: Any) -> None:
    text = f"{PLUGIN_PREFIX} {msg}"
    try:
        if ws is not None and hasattr(ws, "log"):
            ws.log(str(text))
            return
    except Exception:
        pass
    _log(msg)


def _show_error(ws: Any, title: str, exc: Any) -> None:
    text = str(exc)
    try:
        if ws is not None and hasattr(ws, "show_error"):
            ws.show_error(title, text)
            return
    except Exception:
        pass
    _ws_log(ws, f"{title}: {text}")
    try:
        traceback.print_exc()
    except Exception:
        pass


def _get_var(v: Any, default: Any = "") -> Any:
    try:
        if hasattr(v, "get"):
            return v.get()
    except Exception:
        pass
    return default


def _is_tiff_path(path: Any) -> bool:
    try:
        return Path(str(path)).suffix.lower() in TIFF_EXTS
    except Exception:
        return False


def _image_path_from_ws(ws: Any) -> str:
    raw = _get_var(getattr(ws, "image", None), "")
    return str(raw or "").strip().strip('"')


def _tile_settings(ws: Any, settings: Optional[Dict[str, Any]] = None, *, default_tile: int = 1024, default_overlap: int = 128) -> Tuple[int, int, int]:
    settings = settings or {}
    try:
        tile_raw = settings.get("tile", _get_var(getattr(ws, "tile", None), default_tile))
    except Exception:
        tile_raw = default_tile
    try:
        overlap_raw = settings.get("overlap", _get_var(getattr(ws, "overlap", None), default_overlap))
    except Exception:
        overlap_raw = default_overlap
    tile = max(128, int(tile_raw or default_tile))
    overlap = max(0, min(tile - 1, int(overlap_raw or default_overlap)))
    step = max(1, tile - overlap)
    return tile, overlap, step


def _iter_windows(width: int, height: int, tile: int, overlap: int, *, shifted: bool = False) -> List[Tuple[int, int, int, int]]:
    """Return (x, y, w, h) windows covering the image, with optional half-step shifted pass."""
    width = int(width); height = int(height); tile = int(tile); overlap = int(overlap)
    step = max(1, tile - overlap)
    starts: List[Tuple[int, int]] = []

    def add_grid(x0: int = 0, y0: int = 0) -> None:
        seen_y = set()
        y = max(0, int(y0))
        while y < height:
            if y in seen_y:
                break
            seen_y.add(y)
            seen_x = set()
            x = max(0, int(x0))
            while x < width:
                if x in seen_x:
                    break
                seen_x.add(x)
                starts.append((x, y))
                if x + tile >= width:
                    break
                x += step
            if y + tile >= height:
                break
            y += step

    add_grid(0, 0)
    if shifted and tile > 2:
        add_grid(max(1, step // 2), max(1, step // 2))

    out: List[Tuple[int, int, int, int]] = []
    seen = set()
    for x, y in starts:
        if x >= width or y >= height:
            continue
        w = min(tile, width - x)
        h = min(tile, height - y)
        if w <= 0 or h <= 0:
            continue
        key = (int(x), int(y), int(w), int(h))
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _stretch_to_uint8_chw(data: Any) -> Any:
    import numpy as np
    arr = np.asarray(data)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    elif arr.shape[0] >= 3:
        arr = arr[:3]
    else:
        arr = np.repeat(arr[:1], 3, axis=0)

    if arr.dtype == np.uint8:
        out = arr
    else:
        arrf = arr.astype("float32", copy=False)
        out_bands = []
        for i in range(arrf.shape[0]):
            band = arrf[i]
            finite = band[np.isfinite(band)]
            if finite.size == 0:
                out_bands.append(np.zeros(band.shape, dtype="uint8"))
                continue
            try:
                lo, hi = np.percentile(finite, [2, 98])
            except Exception:
                lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                out_bands.append(np.zeros(band.shape, dtype="uint8"))
                continue
            scaled = (band - float(lo)) * (255.0 / max(1e-6, float(hi - lo)))
            out_bands.append(np.clip(scaled, 0, 255).astype("uint8"))
        out = np.stack(out_bands, axis=0)
    return out


def _open_image_reader(path: str, ws: Any = None) -> Dict[str, Any]:
    """Open image for tiled reading. Uses rasterio windows for TIFF when available."""
    path = str(path or "")
    if _is_tiff_path(path):
        try:
            import rasterio  # noqa: F401
            ds = rasterio.open(path)
            return {"kind": "rasterio", "path": path, "ds": ds, "width": int(ds.width), "height": int(ds.height)}
        except Exception as exc:
            _ws_log(ws, "rasterio is missing or could not open TIFF; falling back to PIL crop loading. Install with: python -m pip install rasterio | " + str(exc))
    from PIL import Image
    img = Image.open(path)
    return {"kind": "pil", "path": path, "img": img, "width": int(img.width), "height": int(img.height)}


def _close_image_reader(reader: Optional[Dict[str, Any]]) -> None:
    if not reader:
        return
    for key in ("ds", "img"):
        try:
            obj = reader.get(key)
            if obj is not None and hasattr(obj, "close"):
                obj.close()
        except Exception:
            pass


def _read_window_rgb(reader: Dict[str, Any], x: int, y: int, w: int, h: int):
    """Read one RGB PIL tile from an already opened reader."""
    kind = str(reader.get("kind"))
    if kind == "rasterio":
        import numpy as np
        from PIL import Image
        from rasterio.windows import Window
        ds = reader["ds"]
        # Read first 3 bands if available. For single-band rasters, repeat to RGB.
        if int(getattr(ds, "count", 1) or 1) >= 3:
            bands = [1, 2, 3]
        else:
            bands = [1]
        data = ds.read(bands, window=Window(int(x), int(y), int(w), int(h)), boundless=False)
        chw = _stretch_to_uint8_chw(data)
        hwc = np.transpose(chw, (1, 2, 0))
        return Image.fromarray(hwc, mode="RGB")
    img = reader["img"]
    return img.crop((int(x), int(y), int(x + w), int(y + h))).convert("RGB")


def _safe_refresh_after_detection(ws: Any, mod: Any = None) -> None:
    try:
        if hasattr(ws, "loadprev"):
            ws.loadprev()
    except Exception:
        pass
    try:
        if mod is not None and hasattr(mod, "_refresh_combos"):
            mod._refresh_combos(ws)
    except Exception:
        pass
    try:
        if mod is not None and hasattr(mod, "_redraw"):
            mod._redraw(ws)
            return
    except Exception:
        pass
    for fn in (lambda: ws.redraw(fit=False), lambda: ws.redraw(), lambda: ws.refresh_layers()):
        try:
            fn()
        except Exception:
            pass


def _det_class(mod: Any = None):
    try:
        from mustatil_legacy_backend import Det
        return Det
    except Exception:
        pass
    try:
        g = getattr(mod, "MUSTATIL_GLOBALS", {}) or {}
        backend = g.get("backend")
        Det = getattr(backend, "Det", None)
        if Det is not None:
            return Det
    except Exception:
        pass
    return None


def _make_det(mod: Any, model: str, class_id: int, conf: float, x1: float, y1: float, x2: float, y2: float):
    Det = _det_class(mod)
    if Det is None:
        raise RuntimeError("Could not import mustatil_legacy_backend.Det")
    return Det(0, str(model), int(class_id), float(conf), float(x1), float(y1), float(x2), float(y2))


def _run_owl_detection_image_chunked(ws: Any, *, _mod: Any) -> None:
    try:
        img_path = _image_path_from_ws(ws)
        if not img_path or not Path(img_path).exists():
            raise RuntimeError("No Detection image selected.")
        conf = max(0.001, min(1.0, float(_get_var(getattr(ws, "conf", None), 0.15))))
        tile, overlap, _step = _tile_settings(ws, default_tile=768, default_overlap=128)
        reader = _open_image_reader(img_path, ws)
        try:
            W, H = int(reader["width"]), int(reader["height"])
            windows = _iter_windows(W, H, tile, overlap, shifted=False)
            _ws_log(ws, f"Google OWL Detection chunked input: {Path(img_path).name} {W}x{H}, chunks={len(windows)}, tile={tile}, overlap={overlap}, reader={reader.get('kind')}")
            dets = []
            detect_fn = getattr(_mod, "_owl_detect_pil")
            for idx, (x, y, w, h) in enumerate(windows, 1):
                crop = _read_window_rgb(reader, x, y, w, h)
                local = detect_fn(ws, crop, conf)
                for r in local or []:
                    try:
                        dets.append(_make_det(
                            _mod,
                            str(r.get("model", "Google OWL")),
                            int(r.get("class_id", 0)),
                            float(r.get("confidence", r.get("score", 0.0))),
                            x + float(r["x1"]), y + float(r["y1"]),
                            x + float(r["x2"]), y + float(r["y2"]),
                        ))
                    except Exception:
                        pass
                if idx % 10 == 0 or idx == len(windows):
                    _ws_log(ws, f"Google OWL Detection chunks processed: {idx}/{len(windows)}; detections={len(dets)}")
            try:
                if hasattr(ws, "nms"):
                    dets = ws.nms(dets, 0.45)
            except Exception:
                pass
            ws.dets = list(dets)
            _safe_refresh_after_detection(ws, _mod)
            _ws_log(ws, f"Google OWL Detection finished: {len(dets)} detections")
        finally:
            _close_image_reader(reader)
    except Exception as exc:
        _show_error(ws, "Google OWL Detection", exc)


def _run_generic_detection_image_chunked(ws: Any, model_name: str, detect_fn: Any, *, _mod: Any) -> None:
    try:
        img_path = _image_path_from_ws(ws)
        if not img_path or not Path(img_path).exists():
            raise RuntimeError("No Detection image selected.")
        conf = max(0.001, min(1.0, float(_get_var(getattr(ws, "conf", None), 0.15))))
        tile, overlap, _step = _tile_settings(ws, default_tile=768, default_overlap=128)
        reader = _open_image_reader(img_path, ws)
        try:
            W, H = int(reader["width"]), int(reader["height"])
            windows = _iter_windows(W, H, tile, overlap, shifted=False)
            _ws_log(ws, f"{model_name} Detection chunked input: {Path(img_path).name} {W}x{H}, chunks={len(windows)}, tile={tile}, overlap={overlap}, reader={reader.get('kind')}")
            dets = []
            for idx, (x, y, w, h) in enumerate(windows, 1):
                crop = _read_window_rgb(reader, x, y, w, h)
                local = detect_fn(ws, crop, conf)
                for r in local or []:
                    try:
                        dets.append(_make_det(
                            _mod,
                            str(r.get("model", model_name)),
                            int(r.get("class_id", 0)),
                            float(r.get("confidence", r.get("conf", r.get("score", 0.0)))),
                            x + float(r["x1"]), y + float(r["y1"]),
                            x + float(r["x2"]), y + float(r["y2"]),
                        ))
                    except Exception:
                        pass
                if idx % 10 == 0 or idx == len(windows):
                    _ws_log(ws, f"{model_name}: chunks={idx}/{len(windows)}, detections={len(dets)}")
            try:
                if hasattr(ws, "nms"):
                    dets = ws.nms(dets, 0.45)
            except Exception:
                pass
            ws.dets = list(dets)
            _safe_refresh_after_detection(ws, _mod)
            _ws_log(ws, f"{model_name} Detection finished: {len(dets)} detections")
        finally:
            _close_image_reader(reader)
    except Exception as exc:
        _show_error(ws, str(model_name), exc)


def _run_torchvision_detection_image_chunked(ws: Any, mode: str, settings: Dict[str, Any], *, _mod: Any) -> None:
    try:
        img_path = _image_path_from_ws(ws)
        if not img_path or not Path(img_path).exists():
            raise RuntimeError("No Detection image selected.")
        tile, overlap, _step = _tile_settings(ws, settings, default_tile=1024, default_overlap=128)
        shifted = bool(settings.get("shifted", settings.get("shifted_tiles", False))) if isinstance(settings, dict) else False
        reader = _open_image_reader(img_path, ws)
        try:
            W, H = int(reader["width"]), int(reader["height"])
            windows = _iter_windows(W, H, tile, overlap, shifted=shifted)
            title = str(settings.get("title", mode) if isinstance(settings, dict) else mode)
            records: List[Dict[str, Any]] = []
            prompt_records: List[Dict[str, Any]] = []
            if mode == "sam2" and hasattr(_mod, "_existing_detection_prompt_records"):
                prompt_records = _mod._existing_detection_prompt_records(ws)
                _ws_log(ws, f"SAM2 prompt source boxes available: {len(prompt_records)}")
            _ws_log(ws, f"{title} chunked input: {Path(img_path).name} {W}x{H}, chunks={len(windows)}, tile={tile}, overlap={overlap}, reader={reader.get('kind')}")

            for idx, (x, y, w, h) in enumerate(windows, 1):
                crop = _read_window_rgb(reader, x, y, w, h)
                local_settings = settings
                if mode == "sam2" and hasattr(_mod, "_sam2_settings_for_crop"):
                    local_settings = _mod._sam2_settings_for_crop(settings, prompt_records, x, y, w, h)
                    if local_settings is None:
                        if idx % 10 == 0:
                            _ws_log(ws, f"{title}: chunks={idx}/{len(windows)}; no SAM2 prompts in current chunks")
                        continue
                local = _mod._model_infer_pil(ws, mode, crop, local_settings)
                for r in local or []:
                    try:
                        nr = dict(r)
                        nr["x1"] = x + float(r["x1"]); nr["x2"] = x + float(r["x2"])
                        nr["y1"] = y + float(r["y1"]); nr["y2"] = y + float(r["y2"])
                        records.append(nr)
                    except Exception:
                        pass
                if idx % 5 == 0 or idx == len(windows):
                    _ws_log(ws, f"{title} chunks processed: {idx}/{len(windows)}; detections/components={len(records)}")

            try:
                if hasattr(_mod, "_nms_records_xyxy"):
                    iou = float(settings.get("nms_iou", 0.80 if mode == "sam2" else 0.55) or 0.55)
                    records = _mod._nms_records_xyxy(records, iou)
            except Exception:
                pass
            _mod._append_dets_to_workspace(ws, records, title)
            try:
                ws.mustatil_torchvision_last_records = [dict(r) for r in records]
            except Exception:
                pass
            _safe_refresh_after_detection(ws, _mod)
            _ws_log(ws, f"{title} finished: {len(records)} detections/components")
        finally:
            _close_image_reader(reader)
    except Exception as exc:
        title = str(settings.get("title", mode) if isinstance(settings, dict) else mode)
        if hasattr(_mod, "_show_error"):
            try:
                _mod._show_error(ws, f"{title} Detection", exc)
                return
            except Exception:
                pass
        _show_error(ws, f"{title} Detection", exc)


def _patch_module_functions(mod: Any) -> bool:
    changed = False
    if mod is None or id(mod) in _PATCHED_MODULES:
        return False
    name = str(getattr(mod, "__name__", ""))
    if name == __name__:
        return False

    try:
        if callable(getattr(mod, "_owl_detect_pil", None)) and callable(getattr(mod, "_run_owl_detection_image", None)):
            old = getattr(mod, "_run_owl_detection_image")
            if not getattr(old, "_mustatil_chunked_tiff_patch", False):
                def owl_runner(ws, _mod=mod):
                    return _run_owl_detection_image_chunked(ws, _mod=_mod)
                owl_runner._mustatil_chunked_tiff_patch = True  # type: ignore[attr-defined]
                owl_runner._mustatil_original = old  # type: ignore[attr-defined]
                setattr(mod, "_run_owl_detection_image", owl_runner)
                changed = True
    except Exception as exc:
        _log(f"OWL patch failed for {name}: {exc}")

    try:
        if callable(getattr(mod, "_run_generic_detection_image", None)):
            old = getattr(mod, "_run_generic_detection_image")
            if not getattr(old, "_mustatil_chunked_tiff_patch", False):
                def generic_runner(ws, model_name: str, detect_fn, _mod=mod):
                    return _run_generic_detection_image_chunked(ws, model_name, detect_fn, _mod=_mod)
                generic_runner._mustatil_chunked_tiff_patch = True  # type: ignore[attr-defined]
                generic_runner._mustatil_original = old  # type: ignore[attr-defined]
                setattr(mod, "_run_generic_detection_image", generic_runner)
                changed = True
    except Exception as exc:
        _log(f"generic open-vocab patch failed for {name}: {exc}")

    try:
        # TorchVision/UNet/SAM2 detection tab plugin has all three of these helpers.
        if callable(getattr(mod, "_run_detection_image", None)) and callable(getattr(mod, "_model_infer_pil", None)) and callable(getattr(mod, "_append_dets_to_workspace", None)):
            old = getattr(mod, "_run_detection_image")
            if not getattr(old, "_mustatil_chunked_tiff_patch", False):
                def tv_runner(ws, mode: str, settings: Dict[str, Any], _mod=mod):
                    return _run_torchvision_detection_image_chunked(ws, mode, settings, _mod=_mod)
                tv_runner._mustatil_chunked_tiff_patch = True  # type: ignore[attr-defined]
                tv_runner._mustatil_original = old  # type: ignore[attr-defined]
                setattr(mod, "_run_detection_image", tv_runner)
                changed = True
    except Exception as exc:
        _log(f"TorchVision/UNet/SAM2 patch failed for {name}: {exc}")

    if changed:
        _PATCHED_MODULES.add(id(mod))
        _log(f"Patched detection functions in module: {name}")
    return changed


def _patch_pipeline_class(cls: Any) -> bool:
    if cls is None or id(cls) in _PATCHED_CLASSES:
        return False
    if not callable(getattr(cls, "run_pipeline", None)) or not callable(getattr(cls, "_crop_parent", None)):
        return False
    old_run = getattr(cls, "run_pipeline")
    old_crop = getattr(cls, "_crop_parent")
    if getattr(old_run, "_mustatil_chunked_tiff_patch", False):
        _PATCHED_CLASSES.add(id(cls))
        return False

    def run_pipeline_chunked_tiff(self, *args, _old_run=old_run, **kwargs):
        reader = None
        try:
            image_path = None
            try:
                image_path = self._current_pipeline_image_path()
            except Exception:
                image_path = None
            if image_path and _is_tiff_path(image_path):
                try:
                    reader = _open_image_reader(str(image_path), getattr(self, "ws", None))
                    setattr(self, "_mustatil_chunked_tiff_reader", reader)
                    setattr(self, "_mustatil_chunked_tiff_path", str(image_path))
                    try:
                        self._log(f"Chunked TIFF pipeline crop reader active: {Path(str(image_path)).name}, reader={reader.get('kind')}")
                    except Exception:
                        pass
                except Exception as exc:
                    try:
                        self._log("Chunked TIFF pipeline reader warning: " + str(exc))
                    except Exception:
                        pass
            return _old_run(self, *args, **kwargs)
        finally:
            try:
                if hasattr(self, "_mustatil_chunked_tiff_reader"):
                    delattr(self, "_mustatil_chunked_tiff_reader")
                if hasattr(self, "_mustatil_chunked_tiff_path"):
                    delattr(self, "_mustatil_chunked_tiff_path")
            except Exception:
                pass
            _close_image_reader(reader)

    def crop_parent_chunked_tiff(self, full, rec: Dict[str, Any], pad: int, _old_crop=old_crop):
        try:
            reader = getattr(self, "_mustatil_chunked_tiff_reader", None)
            if isinstance(reader, dict) and reader.get("kind") == "rasterio":
                W, H = int(reader.get("width", getattr(full, "width", 0))), int(reader.get("height", getattr(full, "height", 0)))
                x1 = max(0, int(float(rec.get("x1", 0)) - int(pad or 0)))
                y1 = max(0, int(float(rec.get("y1", 0)) - int(pad or 0)))
                x2 = min(W, int(float(rec.get("x2", W)) + int(pad or 0)))
                y2 = min(H, int(float(rec.get("y2", H)) + int(pad or 0)))
                if x2 <= x1 or y2 <= y1:
                    x1, y1, x2, y2 = 0, 0, W, H
                crop = _read_window_rgb(reader, x1, y1, max(1, x2 - x1), max(1, y2 - y1))
                return crop, (float(x1), float(y1))
        except Exception as exc:
            try:
                self._log("Chunked TIFF crop fallback warning: " + str(exc))
            except Exception:
                pass
        return _old_crop(self, full, rec, pad)

    run_pipeline_chunked_tiff._mustatil_chunked_tiff_patch = True  # type: ignore[attr-defined]
    crop_parent_chunked_tiff._mustatil_chunked_tiff_patch = True  # type: ignore[attr-defined]
    setattr(cls, "run_pipeline", run_pipeline_chunked_tiff)
    setattr(cls, "_crop_parent", crop_parent_chunked_tiff)
    _PATCHED_CLASSES.add(id(cls))
    _log(f"Patched pipeline TIFF crop feeding in class: {getattr(cls, '__module__', '?')}.{getattr(cls, '__name__', '?')}")
    return True


def _scan_and_patch() -> int:
    changed = 0
    for mod in list(sys.modules.values()):
        try:
            if _patch_module_functions(mod):
                changed += 1
        except Exception:
            pass
        try:
            cls = getattr(mod, "YoloPipelineTab", None)
            if cls is not None and _patch_pipeline_class(cls):
                changed += 1
        except Exception:
            pass
    return changed


def _schedule_scan() -> None:
    global _SCAN_TIMER, _SCAN_COUNT
    try:
        from PySide6.QtCore import QTimer
    except Exception:
        try:
            _scan_and_patch()
        except Exception:
            pass
        return

    try:
        if _SCAN_TIMER is None:
            _SCAN_TIMER = QTimer()
            _SCAN_TIMER.setInterval(1200)
            def tick():
                global _SCAN_COUNT
                _SCAN_COUNT += 1
                try:
                    changed = _scan_and_patch()
                    if changed:
                        _log(f"scan patched {changed} target(s).")
                except Exception as exc:
                    _log("scan warning: " + str(exc))
                if _SCAN_COUNT >= _MAX_SCANS:
                    try:
                        _SCAN_TIMER.stop()
                        _log("periodic scan stopped; patch remains active.")
                    except Exception:
                        pass
            _SCAN_TIMER.timeout.connect(tick)
            _SCAN_TIMER.start()
        try:
            QTimer.singleShot(0, _scan_and_patch)
            QTimer.singleShot(250, _scan_and_patch)
            QTimer.singleShot(1000, _scan_and_patch)
        except Exception:
            pass
    except Exception as exc:
        _log("QTimer scan setup failed: " + str(exc))
        try:
            _scan_and_patch()
        except Exception:
            pass


def mustatil_plugin_init() -> None:
    _log("installed. True chunked GeoTIFF/TIFF reading uses rasterio windows; fallback keeps PIL mode alive if rasterio is missing.")
    _scan_and_patch()
    _schedule_scan()


# Also run once when imported, because Mustatil calls plugin init after exec but this
# makes syntax/manual testing easier and harmlessly idempotent.
try:
    _scan_and_patch()
except Exception:
    pass
