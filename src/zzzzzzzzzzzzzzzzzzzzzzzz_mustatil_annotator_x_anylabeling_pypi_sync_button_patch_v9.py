#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mustatil Multiband All-AI + R-CNN Patch v12 LateLoad
=================================

Drop this file into Mustatil's mustatil_plugins/ folder.

Purpose
-------
No new tab. This plugin patches Mustatil's existing training and detection paths
so multiband GeoTIFF/BigTIFF imagery is converted through one shared adapter
before it reaches the AI model.

It affects:
- Existing YOLO Trainer dataset preparation (via backend.prepare_yolo_dataset)
- Existing tiled Detection (via backend.open_img/read_tile/load_preview)
- Other AI plugins that use PIL.Image.open(...) or cv2.imread(...)
- Ultralytics predict(path=multiband.tif) as a safety net

Important model behavior
------------------------
Universal/foundation models (SAM2, Grounding DINO, OWL/OWLv2, LAE-DINO, etc.)
cannot normally accept 4/8/13-channel tensors directly. For these models this
patch uses a 3-channel composite made from selected raster bands.

Native multiband training for models that truly support C>3 is a later model-
specific adapter; this v1 patch prioritizes making every AI model usable on
multiband rasters without breaking RGB models.

Configuration
-------------
The patch searches for mustatil_multiband_profile.json in:
1) MUSTATIL_MULTIBAND_PROFILE environment variable
2) current Mustatil project folder
3) project/_yolo_dataset/
4) next to the input raster/image
5) next to the selected model/weights, when available

If no profile exists and the input GeoTIFF has more than 3 bands, it uses:
- bands from MUSTATIL_MULTIBAND_BANDS, default: 1,2,3
- normalization from MUSTATIL_MULTIBAND_NORMALIZATION, default: percentile_2_98

Minimal profile example:
{
  "enabled": true,
  "selected_bands": [3,4,8],
  "normalization": "percentile_2_98",
  "adapter_mode": "universal_rgb_composite"
}
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PLUGIN_NAME = "Mustatil Multiband All-AI + R-CNN Patch v12 LateLoad"
PROFILE_NAME = "mustatil_multiband_profile.json"

MUSTATIL_GLOBALS = globals().get("MUSTATIL_GLOBALS", {})
MUSTATIL_PLUGIN_GLOBALS = globals().get("MUSTATIL_PLUGIN_GLOBALS", {})

_ORIG: Dict[str, Any] = {}
_PATCHED = False
_CLASS_PATCHED = False
_UI_PATCHED_IDS = set()
_LAST_WORKSPACE = None
_LAST_PROFILE: Optional[Dict[str, Any]] = None
_WARNED: set[str] = set()


def _log(msg: str) -> None:
    print(f"[{PLUGIN_NAME}] {msg}", flush=True)


def _warn_once(key: str, msg: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        print(f"[{PLUGIN_NAME} WARNING] {msg}", flush=True)


def _safe_str(value: Any) -> str:
    try:
        return str(value or "").strip().strip('"')
    except Exception:
        return ""


def _is_tiff(path: Any) -> bool:
    return _safe_str(path).lower().endswith((".tif", ".tiff"))


def _try_import(name: str):
    try:
        return __import__(name)
    except Exception as exc:
        _warn_once(f"missing:{name}", f"Optional dependency missing/unusable: {name}: {exc}")
        return None


def _rasterio():
    return _try_import("rasterio")


def _np():
    return _try_import("numpy")


def _pil_image():
    try:
        from PIL import Image
        return Image
    except Exception as exc:
        _warn_once("missing:PIL", f"Pillow missing/unusable: {exc}")
        return None


def _owner_from_callable(fn: Any) -> Any:
    try:
        return getattr(fn, "__self__", None)
    except Exception:
        return None


def _var_get(obj: Any, name: str, default: Any = "") -> Any:
    try:
        v = getattr(obj, name, None)
        if hasattr(v, "get"):
            return v.get()
        if v is not None:
            return v
    except Exception:
        pass
    return default


def _parse_bands(text_or_list: Any, max_band: Optional[int] = None) -> List[int]:
    if isinstance(text_or_list, (list, tuple)):
        out = []
        for b in text_or_list:
            try:
                bi = int(b)
                if bi > 0 and (max_band is None or bi <= max_band):
                    out.append(bi)
            except Exception:
                pass
        return out or [1, 2, 3]
    text = _safe_str(text_or_list) or os.environ.get("MUSTATIL_MULTIBAND_BANDS", "1,2,3")
    bands: List[int] = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                a, b = part.split("-", 1)
                a, b = int(a), int(b)
                step = 1 if b >= a else -1
                for x in range(a, b + step, step):
                    if x > 0 and (max_band is None or x <= max_band):
                        bands.append(x)
            except Exception:
                continue
        else:
            try:
                x = int(part)
                if x > 0 and (max_band is None or x <= max_band):
                    bands.append(x)
            except Exception:
                continue
    # de-duplicate while preserving order
    out = []
    seen = set()
    for b in bands:
        if b not in seen:
            seen.add(b); out.append(b)
    return out or [1, 2, 3]


def _default_profile(path: Optional[str] = None, max_band: Optional[int] = None) -> Dict[str, Any]:
    bands = _parse_bands(os.environ.get("MUSTATIL_MULTIBAND_BANDS", "1,2,3"), max_band=max_band)
    return {
        "version": 1,
        "enabled": os.environ.get("MUSTATIL_MULTIBAND_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"},
        "source_raster": _safe_str(path),
        "selected_bands": bands,
        "normalization": os.environ.get("MUSTATIL_MULTIBAND_NORMALIZATION", "percentile_2_98"),
        "adapter_mode": "universal_rgb_composite",
        "model_input_channels": 3,
        "applies_to": ["YOLO", "Faster R-CNN", "Mask R-CNN", "U-Net", "SAM2", "Grounding DINO", "OWL/OWLv2", "LAE-DINO", "other PIL/cv2 based plugins"],
        "notes": "Universal all-AI mode uses selected multiband raster bands as a 3-channel composite for RGB/foundation models.",
    }


def _read_profile_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("enabled", True)
                data.setdefault("adapter_mode", "universal_rgb_composite")
                data.setdefault("normalization", "percentile_2_98")
                data["selected_bands"] = _parse_bands(data.get("selected_bands") or data.get("bands") or data.get("band_string") or "1,2,3")
                data["model_input_channels"] = 3
                return data
    except Exception as exc:
        _warn_once(f"profile:{path}", f"Could not read profile {path}: {exc}")
    return None


def _workspace_project_dir(workspace: Any) -> Optional[Path]:
    raw = _safe_str(_var_get(workspace, "project", "")) if workspace is not None else ""
    if raw:
        try:
            return Path(raw).expanduser()
        except Exception:
            return None
    return None


def _workspace_setting(workspace: Any, name: str, default: Any = None) -> Any:
    if workspace is None:
        return default
    try:
        v = getattr(workspace, name, None)
        if hasattr(v, "get"):
            return v.get()
        if v is not None:
            return v
    except Exception:
        pass
    return default


def _model_candidate_dirs(workspace: Any) -> List[Path]:
    out: List[Path] = []
    try:
        for v in getattr(workspace, "models", []) or []:
            raw = _safe_str(v.get() if hasattr(v, "get") else v)
            if raw:
                p = Path(raw).expanduser()
                out.append(p.parent if p.suffix else p)
    except Exception:
        pass
    for attr in ["trainmodel", "sammodel", "form_model_path", "fl_model_path", "auto_annotate_yolo_model"]:
        raw = _safe_str(_var_get(workspace, attr, ""))
        if raw:
            p = Path(raw).expanduser(); out.append(p.parent if p.suffix else p)
    return out


def _find_profile(path: Optional[str] = None, workspace: Any = None, max_band: Optional[int] = None) -> Optional[Dict[str, Any]]:
    global _LAST_PROFILE
    candidates: List[Path] = []
    envp = _safe_str(os.environ.get("MUSTATIL_MULTIBAND_PROFILE", ""))
    if envp:
        candidates.append(Path(envp).expanduser())
    proj = _workspace_project_dir(workspace or _LAST_WORKSPACE)
    if proj:
        candidates.append(proj / PROFILE_NAME)
        candidates.append(proj / "_yolo_dataset" / PROFILE_NAME)
        candidates.append(proj / "_multiband_profile" / PROFILE_NAME)
    if path:
        try:
            p = Path(path).expanduser()
            candidates.append(p.with_suffix(p.suffix + ".multiband.json"))
            candidates.append(p.parent / PROFILE_NAME)
            # Common Mustatil dataset layout:
            # project/
            #   images/<raster>.tif
            #   labels/<raster>.txt
            #   mustatil_multiband_profile.json
            # v1 missed this when testing from CLI without an active workspace.
            try:
                candidates.append(p.parent.parent / PROFILE_NAME)
                if p.parent.name.lower() in {"images", "image", "imgs", "rasters", "raster"}:
                    candidates.append(p.parent.parent / "_yolo_dataset" / PROFILE_NAME)
            except Exception:
                pass
        except Exception:
            pass
    for d in _model_candidate_dirs(workspace or _LAST_WORKSPACE):
        candidates.append(d / PROFILE_NAME)
        candidates.append(d.parent / PROFILE_NAME)

    seen = set()
    for c in candidates:
        try:
            key = str(c.resolve()) if c.exists() else str(c)
        except Exception:
            key = str(c)
        if key in seen:
            continue
        seen.add(key)
        prof = _read_profile_file(c)
        if prof is not None:
            if max_band is not None:
                prof["selected_bands"] = _parse_bands(prof.get("selected_bands", [1, 2, 3]), max_band=max_band)
            _LAST_PROFILE = prof
            return prof

    # GUI inline settings, if the optional compact controls have been injected.
    w = workspace or _LAST_WORKSPACE
    if w is not None:
        try:
            enabled = bool(_workspace_setting(w, "mustatil_mb_enabled", True))
            bands_raw = _workspace_setting(w, "mustatil_mb_bands", None)
            norm_raw = _workspace_setting(w, "mustatil_mb_normalization", None)
            if bands_raw is not None or norm_raw is not None:
                prof = _default_profile(path, max_band=max_band)
                prof["enabled"] = enabled
                if bands_raw is not None:
                    prof["selected_bands"] = _parse_bands(bands_raw, max_band=max_band)
                if norm_raw:
                    prof["normalization"] = _safe_str(norm_raw)
                _LAST_PROFILE = prof
                return prof
        except Exception:
            pass

    # No profile found. For true multiband TIFFs, auto-enable first selected bands.
    if max_band and int(max_band) > 3:
        prof = _default_profile(path, max_band=max_band)
        _LAST_PROFILE = prof
        return prof
    return _LAST_PROFILE


def _save_profile_near_project(workspace: Any, profile: Dict[str, Any]) -> Optional[Path]:
    proj = _workspace_project_dir(workspace)
    if proj is None:
        return None
    try:
        p = proj / PROFILE_NAME
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
        return p
    except Exception as exc:
        _warn_once("profile_save", f"Could not save multiband profile: {exc}")
        return None


def _normalize(arr: Any, method: str) -> Any:
    np = _np()
    if np is None:
        return arr
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    if str(method or "").lower() in {"none", "raw"}:
        return arr
    arr = arr.astype("float32", copy=False)
    out = np.zeros(arr.shape, dtype="uint8")
    for i in range(arr.shape[0]):
        band = arr[i]
        finite = np.isfinite(band)
        if not finite.any():
            continue
        valid = band[finite]
        try:
            if str(method).lower().startswith("percentile"):
                # Accept percentile_2_98, percentile_1_99, percentile_0_100.
                nums = [float(x) for x in str(method).replace("percentile", "").replace("_", " ").split() if x.replace('.', '', 1).isdigit()]
                lo_p, hi_p = (nums[0], nums[1]) if len(nums) >= 2 else (2.0, 98.0)
                lo = float(np.nanpercentile(valid, lo_p)); hi = float(np.nanpercentile(valid, hi_p))
            else:
                lo = float(np.nanmin(valid)); hi = float(np.nanmax(valid))
        except Exception:
            lo = float(np.nanmin(valid)); hi = float(np.nanmax(valid))
        if hi <= lo:
            hi = lo + 1.0
        out[i] = (np.clip((band - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype("uint8")
    return out


def _rgb_from_band_array(arr: Any, profile: Dict[str, Any]) -> Any:
    np = _np()
    if np is None:
        return arr
    arr = _normalize(arr, profile.get("normalization", "percentile_2_98"))
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.shape[0] == 0:
        return np.zeros((1, 1, 3), dtype="uint8")
    if arr.shape[0] == 1:
        arr = np.repeat(arr[:1], 3, axis=0)
    elif arr.shape[0] == 2:
        arr = np.concatenate([arr, arr[1:2]], axis=0)
    elif arr.shape[0] > 3:
        arr = arr[:3]
    return np.transpose(arr[:3], (1, 2, 0)).astype("uint8", copy=False)


def _pil_from_rgb_array(rgb: Any):
    Image = _pil_image()
    if Image is None:
        return rgb
    return Image.fromarray(rgb, mode="RGB")


class _MultibandRasterReader:
    """Small wrapper exposing rasterio-like attributes plus a selected-band profile."""
    def __init__(self, src: Any, path: str, profile: Dict[str, Any]):
        self.src = src
        self.path = str(path)
        self.profile = dict(profile or {})
        self.width = int(src.width)
        self.height = int(src.height)
        self.count = int(src.count)
        self.crs = getattr(src, "crs", None)
        self.transform = getattr(src, "transform", None)
        self.nodata = getattr(src, "nodata", None)

    def close(self):
        try:
            return self.src.close()
        except Exception:
            return None

    def read(self, *args, **kwargs):
        return self.src.read(*args, **kwargs)

    def read_rgb_window(self, x: int, y: int, size: int, W: int, H: int):
        rio = _rasterio()
        if rio is None:
            raise RuntimeError("rasterio is required for multiband window reads")
        from rasterio.windows import Window
        w = max(1, min(int(size), int(W) - int(x)))
        h = max(1, min(int(size), int(H) - int(y)))
        bands = _parse_bands(self.profile.get("selected_bands", [1, 2, 3]), max_band=self.count)
        arr = self.src.read(bands, window=Window(int(x), int(y), w, h), boundless=True, fill_value=0, masked=False)
        rgb = _rgb_from_band_array(arr, self.profile)
        return _pil_from_rgb_array(rgb)


def _tiff_band_count(path: str) -> int:
    rio = _rasterio()
    if rio is None or not _is_tiff(path):
        return 0
    try:
        with rio.open(path) as src:
            return int(src.count)
    except Exception:
        return 0


def _scan_tiff(path: str) -> Dict[str, Any]:
    rio = _rasterio()
    if rio is None:
        raise RuntimeError("rasterio is required")
    with rio.open(path) as src:
        return {
            "path": str(path),
            "width": int(src.width),
            "height": int(src.height),
            "count": int(src.count),
            "crs": str(src.crs) if src.crs else "",
            "transform": list(src.transform)[:6],
            "descriptions": list(src.descriptions or []),
            "dtypes": [str(x) for x in src.dtypes],
            "nodata": src.nodata,
        }


def _profile_for_tiff(path: str, workspace: Any = None) -> Optional[Dict[str, Any]]:
    count = _tiff_band_count(path)
    if count <= 0:
        return None
    prof = _find_profile(path, workspace=workspace, max_band=count)
    if prof is None or not bool(prof.get("enabled", True)):
        return None
    return prof


def _read_full_tiff_as_pil(path: str, workspace: Any = None, max_side: Optional[int] = None):
    rio = _rasterio()
    if rio is None:
        raise RuntimeError("rasterio is required")
    with rio.open(path) as src:
        prof = _find_profile(path, workspace=workspace, max_band=int(src.count)) or _default_profile(path, max_band=int(src.count))
        bands = _parse_bands(prof.get("selected_bands", [1, 2, 3]), max_band=int(src.count))
        out_shape = None
        if max_side:
            scale = min(1.0, float(max_side) / float(max(src.width, src.height)))
            out_w = max(1, int(round(src.width * scale)))
            out_h = max(1, int(round(src.height * scale)))
            out_shape = (len(bands), out_h, out_w)
        arr = src.read(bands, out_shape=out_shape, masked=False)
        return _pil_from_rgb_array(_rgb_from_band_array(arr, prof))


def _get_image_size(path: str) -> Tuple[int, int]:
    if _is_tiff(path):
        rio = _rasterio()
        if rio is not None:
            try:
                with rio.open(path) as src:
                    return int(src.width), int(src.height)
            except Exception:
                pass
    Image = _pil_image()
    if Image is None:
        raise RuntimeError("Pillow is required to read image size")
    orig = _ORIG.get("PIL.Image.open") or Image.open
    im = orig(path)
    try:
        return int(im.size[0]), int(im.size[1])
    finally:
        try: im.close()
        except Exception: pass


def _patch_backend_io(backend: Any) -> None:
    if backend is None:
        return
    if "backend.open_img" not in _ORIG and hasattr(backend, "open_img"):
        _ORIG["backend.open_img"] = backend.open_img
    if "backend.read_tile" not in _ORIG and hasattr(backend, "read_tile"):
        _ORIG["backend.read_tile"] = backend.read_tile
    if "backend.load_preview" not in _ORIG and hasattr(backend, "load_preview"):
        _ORIG["backend.load_preview"] = backend.load_preview
    if "backend.read_boxes" not in _ORIG and hasattr(backend, "read_boxes"):
        _ORIG["backend.read_boxes"] = backend.read_boxes

    def patched_open_img(path, log):
        p = _safe_str(path)
        workspace = _owner_from_callable(log)
        prof = _profile_for_tiff(p, workspace=workspace) if _is_tiff(p) else None
        if prof is not None:
            rio = _rasterio()
            if rio is None:
                return _ORIG["backend.open_img"](path, log)
            src = rio.open(p)
            try:
                bands = _parse_bands(prof.get("selected_bands", [1, 2, 3]), max_band=int(src.count))
                prof["selected_bands"] = bands
                prof["adapter_mode"] = "universal_rgb_composite"
                prof["model_input_channels"] = 3
                _save_profile_near_project(workspace, prof)
                if log:
                    log(f"Multiband All-AI active: {Path(p).name}, raster bands={src.count}, selected={bands}, adapter=RGB composite, normalization={prof.get('normalization')}")
                return int(src.width), int(src.height), "rasterio", _MultibandRasterReader(src, p, prof)
            except Exception:
                try: src.close()
                except Exception: pass
                raise
        return _ORIG["backend.open_img"](path, log)

    def patched_read_tile(reader, mode, x, y, size, W, H):
        if isinstance(reader, _MultibandRasterReader):
            return reader.read_rgb_window(int(x), int(y), int(size), int(W), int(H))
        return _ORIG["backend.read_tile"](reader, mode, x, y, size, W, H)

    def patched_load_preview(path, maxs=2400):
        p = _safe_str(path)
        prof = _profile_for_tiff(p, workspace=_LAST_WORKSPACE) if _is_tiff(p) else None
        if prof is not None:
            im = _read_full_tiff_as_pil(p, workspace=_LAST_WORKSPACE, max_side=int(maxs or 2400))
            W, H = _get_image_size(p)
            return im, W, H
        return _ORIG["backend.load_preview"](path, maxs=maxs)

    def patched_read_boxes(img, label):
        # Same logic as legacy backend.read_boxes, but rasterio supplies TIFF size.
        p = Path(img)
        if not Path(label).exists():
            return []
        W, H = _get_image_size(str(p))
        out = []
        for line in Path(label).read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                c = int(float(parts[0])); cx, cy, bw, bh = map(float, parts[1:])
                out.append([c, (cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H])
            except Exception:
                continue
        return out

    backend.open_img = patched_open_img
    backend.read_tile = patched_read_tile
    backend.load_preview = patched_load_preview
    backend.read_boxes = patched_read_boxes


def _project_has_multiband_images(workspace: Any, root: Path) -> bool:
    imgroot = root / "images"
    if not imgroot.exists():
        return False
    for p in imgroot.rglob("*"):
        if p.suffix.lower() in {".tif", ".tiff"}:
            try:
                if _tiff_band_count(str(p)) > 3:
                    return True
            except Exception:
                pass
    return False


def _patch_backend_training(backend: Any) -> None:
    if backend is None or not hasattr(backend, "GUI"):
        return
    GUI = backend.GUI
    if "backend.GUI.prepare_yolo_dataset" not in _ORIG and hasattr(GUI, "prepare_yolo_dataset"):
        _ORIG["backend.GUI.prepare_yolo_dataset"] = GUI.prepare_yolo_dataset
    else:
        return

    def patched_prepare_yolo_dataset(self):
        global _LAST_WORKSPACE
        _LAST_WORKSPACE = self
        root = Path(_safe_str(_var_get(self, "project", ".")) or ".").expanduser()
        mb = _project_has_multiband_images(self, root)
        old_chunk = None
        if mb:
            prof = _find_profile(workspace=self, max_band=None) or _default_profile(None)
            # Ensure all normal/foundation trainers receive RGB chips, not raw 8/13-band files.
            prof["adapter_mode"] = "universal_rgb_composite"
            prof["model_input_channels"] = 3
            saved = _save_profile_near_project(self, prof)
            try:
                if hasattr(self, "tmsg"):
                    self.tmsg(f"Multiband All-AI patch active for training. Bands={prof.get('selected_bands')} normalization={prof.get('normalization')} profile={saved}")
            except Exception:
                pass
            try:
                # Important: if chunking is disabled, legacy code copies original TIFFs.
                # Force chunking so every model receives RGB composite chips generated by patched read_tile().
                old_chunk = self.train_chunk_enabled.get() if hasattr(self.train_chunk_enabled, "get") else None
                self.train_chunk_enabled.set(True)
            except Exception:
                pass
        yml = _ORIG["backend.GUI.prepare_yolo_dataset"](self)
        if mb:
            try:
                # Copy profile into the dataset for later Detection to pick up the same bands.
                prof = _find_profile(workspace=self) or _default_profile(None)
                yml_p = Path(yml)
                outp = yml_p.parent / PROFILE_NAME
                outp.write_text(json.dumps(prof, indent=2, ensure_ascii=False), encoding="utf-8")
                # Mark YOLO dataset as RGB because this universal patch emits RGB composites.
                txt = yml_p.read_text(encoding="utf-8")
                if "mustatil_multiband_profile" not in txt:
                    txt += f"\n# mustatil_multiband_profile: {PROFILE_NAME}\n# multiband_adapter: universal_rgb_composite\n"
                    yml_p.write_text(txt, encoding="utf-8")
                if hasattr(self, "tmsg"):
                    self.tmsg(f"Multiband profile copied into dataset: {outp}")
            except Exception as exc:
                if hasattr(self, "tmsg"):
                    self.tmsg(f"Multiband profile copy warning: {exc}")
        return yml

    GUI.prepare_yolo_dataset = patched_prepare_yolo_dataset



# ---------------------------------------------------------------------------
# Direct rasterio bridge for R-CNN / Mask R-CNN / U-Net / SAM2 helper plugins
# ---------------------------------------------------------------------------
_RASTERIO_BRIDGE_PATCHED = False
_RASTERIO_BRIDGE_LOG_COUNTS: Dict[str, int] = {}

def _bridge_log(msg: str) -> None:
    try:
        ws = _LAST_WORKSPACE
        if ws is not None and hasattr(ws, "log"):
            ws.log(msg)
            return
    except Exception:
        pass
    try:
        print(msg, flush=True)
    except Exception:
        pass

def _bridge_log_limited(key: str, msg: str) -> None:
    try:
        n = int(_RASTERIO_BRIDGE_LOG_COUNTS.get(key, 0)) + 1
        _RASTERIO_BRIDGE_LOG_COUNTS[key] = n
        if n <= 10 or n in {25, 50, 100, 250, 500, 1000}:
            suffix = "" if n == 1 else f" (count={n})"
            _bridge_log(msg + suffix)
    except Exception:
        _bridge_log(msg)

def _indexes_to_list(indexes: Any) -> Optional[List[int]]:
    if indexes is None:
        return None
    if isinstance(indexes, int):
        return [int(indexes)]
    if isinstance(indexes, range):
        return [int(x) for x in indexes]
    if isinstance(indexes, (list, tuple)):
        try:
            return [int(x) for x in indexes]
        except Exception:
            return None
    return None

class _DirectRasterioMultibandBridge:
    def __init__(self, src: Any, path: str):
        self._src = src
        self._path = str(path or getattr(src, "name", "") or "")
    def __getattr__(self, name: str) -> Any:
        return getattr(self._src, name)
    def __enter__(self):
        try:
            entered = self._src.__enter__()
            if entered is not self._src:
                return _DirectRasterioMultibandBridge(entered, self._path)
        except Exception:
            pass
        return self
    def __exit__(self, exc_type, exc, tb):
        try:
            return self._src.__exit__(exc_type, exc, tb)
        except Exception:
            try: self._src.close()
            except Exception: pass
            return False
    def close(self):
        try: return self._src.close()
        except Exception: return None
    def read(self, indexes=None, *args, **kwargs):
        try:
            count = int(getattr(self._src, "count", 0) or 0)
            if count > 3 and _is_tiff(self._path):
                prof = _find_profile(self._path, workspace=_LAST_WORKSPACE, max_band=count)
                if prof is not None and bool(prof.get("enabled", True)):
                    selected = _parse_bands(prof.get("selected_bands", [1, 2, 3]), max_band=count)[:3]
                    req = _indexes_to_list(indexes)
                    swap_all = str(os.environ.get("MUSTATIL_MULTIBAND_SWAP_RASTERIO_READ_ALL", "1")).strip().lower() not in {"0", "false", "no", "off"}
                    new_indexes = None; old = None
                    if req == [1, 2, 3] and selected != [1, 2, 3]:
                        new_indexes = selected; old = req
                    elif indexes is None and swap_all:
                        new_indexes = selected; old = "all"
                    if new_indexes is not None:
                        key = f"direct_rasterio|{Path(self._path).name}|{old}->{new_indexes}"
                        win = " window=yes" if kwargs.get("window", None) is not None else ""
                        _bridge_log_limited(key, f"Multiband R-CNN/rasterio bridge active: {Path(self._path).name}, raster bands={count}, read {old} -> {new_indexes}{win}")
                        indexes = new_indexes
        except Exception as exc:
            _warn_once("direct_rasterio_bridge_read", f"Direct rasterio bridge warning: {exc}")
        return self._src.read(indexes, *args, **kwargs)

def _patch_direct_rasterio_reads() -> None:
    global _RASTERIO_BRIDGE_PATCHED
    if _RASTERIO_BRIDGE_PATCHED:
        return
    rio = _rasterio()
    if rio is None:
        return
    if getattr(rio, "_mustatil_multiband_direct_rasterio_v5", False):
        _RASTERIO_BRIDGE_PATCHED = True
        return
    orig_open = getattr(rio, "open", None)
    if orig_open is None:
        return
    _ORIG.setdefault("rasterio.open", orig_open)
    def patched_rasterio_open(fp, *args, **kwargs):
        src = orig_open(fp, *args, **kwargs)
        try:
            mode = str(kwargs.get("mode", "") or (args[0] if args else "") or "r")
            if "w" in mode or "+" in mode:
                return src
            if not _is_tiff(fp):
                return src
            count = int(getattr(src, "count", 0) or 0)
            if count <= 3:
                return src
            prof = _find_profile(str(fp), workspace=_LAST_WORKSPACE, max_band=count)
            if prof is None or not bool(prof.get("enabled", True)):
                return src
            selected = _parse_bands(prof.get("selected_bands", [1, 2, 3]), max_band=count)[:3]
            _bridge_log_limited(f"direct_rasterio_open|{Path(str(fp)).name}", f"Multiband R-CNN/rasterio bridge armed: {Path(str(fp)).name}, raster bands={count}, selected={selected}")
            return _DirectRasterioMultibandBridge(src, str(fp))
        except Exception:
            return src
    rio.open = patched_rasterio_open
    setattr(rio, "_mustatil_multiband_direct_rasterio_v5", True)
    _RASTERIO_BRIDGE_PATCHED = True
    _log("Direct rasterio bridge installed for R-CNN / Mask R-CNN / U-Net / SAM2 plugins.")


def _patch_pil_open() -> None:
    Image = _pil_image()
    if Image is None:
        return
    if "PIL.Image.open" not in _ORIG:
        _ORIG["PIL.Image.open"] = Image.open
    else:
        return

    def patched_image_open(fp, *args, **kwargs):
        # Safety net for SAM2/Grounding/OWL/LAE-DINO/RCNN/UNet plugins that read images through PIL.
        try:
            path = None
            if isinstance(fp, (str, bytes, os.PathLike)):
                path = os.fspath(fp)
            if path and _is_tiff(path):
                count = _tiff_band_count(path)
                if count > 3:
                    prof = _find_profile(path, workspace=_LAST_WORKSPACE, max_band=count)
                    if prof is not None and bool(prof.get("enabled", True)):
                        max_side = int(os.environ.get("MUSTATIL_MULTIBAND_PIL_MAX_SIDE", "0") or "0") or None
                        # If an AI plugin calls PIL directly, returning a composite prevents crashes.
                        return _read_full_tiff_as_pil(path, workspace=_LAST_WORKSPACE, max_side=max_side)
        except Exception as exc:
            _warn_once(f"pil:{fp}", f"PIL multiband adapter failed for {fp}: {exc}; using original PIL loader")
        return _ORIG["PIL.Image.open"](fp, *args, **kwargs)

    Image.open = patched_image_open


def _patch_cv2_imread() -> None:
    cv2 = sys.modules.get("cv2")
    if cv2 is None:
        try:
            import cv2 as cv2  # type: ignore
        except Exception:
            return
    if "cv2.imread" in _ORIG or not hasattr(cv2, "imread"):
        return
    _ORIG["cv2.imread"] = cv2.imread

    def patched_imread(filename, flags=None):
        try:
            p = _safe_str(filename)
            if _is_tiff(p):
                count = _tiff_band_count(p)
                if count > 3:
                    prof = _find_profile(p, workspace=_LAST_WORKSPACE, max_band=count)
                    if prof is not None and bool(prof.get("enabled", True)):
                        im = _read_full_tiff_as_pil(p, workspace=_LAST_WORKSPACE, max_side=None)
                        np = _np()
                        if np is not None:
                            arr = np.asarray(im)
                            # cv2 uses BGR by convention.
                            return arr[:, :, ::-1].copy()
        except Exception as exc:
            _warn_once(f"cv2:{filename}", f"cv2 multiband adapter failed for {filename}: {exc}; using original cv2.imread")
        if flags is None:
            return _ORIG["cv2.imread"](filename)
        return _ORIG["cv2.imread"](filename, flags)

    cv2.imread = patched_imread


def _cache_composite_path(path: str, profile: Dict[str, Any]) -> Optional[str]:
    try:
        p = Path(path)
        key = json.dumps({
            "path": str(p.resolve()),
            "mtime": p.stat().st_mtime_ns,
            "size": p.stat().st_size,
            "bands": profile.get("selected_bands"),
            "norm": profile.get("normalization"),
        }, sort_keys=True, default=str)
        h = hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()[:16]
        cache = p.parent / ".mustatil_multiband_cache"
        cache.mkdir(parents=True, exist_ok=True)
        out = cache / f"{p.stem}_mb_{h}.png"
        if not out.exists():
            im = _read_full_tiff_as_pil(str(p), workspace=_LAST_WORKSPACE, max_side=None)
            im.save(out)
        return str(out)
    except Exception as exc:
        _warn_once(f"cache:{path}", f"Could not create multiband composite cache for {path}: {exc}")
        return None


def _patch_ultralytics() -> None:
    try:
        from ultralytics import YOLO
    except Exception:
        return
    if "ultralytics.YOLO.predict" not in _ORIG and hasattr(YOLO, "predict"):
        _ORIG["ultralytics.YOLO.predict"] = YOLO.predict
    if "ultralytics.YOLO.train" not in _ORIG and hasattr(YOLO, "train"):
        _ORIG["ultralytics.YOLO.train"] = YOLO.train

    def patched_predict(self, source=None, *args, **kwargs):
        try:
            src = source if source is not None else kwargs.get("source")
            if isinstance(src, (str, os.PathLike)) and _is_tiff(src):
                p = os.fspath(src)
                count = _tiff_band_count(p)
                if count > 3:
                    prof = _find_profile(p, workspace=_LAST_WORKSPACE, max_band=count)
                    if prof is not None and bool(prof.get("enabled", True)):
                        comp = _cache_composite_path(p, prof)
                        if comp:
                            if source is not None:
                                source = comp
                            else:
                                kwargs["source"] = comp
        except Exception as exc:
            _warn_once("yolo_predict", f"YOLO.predict multiband path adapter warning: {exc}")
        return _ORIG["ultralytics.YOLO.predict"](self, source, *args, **kwargs)

    # Do not over-transform arbitrary YOLO training YAMLs here. The Mustatil
    # trainer is patched at dataset preparation time, which is safer. The train
    # wrapper only logs the profile when present.
    def patched_train(self, *args, **kwargs):
        return _ORIG["ultralytics.YOLO.train"](self, *args, **kwargs)

    YOLO.predict = patched_predict
    YOLO.train = patched_train


def _patch_detection_preview_service() -> None:
    svc = sys.modules.get("detection_preview_service")
    if svc is None:
        return
    if not hasattr(svc, "load_detection_preview_image"):
        return
    if "detection_preview_service.load_detection_preview_image" in _ORIG:
        return
    _ORIG["detection_preview_service.load_detection_preview_image"] = svc.load_detection_preview_image

    def patched_load_detection_preview_image(path, maxs=2400, fallback_loader=None):
        p = _safe_str(path)
        prof = _profile_for_tiff(p, workspace=_LAST_WORKSPACE) if _is_tiff(p) else None
        if prof is not None:
            im = _read_full_tiff_as_pil(p, workspace=_LAST_WORKSPACE, max_side=int(maxs or 2400))
            W, H = _get_image_size(p)
            return im, W, H
        return _ORIG["detection_preview_service.load_detection_preview_image"](path, maxs=maxs, fallback_loader=fallback_loader)

    svc.load_detection_preview_image = patched_load_detection_preview_image


def _patch_workspace_class(cls: Any) -> None:
    global _CLASS_PATCHED
    if cls is None or _CLASS_PATCHED:
        return
    try:
        # Patch run_task so we always know the active workspace. This helps PIL/cv2 based plugins.
        if hasattr(cls, "run_task") and "MustatilQtWorkspace.run_task" not in _ORIG:
            _ORIG["MustatilQtWorkspace.run_task"] = cls.run_task
            def patched_run_task(self, name, func, *args, **kwargs):
                global _LAST_WORKSPACE
                _LAST_WORKSPACE = self
                return _ORIG["MustatilQtWorkspace.run_task"](self, name, func, *args, **kwargs)
            cls.run_task = patched_run_task

        # Patch selected-device detection wrapper to set active workspace.
        if hasattr(cls, "detect_with_selected_device") and "MustatilQtWorkspace.detect_with_selected_device" not in _ORIG:
            _ORIG["MustatilQtWorkspace.detect_with_selected_device"] = cls.detect_with_selected_device
            def patched_detect_with_selected_device(self, *args, **kwargs):
                global _LAST_WORKSPACE
                _LAST_WORKSPACE = self
                return _ORIG["MustatilQtWorkspace.detect_with_selected_device"](self, *args, **kwargs)
            cls.detect_with_selected_device = patched_detect_with_selected_device

        # Satellite Detection uses PIL chunks already; no multiband source there.
        # But if future satellite/AI plugins call TIFF model inputs directly, PIL/cv2/YOLO hooks cover them.
        _CLASS_PATCHED = True
        _log("Workspace class hooks installed: active workspace tracking for all AI tasks.")
    except Exception as exc:
        _warn_once("class_patch", f"Could not patch workspace class: {exc}")


def _install_compact_controls(workspace: Any) -> None:
    """No new tab: add tiny multiband checkbox + bands field to every trainer page."""
    if workspace is None or id(workspace) in _UI_PATCHED_IDS:
        return
    try:
        from PySide6.QtWidgets import QApplication, QTabWidget, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QCheckBox, QComboBox, QPushButton
    except Exception:
        return
    try:
        Var = MUSTATIL_GLOBALS.get("Var")
        if Var is None:
            class Var:  # type: ignore
                def __init__(self, value=None): self._v = value
                def get(self): return self._v
                def set(self, v): self._v = v
        if not hasattr(workspace, "mustatil_mb_enabled"):
            workspace.mustatil_mb_enabled = Var(True)
        if not hasattr(workspace, "mustatil_mb_bands"):
            workspace.mustatil_mb_bands = Var(os.environ.get("MUSTATIL_MULTIBAND_BANDS", "1,2,3"))
        if not hasattr(workspace, "mustatil_mb_normalization"):
            workspace.mustatil_mb_normalization = Var(os.environ.get("MUSTATIL_MULTIBAND_NORMALIZATION", "percentile_2_98"))
        def apply_profile(enabled_widget, bands_widget, norm_widget, verbose=False):
            try:
                workspace.mustatil_mb_enabled.set(bool(enabled_widget.isChecked()))
                workspace.mustatil_mb_bands.set(bands_widget.text().strip() or "1,2,3")
                workspace.mustatil_mb_normalization.set(norm_widget.currentText())
                prof = _default_profile(None)
                prof["enabled"] = bool(enabled_widget.isChecked())
                prof["selected_bands"] = _parse_bands(bands_widget.text().strip() or "1,2,3")
                prof["normalization"] = norm_widget.currentText()
                prof["adapter_mode"] = "universal_rgb_composite"
                prof["model_input_channels"] = 3
                saved = _save_profile_near_project(workspace, prof)
                os.environ["MUSTATIL_MULTIBAND_BANDS"] = bands_widget.text().strip() or "1,2,3"
                os.environ["MUSTATIL_MULTIBAND_NORMALIZATION"] = norm_widget.currentText()
                if verbose and hasattr(workspace, "log"):
                    workspace.log(f"Multiband training profile applied: enabled={enabled_widget.isChecked()}, bands={prof['selected_bands']}, normalization={prof['normalization']}, profile={saved}")
            except Exception as exc:
                if verbose and hasattr(workspace, "log"):
                    workspace.log(f"Multiband training profile warning: {exc}")
        def should_patch_tab(title):
            t = str(title or "").lower()
            positives = ["train", "trainer", "training", "yolo trainer", "rcnn trainer", "r-cnn trainer", "faster r-cnn", "mask r-cnn", "u-net trainer", "unet trainer", "lae-dino trainer", "sam2 trainer", "model trainer"]
            negatives = ["detection", "satellite", "preview", "pipeline", "annotator", "map"]
            if any(n in t for n in negatives) and not any(x in t for x in ["trainer", "training", "train"]):
                return False
            return any(p in t for p in positives)
        tab_widgets=[]
        try:
            main_tabs=getattr(workspace,"tabs",None)
            if main_tabs is not None: tab_widgets.append(main_tabs)
        except Exception: pass
        try:
            for tw in workspace.findChildren(QTabWidget):
                if tw not in tab_widgets: tab_widgets.append(tw)
        except Exception: pass
        inserted=0; patched_titles=[]
        for tabs in tab_widgets:
            try: count=tabs.count()
            except Exception: continue
            for i in range(count):
                try:
                    title=str(tabs.tabText(i) or "")
                    if not should_patch_tab(title): continue
                    page=tabs.widget(i)
                    if page is None or getattr(page,"_mustatil_mb_training_controls_v5",False): continue
                    layout=page.layout()
                    if layout is None or not hasattr(layout,"insertWidget"): continue
                    box=QGroupBox("Multiband")
                    box.setToolTip("Shared Multiband profile for this trainer and all AI models. RGB/foundation models receive a 3-band composite.")
                    row=QHBoxLayout(box); row.setContentsMargins(6,3,6,3)
                    enabled=QCheckBox("Multiband"); enabled.setChecked(bool(workspace.mustatil_mb_enabled.get()))
                    bands=QLineEdit(str(workspace.mustatil_mb_bands.get() or "1,2,3")); bands.setMaximumWidth(110); bands.setPlaceholderText("1,2,4")
                    norm=QComboBox(); norm.addItems(["percentile_2_98", "percentile_1_99", "minmax", "none"])
                    ix=norm.findText(str(workspace.mustatil_mb_normalization.get() or "percentile_2_98"))
                    if ix>=0: norm.setCurrentIndex(ix)
                    apply_btn=QPushButton("Apply"); apply_btn.setMaximumWidth(60)
                    row.addWidget(enabled); row.addWidget(QLabel("Bands:")); row.addWidget(bands); row.addWidget(QLabel("Norm:")); row.addWidget(norm); row.addWidget(apply_btn); row.addStretch(1)
                    enabled.toggled.connect(lambda _=False,e=enabled,b=bands,n=norm: apply_profile(e,b,n,False))
                    bands.editingFinished.connect(lambda e=enabled,b=bands,n=norm: apply_profile(e,b,n,False))
                    norm.currentTextChanged.connect(lambda _txt,e=enabled,b=bands,n=norm: apply_profile(e,b,n,False))
                    apply_btn.clicked.connect(lambda _=False,e=enabled,b=bands,n=norm: apply_profile(e,b,n,True))
                    layout.insertWidget(0,box)
                    setattr(page,"_mustatil_mb_training_controls_v5",True)
                    inserted += 1; patched_titles.append(title)
                except Exception as exc:
                    _warn_once(f"training_ui_{i}", f"Could not patch training UI page {i}: {exc}")
        if inserted:
            _UI_PATCHED_IDS.add(id(workspace))
            if hasattr(workspace,"log"):
                workspace.log(f"Multiband controls inserted into {inserted} trainer page(s): {', '.join(patched_titles[:8])}")
                workspace.log("R-CNN/Mask R-CNN/U-Net direct rasterio bridge is active; reads [1,2,3] are redirected to selected multiband bands when needed.")
    except Exception as exc:
        _warn_once("training_ui_controls", f"Could not add trainer multiband controls: {exc}")

def _watch_qt_app() -> None:
    """Retry after the main class/UI exists. Plugins are loaded before class definition."""
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
    except Exception:
        return

    def tick():
        global _LAST_WORKSPACE
        try:
            cls = None
            g = globals().get("MUSTATIL_GLOBALS") or MUSTATIL_GLOBALS or {}
            cls = g.get("MustatilQtWorkspace")
            if cls is not None:
                _patch_workspace_class(cls)
            app = QApplication.instance()
            if app is not None:
                for w in app.topLevelWidgets():
                    if cls is not None and isinstance(w, cls):
                        _LAST_WORKSPACE = w
                        pass  # v7 disables legacy compact controls
                    elif hasattr(w, "tabs") and hasattr(w, "project"):
                        _LAST_WORKSPACE = w
                        pass  # v7 disables legacy compact controls
            _patch_detection_preview_service()
            _v7_scan(verbose=False)
            _patch_direct_rasterio_reads()
            _patch_ultralytics()
            _patch_cv2_imread()
        except Exception as exc:
            _warn_once("watch", f"Watcher warning: {exc}")
        try:
            QTimer.singleShot(1500, tick)
        except Exception:
            pass

    try:
        QTimer.singleShot(500, tick)
    except Exception:
        pass



# ---------------------------------------------------------------------------
# Robust trainer UI injection v4
# ---------------------------------------------------------------------------
# v3 used a workspace/QTimer scan only. In some Mustatil starts the scan ran
# before all plugin-created trainer tabs existed or before the workspace could
# be identified. v4 additionally patches QTabWidget.addTab globally, so every
# future trainer/training page receives the compact Multiband controls at the
# moment the tab is added.

_QTAB_ADD_PATCHED_V4 = False


def _mb_title_is_training_page(title: Any) -> bool:
    t = str(title or "").lower().strip()
    if not t:
        return False
    # Explicit non-training tabs.
    hard_no = ["console", "detection", "satellite detection", "annotator", "auto updater", "pipeline", "webmap", "map"]
    if t in hard_no:
        return False

    # Strict positives.
    positives = [
        "train", "trainer", "training",
        "yolo trainer", "formtrainer",
        "r-cnn trainer", "rcnn trainer", "faster r-cnn trainer", "mask r-cnn trainer",
        "u-net trainer", "unet trainer", "lae-dino trainer", "sam2 trainer",
    ]
    if any(p in t for p in positives):
        return True

    # Model pages that often appear as trainer pages in add-on plugins.
    model_training_like = ["faster r-cnn", "mask r-cnn", "u-net", "unet", "lae-dino"]
    if any(p in t for p in model_training_like) and "detect" not in t and "satellite" not in t:
        return True

    return False


def _find_workspace_for_widget(widget: Any = None) -> Any:
    global _LAST_WORKSPACE

    # Cached active workspace.
    try:
        if _LAST_WORKSPACE is not None and hasattr(_LAST_WORKSPACE, "tabs"):
            return _LAST_WORKSPACE
    except Exception:
        pass

    # Walk parents from the tab page.
    try:
        w = widget
        for _ in range(20):
            if w is None:
                break
            if hasattr(w, "tabs") and (hasattr(w, "project") or hasattr(w, "log")):
                _LAST_WORKSPACE = w
                return w
            w = w.parent()
    except Exception:
        pass

    # Search top-level windows.
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            for w in app.topLevelWidgets():
                if hasattr(w, "tabs") and (hasattr(w, "project") or hasattr(w, "log")):
                    _LAST_WORKSPACE = w
                    return w
    except Exception:
        pass

    return None


def _ensure_workspace_mb_vars(workspace: Any) -> None:
    if workspace is None:
        return
    try:
        Var = (globals().get("MUSTATIL_GLOBALS") or {}).get("Var")
    except Exception:
        Var = None
    if Var is None:
        class Var:  # type: ignore
            def __init__(self, value=None): self._v = value
            def get(self): return self._v
            def set(self, v): self._v = v
    try:
        if not hasattr(workspace, "mustatil_mb_enabled"):
            workspace.mustatil_mb_enabled = Var(True)
        if not hasattr(workspace, "mustatil_mb_bands"):
            workspace.mustatil_mb_bands = Var(os.environ.get("MUSTATIL_MULTIBAND_BANDS", "1,2,4"))
        if not hasattr(workspace, "mustatil_mb_normalization"):
            workspace.mustatil_mb_normalization = Var(os.environ.get("MUSTATIL_MULTIBAND_NORMALIZATION", "percentile_2_98"))
    except Exception:
        pass


def _compact_multiband_controls_already_present(page: Any) -> bool:
    try:
        if getattr(page, "_mustatil_mb_training_controls_v5", False):
            return True
        from PySide6.QtWidgets import QGroupBox
        for gb in page.findChildren(QGroupBox):
            try:
                if gb.objectName() == "mustatil_multiband_training_controls_v5":
                    return True
                if str(gb.title()).strip().lower() == "multiband":
                    # Avoid duplicates if a previous version inserted controls.
                    return True
            except Exception:
                pass
    except Exception:
        pass
    return False


def _insert_compact_multiband_controls_into_page(page: Any, title: str = "", workspace: Any = None, source: str = "addTab") -> bool:
    """Insert one compact Multiband control row into a specific trainer page."""
    if page is None:
        return False
    try:
        from PySide6.QtWidgets import (
            QGroupBox, QHBoxLayout, QLabel, QLineEdit, QCheckBox,
            QComboBox, QPushButton, QScrollArea, QWidget
        )
    except Exception:
        return False

    try:
        target = page
        # If a plugin adds a QScrollArea as a tab page, patch its contained widget.
        if hasattr(page, "widget") and page.__class__.__name__.lower().endswith("scrollarea"):
            try:
                inner = page.widget()
                if inner is not None:
                    target = inner
            except Exception:
                pass

        if _compact_multiband_controls_already_present(target):
            return False

        layout = target.layout()
        if layout is None:
            return False

        workspace = workspace or _find_workspace_for_widget(target)
        _ensure_workspace_mb_vars(workspace)

        try:
            current_enabled = bool(_workspace_setting(workspace, "mustatil_mb_enabled", True))
            current_bands = str(_workspace_setting(workspace, "mustatil_mb_bands", os.environ.get("MUSTATIL_MULTIBAND_BANDS", "1,2,4")) or "1,2,4")
            current_norm = str(_workspace_setting(workspace, "mustatil_mb_normalization", os.environ.get("MUSTATIL_MULTIBAND_NORMALIZATION", "percentile_2_98")) or "percentile_2_98")
        except Exception:
            current_enabled = True
            current_bands = os.environ.get("MUSTATIL_MULTIBAND_BANDS", "1,2,4")
            current_norm = os.environ.get("MUSTATIL_MULTIBAND_NORMALIZATION", "percentile_2_98")

        box = QGroupBox("Multiband")
        box.setObjectName("mustatil_multiband_training_controls_v5")
        box.setToolTip("Shared Mustatil multiband profile. RGB/foundation models receive a 3-band composite; direct rasterio R-CNN reads are redirected to these bands.")
        row = QHBoxLayout(box)
        row.setContentsMargins(6, 3, 6, 3)

        enabled = QCheckBox("Multiband")
        enabled.setChecked(current_enabled)

        bands = QLineEdit(current_bands)
        bands.setMaximumWidth(110)
        bands.setPlaceholderText("1,2,4")

        norm = QComboBox()
        norm.addItems(["percentile_2_98", "percentile_1_99", "minmax", "none"])
        ix = norm.findText(current_norm)
        if ix >= 0:
            norm.setCurrentIndex(ix)

        apply_btn = QPushButton("Apply")
        apply_btn.setMaximumWidth(64)

        row.addWidget(enabled)
        row.addWidget(QLabel("Bands:"))
        row.addWidget(bands)
        row.addWidget(QLabel("Norm:"))
        row.addWidget(norm)
        row.addWidget(apply_btn)
        row.addStretch(1)

        def apply_profile(verbose=False):
            ws = _find_workspace_for_widget(target)
            _ensure_workspace_mb_vars(ws)
            band_text = bands.text().strip() or "1,2,4"
            norm_text = norm.currentText()
            os.environ["MUSTATIL_MULTIBAND_BANDS"] = band_text
            os.environ["MUSTATIL_MULTIBAND_NORMALIZATION"] = norm_text
            os.environ["MUSTATIL_MULTIBAND_ENABLED"] = "1" if enabled.isChecked() else "0"
            prof = _default_profile(None)
            prof["enabled"] = bool(enabled.isChecked())
            prof["selected_bands"] = _parse_bands(band_text)
            prof["normalization"] = norm_text
            prof["adapter_mode"] = "universal_rgb_composite"
            prof["model_input_channels"] = 3
            saved = None
            try:
                if ws is not None:
                    if hasattr(getattr(ws, "mustatil_mb_enabled", None), "set"):
                        ws.mustatil_mb_enabled.set(bool(enabled.isChecked()))
                    if hasattr(getattr(ws, "mustatil_mb_bands", None), "set"):
                        ws.mustatil_mb_bands.set(band_text)
                    if hasattr(getattr(ws, "mustatil_mb_normalization", None), "set"):
                        ws.mustatil_mb_normalization.set(norm_text)
                    saved = _save_profile_near_project(ws, prof)
            except Exception:
                pass
            if verbose:
                msg = f"Multiband profile applied from {title or 'trainer page'}: enabled={enabled.isChecked()}, bands={prof['selected_bands']}, normalization={norm_text}"
                if saved:
                    msg += f", profile={saved}"
                try:
                    if ws is not None and hasattr(ws, "log"):
                        ws.log(msg)
                    else:
                        print(f"[{PLUGIN_NAME}] {msg}", flush=True)
                except Exception:
                    pass

        enabled.toggled.connect(lambda _=False: apply_profile(False))
        bands.editingFinished.connect(lambda: apply_profile(False))
        norm.currentTextChanged.connect(lambda _txt: apply_profile(False))
        apply_btn.clicked.connect(lambda _=False: apply_profile(True))

        # Put the controls at the top. Support common layouts.
        inserted = False
        try:
            if hasattr(layout, "insertWidget"):
                layout.insertWidget(0, box)
                inserted = True
            elif hasattr(layout, "insertRow"):
                layout.insertRow(0, box)
                inserted = True
            elif hasattr(layout, "addWidget"):
                layout.addWidget(box)
                inserted = True
        except Exception:
            inserted = False

        if not inserted:
            return False

        setattr(target, "_mustatil_mb_training_controls_v5", True)
        setattr(page, "_mustatil_mb_training_controls_v5", True)

        try:
            ws = workspace or _find_workspace_for_widget(target)
            if ws is not None and hasattr(ws, "log"):
                ws.log(f"Multiband controls inserted into trainer page: {title or target.__class__.__name__} ({source})")
            else:
                print(f"[{PLUGIN_NAME}] Multiband controls inserted into trainer page: {title}", flush=True)
        except Exception:
            pass

        # Apply current defaults silently, so env/profile state is coherent.
        try:
            apply_profile(False)
        except Exception:
            pass

        return True

    except Exception as exc:
        _warn_once(f"ui_insert_{title}", f"Could not insert compact multiband controls into {title}: {exc}")
        return False


def _patch_qtabwidget_addtab_for_training_controls() -> bool:
    global _QTAB_ADD_PATCHED_V4
    if _QTAB_ADD_PATCHED_V4:
        return True
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QTabWidget
    except Exception:
        return False

    if getattr(QTabWidget, "_mustatil_multiband_addtab_patch_v5", False):
        _QTAB_ADD_PATCHED_V4 = True
        return True

    original_add_tab = QTabWidget.addTab

    def patched_addTab(self, *args, **kwargs):
        # Determine page/title before calling original.
        page = args[0] if args else None
        title = ""
        try:
            if len(args) >= 2 and isinstance(args[1], str):
                title = args[1]
            elif len(args) >= 3 and isinstance(args[2], str):
                title = args[2]
        except Exception:
            title = ""

        result = original_add_tab(self, *args, **kwargs)

        try:
            if _mb_title_is_training_page(title):
                # Direct attempt plus delayed attempts. Some plugin pages receive
                # their layout shortly after addTab().
                ws = _find_workspace_for_widget(page)
                _insert_compact_multiband_controls_into_page(page, title, ws, source="QTabWidget.addTab")
                for delay in (100, 500, 1500, 3000):
                    QTimer.singleShot(delay, lambda p=page, t=title: _insert_compact_multiband_controls_into_page(p, t, _find_workspace_for_widget(p), source=f"delayed {delay}ms"))
        except Exception as exc:
            _warn_once("qtab_addtab_patch", f"Trainer addTab multiband hook warning: {exc}")

        return result

    try:
        QTabWidget.addTab = patched_addTab
        QTabWidget._mustatil_multiband_addtab_patch_v5 = True
        _QTAB_ADD_PATCHED_V4 = True
        _log("QTabWidget.addTab trainer UI hook installed: compact Multiband controls will be inserted as trainer tabs are created.")
        return True
    except Exception as exc:
        _warn_once("qtab_addtab_patch_install", f"Could not patch QTabWidget.addTab: {exc}")
        return False


def _aggressive_scan_training_controls_v5(verbose=False) -> int:
    """Manual/late scan for already-created trainer pages."""
    inserted = 0
    try:
        from PySide6.QtWidgets import QApplication, QTabWidget
        app = QApplication.instance()
        if app is None:
            return 0
        for w in app.topLevelWidgets():
            ws = _find_workspace_for_widget(w)
            try:
                for tw in w.findChildren(QTabWidget):
                    for i in range(tw.count()):
                        title = str(tw.tabText(i) or "")
                        if _mb_title_is_training_page(title):
                            if _insert_compact_multiband_controls_into_page(tw.widget(i), title, ws, source="late scan"):
                                inserted += 1
            except Exception:
                pass
    except Exception:
        pass
    return inserted



# ---------------------------------------------------------------------------
# Aggressive trainer-page finder v5
# ---------------------------------------------------------------------------
# Some add-on trainer plugins do not expose a simple page layout under a tab
# title like "R-CNN Trainer". They may add nested QScrollArea/QSplitter/QGroupBox
# pages later. v5 therefore scans both tab pages and their child group boxes and
# installs a small Multiband row wherever a training page/panel is recognized.

_AGGRESSIVE_UI_TIMER_V5 = None
_AGGRESSIVE_UI_FILTER_V5 = None
_AGGRESSIVE_UI_SCAN_COUNT_V5 = 0


def _safe_text_v5(obj: Any) -> str:
    parts = []
    for attr in ("objectName", "windowTitle", "title", "text"):
        try:
            v = getattr(obj, attr, None)
            if callable(v):
                v = v()
            if v:
                parts.append(str(v))
        except Exception:
            pass
    return " | ".join(parts)


def _tab_title_for_widget_v5(widget: Any) -> str:
    try:
        from PySide6.QtWidgets import QApplication, QTabWidget
        app = QApplication.instance()
        if app is None:
            return ""
        w = widget
        ancestors = set()
        for _ in range(80):
            if w is None:
                break
            ancestors.add(w)
            try:
                w = w.parent()
            except Exception:
                break
        for top in app.topLevelWidgets():
            for tw in top.findChildren(QTabWidget):
                for i in range(tw.count()):
                    page = tw.widget(i)
                    if page in ancestors or page is widget:
                        return str(tw.tabText(i) or "")
    except Exception:
        pass
    return ""


def _signature_with_context_v5(widget: Any) -> str:
    txt = [_safe_text_v5(widget), _tab_title_for_widget_v5(widget)]
    try:
        p = widget.parent()
        for _ in range(8):
            if p is None:
                break
            s = _safe_text_v5(p)
            if s:
                txt.append(s)
            p = p.parent()
    except Exception:
        pass
    return " | ".join([x for x in txt if x]).lower()


def _training_candidate_v5(widget: Any, strict_page: bool = False) -> bool:
    sig = _signature_with_context_v5(widget)
    own = _safe_text_v5(widget).lower()
    tab = _tab_title_for_widget_v5(widget).lower()

    # Never patch clearly non-training top-level pages unless the widget itself
    # explicitly says trainer/training.
    non_training_tabs = ["detection", "satellite detection", "annotator", "console", "pipeline", "auto updater", "formlearner detection"]
    if tab in non_training_tabs and not any(x in own for x in ["trainer", "training", "train", "create dataset", "start training"]):
        return False

    hard_negative = ["preview", "webmap", "red detection overlay", "post-detection", "formlearner detection"]
    if any(n in own for n in hard_negative) and not any(x in own for x in ["trainer", "training"]):
        return False

    strong = [
        "trainer", "training", "train yolo", "prepare yolo", "start training", "create dataset", "create config",
        "yolo trainer", "r-cnn trainer", "rcnn trainer", "faster r-cnn trainer", "mask r-cnn trainer",
        "u-net trainer", "unet trainer", "lae-dino trainer", "sam2 trainer",
        "anti-freeze panel", "dataset button",
    ]
    if any(x in sig for x in strong):
        return True

    # Ambiguous model names are only accepted when the tab context is training.
    if any(x in sig for x in ["faster r-cnn", "mask r-cnn", "u-net", "unet", "lae-dino"]):
        if any(x in tab for x in ["trainer", "training", "train"]):
            return True
        if any(x in own for x in ["checkpoint", "backbone", "classes", "epochs", "batch", "dataset"]):
            return True

    return False


def _page_has_multiband_controls_v5(page: Any) -> bool:
    try:
        from PySide6.QtWidgets import QGroupBox
        if getattr(page, "_mustatil_mb_training_controls_v5", False):
            return True
        for gb in page.findChildren(QGroupBox):
            try:
                if gb.objectName() == "mustatil_multiband_training_controls_v5":
                    return True
                if str(gb.title()).strip().lower() == "multiband":
                    return True
            except Exception:
                pass
    except Exception:
        pass
    return False


def _aggressive_scan_training_controls_v5(verbose: bool = False) -> int:
    inserted = 0
    try:
        from PySide6.QtWidgets import QApplication, QTabWidget, QGroupBox, QWidget
        app = QApplication.instance()
        if app is None:
            return 0
        for top in app.topLevelWidgets():
            ws = _find_workspace_for_widget(top)

            # 1) First try actual tab pages.
            for tw in top.findChildren(QTabWidget):
                for i in range(tw.count()):
                    page = tw.widget(i)
                    title = str(tw.tabText(i) or "")
                    if _mb_title_is_training_page(title) or _training_candidate_v5(page, strict_page=True):
                        if _insert_compact_multiband_controls_into_page(page, title, ws, source="v5 tab/page scan"):
                            inserted += 1

            # 2) Then try candidate QGroupBoxes, because many add-on trainer
            # pages are nested panels with their own layouts.
            for gb in top.findChildren(QGroupBox):
                try:
                    if gb.objectName() == "mustatil_multiband_training_controls_v5":
                        continue
                    if str(gb.title()).strip().lower() == "multiband":
                        continue
                    if _page_has_multiband_controls_v5(gb):
                        continue
                    if _training_candidate_v5(gb):
                        title = str(gb.title() or _tab_title_for_widget_v5(gb) or gb.objectName() or "Trainer panel")
                        if _insert_compact_multiband_controls_into_page(gb, title, ws, source="v5 groupbox scan"):
                            inserted += 1
                except Exception:
                    pass

            # 3) Last fallback: generic QWidget with layout inside training tab.
            # This catches custom panels that are not QGroupBox.
            for w in top.findChildren(QWidget):
                try:
                    if w is top or isinstance(w, QTabWidget) or isinstance(w, QGroupBox):
                        continue
                    if _page_has_multiband_controls_v5(w):
                        continue
                    if w.layout() is None:
                        continue
                    if _training_candidate_v5(w):
                        title = _tab_title_for_widget_v5(w) or _safe_text_v5(w) or "Trainer widget"
                        if _insert_compact_multiband_controls_into_page(w, title, ws, source="v5 widget scan"):
                            inserted += 1
                except Exception:
                    pass

        if verbose and inserted:
            _log(f"v5 aggressive scan inserted Multiband controls into {inserted} training panel(s).")
    except Exception as exc:
        _warn_once("v5_aggressive_scan", f"v5 aggressive Multiband UI scan warning: {exc}")
    return inserted


def _start_aggressive_training_ui_scanner_v5() -> bool:
    global _AGGRESSIVE_UI_TIMER_V5, _AGGRESSIVE_UI_FILTER_V5
    try:
        from PySide6.QtCore import QObject, QEvent, QTimer
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            # QTabWidget.addTab patch still catches later UI creation.
            return False

        # Burst of delayed scans for plugin tabs created after Mustatil starts.
        for delay in (250, 750, 1500, 3000, 6000, 10000, 15000, 25000, 40000):
            QTimer.singleShot(delay, lambda d=delay: _aggressive_scan_training_controls_v5(verbose=(d >= 3000)))

        if _AGGRESSIVE_UI_TIMER_V5 is None:
            timer = QTimer(app)
            timer.setInterval(1200)
            def tick():
                global _AGGRESSIVE_UI_SCAN_COUNT_V5
                _AGGRESSIVE_UI_SCAN_COUNT_V5 += 1
                _v7_scan(verbose=False)
                if _AGGRESSIVE_UI_SCAN_COUNT_V5 > 45:
                    try:
                        timer.stop()
                    except Exception:
                        pass
            timer.timeout.connect(tick)
            timer.start()
            _AGGRESSIVE_UI_TIMER_V5 = timer

        if _AGGRESSIVE_UI_FILTER_V5 is None:
            class _MBTrainingUiEventFilter(QObject):
                def eventFilter(self, obj, event):
                    try:
                        et = event.type()
                        if et in (QEvent.ChildAdded, QEvent.Show, QEvent.Polish, QEvent.LayoutRequest):
                            QTimer.singleShot(200, lambda: _aggressive_scan_training_controls_v5(verbose=False))
                    except Exception:
                        pass
                    return False
            filt = _MBTrainingUiEventFilter(app)
            app.installEventFilter(filt)
            _AGGRESSIVE_UI_FILTER_V5 = filt

        _log("v5 aggressive trainer UI scanner installed: scans tabs, nested GroupBoxes, custom trainer panels and delayed plugin pages.")
        return True
    except Exception as exc:
        _warn_once("v5_scanner_install", f"Could not install v5 aggressive trainer UI scanner: {exc}")
        return False


def _patch_workspace_showevent_v5() -> bool:
    """Patch the workspace showEvent as a last fallback after the window is visible."""
    try:
        g = globals().get("MUSTATIL_GLOBALS") or {}
        cls = g.get("MustatilQtWorkspace")
        if cls is None or getattr(cls, "_mustatil_multiband_showevent_patch_v5", False):
            return False
        old_show = getattr(cls, "showEvent", None)
        def patched_showEvent(self, event):
            global _LAST_WORKSPACE
            _LAST_WORKSPACE = self
            if old_show is not None:
                try:
                    old_show(self, event)
                except TypeError:
                    old_show(event)
            try:
                from PySide6.QtCore import QTimer
                for delay in (100, 500, 1500, 3000):
                    QTimer.singleShot(delay, lambda: _aggressive_scan_training_controls_v5(verbose=True))
            except Exception:
                pass
        cls.showEvent = patched_showEvent
        cls._mustatil_multiband_showevent_patch_v5 = True
        return True
    except Exception:
        return False



# ---------------------------------------------------------------------------
# v6 guaranteed visible controls: insert-or-wrap trainer / AI Pipeline pages
# ---------------------------------------------------------------------------
# Why: some plugin pages (R-CNN, U-Net, LAE-DINO, AI Pipeline) are not simple
# layouts. v6 can wrap the existing tab page in a QWidget and put the compact
# Multiband bar above it without creating a new tab.

_V6_TAB_PATCHED = False
_V6_SCAN_STARTED = False
_V6_REENTRANT = False
_V6_OBJECT_NAME = "mustatil_multiband_controls_v6"


def _v6_title_or_blob_is_target(title: Any = "", page: Any = None) -> bool:
    t = str(title or "").lower()
    blob = t

    # Add a small text scan so custom pages are detected even with generic tab names.
    if page is not None:
        try:
            bits = [str(getattr(page, "objectName", lambda: "")() or ""), str(getattr(page, "windowTitle", lambda: "")() or "")]
            # Limit to avoid slow full-tree scans.
            for ch in page.findChildren(object):
                try:
                    cls = ch.__class__.__name__.lower()
                    if any(k in cls for k in ["label", "button", "groupbox", "combobox", "lineedit", "checkbox"]):
                        if hasattr(ch, "text"):
                            bits.append(str(ch.text() or ""))
                        if hasattr(ch, "title"):
                            bits.append(str(ch.title() or ""))
                        if hasattr(ch, "objectName"):
                            bits.append(str(ch.objectName() or ""))
                except Exception:
                    pass
                if len(bits) > 220:
                    break
            blob += " " + " ".join(bits).lower()
        except Exception:
            pass

    # Do include AI Pipeline because the user explicitly wants it.
    positive = [
        "trainer", "training", "train ",
        "yolo trainer", "r-cnn", "rcnn", "faster r-cnn", "mask r-cnn",
        "u-net", "unet", "lae-dino", "sam2",
        "create dataset", "start training", "checkpoint", "backbone",
        "ai pipeline", "pipeline",
    ]
    negative_exact = {
        "detection", "satellite detection", "annotator", "auto updater",
        "console", "map", "webmap", "geospatial operations"
    }
    if t.strip() in negative_exact:
        return False
    return any(p in blob for p in positive)


def _v6_has_controls(widget: Any) -> bool:
    if widget is None:
        return False
    try:
        if getattr(widget, "_mustatil_multiband_controls_v6", False):
            return True
    except Exception:
        pass
    try:
        from PySide6.QtWidgets import QWidget
        for w in widget.findChildren(QWidget):
            try:
                if w.objectName() == _V6_OBJECT_NAME:
                    return True
                if getattr(w, "_mustatil_multiband_controls_v6", False):
                    return True
            except Exception:
                pass
    except Exception:
        pass
    return False


def _v6_find_workspace(widget: Any = None) -> Any:
    global _LAST_WORKSPACE
    try:
        if _LAST_WORKSPACE is not None and hasattr(_LAST_WORKSPACE, "log"):
            return _LAST_WORKSPACE
    except Exception:
        pass

    try:
        w = widget
        for _ in range(30):
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


def _v6_widget_layout_target(page: Any) -> Any:
    """Return the best inner widget that has a layout, if possible."""
    if page is None:
        return None
    try:
        # QScrollArea tab page -> contained widget is usually where controls belong.
        if page.__class__.__name__.lower().endswith("scrollarea") and hasattr(page, "widget"):
            inner = page.widget()
            if inner is not None and inner.layout() is not None:
                return inner
    except Exception:
        pass
    try:
        if page.layout() is not None:
            return page
    except Exception:
        pass
    try:
        # Prefer large visible child widgets with layouts.
        candidates = []
        from PySide6.QtWidgets import QWidget
        for ch in page.findChildren(QWidget):
            try:
                if ch.layout() is not None:
                    area = max(1, ch.width()) * max(1, ch.height())
                    candidates.append((area, ch))
            except Exception:
                pass
        if candidates:
            candidates.sort(reverse=True, key=lambda x: x[0])
            return candidates[0][1]
    except Exception:
        pass
    return None


def _v6_make_controls(page: Any = None, title: str = "") -> Any:
    from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QLineEdit, QCheckBox, QComboBox, QPushButton

    ws = _v6_find_workspace(page)

    # Current defaults from workspace/env/profile.
    try:
        enabled_default = bool(_workspace_setting(ws, "mustatil_mb_enabled", True))
    except Exception:
        enabled_default = True
    try:
        bands_default = str(_workspace_setting(ws, "mustatil_mb_bands", os.environ.get("MUSTATIL_MULTIBAND_BANDS", "1,2,4")) or "1,2,4")
    except Exception:
        bands_default = os.environ.get("MUSTATIL_MULTIBAND_BANDS", "1,2,4")
    try:
        norm_default = str(_workspace_setting(ws, "mustatil_mb_normalization", os.environ.get("MUSTATIL_MULTIBAND_NORMALIZATION", "percentile_2_98")) or "percentile_2_98")
    except Exception:
        norm_default = os.environ.get("MUSTATIL_MULTIBAND_NORMALIZATION", "percentile_2_98")

    bar = QWidget()
    bar.setObjectName(_V6_OBJECT_NAME)
    bar.setProperty("mustatil_multiband_bar", True)
    bar._mustatil_multiband_controls_v6 = True
    bar.setMaximumHeight(42)
    bar.setMinimumHeight(32)
    bar.setToolTip("Mustatil Multiband profile. Applies to training, Detection, R-CNN rasterio reads and AI Pipeline image inputs.")

    row = QHBoxLayout(bar)
    row.setContentsMargins(6, 2, 6, 2)

    cb = QCheckBox("Multiband")
    cb.setChecked(enabled_default)
    cb.setToolTip("Enable shared multiband adapter for this workflow.")

    bands = QLineEdit(bands_default)
    bands.setMaximumWidth(105)
    bands.setPlaceholderText("1,2,4")
    bands.setToolTip("Band order for the RGB composite. Example: 1,2,4 or 3,4,8")

    norm = QComboBox()
    norm.addItems(["percentile_2_98", "percentile_1_99", "minmax", "none"])
    ix = norm.findText(norm_default)
    if ix >= 0:
        norm.setCurrentIndex(ix)
    norm.setMaximumWidth(135)

    apply_btn = QPushButton("Apply")
    apply_btn.setMaximumWidth(62)

    row.addWidget(cb)
    row.addWidget(QLabel("Bands:"))
    row.addWidget(bands)
    row.addWidget(QLabel("Norm:"))
    row.addWidget(norm)
    row.addWidget(apply_btn)
    row.addStretch(1)

    def apply_profile(verbose=False):
        ws2 = _v6_find_workspace(page or bar)
        btxt = bands.text().strip() or "1,2,4"
        ntxt = norm.currentText().strip() or "percentile_2_98"
        os.environ["MUSTATIL_MULTIBAND_ENABLED"] = "1" if cb.isChecked() else "0"
        os.environ["MUSTATIL_MULTIBAND_BANDS"] = btxt
        os.environ["MUSTATIL_MULTIBAND_NORMALIZATION"] = ntxt

        prof = _default_profile(None)
        prof["enabled"] = bool(cb.isChecked())
        prof["selected_bands"] = _parse_bands(btxt)
        prof["normalization"] = ntxt
        prof["adapter_mode"] = "universal_rgb_composite"
        prof["model_input_channels"] = 3

        saved = None
        try:
            if ws2 is not None:
                # Create simple Var-compatible attrs if missing.
                class _V:
                    def __init__(self, value=None): self._value = value
                    def get(self): return self._value
                    def set(self, value): self._value = value
                if not hasattr(ws2, "mustatil_mb_enabled"):
                    ws2.mustatil_mb_enabled = _V(cb.isChecked())
                if not hasattr(ws2, "mustatil_mb_bands"):
                    ws2.mustatil_mb_bands = _V(btxt)
                if not hasattr(ws2, "mustatil_mb_normalization"):
                    ws2.mustatil_mb_normalization = _V(ntxt)
                if hasattr(ws2.mustatil_mb_enabled, "set"): ws2.mustatil_mb_enabled.set(bool(cb.isChecked()))
                if hasattr(ws2.mustatil_mb_bands, "set"): ws2.mustatil_mb_bands.set(btxt)
                if hasattr(ws2.mustatil_mb_normalization, "set"): ws2.mustatil_mb_normalization.set(ntxt)
                saved = _save_profile_near_project(ws2, prof)
        except Exception:
            pass

        if verbose:
            msg = f"Multiband applied: enabled={cb.isChecked()}, bands={prof['selected_bands']}, normalization={ntxt}"
            if title:
                msg += f", page={title}"
            if saved:
                msg += f", profile={saved}"
            try:
                if ws2 is not None and hasattr(ws2, "log"):
                    ws2.log(msg)
                else:
                    print(f"[{PLUGIN_NAME}] {msg}", flush=True)
            except Exception:
                pass

    cb.toggled.connect(lambda _=False: apply_profile(False))
    bands.editingFinished.connect(lambda: apply_profile(False))
    norm.currentTextChanged.connect(lambda _=None: apply_profile(False))
    apply_btn.clicked.connect(lambda _=False: apply_profile(True))

    try:
        apply_profile(False)
    except Exception:
        pass

    return bar


def _v6_insert_into_existing_layout(page: Any, title: str = "") -> bool:
    if page is None or _v6_has_controls(page):
        return False
    target = _v6_widget_layout_target(page)
    if target is None:
        return False
    try:
        layout = target.layout()
        if layout is None:
            return False
        bar = _v6_make_controls(target, title)
        inserted = False
        if hasattr(layout, "insertWidget"):
            layout.insertWidget(0, bar)
            inserted = True
        elif hasattr(layout, "insertRow"):
            layout.insertRow(0, bar)
            inserted = True
        elif hasattr(layout, "addWidget"):
            layout.addWidget(bar)
            inserted = True
        if inserted:
            setattr(page, "_mustatil_multiband_controls_v6", True)
            setattr(target, "_mustatil_multiband_controls_v6", True)
            ws = _v6_find_workspace(page)
            msg = f"Multiband v6 controls inserted into existing layout: {title or target.__class__.__name__}"
            if ws is not None and hasattr(ws, "log"):
                ws.log(msg)
            else:
                print(f"[{PLUGIN_NAME}] {msg}", flush=True)
            return True
    except Exception as exc:
        _warn_once(f"v6_layout_insert_{title}", f"v6 layout insert failed for {title}: {exc}")
    return False


def _v6_wrap_tab_page(tabwidget: Any, index: int, reason: str = "") -> bool:
    """Guaranteed visible method: wrap the existing tab page and add bar above it."""
    global _V6_REENTRANT
    if _V6_REENTRANT:
        return False
    try:
        if tabwidget is None or index < 0 or index >= tabwidget.count():
            return False
        page = tabwidget.widget(index)
        title = str(tabwidget.tabText(index) or "")
        if page is None or _v6_has_controls(page):
            return False
        if not _v6_title_or_blob_is_target(title, page):
            return False

        # First try normal insert; safer for pages that have layouts.
        if _v6_insert_into_existing_layout(page, title):
            return True

        from PySide6.QtWidgets import QWidget, QVBoxLayout
        wrapper = QWidget()
        wrapper.setObjectName("mustatil_multiband_wrapper_v6")
        wrapper._mustatil_multiband_controls_v6 = True

        lay = QVBoxLayout(wrapper)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        bar = _v6_make_controls(page, title)
        lay.addWidget(bar)

        icon = tabwidget.tabIcon(index)
        tip = tabwidget.tabToolTip(index)
        enabled = tabwidget.isTabEnabled(index)
        current = tabwidget.currentIndex()

        _V6_REENTRANT = True
        try:
            tabwidget.removeTab(index)
            page.setParent(wrapper)
            lay.addWidget(page, 1)
            tabwidget.insertTab(index, wrapper, icon, title)
            tabwidget.setTabToolTip(index, tip)
            tabwidget.setTabEnabled(index, enabled)
            if current == index:
                tabwidget.setCurrentIndex(index)
        finally:
            _V6_REENTRANT = False

        setattr(page, "_mustatil_multiband_controls_v6", True)
        ws = _v6_find_workspace(wrapper)
        msg = f"Multiband v6 controls WRAPPED into tab: {title} ({reason or 'scan'})"
        if ws is not None and hasattr(ws, "log"):
            ws.log(msg)
        else:
            print(f"[{PLUGIN_NAME}] {msg}", flush=True)
        return True

    except Exception as exc:
        _V6_REENTRANT = False
        _warn_once(f"v6_wrap_{index}", f"v6 tab wrap failed at index {index}: {exc}")
        return False


def _v6_scan_all_tabs(verbose: bool = False) -> int:
    inserted = 0
    try:
        from PySide6.QtWidgets import QApplication, QTabWidget
        app = QApplication.instance()
        if app is None:
            return 0
        for top in app.topLevelWidgets():
            try:
                for tw in top.findChildren(QTabWidget):
                    for i in range(tw.count()):
                        try:
                            title = str(tw.tabText(i) or "")
                            page = tw.widget(i)
                            if _v6_title_or_blob_is_target(title, page):
                                if _v6_wrap_tab_page(tw, i, reason="v6 scan"):
                                    inserted += 1
                        except Exception:
                            pass
            except Exception:
                pass
        if verbose:
            _log(f"v6 scan completed; controls inserted/wrapped={inserted}")
    except Exception as exc:
        _warn_once("v6_scan", f"v6 scan warning: {exc}")
    return inserted


def _v6_patch_qtabwidget() -> bool:
    global _V6_TAB_PATCHED
    if _V6_TAB_PATCHED:
        return True
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QTabWidget
    except Exception:
        return False

    if getattr(QTabWidget, "_mustatil_multiband_v6_tab_patch", False):
        _V6_TAB_PATCHED = True
        return True

    orig_add = QTabWidget.addTab
    orig_insert = QTabWidget.insertTab

    def after(tabw, idx, title, page, reason):
        try:
            if idx is None or idx < 0:
                return
            if _v6_title_or_blob_is_target(title, page):
                _v6_wrap_tab_page(tabw, idx, reason=reason)
                for delay in (100, 500, 1500, 3000, 6000):
                    QTimer.singleShot(delay, lambda tw=tabw, ix=idx: _v6_wrap_tab_page(tw, ix, reason=f"delayed {delay}ms"))
        except Exception as exc:
            _warn_once("v6_after_tab", f"v6 after-tab warning: {exc}")

    def patched_addTab(self, *args, **kwargs):
        global _V6_REENTRANT
        page = args[0] if args else None
        title = ""
        try:
            if len(args) >= 2 and isinstance(args[1], str):
                title = args[1]
            elif len(args) >= 3 and isinstance(args[2], str):
                title = args[2]
        except Exception:
            pass
        idx = orig_add(self, *args, **kwargs)
        if not _V6_REENTRANT:
            after(self, idx, title, page, "addTab")
        return idx

    def patched_insertTab(self, *args, **kwargs):
        # Signature: insertTab(index, widget, label) or insertTab(index, widget, icon, label)
        idx_arg = args[0] if args else -1
        page = args[1] if len(args) >= 2 else None
        title = ""
        try:
            if len(args) >= 3 and isinstance(args[2], str):
                title = args[2]
            elif len(args) >= 4 and isinstance(args[3], str):
                title = args[3]
        except Exception:
            pass
        idx = orig_insert(self, *args, **kwargs)
        if not _V6_REENTRANT:
            after(self, idx, title, page, "insertTab")
        return idx

    try:
        QTabWidget.addTab = patched_addTab
        QTabWidget.insertTab = patched_insertTab
        QTabWidget._mustatil_multiband_v6_tab_patch = True
        _V6_TAB_PATCHED = True
        _log("v6 guaranteed tab UI hook installed: trainer / AI Pipeline pages are inserted or wrapped with Multiband controls.")
        return True
    except Exception as exc:
        _warn_once("v6_patch_qtabwidget", f"Could not install v6 QTabWidget patch: {exc}")
        return False


def _v6_install_event_scanner() -> bool:
    global _V6_SCAN_STARTED
    if _V6_SCAN_STARTED:
        return True
    try:
        from PySide6.QtCore import QObject, QEvent, QTimer
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            return False

        class _V6EventFilter(QObject):
            def eventFilter(self, obj, event):
                try:
                    et = event.type()
                    if et in (QEvent.Show, QEvent.ChildAdded, QEvent.LayoutRequest):
                        QTimer.singleShot(250, lambda: _v6_scan_all_tabs(False))
                except Exception:
                    pass
                return False

        filt = _V6EventFilter(app)
        app.installEventFilter(filt)
        app._mustatil_multiband_v6_event_filter = filt
        _V6_SCAN_STARTED = True

        for delay in (100, 500, 1500, 3000, 6000, 10000, 15000):
            QTimer.singleShot(delay, lambda d=delay: _v6_scan_all_tabs(verbose=(d in (1500, 6000, 15000))))

        _log("v6 event scanner installed: delayed scans will wrap trainer, R-CNN, U-Net, LAE-DINO and AI Pipeline pages.")
        return True
    except Exception as exc:
        _warn_once("v6_event_scanner", f"Could not install v6 event scanner: {exc}")
        return False


def _v6_install_guaranteed_ui() -> None:
    _v6_patch_qtabwidget()
    _v6_install_event_scanner()
    _v6_scan_all_tabs(verbose=True)



# ---------------------------------------------------------------------------
# v8 single-control UI: no duplicate YOLO, targeted LAE-DINO panel support
# ---------------------------------------------------------------------------
# v6 proved wrapping works for R-CNN/U-Net, but older embedded v4/v5 scanners
# could still insert a second row on YOLO, while LAE-DINO can live inside a
# special nested panel that is not reachable by normal tab title logic.
# v7 disables legacy UI scanners and uses exactly one v7 control bar per target.

_V7_OBJECT_NAME = "mustatil_multiband_controls_v7"
_V7_TAB_PATCHED = False
_V7_EVENT_FILTER = None
_V7_REENTRANT = False
_V7_PATCHED_IDS = set()


def _v7_workspace(widget: Any = None) -> Any:
    global _LAST_WORKSPACE
    try:
        if _LAST_WORKSPACE is not None and hasattr(_LAST_WORKSPACE, "log"):
            return _LAST_WORKSPACE
    except Exception:
        pass
    try:
        w = widget
        for _ in range(40):
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


def _v7_blob(widget: Any, limit: int = 180) -> str:
    if widget is None:
        return ""
    bits = []
    def add(x):
        try:
            s = str(x or "").strip()
            if s:
                bits.append(s)
        except Exception:
            pass
    try:
        add(widget.__class__.__name__)
        if hasattr(widget, "objectName"):
            add(widget.objectName())
        if hasattr(widget, "windowTitle"):
            add(widget.windowTitle())
        if hasattr(widget, "text"):
            add(widget.text())
        if hasattr(widget, "title"):
            add(widget.title())
    except Exception:
        pass
    try:
        from PySide6.QtWidgets import QWidget
        for ch in widget.findChildren(QWidget):
            try:
                add(ch.__class__.__name__)
                if hasattr(ch, "objectName"):
                    add(ch.objectName())
                if hasattr(ch, "text"):
                    add(ch.text())
                if hasattr(ch, "title"):
                    add(ch.title())
            except Exception:
                pass
            if len(bits) >= limit:
                break
    except Exception:
        pass
    return " ".join(bits).lower()


def _v7_tab_title_for(widget: Any) -> str:
    try:
        from PySide6.QtWidgets import QTabWidget
        w = widget
        for _ in range(40):
            if w is None:
                return ""
            p = w.parent()
            if isinstance(p, QTabWidget):
                idx = p.indexOf(w)
                if idx >= 0:
                    return str(p.tabText(idx) or "")
            w = p
    except Exception:
        pass
    return ""


def _v7_is_target(title: Any = "", page: Any = None, mode: str = "tab") -> bool:
    title_l = str(title or "").lower().strip()
    blob = (title_l + " " + _v7_blob(page)).lower()

    # Never place bars on pure detection/satellite/map/console pages.
    if title_l in {"detection", "satellite detection", "annotator", "console", "map", "webmap", "geospatial operations"}:
        return False

    # Explicit user-requested target.
    if "ai pipeline" in blob:
        return True

    # Normal trainers.
    if any(x in blob for x in [
        "yolo trainer", "trainer", "training", "start training", "create dataset",
        "prepare dataset", "epochs", "batch", "checkpoint", "backbone",
        "r-cnn", "rcnn", "faster r-cnn", "mask r-cnn", "u-net", "unet", "sam2 trainer"
    ]):
        # Avoid false positives in detection tabs with model names unless training markers exist.
        if "detection" in title_l and not any(x in blob for x in ["trainer", "training", "start training", "create dataset", "epochs", "batch", "checkpoint", "backbone"]):
            return False
        return True

    # LAE-DINO special panels may not expose a normal trainer tab title.
    if "lae-dino" in blob or "lae dino" in blob:
        if any(x in blob for x in ["trainer", "training", "anti-freeze", "create config", "create dataset", "start training", "checkpoint", "dataset"]):
            return True

    return False


def _v7_is_lae_panel(widget: Any) -> bool:
    blob = _v7_blob(widget, limit=240)
    return (
        ("lae-dino" in blob or "lae dino" in blob)
        and any(x in blob for x in ["trainer", "training", "anti-freeze", "create config", "create dataset", "start training", "checkpoint", "dataset"])
    )


def _v7_has_any_mb_control(widget: Any) -> bool:
    if widget is None:
        return False
    try:
        if getattr(widget, "_mustatil_multiband_controls_v7", False):
            return True
    except Exception:
        pass
    try:
        from PySide6.QtWidgets import QWidget, QGroupBox
        for w in widget.findChildren(QWidget):
            try:
                if str(w.objectName() or "") == _V7_OBJECT_NAME:
                    return True
                if getattr(w, "_mustatil_multiband_controls_v7", False):
                    return True
            except Exception:
                pass
        # Treat any old visible Multiband box as existing control to avoid duplicates.
        for gb in widget.findChildren(QGroupBox):
            try:
                if str(gb.title()).strip().lower() == "multiband":
                    return True
                if "mustatil_multiband" in str(gb.objectName() or "").lower():
                    return True
            except Exception:
                pass
    except Exception:
        pass
    return False


def _v7_remove_duplicate_controls(widget: Any) -> int:
    """Remove old v3-v6 duplicate Multiband controls inside one container."""
    if widget is None:
        return 0
    removed = 0
    try:
        from PySide6.QtWidgets import QWidget, QGroupBox
        candidates = []
        for w in widget.findChildren(QWidget):
            try:
                name = str(w.objectName() or "").lower()
                title = ""
                if isinstance(w, QGroupBox):
                    title = str(w.title() or "").strip().lower()
                if "mustatil_multiband" in name or title == "multiband":
                    candidates.append(w)
            except Exception:
                pass
        # Keep first v7 if present, remove everything else. If there is no v7,
        # remove old controls because v7 will insert one after cleanup.
        keep = None
        for w in candidates:
            try:
                if str(w.objectName() or "") == _V7_OBJECT_NAME:
                    keep = w
                    break
            except Exception:
                pass
        for w in candidates:
            if keep is not None and w is keep:
                continue
            try:
                w.setParent(None)
                w.deleteLater()
                removed += 1
            except Exception:
                pass
    except Exception:
        pass
    return removed


def _v7_make_bar(context: Any = None, title: str = "") -> Any:
    from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QLineEdit, QCheckBox, QComboBox, QPushButton

    ws = _v7_workspace(context)
    try:
        enabled_default = bool(_workspace_setting(ws, "mustatil_mb_enabled", True))
    except Exception:
        enabled_default = True
    try:
        bands_default = str(_workspace_setting(ws, "mustatil_mb_bands", os.environ.get("MUSTATIL_MULTIBAND_BANDS", "1,2,4")) or "1,2,4")
    except Exception:
        bands_default = os.environ.get("MUSTATIL_MULTIBAND_BANDS", "1,2,4")
    try:
        norm_default = str(_workspace_setting(ws, "mustatil_mb_normalization", os.environ.get("MUSTATIL_MULTIBAND_NORMALIZATION", "percentile_2_98")) or "percentile_2_98")
    except Exception:
        norm_default = os.environ.get("MUSTATIL_MULTIBAND_NORMALIZATION", "percentile_2_98")

    bar = QWidget()
    bar.setObjectName(_V7_OBJECT_NAME)
    bar.setMaximumHeight(42)
    bar.setMinimumHeight(30)
    bar.setToolTip("Shared Mustatil Multiband profile for Training, Detection, R-CNN/U-Net rasterio paths and AI Pipeline inputs.")
    bar._mustatil_multiband_controls_v7 = True

    row = QHBoxLayout(bar)
    row.setContentsMargins(6, 2, 6, 2)

    cb = QCheckBox("Multiband")
    cb.setChecked(enabled_default)
    bands = QLineEdit(bands_default)
    bands.setMaximumWidth(105)
    bands.setPlaceholderText("1,2,4")
    norm = QComboBox()
    norm.addItems(["percentile_2_98", "percentile_1_99", "minmax", "none"])
    ix = norm.findText(norm_default)
    if ix >= 0:
        norm.setCurrentIndex(ix)
    norm.setMaximumWidth(135)
    apply_btn = QPushButton("Apply")
    apply_btn.setMaximumWidth(62)

    row.addWidget(cb)
    row.addWidget(QLabel("Bands:"))
    row.addWidget(bands)
    row.addWidget(QLabel("Norm:"))
    row.addWidget(norm)
    row.addWidget(apply_btn)
    row.addStretch(1)

    def apply_profile(verbose=False):
        ws2 = _v7_workspace(context or bar)
        btxt = bands.text().strip() or "1,2,4"
        ntxt = norm.currentText().strip() or "percentile_2_98"
        os.environ["MUSTATIL_MULTIBAND_ENABLED"] = "1" if cb.isChecked() else "0"
        os.environ["MUSTATIL_MULTIBAND_BANDS"] = btxt
        os.environ["MUSTATIL_MULTIBAND_NORMALIZATION"] = ntxt

        prof = _default_profile(None)
        prof["enabled"] = bool(cb.isChecked())
        prof["selected_bands"] = _parse_bands(btxt)
        prof["normalization"] = ntxt
        prof["adapter_mode"] = "universal_rgb_composite"
        prof["model_input_channels"] = 3

        saved = None
        try:
            if ws2 is not None:
                class _V:
                    def __init__(self, value=None): self._value = value
                    def get(self): return self._value
                    def set(self, value): self._value = value
                if not hasattr(ws2, "mustatil_mb_enabled"):
                    ws2.mustatil_mb_enabled = _V(cb.isChecked())
                if not hasattr(ws2, "mustatil_mb_bands"):
                    ws2.mustatil_mb_bands = _V(btxt)
                if not hasattr(ws2, "mustatil_mb_normalization"):
                    ws2.mustatil_mb_normalization = _V(ntxt)
                if hasattr(ws2.mustatil_mb_enabled, "set"): ws2.mustatil_mb_enabled.set(bool(cb.isChecked()))
                if hasattr(ws2.mustatil_mb_bands, "set"): ws2.mustatil_mb_bands.set(btxt)
                if hasattr(ws2.mustatil_mb_normalization, "set"): ws2.mustatil_mb_normalization.set(ntxt)
                saved = _save_profile_near_project(ws2, prof)
        except Exception:
            pass

        if verbose:
            msg = f"Multiband v8 applied: enabled={cb.isChecked()}, bands={prof['selected_bands']}, normalization={ntxt}"
            if title:
                msg += f", page={title}"
            if saved:
                msg += f", profile={saved}"
            try:
                if ws2 is not None and hasattr(ws2, "log"):
                    ws2.log(msg)
                else:
                    print(f"[{PLUGIN_NAME}] {msg}", flush=True)
            except Exception:
                pass

    cb.toggled.connect(lambda _=False: apply_profile(False))
    bands.editingFinished.connect(lambda: apply_profile(False))
    norm.currentTextChanged.connect(lambda _=None: apply_profile(False))
    apply_btn.clicked.connect(lambda _=False: apply_profile(True))

    try:
        apply_profile(False)
    except Exception:
        pass

    return bar


def _v7_best_layout_widget(page: Any) -> Any:
    if page is None:
        return None
    try:
        if page.__class__.__name__.lower().endswith("scrollarea") and hasattr(page, "widget"):
            inner = page.widget()
            if inner is not None and inner.layout() is not None:
                return inner
    except Exception:
        pass
    try:
        if page.layout() is not None:
            return page
    except Exception:
        pass
    try:
        from PySide6.QtWidgets import QWidget
        candidates = []
        for ch in page.findChildren(QWidget):
            try:
                if ch.layout() is not None and ch.isVisible():
                    area = max(1, ch.width()) * max(1, ch.height())
                    candidates.append((area, ch))
            except Exception:
                pass
        if candidates:
            candidates.sort(reverse=True, key=lambda x: x[0])
            return candidates[0][1]
    except Exception:
        pass
    return None


def _v7_insert_into_layout(page: Any, title: str = "") -> bool:
    if page is None:
        return False
    if _v7_has_any_mb_control(page):
        return False
    target = _v7_best_layout_widget(page)
    if target is None:
        return False
    try:
        layout = target.layout()
        if layout is None:
            return False
        bar = _v7_make_bar(target, title)
        if hasattr(layout, "insertWidget"):
            layout.insertWidget(0, bar)
        elif hasattr(layout, "insertRow"):
            layout.insertRow(0, bar)
        elif hasattr(layout, "addWidget"):
            layout.addWidget(bar)
        else:
            return False
        setattr(page, "_mustatil_multiband_controls_v7", True)
        setattr(target, "_mustatil_multiband_controls_v7", True)
        ws = _v7_workspace(page)
        msg = f"Multiband v8 controls inserted: {title or target.__class__.__name__}"
        if ws is not None and hasattr(ws, "log"):
            ws.log(msg)
        else:
            print(f"[{PLUGIN_NAME}] {msg}", flush=True)
        return True
    except Exception as exc:
        _warn_once(f"v7_insert_{title}", f"v7 insert failed for {title}: {exc}")
    return False


def _v7_wrap_tab(tabwidget: Any, index: int, reason: str = "") -> bool:
    global _V7_REENTRANT
    if _V7_REENTRANT:
        return False
    try:
        if tabwidget is None or index < 0 or index >= tabwidget.count():
            return False
        page = tabwidget.widget(index)
        title = str(tabwidget.tabText(index) or "")
        if page is None:
            return False
        if not _v7_is_target(title, page):
            return False

        # Remove stale old controls inside this page so YOLO is not double.
        _v7_remove_duplicate_controls(page)
        if _v7_insert_into_layout(page, title):
            return True
        if _v7_has_any_mb_control(page):
            return False

        from PySide6.QtWidgets import QWidget, QVBoxLayout
        wrapper = QWidget()
        wrapper.setObjectName("mustatil_multiband_wrapper_v7")
        wrapper._mustatil_multiband_controls_v7 = True
        lay = QVBoxLayout(wrapper)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(_v7_make_bar(page, title))
        icon = tabwidget.tabIcon(index)
        tip = tabwidget.tabToolTip(index)
        enabled = tabwidget.isTabEnabled(index)
        current = tabwidget.currentIndex()

        _V7_REENTRANT = True
        try:
            tabwidget.removeTab(index)
            page.setParent(wrapper)
            lay.addWidget(page, 1)
            tabwidget.insertTab(index, wrapper, icon, title)
            tabwidget.setTabToolTip(index, tip)
            tabwidget.setTabEnabled(index, enabled)
            if current == index:
                tabwidget.setCurrentIndex(index)
        finally:
            _V7_REENTRANT = False

        setattr(page, "_mustatil_multiband_controls_v7", True)
        ws = _v7_workspace(wrapper)
        msg = f"Multiband v8 controls WRAPPED into tab: {title} ({reason or 'scan'})"
        if ws is not None and hasattr(ws, "log"):
            ws.log(msg)
        else:
            print(f"[{PLUGIN_NAME}] {msg}", flush=True)
        return True
    except Exception as exc:
        _V7_REENTRANT = False
        _warn_once(f"v7_wrap_{index}", f"v7 wrap failed at index {index}: {exc}")
        return False


def _v7_scan_lae_panels(top: Any) -> int:
    inserted = 0
    try:
        from PySide6.QtWidgets import QWidget, QGroupBox
        for w in top.findChildren(QWidget):
            try:
                if _v7_has_any_mb_control(w):
                    continue
                if not _v7_is_lae_panel(w):
                    continue
                # Prefer the actual LAE-DINO groupbox/panel, not tiny buttons.
                if isinstance(w, QGroupBox) or w.layout() is not None:
                    _v7_remove_duplicate_controls(w)
                    title = "LAE-DINO Trainer panel"
                    if hasattr(w, "title"):
                        try:
                            title = str(w.title() or title)
                        except Exception:
                            pass
                    if _v7_insert_into_layout(w, title):
                        inserted += 1
                        break
            except Exception:
                pass
    except Exception:
        pass
    return inserted


def _v7_scan(verbose: bool = False) -> int:
    inserted = 0
    try:
        from PySide6.QtWidgets import QApplication, QTabWidget
        app = QApplication.instance()
        if app is None:
            return 0
        for top in app.topLevelWidgets():
            try:
                for tw in top.findChildren(QTabWidget):
                    for i in range(tw.count()):
                        page = tw.widget(i)
                        title = str(tw.tabText(i) or "")
                        if _v7_is_target(title, page):
                            if _v7_wrap_tab(tw, i, reason="v8 scan"):
                                inserted += 1
                inserted += _v7_scan_lae_panels(top)
            except Exception:
                pass
        if verbose:
            _log(f"v8 scan complete: inserted/wrapped={inserted}")
    except Exception as exc:
        _warn_once("v7_scan", f"v8 scan warning: {exc}")
    return inserted


def _v7_patch_qtabwidget() -> bool:
    global _V7_TAB_PATCHED
    if _V7_TAB_PATCHED:
        return True
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QTabWidget
    except Exception:
        return False
    if getattr(QTabWidget, "_mustatil_multiband_v7_tab_patch", False):
        _V7_TAB_PATCHED = True
        return True

    orig_add = QTabWidget.addTab
    orig_insert = QTabWidget.insertTab

    def after(tabw, idx, title, page, reason):
        try:
            if idx is None or idx < 0:
                return
            if _v7_is_target(title, page):
                _v7_wrap_tab(tabw, idx, reason=reason)
                for delay in (150, 600, 1500, 3000, 6000):
                    QTimer.singleShot(delay, lambda tw=tabw, ix=idx: _v7_wrap_tab(tw, ix, reason=f"delayed {delay}ms"))
        except Exception as exc:
            _warn_once("v7_after_tab", f"v7 after-tab warning: {exc}")

    def patched_addTab(self, *args, **kwargs):
        page = args[0] if args else None
        title = ""
        try:
            if len(args) >= 2 and isinstance(args[1], str):
                title = args[1]
            elif len(args) >= 3 and isinstance(args[2], str):
                title = args[2]
        except Exception:
            pass
        idx = orig_add(self, *args, **kwargs)
        if not _V7_REENTRANT:
            after(self, idx, title, page, "addTab")
        return idx

    def patched_insertTab(self, *args, **kwargs):
        page = args[1] if len(args) >= 2 else None
        title = ""
        try:
            if len(args) >= 3 and isinstance(args[2], str):
                title = args[2]
            elif len(args) >= 4 and isinstance(args[3], str):
                title = args[3]
        except Exception:
            pass
        idx = orig_insert(self, *args, **kwargs)
        if not _V7_REENTRANT:
            after(self, idx, title, page, "insertTab")
        return idx

    try:
        QTabWidget.addTab = patched_addTab
        QTabWidget.insertTab = patched_insertTab
        QTabWidget._mustatil_multiband_v7_tab_patch = True
        _V7_TAB_PATCHED = True
        _log("v8 single-control tab hook installed: avoids duplicate YOLO controls and targets LAE-DINO / AI Pipeline.")
        return True
    except Exception as exc:
        _warn_once("v7_patch_tab", f"v7 QTabWidget patch failed: {exc}")
        return False


def _v7_event_scanner() -> bool:
    global _V7_EVENT_FILTER
    if _V7_EVENT_FILTER is not None:
        return True
    try:
        from PySide6.QtCore import QObject, QEvent, QTimer
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            return False

        class _V7Filter(QObject):
            def eventFilter(self, obj, event):
                try:
                    if event.type() in (QEvent.Show, QEvent.ChildAdded, QEvent.LayoutRequest):
                        QTimer.singleShot(350, lambda: _v7_scan(False))
                except Exception:
                    pass
                return False

        filt = _V7Filter(app)
        app.installEventFilter(filt)
        app._mustatil_multiband_v7_filter = filt
        _V7_EVENT_FILTER = filt
        for delay in (200, 800, 1600, 3000, 6000, 10000, 15000, 25000):
            QTimer.singleShot(delay, lambda d=delay: _v7_scan(verbose=(d in (1600, 6000, 15000, 25000))))
        _log("v8 event scanner installed: YOLO/R-CNN/U-Net/LAE-DINO/AI Pipeline pages will be scanned without duplicates.")
        return True
    except Exception as exc:
        _warn_once("v7_event", f"v8 event scanner failed: {exc}")
    return False


def _v7_install_ui() -> None:
    _v7_patch_qtabwidget()
    _v7_event_scanner()
    _v7_scan(verbose=True)



# ---------------------------------------------------------------------------
# v8 targeted LAE-DINO button-panel injector
# ---------------------------------------------------------------------------
# The LAE-DINO Trainer add-on can inject its training controls as a nested
# "anti-freeze" panel with the actual buttons:
#   Create Dataset, Create Config, Start Training
# This may not have a normal tab title or a layout blob containing "LAE-DINO".
# v8 therefore finds that exact button cluster and inserts the Multiband bar
# into the common parent/panel.

_V8_LAE_OBJECT_NAME = "mustatil_multiband_lae_dino_controls_v8"
_V8_LAE_PATCHED_IDS = set()


def _v8_widget_text(widget: Any) -> str:
    bits = []
    def add(v):
        try:
            s = str(v or "").strip()
            if s:
                bits.append(s)
        except Exception:
            pass
    try:
        add(widget.__class__.__name__)
        if hasattr(widget, "objectName"):
            add(widget.objectName())
        if hasattr(widget, "windowTitle"):
            add(widget.windowTitle())
        if hasattr(widget, "text"):
            add(widget.text())
        if hasattr(widget, "title"):
            add(widget.title())
    except Exception:
        pass
    return " ".join(bits).lower()


def _v8_subtree_button_texts(widget: Any) -> set:
    out = set()
    try:
        from PySide6.QtWidgets import QPushButton, QToolButton
        for b in widget.findChildren(QPushButton):
            try:
                out.add(str(b.text() or "").strip().lower())
            except Exception:
                pass
        for b in widget.findChildren(QToolButton):
            try:
                out.add(str(b.text() or "").strip().lower())
            except Exception:
                pass
    except Exception:
        pass
    return out


def _v8_is_lae_button_cluster(widget: Any) -> bool:
    texts = _v8_subtree_button_texts(widget)
    if not texts:
        return False
    has_dataset = any("create dataset" in t for t in texts)
    has_config = any("create config" in t for t in texts)
    has_start = any("start training" in t or "start train" in t for t in texts)
    has_lae_hint = "lae" in _v8_widget_text(widget) or "dino" in _v8_widget_text(widget)
    # The three-button cluster is distinctive enough; LAE hint makes it safer
    # when present, but we still patch without it because some panels have no title.
    return (has_dataset and has_config and has_start) or (has_lae_hint and has_start and (has_dataset or has_config))


def _v8_ancestor_chain(widget: Any, limit: int = 14) -> list:
    chain = []
    try:
        w = widget
        for _ in range(limit):
            if w is None:
                break
            chain.append(w)
            w = w.parent()
    except Exception:
        pass
    return chain


def _v8_find_best_lae_container_from_button(button: Any) -> Any:
    """Find the smallest parent with layout containing the LAE-DINO button cluster."""
    best = None
    for w in _v8_ancestor_chain(button, limit=18):
        try:
            if w is None or w.layout() is None:
                continue
            if _v8_is_lae_button_cluster(w):
                best = w
                # Prefer a groupbox/frame/panel if found, otherwise keep climbing
                # one or two levels to include the whole row area.
                cls = w.__class__.__name__.lower()
                if any(k in cls for k in ["groupbox", "frame", "widget", "scrollarea"]):
                    return w
        except Exception:
            pass
    return best


def _v8_has_lae_control(container: Any) -> bool:
    if container is None:
        return False
    try:
        from PySide6.QtWidgets import QWidget
        if getattr(container, "_mustatil_multiband_lae_dino_controls_v8", False):
            return True
        for w in container.findChildren(QWidget):
            try:
                if str(w.objectName() or "") == _V8_LAE_OBJECT_NAME:
                    return True
                if str(w.objectName() or "") == _V7_OBJECT_NAME:
                    return True
            except Exception:
                pass
    except Exception:
        pass
    return False


def _v8_insert_lae_bar(container: Any, reason: str = "") -> bool:
    if container is None:
        return False
    try:
        if id(container) in _V8_LAE_PATCHED_IDS or _v8_has_lae_control(container):
            return False
        layout = container.layout()
        if layout is None:
            return False

        # Clean old bars in this panel only. This will not affect YOLO/R-CNN pages.
        try:
            _v7_remove_duplicate_controls(container)
        except Exception:
            pass

        bar = _v7_make_bar(container, "LAE-DINO Trainer")
        bar.setObjectName(_V8_LAE_OBJECT_NAME)
        bar.setToolTip("LAE-DINO Multiband profile: used for LAE-DINO dataset/training helpers and shared Mustatil multiband adapter.")

        inserted = False
        try:
            if hasattr(layout, "insertWidget"):
                layout.insertWidget(0, bar)
                inserted = True
            elif hasattr(layout, "insertRow"):
                layout.insertRow(0, bar)
                inserted = True
            elif hasattr(layout, "addWidget"):
                layout.addWidget(bar)
                inserted = True
        except Exception:
            inserted = False

        if not inserted:
            return False

        _V8_LAE_PATCHED_IDS.add(id(container))
        setattr(container, "_mustatil_multiband_lae_dino_controls_v8", True)

        ws = _v7_workspace(container)
        msg = f"Multiband v8 controls inserted into LAE-DINO button panel ({reason or 'button cluster'})."
        try:
            if ws is not None and hasattr(ws, "log"):
                ws.log(msg)
            else:
                print(f"[{PLUGIN_NAME}] {msg}", flush=True)
        except Exception:
            pass
        return True
    except Exception as exc:
        _warn_once("v8_lae_insert", f"v8 LAE-DINO insert failed: {exc}")
        return False


def _v8_scan_lae_buttons(verbose: bool = False) -> int:
    inserted = 0
    try:
        from PySide6.QtWidgets import QApplication, QPushButton, QToolButton
        app = QApplication.instance()
        if app is None:
            return 0
        button_classes = (QPushButton, QToolButton)
        for top in app.topLevelWidgets():
            try:
                buttons = []
                for cls in button_classes:
                    buttons.extend(top.findChildren(cls))
                for b in buttons:
                    try:
                        txt = str(b.text() or "").strip().lower()
                        if not any(k in txt for k in ["create dataset", "create config", "start training", "start train"]):
                            continue
                        cont = _v8_find_best_lae_container_from_button(b)
                        if cont is not None and _v8_insert_lae_bar(cont, reason=f"button '{txt}'"):
                            inserted += 1
                            # Usually one LAE panel is enough. Continue in case
                            # satellite/detection/trainer variants exist.
                    except Exception:
                        pass
            except Exception:
                pass
        if verbose:
            _log(f"v8 LAE-DINO button scan complete: inserted={inserted}")
    except Exception as exc:
        _warn_once("v8_lae_scan", f"v8 LAE-DINO scan warning: {exc}")
    return inserted


def _v8_patch_button_creation() -> bool:
    """Patch QPushButton/QToolButton.setText so late-created LAE buttons trigger a scan."""
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QPushButton, QToolButton
    except Exception:
        return False

    ok = False
    for cls in (QPushButton, QToolButton):
        try:
            if getattr(cls, "_mustatil_multiband_lae_v8_button_patch", False):
                ok = True
                continue
            original_set_text = cls.setText

            def patched_setText(self, text, _orig=original_set_text):
                res = _orig(self, text)
                try:
                    t = str(text or "").lower()
                    if any(k in t for k in ["create dataset", "create config", "start training", "start train"]):
                        QTimer.singleShot(200, lambda: _v8_scan_lae_buttons(verbose=False))
                        QTimer.singleShot(1200, lambda: _v8_scan_lae_buttons(verbose=False))
                    return res
                except Exception:
                    return res

            cls.setText = patched_setText
            cls._mustatil_multiband_lae_v8_button_patch = True
            ok = True
        except Exception:
            pass

    if ok:
        _log("v8 LAE-DINO button hook installed: Create Dataset/Create Config/Start Training panels will receive Multiband controls.")
    return ok


def _v8_install_lae_targeted_ui() -> None:
    _v8_patch_button_creation()
    try:
        from PySide6.QtCore import QTimer
        # Multiple delayed scans because the LAE-DINO anti-freeze panel is
        # inserted after several other plugins.
        for delay in (250, 800, 1600, 3000, 6000, 10000, 15000, 25000, 40000):
            QTimer.singleShot(delay, lambda d=delay: _v8_scan_lae_buttons(verbose=(d in (1600, 6000, 15000, 40000))))
    except Exception:
        pass
    _v8_scan_lae_buttons(verbose=True)



# ---------------------------------------------------------------------------
# v9 LAE-DINO top-bar repair: avoid CleanLower hiding lower-area controls
# ---------------------------------------------------------------------------
# The log shows:
#   Multiband v8 controls inserted: LAE-DINO
#   [LAE-DINO Trainer CleanLower v1] cleaned ... hidden=7
# Therefore the LAE-DINO multiband bar can be inserted successfully but later
# hidden by the CleanLower lower-area plugin. v9 puts a second LAE-safe top bar
# OUTSIDE the lower panel by wrapping the whole LAE-DINO tab page. It also
# repairs visibility after CleanLower runs.

_V9_LAE_TOP_OBJECT = "mustatil_multiband_lae_dino_topbar_v9"
_V9_LAE_WRAPPER = "mustatil_multiband_lae_dino_topwrap_v9"
_V9_LAE_REENTRANT = False


def _v9_is_lae_title(title: Any) -> bool:
    t = str(title or "").lower()
    return "lae" in t and "dino" in t


def _v9_find_lae_tabwidgets():
    out = []
    try:
        from PySide6.QtWidgets import QApplication, QTabWidget
        app = QApplication.instance()
        if app is None:
            return out
        for top in app.topLevelWidgets():
            for tw in top.findChildren(QTabWidget):
                try:
                    for i in range(tw.count()):
                        title = str(tw.tabText(i) or "")
                        page = tw.widget(i)
                        if _v9_is_lae_title(title):
                            out.append((tw, i, title, page))
                except Exception:
                    pass
    except Exception:
        pass
    return out


def _v9_has_lae_topbar(widget: Any) -> bool:
    if widget is None:
        return False
    try:
        if str(widget.objectName() or "") == _V9_LAE_WRAPPER:
            return True
        from PySide6.QtWidgets import QWidget
        for w in widget.findChildren(QWidget):
            try:
                if str(w.objectName() or "") == _V9_LAE_TOP_OBJECT:
                    return True
            except Exception:
                pass
    except Exception:
        pass
    return False


def _v9_find_or_make_lae_topbar(context: Any, title: str):
    bar = _v7_make_bar(context, "LAE-DINO Trainer")
    bar.setObjectName(_V9_LAE_TOP_OBJECT)
    bar.setMinimumHeight(34)
    bar.setMaximumHeight(42)
    bar.setVisible(True)
    bar.setToolTip("LAE-DINO Multiband top bar. Kept outside the cleaned lower area so CleanLower cannot hide it.")
    try:
        bar.setProperty("mustatil_keep_visible", True)
        bar.setProperty("mustatil_lae_dino_topbar", True)
    except Exception:
        pass
    return bar


def _v9_force_show_lae_topbars() -> int:
    shown = 0
    try:
        from PySide6.QtWidgets import QApplication, QWidget
        app = QApplication.instance()
        if app is None:
            return 0
        for top in app.topLevelWidgets():
            for w in top.findChildren(QWidget):
                try:
                    if str(w.objectName() or "") == _V9_LAE_TOP_OBJECT:
                        w.setVisible(True)
                        w.show()
                        w.setMaximumHeight(42)
                        w.setMinimumHeight(34)
                        shown += 1
                except Exception:
                    pass
    except Exception:
        pass
    return shown


def _v9_remove_lae_lower_multiband_duplicates(page: Any) -> int:
    """Remove old v8/v7 LAE multiband bars inside the lower page, but keep topbar."""
    if page is None:
        return 0
    removed = 0
    try:
        from PySide6.QtWidgets import QWidget, QGroupBox
        candidates = []
        for w in page.findChildren(QWidget):
            try:
                name = str(w.objectName() or "").lower()
                if name == _V9_LAE_TOP_OBJECT.lower():
                    continue
                if "mustatil_multiband" in name or "multiband_lae" in name:
                    candidates.append(w)
            except Exception:
                pass
        for gb in page.findChildren(QGroupBox):
            try:
                if str(gb.title() or "").strip().lower() == "multiband":
                    candidates.append(gb)
            except Exception:
                pass
        seen = set()
        for w in candidates:
            if id(w) in seen:
                continue
            seen.add(id(w))
            try:
                w.setParent(None)
                w.deleteLater()
                removed += 1
            except Exception:
                pass
    except Exception:
        pass
    return removed


def _v9_wrap_lae_tab(tabwidget: Any, index: int, reason: str = "") -> bool:
    global _V9_LAE_REENTRANT
    if _V9_LAE_REENTRANT:
        return False
    try:
        if tabwidget is None or index < 0 or index >= tabwidget.count():
            return False

        page = tabwidget.widget(index)
        title = str(tabwidget.tabText(index) or "")
        if page is None or not _v9_is_lae_title(title):
            return False

        # If already wrapped, just force topbar visible and remove hidden duplicates.
        if str(page.objectName() or "") == _V9_LAE_WRAPPER or _v9_has_lae_topbar(page):
            try:
                _v9_force_show_lae_topbars()
                _v9_remove_lae_lower_multiband_duplicates(page)
            except Exception:
                pass
            return False

        _v9_remove_lae_lower_multiband_duplicates(page)

        from PySide6.QtWidgets import QWidget, QVBoxLayout
        wrapper = QWidget()
        wrapper.setObjectName(_V9_LAE_WRAPPER)
        try:
            wrapper.setProperty("mustatil_keep_visible", True)
        except Exception:
            pass
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        topbar = _v9_find_or_make_lae_topbar(page, title)
        layout.addWidget(topbar)

        icon = tabwidget.tabIcon(index)
        tip = tabwidget.tabToolTip(index)
        enabled = tabwidget.isTabEnabled(index)
        current = tabwidget.currentIndex()

        _V9_LAE_REENTRANT = True
        try:
            tabwidget.removeTab(index)
            page.setParent(wrapper)
            layout.addWidget(page, 1)
            tabwidget.insertTab(index, wrapper, icon, title)
            tabwidget.setTabToolTip(index, tip)
            tabwidget.setTabEnabled(index, enabled)
            if current == index:
                tabwidget.setCurrentIndex(index)
        finally:
            _V9_LAE_REENTRANT = False

        ws = _v7_workspace(wrapper)
        msg = f"Multiband v9 LAE-DINO TOP bar inserted above cleaned lower area ({reason or 'wrap'})."
        try:
            if ws is not None and hasattr(ws, "log"):
                ws.log(msg)
            else:
                print(f"[{PLUGIN_NAME}] {msg}", flush=True)
        except Exception:
            pass
        return True

    except Exception as exc:
        _V9_LAE_REENTRANT = False
        _warn_once("v9_lae_wrap", f"v9 LAE-DINO top-wrap failed: {exc}")
    return False


def _v9_lae_top_scan(verbose: bool = False) -> int:
    inserted = 0
    for tw, i, title, page in _v9_find_lae_tabwidgets():
        try:
            if _v9_wrap_lae_tab(tw, i, reason="v9 scan"):
                inserted += 1
        except Exception:
            pass
    shown = _v9_force_show_lae_topbars()
    if verbose:
        _log(f"v9 LAE-DINO top-bar scan: inserted={inserted}, visible_topbars={shown}")
    return inserted


def _v9_patch_cleanlower_plugin() -> bool:
    """
    Patch the CleanLower plugin after load so every clean pass is followed by
    a LAE-DINO topbar repair. We patch any module whose name contains both
    lae/dino and clean/lower/remove.
    """
    patched = 0
    try:
        import sys
        for modname, mod in list(sys.modules.items()):
            low = str(modname).lower()
            if not (("lae" in low or "dino" in low) and ("clean" in low or "lower" in low or "remove" in low)):
                continue
            for attr in dir(mod):
                if attr.startswith("__"):
                    continue
                try:
                    obj = getattr(mod, attr)
                except Exception:
                    continue
                if not callable(obj):
                    continue
                al = attr.lower()
                if not any(k in al for k in ["clean", "lower", "remove", "hide"]):
                    continue
                if getattr(obj, "_mustatil_v9_lae_repair_wrapped", False):
                    continue

                def make_wrapper(fn):
                    def wrapped(*args, **kwargs):
                        res = fn(*args, **kwargs)
                        try:
                            _v9_lae_top_scan(verbose=False)
                        except Exception:
                            pass
                        return res
                    try:
                        wrapped._mustatil_v9_lae_repair_wrapped = True
                    except Exception:
                        pass
                    return wrapped

                try:
                    setattr(mod, attr, make_wrapper(obj))
                    patched += 1
                except Exception:
                    pass
    except Exception:
        pass

    if patched:
        _log(f"v9 patched LAE-DINO CleanLower/remove-lower callbacks: {patched}; topbar will be repaired after cleanup.")
    return bool(patched)


def _v9_install_lae_topbar_repair() -> None:
    try:
        from PySide6.QtCore import QTimer
        for delay in (150, 500, 1200, 2500, 4000, 7000, 11000, 16000, 25000, 40000):
            QTimer.singleShot(delay, lambda d=delay: (_v9_patch_cleanlower_plugin(), _v9_lae_top_scan(verbose=(d in (1200, 7000, 16000, 40000)))))
    except Exception:
        pass
    _v9_patch_cleanlower_plugin()
    _v9_lae_top_scan(verbose=True)



# ---------------------------------------------------------------------------
# v10 final UI strategy: synchronous tab/page wrapper, no QTimer dependency
# ---------------------------------------------------------------------------
# Observed from the logs:
#   - v9 loaded, but initial topbar scan found inserted=0.
#   - A lower/old "v8 controls inserted: LAE-DINO" row was inserted later.
#   - CleanLower then hid the LAE-DINO lower area.
#   - FormTrainer got a control row, which is not wanted.
#
# v10 avoids all timer-based delayed UI insertion and does not insert into
# FormTrainer. It patches:
#   - QTabWidget.addTab / insertTab / setCurrentIndex
#   - workspace.log()
# The log hook is important because it fires after:
#   [LAE-DINO Trainer CleanLower v1] cleaned ...
# and immediately repairs/wraps the real tab synchronously.

_V10_BAR = "mustatil_multiband_bar_v10"
_V10_LAE_WRAPPER = "mustatil_lae_dino_multiband_top_wrapper_v10"
_V10_REENTRANT = False
_V10_INSTALLED = False


def _v10_ws(widget: Any = None) -> Any:
    return _v7_workspace(widget)


def _v10_log(msg: str) -> None:
    try:
        print(f"[{PLUGIN_NAME}] {msg}", flush=True)
    except Exception:
        pass


def _v10_text(widget: Any, limit: int = 240) -> str:
    bits = []
    def add(v):
        try:
            s = str(v or "").strip()
            if s:
                bits.append(s)
        except Exception:
            pass
    try:
        add(widget.__class__.__name__)
        if hasattr(widget, "objectName"): add(widget.objectName())
        if hasattr(widget, "windowTitle"): add(widget.windowTitle())
        if hasattr(widget, "text"): add(widget.text())
        if hasattr(widget, "title"): add(widget.title())
    except Exception:
        pass
    try:
        from PySide6.QtWidgets import QWidget
        for ch in widget.findChildren(QWidget):
            try:
                add(ch.__class__.__name__)
                if hasattr(ch, "objectName"): add(ch.objectName())
                if hasattr(ch, "text"): add(ch.text())
                if hasattr(ch, "title"): add(ch.title())
            except Exception:
                pass
            if len(bits) >= limit:
                break
    except Exception:
        pass
    return " ".join(bits).lower()


def _v10_tab_blob(title: Any = "", page: Any = None) -> str:
    return (str(title or "") + " " + (_v10_text(page) if page is not None else "")).lower()


def _v10_is_formtrainer(title: Any = "", page: Any = None) -> bool:
    blob = _v10_tab_blob(title, page)
    return "formtrainer" in blob or "form trainer" in blob or "formlearner" in blob


def _v10_is_lae(title: Any = "", page: Any = None) -> bool:
    blob = _v10_tab_blob(title, page)
    if ("lae" in blob and "dino" in blob):
        return True
    # The distinctive LAE-DINO project trainer buttons.
    return ("create dataset" in blob and "create config" in blob and ("start training" in blob or "start train" in blob))


def _v10_is_non_lae_target(title: Any = "", page: Any = None) -> bool:
    if _v10_is_formtrainer(title, page):
        return False
    blob = _v10_tab_blob(title, page)
    title_l = str(title or "").lower().strip()
    if title_l in {"detection", "satellite detection", "annotator", "console", "map", "webmap", "geospatial operations"}:
        return False
    if _v10_is_lae(title, page):
        return False
    positives = [
        "yolo trainer",
        "r-cnn trainer", "rcnn trainer", "faster r-cnn trainer", "mask r-cnn trainer",
        "u-net trainer", "unet trainer",
        "ai pipeline",
    ]
    return any(p in blob for p in positives)


def _v10_find_tab_for_widget(widget: Any):
    """Return (QTabWidget, index, tabPage, title) for an arbitrary nested widget."""
    try:
        from PySide6.QtWidgets import QTabWidget
        w = widget
        for _ in range(60):
            if w is None:
                break
            parent = w.parent()
            if isinstance(parent, QTabWidget):
                idx = parent.indexOf(w)
                if idx >= 0:
                    return parent, idx, w, str(parent.tabText(idx) or "")
            w = parent
    except Exception:
        pass
    return None, -1, None, ""


def _v10_find_lae_button_clusters():
    """Find actual LAE-DINO project trainer button clusters even if hidden."""
    found = []
    try:
        from PySide6.QtWidgets import QApplication, QPushButton, QToolButton
        app = QApplication.instance()
        if app is None:
            return found
        for top in app.topLevelWidgets():
            buttons = []
            for cls in (QPushButton, QToolButton):
                try:
                    buttons.extend(top.findChildren(cls))
                except Exception:
                    pass
            for b in buttons:
                try:
                    txt = str(b.text() or "").lower()
                    if not any(x in txt for x in ["create dataset", "create config", "start training", "start train"]):
                        continue
                    # Check whether the ancestor subtree has the three LAE buttons.
                    for anc in _v8_ancestor_chain(b, limit=25):
                        blob = _v10_text(anc, limit=260)
                        if ("create dataset" in blob and "create config" in blob and ("start training" in blob or "start train" in blob)):
                            found.append(anc)
                            break
                except Exception:
                    pass
    except Exception:
        pass
    # dedupe
    out, seen = [], set()
    for w in found:
        if id(w) not in seen:
            seen.add(id(w))
            out.append(w)
    return out


def _v10_has_bar(widget: Any) -> bool:
    if widget is None:
        return False
    try:
        if getattr(widget, "_mustatil_multiband_bar_v10", False):
            return True
        if str(widget.objectName() or "") in {_V10_BAR, _V10_LAE_WRAPPER}:
            return True
        from PySide6.QtWidgets import QWidget
        for w in widget.findChildren(QWidget):
            try:
                if str(w.objectName() or "") in {_V10_BAR, _V10_LAE_WRAPPER}:
                    return True
                if getattr(w, "_mustatil_multiband_bar_v10", False):
                    return True
            except Exception:
                pass
    except Exception:
        pass
    return False


def _v10_remove_all_mb_controls(widget: Any, keep_v10: bool = True) -> int:
    if widget is None:
        return 0
    removed = 0
    try:
        from PySide6.QtWidgets import QWidget, QGroupBox
        candidates = []
        for w in widget.findChildren(QWidget):
            try:
                name = str(w.objectName() or "").lower()
                if keep_v10 and name in {_V10_BAR.lower(), _V10_LAE_WRAPPER.lower()}:
                    continue
                if "mustatil_multiband" in name or "multiband" in name:
                    candidates.append(w)
            except Exception:
                pass
        for gb in widget.findChildren(QGroupBox):
            try:
                if keep_v10 and str(gb.objectName() or "").lower() == _V10_BAR.lower():
                    continue
                if str(gb.title() or "").strip().lower() == "multiband":
                    candidates.append(gb)
            except Exception:
                pass
        seen = set()
        for w in candidates:
            if id(w) in seen:
                continue
            seen.add(id(w))
            try:
                w.setParent(None)
                w.deleteLater()
                removed += 1
            except Exception:
                pass
    except Exception:
        pass
    return removed


def _v10_make_bar(context: Any = None, title: str = "") -> Any:
    # Reuse the working v7 profile controls, but give the widget a v10 objectName.
    bar = _v7_make_bar(context, title or "Multiband")
    try:
        bar.setObjectName(_V10_BAR)
        bar._mustatil_multiband_bar_v10 = True
        bar.setVisible(True)
        bar.show()
        bar.setMinimumHeight(34)
        bar.setMaximumHeight(44)
        bar.setToolTip("Mustatil Multiband profile. v10 bar; shared by Training, Detection, R-CNN/U-Net and AI Pipeline.")
    except Exception:
        pass
    return bar


def _v10_insert_bar_into_layout(page: Any, title: str = "") -> bool:
    if page is None:
        return False
    if _v10_has_bar(page):
        return False
    try:
        target = _v7_best_layout_widget(page)
    except Exception:
        target = page
    if target is None:
        return False
    try:
        layout = target.layout()
        if layout is None:
            return False
        _v10_remove_all_mb_controls(target, keep_v10=True)
        bar = _v10_make_bar(target, title)
        if hasattr(layout, "insertWidget"):
            layout.insertWidget(0, bar)
        elif hasattr(layout, "insertRow"):
            layout.insertRow(0, bar)
        elif hasattr(layout, "addWidget"):
            layout.addWidget(bar)
        else:
            return False
        setattr(page, "_mustatil_multiband_bar_v10", True)
        setattr(target, "_mustatil_multiband_bar_v10", True)
        ws = _v10_ws(page)
        msg = f"Multiband v10 controls inserted: {title or target.__class__.__name__}"
        if ws is not None and hasattr(ws, "log"):
            try: ws.log(msg)
            except Exception: _v10_log(msg)
        else:
            _v10_log(msg)
        return True
    except Exception as exc:
        _warn_once(f"v10_insert_{title}", f"v10 layout insert failed for {title}: {exc}")
        return False


def _v10_wrap_lae_tab(tabwidget: Any, index: int, reason: str = "") -> bool:
    global _V10_REENTRANT
    if _V10_REENTRANT:
        return False
    try:
        if tabwidget is None or index < 0 or index >= tabwidget.count():
            return False
        page = tabwidget.widget(index)
        title = str(tabwidget.tabText(index) or "")
        if page is None:
            return False
        if not _v10_is_lae(title, page):
            return False

        # If it is already v10-wrapped, just keep it visible and remove lower duplicates.
        try:
            if str(page.objectName() or "") == _V10_LAE_WRAPPER or _v10_has_bar(page):
                _v10_remove_all_mb_controls(page, keep_v10=True)
                for child in page.findChildren(type(page)):
                    pass
                return False
        except Exception:
            pass

        _v10_remove_all_mb_controls(page, keep_v10=False)

        from PySide6.QtWidgets import QWidget, QVBoxLayout
        wrapper = QWidget()
        wrapper.setObjectName(_V10_LAE_WRAPPER)
        wrapper._mustatil_multiband_bar_v10 = True
        lay = QVBoxLayout(wrapper)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        topbar = _v10_make_bar(page, "LAE-DINO Trainer")
        lay.addWidget(topbar)

        icon = tabwidget.tabIcon(index)
        tip = tabwidget.tabToolTip(index)
        enabled = tabwidget.isTabEnabled(index)
        current = tabwidget.currentIndex()

        _V10_REENTRANT = True
        try:
            tabwidget.removeTab(index)
            page.setParent(wrapper)
            lay.addWidget(page, 1)
            tabwidget.insertTab(index, wrapper, icon, title)
            tabwidget.setTabToolTip(index, tip)
            tabwidget.setTabEnabled(index, enabled)
            if current == index:
                tabwidget.setCurrentIndex(index)
        finally:
            _V10_REENTRANT = False

        ws = _v10_ws(wrapper)
        msg = f"Multiband v10 LAE-DINO topbar WRAPPED above real tab/page: {title or 'LAE-DINO'} ({reason or 'sync'})."
        if ws is not None and hasattr(ws, "log"):
            try: ws.log(msg)
            except Exception: _v10_log(msg)
        else:
            _v10_log(msg)
        return True
    except Exception as exc:
        _V10_REENTRANT = False
        _warn_once("v10_lae_wrap", f"v10 LAE-DINO wrap failed: {exc}")
    return False


def _v10_process_tab(tabwidget: Any, index: int, reason: str = "") -> bool:
    try:
        if tabwidget is None or index < 0 or index >= tabwidget.count():
            return False
        page = tabwidget.widget(index)
        title = str(tabwidget.tabText(index) or "")
        if page is None:
            return False

        # Remove wrong control from FormTrainer.
        if _v10_is_formtrainer(title, page):
            removed = _v10_remove_all_mb_controls(page, keep_v10=False)
            if removed:
                _v10_log(f"Removed wrong Multiband controls from FormTrainer: {removed}")
            return False

        if _v10_is_lae(title, page):
            return _v10_wrap_lae_tab(tabwidget, index, reason)

        if _v10_is_non_lae_target(title, page):
            _v10_remove_all_mb_controls(page, keep_v10=False)
            return _v10_insert_bar_into_layout(page, title)

    except Exception as exc:
        _warn_once("v10_process_tab", f"v10 tab process warning: {exc}")
    return False


def _v10_scan_all(reason: str = "scan") -> int:
    inserted = 0
    try:
        from PySide6.QtWidgets import QApplication, QTabWidget
        app = QApplication.instance()
        if app is None:
            return 0
        for top in app.topLevelWidgets():
            try:
                for tw in top.findChildren(QTabWidget):
                    for i in range(tw.count()):
                        if _v10_process_tab(tw, i, reason=reason):
                            inserted += 1
                # Strong fallback: find the LAE button cluster and wrap the tab that owns it.
                for cluster in _v10_find_lae_button_clusters():
                    tw, idx, page, title = _v10_find_tab_for_widget(cluster)
                    if tw is not None and idx >= 0:
                        if _v10_wrap_lae_tab(tw, idx, reason="button-cluster"):
                            inserted += 1
                    else:
                        # Last resort: place the bar into the highest visible cluster layout.
                        _v10_remove_all_mb_controls(cluster, keep_v10=False)
                        if _v10_insert_bar_into_layout(cluster, "LAE-DINO Trainer"):
                            inserted += 1
            except Exception:
                pass
    except Exception as exc:
        _warn_once("v10_scan_all", f"v10 scan warning: {exc}")
    return inserted


def _v10_patch_qtabs() -> bool:
    try:
        from PySide6.QtWidgets import QTabWidget
    except Exception:
        return False

    if getattr(QTabWidget, "_mustatil_multiband_v10_tab_patch", False):
        return True

    orig_add = QTabWidget.addTab
    orig_insert = QTabWidget.insertTab
    orig_set_current = QTabWidget.setCurrentIndex

    def patched_addTab(self, *args, **kwargs):
        idx = orig_add(self, *args, **kwargs)
        if not _V10_REENTRANT:
            try:
                _v10_process_tab(self, idx, reason="addTab")
                _v10_scan_all(reason="addTab-scan")
            except Exception:
                pass
        return idx

    def patched_insertTab(self, *args, **kwargs):
        idx = orig_insert(self, *args, **kwargs)
        if not _V10_REENTRANT:
            try:
                _v10_process_tab(self, idx, reason="insertTab")
                _v10_scan_all(reason="insertTab-scan")
            except Exception:
                pass
        return idx

    def patched_setCurrentIndex(self, index):
        res = orig_set_current(self, index)
        if not _V10_REENTRANT:
            try:
                _v10_process_tab(self, int(index), reason="setCurrentIndex")
                _v10_scan_all(reason="currentIndex-scan")
            except Exception:
                pass
        return res

    try:
        QTabWidget.addTab = patched_addTab
        QTabWidget.insertTab = patched_insertTab
        QTabWidget.setCurrentIndex = patched_setCurrentIndex
        QTabWidget._mustatil_multiband_v10_tab_patch = True
        _v10_log("v10 synchronous QTabWidget hook installed: addTab/insertTab/currentIndex, no QTimer dependency.")
        return True
    except Exception as exc:
        _warn_once("v10_qtab_patch", f"v10 QTabWidget patch failed: {exc}")
        return False


def _v10_patch_workspace_log() -> bool:
    try:
        g = globals().get("MUSTATIL_GLOBALS") or MUSTATIL_GLOBALS or {}
        cls = g.get("MustatilQtWorkspace")
        if cls is None:
            return False
        if getattr(cls, "_mustatil_multiband_v10_log_patch", False):
            return True
        old_log = cls.log

        def patched_log(self, msg="", *args, **kwargs):
            res = old_log(self, msg, *args, **kwargs)
            try:
                s = str(msg or "").lower()
                # Trigger immediately after LAE/AI Pipeline/CleanLower creation messages.
                if any(k in s for k in [
                    "lae-dino trainer cleanlower",
                    "cleaned lae-dino trainer lower area",
                    "lae-dino v18 anti-freeze panel inserted",
                    "ai pipeline hook tab ready",
                    "ai pipeline tab inserted",
                    "inserted next to lae-dino trainer",
                    "multiband v8 controls inserted: lae-dino",
                ]):
                    _v10_scan_all(reason="workspace-log")
            except Exception:
                pass
            return res

        cls.log = patched_log
        cls._mustatil_multiband_v10_log_patch = True
        _v10_log("v10 workspace.log hook installed: repairs LAE-DINO after CleanLower messages.")
        return True
    except Exception as exc:
        _warn_once("v10_log_patch", f"v10 workspace.log patch failed: {exc}")
    return False


def _v10_install_ui() -> None:
    global _V10_INSTALLED
    if _V10_INSTALLED:
        return
    _V10_INSTALLED = True
    _v10_patch_qtabs()
    _v10_patch_workspace_log()
    inserted = _v10_scan_all(reason="install")
    _v10_log(f"v10 initial synchronous UI scan complete: inserted={inserted}. FormTrainer is excluded.")



# ---------------------------------------------------------------------------
# v11 lightweight UI: insert LAE-DINO only AFTER CleanLower has finished
# ---------------------------------------------------------------------------
# v10 slowed startup because it scanned all tabs synchronously on many events.
# v11 keeps the backend/R-CNN multiband patches, but the UI part is minimal:
#   - No QTimer
#   - No setCurrentIndex hook
#   - No full scan on every tab switch
#   - No FormTrainer target
#   - LAE-DINO bar is inserted only after the CleanLower log appears:
#       "[LAE-DINO Trainer CleanLower v1] cleaned LAE-DINO Trainer lower area..."
#
# This matches the user's observation that the right time is after shrinking /
# cleaning the LAE-DINO lower area.

_V11_BAR = "mustatil_multiband_bar_v11"
_V11_LAE_WRAPPER = "mustatil_lae_dino_multiband_after_cleanlower_v11"
_V11_REENTRANT = False
_V11_INSTALLED = False


def _v11_print(msg: str) -> None:
    try:
        print(f"[{PLUGIN_NAME}] {msg}", flush=True)
    except Exception:
        pass


def _v11_workspace(widget: Any = None) -> Any:
    try:
        return _v7_workspace(widget)
    except Exception:
        return None


def _v11_make_bar(context: Any = None, title: str = "Multiband") -> Any:
    bar = _v7_make_bar(context, title)
    try:
        bar.setObjectName(_V11_BAR)
        bar._mustatil_multiband_bar_v11 = True
        bar.setMinimumHeight(34)
        bar.setMaximumHeight(44)
        bar.setVisible(True)
        bar.show()
        bar.setToolTip("Mustatil Multiband profile. v11 lightweight bar; LAE-DINO is inserted after CleanLower.")
    except Exception:
        pass
    return bar


def _v11_has_bar(widget: Any) -> bool:
    if widget is None:
        return False
    try:
        if getattr(widget, "_mustatil_multiband_bar_v11", False):
            return True
        if str(widget.objectName() or "") in {_V11_BAR, _V11_LAE_WRAPPER}:
            return True
        from PySide6.QtWidgets import QWidget
        for w in widget.findChildren(QWidget):
            try:
                if str(w.objectName() or "") in {_V11_BAR, _V11_LAE_WRAPPER}:
                    return True
                if getattr(w, "_mustatil_multiband_bar_v11", False):
                    return True
            except Exception:
                pass
    except Exception:
        pass
    return False


def _v11_remove_old_mb_controls(widget: Any, keep_v11: bool = True) -> int:
    if widget is None:
        return 0
    removed = 0
    try:
        from PySide6.QtWidgets import QWidget, QGroupBox
        candidates = []
        for w in widget.findChildren(QWidget):
            try:
                name = str(w.objectName() or "").lower()
                if keep_v11 and name in {_V11_BAR.lower(), _V11_LAE_WRAPPER.lower()}:
                    continue
                if "mustatil_multiband" in name or ("multiband" in name and "v11" not in name):
                    candidates.append(w)
            except Exception:
                pass
        for gb in widget.findChildren(QGroupBox):
            try:
                if str(gb.title() or "").strip().lower() == "multiband":
                    candidates.append(gb)
            except Exception:
                pass
        seen = set()
        for w in candidates:
            if id(w) in seen:
                continue
            seen.add(id(w))
            try:
                w.setParent(None)
                w.deleteLater()
                removed += 1
            except Exception:
                pass
    except Exception:
        pass
    return removed


def _v11_find_tab_by_title(title_part: str):
    """Small, one-shot tab lookup. Not a full recursive UI scan."""
    out = []
    try:
        from PySide6.QtWidgets import QApplication, QTabWidget
        app = QApplication.instance()
        if app is None:
            return out
        needle = str(title_part or "").lower()
        for top in app.topLevelWidgets():
            try:
                for tw in top.findChildren(QTabWidget):
                    for i in range(tw.count()):
                        title = str(tw.tabText(i) or "")
                        if needle in title.lower():
                            out.append((tw, i, title, tw.widget(i)))
            except Exception:
                pass
    except Exception:
        pass
    return out


def _v11_insert_into_layout(page: Any, title: str) -> bool:
    if page is None or _v11_has_bar(page):
        return False
    try:
        target = _v7_best_layout_widget(page)
    except Exception:
        target = page
    if target is None:
        return False
    try:
        layout = target.layout()
        if layout is None:
            return False
        _v11_remove_old_mb_controls(target, keep_v11=True)
        bar = _v11_make_bar(target, title)
        if hasattr(layout, "insertWidget"):
            layout.insertWidget(0, bar)
        elif hasattr(layout, "insertRow"):
            layout.insertRow(0, bar)
        elif hasattr(layout, "addWidget"):
            layout.addWidget(bar)
        else:
            return False
        setattr(page, "_mustatil_multiband_bar_v11", True)
        setattr(target, "_mustatil_multiband_bar_v11", True)
        ws = _v11_workspace(page)
        msg = f"Multiband v11 controls inserted: {title}"
        try:
            if ws is not None and hasattr(ws, "log"):
                ws.log(msg)
            else:
                _v11_print(msg)
        except Exception:
            _v11_print(msg)
        return True
    except Exception as exc:
        _warn_once(f"v11_insert_{title}", f"v11 insert failed for {title}: {exc}")
    return False


def _v11_wrap_tab(tw: Any, idx: int, title: str, page: Any, reason: str = "") -> bool:
    global _V11_REENTRANT
    if _V11_REENTRANT or tw is None or page is None:
        return False
    if _v11_has_bar(page):
        return False

    # Try simple layout insert first.
    if _v11_insert_into_layout(page, title):
        return True

    try:
        from PySide6.QtWidgets import QWidget, QVBoxLayout
        wrapper = QWidget()
        wrapper.setObjectName(_V11_LAE_WRAPPER if "lae" in title.lower() else "mustatil_multiband_wrapper_v11")
        wrapper._mustatil_multiband_bar_v11 = True

        lay = QVBoxLayout(wrapper)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(_v11_make_bar(page, title))

        icon = tw.tabIcon(idx)
        tip = tw.tabToolTip(idx)
        enabled = tw.isTabEnabled(idx)
        current = tw.currentIndex()

        _V11_REENTRANT = True
        try:
            tw.removeTab(idx)
            page.setParent(wrapper)
            lay.addWidget(page, 1)
            tw.insertTab(idx, wrapper, icon, title)
            tw.setTabToolTip(idx, tip)
            tw.setTabEnabled(idx, enabled)
            if current == idx:
                tw.setCurrentIndex(idx)
        finally:
            _V11_REENTRANT = False

        ws = _v11_workspace(wrapper)
        msg = f"Multiband v11 controls WRAPPED into tab: {title} ({reason or 'sync'})"
        try:
            if ws is not None and hasattr(ws, "log"):
                ws.log(msg)
            else:
                _v11_print(msg)
        except Exception:
            _v11_print(msg)
        return True
    except Exception as exc:
        _V11_REENTRANT = False
        _warn_once(f"v11_wrap_{title}", f"v11 wrap failed for {title}: {exc}")
    return False


def _v11_after_cleanlower_lae() -> int:
    """
    Called only after the CleanLower log message.
    This is the requested timing: insert after LAE-DINO tab has been shrunk.
    """
    inserted = 0
    # First exact LAE-DINO Trainer, then fallback any LAE-DINO tab.
    candidates = _v11_find_tab_by_title("LAE-DINO Trainer")
    if not candidates:
        candidates = _v11_find_tab_by_title("LAE-DINO")

    for tw, idx, title, page in candidates:
        try:
            if _v11_wrap_tab(tw, idx, title or "LAE-DINO Trainer", page, reason="after CleanLower"):
                inserted += 1
        except Exception:
            pass

    if inserted:
        _v11_print(f"LAE-DINO multiband bar inserted after CleanLower: {inserted}")
    else:
        _v11_print("LAE-DINO CleanLower detected, but no LAE-DINO tab/page was found for insertion.")
    return inserted


def _v11_is_regular_training_target(title: Any) -> bool:
    t = str(title or "").lower().strip()
    if not t:
        return False
    if "formtrainer" in t or "form trainer" in t or "formlearner" in t:
        return False
    if "lae" in t and "dino" in t:
        return False  # LAE only after CleanLower.
    allowed = [
        "yolo trainer",
        "r-cnn trainer",
        "rcnn trainer",
        "faster r-cnn trainer",
        "mask r-cnn trainer",
        "u-net trainer",
        "unet trainer",
        "ai pipeline",
    ]
    return any(x in t for x in allowed)


def _v11_patch_qtabwidget_light() -> bool:
    try:
        from PySide6.QtWidgets import QTabWidget
    except Exception:
        return False
    if getattr(QTabWidget, "_mustatil_multiband_v11_light_patch", False):
        return True

    orig_add = QTabWidget.addTab
    orig_insert = QTabWidget.insertTab

    def get_title_from_args(args):
        try:
            if len(args) >= 2 and isinstance(args[1], str):
                return args[1]
            if len(args) >= 3 and isinstance(args[2], str):
                return args[2]
            if len(args) >= 4 and isinstance(args[3], str):
                return args[3]
        except Exception:
            pass
        return ""

    def patched_addTab(self, *args, **kwargs):
        idx = orig_add(self, *args, **kwargs)
        if not _V11_REENTRANT:
            try:
                title = get_title_from_args(args) or str(self.tabText(idx) or "")
                if _v11_is_regular_training_target(title):
                    page = self.widget(idx)
                    _v11_insert_into_layout(page, title)
            except Exception:
                pass
        return idx

    def patched_insertTab(self, *args, **kwargs):
        idx = orig_insert(self, *args, **kwargs)
        if not _V11_REENTRANT:
            try:
                title = get_title_from_args(args) or str(self.tabText(idx) or "")
                if _v11_is_regular_training_target(title):
                    page = self.widget(idx)
                    _v11_insert_into_layout(page, title)
            except Exception:
                pass
        return idx

    try:
        QTabWidget.addTab = patched_addTab
        QTabWidget.insertTab = patched_insertTab
        QTabWidget._mustatil_multiband_v11_light_patch = True
        _v11_print("v11 lightweight QTabWidget hook installed: regular trainers only; LAE-DINO waits for CleanLower.")
        return True
    except Exception as exc:
        _warn_once("v11_qtab", f"v11 QTabWidget hook failed: {exc}")
        return False


def _v11_patch_workspace_log_after_cleanlower() -> bool:
    try:
        g = globals().get("MUSTATIL_GLOBALS") or MUSTATIL_GLOBALS or {}
        cls = g.get("MustatilQtWorkspace")
        if cls is None:
            return False
        if getattr(cls, "_mustatil_multiband_v11_log_patch", False):
            return True
        old_log = cls.log

        def patched_log(self, msg="", *args, **kwargs):
            res = old_log(self, msg, *args, **kwargs)
            try:
                s = str(msg or "").lower()
                # Only after shrink/cleanup. This is intentionally narrow.
                if "cleaned lae-dino trainer lower area" in s or ("lae-dino trainer cleanlower" in s and "cleaned" in s):
                    _v11_after_cleanlower_lae()
            except Exception:
                pass
            return res

        cls.log = patched_log
        cls._mustatil_multiband_v11_log_patch = True
        _v11_print("v11 workspace.log hook installed: LAE-DINO bar will be inserted after CleanLower.")
        return True
    except Exception as exc:
        _warn_once("v11_log", f"v11 workspace.log hook failed: {exc}")
    return False


def _v11_remove_formtrainer_if_present() -> int:
    removed = 0
    try:
        for tw, idx, title, page in _v11_find_tab_by_title("FormTrainer"):
            removed += _v11_remove_old_mb_controls(page, keep_v11=False)
        for tw, idx, title, page in _v11_find_tab_by_title("FormLearner"):
            removed += _v11_remove_old_mb_controls(page, keep_v11=False)
    except Exception:
        pass
    if removed:
        _v11_print(f"Removed unwanted Multiband controls from FormTrainer/FormLearner: {removed}")
    return removed


def _v11_install_ui() -> None:
    global _V11_INSTALLED
    if _V11_INSTALLED:
        return
    _V11_INSTALLED = True
    _v11_patch_qtabwidget_light()
    _v11_patch_workspace_log_after_cleanlower()
    _v11_remove_formtrainer_if_present()
    _v11_print("v11 lightweight UI installed. Startup scan disabled; LAE-DINO inserts only after CleanLower.")



# ---------------------------------------------------------------------------
# v12 late-load LAE-DINO strategy
# ---------------------------------------------------------------------------
# This file must be named with many z's so it loads AFTER:
#   zzzzzzzzzzzzzzz_mustatil_lae_dino_trainer_remove_lower_area_v1.py
#
# Reason:
#   Earlier versions inserted before CleanLower. CleanLower then moved/hidden
#   the lower area. v12 hooks QTabWidget after CleanLower is already wrapped,
#   so the Multiband row is inserted only after the LAE-DINO page has been
#   shrunk/cleaned by the inner hook.

_V12_BAR = "mustatil_multiband_bar_v12_lateload"
_V12_WRAP = "mustatil_multiband_wrapper_v12_lateload"
_V12_REENTRANT = False
_V12_INSTALLED = False


def _v12_print(msg: str) -> None:
    try:
        print(f"[{PLUGIN_NAME}] {msg}", flush=True)
    except Exception:
        pass


def _v12_workspace(widget: Any = None) -> Any:
    try:
        return _v7_workspace(widget)
    except Exception:
        return None


def _v12_make_bar(context: Any = None, title: str = "Multiband") -> Any:
    bar = _v7_make_bar(context, title)
    try:
        bar.setObjectName(_V12_BAR)
        bar._mustatil_multiband_bar_v12 = True
        bar.setMinimumHeight(34)
        bar.setMaximumHeight(44)
        bar.setVisible(True)
        bar.show()
        bar.setToolTip("Mustatil Multiband profile. v12 LateLoad: inserted after LAE-DINO CleanLower.")
    except Exception:
        pass
    return bar


def _v12_has_bar(widget: Any) -> bool:
    if widget is None:
        return False
    try:
        if getattr(widget, "_mustatil_multiband_bar_v12", False):
            return True
        if str(widget.objectName() or "") in {_V12_BAR, _V12_WRAP}:
            return True
        from PySide6.QtWidgets import QWidget
        for w in widget.findChildren(QWidget):
            try:
                if str(w.objectName() or "") in {_V12_BAR, _V12_WRAP}:
                    return True
                if getattr(w, "_mustatil_multiband_bar_v12", False):
                    return True
            except Exception:
                pass
    except Exception:
        pass
    return False


def _v12_title_is_lae(title: Any) -> bool:
    t = str(title or "").lower()
    return "lae" in t and "dino" in t


def _v12_title_is_form(title: Any) -> bool:
    t = str(title or "").lower()
    return "formtrainer" in t or "form trainer" in t or "formlearner" in t or "form learner" in t


def _v12_title_is_regular_target(title: Any) -> bool:
    t = str(title or "").lower()
    if _v12_title_is_form(t):
        return False
    if _v12_title_is_lae(t):
        return True
    return any(x in t for x in [
        "yolo trainer",
        "r-cnn trainer", "rcnn trainer", "faster r-cnn trainer", "mask r-cnn trainer",
        "u-net trainer", "unet trainer",
        "ai pipeline",
    ])


def _v12_remove_old_controls(widget: Any, keep_v12: bool = True) -> int:
    if widget is None:
        return 0
    removed = 0
    try:
        from PySide6.QtWidgets import QWidget, QGroupBox
        candidates = []
        for w in widget.findChildren(QWidget):
            try:
                name = str(w.objectName() or "").lower()
                if keep_v12 and name in {_V12_BAR.lower(), _V12_WRAP.lower()}:
                    continue
                # Remove old v3-v11 bars and wrong FormTrainer bars.
                if "mustatil_multiband" in name or ("multiband" in name and "v12" not in name):
                    candidates.append(w)
            except Exception:
                pass
        for gb in widget.findChildren(QGroupBox):
            try:
                if str(gb.title() or "").strip().lower() == "multiband":
                    candidates.append(gb)
            except Exception:
                pass
        seen = set()
        for w in candidates:
            if id(w) in seen:
                continue
            seen.add(id(w))
            try:
                w.setParent(None)
                w.deleteLater()
                removed += 1
            except Exception:
                pass
    except Exception:
        pass
    return removed


def _v12_best_layout_widget(page: Any) -> Any:
    try:
        return _v7_best_layout_widget(page)
    except Exception:
        pass
    try:
        if page is not None and page.layout() is not None:
            return page
    except Exception:
        pass
    return None


def _v12_insert_into_layout(page: Any, title: str) -> bool:
    if page is None or _v12_has_bar(page):
        return False
    target = _v12_best_layout_widget(page)
    if target is None:
        return False
    try:
        layout = target.layout()
        if layout is None:
            return False
        _v12_remove_old_controls(target, keep_v12=True)
        bar = _v12_make_bar(target, title)
        if hasattr(layout, "insertWidget"):
            layout.insertWidget(0, bar)
        elif hasattr(layout, "insertRow"):
            layout.insertRow(0, bar)
        elif hasattr(layout, "addWidget"):
            layout.addWidget(bar)
        else:
            return False
        setattr(page, "_mustatil_multiband_bar_v12", True)
        setattr(target, "_mustatil_multiband_bar_v12", True)
        ws = _v12_workspace(page)
        msg = f"Multiband v12 LateLoad controls inserted: {title}"
        try:
            if ws is not None and hasattr(ws, "log"):
                ws.log(msg)
            else:
                _v12_print(msg)
        except Exception:
            _v12_print(msg)
        return True
    except Exception as exc:
        _warn_once(f"v12_insert_{title}", f"v12 insert failed for {title}: {exc}")
    return False


def _v12_wrap_tab(tw: Any, idx: int, title: str, page: Any, reason: str = "") -> bool:
    global _V12_REENTRANT
    if _V12_REENTRANT or tw is None or page is None:
        return False
    if _v12_has_bar(page):
        return False

    # Try simple insertion first. If LAE-DINO has a usable final layout after
    # CleanLower, this is best and avoids changing the tab object.
    if _v12_insert_into_layout(page, title):
        return True

    try:
        from PySide6.QtWidgets import QWidget, QVBoxLayout
        wrapper = QWidget()
        wrapper.setObjectName(_V12_WRAP)
        wrapper._mustatil_multiband_bar_v12 = True
        lay = QVBoxLayout(wrapper)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(_v12_make_bar(page, title))

        icon = tw.tabIcon(idx)
        tip = tw.tabToolTip(idx)
        enabled = tw.isTabEnabled(idx)
        current = tw.currentIndex()

        _V12_REENTRANT = True
        try:
            tw.removeTab(idx)
            page.setParent(wrapper)
            lay.addWidget(page, 1)
            tw.insertTab(idx, wrapper, icon, title)
            tw.setTabToolTip(idx, tip)
            tw.setTabEnabled(idx, enabled)
            if current == idx:
                tw.setCurrentIndex(idx)
        finally:
            _V12_REENTRANT = False

        ws = _v12_workspace(wrapper)
        msg = f"Multiband v12 LateLoad controls WRAPPED into tab after inner hooks: {title} ({reason or 'late hook'})"
        try:
            if ws is not None and hasattr(ws, "log"):
                ws.log(msg)
            else:
                _v12_print(msg)
        except Exception:
            _v12_print(msg)
        return True
    except Exception as exc:
        _V12_REENTRANT = False
        _warn_once(f"v12_wrap_{title}", f"v12 wrap failed for {title}: {exc}")
    return False


def _v12_process_tab(tw: Any, idx: int, reason: str = "") -> bool:
    try:
        if tw is None or idx < 0 or idx >= tw.count():
            return False
        title = str(tw.tabText(idx) or "")
        page = tw.widget(idx)
        if page is None:
            return False

        if _v12_title_is_form(title):
            removed = _v12_remove_old_controls(page, keep_v12=False)
            if removed:
                _v12_print(f"Removed unwanted old Multiband controls from {title}: {removed}")
            return False

        if not _v12_title_is_regular_target(title):
            return False

        # LAE-DINO gets wrapped/inserted here AFTER CleanLower because v12 must
        # load alphabetically after the CleanLower plugin.
        return _v12_wrap_tab(tw, idx, title, page, reason=reason)

    except Exception as exc:
        _warn_once("v12_process_tab", f"v12 process warning: {exc}")
    return False


def _v12_patch_qtabwidget_lateload() -> bool:
    try:
        from PySide6.QtWidgets import QTabWidget
    except Exception:
        return False
    if getattr(QTabWidget, "_mustatil_multiband_v12_lateload_patch", False):
        return True

    orig_add = QTabWidget.addTab
    orig_insert = QTabWidget.insertTab

    def patched_addTab(self, *args, **kwargs):
        idx = orig_add(self, *args, **kwargs)
        if not _V12_REENTRANT:
            try:
                _v12_process_tab(self, idx, reason="addTab after CleanLower")
            except Exception:
                pass
        return idx

    def patched_insertTab(self, *args, **kwargs):
        idx = orig_insert(self, *args, **kwargs)
        if not _V12_REENTRANT:
            try:
                _v12_process_tab(self, idx, reason="insertTab after CleanLower")
            except Exception:
                pass
        return idx

    try:
        QTabWidget.addTab = patched_addTab
        QTabWidget.insertTab = patched_insertTab
        QTabWidget._mustatil_multiband_v12_lateload_patch = True
        _v12_print("v12 LateLoad QTabWidget hook installed. It must load after LAE-DINO CleanLower.")
        return True
    except Exception as exc:
        _warn_once("v12_qtab", f"v12 QTabWidget hook failed: {exc}")
        return False


def _v12_patch_cleanlower_module_after_load() -> bool:
    """
    Since v12 loads after CleanLower, patch any already-loaded clean/lower
    callbacks so that, if they run later, the active LAE tab is processed after.
    Lightweight: only wraps functions in modules with lae+dino+lower/clean.
    """
    patched = 0
    try:
        import sys
        for modname, mod in list(sys.modules.items()):
            low = str(modname).lower()
            if not (("lae" in low or "dino" in low) and ("lower" in low or "clean" in low or "remove" in low)):
                continue
            for attr in dir(mod):
                if attr.startswith("__"):
                    continue
                al = attr.lower()
                if not any(k in al for k in ["clean", "lower", "remove", "hide"]):
                    continue
                try:
                    fn = getattr(mod, attr)
                except Exception:
                    continue
                if not callable(fn) or getattr(fn, "_mustatil_v12_after_cleanlower", False):
                    continue

                def make_wrapper(f):
                    def wrapped(*args, **kwargs):
                        res = f(*args, **kwargs)
                        try:
                            # Process visible/existing LAE-DINO tab titles only.
                            for tw, idx, title, page in _v11_find_tab_by_title("LAE-DINO"):
                                _v12_process_tab(tw, idx, reason="after CleanLower callback")
                        except Exception:
                            pass
                        return res
                    try:
                        wrapped._mustatil_v12_after_cleanlower = True
                    except Exception:
                        pass
                    return wrapped

                try:
                    setattr(mod, attr, make_wrapper(fn))
                    patched += 1
                except Exception:
                    pass
    except Exception:
        pass

    if patched:
        _v12_print(f"v12 patched loaded LAE-DINO CleanLower callbacks: {patched}")
    return bool(patched)


def _v12_install_ui() -> None:
    global _V12_INSTALLED
    if _V12_INSTALLED:
        return
    _V12_INSTALLED = True
    _v12_patch_qtabwidget_lateload()
    _v12_patch_cleanlower_module_after_load()
    _v12_print("v12 LateLoad UI installed: no startup scan, no timers, FormTrainer excluded, LAE-DINO handled after CleanLower by load order.")


def _install_all() -> None:
    global _PATCHED
    if _PATCHED:
        return
    g = globals().get("MUSTATIL_GLOBALS") or MUSTATIL_GLOBALS or {}
    backend = g.get("backend") or sys.modules.get("mustatil_legacy_backend")
    # v12 UI only: late-load hooks. File name must sort after LAE-DINO CleanLower plugin.
    _v12_install_ui()
    _patch_backend_io(backend)
    _patch_backend_training(backend)
    _patch_direct_rasterio_reads()
    _patch_pil_open()
    _patch_cv2_imread()
    _patch_ultralytics()
    _patch_detection_preview_service()
    _PATCHED = True
    _log("Installed: backend IO/training hooks + direct R-CNN rasterio bridge + PIL/cv2/YOLO path adapters + v12 LateLoad UI controls. No new tab created.")
    _watch_qt_app()


def mustatil_plugin_init():
    try:
        _install_all()
    except Exception as exc:
        print(f"[{PLUGIN_NAME} ERROR] Init failed: {exc}\n{traceback.format_exc()}", flush=True)


# CLI helpers for testing from PowerShell without opening Mustatil.
def _cmd_scan(argv: List[str]) -> int:
    if not argv:
        print("Usage: python plugin.py scan <multiband.tif>")
        return 2
    p = argv[0]
    info = _scan_tiff(p)
    prof = _find_profile(p, max_band=int(info.get("count") or 0)) or _default_profile(p, max_band=int(info.get("count") or 0))
    print(json.dumps({"raster": info, "active_profile": prof}, indent=2, ensure_ascii=False))
    return 0


def _cmd_make_profile(argv: List[str]) -> int:
    if not argv:
        print("Usage: python plugin.py make-profile <multiband.tif> --bands 3,4,8")
        return 2
    p = argv[0]
    bands = "1,2,3"
    norm = "percentile_2_98"
    out = None
    i = 1
    while i < len(argv):
        if argv[i] == "--bands" and i + 1 < len(argv):
            bands = argv[i+1]; i += 2; continue
        if argv[i] == "--normalization" and i + 1 < len(argv):
            norm = argv[i+1]; i += 2; continue
        if argv[i] == "--out" and i + 1 < len(argv):
            out = argv[i+1]; i += 2; continue
        i += 1
    count = _tiff_band_count(p)
    prof = _default_profile(p, max_band=count)
    prof["selected_bands"] = _parse_bands(bands, max_band=count)
    prof["normalization"] = norm
    if out:
        outp = Path(out)
    else:
        pp = Path(p).expanduser().parent
        # If the raster is inside a normal Mustatil images/ folder, place the
        # profile in the project root so Training and Detection share it.
        if pp.name.lower() in {"images", "image", "imgs", "rasters", "raster"}:
            outp = pp.parent / PROFILE_NAME
        else:
            outp = pp / PROFILE_NAME
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(prof, indent=2, ensure_ascii=False), encoding="utf-8")
    # Also write a tiny sidecar copy next to the raster so CLI tests and model
    # plugins without workspace context can still find the same profile.
    try:
        sidecar = Path(p).expanduser().parent / PROFILE_NAME
        if sidecar.resolve() != outp.resolve():
            sidecar.write_text(json.dumps(prof, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Wrote sidecar profile: {sidecar}")
    except Exception:
        pass
    print(f"Wrote profile: {outp}")
    print(json.dumps(prof, indent=2, ensure_ascii=False))
    return 0


def _cmd_composite(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Usage: python plugin.py composite <multiband.tif> <out.png> --bands 3,4,8")
        return 2
    p, out = argv[0], argv[1]
    bands = None
    i = 2
    while i < len(argv):
        if argv[i] == "--bands" and i + 1 < len(argv):
            bands = argv[i+1]; i += 2; continue
        i += 1
    count = _tiff_band_count(p)
    prof = _default_profile(p, max_band=count)
    if bands:
        prof["selected_bands"] = _parse_bands(bands, max_band=count)
    im = _read_full_tiff_as_pil(p, workspace=None, max_side=None)
    im.save(out)
    print(f"Wrote composite: {out}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1].lower() if len(sys.argv) > 1 else "help"
    args = sys.argv[2:]
    if cmd == "scan":
        raise SystemExit(_cmd_scan(args))
    if cmd in {"make-profile", "profile"}:
        raise SystemExit(_cmd_make_profile(args))
    if cmd in {"composite", "preview"}:
        raise SystemExit(_cmd_composite(args))
    if cmd in {"check-rcnn", "check-rasterio", "check-direct"}:
        if not args:
            print("Usage: python plugin.py check-rcnn <multiband.tif>")
            raise SystemExit(2)
        _patch_direct_rasterio_reads()
        rio = _rasterio()
        if rio is None:
            print("rasterio missing")
            raise SystemExit(1)
        with rio.open(args[0]) as src:
            print("opened:", args[0], "bands=", getattr(src, "count", None))
            arr = src.read([1, 2, 3])
            print("read([1,2,3]) result shape:", getattr(arr, "shape", None))
        raise SystemExit(0)
    print("Mustatil Multiband All-AI + R-CNN Patch v12 LateLoad")
    print("Commands:")
    print("  scan <multiband.tif>")
    print("  make-profile <multiband.tif> --bands 3,4,8 [--normalization percentile_2_98]")
    print("  composite <multiband.tif> <out.png> --bands 3,4,8")
    print("  check-rcnn <multiband.tif>")
    print("  check-ui")
    print("  check-v12-ui")
