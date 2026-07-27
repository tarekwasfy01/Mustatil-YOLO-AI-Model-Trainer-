#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detection preview helper for Mustatil Qt Workspace.

SERVICE-ONLY SAFE VERSION
-------------------------
This file is intentionally compatible with the unchanged main program.
It keeps the old public API:

    is_tiff_path(path)
    load_detection_preview_image(path, maxs=2400, fallback_loader=None)
    compute_zoom_preview_max(...)

Important:
The unchanged Mustatil Qt workspace only calls the two functions above for
Detection preview loading/reloading. It does not pass the current viewport to
this service. Therefore a true QGIS-like tile canvas cannot be implemented by
changing this file alone. To avoid the previous "jump back" behaviour, this
service disables automatic bigger full-preview reloads on mouse-wheel zoom and
loads only one stable overview image initially.

Result:
- Huge TIFF/GeoTIFF files open through Rasterio using overviews/pyramids.
- The preview is stable while panning/zooming in Qt.
- No 12k/full preview reload is triggered during zoom.
- Existing detections remain in the correct scaled preview position.

If the main program is later changed to call load_detection_viewport_image(),
this module also provides a viewport reader that reads only the visible window.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple
from collections import OrderedDict
import os
import time

from PIL import Image

TIFF_EXTENSIONS = (".tif", ".tiff")

# Small in-process viewport cache. The Qt side can request many nearly identical
# viewports while the user zooms/pans. Caching and source-window snapping makes
# these repeated calls return instantly instead of hammering a 100 GB TIFF.
_VIEWPORT_CACHE = OrderedDict()
_VIEWPORT_CACHE_BYTES = 0
_VIEWPORT_CACHE_LIMIT = int(os.environ.get("MUSTATIL_DET_PREVIEW_CACHE_MB", "384")) * 1024 * 1024
_PREVIEW_CACHE = OrderedDict()
_PREVIEW_CACHE_LIMIT = 3


def clear_detection_preview_cache():
    """Clear cached preview/viewport tiles. Safe optional API for the GUI."""
    global _VIEWPORT_CACHE_BYTES
    _VIEWPORT_CACHE.clear()
    _PREVIEW_CACHE.clear()
    _VIEWPORT_CACHE_BYTES = 0


def _file_stamp(path: str):
    try:
        st = os.stat(path)
        return int(st.st_mtime_ns), int(st.st_size)
    except Exception:
        return 0, 0


def _cache_put(key, value, approx_bytes: int):
    global _VIEWPORT_CACHE_BYTES
    if approx_bytes > _VIEWPORT_CACHE_LIMIT:
        return value
    old = _VIEWPORT_CACHE.pop(key, None)
    if old is not None:
        _VIEWPORT_CACHE_BYTES -= int(old[1])
    _VIEWPORT_CACHE[key] = (value, int(approx_bytes), time.time())
    _VIEWPORT_CACHE_BYTES += int(approx_bytes)
    while _VIEWPORT_CACHE_BYTES > _VIEWPORT_CACHE_LIMIT and _VIEWPORT_CACHE:
        _, oldv = _VIEWPORT_CACHE.popitem(last=False)
        _VIEWPORT_CACHE_BYTES -= int(oldv[1])
    return value


def _cache_get(key):
    item = _VIEWPORT_CACHE.pop(key, None)
    if item is None:
        return None
    _VIEWPORT_CACHE[key] = item
    return item[0]


def _qround(value: float, quantum: float) -> int:
    q = max(1.0, float(quantum or 1.0))
    return int(round(float(value) / q))


def is_tiff_path(path: str) -> bool:
    return str(path or "").lower().endswith(TIFF_EXTENSIONS)


def compute_zoom_preview_max(
    zoom: float,
    base_max: int = 2400,
    current_max: int = 2400,
    hard_max: int = 12000,
    reload_factor: float = 1.20,
) -> Optional[int]:
    """Return None to keep the current preview stable during zoom.

    The unchanged main program replaces the whole QGraphicsScene whenever this
    returns a number. On very large TIFFs this causes jumping/back-snapping and
    expensive full-preview reads. QGIS-like behaviour requires viewport/tile
    calls from the main program; with service-only replacement the safest fix is
    to keep the initially loaded overview and let Qt pan/zoom it.
    """
    return None


def _normalize_to_uint8(arr):
    import numpy as np

    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr

    arr = arr.astype("float32", copy=False)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype="uint8")

    # Per-preview robust contrast stretch. Percentiles avoid one bad NoData
    # value making the whole preview black/white.
    valid = arr[finite]
    try:
        lo = float(np.nanpercentile(valid, 1))
        hi = float(np.nanpercentile(valid, 99))
    except Exception:
        lo = float(np.nanmin(valid))
        hi = float(np.nanmax(valid))
    if hi <= lo:
        hi = lo + 1.0
    arr = (np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype("uint8")
    return arr


def _bands_for_rgb(src):
    # Read at most 3 bands. If only one band exists, it is repeated later.
    count = int(getattr(src, "count", 0) or 0)
    if count <= 0:
        raise RuntimeError("TIFF contains no readable raster bands.")
    return list(range(1, min(3, count) + 1))


def _array_to_pil_rgb(arr):
    import numpy as np

    arr = _normalize_to_uint8(arr)
    if arr.ndim != 3:
        raise RuntimeError("Unexpected raster array shape.")
    # Rasterio read shape is bands, height, width.
    if arr.shape[0] == 1:
        img = Image.fromarray(arr[0], mode="L").convert("RGB")
    else:
        if arr.shape[0] < 3:
            arr = np.repeat(arr[:1], 3, axis=0)
        img = Image.fromarray(np.transpose(arr[:3], (1, 2, 0)), mode="RGB")
    return img


def _pick_stable_preview_size(width: int, height: int, maxs: int) -> Tuple[int, int]:
    """Pick a small, stable overview canvas size.

    For huge files this intentionally remains conservative because the unchanged
    GUI cannot request true visible tiles from this service.
    """
    w = max(1, int(width))
    h = max(1, int(height))
    maxs = max(512, int(maxs or 2400))
    # Do not silently obey very high hard reload requests if an older main file
    # passes them. This is the anti-freeze/anti-jump guard.
    maxs = min(maxs, 4096)
    scale = min(1.0, float(maxs) / float(max(w, h)))
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    return out_w, out_h


def load_detection_preview_image(
    path: str,
    maxs: int = 2400,
    fallback_loader: Optional[Callable[..., Tuple[Image.Image, int, int]]] = None,
) -> Tuple[Image.Image, int, int]:
    """Load a stable overview preview and return (PIL image, original W, original H).

    This is intentionally conservative for huge TIFFs. It reads one small
    overview image and caches it; close zoom rendering is handled by
    load_detection_viewport_image().
    """
    p = str(path or "").strip().strip('"')
    if not p:
        raise RuntimeError("No image path supplied for detection preview.")

    if not is_tiff_path(p):
        if fallback_loader is not None:
            return fallback_loader(p, maxs=maxs)
        im = Image.open(p).convert("RGB")
        w, h = im.size
        im.thumbnail((max(512, int(maxs or 2400)), max(512, int(maxs or 2400))), Image.Resampling.LANCZOS)
        return im.copy(), w, h

    cache_key = (p, _file_stamp(p), int(min(max(512, int(maxs or 2400)), 4096)))
    cached = _PREVIEW_CACHE.get(cache_key)
    if cached is not None:
        _PREVIEW_CACHE.move_to_end(cache_key)
        im, w, h = cached
        return im.copy(), w, h

    try:
        import rasterio
        from rasterio.enums import Resampling

        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", NUM_THREADS="ALL_CPUS"):
            with rasterio.open(p, sharing=False) as src:
                w, h = int(src.width), int(src.height)
                out_w, out_h = _pick_stable_preview_size(w, h, maxs)
                bands = _bands_for_rgb(src)
                arr = src.read(
                    bands,
                    out_shape=(len(bands), out_h, out_w),
                    resampling=Resampling.nearest,
                    masked=False,
                )
                im = _array_to_pil_rgb(arr)
                _PREVIEW_CACHE[cache_key] = (im.copy(), w, h)
                while len(_PREVIEW_CACHE) > _PREVIEW_CACHE_LIMIT:
                    _PREVIEW_CACHE.popitem(last=False)
                return im, w, h
    except Exception:
        if fallback_loader is not None:
            return fallback_loader(p, maxs=min(max(512, int(maxs or 2400)), 4096))
        im = Image.open(p).convert("RGB")
        w, h = im.size
        im.thumbnail((min(max(512, int(maxs or 2400)), 4096), min(max(512, int(maxs or 2400)), 4096)), Image.Resampling.LANCZOS)
        return im.copy(), w, h


# ---------------------------------------------------------------------------
# Optional future API: true visible-window reader.
# The current unmodified main program does not call this function, but keeping
# it here makes this service ready for a future QGIS-like viewport integration.
# ---------------------------------------------------------------------------
def load_detection_viewport_image(
    path: str,
    center_x: float = None,
    center_y: float = None,
    view_width: int = None,
    view_height: int = None,
    pixels_per_source_pixel: float = None,
    max_output_px: int = 4096,
    # Newer Mustatil main-program variants pass these keyword names:
    source_width: float = None,
    source_height: float = None,
    out_width: int = None,
    out_height: int = None,
    tile_px: int = 512,
    # Alternative/future names, accepted so the main program cannot crash on kwargs:
    src_width: float = None,
    src_height: float = None,
    viewport_width: int = None,
    viewport_height: int = None,
    **_ignored_kwargs,
):
    """Read only the visible source-pixel window from a TIFF/GeoTIFF.

    Smoother plugin-only behaviour:
    - accepts all known Qt keyword variants;
    - expands the requested source window slightly as a pan buffer;
    - snaps read windows to a grid based on current source-pixels/screen-pixel;
    - caches recent viewport images in memory;
    - returns a 7-tuple so Qt can place the tile in original image coordinates:
      (PIL image, original_width, original_height, x, y, source_width, source_height)
    """
    p = str(path or "").strip().strip('"')
    if not is_tiff_path(p):
        raise RuntimeError("Viewport reading is only implemented for TIFF/GeoTIFF files.")

    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window

    # Normalize keyword aliases.
    if source_width is None:
        source_width = src_width
    if source_height is None:
        source_height = src_height
    if out_width is None:
        out_width = view_width if view_width is not None else viewport_width
    if out_height is None:
        out_height = view_height if view_height is not None else viewport_height

    max_output_px = max(256, min(4096, int(max_output_px or 4096)))
    tile_px = max(128, int(tile_px or 512))
    requested_out_w = max(64, int(out_width or view_width or viewport_width or 1024))
    requested_out_h = max(64, int(out_height or view_height or viewport_height or 768))
    out_w = max(1, min(max_output_px, requested_out_w))
    out_h = max(1, min(max_output_px, requested_out_h))

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", NUM_THREADS="ALL_CPUS"):
        with rasterio.open(p, sharing=False) as src:
            W, H = int(src.width), int(src.height)

            if source_width is not None and source_height is not None:
                req_src_w = max(1.0, float(source_width))
                req_src_h = max(1.0, float(source_height))
            else:
                pps = max(1e-6, float(pixels_per_source_pixel or 1.0))
                req_src_w = max(1.0, requested_out_w / pps)
                req_src_h = max(1.0, requested_out_h / pps)

            cx = float(center_x if center_x is not None else W / 2.0)
            cy = float(center_y if center_y is not None else H / 2.0)

            # Add an internal buffer, like a map renderer cache. This reduces
            # visible stutter because small pans remain inside the already loaded
            # source window. Keep it moderate so close zoom stays sharp.
            buffer_factor = 1.22
            src_w = min(float(W), req_src_w * buffer_factor)
            src_h = min(float(H), req_src_h * buffer_factor)
            left = cx - src_w / 2.0
            top = cy - src_h / 2.0

            if src_w >= W:
                left = 0.0
                src_w = float(W)
            else:
                left = max(0.0, min(float(W) - src_w, left))
            if src_h >= H:
                top = 0.0
                src_h = float(H)
            else:
                top = max(0.0, min(float(H) - src_h, top))

            # Preserve aspect ratio to avoid distorted detection boxes.
            src_aspect = src_w / max(1e-9, src_h)
            out_aspect = out_w / max(1e-9, out_h)
            if abs(src_aspect - out_aspect) > 0.02:
                if out_aspect > src_aspect:
                    out_w = max(1, int(round(out_h * src_aspect)))
                else:
                    out_h = max(1, int(round(out_w / src_aspect)))

            # Snap source reads to a grid proportional to current resolution.
            # At close zoom the grid is small; when zoomed out it is larger.
            spp_x = src_w / max(1.0, float(out_w))
            spp_y = src_h / max(1.0, float(out_h))
            grid = max(8.0, min(4096.0, max(spp_x, spp_y) * 96.0))
            q_left = max(0.0, min(float(W) - src_w, round(left / grid) * grid)) if src_w < W else 0.0
            q_top = max(0.0, min(float(H) - src_h, round(top / grid) * grid)) if src_h < H else 0.0
            # Keep width/height stable across tiny zoom changes.
            q_src_w = min(float(W), max(1.0, round(src_w / grid) * grid))
            q_src_h = min(float(H), max(1.0, round(src_h / grid) * grid))
            if q_left + q_src_w > W:
                q_left = max(0.0, float(W) - q_src_w)
            if q_top + q_src_h > H:
                q_top = max(0.0, float(H) - q_src_h)

            # Quantize output too. It avoids a new cache entry for 1 px viewport
            # changes caused by dock/scrollbar layout updates.
            q_out_w = max(1, int(round(out_w / 16.0) * 16))
            q_out_h = max(1, int(round(out_h / 16.0) * 16))
            q_out_w = min(max_output_px, q_out_w)
            q_out_h = min(max_output_px, q_out_h)

            key = (
                p, _file_stamp(p), W, H,
                _qround(q_left, 1), _qround(q_top, 1),
                _qround(q_src_w, 1), _qround(q_src_h, 1),
                q_out_w, q_out_h, tuple(_bands_for_rgb(src)),
            )
            cached = _cache_get(key)
            if cached is not None:
                im, rx, ry, rw, rh = cached
                return im.copy(), W, H, rx, ry, rw, rh

            bands = _bands_for_rgb(src)
            # Let GDAL/Rasterio use internal overviews/blocks. A single window
            # read is usually smoother than manually reading many tiny pieces.
            # For large output requests, stripe it to avoid memory spikes.
            if max(q_out_w, q_out_h) <= max(1536, tile_px * 3):
                arr = src.read(
                    bands,
                    window=Window(q_left, q_top, q_src_w, q_src_h),
                    out_shape=(len(bands), q_out_h, q_out_w),
                    resampling=Resampling.nearest,
                    boundless=False,
                    masked=False,
                )
                im = _array_to_pil_rgb(arr)
            else:
                import numpy as np
                out = np.zeros((len(bands), q_out_h, q_out_w), dtype="uint8")
                stripe_h = max(128, min(tile_px, q_out_h))
                for oy in range(0, q_out_h, stripe_h):
                    oh = min(stripe_h, q_out_h - oy)
                    sub_top = q_top + (oy / float(q_out_h)) * q_src_h
                    sub_h = q_src_h * (oh / float(q_out_h))
                    sub = src.read(
                        bands,
                        window=Window(q_left, sub_top, q_src_w, sub_h),
                        out_shape=(len(bands), oh, q_out_w),
                        resampling=Resampling.nearest,
                        boundless=False,
                        masked=False,
                    )
                    sub = _normalize_to_uint8(sub)
                    out[:, oy:oy + oh, :] = sub
                im = _array_to_pil_rgb(out)

            value = (im.copy(), float(q_left), float(q_top), float(q_src_w), float(q_src_h))
            approx = int(im.width) * int(im.height) * 4
            _cache_put(key, value, approx)
            return im, W, H, float(q_left), float(q_top), float(q_src_w), float(q_src_h)

