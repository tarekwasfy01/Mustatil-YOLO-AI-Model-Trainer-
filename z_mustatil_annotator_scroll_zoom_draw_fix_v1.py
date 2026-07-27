#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mustatil plugin: Faster R-CNN, Mask R-CNN, U-Net and SAM2 tabs for Detection + Satellite Detection.

Drop this file into Mustatil's `mustatil_plugins` folder and restart Mustatil.

What it does
------------
- Adds additional LEFT-PANEL tabs to the existing Detection and Satellite Detection pages:
    - Faster R-CNN R50-FPN Detection
    - Mask R-CNN R50-FPN Detection / masks-to-boxes
    - U-Net Semantic Segmentation / connected components-to-boxes
    - SAM2 Box-Prompt Segmentation / masks-to-boxes
- Keeps Mustatil's original right preview/map untouched.
- Uses Mustatil's existing tiled image/satellite workflow as much as possible.
- Produces Mustatil detection boxes in `ws.dets` for image Detection.
- Produces Satellite Detection records and writes GPKG through Mustatil's own satellite export method.

Dependencies
------------
Required for Faster R-CNN / Mask R-CNN / U-Net tabs:
    python -m pip install torch torchvision pillow numpy

Optional but useful for faster U-Net connected components:
    python -m pip install opencv-python

SAM2 tab:
    Uses Mustatil/legacy backend SAM2 loading when available. Usually this means
    Ultralytics SAM/SAM2 support is already installed in the Mustatil environment.

Model notes
-----------
- Faster R-CNN / Mask R-CNN can run with torchvision COCO weights immediately.
- For archaeological classes, use custom .pth weights trained for your classes.
- U-Net requires a custom TorchScript or state_dict model. This plugin includes a small
  binary U-Net architecture for state_dict loading, but it cannot magically segment
  archaeology without trained weights.
- SAM2 is prompt-based, not a normal detector. This tab uses existing boxes as prompts
  and converts produced masks back to Mustatil-compatible boxes/GPKG records.
"""
from __future__ import annotations

import math
import os
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_PATCHED_QTAB = False
_ORIG_ADD = None
_ORIG_INSERT = None
_SCAN_TIMER = None
_PATCHED_PAGES: set[int] = set()
WEB_TILE_SIZE = 256


def _log(msg: str) -> None:
    try:
        print("[Mustatil TorchVision/UNet Tabs] " + str(msg))
    except Exception:
        pass


def _get_globals() -> Dict[str, Any]:
    try:
        g = globals().get("MUSTATIL_GLOBALS", None)
        if isinstance(g, dict):
            return g
    except Exception:
        pass
    return {}


def _workspace_from_widget(widget: Any) -> Optional[Any]:
    cur = widget
    for _ in range(160):
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


def _get_var(v: Any, default: Any = "") -> Any:
    try:
        if hasattr(v, "get"):
            return v.get()
    except Exception:
        pass
    return default


def _redraw(ws: Any) -> None:
    for fn in (
        lambda: ws.loadprev(),
        lambda: ws.redraw(fit=False),
        lambda: ws.redraw(),
        lambda: ws.satellite_redraw_detection_overlay(),
        lambda: ws.refresh_layers(),
    ):
        try:
            fn()
        except Exception:
            pass
    try:
        sig = getattr(getattr(ws, "signals", None), "sat_overlay_redraw_requested", None)
        if sig is not None:
            sig.emit(0)
    except Exception:
        pass


def _run_as_task(ws: Any, title: str, fn, allow_parallel: bool = False):
    try:
        return ws.run_task(title, fn, allow_parallel=allow_parallel)
    except TypeError:
        try:
            return ws.run_task(title, fn)
        except Exception:
            pass
    except Exception:
        pass
    return fn()


def _show_error(ws: Any, title: str, exc: Any) -> None:
    text = str(exc)
    try:
        ws.show_error(title, text)
    except Exception:
        try:
            ws.log(f"{title}: {text}")
        except Exception:
            _log(f"{title}: {text}")
    try:
        traceback.print_exc()
    except Exception:
        pass


def _device_from_combo(combo: Any) -> str:
    try:
        raw = str(combo.currentText()).strip().lower()
    except Exception:
        raw = "cpu"
    return raw if raw in {"cpu", "cuda", "auto"} else "cpu"


def _resolve_torch_device(device_name: str):
    import torch
    req = str(device_name or "cpu").lower().strip()
    if req in {"cuda", "auto"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def _load_state_file(path: str, map_location="cpu"):
    import torch
    p = str(path or "").strip().strip('"')
    if not p:
        return None
    obj = torch.load(p, map_location=map_location)
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            val = obj.get(key)
            if isinstance(val, dict):
                return val
    return obj


def _strip_state_dict_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in state.items():
        kk = str(k)
        for pref in ("module.", "model.", "net."):
            if kk.startswith(pref):
                kk = kk[len(pref):]
        out[kk] = v
    return out


def _replace_faster_head(model: Any, num_classes: int):
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, int(num_classes))
    return model


def _replace_mask_head(model: Any, num_classes: int):
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, int(num_classes))
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, int(num_classes))
    return model


def _torchvision_faster(pretrained: bool, num_classes: Optional[int] = None):
    import torchvision
    if pretrained:
        try:
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
            return torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        except Exception:
            return torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    try:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    except TypeError:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    if num_classes and int(num_classes) > 1:
        model = _replace_faster_head(model, int(num_classes))
    return model


def _torchvision_mask(pretrained: bool, num_classes: Optional[int] = None):
    import torchvision
    if pretrained:
        try:
            from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
            return torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        except Exception:
            return torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    try:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    except TypeError:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    if num_classes and int(num_classes) > 1:
        model = _replace_mask_head(model, int(num_classes))
    return model


def _load_detection_model(ws: Any, kind: str, device_name: str, custom_path: str = "", num_classes: int = 2):
    """Load Faster R-CNN or Mask R-CNN; cache per workspace."""
    import torch
    device = _resolve_torch_device(device_name)
    custom_path = str(custom_path or "").strip().strip('"')
    use_custom = bool(custom_path)
    key = (kind, str(device), custom_path, int(num_classes or 2))
    cache = getattr(ws, "_mustatil_torchvision_model_cache", {}) or {}
    if key in cache:
        return cache[key]

    if kind == "faster":
        model = _torchvision_faster(pretrained=not use_custom, num_classes=(int(num_classes) if use_custom else None))
        model_name = "Faster R-CNN ResNet-50-FPN COCO" if not use_custom else Path(custom_path).name
    elif kind == "mask":
        model = _torchvision_mask(pretrained=not use_custom, num_classes=(int(num_classes) if use_custom else None))
        model_name = "Mask R-CNN ResNet-50-FPN COCO" if not use_custom else Path(custom_path).name
    else:
        raise RuntimeError(f"Unknown detection model kind: {kind}")

    if use_custom:
        # First try TorchScript, then fall back to state_dict loading into the torchvision architecture.
        try:
            scripted = torch.jit.load(custom_path, map_location=device)
            model = scripted
        except Exception:
            state = _load_state_file(custom_path, map_location="cpu")
            if not isinstance(state, dict):
                raise RuntimeError("Custom model file is neither TorchScript nor a state_dict dictionary.")
            state = _strip_state_dict_prefix(state)
            missing, unexpected = model.load_state_dict(state, strict=False)
            try:
                ws.log(f"Loaded custom {kind} state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
            except Exception:
                pass

    model.to(device)
    model.eval()
    cache[key] = (model, device, model_name)
    ws._mustatil_torchvision_model_cache = cache
    try:
        ws.log(f"Loaded {model_name} on {device}")
    except Exception:
        pass
    return cache[key]


# -----------------------------------------------------------------------------
# A small binary U-Net implementation for state_dict loading
# -----------------------------------------------------------------------------

def _make_unet(in_channels: int = 3, out_channels: int = 1, base: int = 32):
    import torch
    from torch import nn

    class DoubleConv(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            )
        def forward(self, x):
            return self.net(x)

    class SimpleUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.d1 = DoubleConv(in_channels, base)
            self.p1 = nn.MaxPool2d(2)
            self.d2 = DoubleConv(base, base * 2)
            self.p2 = nn.MaxPool2d(2)
            self.d3 = DoubleConv(base * 2, base * 4)
            self.p3 = nn.MaxPool2d(2)
            self.mid = DoubleConv(base * 4, base * 8)
            self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
            self.c3 = DoubleConv(base * 8, base * 4)
            self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
            self.c2 = DoubleConv(base * 4, base * 2)
            self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
            self.c1 = DoubleConv(base * 2, base)
            self.out = nn.Conv2d(base, out_channels, 1)

        @staticmethod
        def _cat(a, b):
            # Padding guard for odd tile sizes.
            import torch.nn.functional as F
            dy = b.size(2) - a.size(2)
            dx = b.size(3) - a.size(3)
            if dx != 0 or dy != 0:
                a = F.pad(a, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
            return torch.cat([b, a], dim=1)

        def forward(self, x):
            d1 = self.d1(x)
            d2 = self.d2(self.p1(d1))
            d3 = self.d3(self.p2(d2))
            m = self.mid(self.p3(d3))
            x = self.c3(self._cat(self.u3(m), d3))
            x = self.c2(self._cat(self.u2(x), d2))
            x = self.c1(self._cat(self.u1(x), d1))
            return self.out(x)

    return SimpleUNet()


def _load_unet_model(ws: Any, device_name: str, model_path: str, in_channels: int = 3, out_channels: int = 1, base: int = 32):
    import torch
    device = _resolve_torch_device(device_name)
    path = str(model_path or "").strip().strip('"')
    if not path:
        raise RuntimeError("Please choose a trained U-Net .pt/.pth model first.")
    key = ("unet", str(device), path, int(in_channels), int(out_channels), int(base))
    cache = getattr(ws, "_mustatil_unet_model_cache", {}) or {}
    if key in cache:
        return cache[key]

    try:
        model = torch.jit.load(path, map_location=device)
        model_name = "TorchScript U-Net: " + Path(path).name
    except Exception:
        model = _make_unet(in_channels=int(in_channels), out_channels=int(out_channels), base=int(base))
        state = _load_state_file(path, map_location="cpu")
        if not isinstance(state, dict):
            raise RuntimeError("U-Net file is not TorchScript and does not contain a state_dict.")
        state = _strip_state_dict_prefix(state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        model_name = "Simple U-Net state_dict: " + Path(path).name
        try:
            ws.log(f"Loaded U-Net state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
        except Exception:
            pass
    model.to(device)
    model.eval()
    cache[key] = (model, device, model_name)
    ws._mustatil_unet_model_cache = cache
    try:
        ws.log(f"Loaded {model_name} on {device}")
    except Exception:
        pass
    return cache[key]


# -----------------------------------------------------------------------------
# Inference helpers
# -----------------------------------------------------------------------------

def _pil_to_tensor(pil_image):
    import torchvision.transforms.functional as F
    return F.to_tensor(pil_image.convert("RGB"))


def _detect_torchvision_pil(
    ws: Any,
    kind: str,
    pil_image,
    conf: float,
    device_name: str,
    custom_path: str = "",
    num_classes: int = 2,
    mask_threshold: float = 0.50,
) -> List[Dict[str, Any]]:
    import torch
    model, device, model_name = _load_detection_model(ws, kind, device_name, custom_path, num_classes)
    tensor = _pil_to_tensor(pil_image).to(device)
    with torch.no_grad():
        outputs = model([tensor])
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    if isinstance(outputs, list):
        out = outputs[0] if outputs else {}
    elif isinstance(outputs, dict):
        out = outputs
    else:
        raise RuntimeError(f"Unsupported model output type: {type(outputs)}")

    boxes = out.get("boxes", []) if isinstance(out, dict) else []
    labels = out.get("labels", []) if isinstance(out, dict) else []
    scores = out.get("scores", []) if isinstance(out, dict) else []
    masks = out.get("masks", None) if isinstance(out, dict) else None

    boxes = boxes.detach().cpu().tolist() if hasattr(boxes, "detach") else list(boxes or [])
    labels = labels.detach().cpu().tolist() if hasattr(labels, "detach") else list(labels or [])
    scores = scores.detach().cpu().tolist() if hasattr(scores, "detach") else list(scores or [])
    mask_data = None
    if masks is not None and hasattr(masks, "detach"):
        mask_data = masks.detach().cpu()

    result: List[Dict[str, Any]] = []
    w, h = pil_image.size
    for idx, (bb, lab, score) in enumerate(zip(boxes, labels, scores)):
        score = float(score)
        if score < float(conf):
            continue
        x1, y1, x2, y2 = [float(v) for v in bb[:4]]
        x1 = max(0.0, min(float(w), x1)); x2 = max(0.0, min(float(w), x2))
        y1 = max(0.0, min(float(h), y1)); y2 = max(0.0, min(float(h), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        rec = {
            "class_id": int(lab),
            "class_name": f"class_{int(lab)}",
            "label": f"class_{int(lab)}",
            "confidence": score,
            "score": score,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "model": model_name,
        }
        if mask_data is not None and idx < int(mask_data.shape[0]):
            try:
                m = mask_data[idx, 0]
                rec["mask_area_px"] = int((m >= float(mask_threshold)).sum().item())
            except Exception:
                pass
        result.append(rec)
    return result


def _connected_components(mask_bool, prob=None, min_area: int = 64) -> List[Dict[str, Any]]:
    import numpy as np
    mask_bool = np.asarray(mask_bool).astype(bool)
    h, w = mask_bool.shape[:2]
    min_area = max(1, int(min_area))

    # Fast path: OpenCV, if present.
    try:
        import cv2  # type: ignore
        num, labels, stats, _cent = cv2.connectedComponentsWithStats(mask_bool.astype("uint8"), 8)
        out = []
        for i in range(1, int(num)):
            x, y, ww, hh, area = stats[i].tolist()
            if int(area) < min_area:
                continue
            if prob is not None:
                p = float(np.asarray(prob)[labels == i].mean()) if int(area) else 0.0
            else:
                p = 1.0
            out.append({"x1": float(x), "y1": float(y), "x2": float(x + ww), "y2": float(y + hh), "area_px": int(area), "confidence": p})
        return out
    except Exception:
        pass

    # Optional scipy path.
    try:
        from scipy import ndimage  # type: ignore
        labels, num = ndimage.label(mask_bool)
        objs = ndimage.find_objects(labels)
        out = []
        for i, slc in enumerate(objs, start=1):
            if slc is None:
                continue
            yy, xx = slc
            comp = labels[slc] == i
            area = int(comp.sum())
            if area < min_area:
                continue
            p = float(np.asarray(prob)[slc][comp].mean()) if prob is not None else 1.0
            out.append({"x1": float(xx.start), "y1": float(yy.start), "x2": float(xx.stop), "y2": float(yy.stop), "area_px": area, "confidence": p})
        return out
    except Exception:
        pass

    # Pure numpy BFS fallback. Good enough for 1024px tiles if OpenCV/scipy are absent.
    seen = np.zeros((h, w), dtype=bool)
    ys, xs = np.nonzero(mask_bool)
    out = []
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if seen[sy, sx] or not mask_bool[sy, sx]:
            continue
        stack = [(sy, sx)]
        seen[sy, sx] = True
        minx = maxx = sx
        miny = maxy = sy
        area = 0
        vals = []
        while stack:
            y, x = stack.pop()
            area += 1
            minx = min(minx, x); maxx = max(maxx, x)
            miny = min(miny, y); maxy = max(maxy, y)
            if prob is not None:
                vals.append(float(prob[y, x]))
            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if ny == y and nx == x:
                        continue
                    if 0 <= ny < h and 0 <= nx < w and (not seen[ny, nx]) and mask_bool[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        if area >= min_area:
            out.append({"x1": float(minx), "y1": float(miny), "x2": float(maxx + 1), "y2": float(maxy + 1), "area_px": int(area), "confidence": float(sum(vals) / max(1, len(vals))) if vals else 1.0})
    return out


def _unet_segment_pil(
    ws: Any,
    pil_image,
    model_path: str,
    device_name: str,
    threshold: float = 0.50,
    min_area: int = 64,
    in_channels: int = 3,
    out_channels: int = 1,
    base: int = 32,
) -> List[Dict[str, Any]]:
    import torch
    import torch.nn.functional as F
    import numpy as np
    model, device, model_name = _load_unet_model(ws, device_name, model_path, in_channels, out_channels, base)
    x = _pil_to_tensor(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        y = model(x)
    if isinstance(y, (list, tuple)):
        y = y[0]
    if isinstance(y, dict):
        for key in ("out", "logits", "mask", "masks"):
            if key in y:
                y = y[key]
                break
    if y.ndim == 3:
        y = y.unsqueeze(1)
    if y.shape[-2:] != x.shape[-2:]:
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
    if int(y.shape[1]) == 1:
        prob = torch.sigmoid(y[:, 0])[0].detach().cpu().numpy()
        mask = prob >= float(threshold)
        class_id = 0
    else:
        sm = torch.softmax(y, dim=1)[0]
        cls = torch.argmax(sm, dim=0).detach().cpu().numpy()
        prob = torch.max(sm, dim=0).values.detach().cpu().numpy()
        # Treat class 0 as background, everything else as foreground.
        mask = (cls > 0) & (prob >= float(threshold))
        class_id = 1
    comps = _connected_components(mask, prob=prob, min_area=int(min_area))
    result = []
    for comp in comps:
        result.append({
            "class_id": int(class_id),
            "class_name": "segmentation",
            "label": "segmentation",
            "confidence": float(comp.get("confidence", 1.0)),
            "score": float(comp.get("confidence", 1.0)),
            "x1": float(comp["x1"]), "y1": float(comp["y1"]),
            "x2": float(comp["x2"]), "y2": float(comp["y2"]),
            "area_px": int(comp.get("area_px", 0)),
            "model": model_name,
        })
    return result



# -----------------------------------------------------------------------------
# SAM2 prompt-based segmentation helpers
# -----------------------------------------------------------------------------

def _sam_log(ws: Any, msg: str) -> None:
    try:
        if hasattr(ws, "sammsg"):
            ws.sammsg(str(msg))
        else:
            ws.log(str(msg))
    except Exception:
        _log(str(msg))


def _sam2_default_model_path(ws: Any, model_path: str = "") -> str:
    raw = str(model_path or "").strip().strip('"')
    if raw:
        return raw
    try:
        raw = str(_get_var(getattr(ws, "sammodel", None), "") or "").strip().strip('"')
    except Exception:
        raw = ""
    return raw or "sam2_b.pt"


def _sam2_device_name(device_name: str) -> str:
    raw = str(device_name or "cpu").strip().lower()
    aliases = {"gpu": "cuda", "cuda:0": "cuda", "cude": "cuda", "dml": "directml", "direct ml": "directml"}
    raw = aliases.get(raw, raw)
    if raw == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return raw if raw in {"cpu", "cuda", "directml"} else "cpu"


def _load_sam2_model(ws: Any, model_path: str = "", device_name: str = "cpu"):
    path = _sam2_default_model_path(ws, model_path)
    requested = _sam2_device_name(device_name)
    key = ("sam2", path, requested)
    cache = getattr(ws, "_mustatil_sam2_prompt_model_cache", {}) or {}
    if key in cache:
        return cache[key]

    backend = None
    try:
        backend = _get_globals().get("backend")
    except Exception:
        backend = None
    if backend is None:
        try:
            import mustatil_legacy_backend as backend  # type: ignore
        except Exception as exc:
            raise RuntimeError("Could not import Mustatil legacy backend for SAM2 loading: " + str(exc))
    loader = getattr(backend, "load_sam2_model_safe", None)
    if loader is None:
        raise RuntimeError("Mustatil backend has no load_sam2_model_safe(...). SAM2 tab needs the existing Mustatil SAM2 backend.")

    _sam_log(ws, f"SAM2 prompt tab loading model: {path}")
    sam = loader(path, log_fn=lambda m: _sam_log(ws, m))
    active = requested
    device_obj = None
    if hasattr(ws, "_move_sam_model_to_device"):
        try:
            sam, active, device_obj = ws._move_sam_model_to_device(sam, requested)
        except Exception as exc:
            _sam_log(ws, f"SAM2 device setup warning: {exc}; using model default device.")
    else:
        try:
            if requested == "cuda" and hasattr(sam, "to"):
                sam.to("cuda")
            elif requested == "cpu" and hasattr(sam, "to"):
                sam.to("cpu")
        except Exception as exc:
            _sam_log(ws, f"SAM2 .to({requested}) warning: {exc}")
    try:
        ws._sam_active_device = active
        ws._sam_device_obj = device_obj
    except Exception:
        pass
    model_name = "SAM2 " + Path(str(path)).name
    cache[key] = (sam, active, model_name)
    ws._mustatil_sam2_prompt_model_cache = cache
    _sam_log(ws, f"SAM2 prompt tab ready: {model_name} on {active}")
    return cache[key]


def _predict_sam2(ws: Any, sam: Any, image_array, bboxes: List[List[float]], active_device: str):
    if hasattr(ws, "_sam_predict"):
        return ws._sam_predict(sam, image_array, bboxes=bboxes, verbose=False)
    kwargs = {"bboxes": bboxes, "verbose": False}
    if str(active_device).lower() == "cuda":
        kwargs["device"] = "cuda"
    return sam.predict(image_array, **kwargs)


def _poly_area_xy(points: Iterable[Iterable[float]]) -> float:
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) < 3:
        return 0.0
    acc = 0.0
    for i, (x1, y1) in enumerate(pts):
        x2, y2 = pts[(i + 1) % len(pts)]
        acc += x1 * y2 - x2 * y1
    return abs(acc) * 0.5


def _bbox_from_mask_array(mask, threshold: float = 0.50):
    import numpy as np
    arr = np.asarray(mask)
    if arr.ndim > 2:
        arr = arr.squeeze()
    yy, xx = np.nonzero(arr >= float(threshold))
    if len(xx) == 0 or len(yy) == 0:
        return None
    return float(xx.min()), float(yy.min()), float(xx.max() + 1), float(yy.max() + 1), int(len(xx))


def _sam2_segment_pil(ws: Any, pil_image, settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    import numpy as np

    prompt_boxes = [list(map(float, b[:4])) for b in list(settings.get("prompt_boxes") or []) if len(b) >= 4]
    if not prompt_boxes:
        mode = str(settings.get("prompt_mode", "existing_fallback")).lower()
        if mode in {"full", "full_tile", "full image", "full tile only"} or "fallback" in mode:
            w, h = pil_image.size
            prompt_boxes = [[0.0, 0.0, float(max(1, w - 1)), float(max(1, h - 1))]]
    if not prompt_boxes:
        return []

    max_crop = max(128, int(settings.get("max_crop", 1024) or 1024))
    min_area = max(1, int(settings.get("min_area", 64) or 64))
    mask_threshold = float(settings.get("mask_threshold", settings.get("threshold", 0.50)) or 0.50)
    class_id = int(settings.get("sam_class_id", 0) or 0)
    device = str(settings.get("device", "cpu"))
    model_path = str(settings.get("model_path", ""))

    img = pil_image.convert("RGB")
    scale = 1.0
    max_side = max(img.size)
    scaled_boxes = [list(b) for b in prompt_boxes]
    if max_side > max_crop:
        scale = float(max_crop) / float(max_side)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
        scaled_boxes = [[v * scale for v in b] for b in prompt_boxes]

    sam, active, model_name = _load_sam2_model(ws, model_path=model_path, device_name=device)
    res = _predict_sam2(ws, sam, np.asarray(img), bboxes=scaled_boxes, active_device=active)
    if not res:
        return []
    r0 = res[0] if isinstance(res, (list, tuple)) else res
    masks = getattr(r0, "masks", None)
    if masks is None:
        return []

    inv = 1.0 / float(scale or 1.0)
    out: List[Dict[str, Any]] = []
    xy = getattr(masks, "xy", None)
    data = getattr(masks, "data", None)

    if xy is not None:
        try:
            polys = list(xy)
        except Exception:
            polys = []
        for i, poly in enumerate(polys):
            try:
                pts = [(float(x) * inv, float(y) * inv) for x, y in poly]
                if len(pts) < 3:
                    continue
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                area = _poly_area_xy(pts)
                if area < min_area:
                    continue
                out.append({
                    "class_id": class_id,
                    "class_name": "sam2_mask",
                    "label": "sam2_mask",
                    "confidence": 1.0,
                    "score": 1.0,
                    "x1": max(0.0, min(xs)), "y1": max(0.0, min(ys)),
                    "x2": max(xs), "y2": max(ys),
                    "area_px": int(area),
                    "mask_area_px": int(area),
                    "prompt_index": int(i),
                    "model": model_name,
                })
            except Exception:
                pass
        if out:
            return out

    if data is not None:
        try:
            data_cpu = data.detach().cpu().numpy() if hasattr(data, "detach") else np.asarray(data)
        except Exception:
            data_cpu = None
        if data_cpu is not None:
            for i in range(int(data_cpu.shape[0])):
                bb = _bbox_from_mask_array(data_cpu[i], threshold=mask_threshold)
                if bb is None:
                    continue
                x1, y1, x2, y2, area = bb
                area_orig = int(area * inv * inv)
                if area_orig < min_area:
                    continue
                out.append({
                    "class_id": class_id,
                    "class_name": "sam2_mask",
                    "label": "sam2_mask",
                    "confidence": 1.0,
                    "score": 1.0,
                    "x1": x1 * inv, "y1": y1 * inv,
                    "x2": x2 * inv, "y2": y2 * inv,
                    "area_px": area_orig,
                    "mask_area_px": area_orig,
                    "prompt_index": int(i),
                    "model": model_name,
                })
    return out


def _box_iou_xyxy(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in list(a)[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in list(b)[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = aa + bb - inter
    return 0.0 if den <= 0 else float(inter / den)


def _nms_records_xyxy(records: List[Dict[str, Any]], iou_threshold: float = 0.80) -> List[Dict[str, Any]]:
    ordered = sorted(list(records or []), key=lambda r: float(r.get("confidence", r.get("score", 0.0))), reverse=True)
    kept: List[Dict[str, Any]] = []
    for r in ordered:
        box = [r.get("x1", 0), r.get("y1", 0), r.get("x2", 0), r.get("y2", 0)]
        cls = int(r.get("class_id", 0))
        if any(int(k.get("class_id", 0)) == cls and _box_iou_xyxy(box, [k.get("x1", 0), k.get("y1", 0), k.get("x2", 0), k.get("y2", 0)]) >= float(iou_threshold) for k in kept):
            continue
        kept.append(r)
    kept.reverse()
    return kept


def _existing_detection_prompt_records(ws: Any) -> List[Dict[str, Any]]:
    src = []
    try:
        if callable(getattr(ws, "visible", None)):
            src = list(ws.visible() or [])
    except Exception:
        src = []
    if not src:
        try:
            src = list(getattr(ws, "dets", []) or [])
        except Exception:
            src = []
    out: List[Dict[str, Any]] = []
    for d in src:
        try:
            if hasattr(d, "bbox"):
                x1, y1, x2, y2 = d.bbox()
            elif isinstance(d, dict):
                x1, y1, x2, y2 = d.get("x1", d.get("bbox_px_x1")), d.get("y1", d.get("bbox_px_y1")), d.get("x2", d.get("bbox_px_x2")), d.get("y2", d.get("bbox_px_y2"))
            else:
                continue
            out.append({"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)})
        except Exception:
            pass
    return out


def _existing_satellite_prompt_records(ws: Any, z: int, x_min: int, y_min: int) -> List[Dict[str, Any]]:
    src = []
    for attr in ("satellite_detections", "sat_last_records"):
        try:
            vals = list(getattr(ws, attr, []) or [])
            if vals:
                src = vals
                break
        except Exception:
            pass
    out: List[Dict[str, Any]] = []
    for r in src:
        if not isinstance(r, dict):
            continue
        try:
            if "zoom" in r and int(r.get("zoom")) != int(z):
                continue
            if "tile_x_min" in r and int(r.get("tile_x_min")) != int(x_min):
                continue
            if "tile_y_min" in r and int(r.get("tile_y_min")) != int(y_min):
                continue
            x1 = float(r.get("bbox_px_x1")); y1 = float(r.get("bbox_px_y1"))
            x2 = float(r.get("bbox_px_x2")); y2 = float(r.get("bbox_px_y2"))
            out.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
        except Exception:
            pass
    return out


def _local_prompts_from_records(records: List[Dict[str, Any]], off_x: float, off_y: float, width: float, height: float, max_prompts: int = 128) -> List[List[float]]:
    out: List[List[float]] = []
    rx1, ry1, rx2, ry2 = float(off_x), float(off_y), float(off_x + width), float(off_y + height)
    for r in records or []:
        try:
            x1, y1, x2, y2 = float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])
            if x2 < x1: x1, x2 = x2, x1
            if y2 < y1: y1, y2 = y2, y1
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            if not (rx1 <= cx < rx2 and ry1 <= cy < ry2):
                continue
            lx1 = max(0.0, min(float(width), x1 - rx1)); lx2 = max(0.0, min(float(width), x2 - rx1))
            ly1 = max(0.0, min(float(height), y1 - ry1)); ly2 = max(0.0, min(float(height), y2 - ry1))
            if lx2 - lx1 >= 2 and ly2 - ly1 >= 2:
                out.append([lx1, ly1, lx2, ly2])
            if len(out) >= int(max_prompts):
                break
        except Exception:
            pass
    return out


def _sam2_settings_for_crop(settings: Dict[str, Any], prompt_records: List[Dict[str, Any]], x: int, y: int, w: int, h: int) -> Optional[Dict[str, Any]]:
    mode = str(settings.get("prompt_mode", "existing_fallback")).lower()
    local = dict(settings)
    if "full" in mode and "existing" not in mode:
        local["prompt_boxes"] = [[0.0, 0.0, float(max(1, w - 1)), float(max(1, h - 1))]]
        return local
    prompts = _local_prompts_from_records(prompt_records, x, y, w, h, max_prompts=int(settings.get("max_prompts", 128) or 128))
    if not prompts and "fallback" in mode:
        prompts = [[0.0, 0.0, float(max(1, w - 1)), float(max(1, h - 1))]]
    if not prompts:
        return None
    local["prompt_boxes"] = prompts
    return local


def _model_infer_pil(ws: Any, mode: str, pil_image, settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    if mode == "faster":
        return _detect_torchvision_pil(
            ws, "faster", pil_image,
            conf=float(settings.get("conf", 0.25)),
            device_name=str(settings.get("device", "cpu")),
            custom_path=str(settings.get("model_path", "")),
            num_classes=int(settings.get("num_classes", 2)),
        )
    if mode == "mask":
        return _detect_torchvision_pil(
            ws, "mask", pil_image,
            conf=float(settings.get("conf", 0.25)),
            device_name=str(settings.get("device", "cpu")),
            custom_path=str(settings.get("model_path", "")),
            num_classes=int(settings.get("num_classes", 2)),
            mask_threshold=float(settings.get("mask_threshold", 0.50)),
        )
    if mode == "unet":
        return _unet_segment_pil(
            ws, pil_image,
            model_path=str(settings.get("model_path", "")),
            device_name=str(settings.get("device", "cpu")),
            threshold=float(settings.get("threshold", 0.50)),
            min_area=int(settings.get("min_area", 64)),
            in_channels=3,
            out_channels=int(settings.get("num_classes", 1)),
            base=int(settings.get("base", 32)),
        )
    if mode == "sam2":
        return _sam2_segment_pil(ws, pil_image, settings)
    raise RuntimeError(f"Unknown mode: {mode}")


# -----------------------------------------------------------------------------
# Detection image runner
# -----------------------------------------------------------------------------

def _append_dets_to_workspace(ws: Any, records: List[Dict[str, Any]], model_fallback: str = "TorchVision") -> None:
    try:
        from mustatil_legacy_backend import Det
    except Exception:
        g = _get_globals()
        backend = g.get("backend")
        Det = getattr(backend, "Det", None)
        if Det is None:
            raise RuntimeError("Could not import mustatil_legacy_backend.Det")
    dets = []
    for r in records:
        try:
            dets.append(Det(
                0,
                str(r.get("model", model_fallback)),
                int(r.get("class_id", 0)),
                float(r.get("confidence", r.get("score", 0.0))),
                float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]),
            ))
        except Exception:
            pass
    ws.dets = dets


def _run_detection_image(ws: Any, mode: str, settings: Dict[str, Any]) -> None:
    try:
        from PIL import Image
        img_path = str(_get_var(getattr(ws, "image", None), "") or "").strip().strip('"')
        if not img_path or not Path(img_path).exists():
            raise RuntimeError("No Detection image selected.")
        tile = max(128, int(settings.get("tile", _get_var(getattr(ws, "tile", None), 1024)) or 1024))
        overlap = max(0, min(tile - 1, int(settings.get("overlap", _get_var(getattr(ws, "overlap", None), 128)) or 128)))
        step = max(1, tile - overlap)
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        records: List[Dict[str, Any]] = []
        count = 0
        prompt_records: List[Dict[str, Any]] = []
        if mode == "sam2":
            prompt_records = _existing_detection_prompt_records(ws)
            try:
                ws.log(f"SAM2 prompt source boxes available: {len(prompt_records)}")
            except Exception:
                pass
        try:
            ws.log(f"{settings.get('title', mode)} started: {Path(img_path).name} {W}x{H}, tile={tile}, overlap={overlap}")
        except Exception:
            pass
        for y in range(0, H, step):
            for x in range(0, W, step):
                x2 = min(W, x + tile); y2 = min(H, y + tile)
                crop = img.crop((x, y, x2, y2))
                local_settings = settings
                if mode == "sam2":
                    local_settings = _sam2_settings_for_crop(settings, prompt_records, x, y, x2 - x, y2 - y)
                    if local_settings is None:
                        count += 1
                        if x2 >= W:
                            break
                        continue
                local = _model_infer_pil(ws, mode, crop, local_settings)
                for r in local:
                    nr = dict(r)
                    nr["x1"] = x + float(r["x1"]); nr["x2"] = x + float(r["x2"])
                    nr["y1"] = y + float(r["y1"]); nr["y2"] = y + float(r["y2"])
                    records.append(nr)
                count += 1
                if count % 5 == 0:
                    try:
                        ws.log(f"{settings.get('title', mode)} tiles processed: {count}; detections/components={len(records)}")
                    except Exception:
                        pass
                if x2 >= W:
                    break
            if y + tile >= H:
                break
        if mode == "sam2":
            records = _nms_records_xyxy(records, float(settings.get("nms_iou", 0.80) or 0.80))
        _append_dets_to_workspace(ws, records, str(settings.get("title", mode)))
        try:
            ws.mustatil_torchvision_last_records = [dict(r) for r in records]
        except Exception:
            pass
        _redraw(ws)
        try:
            ws.log(f"{settings.get('title', mode)} finished: {len(records)} detections/components")
        except Exception:
            pass
    except Exception as exc:
        _show_error(ws, f"{settings.get('title', mode)} Detection", exc)


# -----------------------------------------------------------------------------
# Satellite runner
# -----------------------------------------------------------------------------

def _sat_helpers(ws: Any) -> Tuple[Any, Any]:
    g = _get_globals()
    sat_tile_bounds_for_bbox = g.get("sat_tile_bounds_for_bbox")
    sat_lonlat_from_world_px = g.get("sat_lonlat_from_world_px")
    if sat_tile_bounds_for_bbox is None or sat_lonlat_from_world_px is None:
        try:
            gg = ws.__class__.satellite_detect_selected.__globals__
            sat_tile_bounds_for_bbox = sat_tile_bounds_for_bbox or gg.get("sat_tile_bounds_for_bbox")
            sat_lonlat_from_world_px = sat_lonlat_from_world_px or gg.get("sat_lonlat_from_world_px")
        except Exception:
            pass
    if sat_tile_bounds_for_bbox is None or sat_lonlat_from_world_px is None:
        raise RuntimeError("Satellite helper functions not found.")
    return sat_tile_bounds_for_bbox, sat_lonlat_from_world_px


def _record_from_local_box(
    r: Dict[str, Any], *,
    z: int, x_min: int, y_min: int, chunk_id: int, chunk_x: int, chunk_y: int,
    cw: int, ch: int, sat_lonlat_from_world_px
) -> Optional[Dict[str, Any]]:
    bx1 = max(0.0, min(float(cw), float(r["x1"]))); bx2 = max(0.0, min(float(cw), float(r["x2"])))
    by1 = max(0.0, min(float(ch), float(r["y1"]))); by2 = max(0.0, min(float(ch), float(r["y2"])))
    if bx2 <= bx1 or by2 <= by1:
        return None
    gx1 = x_min * WEB_TILE_SIZE + chunk_x + bx1
    gy1 = y_min * WEB_TILE_SIZE + chunk_y + by1
    gx2 = x_min * WEB_TILE_SIZE + chunk_x + bx2
    gy2 = y_min * WEB_TILE_SIZE + chunk_y + by2
    lon1, lat1 = sat_lonlat_from_world_px(gx1, gy1, z)
    lon2, lat2 = sat_lonlat_from_world_px(gx2, gy2, z)
    west, east = sorted((float(lon1), float(lon2)))
    south, north = sorted((float(lat1), float(lat2)))
    poly = [(west, north), (east, north), (east, south), (west, south), (west, north)]
    return {
        "class_id": int(r.get("class_id", 0)),
        "class_name": str(r.get("class_name", r.get("label", f"class_{int(r.get('class_id', 0))}"))),
        "confidence": float(r.get("confidence", r.get("score", 0.0))),
        "model": str(r.get("model", "TorchVision/UNet")),
        "model_slot": 1,
        "zoom": int(z),
        "tile_x_min": int(x_min),
        "tile_y_min": int(y_min),
        "chunk_id": int(chunk_id),
        "chunk_px_x": int(chunk_x),
        "chunk_px_y": int(chunk_y),
        "bbox_px_x1": float(chunk_x + bx1),
        "bbox_px_y1": float(chunk_y + by1),
        "bbox_px_x2": float(chunk_x + bx2),
        "bbox_px_y2": float(chunk_y + by2),
        "bbox_lon_min": west,
        "bbox_lat_min": south,
        "bbox_lon_max": east,
        "bbox_lat_max": north,
        "world_px_z": int(z),
        "world_px_x1": float(gx1),
        "world_px_y1": float(gy1),
        "world_px_x2": float(gx2),
        "world_px_y2": float(gy2),
        "polygon_lonlat": poly,
    }


def _run_satellite(ws: Any, mode: str, settings: Dict[str, Any]) -> None:
    try:
        from PIL import Image
        sat_tile_bounds_for_bbox, sat_lonlat_from_world_px = _sat_helpers(ws)
        min_lat, min_lon, max_lat, max_lon = ws._satellite_bbox()
        z = int(_get_var(getattr(ws, "sat_zoom", None), 18))
        x_min, y_min, x_max, y_max = sat_tile_bounds_for_bbox(min_lat, min_lon, max_lat, max_lon, z)
        cols = x_max - x_min + 1; rows = y_max - y_min + 1
        width = cols * WEB_TILE_SIZE; height = rows * WEB_TILE_SIZE
        chunk = max(64, int(settings.get("tile", _get_var(getattr(ws, "tile", None), 1024)) or 1024))
        run_stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1000000000:09d}"
        out_path = ws._satellite_unique_run_output_path(ws._satellite_output_gpkg_path(), run_stamp)
        temp_cache_root = ws._satellite_cache_dir() / "_detection_tmp" / f"tv_unet_z{z}_{run_stamp}_{threading.get_ident()}"
        temp_cache_root.mkdir(parents=True, exist_ok=True)
        old_temp = getattr(ws, "sat_detection_temp_cache_root", None)
        old_thread = getattr(getattr(ws, "_sat_detection_thread_local", None), "cache_root", None)
        ws.sat_detection_temp_cache_root = temp_cache_root
        try:
            ws._sat_detection_thread_local.cache_root = temp_cache_root
        except Exception:
            pass
        records: List[Dict[str, Any]] = []
        prompt_records: List[Dict[str, Any]] = []
        if mode == "sam2":
            prompt_records = _existing_satellite_prompt_records(ws, z, x_min, y_min)
            try:
                ws.log(f"SAM2 satellite prompt source boxes available: {len(prompt_records)}")
            except Exception:
                pass
        try:
            jobs = []
            cid = 0
            for y in range(0, height, chunk):
                for x in range(0, width, chunk):
                    cid += 1
                    jobs.append((cid, x, y, min(chunk, width - x), min(chunk, height - y)))
            total = len(jobs)
            try:
                ws.log(f"{settings.get('title', mode)} Satellite started: z={z}, tiles={cols}x{rows}, chunks={total}")
            except Exception:
                pass
            for cid, x, y, cw, ch in jobs:
                meta = ws._satellite_build_chunk_to_cache(x_min, y_min, z, x, y, cw, ch, cid, temp_cache_root)
                chunk_path = Path(meta["chunk_path"])
                used = list(meta.get("tile_paths") or [])
                im = None
                try:
                    im = Image.open(chunk_path).convert("RGB")
                    local_settings = settings
                    if mode == "sam2":
                        local_settings = _sam2_settings_for_crop(settings, prompt_records, x, y, cw, ch)
                        if local_settings is None:
                            try:
                                ws.log(f"{settings.get('title', mode)} satellite chunk {cid}/{total}: no prompt boxes in chunk")
                            except Exception:
                                pass
                            continue
                    local = _model_infer_pil(ws, mode, im, local_settings)
                    found = 0
                    for r in local:
                        try:
                            ok, _reason = ws._satellite_detection_box_is_valid(float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]), cw, ch)
                            if not ok:
                                continue
                        except Exception:
                            pass
                        rec = _record_from_local_box(r, z=z, x_min=x_min, y_min=y_min, chunk_id=cid, chunk_x=x, chunk_y=y, cw=cw, ch=ch, sat_lonlat_from_world_px=sat_lonlat_from_world_px)
                        if rec is not None:
                            records.append(rec); found += 1
                    try:
                        ws.log(f"{settings.get('title', mode)} satellite chunk {cid}/{total}: found={found}; total={len(records)}")
                    except Exception:
                        pass
                finally:
                    try:
                        if im is not None:
                            im.close()
                    except Exception:
                        pass
                    try:
                        chunk_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    try:
                        ws._satellite_delete_tile_paths(used)
                    except Exception:
                        pass
            try:
                records = ws._satellite_deduplicate_records(records, 0.90)
            except Exception:
                pass
            if mode == "sam2":
                # Pixel-space NMS fallback for SAM2 records, because overlapped chunks may segment the same prompt twice.
                try:
                    tmp = []
                    for rr in records:
                        nr = dict(rr)
                        nr["x1"] = float(rr.get("bbox_px_x1", 0)); nr["y1"] = float(rr.get("bbox_px_y1", 0))
                        nr["x2"] = float(rr.get("bbox_px_x2", 0)); nr["y2"] = float(rr.get("bbox_px_y2", 0))
                        tmp.append(nr)
                    kept = _nms_records_xyxy(tmp, float(settings.get("nms_iou", 0.80) or 0.80))
                    keep_boxes = {(round(k["x1"], 2), round(k["y1"], 2), round(k["x2"], 2), round(k["y2"], 2)) for k in kept}
                    records = [rr for rr in records if (round(float(rr.get("bbox_px_x1", 0)), 2), round(float(rr.get("bbox_px_y1", 0)), 2), round(float(rr.get("bbox_px_x2", 0)), 2), round(float(rr.get("bbox_px_y2", 0)), 2)) in keep_boxes]
                except Exception:
                    pass
            ws.sat_last_x_min = int(x_min); ws.sat_last_y_min = int(y_min); ws.sat_last_z = int(z)
            ws.sat_last_records = [dict(r) for r in records]
            ws.satellite_detections = [dict(r) for r in records]
            if records:
                try:
                    ws._satellite_features_to_file([dict(r) for r in records], out_path)
                    ws.satellite_output_last = str(out_path)
                    ws.log(f"{settings.get('title', mode)} Satellite GeoPackage written: {out_path}")
                except Exception as exc:
                    ws.log(f"Satellite export warning: {exc}")
            else:
                try:
                    ws.log(f"{settings.get('title', mode)} Satellite finished: no detections/components.")
                except Exception:
                    pass
            try:
                ws._satellite_request_map_reload_from_worker(120)
            except Exception:
                pass
            _redraw(ws)
        finally:
            try:
                shutil.rmtree(temp_cache_root, ignore_errors=True)
            except Exception:
                pass
            ws.sat_detection_temp_cache_root = old_temp
            try:
                ws._sat_detection_thread_local.cache_root = old_thread
            except Exception:
                pass
    except Exception as exc:
        _show_error(ws, f"{settings.get('title', mode)} Satellite", exc)


# -----------------------------------------------------------------------------
# UI builders
# -----------------------------------------------------------------------------

def _browse_file(line_edit, caption: str = "Select model file"):
    try:
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(None, caption, "", "Model files (*.pt *.pth *.torchscript *.jit);;All files (*)")
        if path:
            line_edit.setText(path)
    except Exception:
        pass


def _install_deps_button_clicked(ws: Any) -> None:
    import subprocess
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "torch", "torchvision", "numpy", "pillow", "ultralytics"]
    try:
        ws.log("Installing/repairing TorchVision/SAM2 dependencies: " + " ".join(cmd))
    except Exception:
        pass
    subprocess.Popen(cmd)


def _settings_from_widgets(mode: str, widgets: Dict[str, Any]) -> Dict[str, Any]:
    def text(name, default=""):
        try:
            return str(widgets[name].text()).strip()
        except Exception:
            return default
    def value(name, default=0):
        try:
            return widgets[name].value()
        except Exception:
            return default
    def current(name, default="cpu"):
        try:
            return str(widgets[name].currentText()).strip()
        except Exception:
            return default
    title_map = {
        "faster": "Faster R-CNN R50-FPN",
        "mask": "Mask R-CNN R50-FPN",
        "unet": "U-Net Semantic Segmentation",
        "sam2": "SAM2 Box-Prompt Segmentation",
    }
    return {
        "title": title_map.get(mode, mode),
        "device": current("device", "cpu"),
        "model_path": text("model_path", ""),
        "num_classes": int(value("num_classes", 2 if mode in {"faster", "mask"} else 1)),
        "conf": float(value("conf", 0.25)),
        "tile": int(value("tile", 1024)),
        "overlap": int(value("overlap", 128)),
        "mask_threshold": float(value("mask_threshold", 0.50)),
        "threshold": float(value("threshold", 0.50)),
        "min_area": int(value("min_area", 64)),
        "base": int(value("base", 32)),
        "prompt_mode": current("prompt_mode", "existing_fallback"),
        "max_crop": int(value("max_crop", 1024)),
        "sam_class_id": int(value("sam_class_id", value("num_classes", 0))),
        "nms_iou": float(value("nms_iou", 0.80)),
        "max_prompts": int(value("max_prompts", 128)),
    }


def _build_model_tab(ws: Any, tab_kind: str, mode: str):
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton,
        QComboBox, QSpinBox, QDoubleSpinBox
    )
    page = QWidget()
    root = QVBoxLayout(page)
    root.setContentsMargins(8, 8, 8, 8)

    title_map = {
        "faster": "Faster R-CNN ResNet-50-FPN Detection",
        "mask": "Mask R-CNN ResNet-50-FPN Detection",
        "unet": "U-Net Semantic Segmentation",
        "sam2": "SAM2 Box-Prompt Segmentation",
    }
    explain_map = {
        "faster": "Two-stage detector. Outputs boxes. COCO weights can run immediately; custom archaeology weights are recommended.",
        "mask": "Instance segmentation model. This tab uses its boxes and stores mask area metadata. Custom weights are recommended.",
        "unet": "Semantic segmentation. Requires a trained U-Net .pt/.pth/TorchScript. Foreground components are converted to boxes for Mustatil overlay/export.",
    }
    header = QLabel(title_map.get(mode, mode) + "\n" + explain_map.get(mode, ""))
    header.setWordWrap(True)
    root.addWidget(header)

    box = QGroupBox(title_map.get(mode, mode))
    g = QGridLayout(box)
    row = 0

    device = QComboBox(); device.setEditable(True); device.addItems(["cpu", "cuda", "auto"])
    g.addWidget(QLabel("Device"), row, 0); g.addWidget(device, row, 1); row += 1

    model_path = QLineEdit("")
    if mode in {"faster", "mask"}:
        model_path.setPlaceholderText("Optional custom .pth/.pt. Empty = torchvision COCO pretrained")
    elif mode == "sam2":
        model_path.setPlaceholderText("Optional SAM2 .pt. Empty = current Mustatil SAM2 model or sam2_b.pt")
    else:
        model_path.setPlaceholderText("Required trained U-Net .pt/.pth/TorchScript")
    browse = QPushButton("Browse")
    browse.clicked.connect(lambda *_: _browse_file(model_path, "Select model file"))
    g.addWidget(QLabel("Model file"), row, 0); g.addWidget(model_path, row, 1); g.addWidget(browse, row, 2); row += 1

    num_classes = QSpinBox()
    if mode == "sam2":
        num_classes.setRange(0, 10000); num_classes.setValue(0)
        num_label = "Output class id"
    else:
        num_classes.setRange(1, 10000); num_classes.setValue(2 if mode in {"faster", "mask"} else 1)
        num_label = "Num classes for custom head" if mode in {"faster", "mask"} else "U-Net output channels"
    g.addWidget(QLabel(num_label), row, 0); g.addWidget(num_classes, row, 1); row += 1

    conf = QDoubleSpinBox(); conf.setRange(0.001, 1.0); conf.setSingleStep(0.05); conf.setDecimals(3)
    conf.setValue(0.25 if mode in {"faster", "mask"} else 0.50)
    g.addWidget(QLabel("Confidence / threshold"), row, 0); g.addWidget(conf, row, 1); row += 1

    threshold = QDoubleSpinBox(); threshold.setRange(0.001, 1.0); threshold.setSingleStep(0.05); threshold.setDecimals(3); threshold.setValue(0.50)
    mask_threshold = QDoubleSpinBox(); mask_threshold.setRange(0.001, 1.0); mask_threshold.setSingleStep(0.05); mask_threshold.setDecimals(3); mask_threshold.setValue(0.50)
    if mode == "mask":
        g.addWidget(QLabel("Mask threshold metadata"), row, 0); g.addWidget(mask_threshold, row, 1); row += 1
    if mode == "sam2":
        g.addWidget(QLabel("Mask threshold fallback"), row, 0); g.addWidget(mask_threshold, row, 1); row += 1
    if mode == "unet":
        g.addWidget(QLabel("U-Net foreground threshold"), row, 0); g.addWidget(threshold, row, 1); row += 1

    min_area = QSpinBox(); min_area.setRange(1, 10_000_000); min_area.setValue(64)
    base = QSpinBox(); base.setRange(8, 512); base.setValue(32)
    prompt_mode = QComboBox(); prompt_mode.addItems([
        "existing_fallback",
        "existing_only",
        "full_tile_only",
    ])
    max_crop = QSpinBox(); max_crop.setRange(128, 8192); max_crop.setSingleStep(128); max_crop.setValue(1024)
    nms_iou = QDoubleSpinBox(); nms_iou.setRange(0.01, 1.0); nms_iou.setSingleStep(0.05); nms_iou.setDecimals(2); nms_iou.setValue(0.80)
    max_prompts = QSpinBox(); max_prompts.setRange(1, 4096); max_prompts.setValue(128)
    if mode == "unet":
        g.addWidget(QLabel("Minimum component area px"), row, 0); g.addWidget(min_area, row, 1); row += 1
        g.addWidget(QLabel("Simple U-Net base channels"), row, 0); g.addWidget(base, row, 1); row += 1
    if mode == "sam2":
        g.addWidget(QLabel("Prompt mode"), row, 0); g.addWidget(prompt_mode, row, 1); row += 1
        g.addWidget(QLabel("Minimum mask area px"), row, 0); g.addWidget(min_area, row, 1); row += 1
        g.addWidget(QLabel("Max SAM2 image side"), row, 0); g.addWidget(max_crop, row, 1); row += 1
        g.addWidget(QLabel("NMS IoU"), row, 0); g.addWidget(nms_iou, row, 1); row += 1
        g.addWidget(QLabel("Max prompts per tile"), row, 0); g.addWidget(max_prompts, row, 1); row += 1

    tile = QSpinBox(); tile.setRange(64, 8192); tile.setSingleStep(64)
    try:
        tile.setValue(int(_get_var(getattr(ws, "tile", None), 1024) or 1024))
    except Exception:
        tile.setValue(1024)
    overlap = QSpinBox(); overlap.setRange(0, 4096); overlap.setSingleStep(32)
    try:
        overlap.setValue(int(_get_var(getattr(ws, "overlap", None), 128) or 128))
    except Exception:
        overlap.setValue(128)
    g.addWidget(QLabel("Tile size"), row, 0); g.addWidget(tile, row, 1); row += 1
    g.addWidget(QLabel("Overlap"), row, 0); g.addWidget(overlap, row, 1); row += 1

    run_btn = QPushButton("Run on Detection image" if tab_kind == "detection" else "Run on selected satellite map")
    status = QLabel("Idle")
    status.setWordWrap(True)
    g.addWidget(run_btn, row, 0, 1, 3); row += 1
    g.addWidget(status, row, 0, 1, 3); row += 1
    root.addWidget(box)

    dep_box = QGroupBox("Dependencies")
    dg = QGridLayout(dep_box)
    dep_label = QLabel("Needs torch + torchvision + numpy + pillow. SAM2 also uses Mustatil's SAM2/Ultralytics backend. Use this only if imports fail.")
    dep_label.setWordWrap(True)
    dep_btn = QPushButton("Install / repair TorchVision/SAM2 deps")
    dep_btn.clicked.connect(lambda *_: _install_deps_button_clicked(ws))
    dg.addWidget(dep_label, 0, 0, 1, 2); dg.addWidget(dep_btn, 1, 0, 1, 2)
    root.addWidget(dep_box)
    root.addStretch(1)

    widgets = {
        "device": device, "model_path": model_path, "num_classes": num_classes, "conf": conf,
        "threshold": threshold, "mask_threshold": mask_threshold, "tile": tile,
        "overlap": overlap, "min_area": min_area, "base": base,
        "prompt_mode": prompt_mode, "max_crop": max_crop, "nms_iou": nms_iou,
        "max_prompts": max_prompts, "sam_class_id": num_classes,
    }

    def run_clicked():
        settings = _settings_from_widgets(mode, widgets)
        try:
            status.setText("Starting " + settings["title"] + "...")
            ws.log("Starting " + settings["title"] + "...")
        except Exception:
            pass
        if tab_kind == "detection":
            _run_as_task(ws, settings["title"], lambda: _run_detection_image(ws, mode, settings))
        else:
            _run_as_task(ws, settings["title"] + " Satellite", lambda: _run_satellite(ws, mode, settings), allow_parallel=True)

    run_btn.clicked.connect(run_clicked)
    return page


def _tab_names(tw: Any) -> List[str]:
    names = []
    try:
        for i in range(tw.count()):
            names.append(str(tw.tabText(i)).strip().lower())
    except Exception:
        pass
    return names


def _ensure_extra_tabs(inner: Any, ws: Any, tab_kind: str) -> bool:
    changed = False
    names = _tab_names(inner)
    extras = [
        ("faster r-cnn", "Faster R-CNN", lambda: _build_model_tab(ws, tab_kind, "faster")),
        ("mask r-cnn", "Mask R-CNN", lambda: _build_model_tab(ws, tab_kind, "mask")),
        ("u-net segmentation", "U-Net Segmentation", lambda: _build_model_tab(ws, tab_kind, "unet")),
        ("sam2", "SAM2", lambda: _build_model_tab(ws, tab_kind, "sam2")),
    ]
    for needle, label, builder in extras:
        if not any(needle in n for n in names):
            try:
                inner.addTab(builder(), label)
                changed = True
                names = _tab_names(inner)
            except Exception as exc:
                _log(f"Could not add {label}: {exc}")
    return changed


def _wrap_or_extend_page(page: Any, tab_kind: str) -> bool:
    ws = _workspace_from_widget(page)
    if ws is None:
        return False
    try:
        from PySide6.QtWidgets import QSplitter, QTabWidget

        splitter = None
        for sp in page.findChildren(QSplitter):
            try:
                if sp.count() >= 2:
                    splitter = sp
                    break
            except Exception:
                pass
        if splitter is None:
            return False

        left = splitter.widget(0)
        if left is None:
            return False

        # If another model plugin already wrapped the left panel, just extend its inner tabs.
        if isinstance(left, QTabWidget):
            changed = _ensure_extra_tabs(left, ws, tab_kind)
            if changed:
                try:
                    ws.log(f"Faster R-CNN / Mask R-CNN / U-Net / SAM2 tabs added to {tab_kind}. Right preview unchanged.")
                except Exception:
                    pass
            _PATCHED_PAGES.add(id(page))
            return changed

        if id(page) in _PATCHED_PAGES:
            return False

        sizes = []
        try:
            sizes = splitter.sizes()
        except Exception:
            pass

        inner = QTabWidget()
        inner.setObjectName("MustatilTorchVisionLeftTabs")
        returned = None
        try:
            returned = splitter.replaceWidget(0, inner)
        except Exception:
            try:
                left.setParent(None)
                splitter.insertWidget(0, inner)
                returned = left
            except Exception:
                returned = left
        if returned is None:
            returned = left
        inner.addTab(returned, "YOLO / Original")
        _ensure_extra_tabs(inner, ws, tab_kind)
        try:
            if sizes and len(sizes) >= 2:
                splitter.setSizes(sizes)
        except Exception:
            pass
        _PATCHED_PAGES.add(id(page))
        try:
            ws.log(f"Faster R-CNN / Mask R-CNN / U-Net / SAM2 tabs added to {tab_kind}. Right preview unchanged.")
        except Exception:
            pass
        return True
    except Exception as exc:
        _log("wrap/extend page failed: " + str(exc))
        traceback.print_exc()
        return False


def _scan_tabs(root=None) -> None:
    try:
        from PySide6.QtWidgets import QApplication, QTabWidget
        widgets = []
        if root is not None:
            try:
                widgets += root.findChildren(QTabWidget)
            except Exception:
                pass
            try:
                if isinstance(root, QTabWidget):
                    widgets.append(root)
            except Exception:
                pass
        app = QApplication.instance()
        if app is not None:
            for w in app.allWidgets():
                try:
                    if isinstance(w, QTabWidget):
                        widgets.append(w)
                except Exception:
                    pass
        seen = set()
        for tw in widgets:
            if id(tw) in seen:
                continue
            seen.add(id(tw))
            try:
                count = tw.count()
            except Exception:
                continue
            for i in range(count):
                label = str(tw.tabText(i) or "").strip().lower()
                page = tw.widget(i)
                if label == "detection":
                    _wrap_or_extend_page(page, "detection")
                elif label == "satellite detection":
                    _wrap_or_extend_page(page, "satellite")
    except Exception as exc:
        _log("scan tabs failed: " + str(exc))


def _install_hook() -> None:
    global _PATCHED_QTAB, _ORIG_ADD, _ORIG_INSERT, _SCAN_TIMER
    if _PATCHED_QTAB:
        return
    try:
        from PySide6.QtWidgets import QTabWidget
        from PySide6.QtCore import QTimer
    except Exception as exc:
        _log("Qt unavailable: " + str(exc))
        return

    _ORIG_ADD = QTabWidget.addTab
    _ORIG_INSERT = QTabWidget.insertTab

    def after(page, label):
        try:
            norm = str(label or "").strip().lower()
            if norm == "detection":
                QTimer.singleShot(250, lambda p=page: _wrap_or_extend_page(p, "detection"))
                QTimer.singleShot(1500, lambda p=page: _wrap_or_extend_page(p, "detection"))
            elif norm == "satellite detection":
                QTimer.singleShot(250, lambda p=page: _wrap_or_extend_page(p, "satellite"))
                QTimer.singleShot(1500, lambda p=page: _wrap_or_extend_page(p, "satellite"))
        except Exception as exc:
            _log("after hook warning: " + str(exc))

    def addTab_patched(self, page, *args, **kwargs):
        res = _ORIG_ADD(self, page, *args, **kwargs)
        label = ""
        for a in reversed(args):
            if isinstance(a, str):
                label = a; break
        if not label:
            try:
                label = self.tabText(int(res))
            except Exception:
                pass
        after(page, label)
        return res

    def insertTab_patched(self, index, page, *args, **kwargs):
        res = _ORIG_INSERT(self, index, page, *args, **kwargs)
        label = ""
        for a in reversed(args):
            if isinstance(a, str):
                label = a; break
        if not label:
            try:
                label = self.tabText(int(res))
            except Exception:
                pass
        after(page, label)
        return res

    QTabWidget.addTab = addTab_patched
    QTabWidget.insertTab = insertTab_patched
    _PATCHED_QTAB = True
    try:
        _SCAN_TIMER = QTimer()
        _SCAN_TIMER.setInterval(1800)
        _SCAN_TIMER.timeout.connect(lambda: _scan_tabs())
        _SCAN_TIMER.start()
        QTimer.singleShot(300, lambda: _scan_tabs())
        QTimer.singleShot(2500, lambda: _scan_tabs())
    except Exception:
        pass
    _log("hook installed")


def mustatil_plugin_init():
    _install_hook()
    _scan_tabs()


def register_plugin(app=None, main_window=None):
    _install_hook()
    _scan_tabs(main_window or app)
    return True


def init_plugin(app=None, main_window=None):
    return register_plugin(app, main_window)


def load_plugin(app=None, main_window=None):
    return register_plugin(app, main_window)


try:
    _install_hook()
except Exception:
    pass
