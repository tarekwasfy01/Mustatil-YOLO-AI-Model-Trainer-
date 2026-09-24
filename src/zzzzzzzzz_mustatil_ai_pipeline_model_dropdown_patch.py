#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mustatil add-on plugin: Backbone selector + auto-fix for Faster R-CNN / Mask R-CNN tabs.

Drop this file into Mustatil's mustatil_plugins folder together with:
  zzzz_mustatil_torchvision_frcnn_maskrcnn_unet_sam2_tabs.py

What this patch does
--------------------
- Replaces only the Faster R-CNN and Mask R-CNN tab builder from the TorchVision tabs plugin.
- Adds a Backbone dropdown to both tabs.
- Adds a Custom Python Backbone file chooser.
- Infers the internal class count from custom .pth/.pt checkpoints automatically.
- If the selected backbone is wrong for the checkpoint, it scores common candidate backbones
  and automatically loads the best matching one.
- Keeps U-Net and SAM2 tabs unchanged.

Custom backbone contract
------------------------
A custom Python backbone file must expose:

    def build_backbone(pretrained=False, trainable_layers=3, out_channels=256):
        ...
        backbone.out_channels = 256
        return backbone

The returned module must be compatible with torchvision.models.detection.FasterRCNN
or MaskRCNN, i.e. it must return a Tensor or an OrderedDict of feature maps and expose
an integer .out_channels attribute.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_TARGET = None
_ORIG_BUILD_MODEL_TAB = None
_ORIG_MODEL_INFER_PIL = None
_ORIG_SETTINGS_FROM_WIDGETS = None
_PATCHED = False

_BACKBONE_ITEMS: List[Tuple[str, str]] = [
    ("Auto / infer from checkpoint", "auto"),
    ("ResNet50-FPN", "resnet50_fpn"),
    ("ResNet50-FPN v2", "resnet50_fpn_v2"),
    ("ResNet101-FPN", "resnet101_fpn"),
    ("ResNet34-FPN", "resnet34_fpn"),
    ("ResNet18-FPN", "resnet18_fpn"),
    ("ResNeXt50-32x4d-FPN", "resnext50_32x4d_fpn"),
    ("Wide ResNet50-2-FPN", "wide_resnet50_2_fpn"),
    ("MobileNetV3-Large-FPN", "mobilenet_v3_large_fpn"),
    ("MobileNetV3-Large-320-FPN", "mobilenet_v3_large_320_fpn"),
    ("Custom Python Backbone", "custom"),
]

_RESNET_BACKBONE_NAMES = {
    "resnet18_fpn": "resnet18",
    "resnet34_fpn": "resnet34",
    "resnet50_fpn": "resnet50",
    "resnet101_fpn": "resnet101",
    "resnext50_32x4d_fpn": "resnext50_32x4d",
    "wide_resnet50_2_fpn": "wide_resnet50_2",
}

_FASTER_ALLOWED = {
    "resnet18_fpn", "resnet34_fpn", "resnet50_fpn", "resnet50_fpn_v2", "resnet101_fpn",
    "resnext50_32x4d_fpn", "wide_resnet50_2_fpn", "mobilenet_v3_large_fpn",
    "mobilenet_v3_large_320_fpn", "custom",
}

_MASK_ALLOWED = {
    "resnet18_fpn", "resnet34_fpn", "resnet50_fpn", "resnet50_fpn_v2", "resnet101_fpn",
    "resnext50_32x4d_fpn", "wide_resnet50_2_fpn", "custom",
}

_AUTO_CANDIDATES_FAST = [
    "resnet50_fpn", "resnet50_fpn_v2", "resnet101_fpn", "resnet34_fpn", "resnet18_fpn",
    "resnext50_32x4d_fpn", "wide_resnet50_2_fpn", "mobilenet_v3_large_fpn", "mobilenet_v3_large_320_fpn",
]
_AUTO_CANDIDATES_MASK = [
    "resnet50_fpn", "resnet50_fpn_v2", "resnet101_fpn", "resnet34_fpn", "resnet18_fpn",
    "resnext50_32x4d_fpn", "wide_resnet50_2_fpn",
]


def _log(msg: str) -> None:
    try:
        print("[Mustatil R-CNN Backbone Patch] " + str(msg), flush=True)
    except Exception:
        pass


def _ws_log(ws: Any, msg: str) -> None:
    text = str(msg)
    try:
        ws.log(text)
    except Exception:
        _log(text)


def _find_target_module() -> Optional[Any]:
    """Find the already-loaded TorchVision Faster/Mask/U-Net/SAM2 tabs plugin."""
    best = None
    for name, mod in list(sys.modules.items()):
        try:
            if not hasattr(mod, "_build_model_tab"):
                continue
            if not hasattr(mod, "_run_detection_image") or not hasattr(mod, "_run_satellite"):
                continue
            if not hasattr(mod, "_load_detection_model"):
                continue
            text = (str(name) + " " + str(getattr(mod, "__file__", ""))).lower()
            if "torchvision" in text and ("frcnn" in text or "rcnn" in text):
                best = mod
                break
        except Exception:
            continue
    return best


def _safe_path(path: str) -> str:
    return str(path or "").strip().strip('"')


def _normalize_backbone_name(text: str) -> str:
    raw = str(text or "auto").strip().lower()
    for label, key in _BACKBONE_ITEMS:
        if raw == label.lower() or raw == key.lower():
            return key
    compact = raw.replace("-", "_").replace(" ", "_")
    aliases = {
        "r50": "resnet50_fpn",
        "r50_fpn": "resnet50_fpn",
        "resnet_50_fpn": "resnet50_fpn",
        "resnet50": "resnet50_fpn",
        "resnet50_v2": "resnet50_fpn_v2",
        "resnet50_fpn_v2": "resnet50_fpn_v2",
        "r101": "resnet101_fpn",
        "resnet101": "resnet101_fpn",
        "resnet_101_fpn": "resnet101_fpn",
        "r34": "resnet34_fpn",
        "resnet34": "resnet34_fpn",
        "r18": "resnet18_fpn",
        "resnet18": "resnet18_fpn",
        "resnext": "resnext50_32x4d_fpn",
        "resnext50": "resnext50_32x4d_fpn",
        "wide_resnet50": "wide_resnet50_2_fpn",
        "mobilenet": "mobilenet_v3_large_fpn",
        "mobilenetv3": "mobilenet_v3_large_fpn",
        "mobilenet_v3_large": "mobilenet_v3_large_fpn",
        "mobilenet_v3_large_320": "mobilenet_v3_large_320_fpn",
        "custom_python_backbone": "custom",
        "custom_backbone": "custom",
    }
    return aliases.get(compact, "auto")


def _label_for_backbone(key: str) -> str:
    key = _normalize_backbone_name(key)
    for label, val in _BACKBONE_ITEMS:
        if val == key:
            return label
    return str(key)


def _class_count_from_state_dict(state: Dict[str, Any]) -> Optional[int]:
    """Return internal torchvision detection num_classes including background."""
    if not isinstance(state, dict):
        return None
    candidates = [
        "roi_heads.box_predictor.cls_score.weight",
        "module.roi_heads.box_predictor.cls_score.weight",
        "model.roi_heads.box_predictor.cls_score.weight",
        "roi_heads.mask_predictor.mask_fcn_logits.weight",
        "module.roi_heads.mask_predictor.mask_fcn_logits.weight",
        "model.roi_heads.mask_predictor.mask_fcn_logits.weight",
    ]
    for k in candidates:
        v = state.get(k)
        if hasattr(v, "shape") and len(tuple(v.shape)) >= 1:
            n = int(v.shape[0])
            if n >= 1:
                return n
    # Last-resort fuzzy search.
    for k, v in state.items():
        kk = str(k)
        if kk.endswith("roi_heads.box_predictor.cls_score.weight") and hasattr(v, "shape"):
            n = int(v.shape[0])
            if n >= 1:
                return n
    return None


def _classes_from_checkpoint_dict(ckpt: Any) -> Optional[List[str]]:
    if not isinstance(ckpt, dict):
        return None
    for key in ("classes", "class_names", "labels"):
        val = ckpt.get(key)
        if isinstance(val, (list, tuple)) and val:
            return [str(x) for x in val]
    meta = ckpt.get("metadata")
    if isinstance(meta, dict):
        for key in ("classes", "class_names", "labels"):
            val = meta.get(key)
            if isinstance(val, (list, tuple)) and val:
                return [str(x) for x in val]
    return None


def _extract_state_and_meta(path: str, map_location: str = "cpu") -> Tuple[Any, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Load checkpoint once. Returns (raw_ckpt, state_dict_or_none, metadata_or_none)."""
    p = _safe_path(path)
    if not p:
        return None, None, None
    import torch
    ckpt = torch.load(p, map_location=map_location)
    meta = None
    if isinstance(ckpt, dict):
        m = ckpt.get("metadata")
        if isinstance(m, dict):
            meta = dict(m)
        else:
            meta = {}
        for top_key in ("backbone", "backbone_name", "selected_backbone", "model_kind", "classes", "class_names"):
            if top_key in ckpt and top_key not in meta:
                meta[top_key] = ckpt.get(top_key)
        for key in ("model_state_dict", "state_dict", "model"):
            val = ckpt.get(key)
            if isinstance(val, dict):
                return ckpt, _strip_state_dict_prefix_local(val), meta
        # It may already be a plain state_dict.
        tensor_like = any(hasattr(v, "shape") for v in ckpt.values())
        if tensor_like:
            return ckpt, _strip_state_dict_prefix_local(ckpt), meta
    return ckpt, None, meta


def _strip_state_dict_prefix_local(state: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in state.items():
        kk = str(k)
        for pref in ("module.", "model.", "net."):
            if kk.startswith(pref):
                kk = kk[len(pref):]
        out[kk] = v
    return out


def _infer_backbone_from_metadata(meta: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(meta, dict):
        return None
    text_parts = []
    for key in ("backbone", "backbone_name", "selected_backbone", "model_kind", "architecture", "arch"):
        val = meta.get(key)
        if val:
            text_parts.append(str(val))
    text = " ".join(text_parts).lower()
    if not text:
        return None
    checks = [
        ("mobilenet_v3_large_320", "mobilenet_v3_large_320_fpn"),
        ("mobilenet_v3_large", "mobilenet_v3_large_fpn"),
        ("mobilenet", "mobilenet_v3_large_fpn"),
        ("resnet101", "resnet101_fpn"),
        ("resnet34", "resnet34_fpn"),
        ("resnet18", "resnet18_fpn"),
        ("resnet50_fpn_v2", "resnet50_fpn_v2"),
        ("resnet50-v2", "resnet50_fpn_v2"),
        ("resnet50", "resnet50_fpn"),
        ("resnext50", "resnext50_32x4d_fpn"),
        ("resnext", "resnext50_32x4d_fpn"),
        ("wide_resnet50", "wide_resnet50_2_fpn"),
        ("wide-resnet50", "wide_resnet50_2_fpn"),
        ("custom", "custom"),
    ]
    for needle, key in checks:
        if needle in text:
            return key
    return None


def _infer_backbone_from_state_shapes(state: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(state, dict):
        return None
    keys = set(str(k) for k in state.keys())
    key_text = "\n".join(list(keys)[:2000]).lower()
    if "backbone.body.0.0.weight" in key_text or "features.0.0.weight" in key_text:
        return "mobilenet_v3_large_fpn"
    # ResNet family shape/key heuristics.
    if any("backbone.body.layer4.2.conv3.weight" in k for k in keys):
        if any("backbone.body.layer3.22." in k for k in keys):
            return "resnet101_fpn"
        # ResNeXt and Wide ResNet are hard to distinguish reliably from key names alone;
        # shape scoring below will still correct them.
        return "resnet50_fpn"
    if any("backbone.body.layer4.1.conv2.weight" in k for k in keys):
        if any("backbone.body.layer3.5." in k for k in keys):
            return "resnet34_fpn"
        return "resnet18_fpn"
    return None


def _load_custom_backbone_from_py(path: str, pretrained: bool = False, trainable_layers: int = 3, out_channels: int = 256):
    p = Path(_safe_path(path))
    if not p.exists():
        raise RuntimeError("Custom Python Backbone selected, but the .py file does not exist.")
    mod_name = "mustatil_user_custom_backbone_" + str(abs(hash(str(p))))
    spec = importlib.util.spec_from_file_location(mod_name, str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import custom backbone file: " + str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    fn = getattr(mod, "build_backbone", None)
    if not callable(fn):
        raise RuntimeError("Custom backbone file must define build_backbone(...).")
    try:
        backbone = fn(pretrained=bool(pretrained), trainable_layers=int(trainable_layers), out_channels=int(out_channels))
    except TypeError:
        try:
            backbone = fn(pretrained=bool(pretrained), trainable_layers=int(trainable_layers))
        except TypeError:
            backbone = fn()
    if not hasattr(backbone, "out_channels"):
        try:
            backbone.out_channels = int(out_channels)
        except Exception:
            pass
    if not hasattr(backbone, "out_channels"):
        raise RuntimeError("Custom backbone must expose backbone.out_channels.")
    return backbone


def _build_model_for_backbone(kind: str, backbone_key: str, num_classes: int, *, pretrained_coco: bool = False,
                              custom_backbone_path: str = "", trainable_layers: int = 3):
    """Build a torchvision FasterRCNN/MaskRCNN model for a specific backbone key."""
    import torchvision
    from torchvision.models.detection import FasterRCNN, MaskRCNN

    kind = str(kind).lower().strip()
    backbone_key = _normalize_backbone_name(backbone_key)
    num_classes = max(1, int(num_classes or 2))

    if kind == "mask" and backbone_key not in _MASK_ALLOWED:
        raise RuntimeError(f"Backbone {_label_for_backbone(backbone_key)} is not supported for Mask R-CNN in this patch.")
    if kind == "faster" and backbone_key not in _FASTER_ALLOWED:
        raise RuntimeError(f"Backbone {_label_for_backbone(backbone_key)} is not supported for Faster R-CNN in this patch.")

    # Built-in COCO models only when no custom checkpoint is used.
    if pretrained_coco and backbone_key == "resnet50_fpn":
        if kind == "faster":
            try:
                from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
                return torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT), "Faster R-CNN ResNet50-FPN COCO"
            except Exception:
                return torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True), "Faster R-CNN ResNet50-FPN COCO"
        try:
            from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
            return torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT), "Mask R-CNN ResNet50-FPN COCO"
        except Exception:
            return torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True), "Mask R-CNN ResNet50-FPN COCO"

    if pretrained_coco and kind == "faster" and backbone_key == "resnet50_fpn_v2":
        try:
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
            return torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT), "Faster R-CNN ResNet50-FPN v2 COCO"
        except Exception:
            pass

    if pretrained_coco and kind == "mask" and backbone_key == "resnet50_fpn_v2":
        try:
            from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
            return torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT), "Mask R-CNN ResNet50-FPN v2 COCO"
        except Exception:
            pass

    if kind == "faster" and backbone_key == "mobilenet_v3_large_fpn":
        try:
            if pretrained_coco:
                from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights
                return torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT), "Faster R-CNN MobileNetV3-Large-FPN COCO"
        except Exception:
            pass
        try:
            model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None, num_classes=num_classes)
        except TypeError:
            model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=False, pretrained_backbone=False, num_classes=num_classes)
        return model, f"Faster R-CNN MobileNetV3-Large-FPN classes={num_classes}"

    if kind == "faster" and backbone_key == "mobilenet_v3_large_320_fpn":
        try:
            if pretrained_coco:
                from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
                return torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT), "Faster R-CNN MobileNetV3-Large-320-FPN COCO"
        except Exception:
            pass
        try:
            model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None, num_classes=num_classes)
        except TypeError:
            model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=False, pretrained_backbone=False, num_classes=num_classes)
        return model, f"Faster R-CNN MobileNetV3-Large-320-FPN classes={num_classes}"

    if backbone_key == "custom":
        backbone = _load_custom_backbone_from_py(custom_backbone_path, pretrained=False, trainable_layers=trainable_layers)
        model = FasterRCNN(backbone, num_classes=num_classes) if kind == "faster" else MaskRCNN(backbone, num_classes=num_classes)
        return model, f"{kind.title()} R-CNN CustomBackbone({Path(custom_backbone_path).name}) classes={num_classes}"

    if backbone_key == "resnet50_fpn_v2":
        if kind == "faster":
            try:
                model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, num_classes=num_classes)
            except TypeError:
                model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=False, pretrained_backbone=False, num_classes=num_classes)
            return model, f"Faster R-CNN ResNet50-FPN v2 classes={num_classes}"
        try:
            model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, num_classes=num_classes)
        except TypeError:
            model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained=False, pretrained_backbone=False, num_classes=num_classes)
        return model, f"Mask R-CNN ResNet50-FPN v2 classes={num_classes}"

    # Generic ResNet/ResNeXt/WideResNet FPN backbone.
    if backbone_key in _RESNET_BACKBONE_NAMES:
        backbone_name = _RESNET_BACKBONE_NAMES[backbone_key]
        try:
            from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
            try:
                backbone = resnet_fpn_backbone(backbone_name=backbone_name, weights=None, trainable_layers=int(trainable_layers))
            except TypeError:
                backbone = resnet_fpn_backbone(backbone_name, pretrained=False, trainable_layers=int(trainable_layers))
        except Exception as exc:
            raise RuntimeError(f"Could not build {backbone_name}-FPN backbone: {exc}")
        model = FasterRCNN(backbone, num_classes=num_classes) if kind == "faster" else MaskRCNN(backbone, num_classes=num_classes)
        return model, f"{('Faster' if kind == 'faster' else 'Mask')} R-CNN {backbone_name}-FPN classes={num_classes}"

    raise RuntimeError("Unsupported backbone: " + str(backbone_key))


def _state_match_score(model: Any, state: Dict[str, Any]) -> Tuple[int, int, int]:
    """Return (matching_tensors, state_tensors_seen, total_model_tensors)."""
    if not isinstance(state, dict):
        return 0, 0, 0
    cur = model.state_dict()
    match = 0
    seen = 0
    for k, v in state.items():
        if not hasattr(v, "shape"):
            continue
        seen += 1
        if k in cur:
            try:
                if tuple(cur[k].shape) == tuple(v.shape):
                    match += 1
            except Exception:
                pass
    return match, seen, len(cur)


def _candidate_list(kind: str, selected: str, inferred: Optional[str], custom_py: str, auto_fix: bool) -> List[str]:
    selected = _normalize_backbone_name(selected)
    allowed = _FASTER_ALLOWED if kind == "faster" else _MASK_ALLOWED
    base = _AUTO_CANDIDATES_FAST if kind == "faster" else _AUTO_CANDIDATES_MASK
    out: List[str] = []
    for key in (inferred, selected):
        if key and key != "auto" and key in allowed and key not in out:
            out.append(key)
    if selected == "custom" and _safe_path(custom_py):
        out.insert(0, "custom") if "custom" not in out else None
    if auto_fix:
        for key in base:
            if key in allowed and key not in out:
                out.append(key)
    if not out:
        out.append("resnet50_fpn")
    return out


def _load_state_into_model(model: Any, state: Dict[str, Any], ws: Any, label: str) -> None:
    if not isinstance(state, dict):
        return
    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
        _ws_log(ws, f"Loaded {label}: missing={len(missing)}, unexpected={len(unexpected)}")
        return
    except Exception as exc:
        _ws_log(ws, f"Direct load warning for {label}: {exc}; filtering matching tensors.")
    cur = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state.items():
        try:
            if k in cur and tuple(cur[k].shape) == tuple(v.shape):
                filtered[k] = v
            else:
                skipped.append(k)
        except Exception:
            skipped.append(k)
    cur.update(filtered)
    model.load_state_dict(cur, strict=False)
    _ws_log(ws, f"Partially loaded {label}: {len(filtered)} tensors loaded, {len(skipped)} skipped.")


def _load_detection_model_v2(ws: Any, kind: str, device_name: str, custom_path: str = "", num_classes: int = 2,
                             backbone: str = "auto", custom_backbone_path: str = "",
                             auto_class_count: bool = True, auto_fix_backbone: bool = True):
    import torch
    target = _TARGET
    if target is None:
        raise RuntimeError("Backbone patch target module is not available.")
    device = target._resolve_torch_device(device_name) if hasattr(target, "_resolve_torch_device") else torch.device("cuda" if str(device_name).lower() in {"cuda", "auto"} and torch.cuda.is_available() else "cpu")
    kind = "mask" if str(kind).lower().startswith("mask") else "faster"
    custom_path = _safe_path(custom_path)
    selected = _normalize_backbone_name(backbone)
    custom_py = _safe_path(custom_backbone_path)

    # Make cache key include all architecture-affecting inputs.
    key = ("backbone_v2", kind, str(device), custom_path, selected, custom_py, int(num_classes or 2), bool(auto_class_count), bool(auto_fix_backbone))
    cache = getattr(ws, "_mustatil_torchvision_model_cache", {}) or {}
    if key in cache:
        return cache[key]

    use_custom_checkpoint = bool(custom_path)
    ckpt = None
    state = None
    meta = None
    inferred_backbone = None
    inferred_classes = None
    classes_list = None

    if use_custom_checkpoint:
        # TorchScript models already contain their architecture. Try that first.
        try:
            scripted = torch.jit.load(custom_path, map_location=device)
            scripted.to(device)
            scripted.eval()
            model_name = f"TorchScript {Path(custom_path).name}"
            cache[key] = (scripted, device, model_name)
            ws._mustatil_torchvision_model_cache = cache
            _ws_log(ws, f"Loaded {model_name} on {device}; class count/backbone are embedded in TorchScript.")
            return cache[key]
        except Exception:
            pass

        ckpt, state, meta = _extract_state_and_meta(custom_path, map_location="cpu")
        if not isinstance(state, dict):
            raise RuntimeError("Custom model file is not TorchScript and does not contain a compatible state_dict.")
        inferred_classes = _class_count_from_state_dict(state)
        classes_list = _classes_from_checkpoint_dict(ckpt)
        inferred_backbone = _infer_backbone_from_metadata(meta) or _infer_backbone_from_state_shapes(state)
        if classes_list and auto_class_count:
            # Most Mustatil R-CNN checkpoints store user classes without background.
            inferred_classes = max(int(inferred_classes or 0), len(classes_list) + 1)
        if inferred_classes and auto_class_count:
            num_classes = int(inferred_classes)
            _ws_log(ws, f"Auto class count from checkpoint: internal classes={num_classes} (includes background).")
        if inferred_backbone:
            _ws_log(ws, f"Backbone hint from checkpoint: {_label_for_backbone(inferred_backbone)}")
    else:
        # Empty model path means COCO/default model. Auto means ResNet50-FPN because that is the safe built-in default.
        if selected == "auto":
            selected = "resnet50_fpn"
        if kind == "mask" and selected not in _MASK_ALLOWED:
            _ws_log(ws, f"{_label_for_backbone(selected)} is not a safe Mask R-CNN choice here; using ResNet50-FPN.")
            selected = "resnet50_fpn"
        if kind == "faster" and selected not in _FASTER_ALLOWED:
            _ws_log(ws, f"{_label_for_backbone(selected)} is not a safe Faster R-CNN choice here; using ResNet50-FPN.")
            selected = "resnet50_fpn"
        model, model_name = _build_model_for_backbone(kind, selected, int(num_classes or 2), pretrained_coco=True, custom_backbone_path=custom_py)
        model.to(device); model.eval()
        cache[key] = (model, device, model_name)
        ws._mustatil_torchvision_model_cache = cache
        _ws_log(ws, f"Loaded {model_name} on {device}")
        return cache[key]

    # Custom state_dict path: build best matching architecture.
    if selected == "custom" and not custom_py:
        _ws_log(ws, "Custom Python Backbone selected but no .py file was given; trying automatic checkpoint inference instead.")
        selected = inferred_backbone or "auto"

    if kind == "mask" and selected not in _MASK_ALLOWED and selected != "auto":
        _ws_log(ws, f"{_label_for_backbone(selected)} is not supported for Mask R-CNN in this patch; auto-fix is enabled.")
        selected = inferred_backbone or "resnet50_fpn"
    if kind == "faster" and selected not in _FASTER_ALLOWED and selected != "auto":
        _ws_log(ws, f"{_label_for_backbone(selected)} is not supported for Faster R-CNN; auto-fix is enabled.")
        selected = inferred_backbone or "resnet50_fpn"

    candidates = _candidate_list(kind, selected, inferred_backbone, custom_py, bool(auto_fix_backbone))
    scored: List[Tuple[int, int, str, Any, str]] = []
    errors: List[str] = []
    for cand in candidates:
        if cand == "custom" and not custom_py:
            continue
        try:
            model, name = _build_model_for_backbone(kind, cand, int(num_classes or 2), pretrained_coco=False, custom_backbone_path=custom_py)
            score, seen, total = _state_match_score(model, state)
            scored.append((score, seen, cand, model, name))
            _ws_log(ws, f"Backbone check {kind}: {_label_for_backbone(cand)} matches {score}/{seen} checkpoint tensors.")
        except Exception as exc:
            errors.append(f"{_label_for_backbone(cand)}: {exc}")
            continue

    if not scored:
        raise RuntimeError("Could not build any compatible R-CNN backbone. Last errors: " + " | ".join(errors[-5:]))

    # Choose highest matching tensor count. If equal, earlier candidate wins.
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, seen, best_key, model, model_name = scored[0]
    selected_for_msg = selected if selected != "auto" else (inferred_backbone or "auto")
    if selected_for_msg != "auto" and _normalize_backbone_name(selected_for_msg) != best_key:
        _ws_log(ws, f"Selected backbone {_label_for_backbone(selected_for_msg)} did not match best; auto-corrected to {_label_for_backbone(best_key)}.")
    else:
        _ws_log(ws, f"Using backbone {_label_for_backbone(best_key)}.")

    _load_state_into_model(model, state, ws, model_name)
    model.to(device)
    model.eval()
    pretty_classes = f", classes={num_classes}"
    if classes_list:
        pretty_classes += " [" + ", ".join(classes_list[:12]) + ("..." if len(classes_list) > 12 else "") + "]"
    model_name = f"{model_name} | {Path(custom_path).name}{pretty_classes}"
    cache[key] = (model, device, model_name)
    ws._mustatil_torchvision_model_cache = cache
    _ws_log(ws, f"Loaded {model_name} on {device}")
    return cache[key]


def _pil_to_tensor_local(pil_image):
    if _TARGET is not None and hasattr(_TARGET, "_pil_to_tensor"):
        return _TARGET._pil_to_tensor(pil_image)
    import torchvision.transforms.functional as F
    return F.to_tensor(pil_image.convert("RGB"))


def _detect_torchvision_pil_v2(ws: Any, kind: str, pil_image, conf: float, device_name: str,
                               custom_path: str = "", num_classes: int = 2, mask_threshold: float = 0.50,
                               backbone: str = "auto", custom_backbone_path: str = "",
                               auto_class_count: bool = True, auto_fix_backbone: bool = True) -> List[Dict[str, Any]]:
    import torch
    model, device, model_name = _load_detection_model_v2(
        ws, kind, device_name, custom_path, num_classes,
        backbone=backbone,
        custom_backbone_path=custom_backbone_path,
        auto_class_count=auto_class_count,
        auto_fix_backbone=auto_fix_backbone,
    )
    tensor = _pil_to_tensor_local(pil_image).to(device)
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


def _model_infer_pil_v2(ws: Any, mode: str, pil_image, settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    mode = str(mode)
    if mode == "faster":
        return _detect_torchvision_pil_v2(
            ws, "faster", pil_image,
            conf=float(settings.get("conf", 0.25)),
            device_name=str(settings.get("device", "cpu")),
            custom_path=str(settings.get("model_path", "")),
            num_classes=int(settings.get("num_classes", 2)),
            backbone=str(settings.get("backbone", "auto")),
            custom_backbone_path=str(settings.get("custom_backbone_path", "")),
            auto_class_count=bool(settings.get("auto_class_count", True)),
            auto_fix_backbone=bool(settings.get("auto_fix_backbone", True)),
        )
    if mode == "mask":
        return _detect_torchvision_pil_v2(
            ws, "mask", pil_image,
            conf=float(settings.get("conf", 0.25)),
            device_name=str(settings.get("device", "cpu")),
            custom_path=str(settings.get("model_path", "")),
            num_classes=int(settings.get("num_classes", 2)),
            mask_threshold=float(settings.get("mask_threshold", 0.50)),
            backbone=str(settings.get("backbone", "auto")),
            custom_backbone_path=str(settings.get("custom_backbone_path", "")),
            auto_class_count=bool(settings.get("auto_class_count", True)),
            auto_fix_backbone=bool(settings.get("auto_fix_backbone", True)),
        )
    if _ORIG_MODEL_INFER_PIL is not None:
        return _ORIG_MODEL_INFER_PIL(ws, mode, pil_image, settings)
    raise RuntimeError(f"Unknown mode: {mode}")


def _browse_file_to_line(line_edit: Any, caption: str, filt: str) -> None:
    try:
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(None, caption, "", filt)
        if path:
            line_edit.setText(path)
    except Exception:
        pass


def _settings_from_widgets_v2(mode: str, widgets: Dict[str, Any]) -> Dict[str, Any]:
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
            w = widgets[name]
            val = str(w.currentData()).strip() if hasattr(w, "currentData") and w.currentData() is not None else str(w.currentText()).strip()
            return val
        except Exception:
            return default
    def checked(name, default=True):
        try:
            return bool(widgets[name].isChecked())
        except Exception:
            return default
    title_map = {
        "faster": "Faster R-CNN",
        "mask": "Mask R-CNN",
        "unet": "U-Net Semantic Segmentation",
        "sam2": "SAM2 Box-Prompt Segmentation",
    }
    d = {
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
    if mode in {"faster", "mask"}:
        d.update({
            "title": ("Faster R-CNN" if mode == "faster" else "Mask R-CNN") + " / " + _label_for_backbone(current("backbone", "auto")),
            "backbone": current("backbone", "auto"),
            "custom_backbone_path": text("custom_backbone_path", ""),
            "auto_class_count": checked("auto_class_count", True),
            "auto_fix_backbone": checked("auto_fix_backbone", True),
        })
    return d


def _checkpoint_summary_for_ui(model_path: str) -> str:
    p = _safe_path(model_path)
    if not p:
        return "No custom model selected. Empty model path uses the selected built-in COCO/default model where available."
    try:
        ckpt, state, meta = _extract_state_and_meta(p, map_location="cpu")
        n = _class_count_from_state_dict(state or {}) if isinstance(state, dict) else None
        classes = _classes_from_checkpoint_dict(ckpt)
        bb = _infer_backbone_from_metadata(meta) or _infer_backbone_from_state_shapes(state)
        parts = ["Checkpoint inspected."]
        if n:
            parts.append(f"internal classes={n} incl. background")
        if classes:
            parts.append("user classes=" + ", ".join(classes[:8]) + ("..." if len(classes) > 8 else ""))
        if bb:
            parts.append("backbone hint=" + _label_for_backbone(bb))
        return " | ".join(parts)
    except Exception as exc:
        return "Could not inspect checkpoint yet: " + str(exc)


def _build_model_tab_v2(ws: Any, tab_kind: str, mode: str):
    # Leave U-Net and SAM2 untouched.
    if mode not in {"faster", "mask"}:
        return _ORIG_BUILD_MODEL_TAB(ws, tab_kind, mode)

    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton,
        QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox
    )

    target = _TARGET
    page = QWidget()
    root = QVBoxLayout(page)
    root.setContentsMargins(8, 8, 8, 8)

    model_title = "Faster R-CNN" if mode == "faster" else "Mask R-CNN"
    header = QLabel(
        f"{model_title} Detection with selectable backbone. "
        "For custom .pth checkpoints, Mustatil automatically reads the internal class count "
        "from the ROI head and auto-corrects a wrong backbone selection by matching checkpoint tensors."
    )
    header.setWordWrap(True)
    root.addWidget(header)

    box = QGroupBox(model_title + " Settings")
    g = QGridLayout(box)
    row = 0

    device = QComboBox(); device.setEditable(True); device.addItems(["cpu", "cuda", "auto"])
    g.addWidget(QLabel("Device"), row, 0); g.addWidget(device, row, 1, 1, 2); row += 1

    model_path = QLineEdit("")
    model_path.setPlaceholderText("Optional custom .pth/.pt/TorchScript. Empty = built-in COCO/default model.")
    browse_model = QPushButton("Browse")
    browse_model.clicked.connect(lambda *_: _browse_file_to_line(model_path, "Select Faster/Mask R-CNN model", "Model files (*.pt *.pth *.torchscript *.jit);;All files (*)"))
    g.addWidget(QLabel("Model file"), row, 0); g.addWidget(model_path, row, 1); g.addWidget(browse_model, row, 2); row += 1

    backbone = QComboBox()
    for label, key in _BACKBONE_ITEMS:
        backbone.addItem(label, key)
    backbone.setCurrentIndex(0)
    g.addWidget(QLabel("Backbone"), row, 0); g.addWidget(backbone, row, 1, 1, 2); row += 1

    custom_backbone_path = QLineEdit("")
    custom_backbone_path.setPlaceholderText("Only needed when Backbone = Custom Python Backbone")
    browse_backbone = QPushButton("Browse")
    browse_backbone.clicked.connect(lambda *_: _browse_file_to_line(custom_backbone_path, "Select custom Python backbone", "Python files (*.py);;All files (*)"))
    g.addWidget(QLabel("Custom backbone .py"), row, 0); g.addWidget(custom_backbone_path, row, 1); g.addWidget(browse_backbone, row, 2); row += 1

    auto_class_count = QCheckBox("Auto classes from checkpoint")
    auto_class_count.setChecked(True)
    auto_fix_backbone = QCheckBox("Auto-fix wrong backbone")
    auto_fix_backbone.setChecked(True)
    g.addWidget(auto_class_count, row, 0, 1, 2); g.addWidget(auto_fix_backbone, row, 2); row += 1

    num_classes = QSpinBox(); num_classes.setRange(1, 10000); num_classes.setValue(2)
    g.addWidget(QLabel("Manual internal classes"), row, 0); g.addWidget(num_classes, row, 1, 1, 2); row += 1

    conf = QDoubleSpinBox(); conf.setRange(0.001, 1.0); conf.setSingleStep(0.05); conf.setDecimals(3); conf.setValue(0.25)
    g.addWidget(QLabel("Confidence"), row, 0); g.addWidget(conf, row, 1, 1, 2); row += 1

    mask_threshold = QDoubleSpinBox(); mask_threshold.setRange(0.001, 1.0); mask_threshold.setSingleStep(0.05); mask_threshold.setDecimals(3); mask_threshold.setValue(0.50)
    if mode == "mask":
        g.addWidget(QLabel("Mask threshold metadata"), row, 0); g.addWidget(mask_threshold, row, 1, 1, 2); row += 1

    # Hidden/compat widgets expected by the original shared settings structure.
    threshold = QDoubleSpinBox(); threshold.setRange(0.001, 1.0); threshold.setValue(0.50)
    min_area = QSpinBox(); min_area.setRange(1, 10_000_000); min_area.setValue(64)
    base = QSpinBox(); base.setRange(8, 512); base.setValue(32)
    prompt_mode = QComboBox(); prompt_mode.addItems(["existing_fallback", "existing_only", "full_tile_only"])
    max_crop = QSpinBox(); max_crop.setRange(128, 8192); max_crop.setValue(1024)
    nms_iou = QDoubleSpinBox(); nms_iou.setRange(0.01, 1.0); nms_iou.setValue(0.80)
    max_prompts = QSpinBox(); max_prompts.setRange(1, 4096); max_prompts.setValue(128)

    tile = QSpinBox(); tile.setRange(64, 8192); tile.setSingleStep(64)
    try:
        tile.setValue(int(target._get_var(getattr(ws, "tile", None), 1024) or 1024))
    except Exception:
        tile.setValue(1024)
    overlap = QSpinBox(); overlap.setRange(0, 4096); overlap.setSingleStep(32)
    try:
        overlap.setValue(int(target._get_var(getattr(ws, "overlap", None), 128) or 128))
    except Exception:
        overlap.setValue(128)
    g.addWidget(QLabel("Tile size"), row, 0); g.addWidget(tile, row, 1, 1, 2); row += 1
    g.addWidget(QLabel("Overlap"), row, 0); g.addWidget(overlap, row, 1, 1, 2); row += 1

    info = QLabel("Idle")
    info.setWordWrap(True)
    g.addWidget(QLabel("Auto-detect"), row, 0); g.addWidget(info, row, 1, 1, 2); row += 1

    def update_info():
        txt = _checkpoint_summary_for_ui(model_path.text())
        try:
            if auto_class_count.isChecked() and model_path.text().strip():
                ckpt, state, _meta = _extract_state_and_meta(model_path.text(), map_location="cpu")
                n = _class_count_from_state_dict(state or {}) if isinstance(state, dict) else None
                if n:
                    old = num_classes.blockSignals(True)
                    num_classes.setValue(int(n))
                    num_classes.blockSignals(old)
        except Exception:
            pass
        info.setText(txt)

    model_path.editingFinished.connect(update_info)
    browse_model.clicked.connect(lambda *_: update_info())

    run_btn = QPushButton("Run on Detection image" if tab_kind == "detection" else "Run on selected satellite map")
    status = QLabel("Ready")
    status.setWordWrap(True)
    g.addWidget(run_btn, row, 0, 1, 3); row += 1
    g.addWidget(status, row, 0, 1, 3); row += 1
    root.addWidget(box)

    dep_box = QGroupBox("Dependencies / Notes")
    dg = QGridLayout(dep_box)
    dep_label = QLabel(
        "Needs torch + torchvision + numpy + pillow. "
        "If you use a checkpoint trained with another backbone, keep Auto-fix enabled. "
        "Class count is internal torchvision count, so background is included."
    )
    dep_label.setWordWrap(True)
    dep_btn = QPushButton("Install / repair TorchVision deps")
    dep_btn.clicked.connect(lambda *_: target._install_deps_button_clicked(ws) if hasattr(target, "_install_deps_button_clicked") else None)
    dg.addWidget(dep_label, 0, 0, 1, 2); dg.addWidget(dep_btn, 1, 0, 1, 2)
    root.addWidget(dep_box)
    root.addStretch(1)

    widgets = {
        "device": device,
        "model_path": model_path,
        "backbone": backbone,
        "custom_backbone_path": custom_backbone_path,
        "auto_class_count": auto_class_count,
        "auto_fix_backbone": auto_fix_backbone,
        "num_classes": num_classes,
        "conf": conf,
        "threshold": threshold,
        "mask_threshold": mask_threshold,
        "tile": tile,
        "overlap": overlap,
        "min_area": min_area,
        "base": base,
        "prompt_mode": prompt_mode,
        "max_crop": max_crop,
        "nms_iou": nms_iou,
        "max_prompts": max_prompts,
        "sam_class_id": num_classes,
    }

    def run_clicked():
        update_info()
        settings = _settings_from_widgets_v2(mode, widgets)
        try:
            status.setText("Starting " + settings["title"] + "...")
            ws.log("Starting " + settings["title"] + "...")
        except Exception:
            pass
        if tab_kind == "detection":
            target._run_as_task(ws, settings["title"], lambda: target._run_detection_image(ws, mode, settings))
        else:
            target._run_as_task(ws, settings["title"] + " Satellite", lambda: target._run_satellite(ws, mode, settings), allow_parallel=True)

    run_btn.clicked.connect(run_clicked)
    return page


def _patch_target_module(target: Any) -> bool:
    global _TARGET, _ORIG_BUILD_MODEL_TAB, _ORIG_MODEL_INFER_PIL, _ORIG_SETTINGS_FROM_WIDGETS, _PATCHED
    if target is None or _PATCHED:
        return bool(_PATCHED)
    if not hasattr(target, "_build_model_tab") or not hasattr(target, "_model_infer_pil"):
        return False
    _TARGET = target
    _ORIG_BUILD_MODEL_TAB = getattr(target, "_build_model_tab")
    _ORIG_MODEL_INFER_PIL = getattr(target, "_model_infer_pil")
    _ORIG_SETTINGS_FROM_WIDGETS = getattr(target, "_settings_from_widgets", None)

    target._build_model_tab = _build_model_tab_v2
    target._model_infer_pil = _model_infer_pil_v2
    target._settings_from_widgets = _settings_from_widgets_v2
    target._load_detection_model_backbone_v2 = _load_detection_model_v2
    target._detect_torchvision_pil_backbone_v2 = _detect_torchvision_pil_v2
    target.MUSTATIL_BACKBONE_PATCH_ACTIVE = True
    _PATCHED = True
    _log("Patched TorchVision Faster/Mask R-CNN tabs with backbone dropdown + auto class/backbone inference.")
    return True


def _late_patch_scan() -> None:
    try:
        target = _find_target_module()
        if _patch_target_module(target):
            return
    except Exception as exc:
        _log("Late patch scan failed: " + str(exc))
    try:
        from PySide6.QtCore import QTimer
        QTimer.singleShot(1200, _late_patch_scan)
    except Exception:
        pass


def mustatil_plugin_init():
    target = _find_target_module()
    if not _patch_target_module(target):
        _log("Target TorchVision R-CNN tabs plugin not found yet; scheduling late scan.")
        try:
            from PySide6.QtCore import QTimer
            QTimer.singleShot(800, _late_patch_scan)
        except Exception:
            pass
