#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mustatil plugin patch: R-CNN Training Studio + Custom Backbone Creator + optional ONNX export
for the existing Faster R-CNN / Mask R-CNN Trainer plugin.

Install:
  1) Keep the original R-CNN trainer plugin in mustatil_plugins.
  2) Drop this file into the same mustatil_plugins folder.
  3) Restart Mustatil.

What this patch does:
  - Replaces the R-CNN Trainer tab with a workflow-oriented Training Studio layout.
  - Left column: dataset/classes. Middle column: model/backbone/custom creator.
    Right column: training strategy, freeze/warmup controls, and optional ONNX export.
  - Moves the training console to the bottom and mirrors messages to the normal
    Mustatil workspace console via ws.log(...).
  - Adds a Custom Backbone Creator that writes a ready-to-train backbone .py
    file and selects it automatically.
  - Extends model construction to common torchvision backbones plus a custom
    Python backbone hook.

Custom Python backbone file contract:
  def build_backbone(pretrained=False, trainable_layers=3, out_channels=256):
      ...
      backbone.out_channels = 256
      return backbone

Notes:
  - Detection later has to rebuild the same architecture/backbone as training.
    If your detection plugin only knows ResNet50-FPN, add a matching loader patch
    before using non-ResNet50 checkpoints for detection.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import shutil
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

_PATCHED = False
_TARGET_MODULE_NAME = ""


BACKBONE_LABELS = [
    "ResNet50-FPN (standard, recommended)",
    "ResNet50-FPN v2 (newer, stronger if available)",
    "ResNet101-FPN (stronger, slower)",
    "ResNet18-FPN (small / fast)",
    "ResNet34-FPN (medium / fast)",
    "ResNeXt50-32x4d-FPN (strong alternative)",
    "Wide ResNet50-2-FPN (wide features)",
    "MobileNetV3-Large-FPN (fast / low VRAM)",
    "MobileNetV3-Large-320-FPN (very fast / small images)",
    "Custom Python Backbone (.py)",
]

BACKBONE_ID = {
    "ResNet50-FPN (standard, recommended)": "resnet50_fpn",
    "ResNet50-FPN v2 (newer, stronger if available)": "resnet50_fpn_v2",
    "ResNet101-FPN (stronger, slower)": "resnet101_fpn",
    "ResNet18-FPN (small / fast)": "resnet18_fpn",
    "ResNet34-FPN (medium / fast)": "resnet34_fpn",
    "ResNeXt50-32x4d-FPN (strong alternative)": "resnext50_32x4d_fpn",
    "Wide ResNet50-2-FPN (wide features)": "wide_resnet50_2_fpn",
    "MobileNetV3-Large-FPN (fast / low VRAM)": "mobilenet_v3_large_fpn",
    "MobileNetV3-Large-320-FPN (very fast / small images)": "mobilenet_v3_large_320_fpn",
    "Custom Python Backbone (.py)": "custom_python_backbone",
}

ANCHOR_PROFILES = {
    "Torchvision default anchors (recommended)": None,
    "Small objects / satellite structures": [8, 16, 32, 64, 128],
    "Medium objects": [16, 32, 64, 128, 256],
    "Large objects": [32, 64, 128, 256, 512],
    "Extra large objects": [64, 128, 256, 512, 1024],
}


CUSTOM_BACKBONE_TEMPLATES = [
    "Mustatil Small FPN (recommended custom start)",
    "Mustatil Tiny FPN (low VRAM)",
    "Mustatil Wide FPN (more channels, slower)",
]


def _custom_backbone_template(template_name: str, out_channels: int = 256, width: int = 64) -> str:
    # Return a complete custom backbone .py template.
    name = str(template_name or "").lower()
    if "tiny" in name:
        width = min(int(width or 32), 48)
        depth_note = "Tiny low-VRAM custom FPN backbone"
    elif "wide" in name:
        width = max(int(width or 96), 96)
        depth_note = "Wide custom FPN backbone"
    else:
        width = int(width or 64)
        depth_note = "Small recommended custom FPN backbone"
    out_channels = int(out_channels or 256)
    return """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Mustatil Custom Backbone Template
# {depth_note}
#
# This file is meant for the Mustatil R-CNN Trainer custom-backbone mode.
# It defines build_backbone(...), returns five FPN-style feature maps, and sets
# backbone.out_channels. Keep this file next to your trained checkpoints so the
# Detection loader can rebuild exactly the same architecture later.
from __future__ import annotations

from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MustatilCustomFPNBackbone(nn.Module):
    # Compact CNN + FPN-like output for Faster R-CNN / Mask R-CNN.
    # Output maps: 0=stride2, 1=stride4, 2=stride8, 3=stride16, 4=stride32.
    def __init__(self, out_channels: int = {out_channels}, width: int = {width}):
        super().__init__()
        width = int(width)
        self.out_channels = int(out_channels)

        self.stem = ConvBNAct(3, width, stride=2)
        self.layer1 = ConvBNAct(width, width * 2, stride=2)
        self.layer2 = ConvBNAct(width * 2, width * 4, stride=2)
        self.layer3 = ConvBNAct(width * 4, width * 8, stride=2)
        self.layer4 = ConvBNAct(width * 8, width * 8, stride=2)

        self.lateral0 = nn.Conv2d(width, self.out_channels, kernel_size=1)
        self.lateral1 = nn.Conv2d(width * 2, self.out_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(width * 4, self.out_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(width * 8, self.out_channels, kernel_size=1)
        self.lateral4 = nn.Conv2d(width * 8, self.out_channels, kernel_size=1)

        self.smooth0 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.smooth1 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.smooth4 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        c0 = self.stem(x)
        c1 = self.layer1(c0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lateral2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        p1 = self.lateral1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="nearest")
        p0 = self.lateral0(c0) + F.interpolate(p1, size=c0.shape[-2:], mode="nearest")

        return OrderedDict([
            ("0", self.smooth0(p0)),
            ("1", self.smooth1(p1)),
            ("2", self.smooth2(p2)),
            ("3", self.smooth3(p3)),
            ("4", self.smooth4(p4)),
        ])


def build_backbone(pretrained: bool = False, trainable_layers: int = 5, out_channels: int = {out_channels}):
    # Mustatil entry point. The pretrained argument is accepted for compatibility.
    model = MustatilCustomFPNBackbone(out_channels=int(out_channels), width={width})

    blocks = [model.stem, model.layer1, model.layer2, model.layer3, model.layer4]
    trainable_layers = max(0, min(5, int(trainable_layers)))
    freeze_count = max(0, 5 - trainable_layers)
    for block in blocks[:freeze_count]:
        for p in block.parameters():
            p.requires_grad = False

    model.out_channels = int(out_channels)
    return model
""".format(depth_note=depth_note, out_channels=out_channels, width=width)


def _write_custom_backbone_template(path: Path, template_name: str, out_channels: int, width: int) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    code = _custom_backbone_template(template_name, out_channels=out_channels, width=width)
    path.write_text(code, encoding="utf-8")
    return path


def _count_backbone_feature_maps(backbone: Any) -> int:
    # Best-effort feature-map count for custom backbones.
    try:
        import torch
        was_training = bool(getattr(backbone, "training", False))
        try:
            backbone.eval()
        except Exception:
            pass
        with torch.no_grad():
            out = backbone(torch.zeros(1, 3, 256, 256))
        try:
            backbone.train(was_training)
        except Exception:
            pass
        if isinstance(out, dict):
            return max(1, len(out))
        if isinstance(out, (list, tuple)):
            return max(1, len(out))
        return 1
    except Exception as exc:
        _log("Could not count custom backbone feature maps; assuming 5: " + str(exc))
        return 5


def _anchor_sizes_for_count(profile_text: str, count: int):
    sizes = ANCHOR_PROFILES.get(str(profile_text), None) or [32, 64, 128, 256, 512]
    sizes = [int(s) for s in sizes]
    count = max(1, int(count or 1))
    if count == 1:
        return (tuple(sizes),)
    if count == len(sizes):
        return tuple((s,) for s in sizes)
    if count > 1:
        picked = []
        for i in range(count):
            j = int(round(i * (len(sizes) - 1) / max(1, count - 1)))
            picked.append(sizes[j])
        return tuple((s,) for s in picked)
    return tuple((s,) for s in sizes)


def _ensure_custom_anchor_generator(backbone: Any, cfg: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from torchvision.models.detection.rpn import AnchorGenerator
        count = _count_backbone_feature_maps(backbone)
        sizes_tuple = _anchor_sizes_for_count(str(cfg.get("anchor_profile", "")), count)
        ratios_tuple = tuple((0.5, 1.0, 2.0) for _ in sizes_tuple)
        kwargs["rpn_anchor_generator"] = AnchorGenerator(sizes=sizes_tuple, aspect_ratios=ratios_tuple)
    except Exception as exc:
        _log("Custom anchor generator setup failed; torchvision defaults may be used: " + str(exc))
    return kwargs


def _log(msg: str) -> None:
    try:
        print("[Mustatil R-CNN Advanced Backbone Patch] " + str(msg), flush=True)
    except Exception:
        pass


def _find_target_module():
    """Find the original R-CNN trainer plugin module loaded by Mustatil."""
    global _TARGET_MODULE_NAME
    if _TARGET_MODULE_NAME and _TARGET_MODULE_NAME in sys.modules:
        return sys.modules[_TARGET_MODULE_NAME]
    for name, mod in list(sys.modules.items()):
        try:
            if not name.startswith("mustatil_plugin_"):
                continue
            if all(hasattr(mod, attr) for attr in (
                "_build_training_tab",
                "_train_rcnn_worker",
                "_build_detection_model",
                "MustatilYoloDetectionDataset",
            )):
                _TARGET_MODULE_NAME = name
                return mod
        except Exception:
            pass
    return None


def _safe_attr(obj: Any, name: str, default=None):
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _parse_classes(text: str):
    target = _find_target_module()
    if target is not None and hasattr(target, "_parse_classes"):
        return target._parse_classes(text)
    vals = [p.strip() for p in str(text or "").replace(";", ",").split(",") if p.strip()]
    return vals or ["mustatil", "false_positive"]


def _now_stamp() -> str:
    target = _find_target_module()
    if target is not None and hasattr(target, "_now_stamp"):
        return target._now_stamp()
    return time.strftime("%Y%m%d_%H%M%S")


def _safe_name(text: str) -> str:
    target = _find_target_module()
    if target is not None and hasattr(target, "_safe_name"):
        return target._safe_name(text)
    bad = '<>:"/\\|?*\n\r\t'
    out = "".join("_" if c in bad else c for c in str(text))
    out = "_".join(out.strip().split())
    return out or "model"


def _project_guess(ws: Any) -> Path:
    target = _find_target_module()
    if target is not None and hasattr(target, "_project_guess"):
        return target._project_guess(ws)
    try:
        p = Path(str(ws.project.get())).expanduser()
        if p.exists():
            return p
    except Exception:
        pass
    return Path.cwd()


def _guess_image_dir(project: Path) -> Path:
    target = _find_target_module()
    if target is not None and hasattr(target, "_guess_image_dir"):
        return target._guess_image_dir(project)
    return Path(project) / "images"


def _guess_label_dir(project: Path) -> Path:
    target = _find_target_module()
    if target is not None and hasattr(target, "_guess_label_dir"):
        return target._guess_label_dir(project)
    return Path(project) / "labels"


def _make_anchor_generator(profile_text: str):
    sizes = ANCHOR_PROFILES.get(str(profile_text), None)
    if not sizes:
        return None
    from torchvision.models.detection.rpn import AnchorGenerator
    # Standard FPN backbones in torchvision usually expose five feature maps.
    # Each tuple belongs to one feature-map level.
    sizes_tuple = tuple((int(s),) for s in sizes)
    ratios_tuple = tuple((0.5, 1.0, 2.0) for _ in sizes_tuple)
    return AnchorGenerator(sizes=sizes_tuple, aspect_ratios=ratios_tuple)


def _model_kwargs_from_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    try:
        min_size = int(cfg.get("min_size", 0) or 0)
        max_size = int(cfg.get("max_size", 0) or 0)
        if min_size > 0:
            kwargs["min_size"] = min_size
        if max_size > 0:
            kwargs["max_size"] = max_size
    except Exception:
        pass
    try:
        anchor = _make_anchor_generator(str(cfg.get("anchor_profile", "")))
        if anchor is not None:
            kwargs["rpn_anchor_generator"] = anchor
    except Exception as exc:
        _log("Anchor profile ignored: " + str(exc))
    return kwargs


def _try_constructor(ctor, pretrained: bool, kwargs: Dict[str, Any]):
    weights = "DEFAULT" if pretrained else None
    attempts = [
        lambda: ctor(weights=weights, **kwargs),
        lambda: ctor(pretrained=bool(pretrained), **kwargs),
        lambda: ctor(weights=weights),
        lambda: ctor(pretrained=bool(pretrained)),
        lambda: ctor(),
    ]
    last_exc = None
    for attempt in attempts:
        try:
            return attempt()
        except Exception as exc:
            last_exc = exc
    raise last_exc or RuntimeError("Could not construct model.")


def _replace_faster_head(model, num_classes: int):
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, int(num_classes))
    return model


def _replace_mask_head(model, num_classes: int):
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, int(num_classes))
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, int(num_classes))
    return model


def _load_custom_backbone(path: str, pretrained: bool, trainable_layers: int, out_channels: int):
    p = Path(str(path or "")).expanduser()
    if not p.exists():
        raise RuntimeError("Custom Python backbone file not found: " + str(p))
    module_name = "mustatil_custom_backbone_" + re.sub(r"\W+", "_", p.stem) + "_" + str(abs(hash(str(p))))
    spec = importlib.util.spec_from_file_location(module_name, str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import custom backbone file: " + str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "build_backbone"):
        raise RuntimeError("Custom backbone file must define build_backbone(...).")
    builder = mod.build_backbone
    attempts = [
        lambda: builder(pretrained=pretrained, trainable_layers=trainable_layers, out_channels=out_channels),
        lambda: builder(pretrained=pretrained, trainable_layers=trainable_layers),
        lambda: builder(pretrained=pretrained),
        lambda: builder(),
    ]
    last_exc = None
    for attempt in attempts:
        try:
            backbone = attempt()
            if not hasattr(backbone, "out_channels"):
                backbone.out_channels = int(out_channels or 256)
            return backbone
        except Exception as exc:
            last_exc = exc
    raise last_exc or RuntimeError("Could not build custom backbone.")


def _build_resnet_fpn_backbone(backbone_id: str, pretrained: bool, trainable_layers: int):
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    name = backbone_id.replace("_fpn", "")
    attempts = [
        lambda: resnet_fpn_backbone(backbone_name=name, weights="DEFAULT" if pretrained else None, trainable_layers=trainable_layers),
        lambda: resnet_fpn_backbone(name, weights="DEFAULT" if pretrained else None, trainable_layers=trainable_layers),
        lambda: resnet_fpn_backbone(backbone_name=name, pretrained=bool(pretrained), trainable_layers=trainable_layers),
        lambda: resnet_fpn_backbone(name, pretrained=bool(pretrained), trainable_layers=trainable_layers),
        lambda: resnet_fpn_backbone(name, pretrained=bool(pretrained)),
    ]
    last_exc = None
    for attempt in attempts:
        try:
            backbone = attempt()
            if not hasattr(backbone, "out_channels"):
                backbone.out_channels = 256
            return backbone
        except Exception as exc:
            last_exc = exc
    raise last_exc or RuntimeError("Could not construct resnet_fpn_backbone: " + str(backbone_id))


def _infer_architecture(model_kind: str, cfg: Optional[Dict[str, Any]] = None) -> str:
    if cfg and str(cfg.get("architecture", "")).strip():
        raw = str(cfg.get("architecture", "")).lower()
        return "maskrcnn" if "mask" in raw else "fasterrcnn"
    raw = str(model_kind or "").lower()
    return "maskrcnn" if "mask" in raw else "fasterrcnn"


def _infer_backbone(model_kind: str, cfg: Optional[Dict[str, Any]] = None) -> str:
    if cfg and str(cfg.get("backbone", "")).strip():
        return str(cfg.get("backbone")).strip()
    raw = str(model_kind or "").lower().replace("-", "_")
    for val in BACKBONE_ID.values():
        if val in raw:
            return val
    if "mobilenet" in raw and "320" in raw:
        return "mobilenet_v3_large_320_fpn"
    if "mobilenet" in raw:
        return "mobilenet_v3_large_fpn"
    return "resnet50_fpn"


def _advanced_build_detection_model(
    model_kind: str,
    num_user_classes: int,
    pretrained: bool = True,
    cfg: Optional[Dict[str, Any]] = None,
):
    import torchvision
    from torchvision.models.detection import FasterRCNN, MaskRCNN

    num_classes = int(num_user_classes) + 1  # label 0 is background
    cfg = dict(cfg or {})
    architecture = _infer_architecture(model_kind, cfg)
    backbone_id = _infer_backbone(model_kind, cfg)
    trainable_layers = int(cfg.get("trainable_backbone_layers", 3) or 3)
    custom_out_channels = int(cfg.get("custom_out_channels", 256) or 256)
    kwargs = _model_kwargs_from_cfg(cfg)

    if backbone_id == "custom_python_backbone":
        backbone = _load_custom_backbone(
            cfg.get("custom_backbone_path", ""),
            pretrained=bool(pretrained),
            trainable_layers=trainable_layers,
            out_channels=custom_out_channels,
        )
        kwargs = _ensure_custom_anchor_generator(backbone, cfg, kwargs)
        if architecture == "maskrcnn":
            return MaskRCNN(backbone, num_classes=num_classes, **kwargs)
        return FasterRCNN(backbone, num_classes=num_classes, **kwargs)

    if architecture == "fasterrcnn":
        builtins = {
            "resnet50_fpn": getattr(torchvision.models.detection, "fasterrcnn_resnet50_fpn", None),
            "resnet50_fpn_v2": getattr(torchvision.models.detection, "fasterrcnn_resnet50_fpn_v2", None),
            "mobilenet_v3_large_fpn": getattr(torchvision.models.detection, "fasterrcnn_mobilenet_v3_large_fpn", None),
            "mobilenet_v3_large_320_fpn": getattr(torchvision.models.detection, "fasterrcnn_mobilenet_v3_large_320_fpn", None),
        }
        ctor = builtins.get(backbone_id)
        if ctor is not None:
            model = _try_constructor(ctor, bool(pretrained), kwargs)
            return _replace_faster_head(model, num_classes)
        if backbone_id == "resnet50_fpn_v2":
            _log("fasterrcnn_resnet50_fpn_v2 unavailable; falling back to resnet50_fpn.")
            ctor = getattr(torchvision.models.detection, "fasterrcnn_resnet50_fpn", None)
            if ctor is not None:
                model = _try_constructor(ctor, bool(pretrained), kwargs)
                return _replace_faster_head(model, num_classes)
        backbone = _build_resnet_fpn_backbone(backbone_id, bool(pretrained), trainable_layers)
        return FasterRCNN(backbone, num_classes=num_classes, **kwargs)

    # Mask R-CNN
    mask_builtins = {
        "resnet50_fpn": getattr(torchvision.models.detection, "maskrcnn_resnet50_fpn", None),
        "resnet50_fpn_v2": getattr(torchvision.models.detection, "maskrcnn_resnet50_fpn_v2", None),
    }
    ctor = mask_builtins.get(backbone_id)
    if ctor is not None:
        model = _try_constructor(ctor, bool(pretrained), kwargs)
        return _replace_mask_head(model, num_classes)
    if backbone_id == "resnet50_fpn_v2":
        _log("maskrcnn_resnet50_fpn_v2 unavailable; falling back to maskrcnn_resnet50_fpn.")
        ctor = getattr(torchvision.models.detection, "maskrcnn_resnet50_fpn", None)
        if ctor is not None:
            model = _try_constructor(ctor, bool(pretrained), kwargs)
            return _replace_mask_head(model, num_classes)
    backbone = _build_resnet_fpn_backbone(backbone_id, bool(pretrained), trainable_layers)
    return MaskRCNN(backbone, num_classes=num_classes, **kwargs)



def _set_backbone_trainable(model: Any, trainable: bool, logger=None) -> int:
    """Set requires_grad for the complete detection backbone.

    Faster R-CNN / Mask R-CNN is one PyTorch model, but the backbone is still
    accessible as model.backbone. This helper is used for explicit head-only
    training and for warmup-then-unfreeze training.
    """
    count = 0
    try:
        for p in model.backbone.parameters():
            p.requires_grad = bool(trainable)
            count += 1
    except Exception:
        count = 0
    if logger:
        state = "trainable" if trainable else "frozen"
        logger(f"Backbone set to {state}: {count} parameter tensors.")
    return count


def _make_sgd_optimizer(model: Any, lr: float, weight_decay: float):
    import torch
    params = [p for p in model.parameters() if getattr(p, "requires_grad", False)]
    if not params:
        raise RuntimeError("No trainable parameters. Use full training, warmup+unfreeze, or disable backbone freeze.")
    return torch.optim.SGD(params, lr=float(lr), momentum=0.9, weight_decay=float(weight_decay))


def _try_export_detection_onnx(model: Any, onnx_path: Path, img_size: int = 640, opset: int = 17, logger=None) -> bool:
    """Best-effort ONNX export for inference.

    TorchVision detection models include dynamic post-processing. Export may
    depend on installed torch/torchvision versions. Failure must not invalidate
    the finished PyTorch training checkpoint.
    """
    import torch

    def log(msg: str):
        if logger:
            logger(str(msg))
        else:
            _log(str(msg))

    class DetectionOnnxWrapper(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, images):
            # ONNX gets a BCHW tensor. TorchVision detection expects a list of CHW tensors.
            out = self.wrapped([images[0]])
            if isinstance(out, (list, tuple)):
                out = out[0] if out else {}
            if not isinstance(out, dict):
                raise RuntimeError("Detection model did not return a dict-like output.")
            boxes = out.get("boxes")
            labels = out.get("labels")
            scores = out.get("scores")
            if boxes is None:
                boxes = torch.zeros((0, 4), dtype=torch.float32, device=images.device)
            if labels is None:
                labels = torch.zeros((0,), dtype=torch.int64, device=images.device)
            if scores is None:
                scores = torch.zeros((0,), dtype=torch.float32, device=images.device)
            masks = out.get("masks")
            if masks is None:
                masks = torch.zeros((0, 1, images.shape[-2], images.shape[-1]), dtype=torch.float32, device=images.device)
            return boxes, labels, scores, masks

    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cpu")
    dummy = torch.zeros((1, 3, int(img_size), int(img_size)), dtype=torch.float32, device=device)
    wrapper = DetectionOnnxWrapper(model).to(device).eval()
    try:
        log(f"ONNX export started: {onnx_path}")
        torch.onnx.export(
            wrapper,
            dummy,
            str(onnx_path),
            input_names=["images"],
            output_names=["boxes", "labels", "scores", "masks"],
            opset_version=int(opset),
            do_constant_folding=True,
            dynamic_axes={
                "images": {0: "batch", 2: "height", 3: "width"},
                "boxes": {0: "detections"},
                "labels": {0: "detections"},
                "scores": {0: "detections"},
                "masks": {0: "detections", 2: "mask_height", 3: "mask_width"},
            },
        )
        log(f"ONNX export finished: {onnx_path}")
        return True
    except Exception as exc:
        log("ONNX export failed, but the PyTorch .pth checkpoint is still valid: " + str(exc))
        try:
            log(traceback.format_exc())
        except Exception:
            pass
        return False
    finally:
        try:
            model.train(was_training)
        except Exception:
            pass

def _advanced_train_rcnn_worker(cfg: Dict[str, Any], logger=None) -> str:
    target = _find_target_module()
    if target is None:
        raise RuntimeError("Original Mustatil R-CNN trainer plugin was not found.")

    import torch
    from torch.utils.data import DataLoader

    def log(msg):
        if logger:
            logger(str(msg))
        else:
            _log(str(msg))

    image_dir = Path(cfg["image_dir"])
    label_dir = Path(cfg["label_dir"])
    out_dir = Path(cfg["out_dir"])
    classes = list(cfg["classes"])
    architecture = _infer_architecture(str(cfg.get("model_kind", "fasterrcnn")), cfg)
    backbone = _infer_backbone(str(cfg.get("model_kind", "fasterrcnn")), cfg)
    model_kind = str(cfg.get("model_kind") or f"{architecture}_{backbone}")
    pretrained = bool(cfg.get("pretrained", True))
    checkpoint_path = str(cfg.get("checkpoint_path", "") or "")
    freeze_backbone = bool(cfg.get("freeze_backbone", False))
    mask_mode = str(cfg.get("mask_mode", "boxes"))
    epochs = int(cfg.get("epochs", 10))
    batch_size = int(cfg.get("batch", 1))
    lr = float(cfg.get("lr", 0.005))
    weight_decay = float(cfg.get("weight_decay", 0.0005))
    num_workers = int(cfg.get("workers", 0))
    max_images = int(cfg.get("max_images", 0))
    include_empty = bool(cfg.get("include_empty", True))
    save_every = max(1, int(cfg.get("save_every", 1)))
    training_strategy = str(cfg.get("training_strategy", "full_model") or "full_model").lower().strip()
    warmup_epochs = max(0, int(cfg.get("warmup_epochs", 0) or 0))
    export_onnx_after_training = bool(cfg.get("export_onnx_after_training", False))
    onnx_opset = int(cfg.get("onnx_opset", 17) or 17)
    onnx_img_size = int(cfg.get("onnx_img_size", 640) or 640)

    run_name = _safe_name(f"{architecture}_{backbone}_{_now_stamp()}")
    run_dir = out_dir / run_name
    target._ensure_dir(run_dir)

    log(f"Training run folder: {run_dir}")
    log(f"Images: {image_dir}")
    log(f"Labels: {label_dir}")
    log(f"Classes: {classes}")
    log(f"Architecture: {architecture}")
    log(f"Backbone: {backbone}")
    log(f"Model kind: {model_kind}")

    dataset = target.MustatilYoloDetectionDataset(
        str(image_dir), str(label_dir), classes,
        mask_mode=mask_mode, max_images=max_images, include_empty=include_empty,
    )
    if len(dataset) <= 0:
        raise RuntimeError("No training images found. Check image folder and file extensions.")

    log(f"Dataset images: {len(dataset)}")
    sample_nonempty = 0
    sample_boxes = 0
    for i in range(min(len(dataset), 50)):
        try:
            _, t = dataset[i]
            n = int(t["boxes"].shape[0])
            sample_boxes += n
            if n:
                sample_nonempty += 1
        except Exception:
            pass
    log(f"Dataset check first {min(len(dataset),50)} images: non-empty={sample_nonempty}, boxes={sample_boxes}")

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=max(0, num_workers),
        collate_fn=target._collate_fn,
        pin_memory=False,
    )

    device = target._resolve_device(cfg.get("device", "auto"))
    log(f"Device: {device}")
    model = _advanced_build_detection_model(model_kind, len(classes), pretrained=pretrained, cfg=cfg)
    target._load_checkpoint_flexible(model, checkpoint_path, logger=log)

    # Training strategy:
    # - full_model: train backbone + RPN + ROI heads from the beginning
    # - freeze_backbone: head-only training; useful for a quick low-VRAM warm start
    # - warmup_then_unfreeze: train heads first, then unfreeze the whole backbone
    strategy_freeze = freeze_backbone or training_strategy in {"freeze_backbone", "head_only", "head-only"}
    strategy_warmup = training_strategy in {"warmup_then_unfreeze", "warmup", "head_warmup_then_full"}
    if strategy_freeze:
        _set_backbone_trainable(model, False, logger=log)
    elif strategy_warmup and warmup_epochs > 0:
        _set_backbone_trainable(model, False, logger=log)
        log(f"Training strategy: head warmup for {warmup_epochs} epoch(s), then full-model training.")
    else:
        log("Training strategy: full model training from epoch 1.")

    model.to(device)

    optimizer = _make_sgd_optimizer(model, lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)

    metadata = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_kind": model_kind,
        "architecture": architecture,
        "backbone": backbone,
        "classes": classes,
        "num_classes_including_background": len(classes) + 1,
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "mask_mode": mask_mode,
        "pretrained": pretrained,
        "checkpoint_path": checkpoint_path,
        "freeze_backbone": freeze_backbone,
        "training_strategy": training_strategy,
        "warmup_epochs": warmup_epochs,
        "trainable_backbone_layers": int(cfg.get("trainable_backbone_layers", 3) or 3),
        "min_size": int(cfg.get("min_size", 800) or 800),
        "max_size": int(cfg.get("max_size", 1333) or 1333),
        "anchor_profile": str(cfg.get("anchor_profile", "Torchvision default anchors (recommended)")),
        "custom_backbone_path": str(cfg.get("custom_backbone_path", "") or ""),
        "custom_out_channels": int(cfg.get("custom_out_channels", 256) or 256),
        "epochs": epochs,
        "batch": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "device": str(device),
        "export_onnx_after_training": export_onnx_after_training,
        "onnx_opset": onnx_opset,
        "onnx_img_size": onnx_img_size,
        "internal_label_note": "YOLO class 0 -> torchvision label 1; label 0 is background.",
        "detection_loader_note": "Use the same architecture/backbone settings when loading this checkpoint for detection.",
    }
    (run_dir / "classes.json").write_text(json.dumps(classes, indent=2, ensure_ascii=False), encoding="utf-8")
    if backbone == "custom_python_backbone":
        try:
            src_custom = Path(str(cfg.get("custom_backbone_path", "") or "")).expanduser()
            if src_custom.exists():
                dst_custom = run_dir / "custom_backbone_used_for_training.py"
                shutil.copy2(src_custom, dst_custom)
                metadata["custom_backbone_copied_to"] = str(dst_custom)
                log(f"Custom backbone copied into run folder: {dst_custom}")
        except Exception as exc:
            log("Custom backbone copy warning: " + str(exc))
    (run_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    model.train()
    last_ckpt = ""
    for epoch in range(1, epochs + 1):
        if strategy_warmup and warmup_epochs > 0 and epoch == warmup_epochs + 1:
            log("Warmup finished. Unfreezing backbone and rebuilding optimizer for full-model training.")
            _set_backbone_trainable(model, True, logger=log)
            optimizer = _make_sgd_optimizer(model, lr=lr, weight_decay=weight_decay)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, max(1, epochs - warmup_epochs) // 3), gamma=0.1)
        epoch_loss = 0.0
        epoch_batches = 0
        t0 = time.time()
        for step, (images, targets) in enumerate(data_loader, 1):
            images = [img.to(device) for img in images]
            clean_targets = []
            for t in targets:
                ct = {}
                for k, v in t.items():
                    if hasattr(v, "to"):
                        ct[k] = v.to(device)
                    elif k not in {"image_path", "label_path"}:
                        ct[k] = v
                clean_targets.append(ct)

            loss_dict = model(images, clean_targets)
            losses = sum(loss for loss in loss_dict.values())
            if not torch.isfinite(losses):
                log(f"Epoch {epoch}/{epochs} step {step}/{len(data_loader)}: non-finite loss skipped: {float(losses.detach().cpu())}")
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            losses.backward()
            optimizer.step()

            val = float(losses.detach().cpu())
            epoch_loss += val
            epoch_batches += 1
            if step == 1 or step % 10 == 0 or step == len(data_loader):
                parts = []
                for k, v in loss_dict.items():
                    try:
                        parts.append(f"{k}={float(v.detach().cpu()):.4f}")
                    except Exception:
                        pass
                log(f"Epoch {epoch}/{epochs} step {step}/{len(data_loader)} loss={val:.4f} " + " ".join(parts))
        try:
            lr_scheduler.step()
        except Exception:
            pass
        avg = epoch_loss / max(1, epoch_batches)
        log(f"Epoch {epoch}/{epochs} finished: avg_loss={avg:.4f}, time={time.time()-t0:.1f}s")

        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = run_dir / f"{_safe_name(model_kind)}_epoch_{epoch:03d}.pth"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "classes": classes,
                "model_kind": model_kind,
                "architecture": architecture,
                "backbone": backbone,
                "metadata": metadata,
            }, ckpt_path)
            last_ckpt = str(ckpt_path)
            log(f"Checkpoint saved: {ckpt_path}")

    final_path = run_dir / f"{_safe_name(model_kind)}_final.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epochs,
        "classes": classes,
        "model_kind": model_kind,
        "architecture": architecture,
        "backbone": backbone,
        "metadata": metadata,
    }, final_path)
    last_ckpt = str(final_path)
    readme = (
        "Mustatil R-CNN training result\n"
        f"Architecture: {architecture}\n"
        f"Backbone: {backbone}\n"
        f"Model: {model_kind}\n"
        f"Classes: {', '.join(classes)}\n"
        f"Final checkpoint: {final_path.name}\n"
        "Use this .pth in a detection loader with the same architecture/backbone.\n"
    )
    (run_dir / "README_training_result.txt").write_text(readme, encoding="utf-8")
    log(f"Training complete. Final checkpoint: {final_path}")
    return last_ckpt



def _ergonomic_build_training_tab(ws: Any):
    target = _find_target_module()
    if target is None:
        raise RuntimeError("Original Mustatil R-CNN trainer plugin was not found.")

    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel, QLineEdit,
        QPushButton, QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
        QTextEdit, QSplitter, QProgressBar, QScrollArea, QFrame,
    )

    project = _project_guess(ws)
    image_guess = _guess_image_dir(project)
    label_guess = _guess_label_dir(project)
    out_guess = project / "runs" / "rcnn_training"

    page = QWidget()
    page._mustatil_rcnn_custom_backbone_creator_patch = True
    root = QVBoxLayout(page)
    root.setContentsMargins(8, 8, 8, 8)
    root.setSpacing(6)

    header = QLabel(
        "R-CNN Training Studio — Dataset → Model/Backbone → Training Strategy → Checkpoint/ONNX. "
        "The PyTorch checkpoint always contains backbone + RPN + ROI heads together. ONNX export is optional and inference-only."
    )
    header.setWordWrap(True)
    root.addWidget(header)

    main_split = QSplitter(Qt.Vertical)
    root.addWidget(main_split, 1)

    top_split = QSplitter(Qt.Horizontal)
    main_split.addWidget(top_split)

    def make_scroll(title: str):
        outer = QScrollArea()
        outer.setWidgetResizable(True)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(8)
        outer.setWidget(inner)
        outer.setObjectName(title)
        return outer, inner, lay

    data_scroll, data_inner, data_lay = make_scroll("DatasetWorkflowColumn")
    model_scroll, model_inner, model_lay = make_scroll("ModelBackboneWorkflowColumn")
    train_scroll, train_inner, train_lay = make_scroll("TrainingStrategyWorkflowColumn")
    top_split.addWidget(data_scroll)
    top_split.addWidget(model_scroll)
    top_split.addWidget(train_scroll)
    try:
        data_scroll.setMinimumWidth(360)
        model_scroll.setMinimumWidth(460)
        train_scroll.setMinimumWidth(430)
        top_split.setSizes([390, 520, 470])
    except Exception:
        pass

    console_box = QGroupBox("Live Training Console / Workspace Log Mirror")
    console_lay = QVBoxLayout(console_box)
    console_lay.setContentsMargins(8, 8, 8, 8)
    console_lay.setSpacing(5)
    console_top = QHBoxLayout()
    progress = QProgressBar()
    progress.setRange(0, 100)
    progress.setValue(0)
    status_label = QLabel("Idle")
    status_label.setMinimumWidth(190)
    clear_btn = QPushButton("Clear")
    console_top.addWidget(progress, 1)
    console_top.addWidget(status_label)
    console_top.addWidget(clear_btn)
    log_edit = QTextEdit()
    log_edit.setReadOnly(True)
    log_edit.setMinimumHeight(190)
    console_lay.addLayout(console_top)
    console_lay.addWidget(log_edit, 1)
    main_split.addWidget(console_box)
    try:
        main_split.setSizes([650, 250])
    except Exception:
        pass

    def _update_progress_from_text(text: str):
        try:
            m = re.search(r"Epoch\s+(\d+)\s*/\s*(\d+)\s+step\s+(\d+)\s*/\s*(\d+)", text, re.I)
            if m:
                ep, eps, st, steps = [int(x) for x in m.groups()]
                pct = int(round((((ep - 1) + (st / max(1, steps))) / max(1, eps)) * 100.0))
                progress.setValue(max(0, min(100, pct)))
                status_label.setText(f"Epoch {ep}/{eps} · step {st}/{steps}")
                return
            m = re.search(r"Epoch\s+(\d+)\s*/\s*(\d+)\s+finished", text, re.I)
            if m:
                ep, eps = [int(x) for x in m.groups()]
                progress.setValue(max(0, min(100, int(round((ep / max(1, eps)) * 100.0)))))
                status_label.setText(f"Epoch {ep}/{eps} finished")
                return
            if "ONNX export started" in text:
                status_label.setText("Exporting ONNX")
            elif "Training complete" in text or "Training finished successfully" in text:
                progress.setValue(100)
                status_label.setText("Training complete")
            elif "Training failed" in text:
                status_label.setText("Training failed")
        except Exception:
            pass

    def append(msg: str):
        text = str(msg)
        try:
            print("[Mustatil R-CNN Training Studio] " + text, flush=True)
        except Exception:
            pass
        try:
            ws.log(text)
        except Exception:
            pass
        def ui_append(t=text):
            try:
                log_edit.append(t)
                _update_progress_from_text(t)
            except Exception:
                pass
        try:
            QTimer.singleShot(0, ui_append)
        except Exception:
            ui_append()

    clear_btn.clicked.connect(lambda: log_edit.clear())

    def browse_dir(line: QLineEdit):
        base = line.text().strip() or str(project)
        d = QFileDialog.getExistingDirectory(page, "Select folder", base)
        if d:
            line.setText(d)

    def browse_file(line: QLineEdit, caption="Select file", file_filter="All files (*)"):
        base = line.text().strip() or str(project)
        fn, _ = QFileDialog.getOpenFileName(page, caption, base, file_filter)
        if fn:
            line.setText(fn)

    def path_row(line: QLineEdit, mode="dir", file_filter="All files (*)"):
        row = QWidget()
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(4)
        hl.addWidget(line, 1)
        b = QPushButton("…")
        b.setFixedWidth(32)
        if mode == "dir":
            b.clicked.connect(lambda _=False, le=line: browse_dir(le))
        else:
            b.clicked.connect(lambda _=False, le=line: browse_file(le, "Select file", file_filter))
        hl.addWidget(b)
        return row

    def section_title(text: str):
        lab = QLabel(text)
        lab.setWordWrap(True)
        try:
            lab.setStyleSheet("font-weight: 600;")
        except Exception:
            pass
        return lab

    # ------------------------------------------------------------------
    # Column 1: Dataset / classes / validation
    # ------------------------------------------------------------------
    dataset_box = QGroupBox("1 · Dataset and labels")
    df = QFormLayout(dataset_box)
    df.setLabelAlignment(Qt.AlignRight)
    project_line = QLineEdit(str(project))
    image_line = QLineEdit(str(image_guess))
    label_line = QLineEdit(str(label_guess))
    out_line = QLineEdit(str(out_guess))
    classes_line = QLineEdit(", ".join(list(getattr(getattr(ws, "project_state", None), "classes", []) or ["mustatil", "false_positive"])))
    df.addRow("Project", path_row(project_line, "dir"))
    df.addRow("Images", path_row(image_line, "dir"))
    df.addRow("Labels", path_row(label_line, "dir"))
    df.addRow("Output", path_row(out_line, "dir"))
    df.addRow("Classes", classes_line)
    data_lay.addWidget(dataset_box)

    dataset_note = QLabel(
        "Labels are YOLO TXT files. TorchVision reserves class 0 for background, so YOLO class 0 becomes R-CNN label 1. "
        "Keep class order identical for training and detection."
    )
    dataset_note.setWordWrap(True)
    data_lay.addWidget(dataset_note)

    validate_btn = QPushButton("Validate dataset now")
    validate_btn.setMinimumHeight(36)
    data_lay.addWidget(validate_btn)

    quick_preset_box = QGroupBox("2 · Practical presets")
    qf = QFormLayout(quick_preset_box)
    preset_combo = QComboBox()
    preset_combo.addItems([
        "RTX 2080 safe start",
        "CPU / very low VRAM",
        "Accuracy test",
        "Custom manual settings",
    ])
    apply_preset_btn = QPushButton("Apply preset")
    qf.addRow("Preset", preset_combo)
    qf.addRow("", apply_preset_btn)
    data_lay.addWidget(quick_preset_box)
    data_lay.addStretch(1)

    # ------------------------------------------------------------------
    # Column 2: Model / backbone / creator
    # ------------------------------------------------------------------
    model_box = QGroupBox("3 · Model and backbone")
    mf = QFormLayout(model_box)
    mf.setLabelAlignment(Qt.AlignRight)
    architecture_combo = QComboBox()
    architecture_combo.addItems(["Faster R-CNN", "Mask R-CNN"])
    backbone_combo = QComboBox()
    backbone_combo.addItems(BACKBONE_LABELS)
    weights_combo = QComboBox()
    weights_combo.addItems(["COCO/default pretrained", "random init", "custom checkpoint"])
    ckpt_line = QLineEdit("")
    ckpt_line.setPlaceholderText("Optional .pth/.pt checkpoint")
    ckpt_row = path_row(ckpt_line, "file", "PyTorch checkpoints (*.pth *.pt);;All files (*)")
    custom_backbone_line = QLineEdit("")
    custom_backbone_line.setPlaceholderText("Only needed for Custom Python Backbone")
    custom_out_spin = QSpinBox(); custom_out_spin.setRange(16, 4096); custom_out_spin.setSingleStep(16); custom_out_spin.setValue(256)
    min_size_spin = QSpinBox(); min_size_spin.setRange(128, 8192); min_size_spin.setSingleStep(32); min_size_spin.setValue(800)
    max_size_spin = QSpinBox(); max_size_spin.setRange(128, 16384); max_size_spin.setSingleStep(32); max_size_spin.setValue(1333)
    anchor_combo = QComboBox(); anchor_combo.addItems(list(ANCHOR_PROFILES.keys()))
    compat_label = QLabel("")
    compat_label.setWordWrap(True)
    try:
        compat_label.setStyleSheet("color: #666;")
    except Exception:
        pass

    mf.addRow("Architecture", architecture_combo)
    mf.addRow("Backbone", backbone_combo)
    mf.addRow("Weights", weights_combo)
    mf.addRow("Checkpoint", ckpt_row)
    mf.addRow("Custom .py", path_row(custom_backbone_line, "file", "Python files (*.py);;All files (*)"))
    mf.addRow("Custom out channels", custom_out_spin)
    mf.addRow("Min image size", min_size_spin)
    mf.addRow("Max image size", max_size_spin)
    mf.addRow("Anchors", anchor_combo)
    mf.addRow("Compatibility", compat_label)
    model_lay.addWidget(model_box)

    creator_box = QGroupBox("4 · Custom backbone creator")
    cf = QFormLayout(creator_box)
    cf.setLabelAlignment(Qt.AlignRight)
    creator_template_combo = QComboBox()
    creator_template_combo.addItems(CUSTOM_BACKBONE_TEMPLATES)
    creator_name_line = QLineEdit("mustatil_small_fpn_backbone")
    creator_folder_line = QLineEdit(str(project / "custom_backbones"))
    creator_width_spin = QSpinBox(); creator_width_spin.setRange(16, 256); creator_width_spin.setSingleStep(16); creator_width_spin.setValue(64)
    create_select_btn = QPushButton("Create + select")
    create_only_btn = QPushButton("Create only")
    test_custom_btn = QPushButton("Test selected")
    open_creator_btn = QPushButton("Open folder")
    creator_note = QLabel(
        "Creates a trainable FPN-style PyTorch backbone. It is not pretrained; use more epochs/data than with ResNet50."
    )
    creator_note.setWordWrap(True)
    creator_buttons = QWidget()
    creator_buttons_lay = QHBoxLayout(creator_buttons)
    creator_buttons_lay.setContentsMargins(0, 0, 0, 0)
    creator_buttons_lay.addWidget(create_select_btn)
    creator_buttons_lay.addWidget(create_only_btn)
    creator_buttons_lay.addWidget(test_custom_btn)
    cf.addRow("Template", creator_template_combo)
    cf.addRow("File name", creator_name_line)
    cf.addRow("Folder", path_row(creator_folder_line, "dir"))
    cf.addRow("Base width", creator_width_spin)
    cf.addRow("", creator_buttons)
    cf.addRow("", open_creator_btn)
    cf.addRow("Note", creator_note)
    model_lay.addWidget(creator_box)
    model_lay.addStretch(1)

    # ------------------------------------------------------------------
    # Column 3: Training strategy / freeze / ONNX
    # ------------------------------------------------------------------
    strategy_box = QGroupBox("5 · Training strategy")
    sf = QFormLayout(strategy_box)
    sf.setLabelAlignment(Qt.AlignRight)
    device_combo = QComboBox()
    device_combo.setEditable(True)
    device_combo.addItems(["auto", "cpu", "cuda"])
    strategy_combo = QComboBox()
    strategy_combo.addItems([
        "Full model: train backbone + RPN + ROI heads",
        "Freeze backbone: train RPN/ROI heads only",
        "Warmup heads first, then unfreeze backbone",
    ])
    warmup_spin = QSpinBox(); warmup_spin.setRange(0, 1000); warmup_spin.setValue(3)
    trainable_layers_spin = QSpinBox(); trainable_layers_spin.setRange(0, 5); trainable_layers_spin.setValue(3)
    freeze_chk = QCheckBox("Force freeze whole backbone")
    freeze_chk.setChecked(False)
    mask_combo = QComboBox()
    mask_combo.addItems(["box masks from YOLO boxes", "SAM2/JSON polygons if available, else boxes"])
    epochs_spin = QSpinBox(); epochs_spin.setRange(1, 10000); epochs_spin.setValue(30)
    batch_spin = QSpinBox(); batch_spin.setRange(1, 32); batch_spin.setValue(1)
    lr_spin = QDoubleSpinBox(); lr_spin.setDecimals(6); lr_spin.setRange(0.000001, 1.0); lr_spin.setSingleStep(0.0005); lr_spin.setValue(0.005)
    wd_spin = QDoubleSpinBox(); wd_spin.setDecimals(6); wd_spin.setRange(0.0, 1.0); wd_spin.setSingleStep(0.0001); wd_spin.setValue(0.0005)
    workers_spin = QSpinBox(); workers_spin.setRange(0, 16); workers_spin.setValue(0)
    maximg_spin = QSpinBox(); maximg_spin.setRange(0, 10000000); maximg_spin.setValue(0); maximg_spin.setToolTip("0 = all images")
    save_every_spin = QSpinBox(); save_every_spin.setRange(1, 1000); save_every_spin.setValue(1)
    include_empty_chk = QCheckBox("Include empty/negative images")
    include_empty_chk.setChecked(True)

    sf.addRow("Device", device_combo)
    sf.addRow("Strategy", strategy_combo)
    sf.addRow("Warmup epochs", warmup_spin)
    sf.addRow("Trainable ResNet layers", trainable_layers_spin)
    sf.addRow("", freeze_chk)
    sf.addRow("Mask source", mask_combo)
    sf.addRow("Epochs", epochs_spin)
    sf.addRow("Batch", batch_spin)
    sf.addRow("Learning rate", lr_spin)
    sf.addRow("Weight decay", wd_spin)
    sf.addRow("Workers", workers_spin)
    sf.addRow("Max images", maximg_spin)
    sf.addRow("Save every", save_every_spin)
    sf.addRow("", include_empty_chk)
    train_lay.addWidget(strategy_box)

    onnx_box = QGroupBox("6 · Optional export")
    of = QFormLayout(onnx_box)
    of.setLabelAlignment(Qt.AlignRight)
    export_onnx_chk = QCheckBox("Try ONNX export after training")
    export_onnx_chk.setChecked(False)
    onnx_opset_spin = QSpinBox(); onnx_opset_spin.setRange(11, 20); onnx_opset_spin.setValue(17)
    onnx_img_size_spin = QSpinBox(); onnx_img_size_spin.setRange(128, 4096); onnx_img_size_spin.setSingleStep(32); onnx_img_size_spin.setValue(640)
    onnx_note = QLabel(
        "PyTorch .pth is the main training result. ONNX combines backbone + RPN + heads for inference only. "
        "TorchVision detection ONNX export can fail depending on installed torch/torchvision; training remains valid."
    )
    onnx_note.setWordWrap(True)
    of.addRow("", export_onnx_chk)
    of.addRow("Opset", onnx_opset_spin)
    of.addRow("Dummy image size", onnx_img_size_spin)
    of.addRow("Note", onnx_note)
    train_lay.addWidget(onnx_box)

    train_btn = QPushButton("Start R-CNN Training")
    train_btn.setMinimumHeight(44)
    train_lay.addWidget(train_btn)
    train_lay.addStretch(1)

    def _creator_target_path() -> Path:
        folder = Path(creator_folder_line.text().strip() or str(project / "custom_backbones")).expanduser()
        name = _safe_name(creator_name_line.text().strip() or "mustatil_small_fpn_backbone")
        if not name.lower().endswith(".py"):
            name += ".py"
        return folder / name

    def create_custom_backbone(select_after: bool = True):
        try:
            path = _creator_target_path()
            _write_custom_backbone_template(
                path,
                creator_template_combo.currentText(),
                out_channels=custom_out_spin.value(),
                width=creator_width_spin.value(),
            )
            append("Custom backbone file created: " + str(path))
            if select_after:
                custom_backbone_line.setText(str(path))
                ix = backbone_combo.findText("Custom Python Backbone (.py)")
                if ix >= 0:
                    backbone_combo.setCurrentIndex(ix)
                update_ui_notes()
                append("Custom backbone selected for training.")
        except Exception as exc:
            append("Custom backbone creation failed: " + str(exc))
            append(traceback.format_exc())

    def test_selected_custom_backbone():
        try:
            path = custom_backbone_line.text().strip() or str(_creator_target_path())
            bb = _load_custom_backbone(
                path,
                pretrained=False,
                trainable_layers=trainable_layers_spin.value(),
                out_channels=custom_out_spin.value(),
            )
            count = _count_backbone_feature_maps(bb)
            try:
                import torch
                was_training = bool(getattr(bb, "training", False))
                bb.eval()
                with torch.no_grad():
                    out = bb(torch.zeros(1, 3, 256, 256))
                bb.train(was_training)
                if isinstance(out, dict):
                    shapes = [str(k) + ":" + "x".join(map(str, tuple(v.shape))) for k, v in out.items()]
                elif isinstance(out, (list, tuple)):
                    shapes = ["%d:%s" % (i, "x".join(map(str, tuple(v.shape)))) for i, v in enumerate(out)]
                else:
                    shapes = ["tensor:" + "x".join(map(str, tuple(out.shape)))]
                append("Custom backbone test OK: feature_maps=%s | %s" % (count, "; ".join(shapes)))
            except Exception:
                append("Custom backbone test OK: feature_maps=%s" % count)
        except Exception as exc:
            append("Custom backbone test failed: " + str(exc))
            append(traceback.format_exc())

    def open_custom_backbone_folder():
        try:
            folder = Path(creator_folder_line.text().strip() or str(project / "custom_backbones")).expanduser()
            folder.mkdir(parents=True, exist_ok=True)
            if sys.platform.startswith("win"):
                os.startfile(str(folder))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                import subprocess; subprocess.Popen(["open", str(folder)])
            else:
                import subprocess; subprocess.Popen(["xdg-open", str(folder)])
        except Exception as exc:
            append("Open folder failed: " + str(exc))

    create_select_btn.clicked.connect(lambda: create_custom_backbone(True))
    create_only_btn.clicked.connect(lambda: create_custom_backbone(False))
    test_custom_btn.clicked.connect(test_selected_custom_backbone)
    open_creator_btn.clicked.connect(open_custom_backbone_folder)

    def _strategy_id() -> str:
        txt = strategy_combo.currentText().lower()
        if "freeze" in txt:
            return "freeze_backbone"
        if "warmup" in txt or "unfreeze" in txt:
            return "warmup_then_unfreeze"
        return "full_model"

    def update_ui_notes():
        arch = architecture_combo.currentText()
        bb = BACKBONE_ID.get(backbone_combo.currentText(), "resnet50_fpn")
        strat = _strategy_id()
        warmup_spin.setEnabled(strat == "warmup_then_unfreeze")
        freeze_chk.setEnabled(strat != "full_model")
        if arch.startswith("Mask") and bb.startswith("mobilenet"):
            compat_label.setText("Mask R-CNN with MobileNet uses a custom MaskRCNN wrapper. ResNet50-FPN is safer for production.")
        elif bb == "custom_python_backbone":
            compat_label.setText("Custom mode uses your .py. It starts without ImageNet pretraining unless your file loads its own weights.")
        elif bb != "resnet50_fpn":
            compat_label.setText("Training works, but detection must rebuild the same backbone when loading this .pth.")
        else:
            compat_label.setText("Most compatible choice. Existing ResNet50-FPN detection loaders usually support this.")
    architecture_combo.currentTextChanged.connect(lambda _=None: update_ui_notes())
    backbone_combo.currentTextChanged.connect(lambda _=None: update_ui_notes())
    strategy_combo.currentTextChanged.connect(lambda _=None: update_ui_notes())
    update_ui_notes()

    def apply_preset():
        p = preset_combo.currentText().lower()
        if "2080" in p:
            device_combo.setCurrentText("cuda")
            batch_spin.setValue(1)
            epochs_spin.setValue(80)
            backbone_combo.setCurrentText("ResNet50-FPN (standard, recommended)")
            strategy_combo.setCurrentText("Warmup heads first, then unfreeze backbone")
            warmup_spin.setValue(3)
            anchor_combo.setCurrentText("Small objects / satellite structures")
            min_size_spin.setValue(800); max_size_spin.setValue(1333)
        elif "cpu" in p or "low" in p:
            device_combo.setCurrentText("cpu")
            batch_spin.setValue(1)
            epochs_spin.setValue(20)
            backbone_combo.setCurrentText("MobileNetV3-Large-320-FPN (very fast / small images)")
            strategy_combo.setCurrentText("Freeze backbone: train RPN/ROI heads only")
            min_size_spin.setValue(512); max_size_spin.setValue(800)
        elif "accuracy" in p:
            device_combo.setCurrentText("cuda")
            batch_spin.setValue(1)
            epochs_spin.setValue(120)
            backbone_combo.setCurrentText("ResNet101-FPN (stronger, slower)")
            strategy_combo.setCurrentText("Full model: train backbone + RPN + ROI heads")
            anchor_combo.setCurrentText("Small objects / satellite structures")
            min_size_spin.setValue(1024); max_size_spin.setValue(1600)
        update_ui_notes()
    apply_preset_btn.clicked.connect(apply_preset)

    def cfg() -> Dict[str, Any]:
        arch = "maskrcnn" if "mask" in architecture_combo.currentText().lower() else "fasterrcnn"
        backbone = BACKBONE_ID.get(backbone_combo.currentText(), "resnet50_fpn")
        model_kind = f"{arch}_{backbone}"
        wtxt = weights_combo.currentText().lower()
        return {
            "project_dir": project_line.text().strip(),
            "image_dir": image_line.text().strip(),
            "label_dir": label_line.text().strip(),
            "out_dir": out_line.text().strip(),
            "classes": _parse_classes(classes_line.text()),
            "architecture": arch,
            "backbone": backbone,
            "model_kind": model_kind,
            "device": device_combo.currentText().strip(),
            "pretrained": "random" not in wtxt,
            "checkpoint_path": ckpt_line.text().strip() if "custom" in wtxt else "",
            "freeze_backbone": bool(freeze_chk.isChecked() or _strategy_id() == "freeze_backbone"),
            "training_strategy": _strategy_id(),
            "warmup_epochs": warmup_spin.value(),
            "trainable_backbone_layers": trainable_layers_spin.value(),
            "mask_mode": "sam2_polygons" if "sam2" in mask_combo.currentText().lower() else "boxes",
            "epochs": epochs_spin.value(),
            "batch": batch_spin.value(),
            "lr": lr_spin.value(),
            "weight_decay": wd_spin.value(),
            "workers": workers_spin.value(),
            "max_images": maximg_spin.value(),
            "include_empty": include_empty_chk.isChecked(),
            "save_every": save_every_spin.value(),
            "min_size": min_size_spin.value(),
            "max_size": max_size_spin.value(),
            "anchor_profile": anchor_combo.currentText(),
            "custom_backbone_path": custom_backbone_line.text().strip(),
            "custom_out_channels": custom_out_spin.value(),
            "export_onnx_after_training": export_onnx_chk.isChecked(),
            "onnx_opset": onnx_opset_spin.value(),
            "onnx_img_size": onnx_img_size_spin.value(),
        }

    def validate_clicked():
        append("Validating dataset...")
        try:
            target._validate_dataset(cfg(), logger=append)
        except Exception as exc:
            append("Validation failed: " + str(exc))
            append(traceback.format_exc())

    def train_clicked():
        if getattr(page, "_mustatil_rcnn_training_running", False):
            append("Training is already running; duplicate click ignored.")
            return
        c = cfg()
        progress.setValue(0)
        status_label.setText("Starting")
        append("Start button clicked. Launching R-CNN training worker...")
        append("Config: " + json.dumps({k: v for k, v in c.items() if k not in {"classes"}}, ensure_ascii=False))
        page._mustatil_rcnn_training_running = True
        try:
            train_btn.setEnabled(False)
            train_btn.setText("Training running...")
        except Exception:
            pass

        def job():
            try:
                result = _advanced_train_rcnn_worker(c, logger=append)
                append("Training finished successfully: " + str(result))
            except Exception as exc:
                append("Training failed: " + str(exc))
                try:
                    append(traceback.format_exc())
                except Exception:
                    pass
                raise
            finally:
                page._mustatil_rcnn_training_running = False
                def reset_button():
                    try:
                        train_btn.setEnabled(True)
                        train_btn.setText("Start R-CNN Training")
                    except Exception:
                        pass
                try:
                    QTimer.singleShot(0, reset_button)
                except Exception:
                    reset_button()

        try:
            ws.run_task("R-CNN Training Studio " + _now_stamp(), job)
        except Exception as exc:
            append("Mustatil run_task unavailable, starting direct Python thread: " + str(exc))
            threading.Thread(target=job, daemon=True).start()

    validate_btn.clicked.connect(validate_clicked)
    train_btn.clicked.connect(train_clicked)
    return page

def _advanced_install_trainer_tab(ws: Any) -> bool:
    target = _find_target_module()
    if ws is None or target is None:
        return False
    try:
        tabs = getattr(ws, "tabs", None)
        if tabs is None:
            return False
        labels = [str(tabs.tabText(i)).strip().lower() for i in range(tabs.count())]
        if not any(("trainer" in x or "training" in x) for x in labels):
            return False

        current = None
        for i in range(tabs.count()):
            if str(tabs.tabText(i)).strip().lower() == "r-cnn trainer":
                current = i
                break

        if current is not None:
            old_page = tabs.widget(current)
            if getattr(old_page, "_mustatil_rcnn_custom_backbone_creator_patch", False):
                try:
                    target._INSTALLED_WS.add(id(ws))
                    target._move_tab_after_anchor_if_needed(ws)
                except Exception:
                    pass
                return False
            # Replace older/simple R-CNN page with the advanced ergonomic page.
            try:
                tabs.removeTab(current)
                old_page.setParent(None)
                old_page.deleteLater()
            except Exception:
                pass
            insert_at = current
        else:
            anchor = target._find_training_anchor_index(tabs) if hasattr(target, "_find_training_anchor_index") else None
            insert_at = tabs.count() if anchor is None else int(anchor) + 1

        page = _ergonomic_build_training_tab(ws)
        try:
            target._INSTALLING_TAB = True
            tabs.insertTab(insert_at, page, "R-CNN Trainer")
        finally:
            try:
                target._INSTALLING_TAB = False
            except Exception:
                pass
        try:
            target._INSTALLED_WS.add(id(ws))
            target._move_tab_after_anchor_if_needed(ws)
        except Exception:
            pass
        try:
            ws.log("R-CNN Trainer advanced backbone + custom creator UI installed.")
        except Exception:
            pass
        return True
    except Exception as exc:
        _log("install trainer tab failed: " + str(exc))
        try:
            traceback.print_exc()
        except Exception:
            pass
        try:
            target._INSTALLING_TAB = False
        except Exception:
            pass
        return False


def _patch_target() -> bool:
    global _PATCHED
    target = _find_target_module()
    if target is None:
        return False
    try:
        # Keep originals once, so this remains reversible while debugging.
        if not hasattr(target, "_orig_build_training_tab_before_advanced_patch"):
            target._orig_build_training_tab_before_advanced_patch = target._build_training_tab
        if not hasattr(target, "_orig_build_detection_model_before_advanced_patch"):
            target._orig_build_detection_model_before_advanced_patch = target._build_detection_model
        if not hasattr(target, "_orig_train_rcnn_worker_before_advanced_patch"):
            target._orig_train_rcnn_worker_before_advanced_patch = target._train_rcnn_worker
        if not hasattr(target, "_orig_install_trainer_tab_before_advanced_patch"):
            target._orig_install_trainer_tab_before_advanced_patch = target._install_trainer_tab

        target._build_training_tab = _ergonomic_build_training_tab
        target._build_detection_model = _advanced_build_detection_model
        target._train_rcnn_worker = _advanced_train_rcnn_worker
        target._install_trainer_tab = _advanced_install_trainer_tab
        target._MUSTATIL_ADVANCED_BACKBONE_PATCH = True
        _PATCHED = True
        _log("Patched original R-CNN trainer module: " + getattr(target, "__name__", "<unknown>"))
        return True
    except Exception as exc:
        _log("patch failed: " + str(exc))
        return False


def _schedule_retry_patch():
    try:
        from PySide6.QtCore import QTimer
        for delay in (250, 1000, 2500, 5000):
            QTimer.singleShot(delay, lambda: (_patch_target(), _scan_after_patch()))
    except Exception:
        pass


def _scan_after_patch():
    target = _find_target_module()
    if target is None:
        return
    try:
        if hasattr(target, "_scan_for_workspaces"):
            target._scan_for_workspaces()
    except Exception:
        pass


def mustatil_plugin_init():
    if not _patch_target():
        _schedule_retry_patch()
    _scan_after_patch()


try:
    if not _patch_target():
        _schedule_retry_patch()
    _scan_after_patch()
except Exception as exc:
    _log("startup warning: " + str(exc))
