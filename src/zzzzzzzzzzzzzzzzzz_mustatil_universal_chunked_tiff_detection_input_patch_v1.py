#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mustatil plugin patch: AI-Pipeline-style visual Custom Backbone Creator with draggable graph connections for the R-CNN Trainer.

Drop this file into Mustatil's mustatil_plugins folder and restart Mustatil.

What it adds
------------
- A top toolbar button in the R-CNN Trainer tab: "Backbone Creator".
- A block-based dialog where a custom Faster/Mask R-CNN backbone can be assembled.
- Blocks can be added, moved, duplicated, deleted and exported as a Python backbone file.
- Canvas supports left/middle mouse panning and right-click deletion of visual connections.
- Extra experimental blocks/presets help test style-robust satellite backbones.
- Code tab supports editable visual block specs with auto-apply, plus manual Python backbone export.
- The generated file implements the expected contract:
      def build_backbone(pretrained=False, trainable_layers=3, out_channels=256): ...
- The generated backbone returns an OrderedDict of FPN feature maps and sets backbone.out_channels.
- The created .py can be inserted automatically into the existing Custom Python Backbone field.

This is an add-on. Keep the original R-CNN trainer plugin and the R-CNN Training Studio patch installed.
"""
from __future__ import annotations

import json
import pprint
import ast
import os
import sys
import time
import traceback
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

_PATCHED_QTAB = False
_ORIG_ADD_TAB = None
_ORIG_INSERT_TAB = None
_SCAN_TIMER = None
_PATCHED_PAGES: set[int] = set()

PLUGIN_TITLE = "Mustatil R-CNN Backbone Visual Creator v6 Logic Gates Ergonomic"
CUSTOM_BACKBONE_LABEL = "Custom Python Backbone (.py)"

DEFAULT_BLOCKS: List[Dict[str, Any]] = [
    {"type": "Stem Conv", "out": 32, "stride": 2, "repeats": 1, "feature": True},
    {"type": "Residual Block", "out": 64, "stride": 2, "repeats": 2, "feature": True},
    {"type": "Depthwise Separable", "out": 128, "stride": 2, "repeats": 2, "feature": True},
    {"type": "Residual Block", "out": 256, "stride": 2, "repeats": 3, "feature": True},
    {"type": "Bottleneck Block", "out": 512, "stride": 2, "repeats": 2, "feature": True},
]

BLOCK_TYPES = [
    "Stem Conv",
    "Conv Block",
    "Residual Block",
    "Bottleneck Block",
    "Depthwise Separable",
    "Downsample Conv",
    "SE Attention",
    "SPP Block",
    "Style Adapter",
    "Dilated Context",
    "Feature Gate",
    "Residual Add Gate",
    "Concat Fusion Gate",
    "Weighted Sum Gate",
    "Soft AND Gate",
    "Soft OR Gate",
    "Soft NOT Gate",
    "Soft XOR Gate",
    "Mixture-of-Experts Gate",
    "Style Gate",
    "Scale Mix Gate",
    "Skip Bridge",
    "Iterative Refinement",
]


def _log(msg: str) -> None:
    try:
        print(f"[{PLUGIN_TITLE}] {msg}", flush=True)
    except Exception:
        pass


def _workspace_from_widget(widget: Any) -> Optional[Any]:
    cur = widget
    for _ in range(160):
        if cur is None:
            break
        try:
            if hasattr(cur, "tabs") and (hasattr(cur, "project") or hasattr(cur, "project_state")):
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


def _var_get(v: Any, default: str = "") -> str:
    try:
        if hasattr(v, "get"):
            x = v.get()
            if x is not None:
                return str(x)
    except Exception:
        pass
    return str(default or "")


def _project_guess(ws: Any) -> Path:
    for attr in ("project", "project_create_dir"):
        try:
            p = Path(_var_get(getattr(ws, attr, None), "").strip()).expanduser()
            if str(p) and p.exists():
                return p
        except Exception:
            pass
    try:
        p = Path(getattr(getattr(ws, "project_state", None), "project_root", "") or "").expanduser()
        if str(p) and p.exists():
            return p
    except Exception:
        pass
    return Path.home() / "Mustatil_Custom_Backbones"


def _safe_name(text: str) -> str:
    bad = '<>:"/\\|?*\n\r\t '
    out = "".join("_" if c in bad else c for c in str(text or ""))
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "mustatil_block_backbone"


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _normalize_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize a visual backbone graph specification."""
    out: List[Dict[str, Any]] = []
    for b in list(blocks or []):
        typ = str(b.get("type", "Conv Block"))
        if typ not in BLOCK_TYPES:
            typ = "Conv Block"
        try:
            channels = int(b.get("out", 64))
        except Exception:
            channels = 64
        try:
            stride = int(b.get("stride", 1))
        except Exception:
            stride = 1
        try:
            repeats = int(b.get("repeats", 1))
        except Exception:
            repeats = 1
        channels = max(4, min(4096, channels))
        stride = 2 if stride >= 2 else 1
        repeats = max(1, min(32, repeats))
        raw_inputs = b.get("inputs", [])
        if isinstance(raw_inputs, str):
            raw_inputs = [raw_inputs] if raw_inputs.strip() else []
        inputs = []
        seen = set()
        if isinstance(raw_inputs, (list, tuple)):
            for ref in raw_inputs:
                ref = str(ref or "").strip()
                if ref and ref not in seen:
                    seen.add(ref); inputs.append(ref)
        item = {
            "type": typ,
            "out": channels,
            "stride": stride,
            "repeats": repeats,
            "feature": bool(b.get("feature", False)),
            "inputs": inputs,
        }
        for key in ("id", "name", "x", "y", "enabled"):
            if key in b:
                item[key] = b[key]
        out.append(item)
    if not out:
        out = [dict(x) for x in DEFAULT_BLOCKS]
    if not any(b.get("feature") and bool(b.get("enabled", True)) for b in out):
        out[-1]["feature"] = True
    return out


def _render_backbone_code(
    blocks: List[Dict[str, Any]],
    class_name: str,
    default_out_channels: int = 256,
    norm_kind: str = "GroupNorm",
    activation: str = "SiLU",
    note: str = "Generated by Mustatil Backbone Block Creator",
) -> str:
    blocks = _normalize_blocks(blocks)
    class_name = _safe_name(class_name)
    if not class_name[0].isalpha():
        class_name = "Mustatil" + class_name
    class_name = "".join(part[:1].upper() + part[1:] for part in class_name.split("_"))
    default_out_channels = int(max(16, min(1024, default_out_channels)))
    norm_kind = str(norm_kind or "GroupNorm")
    activation = str(activation or "SiLU")
    spec_json = pprint.pformat(blocks, width=120, sort_dicts=False)
    code = RUNTIME_BACKBONE_TEMPLATE
    code = code.replace("@@CLASS_NAME@@", class_name)
    code = code.replace("@@SPEC_JSON@@", spec_json)
    code = code.replace("@@DEFAULT_OUT_CHANNELS@@", str(default_out_channels))
    code = code.replace("@@NORM_KIND@@", norm_kind)
    code = code.replace("@@ACTIVATION@@", activation)
    code = code.replace("@@NOTE@@", note.replace('"', "'"))
    return code


RUNTIME_BACKBONE_TEMPLATE = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@@NOTE@@

This file is a Mustatil / TorchVision custom R-CNN backbone generated from a visual graph.
It can be used in the R-CNN Training Studio with "Custom Python Backbone (.py)".

Contract:
    def build_backbone(pretrained=False, trainable_layers=3, out_channels=256):
        returns nn.Module with .out_channels and OrderedDict feature maps.

Graph notes:
- A block can have one input, several inputs, or no input.
- No-input blocks start from the RGB image, so multiple starts are supported.
- Multiple-input blocks are fused by trainable gate/fusion modules, including soft differentiable logic gates.
- The graph is evaluated in CUSTOM_BLOCK_SPEC order.
- Use FPN output on 3-5 important levels for Faster/Mask R-CNN.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

import torch
from torch import nn
import torch.nn.functional as F

CUSTOM_BLOCK_SPEC = @@SPEC_JSON@@
DEFAULT_OUT_CHANNELS = @@DEFAULT_OUT_CHANNELS@@
DEFAULT_NORM = "@@NORM_KIND@@"
DEFAULT_ACTIVATION = "@@ACTIVATION@@"


def _make_activation(name: str):
    name = str(name or "SiLU").lower()
    if name == "relu": return nn.ReLU(inplace=True)
    if name == "gelu": return nn.GELU()
    if name in {"leakyrelu", "leaky_relu", "leaky relu"}: return nn.LeakyReLU(0.1, inplace=True)
    return nn.SiLU(inplace=True)


def _make_norm(channels: int, kind: str = DEFAULT_NORM):
    kind = str(kind or "GroupNorm").lower(); channels = int(channels)
    if kind in {"batchnorm", "batch_norm", "batch norm", "bn"}: return nn.BatchNorm2d(channels)
    if kind in {"instancenorm", "instance_norm", "instance norm", "in"}: return nn.InstanceNorm2d(channels, affine=True)
    groups = min(32, channels)
    while groups > 1 and channels % groups != 0: groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, norm: str = DEFAULT_NORM, act: str = DEFAULT_ACTIVATION):
        super().__init__()
        pad = int(kernel_size) // 2
        self.block = nn.Sequential(
            nn.Conv2d(int(in_ch), int(out_ch), int(kernel_size), stride=int(stride), padding=pad, bias=False),
            _make_norm(int(out_ch), norm), _make_activation(act),
        )
    def forward(self, x): return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, norm: str = DEFAULT_NORM, act: str = DEFAULT_ACTIVATION):
        super().__init__()
        self.conv1 = ConvNormAct(in_ch, out_ch, 3, stride, norm, act)
        self.conv2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), _make_norm(out_ch, norm))
        self.skip = None
        if int(stride) != 1 or int(in_ch) != int(out_ch):
            self.skip = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=int(stride), bias=False), _make_norm(out_ch, norm))
        self.act = _make_activation(act)
    def forward(self, x):
        identity = x if self.skip is None else self.skip(x)
        return self.act(self.conv2(self.conv1(x)) + identity)


class BottleneckBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, norm: str = DEFAULT_NORM, act: str = DEFAULT_ACTIVATION):
        super().__init__()
        mid = max(8, int(out_ch) // 4)
        self.conv1 = ConvNormAct(in_ch, mid, 1, 1, norm, act)
        self.conv2 = ConvNormAct(mid, mid, 3, stride, norm, act)
        self.conv3 = nn.Sequential(nn.Conv2d(mid, out_ch, 1, bias=False), _make_norm(out_ch, norm))
        self.skip = None
        if int(stride) != 1 or int(in_ch) != int(out_ch):
            self.skip = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=int(stride), bias=False), _make_norm(out_ch, norm))
        self.act = _make_activation(act)
    def forward(self, x):
        identity = x if self.skip is None else self.skip(x)
        return self.act(self.conv3(self.conv2(self.conv1(x))) + identity)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, norm: str = DEFAULT_NORM, act: str = DEFAULT_ACTIVATION):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=int(stride), padding=1, groups=in_ch, bias=False), _make_norm(in_ch, norm), _make_activation(act),
            nn.Conv2d(in_ch, out_ch, 1, bias=False), _make_norm(out_ch, norm), _make_activation(act),
        )
    def forward(self, x): return self.block(x)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(4, int(channels) // int(reduction))
        self.fc1 = nn.Conv2d(channels, hidden, 1); self.fc2 = nn.Conv2d(hidden, channels, 1)
    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1); s = F.silu(self.fc1(s), inplace=True); s = torch.sigmoid(self.fc2(s))
        return x * s


class SPPBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = DEFAULT_NORM, act: str = DEFAULT_ACTIVATION):
        super().__init__()
        self.pre = ConvNormAct(in_ch, out_ch, 1, 1, norm, act)
        self.post = ConvNormAct(out_ch * 4, out_ch, 1, 1, norm, act)
    def forward(self, x):
        x = self.pre(x)
        return self.post(torch.cat([x, F.max_pool2d(x, 5, stride=1, padding=2), F.max_pool2d(x, 9, stride=1, padding=4), F.max_pool2d(x, 13, stride=1, padding=6)], dim=1))


class StyleAdapterBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, norm: str = DEFAULT_NORM, act: str = DEFAULT_ACTIVATION):
        super().__init__()
        self.proj = ConvNormAct(in_ch, out_ch, 3, stride, norm, act)
        self.style_norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.refine = ConvNormAct(out_ch, out_ch, 3, 1, norm, act)
    def forward(self, x):
        y = self.proj(x); return self.refine(self.style_norm(y))


class DilatedContextBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, norm: str = DEFAULT_NORM, act: str = DEFAULT_ACTIVATION):
        super().__init__()
        self.proj = ConvNormAct(in_ch, out_ch, 1, stride, norm, act)
        self.d1 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, padding=1, dilation=1, bias=False), _make_norm(out_ch, norm), _make_activation(act))
        self.d2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, padding=2, dilation=2, bias=False), _make_norm(out_ch, norm), _make_activation(act))
        self.d3 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, padding=3, dilation=3, bias=False), _make_norm(out_ch, norm), _make_activation(act))
        self.fuse = ConvNormAct(out_ch * 3, out_ch, 1, 1, norm, act)
    def forward(self, x):
        y = self.proj(x); return self.fuse(torch.cat([self.d1(y), self.d2(y), self.d3(y)], dim=1))


class FeatureGateBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, norm: str = DEFAULT_NORM, act: str = DEFAULT_ACTIVATION):
        super().__init__()
        self.proj = ConvNormAct(in_ch, out_ch, 3, stride, norm, act) if (in_ch != out_ch or int(stride) != 1) else nn.Identity()
        hidden = max(8, out_ch // 8); self.gate1 = nn.Conv2d(out_ch, hidden, 1); self.gate2 = nn.Conv2d(hidden, out_ch, 1)
    def forward(self, x):
        y = self.proj(x); g = F.adaptive_avg_pool2d(y, 1); g = F.silu(self.gate1(g), inplace=True); g = torch.sigmoid(self.gate2(g))
        return y * (1.0 + g)


class MultiInputFusion(nn.Module):
    def __init__(self, source_channels: List[int], out_ch: int, mode: str, norm: str = DEFAULT_NORM, act: str = DEFAULT_ACTIVATION):
        super().__init__()
        self.mode = str(mode or "Weighted Sum Gate"); self.out_ch = int(out_ch)
        self.proj = nn.ModuleList([ConvNormAct(int(ch), self.out_ch, 1, 1, norm, act) for ch in source_channels])
        self.weights = nn.Parameter(torch.zeros(max(1, len(source_channels))))
        self.concat_fuse = ConvNormAct(self.out_ch * max(1, len(source_channels)), self.out_ch, 1, 1, norm, act)
        self.att = SEBlock(self.out_ch)
        self.refine = ConvNormAct(self.out_ch, self.out_ch, 3, 1, norm, act)
        self.context = DilatedContextBlock(self.out_ch, self.out_ch, 1, norm, act)
        self.style = StyleAdapterBlock(self.out_ch, self.out_ch, 1, norm, act)
    def _align(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        if not tensors: return tensors
        target_h = max(int(t.shape[-2]) for t in tensors); target_w = max(int(t.shape[-1]) for t in tensors)
        out = []
        for t in tensors:
            if int(t.shape[-2]) != target_h or int(t.shape[-1]) != target_w:
                t = F.interpolate(t, size=(target_h, target_w), mode="nearest")
            out.append(t)
        return out
    def forward(self, xs: List[torch.Tensor]):
        ys = [proj(x) for proj, x in zip(self.proj, xs)]
        ys = self._align(ys)
        mode = self.mode.lower()
        if not ys:
            raise RuntimeError("MultiInputFusion received no tensors")
        if "concat" in mode:
            y = self.concat_fuse(torch.cat(ys, dim=1))
        elif "soft and" in mode:
            # Differentiable AND: features pass strongly only where all streams agree.
            probs = [torch.sigmoid(t) for t in ys]
            y = torch.ones_like(probs[0])
            for p in probs:
                y = y * p
            y = self.refine(y)
        elif "soft or" in mode:
            # Differentiable OR: a feature can pass when any stream supports it.
            inv = torch.ones_like(ys[0])
            for t in ys:
                inv = inv * (1.0 - torch.sigmoid(t))
            y = self.refine(1.0 - inv)
        elif "soft not" in mode:
            # Differentiable NOT/suppression: second stream suppresses the first.
            base = ys[0]
            if len(ys) > 1:
                suppress = torch.sigmoid(sum(ys[1:]) / float(max(1, len(ys) - 1)))
                y = base * (1.0 - suppress)
            else:
                y = -base
            y = self.refine(y)
        elif "soft xor" in mode:
            # Disagreement gate: emphasizes places/channels where branches disagree.
            if len(ys) >= 2:
                a = torch.sigmoid(ys[0]); b = torch.sigmoid(ys[1])
                y = torch.abs(a - b)
                if len(ys) > 2:
                    y = y + torch.var(torch.stack([torch.sigmoid(t) for t in ys], dim=0), dim=0)
            else:
                y = ys[0]
            y = self.refine(y)
        elif "mixture" in mode or "experts" in mode:
            # Mixture-of-experts: learn stream weights, then channel-attention refine.
            w = torch.softmax(self.weights[:len(ys)], dim=0)
            y = sum(wi * yi for wi, yi in zip(w, ys))
            y = self.att(self.refine(y))
        else:
            w = torch.softmax(self.weights[:len(ys)], dim=0)
            y = sum(wi * yi for wi, yi in zip(w, ys))
        if "residual" in mode or "skip" in mode: y = self.refine(y + ys[0])
        elif "style" in mode: y = self.style(y)
        elif "scale" in mode: y = self.context(y)
        elif "feature" in mode or "gate" in mode or "weighted" in mode: y = self.att(y)
        return y


def _make_stage(block_type: str, in_ch: int, out_ch: int, stride: int, repeats: int, norm: str, act: str):
    layers = []; block_type = str(block_type); repeats = max(1, int(repeats))
    if block_type == "Stem Conv":
        layers.append(ConvNormAct(in_ch, out_ch, 7, stride, norm, act)); [layers.append(ConvNormAct(out_ch, out_ch, 3, 1, norm, act)) for _ in range(repeats - 1)]
    elif block_type == "Conv Block":
        layers.append(ConvNormAct(in_ch, out_ch, 3, stride, norm, act)); [layers.append(ConvNormAct(out_ch, out_ch, 3, 1, norm, act)) for _ in range(repeats - 1)]
    elif block_type == "Residual Block":
        layers.append(ResidualBlock(in_ch, out_ch, stride, norm, act)); [layers.append(ResidualBlock(out_ch, out_ch, 1, norm, act)) for _ in range(repeats - 1)]
    elif block_type == "Bottleneck Block":
        layers.append(BottleneckBlock(in_ch, out_ch, stride, norm, act)); [layers.append(BottleneckBlock(out_ch, out_ch, 1, norm, act)) for _ in range(repeats - 1)]
    elif block_type == "Depthwise Separable":
        layers.append(DepthwiseSeparableBlock(in_ch, out_ch, stride, norm, act)); [layers.append(DepthwiseSeparableBlock(out_ch, out_ch, 1, norm, act)) for _ in range(repeats - 1)]
    elif block_type == "Downsample Conv":
        layers.append(ConvNormAct(in_ch, out_ch, 3, max(2, stride), norm, act)); [layers.append(ConvNormAct(out_ch, out_ch, 3, 1, norm, act)) for _ in range(repeats - 1)]
    elif block_type == "SE Attention":
        if in_ch != out_ch or stride != 1: layers.append(ConvNormAct(in_ch, out_ch, 3, stride, norm, act))
        [layers.append(SEBlock(out_ch)) for _ in range(repeats)]
    elif block_type == "SPP Block":
        layers.append(SPPBlock(in_ch, out_ch, norm, act)); [layers.append(ConvNormAct(out_ch, out_ch, 3, 1, norm, act)) for _ in range(repeats - 1)]
    elif block_type == "Style Adapter":
        layers.append(StyleAdapterBlock(in_ch, out_ch, stride, norm, act)); [layers.append(StyleAdapterBlock(out_ch, out_ch, 1, norm, act)) for _ in range(repeats - 1)]
    elif block_type == "Dilated Context":
        layers.append(DilatedContextBlock(in_ch, out_ch, stride, norm, act)); [layers.append(DilatedContextBlock(out_ch, out_ch, 1, norm, act)) for _ in range(repeats - 1)]
    elif block_type == "Feature Gate":
        layers.append(FeatureGateBlock(in_ch, out_ch, stride, norm, act)); [layers.append(FeatureGateBlock(out_ch, out_ch, 1, norm, act)) for _ in range(repeats - 1)]
    elif block_type in {"Residual Add Gate", "Skip Bridge"}:
        layers.append(ResidualBlock(in_ch, out_ch, stride, norm, act)); [layers.append(ConvNormAct(out_ch, out_ch, 3, 1, norm, act)) for _ in range(repeats - 1)]
    elif block_type in {"Soft AND Gate", "Soft OR Gate", "Soft NOT Gate", "Soft XOR Gate", "Mixture-of-Experts Gate"}:
        layers.append(FeatureGateBlock(in_ch, out_ch, stride, norm, act)); [layers.append(FeatureGateBlock(out_ch, out_ch, 1, norm, act)) for _ in range(repeats - 1)]
    elif block_type == "Style Gate":
        layers.append(StyleAdapterBlock(in_ch, out_ch, stride, norm, act)); [layers.append(ConvNormAct(out_ch, out_ch, 3, 1, norm, act)) for _ in range(repeats - 1)]
    elif block_type == "Scale Mix Gate":
        layers.append(DilatedContextBlock(in_ch, out_ch, stride, norm, act)); [layers.append(ConvNormAct(out_ch, out_ch, 3, 1, norm, act)) for _ in range(repeats - 1)]
    elif block_type == "Iterative Refinement":
        layers.append(ResidualBlock(in_ch, out_ch, stride, norm, act)); [layers.append(ResidualBlock(out_ch, out_ch, 1, norm, act)) for _ in range(repeats - 1)]
    else:
        layers.append(ConvNormAct(in_ch, out_ch, 3, stride, norm, act)); [layers.append(ConvNormAct(out_ch, out_ch, 3, 1, norm, act)) for _ in range(repeats - 1)]
    return nn.Sequential(*layers)


class @@CLASS_NAME@@(nn.Module):
    def __init__(self, out_channels: int = DEFAULT_OUT_CHANNELS, norm: str = DEFAULT_NORM, activation: str = DEFAULT_ACTIVATION):
        super().__init__()
        self.out_channels = int(out_channels); self.spec = list(CUSTOM_BLOCK_SPEC)
        self.blocks = nn.ModuleList(); self.fusers = nn.ModuleList(); self.runtime_sources: List[List[str]] = []
        self.block_ids: List[str] = []; self.feature_indices: List[int] = []; self.feature_channels: List[int] = []
        known_channels: Dict[str, int] = {"__image__": 3}; previous_id = "__image__"
        for idx, item in enumerate(self.spec):
            bid = str(item.get("id") or f"b{idx}"); btype = str(item.get("type", "Conv Block"))
            out_ch = int(item.get("out", known_channels.get(previous_id, 3))); stride = int(item.get("stride", 1)); repeats = int(item.get("repeats", 1))
            raw_sources = item.get("inputs", []) or []
            if isinstance(raw_sources, str): raw_sources = [raw_sources]
            sources = [str(s) for s in raw_sources if str(s) in known_channels]
            if not sources: sources = [previous_id] if previous_id in known_channels else ["__image__"]
            source_channels = [known_channels.get(s, 3) for s in sources]
            use_fuser = (len(sources) > 1) or btype in {"Residual Add Gate", "Concat Fusion Gate", "Weighted Sum Gate", "Soft AND Gate", "Soft OR Gate", "Soft NOT Gate", "Soft XOR Gate", "Mixture-of-Experts Gate", "Style Gate", "Scale Mix Gate", "Skip Bridge"}
            if use_fuser:
                self.fusers.append(MultiInputFusion(source_channels, out_ch, btype, norm, activation)); stage_in = out_ch
            else:
                self.fusers.append(nn.Identity()); stage_in = int(source_channels[0])
            stage = _make_stage(btype, stage_in, out_ch, stride, repeats, norm, activation)
            self.blocks.append(stage); self.runtime_sources.append(sources); self.block_ids.append(bid)
            known_channels[bid] = out_ch; previous_id = bid
            if bool(item.get("feature", False)): self.feature_indices.append(idx); self.feature_channels.append(out_ch)
        if not self.feature_indices:
            self.feature_indices = [len(self.blocks) - 1]; self.feature_channels = [known_channels.get(previous_id, 3)]
        self.laterals = nn.ModuleList([nn.Conv2d(ch, self.out_channels, 1) for ch in self.feature_channels])
        self.smooths = nn.ModuleList([ConvNormAct(self.out_channels, self.out_channels, 3, 1, norm, activation) for _ in self.feature_channels])

    def forward(self, x):
        cache: Dict[str, torch.Tensor] = {"__image__": x}; previous = x; features = []
        for idx, (bid, sources, fuser, block) in enumerate(zip(self.block_ids, self.runtime_sources, self.fusers, self.blocks)):
            xs = [cache[s] for s in sources if s in cache]
            if not xs: xs = [previous]
            y = xs[0] if isinstance(fuser, nn.Identity) and len(xs) == 1 else fuser(xs)
            y = block(y); cache[bid] = y; previous = y
            if idx in self.feature_indices: features.append(y)
        if not features: features = [previous]
        maps = [lat(feat) for lat, feat in zip(self.laterals, features)]
        for i in range(len(maps) - 2, -1, -1):
            maps[i] = maps[i] + F.interpolate(maps[i + 1], size=maps[i].shape[-2:], mode="nearest")
        maps = [smooth(m) for smooth, m in zip(self.smooths, maps)]
        return OrderedDict((str(i), m) for i, m in enumerate(maps))


def build_backbone(pretrained=False, trainable_layers=3, out_channels=DEFAULT_OUT_CHANNELS):
    model = @@CLASS_NAME@@(out_channels=int(out_channels)); model.out_channels = int(out_channels); return model
'''




class _BackboneBlockCreatorDialog:
    """AI-Pipeline-style visual backbone creator.

    The backbone is still a sequential CNN/FPN graph, but the UI presents it like
    Mustatil's AI Pipeline: visual blocks on a canvas, arrows between blocks,
    selected-block properties on the right, a readable code view, and an explicit
    description panel for every block type and parameter.
    """
    def __init__(self, parent: Any, ws: Any, training_page: Any):
        from PySide6.QtCore import Qt, QRectF, QPointF, QLineF
        from PySide6.QtGui import QColor, QBrush, QPen, QPainter, QPainterPath, QFont
        from PySide6.QtWidgets import (
            QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QGridLayout, QGroupBox,
            QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox, QTextEdit,
            QVBoxLayout, QWidget, QSplitter, QGraphicsView, QGraphicsScene, QGraphicsItem,
            QGraphicsPathItem, QStackedWidget, QPlainTextEdit, QScrollArea, QSizePolicy
        )

        self.Qt = Qt
        self.QFileDialog = QFileDialog
        self.QMessageBox = QMessageBox
        self.ws = ws
        self.training_page = training_page
        self.project = _project_guess(ws)
        self.blocks: List[Dict[str, Any]] = [dict(x) for x in DEFAULT_BLOCKS]
        self.selected_block_id: Optional[str] = None
        self.block_items: Dict[str, Any] = {}
        self.arrow_items: List[Any] = []
        self._updating = False

        block_descriptions = {
            "Stem Conv": (
                "First image-entry block. It converts RGB input into early visual features. "
                "Usually stride 2, so the feature map becomes smaller immediately. Good first block for aerial/satellite images."
            ),
            "Conv Block": (
                "Standard Conv2d + normalization + activation. It learns local edges, textures, shadows and simple shape parts. "
                "Use this when you want a simple, stable building block."
            ),
            "Residual Block": (
                "ResNet-style block with a skip connection. The skip path helps gradients flow through deeper networks, "
                "so this is usually safer than stacking many plain Conv Blocks."
            ),
            "Bottleneck Block": (
                "ResNet bottleneck: 1x1 reduction, 3x3 processing, 1x1 expansion. It gives high capacity with fewer parameters "
                "than a very wide plain block. Good for stronger accuracy models."
            ),
            "Depthwise Separable": (
                "MobileNet-style lightweight block. It separates spatial filtering from channel mixing. Faster and lower VRAM, "
                "but sometimes weaker than Residual/Bottleneck blocks."
            ),
            "Downsample Conv": (
                "Forces resolution reduction with stride 2. This makes later feature maps coarser and cheaper. "
                "Use it when you need the next FPN level / larger receptive field."
            ),
            "SE Attention": (
                "Squeeze-and-Excitation channel attention. It lets the network reweight channels after seeing the whole feature map. "
                "Useful for suppressing irrelevant textures/backgrounds, but it does not reduce resolution by itself."
            ),
            "SPP Block": (
                "Spatial Pyramid Pooling. It pools with several kernel sizes and fuses the result, adding multi-scale context. "
                "Often useful near the end of the backbone for large archaeological structures and context patterns."
            ),
            "Style Adapter": (
                "Experimental style-robustness block. It uses InstanceNorm-based refinement to reduce differences between satellite sources, "
                "sand color, contrast and illumination. Useful when the same object looks different across deserts/tiles."
            ),
            "Dilated Context": (
                "Experimental context block with parallel dilated convolutions. It sees wider patterns without another stride-2 downsample, "
                "which can help faint long rectangular structures and surrounding landscape context."
            ),
            "Feature Gate": (
                "Trainable channel gate. It is not an if/else loop; it learns which feature channels should be amplified. Useful after weak/faint line features or style-adapter blocks."
            ),
            "Residual Add Gate": (
                "Fusion gate for two or more incoming lines. It projects all inputs to the same channel count, adds them, then refines the result with a residual block. Useful for skip connections from early detail blocks."
            ),
            "Concat Fusion Gate": (
                "Fusion gate for multiple incoming lines. It concatenates projected inputs and compresses them back to the chosen channel count. Useful when you want to keep detail + context rather than simply averaging them."
            ),
            "Weighted Sum Gate": (
                "Fusion gate with trainable weights. The model learns how much each incoming branch contributes. Good default for experimental multi-connection graphs."
            ),
            "Soft AND Gate": (
                "Differentiable logic-style AND. It emphasizes features that several incoming branches agree on. Useful for 'line detail AND context shape' checks without hard if/else code."
            ),
            "Soft OR Gate": (
                "Differentiable logic-style OR. It lets features pass when any branch supports them. Useful when Mustatils appear in several visual styles or scales."
            ),
            "Soft NOT Gate": (
                "Differentiable suppression gate. With two or more inputs, later inputs suppress the first input. Useful for negative evidence such as sand waves, wadis, roads, or false-positive texture branches."
            ),
            "Soft XOR Gate": (
                "Differentiable disagreement gate. It emphasizes features where two branches disagree. Experimental: can highlight uncertain or ambiguous areas, but may be noisy."
            ),
            "Mixture-of-Experts Gate": (
                "Trainable expert selector. It learns how much to trust detail/context/style branches. This is often the most useful 'fancy' gate for changing desert satellite styles."
            ),
            "Style Gate": (
                "Fusion/refinement gate with InstanceNorm-style adaptation. Useful when satellite colors, contrast and illumination change strongly."
            ),
            "Scale Mix Gate": (
                "Fusion gate followed by dilated context processing. Useful when incoming lines represent different scales or object sizes."
            ),
            "Skip Bridge": (
                "Light skip-fusion block. Use it to bring an earlier high-resolution detail stream into a later block without making the graph too heavy."
            ),
            "Iterative Refinement": (
                "A fixed, unrolled refinement loop: repeated residual processing without an actual recurrent cycle. Good when you want the model to 'check again' several times in a trainable and exportable way."
            ),
        }
        param_descriptions = {
            "Channels": "Number of output feature channels for this block. More channels = more capacity, slower training, more VRAM.",
            "Stride": "Stride 2 halves width/height and increases receptive field. Stride 1 keeps the same resolution.",
            "Repeats": "How many times this block is stacked. More repeats = deeper model and usually better capacity, but slower.",
            "FPN output": "Marks this block's output as a feature map for the Feature Pyramid Network. Use several levels for small and large objects.",
            "FPN out channels": "All selected feature maps are converted to this channel count before Faster/Mask R-CNN uses them. 256 is a strong default.",
            "Normalization": "GroupNorm is robust for batch size 1. BatchNorm can work with larger batches. InstanceNorm is an alternative for style/contrast variation.",
            "Activation": "Nonlinearity inside blocks. SiLU is a good modern default; ReLU is simpler; GELU is smoother; LeakyReLU keeps negative gradients.",
        }
        self.block_descriptions = block_descriptions
        self.param_descriptions = param_descriptions

        def ensure_ids(blocks: List[Dict[str, Any]]) -> None:
            import uuid
            for i, b in enumerate(blocks):
                if not b.get("id"):
                    b["id"] = "bb_" + uuid.uuid4().hex[:8]
                if not b.get("name"):
                    b["name"] = f"{i + 1}. {b.get('type', 'Block')}"
                if "x" not in b:
                    b["x"] = 80.0 + i * 240.0
                if "y" not in b:
                    b["y"] = 120.0
                if "enabled" not in b:
                    b["enabled"] = True
            for i, b in enumerate(blocks):
                if "inputs" not in b or b.get("inputs") is None:
                    b["inputs"] = [] if i == 0 else [str(blocks[i - 1].get("id"))]
        ensure_ids(self.blocks)
        self._ensure_ids = ensure_ids

        class VisualBackboneBlockItem(QGraphicsItem):
            WIDTH = 215
            HEIGHT = 128
            TYPE_COLORS = {
                "Stem Conv": QColor(75, 130, 255),
                "Conv Block": QColor(70, 155, 210),
                "Residual Block": QColor(60, 170, 120),
                "Bottleneck Block": QColor(180, 105, 220),
                "Depthwise Separable": QColor(70, 170, 170),
                "Downsample Conv": QColor(235, 150, 60),
                "SE Attention": QColor(225, 105, 140),
                "SPP Block": QColor(150, 120, 60),
                "Style Adapter": QColor(90, 120, 210),
                "Dilated Context": QColor(135, 95, 200),
                "Feature Gate": QColor(215, 95, 55),
                "Residual Add Gate": QColor(205, 120, 55),
                "Concat Fusion Gate": QColor(205, 85, 135),
                "Weighted Sum Gate": QColor(95, 135, 215),
                "Soft AND Gate": QColor(70, 150, 90),
                "Soft OR Gate": QColor(80, 165, 165),
                "Soft NOT Gate": QColor(190, 85, 85),
                "Soft XOR Gate": QColor(150, 90, 220),
                "Mixture-of-Experts Gate": QColor(85, 105, 225),
                "Style Gate": QColor(70, 120, 190),
                "Scale Mix Gate": QColor(120, 90, 210),
                "Skip Bridge": QColor(90, 150, 95),
                "Iterative Refinement": QColor(110, 110, 180),
            }
            def __init__(self, block: Dict[str, Any], owner: "_BackboneBlockCreatorDialog"):
                super().__init__()
                self.block = block
                self.owner = owner
                self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemSendsGeometryChanges)
                self.setAcceptHoverEvents(True)
                self.setPos(float(block.get("x", 0)), float(block.get("y", 0)))
                self.setToolTip(owner.block_descriptions.get(str(block.get("type")), "Backbone block"))

            def boundingRect(self):
                return QRectF(0, 0, self.WIDTH, self.HEIGHT)

            def input_anchor(self) -> QPointF:
                return self.scenePos() + QPointF(0, self.HEIGHT / 2)

            def output_anchor(self) -> QPointF:
                return self.scenePos() + QPointF(self.WIDTH, self.HEIGHT / 2)

            def paint(self, painter: QPainter, option, widget=None):
                b = self.block
                typ = str(b.get("type", "Conv Block"))
                color = QColor(self.TYPE_COLORS.get(typ, QColor(120, 120, 120)))
                if not bool(b.get("enabled", True)):
                    color = QColor(145, 145, 145)
                header = QColor(max(0, color.red() - 35), max(0, color.green() - 35), max(0, color.blue() - 35))
                painter.setRenderHint(QPainter.Antialiasing, True)
                painter.setPen(QPen(QColor(20, 20, 20) if self.isSelected() else QColor(80, 80, 80), 3 if self.isSelected() else 1.4))
                painter.setBrush(QBrush(color))
                painter.drawRoundedRect(self.boundingRect(), 12, 12)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(header))
                path = QPainterPath()
                path.addRoundedRect(QRectF(0, 0, self.WIDTH, 30), 12, 12)
                path.addRect(QRectF(0, 14, self.WIDTH, 18))
                painter.drawPath(path)
                painter.setPen(QPen(QColor(45, 45, 45), 1))
                painter.setBrush(QBrush(QColor(245, 245, 245)))
                painter.drawEllipse(QPointF(0, self.HEIGHT / 2), 7, 7)
                painter.drawEllipse(QPointF(self.WIDTH, self.HEIGHT / 2), 7, 7)
                painter.setPen(QPen(QColor("white")))
                painter.setFont(QFont("Arial", 8, QFont.Bold))
                painter.drawText(QRectF(10, 4, self.WIDTH - 20, 22), Qt.AlignLeft | Qt.AlignVCenter, typ.upper())
                painter.setFont(QFont("Arial", 10, QFont.Bold))
                painter.drawText(QRectF(12, 36, self.WIDTH - 24, 22), Qt.AlignLeft | Qt.AlignVCenter, str(b.get("name", typ))[:26])
                painter.setFont(QFont("Arial", 8))
                painter.setPen(QPen(QColor(30, 30, 30)))
                line1 = f"channels {int(b.get('out', 64))} | stride {int(b.get('stride', 1))} | x{int(b.get('repeats', 1))}"
                line2 = "FPN feature output" if bool(b.get("feature", False)) else "internal block only"
                painter.drawText(QRectF(12, 65, self.WIDTH - 24, 18), Qt.AlignLeft | Qt.AlignVCenter, line1)
                painter.drawText(QRectF(12, 86, self.WIDTH - 24, 18), Qt.AlignLeft | Qt.AlignVCenter, line2)
                if bool(b.get("feature", False)):
                    painter.setBrush(QBrush(QColor(255, 255, 255, 220)))
                    painter.setPen(QPen(QColor(40, 100, 40), 1))
                    painter.drawRoundedRect(QRectF(self.WIDTH - 54, self.HEIGHT - 27, 43, 18), 5, 5)
                    painter.setPen(QPen(QColor(20, 100, 20)))
                    painter.drawText(QRectF(self.WIDTH - 54, self.HEIGHT - 27, 43, 18), Qt.AlignCenter, "FPN")

            def mouseDoubleClickEvent(self, event):
                self.owner.select_block(str(self.block.get("id")))
                super().mouseDoubleClickEvent(event)

            def itemChange(self, change, value):
                if change == QGraphicsItem.ItemPositionHasChanged:
                    p = self.pos()
                    self.block["x"] = float(p.x())
                    self.block["y"] = float(p.y())
                    try:
                        self.owner.refresh_arrows()
                    except Exception:
                        pass
                if change == QGraphicsItem.ItemSelectedHasChanged and bool(value):
                    try:
                        self.owner.select_block(str(self.block.get("id")), from_scene=True)
                    except Exception:
                        pass
                return super().itemChange(change, value)

        class BackboneArrowItem(QGraphicsPathItem):
            def __init__(self, start_item: VisualBackboneBlockItem, end_item: VisualBackboneBlockItem, owner: "_BackboneBlockCreatorDialog"):
                super().__init__()
                self.start_item = start_item
                self.end_item = end_item
                self.owner = owner
                self.source_id = str(start_item.block.get("id"))
                self.target_id = str(end_item.block.get("id"))
                self.setZValue(-10)
                self._normal_pen = QPen(QColor(55, 55, 55), 2.2)
                self._hover_pen = QPen(QColor(210, 60, 45), 3.0)
                self.setPen(self._normal_pen)
                self.setAcceptHoverEvents(True)
                self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
                self.setToolTip("Right-click to delete this connection from the active backbone path")
                self.update_path()

            def shape(self):
                try:
                    from PySide6.QtGui import QPainterPathStroker
                    stroker = QPainterPathStroker()
                    stroker.setWidth(16)
                    return stroker.createStroke(self.path())
                except Exception:
                    return super().shape()

            def hoverEnterEvent(self, event):
                self.setPen(self._hover_pen)
                super().hoverEnterEvent(event)

            def hoverLeaveEvent(self, event):
                self.setPen(self._normal_pen)
                super().hoverLeaveEvent(event)

            def mousePressEvent(self, event):
                if event.button() == Qt.RightButton:
                    try:
                        self.owner.remove_connection(self.source_id, self.target_id)
                    except Exception:
                        pass
                    event.accept()
                    return
                super().mousePressEvent(event)

            def update_path(self):
                s = self.start_item.output_anchor()
                e = self.end_item.input_anchor()
                dx = max(70.0, abs(e.x() - s.x()) * 0.45)
                path = QPainterPath(s)
                path.cubicTo(QPointF(s.x() + dx, s.y()), QPointF(e.x() - dx, e.y()), e)
                line = QLineF(path.pointAtPercent(0.985), e)
                angle = math.atan2(-line.dy(), line.dx())
                size = 10
                p1 = e - QPointF(math.sin(angle + math.pi / 3) * size, math.cos(angle + math.pi / 3) * size)
                p2 = e - QPointF(math.sin(angle + math.pi - math.pi / 3) * size, math.cos(angle + math.pi - math.pi / 3) * size)
                head = QPainterPath(e); head.lineTo(p1); head.lineTo(p2); head.closeSubpath()
                path.addPath(head)
                self.setPath(path)

        class BackboneGraphicsView(QGraphicsView):
            def __init__(self, scene: QGraphicsScene, owner: "_BackboneBlockCreatorDialog"):
                super().__init__(scene)
                self.owner = owner
                self.setRenderHint(QPainter.Antialiasing, True)
                self.setDragMode(QGraphicsView.RubberBandDrag)
                self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
                self.setBackgroundBrush(QBrush(QColor(248, 248, 248)))
                self._panning = False
                self._pan_last = None
                self._connect_source = None
                self._temp_connection = None

            def _event_pos(self, event):
                try: return event.position().toPoint()
                except Exception: return event.pos()

            def _scene_pos(self, event):
                return self.mapToScene(self._event_pos(event))

            def _socket_item_at(self, scene_pos, socket: str):
                best = None; best_dist = 999999.0
                hit_radius = max(16.0, 16.0 / max(0.25, abs(float(self.transform().m11())) or 1.0))
                for item in self.owner.block_items.values():
                    try:
                        anchor = item.output_anchor() if socket == "output" else item.input_anchor()
                        dist = QLineF(scene_pos, anchor).length()
                        if dist < hit_radius and dist < best_dist:
                            best = item; best_dist = dist
                    except Exception:
                        pass
                return best

            def _connection_path(self, start, end):
                dx = max(70.0, abs(end.x() - start.x()) * 0.45)
                path = QPainterPath(start)
                path.cubicTo(QPointF(start.x() + dx, start.y()), QPointF(end.x() - dx, end.y()), end)
                return path

            def wheelEvent(self, event):
                factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
                self.scale(factor, factor)
                event.accept()

            def mousePressEvent(self, event):
                pos = self._event_pos(event)
                scene_pos = self.mapToScene(pos)
                if event.button() == Qt.LeftButton:
                    source = self._socket_item_at(scene_pos, "output")
                    if source is not None:
                        self._connect_source = source
                        self._temp_connection = QGraphicsPathItem()
                        self._temp_connection.setZValue(-4)
                        pen = QPen(QColor(25, 115, 255), 2.8); pen.setStyle(Qt.DashLine)
                        self._temp_connection.setPen(pen)
                        self._temp_connection.setPath(self._connection_path(source.output_anchor(), scene_pos))
                        self.scene().addItem(self._temp_connection)
                        self.setDragMode(QGraphicsView.NoDrag)
                        event.accept(); return
                if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and self.itemAt(pos) is None):
                    self._panning = True
                    self._pan_last = pos
                    self.setCursor(Qt.ClosedHandCursor)
                    self.setDragMode(QGraphicsView.NoDrag)
                    event.accept(); return
                super().mousePressEvent(event)

            def mouseMoveEvent(self, event):
                if self._connect_source is not None and self._temp_connection is not None:
                    scene_pos = self._scene_pos(event)
                    self._temp_connection.setPath(self._connection_path(self._connect_source.output_anchor(), scene_pos))
                    event.accept(); return
                if self._panning and self._pan_last is not None:
                    pos = self._event_pos(event)
                    dx = pos.x() - self._pan_last.x(); dy = pos.y() - self._pan_last.y()
                    self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - dx)
                    self.verticalScrollBar().setValue(self.verticalScrollBar().value() - dy)
                    self._pan_last = pos
                    event.accept(); return
                super().mouseMoveEvent(event)

            def mouseReleaseEvent(self, event):
                if self._connect_source is not None:
                    scene_pos = self._scene_pos(event)
                    target = self._socket_item_at(scene_pos, "input")
                    if self._temp_connection is not None:
                        try: self.scene().removeItem(self._temp_connection)
                        except Exception: pass
                    source = self._connect_source
                    self._connect_source = None
                    self._temp_connection = None
                    self.setDragMode(QGraphicsView.RubberBandDrag)
                    if target is not None and target.block.get("id") != source.block.get("id"):
                        self.owner.add_connection(str(source.block.get("id")), str(target.block.get("id")))
                    event.accept(); return
                if self._panning and event.button() in (Qt.MiddleButton, Qt.LeftButton):
                    self._panning = False
                    self._pan_last = None
                    self.unsetCursor()
                    self.setDragMode(QGraphicsView.RubberBandDrag)
                    event.accept(); return
                super().mouseReleaseEvent(event)

        self.VisualBackboneBlockItem = VisualBackboneBlockItem
        self.BackboneArrowItem = BackboneArrowItem
        self.BackboneGraphicsView = BackboneGraphicsView

        dlg = QDialog(parent)
        dlg.setWindowTitle("Backbone Creator — Visual Block Pipeline")
        dlg.resize(1500, 900)
        dlg.setMinimumSize(980, 620)
        self.dialog = dlg

        root = QVBoxLayout(dlg)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        toolbar = QHBoxLayout()
        root.addLayout(toolbar)

        self.block_category_combo = QComboBox()
        self.block_category_combo.addItems(["Core CNN", "Attention / Context", "Fusion Gates", "Soft Logic Gates", "Experimental"] )
        self.block_type_add_combo = QComboBox()
        self.block_type_add_combo.setMinimumWidth(220)
        block_categories = {
            "Core CNN": ["Stem Conv", "Conv Block", "Residual Block", "Bottleneck Block", "Depthwise Separable", "Downsample Conv"],
            "Attention / Context": ["SE Attention", "SPP Block", "Style Adapter", "Dilated Context", "Feature Gate", "Style Gate", "Scale Mix Gate"],
            "Fusion Gates": ["Residual Add Gate", "Concat Fusion Gate", "Weighted Sum Gate", "Mixture-of-Experts Gate", "Skip Bridge"],
            "Soft Logic Gates": ["Soft AND Gate", "Soft OR Gate", "Soft NOT Gate", "Soft XOR Gate"],
            "Experimental": ["Iterative Refinement", "Mixture-of-Experts Gate", "Scale Mix Gate", "Style Gate"],
        }
        def _refresh_add_combo():
            cat = str(self.block_category_combo.currentText() or "Core CNN")
            self.block_type_add_combo.clear()
            for typ in block_categories.get(cat, BLOCK_TYPES):
                self.block_type_add_combo.addItem(typ)
            try:
                typ = str(self.block_type_add_combo.currentText())
                self.block_type_add_combo.setToolTip(block_descriptions.get(typ, "Add backbone block"))
            except Exception:
                pass
        self.block_category_combo.currentTextChanged.connect(lambda *_: _refresh_add_combo())
        self.block_type_add_combo.currentTextChanged.connect(lambda t: self.block_type_add_combo.setToolTip(block_descriptions.get(str(t), "Add backbone block")))
        _refresh_add_combo()
        add_btn = QPushButton("+ Add block")
        add_btn.clicked.connect(lambda *_: self.add_block(str(self.block_type_add_combo.currentText() or "Conv Block")))
        toolbar.addWidget(QLabel("Block category")); toolbar.addWidget(self.block_category_combo)
        toolbar.addWidget(QLabel("Block")); toolbar.addWidget(self.block_type_add_combo); toolbar.addWidget(add_btn)
        dup_btn = QPushButton("Duplicate"); dup_btn.clicked.connect(self.duplicate_selected); toolbar.addWidget(dup_btn)
        del_btn = QPushButton("Delete"); del_btn.clicked.connect(self.delete_selected); toolbar.addWidget(del_btn)
        left_btn = QPushButton("←"); left_btn.setToolTip("Move selected block earlier in the execution order"); left_btn.clicked.connect(lambda: self.move_selected(-1)); toolbar.addWidget(left_btn)
        right_btn = QPushButton("→"); right_btn.setToolTip("Move selected block later in the execution order"); right_btn.clicked.connect(lambda: self.move_selected(1)); toolbar.addWidget(right_btn)
        layout_btn = QPushButton("Auto layout"); layout_btn.clicked.connect(self.auto_layout); toolbar.addWidget(layout_btn)
        self.view_mode_btn = QPushButton("View: Code")
        self.view_mode_btn.clicked.connect(self.toggle_code_view)
        toolbar.addWidget(self.view_mode_btn)
        toolbar.addStretch(1)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "satellite accuracy", "fast", "wide", "desert style robust", "tiny/faint objects",
            "context heavy", "graph fusion", "logic detail+context", "expert style ensemble"
        ])
        preset_btn = QPushButton("Load preset")
        preset_btn.clicked.connect(self.load_selected_preset)
        toolbar.addWidget(QLabel("Preset")); toolbar.addWidget(self.preset_combo); toolbar.addWidget(preset_btn)

        hint = QLabel(
            "Build the backbone like a pipeline: add blocks from the toolbar, reorder them with Move left/right, mark important outputs as FPN feature maps, then export a Custom Python Backbone. "
            "Pan the canvas with middle mouse or left mouse on empty space. Right-click a line to remove that connection from the active exported path. "
            "Backbone graphs can branch and merge. Soft logic gates are differentiable feature gates, not hard Python if/else loops, so they remain trainable."
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        self.mode_stack = QStackedWidget()
        root.addWidget(self.mode_stack, 1)

        split = QSplitter(Qt.Horizontal)
        self.scene = QGraphicsScene(self.dialog)
        self.scene.setSceneRect(-200, -160, 2000, 1100)
        self.view = BackboneGraphicsView(self.scene, self)
        split.addWidget(self.view)

        right = QWidget()
        right.setMinimumWidth(340)
        right.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(8, 8, 8, 8)
        right_lay.setSpacing(8)

        output_box = QGroupBox("Output and global backbone settings")
        og = QGridLayout(output_box)
        self.name_line = QLineEdit("mustatil_visual_backbone_" + _timestamp())
        self.folder_line = QLineEdit(str(self.project / "custom_backbones"))
        browse_folder = QPushButton("…"); browse_folder.setFixedWidth(32); browse_folder.clicked.connect(self.browse_folder)
        folder_row = QWidget(); fr = QHBoxLayout(folder_row); fr.setContentsMargins(0, 0, 0, 0); fr.addWidget(self.folder_line, 1); fr.addWidget(browse_folder)
        self.out_channels_spin = QSpinBox(); self.out_channels_spin.setRange(16, 1024); self.out_channels_spin.setSingleStep(16); self.out_channels_spin.setValue(256)
        self.norm_combo = QComboBox(); self.norm_combo.addItems(["GroupNorm", "BatchNorm", "InstanceNorm"]); self.norm_combo.setCurrentText("GroupNorm")
        self.act_combo = QComboBox(); self.act_combo.addItems(["SiLU", "ReLU", "GELU", "LeakyReLU"]); self.act_combo.setCurrentText("SiLU")
        for w in (self.out_channels_spin, self.norm_combo, self.act_combo, self.name_line, self.folder_line):
            try:
                if hasattr(w, "valueChanged"):
                    w.valueChanged.connect(self.refresh_code_from_blocks)
                elif hasattr(w, "currentTextChanged"):
                    w.currentTextChanged.connect(self.refresh_code_from_blocks)
                elif hasattr(w, "textChanged"):
                    w.textChanged.connect(self.refresh_code_from_blocks)
            except Exception:
                pass
        og.addWidget(QLabel("File name"), 0, 0); og.addWidget(self.name_line, 0, 1)
        og.addWidget(QLabel("Folder"), 1, 0); og.addWidget(folder_row, 1, 1)
        og.addWidget(QLabel("FPN out channels"), 2, 0); og.addWidget(self.out_channels_spin, 2, 1)
        og.addWidget(QLabel("Normalization"), 3, 0); og.addWidget(self.norm_combo, 3, 1)
        og.addWidget(QLabel("Activation"), 4, 0); og.addWidget(self.act_combo, 4, 1)
        right_lay.addWidget(output_box, 0)

        prop_box = QGroupBox("Selected block properties")
        form = QFormLayout(prop_box)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.enabled_chk = QCheckBox("enabled")
        self.name_edit = QLineEdit()
        self.type_combo = QComboBox(); self.type_combo.addItems(BLOCK_TYPES)
        self.channels_spin = QSpinBox(); self.channels_spin.setRange(4, 4096); self.channels_spin.setSingleStep(8)
        self.stride_combo = QComboBox(); self.stride_combo.addItems(["1", "2"])
        self.repeats_spin = QSpinBox(); self.repeats_spin.setRange(1, 32)
        self.feature_chk = QCheckBox("Use this block output as FPN feature map")
        form.addRow("", self.enabled_chk)
        form.addRow("Name", self.name_edit)
        form.addRow("Block type", self.type_combo)
        form.addRow("Channels", self.channels_spin)
        form.addRow("Stride", self.stride_combo)
        form.addRow("Repeats", self.repeats_spin)
        form.addRow("FPN output", self.feature_chk)
        for w in (self.enabled_chk, self.name_edit, self.type_combo, self.channels_spin, self.stride_combo, self.repeats_spin, self.feature_chk):
            try:
                if hasattr(w, "toggled"):
                    w.toggled.connect(self.save_editor_to_block)
                elif hasattr(w, "textChanged"):
                    w.textChanged.connect(self.save_editor_to_block)
                elif hasattr(w, "currentTextChanged"):
                    w.currentTextChanged.connect(self.save_editor_to_block)
                elif hasattr(w, "valueChanged"):
                    w.valueChanged.connect(self.save_editor_to_block)
            except Exception:
                pass
        prop_scroll = QScrollArea(); prop_scroll.setWidgetResizable(True); prop_scroll.setMinimumHeight(250); prop_scroll.setWidget(prop_box)
        right_lay.addWidget(prop_scroll, 1)

        desc_box = QGroupBox("What this block/function does")
        dl = QVBoxLayout(desc_box)
        self.description = QTextEdit(); self.description.setReadOnly(True); self.description.setMinimumHeight(230)
        dl.addWidget(self.description, 1)
        right_lay.addWidget(desc_box, 1)

        actions = QGroupBox("Actions")
        ag = QGridLayout(actions)
        create_btn = QPushButton("Create .py")
        create_use_btn = QPushButton("Create .py + use in Training tab")
        test_btn = QPushButton("Test generated backbone")
        fit_btn = QPushButton("Fit canvas")
        close_btn = QPushButton("Close")
        create_btn.clicked.connect(lambda: self.write_file(use_in_tab=False))
        create_use_btn.clicked.connect(lambda: self.write_file(use_in_tab=True))
        test_btn.clicked.connect(self.test_generated)
        fit_btn.clicked.connect(self.fit_canvas)
        close_btn.clicked.connect(dlg.close)
        ag.addWidget(create_btn, 0, 0)
        ag.addWidget(create_use_btn, 0, 1)
        ag.addWidget(test_btn, 1, 0)
        ag.addWidget(fit_btn, 1, 1)
        ag.addWidget(close_btn, 2, 0, 1, 2)
        right_lay.addWidget(actions, 0)

        self.summary = QTextEdit(); self.summary.setReadOnly(True); self.summary.setMinimumHeight(120)
        right_lay.addWidget(QLabel("Backbone Creator log"))
        right_lay.addWidget(self.summary, 0)
        split.addWidget(right)
        split.setSizes([900, 540])
        self.mode_stack.addWidget(split)

        code_page = QWidget()
        code_lay = QVBoxLayout(code_page)
        code_help = QLabel(
            "Code view. In 'Visual block spec' mode you can write/edit the block architecture as Python/JSON data and it is applied back to the canvas. "
            "In 'Manual Python backbone' mode you can edit the real .py code directly; that can be saved/tested/used, but arbitrary Python cannot be converted back into visual blocks."
        )
        code_help.setWordWrap(True)
        code_lay.addWidget(code_help)
        code_btn_row = QHBoxLayout()
        self.code_mode_combo = QComboBox()
        self.code_mode_combo.addItems(["Visual block spec (editable)", "Manual Python backbone (.py)"])
        self.code_mode_combo.currentTextChanged.connect(self._code_mode_changed)
        code_btn_row.addWidget(QLabel("Code mode"))
        code_btn_row.addWidget(self.code_mode_combo)
        refresh_btn = QPushButton("Refresh from blocks")
        refresh_btn.clicked.connect(self.refresh_code_from_blocks)
        code_btn_row.addWidget(refresh_btn)
        apply_btn = QPushButton("Apply code to blocks")
        apply_btn.clicked.connect(lambda: self.apply_code_to_blocks(silent=False))
        code_btn_row.addWidget(apply_btn)
        self.auto_apply_code_chk = QCheckBox("Auto-apply block code")
        self.auto_apply_code_chk.setChecked(True)
        self.auto_apply_code_chk.setToolTip("When enabled, changes in Visual block spec mode are applied to the canvas after a short pause.")
        code_btn_row.addWidget(self.auto_apply_code_chk)
        save_manual_btn = QPushButton("Save current code as .py")
        save_manual_btn.clicked.connect(lambda: self.write_file(use_in_tab=False))
        code_btn_row.addWidget(save_manual_btn)
        use_manual_btn = QPushButton("Save + use in Training")
        use_manual_btn.clicked.connect(lambda: self.write_file(use_in_tab=True))
        code_btn_row.addWidget(use_manual_btn)
        code_btn_row.addStretch(1)
        code_lay.addLayout(code_btn_row)
        self.generated_code = QPlainTextEdit()
        self.generated_code.setMinimumHeight(650)
        self.generated_code.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.generated_code.setPlaceholderText("backbone = {\n  'version': 1,\n  'globals': {'fpn_out_channels': 256, 'normalization': 'GroupNorm', 'activation': 'SiLU'},\n  'blocks': []\n}")
        code_lay.addWidget(self.generated_code, 1)
        try:
            from PySide6.QtCore import QTimer as _QTimer
            self._code_apply_timer = _QTimer(self.dialog)
            self._code_apply_timer.setSingleShot(True)
            self._code_apply_timer.setInterval(750)
            self._code_apply_timer.timeout.connect(lambda: self.apply_code_to_blocks(silent=True))
            self._code_text_updating = False
            self.generated_code.textChanged.connect(self._code_text_changed)
        except Exception:
            self._code_apply_timer = None
            self._code_text_updating = False
        self.mode_stack.addWidget(code_page)

        self._ensure_ids(self.blocks)
        self._rebuild_scene()
        if self.blocks:
            self.select_block(str(self.blocks[0].get("id")))
        self.refresh_code_from_blocks()
        self.write_intro()
        self.auto_layout()

    def exec(self):
        return self.dialog.exec()

    def _log_to_ws(self, msg: str) -> None:
        self.summary.append(str(msg))
        try:
            self.ws.log("Backbone Creator: " + str(msg).replace("\n", " | "))
        except Exception:
            _log(str(msg))

    def write_intro(self):
        self._log_to_ws("Visual Backbone Creator ready. Add blocks like the AI Pipeline, select a block to see what it does, then export .py for the R-CNN Trainer.")

    def browse_folder(self):
        d = self.QFileDialog.getExistingDirectory(self.dialog, "Select custom backbone folder", self.folder_line.text().strip() or str(self.project))
        if d:
            self.folder_line.setText(d)

    def _target_path(self) -> Path:
        folder = Path(self.folder_line.text().strip() or str(self.project / "custom_backbones")).expanduser()
        name = _safe_name(self.name_line.text().strip() or "mustatil_visual_backbone")
        if not name.lower().endswith(".py"):
            name += ".py"
        return folder / name


    def _normalize_visual_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        base = _normalize_blocks(blocks)
        out: List[Dict[str, Any]] = []
        for old, clean in zip(list(blocks or []), base):
            merged = dict(clean)
            for key in ("id", "name", "x", "y", "enabled", "inputs"):
                if key in old:
                    merged[key] = old[key]
            out.append(merged)
        if not out:
            out = [dict(x) for x in DEFAULT_BLOCKS]
        self._ensure_ids(out)
        return out

    def current_block(self) -> Optional[Dict[str, Any]]:
        for b in self.blocks:
            if str(b.get("id")) == str(self.selected_block_id):
                return b
        return None

    def add_block(self, typ: str):
        import uuid
        idx = len(self.blocks) + 1
        defaults = {
            "Stem Conv": (32, 2, 1, True),
            "Conv Block": (64, 1, 1, False),
            "Residual Block": (128, 2, 2, True),
            "Bottleneck Block": (256, 2, 2, True),
            "Depthwise Separable": (128, 2, 1, True),
            "Downsample Conv": (128, 2, 1, True),
            "SE Attention": (128, 1, 1, False),
            "SPP Block": (512, 1, 1, True),
            "Style Adapter": (128, 1, 1, False),
            "Dilated Context": (256, 1, 1, True),
            "Feature Gate": (256, 1, 1, False),
            "Residual Add Gate": (256, 1, 1, True),
            "Concat Fusion Gate": (256, 1, 1, True),
            "Weighted Sum Gate": (256, 1, 1, True),
            "Soft AND Gate": (256, 1, 1, True),
            "Soft OR Gate": (256, 1, 1, True),
            "Soft NOT Gate": (256, 1, 1, False),
            "Soft XOR Gate": (256, 1, 1, False),
            "Mixture-of-Experts Gate": (256, 1, 1, True),
            "Style Gate": (128, 1, 1, False),
            "Scale Mix Gate": (256, 1, 1, True),
            "Skip Bridge": (128, 1, 1, False),
            "Iterative Refinement": (256, 1, 3, True),
        }
        ch, st, rep, feat = defaults.get(str(typ), (64, 1, 1, False))
        b = {"id": "bb_" + uuid.uuid4().hex[:8], "name": f"{idx}. {typ}", "type": typ, "out": ch, "stride": st, "repeats": rep, "feature": feat, "enabled": True, "x": 80 + (idx - 1) * 250, "y": 120, "inputs": []}
        cur = self.current_block()
        if cur is not None:
            b["inputs"] = [str(cur.get("id"))]
        elif self.blocks:
            b["inputs"] = [str(self.blocks[-1].get("id"))]
        if cur in self.blocks:
            pos = self.blocks.index(cur) + 1
            self.blocks.insert(pos, b)
        else:
            self.blocks.append(b)
        self._rebuild_scene()
        self.select_block(str(b["id"]))
        self.refresh_code_from_blocks()

    def duplicate_selected(self):
        import uuid
        b = self.current_block()
        if not b:
            return
        nb = dict(b)
        nb["id"] = "bb_" + uuid.uuid4().hex[:8]
        nb["name"] = str(b.get("name", b.get("type", "Block"))) + " copy"
        nb["x"] = float(b.get("x", 0)) + 40
        nb["y"] = float(b.get("y", 0)) + 40
        self.blocks.insert(self.blocks.index(b) + 1, nb)
        self._rebuild_scene(); self.select_block(str(nb["id"])); self.refresh_code_from_blocks()

    def delete_selected(self):
        b = self.current_block()
        if not b:
            return
        self.blocks.remove(b)
        if not self.blocks:
            self.blocks = [dict(x) for x in DEFAULT_BLOCKS]
            self._ensure_ids(self.blocks)
        self._rebuild_scene()
        self.select_block(str(self.blocks[min(0, len(self.blocks)-1)].get("id")))
        self.refresh_code_from_blocks()

    def move_selected(self, delta: int):
        b = self.current_block()
        if not b:
            return
        i = self.blocks.index(b)
        j = i + int(delta)
        if j < 0 or j >= len(self.blocks):
            return
        self.blocks[i], self.blocks[j] = self.blocks[j], self.blocks[i]
        self._rebuild_scene(); self.select_block(str(b.get("id"))); self.refresh_code_from_blocks(); self.auto_layout()

    def _create_item_for_block(self, b: Dict[str, Any]):
        item = self.VisualBackboneBlockItem(b, self)
        self.block_items[str(b.get("id"))] = item
        self.scene.addItem(item)

    def _rebuild_scene(self):
        self.scene.clear()
        self.block_items = {}
        self.arrow_items = []
        self._ensure_ids(self.blocks)
        self.blocks = self._normalize_visual_blocks(self.blocks)
        for b in self.blocks:
            self._create_item_for_block(b)
        self.refresh_arrows()

    def _block_inputs(self, block: Dict[str, Any]) -> List[str]:
        raw = block.get("inputs", []) or []
        if isinstance(raw, str): raw = [raw] if raw.strip() else []
        out, seen = [], set()
        valid = {str(b.get("id")) for b in self.blocks}
        for ref in raw:
            ref = str(ref or "").strip()
            if ref and ref in valid and ref != str(block.get("id")) and ref not in seen:
                seen.add(ref); out.append(ref)
        return out

    def refresh_arrows(self):
        for a in list(self.arrow_items):
            try: self.scene.removeItem(a)
            except Exception: pass
        self.arrow_items = []
        for target_block in self.blocks:
            if not bool(target_block.get("enabled", True)): continue
            target_item = self.block_items.get(str(target_block.get("id")))
            if target_item is None: continue
            for ref in self._block_inputs(target_block):
                source_block = next((b for b in self.blocks if str(b.get("id")) == ref), None)
                if source_block is not None and not bool(source_block.get("enabled", True)): continue
                source_item = self.block_items.get(ref)
                if source_item is not None:
                    arr = self.BackboneArrowItem(source_item, target_item, self)
                    self.arrow_items.append(arr); self.scene.addItem(arr)
        for item in self.block_items.values():
            try: item.update()
            except Exception: pass

    def add_connection(self, source_id: str, target_id: str):
        if str(source_id) == str(target_id): return
        target = next((b for b in self.blocks if str(b.get("id")) == str(target_id)), None)
        if target is None: return
        inputs = self._block_inputs(target)
        if str(source_id) not in inputs: inputs.append(str(source_id))
        target["inputs"] = inputs; target["enabled"] = True
        self.refresh_arrows(); self.select_block(str(target.get("id"))); self.refresh_code_from_blocks()
        self._log_to_ws(f"Connected: {self.block_name(source_id)} → {self.block_name(target_id)}")

    def block_name(self, block_id: str) -> str:
        for b in self.blocks:
            if str(b.get("id")) == str(block_id):
                return str(b.get("name") or b.get("type") or block_id)
        return str(block_id)

    def remove_connection(self, source_id: str, target_id: str):
        target = next((b for b in self.blocks if str(b.get("id")) == str(target_id)), None)
        if target is None: return
        target["inputs"] = [x for x in self._block_inputs(target) if str(x) != str(source_id)]
        self.refresh_arrows(); self.select_block(str(target.get("id"))); self.refresh_code_from_blocks()
        try:
            self.description.setPlainText(
                "Connection removed. Only the clicked line was deleted.\n\n"
                "If the block has no remaining input lines, it becomes a new start from the RGB image during export. "
                "For normal backbones, keep one clean main path and use extra lines mainly into fusion/gate blocks."
            )
        except Exception: pass
        self._log_to_ws(f"Removed connection: {self.block_name(source_id)} → {self.block_name(target_id)}")

    def select_block(self, block_id: str, from_scene: bool = False):
        self.selected_block_id = block_id
        if not from_scene:
            for bid, item in self.block_items.items():
                item.setSelected(str(bid) == str(block_id))
        b = self.current_block()
        if b:
            self.load_block_to_editor(b)

    def load_block_to_editor(self, b: Dict[str, Any]):
        self._updating = True
        try:
            self.enabled_chk.setChecked(bool(b.get("enabled", True)))
            self.name_edit.setText(str(b.get("name", b.get("type", "Block"))))
            self.type_combo.setCurrentText(str(b.get("type", "Conv Block")))
            self.channels_spin.setValue(int(b.get("out", 64)))
            self.stride_combo.setCurrentText(str(int(b.get("stride", 1))))
            self.repeats_spin.setValue(int(b.get("repeats", 1)))
            self.feature_chk.setChecked(bool(b.get("feature", False)))
        finally:
            self._updating = False
        self.update_description()

    def save_editor_to_block(self, *_):
        if self._updating:
            return
        b = self.current_block()
        if not b:
            return
        b["enabled"] = bool(self.enabled_chk.isChecked())
        b["name"] = self.name_edit.text().strip() or str(b.get("type", "Block"))
        b["type"] = self.type_combo.currentText().strip() or "Conv Block"
        b["out"] = int(self.channels_spin.value())
        b["stride"] = int(self.stride_combo.currentText())
        b["repeats"] = int(self.repeats_spin.value())
        b["feature"] = bool(self.feature_chk.isChecked())
        self.blocks = self._normalize_visual_blocks(self.blocks)
        try:
            item = self.block_items.get(str(b.get("id")))
            if item:
                item.update()
                item.setToolTip(self.block_descriptions.get(str(b.get("type")), "Backbone block"))
        except Exception:
            pass
        self.refresh_arrows()
        self.update_description()
        self.refresh_code_from_blocks()

    def update_description(self):
        b = self.current_block()
        if not b:
            self.description.setPlainText("No block selected.")
            return
        typ = str(b.get("type", "Conv Block"))
        lines = [
            f"{typ}",
            "",
            self.block_descriptions.get(typ, "No description available."),
            "",
            "Current settings:",
            f"- Channels: {int(b.get('out', 64))} — {self.param_descriptions['Channels']}",
            f"- Stride: {int(b.get('stride', 1))} — {self.param_descriptions['Stride']}",
            f"- Repeats: {int(b.get('repeats', 1))} — {self.param_descriptions['Repeats']}",
            f"- FPN output: {'yes' if bool(b.get('feature', False)) else 'no'} — {self.param_descriptions['FPN output']}",
            f"- Incoming lines: {len(self._block_inputs(b))}. Several inputs are fused automatically; use Gate/Fusion blocks for the clearest architecture.",
            "",
            "Gate quick guide:",
            "- Feature Gate: channel attention for one stream; not a hard if-loop.",
            "- Residual Add Gate: adds skip/detail streams and refines them; good for early-detail skip connections.",
            "- Concat Fusion Gate: preserves several streams by concatenating before compression; high capacity, more VRAM.",
            "- Weighted Sum Gate: learns how much each incoming line contributes; safest experimental multi-input gate.",
            "- Style Gate: helps with satellite-style changes: contrast, sand color, source variation.",
            "- Scale Mix Gate: fuses streams and applies dilated context; useful for objects at different sizes.",
            "- Skip Bridge: lightweight skip fusion from an earlier block into a later block.",
            "",
            "Global functions/settings:",
            f"- FPN out channels — {self.param_descriptions['FPN out channels']}",
            f"- Normalization — {self.param_descriptions['Normalization']}",
            f"- Activation — {self.param_descriptions['Activation']}",
        ]
        self.description.setPlainText("\n".join(lines))

    def auto_layout(self):
        x0, y0 = 70, 130
        step_x = 255
        wrap = 4
        for i, b in enumerate(self.blocks):
            b["x"] = float(x0 + (i % wrap) * step_x)
            b["y"] = float(y0 + (i // wrap) * 190)
        for b in self.blocks:
            item = self.block_items.get(str(b.get("id")))
            if item:
                item.setPos(float(b.get("x", 0)), float(b.get("y", 0)))
        self.refresh_arrows()
        self.fit_canvas()

    def fit_canvas(self):
        try:
            rect = self.scene.itemsBoundingRect().adjusted(-80, -80, 80, 80)
            self.view.fitInView(rect, self.Qt.KeepAspectRatio)
        except Exception:
            pass

    def toggle_code_view(self):
        if self.mode_stack.currentIndex() == 0:
            self.refresh_code_from_blocks()
            self.mode_stack.setCurrentIndex(1)
            self.view_mode_btn.setText("View: Blocks")
        else:
            try:
                if hasattr(self, "code_mode_combo") and "Visual block spec" in str(self.code_mode_combo.currentText()):
                    self.apply_code_to_blocks(silent=True)
            except Exception:
                pass
            self.mode_stack.setCurrentIndex(0)
            self.view_mode_btn.setText("View: Code")

    def _code(self) -> str:
        enabled_blocks = [dict(b) for b in self.blocks if bool(b.get("enabled", True))]
        if not enabled_blocks:
            enabled_blocks = [dict(x) for x in DEFAULT_BLOCKS]
        return _render_backbone_code(
            enabled_blocks,
            self.name_line.text().strip() or "mustatil_visual_backbone",
            default_out_channels=self.out_channels_spin.value(),
            norm_kind=self.norm_combo.currentText(),
            activation=self.act_combo.currentText(),
            note="Generated by Mustatil Visual Backbone Creator",
        )

    def _block_spec_dict(self) -> Dict[str, Any]:
        self._ensure_ids(self.blocks)
        return {
            "version": 1,
            "description": "Editable Mustatil visual backbone block spec. Change blocks/globals, then Apply code to blocks or leave Auto-apply enabled.",
            "name": self.name_line.text().strip() or "mustatil_visual_backbone",
            "globals": {
                "fpn_out_channels": int(self.out_channels_spin.value()),
                "normalization": str(self.norm_combo.currentText()),
                "activation": str(self.act_combo.currentText()),
            },
            "blocks": [dict(b) for b in self.blocks],
        }

    def _block_spec_code(self) -> str:
        return (
            "# Mustatil Visual Backbone Creator - editable block code\n"
            "# Edit this Python data, then click 'Apply code to blocks'.\n"
            "# Auto-apply updates the visual canvas after a short pause.\n"
            "# Supported block types: " + ", ".join(BLOCK_TYPES) + "\n\n"
            "backbone = " + pprint.pformat(self._block_spec_dict(), width=120, sort_dicts=False) + "\n"
        )

    def _code_mode_is_block_spec(self) -> bool:
        try:
            return "Visual block spec" in str(self.code_mode_combo.currentText())
        except Exception:
            return True

    def _code_mode_changed(self, *_):
        self.refresh_code_from_blocks()

    def _code_text_changed(self):
        if getattr(self, "_code_text_updating", False):
            return
        if not self._code_mode_is_block_spec():
            return
        try:
            if self.auto_apply_code_chk.isChecked() and self._code_apply_timer is not None:
                self._code_apply_timer.start()
        except Exception:
            pass

    def _parse_block_spec_text(self, text: str) -> Dict[str, Any]:
        raw = str(text or "").strip()
        if not raw:
            raise ValueError("Code field is empty.")
        if raw.startswith("{") or raw.startswith("["):
            data = json.loads(raw)
            if isinstance(data, list):
                return {"blocks": data}
            return data
        tree = ast.parse(raw, mode="exec")
        node = None
        for stmt in tree.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id in {"backbone", "pipeline", "blocks"}:
                        node = stmt.value
                        break
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id in {"backbone", "pipeline", "blocks"}:
                node = stmt.value
            elif isinstance(stmt, ast.Expr) and node is None:
                node = stmt.value
            if node is not None:
                break
        if node is None:
            raise ValueError("Expected code like: backbone = {'globals': {...}, 'blocks': [...]}.")
        data = ast.literal_eval(node)
        if isinstance(data, list):
            return {"blocks": data}
        if not isinstance(data, dict):
            raise ValueError("Backbone code must evaluate to a dict or list of blocks.")
        return data

    def apply_code_to_blocks(self, silent: bool = False) -> bool:
        if not self._code_mode_is_block_spec():
            if not silent:
                self.QMessageBox.information(self.dialog, "Manual Python mode", "Manual Python code can be saved/tested, but arbitrary PyTorch code cannot be converted back into visual blocks.")
            return False
        try:
            data = self._parse_block_spec_text(self.generated_code.toPlainText())
            globals_cfg = data.get("globals", {}) if isinstance(data, dict) else {}
            blocks = data.get("blocks", data if isinstance(data, list) else [])
            if not isinstance(blocks, list):
                raise ValueError("Expected a 'blocks' list.")
            self.blocks = self._normalize_visual_blocks([dict(b) for b in blocks])
            try:
                if data.get("name"):
                    self.name_line.setText(str(data.get("name")))
            except Exception:
                pass
            try:
                self.out_channels_spin.setValue(int(globals_cfg.get("fpn_out_channels", globals_cfg.get("out_channels", self.out_channels_spin.value()))))
            except Exception:
                pass
            try:
                norm = str(globals_cfg.get("normalization", self.norm_combo.currentText()))
                ix = self.norm_combo.findText(norm)
                if ix >= 0:
                    self.norm_combo.setCurrentIndex(ix)
            except Exception:
                pass
            try:
                act = str(globals_cfg.get("activation", self.act_combo.currentText()))
                ix = self.act_combo.findText(act)
                if ix >= 0:
                    self.act_combo.setCurrentIndex(ix)
            except Exception:
                pass
            self._rebuild_scene()
            if self.blocks:
                self.select_block(str(self.blocks[0].get("id")))
            self.refresh_arrows()
            if not silent:
                self._log_to_ws(f"Applied code to visual backbone: {len(self.blocks)} block(s).")
            return True
        except Exception as exc:
            if not silent:
                self.QMessageBox.critical(self.dialog, "Could not apply code", str(exc) + "\n\n" + traceback.format_exc())
            return False

    def refresh_code_from_blocks(self, *_):
        try:
            code = self._block_spec_code() if self._code_mode_is_block_spec() else self._code()
            self._code_text_updating = True
            self.generated_code.setPlainText(code)
            self._code_text_updating = False
        except Exception as exc:
            try:
                self._code_text_updating = True
                self.generated_code.setPlainText("Preview failed: " + str(exc))
                self._code_text_updating = False
            except Exception:
                pass

    def _current_code_for_writing(self) -> str:
        if self._code_mode_is_block_spec():
            self.apply_code_to_blocks(silent=True)
            return self._code()
        code = self.generated_code.toPlainText()
        if "def build_backbone" not in code:
            raise RuntimeError("Manual Python mode must contain def build_backbone(...).")
        return code

    def write_file(self, use_in_tab: bool) -> Optional[Path]:
        try:
            path = self._target_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            code = self._current_code_for_writing()
            compile(code, str(path), "exec")
            path.write_text(code, encoding="utf-8")
            self._log_to_ws(f"Wrote custom visual backbone: {path}")
            if use_in_tab:
                _use_backbone_file_in_training_tab(self.training_page, path, self.out_channels_spin.value())
                self._log_to_ws("Selected generated .py in the R-CNN Trainer tab.")
            self.QMessageBox.information(self.dialog, "Backbone created", f"Created:\n{path}")
            return path
        except Exception as exc:
            self.QMessageBox.critical(self.dialog, "Backbone creation failed", str(exc) + "\n\n" + traceback.format_exc())
            return None

    def test_generated(self):
        path = self.write_file(use_in_tab=False)
        if not path:
            return
        try:
            import importlib.util
            import torch
            spec = importlib.util.spec_from_file_location("mustatil_visual_backbone_test", str(path))
            if spec is None or spec.loader is None:
                raise RuntimeError("Could not load generated file as Python module.")
            mod = importlib.util.module_from_spec(spec)
            sys.modules["mustatil_visual_backbone_test"] = mod
            spec.loader.exec_module(mod)
            bb = mod.build_backbone(out_channels=int(self.out_channels_spin.value()))
            bb.eval()
            with torch.no_grad():
                out = bb(torch.zeros(1, 3, 256, 256))
            if isinstance(out, dict):
                summary = "; ".join(f"{k}: {tuple(v.shape)}" for k, v in out.items())
            else:
                summary = str(type(out))
            self.QMessageBox.information(self.dialog, "Backbone test OK", "Forward test passed.\n\n" + summary)
            self._log_to_ws("Backbone test OK: " + summary)
        except Exception as exc:
            self.QMessageBox.critical(self.dialog, "Backbone test failed", str(exc) + "\n\n" + traceback.format_exc())
            self._log_to_ws("Backbone test failed: " + str(exc))

    def preset_satellite_accuracy(self):
        self.blocks = [
            {"type": "Stem Conv", "out": 48, "stride": 2, "repeats": 1, "feature": True, "name": "Stem: satellite texture"},
            {"type": "Residual Block", "out": 96, "stride": 2, "repeats": 2, "feature": True, "name": "Low-level shapes"},
            {"type": "SE Attention", "out": 96, "stride": 1, "repeats": 1, "feature": False, "name": "Channel attention"},
            {"type": "Depthwise Separable", "out": 192, "stride": 2, "repeats": 2, "feature": True, "name": "Efficient mid features"},
            {"type": "Bottleneck Block", "out": 384, "stride": 2, "repeats": 3, "feature": True, "name": "High-level features"},
            {"type": "SPP Block", "out": 512, "stride": 1, "repeats": 1, "feature": True, "name": "Multi-scale context"},
        ]
        self._ensure_ids(self.blocks); self.out_channels_spin.setValue(256); self.norm_combo.setCurrentText("GroupNorm"); self.act_combo.setCurrentText("SiLU")
        self._rebuild_scene(); self.auto_layout(); self.select_block(str(self.blocks[0].get("id"))); self.refresh_code_from_blocks()

    def preset_fast(self):
        self.blocks = [
            {"type": "Stem Conv", "out": 24, "stride": 2, "repeats": 1, "feature": True, "name": "Fast stem"},
            {"type": "Depthwise Separable", "out": 48, "stride": 2, "repeats": 1, "feature": True, "name": "Fast level 1"},
            {"type": "Depthwise Separable", "out": 96, "stride": 2, "repeats": 1, "feature": True, "name": "Fast level 2"},
            {"type": "Depthwise Separable", "out": 192, "stride": 2, "repeats": 1, "feature": True, "name": "Fast level 3"},
            {"type": "Conv Block", "out": 256, "stride": 2, "repeats": 1, "feature": True, "name": "Final feature"},
        ]
        self._ensure_ids(self.blocks); self.out_channels_spin.setValue(128); self.norm_combo.setCurrentText("GroupNorm"); self.act_combo.setCurrentText("SiLU")
        self._rebuild_scene(); self.auto_layout(); self.select_block(str(self.blocks[0].get("id"))); self.refresh_code_from_blocks()

    def preset_wide(self):
        self.blocks = [
            {"type": "Stem Conv", "out": 64, "stride": 2, "repeats": 1, "feature": True, "name": "Wide stem"},
            {"type": "Residual Block", "out": 128, "stride": 2, "repeats": 3, "feature": True, "name": "Wide low-level"},
            {"type": "SE Attention", "out": 128, "stride": 1, "repeats": 1, "feature": False, "name": "Attention"},
            {"type": "Bottleneck Block", "out": 256, "stride": 2, "repeats": 4, "feature": True, "name": "Wide mid-level"},
            {"type": "Bottleneck Block", "out": 512, "stride": 2, "repeats": 4, "feature": True, "name": "Wide high-level"},
            {"type": "SPP Block", "out": 768, "stride": 1, "repeats": 1, "feature": True, "name": "Wide SPP context"},
        ]
        self._ensure_ids(self.blocks); self.out_channels_spin.setValue(256); self.norm_combo.setCurrentText("GroupNorm"); self.act_combo.setCurrentText("SiLU")
        self._rebuild_scene(); self.auto_layout(); self.select_block(str(self.blocks[0].get("id"))); self.refresh_code_from_blocks()

    def preset_desert_style_robust(self):
        self.blocks = [
            {"type": "Stem Conv", "out": 40, "stride": 2, "repeats": 1, "feature": True, "name": "Stem: desert texture"},
            {"type": "Style Adapter", "out": 64, "stride": 1, "repeats": 1, "feature": False, "name": "Normalize satellite style"},
            {"type": "Residual Block", "out": 96, "stride": 2, "repeats": 2, "feature": True, "name": "Low-level faint edges"},
            {"type": "Feature Gate", "out": 96, "stride": 1, "repeats": 1, "feature": False, "name": "Soft gate: useful channels"},
            {"type": "Dilated Context", "out": 192, "stride": 2, "repeats": 2, "feature": True, "name": "Wider desert context"},
            {"type": "Bottleneck Block", "out": 384, "stride": 2, "repeats": 3, "feature": True, "name": "High-level archaeology"},
            {"type": "SPP Block", "out": 512, "stride": 1, "repeats": 1, "feature": True, "name": "Multi-scale mustatil context"},
        ]
        self._ensure_ids(self.blocks); self.out_channels_spin.setValue(256); self.norm_combo.setCurrentText("GroupNorm"); self.act_combo.setCurrentText("SiLU")
        self._rebuild_scene(); self.auto_layout(); self.select_block(str(self.blocks[0].get("id"))); self.refresh_code_from_blocks()

    def preset_tiny_faint_objects(self):
        self.blocks = [
            {"type": "Stem Conv", "out": 32, "stride": 1, "repeats": 1, "feature": True, "name": "Detail-preserving stem"},
            {"type": "Residual Block", "out": 64, "stride": 1, "repeats": 2, "feature": True, "name": "Faint line details"},
            {"type": "Style Adapter", "out": 64, "stride": 1, "repeats": 1, "feature": False, "name": "Style adapter"},
            {"type": "Residual Block", "out": 128, "stride": 2, "repeats": 2, "feature": True, "name": "Small object level"},
            {"type": "Depthwise Separable", "out": 192, "stride": 2, "repeats": 2, "feature": True, "name": "Efficient mid level"},
            {"type": "Dilated Context", "out": 256, "stride": 1, "repeats": 1, "feature": True, "name": "No-downsample context"},
            {"type": "Bottleneck Block", "out": 384, "stride": 2, "repeats": 2, "feature": True, "name": "High-level faint objects"},
        ]
        self._ensure_ids(self.blocks); self.out_channels_spin.setValue(256); self.norm_combo.setCurrentText("GroupNorm"); self.act_combo.setCurrentText("SiLU")
        self._rebuild_scene(); self.auto_layout(); self.select_block(str(self.blocks[0].get("id"))); self.refresh_code_from_blocks()

    def preset_context_heavy(self):
        self.blocks = [
            {"type": "Stem Conv", "out": 48, "stride": 2, "repeats": 1, "feature": True, "name": "Context stem"},
            {"type": "Residual Block", "out": 96, "stride": 2, "repeats": 2, "feature": True, "name": "Local context"},
            {"type": "Dilated Context", "out": 192, "stride": 2, "repeats": 2, "feature": True, "name": "Dilated context 1"},
            {"type": "Feature Gate", "out": 192, "stride": 1, "repeats": 1, "feature": False, "name": "Context gate"},
            {"type": "Dilated Context", "out": 384, "stride": 2, "repeats": 2, "feature": True, "name": "Dilated context 2"},
            {"type": "Bottleneck Block", "out": 512, "stride": 1, "repeats": 3, "feature": True, "name": "Deep context"},
            {"type": "SPP Block", "out": 640, "stride": 1, "repeats": 1, "feature": True, "name": "SPP long-range context"},
        ]
        self._ensure_ids(self.blocks); self.out_channels_spin.setValue(256); self.norm_combo.setCurrentText("GroupNorm"); self.act_combo.setCurrentText("SiLU")
        self._rebuild_scene(); self.auto_layout(); self.select_block(str(self.blocks[0].get("id"))); self.refresh_code_from_blocks()


    def preset_graph_fusion(self):
        """Experimental multi-connection preset for faint desert structures."""
        self.blocks = [
            {"type": "Stem Conv", "out": 40, "stride": 2, "repeats": 1, "feature": True, "name": "Input detail stem"},
            {"type": "Style Adapter", "out": 64, "stride": 1, "repeats": 1, "feature": False, "name": "Desert style adapter"},
            {"type": "Residual Block", "out": 96, "stride": 2, "repeats": 2, "feature": True, "name": "Faint edge level"},
            {"type": "Dilated Context", "out": 192, "stride": 2, "repeats": 2, "feature": True, "name": "Context branch"},
            {"type": "Weighted Sum Gate", "out": 192, "stride": 1, "repeats": 1, "feature": True, "name": "Detail + context gate"},
            {"type": "Scale Mix Gate", "out": 384, "stride": 2, "repeats": 2, "feature": True, "name": "Scale mix gate"},
            {"type": "SPP Block", "out": 512, "stride": 1, "repeats": 1, "feature": True, "name": "Long context SPP"},
        ]
        self._ensure_ids(self.blocks)
        ids = [str(b.get("id")) for b in self.blocks]
        for i, b in enumerate(self.blocks):
            b["inputs"] = [] if i == 0 else [ids[i - 1]]
        self.blocks[4]["inputs"] = [ids[2], ids[3]]
        self.blocks[5]["inputs"] = [ids[1], ids[4]]
        self.out_channels_spin.setValue(256); self.norm_combo.setCurrentText("GroupNorm"); self.act_combo.setCurrentText("SiLU")
        self._rebuild_scene(); self.auto_layout(); self.select_block(str(self.blocks[0].get("id"))); self.refresh_code_from_blocks()


    def load_selected_preset(self):
        name = str(getattr(self, "preset_combo", None).currentText() if hasattr(self, "preset_combo") else "satellite accuracy").lower()
        if "logic" in name:
            return self.preset_logic_detail_context()
        if "expert" in name:
            return self.preset_expert_style_ensemble()
        if "graph" in name:
            return self.preset_graph_fusion()
        if "context" in name:
            return self.preset_context_heavy()
        if "tiny" in name or "faint" in name:
            return self.preset_tiny_faint_objects()
        if "desert" in name or "style" in name:
            return self.preset_desert_style_robust()
        if "wide" in name:
            return self.preset_wide()
        if "fast" in name:
            return self.preset_fast()
        return self.preset_satellite_accuracy()

    def preset_logic_detail_context(self):
        """Two-branch double-check preset using soft AND/OR/NOT logic gates."""
        self.blocks = [
            {"type": "Stem Conv", "out": 40, "stride": 2, "repeats": 1, "feature": True, "name": "Stem: shared satellite features"},
            {"type": "Residual Block", "out": 80, "stride": 1, "repeats": 2, "feature": True, "name": "Detail path: faint line evidence"},
            {"type": "Dilated Context", "out": 128, "stride": 2, "repeats": 2, "feature": True, "name": "Context path: long shape evidence"},
            {"type": "Style Adapter", "out": 80, "stride": 1, "repeats": 1, "feature": False, "name": "Style path: desert normalization"},
            {"type": "Soft AND Gate", "out": 192, "stride": 1, "repeats": 1, "feature": True, "name": "Soft AND: detail + context agree"},
            {"type": "Soft OR Gate", "out": 192, "stride": 1, "repeats": 1, "feature": True, "name": "Soft OR: style fallback"},
            {"type": "Bottleneck Block", "out": 320, "stride": 2, "repeats": 3, "feature": True, "name": "Deep Mustatil shape"},
            {"type": "Iterative Refinement", "out": 384, "stride": 1, "repeats": 3, "feature": True, "name": "Refine / check again"},
            {"type": "SPP Block", "out": 512, "stride": 1, "repeats": 1, "feature": True, "name": "Final SPP context"},
        ]
        self._ensure_ids(self.blocks)
        ids = [str(b.get("id")) for b in self.blocks]
        self.blocks[0]["inputs"] = []
        self.blocks[1]["inputs"] = [ids[0]]
        self.blocks[2]["inputs"] = [ids[0]]
        self.blocks[3]["inputs"] = [ids[0]]
        self.blocks[4]["inputs"] = [ids[1], ids[2]]
        self.blocks[5]["inputs"] = [ids[4], ids[3]]
        self.blocks[6]["inputs"] = [ids[5]]
        self.blocks[7]["inputs"] = [ids[6]]
        self.blocks[8]["inputs"] = [ids[7]]
        self.out_channels_spin.setValue(256); self.norm_combo.setCurrentText("GroupNorm"); self.act_combo.setCurrentText("SiLU")
        self._rebuild_scene(); self.auto_layout(); self.select_block(str(self.blocks[0].get("id"))); self.refresh_code_from_blocks()

    def preset_expert_style_ensemble(self):
        """Three-branch mixture-of-experts preset for changing desert styles."""
        self.blocks = [
            {"type": "Stem Conv", "out": 48, "stride": 2, "repeats": 1, "feature": True, "name": "Shared stem"},
            {"type": "Residual Block", "out": 96, "stride": 1, "repeats": 2, "feature": True, "name": "Expert A: fine lines"},
            {"type": "Dilated Context", "out": 160, "stride": 2, "repeats": 2, "feature": True, "name": "Expert B: context"},
            {"type": "Style Gate", "out": 128, "stride": 1, "repeats": 1, "feature": True, "name": "Expert C: style"},
            {"type": "Mixture-of-Experts Gate", "out": 256, "stride": 1, "repeats": 1, "feature": True, "name": "Mixture: choose expert weights"},
            {"type": "Soft NOT Gate", "out": 256, "stride": 1, "repeats": 1, "feature": False, "name": "Suppress false textures"},
            {"type": "Scale Mix Gate", "out": 384, "stride": 2, "repeats": 2, "feature": True, "name": "Scale mix"},
            {"type": "Bottleneck Block", "out": 512, "stride": 2, "repeats": 3, "feature": True, "name": "Deep archaeology"},
            {"type": "SPP Block", "out": 640, "stride": 1, "repeats": 1, "feature": True, "name": "SPP long context"},
        ]
        self._ensure_ids(self.blocks)
        ids = [str(b.get("id")) for b in self.blocks]
        self.blocks[0]["inputs"] = []
        self.blocks[1]["inputs"] = [ids[0]]
        self.blocks[2]["inputs"] = [ids[0]]
        self.blocks[3]["inputs"] = [ids[0]]
        self.blocks[4]["inputs"] = [ids[1], ids[2], ids[3]]
        self.blocks[5]["inputs"] = [ids[4], ids[3]]
        self.blocks[6]["inputs"] = [ids[5]]
        self.blocks[7]["inputs"] = [ids[6]]
        self.blocks[8]["inputs"] = [ids[7]]
        self.out_channels_spin.setValue(256); self.norm_combo.setCurrentText("GroupNorm"); self.act_combo.setCurrentText("SiLU")
        self._rebuild_scene(); self.auto_layout(); self.select_block(str(self.blocks[0].get("id"))); self.refresh_code_from_blocks()


def _find_custom_backbone_line(page: Any):
    try:
        from PySide6.QtWidgets import QLineEdit
    except Exception:
        return None
    candidates = []
    try:
        for le in page.findChildren(QLineEdit):
            ph = str(le.placeholderText() or "").lower()
            txt = str(le.text() or "").lower()
            if "custom python backbone" in ph or "custom backbone" in ph:
                return le
            if txt.endswith(".py") or "backbone" in ph:
                candidates.append(le)
    except Exception:
        pass
    return candidates[0] if candidates else None


def _find_form_field_by_label(page: Any, label_substring: str):
    try:
        from PySide6.QtWidgets import QFormLayout
    except Exception:
        return None
    want = str(label_substring).lower()
    try:
        for layout in page.findChildren(QFormLayout):
            for row in range(layout.rowCount()):
                lab = layout.itemAt(row, QFormLayout.LabelRole)
                fld = layout.itemAt(row, QFormLayout.FieldRole)
                if lab is None or fld is None:
                    continue
                lw = lab.widget()
                text = ""
                try:
                    text = str(lw.text())
                except Exception:
                    pass
                if want in text.lower():
                    return fld.widget()
    except Exception:
        pass
    return None


def _use_backbone_file_in_training_tab(page: Any, path: Path, out_channels: int = 256) -> None:
    try:
        from PySide6.QtWidgets import QComboBox, QSpinBox
    except Exception:
        return
    # Select custom backbone combo.
    try:
        for combo in page.findChildren(QComboBox):
            for i in range(combo.count()):
                if "custom python backbone" in str(combo.itemText(i)).lower():
                    combo.setCurrentIndex(i)
                    break
    except Exception:
        pass
    # Fill custom .py line.
    line = _find_custom_backbone_line(page)
    if line is not None:
        try:
            line.setText(str(path))
        except Exception:
            pass
    # Try to set the custom out-channels field without touching unrelated spinners.
    field = _find_form_field_by_label(page, "Custom out channels")
    if isinstance(field, QSpinBox):
        try:
            field.setValue(int(out_channels))
        except Exception:
            pass


def _open_backbone_creator(parent_page: Any) -> None:
    try:
        ws = _workspace_from_widget(parent_page)
        dlg = _BackboneBlockCreatorDialog(parent_page, ws, parent_page)
        dlg.exec()
    except Exception as exc:
        try:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(parent_page, PLUGIN_TITLE, str(exc) + "\n\n" + traceback.format_exc())
        except Exception:
            _log("Could not open creator: " + str(exc))


def _patch_training_page(page: Any) -> bool:
    if page is None or id(page) in _PATCHED_PAGES:
        return False
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout
    except Exception:
        return False
    try:
        # Only patch the R-CNN Trainer page / R-CNN Training Studio pages.
        text_blob = ""
        try:
            for lab in page.findChildren(QLabel):
                text_blob += "\n" + str(lab.text())
        except Exception:
            pass
        if "r-cnn" not in text_blob.lower() and "backbone" not in text_blob.lower():
            # Object attribute from previous patches is also accepted.
            if not any(str(getattr(page, name, "")).lower() for name in ("_mustatil_rcnn_custom_backbone_creator_patch", "_mustatil_rcnn_training_studio")):
                return False

        root = page.layout()
        if root is None or not isinstance(root, QVBoxLayout):
            return False
        try:
            for old_bar in page.findChildren(QFrame, "MustatilBackboneBlockCreatorTopBar"):
                old_bar.setVisible(False)
        except Exception:
            pass

        bar = QFrame(page)
        bar.setObjectName("MustatilBackboneVisualPipelineCreatorTopBar")
        try:
            bar.setFrameShape(QFrame.StyledPanel)
        except Exception:
            pass
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(8, 5, 8, 5)
        lay.setSpacing(6)
        title = QLabel("Backbone tools")
        try:
            title.setStyleSheet("font-weight: 600;")
        except Exception:
            pass
        btn = QPushButton("Backbone Creator")
        btn.setMinimumHeight(34)
        btn.setToolTip("Open an interactive block editor to create a Custom Python Backbone for this R-CNN training tab.")
        note = QLabel("Open the visual block pipeline: add CNN/FPN blocks, read function descriptions, export .py, and use it as Custom Python Backbone.")
        note.setWordWrap(True)
        lay.addWidget(title)
        lay.addWidget(btn)
        lay.addWidget(note, 1)
        btn.clicked.connect(lambda _=False, p=page: _open_backbone_creator(p))

        insert_at = 1 if root.count() >= 1 else 0
        root.insertWidget(insert_at, bar)
        _PATCHED_PAGES.add(id(page))
        try:
            ws = _workspace_from_widget(page)
            if ws is not None:
                ws.log("Visual Backbone Creator button added to R-CNN Trainer.")
        except Exception:
            pass
        return True
    except Exception as exc:
        _log("patch page failed: " + str(exc))
        return False


def _scan_for_training_tabs() -> None:
    try:
        from PySide6.QtWidgets import QApplication, QTabWidget
    except Exception:
        return
    app = QApplication.instance()
    if app is None:
        return
    for tw in app.allWidgets():
        try:
            if not isinstance(tw, QTabWidget):
                continue
            for i in range(tw.count()):
                label = str(tw.tabText(i) or "").lower()
                if "r-cnn trainer" in label or "r-cnn" in label and "train" in label:
                    _patch_training_page(tw.widget(i))
        except Exception:
            pass


def _patch_qtabwidget() -> None:
    global _PATCHED_QTAB, _ORIG_ADD_TAB, _ORIG_INSERT_TAB
    if _PATCHED_QTAB:
        return
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QTabWidget
    except Exception as exc:
        _log("Qt unavailable; cannot patch tabs: " + str(exc))
        return

    _ORIG_ADD_TAB = QTabWidget.addTab
    _ORIG_INSERT_TAB = QTabWidget.insertTab

    def add_tab(self, *args, **kwargs):
        res = _ORIG_ADD_TAB(self, *args, **kwargs)
        try:
            label = str(args[1] if len(args) >= 2 else "")
            widget = args[0] if args else None
            if "r-cnn" in label.lower() and "trainer" in label.lower():
                QTimer.singleShot(0, lambda w=widget: _patch_training_page(w))
        except Exception:
            pass
        return res

    def insert_tab(self, *args, **kwargs):
        res = _ORIG_INSERT_TAB(self, *args, **kwargs)
        try:
            widget = args[1] if len(args) >= 2 else None
            label = str(args[2] if len(args) >= 3 else "")
            if "r-cnn" in label.lower() and "trainer" in label.lower():
                QTimer.singleShot(0, lambda w=widget: _patch_training_page(w))
        except Exception:
            pass
        return res

    QTabWidget.addTab = add_tab
    QTabWidget.insertTab = insert_tab
    _PATCHED_QTAB = True

    global _SCAN_TIMER
    _SCAN_TIMER = QTimer()
    _SCAN_TIMER.setInterval(1200)
    _SCAN_TIMER.timeout.connect(_scan_for_training_tabs)
    _SCAN_TIMER.start()
    QTimer.singleShot(0, _scan_for_training_tabs)
    QTimer.singleShot(2500, _scan_for_training_tabs)
    _log("Backbone Creator tab patch installed.")


def mustatil_plugin_init():
    _patch_qtabwidget()
