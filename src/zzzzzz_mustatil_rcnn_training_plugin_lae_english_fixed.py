#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mustatil plugin: U-Net Semantic Segmentation Trainer

Drop this file into Mustatil's mustatil_plugins folder and restart Mustatil.
It adds a new top-level tab named "U-Net Trainer" and places it next to the
R-CNN Trainer tab when present, with LAE-DINO Trainer / YOLO Trainer as fallback.
It does not modify Detection preview, Satellite preview, or project files.

Training input:
  - Images: jpg/png/tif/tiff/webp/bmp, recursively scanned
  - Labels: YOLO txt labels with normalized x_center y_center width height
  - Optional SAM2/JSON polygon sidecars can be used when present; otherwise box masks are used.

Outputs:
  - .pth checkpoints compatible with the companion U-Net Segmentation detection tab
  - classes.json and training_metadata.json next to checkpoints

Compatibility note:
  - Binary mode saves a Simple U-Net with output_channels=1. In the Detection U-Net tab,
    set "U-Net output channels" to 1 and the same "Simple U-Net base channels" value.
  - Multiclass mode saves output_channels = number_of_classes + 1 because channel 0 is background.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_PATCHED_QTAB = False
_ORIG_ADD_TAB = None
_ORIG_INSERT_TAB = None
_SCAN_TIMER = None
_INSTALLING_TAB = False
_INSTALLED_WS = set()

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def _log(msg: str) -> None:
    try:
        print("[Mustatil U-Net Trainer] " + str(msg), flush=True)
    except Exception:
        pass


def _workspace_from_widget(widget: Any) -> Optional[Any]:
    cur = widget
    for _ in range(160):
        if cur is None:
            break
        try:
            if hasattr(cur, "tabs") and hasattr(cur, "project") and hasattr(cur, "run_task"):
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


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _safe_name(text: str) -> str:
    bad = '<>:"/\\|?*\n\r\t'
    out = "".join("_" if c in bad else c for c in str(text))
    out = "_".join(out.strip().split())
    return out or "model"


def _parse_classes(text: str) -> List[str]:
    vals = [p.strip() for p in str(text or "").replace(";", ",").split(",") if p.strip()]
    return vals or ["mustatil", "false_positive"]


def _project_guess(ws: Any) -> Path:
    for attr in ("project", "project_create_dir"):
        try:
            p = Path(_var_get(getattr(ws, attr, None), "").strip())
            if str(p) and p.exists():
                return p
        except Exception:
            pass
    try:
        p = Path(getattr(getattr(ws, "project_state", None), "project_root", "") or "")
        if str(p) and p.exists():
            return p
    except Exception:
        pass
    return Path.cwd()


def _guess_image_dir(project: Path) -> Path:
    candidates = [
        project / "images",
        project / "image",
        project / "dataset" / "images",
        project / "train" / "images",
        project,
    ]
    for p in candidates:
        try:
            if p.exists() and any(p.rglob("*" + ext) for ext in IMG_EXTS):
                return p
        except Exception:
            pass
    return project / "images"


def _guess_label_dir(project: Path) -> Path:
    candidates = [
        project / "labels",
        project / "label",
        project / "dataset" / "labels",
        project / "train" / "labels",
        project,
    ]
    for p in candidates:
        try:
            if p.exists() and any(p.rglob("*.txt")):
                return p
        except Exception:
            pass
    return project / "labels"


def _find_label_for_image(img: Path, image_dir: Path, label_dir: Path) -> Path:
    try:
        rel = img.relative_to(image_dir).with_suffix(".txt")
        cand = label_dir / rel
        if cand.exists():
            return cand
    except Exception:
        pass
    cand = label_dir / (img.stem + ".txt")
    if cand.exists():
        return cand
    try:
        matches = sorted(label_dir.rglob(img.stem + ".txt"), key=lambda p: len(str(p)))
        if matches:
            return matches[0]
    except Exception:
        pass
    return cand


def _collect_images(image_dir: Path, max_images: int = 0) -> List[Path]:
    imgs: List[Path] = []
    if image_dir.exists():
        for p in image_dir.rglob("*"):
            try:
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    imgs.append(p)
            except Exception:
                pass
    imgs = sorted(imgs, key=lambda p: str(p).lower())
    if max_images and max_images > 0:
        imgs = imgs[: int(max_images)]
    return imgs


def _read_yolo_txt(label_path: Path, width: int, height: int, num_classes: int, min_box_px: float = 1.0):
    boxes: List[List[float]] = []
    labels: List[int] = []
    if not label_path.exists():
        return boxes, labels
    try:
        lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        lines = []
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.replace(",", " ").split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])
        except Exception:
            continue
        if cls < 0 or cls >= max(1, num_classes):
            continue
        x1 = (xc - bw / 2.0) * width
        y1 = (yc - bh / 2.0) * height
        x2 = (xc + bw / 2.0) * width
        y2 = (yc + bh / 2.0) * height
        x1 = max(0.0, min(float(width - 1), x1))
        y1 = max(0.0, min(float(height - 1), y1))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))
        if x2 - x1 < min_box_px or y2 - y1 < min_box_px:
            continue
        boxes.append([float(x1), float(y1), float(x2), float(y2)])
        labels.append(int(cls))
    return boxes, labels


def _load_optional_polygons(img: Path, label_path: Path, width: int, height: int) -> List[List[Tuple[float, float]]]:
    """Best-effort parser for SAM2/Mustatil sidecar JSON polygon files.

    Accepted loose formats:
      - [{"points": [[x,y], ...]}, ...]
      - [{"polygon": [[x,y], ...]}, ...]
      - {"polygons": [...]} / {"masks": [...]} / GeoJSON-ish features
    Normalized 0..1 coordinates are scaled to pixels.
    """
    candidates = [
        img.with_suffix(".sam2.json"),
        img.with_suffix(".json"),
        label_path.with_suffix(".sam2.json"),
        label_path.with_suffix(".json"),
    ]
    data = None
    for p in candidates:
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                break
            except Exception:
                data = None
    if data is None:
        return []

    def unwrap(x):
        if isinstance(x, dict):
            for key in ("polygons", "masks", "items", "annotations", "features"):
                if key in x:
                    return unwrap(x[key])
            return [x]
        if isinstance(x, list):
            return x
        return []

    def get_points(item):
        if isinstance(item, dict):
            for key in ("points", "polygon", "poly", "coordinates", "mask_polygon"):
                if key in item:
                    return item.get(key)
            geom = item.get("geometry")
            if isinstance(geom, dict):
                coords = geom.get("coordinates")
                typ = str(geom.get("type", "")).lower()
                if typ == "polygon" and coords:
                    return coords[0]
                if typ == "multipolygon" and coords:
                    return coords[0][0]
        return item

    polys: List[List[Tuple[float, float]]] = []
    for it in unwrap(data):
        pts_raw = get_points(it)
        if not isinstance(pts_raw, (list, tuple)):
            continue
        if pts_raw and isinstance(pts_raw[0], (list, tuple)) and pts_raw[0] and isinstance(pts_raw[0][0], (list, tuple)):
            pts_raw = pts_raw[0]
        pts: List[Tuple[float, float]] = []
        for p in pts_raw:
            try:
                x, y = float(p[0]), float(p[1])
                pts.append((x, y))
            except Exception:
                pass
        if len(pts) < 3:
            continue
        max_abs = max(max(abs(x), abs(y)) for x, y in pts)
        if max_abs <= 1.5:
            pts = [(x * width, y * height) for x, y in pts]
        pts = [(max(0.0, min(float(width), x)), max(0.0, min(float(height), y))) for x, y in pts]
        if len(pts) >= 3:
            polys.append(pts)
    return polys


# -----------------------------------------------------------------------------
# Simple U-Net architecture. Keep names identical to the Detection U-Net plugin.
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
            import torch
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


def _strip_state_dict_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in state.items():
        kk = str(k)
        for pref in ("module.", "model.", "net."):
            if kk.startswith(pref):
                kk = kk[len(pref):]
        out[kk] = v
    return out


def _load_checkpoint_flexible(model, checkpoint_path: str, logger=None) -> None:
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return
    import torch
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state = ckpt
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint has no compatible state_dict.")
    state = _strip_state_dict_prefix(state)
    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if logger:
            logger(f"Checkpoint loaded with strict=False: missing={len(missing)}, unexpected={len(unexpected)}")
        return
    except Exception as exc:
        if logger:
            logger("Direct checkpoint load failed, filtering incompatible tensors: " + str(exc))
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
    if logger:
        logger(f"Checkpoint partially loaded: {len(filtered)} tensors loaded, {len(skipped)} skipped.")


def _resolve_device(text: str):
    import torch
    raw = str(text or "auto").lower().strip()
    if raw == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if raw == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MustatilYoloSemanticTileDataset:
    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        classes: Sequence[str],
        target_mode: str = "binary",
        mask_source: str = "boxes",
        tile_size: int = 512,
        samples_per_image: int = 4,
        max_images: int = 0,
        include_empty: bool = True,
        positive_crop_probability: float = 0.70,
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.classes = list(classes)
        self.num_classes = len(self.classes)
        self.target_mode = str(target_mode or "binary").lower()
        self.mask_source = str(mask_source or "boxes").lower()
        self.tile_size = max(64, int(tile_size or 512))
        self.samples_per_image = max(1, int(samples_per_image or 1))
        self.include_empty = bool(include_empty)
        self.positive_crop_probability = max(0.0, min(1.0, float(positive_crop_probability)))
        imgs = _collect_images(self.image_dir, int(max_images or 0))
        if not self.include_empty:
            kept = []
            for img in imgs:
                lab = _find_label_for_image(img, self.image_dir, self.label_dir)
                try:
                    if lab.exists() and lab.read_text(encoding="utf-8", errors="ignore").strip():
                        kept.append(img)
                except Exception:
                    pass
            imgs = kept
        self.images = imgs

    def __len__(self):
        return len(self.images) * self.samples_per_image

    def _choose_crop(self, width: int, height: int, boxes: Sequence[Sequence[float]]):
        ts = self.tile_size
        max_x = max(0, width - ts)
        max_y = max(0, height - ts)
        if boxes and random.random() < self.positive_crop_probability:
            box = random.choice(list(boxes))
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            cx = random.uniform(x1, x2) if x2 > x1 else x1
            cy = random.uniform(y1, y2) if y2 > y1 else y1
            x0 = int(round(cx - random.uniform(0.25, 0.75) * ts))
            y0 = int(round(cy - random.uniform(0.25, 0.75) * ts))
        else:
            x0 = random.randint(0, max_x) if max_x > 0 else 0
            y0 = random.randint(0, max_y) if max_y > 0 else 0
        x0 = max(0, min(max_x, x0))
        y0 = max(0, min(max_y, y0))
        return x0, y0, min(width, x0 + ts), min(height, y0 + ts)

    def _crop_image_padded(self, pil, crop):
        from PIL import Image
        ts = self.tile_size
        x0, y0, x1, y1 = crop
        c = pil.crop((int(x0), int(y0), int(x1), int(y1))).convert("RGB")
        if c.size == (ts, ts):
            return c
        out = Image.new("RGB", (ts, ts), (0, 0, 0))
        out.paste(c, (0, 0))
        return out

    def _make_crop_mask(self, width: int, height: int, img_path: Path, label_path: Path, boxes, labels, crop):
        from PIL import Image, ImageDraw
        ts = self.tile_size
        x0, y0, x1, y1 = [float(v) for v in crop]
        target_mode = self.target_mode
        mask = Image.new("L", (ts, ts), 0)
        draw = ImageDraw.Draw(mask)

        def value_for(cls: int) -> int:
            if target_mode.startswith("multi"):
                return int(cls) + 1
            return 1

        polys: List[List[Tuple[float, float]]] = []
        if "sam" in self.mask_source or "polygon" in self.mask_source:
            polys = _load_optional_polygons(img_path, label_path, width, height)

        used_poly = False
        if polys:
            # Best-effort: if polygon count matches/approximates label count, pair by order.
            for i, poly in enumerate(polys):
                cls = labels[i] if i < len(labels) else 0
                pts = [(float(px) - x0, float(py) - y0) for px, py in poly]
                if len(pts) >= 3:
                    try:
                        draw.polygon(pts, outline=value_for(cls), fill=value_for(cls))
                        used_poly = True
                    except Exception:
                        pass

        if not used_poly:
            for box, cls in zip(boxes, labels):
                bx1, by1, bx2, by2 = [float(v) for v in box[:4]]
                ix1 = max(0.0, bx1 - x0); iy1 = max(0.0, by1 - y0)
                ix2 = min(float(ts), bx2 - x0); iy2 = min(float(ts), by2 - y0)
                if ix2 > ix1 and iy2 > iy1:
                    draw.rectangle([ix1, iy1, ix2, iy2], outline=value_for(cls), fill=value_for(cls))
        return mask

    def __getitem__(self, idx: int):
        import numpy as np
        import torch
        from PIL import Image
        if not self.images:
            raise IndexError("Dataset is empty.")
        img_path = self.images[int(idx) % len(self.images)]
        pil = Image.open(img_path).convert("RGB")
        width, height = pil.size
        label_path = _find_label_for_image(img_path, self.image_dir, self.label_dir)
        boxes, labels = _read_yolo_txt(label_path, width, height, self.num_classes)
        crop = self._choose_crop(width, height, boxes)
        pil_crop = self._crop_image_padded(pil, crop)
        mask_pil = self._make_crop_mask(width, height, img_path, label_path, boxes, labels, crop)

        img_arr = np.array(pil_crop, dtype="uint8")
        x = torch.from_numpy(img_arr).permute(2, 0, 1).float().div(255.0)
        m = torch.from_numpy(np.array(mask_pil, dtype="uint8"))
        if self.target_mode.startswith("multi"):
            y = m.long()
        else:
            y = (m > 0).float().unsqueeze(0)
        return x, y


def _dice_loss_binary(logits, targets, eps: float = 1e-6):
    import torch
    probs = torch.sigmoid(logits)
    targets = targets.float()
    dims = tuple(range(1, probs.ndim))
    inter = (probs * targets).sum(dim=dims)
    den = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * inter + eps) / (den + eps)
    return 1.0 - dice.mean()


def _dice_loss_multiclass(logits, targets, num_classes: int, eps: float = 1e-6):
    import torch
    import torch.nn.functional as F
    probs = torch.softmax(logits, dim=1)
    total = 0.0
    count = 0
    # skip background channel 0
    for c in range(1, int(num_classes)):
        pc = probs[:, c]
        tc = (targets == c).float()
        inter = (pc * tc).sum(dim=(1, 2))
        den = pc.sum(dim=(1, 2)) + tc.sum(dim=(1, 2))
        total = total + (1.0 - ((2.0 * inter + eps) / (den + eps))).mean()
        count += 1
    if count <= 0:
        return logits.sum() * 0.0
    return total / float(count)


def _train_unet_worker(cfg: Dict[str, Any], logger=None) -> str:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    def log(msg):
        if logger:
            logger(str(msg))
        else:
            _log(str(msg))

    image_dir = Path(cfg["image_dir"])
    label_dir = Path(cfg["label_dir"])
    out_dir = Path(cfg["out_dir"])
    classes = list(cfg.get("classes") or ["mustatil", "false_positive"])
    target_mode = str(cfg.get("target_mode", "binary")).lower()
    out_channels = 1 if target_mode.startswith("binary") else len(classes) + 1
    base = int(cfg.get("base", 32) or 32)
    tile_size = int(cfg.get("tile_size", 512) or 512)

    if not image_dir.exists():
        raise RuntimeError(f"Image folder does not exist: {image_dir}")
    if not label_dir.exists():
        raise RuntimeError(f"Label folder does not exist: {label_dir}")
    _ensure_dir(out_dir)

    run_dir = out_dir / f"unet_{'binary' if out_channels == 1 else 'multiclass'}_b{base}_{_now_stamp()}"
    _ensure_dir(run_dir)
    log(f"Output run folder: {run_dir}")

    dataset = MustatilYoloSemanticTileDataset(
        image_dir=str(image_dir),
        label_dir=str(label_dir),
        classes=classes,
        target_mode=target_mode,
        mask_source=str(cfg.get("mask_source", "boxes")),
        tile_size=tile_size,
        samples_per_image=int(cfg.get("samples_per_image", 4) or 4),
        max_images=int(cfg.get("max_images", 0) or 0),
        include_empty=bool(cfg.get("include_empty", True)),
        positive_crop_probability=float(cfg.get("positive_crop_probability", 0.70)),
    )
    if len(dataset.images) <= 0:
        raise RuntimeError("No training images found.")
    if len(dataset) <= 0:
        raise RuntimeError("Dataset is empty after filtering.")
    log(f"Images: {len(dataset.images)}")
    log(f"Training samples per epoch: {len(dataset)}")
    log(f"Target mode: {target_mode}; U-Net output channels: {out_channels}; base channels: {base}")

    device = _resolve_device(str(cfg.get("device", "auto")))
    log(f"Selected device: {device}")
    model = _make_unet(in_channels=3, out_channels=out_channels, base=base).to(device)
    ckpt_path = str(cfg.get("checkpoint_path", "")).strip().strip('"')
    if ckpt_path:
        _load_checkpoint_flexible(model, ckpt_path, logger=log)

    batch = int(cfg.get("batch", 2) or 2)
    workers = int(cfg.get("workers", 0) or 0)
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=(str(device) == "cuda"))

    lr = float(cfg.get("lr", 0.001) or 0.001)
    wd = float(cfg.get("weight_decay", 0.0001) or 0.0001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    epochs = int(cfg.get("epochs", 30) or 30)
    save_every = max(1, int(cfg.get("save_every", 1) or 1))
    pos_weight_value = max(0.1, float(cfg.get("positive_weight", 5.0) or 5.0))

    metadata = {
        "model": "Simple U-Net",
        "in_channels": 3,
        "out_channels": out_channels,
        "base_channels": base,
        "target_mode": target_mode,
        "classes": classes,
        "background_channel": 0 if out_channels > 1 else None,
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "tile_size": tile_size,
        "samples_per_image": int(cfg.get("samples_per_image", 4) or 4),
        "mask_source": str(cfg.get("mask_source", "boxes")),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "compatibility_note": "Use the companion U-Net Segmentation tab with matching output_channels and base_channels.",
    }
    (run_dir / "classes.json").write_text(json.dumps({"classes": classes}, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    model.train()
    log(f"Training started: epochs={epochs}, batch={batch}, lr={lr}, weight_decay={wd}")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_bce_ce = 0.0
        total_dice = 0.0
        n_batches = 0
        t0 = time.time()
        for bi, (images, masks) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            if isinstance(logits, dict):
                logits = logits.get("out", next(iter(logits.values())))
            if logits.shape[-2:] != images.shape[-2:]:
                logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
            if out_channels == 1:
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1).float()
                pw = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
                bce = F.binary_cross_entropy_with_logits(logits, masks.float(), pos_weight=pw)
                dice = _dice_loss_binary(logits, masks)
                loss = bce + dice
            else:
                if masks.ndim == 4:
                    masks = masks[:, 0]
                ce = F.cross_entropy(logits, masks.long())
                dice = _dice_loss_multiclass(logits, masks.long(), num_classes=out_channels)
                bce = ce
                loss = ce + dice
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += float(loss.detach().cpu().item())
            total_bce_ce += float(bce.detach().cpu().item())
            total_dice += float(dice.detach().cpu().item())
            n_batches += 1
            if bi == 1 or bi % 20 == 0:
                log(f"Epoch {epoch}/{epochs} batch {bi}/{len(loader)} loss={total_loss/max(1,n_batches):.5f}")

        avg = total_loss / max(1, n_batches)
        avg_a = total_bce_ce / max(1, n_batches)
        avg_d = total_dice / max(1, n_batches)
        log(f"Epoch {epoch}/{epochs} finished: loss={avg:.5f}, BCE/CE={avg_a:.5f}, dice={avg_d:.5f}, time={time.time()-t0:.1f}s")

        if epoch % save_every == 0 or epoch == epochs:
            ckpt = run_dir / f"unet_{'binary' if out_channels == 1 else 'multiclass'}_b{base}_epoch_{epoch:03d}.pth"
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "metadata": metadata,
                "epoch": epoch,
            }, str(ckpt))
            log(f"Checkpoint saved: {ckpt}")

    final = run_dir / f"unet_{'binary' if out_channels == 1 else 'multiclass'}_b{base}_final.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "metadata": metadata,
        "epoch": epochs,
    }, str(final))
    readme = run_dir / "README_training_result.txt"
    readme.write_text(
        "Mustatil U-Net training result\n"
        "================================\n\n"
        f"Final checkpoint: {final.name}\n"
        f"Target mode: {target_mode}\n"
        f"Output channels: {out_channels}\n"
        f"Base channels: {base}\n\n"
        "Use this checkpoint in the U-Net Segmentation tab. Set the same output channels and base channels.\n",
        encoding="utf-8",
    )
    log(f"Final checkpoint saved: {final}")
    return str(final)


def _validate_dataset(cfg: Dict[str, Any], logger=None) -> None:
    from PIL import Image
    def log(msg):
        if logger:
            logger(str(msg))
        else:
            _log(str(msg))
    image_dir = Path(cfg["image_dir"])
    label_dir = Path(cfg["label_dir"])
    classes = list(cfg.get("classes") or ["mustatil", "false_positive"])
    images = _collect_images(image_dir, int(cfg.get("max_images", 0) or 0))
    log(f"Image folder: {image_dir}")
    log(f"Label folder: {label_dir}")
    log(f"Images found: {len(images)}")
    if not images:
        return
    missing_labels = 0
    bad_images = 0
    nonempty = 0
    total_boxes = 0
    class_counts = {i: 0 for i in range(len(classes))}
    for img in images[:500]:
        try:
            pil = Image.open(img)
            w, h = pil.size
            lab = _find_label_for_image(img, image_dir, label_dir)
            if not lab.exists():
                missing_labels += 1
            boxes, labels = _read_yolo_txt(lab, w, h, len(classes))
            total_boxes += len(boxes)
            if boxes:
                nonempty += 1
            for c in labels:
                if c in class_counts:
                    class_counts[c] += 1
        except Exception:
            bad_images += 1
    log(f"Checked images: {min(len(images), 500)}")
    log(f"Images with foreground labels: {nonempty}")
    log(f"YOLO boxes: {total_boxes}")
    log("Class counts: " + json.dumps({classes[k] if k < len(classes) else str(k): v for k, v in class_counts.items()}, ensure_ascii=False))
    log(f"Missing label txt: {missing_labels}")
    log(f"Unreadable images: {bad_images}")
    log("Binary target mode creates foreground/background masks from all non-empty labels.")


def _build_training_tab(ws: Any):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel, QLineEdit,
        QPushButton, QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
        QTextEdit, QSplitter
    )

    project = _project_guess(ws)
    image_guess = _guess_image_dir(project)
    label_guess = _guess_label_dir(project)
    out_guess = project / "runs" / "unet_training"

    page = QWidget()
    root = QVBoxLayout(page)
    root.setContentsMargins(8, 8, 8, 8)

    header = QLabel(
        "U-Net Semantic Segmentation Trainer for Mustatil/YOLO labels. "
        "It converts YOLO boxes or optional SAM2/JSON polygons into semantic masks. "
        "The trained checkpoint is compatible with the U-Net Segmentation tab in Detection and Satellite Detection."
    )
    header.setWordWrap(True)
    root.addWidget(header)

    splitter = QSplitter()
    root.addWidget(splitter, 1)

    controls = QWidget(); left = QVBoxLayout(controls); left.setContentsMargins(0, 0, 8, 0)
    splitter.addWidget(controls)

    log_edit = QTextEdit(); log_edit.setReadOnly(True); log_edit.setMinimumWidth(520)
    splitter.addWidget(log_edit)
    try:
        splitter.setSizes([540, 720])
    except Exception:
        pass

    def append(msg: str):
        text = str(msg)
        try:
            print("[Mustatil U-Net Trainer] " + text, flush=True)
        except Exception:
            pass
        try:
            ws.log(text)
        except Exception:
            pass
        try:
            QTimer.singleShot(0, lambda t=text: log_edit.append(t))
        except Exception:
            try:
                log_edit.append(text)
            except Exception:
                pass

    def browse_dir(line: QLineEdit):
        base = line.text().strip() or str(project)
        d = QFileDialog.getExistingDirectory(page, "Select folder", base)
        if d:
            line.setText(d)

    def browse_file(line: QLineEdit):
        base = line.text().strip() or str(project)
        fn, _ = QFileDialog.getOpenFileName(page, "Select checkpoint", base, "PyTorch checkpoints (*.pth *.pt);;All files (*)")
        if fn:
            line.setText(fn)

    data_box = QGroupBox("Dataset")
    df = QFormLayout(data_box)
    project_line = QLineEdit(str(project))
    image_line = QLineEdit(str(image_guess))
    label_line = QLineEdit(str(label_guess))
    out_line = QLineEdit(str(out_guess))
    classes_line = QLineEdit(", ".join(list(getattr(getattr(ws, "project_state", None), "classes", []) or ["mustatil", "false_positive"])))

    for title, line in [
        ("Project folder", project_line),
        ("Image folder", image_line),
        ("Label folder", label_line),
        ("Output folder", out_line),
    ]:
        row = QWidget(); hl = QHBoxLayout(row); hl.setContentsMargins(0,0,0,0)
        hl.addWidget(line, 1)
        b = QPushButton("..."); b.setFixedWidth(32); b.clicked.connect(lambda _=False, le=line: browse_dir(le)); hl.addWidget(b)
        df.addRow(title, row)
    df.addRow("Classes", classes_line)
    left.addWidget(data_box)

    model_box = QGroupBox("Training")
    mf = QFormLayout(model_box)
    target_combo = QComboBox(); target_combo.addItems(["Binary foreground/background", "Multiclass semantic mask"])
    mask_combo = QComboBox(); mask_combo.addItems(["box masks from YOLO boxes", "SAM2/JSON polygons if available, else boxes"])
    device_combo = QComboBox(); device_combo.setEditable(True); device_combo.addItems(["auto", "cpu", "cuda"])
    ckpt_line = QLineEdit("")
    ckpt_row = QWidget(); ch = QHBoxLayout(ckpt_row); ch.setContentsMargins(0,0,0,0); ch.addWidget(ckpt_line, 1); cb = QPushButton("..."); cb.setFixedWidth(32); cb.clicked.connect(lambda: browse_file(ckpt_line)); ch.addWidget(cb)

    epochs_spin = QSpinBox(); epochs_spin.setRange(1, 10000); epochs_spin.setValue(30)
    batch_spin = QSpinBox(); batch_spin.setRange(1, 64); batch_spin.setValue(2)
    tile_spin = QSpinBox(); tile_spin.setRange(64, 4096); tile_spin.setSingleStep(64); tile_spin.setValue(512)
    samples_spin = QSpinBox(); samples_spin.setRange(1, 1000); samples_spin.setValue(4)
    base_spin = QSpinBox(); base_spin.setRange(8, 512); base_spin.setSingleStep(8); base_spin.setValue(32)
    lr_spin = QDoubleSpinBox(); lr_spin.setDecimals(6); lr_spin.setRange(0.000001, 1.0); lr_spin.setSingleStep(0.0005); lr_spin.setValue(0.001)
    wd_spin = QDoubleSpinBox(); wd_spin.setDecimals(6); wd_spin.setRange(0.0, 1.0); wd_spin.setSingleStep(0.0001); wd_spin.setValue(0.0001)
    pos_weight_spin = QDoubleSpinBox(); pos_weight_spin.setDecimals(2); pos_weight_spin.setRange(0.1, 1000.0); pos_weight_spin.setSingleStep(1.0); pos_weight_spin.setValue(5.0)
    pos_prob_spin = QDoubleSpinBox(); pos_prob_spin.setDecimals(2); pos_prob_spin.setRange(0.0, 1.0); pos_prob_spin.setSingleStep(0.05); pos_prob_spin.setValue(0.70)
    workers_spin = QSpinBox(); workers_spin.setRange(0, 16); workers_spin.setValue(0)
    maximg_spin = QSpinBox(); maximg_spin.setRange(0, 10000000); maximg_spin.setValue(0); maximg_spin.setToolTip("0 = all images")
    save_every_spin = QSpinBox(); save_every_spin.setRange(1, 1000); save_every_spin.setValue(1)
    include_empty_chk = QCheckBox("Include empty/negative images"); include_empty_chk.setChecked(True)

    mf.addRow("Target mode", target_combo)
    mf.addRow("Mask source", mask_combo)
    mf.addRow("Device", device_combo)
    mf.addRow("Resume/custom checkpoint", ckpt_row)
    mf.addRow("Epochs", epochs_spin)
    mf.addRow("Batch", batch_spin)
    mf.addRow("Tile size", tile_spin)
    mf.addRow("Samples per image", samples_spin)
    mf.addRow("Base channels", base_spin)
    mf.addRow("Learning rate", lr_spin)
    mf.addRow("Weight decay", wd_spin)
    mf.addRow("Binary positive weight", pos_weight_spin)
    mf.addRow("Positive crop probability", pos_prob_spin)
    mf.addRow("Workers", workers_spin)
    mf.addRow("Max images", maximg_spin)
    mf.addRow("Save every N epochs", save_every_spin)
    mf.addRow("", include_empty_chk)
    left.addWidget(model_box)

    hint = QLabel(
        "Detection tab settings after training: Binary = output channels 1. "
        "Multiclass = output channels = classes + 1. Base channels must match this trainer."
    )
    hint.setWordWrap(True)
    left.addWidget(hint)

    buttons = QHBoxLayout()
    validate_btn = QPushButton("Validate Dataset")
    train_btn = QPushButton("Start Training")
    buttons.addWidget(validate_btn)
    buttons.addWidget(train_btn)
    left.addLayout(buttons)
    left.addStretch(1)

    def cfg() -> Dict[str, Any]:
        target_mode = "multiclass" if "multi" in target_combo.currentText().lower() else "binary"
        return {
            "project_dir": project_line.text().strip(),
            "image_dir": image_line.text().strip(),
            "label_dir": label_line.text().strip(),
            "out_dir": out_line.text().strip(),
            "classes": _parse_classes(classes_line.text()),
            "target_mode": target_mode,
            "mask_source": "sam2_polygons" if "sam2" in mask_combo.currentText().lower() else "boxes",
            "device": device_combo.currentText().strip(),
            "checkpoint_path": ckpt_line.text().strip(),
            "epochs": epochs_spin.value(),
            "batch": batch_spin.value(),
            "tile_size": tile_spin.value(),
            "samples_per_image": samples_spin.value(),
            "base": base_spin.value(),
            "lr": lr_spin.value(),
            "weight_decay": wd_spin.value(),
            "positive_weight": pos_weight_spin.value(),
            "positive_crop_probability": pos_prob_spin.value(),
            "workers": workers_spin.value(),
            "max_images": maximg_spin.value(),
            "include_empty": include_empty_chk.isChecked(),
            "save_every": save_every_spin.value(),
        }

    def validate_clicked():
        append("Validating U-Net dataset...")
        _validate_dataset(cfg(), logger=append)

    def train_clicked():
        import threading
        if getattr(page, "_mustatil_unet_training_running", False):
            append("Training is already running; duplicate click ignored.")
            return
        c = cfg()
        append("Start button clicked. Launching U-Net training worker...")
        append("Config: " + json.dumps({k: v for k, v in c.items() if k not in {"classes"}}, ensure_ascii=False))
        page._mustatil_unet_training_running = True
        try:
            train_btn.setEnabled(False)
            train_btn.setText("Training running...")
        except Exception:
            pass

        def job():
            try:
                result = _train_unet_worker(c, logger=append)
                append("Training finished successfully: " + str(result))
            except Exception as exc:
                append("Training failed: " + str(exc))
                try:
                    append(traceback.format_exc())
                except Exception:
                    pass
                raise
            finally:
                page._mustatil_unet_training_running = False
                try:
                    QTimer.singleShot(0, lambda: (train_btn.setEnabled(True), train_btn.setText("Start Training")))
                except Exception:
                    pass

        try:
            ws.run_task("U-Net Training " + _now_stamp(), job)
        except Exception as exc:
            append("Mustatil run_task unavailable, starting direct Python thread: " + str(exc))
            threading.Thread(target=job, daemon=True).start()

    validate_btn.clicked.connect(validate_clicked)
    train_btn.clicked.connect(train_clicked)
    return page


def _has_tab(tab_widget: Any, label: str) -> bool:
    try:
        want = str(label).strip().lower()
        for i in range(tab_widget.count()):
            if str(tab_widget.tabText(i)).strip().lower() == want:
                return True
    except Exception:
        pass
    return False


def _tab_labels(tab_widget: Any) -> List[str]:
    try:
        return [str(tab_widget.tabText(i)).strip() for i in range(tab_widget.count())]
    except Exception:
        return []


def _find_training_anchor_index(tabs: Any) -> Optional[int]:
    """Return the tab index after which the U-Net trainer should be inserted.

    Priority:
      1. R-CNN Trainer
      2. LAE-DINO Trainer / LAE DINO Trainer
      3. YOLO Trainer
      4. any trainer tab
    """
    labels = _tab_labels(tabs)
    lows = [x.lower().replace("_", "-") for x in labels]

    def is_lae(text: str) -> bool:
        compact = text.replace(" ", "").replace("_", "-")
        return "lae-dino" in compact or "laedino" in compact

    for i, low in enumerate(lows):
        if "r-cnn" in low and "trainer" in low:
            return i
    for i, low in enumerate(lows):
        if is_lae(low) and "trainer" in low:
            return i
    for i, low in enumerate(lows):
        if is_lae(low) and ("train" in low or "trainer" in low):
            return i
    for i, low in enumerate(lows):
        if "yolo" in low and "trainer" in low:
            return i
    for i, low in enumerate(lows):
        if "trainer" in low or "training" in low:
            return i
    return None


def _move_tab_after_anchor_if_needed(ws: Any) -> None:
    try:
        tabs = getattr(ws, "tabs", None)
        if tabs is None:
            return
        current = None
        for i in range(tabs.count()):
            if str(tabs.tabText(i)).strip().lower() == "u-net trainer":
                current = i
                break
        if current is None:
            return
        anchor = _find_training_anchor_index(tabs)
        if anchor is None:
            return
        desired = anchor + 1
        if current == desired:
            return
        page = tabs.widget(current)
        label = tabs.tabText(current)
        icon = tabs.tabIcon(current)
        tooltip = tabs.tabToolTip(current)
        tabs.removeTab(current)
        if current < desired:
            desired -= 1
        tabs.insertTab(desired, page, icon, label)
        try:
            tabs.setTabToolTip(desired, tooltip)
        except Exception:
            pass
    except Exception as exc:
        _log("tab reposition warning: " + str(exc))


def _install_trainer_tab(ws: Any) -> bool:
    global _INSTALLING_TAB
    if ws is None:
        return False
    try:
        tabs = getattr(ws, "tabs", None)
        if tabs is None:
            return False
        labels = [x.lower() for x in _tab_labels(tabs)]
        if not any(("trainer" in x or "training" in x) for x in labels):
            return False
        if _has_tab(tabs, "U-Net Trainer"):
            _INSTALLED_WS.add(id(ws))
            _move_tab_after_anchor_if_needed(ws)
            return False
        page = _build_training_tab(ws)
        _INSTALLING_TAB = True
        try:
            anchor = _find_training_anchor_index(tabs)
            insert_at = tabs.count() if anchor is None else anchor + 1
            tabs.insertTab(insert_at, page, "U-Net Trainer")
        finally:
            _INSTALLING_TAB = False
        _INSTALLED_WS.add(id(ws))
        _move_tab_after_anchor_if_needed(ws)
        try:
            ws.log("U-Net Trainer tab added next to the available training tabs.")
        except Exception:
            pass
        return True
    except Exception as exc:
        _log("install trainer tab failed: " + str(exc))
        try:
            traceback.print_exc()
        except Exception:
            pass
        _INSTALLING_TAB = False
        return False


def _scan_for_workspaces(root=None):
    try:
        from PySide6.QtWidgets import QApplication, QTabWidget
        app = QApplication.instance()
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
            ws = _workspace_from_widget(tw)
            if ws is not None:
                _install_trainer_tab(ws)
    except Exception as exc:
        _log("scan failed: " + str(exc))


def _install_hook():
    global _PATCHED_QTAB, _ORIG_ADD_TAB, _ORIG_INSERT_TAB, _SCAN_TIMER
    if _PATCHED_QTAB:
        return
    try:
        from PySide6.QtWidgets import QTabWidget
        from PySide6.QtCore import QTimer
    except Exception as exc:
        _log("Qt unavailable: " + str(exc))
        return

    _ORIG_ADD_TAB = QTabWidget.addTab
    _ORIG_INSERT_TAB = QTabWidget.insertTab

    def addTab_patched(self, page, *args, **kwargs):
        res = _ORIG_ADD_TAB(self, page, *args, **kwargs)
        if not _INSTALLING_TAB:
            try:
                QTimer.singleShot(250, lambda root=self: _scan_for_workspaces(root))
                QTimer.singleShot(1200, lambda root=self: _scan_for_workspaces(root))
            except Exception:
                pass
        return res

    def insertTab_patched(self, index, page, *args, **kwargs):
        res = _ORIG_INSERT_TAB(self, index, page, *args, **kwargs)
        if not _INSTALLING_TAB:
            try:
                QTimer.singleShot(250, lambda root=self: _scan_for_workspaces(root))
                QTimer.singleShot(1200, lambda root=self: _scan_for_workspaces(root))
            except Exception:
                pass
        return res

    QTabWidget.addTab = addTab_patched
    QTabWidget.insertTab = insertTab_patched
    _PATCHED_QTAB = True

    try:
        _SCAN_TIMER = QTimer()
        _SCAN_TIMER.setInterval(1500)
        _SCAN_TIMER.timeout.connect(lambda: _scan_for_workspaces())
        _SCAN_TIMER.start()
        QTimer.singleShot(300, lambda: _scan_for_workspaces())
        QTimer.singleShot(2500, lambda: _scan_for_workspaces())
    except Exception as exc:
        _log("timer setup warning: " + str(exc))
    _log("QTabWidget hook installed.")


def mustatil_plugin_init():
    _install_hook()


try:
    _install_hook()
except Exception:
    pass
