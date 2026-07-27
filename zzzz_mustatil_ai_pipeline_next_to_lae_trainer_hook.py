#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mustatil plugin: Faster R-CNN / Mask R-CNN Trainer

Drop this file into Mustatil's mustatil_plugins folder and restart Mustatil.
It adds a new top-level tab named "R-CNN Trainer" and places it next to the
LAE-DINO Trainer tab when present, with YOLO Trainer as fallback. It does not
modify Detection preview, Satellite preview, or project files.

Training input:
  - Images: jpg/png/tif/tiff/webp/bmp, recursively scanned
  - Labels: YOLO txt labels with normalized x_center y_center width height
  - Mask R-CNN masks: either rectangular masks from boxes, or optional SAM2/JSON
    polygon sidecars when available, with rectangle fallback.

Outputs:
  - .pth checkpoints usable by the companion Faster R-CNN / Mask R-CNN detection plugin
  - classes.json and training_metadata.json next to checkpoints

Notes:
  - Faster R-CNN and Mask R-CNN use class index 0 as background internally.
    YOLO class 0 becomes training label 1, YOLO class 1 becomes label 2, etc.
  - For real archaeological detection, train with your own Mustatil/Cairn/etc.
    dataset; COCO pretrained weights are only a starting backbone.
"""
from __future__ import annotations

import json
import math
import os
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
        print("[Mustatil R-CNN Trainer] " + str(msg))
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
        if p.exists() and any(p.rglob("*" + ext) for ext in IMG_EXTS):
            return p
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
        if p.exists() and any(p.rglob("*.txt")):
            return p
    return project / "labels"


def _find_label_for_image(img: Path, image_dir: Path, label_dir: Path) -> Path:
    # Preserve subfolder structure where possible: images/a/b.png -> labels/a/b.txt
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
    # Last fallback: recursive same stem. Use shortest path first.
    try:
        matches = sorted(label_dir.rglob(img.stem + ".txt"), key=lambda p: len(str(p)))
        if matches:
            return matches[0]
    except Exception:
        pass
    return cand


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
        # YOLO normalized coordinates -> absolute xyxy
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
        # torchvision detection reserves 0 for background
        labels.append(int(cls) + 1)
    return boxes, labels


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


def _load_optional_polygons(img: Path, label_path: Path, width: int, height: int) -> List[List[Tuple[float, float]]]:
    """Best-effort parser for SAM2/Mustatil sidecar JSON polygon files.

    The function accepts many loose shapes:
      - [{"points": [[x,y], ...]}, ...]
      - [{"polygon": [[x,y], ...]}, ...]
      - {"polygons": [...]} or {"masks": [...]}
    If coordinates are normalized 0..1, they are scaled to pixels.
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

    items = unwrap(data)
    polys: List[List[Tuple[float, float]]] = []

    def get_points(item):
        if isinstance(item, dict):
            for key in ("points", "polygon", "poly", "coordinates", "mask_polygon"):
                if key in item:
                    return item.get(key)
            # GeoJSON-ish feature
            geom = item.get("geometry")
            if isinstance(geom, dict):
                coords = geom.get("coordinates")
                if geom.get("type", "").lower() == "polygon" and coords:
                    return coords[0]
                if geom.get("type", "").lower() == "multipolygon" and coords:
                    return coords[0][0]
        return item

    for it in items:
        pts_raw = get_points(it)
        if not isinstance(pts_raw, (list, tuple)):
            continue
        # Sometimes nested as [[[x,y], ...]]
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


def _polygon_to_mask(width: int, height: int, points: Sequence[Tuple[float, float]]):
    from PIL import Image, ImageDraw
    im = Image.new("L", (int(width), int(height)), 0)
    draw = ImageDraw.Draw(im)
    draw.polygon([(float(x), float(y)) for x, y in points], outline=1, fill=1)
    import numpy as np
    return np.array(im, dtype="uint8")


def _box_to_mask(width: int, height: int, box: Sequence[float]):
    import numpy as np
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(width, x1)); x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1)); y2 = max(0, min(height, y2))
    m = np.zeros((int(height), int(width)), dtype="uint8")
    if x2 > x1 and y2 > y1:
        m[y1:y2, x1:x2] = 1
    return m


class MustatilYoloDetectionDataset:
    def __init__(self, image_dir: str, label_dir: str, classes: Sequence[str], mask_mode: str = "boxes", max_images: int = 0, include_empty: bool = True):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.classes = list(classes)
        self.num_classes = len(self.classes)
        self.mask_mode = str(mask_mode or "boxes").lower()
        self.include_empty = bool(include_empty)
        self.images = _collect_images(self.image_dir, int(max_images or 0))
        if not self.include_empty:
            kept = []
            for img in self.images:
                lab = _find_label_for_image(img, self.image_dir, self.label_dir)
                if lab.exists() and lab.read_text(encoding="utf-8", errors="ignore").strip():
                    kept.append(img)
            self.images = kept

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        import torch
        import numpy as np
        from PIL import Image
        img_path = self.images[int(idx)]
        pil = Image.open(img_path).convert("RGB")
        width, height = pil.size
        label_path = _find_label_for_image(img_path, self.image_dir, self.label_dir)
        boxes, labels = _read_yolo_txt(label_path, width, height, self.num_classes)

        # Image -> torch float tensor C,H,W in 0..1. Avoid torchvision.transforms dependency variants.
        arr = np.array(pil, dtype="uint8")
        image_tensor = torch.from_numpy(arr).permute(2, 0, 1).float().div(255.0)

        if boxes:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes_t[:, 2] - boxes_t[:, 0]).clamp(min=0) * (boxes_t[:, 3] - boxes_t[:, 1]).clamp(min=0)
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target: Dict[str, Any] = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([int(idx)], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
        }

        # Mask R-CNN requires masks. Faster R-CNN ignores them.
        masks_np = []
        if boxes:
            polys: List[List[Tuple[float, float]]] = []
            if "sam" in self.mask_mode or "polygon" in self.mask_mode:
                polys = _load_optional_polygons(img_path, label_path, width, height)
            for i, box in enumerate(boxes):
                m = None
                if polys and i < len(polys):
                    try:
                        m = _polygon_to_mask(width, height, polys[i])
                    except Exception:
                        m = None
                if m is None:
                    m = _box_to_mask(width, height, box)
                masks_np.append(m)
        if masks_np:
            target["masks"] = torch.as_tensor(np.stack(masks_np, axis=0), dtype=torch.uint8)
        else:
            target["masks"] = torch.zeros((0, int(height), int(width)), dtype=torch.uint8)

        target["image_path"] = str(img_path)
        target["label_path"] = str(label_path)
        return image_tensor, target


def _collate_fn(batch):
    return tuple(zip(*batch))


def _build_detection_model(model_kind: str, num_user_classes: int, pretrained: bool = True):
    import torch
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    # +1 for background
    num_classes = int(num_user_classes) + 1
    kind = str(model_kind).lower()

    if "mask" in kind:
        # torchvision changed from pretrained=True to weights=...; support both.
        try:
            weights = "DEFAULT" if pretrained else None
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
        except TypeError:
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=bool(pretrained))
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
        return model

    try:
        weights = "DEFAULT" if pretrained else None
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    except TypeError:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=bool(pretrained))
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


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

    try:
        model.load_state_dict(state, strict=False)
        if logger:
            logger("Custom checkpoint loaded with strict=False.")
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


def _freeze_backbone_if_requested(model, freeze: bool, logger=None) -> None:
    if not freeze:
        return
    frozen = 0
    try:
        for name, p in model.backbone.named_parameters():
            p.requires_grad = False
            frozen += 1
    except Exception:
        pass
    if logger:
        logger(f"Backbone frozen: {frozen} parameter tensors.")


def _resolve_device(text: str):
    import torch
    raw = str(text or "auto").lower().strip()
    if raw == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if raw == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _train_rcnn_worker(cfg: Dict[str, Any], logger=None) -> str:
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
    model_kind = str(cfg.get("model_kind", "fasterrcnn"))
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

    run_name = _safe_name(f"{model_kind}_{_now_stamp()}")
    run_dir = out_dir / run_name
    _ensure_dir(run_dir)

    log(f"Training run folder: {run_dir}")
    log(f"Images: {image_dir}")
    log(f"Labels: {label_dir}")
    log(f"Classes: {classes}")
    log(f"Model: {model_kind}")

    dataset = MustatilYoloDetectionDataset(
        str(image_dir), str(label_dir), classes,
        mask_mode=mask_mode, max_images=max_images, include_empty=include_empty
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
        collate_fn=_collate_fn,
        pin_memory=False,
    )

    device = _resolve_device(cfg.get("device", "auto"))
    log(f"Device: {device}")
    model = _build_detection_model(model_kind, len(classes), pretrained=pretrained)
    _load_checkpoint_flexible(model, checkpoint_path, logger=log)
    _freeze_backbone_if_requested(model, freeze_backbone, logger=log)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)

    metadata = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_kind": model_kind,
        "classes": classes,
        "num_classes_including_background": len(classes) + 1,
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "mask_mode": mask_mode,
        "pretrained": pretrained,
        "checkpoint_path": checkpoint_path,
        "freeze_backbone": freeze_backbone,
        "epochs": epochs,
        "batch": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "device": str(device),
        "internal_label_note": "YOLO class 0 -> torchvision label 1; label 0 is background.",
    }
    (run_dir / "classes.json").write_text(json.dumps(classes, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    model.train()
    last_ckpt = ""
    for epoch in range(1, epochs + 1):
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
                log(f"Epoch {epoch} step {step}: non-finite loss skipped: {float(losses.detach().cpu())}")
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            losses.backward()
            optimizer.step()

            val = float(losses.detach().cpu())
            epoch_loss += val
            epoch_batches += 1
            if step == 1 or step % 10 == 0:
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
        "metadata": metadata,
    }, final_path)
    last_ckpt = str(final_path)
    readme = (
        "Mustatil R-CNN training result\n"
        f"Model: {model_kind}\n"
        f"Classes: {', '.join(classes)}\n"
        f"Final checkpoint: {final_path.name}\n"
        "Use this .pth in the Faster R-CNN / Mask R-CNN detection plugin as custom checkpoint.\n"
    )
    (run_dir / "README_training_result.txt").write_text(readme, encoding="utf-8")
    log(f"Training complete. Final checkpoint: {final_path}")
    return last_ckpt


def _validate_dataset(cfg: Dict[str, Any], logger=None) -> None:
    def log(msg):
        if logger:
            logger(str(msg))
        else:
            _log(str(msg))
    image_dir = Path(cfg["image_dir"])
    label_dir = Path(cfg["label_dir"])
    classes = list(cfg["classes"])
    max_images = int(cfg.get("max_images", 0) or 0)
    images = _collect_images(image_dir, max_images=max_images)
    log(f"Image folder: {image_dir}")
    log(f"Label folder: {label_dir}")
    log(f"Images found: {len(images)}")
    log(f"Classes: {classes}")
    if not images:
        log("No images found.")
        return
    from PIL import Image
    total_boxes = 0
    nonempty = 0
    missing_labels = 0
    bad_images = 0
    for img in images[: min(len(images), 500)]:
        lab = _find_label_for_image(img, image_dir, label_dir)
        if not lab.exists():
            missing_labels += 1
        try:
            with Image.open(img) as im:
                w, h = im.size
            boxes, _labels = _read_yolo_txt(lab, w, h, len(classes))
            total_boxes += len(boxes)
            if boxes:
                nonempty += 1
        except Exception:
            bad_images += 1
    checked = min(len(images), 500)
    log(f"Checked images: {checked}")
    log(f"Images with objects: {nonempty}")
    log(f"Boxes: {total_boxes}")
    log(f"Missing label txt: {missing_labels}")
    log(f"Unreadable images: {bad_images}")
    if missing_labels:
        log("Missing labels are allowed if 'Include empty/negative images' is enabled.")


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
    out_guess = project / "runs" / "rcnn_training"

    page = QWidget()
    root = QVBoxLayout(page)
    root.setContentsMargins(8, 8, 8, 8)

    header = QLabel(
        "Faster R-CNN / Mask R-CNN Trainer for Mustatil/YOLO labels. "
        "The existing YOLO Trainer stays unchanged. Mask R-CNN can use rectangular masks from YOLO boxes "
        "or existing SAM2/JSON polygons when available."
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
        splitter.setSizes([520, 720])
    except Exception:
        pass

    def append(msg: str):
        text = str(msg)
        try:
            print("[Mustatil R-CNN Trainer] " + text, flush=True)
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

    for title, line, kind in [
        ("Project folder", project_line, "dir"),
        ("Image folder", image_line, "dir"),
        ("Label folder", label_line, "dir"),
        ("Output folder", out_line, "dir"),
    ]:
        row = QWidget(); hl = QHBoxLayout(row); hl.setContentsMargins(0,0,0,0)
        hl.addWidget(line, 1)
        b = QPushButton("..."); b.setFixedWidth(32); b.clicked.connect(lambda _=False, le=line: browse_dir(le)); hl.addWidget(b)
        df.addRow(title, row)
    df.addRow("Classes", classes_line)
    left.addWidget(data_box)

    model_box = QGroupBox("Training")
    mf = QFormLayout(model_box)
    model_combo = QComboBox(); model_combo.addItems(["Faster R-CNN ResNet50-FPN", "Mask R-CNN ResNet50-FPN"])
    device_combo = QComboBox(); device_combo.setEditable(True); device_combo.addItems(["auto", "cpu", "cuda"])
    weights_combo = QComboBox(); weights_combo.addItems(["COCO/default pretrained", "random init", "custom checkpoint"])
    ckpt_line = QLineEdit("")
    ckpt_row = QWidget(); ch = QHBoxLayout(ckpt_row); ch.setContentsMargins(0,0,0,0); ch.addWidget(ckpt_line, 1); cb = QPushButton("..."); cb.setFixedWidth(32); cb.clicked.connect(lambda: browse_file(ckpt_line)); ch.addWidget(cb)
    mask_combo = QComboBox(); mask_combo.addItems(["box masks from YOLO boxes", "SAM2/JSON polygons if available, else boxes"])

    epochs_spin = QSpinBox(); epochs_spin.setRange(1, 10000); epochs_spin.setValue(30)
    batch_spin = QSpinBox(); batch_spin.setRange(1, 32); batch_spin.setValue(1)
    lr_spin = QDoubleSpinBox(); lr_spin.setDecimals(6); lr_spin.setRange(0.000001, 1.0); lr_spin.setSingleStep(0.0005); lr_spin.setValue(0.005)
    wd_spin = QDoubleSpinBox(); wd_spin.setDecimals(6); wd_spin.setRange(0.0, 1.0); wd_spin.setSingleStep(0.0001); wd_spin.setValue(0.0005)
    workers_spin = QSpinBox(); workers_spin.setRange(0, 16); workers_spin.setValue(0)
    maximg_spin = QSpinBox(); maximg_spin.setRange(0, 10000000); maximg_spin.setValue(0); maximg_spin.setToolTip("0 = alle Bilder")
    save_every_spin = QSpinBox(); save_every_spin.setRange(1, 1000); save_every_spin.setValue(1)
    include_empty_chk = QCheckBox("Include empty/negative images"); include_empty_chk.setChecked(True)
    freeze_chk = QCheckBox("Freeze backbone first run / low VRAM"); freeze_chk.setChecked(False)

    mf.addRow("Model", model_combo)
    mf.addRow("Device", device_combo)
    mf.addRow("Weights", weights_combo)
    mf.addRow("Custom checkpoint", ckpt_row)
    mf.addRow("Mask source", mask_combo)
    mf.addRow("Epochs", epochs_spin)
    mf.addRow("Batch", batch_spin)
    mf.addRow("Learning rate", lr_spin)
    mf.addRow("Weight decay", wd_spin)
    mf.addRow("Workers", workers_spin)
    mf.addRow("Max images", maximg_spin)
    mf.addRow("Save every N epochs", save_every_spin)
    mf.addRow("", include_empty_chk)
    mf.addRow("", freeze_chk)
    left.addWidget(model_box)

    buttons = QHBoxLayout()
    validate_btn = QPushButton("Validate Dataset")
    train_btn = QPushButton("Start Training")
    buttons.addWidget(validate_btn)
    buttons.addWidget(train_btn)
    left.addLayout(buttons)
    left.addStretch(1)

    def cfg() -> Dict[str, Any]:
        model_kind = "maskrcnn_resnet50_fpn" if "mask" in model_combo.currentText().lower() else "fasterrcnn_resnet50_fpn"
        wtxt = weights_combo.currentText().lower()
        return {
            "project_dir": project_line.text().strip(),
            "image_dir": image_line.text().strip(),
            "label_dir": label_line.text().strip(),
            "out_dir": out_line.text().strip(),
            "classes": _parse_classes(classes_line.text()),
            "model_kind": model_kind,
            "device": device_combo.currentText().strip(),
            "pretrained": "random" not in wtxt,
            "checkpoint_path": ckpt_line.text().strip() if "custom" in wtxt else "",
            "freeze_backbone": freeze_chk.isChecked(),
            "mask_mode": "sam2_polygons" if "sam2" in mask_combo.currentText().lower() else "boxes",
            "epochs": epochs_spin.value(),
            "batch": batch_spin.value(),
            "lr": lr_spin.value(),
            "weight_decay": wd_spin.value(),
            "workers": workers_spin.value(),
            "max_images": maximg_spin.value(),
            "include_empty": include_empty_chk.isChecked(),
            "save_every": save_every_spin.value(),
        }

    def validate_clicked():
        append("Validating dataset...")
        _validate_dataset(cfg(), logger=append)

    def train_clicked():
        import threading
        if getattr(page, "_mustatil_rcnn_training_running", False):
            append("Training is already running; duplicate click ignored.")
            return
        c = cfg()
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
                result = _train_rcnn_worker(c, logger=append)
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
                try:
                    QTimer.singleShot(0, lambda: (train_btn.setEnabled(True), train_btn.setText("Start Training")))
                except Exception:
                    pass

        # Prefer Mustatil's task runner for status integration. Use a unique task name
        # so a stale duplicate name cannot prevent the start. Fall back to a direct thread.
        try:
            ws.run_task("R-CNN Training " + _now_stamp(), job)
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
    """Return the tab index after which the R-CNN trainer should be inserted.

    Priority:
      1. LAE-DINO Trainer / LAE DINO Trainer
      2. any LAE-DINO training-related tab
      3. YOLO Trainer
      4. any trainer tab
    """
    labels = _tab_labels(tabs)
    lows = [x.lower().replace("_", "-") for x in labels]

    def is_lae(text: str) -> bool:
        compact = text.replace(" ", "").replace("_", "-")
        return "lae-dino" in compact or "laedino" in compact

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
            if str(tabs.tabText(i)).strip().lower() == "r-cnn trainer":
                current = i
                break
        if current is None:
            return
        anchor = _find_training_anchor_index(tabs)
        if anchor is None:
            return
        # If R-CNN itself is before the anchor, moving it removes one index before reinsert.
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
        if _has_tab(tabs, "R-CNN Trainer"):
            _INSTALLED_WS.add(id(ws))
            _move_tab_after_anchor_if_needed(ws)
            return False
        page = _build_training_tab(ws)
        _INSTALLING_TAB = True
        try:
            anchor = _find_training_anchor_index(tabs)
            insert_at = tabs.count() if anchor is None else anchor + 1
            tabs.insertTab(insert_at, page, "R-CNN Trainer")
        finally:
            _INSTALLING_TAB = False
        _INSTALLED_WS.add(id(ws))
        _move_tab_after_anchor_if_needed(ws)
        try:
            ws.log("R-CNN Trainer tab added next to the available training tabs.")
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


# Also install on import, because Mustatil plugin loader imports modules before
# the workspace object is created.
try:
    _install_hook()
except Exception:
    pass
