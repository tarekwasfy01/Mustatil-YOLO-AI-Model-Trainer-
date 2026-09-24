#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PLUGIN_NAME = "Mustatil LAE-DINO Project Trainer V18 Anti-Freeze Insert Dataset Button"
PLUGIN_VERSION = "2026-06-16-v18-antifreeze-insert-dataset-button"


# Mustatil LAE-DINO Project Trainer V18 Anti-Freeze Insert Dataset + Config + Train
#
# Fix versus V5:
# - V5 could miss the tab because it relied only on delayed scans.
# - V6 installs a very light QTabWidget addTab/insertTab hook.
# - The hook only watches tab labels; it does NOT scan files, weights, configs, or runtime.
# - Heavy project import/config generation still runs only after button click in a background thread.

_PATCHED_QTAB = False
_ORIG_ADD = None
_ORIG_INSERT = None
_INSERTED_WIDGETS = set()

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
DEFAULT_CLASS_NAMES = [
    "mustatil",
    "burial mound",
    "tumulus",
    "stone enclosure",
    "rectangular structure",
    "false_positive",
]


def _log(x: Any):
    try:
        print("[Mustatil LAE-DINO Project Trainer V17] " + str(x))
    except Exception:
        pass


def _pdir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except Exception:
        return Path.cwd()


def _runtime_root() -> Path:
    return _pdir() / "mustatil_model_runtimes" / "LAE-DINO-PATCH"


def _repo_root() -> Path:
    return _runtime_root() / "repo"


def _py() -> Path:
    return _runtime_root() / "python310" / ("python.exe" if os.name == "nt" else "python")


def _mmd(repo: Optional[Path] = None) -> Path:
    repo = Path(repo or _repo_root())
    return repo / "mmdetection_lae" if (repo / "mmdetection_lae").exists() else repo


def _workspace_from_widget(widget):
    cur = widget
    for _ in range(180):
        if cur is None:
            break
        try:
            if hasattr(cur, "tabs") and (hasattr(cur, "project") or hasattr(cur, "image") or hasattr(cur, "dets")):
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


def _project_from_workspace(ws) -> Path:
    candidates: List[str] = []
    try:
        if ws is not None and hasattr(ws, "project") and hasattr(ws.project, "get"):
            v = ws.project.get()
            if v:
                candidates.append(str(v))
    except Exception:
        pass
    for attr in (
        "project_dir", "project_path", "current_project", "project_folder",
        "workspace_dir", "mustatil_project_dir", "mustatil_current_project",
        "mustatil_lae_trainer_project_folder",
    ):
        try:
            v = getattr(ws, attr, None) if ws is not None else None
            if hasattr(v, "get"):
                v = v.get()
            if v:
                candidates.append(str(v))
        except Exception:
            pass
    for raw in candidates:
        try:
            raw = str(raw or "").strip().strip('"')
            if not raw:
                continue
            p = Path(raw).expanduser()
            if p.suffix and not p.is_dir():
                p = p.parent
            if p.exists():
                return p
            return p
        except Exception:
            pass
    return Path.home()


def _emit(bridge, sig: str, *args):
    try:
        getattr(bridge, sig).emit(*[str(a) for a in args])
    except Exception:
        if args:
            _log(args[-1])


def _env(repo: Path, device: str = "auto") -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    mmd = _mmd(repo)
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([str(mmd), str(mmd.parent)] + ([old] if old else []))

    dev = str(device or "auto").strip().lower()
    if dev == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    elif dev in {"cuda", "cuda:0", "gpu", "auto"}:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    elif dev.startswith("cuda:"):
        try:
            env["CUDA_VISIBLE_DEVICES"] = dev.split(":", 1)[1]
        except Exception:
            env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def _run(cmd: List[Any], bridge, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None, title: str = "command") -> str:
    cmd = [str(x) for x in cmd]
    _emit(bridge, "log", "RUN: " + " ".join(('"' + c + '"') if " " in c else c for c in cmd))
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    lines: List[str] = []
    last = time.time()
    assert p.stdout is not None
    for line in p.stdout:
        line = line.rstrip("\r\n")
        lines.append(line)
        _emit(bridge, "log", line)
        if time.time() - last > 4:
            _emit(bridge, "status", title + " läuft... " + line[:160])
            last = time.time()
    rc = p.wait()
    if rc:
        raise RuntimeError(title + " failed with exit code " + str(rc) + "\n" + "\n".join(lines[-160:]))
    return "\n".join(lines)


def _bg(project_text: str, bridge, button=None):
    project_text = str(project_text or "").strip().strip('"')
    if button is not None:
        try:
            button.setEnabled(False)
        except Exception:
            pass

    def runner():
        try:
            _emit(bridge, "clear")
            _emit(bridge, "status", "Erzeuge LAE-DINO Training-Config aus Projektordner...")
            _emit(bridge, "log", "Project folder: " + project_text)
            project = Path(project_text)
            if not project.exists():
                raise RuntimeError("Projektordner nicht gefunden: " + str(project))
            result = _create_everything_from_project(project, bridge)
            _emit(bridge, "status", "Fertig: " + str(result.get("config")))
            _emit(bridge, "log", "")
            _emit(bridge, "log", "=== DONE ===")
            for k, v in result.items():
                _emit(bridge, "log", f"{k}: {v}")
        except Exception as exc:
            _emit(bridge, "status", "Fehler: " + str(exc))
            _emit(bridge, "log", "ERROR: " + str(exc))
            _emit(bridge, "log", traceback.format_exc())
        finally:
            if button is not None:
                _emit(bridge, "enable_button", "1")

    threading.Thread(target=runner, daemon=True, name="LAE_DINO_Project_Config").start()


def _bg_dataset(project_text: str, bridge, button=None):
    """Create only the project COCO dataset from project/images + project/labels.

    This is the explicit button the user asked for. It does not start training,
    but it writes/refreshes project/lae_dino_dataset/images, annotations and meta.
    """
    project_text = str(project_text or "").strip().strip('"')
    if button is not None:
        try:
            button.setEnabled(False)
        except Exception:
            pass

    def runner():
        try:
            _emit(bridge, "clear")
            _emit(bridge, "status", "Erzeuge Dataset aus Projektbildern...")
            project = Path(project_text).expanduser()
            _emit(bridge, "log", "")
            _emit(bridge, "log", "=== CREATE DATASET FROM PROJECT ===")
            _emit(bridge, "log", "Project folder: " + str(project))
            if not project.exists():
                raise RuntimeError("Projektordner nicht gefunden: " + str(project))

            pairs, label_files = _discover_pairs(project, bridge)
            classes = _load_classes(project, label_files, bridge)
            _emit(bridge, "log", "Classes: " + ", ".join(classes))
            meta = _write_coco_dataset(project, pairs, classes, bridge)

            _emit(bridge, "status", "Dataset fertig: " + str(meta.get("dataset_root")))
            _emit(bridge, "log", "")
            _emit(bridge, "log", "=== DATASET DONE ===")
            for k, v in meta.items():
                _emit(bridge, "log", f"{k}: {v}")
        except Exception as exc:
            _emit(bridge, "status", "Dataset Fehler: " + str(exc))
            _emit(bridge, "log", "ERROR: " + str(exc))
            _emit(bridge, "log", traceback.format_exc())
        finally:
            if button is not None:
                _emit(bridge, "enable_dataset_button", "1")

    threading.Thread(target=runner, daemon=True, name="LAE_DINO_Create_Dataset_From_Project_V17").start()


# ------------------------------
# Project discovery
# ------------------------------

def _find_images(root: Path) -> List[Path]:
    out: List[Path] = []
    for ext in sorted(IMG_EXTS):
        try:
            out += list(root.rglob("*" + ext))
            out += list(root.rglob("*" + ext.upper()))
        except Exception:
            pass
    filtered = []
    for p in out:
        s = str(p).replace("\\", "/").lower()
        if "/lae_dino_dataset/" in s or "/yolo_dataset_lae/" in s or "/work_dirs/" in s:
            continue
        filtered.append(p)
    filtered = [p for p in dict.fromkeys(filtered) if p.is_file()]
    filtered.sort(key=lambda p: str(p).lower())
    return filtered


def _find_label_for_image(img: Path, label_roots: List[Path]) -> Optional[Path]:
    for root in label_roots:
        p = root / (img.stem + ".txt")
        if p.exists():
            return p
    for root in label_roots:
        try:
            hits = list(root.rglob(img.stem + ".txt"))
            if hits:
                return hits[0]
        except Exception:
            pass
    return None


def _candidate_label_roots(project: Path) -> List[Path]:
    roots = []
    for name in ["labels", "Labels", "yolo_labels", "annotations", "Annotations"]:
        p = project / name
        if p.exists():
            roots.append(p)
    roots.append(project)
    return list(dict.fromkeys(roots))


def _candidate_image_roots(project: Path) -> List[Path]:
    roots = []
    for name in ["images", "Images", "imgs", "train_images"]:
        p = project / name
        if p.exists():
            roots.append(p)
    roots.append(project)
    return list(dict.fromkeys(roots))


def _read_yaml_simple(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
        return yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore")) or {}
    except Exception:
        pass
    data: Dict[str, Any] = {}
    key = None
    if not path.exists():
        return data
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line and not line.startswith("-"):
            k, v = line.split(":", 1)
            key = k.strip()
            v = v.strip().strip('"').strip("'")
            if v.startswith("[") and v.endswith("]"):
                data[key] = [x.strip().strip('"').strip("'") for x in v[1:-1].split(",") if x.strip()]
            elif v:
                data[key] = v
            else:
                data[key] = []
        elif key and line.startswith("-"):
            data.setdefault(key, []).append(line[1:].strip().strip('"').strip("'"))
    return data


def _load_classes(project: Path, labels: List[Path], bridge) -> List[str]:
    for rel in ["data.yaml", "dataset.yaml", "yolo.yaml"]:
        p = project / rel
        if p.exists():
            data = _read_yaml_simple(p)
            names = data.get("names")
            if isinstance(names, list) and names:
                _emit(bridge, "log", "Classes from " + str(p))
                return [str(x) for x in names]
            if isinstance(names, dict) and names:
                try:
                    return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
                except Exception:
                    return [str(v) for v in names.values()]

    for rel in ["classes.txt", "names.txt", "class_names.txt"]:
        p = project / rel
        if p.exists():
            vals = [x.strip() for x in p.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
            if vals:
                _emit(bridge, "log", "Classes from " + str(p))
                return vals

    for rel in ["classes.json", "class_names.json", "project.json", "metadata.json", "mustatil_project.json"]:
        p = project / rel
        if not p.exists():
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            for key in ["classes", "class_names", "names", "labels"]:
                vals = obj.get(key) if isinstance(obj, dict) else None
                if isinstance(vals, list) and vals:
                    _emit(bridge, "log", "Classes from " + str(p))
                    return [str(x.get("name", x)) if isinstance(x, dict) else str(x) for x in vals]
                if isinstance(vals, dict) and vals:
                    try:
                        return [str(vals[k]) for k in sorted(vals, key=lambda x: int(x))]
                    except Exception:
                        return [str(v) for v in vals.values()]
        except Exception:
            pass

    max_id = -1
    for lab in labels:
        try:
            for raw in lab.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = raw.split()
                if not parts:
                    continue
                try:
                    max_id = max(max_id, int(float(parts[0])))
                except Exception:
                    pass
        except Exception:
            pass
    if max_id >= 0:
        if max_id + 1 <= len(DEFAULT_CLASS_NAMES):
            _emit(bridge, "log", "Classes inferred from label ids; using default archaeology names.")
            return DEFAULT_CLASS_NAMES[:max_id + 1]
        _emit(bridge, "log", "Classes inferred from label ids; using generic names.")
        return [f"class_{i}" for i in range(max_id + 1)]

    _emit(bridge, "log", "No class metadata found; using default class list.")
    return DEFAULT_CLASS_NAMES[:]


def _image_size(path: Path) -> Tuple[int, int]:
    from PIL import Image
    with Image.open(path) as im:
        return int(im.width), int(im.height)


def _copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or src.stat().st_size != dst.stat().st_size or src.stat().st_mtime > dst.stat().st_mtime:
        shutil.copy2(src, dst)


def _discover_pairs(project: Path, bridge) -> Tuple[List[Tuple[Path, Optional[Path]]], List[Path]]:
    image_roots = _candidate_image_roots(project)
    label_roots = _candidate_label_roots(project)

    images: List[Path] = []
    for root in image_roots:
        images.extend(_find_images(root))
    images = list(dict.fromkeys(images))
    images.sort(key=lambda p: str(p).lower())

    pairs: List[Tuple[Path, Optional[Path]]] = []
    labels: List[Path] = []
    for im in images:
        lab = _find_label_for_image(im, label_roots)
        if lab is not None:
            labels.append(lab)
        pairs.append((im, lab))

    if not pairs:
        raise RuntimeError("Keine Trainingsbilder im Projekt gefunden. Erwartet z.B. project\\images und project\\labels.")
    if not labels:
        raise RuntimeError("Keine YOLO-Labels gefunden. Erwartet .txt Dateien passend zu den Bildnamen.")

    _emit(bridge, "log", f"Found images: {len(pairs)}")
    _emit(bridge, "log", f"Found label files: {len(labels)}")
    return pairs, labels


def _split_pairs(pairs: List[Tuple[Path, Optional[Path]]]) -> Tuple[List[Tuple[Path, Optional[Path]]], List[Tuple[Path, Optional[Path]]]]:
    train = []
    val = []
    for im, lab in pairs:
        s = str(im).replace("\\", "/").lower()
        if "/val/" in s or "/valid/" in s or "/validation/" in s:
            val.append((im, lab))
        elif "/train/" in s:
            train.append((im, lab))
    if train and val:
        return train, val

    pairs2 = pairs[:]
    random.seed(42)
    random.shuffle(pairs2)
    if len(pairs2) == 1:
        return pairs2, pairs2
    nval = max(1, int(len(pairs2) * 0.15))
    nval = min(nval, len(pairs2) - 1)
    return pairs2[nval:], pairs2[:nval]


def _write_coco_dataset(project: Path, pairs: List[Tuple[Path, Optional[Path]]], classes: List[str], bridge) -> Dict[str, Any]:
    out = project / "lae_dino_dataset"
    for sub in ["images/train", "images/val", "annotations", "checkpoints", "work_dirs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    train_pairs, val_pairs = _split_pairs(pairs)
    categories = [{"id": i + 1, "name": name, "supercategory": "mustatil"} for i, name in enumerate(classes)]

    def split_to_coco(split: str, items: List[Tuple[Path, Optional[Path]]]) -> Tuple[int, int]:
        coco = {
            "info": {"description": "Mustatil Annotator project converted to LAE-DINO COCO", "version": "1.0"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": categories,
        }
        ann_id = 1
        for img_id, (im, lab) in enumerate(items, 1):
            W, H = _image_size(im)
            dst_name = im.name
            _copy(im, out / "images" / split / dst_name)
            coco["images"].append({"id": img_id, "file_name": dst_name, "width": W, "height": H})
            if lab and lab.exists():
                for raw in lab.read_text(encoding="utf-8", errors="ignore").splitlines():
                    parts = raw.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls_id = int(float(parts[0]))
                        cx, cy, bw, bh = map(float, parts[1:5])
                    except Exception:
                        continue
                    if cls_id < 0 or cls_id >= len(classes):
                        continue
                    x = max(0.0, (cx - bw / 2.0) * W)
                    y = max(0.0, (cy - bh / 2.0) * H)
                    w = min(float(W) - x, bw * W)
                    h = min(float(H) - y, bh * H)
                    if w <= 1 or h <= 1:
                        continue
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cls_id + 1,
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "area": float(w * h),
                        "iscrowd": 0,
                        "segmentation": [],
                    })
                    ann_id += 1
            if img_id % 20 == 0:
                _emit(bridge, "status", f"COCO {split}: {img_id}/{len(items)}")
        (out / "annotations" / f"instances_{split}.json").write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
        return len(coco["images"]), len(coco["annotations"])

    train_i, train_a = split_to_coco("train", train_pairs)
    val_i, val_a = split_to_coco("val", val_pairs)

    if train_a == 0:
        raise RuntimeError("Keine gültigen Trainingsannotation gefunden. Prüfe YOLO .txt Dateien.")

    if val_i == 0 or val_a == 0:
        _emit(bridge, "log", "Val split empty; copying train annotations as validation fallback.")
        shutil.copy2(out / "annotations" / "instances_train.json", out / "annotations" / "instances_val.json")
        for im in list((out / "images" / "train").glob("*"))[:min(32, train_i)]:
            if im.is_file():
                _copy(im, out / "images" / "val" / im.name)
        val_i, val_a = train_i, train_a

    meta = {
        "project_root": str(project),
        "dataset_root": str(out),
        "classes": classes,
        "train_images": train_i,
        "train_annotations": train_a,
        "val_images": val_i,
        "val_annotations": val_a,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "mustatil_lae_dataset_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _emit(bridge, "log", f"COCO dataset: {out}")
    _emit(bridge, "log", f"train images/annotations: {train_i}/{train_a}")
    _emit(bridge, "log", f"val images/annotations: {val_i}/{val_a}")
    return meta


# ------------------------------
# LAE-DINO config/checkpoint
# ------------------------------

def _default_base_config(repo: Path) -> Path:
    mmd = _mmd(repo)
    candidates = [
        mmd / "configs" / "lae_dino" / "lae_dino_swin-t_finetune_DIOR.py",
        mmd / "configs" / "lae_dino" / "lae_dino_swin-t_pretrain_LAE-1M.py",
        mmd / "configs" / "lae_dino" / "lae_dino_swin-t_finetune_DOTA.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    d = mmd / "configs" / "lae_dino"
    hits = list(d.glob("*.py")) if d.exists() else []
    if hits:
        return hits[0]
    raise RuntimeError("Keine LAE-DINO Base-Config gefunden in: " + str(d))


def _default_checkpoint(repo: Path) -> Optional[Path]:
    roots = [repo / "weights" / "checkpoints", repo / "weights"]
    hits = []
    for root in roots:
        try:
            if root.exists():
                hits += list(root.rglob("*.pth"))
                hits += list(root.rglob("*.pt"))
        except Exception:
            pass
    hits = [p for p in dict.fromkeys(hits) if p.is_file()]
    if not hits:
        return None

    def key(p: Path):
        s = p.name.lower()
        score = (1000 if "dior" in s else 0) + (100 if "swin" in s else 0)
        try:
            return (-score, -p.stat().st_size, str(p).lower())
        except Exception:
            return (-score, 0, str(p).lower())

    hits.sort(key=key)
    return hits[0]


def _sanitize_checkpoint(py: Path, repo: Path, src: Path, out_dir: Path, bridge) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / (src.stem + "_mustatil_sanitized.pth")
    if dst.exists() and dst.stat().st_size > 1024:
        _emit(bridge, "log", "Using existing sanitized checkpoint: " + str(dst))
        return dst

    code = (
        "import sys, torch\n"
        "src,dst=sys.argv[1],sys.argv[2]\n"
        "ck=torch.load(src,map_location='cpu')\n"
        "sd=ck.get('state_dict',ck)\n"
        "patterns=['dn_query_generator.label_embedding','label_embedding.weight','cls_branches','bbox_head_teacher','teacher','log_scale']\n"
        "removed=[]\n"
        "for k in list(sd.keys()):\n"
        "    if any(p.lower() in k.lower() for p in patterns):\n"
        "        removed.append(k); sd.pop(k,None)\n"
        "if isinstance(ck,dict) and 'state_dict' in ck: ck['state_dict']=sd\n"
        "else: ck={'state_dict':sd}\n"
        "ck.setdefault('meta',{})\n"
        "ck['meta']['mustatil_sanitized']=True\n"
        "ck['meta']['removed_keys_count']=len(removed)\n"
        "torch.save(ck,dst)\n"
        "print('Sanitized checkpoint:',dst)\n"
        "print('Removed keys:',len(removed))\n"
    )
    _run([py, "-c", code, src, dst], bridge, cwd=_mmd(repo), env=_env(repo), title="sanitize checkpoint")
    return dst


def _write_training_config(project: Path, meta: Dict[str, Any], bridge) -> Path:
    repo = _repo_root()
    py = _py()
    base_cfg = _default_base_config(repo)
    checkpoint = _default_checkpoint(repo)

    classes = list(meta["classes"])
    num_classes = len(classes)
    class_tuple = "(" + ", ".join(repr(c) for c in classes) + ("," if len(classes) == 1 else "") + ")"
    dataset_root = Path(meta["dataset_root"])
    data_root = dataset_root.as_posix() + "/"
    work_dir = dataset_root / "work_dirs" / "lae_dino_mustatil"
    work_dir.mkdir(parents=True, exist_ok=True)

    load_from = ""
    if checkpoint and checkpoint.exists():
        if num_classes != 20 or "dior" in checkpoint.name.lower():
            if py.exists():
                checkpoint = _sanitize_checkpoint(py, repo, checkpoint, dataset_root / "checkpoints", bridge)
            else:
                _emit(bridge, "log", "Python runtime missing; using checkpoint without sanitizing.")
        load_from = str(checkpoint).replace("\\", "/")

    cfg = dataset_root / "lae_dino_mustatil_train_from_project.py"
    lines = [
        "# Auto-generated by Mustatil LAE-DINO Project Trainer V17",
        "# Project: " + str(project),
        "# Base config: " + str(base_cfg),
        "_base_ = r" + repr(str(base_cfg).replace("\\", "/")),
        "data_root = r" + repr(data_root),
        "num_classes = " + str(num_classes),
        "metainfo = dict(classes=" + class_tuple + ")",
        "",
        "mustatil_train_pipeline = [",
        "    dict(type='LoadImageFromFile', backend_args=None),",
        "    dict(type='LoadAnnotations', with_bbox=True),",
        "    dict(type='RandomFlip', prob=0.5),",
        "    dict(type='RandomChoiceResize', scales=[(800, 800), (640, 640), (1000, 1000)], keep_ratio=True),",
        "    dict(type='PackDetInputs'),",
        "]",
        "",
        "mustatil_test_pipeline = [",
        "    dict(type='LoadImageFromFile', backend_args=None),",
        "    dict(type='Resize', scale=(800, 800), keep_ratio=True),",
        "    dict(type='LoadAnnotations', with_bbox=True),",
        "    dict(type='PackDetInputs'),",
        "]",
        "",
        "train_dataloader = dict(",
        "    _delete_=True, batch_size=1, num_workers=2, persistent_workers=False,",
        "    sampler=dict(type='DefaultSampler', shuffle=True),",
        "    batch_sampler=dict(type='AspectRatioBatchSampler'),",
        "    dataset=dict(",
        "        type='CocoDataset', data_root=data_root,",
        "        ann_file='annotations/instances_train.json',",
        "        data_prefix=dict(img='images/train/'),",
        "        metainfo=metainfo,",
        "        filter_cfg=dict(filter_empty_gt=False, min_size=1),",
        "        pipeline=mustatil_train_pipeline))",
        "",
        "val_dataloader = dict(",
        "    _delete_=True, batch_size=1, num_workers=2, persistent_workers=False,",
        "    sampler=dict(type='DefaultSampler', shuffle=False),",
        "    dataset=dict(",
        "        type='CocoDataset', data_root=data_root,",
        "        ann_file='annotations/instances_val.json',",
        "        data_prefix=dict(img='images/val/'),",
        "        metainfo=metainfo, test_mode=True,",
        "        pipeline=mustatil_test_pipeline))",
        "",
        "test_dataloader = val_dataloader",
        "",
        "val_evaluator = dict(_delete_=True, type='CocoMetric', ann_file=data_root + 'annotations/instances_val.json', metric='bbox', format_only=False)",
        "test_evaluator = dict(_delete_=True, type='CocoMetric', ann_file=data_root + 'annotations/instances_val.json', metric='bbox', format_only=False)",
        "",
        "train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)",
        "val_cfg = dict(type='ValLoop')",
        "test_cfg = dict(type='TestLoop')",
        "",
        "default_hooks = dict(",
        "    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_best='auto'),",
        "    logger=dict(type='LoggerHook', interval=10))",
        "",
        "optim_wrapper = dict(optimizer=dict(lr=0.00005))",
        "auto_scale_lr = dict(enable=False)",
        "work_dir = r" + repr(work_dir.as_posix()),
        "",
        "def _mustatil_patch_num_classes(obj):",
        "    if isinstance(obj, dict):",
        "        for k, v in list(obj.items()):",
        "            if k in ('num_classes', 'num_things_classes'): obj[k] = num_classes",
        "            else: _mustatil_patch_num_classes(v)",
        "    elif isinstance(obj, (list, tuple)):",
        "        for v in obj: _mustatil_patch_num_classes(v)",
        "",
        "try: _mustatil_patch_num_classes(model)",
        "except Exception: pass",
        "",
        "try:",
        "    model.setdefault('bbox_head', dict())",
        "    model['bbox_head']['num_classes'] = num_classes",
        "except Exception: pass",
        "",
        "try:",
        "    model.setdefault('dn_query_generator', dict())",
        "    model['dn_query_generator']['num_classes'] = num_classes",
        "except Exception: pass",
    ]
    if load_from:
        lines.append("")
        lines.append("load_from = r" + repr(load_from))

    cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta["config"] = str(cfg)
    meta["base_config"] = str(base_cfg)
    meta["load_from"] = load_from
    meta["work_dir"] = str(work_dir)
    (dataset_root / "mustatil_lae_dataset_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _emit(bridge, "log", "Training config written: " + str(cfg))
    return cfg


def _create_everything_from_project(project: Path, bridge) -> Dict[str, Any]:
    _emit(bridge, "status", "Scanne Annotator-Projekt...")
    pairs, label_files = _discover_pairs(project, bridge)
    classes = _load_classes(project, label_files, bridge)
    _emit(bridge, "log", "Classes: " + ", ".join(classes))

    _emit(bridge, "status", "Erzeuge COCO/LAE-DINO Dataset...")
    meta = _write_coco_dataset(project, pairs, classes, bridge)

    _emit(bridge, "status", "Erzeuge sichere LAE-DINO Training-Config...")
    cfg = _write_training_config(project, meta, bridge)
    return {
        "project": str(project),
        "dataset": meta.get("dataset_root"),
        "classes": ", ".join(classes),
        "config": str(cfg),
        "work_dir": meta.get("work_dir"),
        "load_from": meta.get("load_from", ""),
    }



# ------------------------------
# Integrated train button helpers
# ------------------------------

HOTFIX_MARKER_INTEGRATED_TRAIN = "# === Mustatil LAE-DINO integrated train hotfix V12 ==="


def _detect_train_config(project: Path) -> Optional[Path]:
    candidates = [
        project / "lae_dino_dataset" / "lae_dino_mustatil_train_from_project.py",
        project / "lae_dino_dataset" / "lae_dino_mustatil_train_auto_v6.py",
        project / "lae_dino_dataset" / "lae_dino_mustatil_train_auto_v5.py",
        project / "lae_dino_dataset" / "lae_dino_mustatil_train_auto_v4.py",
        project / "lae_dino_dataset" / "lae_dino_mustatil_train_auto_v3.py",
        project / "lae_dino_dataset" / "lae_dino_mustatil_train_auto.py",
    ]
    for p in candidates:
        if p.exists():
            return p

    folder = project / "lae_dino_dataset"
    try:
        if folder.exists():
            hits = [p for p in folder.glob("*.py") if "train" in p.name.lower() or "mustatil" in p.name.lower()]
            if hits:
                hits.sort(key=lambda p: -p.stat().st_mtime)
                return hits[0]
    except Exception:
        pass
    return None


def _detect_workdir_from_config(config: Optional[Path], project: Path) -> Path:
    if config and config.exists():
        try:
            import re
            txt = config.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r"work_dir\s*=\s*r?[\"']([^\"']+)[\"']", txt)
            if m:
                return Path(m.group(1))
        except Exception:
            pass
        return config.parent / "work_dirs" / "lae_dino_mustatil"
    return project / "lae_dino_dataset" / "work_dirs" / "lae_dino_mustatil"




def _patch_config_for_lae_training(config: Path, bridge, settings: Optional[Dict[str, Any]] = None) -> None:
    settings = settings or {}

    img_size = int(settings.get("img_size") or 800)
    img_size = max(256, min(4096, img_size))
    small_size = max(256, int(img_size * 0.8))
    large_size = min(4096, int(img_size * 1.25))

    batch_size = int(settings.get("batch_size") or 1)
    batch_size = max(1, min(64, batch_size))

    workers = int(settings.get("workers") or 2)
    workers = max(0, min(32, workers))

    epochs = int(settings.get("epochs") or 20)
    epochs = max(1, min(1000, epochs))

    lr = float(settings.get("lr") or 0.00005)

    val_interval = int(settings.get("val_interval") or 1)
    val_interval = max(1, min(1000, val_interval))

    num_queries = int(settings.get("num_queries") or 50)
    num_queries = max(10, min(900, num_queries))

    max_per_img = int(settings.get("max_per_img") or min(50, num_queries))
    max_per_img = max(1, min(900, max_per_img))

    dn_queries = int(settings.get("dn_queries") or max(0, min(25, num_queries // 2)))
    dn_queries = max(0, min(900, dn_queries))

    text = config.read_text(encoding="utf-8", errors="ignore")
    import re

    # Remove older appended hotfix blocks first.
    text = re.sub(
        r"\n# === Mustatil LAE-DINO integrated train hotfix.*?(?=\n# === |\Z)",
        "",
        text,
        flags=re.S,
    )
    text = re.sub(
        r"\n# === Mustatil LAE-DINO query-count hotfix.*?(?=\n# === |\Z)",
        "",
        text,
        flags=re.S,
    )
    text = re.sub(
        r"\n# === Mustatil LAE-DINO Train Button V[23] hotfix.*?(?=\n# === |\Z)",
        "",
        text,
        flags=re.S,
    )

    # Repair broken placeholder strings if an old V8 config is still around.
    text = text.replace("{img_size}", str(img_size))
    text = text.replace("{small_size}", str(small_size))
    text = text.replace("{large_size}", str(large_size))

    # V12 hard source-level edits:
    # These make the final MMEngine printed config show the right values.
    text = re.sub(r"num_queries\s*=\s*900", f"num_queries={num_queries}", text)
    text = re.sub(r"num_queries\s*=\s*\d+", f"num_queries={num_queries}", text)

    text = re.sub(r"max_per_img\s*=\s*300", f"max_per_img={max_per_img}", text)
    text = re.sub(r"max_per_img\s*=\s*\d+", f"max_per_img={max_per_img}", text)

    text = re.sub(r"num_dn_queries\s*=\s*100", f"num_dn_queries={dn_queries}", text)
    text = re.sub(r"num_dn_queries\s*=\s*\d+", f"num_dn_queries={dn_queries}", text)

    # Also directly rewrite generated Mustatil pipeline image-size tuples.
    text = re.sub(
        r"scales=\[\(\s*\d+\s*,\s*\d+\s*\),\s*\(\s*\d+\s*,\s*\d+\s*\),\s*\(\s*\d+\s*,\s*\d+\s*\)\]",
        f"scales=[({img_size}, {img_size}), ({small_size}, {small_size}), ({large_size}, {large_size})]",
        text,
    )
    text = re.sub(
        r"scale=\(\s*\d+\s*,\s*\d+\s*\)",
        f"scale=({img_size}, {img_size})",
        text,
    )

    patch_lines = [
        "",
        "# === Mustatil LAE-DINO integrated train hotfix V12 ===",
        "# V12 writes query/image/training settings both by source rewrite and by final runtime override.",
        "# V12 also clamps torch.topk(k) to the available candidate count.",
        "",
        "# Robust guard for LAE-DINO pre_decoder topk.",
        "import torch as _mustatil_torch",
        "if not hasattr(_mustatil_torch, '_mustatil_original_topk'):",
        "    _mustatil_torch._mustatil_original_topk = _mustatil_torch.topk",
        "    def _mustatil_safe_topk(input, k, dim=None, largest=True, sorted=True, *, out=None):",
        "        try:",
        "            d = -1 if dim is None else dim",
        "            kk = int(k)",
        "            size = int(input.size(d))",
        "            if size > 0 and kk > size:",
        "                kk = size",
        "            if out is not None:",
        "                return _mustatil_torch._mustatil_original_topk(input, kk, dim=dim, largest=largest, sorted=sorted, out=out)",
        "            return _mustatil_torch._mustatil_original_topk(input, kk, dim=dim, largest=largest, sorted=sorted)",
        "        except Exception:",
        "            if out is not None:",
        "                return _mustatil_torch._mustatil_original_topk(input, k, dim=dim, largest=largest, sorted=sorted, out=out)",
        "            return _mustatil_torch._mustatil_original_topk(input, k, dim=dim, largest=largest, sorted=sorted)",
        "    _mustatil_torch.topk = _mustatil_safe_topk",
        "",
        "mustatil_lae_train_meta_keys = (",
        "    'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',",
        "    'flip', 'flip_direction', 'text', 'custom_entities',",
        ")",
        "",
        "mustatil_lae_test_meta_keys = (",
        "    'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',",
        "    'text', 'custom_entities',",
        ")",
        "",
        "mustatil_train_pipeline_text = [",
        "    dict(type='LoadImageFromFile', backend_args=None),",
        "    dict(type='LoadAnnotations', with_bbox=True),",
        "    dict(type='RandomFlip', prob=0.5),",
        f"    dict(type='RandomChoiceResize', scales=[({img_size}, {img_size}), ({small_size}, {small_size}), ({large_size}, {large_size})], keep_ratio=True),",
        "    dict(type='PackDetInputs', meta_keys=mustatil_lae_train_meta_keys),",
        "]",
        "",
        "mustatil_test_pipeline_text = [",
        "    dict(type='LoadImageFromFile', backend_args=None),",
        f"    dict(type='Resize', scale=({img_size}, {img_size}), keep_ratio=True),",
        "    dict(type='LoadAnnotations', with_bbox=True),",
        "    dict(type='PackDetInputs', meta_keys=mustatil_lae_test_meta_keys),",
        "]",
        "",
        "def _mustatil_get(obj, key, default=None):",
        "    try:",
        "        return obj[key]",
        "    except Exception:",
        "        return getattr(obj, key, default)",
        "",
        "def _mustatil_set(obj, key, value):",
        "    try:",
        "        obj[key] = value",
        "        return",
        "    except Exception:",
        "        pass",
        "    try:",
        "        setattr(obj, key, value)",
        "    except Exception:",
        "        pass",
        "",
        "def _mustatil_patch_dataset(dl, pipeline):",
        "    ds = _mustatil_get(dl, 'dataset')",
        "    if ds is None:",
        "        return",
        "    _mustatil_set(ds, 'pipeline', pipeline)",
        "    _mustatil_set(ds, 'return_classes', True)",
        "    _mustatil_set(ds, 'metainfo', metainfo)",
        "",
        "try:",
        "    _mustatil_patch_dataset(train_dataloader, mustatil_train_pipeline_text)",
        "except Exception:",
        "    pass",
        "try:",
        "    _mustatil_patch_dataset(val_dataloader, mustatil_test_pipeline_text)",
        "except Exception:",
        "    pass",
        "try:",
        "    test_dataloader = val_dataloader",
        "except Exception:",
        "    pass",
        "",
        "def _mustatil_patch_num_classes(obj):",
        "    if isinstance(obj, dict):",
        "        for k, v in list(obj.items()):",
        "            try:",
        "                if k in ('num_classes', 'num_things_classes'):",
        "                    obj[k] = num_classes",
        "                else:",
        "                    _mustatil_patch_num_classes(v)",
        "            except Exception:",
        "                pass",
        "    elif isinstance(obj, (list, tuple)):",
        "        for v in obj:",
        "            _mustatil_patch_num_classes(v)",
        "",
        "try:",
        "    _mustatil_patch_num_classes(model)",
        "except Exception:",
        "    pass",
        "",
        "try:",
        "    model['bbox_head']['num_classes'] = num_classes",
        "except Exception:",
        "    pass",
        "try:",
        "    model.bbox_head.num_classes = num_classes",
        "except Exception:",
        "    pass",
        "",
        "# Final hard query override.",
        "try:",
        f"    model['num_queries'] = {num_queries}",
        "except Exception:",
        "    pass",
        "try:",
        f"    model.num_queries = {num_queries}",
        "except Exception:",
        "    pass",
        "",
        "try:",
        f"    model['test_cfg']['max_per_img'] = {max_per_img}",
        "except Exception:",
        "    pass",
        "try:",
        f"    model.test_cfg.max_per_img = {max_per_img}",
        "except Exception:",
        "    pass",
        "",
        "try:",
        f"    model['dn_cfg']['group_cfg']['num_dn_queries'] = {dn_queries}",
        "except Exception:",
        "    pass",
        "try:",
        f"    model.dn_cfg.group_cfg.num_dn_queries = {dn_queries}",
        "except Exception:",
        "    pass",
        "",
        "try:",
        f"    train_dataloader['batch_size'] = {batch_size}",
        "except Exception:",
        "    pass",
        "try:",
        f"    train_dataloader['num_workers'] = {workers}",
        "except Exception:",
        "    pass",
        "try:",
        f"    val_dataloader['num_workers'] = min({workers}, 4)",
        "except Exception:",
        "    pass",
        "try:",
        f"    train_cfg['max_epochs'] = {epochs}",
        "except Exception:",
        "    pass",
        "try:",
        f"    train_cfg['val_interval'] = {val_interval}",
        "except Exception:",
        "    pass",
        "try:",
        f"    optim_wrapper['optimizer']['lr'] = {lr!r}",
        "except Exception:",
        "    pass",
        "",
        "val_evaluator = dict(",
        "    _delete_=True,",
        "    type='CocoMetric',",
        "    ann_file=data_root + 'annotations/instances_val.json',",
        "    metric='bbox',",
        "    format_only=False",
        ")",
        "test_evaluator = val_evaluator",
        "try:",
        "    dataset_prefixes = []",
        "except Exception:",
        "    pass",
        "try:",
        "    all_metrics = []",
        "except Exception:",
        "    pass",
        "",
    ]

    patch = "\n".join(patch_lines)
    config.write_text(text.rstrip() + "\n" + patch + "\n", encoding="utf-8")

    # Log what V12 actually wrote, so the user can see it before train.py starts.
    _emit(bridge, "log", "Training hotfix/settings V17 wurden hart in die Config geschrieben:")
    _emit(bridge, "log", "Config: " + str(config))
    _emit(bridge, "log", f"V17 num_queries={num_queries}, max_per_img={max_per_img}, dn_queries={dn_queries}, image_size={img_size}")


def _patch_lae_dino_source_safe_topk(repo: Path, bridge) -> None:
    """Patch LAE-DINO source once so topk cannot request more candidates than exist."""
    try:
        lae_file = _mmd(repo) / "mmdet" / "models" / "detectors" / "lae_dino.py"
        if not lae_file.exists():
            _emit(bridge, "log", "safe_topk source patch skipped; file not found: " + str(lae_file))
            return

        text = lae_file.read_text(encoding="utf-8", errors="ignore")
        if "# Mustatil safe_topk helper" not in text:
            marker = "import torch"
            helper = """

# Mustatil safe_topk helper
def _mustatil_safe_topk(input, k, dim=None, largest=True, sorted=True):
    try:
        d = -1 if dim is None else dim
        kk = int(k)
        size = int(input.size(d))
        if size > 0 and kk > size:
            kk = size
        return torch.topk(input, kk, dim=dim, largest=largest, sorted=sorted)
    except Exception:
        return torch.topk(input, k, dim=dim, largest=largest, sorted=sorted)
"""
            if marker in text:
                text = text.replace(marker, marker + helper, 1)
            else:
                text = helper + "\n" + text

        # Target the common LAE-DINO expression:
        # torch.topk(output_memory_class.max(-1)[0], self.num_queries, dim=1)
        text = text.replace(
            "torch.topk(\\n            enc_outputs_class.max(-1)[0], self.num_queries, dim=1)",
            "_mustatil_safe_topk(\\n            enc_outputs_class.max(-1)[0], self.num_queries, dim=1)",
        )
        text = text.replace(
            "torch.topk(enc_outputs_class.max(-1)[0], self.num_queries, dim=1)",
            "_mustatil_safe_topk(enc_outputs_class.max(-1)[0], self.num_queries, dim=1)",
        )
        text = text.replace(
            "torch.topk(\\n            output_memory_class.max(-1)[0], self.num_queries, dim=1)",
            "_mustatil_safe_topk(\\n            output_memory_class.max(-1)[0], self.num_queries, dim=1)",
        )
        text = text.replace(
            "torch.topk(output_memory_class.max(-1)[0], self.num_queries, dim=1)",
            "_mustatil_safe_topk(output_memory_class.max(-1)[0], self.num_queries, dim=1)",
        )

        # Generic line-level fallback: replace only topk calls that contain self.num_queries nearby.
        lines = text.splitlines()
        changed = False
        for i, line in enumerate(lines):
            if "torch.topk" in line and "self.num_queries" in "".join(lines[i:i+4]):
                lines[i] = line.replace("torch.topk", "_mustatil_safe_topk")
                changed = True
        if changed:
            text = "\n".join(lines) + "\n"

        lae_file.write_text(text, encoding="utf-8")
        _emit(bridge, "log", "safe_topk source patch active: " + str(lae_file))
    except Exception as exc:
        _emit(bridge, "log", "safe_topk source patch failed: " + str(exc))
        try:
            _emit(bridge, "log", traceback.format_exc())
        except Exception:
            pass





def _v14_int_setting(settings: Dict[str, Any], key: str, default: int, lo: int, hi: int) -> int:
    try:
        v = int(settings.get(key) or default)
    except Exception:
        v = default
    return max(lo, min(hi, v))


def _v14_float_setting(settings: Dict[str, Any], key: str, default: float, lo: float, hi: float) -> float:
    try:
        v = float(settings.get(key) or default)
    except Exception:
        v = default
    return max(lo, min(hi, v))


def _write_v14_final_config_with_mmengine(
    py: Path,
    repo: Path,
    base_config: Path,
    bridge,
    settings: Optional[Dict[str, Any]] = None,
) -> Path:
    # V14: LAE runtime/MMEngine loads the generated config, mutates the Config object,
    # dumps a complete final config, then verifies that dumped config.
    settings = settings or {}

    img_size = _v14_int_setting(settings, "img_size", 800, 256, 4096)
    small_size = max(256, int(img_size * 0.8))
    large_size = min(4096, int(img_size * 1.25))
    batch_size = _v14_int_setting(settings, "batch_size", 1, 1, 64)
    workers = _v14_int_setting(settings, "workers", 2, 0, 32)
    epochs = _v14_int_setting(settings, "epochs", 20, 1, 1000)
    val_interval = _v14_int_setting(settings, "val_interval", 5, 1, 1000)
    lr = _v14_float_setting(settings, "lr", 0.00005, 0.00000001, 1.0)
    num_queries = _v14_int_setting(settings, "num_queries", 50, 10, 900)
    max_per_img = _v14_int_setting(settings, "max_per_img", min(50, num_queries), 1, 900)
    dn_queries = _v14_int_setting(settings, "dn_queries", max(0, min(25, num_queries // 2)), 0, 900)
    max_per_img = min(max_per_img, num_queries)

    final_config = base_config.with_name("lae_dino_mustatil_train_v14_mmengine_final.py")
    build_script = base_config.with_name("_v14_build_final_config.py")
    ann_val = (base_config.parent / "annotations" / "instances_val.json").as_posix()

    L = []
    L.append("# Auto-generated by Mustatil LAE-DINO Trainer V17.")
    L.append("from mmengine.config import Config, ConfigDict")
    L.append("from pathlib import Path")
    L.append("import copy")
    L.append("")
    L.append("base_config = " + repr(str(base_config)))
    L.append("final_config = " + repr(str(final_config)))
    L.append("img_size = " + repr(img_size))
    L.append("small_size = " + repr(small_size))
    L.append("large_size = " + repr(large_size))
    L.append("batch_size = " + repr(batch_size))
    L.append("workers = " + repr(workers))
    L.append("epochs = " + repr(epochs))
    L.append("val_interval = " + repr(val_interval))
    L.append("lr = " + repr(lr))
    L.append("num_queries = " + repr(num_queries))
    L.append("max_per_img = " + repr(max_per_img))
    L.append("dn_queries = " + repr(dn_queries))
    L.append("ann_val = " + repr(ann_val))
    L.append("")
    L.append("cfg = Config.fromfile(base_config)")
    L.append("")
    L.append("def ensure_dict(obj, key):")
    L.append("    if key not in obj or obj[key] is None:")
    L.append("        obj[key] = ConfigDict()")
    L.append("    return obj[key]")
    L.append("")
    L.append("def recursive_set_classes(obj):")
    L.append("    if isinstance(obj, dict):")
    L.append("        for k in list(obj.keys()):")
    L.append("            if k in ('num_classes', 'num_things_classes'):")
    L.append("                obj[k] = 2")
    L.append("            else:")
    L.append("                recursive_set_classes(obj[k])")
    L.append("    elif isinstance(obj, (list, tuple)):")
    L.append("        for v in obj:")
    L.append("            recursive_set_classes(v)")
    L.append("")
    L.append("def patch_model(cfg):")
    L.append("    if 'model' not in cfg or cfg['model'] is None:")
    L.append("        raise RuntimeError('No model object found in loaded config')")
    L.append("    model = cfg['model']")
    L.append("    model['num_queries'] = num_queries")
    L.append("    bbox_head = ensure_dict(model, 'bbox_head')")
    L.append("    bbox_head['num_classes'] = 2")
    L.append("    test_cfg = ensure_dict(model, 'test_cfg')")
    L.append("    test_cfg['max_per_img'] = max_per_img")
    L.append("    dn_cfg = ensure_dict(model, 'dn_cfg')")
    L.append("    group_cfg = ensure_dict(dn_cfg, 'group_cfg')")
    L.append("    group_cfg['num_dn_queries'] = dn_queries")
    L.append("    recursive_set_classes(model)")
    L.append("")
    L.append("def patch_training(cfg):")
    L.append("    cfg['num_classes'] = 2")
    L.append("    cfg['metainfo'] = dict(classes=('mustatil', 'false_positive'))")
    L.append("    train_meta_keys = ('img_id','img_path','ori_shape','img_shape','scale_factor','flip','flip_direction','text','custom_entities')")
    L.append("    test_meta_keys = ('img_id','img_path','ori_shape','img_shape','scale_factor','text','custom_entities')")
    L.append("    train_pipeline = [")
    L.append("        dict(type='LoadImageFromFile', backend_args=None),")
    L.append("        dict(type='LoadAnnotations', with_bbox=True),")
    L.append("        dict(type='RandomFlip', prob=0.5),")
    L.append("        dict(type='RandomChoiceResize', scales=[(img_size,img_size),(small_size,small_size),(large_size,large_size)], keep_ratio=True),")
    L.append("        dict(type='PackDetInputs', meta_keys=train_meta_keys),")
    L.append("    ]")
    L.append("    test_pipeline = [")
    L.append("        dict(type='LoadImageFromFile', backend_args=None),")
    L.append("        dict(type='Resize', scale=(img_size,img_size), keep_ratio=True),")
    L.append("        dict(type='LoadAnnotations', with_bbox=True),")
    L.append("        dict(type='PackDetInputs', meta_keys=test_meta_keys),")
    L.append("    ]")
    L.append("    cfg['mustatil_train_pipeline_v14'] = train_pipeline")
    L.append("    cfg['mustatil_test_pipeline_v14'] = test_pipeline")
    L.append("    if 'train_cfg' in cfg and cfg['train_cfg'] is not None:")
    L.append("        cfg['train_cfg']['max_epochs'] = epochs")
    L.append("        cfg['train_cfg']['val_interval'] = val_interval")
    L.append("    else:")
    L.append("        cfg['train_cfg'] = dict(type='EpochBasedTrainLoop', max_epochs=epochs, val_interval=val_interval)")
    L.append("    if 'optim_wrapper' in cfg and cfg['optim_wrapper'] is not None:")
    L.append("        if 'optimizer' not in cfg['optim_wrapper'] or cfg['optim_wrapper']['optimizer'] is None:")
    L.append("            cfg['optim_wrapper']['optimizer'] = ConfigDict()")
    L.append("        cfg['optim_wrapper']['optimizer']['lr'] = lr")
    L.append("    def patch_dl(name, pipeline, is_train):")
    L.append("        if name not in cfg or cfg[name] is None:")
    L.append("            return")
    L.append("        dl = cfg[name]")
    L.append("        dl['persistent_workers'] = False")
    L.append("        if is_train:")
    L.append("            dl['batch_size'] = batch_size")
    L.append("            dl['num_workers'] = workers")
    L.append("        else:")
    L.append("            dl['num_workers'] = min(workers, 4)")
    L.append("        ds = dl.get('dataset', None)")
    L.append("        if ds is not None:")
    L.append("            ds['metainfo'] = cfg['metainfo']")
    L.append("            ds['return_classes'] = True")
    L.append("            ds['pipeline'] = pipeline")
    L.append("            if is_train:")
    L.append("                ds['filter_cfg'] = dict(filter_empty_gt=False, min_size=1)")
    L.append("    patch_dl('train_dataloader', train_pipeline, True)")
    L.append("    patch_dl('val_dataloader', test_pipeline, False)")
    L.append("    try:")
    L.append("        cfg['test_dataloader'] = copy.deepcopy(cfg['val_dataloader'])")
    L.append("    except Exception:")
    L.append("        cfg['test_dataloader'] = cfg.get('val_dataloader', None)")
    L.append("    cfg['val_evaluator'] = dict(type='CocoMetric', ann_file=ann_val, metric='bbox', format_only=False)")
    L.append("    cfg['test_evaluator'] = cfg['val_evaluator']")
    L.append("    cfg['dataset_prefixes'] = []")
    L.append("    cfg['all_metrics'] = []")
    L.append("    cfg['mustatil_v14_num_queries'] = num_queries")
    L.append("    cfg['mustatil_v14_max_per_img'] = max_per_img")
    L.append("    cfg['mustatil_v14_dn_queries'] = dn_queries")
    L.append("    cfg['mustatil_v14_num_classes'] = 2")
    L.append("")
    L.append("def strip_delete_keys(obj):")
    L.append("    if isinstance(obj, dict):")
    L.append("        obj.pop('_delete_', None)")
    L.append("        for v in list(obj.values()):")
    L.append("            strip_delete_keys(v)")
    L.append("    elif isinstance(obj, (list, tuple)):")
    L.append("        for v in obj:")
    L.append("            strip_delete_keys(v)")
    L.append("")
    L.append("patch_model(cfg)")
    L.append("patch_training(cfg)")
    L.append("strip_delete_keys(cfg)")
    L.append("Path(final_config).parent.mkdir(parents=True, exist_ok=True)")
    L.append("cfg.dump(final_config)")
    L.append("cfg2 = Config.fromfile(final_config)")
    L.append("m = cfg2['model']")
    L.append("print('V14_BUILD final_config=' + final_config)")
    L.append("print('V14_BUILD num_queries=' + str(m.get('num_queries')))")
    L.append("print('V14_BUILD max_per_img=' + str(m.get('test_cfg', {}).get('max_per_img')))")
    L.append("print('V14_BUILD dn_queries=' + str(m.get('dn_cfg', {}).get('group_cfg', {}).get('num_dn_queries')))")
    L.append("print('V14_BUILD num_classes=' + str(m.get('bbox_head', {}).get('num_classes')))")
    L.append("if str(m.get('num_queries')) == '900':")
    L.append("    raise RuntimeError('V14 build failed: num_queries is still 900')")
    L.append("if str(m.get('bbox_head', {}).get('num_classes')) == '20':")
    L.append("    raise RuntimeError('V14 build failed: num_classes is still 20')")

    build_script.write_text("\n".join(L) + "\n", encoding="utf-8")
    _emit(bridge, "log", "V14 MMEngine builder script:")
    _emit(bridge, "log", str(build_script))

    p = subprocess.run(
        [str(py), str(build_script)],
        cwd=str(_mmd(repo)),
        env=_env(repo, "cpu"),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
    )
    out = p.stdout or ""
    for line in out.splitlines():
        _emit(bridge, "log", line)

    if p.returncode != 0:
        raise RuntimeError("V14 final config build failed:\n" + out)

    _emit(bridge, "log", "V14 final MMEngine config geschrieben:")
    _emit(bridge, "log", str(final_config))
    return final_config


def _verify_v14_config_with_runtime(py: Path, repo: Path, train_config: Path, bridge) -> None:
    check_script = train_config.with_name("_v14_verify_config.py")
    script_lines = [
        "from mmengine.config import Config",
        "cfg = Config.fromfile(" + repr(str(train_config)) + ")",
        "print('V14_VERIFY keys=' + ','.join(list(cfg.keys())[:50]))",
        "m = cfg.get('model', None)",
        "if m is None:",
        "    print('V14_VERIFY model=None')",
        "else:",
        "    print('V14_VERIFY num_queries=' + str(m.get('num_queries')))",
        "    print('V14_VERIFY max_per_img=' + str(m.get('test_cfg', {}).get('max_per_img')))",
        "    print('V14_VERIFY dn_queries=' + str(m.get('dn_cfg', {}).get('group_cfg', {}).get('num_dn_queries')))",
        "    print('V14_VERIFY num_classes=' + str(m.get('bbox_head', {}).get('num_classes')))",
        "print('V14_VERIFY val_evaluator_keys=' + ','.join(sorted(list(cfg.get('val_evaluator', {}).keys()))))",
    ]
    check_script.write_text("\n".join(script_lines) + "\n", encoding="utf-8")

    p = subprocess.run(
        [str(py), str(check_script)],
        cwd=str(_mmd(repo)),
        env=_env(repo, "cpu"),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )
    out = p.stdout or ""
    for line in out.splitlines():
        _emit(bridge, "log", line)

    if p.returncode != 0:
        raise RuntimeError("V14 Config Verify konnte nicht geladen werden:\n" + out)

    bad = (
        "V14_VERIFY model=None" in out
        or "V14_VERIFY num_queries=900" in out
        or "V14_VERIFY num_classes=20" in out
        or "V14_VERIFY num_queries=None" in out
        or "V14_VERIFY num_classes=None" in out
        or "_delete_" in out
    )
    if bad:
        raise RuntimeError("V14 Verify fehlgeschlagen: Config enthält weiterhin alte/fehlende LAE-DINO-Werte. Training wird gestoppt.")


def _bg_train(project_text: str, bridge, button=None, settings: Optional[Dict[str, Any]] = None):
    settings = settings or {}
    project_text = str(project_text or "").strip().strip('"')
    if button is not None:
        try:
            button.setEnabled(False)
        except Exception:
            pass

    def runner():
        try:
            _emit(bridge, "clear")
            _emit(bridge, "status", "Training: Dataset + Config werden zuerst aus Projektbildern aktualisiert...")
            project = Path(project_text).expanduser()
            if not project.exists():
                raise RuntimeError("Projektordner nicht gefunden: " + str(project))

            _emit(bridge, "log", "")
            _emit(bridge, "log", "=== AUTO CREATE DATASET + CONFIG BEFORE TRAINING ===")
            _emit(bridge, "log", "Project folder: " + str(project))
            _emit(bridge, "log", "Nutze Projektbilder/Labels direkt aus dem Projektordner: erst Dataset, dann Config, dann Training.")
            result = _create_everything_from_project(project, bridge)
            _emit(bridge, "log", "")
            _emit(bridge, "log", "=== DATASET + CONFIG READY ===")
            for k, v in result.items():
                _emit(bridge, "log", f"{k}: {v}")

            config = _detect_train_config(project)
            if config is None or not config.exists():
                raise RuntimeError(
                    "Training-Config wurde nach dem Projektimport nicht gefunden. Erwartet: "
                    + str(project / "lae_dino_dataset" / "lae_dino_mustatil_train_from_project.py")
                )

            repo = _repo_root()
            py = _py()
            train_py = _mmd(repo) / "tools" / "train.py"
            workdir = _detect_workdir_from_config(config, project)

            _patch_lae_dino_source_safe_topk(repo, bridge)

            _emit(bridge, "log", "")
            _emit(bridge, "log", "=== LAE-DINO TRAINING ===")
            _emit(bridge, "log", "Project: " + str(project))
            _emit(bridge, "log", "Runtime Python: " + str(py))
            _emit(bridge, "log", "train.py: " + str(train_py))
            _emit(bridge, "log", "Config: " + str(config))
            _emit(bridge, "log", "Work dir: " + str(workdir))
            _emit(bridge, "log", "Device: " + str(settings.get("device", "auto")))
            _emit(bridge, "log", "Image size: " + str(settings.get("img_size", 800)))
            _emit(bridge, "log", "Epochs: " + str(settings.get("epochs", 20)))
            _emit(bridge, "log", "Batch size: " + str(settings.get("batch_size", 1)))
            _emit(bridge, "log", "Workers: " + str(settings.get("workers", 2)))
            _emit(bridge, "log", "LR: " + str(settings.get("lr", 0.00005)))
            _emit(bridge, "log", "Num queries: " + str(settings.get("num_queries", 50)))
            _emit(bridge, "log", "Max per image: " + str(settings.get("max_per_img", 50)))
            _emit(bridge, "log", "DN queries: " + str(settings.get("dn_queries", 25)))
            _emit(bridge, "log", "")

            if not py.exists():
                raise RuntimeError("LAE-DINO-PATCH Python nicht gefunden: " + str(py))
            if not train_py.exists():
                raise RuntimeError("LAE-DINO train.py nicht gefunden: " + str(train_py))

            # V16: patch base config lightly, then let LAE runtime/MMEngine build a real final config.
            _patch_config_for_lae_training(config, bridge, settings)
            train_config = _write_v14_final_config_with_mmengine(py, repo, config, bridge, settings)
            _verify_v14_config_with_runtime(py, repo, train_config, bridge)

            workdir.mkdir(parents=True, exist_ok=True)

            cmd = [str(py), str(train_py), str(train_config), "--work-dir", str(workdir)]
            _emit(bridge, "log", "RUN: " + " ".join(('"' + c + '"') if " " in c else c for c in cmd))

            p = subprocess.Popen(
                cmd,
                cwd=str(_mmd(repo)),
                env=_env(repo, str(settings.get("device", "auto"))),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert p.stdout is not None
            lines = []
            last = 0.0
            for line in p.stdout:
                line = line.rstrip("\r\n")
                lines.append(line)
                _emit(bridge, "log", line)
                now = time.time()
                if now - last > 4:
                    _emit(bridge, "status", "Training läuft... " + line[:160])
                    last = now

            rc = p.wait()
            if rc != 0:
                raise RuntimeError("LAE-DINO training failed with exit code " + str(rc) + "\n" + "\n".join(lines[-120:]))

            _emit(bridge, "status", "Training fertig: " + str(workdir))
            _emit(bridge, "log", "")
            _emit(bridge, "log", "=== TRAINING DONE ===")
            _emit(bridge, "log", "Work dir: " + str(workdir))
        except Exception as exc:
            _emit(bridge, "status", "Training Fehler: " + str(exc))
            _emit(bridge, "log", "ERROR: " + str(exc))
            _emit(bridge, "log", traceback.format_exc())
        finally:
            if button is not None:
                _emit(bridge, "enable_train_button", "1")

    threading.Thread(target=runner, daemon=True, name="LAE_DINO_Integrated_Training_V17_AutoDatasetConfig").start()




# ------------------------------
# UI - V18 anti-freeze insert-only
# ------------------------------
# V18 intentionally does NOT hard-replace tabs and does NOT scan QApplication.allWidgets
# repeatedly. Freezes in V17 were most likely caused by replacing/inserting tabs while
# QTabWidget hooks were also scheduling broad rescans. V18 only inserts a compact panel
# into an existing LAE-DINO trainer tab, or creates ONE LAE-DINO tab on the main tab bar.

_PATCHED_QTAB_V18 = False
_ORIG_ADD_V18 = None
_ORIG_INSERT_V18 = None
_DONE_PAGES_V18 = set()
_DONE_TABS_V18 = set()


def _is_lae_trainer_label(text: str) -> bool:
    low = str(text or "").strip().lower()
    return ("lae" in low and "dino" in low and ("train" in low or "trainer" in low or "training" in low))


def _is_normal_trainer_label(text: str) -> bool:
    low = str(text or "").strip().lower()
    return (
        low in {"trainer", "training", "yolo trainer", "yolo training"}
        or ("yolo" in low and "train" in low)
    )


def _looks_like_main_tab_widget(tw) -> bool:
    try:
        labels = [str(tw.tabText(i) or "").strip().lower() for i in range(tw.count())]
    except Exception:
        return False
    if not labels:
        return False
    # Stronger test than V17: avoid random nested QTabWidgets.
    has_detection = any("detection" in x or "detect" in x for x in labels)
    has_satellite = any("satellite" in x or "remote" in x or "sensing" in x for x in labels)
    has_training = any("train" in x or "trainer" in x for x in labels)
    has_pipeline = any("pipeline" in x or "annotator" in x or "annotation" in x for x in labels)
    return (has_detection and has_training) or (has_detection and has_satellite and has_pipeline)


def _page_inner_widget(page):
    try:
        from PySide6.QtWidgets import QScrollArea
        if isinstance(page, QScrollArea) and page.widget() is not None:
            return page.widget()
    except Exception:
        pass
    return page


def _find_or_make_layout(page):
    from PySide6.QtWidgets import QVBoxLayout
    inner = _page_inner_widget(page)
    try:
        layout = inner.layout()
    except Exception:
        layout = None
    if layout is None:
        layout = QVBoxLayout(inner)
        try:
            inner.setLayout(layout)
        except Exception:
            pass
    return inner, layout


def _settings_from_controls(device_combo, img_size_spin, epochs_spin, batch_spin, workers_spin, lr_spin, val_interval_spin, num_queries_spin, max_per_img_spin, dn_queries_spin) -> Dict[str, Any]:
    try:
        device = str(device_combo.currentText())
    except Exception:
        device = "cuda"
    return {
        "device": device,
        "img_size": int(img_size_spin.value()),
        "epochs": int(epochs_spin.value()),
        "batch_size": int(batch_spin.value()),
        "workers": int(workers_spin.value()),
        "lr": float(lr_spin.value()),
        "val_interval": int(val_interval_spin.value()),
        "num_queries": int(num_queries_spin.value()),
        "max_per_img": int(max_per_img_spin.value()),
        "dn_queries": int(dn_queries_spin.value()),
    }


def _make_v18_panel(page, ws):
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton,
        QFileDialog, QTextEdit, QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox
    )
    from PySide6.QtCore import QObject, Signal

    class Bridge(QObject):
        status = Signal(str)
        log = Signal(str)
        clear = Signal()
        enable_button = Signal(str)
        enable_dataset_button = Signal(str)
        enable_train_button = Signal(str)

    panel = QGroupBox("LAE-DINO Project Trainer V18 Anti-Freeze: Project Dataset + Config + Training")
    panel.setObjectName("MustatilLAEDINOProjectTrainerV18AntiFreezePanel")
    root = QVBoxLayout(panel)

    row = QHBoxLayout()
    row.addWidget(QLabel("Project folder"))
    project_edit = QLineEdit(str(_project_from_workspace(ws)))
    project_edit.setObjectName("MustatilLAEDINOV18ProjectFolder")
    row.addWidget(project_edit, 1)
    browse = QPushButton("Select project folder")
    row.addWidget(browse)
    root.addLayout(row)

    hint = QLabel("Anti-Freeze-Version: keine Tab-Ersetzung, keine schweren Arbeiten im UI-Thread. Erwartet project\\images und project\\labels. Training erzeugt Dataset + Config automatisch vorher neu.")
    hint.setWordWrap(True)
    root.addWidget(hint)

    actions = QHBoxLayout()
    dataset_btn = QPushButton("Create Dataset from project images")
    dataset_btn.setObjectName("MustatilLAEDINOCreateDatasetButtonV18")
    config_btn = QPushButton("Create safe training config")
    config_btn.setObjectName("MustatilLAEDINOCreateConfigButtonV18")
    train_btn = QPushButton("Start LAE-DINO training from project images")
    train_btn.setObjectName("MustatilLAEDINOIntegratedTrainButtonV18")
    actions.addWidget(dataset_btn)
    actions.addWidget(config_btn)
    actions.addWidget(train_btn)
    root.addLayout(actions)

    settings_box = QGroupBox("Training settings")
    settings_grid = QGridLayout(settings_box)

    device_combo = QComboBox()
    device_combo.addItems(["auto", "cuda", "cuda:0", "cpu"])
    try:
        device_combo.setCurrentText("cuda")
    except Exception:
        pass

    img_size_spin = QSpinBox(); img_size_spin.setRange(256, 4096); img_size_spin.setSingleStep(64); img_size_spin.setValue(800)
    epochs_spin = QSpinBox(); epochs_spin.setRange(1, 1000); epochs_spin.setValue(20)
    batch_spin = QSpinBox(); batch_spin.setRange(1, 64); batch_spin.setValue(1)
    workers_spin = QSpinBox(); workers_spin.setRange(0, 32); workers_spin.setValue(0)
    lr_spin = QDoubleSpinBox(); lr_spin.setDecimals(8); lr_spin.setRange(0.00000001, 1.0); lr_spin.setSingleStep(0.00001); lr_spin.setValue(0.00005)
    val_interval_spin = QSpinBox(); val_interval_spin.setRange(1, 1000); val_interval_spin.setValue(5)
    num_queries_spin = QSpinBox(); num_queries_spin.setRange(10, 900); num_queries_spin.setSingleStep(10); num_queries_spin.setValue(50)
    max_per_img_spin = QSpinBox(); max_per_img_spin.setRange(1, 900); max_per_img_spin.setSingleStep(10); max_per_img_spin.setValue(50)
    dn_queries_spin = QSpinBox(); dn_queries_spin.setRange(0, 900); dn_queries_spin.setSingleStep(10); dn_queries_spin.setValue(25)

    rows = [
        ("Device", device_combo),
        ("Image size", img_size_spin),
        ("Epochs", epochs_spin),
        ("Batch size", batch_spin),
        ("Workers", workers_spin),
        ("Learning rate", lr_spin),
        ("Val interval", val_interval_spin),
        ("Num queries", num_queries_spin),
        ("Max detections/image", max_per_img_spin),
        ("DN queries", dn_queries_spin),
    ]
    for i, (label, widget) in enumerate(rows):
        settings_grid.addWidget(QLabel(label), i // 2, (i % 2) * 2)
        settings_grid.addWidget(widget, i // 2, (i % 2) * 2 + 1)
    root.addWidget(settings_box)

    status = QLabel("Ready. Click Create Dataset, Create Config, or Start Training.")
    status.setWordWrap(True)
    root.addWidget(status)

    console = QTextEdit()
    console.setObjectName("MustatilLAEDINOV18Console")
    console.setReadOnly(True)
    console.setMinimumHeight(180)
    root.addWidget(console)

    bridge = Bridge(panel)
    bridge.status.connect(lambda s: status.setText(str(s)))
    bridge.log.connect(lambda s: console.append(str(s)))
    bridge.clear.connect(lambda: console.clear())
    bridge.enable_button.connect(lambda _s: config_btn.setEnabled(True))
    bridge.enable_dataset_button.connect(lambda _s: dataset_btn.setEnabled(True))
    bridge.enable_train_button.connect(lambda _s: train_btn.setEnabled(True))

    def browse_dir():
        try:
            p = QFileDialog.getExistingDirectory(panel, "Select Mustatil project folder", project_edit.text() or str(Path.home()))
            if p:
                project_edit.setText(p)
        except Exception as exc:
            status.setText("Browse failed: " + str(exc))

    browse.clicked.connect(browse_dir)

    def run_dataset():
        try:
            dataset_btn.setEnabled(False)
            _bg_dataset(project_edit.text(), bridge, dataset_btn)
        except Exception as exc:
            dataset_btn.setEnabled(True)
            status.setText("Dataset start failed: " + str(exc))

    def run_config():
        try:
            config_btn.setEnabled(False)
            _bg(project_edit.text(), bridge, config_btn)
        except Exception as exc:
            config_btn.setEnabled(True)
            status.setText("Config start failed: " + str(exc))

    def run_train():
        try:
            train_btn.setEnabled(False)
            settings = _settings_from_controls(device_combo, img_size_spin, epochs_spin, batch_spin, workers_spin, lr_spin, val_interval_spin, num_queries_spin, max_per_img_spin, dn_queries_spin)
            _bg_train(project_edit.text(), bridge, train_btn, settings)
        except Exception as exc:
            train_btn.setEnabled(True)
            status.setText("Training start failed: " + str(exc))

    dataset_btn.clicked.connect(run_dataset)
    config_btn.clicked.connect(run_config)
    train_btn.clicked.connect(run_train)
    return panel


def _insert_panel_into_page(page, ws=None) -> bool:
    try:
        from PySide6.QtWidgets import QWidget
        if page is None:
            return False
        inner, layout = _find_or_make_layout(page)
        key = id(inner)
        if key in _DONE_PAGES_V18:
            return True
        try:
            existing = inner.findChild(QWidget, "MustatilLAEDINOProjectTrainerV18AntiFreezePanel")
            if existing is not None:
                _DONE_PAGES_V18.add(key)
                return True
        except Exception:
            pass
        panel = _make_v18_panel(page, ws)
        try:
            layout.insertWidget(0, panel)
        except Exception:
            layout.addWidget(panel)
        _DONE_PAGES_V18.add(key)
        try:
            if ws is not None and hasattr(ws, "log"):
                ws.log("LAE-DINO V18 Anti-Freeze panel inserted. Buttons: Create Dataset, Create Config, Start Training.")
        except Exception:
            pass
        return True
    except Exception as exc:
        _log("V18 insert panel failed: " + str(exc))
        try:
            traceback.print_exc()
        except Exception:
            pass
        return False


def _create_own_tab_once(tw) -> bool:
    try:
        from PySide6.QtWidgets import QWidget, QVBoxLayout
        if tw is None or id(tw) in _DONE_TABS_V18:
            return False
        if not _looks_like_main_tab_widget(tw):
            return False
        # Never create duplicates.
        for i in range(tw.count()):
            if _is_lae_trainer_label(tw.tabText(i)):
                return False
        ws = _workspace_from_widget(tw)
        page = QWidget()
        page.setObjectName("MustatilLAEDINOProjectTrainerV18AntiFreezeTab")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.addWidget(_make_v18_panel(page, ws))
        insert_at = tw.count()
        for i in range(tw.count()):
            if _is_normal_trainer_label(tw.tabText(i)):
                insert_at = i + 1
        # Use the original insert function if available, so our hook does not recursively schedule more work.
        if _ORIG_INSERT_V18 is not None:
            _ORIG_INSERT_V18(tw, insert_at, page, "LAE-DINO Trainer")
        else:
            tw.insertTab(insert_at, page, "LAE-DINO Trainer")
        _DONE_TABS_V18.add(id(tw))
        return True
    except Exception as exc:
        _log("V18 create tab failed: " + str(exc))
        return False


def _process_tabwidget(tw) -> bool:
    try:
        if tw is None:
            return False
        ok = False
        # Prefer inserting into an existing LAE-DINO Trainer page.
        for i in range(tw.count()):
            label = tw.tabText(i)
            if _is_lae_trainer_label(label):
                ws = _workspace_from_widget(tw) or _workspace_from_widget(tw.widget(i))
                if _insert_panel_into_page(tw.widget(i), ws):
                    ok = True
        if not ok:
            ok = _create_own_tab_once(tw)
        return ok
    except Exception as exc:
        _log("V18 process tabwidget failed: " + str(exc))
        return False


def _scan_local(root=None) -> bool:
    # Strictly local, bounded scan. No QApplication.allWidgets() storm.
    try:
        from PySide6.QtWidgets import QTabWidget
        widgets = []
        if root is not None:
            try:
                if isinstance(root, QTabWidget):
                    widgets.append(root)
            except Exception:
                pass
            try:
                widgets.extend(root.findChildren(QTabWidget))
            except Exception:
                pass
        # If register_plugin has no root, do one limited top-level scan only.
        if not widgets:
            try:
                from PySide6.QtWidgets import QApplication
                app = QApplication.instance()
                if app:
                    for w in app.topLevelWidgets():
                        try:
                            widgets.extend(w.findChildren(QTabWidget))
                        except Exception:
                            pass
            except Exception:
                pass
        seen = set()
        ok = False
        for tw in widgets[:20]:
            if id(tw) in seen:
                continue
            seen.add(id(tw))
            if _process_tabwidget(tw):
                ok = True
        return ok
    except Exception as exc:
        _log("V18 local scan failed: " + str(exc))
        return False


def _install_hook(root=None):
    global _PATCHED_QTAB_V18, _ORIG_ADD_V18, _ORIG_INSERT_V18
    try:
        from PySide6.QtWidgets import QTabWidget
        from PySide6.QtCore import QTimer
    except Exception as exc:
        _log("Qt unavailable: " + str(exc))
        return

    def schedule_for(tw=None):
        # Only a few delayed attempts. No endless timer, no broad rescan after every unrelated tab.
        try:
            if tw is not None:
                for ms in (100, 500, 1200):
                    QTimer.singleShot(ms, lambda tw=tw: _process_tabwidget(tw))
            else:
                for ms in (300, 1200, 3000):
                    QTimer.singleShot(ms, lambda root=root: _scan_local(root))
        except Exception:
            pass

    if not _PATCHED_QTAB_V18:
        _ORIG_ADD_V18 = QTabWidget.addTab
        _ORIG_INSERT_V18 = QTabWidget.insertTab

        def addTab_patched(self, page, *args, **kwargs):
            res = _ORIG_ADD_V18(self, page, *args, **kwargs)
            try:
                label = ""
                for a in reversed(args):
                    if isinstance(a, str):
                        label = a
                        break
                if not label:
                    label = str(self.tabText(int(res)))
                # Schedule only when this tab bar is relevant.
                if _is_lae_trainer_label(label) or _is_normal_trainer_label(label) or _looks_like_main_tab_widget(self):
                    schedule_for(self)
            except Exception:
                pass
            return res

        def insertTab_patched(self, index, page, *args, **kwargs):
            res = _ORIG_INSERT_V18(self, index, page, *args, **kwargs)
            try:
                label = ""
                for a in reversed(args):
                    if isinstance(a, str):
                        label = a
                        break
                if not label:
                    label = str(self.tabText(int(res)))
                if _is_lae_trainer_label(label) or _is_normal_trainer_label(label) or _looks_like_main_tab_widget(self):
                    schedule_for(self)
            except Exception:
                pass
            return res

        QTabWidget.addTab = addTab_patched
        QTabWidget.insertTab = insertTab_patched
        _PATCHED_QTAB_V18 = True
        _log("V18 anti-freeze insert hook installed")

    schedule_for(None)


def mustatil_plugin_init():
    _install_hook(None)


def register_plugin(app=None, main_window=None):
    _install_hook(main_window or app)
    return True
