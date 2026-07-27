#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mustatil plugin: YOLO Trainer use empty / negative images v3
===========================================================

Drop this file into Mustatil's mustatil_plugins folder and restart Mustatil.

This version is made for the current MustatilQtWorkspace layout where:
- the trainer tab is added as: self.tabs.addTab(page, "YOLO Trainer")
- training calls: y = self.prepare_yolo_dataset()
- then Ultralytics starts with: model.train(data=str(y), ...)

What it adds:
- A visible checkbox inside the existing "YOLO Trainer" panel:
    "Use empty/unlabeled images in YOLO training"
- A manual button:
    "Patch empty labels now"

What it does when the checkbox is enabled:
- calls the original prepare_yolo_dataset()
- then post-processes the generated YOLO dataset
- creates empty .txt label files for images without boxes
- optionally copies negative/empty source images from the Mustatil project into the training dataset
- rewrites data.yaml to explicit image-list manifests so the added empty images are included

It does not overwrite non-empty labels.
"""

from __future__ import annotations

import os
import re
import sys
import time
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PLUGIN_TITLE = "Mustatil YOLO Trainer Empty Images Hook v3"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
YAML_NAMES = ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml")
MANIFEST_PREFIX = "_mustatil_train_with_empty_"

_PATCHED_QTAB = False
_ORIG_ADD_TAB = None
_ORIG_INSERT_TAB = None
_INSTALLED_WORKSPACES = set()


def _log(msg: str) -> None:
    line = f"[{PLUGIN_TITLE}] {msg}"
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        base = Path(__file__).resolve().parent
        with (base / "mustatil_yolo_empty_images_hook_v3.log").open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _qt():
    try:
        from PySide6.QtWidgets import (
            QTabWidget, QCheckBox, QPushButton, QLabel, QGroupBox,
            QFormLayout, QVBoxLayout, QHBoxLayout, QWidget, QMessageBox
        )
        from PySide6.QtCore import Qt
        return {
            "QTabWidget": QTabWidget,
            "QCheckBox": QCheckBox,
            "QPushButton": QPushButton,
            "QLabel": QLabel,
            "QGroupBox": QGroupBox,
            "QFormLayout": QFormLayout,
            "QVBoxLayout": QVBoxLayout,
            "QHBoxLayout": QHBoxLayout,
            "QWidget": QWidget,
            "QMessageBox": QMessageBox,
            "Qt": Qt,
        }
    except Exception as exc:
        _log("PySide6 import failed: " + repr(exc))
        return None


def _var_get(obj: Any, default: str = "") -> str:
    try:
        if hasattr(obj, "get"):
            val = obj.get()
            if val is not None:
                return str(val)
    except Exception:
        pass
    try:
        if obj is not None:
            return str(obj)
    except Exception:
        pass
    return str(default or "")


def _var_set(obj: Any, value: Any) -> None:
    try:
        if hasattr(obj, "set"):
            obj.set(value)
            return
    except Exception:
        pass


def _workspace_from_widget(widget: Any) -> Optional[Any]:
    cur = widget
    for _ in range(100):
        if cur is None:
            return None
        try:
            if hasattr(cur, "tabs") and hasattr(cur, "project") and hasattr(cur, "prepare_yolo_dataset"):
                return cur
        except Exception:
            pass
        try:
            cur = cur.parentWidget()
        except Exception:
            try:
                cur = cur.parent()
            except Exception:
                return None
    return None


def _tab_title_from_add_args(args: Tuple[Any, ...]) -> str:
    if not args:
        return ""
    # QTabWidget.addTab(widget, "Title")
    # QTabWidget.addTab(widget, icon, "Title")
    try:
        return str(args[-1] or "")
    except Exception:
        return ""


def _is_yolo_trainer_title(title: str) -> bool:
    title = str(title or "").strip().lower()
    return title == "yolo trainer" or ("yolo" in title and "trainer" in title)


def _status(ws: Any, msg: str) -> None:
    _log(msg)
    try:
        if hasattr(ws, "log"):
            ws.log(msg)
    except Exception:
        pass
    try:
        sb = ws.statusBar()
        if sb is not None:
            sb.showMessage(msg, 9000)
    except Exception:
        pass
    try:
        if hasattr(ws, "tmsg"):
            ws.tmsg(msg)
    except Exception:
        pass


def _safe_rel_to(path: Path, root: Path) -> Optional[Path]:
    try:
        return path.resolve().relative_to(root.resolve())
    except Exception:
        return None


def _collect_images(root: Path, recursive: bool = True) -> List[Path]:
    if not root.exists():
        return []
    images: List[Path] = []
    try:
        iterator = root.rglob("*") if recursive else root.glob("*")
        for p in iterator:
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                low = str(p).lower()
                # Avoid accidental runtime/cache/system folders.
                if any(skip in low for skip in (
                    ".venv", "__pycache__", "site-packages", ".git",
                    "mustatil_model_runtimes", "onnx_models", "satellite_cache",
                    "runs/train_mustatil", "weights/best", "weights/last",
                    "temp", "tmp"
                )):
                    continue
                images.append(p.resolve())
    except Exception:
        pass
    return sorted(set(images), key=lambda x: str(x).lower())


def _split_from_path(p: Path) -> str:
    parts = [x.lower() for x in p.parts]
    if "test" in parts:
        return "test"
    if "val" in parts or "valid" in parts:
        return "val"
    return "train"


def _dataset_images_by_split(dataset_root: Path) -> Dict[str, List[Path]]:
    out = {"train": [], "val": [], "test": []}

    locations = [
        ("train", dataset_root / "images" / "train"),
        ("val", dataset_root / "images" / "val"),
        ("val", dataset_root / "images" / "valid"),
        ("test", dataset_root / "images" / "test"),
        ("train", dataset_root / "train" / "images"),
        ("val", dataset_root / "val" / "images"),
        ("val", dataset_root / "valid" / "images"),
        ("test", dataset_root / "test" / "images"),
    ]

    found = False
    for split, folder in locations:
        imgs = _collect_images(folder, recursive=True)
        if imgs:
            found = True
            out[split].extend(imgs)

    if not found:
        imgs = _collect_images(dataset_root / "images", recursive=True)
        for img in imgs:
            out[_split_from_path(img)].append(img)

    for k in list(out.keys()):
        out[k] = sorted(set(out[k]), key=lambda x: str(x).lower())
    return out


def _label_path_for_dataset_image(dataset_root: Path, image_path: Path) -> Path:
    parts = list(image_path.parts)
    low = [p.lower() for p in parts]

    if "images" in low:
        idx = low.index("images")
        new = parts[:]
        new[idx] = "labels"
        return Path(*new).with_suffix(".txt")

    for split in ("train", "val", "valid", "test"):
        if split in low:
            idx = low.index(split)
            if idx + 1 < len(low) and low[idx + 1] == "images":
                new = parts[:]
                new[idx + 1] = "labels"
                return Path(*new).with_suffix(".txt")

    rel = _safe_rel_to(image_path, dataset_root)
    if rel is None:
        rel = Path(image_path.name)
    return (dataset_root / "labels" / rel).with_suffix(".txt")


def _source_label_candidates(project_root: Path, image_path: Path) -> List[Path]:
    cands: List[Path] = []

    # Same folder .txt
    cands.append(image_path.with_suffix(".txt"))

    # project/images/foo.jpg -> project/labels/foo.txt
    rel = _safe_rel_to(image_path, project_root / "images")
    if rel is not None:
        cands.append((project_root / "labels" / rel).with_suffix(".txt"))

    # project/crops/foo.jpg -> project/labels/foo.txt and project/crop_labels/foo.txt
    rel = _safe_rel_to(image_path, project_root / "crops")
    if rel is not None:
        cands.append((project_root / "labels" / rel).with_suffix(".txt"))
        cands.append((project_root / "crop_labels" / rel).with_suffix(".txt"))

    # Generic images -> labels replacement.
    parts = list(image_path.parts)
    low = [x.lower() for x in parts]
    if "images" in low:
        i = low.index("images")
        new = parts[:]
        new[i] = "labels"
        cands.append(Path(*new).with_suffix(".txt"))
    if "crops" in low:
        i = low.index("crops")
        new = parts[:]
        new[i] = "labels"
        cands.append(Path(*new).with_suffix(".txt"))

    # Deduplicate.
    out = []
    seen = set()
    for p in cands:
        try:
            key = str(p.resolve()).lower()
        except Exception:
            key = str(p).lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _has_nonempty_label(label_path: Path) -> bool:
    try:
        return label_path.exists() and bool(label_path.read_text(encoding="utf-8", errors="ignore").strip())
    except Exception:
        return False


def _source_image_is_empty(project_root: Path, image_path: Path) -> bool:
    # Empty/negative means: no label file OR all matching label files are empty.
    found_any = False
    for cand in _source_label_candidates(project_root, image_path):
        try:
            if cand.exists():
                found_any = True
                if _has_nonempty_label(cand):
                    return False
        except Exception:
            pass
    return True if not found_any else True


def _unique_destination(dest_dir: Path, src: Path) -> Path:
    base = dest_dir / src.name
    if not base.exists():
        return base
    stem = src.stem
    suffix = src.suffix
    for i in range(1, 100000):
        cand = dest_dir / f"{stem}_empty_{i}{suffix}"
        if not cand.exists():
            return cand
    return dest_dir / f"{stem}_{int(time.time())}{suffix}"


def _existing_dataset_image_names(dataset_root: Path) -> set[str]:
    names = set()
    for imgs in _dataset_images_by_split(dataset_root).values():
        for img in imgs:
            names.add(img.name.lower())
    return names


def _copy_empty_source_images(ws: Any, dataset_root: Path, limit: int = 100000) -> int:
    project_raw = _var_get(getattr(ws, "project", None), "").strip()
    if not project_raw:
        return 0
    project_root = Path(project_raw).expanduser()
    if not project_root.exists():
        return 0

    source_dirs: List[Path] = []

    # Strong project folders first.
    for sub in ("images", "crops"):
        p = project_root / sub
        if p.exists():
            source_dirs.append(p)

    # Project-state folders if present.
    try:
        ps = getattr(ws, "project_state", None)
        for attr in ("image_folder", "crop_folder"):
            raw = str(getattr(ps, attr, "") or "").strip()
            if raw:
                p = Path(raw).expanduser()
                if p.exists():
                    source_dirs.append(p)
    except Exception:
        pass

    # Deduplicate dirs.
    unique_dirs = []
    seen_dirs = set()
    for d in source_dirs:
        try:
            key = str(d.resolve()).lower()
        except Exception:
            key = str(d).lower()
        if key not in seen_dirs:
            seen_dirs.add(key)
            unique_dirs.append(d)

    if not unique_dirs:
        return 0

    dest_img_dir = dataset_root / "images" / "train"
    dest_lab_dir = dataset_root / "labels" / "train"
    dest_img_dir.mkdir(parents=True, exist_ok=True)
    dest_lab_dir.mkdir(parents=True, exist_ok=True)

    existing_names = _existing_dataset_image_names(dataset_root)
    copied = 0

    for src_dir in unique_dirs:
        for img in _collect_images(src_dir, recursive=True):
            if copied >= limit:
                return copied

            # Add only negative/empty source images. Annotated source images are already handled by normal preparation.
            if not _source_image_is_empty(project_root, img):
                continue

            # Avoid copying the same source file by simple filename when already present in dataset.
            # This keeps the patch from duplicating existing negative images on repeated runs.
            if img.name.lower() in existing_names:
                # Still make sure the dataset label exists if the image is already there.
                try:
                    for ds_img in _dataset_images_by_split(dataset_root).get("train", []):
                        if ds_img.name.lower() == img.name.lower():
                            lab = _label_path_for_dataset_image(dataset_root, ds_img)
                            lab.parent.mkdir(parents=True, exist_ok=True)
                            if not lab.exists():
                                lab.write_text("", encoding="utf-8")
                except Exception:
                    pass
                continue

            dst = _unique_destination(dest_img_dir, img)
            try:
                shutil.copy2(img, dst)
                (dest_lab_dir / dst.with_suffix(".txt").name).write_text("", encoding="utf-8")
                existing_names.add(dst.name.lower())
                copied += 1
            except Exception as exc:
                _log(f"Could not copy empty source image {img}: {exc}")

    return copied


def _write_manifest(dataset_root: Path, split: str, images: List[Path]) -> Optional[Path]:
    if not images:
        return None
    p = dataset_root / f"{MANIFEST_PREFIX}{split}.txt"
    try:
        txt = "\n".join(str(x.resolve()).replace("\\", "/") for x in images) + "\n"
        p.write_text(txt, encoding="utf-8")
        return p
    except Exception as exc:
        _log(f"Could not write manifest {p}: {exc}")
        return None


def _patch_yaml(yaml_path: Path, manifests: Dict[str, Path]) -> bool:
    if not yaml_path.exists():
        return False
    try:
        old = yaml_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False

    new = old

    def repl(text: str, key: str, value: Path) -> str:
        val = str(value.resolve()).replace("\\", "/")
        pat = re.compile(rf"^(\s*{re.escape(key)}\s*:\s*).*$", re.MULTILINE)
        if pat.search(text):
            return pat.sub(rf"\1{val}", text)
        return f"{key}: {val}\n" + text

    if "train" in manifests:
        new = repl(new, "train", manifests["train"])
    if "val" in manifests:
        new = repl(new, "val", manifests["val"])
    if "test" in manifests:
        new = repl(new, "test", manifests["test"])

    if new == old:
        return True

    try:
        bak = yaml_path.with_suffix(yaml_path.suffix + ".before_empty_images_hook_v3.bak")
        if not bak.exists():
            shutil.copy2(yaml_path, bak)
        yaml_path.write_text(new, encoding="utf-8")
        return True
    except Exception as exc:
        _log(f"Could not patch YAML {yaml_path}: {exc}")
        return False


def _find_yaml_from_result(ws: Any, result: Any) -> Optional[Path]:
    # prepare_yolo_dataset normally returns data.yaml.
    try:
        if result:
            p = Path(str(result)).expanduser()
            if p.is_file() and p.suffix.lower() in (".yaml", ".yml"):
                return p.resolve()
            if p.is_dir():
                for name in YAML_NAMES:
                    q = p / name
                    if q.exists():
                        return q.resolve()
    except Exception:
        pass

    # Fallback scan inside project root.
    project_raw = _var_get(getattr(ws, "project", None), "").strip()
    if project_raw:
        root = Path(project_raw).expanduser()
        likely = [
            root / "data.yaml",
            root / "dataset.yaml",
            root / "yolo_dataset" / "data.yaml",
            root / "dataset" / "data.yaml",
            root / "training" / "data.yaml",
        ]
        for p in likely:
            if p.exists():
                return p.resolve()
        try:
            yamls = []
            for pat in ("*.yaml", "*.yml"):
                yamls.extend(root.rglob(pat))
            yamls = [p for p in yamls if p.name.lower() in YAML_NAMES]
            if yamls:
                yamls.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return yamls[0].resolve()
        except Exception:
            pass
    return None


def _postprocess_yolo_dataset(ws: Any, yaml_result: Any = None, manual: bool = False) -> Optional[Path]:
    enabled = bool(getattr(ws, "_mustatil_yolo_use_empty_images", False))
    if not enabled and not manual:
        return None

    yaml_path = _find_yaml_from_result(ws, yaml_result)
    if yaml_path is None:
        _status(ws, "Empty-image patch: data.yaml was not found. First select a Project and prepare the YOLO dataset.")
        return None

    dataset_root = yaml_path.parent
    if not dataset_root.exists():
        _status(ws, f"Empty-image patch: dataset folder not found: {dataset_root}")
        return None

    copied = _copy_empty_source_images(ws, dataset_root)

    split_imgs = _dataset_images_by_split(dataset_root)
    all_imgs = sorted(set(img for imgs in split_imgs.values() for img in imgs), key=lambda x: str(x).lower())

    created_labels = 0
    existing_nonempty = 0
    existing_empty = 0

    for img in all_imgs:
        lab = _label_path_for_dataset_image(dataset_root, img)
        try:
            lab.parent.mkdir(parents=True, exist_ok=True)
            if lab.exists():
                if _has_nonempty_label(lab):
                    existing_nonempty += 1
                else:
                    existing_empty += 1
            else:
                lab.write_text("", encoding="utf-8")
                created_labels += 1
                existing_empty += 1
        except Exception as exc:
            _log(f"Could not ensure empty label for {img}: {exc}")

    # Refresh after possible copy.
    split_imgs = _dataset_images_by_split(dataset_root)
    manifests: Dict[str, Path] = {}
    for split, imgs in split_imgs.items():
        if imgs:
            mf = _write_manifest(dataset_root, split, imgs)
            if mf is not None:
                manifests[split] = mf

    yaml_ok = _patch_yaml(yaml_path, manifests)

    total = sum(len(v) for v in split_imgs.values())
    _status(
        ws,
        "Empty-image YOLO training active: "
        f"{total} image(s), {copied} empty source image(s) copied, "
        f"{created_labels} empty label file(s) created, "
        f"{existing_nonempty} non-empty label file(s), data.yaml patched={yaml_ok}."
    )
    return yaml_path


def _patch_workspace_methods(ws: Any) -> None:
    if id(ws) in _INSTALLED_WORKSPACES:
        return
    _INSTALLED_WORKSPACES.add(id(ws))

    # Store setting on workspace.
    if not hasattr(ws, "_mustatil_yolo_use_empty_images"):
        ws._mustatil_yolo_use_empty_images = False

    # Patch prepare_yolo_dataset: train() calls this dynamically, so training will be covered.
    try:
        original_prepare = ws.prepare_yolo_dataset

        def prepare_wrapper(*args, **kwargs):
            y = original_prepare(*args, **kwargs)
            try:
                _postprocess_yolo_dataset(ws, y, manual=False)
            except Exception:
                _log("postprocess after prepare failed:\n" + traceback.format_exc())
            return y

        prepare_wrapper._mustatil_empty_images_wrapped_v3 = True
        ws.prepare_yolo_dataset = prepare_wrapper
        _log("Workspace prepare_yolo_dataset patched.")
    except Exception as exc:
        _log("Could not patch prepare_yolo_dataset: " + repr(exc))

    # Patch run_task so the separately captured "Prepare YOLO Dataset" button is also covered.
    try:
        original_run_task = ws.run_task

        def run_task_wrapper(name: str, func, allow_parallel: bool = False):
            def wrapped_func():
                result = func()
                try:
                    fname = str(getattr(func, "__name__", "") or "")
                    n = str(name or "").lower()
                    if "prepare" in fname.lower() or "prepare" in n or "train" in fname.lower() or "train" in n:
                        _postprocess_yolo_dataset(ws, result, manual=False)
                except Exception:
                    _log("postprocess from run_task failed:\n" + traceback.format_exc())
                return result
            return original_run_task(name, wrapped_func, allow_parallel=allow_parallel)

        run_task_wrapper._mustatil_empty_images_run_task_wrapped_v3 = True
        ws.run_task = run_task_wrapper
        _log("Workspace run_task patched.")
    except Exception as exc:
        _log("Could not patch run_task: " + repr(exc))


def _find_yolo_trainer_group(page: Any) -> Optional[Any]:
    q = _qt()
    if q is None:
        return None
    QGroupBox = q["QGroupBox"]
    try:
        groups = page.findChildren(QGroupBox)
    except Exception:
        groups = []
    for g in groups:
        try:
            title = str(g.title() or "").strip().lower()
            if title == "yolo trainer":
                return g
        except Exception:
            pass
    return groups[0] if groups else None


def _install_ui(tab_widget: Any, page: Any) -> None:
    q = _qt()
    if q is None:
        return

    QCheckBox = q["QCheckBox"]
    QPushButton = q["QPushButton"]
    QLabel = q["QLabel"]
    QFormLayout = q["QFormLayout"]
    QHBoxLayout = q["QHBoxLayout"]
    QMessageBox = q["QMessageBox"]

    try:
        if getattr(page, "_mustatil_empty_images_ui_installed_v3", False):
            return
        page._mustatil_empty_images_ui_installed_v3 = True
    except Exception:
        pass

    ws = _workspace_from_widget(tab_widget) or _workspace_from_widget(page)
    if ws is None:
        _log("YOLO Trainer page found, but workspace object was not found.")
        return

    _patch_workspace_methods(ws)

    try:
        trainer_group = _find_yolo_trainer_group(page)
        if trainer_group is None:
            _log("YOLO Trainer group not found on page.")
            return

        layout = trainer_group.layout()
        if not isinstance(layout, QFormLayout):
            _log("YOLO Trainer group layout is not QFormLayout.")
            return

        cb = QCheckBox("Use empty/unlabeled images in YOLO training")
        cb.setObjectName("mustatil_yolo_use_empty_images_checkbox_v3")
        cb.setToolTip(
            "If enabled, Mustatil post-processes the generated YOLO dataset before training: "
            "missing label files are created as empty .txt files and empty source images from the project "
            "can be copied into the training dataset."
        )
        cb.setChecked(bool(getattr(ws, "_mustatil_yolo_use_empty_images", False)))

        def on_toggle(state: bool):
            ws._mustatil_yolo_use_empty_images = bool(state)
            try:
                ws.train_keep_negative_chunks.set(bool(state))
            except Exception:
                pass
            _status(ws, f"Use empty/unlabeled YOLO images: {'ON' if bool(state) else 'OFF'}")

        cb.toggled.connect(on_toggle)

        patch_btn = QPushButton("Patch empty labels now")
        patch_btn.setObjectName("mustatil_yolo_patch_empty_labels_now_v3")
        patch_btn.setToolTip("Runs prepare_yolo_dataset() and then adds empty-label images into the generated YOLO training dataset.")

        def do_manual_patch():
            try:
                # Use the original workflow first if possible, then postprocess.
                result = None
                try:
                    result = ws.prepare_yolo_dataset()
                except Exception as exc:
                    _status(ws, "Prepare YOLO Dataset failed before empty-image patch: " + str(exc))
                    raise
                ws._mustatil_yolo_use_empty_images = True
                cb.setChecked(True)
                yaml_path = _postprocess_yolo_dataset(ws, result, manual=True)
                try:
                    QMessageBox.information(ws, "YOLO empty images", f"Finished.\n\nDataset YAML:\n{yaml_path}")
                except Exception:
                    pass
            except Exception as exc:
                _log("Manual patch failed:\n" + traceback.format_exc())
                try:
                    QMessageBox.critical(ws, "YOLO empty images", str(exc))
                except Exception:
                    pass

        patch_btn.clicked.connect(do_manual_patch)

        row_widget = q["QWidget"]()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(cb)
        row.addWidget(patch_btn)
        row.addStretch(1)

        hint = QLabel("Keeps negative images by creating empty YOLO .txt labels before training.")
        hint.setWordWrap(True)

        layout.addRow("", row_widget)
        layout.addRow("", hint)

        # Keep handles on workspace so Qt does not garbage-collect anything unexpectedly.
        ws._mustatil_yolo_empty_images_checkbox_v3 = cb
        ws._mustatil_yolo_empty_images_patch_button_v3 = patch_btn

        _status(ws, "YOLO Trainer empty-image checkbox inserted.")
    except Exception:
        _log("UI install failed:\n" + traceback.format_exc())


def _patch_qtabwidget() -> None:
    global _PATCHED_QTAB, _ORIG_ADD_TAB, _ORIG_INSERT_TAB
    if _PATCHED_QTAB:
        return
    q = _qt()
    if q is None:
        return

    QTabWidget = q["QTabWidget"]
    _ORIG_ADD_TAB = QTabWidget.addTab
    _ORIG_INSERT_TAB = QTabWidget.insertTab

    def addTab_patched(self, page, *args, **kwargs):
        title = _tab_title_from_add_args(args)
        idx = _ORIG_ADD_TAB(self, page, *args, **kwargs)
        try:
            if _is_yolo_trainer_title(title):
                _install_ui(self, page)
        except Exception:
            _log("addTab hook failed:\n" + traceback.format_exc())
        return idx

    def insertTab_patched(self, index, page, *args, **kwargs):
        title = _tab_title_from_add_args(args)
        idx = _ORIG_INSERT_TAB(self, index, page, *args, **kwargs)
        try:
            if _is_yolo_trainer_title(title):
                _install_ui(self, page)
        except Exception:
            _log("insertTab hook failed:\n" + traceback.format_exc())
        return idx

    QTabWidget.addTab = addTab_patched
    QTabWidget.insertTab = insertTab_patched
    _PATCHED_QTAB = True
    _log("QTabWidget.addTab/insertTab hooked.")


def mustatil_plugin_init():
    _patch_qtabwidget()
    _log("Plugin init complete.")


# Auto-init, because Mustatil's plugin loader imports files before the main window is created.
try:
    mustatil_plugin_init()
except Exception:
    _log("Auto init failed:\n" + traceback.format_exc())
