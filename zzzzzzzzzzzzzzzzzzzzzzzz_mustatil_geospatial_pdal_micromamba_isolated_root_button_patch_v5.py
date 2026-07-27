# -*- coding: utf-8 -*-
"""
Modified for Mustatil by Tarek Wasfy, 2026.

Mustatil Annotator X-AnyLabeling PyPI sync-button patch v9

This plugin does NOT add a separate X-AnyLabeling tab.  Instead it patches the
existing Mustatil Annotator/Annotation page and inserts a compact toolbar at the
top:

    Install/Start X-AnyLabeling    Sync back to Mustatil labels

Workflow
--------
1. The user selects/opens the normal Mustatil project in the existing Annotator.
2. The toolbar auto-detects project/data.yaml, images/, labels/ and classes.
3. Before starting X-AnyLabeling, existing YOLO/Mustatil .txt labels are converted
   to X-AnyLabeling/LabelMe-style JSON so the labels are visible in X-AnyLabeling.
4. X-AnyLabeling is installed from PyPI into a local Mustatil runtime and started
   as an external Qt process.
5. When X-AnyLabeling closes, JSON annotations are automatically synchronized
   back to the existing YOLO labels/ folder and the Annotator page is refreshed
   where possible.

Third-party source notice
-------------------------
X-AnyLabeling / CVHub520
Project URL: https://github.com/CVHub520/X-AnyLabeling
PyPI package: x-anylabeling-cvhub
License notice: GPL-3.0; see LICENSES/GPL-3.0.txt and THIRD_PARTY_NOTICES.md

Architecture note
-----------------
Mustatil uses PySide6 while X-AnyLabeling is a Qt application.  To preserve all
original X-AnyLabeling functions and avoid unsafe Qt binding mixing, this plugin
launches X-AnyLabeling as its own process instead of importing it into Mustatil.
"""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PLUGIN_NAME = "Mustatil Annotator X-AnyLabeling PyPI Sync Button Patch v9"
_RUNTIME_DIR_NAME = "X-AnyLabeling-PyPI"
DEFAULT_PYPI_PACKAGE = "x-anylabeling-cvhub[cpu]"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DATASET_YAML_NAMES = ["data.yaml", "dataset.yaml", "data.yml", "dataset.yml"]
_PATCHED_QTAB = False
_PATCHED_PAGES: set[int] = set()
_SCAN_TIMER = None
_ORIG_ADD_TAB = None
_ORIG_INSERT_TAB = None


def _log(msg: str) -> None:
    try:
        print(f"[{PLUGIN_NAME}] {msg}", flush=True)
    except Exception:
        pass


def _is_windows() -> bool:
    return os.name == "nt" or sys.platform.startswith("win")


def _plugin_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except Exception:
        return Path.cwd()


def _mustatil_root() -> Path:
    p = _plugin_dir()
    if p.name.lower() == "mustatil_plugins":
        return p.parent
    return p


def _runtime_root() -> Path:
    return _mustatil_root() / "mustatil_model_runtimes" / _RUNTIME_DIR_NAME


def _work_dir() -> Path:
    return _runtime_root() / "work"


def _project_cache_dir() -> Path:
    return _runtime_root() / "project_import"


def _venv_dir() -> Path:
    return _runtime_root() / ".venv"


def _venv_python() -> Path:
    if _is_windows():
        return _venv_dir() / "Scripts" / "python.exe"
    return _venv_dir() / "bin" / "python"


def _venv_script(name: str) -> Path:
    if _is_windows():
        exe = name + ".exe" if not name.lower().endswith((".exe", ".bat", ".cmd")) else name
        return _venv_dir() / "Scripts" / exe
    return _venv_dir() / "bin" / name


def _runtime_env() -> Dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["QT_API"] = "pyqt5"
    env["MUSTATIL_X_ANYLABELING_RUNTIME"] = str(_runtime_root())
    env["MUSTATIL_X_ANYLABELING_NOTICE"] = "Modified for Mustatil by Tarek Wasfy, 2026. Source: X-AnyLabeling / CVHub520, GPL-3.0."
    env.setdefault("XDG_CACHE_HOME", str(_runtime_root() / "cache"))
    env.setdefault("HF_HOME", str(_runtime_root() / "cache" / "huggingface"))
    if _venv_python().exists():
        scripts = _venv_python().parent
        env["PATH"] = str(scripts) + os.pathsep + env.get("PATH", "")
        env["VIRTUAL_ENV"] = str(_venv_dir())
    return env


def _quote_cmd(s: Any) -> str:
    text = str(s or "")
    if _is_windows():
        return '"' + text.replace('"', '""') + '"'
    import shlex
    return shlex.quote(text)


def _write_runtime_notices() -> None:
    rt = _runtime_root()
    (rt / "LICENSES").mkdir(parents=True, exist_ok=True)
    _work_dir().mkdir(parents=True, exist_ok=True)
    _project_cache_dir().mkdir(parents=True, exist_ok=True)
    (rt / "THIRD_PARTY_NOTICES.md").write_text(
        "# Third-party notices for Mustatil X-AnyLabeling PyPI integration\n\n"
        "This Mustatil plugin downloads and installs X-AnyLabeling from PyPI at runtime.\n\n"
        "- Source project: X-AnyLabeling / CVHub520\n"
        "- Upstream URL: https://github.com/CVHub520/X-AnyLabeling\n"
        "- PyPI package: x-anylabeling-cvhub\n"
        "- License notice: GPL-3.0. See `LICENSES/GPL-3.0.txt`.\n\n"
        "This plugin does not bundle X-AnyLabeling source code. It starts the original X-AnyLabeling app as an external process and synchronizes YOLO labels before/after editing.\n\n"
        "Modified for Mustatil by Tarek Wasfy, 2026.\n",
        encoding="utf-8",
    )
    (rt / "ABOUT_X_ANYLABELING_MUSTATIL.txt").write_text(
        "X-AnyLabeling Annotator bridge for Mustatil\n"
        "===========================================\n\n"
        "This integration installs X-AnyLabeling from PyPI into a local Mustatil runtime.\n"
        "It adds buttons to Mustatil's existing Annotator page, converts existing YOLO labels to X-AnyLabeling JSON before editing, and synchronizes JSON annotations back to YOLO labels after X-AnyLabeling closes.\n\n"
        "Source: https://github.com/CVHub520/X-AnyLabeling\n"
        "PyPI: x-anylabeling-cvhub\n"
        "License: GPL-3.0\n\n"
        "Modified for Mustatil by Tarek Wasfy, 2026.\n",
        encoding="utf-8",
    )
    try:
        src = _plugin_dir() / "LICENSES" / "GPL-3.0.txt"
        dst = rt / "LICENSES" / "GPL-3.0.txt"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    except Exception:
        pass


try:
    from PySide6.QtCore import Qt, QProcess, QTimer
    from PySide6.QtGui import QImage, QTextCursor
    from PySide6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QPlainTextEdit, QFileDialog, QTabWidget, QMessageBox, QCheckBox,
        QGroupBox, QFrame, QSizePolicy, QScrollArea, QLineEdit, QComboBox,
    )
except Exception:  # keep importable for py_compile outside Mustatil
    Qt = QProcess = QTimer = None  # type: ignore
    QImage = QTextCursor = None  # type: ignore
    QApplication = None  # type: ignore
    QWidget = object  # type: ignore
    QVBoxLayout = QHBoxLayout = QLabel = QPushButton = QPlainTextEdit = QFileDialog = None  # type: ignore
    QTabWidget = QMessageBox = QCheckBox = QGroupBox = QFrame = QSizePolicy = QScrollArea = QLineEdit = QComboBox = None  # type: ignore


# -----------------------------------------------------------------------------
# Dataset/project discovery
# -----------------------------------------------------------------------------

def _read_yaml_like(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass
    data: Dict[str, Any] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        i += 1
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip(); val = val.strip()
        if key == "names":
            if val:
                try:
                    data[key] = ast.literal_eval(val)
                except Exception:
                    data[key] = [x.strip().strip("'\"") for x in val.strip("[]").split(",") if x.strip()]
            else:
                names: Dict[int, str] = {}
                while i < len(lines):
                    nxt = lines[i]
                    if not nxt.startswith((" ", "\t", "-")):
                        break
                    s = nxt.strip(); i += 1
                    if not s or s.startswith("#"):
                        continue
                    if s.startswith("-"):
                        names[len(names)] = s[1:].strip().strip("'\"")
                    elif ":" in s:
                        k, v = s.split(":", 1)
                        try: idx = int(k.strip().strip("'\""))
                        except Exception: idx = len(names)
                        names[idx] = v.strip().strip("'\"")
                data[key] = names
        elif val:
            data[key] = val.strip("'\"")
    return data


def _classes_from_names(names: Any) -> List[str]:
    if isinstance(names, dict):
        items: List[Tuple[int, str]] = []
        for k, v in names.items():
            try: idx = int(k)
            except Exception: idx = len(items)
            items.append((idx, str(v)))
        return [v for _, v in sorted(items)]
    if isinstance(names, list):
        return [str(x) for x in names]
    if isinstance(names, str):
        s = names.strip()
        if s.startswith("[") or s.startswith("{"):
            try: return _classes_from_names(ast.literal_eval(s))
            except Exception: pass
        return [x.strip() for x in re.split(r"[,;]", s) if x.strip()]
    return []


def _make_abs_path(base: Path, p: Any) -> Path:
    if isinstance(p, (list, tuple)) and p:
        p = p[0]
    text = str(p or "").strip().strip("'\"")
    if not text:
        return base
    q = Path(text).expanduser()
    if not q.is_absolute():
        q = (base / q).resolve()
    if q.suffix.lower() == ".txt":
        try:
            first = q.read_text(encoding="utf-8", errors="replace").splitlines()[0].strip()
            if first:
                fp = Path(first)
                if not fp.is_absolute(): fp = (base / fp).resolve()
                return fp.parent
        except Exception:
            pass
        return q.parent
    return q


def _folder_has_images(folder: Path) -> bool:
    try:
        if not folder.exists() or not folder.is_dir(): return False
        for p in folder.iterdir():
            if p.suffix.lower() in IMAGE_EXTS: return True
    except Exception:
        pass
    return False


def _find_dataset_yaml(folder: Path) -> Optional[Path]:
    for name in DATASET_YAML_NAMES:
        p = folder / name
        if p.exists(): return p
    try:
        for p in folder.glob("*.yaml"):
            try:
                txt = p.read_text(encoding="utf-8", errors="replace")[:2500].lower()
                if "names:" in txt and ("train:" in txt or "val:" in txt or "path:" in txt):
                    return p
            except Exception:
                pass
    except Exception:
        pass
    return None


def _labels_from_images_dir(images_dir: Path) -> Path:
    s = str(images_dir)
    variants: List[Path] = []
    for token in [os.sep + "images" + os.sep, "/images/", "\\images\\"]:
        if token in s:
            variants.append(Path(s.replace(token, token.replace("images", "labels"))))
    parts = list(images_dir.parts)
    for i, part in enumerate(parts):
        if part.lower() == "images":
            p2 = Path(*parts[:i], "labels", *parts[i+1:]) if i > 0 else Path("labels", *parts[i+1:])
            variants.append(p2)
    variants += [images_dir.parent / "labels", images_dir.parent.parent / "labels" / images_dir.name, images_dir.parent / "labels" / images_dir.name]
    for p in variants:
        try:
            if p.exists() and p.is_dir(): return p.resolve()
        except Exception:
            pass
    return variants[0] if variants else images_dir.parent / "labels"


def _find_images_dir(project: Path) -> Path:
    candidates = [project / "images" / "train", project / "images" / "val", project / "images", project / "train" / "images", project / "valid" / "images", project / "val" / "images", project / "test" / "images", project]
    for p in candidates:
        if _folder_has_images(p): return p.resolve()
    try:
        for p in project.rglob("images"):
            if _folder_has_images(p): return p.resolve()
    except Exception:
        pass
    return project / "images"


def _find_labels_dir(project: Path, images_dir: Optional[Path] = None) -> Path:
    if images_dir:
        p = _labels_from_images_dir(images_dir)
        if p.exists(): return p.resolve()
    candidates = [project / "labels" / "train", project / "labels" / "val", project / "labels", project / "train" / "labels", project / "valid" / "labels", project / "val" / "labels", project / "test" / "labels"]
    for p in candidates:
        try:
            if p.exists() and p.is_dir(): return p.resolve()
        except Exception:
            pass
    return project / "labels"


def _classes_from_project_files(project: Path, yaml_path: Optional[Path] = None) -> List[str]:
    if yaml_path and yaml_path.exists():
        data = _read_yaml_like(yaml_path)
        classes = _classes_from_names(data.get("names"))
        if classes: return classes
    for name in ["classes.txt", "obj.names", "labels.txt", "classes.names", "names.txt"]:
        for p in [project / name, project.parent / name, project / "labels" / name]:
            try:
                if p.exists():
                    return [x.strip() for x in p.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip() and not x.strip().startswith("#")]
            except Exception:
                pass
    return []


def _infer_classes_from_label_ids(labels_dir: Path) -> List[str]:
    max_id = -1
    try:
        for txt in labels_dir.rglob("*.txt"):
            try:
                for line in txt.read_text(encoding="utf-8", errors="replace").splitlines():
                    parts = line.strip().split()
                    if not parts: continue
                    try: max_id = max(max_id, int(float(parts[0])))
                    except Exception: pass
            except Exception:
                pass
    except Exception:
        pass
    return [f"class_{i}" for i in range(max_id + 1)] if max_id >= 0 else []


def _detect_project_from_path(raw_text: str) -> Dict[str, Any]:
    raw = Path(str(raw_text or "").strip().strip('"')).expanduser()
    result: Dict[str, Any] = {"project": str(raw), "yaml": "", "images": "", "labels": "", "classes": []}
    if not raw.exists(): return result
    yaml_path: Optional[Path] = None
    project_dir = raw
    if raw.is_file() and raw.suffix.lower() in {".yaml", ".yml"}:
        yaml_path = raw.resolve(); project_dir = raw.parent.resolve()
    elif raw.is_file():
        project_dir = raw.parent.resolve(); yaml_path = _find_dataset_yaml(project_dir)
    else:
        project_dir = raw.resolve(); yaml_path = _find_dataset_yaml(project_dir)

    images_dir: Optional[Path] = None
    classes: List[str] = []
    if yaml_path:
        result["yaml"] = str(yaml_path)
        data = _read_yaml_like(yaml_path)
        base = yaml_path.parent
        if data.get("path"): base = _make_abs_path(yaml_path.parent, data.get("path"))
        for key in ["train", "val", "valid", "test"]:
            if data.get(key):
                cand = _make_abs_path(base, data.get(key))
                if cand.exists() and cand.is_file(): cand = cand.parent
                if cand.exists() and cand.is_dir():
                    images_dir = cand; break
        classes = _classes_from_names(data.get("names"))
    if not images_dir:
        # If raw is already an images folder, keep it.
        if raw.is_dir() and _folder_has_images(raw): images_dir = raw.resolve()
        else: images_dir = _find_images_dir(project_dir)
    labels_dir = _find_labels_dir(project_dir, images_dir)
    if not classes: classes = _classes_from_project_files(project_dir, yaml_path)
    if not classes: classes = _infer_classes_from_label_ids(labels_dir)
    result.update({"project": str(project_dir), "images": str(images_dir), "labels": str(labels_dir), "classes": classes})
    return result


def _xlabels_dir_for_project(project: Path) -> Path:
    p = project / "x_anylabeling_json"
    try: p.mkdir(parents=True, exist_ok=True)
    except Exception: pass
    return p


def _write_classes_file(classes: List[str], project: Path) -> Path:
    if project.is_file(): project = project.parent
    project.mkdir(parents=True, exist_ok=True)
    p = project / "classes.txt"
    clean = [str(x).strip() for x in classes if str(x).strip()]
    p.write_text("\n".join(clean) + ("\n" if clean else ""), encoding="utf-8")
    _project_cache_dir().mkdir(parents=True, exist_ok=True)
    cache = _project_cache_dir() / (re.sub(r"[^A-Za-z0-9_.-]+", "_", project.name or "project") + "_classes.txt")
    cache.write_text(p.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    return p


def _candidate_values_from_object(obj: Any) -> List[Any]:
    vals: List[Any] = []
    if obj is None: return vals
    names = [
        "project_path", "project_dir", "project_folder", "current_project", "project_file", "dataset_yaml", "data_yaml", "yaml_path",
        "images_dir", "image_dir", "image_folder", "labels_dir", "label_dir", "label_folder", "dataset_dir", "dataset_path",
    ]
    for name in names:
        try:
            v = getattr(obj, name, None)
            if callable(v):
                try: v = v()
                except Exception: continue
            if v: vals.append(v)
        except Exception:
            pass
    try:
        for name in dir(obj):
            lname = name.lower()
            if any(k in lname for k in ["project", "dataset", "data_yaml", "image_dir", "images_dir", "label_dir", "labels_dir"]):
                try:
                    v = getattr(obj, name)
                    if callable(v): continue
                    if isinstance(v, (str, Path)) and str(v): vals.append(v)
                except Exception:
                    pass
    except Exception:
        pass
    return vals


def _candidate_values_from_widgets(page: Any) -> List[str]:
    vals: List[str] = []
    if QLineEdit is None or page is None: return vals
    try:
        for le in page.findChildren(QLineEdit):
            try:
                t = le.text().strip().strip('"')
                if t and (":" in t or "/" in t or "\\" in t or t.endswith((".yaml", ".yml"))): vals.append(t)
            except Exception:
                pass
    except Exception:
        pass
    try:
        for combo in page.findChildren(QComboBox):
            try:
                t = combo.currentText().strip().strip('"')
                if t and (":" in t or "/" in t or "\\" in t): vals.append(t)
            except Exception:
                pass
    except Exception:
        pass
    return vals


def _find_workspace_from_page(page: Any) -> Optional[Any]:
    try:
        p = page
        for _ in range(30):
            if p is None: break
            if hasattr(p, "tabs") or "workspace" in p.__class__.__name__.lower() or "mustatil" in p.__class__.__name__.lower():
                return p
            p = p.parent()
    except Exception:
        pass
    try:
        app = QApplication.instance() if QApplication is not None else None
        if app:
            for w in app.allWidgets():
                try:
                    if hasattr(w, "tabs") or "workspace" in w.__class__.__name__.lower(): return w
                except Exception:
                    pass
    except Exception:
        pass
    return None


def _project_score(info: Dict[str, Any]) -> int:
    score = 0
    try:
        if info.get("yaml"): score += 40
        if info.get("images") and Path(info["images"]).exists(): score += 40
        if info.get("labels") and Path(info["labels"]).exists(): score += 40
        if info.get("classes"): score += 20
    except Exception:
        pass
    return score


def _detect_project_from_context(page: Any, workspace: Any = None) -> Dict[str, Any]:
    candidates: List[Any] = []
    candidates += _candidate_values_from_object(page)
    candidates += _candidate_values_from_object(workspace)
    candidates += _candidate_values_from_widgets(page)
    # Also test parent folders of image/label paths.
    expanded: List[str] = []
    for c in candidates:
        try:
            p = Path(str(c).strip().strip('"')).expanduser()
            expanded.append(str(p))
            if p.exists():
                if p.is_file(): expanded.append(str(p.parent))
                elif p.name.lower() in {"images", "labels", "train", "val", "valid", "test"}: expanded.append(str(p.parent))
                if p.parent.exists(): expanded.append(str(p.parent))
                if p.parent.parent.exists(): expanded.append(str(p.parent.parent))
        except Exception:
            pass
    best: Dict[str, Any] = {}
    best_score = -1
    seen = set()
    for raw in expanded:
        if not raw or raw in seen: continue
        seen.add(raw)
        try:
            info = _detect_project_from_path(raw)
            score = _project_score(info)
            if score > best_score:
                best_score = score; best = info
        except Exception:
            pass
    return best if best_score > 0 else {}


# -----------------------------------------------------------------------------
# YOLO <-> X-AnyLabeling JSON synchronization
# -----------------------------------------------------------------------------

def _image_size(path: Path) -> Tuple[int, int]:
    try:
        if QImage is not None:
            img = QImage(str(path))
            if not img.isNull(): return int(img.width()), int(img.height())
    except Exception:
        pass
    try:
        from PIL import Image  # type: ignore
        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        return (0, 0)


def _image_index(images_dir: Path) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    try:
        for p in images_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                idx.setdefault(p.stem, p.resolve())
    except Exception:
        pass
    return idx


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _yolo_to_xjson(images_dir: Path, labels_dir: Path, classes: List[str], out_dir: Path) -> Tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    image_map = _image_index(images_dir)
    converted = 0; skipped = 0
    for txt in labels_dir.rglob("*.txt") if labels_dir.exists() else []:
        try:
            img = image_map.get(txt.stem)
            if not img:
                skipped += 1; continue
            w, h = _image_size(img)
            if w <= 0 or h <= 0:
                skipped += 1; continue
            shapes: List[Dict[str, Any]] = []
            for line in txt.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.strip().split()
                if len(parts) < 5: continue
                try: cid = int(float(parts[0]))
                except Exception: continue
                label = classes[cid] if 0 <= cid < len(classes) else f"class_{cid}"
                vals = [float(x) for x in parts[1:]]
                if len(vals) == 4:
                    xc, yc, bw, bh = vals
                    x1 = (xc - bw / 2.0) * w; y1 = (yc - bh / 2.0) * h
                    x2 = (xc + bw / 2.0) * w; y2 = (yc + bh / 2.0) * h
                    shapes.append({
                        "label": label,
                        "points": [[round(x1, 3), round(y1, 3)], [round(x2, 3), round(y2, 3)]],
                        "group_id": None,
                        "description": "",
                        "shape_type": "rectangle",
                        "flags": {},
                    })
                elif len(vals) >= 6 and len(vals) % 2 == 0:
                    pts = []
                    for i in range(0, len(vals), 2):
                        pts.append([round(vals[i] * w, 3), round(vals[i+1] * h, 3)])
                    shapes.append({
                        "label": label,
                        "points": pts,
                        "group_id": None,
                        "description": "",
                        "shape_type": "polygon",
                        "flags": {},
                    })
            data = {
                "version": "x-anylabeling-mustatil-sync-v3",
                "flags": {},
                "shapes": shapes,
                "imagePath": img.name,
                "imageData": None,
                "imageHeight": h,
                "imageWidth": w,
            }
            (out_dir / f"{txt.stem}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            converted += 1
        except Exception:
            skipped += 1
    return converted, skipped


def _points_bbox(points: Sequence[Any]) -> Optional[Tuple[float, float, float, float]]:
    xs: List[float] = []; ys: List[float] = []
    for p in points:
        try:
            xs.append(float(p[0])); ys.append(float(p[1]))
        except Exception:
            pass
    if not xs or not ys: return None
    return min(xs), min(ys), max(xs), max(ys)


def _xjson_to_yolo(xjson_dir: Path, images_dir: Path, labels_dir: Path, classes: List[str]) -> Tuple[int, int, List[str]]:
    labels_dir.mkdir(parents=True, exist_ok=True)
    image_map = _image_index(images_dir)
    class_list = list(classes)
    written = 0; skipped = 0
    json_files: List[Path] = []
    if xjson_dir.exists():
        json_files += list(xjson_dir.rglob("*.json"))
    # Robust fallback: X-AnyLabeling may write JSON next to images if output handling differs.
    try:
        json_files += [p for p in images_dir.rglob("*.json") if p not in json_files]
    except Exception:
        pass
    seen = set()
    for js in json_files:
        if str(js.resolve()) in seen: continue
        seen.add(str(js.resolve()))
        try:
            data = json.loads(js.read_text(encoding="utf-8", errors="replace"))
            stem = Path(str(data.get("imagePath") or js.stem)).stem
            img = image_map.get(stem)
            w = int(data.get("imageWidth") or 0); h = int(data.get("imageHeight") or 0)
            if (w <= 0 or h <= 0) and img:
                w, h = _image_size(img)
            if w <= 0 or h <= 0:
                skipped += 1; continue
            lines: List[str] = []
            for shape in data.get("shapes") or []:
                if not isinstance(shape, dict): continue
                label = str(shape.get("label") or "").strip()
                if not label: continue
                if label not in class_list:
                    class_list.append(label)
                cid = class_list.index(label)
                bbox = _points_bbox(shape.get("points") or [])
                if not bbox: continue
                x1, y1, x2, y2 = bbox
                x1, x2 = sorted([x1, x2]); y1, y2 = sorted([y1, y2])
                xc = _clip01(((x1 + x2) / 2.0) / w)
                yc = _clip01(((y1 + y2) / 2.0) / h)
                bw = _clip01(abs(x2 - x1) / w)
                bh = _clip01(abs(y2 - y1) / h)
                if bw <= 0 or bh <= 0: continue
                lines.append(f"{cid} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")
            (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            written += 1
        except Exception:
            skipped += 1
    return written, skipped, class_list


class XAnyLabelingAnnotatorBridge(QWidget):  # type: ignore[misc,valid-type]
    def __init__(self, annotator_page: Any, workspace: Any = None, parent: Any = None):
        super().__init__(parent)
        self.annotator_page = annotator_page
        self.workspace = workspace
        self.proc: Optional[Any] = None
        self.install_proc: Optional[Any] = None
        self.project_info: Dict[str, Any] = {}
        self.classes_file: Optional[Path] = None
        self.xlabels_dir: Optional[Path] = None
        self._start_after_install = False  # v4: never auto-start on Mustatil startup; set only during explicit button workflows if needed.
        self._build_ui()
        _write_runtime_notices()

    def _build_ui(self) -> None:
        # v9: ultra-compact non-expanding row.  The previous multi-button row
        # inserted above Project root could increase the form/splitter minimum
        # width and make the left Annotator side look narrower.  This widget is
        # intentionally small, maximum-width limited, and inserted as the FIELD
        # of the existing Project-root QFormLayout row, not as a full-width span.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(3)

        self.status = QLabel("")
        self.status.setVisible(False)
        self.auto_sync = QCheckBox("auto")
        self.auto_sync.setChecked(True)
        self.auto_sync.setToolTip("Automatically sync X-AnyLabeling JSON back to YOLO labels after closing X-AnyLabeling.")

        self.start_btn = QPushButton("X-AnyLabeling")
        self.start_btn.setToolTip("Install if needed, pre-sync YOLO labels to X-AnyLabeling JSON, then start X-AnyLabeling.")
        self.sync_btn = QPushButton("Sync")
        self.sync_btn.setToolTip("Manually sync X-AnyLabeling JSON back to Mustatil/YOLO labels.")
        self.open_btn = QPushButton("JSON")
        self.open_btn.setToolTip("Open the X-AnyLabeling JSON folder.")
        self.about_btn = QPushButton("?")

        try:
            self.start_btn.setMaximumWidth(112)
            self.sync_btn.setMaximumWidth(48)
            self.open_btn.setMaximumWidth(48)
            self.about_btn.setMaximumWidth(24)
            self.auto_sync.setMaximumWidth(54)
            for b in [self.start_btn, self.sync_btn, self.open_btn, self.about_btn]:
                b.setMinimumHeight(22)
                b.setMaximumHeight(24)
                b.setMinimumWidth(0)
            self.setMinimumWidth(0)
            self.setMaximumWidth(310)
            self.setMinimumHeight(24)
            self.setMaximumHeight(28)
            if QSizePolicy is not None:
                self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        except Exception:
            pass

        self.start_btn.clicked.connect(self.install_or_start)
        self.sync_btn.clicked.connect(self.sync_back_now)
        self.open_btn.clicked.connect(self.open_xjson_folder)
        self.about_btn.clicked.connect(self.show_about)

        row.addWidget(self.start_btn)
        row.addWidget(self.sync_btn)
        row.addWidget(self.open_btn)
        row.addWidget(self.auto_sync)
        row.addWidget(self.about_btn)
        row.addWidget(self.status)
        outer.addLayout(row)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(1)
        self.log.setVisible(False)
        self.log.setMaximumBlockCount(1000)
        outer.addWidget(self.log)
        try:
            self.setObjectName("MustatilXAnyAnnotatorBridge")
            self.setStyleSheet(
                "QWidget#MustatilXAnyAnnotatorBridge { border: 0px; margin: 0px; padding: 0px; } "
                "QPushButton { padding: 1px 4px; } QCheckBox { padding: 0px; margin: 0px; }"
            )
        except Exception:
            pass

    def _append(self, text: str) -> None:
        try:
            s = str(text)
            # Keep detailed log internally, but do not show a large text box on
            # the Annotator page.  Show only the last useful status line.
            try:
                last = [x.strip() for x in s.splitlines() if x.strip()]
                if last:
                    self.status.setText(last[-1][:160])
            except Exception:
                pass
            try:
                if QTextCursor is not None:
                    self.log.moveCursor(QTextCursor.End)
                    self.log.insertPlainText(s)
                    self.log.moveCursor(QTextCursor.End)
                else:
                    self.log.appendPlainText(s.rstrip("\n"))
            except Exception:
                pass
            try:
                print("[X-AnyLabeling Annotator Sync] " + s, end="" if s.endswith("\n") else "\n", flush=True)
            except Exception:
                pass
        except Exception:
            pass

    def _make_process(self) -> Any:
        proc = QProcess(self)
        try:
            proc.setProcessChannelMode(QProcess.MergedChannels)
        except Exception:
            pass
        env = proc.processEnvironment()
        for k, v in _runtime_env().items():
            try: env.insert(k, str(v))
            except Exception: pass
        proc.setProcessEnvironment(env)
        proc.readyReadStandardOutput.connect(lambda p=proc: self._read_process(p))
        proc.readyReadStandardError.connect(lambda p=proc: self._read_process(p))
        return proc

    def _read_process(self, proc: Any) -> None:
        for meth in ["readAllStandardOutput", "readAllStandardError"]:
            try:
                data = bytes(getattr(proc, meth)()).decode("utf-8", errors="replace")
                if data: self._append(data)
            except Exception:
                pass

    def _ensure_project(self, allow_dialog: bool = True) -> Optional[Dict[str, Any]]:
        info = _detect_project_from_context(self.annotator_page, self.workspace)
        if not info and allow_dialog and QFileDialog is not None:
            d = QFileDialog.getExistingDirectory(self, "Choose current Mustatil/YOLO project folder", str(Path.home()))
            if d: info = _detect_project_from_path(d)
        if not info:
            self._append("[project] Could not detect project. Choose/open the project in Annotator first.\n")
            self.status.setText("project not detected")
            return None
        project = Path(info.get("project") or "")
        images = Path(info.get("images") or "")
        labels = Path(info.get("labels") or "")
        classes = list(info.get("classes") or [])
        if not images.exists():
            self._append(f"[project] images folder not found: {images}\n")
        if not labels.exists():
            self._append(f"[project] labels folder not found yet, will create/use: {labels}\n")
        if not classes:
            classes = _infer_classes_from_label_ids(labels)
        if not classes:
            classes = ["object"]
        info["classes"] = classes
        self.classes_file = _write_classes_file(classes, project if project.exists() else images.parent)
        self.xlabels_dir = _xlabels_dir_for_project(project if project.exists() and project.is_dir() else images.parent)
        self.project_info = info
        self.status.setText(f"Project: {project.name if project else 'project'} | images={images.name} | labels={labels.name} | classes={len(classes)}")
        self._append("[project detected] " + json.dumps(info, ensure_ascii=False, indent=2) + "\n")
        return info

    def _xany_installed(self) -> bool:
        for name in ["xanylabeling", "anylabeling"]:
            if _venv_script(name).exists(): return True
        if _venv_python().exists(): return True
        return False

    def _xany_program_args(self, extra_args: List[str]) -> Tuple[str, List[str], Path]:
        py = _venv_python()
        if not py.exists(): py = Path(sys.executable or "python")
        for name in ["xanylabeling", "anylabeling"]:
            exe = _venv_script(name)
            if exe.exists(): return str(exe), extra_args, _runtime_root()
        return str(py), ["-m", "anylabeling.app"] + extra_args, _runtime_root()

    def _write_install_script(self) -> Tuple[str, List[str], Path]:
        rt = _runtime_root(); rt.mkdir(parents=True, exist_ok=True); _write_runtime_notices()
        py = sys.executable or "python"
        vpy = _venv_python()
        pkg = DEFAULT_PYPI_PACKAGE
        if _is_windows():
            script = rt / "install_x_anylabeling_from_pypi_for_annotator.cmd"
            script.write_text(
                "@echo off\r\n"
                "chcp 65001 >nul\r\n"
                "echo ================================================\r\n"
                "echo Mustatil X-AnyLabeling PyPI runtime installer\r\n"
                "echo ================================================\r\n"
                f"echo Runtime: {rt}\r\n"
                f"\"{py}\" -m venv \"{_venv_dir()}\"\r\n"
                "if errorlevel 1 exit /b 10\r\n"
                f"\"{vpy}\" -m pip install --upgrade pip setuptools wheel\r\n"
                "if errorlevel 1 exit /b 11\r\n"
                f"\"{vpy}\" -m pip install --pre --upgrade \"{pkg}\"\r\n"
                "if errorlevel 1 exit /b 12\r\n"
                f"\"{vpy}\" -m pip show x-anylabeling-cvhub\r\n"
                f"\"{vpy}\" -m anylabeling.app version\r\n"
                "exit /b 0\r\n",
                encoding="utf-8",
            )
            return "cmd.exe", ["/d", "/c", str(script)], rt
        script = rt / "install_x_anylabeling_from_pypi_for_annotator.sh"
        script.write_text(
            "#!/usr/bin/env bash\nset -e\n"
            f"{_quote_cmd(py)} -m venv {_quote_cmd(_venv_dir())}\n"
            f"{_quote_cmd(vpy)} -m pip install --upgrade pip setuptools wheel\n"
            f"{_quote_cmd(vpy)} -m pip install --pre --upgrade {_quote_cmd(pkg)}\n"
            f"{_quote_cmd(vpy)} -m anylabeling.app version\n",
            encoding="utf-8",
        )
        try: script.chmod(0o755)
        except Exception: pass
        return "bash", [str(script)], rt

    def install_or_start(self) -> None:
        try:
            if self.proc is not None and self.proc.state() != QProcess.NotRunning:
                self._append("[running] X-AnyLabeling is already running. Close it first; sync will run after close.\n")
                return
            info = self._ensure_project(allow_dialog=True)
            if not info: return
            # Always pre-sync existing Mustatil/YOLO labels so X-AnyLabeling opens with them visible.
            self.convert_yolo_to_xanylabeling(info)
            if not self._xany_installed():
                self._append("[install] X-AnyLabeling runtime not found; installing from PyPI first.\n")
                self._start_after_install = False  # v4: install only; user can click again to start after install.
                program, args, cwd = self._write_install_script()
                self.install_proc = self._make_process()
                self.install_proc.setWorkingDirectory(str(cwd))
                self.install_proc.finished.connect(self._install_finished)
                self._append("> " + program + " " + " ".join(args) + "\n")
                self.install_proc.start(program, args)
                return
            self.start_xanylabeling(info)
        except Exception:
            self._append("[Install/Start failed]\n" + traceback.format_exc() + "\n")

    def _install_finished(self, code: int, status: Any = None) -> None:
        self._read_process(self.install_proc)
        self._append(f"\n[install finished exit_code={code}]\n")
        if code == 0 and self._start_after_install:
            self._start_after_install = False  # v4: never auto-start on Mustatil startup; set only during explicit button workflows if needed.
            QTimer.singleShot(500, lambda: self.start_xanylabeling(self.project_info))

    def convert_yolo_to_xanylabeling(self, info: Optional[Dict[str, Any]] = None) -> None:
        try:
            info = info or self._ensure_project(allow_dialog=True)
            if not info: return
            images = Path(info.get("images") or "")
            labels = Path(info.get("labels") or "")
            project = Path(info.get("project") or images.parent)
            classes = list(info.get("classes") or [])
            if not classes: classes = _infer_classes_from_label_ids(labels) or ["object"]
            self.classes_file = _write_classes_file(classes, project)
            self.xlabels_dir = _xlabels_dir_for_project(project)
            if not images.exists():
                self._append(f"[pre-sync] images folder missing: {images}\n"); return
            if not labels.exists():
                labels.mkdir(parents=True, exist_ok=True)
            converted, skipped = _yolo_to_xjson(images, labels, classes, self.xlabels_dir)
            self._append(f"[pre-sync YOLO -> X-AnyLabeling JSON] converted={converted}, skipped={skipped}, out={self.xlabels_dir}\n")
        except Exception:
            self._append("[pre-sync failed]\n" + traceback.format_exc() + "\n")

    def start_xanylabeling(self, info: Optional[Dict[str, Any]] = None) -> None:
        try:
            info = info or self._ensure_project(allow_dialog=True)
            if not info: return
            images = Path(info.get("images") or "")
            project = Path(info.get("project") or images.parent)
            xlabels = self.xlabels_dir or _xlabels_dir_for_project(project)
            classes_path = self.classes_file or _write_classes_file(list(info.get("classes") or ["object"]), project)
            args: List[str] = ["--work-dir", str(_work_dir())]
            if images.exists(): args += ["--filename", str(images)]
            xlabels.mkdir(parents=True, exist_ok=True)
            args += ["--output", str(xlabels)]
            if classes_path.exists(): args += ["--labels", str(classes_path), "--validatelabel", "exact"]
            args += ["--autosave", "--no-auto-update-check"]
            program, final_args, cwd = self._xany_program_args(args)
            self.proc = self._make_process()
            self.proc.setWorkingDirectory(str(cwd))
            self.proc.finished.connect(self._xany_finished)
            self._append("> " + program + " " + " ".join([_quote_cmd(a) if " " in str(a) else str(a) for a in final_args]) + "\n")
            self.proc.start(program, final_args)
        except Exception:
            self._append("[start failed]\n" + traceback.format_exc() + "\n")

    def _xany_finished(self, code: int, status: Any = None) -> None:
        try: self._read_process(self.proc)
        except Exception: pass
        self._append(f"\n[X-AnyLabeling closed exit_code={code}]\n")
        if self.auto_sync.isChecked():
            self._append("[auto-sync] syncing JSON annotations back to Mustatil YOLO labels...\n")
            self.sync_back_now()

    def sync_back_now(self) -> None:
        try:
            info = self.project_info or self._ensure_project(allow_dialog=True)
            if not info: return
            images = Path(info.get("images") or "")
            labels = Path(info.get("labels") or "")
            project = Path(info.get("project") or images.parent)
            classes = list(info.get("classes") or [])
            if self.classes_file and self.classes_file.exists():
                classes = [x.strip() for x in self.classes_file.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()]
            xlabels = self.xlabels_dir or _xlabels_dir_for_project(project)
            written, skipped, new_classes = _xjson_to_yolo(xlabels, images, labels, classes)
            self.classes_file = _write_classes_file(new_classes, project)
            info["classes"] = new_classes
            self.project_info = info
            self._append(f"[sync X-AnyLabeling JSON -> YOLO labels] written={written}, skipped={skipped}, labels={labels}, classes={len(new_classes)}\n")
            self._refresh_annotator()
        except Exception:
            self._append("[sync failed]\n" + traceback.format_exc() + "\n")

    def _refresh_annotator(self) -> None:
        targets = [self.annotator_page, self.workspace]
        method_names = [
            "reload_labels", "load_labels", "refresh_labels", "reload_annotations", "refresh_annotations", "reload_current_image", "load_current_image", "refresh", "update_view", "redraw", "update",
        ]
        for obj in targets:
            if obj is None: continue
            for name in method_names:
                try:
                    fn = getattr(obj, name, None)
                    if callable(fn):
                        try: fn()
                        except TypeError:
                            try: fn(False)
                            except Exception: pass
                except Exception:
                    pass
        self._append("[Mustatil Annotator refresh requested]\n")

    def open_xjson_folder(self) -> None:
        try:
            info = self.project_info or self._ensure_project(allow_dialog=True)
            if not info: return
            project = Path(info.get("project") or "")
            folder = self.xlabels_dir or _xlabels_dir_for_project(project)
            folder.mkdir(parents=True, exist_ok=True)
            if _is_windows(): subprocess.Popen(["explorer", str(folder)])
            elif sys.platform == "darwin": subprocess.Popen(["open", str(folder)])
            else: subprocess.Popen(["xdg-open", str(folder)])
        except Exception as exc:
            self._append(f"[open folder failed] {exc}\n")

    def show_about(self) -> None:
        msg = (
            "X-AnyLabeling bridge for Mustatil Annotator\n\n"
            "Installed from PyPI package x-anylabeling-cvhub into a local runtime.\n"
            "Existing YOLO labels are converted to X-AnyLabeling JSON before start.\n"
            "After X-AnyLabeling closes, JSON annotations are synchronized back to the existing Mustatil/YOLO labels folder.\n\n"
            "Source: X-AnyLabeling / CVHub520\n"
            "URL: https://github.com/CVHub520/X-AnyLabeling\n"
            "License notice: GPL-3.0. This plugin ZIP includes LICENSES/GPL-3.0.txt and THIRD_PARTY_NOTICES.md.\n\n"
            "Modified for Mustatil by Tarek Wasfy, 2026."
        )
        try: QMessageBox.information(self, "About X-AnyLabeling in Mustatil", msg)
        except Exception: self._append(msg + "\n")


# -----------------------------------------------------------------------------
# Annotator-page patching
# -----------------------------------------------------------------------------

def _tab_label_is_annotator(label: str) -> bool:
    low = str(label or "").strip().lower()
    if not low:
        return False
    if "x-any" in low or "anylabel" in low:
        return False
    if "trainer" in low or "training" in low or "pipeline" in low:
        return False
    # v5: strict.  Do not patch every page containing the words label/class.
    return low == "annotator" or low.endswith(" annotator") or " annotator" in low


def _page_text(page: Any) -> str:
    parts: List[str] = []
    try:
        parts.append(str(page.objectName()))
        parts.append(str(page.windowTitle()))
    except Exception:
        pass
    try:
        for child in page.findChildren(QWidget):
            try:
                for attr in ["text", "title", "objectName", "windowTitle", "placeholderText"]:
                    fn = getattr(child, attr, None)
                    if callable(fn):
                        v = fn()
                        if v: parts.append(str(v))
            except Exception:
                pass
    except Exception:
        pass
    return " ".join(parts).lower()


def _button_text(widget: Any) -> str:
    try:
        if hasattr(widget, "text"):
            return str(widget.text() or "")
    except Exception:
        pass
    return ""


def _find_manage_classes_group(page: Any) -> Optional[Any]:
    """Find the Manage Classes QGroupBox on the real Annotator tab.

    In the user's current UI, "Manage Classes" is a group/box, not a button.
    The stable controls inside it are buttons such as "Add Class" and
    "Redraw Preview".  v7 therefore anchors to that group first and inserts the
    X-AnyLabeling row inside/under it.
    """
    try:
        for gb in page.findChildren(QGroupBox):
            title = ""
            try:
                title = str(gb.title() or "")
            except Exception:
                pass
            low = title.strip().lower().replace("&", "")
            if ("manage" in low and "class" in low) or ("klasse" in low and ("manage" in low or "verwal" in low)):
                return gb
    except Exception:
        pass
    # Fallback: find a group that contains the known class-manager buttons.
    try:
        for gb in page.findChildren(QGroupBox):
            texts = []
            try:
                texts = [_button_text(b).strip().lower().replace("&", "") for b in gb.findChildren(QPushButton)]
            except Exception:
                texts = []
            if any("add class" in t or "klasse" in t for t in texts) and any("redraw" in t or "preview" in t for t in texts):
                return gb
            if any("add class" in t for t in texts):
                return gb
    except Exception:
        pass
    return None


def _find_class_manager_anchor_button(root: Any) -> Optional[Any]:
    """Prefer Add Class, otherwise Redraw Preview, inside Manage Classes."""
    try:
        buttons = list(root.findChildren(QPushButton))
    except Exception:
        buttons = []
    ordered_needles = [
        ("add class",),
        ("add", "class"),
        ("klasse",),
        ("redraw preview",),
        ("redraw", "preview"),
        ("preview",),
    ]
    for needles in ordered_needles:
        for b in buttons:
            txt = _button_text(b).strip().lower().replace("&", "")
            if all(n in txt for n in needles):
                return b
    return None


def _find_manage_classes_button(page: Any) -> Optional[Any]:
    """Compatibility fallback for older class-manager patches with a button."""
    try:
        for child in page.findChildren(QPushButton):
            txt = _button_text(child).strip().lower().replace("&", "")
            if txt in {"manage classes", "classes", "class manager", "manage class", "class manager…", "manage classes…"}:
                return child
            if "manage" in txt and "class" in txt:
                return child
    except Exception:
        pass
    return None

def _find_annotations_group(page: Any) -> Optional[Any]:
    """Find Mustatil's exact Annotator annotations group.

    In the uploaded workspace this is created in _build_annotator_tab as
    QGroupBox("Annotations for current image"), with the class dropdown, the
    annotation list and the edit buttons.  This is the safest fallback anchor
    when no Manage Classes button exists in the active build.
    """
    try:
        for gb in page.findChildren(QGroupBox):
            title = ""
            try: title = str(gb.title() or "")
            except Exception: title = ""
            low = title.strip().lower()
            if low == "annotations for current image":
                return gb
            if "annotation" in low and "current" in low and "image" in low:
                return gb
    except Exception:
        pass
    return None


def _looks_like_annotator_page(page: Any, label: str = "") -> bool:
    # v7: only the real Annotator tab label is accepted.  No generic widget scan.
    if not _tab_label_is_annotator(label):
        return False
    target = _target_widget_for_page(page)
    if target is None:
        return False
    if _find_manage_classes_group(target) is not None:
        return True
    if _find_manage_classes_button(target) is not None:
        return True
    # Last fallback for plain uploaded workspace builds without class-manager group.
    if _find_annotations_group(target) is not None:
        return True
    return False

def _target_widget_for_page(page: Any) -> Any:
    try:
        if isinstance(page, QScrollArea) and page.widget() is not None:
            return page.widget()
    except Exception:
        pass
    return page


def _direct_layout_index_of_widget(layout: Any, widget: Any) -> int:
    try:
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item is not None and item.widget() is widget:
                return i
    except Exception:
        pass
    return -1


def _layout_class_name(layout: Any) -> str:
    try:
        mo = layout.metaObject()
        if mo is not None:
            return str(mo.className())
    except Exception:
        pass
    try:
        return layout.__class__.__name__
    except Exception:
        return ""


def _layout_is_horizontal(layout: Any) -> bool:
    name = _layout_class_name(layout).lower()
    return "hbox" in name or "horizontal" in name


def _layout_is_vertical(layout: Any) -> bool:
    name = _layout_class_name(layout).lower()
    return "vbox" in name or "vertical" in name


def _find_layout_path_to_widget(layout: Any, widget: Any, path: Optional[List[Tuple[Any, int, Any]]] = None) -> Optional[List[Tuple[Any, int, Any]]]:
    if layout is None:
        return None
    if path is None:
        path = []
    try:
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item is None:
                continue
            try:
                if item.widget() is widget:
                    return path + [(layout, i, item)]
            except Exception:
                pass
            try:
                sub = item.layout()
                if sub is not None:
                    found = _find_layout_path_to_widget(sub, widget, path + [(layout, i, item)])
                    if found:
                        return found
            except Exception:
                pass
            try:
                w = item.widget()
                if w is not None and hasattr(w, "layout") and w.layout() is not None:
                    found = _find_layout_path_to_widget(w.layout(), widget, path + [(layout, i, item)])
                    if found:
                        return found
            except Exception:
                pass
    except Exception:
        pass
    return None


def _insert_widget_in_layout(layout: Any, index: int, box: Any) -> bool:
    try:
        if layout is None:
            return False
        if hasattr(layout, "insertWidget"):
            layout.insertWidget(max(0, int(index)), box)
            return True
        # QGridLayout / QFormLayout fallback.  This is not ideal but still keeps
        # the widget anchored near the requested class-management area.
        if hasattr(layout, "addWidget") and hasattr(layout, "getItemPosition"):
            try:
                row, col, row_span, col_span = layout.getItemPosition(max(0, int(index)))
                layout.addWidget(box, max(0, int(row)) + 1, 0, 1, max(2, int(col_span) + int(col) + 1))
                return True
            except Exception:
                pass
    except Exception:
        pass
    return False


def _insert_box_below_manage_classes(target: Any, manage_btn: Any, box: Any) -> bool:
    """Insert directly below Manage Classes, never at the page top."""
    # 1) If Manage Classes is inside a row-widget, put our widget directly below
    # that row-widget in the parent VBox.
    try:
        row_widget = manage_btn.parentWidget()
        if row_widget is not None and row_widget is not target:
            parent = row_widget.parentWidget()
            if parent is not None and hasattr(parent, "layout"):
                parent_lay = parent.layout()
                idx = _direct_layout_index_of_widget(parent_lay, row_widget)
                if idx >= 0 and _insert_widget_in_layout(parent_lay, idx + 1, box):
                    _log("v9 placed X-AnyLabeling directly below Manage Classes parent row")
                    return True
    except Exception:
        pass

    # 2) Path-based layout insertion.
    try:
        root_layout = target.layout() if hasattr(target, "layout") else None
        path = _find_layout_path_to_widget(root_layout, manage_btn)
        if path:
            containing_layout, btn_idx, _btn_item = path[-1]
            if _layout_is_horizontal(containing_layout) and len(path) >= 2:
                parent_layout, row_idx, _row_item = path[-2]
                if _insert_widget_in_layout(parent_layout, row_idx + 1, box):
                    _log("v9 placed X-AnyLabeling below Manage Classes horizontal row")
                    return True
            if _layout_is_vertical(containing_layout):
                if _insert_widget_in_layout(containing_layout, btn_idx + 1, box):
                    _log("v9 placed X-AnyLabeling directly after Manage Classes in vertical layout")
                    return True
            if len(path) >= 2:
                parent_layout, row_idx, _row_item = path[-2]
                if _insert_widget_in_layout(parent_layout, row_idx + 1, box):
                    _log("v9 placed X-AnyLabeling below Manage Classes parent layout item")
                    return True
            if _insert_widget_in_layout(containing_layout, btn_idx + 1, box):
                _log("v9 placed X-AnyLabeling immediately after Manage Classes")
                return True
    except Exception:
        pass

    return False


def _insert_box_in_manage_classes_group(target: Any, box: Any) -> bool:
    """Place compact X-AnyLabeling row directly under Manage Classes.

    Preferred placement:
      QGroupBox("Manage Classes")
        ... Add Class / Redraw Preview buttons ...
        [X-AnyLabeling row]

    If the class-manager group uses nested horizontal rows, the row is inserted
    immediately below the Add Class/Redraw Preview row.  If that is not possible,
    it is appended at the end of the Manage Classes group.
    """
    group = _find_manage_classes_group(target)
    if group is None:
        return False
    lay = group.layout() if hasattr(group, "layout") else None
    if lay is None:
        return False

    # If Add Class / Redraw Preview is inside a sub-row, insert below that row.
    anchor = _find_class_manager_anchor_button(group)
    if anchor is not None:
        try:
            path = _find_layout_path_to_widget(lay, anchor)
            if path:
                containing_layout, btn_idx, _btn_item = path[-1]
                if _layout_is_horizontal(containing_layout) and len(path) >= 2:
                    parent_layout, row_idx, _row_item = path[-2]
                    if _insert_widget_in_layout(parent_layout, row_idx + 1, box):
                        _log("v9 placed X-AnyLabeling directly below Add Class/Redraw Preview row in Manage Classes")
                        return True
                if _layout_is_vertical(containing_layout):
                    if _insert_widget_in_layout(containing_layout, btn_idx + 1, box):
                        _log("v9 placed X-AnyLabeling directly below Add Class/Redraw Preview in Manage Classes")
                        return True
                if len(path) >= 2:
                    parent_layout, row_idx, _row_item = path[-2]
                    if _insert_widget_in_layout(parent_layout, row_idx + 1, box):
                        _log("v9 placed X-AnyLabeling below class-manager parent row")
                        return True
        except Exception:
            pass

    # Direct group-layout append/insert fallback.
    try:
        if hasattr(lay, "insertWidget"):
            # Put it low in the group, under existing class controls/buttons.
            lay.insertWidget(max(0, lay.count()), box)
            _log("v9 placed X-AnyLabeling at bottom of Manage Classes group")
            return True
        if hasattr(lay, "addWidget") and hasattr(lay, "rowCount"):
            try:
                lay.addWidget(box, int(lay.rowCount()), 0, 1, 2)
                _log("v9 placed X-AnyLabeling in Manage Classes grid/form layout")
                return True
            except Exception:
                pass
        lay.addWidget(box)
        _log("v9 placed X-AnyLabeling in Manage Classes using addWidget fallback")
        return True
    except Exception:
        return False


def _insert_box_in_annotations_group(target: Any, box: Any) -> bool:
    """Fallback for the uploaded base workspace where no Manage Classes button exists.

    Exact uploaded structure:
      ann_box = QGroupBox("Annotations for current image")
      ann_lay = QVBoxLayout(ann_box)
      index 0: class row
      index 1: annotation list
      index 2: edit button grid

    Put the X-AnyLabeling row at index 1: directly under the class-control row,
    and above the list of boxes.  This is the nearest stable place to "Manage
    Classes" in the current Annotator build.
    """
    try:
        group = _find_annotations_group(target)
        if group is None:
            return False
        lay = group.layout() if hasattr(group, "layout") else None
        if lay is None:
            return False
        # If some other class-manager patch inserted a Manage Classes button into
        # this group, prefer direct placement below it.
        manage_btn = _find_manage_classes_button(group)
        if manage_btn is not None and _insert_box_below_manage_classes(group, manage_btn, box):
            return True
        if hasattr(lay, "insertWidget"):
            lay.insertWidget(1, box)
            _log("v9 placed X-AnyLabeling inside 'Annotations for current image' below class row")
            return True
        lay.addWidget(box)
        _log("v9 placed X-AnyLabeling inside annotations group using addWidget fallback")
        return True
    except Exception:
        pass
    return False



def _text_of_widget(widget: Any) -> str:
    try:
        fn = getattr(widget, "text", None)
        if callable(fn):
            return str(fn() or "")
    except Exception:
        pass
    try:
        fn = getattr(widget, "title", None)
        if callable(fn):
            return str(fn() or "")
    except Exception:
        pass
    return ""


def _find_project_root_label(target: Any) -> Optional[Any]:
    """Find the real 'Project root' label in the Annotator top project box."""
    try:
        for lab in target.findChildren(QLabel):
            txt = _text_of_widget(lab).strip().replace("&", "").lower()
            if txt == "project root":
                return lab
    except Exception:
        pass
    return None


def _find_project_image_group(target: Any) -> Optional[Any]:
    """Find QGroupBox('Project / image folder') on the real Annotator tab."""
    try:
        for gb in target.findChildren(QGroupBox):
            title = _text_of_widget(gb).strip().lower()
            if "project" in title and ("image" in title or "folder" in title):
                if _find_project_root_label(gb) is not None:
                    return gb
    except Exception:
        pass
    # Fallback: ascend from the Project root label until a group box is reached.
    try:
        lab = _find_project_root_label(target)
        p = lab.parentWidget() if lab is not None and hasattr(lab, "parentWidget") else None
        for _ in range(12):
            if p is None:
                break
            try:
                if isinstance(p, QGroupBox):
                    return p
            except Exception:
                pass
            p = p.parentWidget() if hasattr(p, "parentWidget") else None
    except Exception:
        pass
    return None


def _insert_box_above_project_root(target: Any, box: Any) -> bool:
    """Place a tiny X-AnyLabeling control above Project root without resizing the left pane.

    Critical v9 change: do not insert a full-width frame row.  Instead insert a
    normal two-column form row:

        X-AnyLabeling: [small buttons]
        Project root: [...]

    This keeps the original form geometry and avoids making the Annotator left
    side visually narrower.
    """
    group = _find_project_image_group(target)
    if group is None:
        return False
    lay = group.layout() if hasattr(group, "layout") else None
    if lay is not None:
        try:
            if hasattr(lay, "insertRow"):
                label = QLabel("X-AnyLabeling")
                try:
                    label.setMinimumWidth(0)
                    label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum)
                    box.setMinimumWidth(0)
                    box.setMaximumWidth(320)
                    box.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
                except Exception:
                    pass
                lay.insertRow(0, label, box)
                _log("v9 placed tiny X-AnyLabeling form row above Project root")
                return True
        except Exception as exc:
            _log("v9 form-row insert failed: " + str(exc))
        try:
            if hasattr(lay, "insertWidget"):
                # fallback, still keep width limited
                try:
                    box.setMaximumWidth(320)
                    box.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
                except Exception:
                    pass
                lay.insertWidget(0, box)
                _log("v9 placed limited-width X-AnyLabeling widget above Project root fallback")
                return True
        except Exception:
            pass
    return False

def _remove_old_xany_boxes(root: Any) -> None:
    try:
        for child in root.findChildren(QWidget):
            name = str(child.objectName() or "")
            if "MustatilXAnyAnnotatorBridge" in name:
                try:
                    child.setParent(None)
                    child.deleteLater()
                except Exception:
                    try: child.hide()
                    except Exception: pass
    except Exception:
        pass

def _insert_bridge_on_page(page: Any, label: str = "") -> bool:
    try:
        if id(page) in _PATCHED_PAGES:
            return False
        if not _looks_like_annotator_page(page, label):
            return False
        target = _target_widget_for_page(page)
        if target is None:
            return False

        # Remove all older misplaced bridge boxes first. v9 creates exactly one
        # compact row at the top of the Annotator page, above Project root.
        _remove_old_xany_boxes(target)

        try:
            for child in target.findChildren(QWidget):
                if str(child.objectName()) in {"MustatilXAnyAnnotatorBridgeV9", "MustatilXAnyAnnotatorBridgeBoxV9"}:
                    _PATCHED_PAGES.add(id(page))
                    return False
        except Exception:
            pass

        # Only patch pages that really have the Project root row. This prevents
        # accidental placement on Detection/Trainer/other tabs.
        project_group = _find_project_image_group(target)
        if project_group is None:
            return False

        ws = _find_workspace_from_page(page)
        box = QWidget(target)
        box.setObjectName("MustatilXAnyAnnotatorBridgeBoxV9")
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(1, 1, 1, 1)
        box_layout.setSpacing(0)
        bridge = XAnyLabelingAnnotatorBridge(page, ws, box)
        try:
            bridge.setObjectName("MustatilXAnyAnnotatorBridgeV9")
        except Exception:
            pass
        box_layout.addWidget(bridge)
        try:
            box.setMinimumWidth(0)
            box.setMaximumWidth(320)
            box.setMaximumHeight(30)
            box.setMinimumHeight(22)
            if QSizePolicy is not None:
                box.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
            box.setStyleSheet("QWidget#MustatilXAnyAnnotatorBridgeBoxV9 { border: 0px; margin: 0px; padding: 0px; }")
        except Exception:
            pass

        inserted = _insert_box_above_project_root(target, box)
        if inserted:
            _PATCHED_PAGES.add(id(page))
            _log(f"inserted compact X-AnyLabeling buttons above Project root in Annotator: {label}")
            return True
        try:
            box.setParent(None)
            box.deleteLater()
        except Exception:
            pass
    except Exception as exc:
        _log("insert bridge failed: " + str(exc))
        try:
            traceback.print_exc()
        except Exception:
            pass
    return False

def _scan_tabwidget(tw: Any) -> int:
    n = 0
    try:
        for i in range(tw.count()):
            page = tw.widget(i)
            label = str(tw.tabText(i))
            if _insert_bridge_on_page(page, label): n += 1
    except Exception:
        pass
    return n


def _scan_application() -> int:
    if QApplication is None or QTabWidget is None: return 0
    app = QApplication.instance()
    if app is None: return 0
    widgets: List[Any] = []
    try:
        for w in app.topLevelWidgets():
            if isinstance(w, QTabWidget): widgets.append(w)
            try: widgets.extend(w.findChildren(QTabWidget))
            except Exception: pass
    except Exception:
        pass
    total = 0; seen = set()
    for tw in widgets:
        if id(tw) in seen: continue
        seen.add(id(tw))
        total += _scan_tabwidget(tw)
    return total


def _install_qtab_hook() -> None:
    global _PATCHED_QTAB, _ORIG_ADD_TAB, _ORIG_INSERT_TAB
    if QTabWidget is None or QTimer is None or _PATCHED_QTAB: return
    try:
        _ORIG_ADD_TAB = QTabWidget.addTab
        _ORIG_INSERT_TAB = QTabWidget.insertTab

        def _try_insert_for_new_tab(page: Any, label: str) -> None:
            # Fast path: only check the newly added Annotator page, not the whole app.
            try:
                if _tab_label_is_annotator(label):
                    _insert_bridge_on_page(page, label)
            except Exception:
                pass

        def addTab_patched(self, page, *args, **kwargs):
            res = _ORIG_ADD_TAB(self, page, *args, **kwargs)
            try:
                label = str(args[-1]) if args else ""
                QTimer.singleShot(0, lambda p=page, l=label: _try_insert_for_new_tab(p, l))
                QTimer.singleShot(120, lambda p=page, l=label: _try_insert_for_new_tab(p, l))
            except Exception: pass
            return res

        def insertTab_patched(self, index, page, *args, **kwargs):
            res = _ORIG_INSERT_TAB(self, index, page, *args, **kwargs)
            try:
                label = str(args[-1]) if args else ""
                QTimer.singleShot(0, lambda p=page, l=label: _try_insert_for_new_tab(p, l))
                QTimer.singleShot(120, lambda p=page, l=label: _try_insert_for_new_tab(p, l))
            except Exception: pass
            return res

        QTabWidget.addTab = addTab_patched
        QTabWidget.insertTab = insertTab_patched
        _PATCHED_QTAB = True
        _log("QTabWidget fast Annotator-only hook installed")
    except Exception as exc:
        _log("QTabWidget hook failed: " + str(exc))


def install_patch() -> bool:
    global _SCAN_TIMER
    if QApplication is None or QTimer is None:
        _log("PySide6 not available yet")
        return False
    try:
        _write_runtime_notices()
        _install_qtab_hook()
        # Fast startup: no permanent scanner and no long 20s sweep.  Plugins load
        # before MustatilQtWorkspace is fully built, so the QTabWidget hook above
        # handles the Annotator tab at addTab-time.  These few one-shots cover
        # already-open windows without causing repeated console spam.
        for ms in [0, 120, 600]:
            QTimer.singleShot(ms, _scan_application)
        _log("installed fast Annotator-only X-AnyLabeling sync-button patch v9")
        return True
    except Exception:
        _log("install failed\n" + traceback.format_exc())
        return False


def mustatil_plugin_init():
    return install_patch()


def register_plugin(app=None, main_window=None):
    return install_patch()


def init_plugin(app=None, main_window=None):
    return install_patch()


def load_plugin(app=None, main_window=None):
    return install_patch()


try:
    install_patch()
except Exception:
    pass
