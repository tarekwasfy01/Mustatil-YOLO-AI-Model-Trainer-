#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mustatil LAE-DINO custom trained model selector add-on V3 force selected model

Purpose:
  - add a small "Custom trained model" group into existing LAE-DINO tabs
  - allow selecting project / config / checkpoint manually
  - auto-import trained model from a Mustatil project
  - Run with selected model button triggers the existing LAE-DINO run button
  - patches V8/V9 AutoMatch so selected project model is not replaced by DIOR pretrained
  - set fields for V9 strict plugin:
      ws.mustatil_lae_existing_v9_python
      ws.mustatil_lae_existing_v9_repo
      ws.mustatil_lae_existing_v9_config
      ws.mustatil_lae_existing_v9_weights
      ws.mustatil_lae_existing_v9_classes_text
  - also set V8/canonical aliases for compatibility

This add-on does NOT:
  - replace tabs
  - replace Detection/Satellite layout
  - touch the right preview/map
  - install dependencies
  - globally redirect subprocesses
"""

from __future__ import annotations

import json
import os
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PLUGIN_NAME = "Mustatil LAE-DINO Custom Trained Model Selector V3 Force Selected"

_PATCHED_QTAB = False
_ORIG_ADD = None
_ORIG_INSERT = None
_PATCHED_PAGES = set()


def _log(msg: Any) -> None:
    try:
        print("[LAE-DINO Custom Model Selector] " + str(msg))
    except Exception:
        pass


def _plugin_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except Exception:
        return Path.cwd()


def _runtime_root() -> Path:
    return _plugin_dir() / "mustatil_model_runtimes" / "LAE-DINO-PATCH"


def _runtime_python() -> Path:
    return _runtime_root() / "python310" / ("python.exe" if os.name == "nt" else "python")


def _runtime_repo() -> Path:
    return _runtime_root() / "repo"


def _qt():
    try:
        from PySide6.QtWidgets import (
            QApplication, QWidget, QVBoxLayout, QGridLayout, QHBoxLayout, QLabel,
            QLineEdit, QPushButton, QFileDialog, QGroupBox, QTabWidget,
            QPlainTextEdit, QTextEdit
        )
        from PySide6.QtCore import QTimer
        return {
            "api": "PySide6",
            "QApplication": QApplication,
            "QWidget": QWidget,
            "QVBoxLayout": QVBoxLayout,
            "QGridLayout": QGridLayout,
            "QHBoxLayout": QHBoxLayout,
            "QLabel": QLabel,
            "QLineEdit": QLineEdit,
            "QPushButton": QPushButton,
            "QFileDialog": QFileDialog,
            "QGroupBox": QGroupBox,
            "QTabWidget": QTabWidget,
            "QPlainTextEdit": QPlainTextEdit,
            "QTextEdit": QTextEdit,
            "QTimer": QTimer,
        }
    except Exception:
        from PyQt5.QtWidgets import (
            QApplication, QWidget, QVBoxLayout, QGridLayout, QHBoxLayout, QLabel,
            QLineEdit, QPushButton, QFileDialog, QGroupBox, QTabWidget,
            QPlainTextEdit, QTextEdit
        )
        from PyQt5.QtCore import QTimer
        return {
            "api": "PyQt5",
            "QApplication": QApplication,
            "QWidget": QWidget,
            "QVBoxLayout": QVBoxLayout,
            "QGridLayout": QGridLayout,
            "QHBoxLayout": QHBoxLayout,
            "QLabel": QLabel,
            "QLineEdit": QLineEdit,
            "QPushButton": QPushButton,
            "QFileDialog": QFileDialog,
            "QGroupBox": QGroupBox,
            "QTabWidget": QTabWidget,
            "QPlainTextEdit": QPlainTextEdit,
            "QTextEdit": QTextEdit,
            "QTimer": QTimer,
        }


def _workspace_from_widget(widget: Any) -> Optional[Any]:
    cur = widget
    for _ in range(180):
        if cur is None:
            break
        try:
            if hasattr(cur, "tabs") or hasattr(cur, "dets") or hasattr(cur, "satellite_detections") or hasattr(cur, "project"):
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


def _emit(ws: Any, msg: str, label: Any = None, page: Any = None) -> None:
    try:
        if label is not None:
            label.setText(str(msg))
    except Exception:
        pass
    try:
        if ws is not None and hasattr(ws, "log"):
            ws.log("[LAE-DINO Custom Selector] " + str(msg))
            return
    except Exception:
        pass
    try:
        if page is not None:
            q = _qt()
            for clsname in ("QTextEdit", "QPlainTextEdit"):
                cls = q[clsname]
                logs = page.findChildren(cls)
                if logs:
                    try:
                        logs[-1].append("[LAE-DINO Custom Selector] " + str(msg))
                    except Exception:
                        try:
                            logs[-1].appendPlainText("[LAE-DINO Custom Selector] " + str(msg))
                        except Exception:
                            pass
                    return
    except Exception:
        pass
    _log(msg)


def _safe_text(obj: Any) -> str:
    try:
        return str(obj.text())
    except Exception:
        return ""


def _read_project_classes(project: Path) -> List[str]:
    for name in ("project.json", "mustatil_project.json", "classes.json"):
        p = project / name
        try:
            if not p.exists():
                continue
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(data, dict):
                for key in ("classes", "names", "class_names"):
                    val = data.get(key)
                    if isinstance(val, list):
                        out = []
                        for item in val:
                            if isinstance(item, str):
                                out.append(item)
                            elif isinstance(item, dict):
                                n = item.get("name") or item.get("class") or item.get("label")
                                if n:
                                    out.append(str(n))
                        if out:
                            return out
                    if isinstance(val, dict):
                        out = [str(v) for _, v in sorted(val.items(), key=lambda kv: str(kv[0]))]
                        if out:
                            return out
                for key, val in data.items():
                    if "class" in str(key).lower() and isinstance(val, dict):
                        out = []
                        for _, item in sorted(val.items(), key=lambda kv: str(kv[0])):
                            if isinstance(item, str):
                                out.append(item)
                            elif isinstance(item, dict) and item.get("name"):
                                out.append(str(item.get("name")))
                        if out:
                            return out
        except Exception:
            pass
    return []


def _epoch_number(path: Path) -> int:
    try:
        m = re.search(r"epoch[_\- ]?(\d+)", path.name.lower())
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return -1


def _score_checkpoint(path: Path) -> Tuple[int, float, str]:
    s = str(path).replace("\\", "/").lower()
    n = path.name.lower()
    score = 0
    if "/work_dirs/" in s:
        score += 1000
    if "lae_dino_mustatil" in s:
        score += 400
    if n.startswith("best") or "best_" in n:
        score += 900
    if n == "latest.pth":
        score += 800
    if "epoch" in n:
        score += 500 + max(0, min(999, _epoch_number(path)))
    if "sanitized" in n or "dior" in n or "pretrain" in n or "fintune" in n or "finetune" in n:
        score -= 900
    try:
        mtime = path.stat().st_mtime
    except Exception:
        mtime = 0.0
    return (-score, -mtime, str(path).lower())


def _find_checkpoints(project: Path) -> List[Path]:
    roots = [
        project / "lae_dino_dataset" / "work_dirs" / "lae_dino_mustatil",
        project / "lae_dino_dataset" / "work_dirs",
        project / "lae_dino_dataset",
        project,
    ]
    out: List[Path] = []
    for root in roots:
        try:
            if not root.exists():
                continue
            for pat in ("best*.pth", "latest.pth", "epoch_*.pth", "epoch*.pth", "*.pth", "*.pt"):
                out.extend([p for p in root.rglob(pat) if p.is_file()])
        except Exception:
            pass
    uniq: List[Path] = []
    seen = set()
    for p in out:
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    uniq.sort(key=_score_checkpoint)
    return uniq


def _score_config(path: Path, project: Path) -> Tuple[int, float, str]:
    s = str(path).replace("\\", "/").lower()
    n = path.name.lower()
    score = 0
    if str(project).replace("\\", "/").lower() in s:
        score += 1000
    if "v14" in n and "final" in n:
        score += 1300
    elif "v13" in n and "final" in n:
        score += 1200
    elif "v12" in n and "final" in n:
        score += 700
    if "mmengine_final" in n:
        score += 500
    if "mustatil_train" in n:
        score += 400
    if "train_from_project" in n:
        score += 200
    if "verify_config" in n or "build_final_config" in n:
        score -= 1000
    if "_base_" in s or "__pycache__" in s:
        score -= 1000
    try:
        mtime = path.stat().st_mtime
    except Exception:
        mtime = 0.0
    return (-score, -mtime, str(path).lower())


def _find_configs(project: Path) -> List[Path]:
    dataset = project / "lae_dino_dataset"
    explicit = [
        dataset / "lae_dino_mustatil_train_v14_mmengine_final.py",
        dataset / "lae_dino_mustatil_train_v13_mmengine_final.py",
        dataset / "lae_dino_mustatil_train_v12d_final.py",
        dataset / "lae_dino_mustatil_train_v12c_final.py",
        dataset / "lae_dino_mustatil_train_v12_final.py",
        dataset / "lae_dino_mustatil_train_from_project.py",
    ]
    out = [p for p in explicit if p.exists()]
    for root in [dataset, dataset / "work_dirs", project]:
        try:
            if not root.exists():
                continue
            for p in root.rglob("*.py"):
                low = str(p).replace("\\", "/").lower()
                if "__pycache__" in low or "verify_config" in low or "build_final_config" in low:
                    continue
                if "lae_dino" in p.name.lower() or "mustatil_train" in p.name.lower():
                    out.append(p)
        except Exception:
            pass
    uniq: List[Path] = []
    seen = set()
    for p in out:
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    uniq.sort(key=lambda p: _score_config(p, project))
    return uniq


def _guess_project(ws: Any, page: Any = None) -> str:
    for obj in (ws, page):
        if obj is None:
            continue
        for name in ("mustatil_lae_existing_v9_project", "project_dir", "project_folder", "project_path", "current_project_dir", "last_project_dir"):
            try:
                v = str(getattr(obj, name, "") or "").strip().strip('"')
                if v and ((Path(v) / "project.json").exists() or (Path(v) / "lae_dino_dataset").exists()):
                    return v
            except Exception:
                pass
    try:
        if (Path("G:/Khirigsuurs") / "project.json").exists() or (Path("G:/Khirigsuurs") / "lae_dino_dataset").exists():
            return "G:/Khirigsuurs"
    except Exception:
        pass
    return ""


def _is_python_edit(edit: Any) -> bool:
    name = ""
    try: name += " " + str(edit.objectName())
    except Exception: pass
    try: name += " " + str(edit.placeholderText())
    except Exception: pass
    text = _safe_text(edit)
    low = (name + " " + text).replace("\\", "/").lower()
    return "python" in low or "python.exe" in low or "/python310/" in low or "/.venv/" in low


def _is_repo_edit(edit: Any) -> bool:
    name = ""
    try: name += " " + str(edit.objectName())
    except Exception: pass
    try: name += " " + str(edit.placeholderText())
    except Exception: pass
    text = _safe_text(edit)
    low = (name + " " + text).replace("\\", "/").lower()
    return "repo" in low or "repository" in low or "/mmdetection_lae" in low or "/mustatil_model_runtimes/lae-dino" in low


def _is_config_edit(edit: Any) -> bool:
    name = ""
    try: name += " " + str(edit.objectName())
    except Exception: pass
    try: name += " " + str(edit.placeholderText())
    except Exception: pass
    text = _safe_text(edit)
    low = (name + " " + text).lower()
    return "config" in low or text.strip().lower().endswith(".py")


def _is_weights_edit(edit: Any) -> bool:
    name = ""
    try: name += " " + str(edit.objectName())
    except Exception: pass
    try: name += " " + str(edit.placeholderText())
    except Exception: pass
    text = _safe_text(edit)
    low = (name + " " + text).lower()
    return "weight" in low or "checkpoint" in low or text.strip().lower().endswith(".pth") or text.strip().lower().endswith(".pt")


def _is_classes_edit(edit: Any) -> bool:
    name = ""
    try: name += " " + str(edit.objectName())
    except Exception: pass
    try: name += " " + str(edit.placeholderText())
    except Exception: pass
    text = _safe_text(edit)
    low = (name + " " + text).lower()
    return "class" in low or "classes" in low or "text" in low or "prompt" in low


def _set_workspace_fields(ws: Any, project: str, config: str, weights: str, classes: List[str]) -> None:
    py = str(_runtime_python())
    repo = str(_runtime_repo())
    class_text = ", ".join(classes) if classes else ""

    # V9 strict plugin fields
    fields = {
        "mustatil_lae_existing_v9_project": project,
        "mustatil_lae_existing_v9_python": py,
        "mustatil_lae_existing_v9_repo": repo,
        "mustatil_lae_existing_v9_config": config,
        "mustatil_lae_existing_v9_weights": weights,
        # V8 compatibility, in case the older strict patch is active.
        "mustatil_lae_existing_v8_project": project,
        "mustatil_lae_existing_v8_python": py,
        "mustatil_lae_existing_v8_repo": repo,
        "mustatil_lae_existing_v8_config": config,
        "mustatil_lae_existing_v8_weights": weights,
        # Older source-runtime canonical fields.
        "mustatil_lae_python": py,
        "mustatil_lae_repo": repo,
        "mustatil_lae_config": config,
        "mustatil_lae_weights": weights,
        "mustatil_lae_checkpoint": weights,
        "mustatil_lae_pretrain": weights,
    }
    if class_text:
        fields.update({
            "mustatil_lae_existing_v9_classes_text": class_text,
            "mustatil_lae_existing_v8_classes_text": class_text,
            "mustatil_lae_classes_text": class_text,
        })
    for k, v in fields.items():
        try:
            setattr(ws, k, v)
        except Exception:
            pass

    try:
        os.environ["MUSTATIL_LAE_DINO_PYTHON"] = py
        os.environ["MUSTATIL_LAE_DINO_REPO"] = repo
        os.environ["MUSTATIL_LAE_DINO_CONFIG"] = config
        os.environ["MUSTATIL_LAE_DINO_WEIGHTS"] = weights
        os.environ["MUSTATIL_LAE_DINO_FORCE_CUSTOM_MODEL"] = "1"
    except Exception:
        pass

    _force_lock_custom_model(ws, project, config, weights)
    _patch_lae_automatch_modules(ws)


def _apply_to_visible_edits(page: Any, config: str, weights: str, classes: List[str]) -> Tuple[int, int, int, int, int]:
    q = _qt()
    QLineEdit = q["QLineEdit"]
    py = str(_runtime_python())
    repo = str(_runtime_repo())
    class_text = ", ".join(classes) if classes else ""

    counts = [0, 0, 0, 0, 0]  # py, repo, cfg, weights, classes
    try:
        edits = page.findChildren(QLineEdit)
    except Exception:
        edits = []
    for e in edits:
        try:
            # Runtime first, otherwise LAE-DINO-PATCH paths can be mis-detected as config-ish.
            if _is_python_edit(e):
                e.setText(py); counts[0] += 1
            elif _is_repo_edit(e):
                e.setText(repo); counts[1] += 1
            elif _is_config_edit(e):
                e.setText(config); counts[2] += 1
            elif _is_weights_edit(e):
                e.setText(weights); counts[3] += 1
            elif class_text and _is_classes_edit(e):
                e.setText(class_text); counts[4] += 1
        except Exception:
            pass
    return tuple(counts)  # type: ignore[return-value]


def _looks_like_lae_page(page: Any, tab_label: str = "") -> bool:
    try:
        obj = str(page.objectName()).lower()
    except Exception:
        obj = ""
    blob = (str(tab_label or "") + " " + obj).lower()
    if "train" in blob or "trainer" in blob:
        return False
    if "lae" in blob and "dino" in blob:
        return True
    try:
        # The V9 strict tab has this objectName.
        if "mustatillaedinoexistingtab" in obj:
            return True
    except Exception:
        pass
    return False




# ============================================================
# V3: force selected custom model against V8/V9 AutoMatch override
# ============================================================

_PATCHED_LAE_MODULES = set()


def _selected_custom_pair(ws: Any) -> Optional[Tuple[Path, Path]]:
    """Return selected custom config/checkpoint if user explicitly locked it."""
    try:
        locked = (
            bool(getattr(ws, "mustatil_lae_force_custom_model", False))
            or bool(getattr(ws, "mustatil_lae_custom_model_locked", False))
            or bool(getattr(ws, "mustatil_lae_existing_v8_force_custom_model", False))
            or bool(getattr(ws, "mustatil_lae_existing_v9_force_custom_model", False))
        )
    except Exception:
        locked = False
    if not locked:
        return None

    cfg_candidates = [
        "mustatil_lae_existing_v9_config",
        "mustatil_lae_existing_v8_config",
        "mustatil_lae_config",
    ]
    weight_candidates = [
        "mustatil_lae_existing_v9_weights",
        "mustatil_lae_existing_v8_weights",
        "mustatil_lae_weights",
        "mustatil_lae_checkpoint",
    ]

    cfg = ""
    weights = ""
    for name in cfg_candidates:
        try:
            v = str(getattr(ws, name, "") or "").strip().strip('"')
            if v and Path(v).exists():
                cfg = v
                break
        except Exception:
            pass
    for name in weight_candidates:
        try:
            v = str(getattr(ws, name, "") or "").strip().strip('"')
            if v and Path(v).exists():
                weights = v
                break
        except Exception:
            pass

    if cfg and weights:
        return Path(cfg), Path(weights)
    return None


def _force_lock_custom_model(ws: Any, project: str, config: str, weights: str) -> None:
    """Mark the selected custom model as locked so V8/V9 AutoMatch may not replace it."""
    flags = [
        "mustatil_lae_force_custom_model",
        "mustatil_lae_custom_model_locked",
        "mustatil_lae_existing_v8_force_custom_model",
        "mustatil_lae_existing_v9_force_custom_model",
    ]
    for f in flags:
        try:
            setattr(ws, f, True)
        except Exception:
            pass
    try:
        setattr(ws, "mustatil_lae_custom_project", project)
        setattr(ws, "mustatil_lae_custom_config", config)
        setattr(ws, "mustatil_lae_custom_weights", weights)
    except Exception:
        pass


def _runtime_mmdet_dir(repo: Path) -> Path:
    return repo / "mmdetection_lae" if (repo / "mmdetection_lae").exists() else repo


def _force_runtime_paths_for_module(mod: Any, ws: Any):
    """Replacement/wrapper for V8/V9 _runtime_paths(ws).

    If a custom model is locked, return exactly that config/checkpoint pair.
    Otherwise delegate to the original module _runtime_paths.
    """
    pair = _selected_custom_pair(ws)
    if pair is not None:
        cfg, weights = pair
        py = Path(str(getattr(ws, "mustatil_lae_existing_v9_python", "") or getattr(ws, "mustatil_lae_existing_v8_python", "") or getattr(ws, "mustatil_lae_python", "") or _runtime_python()).strip().strip('"'))
        repo = Path(str(getattr(ws, "mustatil_lae_existing_v9_repo", "") or getattr(ws, "mustatil_lae_existing_v8_repo", "") or getattr(ws, "mustatil_lae_repo", "") or _runtime_repo()).strip().strip('"'))
        mmd = None
        demo = None
        try:
            if hasattr(mod, "_mmdet_dir"):
                mmd = mod._mmdet_dir(repo)
        except Exception:
            mmd = None
        if mmd is None:
            mmd = _runtime_mmdet_dir(repo)
        try:
            if hasattr(mod, "_demo_path"):
                demo = mod._demo_path(repo)
        except Exception:
            demo = None
        if demo is None:
            demo = mmd / "demo" / "image_demo.py"

        if "LAE-DINO\\.venv" in str(py) or "LAE-DINO/.venv" in str(py).replace("\\", "/"):
            py = _runtime_python()
        if not py.exists():
            raise RuntimeError("LAE-DINO-PATCH Python missing: " + str(py))
        if not demo.exists():
            raise RuntimeError("LAE-DINO image_demo.py missing: " + str(demo))
        if not cfg.exists():
            raise RuntimeError("Selected custom LAE-DINO config missing: " + str(cfg))
        if not weights.exists():
            raise RuntimeError("Selected custom LAE-DINO checkpoint missing: " + str(weights))

        try:
            if hasattr(ws, "log"):
                ws.log("[LAE-DINO Custom Selector V3] FORCE selected custom config: " + str(cfg))
                ws.log("[LAE-DINO Custom Selector V3] FORCE selected custom checkpoint: " + str(weights))
        except Exception:
            pass

        return py, mmd, demo, cfg, weights

    orig = getattr(mod, "_mustatil_custom_selector_orig_runtime_paths", None)
    if orig is not None:
        return orig(ws)
    raise RuntimeError("Original LAE-DINO _runtime_paths not available.")


def _force_auto_match_for_module(mod: Any, ws: Any, status_label=None, preferred_weight: Optional[Path] = None):
    """Replacement/wrapper for V8/V9 _auto_match_lae_paths(ws,...).

    V8 was overriding project models with DIOR because its repo AutoMatch scored
    the pretrained DIOR pair higher. This function blocks that when custom model
    lock is active.
    """
    pair = _selected_custom_pair(ws)
    if pair is not None:
        cfg, weights = pair
        msg = "Custom model lock active: keeping selected checkpoint/config; AutoMatch not allowed to replace it."
        try:
            if hasattr(ws, "log"):
                ws.log("[LAE-DINO Custom Selector V3] " + msg)
        except Exception:
            pass
        try:
            if status_label is not None and hasattr(status_label, "setText"):
                status_label.setText(msg)
        except Exception:
            pass
        return cfg, weights

    orig = getattr(mod, "_mustatil_custom_selector_orig_auto_match", None)
    if orig is not None:
        return orig(ws, status_label, preferred_weight)
    raise RuntimeError("Original LAE-DINO _auto_match_lae_paths not available.")


def _patch_lae_automatch_modules(ws: Any = None) -> int:
    """Patch already loaded V8/V9 strict LAE-DINO modules.

    The patch is deliberately narrow:
      - only modules exposing _runtime_paths and/or _auto_match_lae_paths
      - only affects behavior when custom model lock is active
    """
    import sys
    count = 0
    for name, mod in list(sys.modules.items()):
        try:
            if mod is None:
                continue
            mod_file = str(getattr(mod, "__file__", "") or "").replace("\\", "/").lower()
            mod_name = str(name).lower()
            looks_lae = (
                "lae_dino" in mod_file
                or "lae-dino" in mod_file
                or "lae_dino" in mod_name
                or "lae" in mod_name and "dino" in mod_name
            )
            if not looks_lae:
                continue

            patched_any = False
            if hasattr(mod, "_runtime_paths") and not hasattr(mod, "_mustatil_custom_selector_orig_runtime_paths"):
                orig_runtime = getattr(mod, "_runtime_paths")
                setattr(mod, "_mustatil_custom_selector_orig_runtime_paths", orig_runtime)

                def runtime_wrapper(_ws, _mod=mod):
                    return _force_runtime_paths_for_module(_mod, _ws)

                setattr(mod, "_runtime_paths", runtime_wrapper)
                patched_any = True

            if hasattr(mod, "_auto_match_lae_paths") and not hasattr(mod, "_mustatil_custom_selector_orig_auto_match"):
                orig_auto = getattr(mod, "_auto_match_lae_paths")
                setattr(mod, "_mustatil_custom_selector_orig_auto_match", orig_auto)

                def auto_wrapper(_ws, _status_label=None, _preferred_weight=None, _mod=mod):
                    return _force_auto_match_for_module(_mod, _ws, _status_label, _preferred_weight)

                setattr(mod, "_auto_match_lae_paths", auto_wrapper)
                patched_any = True

            if patched_any:
                _PATCHED_LAE_MODULES.add(name)
                count += 1
        except Exception:
            pass

    try:
        if ws is not None and count and hasattr(ws, "log"):
            ws.log(f"[LAE-DINO Custom Selector V3] Patched {count} LAE-DINO AutoMatch/runtime module(s).")
    except Exception:
        pass
    return count


def _click_existing_lae_run_button(page: Any, self_button: Any = None) -> bool:
    """Find and click the existing V8/V9 LAE-DINO Run button in the same tab.

    This add-on intentionally does not duplicate the inference implementation.
    It applies the selected model fields, then triggers the original button so
    the proven V9 strict pipeline / satellite logic stays in control.
    """
    try:
        q = _qt()
        QPushButton = q["QPushButton"]
        buttons = page.findChildren(QPushButton)
    except Exception:
        buttons = []

    candidates = []
    for b in buttons:
        if self_button is not None and b is self_button:
            continue
        try:
            txt = str(b.text() or "").strip()
        except Exception:
            txt = ""
        low = txt.lower()
        if not txt:
            continue

        # Positive match: the real button normally contains Run + LAE-DINO.
        if ("run" in low or "detect" in low or "start" in low) and ("lae" in low or "dino" in low):
            # Exclude helper/config buttons.
            bad = ("runtime", "check", "auto", "match", "import", "select", "apply", "build", "install", "repair", "browse")
            if not any(x in low for x in bad):
                candidates.append((0, b, txt))
            elif "run" in low and ("lae" in low or "dino" in low):
                candidates.append((1, b, txt))

    # If button text is only "Run", accept it if it is inside a LAE-DINO tab.
    if not candidates:
        for b in buttons:
            if self_button is not None and b is self_button:
                continue
            try:
                txt = str(b.text() or "").strip()
            except Exception:
                txt = ""
            low = txt.lower()
            if low in {"run", "start", "detect"} or low.startswith("run "):
                candidates.append((2, b, txt))

    if not candidates:
        return False

    candidates.sort(key=lambda x: x[0])
    btn = candidates[0][1]
    try:
        btn.click()
        return True
    except Exception:
        try:
            btn.animateClick()
            return True
        except Exception:
            return False


def _inject(page: Any, tab_label: str = "", tab_widget: Any = None) -> bool:
    if page is None or id(page) in _PATCHED_PAGES:
        return False
    if not _looks_like_lae_page(page, tab_label):
        return False

    q = _qt()
    QWidget = q["QWidget"]
    QVBoxLayout = q["QVBoxLayout"]
    QGridLayout = q["QGridLayout"]
    QHBoxLayout = q["QHBoxLayout"]
    QLabel = q["QLabel"]
    QLineEdit = q["QLineEdit"]
    QPushButton = q["QPushButton"]
    QFileDialog = q["QFileDialog"]
    QGroupBox = q["QGroupBox"]

    try:
        for ch in page.findChildren(QWidget):
            try:
                if str(ch.objectName()) == "MustatilLAEDINOCustomTrainedModelSelectorBox":
                    _PATCHED_PAGES.add(id(page))
                    return False
            except Exception:
                pass
    except Exception:
        pass

    ws = _workspace_from_widget(page)
    if ws is None:
        return False

    lay = page.layout()
    if lay is None:
        lay = QVBoxLayout(page)

    box = QGroupBox("Custom trained model selector")
    box.setObjectName("MustatilLAEDINOCustomTrainedModelSelectorBox")
    grid = QGridLayout(box)

    desc = QLabel("Wählt ein eigenes trainiertes LAE-DINO-Modell und setzt Python/Repo/Config/Checkpoint in diesem vorhandenen LAE-DINO-Tab.")
    desc.setWordWrap(True)
    grid.addWidget(desc, 0, 0, 1, 4)

    project_edit = QLineEdit(_guess_project(ws, page))
    project_edit.setObjectName("MustatilLAEProjectForCustomModel")
    config_edit = QLineEdit("")
    config_edit.setObjectName("MustatilLAECustomModelConfig")
    weights_edit = QLineEdit("")
    weights_edit.setObjectName("MustatilLAECustomModelWeights")
    status = QLabel("Projekt auswählen oder Config/Checkpoint manuell wählen.")
    status.setWordWrap(True)

    def browse_project():
        start = project_edit.text().strip() or str(Path.home())
        p = QFileDialog.getExistingDirectory(page, "Select Mustatil project folder", start)
        if p:
            project_edit.setText(p)
            _emit(ws, "Project selected: " + p, status, page)

    def browse_config():
        start = str(Path(config_edit.text()).parent) if config_edit.text().strip() else project_edit.text().strip() or str(Path.home())
        p, _ = QFileDialog.getOpenFileName(page, "Select LAE-DINO config", start, "Python config (*.py);;All files (*)")
        if p:
            config_edit.setText(p)
            _emit(ws, "Config selected: " + p, status, page)

    def browse_weights():
        start = str(Path(weights_edit.text()).parent) if weights_edit.text().strip() else project_edit.text().strip() or str(Path.home())
        p, _ = QFileDialog.getOpenFileName(page, "Select LAE-DINO checkpoint", start, "Checkpoint (*.pth *.pt);;All files (*)")
        if p:
            weights_edit.setText(p)
            _emit(ws, "Checkpoint selected: " + p, status, page)

    def auto_find_from_project():
        project = Path(project_edit.text().strip().strip('"'))
        if not project.exists():
            _emit(ws, "Project folder missing: " + str(project), status, page)
            return
        configs = _find_configs(project)
        checkpoints = _find_checkpoints(project)
        if configs:
            config_edit.setText(str(configs[0]))
        if checkpoints:
            weights_edit.setText(str(checkpoints[0]))
        msg = []
        msg.append("config=" + (configs[0].name if configs else "NOT FOUND"))
        msg.append("checkpoint=" + (checkpoints[0].name if checkpoints else "NOT FOUND"))
        classes = _read_project_classes(project)
        if classes:
            msg.append("classes=" + ", ".join(classes))
        _emit(ws, "Auto-found project model: " + "; ".join(msg), status, page)

    def apply_selected():
        project = project_edit.text().strip().strip('"')
        config = config_edit.text().strip().strip('"')
        weights = weights_edit.text().strip().strip('"')
        problems = []
        if not config or not Path(config).exists():
            problems.append("Config missing")
        if not weights or not Path(weights).exists():
            problems.append("Checkpoint missing")
        if problems:
            _emit(ws, "Cannot apply: " + "; ".join(problems), status, page)
            return
        classes = _read_project_classes(Path(project)) if project else []
        _set_workspace_fields(ws, project, config, weights, classes)
        py_n, repo_n, cfg_n, w_n, cls_n = _apply_to_visible_edits(page, config, weights, classes)
        _emit(
            ws,
            "Custom model applied. "
            f"python={py_n}, repo={repo_n}, config={cfg_n}, checkpoint={w_n}, classes={cls_n}. "
            f"Run LAE-DINO now. Custom-model lock is ACTIVE; V8/V9 AutoMatch cannot replace it. Runtime={_runtime_python()}",
            status,
            page,
        )

    def auto_find_and_apply():
        auto_find_from_project()
        apply_selected()

    def run_with_selected_model():
        # Do not auto-find here; respect manual config/checkpoint if user selected them.
        apply_selected()
        _patch_lae_automatch_modules(ws)
        ok = _click_existing_lae_run_button(page, b_run_selected)
        if ok:
            _emit(ws, "Run with selected model gestartet über den vorhandenen LAE-DINO-Run-Button.", status, page)
        else:
            _emit(ws, "Modell wurde gesetzt, aber kein vorhandener LAE-DINO-Run-Button wurde gefunden. Bitte den normalen Run-LAE-DINO-Button darunter drücken.", status, page)

    grid.addWidget(QLabel("Project"), 1, 0)
    grid.addWidget(project_edit, 1, 1)
    b_proj = QPushButton("Select project")
    b_proj.clicked.connect(browse_project)
    grid.addWidget(b_proj, 1, 2)
    b_auto = QPushButton("Auto find")
    b_auto.clicked.connect(auto_find_from_project)
    grid.addWidget(b_auto, 1, 3)

    grid.addWidget(QLabel("Config .py"), 2, 0)
    grid.addWidget(config_edit, 2, 1, 1, 2)
    b_cfg = QPushButton("...")
    b_cfg.clicked.connect(browse_config)
    grid.addWidget(b_cfg, 2, 3)

    grid.addWidget(QLabel("Checkpoint .pth"), 3, 0)
    grid.addWidget(weights_edit, 3, 1, 1, 2)
    b_w = QPushButton("...")
    b_w.clicked.connect(browse_weights)
    grid.addWidget(b_w, 3, 3)

    b_import = QPushButton("Import trained project model")
    b_import.clicked.connect(auto_find_and_apply)
    grid.addWidget(b_import, 4, 0, 1, 2)

    b_apply = QPushButton("Apply selected config/checkpoint")
    b_apply.clicked.connect(apply_selected)
    grid.addWidget(b_apply, 4, 2, 1, 2)

    b_run_selected = QPushButton("Run with selected model")
    b_run_selected.setObjectName("MustatilLAERunWithSelectedModelButton")
    b_run_selected.clicked.connect(run_with_selected_model)
    grid.addWidget(b_run_selected, 5, 0, 1, 4)

    grid.addWidget(status, 6, 0, 1, 4)

    # Insert near top, but below existing header if possible.
    try:
        lay.insertWidget(1, box)
    except Exception:
        try:
            lay.addWidget(box)
        except Exception:
            return False

    _PATCHED_PAGES.add(id(page))
    _emit(ws, "Custom selector inserted.", status, page)
    return True


def _scan(root: Any = None) -> None:
    try:
        q = _qt()
        QApplication = q["QApplication"]
        QTabWidget = q["QTabWidget"]
    except Exception:
        return

    widgets = []
    try:
        if root is not None:
            if isinstance(root, QTabWidget):
                widgets.append(root)
            widgets.extend(root.findChildren(QTabWidget))
    except Exception:
        pass

    try:
        app = QApplication.instance()
        if app is not None:
            for w in app.allWidgets():
                try:
                    if isinstance(w, QTabWidget):
                        widgets.append(w)
                except Exception:
                    pass
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
            try:
                label = str(tw.tabText(i) or "")
                page = tw.widget(i)
                if "lae" in label.lower() and "dino" in label.lower():
                    _inject(page, label, tw)
                else:
                    # Also inspect V9 objectName even if tab label differs.
                    try:
                        if page is not None and "mustatillaedinoexistingtab" in str(page.objectName()).lower():
                            _inject(page, label, tw)
                    except Exception:
                        pass
            except Exception:
                traceback.print_exc()


def _install_hook() -> bool:
    global _PATCHED_QTAB, _ORIG_ADD, _ORIG_INSERT
    if _PATCHED_QTAB:
        return True
    try:
        q = _qt()
        QTabWidget = q["QTabWidget"]
        QTimer = q["QTimer"]
    except Exception as exc:
        _log("Qt unavailable: " + str(exc))
        return False

    _ORIG_ADD = QTabWidget.addTab
    _ORIG_INSERT = QTabWidget.insertTab

    def schedule(label: str = ""):
        try:
            if "lae" in str(label).lower() or "dino" in str(label).lower():
                QTimer.singleShot(250, lambda: _scan())
                QTimer.singleShot(1200, lambda: _scan())
        except Exception:
            pass

    def addTab_patched(self, page, *args, **kwargs):
        res = _ORIG_ADD(self, page, *args, **kwargs)
        label = ""
        for a in reversed(args):
            if isinstance(a, str):
                label = a
                break
        if not label:
            try:
                label = self.tabText(int(res))
            except Exception:
                pass
        schedule(label)
        return res

    def insertTab_patched(self, index, page, *args, **kwargs):
        res = _ORIG_INSERT(self, index, page, *args, **kwargs)
        label = ""
        for a in reversed(args):
            if isinstance(a, str):
                label = a
                break
        if not label:
            try:
                label = self.tabText(int(res))
            except Exception:
                pass
        schedule(label)
        return res

    QTabWidget.addTab = addTab_patched
    QTabWidget.insertTab = insertTab_patched
    _PATCHED_QTAB = True

    try:
        QTimer.singleShot(500, lambda: (_patch_lae_automatch_modules(), _scan()))
        QTimer.singleShot(2000, lambda: (_patch_lae_automatch_modules(), _scan()))
        QTimer.singleShot(5000, lambda: (_patch_lae_automatch_modules(), _scan()))
    except Exception:
        pass

    _log("hook installed")
    return True


def mustatil_plugin_init():
    _install_hook()
    _scan()
    return True


def register_plugin(app=None, main_window=None):
    _install_hook()
    _patch_lae_automatch_modules()
    _scan(main_window or app)
    return True


def init_plugin(app=None, main_window=None):
    return register_plugin(app, main_window)


def load_plugin(app=None, main_window=None):
    return register_plugin(app, main_window)


try:
    _install_hook()
except Exception as exc:
    _log("install failed: " + str(exc))
