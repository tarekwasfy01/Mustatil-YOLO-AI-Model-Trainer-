# -*- coding: utf-8 -*-
"""
Mustatil add-on plugin: compact LAE-DINO config/checkpoint rows for Video Detection.

Drop this file into mustatil_plugins in addition to the Video Detection plugin.
It scans for the Video Detection -> LAE-DINO tab and inserts two compact visible rows:

    Config      [path field] [...] 
    Checkpoint  [path field] [...] [Import from project]

It also hides the older "buttons only" LAE-DINO button grid when present, so the
LAE-DINO area stays small and usable. The selected paths are synchronized into
hidden LAE-DINO fields and common Mustatil LAE attributes/environment variables.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

PLUGIN_NAME = "Mustatil Video LAE-DINO Config/Checkpoint Rows Patch v1"
_PATCHED = False
_SCAN_TIMER = None
_ORIG_ADD_TAB = None
_ORIG_INSERT_TAB = None


def _import_qt():
    """Prefer Mustatil's existing Qt binding; fall back safely."""
    for prefix in ("PySide6", "PyQt6", "PySide2", "PyQt5"):
        try:
            if prefix.startswith("PySide"):
                qtwidgets = __import__(prefix + ".QtWidgets", fromlist=["*"])
                qtcore = __import__(prefix + ".QtCore", fromlist=["*"])
                qtgui = __import__(prefix + ".QtGui", fromlist=["*"])
            else:
                qtwidgets = __import__(prefix + ".QtWidgets", fromlist=["*"])
                qtcore = __import__(prefix + ".QtCore", fromlist=["*"])
                qtgui = __import__(prefix + ".QtGui", fromlist=["*"])
            return prefix, qtwidgets, qtcore, qtgui
        except Exception:
            continue
    return None, None, None, None

_QT_PREFIX, QtWidgets, QtCore, QtGui = _import_qt()


def _log(msg: str) -> None:
    try:
        print(f"[{PLUGIN_NAME}] {msg}")
    except Exception:
        pass


def _norm(s: Any) -> str:
    return str(s or "").strip().lower().replace("_", " ").replace("-", " ")


def _get_app():
    try:
        return QtWidgets.QApplication.instance() if QtWidgets is not None else None
    except Exception:
        return None


def _all_tabwidgets() -> List[Any]:
    app = _get_app()
    if app is None:
        return []
    out: List[Any] = []
    try:
        tops = list(app.topLevelWidgets())
    except Exception:
        tops = []
    seen = set()
    for top in tops:
        try:
            widgets = [top] if isinstance(top, QtWidgets.QTabWidget) else []
            widgets += list(top.findChildren(QtWidgets.QTabWidget))
            for tw in widgets:
                ident = id(tw)
                if ident not in seen:
                    seen.add(ident)
                    out.append(tw)
        except Exception:
            pass
    return out


def _tab_texts(tw: Any) -> List[str]:
    try:
        return [str(tw.tabText(i) or "") for i in range(int(tw.count()))]
    except Exception:
        return []


def _tab_index(tw: Any, names: Iterable[str]) -> int:
    wanted = [_norm(n) for n in names]
    try:
        for i in range(int(tw.count())):
            txt = _norm(tw.tabText(i))
            if any(w in txt for w in wanted):
                return i
    except Exception:
        pass
    return -1


def _widget_lineage(w: Any, limit: int = 12) -> List[Any]:
    arr = []
    cur = w
    for _ in range(limit):
        if cur is None:
            break
        arr.append(cur)
        try:
            cur = cur.parentWidget()
        except Exception:
            break
    return arr


def _find_video_page_for_lae_tab(lae_tab: Any) -> Optional[Any]:
    # Usually: LAE tab -> model QTabWidget -> left/control page -> Video Detection page.
    for anc in _widget_lineage(lae_tab, 20):
        try:
            if getattr(anc, "objectName", lambda: "")() in (
                "MustatilVideoDetectionPage",
                "MustatilVideoDetectionPageV15",
                "MustatilVideoDetectionPageV14",
            ):
                return anc
        except Exception:
            pass
    # Fallback: first ancestor whose children include the video preview label or frame controls.
    for anc in _widget_lineage(lae_tab, 20):
        try:
            texts = [str(x.text()) for x in anc.findChildren(QtWidgets.QLabel)[:200] if hasattr(x, "text")]
            joined = "\n".join(texts).lower()
            if "video preview" in joined or "frame review" in joined or "detect full video" in joined:
                return anc
        except Exception:
            pass
    return None


def _set_text(widget: Any, text: str) -> None:
    try:
        widget.setText(str(text or ""))
    except Exception:
        pass


def _text(widget: Any) -> str:
    try:
        return str(widget.text() or "").strip()
    except Exception:
        return ""


def _short(path_text: str, max_len: int = 72) -> str:
    s = str(path_text or "").strip()
    if len(s) <= max_len:
        return s
    return "..." + s[-(max_len - 3):]


def _set_status_near(page: Any, text: str) -> None:
    # Try common status labels/log functions but keep silent if none exist.
    for obj in _widget_lineage(page, 15):
        for name in ("set_status", "append_log", "log", "_log"):
            try:
                fn = getattr(obj, name, None)
                if callable(fn):
                    fn(str(text))
                    return
            except Exception:
                pass
    _log(text)


def _sync_paths(lae_tab: Any, cfg: str = "", ckpt: str = "", project: str = "") -> None:
    """Push selected paths into visible fields, hidden fields, attributes and env."""
    cfg = str(cfg or "").strip()
    ckpt = str(ckpt or "").strip()
    project = str(project or "").strip()

    try:
        if cfg:
            os.environ["MUSTATIL_LAE_DINO_CONFIG"] = cfg
        if ckpt:
            os.environ["MUSTATIL_LAE_DINO_CHECKPOINT"] = ckpt
        if project:
            os.environ["MUSTATIL_LAE_DINO_PROJECT"] = project
    except Exception:
        pass

    page = _find_video_page_for_lae_tab(lae_tab) or lae_tab

    # Store on likely relevant objects. This helps older/newer Mustatil plugins read the same paths.
    targets = []
    for obj in _widget_lineage(lae_tab, 20):
        if obj not in targets:
            targets.append(obj)
    if page not in targets:
        targets.append(page)

    cfg_names = (
        "mustatil_lae_config", "mustatil_lae_dino_config", "mustatil_lae_config_path",
        "mustatil_lae_dino_config_path", "mustatil_lae_existing_v9_config",
        "mustatil_lae_existing_v8_config", "lae_config", "lae_config_path",
        "lae_dino_config", "lae_dino_config_path", "_mustatil_video_lae_config",
    )
    ckpt_names = (
        "mustatil_lae_weights", "mustatil_lae_checkpoint", "mustatil_lae_dino_checkpoint",
        "mustatil_lae_checkpoint_path", "mustatil_lae_weights_path",
        "mustatil_lae_existing_v9_weights", "mustatil_lae_existing_v8_weights",
        "lae_weights", "lae_weights_path", "lae_checkpoint", "lae_checkpoint_path",
        "lae_dino_checkpoint", "lae_dino_checkpoint_path", "_mustatil_video_lae_checkpoint",
    )
    project_names = (
        "mustatil_lae_project", "mustatil_lae_existing_v9_project", "mustatil_lae_existing_v8_project",
        "lae_project", "lae_project_path", "_mustatil_video_lae_project",
    )
    for target in targets:
        try:
            if cfg:
                for n in cfg_names:
                    setattr(target, n, cfg)
            if ckpt:
                for n in ckpt_names:
                    setattr(target, n, ckpt)
            if project:
                for n in project_names:
                    setattr(target, n, project)
        except Exception:
            pass

    # Synchronize hidden QLineEdits in the existing LAE tab. In v14/v15 the original
    # runtime fields are hidden, so this is the most important integration point.
    try:
        lines = list(lae_tab.findChildren(QtWidgets.QLineEdit))
        hidden = [le for le in lines if not le.isVisible()]
        named_cfg = [le for le in lines if "config" in _norm(getattr(le, "objectName", lambda: "")())]
        named_ckpt = [le for le in lines if any(k in _norm(getattr(le, "objectName", lambda: "")()) for k in ("checkpoint", "ckpt", "weight"))]
        if cfg:
            for le in named_cfg[:2]:
                _set_text(le, cfg)
            if not named_cfg and len(hidden) >= 1:
                _set_text(hidden[0], cfg)
        if ckpt:
            for le in named_ckpt[:2]:
                _set_text(le, ckpt)
            if not named_ckpt and len(hidden) >= 2:
                _set_text(hidden[1], ckpt)
    except Exception:
        pass


def _current_project_candidates() -> List[str]:
    cand: List[str] = []
    # Env first.
    for key in ("MUSTATIL_PROJECT", "MUSTATIL_PROJECT_DIR", "MUSTATIL_CURRENT_PROJECT", "MUSTATIL_LAE_DINO_PROJECT"):
        v = os.environ.get(key, "")
        if v:
            cand.append(v)
    # Attributes on all top-level widgets and their children.
    attr_names = (
        "project_dir", "project_folder", "project_path", "current_project_dir", "current_project_folder",
        "current_project_path", "mustatil_project_dir", "mustatil_project_folder", "mustatil_project_path",
        "lae_project", "lae_project_path", "_mustatil_video_lae_project",
    )
    app = _get_app()
    tops = []
    try:
        tops = list(app.topLevelWidgets()) if app is not None else []
    except Exception:
        tops = []
    for top in tops:
        objs = [top]
        try:
            objs += list(top.findChildren(QtWidgets.QWidget))[:1500]
        except Exception:
            pass
        for obj in objs:
            for name in attr_names:
                try:
                    v = getattr(obj, name, None)
                    if isinstance(v, (str, os.PathLike)) and str(v):
                        cand.append(str(v))
                except Exception:
                    pass
    # Current working directory as last resort.
    try:
        cand.append(os.getcwd())
    except Exception:
        pass
    out = []
    seen = set()
    for c in cand:
        try:
            p = Path(str(c)).expanduser()
            if p.is_file():
                p = p.parent
            if p.exists() and p.is_dir():
                s = str(p.resolve())
                if s not in seen:
                    seen.add(s); out.append(s)
        except Exception:
            pass
    return out


def _score_config(p: Path) -> int:
    s = str(p).lower().replace("\\", "/")
    name = p.name.lower()
    score = 0
    if p.suffix.lower() == ".py": score += 20
    if p.suffix.lower() in (".yaml", ".yml", ".json"): score += 12
    if "config" in name or "cfg" in name: score += 50
    if "lae" in s or "dino" in s: score += 40
    if "work_dirs" in s or "work-dir" in s: score += 25
    if "dataset" in s or "project" in s: score += 10
    if "__pycache__" in s or "site-packages" in s or "venv" in s: score -= 80
    try:
        score += min(20, int(p.stat().st_mtime // 86400) % 20)  # weak tie breaker
    except Exception:
        pass
    return score


def _score_checkpoint(p: Path) -> int:
    s = str(p).lower().replace("\\", "/")
    name = p.name.lower()
    score = 0
    if p.suffix.lower() in (".pth", ".pt", ".ckpt"): score += 30
    if any(x in name for x in ("latest", "best", "final", "epoch", "checkpoint", "ckpt")): score += 60
    if "lae" in s or "dino" in s: score += 40
    if "work_dirs" in s or "runs" in s or "weights" in s: score += 35
    if "__pycache__" in s or "site-packages" in s or "venv" in s: score -= 80
    try:
        score += min(40, int(time.time() - p.stat().st_mtime) * -1 // 86400)  # newer weakly preferred
    except Exception:
        pass
    return score


def _find_lae_files(project: str) -> Tuple[str, str]:
    root = Path(project)
    configs: List[Path] = []
    ckpts: List[Path] = []
    visited = 0
    skip_dirs = {".git", "__pycache__", ".venv", "venv", "env", "node_modules", "site-packages"}
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
            visited += 1
            if visited > 6000:
                break
            dp = Path(dirpath)
            lower_dp = str(dp).lower().replace("\\", "/")
            for fn in filenames:
                p = dp / fn
                suffix = p.suffix.lower()
                lfn = fn.lower()
                if suffix in (".py", ".yaml", ".yml", ".json") and ("config" in lfn or "cfg" in lfn or "lae" in lower_dp or "dino" in lower_dp):
                    configs.append(p)
                elif suffix in (".pth", ".pt", ".ckpt"):
                    ckpts.append(p)
    except Exception:
        pass
    cfg = ""
    ckpt = ""
    if configs:
        cfg = str(sorted(configs, key=_score_config, reverse=True)[0])
    if ckpts:
        ckpt = str(sorted(ckpts, key=_score_checkpoint, reverse=True)[0])
    return cfg, ckpt


def _choose_project_dialog(parent: Any) -> str:
    try:
        start = _current_project_candidates()[0] if _current_project_candidates() else ""
        d = QtWidgets.QFileDialog.getExistingDirectory(parent, "Select Mustatil / LAE-DINO project folder", start)
        return str(d or "")
    except Exception:
        return ""


def _hide_old_lae_buttons(lae_tab: Any, our_frame: Any) -> None:
    """Hide v14/v15 button-only LAE grid so the new two-row editor is not squeezed."""
    try:
        parents = set()
        for btn in lae_tab.findChildren(QtWidgets.QPushButton):
            if btn is None or btn is our_frame:
                continue
            try:
                if our_frame is not None and our_frame.isAncestorOf(btn):
                    continue
            except Exception:
                pass
            txt = _norm(btn.text() if hasattr(btn, "text") else "")
            if (txt.startswith("1 import from project") or txt.startswith("2 config") or txt.startswith("3 checkpoint")
                    or txt.startswith("4 project") or txt in ("clear", "ready no", "ready yes") or txt.startswith("ready")):
                try:
                    parents.add(btn.parentWidget())
                    btn.setVisible(False)
                except Exception:
                    pass
        for p in list(parents):
            if p is not None and p is not lae_tab:
                try:
                    if our_frame is not None and (p is our_frame or our_frame.isAncestorOf(p) or p.isAncestorOf(our_frame)):
                        continue
                except Exception:
                    pass
                # Hide only anonymous grid holders; leave real LAE tab/page visible.
                try:
                    if _norm(p.objectName()) in ("", "qt_scrollarea_viewport"):
                        p.setVisible(False)
                except Exception:
                    pass
    except Exception:
        pass


def _install_rows_on_lae_tab(lae_tab: Any) -> bool:
    try:
        if getattr(lae_tab, "_mustatil_lae_rows_patch_v1", False):
            return False
        if getattr(lae_tab, "objectName", lambda: "")() == "MustatilVideoLAERowsPatchFrame":
            return False

        # Confirm this really looks like a LAE-DINO tab.
        text_blob = ""
        try:
            text_blob += " ".join([str(x.text()) for x in lae_tab.findChildren(QtWidgets.QLabel)[:120] if hasattr(x, "text")])
            text_blob += " " + " ".join([str(x.text()) for x in lae_tab.findChildren(QtWidgets.QPushButton)[:120] if hasattr(x, "text")])
        except Exception:
            pass
        if "lae" not in _norm(text_blob + " " + str(getattr(lae_tab, "objectName", lambda: "")())) and "dino" not in _norm(text_blob):
            # If it is the QTabWidget page named LAE-DINO, the caller can still force install.
            pass

        # Build compact frame.
        frame = QtWidgets.QFrame(lae_tab)
        frame.setObjectName("MustatilVideoLAERowsPatchFrame")
        try:
            frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        except Exception:
            try:
                frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
            except Exception:
                pass
        frame.setMaximumHeight(84)
        frame.setMinimumHeight(64)
        frame.setStyleSheet(
            "QFrame#MustatilVideoLAERowsPatchFrame{background:#f6f6f6;border:1px solid #c9c9c9;}"
            "QLineEdit{min-height:20px; max-height:24px;}"
            "QPushButton{min-height:22px; max-height:26px; padding:2px 6px;}"
            "QLabel{padding:0px;}"
        )

        grid = QtWidgets.QGridLayout(frame)
        grid.setContentsMargins(6, 4, 6, 4)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(3)

        cfg_line = QtWidgets.QLineEdit(frame)
        ckpt_line = QtWidgets.QLineEdit(frame)
        cfg_line.setObjectName("mustatil_video_lae_config_visible_line")
        ckpt_line.setObjectName("mustatil_video_lae_checkpoint_visible_line")
        cfg_line.setPlaceholderText("LAE-DINO config (.py/.yaml/.json)")
        ckpt_line.setPlaceholderText("LAE-DINO checkpoint (.pth/.pt/.ckpt)")

        cfg_browse = QtWidgets.QPushButton("...", frame)
        ckpt_browse = QtWidgets.QPushButton("...", frame)
        import_btn = QtWidgets.QPushButton("Import from project", frame)
        cfg_browse.setFixedWidth(30)
        ckpt_browse.setFixedWidth(30)
        import_btn.setMaximumWidth(150)

        # Pre-fill from hidden fields if available.
        try:
            hidden = [le for le in lae_tab.findChildren(QtWidgets.QLineEdit) if le is not cfg_line and le is not ckpt_line and not le.isVisible()]
            if len(hidden) >= 1 and _text(hidden[0]):
                _set_text(cfg_line, _text(hidden[0]))
            if len(hidden) >= 2 and _text(hidden[1]):
                _set_text(ckpt_line, _text(hidden[1]))
        except Exception:
            pass

        grid.addWidget(QtWidgets.QLabel("Config", frame), 0, 0)
        grid.addWidget(cfg_line, 0, 1)
        grid.addWidget(cfg_browse, 0, 2)
        grid.addWidget(import_btn, 0, 3, 2, 1)
        grid.addWidget(QtWidgets.QLabel("Checkpoint", frame), 1, 0)
        grid.addWidget(ckpt_line, 1, 1)
        grid.addWidget(ckpt_browse, 1, 2)
        try:
            grid.setColumnStretch(1, 1)
        except Exception:
            pass

        def apply_now() -> None:
            _sync_paths(lae_tab, _text(cfg_line), _text(ckpt_line), "")

        def browse_cfg() -> None:
            try:
                start = str(Path(_text(cfg_line)).parent) if _text(cfg_line) else ""
                fn, _ = QtWidgets.QFileDialog.getOpenFileName(
                    lae_tab, "LAE-DINO config", start, "LAE-DINO config (*.py *.yaml *.yml *.json);;All files (*)"
                )
                if fn:
                    _set_text(cfg_line, fn)
                    _sync_paths(lae_tab, fn, _text(ckpt_line), "")
                    _set_status_near(lae_tab, "LAE-DINO config set: " + Path(fn).name)
            except Exception as exc:
                _set_status_near(lae_tab, "LAE-DINO config browse failed: " + str(exc))

        def browse_ckpt() -> None:
            try:
                start = str(Path(_text(ckpt_line)).parent) if _text(ckpt_line) else ""
                fn, _ = QtWidgets.QFileDialog.getOpenFileName(
                    lae_tab, "LAE-DINO checkpoint", start, "LAE-DINO checkpoint (*.pth *.pt *.ckpt);;All files (*)"
                )
                if fn:
                    _set_text(ckpt_line, fn)
                    _sync_paths(lae_tab, _text(cfg_line), fn, "")
                    _set_status_near(lae_tab, "LAE-DINO checkpoint set: " + Path(fn).name)
            except Exception as exc:
                _set_status_near(lae_tab, "LAE-DINO checkpoint browse failed: " + str(exc))

        def import_from_project() -> None:
            try:
                project = ""
                for c in _current_project_candidates():
                    cfg, ckpt = _find_lae_files(c)
                    if cfg or ckpt:
                        project = c
                        break
                if not project:
                    project = _choose_project_dialog(lae_tab)
                    if not project:
                        _set_status_near(lae_tab, "LAE-DINO import cancelled.")
                        return
                    cfg, ckpt = _find_lae_files(project)
                if cfg:
                    _set_text(cfg_line, cfg)
                if ckpt:
                    _set_text(ckpt_line, ckpt)
                _sync_paths(lae_tab, _text(cfg_line), _text(ckpt_line), project)
                msg = "LAE-DINO import from project: "
                msg += ("config=yes" if cfg else "config=no") + ", " + ("checkpoint=yes" if ckpt else "checkpoint=no")
                _set_status_near(lae_tab, msg)
            except Exception as exc:
                _set_status_near(lae_tab, "LAE-DINO import from project failed: " + str(exc))

        try:
            cfg_browse.clicked.connect(browse_cfg)
            ckpt_browse.clicked.connect(browse_ckpt)
            import_btn.clicked.connect(import_from_project)
            cfg_line.editingFinished.connect(apply_now)
            ckpt_line.editingFinished.connect(apply_now)
        except Exception:
            pass

        # Insert at top of current layout or create one.
        layout = lae_tab.layout()
        if layout is None:
            layout = QtWidgets.QVBoxLayout(lae_tab)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(4)
        try:
            layout.insertWidget(0, frame)
        except Exception:
            try:
                layout.addWidget(frame)
            except Exception:
                pass

        # Hide old v14/v15 buttons-only grid to avoid duplicated/gequetscht LAE controls.
        _hide_old_lae_buttons(lae_tab, frame)
        _sync_paths(lae_tab, _text(cfg_line), _text(ckpt_line), "")
        setattr(lae_tab, "_mustatil_lae_rows_patch_v1", True)
        _log("Installed compact config/checkpoint rows on LAE-DINO video tab.")
        return True
    except Exception:
        _log("Failed to install rows:\n" + traceback.format_exc())
        return False


def _scan_and_patch() -> None:
    try:
        for tw in _all_tabwidgets():
            idx = _tab_index(tw, ["LAE-DINO", "LAE DINO"])
            if idx >= 0:
                try:
                    page = tw.widget(idx)
                    if page is not None:
                        _install_rows_on_lae_tab(page)
                except Exception:
                    pass
    except Exception:
        _log("scan failed:\n" + traceback.format_exc())


def _schedule_scan(delay_ms: int = 0) -> None:
    try:
        if QtCore is None:
            return
        QtCore.QTimer.singleShot(int(delay_ms), _scan_and_patch)
    except Exception:
        pass


def _start_periodic_scanner() -> None:
    global _SCAN_TIMER
    try:
        if _SCAN_TIMER is not None or QtCore is None:
            return
        timer = QtCore.QTimer()
        timer.setInterval(1200)
        timer.timeout.connect(_scan_and_patch)
        timer.start()
        _SCAN_TIMER = timer
        # also run multiple early scans because Mustatil creates tabs asynchronously via plugins.
        for ms in (0, 250, 800, 1600, 3200, 6000):
            _schedule_scan(ms)
    except Exception:
        pass


def _patch_qtabwidget() -> None:
    global _ORIG_ADD_TAB, _ORIG_INSERT_TAB
    if QtWidgets is None:
        return
    try:
        if getattr(QtWidgets.QTabWidget, "_mustatil_lae_rows_patch_wrapped", False):
            return
        _ORIG_ADD_TAB = QtWidgets.QTabWidget.addTab
        _ORIG_INSERT_TAB = QtWidgets.QTabWidget.insertTab

        def addTab_patched(self, *args, **kwargs):
            res = _ORIG_ADD_TAB(self, *args, **kwargs)
            _schedule_scan(0)
            _schedule_scan(250)
            return res

        def insertTab_patched(self, *args, **kwargs):
            res = _ORIG_INSERT_TAB(self, *args, **kwargs)
            _schedule_scan(0)
            _schedule_scan(250)
            return res

        QtWidgets.QTabWidget.addTab = addTab_patched
        QtWidgets.QTabWidget.insertTab = insertTab_patched
        QtWidgets.QTabWidget._mustatil_lae_rows_patch_wrapped = True
    except Exception:
        _log("QTabWidget patch failed:\n" + traceback.format_exc())


def install_plugin() -> bool:
    global _PATCHED
    if _PATCHED:
        _schedule_scan(0)
        return True
    if QtWidgets is None or QtCore is None:
        _log("No Qt binding found. Plugin did not install.")
        return False
    _PATCHED = True
    _patch_qtabwidget()
    _start_periodic_scanner()
    _log("Installed. Waiting for Video Detection / LAE-DINO tab.")
    return True


def mustatil_plugin_init(*args, **kwargs):
    return install_plugin()


def init_plugin(*args, **kwargs):
    return install_plugin()


def register_plugin(*args, **kwargs):
    return install_plugin()


# Some Mustatil plugin loaders simply import plugin files. Make import-time install safe.
try:
    install_plugin()
except Exception:
    _log("import-time install failed:\n" + traceback.format_exc())
