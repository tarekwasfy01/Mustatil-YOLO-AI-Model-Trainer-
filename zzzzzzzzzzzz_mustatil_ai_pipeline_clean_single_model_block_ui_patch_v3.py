# -*- coding: utf-8 -*-
"""
Mustatil plugin: LAE-DINO Trainer Remove Lower Area v1

Purpose
-------
Drop-in cleanup patch for Mustatil's existing LAE-DINO Trainer tab.
It keeps the main LAE-DINO trainer control panel and hides/removes the
extra lower area that remains below it in the tab.

Install
-------
Copy this file into Mustatil's `mustatil_plugins` folder and restart Mustatil.
It is intentionally named with many leading z's so it loads after the LAE-DINO
trainer plugin.

Notes
-----
This plugin does not replace the LAE-DINO trainer. It only cleans the UI after
Mustatil and the trainer plugin have finished building the tab.
"""

from __future__ import annotations

import traceback

PLUGIN_NAME = "Mustatil LAE-DINO Trainer Remove Lower Area v1"
PLUGIN_PREFIX = "[LAE-DINO Trainer CleanLower v1]"

_PATCHED = False
_ORIG_ADD = None
_ORIG_INSERT = None
_DONE = set()
_HIDDEN_WIDGETS = []  # keep references so Qt/Python do not immediately GC detached widgets


def _log(msg):
    try:
        print(f"{PLUGIN_PREFIX} {msg}")
    except Exception:
        pass


def _qt_imports():
    try:
        from PySide6.QtWidgets import (
            QApplication, QTabWidget, QScrollArea, QWidget, QGroupBox,
            QSizePolicy, QLabel
        )
        from PySide6.QtCore import QTimer
        return QApplication, QTabWidget, QScrollArea, QWidget, QGroupBox, QSizePolicy, QLabel, QTimer
    except Exception:
        return None


def _tab_label_is_lae_trainer(text) -> bool:
    low = str(text or "").strip().lower().replace("_", "-")
    return ("lae" in low and "dino" in low and ("train" in low or "trainer" in low or "training" in low))


def _page_inner_widget(page):
    qt = _qt_imports()
    if not qt:
        return page
    _QApplication, _QTabWidget, QScrollArea, _QWidget, _QGroupBox, _QSizePolicy, _QLabel, _QTimer = qt
    try:
        # Unwrap nested QScrollAreas if present.
        seen = set()
        cur = page
        while isinstance(cur, QScrollArea) and cur.widget() is not None and id(cur) not in seen:
            seen.add(id(cur))
            cur = cur.widget()
        return cur
    except Exception:
        return page


def _text_of_widget(w) -> str:
    parts = []
    try:
        name = str(w.objectName() or "")
        if name:
            parts.append(name)
    except Exception:
        pass
    try:
        title = str(w.title() or "")
        if title:
            parts.append(title)
    except Exception:
        pass
    try:
        text = str(w.text() or "")
        if text:
            parts.append(text)
    except Exception:
        pass
    return " ".join(parts).lower()


def _is_primary_lae_trainer_panel(w) -> bool:
    """Return True for the top trainer panel that should be kept."""
    try:
        s = _text_of_widget(w)
        obj = str(getattr(w, "objectName", lambda: "")() or "")

        # Known panel from the user's current LAE-DINO trainer plugin.
        if obj == "MustatilLAEDINOProjectTrainerV18AntiFreezePanel":
            return True
        if obj.startswith("MustatilLAEDINOProjectTrainer") and "Panel" in obj:
            return True
        if obj.startswith("MustatilLAEDINO") and "Trainer" in obj and "Panel" in obj:
            return True

        # Fallback: a group/panel whose visible title clearly says LAE-DINO Project Trainer.
        if "lae" in s and "dino" in s and "trainer" in s:
            if "project" in s or "dataset" in s or "config" in s or "training" in s:
                return True

        # If the panel contains the known V18 controls, it is the primary panel.
        try:
            if w.findChild(type(w), "MustatilLAEDINOProjectTrainerV18AntiFreezePanel") is not None:
                return True
        except Exception:
            pass
        try:
            # Avoid importing classes here; findChild can find by objectName with base QWidget below.
            qt = _qt_imports()
            if qt:
                _QA, _QTW, _QSA, QWidget, _QGB, _QSP, _QL, _QT = qt
                known_names = [
                    "MustatilLAEDINOV18ProjectFolder",
                    "MustatilLAEDINOCreateDatasetButtonV18",
                    "MustatilLAEDINOCreateConfigButtonV18",
                    "MustatilLAEDINOIntegratedTrainButtonV18",
                ]
                hits = 0
                for n in known_names:
                    if w.findChild(QWidget, n) is not None:
                        hits += 1
                if hits >= 2:
                    return True
        except Exception:
            pass
    except Exception:
        pass
    return False


def _iter_layout_items(layout):
    try:
        for i in range(layout.count()):
            yield i, layout.itemAt(i)
    except Exception:
        return


def _hide_widget(w, reason="lower area"):
    qt = _qt_imports()
    if not qt or w is None:
        return False
    _QA, _QTW, _QSA, _QWidget, _QGB, QSizePolicy, _QL, _QT = qt
    try:
        if bool(w.property("MustatilLAEDINOCleanLowerKeep")):
            return False
    except Exception:
        pass
    try:
        if bool(w.property("MustatilLAEDINOCleanLowerHidden")):
            return False
    except Exception:
        pass

    try:
        w.setProperty("MustatilLAEDINOCleanLowerHidden", True)
    except Exception:
        pass
    try:
        w.hide()
    except Exception:
        pass
    try:
        w.setMinimumHeight(0)
        w.setMaximumHeight(0)
    except Exception:
        pass
    try:
        w.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
    except Exception:
        try:
            w.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        except Exception:
            pass
    try:
        _HIDDEN_WIDGETS.append(w)
    except Exception:
        pass
    return True


def _find_primary_panel(page):
    qt = _qt_imports()
    if not qt or page is None:
        return None
    _QA, _QTW, _QSA, QWidget, QGroupBox, _QSP, _QL, _QT = qt
    inner = _page_inner_widget(page)

    # Prefer exact known object name.
    try:
        found = inner.findChild(QWidget, "MustatilLAEDINOProjectTrainerV18AntiFreezePanel")
        if found is not None:
            return found
    except Exception:
        pass

    # Any direct/indirect widget that looks like the primary panel.
    try:
        for w in inner.findChildren(QWidget):
            if _is_primary_lae_trainer_panel(w):
                return w
    except Exception:
        pass

    # Fallback: if the first layout widget itself looks like a LAE-DINO trainer panel.
    try:
        lay = inner.layout()
        if lay is not None:
            for _i, item in _iter_layout_items(lay):
                w = item.widget()
                if w is not None and _is_primary_lae_trainer_panel(w):
                    return w
    except Exception:
        pass
    return None


def _cleanup_layout_below(panel) -> int:
    """Hide widgets after the primary panel in its parent layout."""
    if panel is None:
        return 0
    hidden = 0
    try:
        panel.setProperty("MustatilLAEDINOCleanLowerKeep", True)
    except Exception:
        pass

    parent = None
    try:
        parent = panel.parentWidget()
    except Exception:
        parent = None
    if parent is None:
        return 0

    try:
        layout = parent.layout()
    except Exception:
        layout = None
    if layout is None:
        return 0

    seen_panel = False
    for _idx, item in _iter_layout_items(layout):
        try:
            w = item.widget()
        except Exception:
            w = None
        if w is None:
            # Leave spacers/stretches alone; they normally do not create a visible bottom panel.
            continue
        if w is panel:
            seen_panel = True
            continue
        if seen_panel:
            # Only remove/hide siblings below the main LAE-DINO trainer panel.
            if _hide_widget(w):
                hidden += 1
    return hidden


def _try_cleanup_page(page, label="") -> bool:
    qt = _qt_imports()
    if not qt or page is None:
        return False
    try:
        key = id(page)
        # Re-run a few times in case other plugins add lower content after this plugin.
        panel = _find_primary_panel(page)
        if panel is None:
            return False
        hidden = _cleanup_layout_below(panel)
        if hidden or key not in _DONE:
            _DONE.add(key)
            _log(f"cleaned LAE-DINO Trainer lower area: hidden={hidden}, label={label!r}")
        return True
    except Exception as exc:
        _log("cleanup page failed: " + str(exc))
        try:
            traceback.print_exc()
        except Exception:
            pass
        return False


def _process_tabwidget(tw) -> bool:
    try:
        if tw is None:
            return False
        ok = False
        count = int(tw.count())
        for i in range(count):
            label = str(tw.tabText(i) or "")
            if _tab_label_is_lae_trainer(label):
                page = tw.widget(i)
                ok = _try_cleanup_page(page, label) or ok
        return ok
    except Exception as exc:
        _log("process tabwidget failed: " + str(exc))
        return False


def _scan(root=None) -> bool:
    qt = _qt_imports()
    if not qt:
        _log("PySide6 Qt widgets unavailable")
        return False
    QApplication, QTabWidget, _QSA, _QWidget, _QGB, _QSP, _QL, _QT = qt
    widgets = []
    try:
        if root is not None:
            if isinstance(root, QTabWidget):
                widgets.append(root)
            try:
                widgets.extend(root.findChildren(QTabWidget))
            except Exception:
                pass
    except Exception:
        pass
    if not widgets:
        try:
            app = QApplication.instance()
            if app:
                for top in app.topLevelWidgets():
                    try:
                        if isinstance(top, QTabWidget):
                            widgets.append(top)
                        widgets.extend(top.findChildren(QTabWidget))
                    except Exception:
                        pass
        except Exception:
            pass
    seen = set()
    ok = False
    for tw in widgets[:80]:
        if id(tw) in seen:
            continue
        seen.add(id(tw))
        if _process_tabwidget(tw):
            ok = True
    return ok


def _schedule(root=None):
    qt = _qt_imports()
    if not qt:
        return
    _QA, _QTW, _QSA, _QW, _QGB, _QSP, _QL, QTimer = qt
    try:
        for ms in (100, 300, 700, 1200, 2500, 5000, 9000):
            QTimer.singleShot(ms, lambda root=root: _scan(root))
    except Exception:
        pass


def _install(root=None):
    global _PATCHED, _ORIG_ADD, _ORIG_INSERT
    qt = _qt_imports()
    if not qt:
        _log("Qt unavailable; plugin will not run")
        return False
    _QA, QTabWidget, _QSA, _QW, _QGB, _QSP, _QL, _QT = qt

    if not _PATCHED:
        try:
            _ORIG_ADD = QTabWidget.addTab
            _ORIG_INSERT = QTabWidget.insertTab

            def addTab_patched(self, page, *args, **kwargs):
                res = _ORIG_ADD(self, page, *args, **kwargs)
                try:
                    label = ""
                    for a in reversed(args):
                        if isinstance(a, str):
                            label = a
                            break
                    if not label:
                        try:
                            label = str(self.tabText(int(res)) or "")
                        except Exception:
                            label = ""
                    if _tab_label_is_lae_trainer(label):
                        _schedule(self)
                    else:
                        # cheap delayed scan: LAE tab may already exist and receive content later
                        _schedule(self)
                except Exception:
                    pass
                return res

            def insertTab_patched(self, index, page, *args, **kwargs):
                res = _ORIG_INSERT(self, index, page, *args, **kwargs)
                try:
                    label = ""
                    for a in reversed(args):
                        if isinstance(a, str):
                            label = a
                            break
                    if not label:
                        try:
                            label = str(self.tabText(int(res)) or "")
                        except Exception:
                            label = ""
                    if _tab_label_is_lae_trainer(label):
                        _schedule(self)
                    else:
                        _schedule(self)
                except Exception:
                    pass
                return res

            QTabWidget.addTab = addTab_patched
            QTabWidget.insertTab = insertTab_patched
            _PATCHED = True
            _log("hook installed")
        except Exception as exc:
            _log("hook install failed: " + str(exc))
            return False

    _schedule(root)
    return True


def mustatil_plugin_init():
    return _install(None)


def register_plugin(app=None, main_window=None):
    return _install(main_window or app)
