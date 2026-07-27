#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mustatil Auto Updater GUI plugin.

Use:
    Put this file into mustatil_plugins/ and rename it to:
        mustatil_auto_updater_gui.py

It is intentionally self-contained and loader-safe:
- no dataclass usage, so the old plugin loader cannot crash with NoneType.__dict__
- when imported as a plugin, it patches the main window's Auto Updater button
  to launch this file from mustatil_plugins/
- when executed as a script, it opens its own PySide6 updater window
- downloaded setup EXE/MSI is opened normally; the user clicks through setup manually
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

APP_VERSION = "5.6.0"
CONFIG_NAME = "github_update_config.json"
DEFAULT_REPO = "tarekwasfy01/Mustatil---YOLO-AI-Model-Trainer-"
SETUP_EXTENSIONS = (".exe", ".msi")


class ReleaseInfo:
    def __init__(self, tag="", name="", html_url="", asset_name="", asset_url="", body=""):
        self.tag = str(tag or "")
        self.name = str(name or "")
        self.html_url = str(html_url or "")
        self.asset_name = str(asset_name or "")
        self.asset_url = str(asset_url or "")
        self.body = str(body or "")

    def display_name(self) -> str:
        label = self.tag or self.name or "Release"
        if self.asset_name:
            label += "  (" + self.asset_name + ")"
        return label


def _plugin_file() -> Path:
    try:
        return Path(__file__).resolve()
    except Exception:
        return Path.cwd() / "mustatil_auto_updater_gui.py"


def _app_base_dir() -> Path:
    """Return the main app folder when the script is inside mustatil_plugins."""
    p = _plugin_file().parent
    if p.name.lower() == "mustatil_plugins":
        return p.parent
    return p


def _plugin_dir() -> Path:
    p = _plugin_file().parent
    if p.name.lower() == "mustatil_plugins":
        return p
    d = p / "mustatil_plugins"
    return d if d.exists() else p


def open_auto_updater():
    """Public entry point usable by the main program or other plugins."""
    return launch_auto_updater()


def start_auto_updater():
    return launch_auto_updater()


def launch_auto_updater():
    """Launch this updater as a separate process/window."""
    script = _plugin_file()
    kwargs = {}
    try:
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        return subprocess.Popen([sys.executable, str(script)], cwd=str(_app_base_dir()), **kwargs)
    except Exception:
        # Last fallback: run without CREATE_NEW_CONSOLE.
        return subprocess.Popen([sys.executable, str(script)], cwd=str(_app_base_dir()))


def mustatil_plugin_init():
    """Patch MustatilQtWorkspace.open_auto_updater_window while loaded as plugin.

    The unmodified main program only searches next to mustatil_qt_workspace.py.
    This patch makes the existing Auto Updater tab button start the plugin file
    from mustatil_plugins/ without changing the main program.
    """
    g = globals().get("MUSTATIL_GLOBALS") or {}
    cls = g.get("MustatilQtWorkspace")
    if cls is None:
        return

    def _patched_open_auto_updater_window(self):
        candidates = [
            _plugin_dir() / "mustatil_auto_updater_gui.py",
            _plugin_dir() / "github_auto_updater_gui.py",
            _plugin_dir() / "github_auto_updater.py",
            _plugin_dir() / "auto_updater.py",
            _app_base_dir() / "mustatil_auto_updater_gui.py",
            _app_base_dir() / "github_auto_updater_gui.py",
            _app_base_dir() / "github_auto_updater.py",
            _app_base_dir() / "auto_updater.py",
        ]
        script = next((p for p in candidates if p.is_file()), _plugin_file())
        try:
            kwargs = {}
            if os.name == "nt":
                kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
            subprocess.Popen([sys.executable, str(script)], cwd=str(_app_base_dir()), **kwargs)
            if hasattr(self, "log"):
                self.log("Auto Updater gestartet: " + str(script.name))
        except Exception as exc:
            msg = "Auto-Updater konnte nicht gestartet werden:\n" + str(exc)
            if hasattr(self, "show_error"):
                self.show_error("Auto Updater", msg)
            else:
                print(msg)

    try:
        cls.open_auto_updater_window = _patched_open_auto_updater_window
        print("[Mustatil Auto Updater Plugin] Auto Updater button patched.")
    except Exception as exc:
        print("[Mustatil Auto Updater Plugin] Patch failed:", exc)


def _load_qt():
    from PySide6.QtCore import Signal, QObject
    from PySide6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QTextEdit, QMessageBox, QComboBox, QGroupBox
    )
    return {
        "Signal": Signal,
        "QObject": QObject,
        "QApplication": QApplication,
        "QWidget": QWidget,
        "QVBoxLayout": QVBoxLayout,
        "QHBoxLayout": QHBoxLayout,
        "QLabel": QLabel,
        "QPushButton": QPushButton,
        "QTextEdit": QTextEdit,
        "QMessageBox": QMessageBox,
        "QComboBox": QComboBox,
        "QGroupBox": QGroupBox,
    }


def _build_classes(qt):
    Signal = qt["Signal"]
    QObject = qt["QObject"]
    QApplication = qt["QApplication"]
    QWidget = qt["QWidget"]
    QVBoxLayout = qt["QVBoxLayout"]
    QHBoxLayout = qt["QHBoxLayout"]
    QLabel = qt["QLabel"]
    QPushButton = qt["QPushButton"]
    QTextEdit = qt["QTextEdit"]
    QMessageBox = qt["QMessageBox"]
    QComboBox = qt["QComboBox"]
    QGroupBox = qt["QGroupBox"]

    class Signals(QObject):
        log = Signal(str)
        releases_loaded = Signal(object)
        newer_check_done = Signal(object)
        install_done = Signal(str)
        error = Signal(str)

    class AutoUpdaterWindow(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Mustatil Auto Updater")
            self.resize(760, 460)

            self.base = _app_base_dir()
            self.plugin_base = _plugin_dir()
            self.config_path = self.base / CONFIG_NAME
            self.repo = DEFAULT_REPO
            self.asset_pattern = ".exe"
            self.releases: List[ReleaseInfo] = []

            self.signals = Signals()
            self.signals.log.connect(self.log)
            self.signals.releases_loaded.connect(self.on_releases_loaded)
            self.signals.newer_check_done.connect(self.on_newer_check_done)
            self.signals.install_done.connect(self.on_install_done)
            self.signals.error.connect(lambda txt: QMessageBox.critical(self, "Auto Updater", txt))

            self._build_ui()
            self.load_config()
            self.refresh_releases()

        def _build_ui(self):
            root = QVBoxLayout(self)

            newest_box = QGroupBox("Newest Release")
            newest_row = QHBoxLayout(newest_box)
            newest_row.addWidget(QLabel("Download the newest GitHub setup file and open it. The user clicks through setup manually."), 1)
            self.check_newer_btn = QPushButton("Check for Newer Version")
            self.check_newer_btn.clicked.connect(self.check_for_newer_version)
            newest_row.addWidget(self.check_newer_btn)
            self.install_newest_btn = QPushButton("Open Newest Setup")
            self.install_newest_btn.clicked.connect(self.install_newest_release)
            newest_row.addWidget(self.install_newest_btn)
            root.addWidget(newest_box)

            older_box = QGroupBox("Open Older Release Setup")
            older_col = QVBoxLayout(older_box)
            older_row = QHBoxLayout()
            self.older_combo = QComboBox()
            self.older_combo.setMinimumWidth(440)
            older_row.addWidget(QLabel("Older Release:"))
            older_row.addWidget(self.older_combo, 1)
            self.refresh_btn = QPushButton("Refresh Releases")
            self.refresh_btn.clicked.connect(self.refresh_releases)
            older_row.addWidget(self.refresh_btn)
            self.install_older_btn = QPushButton("Open Older Setup")
            self.install_older_btn.clicked.connect(self.install_older_release)
            older_row.addWidget(self.install_older_btn)
            older_col.addLayout(older_row)
            root.addWidget(older_box)

            self.log_box = QTextEdit()
            self.log_box.setReadOnly(True)
            root.addWidget(self.log_box, 1)

        def log(self, text: str):
            self.log_box.append("[" + time.strftime("%H:%M:%S") + "] " + str(text))

        def load_config(self):
            if not self.config_path.is_file():
                self.save_config()
                self.log("No config found. Default repository set: " + self.repo)
                return
            try:
                cfg = json.loads(self.config_path.read_text(encoding="utf-8"))
                self.repo = str(cfg.get("repo", DEFAULT_REPO)).strip() or DEFAULT_REPO
                pattern = str(cfg.get("asset_pattern", ".exe")).strip().lower()
                self.asset_pattern = pattern or ".exe"
                self.log("Config loaded. Repository: " + self.repo)
            except Exception as exc:
                self.log("Config could not be read; using default repository: " + str(exc))

        def save_config(self):
            cfg = {
                "repo": self.repo,
                "current_version": APP_VERSION,
                "asset_pattern": self.asset_pattern,
                "install_dir": str(self.base),
            }
            try:
                self.config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass

        def set_busy(self, busy: bool):
            self.install_newest_btn.setEnabled(not busy)
            self.check_newer_btn.setEnabled(not busy)
            self.install_older_btn.setEnabled(not busy and self.older_combo.count() > 0)
            self.refresh_btn.setEnabled(not busy)
            self.older_combo.setEnabled(not busy and self.older_combo.count() > 0)

        def refresh_releases(self):
            self.set_busy(True)
            self.log("Loading GitHub releases ...")
            threading.Thread(target=self._refresh_releases_worker, daemon=True).start()

        def _refresh_releases_worker(self):
            try:
                releases = self.fetch_releases()
                self.signals.releases_loaded.emit(releases)
            except Exception:
                self.signals.error.emit(traceback.format_exc())
                self.signals.releases_loaded.emit([])

        def github_get_json(self, url: str) -> Any:
            req = urllib.request.Request(url, headers={"User-Agent": "MustatilAutoUpdater/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8", errors="replace"))

        def fetch_releases(self) -> List[ReleaseInfo]:
            url = "https://api.github.com/repos/" + self.repo + "/releases?per_page=50"
            data = self.github_get_json(url)
            releases: List[ReleaseInfo] = []
            for item in data:
                release = self.release_from_json(item)
                if release.asset_url:
                    releases.append(release)
            return releases

        def release_from_json(self, data: Dict[str, Any]) -> ReleaseInfo:
            picked = self.pick_setup_asset(data.get("assets", []))
            return ReleaseInfo(
                tag=data.get("tag_name", ""),
                name=data.get("name", ""),
                html_url=data.get("html_url", ""),
                asset_name=(picked or {}).get("name", ""),
                asset_url=(picked or {}).get("browser_download_url", ""),
                body=data.get("body", ""),
            )

        def pick_setup_asset(self, assets: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            for a in assets:
                name = str(a.get("name", "")).lower()
                if name.endswith(SETUP_EXTENSIONS) and ("setup" in name or "install" in name or "installer" in name):
                    return a
            for a in assets:
                name = str(a.get("name", "")).lower()
                if name.endswith(SETUP_EXTENSIONS):
                    return a
            pattern = self.asset_pattern.replace("*", "").lower()
            if pattern:
                for a in assets:
                    if pattern in str(a.get("name", "")).lower():
                        return a
            return None

        def on_releases_loaded(self, releases_obj: object):
            self.releases = list(releases_obj or [])
            self.older_combo.clear()

            if not self.releases:
                self.log("No releases with setup EXE/MSI were found.")
                self.set_busy(False)
                return

            older = self.releases[1:] if len(self.releases) > 1 else []
            for release in older:
                self.older_combo.addItem(release.display_name(), release)

            newest = self.releases[0]
            self.log("Newest release found: " + newest.display_name())
            if older:
                self.log("Older releases in dropdown: " + str(len(older)))
            else:
                self.log("There is currently no older release available in the dropdown.")
            self.set_busy(False)

        def normalize_version_parts(self, text: str) -> List[int]:
            import re
            nums = re.findall(r"\d+", text or "")
            return [int(x) for x in nums[:4]] or [0]

        def version_is_newer(self, latest: str, current: str) -> bool:
            a = self.normalize_version_parts(latest)
            b = self.normalize_version_parts(current)
            max_len = max(len(a), len(b))
            a += [0] * (max_len - len(a))
            b += [0] * (max_len - len(b))
            return a > b

        def check_for_newer_version(self):
            self.set_busy(True)
            self.log("Checking for a newer version ...")
            threading.Thread(target=self._check_for_newer_version_worker, daemon=True).start()

        def _check_for_newer_version_worker(self):
            try:
                releases = self.fetch_releases()
                newest = releases[0] if releases else None
                self.signals.newer_check_done.emit(newest)
                self.signals.releases_loaded.emit(releases)
            except Exception:
                self.signals.error.emit(traceback.format_exc())
                self.signals.releases_loaded.emit(self.releases)

        def on_newer_check_done(self, release_obj: object):
            if not isinstance(release_obj, ReleaseInfo):
                self.log("No installable release was found.")
                QMessageBox.information(self, "Auto Updater", "No installable release was found.")
                return
            latest_label = release_obj.tag or release_obj.name or "unknown"
            if self.version_is_newer(latest_label, APP_VERSION):
                msg = "A newer version is available: " + latest_label + "\nCurrent version: " + APP_VERSION
            else:
                msg = "You are already on the newest known version.\nCurrent version: " + APP_VERSION + "\nNewest release: " + latest_label
            self.log(msg.replace("\n", " | "))
            QMessageBox.information(self, "Auto Updater", msg)

        def install_newest_release(self):
            self.set_busy(True)
            threading.Thread(target=self._install_newest_worker, daemon=True).start()

        def _install_newest_worker(self):
            try:
                releases = self.fetch_releases()
                if not releases:
                    raise RuntimeError("No release with setup EXE/MSI was found.")
                self.install_release(releases[0])
                self.signals.install_done.emit(releases[0].display_name())
                self.signals.releases_loaded.emit(releases)
            except Exception:
                self.signals.error.emit(traceback.format_exc())
                self.signals.releases_loaded.emit(self.releases)

        def install_older_release(self):
            release = self.older_combo.currentData()
            if not isinstance(release, ReleaseInfo):
                QMessageBox.warning(self, "Auto Updater", "Please select an older release first.")
                return
            self.set_busy(True)
            threading.Thread(target=self._install_release_worker, args=(release,), daemon=True).start()

        def _install_release_worker(self, release: ReleaseInfo):
            try:
                self.install_release(release)
                self.signals.install_done.emit(release.display_name())
            except Exception:
                self.signals.error.emit(traceback.format_exc())
            finally:
                self.signals.releases_loaded.emit(self.releases)

        def install_release(self, release: ReleaseInfo):
            if not release.asset_url:
                raise RuntimeError("The selected release has no setup asset.")
            downloads = self.base / "_mustatil_update_downloads"
            downloads.mkdir(exist_ok=True)
            filename = release.asset_name or ("Mustatil_Setup_" + (release.tag or "release") + ".exe")
            target = downloads / self.safe_filename(filename)
            self.signals.log.emit("Download: " + release.display_name())
            self.download_file(release.asset_url, target)
            self.signals.log.emit("Opening setup for manual installation: " + target.name)
            self.open_installer_manual(target)

        def safe_filename(self, name: str) -> str:
            bad = '<>:"/\\|?*'
            out = "".join("_" if c in bad else c for c in str(name)).strip()
            return out or "Mustatil_Setup.exe"

        def download_file(self, url: str, target: Path):
            req = urllib.request.Request(url, headers={"User-Agent": "MustatilAutoUpdater/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, target.open("wb") as f:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                last_log = 0
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    now = time.time()
                    if total and now - last_log > 1.0:
                        self.signals.log.emit("Download: " + str(done * 100 // total) + "%")
                        last_log = now

        def open_installer_manual(self, path: Path):
            if not path.exists():
                raise RuntimeError("Downloaded setup file does not exist: " + str(path))
            if sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            else:
                subprocess.Popen([str(path)], cwd=str(path.parent))
            self.signals.log.emit("Setup opened. Please continue manually in the installer window.")

        def on_install_done(self, release_name: str):
            self.log("Setup opened manually: " + release_name)
            QMessageBox.information(self, "Auto Updater", "Setup was opened. Please click through the installer manually.")

    return QApplication, AutoUpdaterWindow


def main():
    try:
        qt = _load_qt()
    except Exception as exc:
        raise SystemExit("PySide6 is missing. Start the Mustatil launcher so dependencies can be installed.\n" + str(exc))

    QApplication, AutoUpdaterWindow = _build_classes(qt)
    app = QApplication.instance() or QApplication(sys.argv)
    w = AutoUpdaterWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
