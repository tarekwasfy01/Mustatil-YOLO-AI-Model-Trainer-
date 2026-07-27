# -*- coding: utf-8 -*-
"""
Mustatil patch plugin: direct PDAL conda-forge button for Geospatial Operations.

Purpose
-------
This is a small additive patch for the existing Geospatial Operations tab plugin
(v17 simple QGIS/GDAL installer).  It patches the concrete ConsoleHubWidget from
that plugin and inserts a visible button directly in the top console toolbar,
next to "Check all consoles" when possible.  It does not create a second tab and
it does not replace the Geospatial Operations tab.

Button label:
    Install PDAL conda-forge

The button installs/repairs a local conda-forge PDAL runtime under:
    <Mustatil root>/mustatil_model_runtimes/PDAL-CONDAFORGE

It then makes the Geospatial Operations PDAL discovery prefer:
    .../envs/mustatil-pdal/Library/bin/pdal.exe

Install: copy this .py into Mustatil's mustatil_plugins folder and restart.
"""
from __future__ import annotations

import os
import sys
import json
import traceback
from pathlib import Path
from typing import Any, Iterable, List, Optional

PLUGIN_NAME = "Mustatil Geospatial PDAL conda-forge direct button patch"
_PATCHED_MODULE_IDS = set()
_PATCHED_QTAB = False
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


def _mustatil_root() -> Path:
    try:
        here = Path(__file__).resolve()
        if here.parent.name.lower() == "mustatil_plugins":
            return here.parent.parent
        return here.parent
    except Exception:
        return Path.cwd()


def _runtime_root() -> Path:
    return _mustatil_root() / "mustatil_model_runtimes" / "PDAL-CONDAFORGE"


def _env_root() -> Path:
    return _runtime_root() / "envs" / "mustatil-pdal"


def _local_pdal_dirs() -> List[str]:
    env = _env_root()
    rt = _runtime_root()
    if _is_windows():
        candidates = [
            env / "Library" / "bin",
            env / "Scripts",
            env,
            rt / "Library" / "bin",
            rt / "Scripts",
            rt,
        ]
    else:
        candidates = [
            env / "bin",
            env / "lib",
            rt / "bin",
            rt,
        ]
    out: List[str] = []
    seen = set()
    for p in candidates:
        try:
            if p.exists() and p.is_dir():
                s = str(p.resolve())
                k = s.lower()
                if k not in seen:
                    seen.add(k)
                    out.append(s)
        except Exception:
            pass
    return out


def _local_pdal_exe() -> str:
    env = _env_root()
    if _is_windows():
        candidates = [
            env / "Library" / "bin" / "pdal.exe",
            env / "Scripts" / "pdal.exe",
            _runtime_root() / "Library" / "bin" / "pdal.exe",
        ]
    else:
        candidates = [env / "bin" / "pdal", _runtime_root() / "bin" / "pdal"]
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                return str(p.resolve())
        except Exception:
            pass
    return ""


def _quote_cmd(s: Any) -> str:
    text = str(s or "")
    if _is_windows():
        return '"' + text.replace('"', '""') + '"'
    import shlex
    return shlex.quote(text)


# Qt imports are delayed enough that py_compile works outside Mustatil.
try:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QPushButton, QHBoxLayout, QVBoxLayout, QTabWidget
except Exception:  # pragma: no cover
    QTimer = None  # type: ignore
    QApplication = None  # type: ignore
    QPushButton = None  # type: ignore
    QHBoxLayout = None  # type: ignore
    QVBoxLayout = None  # type: ignore
    QTabWidget = None  # type: ignore


def _write_windows_install_script() -> Path:
    rt = _runtime_root()
    rt.mkdir(parents=True, exist_ok=True)
    script = rt / "install_or_repair_pdal_condaforge.cmd"
    installer = rt / "Miniforge3-Windows-x86_64.exe"
    install_dir = rt
    env_name = "mustatil-pdal"
    env_dir = _env_root()
    pdal_exe = env_dir / "Library" / "bin" / "pdal.exe"
    py_exe = env_dir / "python.exe"
    meta_json = rt / "mustatil_pdal_condaforge_runtime.json"

    lines = [
        "@echo off",
        "chcp 65001 >nul",
        "setlocal EnableExtensions EnableDelayedExpansion",
        "echo ================================================",
        "echo Mustatil PDAL conda-forge installer / repair",
        "echo ================================================",
        f"set \"RUNTIME={install_dir}\"",
        f"set \"ENVNAME={env_name}\"",
        f"set \"ENVPATH={env_dir}\"",
        f"set \"INSTALLER={installer}\"",
        f"set \"PDAL_EXE={pdal_exe}\"",
        "set \"CONDA_EXE=\"",
        "",
        "if exist \"%RUNTIME%\\Scripts\\conda.exe\" set \"CONDA_EXE=%RUNTIME%\\Scripts\\conda.exe\"",
        "if not defined CONDA_EXE if exist \"%USERPROFILE%\\miniforge3\\Scripts\\conda.exe\" set \"CONDA_EXE=%USERPROFILE%\\miniforge3\\Scripts\\conda.exe\"",
        "if not defined CONDA_EXE if exist \"%USERPROFILE%\\miniconda3\\Scripts\\conda.exe\" set \"CONDA_EXE=%USERPROFILE%\\miniconda3\\Scripts\\conda.exe\"",
        "if not defined CONDA_EXE if exist \"%USERPROFILE%\\anaconda3\\Scripts\\conda.exe\" set \"CONDA_EXE=%USERPROFILE%\\anaconda3\\Scripts\\conda.exe\"",
        "if not defined CONDA_EXE for %%C in (conda.exe) do if not \"%%~$PATH:C\"==\"\" set \"CONDA_EXE=%%~$PATH:C\"",
        "",
        "if defined CONDA_EXE (",
        "  echo [conda found] %CONDA_EXE%",
        ") else (",
        "  echo [conda missing] Installing local Miniforge runtime for Mustatil...",
        "  if not exist \"%RUNTIME%\" mkdir \"%RUNTIME%\"",
        "  if not exist \"%INSTALLER%\" (",
        "    echo [download] Miniforge3-Windows-x86_64.exe",
        "    powershell -NoProfile -ExecutionPolicy Bypass -Command \"$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe' -OutFile '%INSTALLER%'\"",
        "    if errorlevel 1 ( echo [ERROR] Miniforge download failed. & exit /b 11 )",
        "  )",
        "  echo [install] Local Miniforge into %RUNTIME%",
        "  start /wait \"\" \"%INSTALLER%\" /S /InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /D=%RUNTIME%",
        "  if not exist \"%RUNTIME%\\Scripts\\conda.exe\" ( echo [ERROR] Local conda was not created. & exit /b 12 )",
        "  set \"CONDA_EXE=%RUNTIME%\\Scripts\\conda.exe\"",
        ")",
        "",
        "echo [conda] %CONDA_EXE%",
        "call \"%CONDA_EXE%\" --version",
        "",
        "if exist \"%ENVPATH%\\conda-meta\" (",
        "  echo [env exists] Repairing/updating %ENVPATH%",
        "  call \"%CONDA_EXE%\" install -y -p \"%ENVPATH%\" -c conda-forge pdal python-pdal gdal rasterio numpy python=3.11",
        ") else (",
        "  echo [env create] %ENVPATH%",
        "  call \"%CONDA_EXE%\" create -y -p \"%ENVPATH%\" -c conda-forge pdal python-pdal gdal rasterio numpy python=3.11",
        ")",
        "if errorlevel 1 ( echo [ERROR] conda PDAL install failed. & exit /b 20 )",
        "",
        "echo [test] PDAL executable",
        "if not exist \"%PDAL_EXE%\" ( echo [ERROR] pdal.exe was not found at %PDAL_EXE% & exit /b 21 )",
        "\"%PDAL_EXE%\" --version",
        "if errorlevel 1 ( echo [ERROR] pdal --version failed. & exit /b 22 )",
        "",
        "echo [test] python-pdal import",
        f"\"{py_exe}\" -c \"import sys, pdal; print('python', sys.executable); print('pdal module OK', getattr(pdal, '__version__', 'OK'))\"",
        "if errorlevel 1 echo [warning] python-pdal import test failed, but pdal.exe may still work for CLI conversion.",
        "",
        f"powershell -NoProfile -ExecutionPolicy Bypass -Command \"$o=@{{runtime='%RUNTIME%'; env='%ENVPATH%'; pdal='%PDAL_EXE%'; python='{py_exe}'}}; $o | ConvertTo-Json | Set-Content -Encoding UTF8 '{meta_json}'\"",
        "echo.",
        "echo [OK] PDAL conda-forge runtime is ready.",
        "echo [PDAL] %PDAL_EXE%",
        "echo Restart Mustatil if the LiDAR converter still does not see pdal.exe.",
        "exit /b 0",
    ]
    script.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
    return script


def _write_posix_install_script() -> Path:
    rt = _runtime_root()
    rt.mkdir(parents=True, exist_ok=True)
    script = rt / "install_or_repair_pdal_condaforge.sh"
    env_dir = _env_root()
    meta_json = rt / "mustatil_pdal_condaforge_runtime.json"
    installer = rt / "Miniforge3.sh"
    arch = "$(uname -m)"
    text = f"""#!/usr/bin/env bash
set -e
export RUNTIME={_quote_cmd(rt)}
export ENVPATH={_quote_cmd(env_dir)}
export INSTALLER={_quote_cmd(installer)}
echo "================================================"
echo "Mustatil PDAL conda-forge installer / repair"
echo "================================================"
mkdir -p "$RUNTIME"
CONDA_EXE=""
if [ -x "$RUNTIME/bin/conda" ]; then CONDA_EXE="$RUNTIME/bin/conda"; fi
if [ -z "$CONDA_EXE" ] && command -v conda >/dev/null 2>&1; then CONDA_EXE="$(command -v conda)"; fi
if [ -z "$CONDA_EXE" ]; then
  echo "[conda missing] Installing local Miniforge runtime for Mustatil..."
  URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-${{arch}}.sh"
  if [ ! -f "$INSTALLER" ]; then
    if command -v curl >/dev/null 2>&1; then curl -L "$URL" -o "$INSTALLER"; else wget -O "$INSTALLER" "$URL"; fi
  fi
  bash "$INSTALLER" -b -p "$RUNTIME"
  CONDA_EXE="$RUNTIME/bin/conda"
fi
"$CONDA_EXE" --version
if [ -d "$ENVPATH/conda-meta" ]; then
  "$CONDA_EXE" install -y -p "$ENVPATH" -c conda-forge pdal python-pdal gdal rasterio numpy python=3.11
else
  "$CONDA_EXE" create -y -p "$ENVPATH" -c conda-forge pdal python-pdal gdal rasterio numpy python=3.11
fi
PDAL_EXE="$ENVPATH/bin/pdal"
PY_EXE="$ENVPATH/bin/python"
"$PDAL_EXE" --version
"$PY_EXE" - <<'EOF' || true
import sys, pdal
print('python', sys.executable)
print('pdal module OK', getattr(pdal, '__version__', 'OK'))
EOF
cat > {_quote_cmd(meta_json)} <<EOF
{{"runtime":"$RUNTIME","env":"$ENVPATH","pdal":"$PDAL_EXE","python":"$PY_EXE"}}
EOF
echo "[OK] PDAL conda-forge runtime is ready: $PDAL_EXE"
"""
    script.write_text(text, encoding="utf-8")
    try:
        script.chmod(0o755)
    except Exception:
        pass
    return script


def _install_script_path() -> Path:
    if _is_windows():
        return _write_windows_install_script()
    return _write_posix_install_script()


def _prepend_runtime_to_geo_module(mod: Any) -> None:
    """Make the original Geospatial Operations module prefer the local PDAL runtime."""
    try:
        dirs = _local_pdal_dirs()
        # Add expected dirs even before they exist, so the original functions see them after install.
        env = _env_root()
        if _is_windows():
            expected = [env / "Library" / "bin", env / "Scripts", env]
        else:
            expected = [env / "bin", env]
        for p in expected:
            s = str(p)
            if s not in dirs:
                dirs.append(s)
        extra = getattr(mod, "_EXTRA_TOOL_DIRS", None)
        if isinstance(extra, list):
            for d in reversed(dirs):
                if d and d not in extra:
                    extra.insert(0, d)
    except Exception:
        pass

    try:
        if not getattr(mod, "_MUSTATIL_PDAL_DIRECT_PATCH_FIND", False):
            orig_find_pdal = getattr(mod, "_find_pdal_executable", None)

            def _find_pdal_executable_patched(name: str = "pdal", extra_dirs: Optional[List[Any]] = None) -> str:
                exe = _local_pdal_exe()
                if exe:
                    return exe
                merged = list(extra_dirs or []) + _local_pdal_dirs()
                if callable(orig_find_pdal):
                    try:
                        return orig_find_pdal(name, merged)
                    except TypeError:
                        return orig_find_pdal(name)
                    except Exception:
                        pass
                return ""

            setattr(mod, "_find_pdal_executable", _find_pdal_executable_patched)
            setattr(mod, "_MUSTATIL_PDAL_DIRECT_PATCH_FIND", True)
    except Exception:
        pass

    try:
        if not getattr(mod, "_MUSTATIL_PDAL_DIRECT_PATCH_ENV", False):
            orig_env = getattr(mod, "_env_for_process", None)

            def _env_for_process_patched(extra_paths: Optional[List[Any]] = None) -> dict:
                env = orig_env((extra_paths or []) + _local_pdal_dirs()) if callable(orig_env) else dict(os.environ)
                dirs = _local_pdal_dirs()
                if dirs:
                    old = env.get("PATH", "")
                    env["PATH"] = os.pathsep.join(dirs + ([old] if old else []))
                exe = _local_pdal_exe()
                if exe:
                    env["MUSTATIL_PDAL_EXE"] = exe
                    env["MUSTATIL_PDAL_CONDA_ENV"] = str(_env_root())
                    env["MUSTATIL_PDAL_CONDA_RUNTIME"] = str(_runtime_root())
                return env

            setattr(mod, "_env_for_process", _env_for_process_patched)
            setattr(mod, "_MUSTATIL_PDAL_DIRECT_PATCH_ENV", True)
    except Exception:
        pass


def _run_pdal_install_from_hub(hub: Any, mod: Any = None) -> None:
    try:
        if mod is not None:
            _prepend_runtime_to_geo_module(mod)
        console = getattr(hub, "gdal_console", None)
        if console is None:
            _log("No gdal_console on ConsoleHubWidget")
            return
        try:
            tabs = getattr(hub, "console_tabs", None)
            if tabs is not None and hasattr(tabs, "setCurrentWidget"):
                tabs.setCurrentWidget(console)
        except Exception:
            pass
        try:
            out = getattr(console, "output", None)
            if out is not None:
                out.clear()
        except Exception:
            pass
        try:
            console._append_line("[PDAL conda-forge] Installing/repairing local PDAL runtime for Mustatil...")
            console._append_line(f"[runtime] {_runtime_root()}")
        except Exception:
            pass
        script = _install_script_path()
        if _is_windows():
            cmd = str(script)
        else:
            cmd = "bash " + _quote_cmd(script)
        try:
            if hasattr(hub, "_run_in_shell_console"):
                hub._run_in_shell_console(console, cmd)
            else:
                console.command_edit.setText(cmd)
                console.run_command()
        except Exception as exc:
            try:
                console._append_line(f"[PDAL install failed to start] {exc}")
            except Exception:
                _log("PDAL install failed to start: " + str(exc))
    except Exception as exc:
        _log("button action failed: " + str(exc))
        try:
            traceback.print_exc()
        except Exception:
            pass


def _widget_text(w: Any) -> str:
    try:
        if hasattr(w, "text"):
            return str(w.text())
    except Exception:
        pass
    return ""


def _layout_item_widget(layout: Any, index: int) -> Any:
    try:
        item = layout.itemAt(index)
        if item is not None:
            return item.widget()
    except Exception:
        pass
    return None


def _iter_layouts(layout: Any) -> Iterable[Any]:
    if layout is None:
        return
    yield layout
    try:
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item is None:
                continue
            sub = item.layout()
            if sub is not None:
                yield from _iter_layouts(sub)
            w = item.widget()
            if w is not None:
                try:
                    wl = w.layout()
                    if wl is not None:
                        yield from _iter_layouts(wl)
                except Exception:
                    pass
    except Exception:
        pass


def _button_already_exists(hub: Any) -> bool:
    try:
        for b in hub.findChildren(QPushButton):
            if "pdal" in _widget_text(b).lower() and "conda" in _widget_text(b).lower():
                return True
    except Exception:
        pass
    return False


def _make_pdal_button(hub: Any, mod: Any) -> Any:
    btn = QPushButton("Install PDAL conda-forge")
    try:
        btn.setMinimumHeight(32)
        btn.setToolTip(
            "Install/repair a local conda-forge PDAL runtime for LiDAR LAS/LAZ conversion. "
            "The runtime is stored under mustatil_model_runtimes/PDAL-CONDAFORGE and is preferred by the Geospatial Operations PDAL lookup."
        )
        btn.setStyleSheet("QPushButton { font-weight: 700; padding: 6px 10px; }")
    except Exception:
        pass
    btn.clicked.connect(lambda _checked=False, h=hub, m=mod: _run_pdal_install_from_hub(h, m))
    return btn


def _install_button_on_hub(hub: Any, mod: Any = None) -> bool:
    if QPushButton is None:
        return False
    try:
        if hub is None or _button_already_exists(hub):
            return False
        if mod is not None:
            _prepend_runtime_to_geo_module(mod)

        btn = _make_pdal_button(hub, mod)
        root_layout = hub.layout() if hasattr(hub, "layout") else None
        inserted = False

        # Best placement: same toolbar layout, immediately after "Check all consoles".
        try:
            for lay in _iter_layouts(root_layout):
                n = lay.count()
                texts = [_widget_text(_layout_item_widget(lay, i)).lower() for i in range(n)]
                check_idx = -1
                for i, txt in enumerate(texts):
                    if "check all consoles" in txt:
                        check_idx = i
                        break
                if check_idx >= 0:
                    lay.insertWidget(check_idx + 1, btn, 0)
                    inserted = True
                    break
        except Exception:
            inserted = False

        # Second placement: same top toolbar after the big Install/Repair button or QGIS installer button.
        if not inserted:
            try:
                for lay in _iter_layouts(root_layout):
                    n = lay.count()
                    texts = [_widget_text(_layout_item_widget(lay, i)).lower() for i in range(n)]
                    if any("install" in t or "repair" in t or "qgis" in t for t in texts):
                        # Insert before first stretch if possible, otherwise append.
                        insert_at = n
                        for i in range(n):
                            try:
                                item = lay.itemAt(i)
                                if item is not None and item.spacerItem() is not None:
                                    insert_at = i
                                    break
                            except Exception:
                                pass
                        lay.insertWidget(insert_at, btn, 0)
                        inserted = True
                        break
            except Exception:
                inserted = False

        # Final visible fallback: insert as second row in the ConsoleHub widget itself.
        if not inserted:
            try:
                if root_layout is not None:
                    if hasattr(root_layout, "insertWidget"):
                        root_layout.insertWidget(1, btn, 0)
                    else:
                        root_layout.addWidget(btn)
                    inserted = True
            except Exception:
                inserted = False

        if inserted:
            try:
                setattr(hub, "_mustatil_pdal_condaforge_button", btn)
            except Exception:
                pass
            _log("PDAL conda-forge button inserted into ConsoleHubWidget")
            return True
    except Exception as exc:
        _log("insert button failed: " + str(exc))
        try:
            traceback.print_exc()
        except Exception:
            pass
    return False


def _looks_like_geospatial_module(mod: Any) -> bool:
    try:
        if getattr(mod, "PLUGIN_NAME", "") == "Mustatil Geospatial Operations":
            return True
        if getattr(mod, "TAB_LABEL", "") == "Geospatial Operations" and hasattr(mod, "ConsoleHubWidget"):
            return True
        if hasattr(mod, "ConsoleHubWidget") and hasattr(mod, "GeospatialOperationsTab") and hasattr(mod, "_find_pdal_executable"):
            return True
    except Exception:
        pass
    return False


def _find_geospatial_modules() -> List[Any]:
    out = []
    seen = set()
    for mod in list(sys.modules.values()):
        try:
            if mod is None or id(mod) in seen:
                continue
            if _looks_like_geospatial_module(mod):
                seen.add(id(mod))
                out.append(mod)
        except Exception:
            pass
    return out


def _patch_geospatial_module(mod: Any) -> bool:
    try:
        if not _looks_like_geospatial_module(mod):
            return False
        _prepend_runtime_to_geo_module(mod)
        cls = getattr(mod, "ConsoleHubWidget", None)
        if cls is None:
            return False
        if id(cls) not in _PATCHED_MODULE_IDS:
            orig_build = getattr(cls, "_build_ui", None)
            if callable(orig_build):
                def _build_ui_patched(self, *args, __orig=orig_build, __mod=mod, **kwargs):
                    res = __orig(self, *args, **kwargs)
                    try:
                        _install_button_on_hub(self, __mod)
                    except Exception:
                        pass
                    return res
                setattr(cls, "_build_ui", _build_ui_patched)
                _PATCHED_MODULE_IDS.add(id(cls))
                _log("ConsoleHubWidget._build_ui patched")
        return True
    except Exception as exc:
        _log("module patch failed: " + str(exc))
        return False


def _scan_existing_hubs() -> int:
    if QApplication is None:
        return 0
    count = 0
    try:
        app = QApplication.instance()
        if not app:
            return 0
        mods = _find_geospatial_modules()
        for mod in mods:
            _patch_geospatial_module(mod)
        for w in app.allWidgets():
            try:
                # Concrete attributes of ConsoleHubWidget from v17.
                if hasattr(w, "console_tabs") and hasattr(w, "gdal_console") and hasattr(w, "run_all_console_checks"):
                    mod = mods[0] if mods else None
                    if _install_button_on_hub(w, mod):
                        count += 1
            except Exception:
                pass
    except Exception as exc:
        _log("hub scan failed: " + str(exc))
    return count


def _periodic_scan() -> None:
    try:
        for mod in _find_geospatial_modules():
            _patch_geospatial_module(mod)
        _scan_existing_hubs()
    except Exception:
        pass


def _install_qtab_hook() -> None:
    global _PATCHED_QTAB, _ORIG_ADD_TAB, _ORIG_INSERT_TAB
    if QTabWidget is None or QTimer is None or _PATCHED_QTAB:
        return
    try:
        _ORIG_ADD_TAB = QTabWidget.addTab
        _ORIG_INSERT_TAB = QTabWidget.insertTab

        def addTab_patched(self, page, *args, **kwargs):
            res = _ORIG_ADD_TAB(self, page, *args, **kwargs)
            try:
                QTimer.singleShot(100, _periodic_scan)
                QTimer.singleShot(700, _periodic_scan)
            except Exception:
                pass
            return res

        def insertTab_patched(self, index, page, *args, **kwargs):
            res = _ORIG_INSERT_TAB(self, index, page, *args, **kwargs)
            try:
                QTimer.singleShot(100, _periodic_scan)
                QTimer.singleShot(700, _periodic_scan)
            except Exception:
                pass
            return res

        QTabWidget.addTab = addTab_patched
        QTabWidget.insertTab = insertTab_patched
        _PATCHED_QTAB = True
        _log("QTabWidget scan hook installed")
    except Exception as exc:
        _log("QTabWidget hook failed: " + str(exc))


def _install_patch(root: Any = None) -> bool:
    global _SCAN_TIMER
    try:
        for mod in _find_geospatial_modules():
            _patch_geospatial_module(mod)
        _scan_existing_hubs()
        _install_qtab_hook()
        if QTimer is not None:
            for ms in (100, 300, 800, 1500, 3000, 6000, 10000, 15000, 25000):
                QTimer.singleShot(ms, _periodic_scan)
            try:
                if _SCAN_TIMER is None:
                    _SCAN_TIMER = QTimer()
                    _SCAN_TIMER.setInterval(2500)
                    _SCAN_TIMER.timeout.connect(_periodic_scan)
                    _SCAN_TIMER.start()
            except Exception:
                pass
        _log("patch installed")
        return True
    except Exception as exc:
        _log("install patch failed: " + str(exc))
        try:
            traceback.print_exc()
        except Exception:
            pass
        return False


def mustatil_plugin_init():
    return _install_patch()


def register_plugin(app=None, main_window=None):
    return _install_patch(main_window or app)


def init_plugin(app=None, main_window=None):
    return register_plugin(app, main_window)


def load_plugin(app=None, main_window=None):
    return register_plugin(app, main_window)


try:
    _install_patch()
except Exception:
    pass
