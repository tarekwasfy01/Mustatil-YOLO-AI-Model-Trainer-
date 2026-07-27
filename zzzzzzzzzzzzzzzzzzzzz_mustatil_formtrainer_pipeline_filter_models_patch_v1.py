# -*- coding: utf-8 -*-
"""
Mustatil patch plugin: PDAL micromamba isolated-root button for Geospatial Operations.

Small additive patch for the existing Geospatial Operations tab plugin. It adds
one button near "Check all consoles" and uses micromamba with a fresh
isolated root/cache for every install run. This avoids stale libmamba package
cache folders that cannot be removed while Mustatil/Windows still has handles open.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable, List, Optional

PLUGIN_NAME = "Mustatil Geospatial PDAL micromamba isolated-root button patch"
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


def _micromamba_exe() -> Path:
    rt = _runtime_root()
    if _is_windows():
        return rt / "micromamba" / "Library" / "bin" / "micromamba.exe"
    return rt / "micromamba" / "bin" / "micromamba"


def _local_pdal_dirs() -> List[str]:
    env = _env_root()
    rt = _runtime_root()
    mm = _micromamba_exe().parent
    if _is_windows():
        candidates = [
            env / "Library" / "bin",
            env / "Scripts",
            env,
            mm,
            rt / "Library" / "bin",
            rt / "Scripts",
            rt,
        ]
    else:
        candidates = [env / "bin", env / "lib", mm, rt / "bin", rt]
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


try:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QPushButton, QTabWidget
except Exception:  # pragma: no cover
    QTimer = None  # type: ignore
    QApplication = None  # type: ignore
    QPushButton = None  # type: ignore
    QTabWidget = None  # type: ignore


def _write_windows_install_script() -> Path:
    rt = _runtime_root()
    rt.mkdir(parents=True, exist_ok=True)
    script = rt / "install_or_repair_pdal_micromamba_isolated.cmd"
    archive = rt / "micromamba-win-64-latest.tar.bz2"
    extract_dir = rt / "micromamba"
    env_dir = _env_root()
    # IMPORTANT: do NOT use the old persistent mamba-root for installs.  The
    # user's logs show Windows/libmamba repeatedly reusing broken extracted
    # package directories under PDAL-CONDAFORGE\mamba-root\pkgs.  Use a fresh
    # root and fresh pkgs dir in %TEMP% for every run, then only keep the final
    # env under Mustatil.
    micromamba = _micromamba_exe()
    pdal_exe = env_dir / "Library" / "bin" / "pdal.exe"
    meta_json = rt / "mustatil_pdal_condaforge_runtime.json"
    log_file = rt / "pdal_micromamba_isolated_install_last.log"

    lines = [
        "@echo off",
        "chcp 65001 >nul",
        "setlocal EnableExtensions EnableDelayedExpansion",
        "echo ================================================",
        "echo Mustatil PDAL micromamba isolated-root repair",
        "echo ================================================",
        f"set \"RUNTIME={rt}\"",
        f"set \"ARCHIVE={archive}\"",
        f"set \"EXTRACT={extract_dir}\"",
        f"set \"ENVPATH={env_dir}\"",
        f"set \"MICROMAMBA_EXE={micromamba}\"",
        f"set \"PDAL_EXE={pdal_exe}\"",
        f"set \"LOGFILE={log_file}\"",
        f"set \"META_JSON={meta_json}\"",
        "set \"RUNID=%RANDOM%%RANDOM%\"",
        "set \"MAMBA_ROOT_PREFIX=%TEMP%\\mustatil-pdal-mamba-root-!RUNID!\"",
        "set \"MAMBA_PKGS_DIRS=%TEMP%\\mustatil-pdal-mamba-pkgs-!RUNID!\"",
        "set \"MAMBA_NO_BANNER=1\"",
        "set \"MAMBA_ALWAYS_YES=true\"",
        "set \"SPECS=pdal\"",
        "echo [runtime] %RUNTIME%",
        "echo [env] %ENVPATH%",
        "echo [fresh-root] %MAMBA_ROOT_PREFIX%",
        "echo [fresh-cache] %MAMBA_PKGS_DIRS%",
        "echo [specs] %SPECS%",
        "if not exist \"%RUNTIME%\" mkdir \"%RUNTIME%\"",
        "if not exist \"%MAMBA_ROOT_PREFIX%\" mkdir \"%MAMBA_ROOT_PREFIX%\"",
        "if not exist \"%MAMBA_PKGS_DIRS%\" mkdir \"%MAMBA_PKGS_DIRS%\"",
        "echo Mustatil PDAL micromamba isolated-root repair > \"%LOGFILE%\"",
        "echo [note] Old broken cache under %RUNTIME%\\mamba-root is intentionally ignored.",
        "",
        "if not exist \"%MICROMAMBA_EXE%\" (",
        "  echo [micromamba] downloading standalone micromamba...",
        "  if exist \"%ARCHIVE%\" del /f /q \"%ARCHIVE%\" >nul 2>nul",
        "  powershell -NoProfile -ExecutionPolicy Bypass -Command \"$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://micro.mamba.pm/api/micromamba/win-64/latest' -OutFile $env:ARCHIVE\"",
        "  if errorlevel 1 ( echo [ERROR] micromamba download failed. & exit /b 11 )",
        "  if exist \"%EXTRACT%\" powershell -NoProfile -ExecutionPolicy Bypass -Command \"$ErrorActionPreference='SilentlyContinue'; Remove-Item -LiteralPath $env:EXTRACT -Recurse -Force\"",
        "  mkdir \"%EXTRACT%\"",
        "  echo [micromamba] extracting archive with Windows tar.exe...",
        "  where tar >nul 2>nul",
        "  if errorlevel 1 ( echo [ERROR] tar.exe is missing. & exit /b 12 )",
        "  tar -xjf \"%ARCHIVE%\" -C \"%EXTRACT%\"",
        "  if errorlevel 1 ( echo [ERROR] micromamba extraction failed. & exit /b 13 )",
        ") else (",
        "  echo [micromamba] already present: %MICROMAMBA_EXE%",
        ")",
        "",
        "if not exist \"%MICROMAMBA_EXE%\" ( echo [ERROR] micromamba.exe was not created at %MICROMAMBA_EXE% & dir /s /b \"%EXTRACT%\" & exit /b 14 )",
        "\"%MICROMAMBA_EXE%\" --version",
        "if errorlevel 1 ( echo [ERROR] micromamba --version failed. & exit /b 15 )",
        "",
        "echo [cleanup] removing incomplete env if pdal.exe is missing...",
        "if exist \"%ENVPATH%\" if not exist \"%PDAL_EXE%\" (",
        "  powershell -NoProfile -ExecutionPolicy Bypass -Command \"$ErrorActionPreference='SilentlyContinue'; if(Test-Path -LiteralPath $env:ENVPATH){ Remove-Item -LiteralPath $env:ENVPATH -Recurse -Force }\"",
        ")",
        "",
        "echo [env] creating/updating minimal PDAL CLI environment via conda-forge...",
        "call :DO_INSTALL",
        "if errorlevel 1 (",
        "  echo [repair] first isolated-root install failed.",
        "  echo [repair] using second fresh root/cache and retrying once...",
        "  set \"RUNID=%RANDOM%%RANDOM%\"",
        "  set \"MAMBA_ROOT_PREFIX=%TEMP%\\mustatil-pdal-mamba-root-!RUNID!\"",
        "  set \"MAMBA_PKGS_DIRS=%TEMP%\\mustatil-pdal-mamba-pkgs-!RUNID!\"",
        "  powershell -NoProfile -ExecutionPolicy Bypass -Command \"$ErrorActionPreference='SilentlyContinue'; foreach($p in @($env:ENVPATH,$env:MAMBA_ROOT_PREFIX,$env:MAMBA_PKGS_DIRS)){ if($p -and (Test-Path -LiteralPath $p)){ Remove-Item -LiteralPath $p -Recurse -Force } }; New-Item -ItemType Directory -Force -Path $env:MAMBA_ROOT_PREFIX,$env:MAMBA_PKGS_DIRS | Out-Null\"",
        "  call :DO_INSTALL",
        "  if errorlevel 1 ( echo [ERROR] micromamba PDAL isolated install failed after retry. & echo [hint] Close Mustatil and manually delete %ENVPATH%, then try again. Old %RUNTIME%\\mamba-root is no longer used. & exit /b 20 )",
        ")",
        "goto AFTER_INSTALL",
        "",
        ":DO_INSTALL",
        "echo [using-root] %MAMBA_ROOT_PREFIX%",
        "echo [using-cache] %MAMBA_PKGS_DIRS%",
        "if exist \"%ENVPATH%\\conda-meta\" (",
        "  \"%MICROMAMBA_EXE%\" --no-rc install -y -r \"%MAMBA_ROOT_PREFIX%\" -p \"%ENVPATH%\" -c conda-forge --override-channels --strict-channel-priority %SPECS%",
        ") else (",
        "  \"%MICROMAMBA_EXE%\" --no-rc create -y -r \"%MAMBA_ROOT_PREFIX%\" -p \"%ENVPATH%\" -c conda-forge --override-channels --strict-channel-priority %SPECS%",
        ")",
        "exit /b %ERRORLEVEL%",
        "",
        ":AFTER_INSTALL",
        "echo [test] PDAL executable",
        "if not exist \"%PDAL_EXE%\" ( echo [ERROR] pdal.exe was not found at %PDAL_EXE% & dir /s /b \"%ENVPATH%\" | findstr /i \"pdal\" & exit /b 21 )",
        "\"%PDAL_EXE%\" --version",
        "if errorlevel 1 ( echo [ERROR] pdal --version failed. & exit /b 22 )",
        "echo [test] PDAL LAS/GDAL/SMRF driver availability",
        "\"%PDAL_EXE%\" --drivers | findstr /i \"readers.las writers.gdal filters.smrf\"",
        f"powershell -NoProfile -ExecutionPolicy Bypass -Command \"$o=@{{runtime='%RUNTIME%'; env='%ENVPATH%'; pdal='%PDAL_EXE%'; micromamba='%MICROMAMBA_EXE%'; root='%MAMBA_ROOT_PREFIX%'; cache='%MAMBA_PKGS_DIRS%'; specs='%SPECS%'}}; $o | ConvertTo-Json | Set-Content -Encoding UTF8 '{meta_json}'\"",
        "echo.",
        "echo [OK] PDAL micromamba isolated-root runtime is ready.",
        "echo [PDAL] %PDAL_EXE%",
        "echo Restart Mustatil if the LiDAR converter still does not see pdal.exe.",
        "exit /b 0",
    ]
    script.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
    return script

def _write_posix_install_script() -> Path:
    rt = _runtime_root()
    rt.mkdir(parents=True, exist_ok=True)
    script = rt / "install_or_repair_pdal_micromamba_isolated.sh"
    env_dir = _env_root()
    mamba_root = rt / "mamba-root"
    extract_dir = rt / "micromamba"
    archive = rt / "micromamba-latest.tar.bz2"
    micromamba = _micromamba_exe()
    meta_json = rt / "mustatil_pdal_condaforge_runtime.json"
    text = f"""#!/usr/bin/env bash
set -e
RUNTIME={_quote_cmd(rt)}
ENVPATH={_quote_cmd(env_dir)}
MAMBA_ROOT_PREFIX={_quote_cmd(mamba_root)}
EXTRACT={_quote_cmd(extract_dir)}
ARCHIVE={_quote_cmd(archive)}
MICROMAMBA_EXE={_quote_cmd(micromamba)}
mkdir -p "$RUNTIME" "$MAMBA_ROOT_PREFIX" "$EXTRACT"
echo "================================================"
echo "Mustatil PDAL micromamba / conda-forge repair"
echo "================================================"
echo "[runtime] $RUNTIME"
if [ ! -x "$MICROMAMBA_EXE" ]; then
  echo "[micromamba] downloading standalone micromamba..."
  OS=$(uname | tr '[:upper:]' '[:lower:]')
  ARCH=$(uname -m)
  if [ "$ARCH" = "x86_64" ] || [ "$ARCH" = "amd64" ]; then ARCH="64"; fi
  if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then ARCH="aarch64"; fi
  URL="https://micro.mamba.pm/api/micromamba/${{OS}}-${{ARCH}}/latest"
  if command -v curl >/dev/null 2>&1; then curl -L "$URL" -o "$ARCHIVE"; else wget -O "$ARCHIVE" "$URL"; fi
  tar -xjf "$ARCHIVE" -C "$EXTRACT"
fi
"$MICROMAMBA_EXE" --version
if [ -d "$ENVPATH" ] && [ ! -d "$ENVPATH/conda-meta" ]; then
  echo "[cleanup] removing partial broken env: $ENVPATH"
  rm -rf "$ENVPATH"
fi
if [ -d "$ENVPATH/conda-meta" ]; then
  "$MICROMAMBA_EXE" install -y -r "$MAMBA_ROOT_PREFIX" -p "$ENVPATH" -c conda-forge --strict-channel-priority pdal
else
  "$MICROMAMBA_EXE" create -y -r "$MAMBA_ROOT_PREFIX" -p "$ENVPATH" -c conda-forge --strict-channel-priority pdal
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
{{"runtime":"$RUNTIME","env":"$ENVPATH","pdal":"$PDAL_EXE","python":"$PY_EXE","micromamba":"$MICROMAMBA_EXE"}}
EOF
echo "[OK] PDAL micromamba runtime is ready: $PDAL_EXE"
"""
    script.write_text(text, encoding="utf-8")
    try:
        script.chmod(0o755)
    except Exception:
        pass
    return script


def _install_script_path() -> Path:
    return _write_windows_install_script() if _is_windows() else _write_posix_install_script()


def _prepend_runtime_to_geo_module(mod: Any) -> None:
    try:
        dirs = _local_pdal_dirs()
        env = _env_root()
        expected = ([env / "Library" / "bin", env / "Scripts", env, _micromamba_exe().parent] if _is_windows()
                    else [env / "bin", env, _micromamba_exe().parent])
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
        if not getattr(mod, "_MUSTATIL_PDAL_MICROMAMBA_PATCH_FIND", False):
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
            setattr(mod, "_MUSTATIL_PDAL_MICROMAMBA_PATCH_FIND", True)
    except Exception:
        pass

    try:
        if not getattr(mod, "_MUSTATIL_PDAL_MICROMAMBA_PATCH_ENV", False):
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
                    env["MUSTATIL_PDAL_MICROMAMBA"] = str(_micromamba_exe())
                return env

            setattr(mod, "_env_for_process", _env_for_process_patched)
            setattr(mod, "_MUSTATIL_PDAL_MICROMAMBA_PATCH_ENV", True)
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
            console._append_line("[PDAL micromamba isolated] Installing/repairing local PDAL runtime for Mustatil...")
            console._append_line(f"[runtime] {_runtime_root()}")
        except Exception:
            pass
        script = _install_script_path()
        cmd = str(script) if _is_windows() else "bash " + _quote_cmd(script)
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


def _hide_old_pdal_buttons(hub: Any) -> None:
    if QPushButton is None:
        return
    try:
        for b in hub.findChildren(QPushButton):
            t = _widget_text(b).lower()
            if "pdal" in t and (("conda" in t and "micromamba" not in t) or ("micromamba" in t and "isolated" not in t)):
                try:
                    b.setVisible(False)
                    b.setEnabled(False)
                except Exception:
                    pass
    except Exception:
        pass


def _button_already_exists(hub: Any) -> bool:
    try:
        for b in hub.findChildren(QPushButton):
            t = _widget_text(b).lower()
            if "pdal" in t and "micromamba" in t and "isolated" in t:
                return True
    except Exception:
        pass
    return False


def _make_pdal_button(hub: Any, mod: Any) -> Any:
    btn = QPushButton("Install PDAL micromamba isolated")
    try:
        btn.setMinimumHeight(32)
        btn.setToolTip(
            "Install/repair a local micromamba + conda-forge PDAL runtime for LiDAR LAS/LAZ conversion. "
            "This avoids old broken micromamba caches by using a fresh isolated root/cache for every install; only the final PDAL env is kept under mustatil_model_runtimes/PDAL-CONDAFORGE."
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
        if hub is None:
            return False
        _hide_old_pdal_buttons(hub)
        if _button_already_exists(hub):
            return False
        if mod is not None:
            _prepend_runtime_to_geo_module(mod)

        btn = _make_pdal_button(hub, mod)
        root_layout = hub.layout() if hasattr(hub, "layout") else None
        inserted = False

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

        if not inserted:
            try:
                for lay in _iter_layouts(root_layout):
                    n = lay.count()
                    texts = [_widget_text(_layout_item_widget(lay, i)).lower() for i in range(n)]
                    if any("install" in t or "repair" in t or "qgis" in t for t in texts):
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
                setattr(hub, "_mustatil_pdal_micromamba_button", btn)
            except Exception:
                pass
            _log("PDAL micromamba button inserted into ConsoleHubWidget")
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
