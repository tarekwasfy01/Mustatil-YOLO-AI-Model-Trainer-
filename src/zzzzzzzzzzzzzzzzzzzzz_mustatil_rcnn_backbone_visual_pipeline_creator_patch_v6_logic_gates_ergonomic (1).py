# -*- coding: utf-8 -*-
"""
Mustatil plugin: Geospatial Operations tab

Adds a top-level "Geospatial Operations" tab next to FormLearner / trainer tabs.

V17 simple QGIS/GDAL-first consoles:
- Top tool area uses tabs for LiDAR, model conversion, GPKG/vector, raster/DEM, tile/AI prep and environment checks.
- Bottom area is one large selectable console deck: Python/Venv, Rasterio/GIS Python, GDAL/OGR/PDAL and GMT.
- One large Install/Repair button repairs the Python GIS stack, prefers GDAL from QGIS/OSGeo4W when installed, falls back to existing standalone/GMT GDAL tools, and opens manual installer pages when GDAL/GMT is missing. No winget is used. Commands are launched with direct program+args rather than broken quoted cmd strings.
- Robust process output, absolute GDAL/GMT command resolution, portable Mustatil venv detection, QGIS/OSGeo4W-first provider discovery, optional manual installer buttons, and console commands that avoid PATH-only assumptions.

Features:
- Robust Python script console using Mustatil's current Python / venv; multi-line code runs from a temporary script instead of fragile REPL input. Python/Rasterio consoles use an isolated Python-GIS environment so external GMT/QGIS PROJ_LIB cannot break rasterio.
- GDAL/OGR/PDAL command console for gdalinfo, gdal_translate, gdalwarp, ogr2ogr and PDAL point-cloud commands. It prefers QGIS/OSGeo4W GDAL if present and otherwise uses standalone/GMT GDAL.
- GMT command console for Generic Mapping Tools commands.
- LiDAR LAS/LAZ -> GeoTIFF converter using PDAL when available.
- Model -> ONNX converter for Ultralytics YOLO, TorchScript, and generic pickled nn.Module models.
- GeoPackage converter using ogr2ogr to GeoJSON, CSV, KML, Shapefile, SQLite, or GPKG.
- Raster/DEM tools: COG, VRT, warp, clip, overviews, hillshade, slope, contours, polygonize/rasterize.
- Vector/tile tools: make valid, assign/reproject CRS, gdal2tiles, gdal_retile and tile index.
- Environment check for GDAL, GMT, PDAL, Torch, Ultralytics, ONNX, rasterio/geopandas.

Install: copy this .py into Mustatil's mustatil_plugins folder and restart Mustatil.
"""

from __future__ import annotations

import os
import sys
import json
import shlex
import traceback
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional, Dict, List

PLUGIN_NAME = "Mustatil Geospatial Operations"
TAB_LABEL = "Geospatial Operations"

_PATCHED_QTAB = False
_ORIG_ADD_TAB = None
_ORIG_INSERT_TAB = None
_SCAN_TIMER = None
_INSTALLING_TAB = False
_INSTALLED_TABWIDGETS = set()


def _log(msg: str) -> None:
    try:
        print(f"[{PLUGIN_NAME}] {msg}", flush=True)
    except Exception:
        pass


def _is_windows() -> bool:
    return os.name == "nt" or sys.platform.startswith("win")


def _q(s: str) -> str:
    """Quote a path/arg for shell commands.

    Windows fix: cmd.exe needs plain double quotes around paths. Do not
    escape quotes as \"...\", because then Windows tries to execute a
    program whose name literally contains quote characters.
    """
    s = str(s or "")
    if _is_windows():
        return '"' + s.replace('"', '""') + '"'
    return shlex.quote(s)


def _python_exe() -> str:
    """Return the best Python executable for Mustatil.

    Prefer the running interpreter, but also try the expected Mustatil root/.venv
    location when this plugin is loaded from mustatil_plugins.
    """
    candidates: List[Any] = []
    try:
        candidates.append(sys.executable)
    except Exception:
        pass
    try:
        here = Path(__file__).resolve()
        root = here.parent.parent if here.parent.name.lower() == "mustatil_plugins" else here.parent
        if _is_windows():
            candidates += [root / ".venv" / "Scripts" / "python.exe", root / "venv" / "Scripts" / "python.exe"]
        else:
            candidates += [root / ".venv" / "bin" / "python", root / "venv" / "bin" / "python"]
    except Exception:
        pass
    for c in candidates:
        try:
            if c and Path(str(c)).exists():
                return str(Path(str(c)).resolve())
        except Exception:
            continue
    return sys.executable or "python"


_EXTRA_TOOL_DIRS: List[str] = []


def _existing_dirs(paths: List[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for p in paths:
        try:
            if not p:
                continue
            pp = Path(str(p)).expanduser()
            if pp.is_file():
                pp = pp.parent
            if pp.exists() and pp.is_dir():
                s = str(pp.resolve())
                key = s.lower()
                if key not in seen:
                    seen.add(key)
                    out.append(s)
        except Exception:
            pass
    return out


def _qgis_osgeo_roots() -> List[str]:
    """Return likely QGIS/OSGeo4W installation roots.

    This is deliberately portable: it does not assume Tarek's S: drive.  It
    checks environment variables, PATH entries, Program Files, C:\OSGeo4W*, and
    C:\Programs.  The result is used for GDAL/OGR command-line tools and, when
    compatible, QGIS' Python osgeo bindings.
    """
    roots: List[Any] = []
    if _is_windows():
        for env_name in ["QGIS_PREFIX_PATH", "OSGEO4W_ROOT", "OSGEO4W_HOME", "QGIS_ROOT"]:
            v = os.environ.get(env_name)
            if v:
                roots.append(Path(v))
        # PATH often contains QGIS/OSGeo4W bin.  Walk one level up to root.
        for p in os.environ.get("PATH", "").split(os.pathsep):
            try:
                pp = Path(p)
                if not pp.exists():
                    continue
                low = str(pp).lower()
                if "qgis" in low or "osgeo4w" in low:
                    roots += [pp, pp.parent, pp.parent.parent]
            except Exception:
                pass
        bases = [
            Path(r"C:\Program Files"), Path(r"C:\Program Files (x86)"),
            Path(r"C:\Programs"), Path(r"C:\ProgramData"),
            Path(r"C:\OSGeo4W"), Path(r"C:\OSGeo4W64"),
        ]
        for base in bases:
            try:
                if base.name.lower().startswith("osgeo4w") and base.exists():
                    roots.append(base)
                if not base.exists() or not base.is_dir():
                    continue
                for pat in ["QGIS*", "OSGeo4W*", "qgis*"]:
                    roots.extend(base.glob(pat))
            except Exception:
                pass
    else:
        for p in ["/usr", "/usr/local", "/opt/qgis", "/Applications/QGIS.app/Contents/MacOS"]:
            roots.append(Path(p))
    return _existing_dirs(roots)


def _qgis_gdal_dirs() -> List[str]:
    """Prefer QGIS/OSGeo4W GDAL/OGR folders before generic PATH tools."""
    dirs: List[Any] = []
    for r in _qgis_osgeo_roots():
        root = Path(r)
        dirs += [
            root / "bin",
            root / "apps" / "gdal" / "bin",
            root / "Library" / "bin",
            root,
        ]
        # Some standalone QGIS layouts have versioned app folders below root/apps.
        try:
            for app in (root / "apps").glob("*"):
                dirs += [app / "bin", app / "Scripts"]
        except Exception:
            pass
    return _existing_dirs(dirs)


def _qgis_pythonpath_dirs() -> List[str]:
    """Return QGIS Python paths only when they look compatible with this Python.

    QGIS' osgeo bindings are compiled.  Adding a Python311 site-packages folder
    to Python312 would be unsafe.  Therefore this only adds PythonXY folders
    matching the running Mustatil interpreter, and only if an osgeo package is
    present.
    """
    dirs: List[Any] = []
    tag = f"Python{sys.version_info.major}{sys.version_info.minor}"
    for r in _qgis_osgeo_roots():
        root = Path(r)
        candidates = [
            root / "apps" / tag / "Lib" / "site-packages",
            root / "apps" / tag / "lib" / "site-packages",
            root / "apps" / "qgis" / "python",
            root / "python",
        ]
        for c in candidates:
            try:
                if c.exists() and ((c / "osgeo").exists() or "qgis" in str(c).lower()):
                    dirs.append(c)
            except Exception:
                pass
    return _existing_dirs(dirs)


def _common_tool_dirs() -> List[str]:
    """Common Windows/Linux folders for GDAL/GMT/QGIS/OSGeo4W tools."""
    dirs: List[Any] = []
    py = Path(_python_exe()).resolve()
    scripts = py.parent
    root = scripts.parent if scripts.name.lower() in {"scripts", "bin"} else scripts
    dirs += [scripts, root, root / "Scripts", root / "bin", root / "Library" / "bin"]
    # Prefer QGIS/OSGeo4W GDAL folders when present, but still include other
    # bundles like GMT's GDAL binaries and standalone GDAL installs.
    dirs += _qgis_gdal_dirs()
    if _is_windows():
        bases = [Path(r"C:\Program Files"), Path(r"C:\Program Files (x86)"), Path(r"C:\Programs"), Path(r"C:\ProgramData"), Path(r"C:\OSGeo4W"), Path(r"C:\OSGeo4W64")]
        dirs += [Path(r"C:\OSGeo4W\bin"), Path(r"C:\OSGeo4W64\bin"), Path(r"C:\programs\gmt6\bin"), Path(r"C:\Programs\GMT6\bin"), Path(r"C:\Programs\QGIS\bin")]
        for base in bases:
            try:
                if not base.exists():
                    continue
                for qgis in base.glob("QGIS*"):
                    dirs += [qgis / "bin", qgis / "apps" / "gdal" / "bin", qgis / "apps" / "Python312" / "Scripts", qgis / "apps" / "Python311" / "Scripts", qgis / "apps" / "Python39" / "Scripts"]
                for gmt in list(base.glob("GMT*")) + list(base.glob("Generic Mapping Tools*")):
                    dirs += [gmt / "bin", gmt]
                for gdal in base.glob("GDAL*"):
                    dirs += [gdal / "bin", gdal]
            except Exception:
                pass
    else:
        dirs += [Path("/usr/bin"), Path("/usr/local/bin"), Path("/opt/homebrew/bin"), Path("/opt/local/bin")]
        for d in [Path("/opt/gmt"), Path("/usr/local/gmt")]:
            dirs += [d / "bin", d]
    return _existing_dirs([*_EXTRA_TOOL_DIRS, *dirs])


def _find_gdal_executable(name: str, extra_dirs: Optional[List[Any]] = None) -> str:
    """Find GDAL/OGR tools, preferring QGIS/OSGeo4W if present.

    GMT can ship gdalinfo/ogr2ogr too.  Those are usable, but QGIS/OSGeo4W is
    usually a fuller GDAL/OGR environment, so this function tries QGIS first for
    GDAL-family tools and falls back to generic PATH/GMT/venv discovery.
    """
    names = [name]
    if _is_windows() and not name.lower().endswith((".exe", ".bat", ".cmd", ".py")):
        names += [name + ".exe", name + ".bat", name + ".cmd"]
    preferred = _existing_dirs((extra_dirs or []) + _qgis_gdal_dirs())
    for d in preferred:
        for n in names:
            try:
                p = Path(d) / n
                if p.exists() and p.is_file():
                    return str(p.resolve())
            except Exception:
                pass
    return _find_executable(name, extra_dirs or [])

def _find_pdal_executable(name: str = "pdal", extra_dirs: Optional[List[Any]] = None) -> str:
    """Find PDAL in PATH, QGIS/OSGeo4W, common Program Files folders, or conda-style folders.

    PDAL is treated as an optional LiDAR/point-cloud tool.  It belongs in the
    GDAL console because it is usually used together with GDAL outputs
    (LAS/LAZ -> GeoTIFF/DEM/DSM).
    """
    names = [name]
    if _is_windows() and not name.lower().endswith((".exe", ".bat", ".cmd")):
        names += [name + ".exe", name + ".bat", name + ".cmd"]
    dirs: List[Any] = []
    dirs += list(extra_dirs or [])
    dirs += _common_tool_dirs()
    if _is_windows():
        # OSGeo4W and QGIS often provide PDAL or compatible PATH setup.  Conda
        # environments are also common for PDAL on Windows.
        dirs += [
            Path(r"C:\OSGeo4W\bin"),
            Path(r"C:\OSGeo4W64\bin"),
            Path(r"C:\Program Files\PDAL\bin"),
            Path(r"C:\Program Files\PDAL"),
            Path(r"C:\Programs\PDAL\bin"),
            Path(r"C:\Programs\PDAL"),
        ]
        for base in [Path(r"C:\Program Files"), Path(r"C:\Program Files (x86)"), Path(r"C:\Programs"), Path.home() / "miniconda3", Path.home() / "anaconda3"]:
            try:
                if not base.exists():
                    continue
                for d in base.glob("**/pdal*"):
                    if d.is_dir():
                        dirs += [d, d / "bin", d / "Scripts", d / "Library" / "bin"]
            except Exception:
                pass
    for n in names:
        try:
            w = shutil.which(n)
            if w:
                return str(Path(w).resolve())
        except Exception:
            pass
    for d in _existing_dirs(dirs):
        for n in names:
            try:
                p = Path(d) / n
                if p.exists() and p.is_file():
                    return str(p.resolve())
            except Exception:
                pass
    return ""

def _find_executable(name: str, extra_dirs: Optional[List[Any]] = None) -> str:
    """Find an executable by PATH plus common GIS/GMT/QGIS folders."""
    names = [name]
    if _is_windows() and not name.lower().endswith((".exe", ".bat", ".cmd", ".py")):
        names.append(name + ".exe")
        names.append(name + ".bat")
        names.append(name + ".cmd")
    for n in names:
        try:
            w = shutil.which(n)
            if w:
                return str(Path(w).resolve())
        except Exception:
            pass
    for d in _existing_dirs((extra_dirs or []) + _common_tool_dirs()):
        for n in names:
            try:
                p = Path(d) / n
                if p.exists() and p.is_file():
                    return str(p.resolve())
            except Exception:
                pass
    return ""

def _cmd_token_starts(command: str, token: str) -> bool:
    c = (command or "").lstrip()
    t = (token or "").strip()
    if not c or not t:
        return False
    cl = c.lower()
    tl = t.lower()
    return cl == tl or cl.startswith(tl + " ") or cl.startswith(tl + "	")


def _replace_first_token(command: str, token: str, replacement: str) -> str:
    c = (command or "").lstrip()
    prefix_len = len(command or "") - len(c)
    if not _cmd_token_starts(c, token):
        return command
    return (command or "")[:prefix_len] + _q(replacement) + c[len(token):]


def _tool_version_command(exe_or_name: str) -> str:
    exe = str(exe_or_name or "")
    low = Path(exe).name.lower()
    if "gmt" in low or low == "gmt":
        return f"{_q(exe)} --version"
    if low.startswith("gdalinfo") or low == "gdalinfo":
        return f"{_q(exe)} --version"
    return f"{_q(exe)} --version"


def _remember_tool_path(path: str) -> None:
    try:
        p = Path(str(path)).expanduser()
        d = p.parent if p.is_file() else p
        if d.exists() and d.is_dir():
            s = str(d.resolve())
            if s not in _EXTRA_TOOL_DIRS:
                _EXTRA_TOOL_DIRS.insert(0, s)
    except Exception:
        pass


def _env_for_process(extra_paths: Optional[List[Any]] = None) -> Dict[str, str]:
    env = dict(os.environ)
    py = Path(_python_exe()).resolve()
    env["MUSTATIL_PYTHON"] = str(py)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    scripts = py.parent
    root = scripts.parent if scripts.name.lower() in {"scripts", "bin"} else scripts
    extra = [str(scripts), str(root), str(root / "Scripts"), str(root / "bin"), str(root / "Library" / "bin")]
    extra += _existing_dirs(extra_paths or [])
    extra += _common_tool_dirs()
    old_path = env.get("PATH", "")
    sep = os.pathsep
    ordered: List[str] = []
    seen = set()
    for p in extra:
        try:
            if not p:
                continue
            pp = Path(str(p))
            if not pp.exists():
                continue
            s = str(pp.resolve())
            key = s.lower()
            if key not in seen:
                seen.add(key)
                ordered.append(s)
        except Exception:
            pass
    env["PATH"] = sep.join(ordered + [old_path])

    # Optional QGIS Python/osgeo bridge.  Only compatible PythonXY site-packages
    # folders are added; this lets another user with QGIS Python 3.12 import
    # `osgeo` from QGIS while avoiding cross-version DLL crashes.
    try:
        qpy = _qgis_pythonpath_dirs()
        if qpy:
            old_py_path = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = sep.join(qpy + ([old_py_path] if old_py_path else []))
            env["MUSTATIL_QGIS_PYTHONPATH"] = sep.join(qpy)
    except Exception:
        pass

    # Help GDAL/PROJ command-line tools started from the Mustatil process.
    # Some standalone Windows bundles (for example GMT with GDAL tools) have
    # gdalinfo.exe/ogr2ogr.exe in bin, while data folders live in ../share/* or
    # ../Library/share/*.  If these variables are missing, GDAL commands may be
    # found but fail or print incomplete output.
    try:
        for d in ordered:
            base = Path(d)
            roots = [base, base.parent]
            for r in list(roots):
                roots.append(r / "Library")
            if not env.get("GDAL_DATA"):
                for r in roots:
                    cand = r / "share" / "gdal"
                    if cand.exists() and cand.is_dir():
                        env["GDAL_DATA"] = str(cand.resolve())
                        break
            if not env.get("PROJ_LIB"):
                for r in roots:
                    cand = r / "share" / "proj"
                    if cand.exists() and cand.is_dir():
                        env["PROJ_LIB"] = str(cand.resolve())
                        break
            if env.get("GDAL_DATA") and env.get("PROJ_LIB"):
                break
    except Exception:
        pass
    return env


def _env_for_python_process(extra_paths: Optional[List[Any]] = None) -> Dict[str, str]:
    """Environment for Python/Rasterio/GIS packages.

    Critical Windows fix: external GIS suites such as GMT/QGIS/OSGeo4W may set
    PROJ_LIB/GDAL_DATA or put their DLL folders early on PATH.  That is useful
    for the GDAL/GMT command consoles, but it can break rasterio/pyproj because
    wheel packages ship/expect their own compatible PROJ database.  The user log
    showed rasterio reading C:\programs\gmt6\share\proj\proj.db and failing
    with DATABASE.LAYOUT.VERSION mismatch.  Therefore Python consoles remove
    external PROJ/GDAL variables and prioritize the Mustatil venv.
    """
    env = dict(os.environ)
    py = Path(_python_exe()).resolve()
    env["MUSTATIL_PYTHON"] = str(py)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    # Do not let GMT/QGIS/OSGeo4W PROJ/GDAL data override rasterio/pyproj wheels.
    for k in ["PROJ_LIB", "PROJ_DATA", "GDAL_DATA", "GDAL_DRIVER_PATH"]:
        env.pop(k, None)
    env.setdefault("PROJ_NETWORK", "ON")

    scripts = py.parent
    root = scripts.parent if scripts.name.lower() in {"scripts", "bin"} else scripts
    ordered: List[str] = []
    seen = set()

    def add_path(p: Any) -> None:
        try:
            if not p:
                return
            pp = Path(str(p))
            if pp.exists() and pp.is_dir():
                s = str(pp.resolve())
                key = s.lower()
                if key not in seen:
                    seen.add(key)
                    ordered.append(s)
        except Exception:
            pass

    # Venv first.  Keep user/system PATH afterwards, but remove obvious external
    # GIS bin folders from the front so rasterio does not accidentally load their
    # PROJ/GDAL DLLs before its own package DLLs.
    for p in [scripts, root, root / "Scripts", root / "bin", root / "Library" / "bin"]:
        add_path(p)
    for p in _existing_dirs(extra_paths or []):
        add_path(p)

    filtered_old: List[str] = []
    for p in env.get("PATH", "").split(os.pathsep):
        if not p:
            continue
        low = p.lower().replace("/", "\\")
        # Keep venv/system paths, drop external GIS DLL folders for Python package runs.
        if ("gmt" in low or "osgeo4w" in low or "qgis" in low) and ("\\bin" in low or "library\\bin" in low):
            continue
        filtered_old.append(p)

    env["PATH"] = os.pathsep.join(ordered + filtered_old)

    # Optional compatible QGIS Python path only.  Do not force it; osgeo is optional.
    try:
        qpy = _qgis_pythonpath_dirs()
        if qpy:
            old_py_path = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = os.pathsep.join(qpy + ([old_py_path] if old_py_path else []))
            env["MUSTATIL_QGIS_PYTHONPATH"] = os.pathsep.join(qpy)
    except Exception:
        pass
    return env


# -----------------------------------------------------------------------------
# Robust command runner helpers
# -----------------------------------------------------------------------------
def _strip_outer_quotes(s: str) -> str:
    s = str(s or "").strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1]
    return s


def _command_has_shell_syntax(cmd: str) -> bool:
    """Return True when a command needs a real shell script.

    Simple commands such as `gmt --version` or `"C:\\x\\gdalinfo.exe" --version`
    are started directly through QProcess(program, args).  Shell syntax such as
    &&, pipes, redirection, or multiline code is written to a temporary .cmd/.sh
    file and that script is launched.  This avoids the broken Windows case where
    cmd.exe received paths as \"C:\\...\" and then tried to execute the quotes.
    """
    c = str(cmd or "")
    if "\n" in c or "\r" in c:
        return True
    for op in ["&&", "||", "|", ">", "<"]:
        if op in c:
            return True
    # Single & is a Windows command separator.  Ignore it inside obvious URLs.
    if " & " in c or c.strip().startswith("& "):
        return True
    return False


def _split_simple_command(cmd: str) -> List[str]:
    raw = str(cmd or "").strip()
    if not raw:
        return []
    # remove a leading CMD call if a script path was generated by the plugin
    if raw.lower().startswith("call "):
        raw = raw[5:].strip()
    try:
        # On Windows, posix=True treats backslashes in C:\paths as escapes.
        # posix=False keeps Windows paths intact; surrounding quotes are stripped below.
        parts = shlex.split(raw, posix=(not _is_windows()))
    except Exception:
        try:
            parts = shlex.split(raw, posix=False)
        except Exception:
            parts = raw.split()
    return [_strip_outer_quotes(p) for p in parts if str(p).strip()]


def _resolve_command_program(program: str, extra_dirs: Optional[List[Any]] = None) -> str:
    p = _strip_outer_quotes(program)
    if not p:
        return p
    try:
        pp = Path(p)
        if pp.exists() and pp.is_file():
            return str(pp.resolve())
    except Exception:
        pass
    found = _find_executable(p, extra_dirs or [])
    if found:
        return found
    return p


def _write_temp_shell_script(command: str) -> str:
    tmp_dir = Path(tempfile.gettempdir()) / "mustatil_geospatial_ops"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    if _is_windows():
        script = tmp_dir / "mustatil_run_command.cmd"
        script.write_text("@echo off\r\nchcp 65001 >nul\r\n" + str(command) + "\r\n", encoding="utf-8")
        return str(script)
    script = tmp_dir / "mustatil_run_command.sh"
    script.write_text("#!/usr/bin/env bash\nset +e\n" + str(command) + "\n", encoding="utf-8")
    try:
        script.chmod(0o755)
    except Exception:
        pass
    return str(script)


def _program_args_for_qprocess(command: str, extra_dirs: Optional[List[Any]] = None) -> tuple[str, List[str], str]:
    raw = str(command or "").strip()
    if not raw:
        return ("cmd.exe" if _is_windows() else "bash", ["/d", "/c", "echo empty command"] if _is_windows() else ["-lc", "echo empty command"], "empty")

    # Generated .cmd/.sh scripts should be launched as scripts, not through a quoted cmd string.
    raw_no_call = raw[5:].strip() if raw.lower().startswith("call ") else raw
    maybe_path = _strip_outer_quotes(raw_no_call)
    if len(_split_simple_command(raw_no_call)) == 1:
        try:
            p = Path(maybe_path)
            if p.exists() and p.is_file() and p.suffix.lower() in {".cmd", ".bat"}:
                return "cmd.exe", ["/d", "/c", str(p)], "cmd.exe /d /c <script>"
            if p.exists() and p.is_file() and p.suffix.lower() == ".sh":
                return "bash", [str(p)], "bash <script>"
        except Exception:
            pass

    if _command_has_shell_syntax(raw):
        script = _write_temp_shell_script(raw)
        if _is_windows():
            return "cmd.exe", ["/d", "/c", script], "cmd.exe /d /c <temp .cmd>"
        return "bash", [script], "bash <temp .sh>"

    parts = _split_simple_command(raw)
    if not parts:
        return ("cmd.exe" if _is_windows() else "bash", ["/d", "/c", "echo empty command"] if _is_windows() else ["-lc", "echo empty command"], "empty")
    program = _resolve_command_program(parts[0], extra_dirs or [])
    args = parts[1:]
    try:
        suffix = Path(program).suffix.lower()
        if _is_windows() and suffix in {".cmd", ".bat"}:
            return "cmd.exe", ["/d", "/c", program] + args, "cmd.exe /d /c <batch>"
    except Exception:
        pass
    return program, args, "direct QProcess(program, args)"


def _start_qprocess_command(proc: Any, command: str, append_line: Optional[Any] = None, extra_dirs: Optional[List[Any]] = None) -> None:
    program, args, note = _program_args_for_qprocess(command, extra_dirs or [])
    try:
        if append_line:
            append_line(f"[runner] {note}")
            append_line(f"[program] {program}")
            if args:
                append_line(f"[args] {args}")
    except Exception:
        pass
    proc.start(program, args)

def _browse_filter_all() -> str:
    return "All files (*.*);;All files (*)"


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _find_workspace_from_tabwidget(tw: Any) -> Optional[Any]:
    try:
        p = tw.parent()
        for _ in range(25):
            if p is None:
                break
            if hasattr(p, "tabs") and getattr(p, "tabs", None) is tw:
                return p
            p = p.parent()
    except Exception:
        pass
    try:
        app = QApplication.instance()  # type: ignore[name-defined]
        if app:
            for w in app.allWidgets():
                try:
                    if getattr(w, "tabs", None) is tw:
                        return w
                except Exception:
                    pass
    except Exception:
        pass
    return None


def _tab_labels(tw: Any) -> List[str]:
    labels = []
    try:
        for i in range(tw.count()):
            labels.append(str(tw.tabText(i)))
    except Exception:
        pass
    return labels


def _has_tab(tw: Any, label: str = TAB_LABEL) -> bool:
    for t in _tab_labels(tw):
        if t.strip().lower() == label.strip().lower():
            return True
    return False


def _find_anchor_index(tw: Any) -> Optional[int]:
    """Prefer placing after FormLearner/Form Trainer, then after any trainer tab."""
    labels = _tab_labels(tw)
    low = [x.lower() for x in labels]
    preferred = [
        "form trainer",
        "formlearner trainer",
        "form learner trainer",
        "formlearner",
        "form learner",
    ]
    for needle in preferred:
        for i, text in enumerate(low):
            if needle in text:
                return i
    trainer_needles = ["r-cnn trainer", "rcnn trainer", "trainer", "training", "yolo trainer", "lae-dino trainer"]
    last = None
    for i, text in enumerate(low):
        if any(n in text for n in trainer_needles):
            last = i
    if last is not None:
        return last
    # Fallback: put near the end, but not before Detection if there are only few tabs.
    try:
        return max(0, tw.count() - 1)
    except Exception:
        return None


# Qt imports are delayed enough that py_compile works without PySide6 installed.
try:
    from PySide6.QtCore import Qt, QProcess, QTimer, QSize, QProcessEnvironment
    from PySide6.QtGui import QTextCursor
    from PySide6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
        QLabel, QPushButton, QLineEdit, QPlainTextEdit, QTextEdit, QFileDialog,
        QTabWidget, QSplitter, QGroupBox, QComboBox, QCheckBox, QSpinBox,
        QDoubleSpinBox, QToolButton, QSizePolicy, QMessageBox, QScrollArea
    )
except Exception:  # noqa: BLE001
    # Keep the module importable for syntax checks outside Mustatil/PySide6.
    class _QtDummyBase:  # pragma: no cover
        pass
    Qt = QProcess = QTimer = QSize = QProcessEnvironment = QTextCursor = None  # type: ignore
    QApplication = None  # type: ignore
    QWidget = _QtDummyBase  # type: ignore
    QVBoxLayout = QHBoxLayout = QGridLayout = QFormLayout = None  # type: ignore
    QLabel = QPushButton = QLineEdit = QPlainTextEdit = QTextEdit = QFileDialog = None  # type: ignore
    QTabWidget = QSplitter = QGroupBox = QComboBox = QCheckBox = QSpinBox = None  # type: ignore
    QDoubleSpinBox = QToolButton = QSizePolicy = QMessageBox = QScrollArea = None  # type: ignore


class _RunnerMixin:
    def _append(self, text: str) -> None:
        """Append process output reliably.

        Earlier versions used self.output.textCursor().End, which can fail silently in
        PySide6.  This version uses QTextCursor.End and falls back to appendPlainText,
        so console output is visible even when a command fails quickly.
        """
        try:
            if text is None:
                return
            text = str(text)
            out = getattr(self, "output", None)
            if out is None:
                print(text, end="", flush=True)
                return
            try:
                out.moveCursor(QTextCursor.End)  # type: ignore[union-attr]
                out.insertPlainText(text)
                out.moveCursor(QTextCursor.End)  # type: ignore[union-attr]
                out.ensureCursorVisible()
            except Exception:
                # Last-resort UI fallback: less pretty, but visible.
                try:
                    out.appendPlainText(text.rstrip("\n"))
                    out.ensureCursorVisible()
                except Exception:
                    print(text, end="", flush=True)
        except Exception as exc:
            try:
                print(f"[{PLUGIN_NAME} console append failed] {exc}", flush=True)
            except Exception:
                pass

    def _append_line(self, text: str = "") -> None:
        self._append(str(text) + "\n")

    def _set_env_on_process(self, proc: Any) -> None:
        try:
            try:
                env = QProcessEnvironment.systemEnvironment()  # type: ignore[union-attr]
            except Exception:
                env = proc.processEnvironment()
            for k, v in _env_for_process().items():
                env.insert(k, str(v))
            proc.setProcessEnvironment(env)
            try:
                proc.setProcessChannelMode(QProcess.MergedChannels)  # type: ignore[union-attr]
            except Exception:
                pass
        except Exception as exc:
            try:
                self._append_line(f"[could not set process environment] {exc}")
            except Exception:
                pass


class ShellConsoleWidget(QWidget, _RunnerMixin):  # type: ignore[misc,valid-type]
    def __init__(self, title: str, help_text: str, snippets: Dict[str, str], parent: Optional[Any] = None):
        super().__init__(parent)
        self.title = title
        self.proc = None
        self.snippets = snippets or {}
        self._build_ui(help_text)

    def _build_ui(self, help_text: str) -> None:
        layout = QVBoxLayout(self)
        header = QLabel(f"<b>{self.title}</b><br>{help_text}")
        header.setWordWrap(True)
        layout.addWidget(header)

        top = QHBoxLayout()
        self.cwd_edit = QLineEdit(str(Path.cwd()))
        browse = QPushButton("Working folder…")
        browse.clicked.connect(self._browse_cwd)
        top.addWidget(QLabel("CWD"))
        top.addWidget(self.cwd_edit, 1)
        top.addWidget(browse)
        layout.addLayout(top)

        row = QHBoxLayout()
        self.snip_combo = QComboBox()
        self.snip_combo.addItem("Insert command template…", "")
        for name, cmd in self.snippets.items():
            self.snip_combo.addItem(name, cmd)
        self.snip_combo.currentIndexChanged.connect(self._insert_snippet)
        row.addWidget(self.snip_combo, 1)
        self.run_btn = QPushButton("Run")
        self.stop_btn = QPushButton("Stop")
        self.clear_btn = QPushButton("Clear")
        self.run_btn.clicked.connect(self.run_command)
        self.stop_btn.clicked.connect(self.stop_process)
        self.clear_btn.clicked.connect(lambda: self.output.clear())
        row.addWidget(self.run_btn)
        row.addWidget(self.stop_btn)
        row.addWidget(self.clear_btn)
        layout.addLayout(row)

        self.command_edit = QLineEdit()
        self.command_edit.setPlaceholderText("Type a command and press Enter…")
        self.command_edit.returnPressed.connect(self.run_command)
        layout.addWidget(self.command_edit)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.output.setPlaceholderText("Command output appears here.")
        layout.addWidget(self.output, 1)

    def _browse_cwd(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Working folder", self.cwd_edit.text() or str(Path.cwd()))
        if d:
            self.cwd_edit.setText(d)

    def _insert_snippet(self) -> None:
        cmd = self.snip_combo.currentData()
        if cmd:
            self.command_edit.setText(str(cmd))
            self.command_edit.setFocus()
            try:
                self.snip_combo.setCurrentIndex(0)
            except Exception:
                pass

    def _prepare_command(self, cmd: str) -> str:
        return cmd

    def _process_ready(self) -> None:
        try:
            data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
            if data:
                self._append(data)
        except Exception:
            pass
        try:
            data = bytes(self.proc.readAllStandardError()).decode("utf-8", errors="replace")
            if data:
                self._append(data)
        except Exception:
            pass

    def _process_finished(self, code: int, status: Any = None) -> None:
        # Read any remaining buffered stdout/stderr before printing the final line.
        # Very short commands such as `gdalinfo --version` or `gmt --version` can
        # otherwise finish before readyRead is delivered, which looks like "no output".
        try:
            self._process_ready()
        except Exception:
            pass
        self._append_line(f"\n[finished exit_code={code}]\n")
        self.proc = None

    def run_command(self) -> None:
        cmd = self.command_edit.text().strip()
        if not cmd:
            return
        if self.proc is not None:
            self._append_line("[A command is already running. Stop it first.]\n")
            return
        cmd = self._prepare_command(cmd)
        cwd = self.cwd_edit.text().strip() or str(Path.cwd())
        self._append_line(f"\n> {cmd}")
        self.proc = QProcess(self)
        self._set_env_on_process(self.proc)
        try:
            self.proc.setWorkingDirectory(cwd)
        except Exception:
            pass
        self.proc.readyReadStandardOutput.connect(self._process_ready)
        self.proc.readyReadStandardError.connect(self._process_ready)
        self.proc.finished.connect(self._process_finished)
        try:
            self.proc.errorOccurred.connect(lambda e: self._append_line(f"[QProcess error] {e}"))
        except Exception:
            pass
        extra_dirs = []
        try:
            extra_dirs = self._tool_extra_dirs()  # type: ignore[attr-defined]
        except Exception:
            extra_dirs = []
        _start_qprocess_command(self.proc, cmd, self._append_line, extra_dirs)

    def stop_process(self) -> None:
        if self.proc is not None:
            try:
                self.proc.terminate()
                QTimer.singleShot(2000, lambda p=self.proc: p.kill() if p is not None and p.state() != QProcess.NotRunning else None)
                self._append_line("[stop requested]")
            except Exception:
                pass


class ToolShellConsoleWidget(ShellConsoleWidget):  # type: ignore[misc,valid-type]
    """Small command console with executable auto-detection for GDAL/GMT-style tools."""
    def __init__(self, title: str, help_text: str, snippets: Dict[str, str], tool_name: str, family: str = "tool", parent: Optional[Any] = None):
        self.tool_name = tool_name
        self.family = family
        super().__init__(title, help_text, snippets, parent)
        try:
            self.find_tool()
        except Exception:
            pass

    def _build_ui(self, help_text: str) -> None:
        layout = QVBoxLayout(self)
        header = QLabel(f"<b>{self.title}</b><br>{help_text}")
        header.setWordWrap(True)
        layout.addWidget(header)

        exe_row = QHBoxLayout()
        self.exe_edit = QLineEdit()
        self.exe_edit.setPlaceholderText(f"Optional path to {self.tool_name}.exe or its bin folder. Auto-detected if empty.")
        find_btn = QPushButton(f"Find {self.tool_name}")
        browse_btn = QPushButton("Browse…")
        test_btn = QPushButton("Test")
        find_btn.clicked.connect(self.find_tool)
        browse_btn.clicked.connect(self._browse_exe_or_dir)
        test_btn.clicked.connect(self.test_tool)
        exe_row.addWidget(QLabel("Tool"))
        exe_row.addWidget(self.exe_edit, 1)
        exe_row.addWidget(find_btn)
        exe_row.addWidget(browse_btn)
        exe_row.addWidget(test_btn)
        layout.addLayout(exe_row)

        top = QHBoxLayout()
        self.cwd_edit = QLineEdit(str(Path.cwd()))
        browse = QPushButton("Working folder…")
        browse.clicked.connect(self._browse_cwd)
        top.addWidget(QLabel("CWD"))
        top.addWidget(self.cwd_edit, 1)
        top.addWidget(browse)
        layout.addLayout(top)

        row = QHBoxLayout()
        self.snip_combo = QComboBox()
        self.snip_combo.addItem("Insert command template…", "")
        for name, cmd in self.snippets.items():
            self.snip_combo.addItem(name, cmd)
        self.snip_combo.currentIndexChanged.connect(self._insert_snippet)
        row.addWidget(self.snip_combo, 1)
        self.run_btn = QPushButton("Run")
        self.stop_btn = QPushButton("Stop")
        self.clear_btn = QPushButton("Clear")
        self.run_btn.clicked.connect(self.run_command)
        self.stop_btn.clicked.connect(self.stop_process)
        self.clear_btn.clicked.connect(lambda: self.output.clear())
        row.addWidget(self.run_btn)
        row.addWidget(self.stop_btn)
        row.addWidget(self.clear_btn)
        layout.addLayout(row)

        self.command_edit = QLineEdit()
        self.command_edit.setPlaceholderText("Type a command and press Enter…")
        self.command_edit.returnPressed.connect(self.run_command)
        layout.addWidget(self.command_edit)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.output.setPlaceholderText("Command output appears here.")
        layout.addWidget(self.output, 1)

    def _tool_extra_dirs(self) -> List[str]:
        p = self.exe_edit.text().strip()
        if p:
            _remember_tool_path(p)
            return _existing_dirs([p])
        return []

    def _set_env_on_process(self, proc: Any) -> None:
        try:
            try:
                env = QProcessEnvironment.systemEnvironment()  # type: ignore[union-attr]
            except Exception:
                env = proc.processEnvironment()
            for k, v in _env_for_process(self._tool_extra_dirs()).items():
                env.insert(k, str(v))
            proc.setProcessEnvironment(env)
            try:
                proc.setProcessChannelMode(QProcess.MergedChannels)  # type: ignore[union-attr]
            except Exception:
                pass
        except Exception as exc:
            try:
                self._append_line(f"[could not set process environment] {exc}")
            except Exception:
                pass

    def _browse_exe_or_dir(self) -> None:
        start = self.exe_edit.text().strip() or str(Path.cwd())
        f, _ = QFileDialog.getOpenFileName(self, f"Select {self.tool_name} executable", start, "Executables (*.exe *.bat *.cmd);;All files (*.*);;All files (*)")
        if f:
            self.exe_edit.setText(f)
            _remember_tool_path(f)
            return
        d = QFileDialog.getExistingDirectory(self, f"Select {self.family} bin folder", start)
        if d:
            self.exe_edit.setText(d)
            _remember_tool_path(d)

    def find_tool(self) -> None:
        # For GDAL-family consoles, deliberately prefer QGIS/OSGeo4W first.
        # This keeps GDAL simple for other users: install QGIS/OSGeo4W, restart Mustatil, press Find/Test.
        if str(getattr(self, "family", "")).lower() == "gdal":
            found = _find_gdal_executable("gdalinfo", self._tool_extra_dirs())
        elif str(getattr(self, "family", "")).lower() == "gmt":
            found = _find_executable("gmt", self._tool_extra_dirs())
        else:
            found = _find_executable(self.tool_name, self._tool_extra_dirs())
        if found:
            self.exe_edit.setText(found)
            _remember_tool_path(found)
            self._append_line(f"[found {self.tool_name}] {found}")
            if str(getattr(self, "family", "")).lower() == "gdal":
                self._append_line("[GDAL provider] QGIS/OSGeo4W is preferred when present; otherwise standalone/GMT GDAL is used.")
        else:
            self._append_line(f"[{self.tool_name} not found] Add its bin folder/path or install QGIS/OSGeo4W/GDAL.")

    def _prepare_command(self, cmd: str) -> str:
        """Resolve common GIS/GMT commands to absolute executables before cmd.exe runs them.

        This avoids PATH problems inside the Mustatil venv and fixes the case where GMT
        is found at C:/programs/gmt6/bin/gmt.exe but plain `gmt` is still unknown.
        """
        raw = (cmd or "").strip()
        if not raw:
            return cmd
        extra = self._tool_extra_dirs()
        fam = (self.family or "").lower()
        if fam == "gmt":
            exe = _find_executable("gmt", extra) or self.exe_edit.text().strip()
            if exe and _cmd_token_starts(raw, "gmt"):
                self.exe_edit.setText(exe)
                _remember_tool_path(exe)
                return _replace_first_token(cmd, "gmt", exe)
            return cmd
        if fam == "gdal":
            aliases = [
                "gdalinfo", "ogrinfo", "ogr2ogr", "gdal_translate", "gdalwarp", "gdaldem",
                "gdal_contour", "gdal_rasterize", "gdal_polygonize.py", "gdalbuildvrt",
                "gdaladdo", "gdal2tiles.py", "gdal_retile.py", "rio", "pdal"
            ]
            for alias in aliases:
                if _cmd_token_starts(raw, alias):
                    if alias.lower().startswith("pdal"):
                        exe = _find_pdal_executable("pdal", extra)
                    else:
                        exe = _find_gdal_executable(alias, extra)
                    if exe:
                        _remember_tool_path(exe)
                        return _replace_first_token(cmd, alias, exe)
            return cmd
        exe = _find_executable(self.tool_name, extra)
        if exe and _cmd_token_starts(raw, self.tool_name):
            return _replace_first_token(cmd, self.tool_name, exe)
        return cmd

    def test_tool(self) -> None:
        if self.family.lower() == "gmt":
            self.command_edit.setText("gmt --version && gmt defaults -D")
        elif self.family.lower() == "gdal":
            self.command_edit.setText("gdalinfo --version && ogr2ogr --version && gdalwarp --version && gdal_translate --version")
        else:
            self.command_edit.setText(f"{self.tool_name} --version")
        self.run_command()

    def install_gmt_silent(self) -> None:
        found = _find_executable("gmt", self._tool_extra_dirs())
        if found:
            self.exe_edit.setText(found)
            _remember_tool_path(found)
            self._append_line(f"[GMT already found] {found}")
            self._append_line("[skip install] Running GMT version test instead of reinstalling.")
            self.command_edit.setText(f"{_q(found)} --version")
            self.run_command()
            return
        self._append_line("[no winget] GMT was not found. Opening the official GMT download page.")
        if _is_windows():
            self.command_edit.setText('start "" "https://www.generic-mapping-tools.org/download/"')
            self.run_command()
        else:
            self._append_line("Open: https://www.generic-mapping-tools.org/download/")




class PythonReplWidget(QWidget, _RunnerMixin):  # type: ignore[misc,valid-type]
    def __init__(self, parent: Optional[Any] = None):
        super().__init__(parent)
        self.proc = None
        self._build_ui()
        QTimer.singleShot(250, self.start_repl)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        header = QLabel(
            "<b>Python / GDAL Console</b><br>"
            "Interactive Python running with Mustatil's current interpreter/venv. "
            "Use it for <code>osgeo.gdal</code>, <code>rasterio</code>, <code>geopandas</code>, <code>torch</code>, scripts, and quick checks."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        info = QHBoxLayout()
        self.py_edit = QLineEdit(_python_exe())
        browse = QPushButton("Python…")
        browse.clicked.connect(self._browse_python)
        info.addWidget(QLabel("Python"))
        info.addWidget(self.py_edit, 1)
        info.addWidget(browse)
        layout.addLayout(info)

        tools = QHBoxLayout()
        self.snip_combo = QComboBox()
        snippets = {
            "Check interpreter / venv": "import sys, os; print(sys.executable); print(os.getcwd())",
            "Check GDAL Python bindings": "from osgeo import gdal, ogr, osr; print('GDAL', gdal.VersionInfo('--version')); print('OGR drivers', ogr.GetDriverCount())",
            "Check rasterio/geopandas": "import rasterio, geopandas as gpd; print('rasterio', rasterio.__version__); print('geopandas', gpd.__version__)",
            "Check Torch CUDA": "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')",
            "List installed packages": "import subprocess, sys; subprocess.run([sys.executable, '-m', 'pip', 'list'])",
            "Run shell command from Python": "import subprocess; subprocess.run('gdalinfo --version', shell=True)",
        }
        self.snip_combo.addItem("Insert Python snippet…", "")
        for k, v in snippets.items():
            self.snip_combo.addItem(k, v)
        self.snip_combo.currentIndexChanged.connect(self._insert_snippet)
        self.start_btn = QPushButton("Start / Restart")
        self.stop_btn = QPushButton("Stop")
        self.clear_btn = QPushButton("Clear")
        self.start_btn.clicked.connect(self.restart_repl)
        self.stop_btn.clicked.connect(self.stop_repl)
        self.clear_btn.clicked.connect(lambda: self.output.clear())
        tools.addWidget(self.snip_combo, 1)
        tools.addWidget(self.start_btn)
        tools.addWidget(self.stop_btn)
        tools.addWidget(self.clear_btn)
        layout.addLayout(tools)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(self.output, 1)

        row = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Python command. Enter sends it to the REPL. Use exec(...) for multi-line code.")
        self.input_edit.returnPressed.connect(self.send_line)
        send = QPushButton("Send")
        send.clicked.connect(self.send_line)
        row.addWidget(self.input_edit, 1)
        row.addWidget(send)
        layout.addLayout(row)

    def _browse_python(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "Python executable", str(Path(_python_exe()).parent), _browse_filter_all())
        if f:
            self.py_edit.setText(f)

    def _insert_snippet(self) -> None:
        code = self.snip_combo.currentData()
        if code:
            # QLineEdit/interactive REPLs are poor at compound multi-line blocks.
            # Wrap multi-line snippets in exec(...) so they execute as one command.
            s = str(code)
            if "\n" in s or "\r" in s:
                s = "exec(" + repr(s) + ")"
            self.input_edit.setText(s)
            self.input_edit.setFocus()
            try:
                self.snip_combo.setCurrentIndex(0)
            except Exception:
                pass

    def _ready(self) -> None:
        try:
            data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
            if data:
                self._append(data)
        except Exception:
            pass
        try:
            data = bytes(self.proc.readAllStandardError()).decode("utf-8", errors="replace")
            if data:
                self._append(data)
        except Exception:
            pass

    def start_repl(self) -> None:
        if self.proc is not None:
            return
        py = self.py_edit.text().strip() or _python_exe()
        self._append_line(f"[starting Python REPL: {py}]\n")
        self.proc = QProcess(self)
        self._set_env_on_process(self.proc)
        self.proc.readyReadStandardOutput.connect(self._ready)
        self.proc.readyReadStandardError.connect(self._ready)
        self.proc.finished.connect(lambda code, status=None: self._on_finished(code))
        self.proc.start(py, ["-i", "-u"])

    def _on_finished(self, code: int) -> None:
        self._append_line(f"\n[Python REPL finished exit_code={code}]\n")
        self.proc = None

    def restart_repl(self) -> None:
        self.stop_repl()
        QTimer.singleShot(250, self.start_repl)

    def stop_repl(self) -> None:
        if self.proc is not None:
            try:
                self.proc.terminate()
                QTimer.singleShot(1500, lambda p=self.proc: p.kill() if p is not None and p.state() != QProcess.NotRunning else None)
            except Exception:
                pass

    def send_line(self) -> None:
        code = self.input_edit.text()
        if not code.strip():
            return
        # Multi-line code pasted into the single-line command field is executed
        # as one Python statement. This prevents SyntaxError after `with`, `for`,
        # `try`, etc. in the Rasterio/GIS snippets.
        send_code = code
        if "\n" in send_code or "\r" in send_code:
            send_code = "exec(" + repr(send_code) + ")"
        if self.proc is None:
            self.start_repl()
        self._append_line(f">>> {send_code}")
        try:
            self.proc.write((send_code + "\n").encode("utf-8"))
        except Exception as exc:
            self._append_line(f"[send failed] {exc}")
        self.input_edit.clear()


class GISPythonConsoleWidget(PythonReplWidget):  # type: ignore[misc,valid-type]
    """Rasterio/GIS-focused Python REPL using the same robust process handling."""
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        header = QLabel(
            "<b>Rasterio / GIS Python Console</b><br>"
            "Python console focused on packages that often ship with binary GDAL support in the venv: "
            "<code>rasterio</code>, <code>pyogrio</code>, <code>geopandas</code>, <code>shapely</code>, "
            "<code>pyproj</code>, <code>laspy</code>, <code>pandas</code>. Use the Install/Repair tab if imports fail."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        info = QHBoxLayout()
        self.py_edit = QLineEdit(_python_exe())
        browse = QPushButton("Python…")
        browse.clicked.connect(self._browse_python)
        info.addWidget(QLabel("Python"))
        info.addWidget(self.py_edit, 1)
        info.addWidget(browse)
        layout.addLayout(info)

        tools = QHBoxLayout()
        self.snip_combo = QComboBox()
        snippets = {
            "Check GIS Python stack": "import sys, importlib; print(sys.executable);\nfor p in ['rasterio','pyogrio','geopandas','shapely','pyproj','fiona','laspy','lazrs','pandas','numpy']:\n    try:\n        m=importlib.import_module(p); print(p, getattr(m, '__version__', 'OK'))\n    except Exception as e:\n        print(p, 'MISSING', e)",
            "Rasterio info": "import rasterio; path=r'input.tif';\nwith rasterio.open(path) as ds:\n    print(ds.profile); print('bounds', ds.bounds); print('crs', ds.crs); print('res', ds.res)",
            "Read raster window": "import rasterio; from rasterio.windows import Window;\nwith rasterio.open(r'input.tif') as ds:\n    arr=ds.read(1, window=Window(0,0,512,512)); print(arr.shape, arr.dtype, arr.min(), arr.max())",
            "Write small GeoTIFF copy": "import rasterio;\nwith rasterio.open(r'input.tif') as src:\n    profile=src.profile.copy(); data=src.read();\n    profile.update(compress='deflate', tiled=True, bigtiff='IF_SAFER');\n    with rasterio.open(r'output.tif','w',**profile) as dst: dst.write(data);\nprint('written output.tif')",
            "Vector layers with pyogrio": "import pyogrio; print(pyogrio.list_layers(r'input.gpkg'))",
            "GPKG to GeoJSON with geopandas": "import geopandas as gpd; gdf=gpd.read_file(r'input.gpkg'); print(gdf.head()); gdf.to_file(r'output.geojson', driver='GeoJSON')",
            "LAS/LAZ quick check": "import laspy; las=laspy.read(r'input.laz'); print(las.header); print(len(las.points)); print(las.x.min(), las.x.max(), las.y.min(), las.y.max())",
            "Run rio CLI from Python": "import subprocess, sys; subprocess.run(['rio','--version'], shell=True)",
        }
        self.snip_combo.addItem("Insert GIS Python snippet…", "")
        for k, v in snippets.items():
            self.snip_combo.addItem(k, v)
        self.snip_combo.currentIndexChanged.connect(self._insert_snippet)
        self.start_btn = QPushButton("Start / Restart")
        self.stop_btn = QPushButton("Stop")
        self.clear_btn = QPushButton("Clear")
        self.start_btn.clicked.connect(self.restart_repl)
        self.stop_btn.clicked.connect(self.stop_repl)
        self.clear_btn.clicked.connect(lambda: self.output.clear())
        tools.addWidget(self.snip_combo, 1)
        tools.addWidget(self.start_btn)
        tools.addWidget(self.stop_btn)
        tools.addWidget(self.clear_btn)
        layout.addLayout(tools)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(self.output, 1)

        row = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("GIS Python command. Enter sends it to the REPL. Use exec(...) for multi-line code.")
        self.input_edit.returnPressed.connect(self.send_line)
        send = QPushButton("Send")
        send.clicked.connect(self.send_line)
        row.addWidget(self.input_edit, 1)
        row.addWidget(send)
        layout.addLayout(row)


class PackageInstallWidget(QWidget, _RunnerMixin):  # type: ignore[misc,valid-type]
    """Installer/repair panel for the Mustatil venv and external GIS command tools."""
    def __init__(self, parent: Optional[Any] = None):
        super().__init__(parent)
        self.proc = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        info = QLabel(
            "<b>Install / Repair GIS Tools</b><br>"
            "GDAL exists in two practical forms on Windows: Python-GDAL APIs inside packages such as rasterio/pyogrio, "
            "and command-line tools such as gdalinfo/ogr2ogr from QGIS/OSGeo4W/GDAL builds. "
            "Use the safe venv stack first; use QGIS/GDAL tools if you need gdalinfo/ogr2ogr commands."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        row1 = QHBoxLayout()
        b1 = QPushButton("Install/repair venv GIS stack")
        b1.setToolTip("Installs rasterio/pyogrio/geopandas/shapely/pyproj/laspy/lazrs etc. This is the recommended first step.")
        b1.clicked.connect(self.install_safe_gis_stack)
        b2 = QPushButton("Install osgeo helper")
        b2.setToolTip("Checks osgeo if already present. Stable default is rasterio/pyogrio/GDAL CLI; no direct pip GDAL compile is attempted.")
        b2.clicked.connect(self.try_pip_gdal)
        b3 = QPushButton("Install QGIS/GDAL tools")
        b3.setToolTip("Opens QGIS/OSGeo4W download pages. QGIS/OSGeo environments normally include GDAL/OGR command-line tools.")
        b3.clicked.connect(self.install_qgis_tools)
        b4 = QPushButton("Open GMT installer page")
        b4.clicked.connect(self.install_gmt)
        row1.addWidget(b1); row1.addWidget(b2); row1.addWidget(b3); row1.addWidget(b4)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        b5 = QPushButton("Open GIS download pages")
        b5.clicked.connect(self.search_winget)
        b6 = QPushButton("Check after install")
        b6.clicked.connect(self.check_after_install)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_process)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(lambda: self.output.clear())
        row2.addWidget(b5); row2.addWidget(b6); row2.addStretch(1); row2.addWidget(self.stop_btn); row2.addWidget(self.clear_btn)
        layout.addLayout(row2)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.output.setPlaceholderText("Installer output appears here. Nothing is installed until you click a button.")
        layout.addWidget(self.output, 1)

    def _run(self, cmd: str) -> None:
        if self.proc is not None:
            self._append_line("[process already running]")
            return
        self._append_line("\n> " + cmd)
        self.proc = QProcess(self)
        self._set_env_on_process(self.proc)
        self.proc.readyReadStandardOutput.connect(self._ready)
        self.proc.readyReadStandardError.connect(self._ready)
        self.proc.finished.connect(self._finished)
        try:
            self.proc.errorOccurred.connect(lambda e: self._append_line(f"[QProcess error] {e}"))
        except Exception:
            pass
        extra_dirs = []
        try:
            extra_dirs = self._tool_extra_dirs()  # type: ignore[attr-defined]
        except Exception:
            extra_dirs = []
        _start_qprocess_command(self.proc, cmd, self._append_line, extra_dirs)

    def _ready(self) -> None:
        try:
            out = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
            if out:
                self._append(out)
        except Exception:
            pass
        try:
            err = bytes(self.proc.readAllStandardError()).decode("utf-8", errors="replace")
            if err:
                self._append(err)
        except Exception:
            pass

    def _finished(self, code: int, status: Any = None) -> None:
        try:
            self._ready()
        except Exception:
            pass
        self._append_line(f"\n[finished exit_code={code}]\n")
        self.proc = None

    def stop_process(self) -> None:
        if self.proc is not None:
            try:
                self.proc.terminate()
                QTimer.singleShot(2000, lambda p=self.proc: p.kill() if p is not None and p.state() != QProcess.NotRunning else None)
            except Exception:
                pass

    def install_safe_gis_stack(self) -> None:
        py = _q(_python_exe())
        cmd = (
            f"{py} -m pip install --upgrade pip setuptools wheel && "
            f"{py} -m pip install --upgrade numpy pandas pyproj shapely rasterio pyogrio fiona geopandas laspy lazrs affine click requests tqdm"
        )
        self._append_line("[recommended] Installs a rasterio/pyogrio GIS stack into the current Mustatil venv.")
        self._run(cmd)

    def try_pip_gdal(self) -> None:
        py = _q(_python_exe())
        # On Windows/Python 3.12 direct pip GDAL/gdal-utils often tries to compile C++.
        # Keep this button as a safe diagnostic only; use rasterio/pyogrio/GDAL CLI as default.
        check_code = "import importlib.util as u; print('osgeo importable:', bool(u.find_spec('osgeo'))); print('osgeo_utils importable:', bool(u.find_spec('osgeo_utils'))); print('Stable default: rasterio + pyogrio + external GDAL CLI')"
        cmd = f"{py} -c {_q(check_code)}"
        self._append_line("[osgeo check] No pip GDAL/gdal-utils compile is attempted here. This avoids the MSVC Build Tools failure on Windows.")
        self._run(cmd)

    def install_qgis_tools(self) -> None:
        self._append_line("[no winget] Opening QGIS and OSGeo4W download pages for manual installer use.")
        if _is_windows():
            self._run('start "" "https://qgis.org/download/" & start "" "https://trac.osgeo.org/osgeo4w/"')
        else:
            self._append_line("Open: https://qgis.org/download/ and https://trac.osgeo.org/osgeo4w/")

    def install_gmt(self) -> None:
        self._append_line("[no winget] Opening the official GMT download page for manual installer use.")
        if _is_windows():
            self._run('start "" "https://www.generic-mapping-tools.org/download/"')
        else:
            self._append_line("Open: https://www.generic-mapping-tools.org/download/")

    def search_winget(self) -> None:
        self._append_line("[no winget] Opening GIS download pages instead of searching/installing packages.")
        if _is_windows():
            self._run('start "" "https://qgis.org/download/" & start "" "https://trac.osgeo.org/osgeo4w/" & start "" "https://www.generic-mapping-tools.org/download/"')
        else:
            self._append_line("Open: https://qgis.org/download/ ; https://trac.osgeo.org/osgeo4w/ ; https://www.generic-mapping-tools.org/download/")

    def check_after_install(self) -> None:
        py = _q(_python_exe())
        check_code = (
            "import sys,shutil,importlib\n"
            "print('Python:', sys.executable)\n"
            "for exe in ['gdalinfo','ogr2ogr','gmt','rio','pdal']:\n"
            "    print(exe, shutil.which(exe))\n"
            "for p in ['rasterio','pyogrio','geopandas','shapely','pyproj','fiona','laspy','numpy','osgeo']:\n"
            "    try:\n"
            "        m=importlib.import_module(p); print(p, getattr(m,'__version__','OK'))\n"
            "    except Exception as e:\n"
            "        print(p, 'MISSING', e)\n"
        )
        cmd = f"{py} -c {_q(check_code)}"
        self._run(cmd)



class LidarToTiffWidget(QWidget, _RunnerMixin):  # type: ignore[misc,valid-type]
    def __init__(self, parent: Optional[Any] = None):
        super().__init__(parent)
        self.proc = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        header = QLabel(
            "<b>LiDAR LAS/LAZ → GeoTIFF</b><br>"
            "Uses PDAL when available. Output can be DSM/DTM/mean/count/intensity-style rasters. "
            "For DTM, enable ground-only filtering if the LAS has classification values."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        box = QGroupBox("Conversion")
        form = QFormLayout(box)
        self.in_edit = QLineEdit()
        self.out_edit = QLineEdit()
        self.res_spin = QDoubleSpinBox(); self.res_spin.setRange(0.01, 100000.0); self.res_spin.setDecimals(3); self.res_spin.setValue(1.0)
        self.radius_spin = QDoubleSpinBox(); self.radius_spin.setRange(0.0, 100000.0); self.radius_spin.setDecimals(3); self.radius_spin.setValue(1.5)
        self.metric_combo = QComboBox()
        for k, v in [
            ("DSM / highest point", "max"),
            ("DTM-like / lowest point", "min"),
            ("Mean elevation", "mean"),
            ("IDW elevation", "idw"),
            ("Point count", "count"),
        ]:
            self.metric_combo.addItem(k, v)
        self.ground_check = QCheckBox("Use ground classification only (Classification == 2)")
        self.smrf_check = QCheckBox("Try PDAL SMRF ground filter before rasterizing")
        self.assign_srs_edit = QLineEdit(); self.assign_srs_edit.setPlaceholderText("optional, e.g. EPSG:32637")
        self.nodata_edit = QLineEdit("-9999")

        form.addRow("LAS/LAZ input", self._path_row(self.in_edit, self._browse_lidar, "LAS/LAZ (*.las *.laz);;All files (*)"))
        form.addRow("GeoTIFF output", self._save_row(self.out_edit, self._browse_out_tif, "GeoTIFF (*.tif *.tiff);;All files (*)"))
        form.addRow("Resolution", self.res_spin)
        form.addRow("Radius", self.radius_spin)
        form.addRow("Raster metric", self.metric_combo)
        form.addRow("Assign SRS", self.assign_srs_edit)
        form.addRow("NoData", self.nodata_edit)
        form.addRow("", self.ground_check)
        form.addRow("", self.smrf_check)
        layout.addWidget(box)

        buttons = QHBoxLayout()
        build = QPushButton("Preview command/script")
        run = QPushButton("Convert LiDAR to TIFF")
        stop = QPushButton("Stop")
        clear = QPushButton("Clear")
        build.clicked.connect(self.preview_script)
        run.clicked.connect(self.run_conversion)
        stop.clicked.connect(self.stop_process)
        clear.clicked.connect(lambda: self.output.clear())
        buttons.addWidget(build)
        buttons.addWidget(run)
        buttons.addWidget(stop)
        buttons.addStretch(1)
        buttons.addWidget(clear)
        layout.addLayout(buttons)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(self.output, 1)

    def _path_row(self, edit: Any, cb: Any, filt: str) -> Any:
        w = QWidget(); l = QHBoxLayout(w); l.setContentsMargins(0,0,0,0); l.addWidget(edit,1); b=QPushButton("…"); b.clicked.connect(lambda: cb(edit, filt)); l.addWidget(b); return w
    def _save_row(self, edit: Any, cb: Any, filt: str) -> Any:
        return self._path_row(edit, cb, filt)
    def _browse_lidar(self, edit: Any, filt: str) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "LAS/LAZ input", "", filt)
        if f:
            edit.setText(f)
            if not self.out_edit.text().strip():
                self.out_edit.setText(str(Path(f).with_suffix(".tif")))
    def _browse_out_tif(self, edit: Any, filt: str) -> None:
        f, _ = QFileDialog.getSaveFileName(self, "GeoTIFF output", edit.text() or "output.tif", filt)
        if f:
            edit.setText(f)

    def _make_script(self) -> str:
        inp = self.in_edit.text().strip()
        out = self.out_edit.text().strip()
        res = float(self.res_spin.value())
        radius = float(self.radius_spin.value())
        output_type = self.metric_combo.currentData() or "max"
        ground_only = bool(self.ground_check.isChecked())
        smrf = bool(self.smrf_check.isChecked())
        assign_srs = self.assign_srs_edit.text().strip()
        nodata = self.nodata_edit.text().strip() or "-9999"
        pipe: List[Dict[str, Any]] = [{"type": "readers.las", "filename": inp}]
        if assign_srs:
            pipe[0]["override_srs"] = assign_srs
        if smrf:
            pipe.append({"type": "filters.smrf"})
        if ground_only:
            pipe.append({"type": "filters.range", "limits": "Classification[2:2]"})
        writer: Dict[str, Any] = {
            "type": "writers.gdal",
            "filename": out,
            "resolution": res,
            "radius": radius,
            "dimension": "Z",
            "output_type": output_type,
            "data_type": "float32",
            "gdaldriver": "GTiff",
            "nodata": nodata,
        }
        pipe.append(writer)
        pipeline_json = json.dumps(pipe, indent=2)
        return f'''
import json, sys, subprocess, shutil, tempfile, os
pipeline = {pipeline_json!r}
print("PDAL pipeline:")
print(pipeline)
try:
    import pdal
    p = pdal.Pipeline(pipeline)
    count = p.execute()
    print("PDAL Python pipeline finished. Points processed:", count)
except Exception as py_exc:
    print("PDAL Python failed:", py_exc)
    exe = shutil.which("pdal")
    if not exe:
        raise RuntimeError("PDAL is not available. Install PDAL in this venv/environment or add pdal.exe to PATH.")
    print("Trying pdal executable:", exe)
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as f:
        f.write(pipeline)
        tmp = f.name
    try:
        r = subprocess.run([exe, "pipeline", tmp], text=True, capture_output=True)
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        if r.returncode != 0:
            raise SystemExit(r.returncode)
    finally:
        try: os.remove(tmp)
        except Exception: pass
print("Output:", {out!r})
'''.strip()

    def preview_script(self) -> None:
        self.output.clear()
        self._append_line(self._make_script())

    def run_conversion(self) -> None:
        if self.proc is not None:
            self._append_line("[process already running]")
            return
        if not self.in_edit.text().strip() or not self.out_edit.text().strip():
            self._append_line("Choose input LAS/LAZ and output GeoTIFF first.")
            return
        script = self._make_script()
        self._append_line("[running LiDAR conversion]\n")
        self.proc = QProcess(self)
        self._set_env_on_process(self.proc)
        self.proc.readyReadStandardOutput.connect(self._ready)
        self.proc.readyReadStandardError.connect(self._ready)
        self.proc.finished.connect(self._finished)
        self.proc.start(_python_exe(), ["-u", "-c", script])

    def _ready(self) -> None:
        try:
            out = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
            err = bytes(self.proc.readAllStandardError()).decode("utf-8", errors="replace")
            if out: self._append(out)
            if err: self._append(err)
        except Exception:
            pass

    def _finished(self, code: int, status: Any = None) -> None:
        try:
            self._ready()
        except Exception:
            pass
        self._append_line(f"\n[finished exit_code={code}]\n")
        self.proc = None

    def stop_process(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            QTimer.singleShot(2000, lambda p=self.proc: p.kill() if p is not None and p.state() != QProcess.NotRunning else None)


class ModelOnnxWidget(QWidget, _RunnerMixin):  # type: ignore[misc,valid-type]
    def __init__(self, parent: Optional[Any] = None):
        super().__init__(parent)
        self.proc = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        header = QLabel(
            "<b>Model → ONNX</b><br>"
            "Best support: Ultralytics YOLO .pt. TorchScript and pickled nn.Module models can also be exported with a dummy image. "
            "A plain Faster/Mask R-CNN state_dict needs its architecture/class metadata and cannot be safely exported generically."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        box = QGroupBox("ONNX Export")
        form = QFormLayout(box)
        self.in_edit = QLineEdit()
        self.out_edit = QLineEdit()
        self.mode_combo = QComboBox()
        for name, val in [
            ("Ultralytics YOLO .pt", "yolo"),
            ("TorchScript .pt/.pth", "torchscript"),
            ("Generic pickled torch.nn.Module", "module"),
            ("Try auto-detect", "auto"),
        ]:
            self.mode_combo.addItem(name, val)
        self.imgsz_spin = QSpinBox(); self.imgsz_spin.setRange(32, 8192); self.imgsz_spin.setSingleStep(32); self.imgsz_spin.setValue(640)
        self.opset_spin = QSpinBox(); self.opset_spin.setRange(9, 21); self.opset_spin.setValue(17)
        self.batch_spin = QSpinBox(); self.batch_spin.setRange(1, 16); self.batch_spin.setValue(1)
        self.device_combo = QComboBox(); self.device_combo.addItems(["cpu", "cuda:0"])
        self.dynamic_check = QCheckBox("Dynamic axes")
        self.simplify_check = QCheckBox("Simplify ONNX if supported")
        self.simplify_check.setChecked(True)
        form.addRow("Input model", self._file_row(self.in_edit, self._browse_input, "Models (*.pt *.pth *.torchscript *.onnx);;All files (*)"))
        form.addRow("Output ONNX", self._save_row(self.out_edit, self._browse_output, "ONNX (*.onnx);;All files (*)"))
        form.addRow("Mode", self.mode_combo)
        form.addRow("Image size", self.imgsz_spin)
        form.addRow("Batch", self.batch_spin)
        form.addRow("Opset", self.opset_spin)
        form.addRow("Device", self.device_combo)
        form.addRow("", self.dynamic_check)
        form.addRow("", self.simplify_check)
        layout.addWidget(box)

        buttons = QHBoxLayout()
        preview = QPushButton("Preview export script")
        run = QPushButton("Export ONNX")
        stop = QPushButton("Stop")
        clear = QPushButton("Clear")
        preview.clicked.connect(self.preview_script)
        run.clicked.connect(self.run_export)
        stop.clicked.connect(self.stop_process)
        clear.clicked.connect(lambda: self.output.clear())
        buttons.addWidget(preview)
        buttons.addWidget(run)
        buttons.addWidget(stop)
        buttons.addStretch(1)
        buttons.addWidget(clear)
        layout.addLayout(buttons)

        self.output = QPlainTextEdit(); self.output.setReadOnly(True); self.output.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(self.output, 1)

    def _file_row(self, edit: Any, cb: Any, filt: str) -> Any:
        w=QWidget(); l=QHBoxLayout(w); l.setContentsMargins(0,0,0,0); l.addWidget(edit,1); b=QPushButton("…"); b.clicked.connect(lambda: cb(edit, filt)); l.addWidget(b); return w
    def _save_row(self, edit: Any, cb: Any, filt: str) -> Any:
        return self._file_row(edit, cb, filt)
    def _browse_input(self, edit: Any, filt: str) -> None:
        f,_=QFileDialog.getOpenFileName(self,"Input model","",filt)
        if f:
            edit.setText(f)
            if not self.out_edit.text().strip():
                self.out_edit.setText(str(Path(f).with_suffix(".onnx")))
    def _browse_output(self, edit: Any, filt: str) -> None:
        f,_=QFileDialog.getSaveFileName(self,"Output ONNX", edit.text() or "model.onnx",filt)
        if f: edit.setText(f)

    def _make_script(self) -> str:
        inp = self.in_edit.text().strip()
        out = self.out_edit.text().strip() or str(Path(inp).with_suffix(".onnx"))
        mode = self.mode_combo.currentData() or "auto"
        imgsz = int(self.imgsz_spin.value())
        batch = int(self.batch_spin.value())
        opset = int(self.opset_spin.value())
        device = self.device_combo.currentText().strip() or "cpu"
        dynamic = bool(self.dynamic_check.isChecked())
        simplify = bool(self.simplify_check.isChecked())
        return f'''
import sys, os, traceback
from pathlib import Path
inp = {inp!r}
out = {out!r}
mode = {mode!r}
imgsz = {imgsz!r}
batch = {batch!r}
opset = {opset!r}
device = {device!r}
dynamic = {dynamic!r}
simplify = {simplify!r}
print("Input:", inp)
print("Output:", out)
print("Mode:", mode)
if not inp:
    raise RuntimeError("No input model selected.")
Path(out).parent.mkdir(parents=True, exist_ok=True)

def export_yolo():
    from ultralytics import YOLO
    model = YOLO(inp)
    result = model.export(format="onnx", imgsz=imgsz, opset=opset, simplify=simplify, dynamic=dynamic, device=device)
    print("Ultralytics exported:", result)
    # Ultralytics may choose its own output name; copy/rename if needed.
    import shutil
    if result and str(result) != out and Path(result).exists():
        shutil.copy2(str(result), out)
        print("Copied to:", out)

def export_torchscript_or_module(load_mode):
    import torch
    dev = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    if load_mode == "torchscript":
        model = torch.jit.load(inp, map_location=dev)
    else:
        obj = torch.load(inp, map_location=dev)
        if hasattr(obj, "eval"):
            model = obj
        else:
            raise RuntimeError("Loaded object is not a torch.nn.Module. A state_dict/checkpoint needs architecture-specific loader code.")
    model.eval().to(dev)
    dummy = torch.zeros(batch, 3, imgsz, imgsz, device=dev)
    input_names = ["images"]
    output_names = ["output"]
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {"images": {0: "batch", 2: "height", 3: "width"}, "output": {0: "batch"}}
    torch.onnx.export(
        model, dummy, out,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print("Torch ONNX exported:", out)

try:
    if mode == "yolo":
        export_yolo()
    elif mode == "torchscript":
        export_torchscript_or_module("torchscript")
    elif mode == "module":
        export_torchscript_or_module("module")
    else:
        try:
            export_yolo()
        except Exception as yolo_exc:
            print("YOLO export failed, trying TorchScript:", yolo_exc)
            try:
                export_torchscript_or_module("torchscript")
            except Exception as ts_exc:
                print("TorchScript export failed, trying generic nn.Module:", ts_exc)
                export_torchscript_or_module("module")
    print("Done.")
except Exception:
    traceback.print_exc()
    raise
'''.strip()

    def preview_script(self) -> None:
        self.output.clear(); self._append_line(self._make_script())
    def run_export(self) -> None:
        if self.proc is not None:
            self._append_line("[process already running]"); return
        script = self._make_script()
        self._append_line("[running ONNX export]\n")
        self.proc = QProcess(self); self._set_env_on_process(self.proc)
        self.proc.readyReadStandardOutput.connect(self._ready); self.proc.readyReadStandardError.connect(self._ready); self.proc.finished.connect(self._finished)
        self.proc.start(_python_exe(), ["-u", "-c", script])
    def _ready(self) -> None:
        try:
            out=bytes(self.proc.readAllStandardOutput()).decode("utf-8",errors="replace"); err=bytes(self.proc.readAllStandardError()).decode("utf-8",errors="replace")
            if out: self._append(out)
            if err: self._append(err)
        except Exception: pass
    def _finished(self, code: int, status: Any = None) -> None:
        self._append_line(f"\n[finished exit_code={code}]\n"); self.proc=None
    def stop_process(self) -> None:
        if self.proc is not None:
            self.proc.terminate(); QTimer.singleShot(2000, lambda p=self.proc: p.kill() if p is not None and p.state()!=QProcess.NotRunning else None)


class GpkgConverterWidget(QWidget, _RunnerMixin):  # type: ignore[misc,valid-type]
    def __init__(self, parent: Optional[Any] = None):
        super().__init__(parent)
        self.proc = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        header = QLabel(
            "<b>GeoPackage Converter</b><br>"
            "Converts GPKG/vector files using ogr2ogr. CSV export uses WKT geometry so coordinates are not lost."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        box = QGroupBox("Vector conversion")
        form = QFormLayout(box)
        self.in_edit = QLineEdit(); self.out_edit = QLineEdit(); self.layer_edit = QLineEdit(); self.t_srs_edit = QLineEdit()
        self.format_combo = QComboBox()
        for name, val, ext in [
            ("GeoJSON", "GeoJSON", ".geojson"),
            ("Flat GeoJSONSeq", "GeoJSONSeq", ".geojsonl"),
            ("CSV with WKT geometry", "CSV", ".csv"),
            ("KML", "KML", ".kml"),
            ("ESRI Shapefile", "ESRI Shapefile", ".shp"),
            ("SQLite", "SQLite", ".sqlite"),
            ("GeoPackage", "GPKG", ".gpkg"),
        ]:
            self.format_combo.addItem(name, {"driver": val, "ext": ext})
        self.format_combo.currentIndexChanged.connect(self._suggest_output)
        self.overwrite_check = QCheckBox("Overwrite output if it exists"); self.overwrite_check.setChecked(True)
        form.addRow("Input GPKG/vector", self._file_row(self.in_edit, self._browse_input, "Vector files (*.gpkg *.geojson *.json *.shp *.kml *.sqlite);;All files (*)"))
        form.addRow("Layer name", self.layer_edit)
        form.addRow("Output", self._file_row(self.out_edit, self._browse_output, "All files (*)"))
        form.addRow("Format", self.format_combo)
        form.addRow("Target CRS", self.t_srs_edit)
        form.addRow("", self.overwrite_check)
        layout.addWidget(box)

        buttons=QHBoxLayout(); preview=QPushButton("Preview ogr2ogr command"); run=QPushButton("Convert"); listb=QPushButton("List layers"); stop=QPushButton("Stop"); clear=QPushButton("Clear")
        preview.clicked.connect(self.preview_command); run.clicked.connect(self.run_conversion); listb.clicked.connect(self.list_layers); stop.clicked.connect(self.stop_process); clear.clicked.connect(lambda: self.output.clear())
        buttons.addWidget(preview); buttons.addWidget(run); buttons.addWidget(listb); buttons.addWidget(stop); buttons.addStretch(1); buttons.addWidget(clear)
        layout.addLayout(buttons)
        self.output=QPlainTextEdit(); self.output.setReadOnly(True); self.output.setLineWrapMode(QPlainTextEdit.NoWrap); layout.addWidget(self.output,1)

    def _file_row(self, edit: Any, cb: Any, filt: str) -> Any:
        w=QWidget(); l=QHBoxLayout(w); l.setContentsMargins(0,0,0,0); l.addWidget(edit,1); b=QPushButton("…"); b.clicked.connect(lambda: cb(edit, filt)); l.addWidget(b); return w
    def _browse_input(self, edit: Any, filt: str) -> None:
        f,_=QFileDialog.getOpenFileName(self,"Input vector/GPKG","",filt)
        if f:
            edit.setText(f); self._suggest_output()
    def _browse_output(self, edit: Any, filt: str) -> None:
        f,_=QFileDialog.getSaveFileName(self,"Output", edit.text() or "converted.geojson",filt)
        if f: edit.setText(f)
    def _suggest_output(self) -> None:
        inp=self.in_edit.text().strip()
        if not inp: return
        meta=self.format_combo.currentData() or {"ext":".geojson"}
        self.out_edit.setText(str(Path(inp).with_suffix(str(meta.get("ext", ".geojson")))))
    def _command(self, list_layers: bool=False) -> str:
        inp=self.in_edit.text().strip(); out=self.out_edit.text().strip(); layer=self.layer_edit.text().strip(); t_srs=self.t_srs_edit.text().strip()
        if list_layers:
            return f"ogrinfo -so {_q(inp)}"
        meta=self.format_combo.currentData() or {"driver":"GeoJSON"}; driver=meta.get("driver","GeoJSON")
        parts=["ogr2ogr"]
        if self.overwrite_check.isChecked(): parts.append("-overwrite")
        parts += ["-f", _q(driver), _q(out), _q(inp)]
        if layer: parts.append(_q(layer))
        if t_srs: parts += ["-t_srs", _q(t_srs)]
        if driver == "CSV":
            parts += ["-lco", "GEOMETRY=AS_WKT"]
        return " ".join(parts)
    def preview_command(self) -> None:
        self._append_line(self._command(False))
    def list_layers(self) -> None:
        self._run_shell(self._command(True))
    def run_conversion(self) -> None:
        self._run_shell(self._command(False))
    def _run_shell(self, cmd: str) -> None:
        if self.proc is not None: self._append_line("[process already running]"); return
        self._append_line(f"\n> {cmd}")
        self.proc=QProcess(self); self._set_env_on_process(self.proc); self.proc.readyReadStandardOutput.connect(self._ready); self.proc.readyReadStandardError.connect(self._ready); self.proc.finished.connect(self._finished)
        _start_qprocess_command(self.proc, cmd, self._append_line, [])
    def _ready(self) -> None:
        try:
            out=bytes(self.proc.readAllStandardOutput()).decode("utf-8",errors="replace"); err=bytes(self.proc.readAllStandardError()).decode("utf-8",errors="replace")
            if out: self._append(out)
            if err: self._append(err)
        except Exception: pass
    def _finished(self, code: int, status: Any = None) -> None:
        self._append_line(f"\n[finished exit_code={code}]\n"); self.proc=None
    def stop_process(self) -> None:
        if self.proc is not None:
            self.proc.terminate(); QTimer.singleShot(2000, lambda p=self.proc: p.kill() if p is not None and p.state()!=QProcess.NotRunning else None)




def _set_widget_compact(widget: Any, output_height: int = 76) -> Any:
    """Make existing tool widgets behave like compact cards in the operations dashboard."""
    try:
        widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
    except Exception:
        pass
    try:
        for edit in widget.findChildren(QPlainTextEdit):
            try:
                edit.setMaximumHeight(output_height)
                edit.setMinimumHeight(42)
            except Exception:
                pass
        for edit in widget.findChildren(QTextEdit):
            try:
                edit.setMaximumHeight(output_height)
                edit.setMinimumHeight(42)
            except Exception:
                pass
        for label in widget.findChildren(QLabel):
            try:
                label.setWordWrap(True)
            except Exception:
                pass
    except Exception:
        pass
    return widget


def _card(title: str, child: Any, note: str = "") -> Any:
    box = QGroupBox(title)
    lay = QVBoxLayout(box)
    lay.setContentsMargins(8, 8, 8, 8)
    lay.setSpacing(6)
    if note:
        lbl = QLabel(note)
        lbl.setWordWrap(True)
        lay.addWidget(lbl)
    lay.addWidget(child)
    try:
        box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
    except Exception:
        pass
    return box


class RasterDemToolsWidget(QWidget, _RunnerMixin):  # type: ignore[misc,valid-type]
    """Compact GDAL raster/DEM preparation panel."""
    def __init__(self, parent: Optional[Any] = None):
        super().__init__(parent)
        self.proc = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        help_lbl = QLabel(
            "Raster/DEM operations: info, COG/GeoTIFF, reproject, clip, VRT/merge, overviews, hillshade, slope and contours."
        )
        help_lbl.setWordWrap(True)
        layout.addWidget(help_lbl)
        form = QFormLayout()
        self.op_combo = QComboBox()
        ops = [
            ("Raster info", "info"),
            ("Translate to GeoTIFF", "translate_gtiff"),
            ("Create Cloud Optimized GeoTIFF", "cog"),
            ("Reproject / warp raster", "warp"),
            ("Clip by bbox", "clip_bbox"),
            ("Clip by vector cutline", "clip_cutline"),
            ("Build VRT from rasters", "vrt"),
            ("Build overviews", "overviews"),
            ("DEM hillshade", "hillshade"),
            ("DEM slope", "slope"),
            ("DEM contours to GPKG", "contours"),
            ("Polygonize raster", "polygonize"),
            ("Rasterize vector", "rasterize"),
        ]
        for name, val in ops:
            self.op_combo.addItem(name, val)
        self.in_edit = QLineEdit(); self.in_edit.setPlaceholderText("Input raster/vector or multiple rasters separated by ;")
        self.out_edit = QLineEdit(); self.out_edit.setPlaceholderText("Output file")
        self.cutline_edit = QLineEdit(); self.cutline_edit.setPlaceholderText("Optional cutline/vector/reference")
        self.crs_edit = QLineEdit(); self.crs_edit.setPlaceholderText("EPSG:4326 / EPSG:32637 / optional")
        self.bbox_edit = QLineEdit(); self.bbox_edit.setPlaceholderText("xmin ymin xmax ymax")
        self.res_edit = QLineEdit(); self.res_edit.setPlaceholderText("resolution, e.g. 0.5 or 10")
        self.interval_edit = QLineEdit("1.0")
        self.compress_combo = QComboBox(); self.compress_combo.addItems(["LZW", "DEFLATE", "ZSTD", "JPEG", "NONE"])
        form.addRow("Operation", self.op_combo)
        form.addRow("Input", self._file_row(self.in_edit, self._browse_input))
        form.addRow("Output", self._file_row(self.out_edit, self._browse_output, save=True))
        form.addRow("Cutline/ref", self._file_row(self.cutline_edit, self._browse_cutline))
        form.addRow("Target/assign CRS", self.crs_edit)
        form.addRow("BBox", self.bbox_edit)
        form.addRow("Resolution", self.res_edit)
        form.addRow("Contour interval", self.interval_edit)
        form.addRow("Compression", self.compress_combo)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        preview = QPushButton("Preview")
        run = QPushButton("Run")
        stop = QPushButton("Stop")
        clear = QPushButton("Clear")
        preview.clicked.connect(self.preview_command)
        run.clicked.connect(self.run_command)
        stop.clicked.connect(self.stop_process)
        clear.clicked.connect(lambda: self.output.clear())
        buttons.addWidget(preview); buttons.addWidget(run); buttons.addWidget(stop); buttons.addStretch(1); buttons.addWidget(clear)
        layout.addLayout(buttons)
        self.output = QPlainTextEdit(); self.output.setReadOnly(True); self.output.setLineWrapMode(QPlainTextEdit.NoWrap); self.output.setMaximumHeight(90)
        layout.addWidget(self.output)

    def _file_row(self, edit: Any, cb: Any, save: bool = False) -> Any:
        w = QWidget(); l = QHBoxLayout(w); l.setContentsMargins(0, 0, 0, 0); l.addWidget(edit, 1)
        b = QPushButton("…"); b.clicked.connect(lambda: cb(edit)); l.addWidget(b)
        return w

    def _browse_input(self, edit: Any) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Input files", "", "Geo files (*.tif *.tiff *.vrt *.gpkg *.shp *.geojson *.json *.asc *.img);;All files (*)")
        if files:
            edit.setText(";".join(files))
            if not self.out_edit.text().strip():
                self.out_edit.setText(str(Path(files[0]).with_name(Path(files[0]).stem + "_out.tif")))

    def _browse_output(self, edit: Any) -> None:
        f, _ = QFileDialog.getSaveFileName(self, "Output", edit.text() or "output.tif", "Geo files (*.tif *.tiff *.vrt *.gpkg *.geojson *.shp);;All files (*)")
        if f:
            edit.setText(f)

    def _browse_cutline(self, edit: Any) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "Cutline/reference", "", "Geo files (*.gpkg *.shp *.geojson *.json *.tif *.tiff);;All files (*)")
        if f:
            edit.setText(f)

    def _inputs(self) -> List[str]:
        raw = self.in_edit.text().strip()
        return [p.strip() for p in raw.split(";") if p.strip()]

    def _command(self) -> str:
        op = self.op_combo.currentData() or "info"
        inputs = self._inputs()
        inp = inputs[0] if inputs else "input.tif"
        out = self.out_edit.text().strip() or "output.tif"
        cut = self.cutline_edit.text().strip()
        crs = self.crs_edit.text().strip()
        bbox = self.bbox_edit.text().strip()
        res = self.res_edit.text().strip()
        interval = self.interval_edit.text().strip() or "1.0"
        comp = self.compress_combo.currentText().strip()
        co = [] if comp == "NONE" else ["-co", f"COMPRESS={comp}"]
        tiled = ["-co", "TILED=YES"]
        if op == "info":
            return f"gdalinfo {_q(inp)}"
        if op == "translate_gtiff":
            parts = ["gdal_translate", "-of", "GTiff"] + co + tiled + [_q(inp), _q(out)]
            return " ".join(parts)
        if op == "cog":
            parts = ["gdal_translate", "-of", "COG"] + ([] if comp == "NONE" else ["-co", f"COMPRESS={comp}"]) + [_q(inp), _q(out)]
            return " ".join(parts)
        if op == "warp":
            parts = ["gdalwarp", "-multi", "-overwrite"] + co + tiled
            if crs:
                parts += ["-t_srs", _q(crs)]
            if res:
                parts += ["-tr", res, res]
            parts += [_q(inp), _q(out)]
            return " ".join(parts)
        if op == "clip_bbox":
            parts = ["gdalwarp", "-overwrite"] + co + tiled
            if bbox:
                parts += ["-te"] + bbox.split()
            parts += [_q(inp), _q(out)]
            return " ".join(parts)
        if op == "clip_cutline":
            parts = ["gdalwarp", "-overwrite", "-crop_to_cutline"] + co + tiled
            if cut:
                parts += ["-cutline", _q(cut)]
            parts += [_q(inp), _q(out)]
            return " ".join(parts)
        if op == "vrt":
            parts = ["gdalbuildvrt", _q(out)] + [_q(p) for p in inputs]
            return " ".join(parts)
        if op == "overviews":
            return f"gdaladdo -r average {_q(inp)} 2 4 8 16 32"
        if op == "hillshade":
            return f"gdaldem hillshade {_q(inp)} {_q(out)}"
        if op == "slope":
            return f"gdaldem slope {_q(inp)} {_q(out)}"
        if op == "contours":
            return f"gdal_contour -a elev -f GPKG -i {interval} {_q(inp)} {_q(out)}"
        if op == "polygonize":
            return f"gdal_polygonize.py {_q(inp)} -f GPKG {_q(out)} polygons DN"
        if op == "rasterize":
            parts = ["gdal_rasterize"]
            if res:
                parts += ["-tr", res, res]
            if crs:
                parts += ["-a_srs", _q(crs)]
            parts += ["-burn", "1", _q(inp), _q(out)]
            return " ".join(parts)
        return "gdalinfo --version"

    def preview_command(self) -> None:
        self.output.clear(); self._append_line(self._command())

    def run_command(self) -> None:
        cmd = self._command()
        if self.proc is not None:
            self._append_line("[process already running]"); return
        self._append_line(f"\n> {cmd}")
        self.proc = QProcess(self); self._set_env_on_process(self.proc)
        self.proc.readyReadStandardOutput.connect(self._ready); self.proc.readyReadStandardError.connect(self._ready); self.proc.finished.connect(self._finished)
        _start_qprocess_command(self.proc, cmd, self._append_line, [])

    def _ready(self) -> None:
        try:
            out = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
            err = bytes(self.proc.readAllStandardError()).decode("utf-8", errors="replace")
            if out: self._append(out)
            if err: self._append(err)
        except Exception:
            pass

    def _finished(self, code: int, status: Any = None) -> None:
        self._append_line(f"\n[finished exit_code={code}]\n"); self.proc = None

    def stop_process(self) -> None:
        if self.proc is not None:
            self.proc.terminate(); QTimer.singleShot(2000, lambda p=self.proc: p.kill() if p is not None and p.state() != QProcess.NotRunning else None)


class VectorTileToolsWidget(QWidget, _RunnerMixin):  # type: ignore[misc,valid-type]
    """Compact vector cleanup/tile/training-prep panel."""
    def __init__(self, parent: Optional[Any] = None):
        super().__init__(parent)
        self.proc = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        help_lbl = QLabel("Vector cleanup, CRS assignment/reprojection, tile creation and dataset prep helpers.")
        help_lbl.setWordWrap(True); layout.addWidget(help_lbl)
        form = QFormLayout()
        self.op_combo = QComboBox()
        for name, val in [
            ("Vector info / layers", "vinfo"),
            ("Make valid geometries", "makevalid"),
            ("Reproject vector", "vreproject"),
            ("Assign vector CRS", "vassign"),
            ("Clip vector by bbox", "vclip_bbox"),
            ("Create XYZ web tiles", "gdal2tiles"),
            ("Retile raster for AI", "retile"),
            ("Raster footprint polygons", "footprint"),
            ("Tile index from rasters", "tileindex"),
        ]:
            self.op_combo.addItem(name, val)
        self.in_edit = QLineEdit(); self.in_edit.setPlaceholderText("Input file or raster folder")
        self.out_edit = QLineEdit(); self.out_edit.setPlaceholderText("Output file/folder")
        self.crs_edit = QLineEdit(); self.crs_edit.setPlaceholderText("EPSG:4326 / EPSG:32637 / optional")
        self.bbox_edit = QLineEdit(); self.bbox_edit.setPlaceholderText("xmin ymin xmax ymax")
        self.tile_size = QSpinBox(); self.tile_size.setRange(128, 8192); self.tile_size.setSingleStep(128); self.tile_size.setValue(1024)
        self.overlap = QSpinBox(); self.overlap.setRange(0, 4096); self.overlap.setValue(160)
        form.addRow("Operation", self.op_combo)
        form.addRow("Input", self._path_row(self.in_edit, False))
        form.addRow("Output", self._path_row(self.out_edit, True))
        form.addRow("CRS", self.crs_edit)
        form.addRow("BBox", self.bbox_edit)
        form.addRow("Tile size", self.tile_size)
        form.addRow("Overlap", self.overlap)
        layout.addLayout(form)
        row = QHBoxLayout(); preview = QPushButton("Preview"); run = QPushButton("Run"); stop = QPushButton("Stop"); clear = QPushButton("Clear")
        preview.clicked.connect(self.preview_command); run.clicked.connect(self.run_command); stop.clicked.connect(self.stop_process); clear.clicked.connect(lambda: self.output.clear())
        row.addWidget(preview); row.addWidget(run); row.addWidget(stop); row.addStretch(1); row.addWidget(clear); layout.addLayout(row)
        self.output = QPlainTextEdit(); self.output.setReadOnly(True); self.output.setLineWrapMode(QPlainTextEdit.NoWrap); self.output.setMaximumHeight(90); layout.addWidget(self.output)

    def _path_row(self, edit: Any, save: bool = False) -> Any:
        w = QWidget(); l = QHBoxLayout(w); l.setContentsMargins(0, 0, 0, 0); l.addWidget(edit, 1)
        b1 = QPushButton("File…"); b2 = QPushButton("Folder…")
        b1.clicked.connect(lambda: self._browse_file(edit, save)); b2.clicked.connect(lambda: self._browse_folder(edit))
        l.addWidget(b1); l.addWidget(b2); return w

    def _browse_file(self, edit: Any, save: bool) -> None:
        if save:
            f, _ = QFileDialog.getSaveFileName(self, "Output", edit.text() or "output.gpkg", "Geo files (*.gpkg *.geojson *.shp *.tif *.tiff);;All files (*)")
        else:
            f, _ = QFileDialog.getOpenFileName(self, "Input", "", "Geo files (*.gpkg *.geojson *.shp *.tif *.tiff *.vrt);;All files (*)")
        if f: edit.setText(f)

    def _browse_folder(self, edit: Any) -> None:
        d = QFileDialog.getExistingDirectory(self, "Folder", edit.text() or str(Path.cwd()))
        if d: edit.setText(d)

    def _command(self) -> str:
        op = self.op_combo.currentData() or "vinfo"
        inp = self.in_edit.text().strip() or "input.gpkg"
        out = self.out_edit.text().strip() or "output.gpkg"
        crs = self.crs_edit.text().strip()
        bbox = self.bbox_edit.text().strip()
        tile = int(self.tile_size.value())
        ov = int(self.overlap.value())
        if op == "vinfo":
            return f"ogrinfo -so {_q(inp)}"
        if op == "makevalid":
            return f"ogr2ogr -overwrite -makevalid {_q(out)} {_q(inp)}"
        if op == "vreproject":
            return f"ogr2ogr -overwrite -t_srs {_q(crs or 'EPSG:4326')} {_q(out)} {_q(inp)}"
        if op == "vassign":
            return f"ogr2ogr -overwrite -a_srs {_q(crs or 'EPSG:4326')} {_q(out)} {_q(inp)}"
        if op == "vclip_bbox":
            parts = ["ogr2ogr", "-overwrite"]
            if bbox: parts += ["-spat"] + bbox.split()
            parts += [_q(out), _q(inp)]
            return " ".join(parts)
        if op == "gdal2tiles":
            return f"gdal2tiles.py --xyz --processes=4 {_q(inp)} {_q(out)}"
        if op == "retile":
            return f"gdal_retile.py -targetDir {_q(out)} -ps {tile} {tile} -overlap {ov} -of GTiff {_q(inp)}"
        if op == "footprint":
            return f"gdaltindex {_q(out)} {_q(inp)}"
        if op == "tileindex":
            return f"gdaltindex {_q(out)} {_q(str(Path(inp) / '*.tif'))}"
        return "ogrinfo --version"

    def preview_command(self) -> None:
        self.output.clear(); self._append_line(self._command())

    def run_command(self) -> None:
        cmd = self._command()
        if self.proc is not None:
            self._append_line("[process already running]"); return
        self._append_line(f"\n> {cmd}")
        self.proc = QProcess(self); self._set_env_on_process(self.proc)
        self.proc.readyReadStandardOutput.connect(self._ready); self.proc.readyReadStandardError.connect(self._ready); self.proc.finished.connect(self._finished)
        _start_qprocess_command(self.proc, cmd, self._append_line, [])

    def _ready(self) -> None:
        try:
            out = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
            err = bytes(self.proc.readAllStandardError()).decode("utf-8", errors="replace")
            if out: self._append(out)
            if err: self._append(err)
        except Exception: pass

    def _finished(self, code: int, status: Any = None) -> None:
        self._append_line(f"\n[finished exit_code={code}]\n"); self.proc = None

    def stop_process(self) -> None:
        if self.proc is not None:
            self.proc.terminate(); QTimer.singleShot(2000, lambda p=self.proc: p.kill() if p is not None and p.state() != QProcess.NotRunning else None)



_ENV_CHECK_SCRIPT = 'import sys, os, shutil, importlib, subprocess\nprint("Python:", sys.executable)\nprint("Version:", sys.version.replace("\\n", " "))\nprint("PATH first entries:")\nfor p in os.environ.get("PATH", "").split(os.pathsep)[:20]: print("  ", p)\nprint("\\nCommand tools:")\nfor exe in ["gdalinfo", "ogr2ogr", "gdalwarp", "gdal_translate", "gdaldem", "gdal2tiles.py", "gdal_retile.py", "rio", "pdal", "gmt", "python"]:\n    print(f"  {exe:18s}", shutil.which(exe) or "NOT FOUND")\nprint("\\nPython packages:")\nfor pkg in ["osgeo", "rasterio", "pyogrio", "geopandas", "shapely", "fiona", "pyproj", "laspy", "lazrs", "pdal", "torch", "ultralytics", "onnx", "onnxruntime", "numpy"]:\n    try:\n        mod = importlib.import_module(pkg)\n        ver = getattr(mod, "__version__", "OK")\n        print(f"  {pkg:18s} {ver}")\n    except Exception as exc:\n        print(f"  {pkg:18s} MISSING ({exc})")\nfor label, cmd in [("GDAL", ["gdalinfo", "--version"]), ("OGR", ["ogr2ogr", "--version"]), ("GMT", ["gmt", "--version"]), ("PDAL", ["pdal", "--version"])]:\n    try:\n        r = subprocess.run(cmd, text=True, capture_output=True, timeout=10)\n        print(f"\\n{label} check:", (r.stdout or r.stderr).strip())\n    except Exception as exc:\n        print(f"\\n{label} check failed:", exc)'

class EnvironmentCheckWidget(QWidget, _RunnerMixin):  # type: ignore[misc,valid-type]
    def __init__(self, parent: Optional[Any] = None):
        super().__init__(parent)
        self.proc = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel("Checks whether Python packages and command line tools are visible from Mustatil's current venv/PATH.")
        lbl.setWordWrap(True)
        layout.addWidget(lbl)
        row = QHBoxLayout()
        check = QPushButton("Check environment")
        clear = QPushButton("Clear")
        check.clicked.connect(self.run_check)
        clear.clicked.connect(lambda: self.output.clear())
        row.addWidget(check)
        row.addStretch(1)
        row.addWidget(clear)
        layout.addLayout(row)
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.output.setMaximumHeight(140)
        layout.addWidget(self.output)

    def _script(self) -> str:
        return _ENV_CHECK_SCRIPT

    def run_check(self) -> None:
        if self.proc is not None:
            self._append_line("[process already running]")
            return
        self.output.clear()
        self._append_line("[checking environment]\n")
        self.proc = QProcess(self)
        self._set_env_on_process(self.proc)
        self.proc.readyReadStandardOutput.connect(self._ready)
        self.proc.readyReadStandardError.connect(self._ready)
        self.proc.finished.connect(self._finished)
        self.proc.start(_python_exe(), ["-u", "-c", self._script()])

    def _ready(self) -> None:
        try:
            out = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
            err = bytes(self.proc.readAllStandardError()).decode("utf-8", errors="replace")
            if out:
                self._append(out)
            if err:
                self._append(err)
        except Exception:
            pass

    def _finished(self, code: int, status: Any = None) -> None:
        try:
            self._ready()
        except Exception:
            pass
        self._append_line(f"\n[finished exit_code={code}]\n")
        self.proc = None



class PythonScriptConsoleWidget(QWidget, _RunnerMixin):  # type: ignore[misc,valid-type]
    """Reliable Python console for Mustatil.

    This intentionally does NOT depend on an interactive REPL.  Multi-line code,
    `with` blocks, `for` loops and pasted scripts are written to a temporary
    .py file and executed with QProcess(program, ["-u", script]).  This avoids
    the REPL prompt/SyntaxError issue that happens when compound statements are
    pasted line-by-line.
    """
    def __init__(self, title: str = "Python / Venv Console", help_text: str = "", snippets: Optional[Dict[str, str]] = None, parent: Optional[Any] = None):
        super().__init__(parent)
        self.title = title
        self.help_text = help_text or "Runs Python code with Mustatil's current interpreter/venv. Multi-line code is supported."
        self.snippets = snippets or self._default_snippets()
        self.proc = None
        self._last_script = None
        self._build_ui()

    def _default_snippets(self) -> Dict[str, str]:
        return {
            "Check interpreter / venv": "import sys, os\nprint('Python console OK:', sys.executable)\nprint('CWD:', os.getcwd())",
            "Check GIS imports": "import importlib\nfor p in ['rasterio','pyogrio','geopandas','shapely','pyproj','fiona','laspy','lazrs','numpy','pandas','osgeo']:\n    try:\n        m=importlib.import_module(p)\n        print(p, getattr(m, '__version__', 'OK'))\n    except Exception as e:\n        print(p, 'MISSING', e)",
            "Check GDAL CLI from Python": "import subprocess\nfor cmd in [['gdalinfo','--version'], ['ogr2ogr','--version'], ['gdalwarp','--version'], ['gmt','--version'], ['pdal','--version']]:\n    print('>', ' '.join(cmd))\n    subprocess.run(cmd, shell=False)",
            "Check Torch CUDA": "import torch\nprint(torch.__version__)\nprint('cuda', torch.cuda.is_available())\nprint(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')",
            "List installed packages": "import subprocess, sys\nsubprocess.run([sys.executable, '-m', 'pip', 'list'])",
        }

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        header = QLabel(f"<b>{self.title}</b><br>{self.help_text}")
        header.setWordWrap(True)
        layout.addWidget(header)

        path_row = QHBoxLayout()
        self.py_edit = QLineEdit(_python_exe())
        self.cwd_edit = QLineEdit(str(Path.cwd()))
        py_btn = QPushButton("Python…")
        cwd_btn = QPushButton("Folder…")
        py_btn.clicked.connect(self._browse_python)
        cwd_btn.clicked.connect(self._browse_cwd)
        path_row.addWidget(QLabel("Python"))
        path_row.addWidget(self.py_edit, 2)
        path_row.addWidget(py_btn)
        path_row.addWidget(QLabel("CWD"))
        path_row.addWidget(self.cwd_edit, 1)
        path_row.addWidget(cwd_btn)
        layout.addLayout(path_row)

        tools = QHBoxLayout()
        self.snip_combo = QComboBox()
        self.snip_combo.addItem("Insert Python script/snippet…", "")
        for name, code in self.snippets.items():
            self.snip_combo.addItem(name, code)
        self.snip_combo.currentIndexChanged.connect(self._insert_snippet)
        self.run_btn = QPushButton("Run Python code")
        self.stop_btn = QPushButton("Stop")
        self.clear_btn = QPushButton("Clear")
        self.run_btn.clicked.connect(self.run_code)
        self.stop_btn.clicked.connect(self.stop_process)
        self.clear_btn.clicked.connect(lambda: self.output.clear())
        tools.addWidget(self.snip_combo, 1)
        tools.addWidget(self.run_btn)
        tools.addWidget(self.stop_btn)
        tools.addWidget(self.clear_btn)
        layout.addLayout(tools)

        self.code_edit = QPlainTextEdit()
        self.code_edit.setPlaceholderText("Paste Python here. Multi-line code, for-loops and with-blocks work.")
        self.code_edit.setPlainText("import sys\nprint('Python console OK:', sys.executable)")
        self.code_edit.setMaximumHeight(115)
        self.code_edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(self.code_edit)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.output.setPlaceholderText("Python output appears here.")
        layout.addWidget(self.output, 1)

    def _browse_python(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "Python executable", str(Path(_python_exe()).parent), _browse_filter_all())
        if f:
            self.py_edit.setText(f)

    def _browse_cwd(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Working folder", self.cwd_edit.text() or str(Path.cwd()))
        if d:
            self.cwd_edit.setText(d)

    def _insert_snippet(self) -> None:
        code = self.snip_combo.currentData()
        if code:
            self.code_edit.setPlainText(str(code))
            self.code_edit.setFocus()
            try:
                self.snip_combo.setCurrentIndex(0)
            except Exception:
                pass

    def _ready(self) -> None:
        try:
            data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
            if data:
                self._append(data)
        except Exception:
            pass
        try:
            data = bytes(self.proc.readAllStandardError()).decode("utf-8", errors="replace")
            if data:
                self._append(data)
        except Exception:
            pass

    def _finished(self, code: int, status: Any = None) -> None:
        try:
            self._ready()
        except Exception:
            pass
        self._append_line(f"\n[Python finished exit_code={code}]\n")
        self.proc = None

    def _write_temp_script(self, code: str) -> str:
        tmp_dir = Path(tempfile.gettempdir()) / "mustatil_geospatial_ops"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        script = tmp_dir / "mustatil_python_console_run.py"
        script.write_text("# -*- coding: utf-8 -*-\n" + str(code).replace("\r\n", "\n") + "\n", encoding="utf-8")
        return str(script)

    def run_text(self, code: str) -> None:
        self.code_edit.setPlainText(str(code))
        self.run_code()

    def run_code(self) -> None:
        if self.proc is not None:
            self._append_line("[A Python command is already running. Stop it first.]\n")
            return
        py = self.py_edit.text().strip() or _python_exe()
        code = self.code_edit.toPlainText()
        if not str(code).strip():
            self._append_line("[empty Python code]")
            return
        script = self._write_temp_script(code)
        self._last_script = script
        cwd = self.cwd_edit.text().strip() or str(Path.cwd())
        self._append_line(f"\n[python script console]")
        self._append_line(f"[program] {py}")
        self._append_line(f"[script] {script}")
        self.proc = QProcess(self)
        # Python/Rasterio scripts must not inherit external GMT/QGIS PROJ_LIB/GDAL_DATA.
        # Use isolated Python-GIS environment; GDAL/GMT command consoles still use _env_for_process().
        try:
            try:
                env = QProcessEnvironment.systemEnvironment()  # type: ignore[union-attr]
            except Exception:
                env = self.proc.processEnvironment()
            for k, v in _env_for_python_process().items():
                env.insert(k, str(v))
            self.proc.setProcessEnvironment(env)
            try:
                self.proc.setProcessChannelMode(QProcess.MergedChannels)  # type: ignore[union-attr]
            except Exception:
                pass
        except Exception as exc:
            self._append_line(f"[could not set isolated Python environment] {exc}")
        try:
            self.proc.setWorkingDirectory(cwd)
        except Exception:
            pass
        self.proc.readyReadStandardOutput.connect(self._ready)
        self.proc.readyReadStandardError.connect(self._ready)
        self.proc.finished.connect(self._finished)
        try:
            self.proc.errorOccurred.connect(lambda e: self._append_line(f"[QProcess error] {e}"))
        except Exception:
            pass
        self.proc.start(py, ["-u", script])

    def stop_process(self) -> None:
        if self.proc is not None:
            try:
                self.proc.terminate()
                QTimer.singleShot(2000, lambda p=self.proc: p.kill() if p is not None and p.state() != QProcess.NotRunning else None)
                self._append_line("[stop requested]")
            except Exception:
                pass


class GISPythonScriptConsoleWidget(PythonScriptConsoleWidget):  # type: ignore[misc,valid-type]
    def __init__(self, parent: Optional[Any] = None):
        snippets = {
            "Check GIS Python stack": "import sys, importlib\nprint('Python:', sys.executable)\nfor p in ['rasterio','pyogrio','geopandas','shapely','pyproj','fiona','laspy','lazrs','pandas','numpy','osgeo','osgeo_utils']:\n    try:\n        m=importlib.import_module(p)\n        print(p, getattr(m, '__version__', 'OK'))\n    except Exception as e:\n        print(p, 'MISSING', e)",
            "Rasterio self-test (creates temp GeoTIFF)": "import os, tempfile, numpy as np, rasterio\nfrom rasterio.transform import from_origin\nprint('PROJ_LIB:', os.environ.get('PROJ_LIB'))\nprint('PROJ_DATA:', os.environ.get('PROJ_DATA'))\nprint('GDAL_DATA:', os.environ.get('GDAL_DATA'))\nout = tempfile.gettempdir() + r'\\mustatil_rasterio_selftest.tif'\narr = np.arange(100, dtype='uint16').reshape(10, 10)\nprofile = dict(driver='GTiff', height=10, width=10, count=1, dtype='uint16', transform=from_origin(10, 50, 0.01, 0.01), compress='deflate')\nwith rasterio.open(out, 'w', **profile) as dst:\n    dst.write(arr, 1)\nwith rasterio.open(out) as ds:\n    print('Rasterio OK:', rasterio.__version__)\n    print('created:', out)\n    print('profile:', ds.profile)\n    print('bounds:', ds.bounds)\n    print('crs:', ds.crs)\n    print('res:', ds.res)",
            "Rasterio info (edit path first)": "import os, rasterio\npath = r''  # <- paste your GeoTIFF path here, e.g. r'C:\\data\\input.tif'\nif not path or not os.path.exists(path):\n    raise SystemExit('Please set path to an existing .tif/.tiff file. Current path=' + repr(path))\nwith rasterio.open(path) as ds:\n    print(ds.profile)\n    print('bounds', ds.bounds)\n    print('crs', ds.crs)\n    print('res', ds.res)",
            "Read raster window (edit path first)": "import os, rasterio\nfrom rasterio.windows import Window\npath = r''  # <- paste your GeoTIFF path here\nif not path or not os.path.exists(path):\n    raise SystemExit('Please set path to an existing .tif/.tiff file.')\nwith rasterio.open(path) as ds:\n    arr = ds.read(1, window=Window(0, 0, min(512, ds.width), min(512, ds.height)))\n    print(arr.shape, arr.dtype, arr.min(), arr.max())",
            "Write compressed GeoTIFF copy (edit paths)": "import os, rasterio\ninp = r''   # <- input GeoTIFF\nout = r'output_compressed.tif'\nif not inp or not os.path.exists(inp):\n    raise SystemExit('Please set inp to an existing .tif/.tiff file.')\nwith rasterio.open(inp) as src:\n    profile = src.profile.copy()\n    data = src.read()\n    profile.update(compress='deflate', tiled=True, bigtiff='IF_SAFER')\n    with rasterio.open(out, 'w', **profile) as dst:\n        dst.write(data)\nprint('written', out)",
            "Vector layers with pyogrio (edit path first)": "import os, pyogrio\npath = r''  # <- paste your .gpkg/.shp/.geojson path here\nif not path or not os.path.exists(path):\n    raise SystemExit('Please set path to an existing vector file. Current path=' + repr(path))\nprint(pyogrio.list_layers(path))",
            "GPKG to GeoJSON with geopandas (edit paths)": "import os, geopandas as gpd\ninp = r''   # <- input .gpkg\nout = r'output.geojson'\nif not inp or not os.path.exists(inp):\n    raise SystemExit('Please set inp to an existing .gpkg file.')\ngdf = gpd.read_file(inp)\nprint(gdf.head())\ngdf.to_file(out, driver='GeoJSON')\nprint('written', out)",
            "LAS/LAZ quick check": "import laspy\nlas = laspy.read(r'input.laz')\nprint(las.header)\nprint('points', len(las.points))\nprint('x', las.x.min(), las.x.max())\nprint('y', las.y.min(), las.y.max())",
            "Check GDAL/GMT CLIs from Python": "import subprocess\nfor cmd in [['gdalinfo','--version'], ['ogr2ogr','--version'], ['gdalwarp','--version'], ['gmt','--version'], ['pdal','--version']]:\n    print('>', ' '.join(cmd))\n    subprocess.run(cmd, shell=False)",
        }
        super().__init__(
            "Rasterio / GIS Python Console",
            "Runs GIS Python scripts in Mustatil's venv. This is not a fragile REPL; multi-line code is saved to a temporary script and executed cleanly.",
            snippets,
            parent,
        )


class ConsoleHubWidget(QWidget, _RunnerMixin):  # type: ignore[misc,valid-type]
    """Bottom console deck: one large console at a time plus direct install/repair buttons.

    V5 intentionally removes the old upper installer/output console. Install and check
    commands are launched directly into the visible shell-style console area, so the
    lower console has much more room and the output is always visible in one place.
    """
    def __init__(self, parent: Optional[Any] = None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        top = QHBoxLayout()
        top.setSpacing(6)
        title = QLabel("<b>Consoles</b>")
        title.setToolTip("Choose one large console below. Install/download buttons write their output into the shell console.")
        top.addWidget(title)

        self.console_tabs = QTabWidget()
        self.console_tabs.setDocumentMode(True)

        self.py_console = PythonScriptConsoleWidget()
        self.gis_console = GISPythonScriptConsoleWidget()

        gdal_snips = {
            "Check GDAL version": "gdalinfo --version",
            "Check all GDAL tools": "gdalinfo --version && ogr2ogr --version && gdalwarp --version && gdal_translate --version",
            "Check GDAL + GMT + Rasterio": "gdalinfo --version && ogr2ogr --version && gmt --version && rio --version",
            "Check PDAL version optional": "where pdal >nul 2>nul && pdal --version || echo PDAL optional: not installed",
            "Check GDAL + PDAL optional": "gdalinfo --version && ogr2ogr --version && (where pdal >nul 2>nul && pdal --version || echo PDAL optional: not installed)",
            "Show GDAL/PDAL provider paths": "where gdalinfo && where ogr2ogr && where gdalwarp && (where pdal || echo PDAL optional: not installed)",
            "Check rasterio rio CLI": "rio --version",
            "Open PDAL download page": 'start "" "https://pdal.io/en/stable/download.html"',
            "GDAL raster info (edit path first)": r"gdalinfo \"C:\\path\\input.tif\"",
            "Translate raster to GeoTIFF (edit paths)": r"gdal_translate -of GTiff \"C:\\path\\input.tif\" \"C:\\path\\output.tif\"",
            "Create COG (edit paths)": r"gdal_translate \"C:\\path\\input.tif\" \"C:\\path\\output_cog.tif\" -of COG -co COMPRESS=DEFLATE -co BIGTIFF=IF_SAFER",
            "Warp/reproject raster (edit paths)": r"gdalwarp -t_srs EPSG:4326 \"C:\\path\\input.tif\" \"C:\\path\\output_4326.tif\"",
            "Build hillshade": "gdaldem hillshade dem.tif hillshade.tif",
            "Contour DEM": "gdal_contour -a elev dem.tif contours.gpkg -f GPKG -i 1.0",
            "Vector info": "ogrinfo -so input.gpkg",
            "GPKG to GeoJSON": "ogr2ogr -f GeoJSON output.geojson input.gpkg",
            "GPKG to CSV with WKT": "ogr2ogr -f CSV output.csv input.gpkg -lco GEOMETRY=AS_WKT",
            "PDAL point-cloud info (edit path)": r"pdal info \"C:\\path\\input.laz\"",
            "PDAL pipeline from JSON (edit path)": r"pdal pipeline \"C:\\path\\pipeline.json\"",
            "PDAL translate LAS/LAZ to COPC (edit paths)": r"pdal translate \"C:\\path\\input.laz\" \"C:\\path\\output.copc.laz\"",
            "PDAL LAS/LAZ to raster pipeline template": "echo Create a pipeline.json in the working folder, then run: pdal pipeline pipeline.json",
        }
        self.gdal_console = ToolShellConsoleWidget(
            "GDAL / OGR / PDAL Console",
            "Command console for GDAL/OGR and optional PDAL. Auto-detects GDAL from QGIS/OSGeo4W/GMT/PATH and PDAL from PATH/OSGeo4W/Program Files. PDAL is optional and used for LiDAR/point-cloud workflows.",
            gdal_snips,
            "gdalinfo",
            "gdal",
        )

        gmt_snips = {
            "Check GMT version": "gmt --version",
            "GMT defaults": "gmt defaults -D",
            "GMT full config": "gmt --show-library && gmt --show-bindir",
            "Grid info": "gmt grdinfo input.grd",
            "XYZ to grid": "gmt xyz2grd input.xyz -Goutput.grd -I10 -R0/100/0/100",
            "Surface interpolation": "gmt surface input.xyz -Gsurface.grd -I10 -R0/100/0/100",
            "Make color palette": "gmt makecpt -Cgeo -T0/1000/50 > palette.cpt",
            "Quick coast map": "gmt begin map png && gmt coast -Rg -JH15c -Baf -W0.5p -Glightgray -Swhite && gmt end show",
            "Open GMT download page": "start \"\" \"https://www.generic-mapping-tools.org/download/\"",
            "GMT version absolute path test": "gmt --version",
        }
        self.gmt_console = ToolShellConsoleWidget(
            "GMT Console",
            "Console for Generic Mapping Tools. Use Find GMT, Test, or Install GMT if it is missing.",
            gmt_snips,
            "gmt",
            "gmt",
        )

        self.console_tabs.addTab(self.py_console, "Python / Venv")
        self.console_tabs.addTab(self.gis_console, "Rasterio / GIS")
        self.console_tabs.addTab(self.gdal_console, "GDAL / OGR / PDAL")
        self.console_tabs.addTab(self.gmt_console, "GMT")

        # One-click install/repair control.  V8 intentionally keeps the main UI simple:
        # one large button repairs the venv GIS stack, QGIS/GDAL command tools, GMT,
        # PATH discovery and final checks.  Individual manual commands remain available
        # inside the consoles/snippet dropdowns for advanced use.
        b_install_all = QPushButton("INSTALL / REPAIR + TEST ALL CONSOLES")
        b_install_all.setMinimumHeight(40)
        try:
            b_install_all.setStyleSheet("QPushButton { font-weight: 700; padding: 8px 14px; }")
        except Exception:
            pass
        b_use_qgis = QPushButton("Use/Test QGIS-GDAL")
        b_open_installers = QPushButton("Open GDAL/QGIS installer")
        b_check = QPushButton("Check all consoles")
        b_clear = QPushButton("Clear visible")

        b_install_all.setToolTip(
            "Runs the complete repair sequence: Python GIS packages in the Mustatil venv, "
            "portable Mustatil venv discovery, QGIS/OSGeo4W/GMT discovery, optional installer download pages if missing, and final checks. No winget is used. QGIS/OSGeo4W GDAL is preferred when present and commands are launched with direct program+args rather than broken quoted cmd strings."
        )
        b_use_qgis.setToolTip("Finds GDAL again, preferring QGIS/OSGeo4W. If QGIS/OSGeo4W is installed, this should immediately make gdalinfo/ogr2ogr/gdalwarp work.")
        b_open_installers.setToolTip("Opens official QGIS/OSGeo4W/GDAL download pages so the user can install manually by clicking through. No winget.")
        b_check.setToolTip("Runs a combined environment check and version tests for Python, Rasterio/GIS, GDAL/OGR and GMT.")

        b_install_all.clicked.connect(self.install_repair_all_tools)
        b_use_qgis.clicked.connect(self.use_qgis_gdal_if_installed)
        b_open_installers.clicked.connect(self.open_gdal_installer_pages)
        b_check.clicked.connect(self.run_all_console_checks)
        b_clear.clicked.connect(self.clear_visible_console)

        top.addWidget(b_install_all, 0)
        top.addWidget(b_use_qgis, 0)
        top.addWidget(b_open_installers, 0)
        top.addWidget(b_check, 0)
        top.addStretch(1)
        top.addWidget(b_clear, 0)
        layout.addLayout(top)

        hint = QLabel(
            "Der große Install/Repair-Button schreibt alles in die untere GDAL/OGR-Konsole. "
            "Danach laufen Check-Befehle für Python, Rasterio/GIS, GDAL/OGR und GMT."
        )
        hint.setWordWrap(True)
        hint.setMaximumHeight(38)
        layout.addWidget(hint)
        layout.addWidget(self.console_tabs, 1)

    def _select_console(self, widget: Any) -> None:
        try:
            idx = self.console_tabs.indexOf(widget)
            if idx >= 0:
                self.console_tabs.setCurrentIndex(idx)
        except Exception:
            pass

    def _run_in_shell_console(self, console: Any, cmd: str) -> None:
        self._select_console(console)
        try:
            console.command_edit.setText(cmd)
            console.run_command()
        except Exception as exc:
            try:
                console._append_line(f"[could not run command] {exc}")
            except Exception:
                pass

    def _shell_quote_code(self, code: str) -> str:
        try:
            return _q(code)
        except Exception:
            return repr(code)

    def _quote_for_py_code(self, s: str) -> str:
        try:
            return repr(str(s))
        except Exception:
            return "''"

    def _cmd_exists_or_where(self, exe_name: str) -> str:
        """Return an existing executable path or the bare name for post-install checks."""
        try:
            low = str(exe_name).lower()
            if low.startswith(("gdal", "ogr")):
                found = _find_gdal_executable(exe_name)
            elif low.startswith("pdal"):
                found = _find_pdal_executable("pdal")
            else:
                found = _find_executable(exe_name)
            if found:
                _remember_tool_path(found)
                return found
        except Exception:
            pass
        return exe_name

    def _cmd_path(self, p: str) -> str:
        return str(p).replace('"', '')

    def _write_no_winget_repair_script(self) -> str:
        """Create a temporary .cmd/.sh repair script and return its path.

        This avoids the broken huge cmd.exe one-liner problem.  Paths are placed
        in SET variables, not escaped as \"...\".  External installers are not
        run through winget; if GDAL/GMT command tools are missing, official
        download pages are opened for manual installer use.
        """
        py = _python_exe()
        gdal_found = _find_gdal_executable("gdalinfo") or _find_gdal_executable("ogr2ogr")
        ogr_found = _find_gdal_executable("ogr2ogr")
        gmt_found = _find_executable("gmt")
        pdal_found = _find_pdal_executable("pdal")
        for found in [gdal_found, ogr_found, gmt_found, pdal_found]:
            if found:
                _remember_tool_path(found)
        try:
            if gdal_found:
                self.gdal_console.exe_edit.setText(gdal_found)
            if gmt_found:
                self.gmt_console.exe_edit.setText(gmt_found)
        except Exception:
            pass

        tool_dirs = _common_tool_dirs()
        for found in [gdal_found, ogr_found, gmt_found, pdal_found]:
            try:
                if found:
                    d = str(Path(found).parent)
                    if d not in tool_dirs:
                        tool_dirs.insert(0, d)
            except Exception:
                pass
        path_prefix = os.pathsep.join([d for d in tool_dirs if d])

        py_check = (
            "import sys, os, shutil, importlib\n"
            "print('Python:', sys.executable)\n"
            "print('Venv prefix:', sys.prefix)\n"
            "print('PATH first entries:', os.environ.get('PATH','').split(os.pathsep)[:18])\nprint('GDAL_DATA:', os.environ.get('GDAL_DATA'))\nprint('PROJ_LIB:', os.environ.get('PROJ_LIB'))\nprint('QGIS PYTHONPATH:', os.environ.get('MUSTATIL_QGIS_PYTHONPATH'))\n"
            "for exe in ['gdalinfo','ogr2ogr','gdalwarp','gdal_translate','gdaldem','gdal_contour','rio','gmt','pdal']:\n"
            "    print(exe, '=>', shutil.which(exe))\n"
            "for p in ['rasterio','pyogrio','geopandas','shapely','pyproj','fiona','laspy','lazrs','pandas','numpy','osgeo','torch','ultralytics','onnx','onnxruntime']:\n"
            "    try:\n"
            "        m=importlib.import_module(p); print(p, getattr(m,'__version__','OK'))\n"
            "    except Exception as e:\n"
            "        print(p, 'MISSING', e)\n"
        )

        tmp_dir = Path(tempfile.gettempdir()) / "mustatil_geospatial_ops"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        if _is_windows():
            script = tmp_dir / "mustatil_gis_repair_no_winget.cmd"
            py_check_file = tmp_dir / "mustatil_gis_final_check.py"
            py_check_file.write_text(py_check, encoding="utf-8")
            lines = [
                "@echo off",
                "setlocal EnableExtensions",
                "chcp 65001 >nul",
                "echo ===== MUSTATIL GIS ONE-CLICK INSTALL/REPAIR - QGIS/GDAL FIRST =====",
                f"set \"MUSTATIL_PYTHON={self._cmd_path(py)}\"",
                f"set \"PATH={self._cmd_path(path_prefix)};%PATH%\"",
                "echo Python candidate: %MUSTATIL_PYTHON%",
                "if not exist \"%MUSTATIL_PYTHON%\" (",
                "  echo ERROR: Mustatil Python was not found at: %MUSTATIL_PYTHON%",
                "  echo The plugin will continue with plain python from PATH if available.",
                "  set \"MUSTATIL_PYTHON=python\"",
                ")",
                "echo.",
                "echo [1/6] Upgrade pip/build tools in Mustatil venv...",
                "\"%MUSTATIL_PYTHON%\" -m pip install --upgrade pip setuptools wheel",
                "echo.",
                "echo [2/6] Install/repair safe Python GIS stack...",
                "\"%MUSTATIL_PYTHON%\" -m pip install --upgrade numpy pandas affine click requests tqdm pyproj shapely rasterio pyogrio fiona geopandas laspy lazrs onnx onnxruntime ultralytics",
                "echo.",
                'echo [3/6] Python osgeo bindings / GDAL utilities...',
                'echo Checking osgeo only. This repair does NOT run pip install GDAL/gdal-utils, because on Windows/Python 3.12 it often tries to compile C++ and fails.',
                '"%MUSTATIL_PYTHON%" -c "import importlib.util as u, os; print(\'osgeo importable:\', bool(u.find_spec(\'osgeo\'))); print(\'osgeo_utils importable:\', bool(u.find_spec(\'osgeo_utils\'))); print(\'QGIS PYTHONPATH:\', os.environ.get(\'MUSTATIL_QGIS_PYTHONPATH\'))"',
                'echo Using rasterio/pyogrio/geopandas plus external GDAL CLI as the stable default.',
                "echo.",
                "echo [4/6] Check GDAL/OGR command tools...",
                "where gdalinfo >nul 2>nul",
                "if errorlevel 1 (",
                "  echo GDAL command tools are not found in PATH/common folders.",
                "  echo Opening QGIS and OSGeo4W download pages. Install QGIS or OSGeo4W, then restart Mustatil and press Check all consoles.",
                "  start \"\" \"https://qgis.org/download/\"",
                "  start \"\" \"https://trac.osgeo.org/osgeo4w/\"",
                ") else (",
                "  for /f \"delims=\" %%I in ('where gdalinfo 2^>nul') do (echo GDAL found: %%I & goto :after_gdal_where)",
                ")",
                ":after_gdal_where",
                "gdalinfo --version",
                "where ogr2ogr >nul 2>nul && ogr2ogr --version",
                "echo.",
                "echo [5/6] Check GMT command tools...",
                "where gmt >nul 2>nul",
                "if errorlevel 1 (",
                "  echo GMT is not found in PATH/common folders.",
                "  echo Opening GMT download page. Install GMT, then restart Mustatil and press Check all consoles.",
                "  start \"\" \"https://www.generic-mapping-tools.org/download/\"",
                ") else (",
                "  for /f \"delims=\" %%I in ('where gmt 2^>nul') do (echo GMT found: %%I & goto :after_gmt_where)",
                ")",
                ":after_gmt_where",
                "gmt --version",
                "echo.",
                "echo [5b/6] Check PDAL command tools for LiDAR workflows...",
                "where pdal >nul 2>nul",
                "if errorlevel 1 (",
                "  echo PDAL optional: not found. GDAL/GMT/Rasterio still work. Install PDAL only for advanced LiDAR point-cloud workflows.",
                "  echo PDAL download: https://pdal.io/en/stable/download.html",
                ") else (",
                "  for /f \"delims=\" %%I in ('where pdal 2^>nul') do (echo PDAL found: %%I & goto :after_pdal_where)",
                "  :after_pdal_where",
                "  pdal --version",
                ")",
                "echo.",
                "echo [6/6] Final Python/CLI checks...",
                f"\"%MUSTATIL_PYTHON%\" -u \"{self._cmd_path(str(py_check_file))}\"",
                "echo.",
                "echo ===== FINISHED =====",
                "echo If installer pages opened, install those tools, then restart Mustatil once.",
                "endlocal",
            ]
            script.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
            return str(script)

        script = tmp_dir / "mustatil_gis_repair_no_winget.sh"
        py_check_file = tmp_dir / "mustatil_gis_final_check.py"
        py_check_file.write_text(py_check, encoding="utf-8")
        lines = [
            "#!/usr/bin/env bash",
            "set +e",
            "echo '===== MUSTATIL GIS ONE-CLICK INSTALL/REPAIR - QGIS/GDAL FIRST ====='",
            f"export MUSTATIL_PYTHON={shlex.quote(py)}",
            f"export PATH={shlex.quote(path_prefix)}:$PATH",
            "echo Python candidate: $MUSTATIL_PYTHON",
            'if [ ! -x "$MUSTATIL_PYTHON" ]; then export MUSTATIL_PYTHON=python; fi',
            '"$MUSTATIL_PYTHON" -m pip install --upgrade pip setuptools wheel',
            '"$MUSTATIL_PYTHON" -m pip install --upgrade numpy pandas affine click requests tqdm pyproj shapely rasterio pyogrio fiona geopandas laspy lazrs onnx onnxruntime ultralytics',
            "gdalinfo --version || true",
            "ogr2ogr --version || true",
            "gmt --version || true",
            "pdal --version || true",
            f'"$MUSTATIL_PYTHON" -u {shlex.quote(str(py_check_file))}',
            "echo '===== FINISHED ====='",
        ]
        script.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            script.chmod(0o755)
        except Exception:
            pass
        return str(script)

    def _one_click_install_command(self) -> str:
        script = self._write_no_winget_repair_script()
        if _is_windows():
            return script
        return f"bash {_q(script)}"

    def open_gdal_installer_pages(self) -> None:
        """Open manual installer pages instead of using winget.

        This is intentionally simple for other users: install QGIS or OSGeo4W,
        restart Mustatil, then click Use/Test QGIS-GDAL.
        """
        self._select_console(self.gdal_console)
        try:
            self.gdal_console.output.clear()
        except Exception:
            pass
        self.gdal_console._append_line("[manual installer] Opening QGIS / OSGeo4W / GDAL download pages. Install QGIS or OSGeo4W, restart Mustatil, then press Use/Test QGIS-GDAL.")
        if _is_windows():
            self._run_in_shell_console(
                self.gdal_console,
                'start "" "https://qgis.org/download/" & start "" "https://trac.osgeo.org/osgeo4w/" & start "" "https://gdal.org/download.html"'
            )
        else:
            self.gdal_console._append_line("Open: https://qgis.org/download/")
            self.gdal_console._append_line("Open: https://gdal.org/download.html")

    def use_qgis_gdal_if_installed(self) -> None:
        """Prefer QGIS/OSGeo4W GDAL if present, otherwise open installer pages."""
        self._select_console(self.gdal_console)
        try:
            self.gdal_console.output.clear()
        except Exception:
            pass
        gdalinfo = _find_gdal_executable("gdalinfo")
        ogr2ogr = _find_gdal_executable("ogr2ogr")
        gdalwarp = _find_gdal_executable("gdalwarp")
        gdal_translate = _find_gdal_executable("gdal_translate")
        found_any = bool(gdalinfo or ogr2ogr or gdalwarp or gdal_translate)
        if not found_any:
            self.gdal_console._append_line("[GDAL not found] No QGIS/OSGeo4W/GDAL tools found. Opening installer pages.")
            self.open_gdal_installer_pages()
            return
        if gdalinfo:
            self.gdal_console.exe_edit.setText(gdalinfo)
            _remember_tool_path(gdalinfo)
            self.gdal_console._append_line(f"[using GDAL] {gdalinfo}")
        for exe in [gdalinfo, ogr2ogr, gdalwarp, gdal_translate]:
            if exe:
                _remember_tool_path(exe)
        self.gdal_console.command_edit.setText("gdalinfo --version && ogr2ogr --version && gdalwarp --version && gdal_translate --version")
        self.gdal_console.run_command()

    def install_repair_all_tools(self) -> None:
        """One large user-facing repair button for all consoles and tools."""
        self._select_console(self.gdal_console)
        cmd = self._one_click_install_command()
        try:
            self.gdal_console.output.clear()
        except Exception:
            pass
        try:
            self.gdal_console._append_line(
                "[one-click repair] This installs/repairs the Mustatil venv GIS stack, "
                "Python GIS stack, finds existing GDAL/GMT/optional PDAL, then runs checks. No winget and no pip GDAL compile.\n"
            )
        except Exception:
            pass
        self._run_in_shell_console(self.gdal_console, cmd)

    def run_all_console_checks(self) -> None:
        """Run lightweight checks for every console category.

        This does not install anything; it only verifies that output and command
        resolution work in all console tabs.
        """
        # Python REPL checks are sent directly to their REPLs.  Shell consoles run
        # version commands with absolute path replacement where possible.
        try:
            self.py_console.run_text("import sys\nprint('Python console OK:', sys.executable)")
        except Exception:
            pass
        try:
            self.gis_console.run_text("import importlib\nprint('GIS console OK')\nprint('rasterio', importlib.import_module('rasterio').__version__)")
        except Exception:
            pass
        try:
            # Check core GDAL/OGR, rasterio/rio and optional PDAL without requiring an input.tif placeholder.
            self.gdal_console.command_edit.setText("gdalinfo --version && ogr2ogr --version && gdalwarp --version && gdal_translate --version && rio --version && (where pdal >nul 2>nul && pdal --version || echo PDAL optional: not installed)")
            self.gdal_console.run_command()
        except Exception:
            pass
        try:
            self.gmt_console.command_edit.setText("gmt --version && gmt defaults -D")
            self.gmt_console.run_command()
        except Exception:
            pass

    def run_combined_check(self) -> None:
        py = _q(_python_exe())
        check_code = (
            "import sys, os, shutil, importlib\\n"
            "print('Python:', sys.executable)\\n"
            "print('CWD:', os.getcwd())\\n"
            "print('PATH first entries:', os.environ.get('PATH','').split(os.pathsep)[:8])\\n"
            "for exe in ['gdalinfo','ogr2ogr','gdalwarp','gdal_translate','rio','gmt','pdal']:\\n"
            "    print(exe, '=>', shutil.which(exe))\\n"
            "for p in ['rasterio','pyogrio','geopandas','shapely','pyproj','fiona','laspy','lazrs','pandas','numpy','osgeo','torch','ultralytics','onnx']:\\n"
            "    try:\\n"
            "        m=importlib.import_module(p); print(p, getattr(m,'__version__','OK'))\\n"
            "    except Exception as e:\\n"
            "        print(p, 'MISSING', e)\\n"
        )
        cmd = f"{py} -c {_q(check_code)}"
        self._run_in_shell_console(self.gdal_console, cmd)

    def install_venv_gis_stack(self) -> None:
        py = _q(_python_exe())
        cmd = (
            f"{py} -m pip install --upgrade pip setuptools wheel & "
            f"{py} -m pip install --upgrade numpy pandas pyproj shapely rasterio pyogrio fiona geopandas laspy lazrs affine click requests tqdm"
        )
        self._run_in_shell_console(self.gdal_console, cmd)

    def install_qgis_gdal_tools(self) -> None:
        found = _find_gdal_executable("gdalinfo") or _find_gdal_executable("ogr2ogr")
        if found:
            _remember_tool_path(found)
            try:
                self.gdal_console.exe_edit.setText(found)
            except Exception:
                pass
            self._run_in_shell_console(self.gdal_console, f"{_q(found)} --version")
            try:
                self.gdal_console._append_line("[skip install] GDAL/OGR tools are already available. No installer needed.")
            except Exception:
                pass
            return
        if _is_windows():
            self._run_in_shell_console(self.gdal_console, 'start "" "https://qgis.org/download/" & start "" "https://trac.osgeo.org/osgeo4w/"')
        else:
            self._run_in_shell_console(self.gdal_console, "echo Open QGIS/OSGeo4W download pages manually.")

    def install_gmt_silent(self) -> None:
        found = _find_executable("gmt")
        if found:
            _remember_tool_path(found)
            try:
                self.gmt_console.exe_edit.setText(found)
            except Exception:
                pass
            self._run_in_shell_console(self.gmt_console, f"{_q(found)} --version")
            try:
                self.gmt_console._append_line("[skip install] GMT is already available. No installer needed.")
            except Exception:
                pass
            return
        if _is_windows():
            self._run_in_shell_console(self.gmt_console, 'start "" "https://www.generic-mapping-tools.org/download/"')
        else:
            self._run_in_shell_console(self.gmt_console, "echo Open GMT download page manually.")

    def find_gdal(self) -> None:
        self._select_console(self.gdal_console)
        try:
            self.gdal_console.find_tool()
        except Exception as exc:
            self.gdal_console._append_line(f"[find GDAL failed] {exc}")

    def find_gmt(self) -> None:
        self._select_console(self.gmt_console)
        try:
            self.gmt_console.find_tool()
        except Exception as exc:
            self.gmt_console._append_line(f"[find GMT failed] {exc}")

    def clear_visible_console(self) -> None:
        try:
            w = self.console_tabs.currentWidget()
            out = getattr(w, "output", None)
            if out is not None:
                out.clear()
        except Exception:
            pass


class GeospatialOperationsTab(QWidget):  # type: ignore[misc,valid-type]
    def __init__(self, ws: Optional[Any] = None, parent: Optional[Any] = None):
        super().__init__(parent)
        self.ws = ws
        self._build_ui()

    def _wrap_tool_tab(self, widget: Any, note: str = "") -> Any:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)
        if note:
            lbl = QLabel(note)
            lbl.setWordWrap(True)
            lay.addWidget(lbl)
        lay.addWidget(widget, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        return scroll

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        title = QLabel(
            "<h2>Geospatial Operations</h2>"
            "Oben: nur Werkzeug-Tabs für LiDAR, Modell/ONNX, Vector/GPKG, Raster/DEM, Tiles/AI Prep und Environment. "
            "Unten: eine große umschaltbare Konsolenfläche mit Download-/Install-Buttons direkt daneben."
        )
        title.setWordWrap(True)
        title.setMaximumHeight(72)
        layout.addWidget(title)

        main_split = QSplitter(Qt.Vertical)
        layout.addWidget(main_split, 1)

        tools_tabs = QTabWidget()
        tools_tabs.setDocumentMode(True)
        tools_tabs.addTab(self._wrap_tool_tab(LidarToTiffWidget(), "LiDAR LAS/LAZ zu GeoTIFF/DEM. Bevorzugt PDAL; optional Ground/SMRF-Filter."), "LiDAR → TIFF")
        tools_tabs.addTab(self._wrap_tool_tab(ModelOnnxWidget(), "Konvertiert YOLO/TorchScript/generische Torch-Modelle nach ONNX, wenn die jeweiligen Pakete verfügbar sind."), "Model → ONNX")
        tools_tabs.addTab(self._wrap_tool_tab(GpkgConverterWidget(), "GeoPackage zu GeoJSON, CSV/WKT, KML, Shapefile, SQLite oder GPKG."), "GPKG / Vector")
        tools_tabs.addTab(self._wrap_tool_tab(RasterDemToolsWidget(), "GeoTIFF/COG/VRT/Warp/Clip/Overviews/Hillshade/Slope/Contours/Polygonize/Rasterize."), "Raster / DEM")
        tools_tabs.addTab(self._wrap_tool_tab(VectorTileToolsWidget(), "Vector cleanup, CRS, clipping, gdal2tiles, gdal_retile, Tileindex und AI-Tiling."), "Tiles / AI Prep")
        tools_tabs.addTab(self._wrap_tool_tab(EnvironmentCheckWidget(), "Prüft, ob GDAL, GMT, PDAL und Python-Pakete wirklich aus Mustatil sichtbar sind."), "Environment")
        main_split.addWidget(tools_tabs)

        console_hub = ConsoleHubWidget()
        main_split.addWidget(console_hub)
        try:
            main_split.setSizes([520, 480])
            main_split.setStretchFactor(0, 3)
            main_split.setStretchFactor(1, 3)
        except Exception:
            pass

def _install_geospatial_tab_on_widget(tw: Any) -> bool:
    global _INSTALLING_TAB
    try:
        if tw is None or _has_tab(tw):
            _INSTALLED_TABWIDGETS.add(id(tw))
            return False
        labels = _tab_labels(tw)
        # Only install on the main Mustatil tab widget, not every nested QTabWidget.
        label_blob = " | ".join(labels).lower()
        if not any(k in label_blob for k in ["detection", "trainer", "training", "annotator", "satellite", "formlearner", "pipeline"]):
            return False
        ws = _find_workspace_from_tabwidget(tw)
        page = GeospatialOperationsTab(ws)
        anchor = _find_anchor_index(tw)
        insert_at = tw.count() if anchor is None else int(anchor) + 1
        _INSTALLING_TAB = True
        try:
            tw.insertTab(insert_at, page, TAB_LABEL)
        finally:
            _INSTALLING_TAB = False
        _INSTALLED_TABWIDGETS.add(id(tw))
        try:
            if ws is not None and hasattr(ws, "log"):
                ws.log("Geospatial Operations tab added.")
        except Exception:
            pass
        _log("tab inserted")
        return True
    except Exception as exc:
        _INSTALLING_TAB = False
        _log("install failed: " + str(exc))
        try:
            traceback.print_exc()
        except Exception:
            pass
        return False


def _scan_for_tabwidgets(root: Optional[Any] = None) -> bool:
    try:
        if QApplication is None or QTabWidget is None:
            return False
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
        app = QApplication.instance()
        if app:
            for w in app.allWidgets():
                try:
                    if isinstance(w, QTabWidget):
                        widgets.append(w)
                except Exception:
                    pass
        seen = set(); ok = False
        # Larger tabwidgets first, to prefer main tabs over nested conversion tabs.
        widgets.sort(key=lambda x: getattr(x, "count", lambda: 0)(), reverse=True)
        for tw in widgets:
            if id(tw) in seen:
                continue
            seen.add(id(tw))
            try:
                if _install_geospatial_tab_on_widget(tw):
                    ok = True
            except Exception:
                pass
        return ok
    except Exception as exc:
        _log("scan failed: " + str(exc))
        return False


def _install_hook(root: Optional[Any] = None) -> bool:
    global _PATCHED_QTAB, _ORIG_ADD_TAB, _ORIG_INSERT_TAB, _SCAN_TIMER
    if QTabWidget is None or QTimer is None:
        _log("PySide6 unavailable; hook not installed")
        return False
    if not _PATCHED_QTAB:
        _ORIG_ADD_TAB = QTabWidget.addTab
        _ORIG_INSERT_TAB = QTabWidget.insertTab

        def addTab_patched(self, page, *args, **kwargs):
            res = _ORIG_ADD_TAB(self, page, *args, **kwargs)
            if not _INSTALLING_TAB:
                try:
                    QTimer.singleShot(100, lambda root=self: _scan_for_tabwidgets(root))
                    QTimer.singleShot(800, lambda root=self: _scan_for_tabwidgets(root))
                except Exception:
                    pass
            return res

        def insertTab_patched(self, index, page, *args, **kwargs):
            res = _ORIG_INSERT_TAB(self, index, page, *args, **kwargs)
            if not _INSTALLING_TAB:
                try:
                    QTimer.singleShot(100, lambda root=self: _scan_for_tabwidgets(root))
                    QTimer.singleShot(800, lambda root=self: _scan_for_tabwidgets(root))
                except Exception:
                    pass
            return res

        QTabWidget.addTab = addTab_patched
        QTabWidget.insertTab = insertTab_patched
        _PATCHED_QTAB = True
        _log("QTabWidget hook installed")

    try:
        for ms in (100, 300, 800, 1500, 3000, 6000, 10000, 15000):
            QTimer.singleShot(ms, lambda root=root: _scan_for_tabwidgets(root))
    except Exception:
        pass
    try:
        if _SCAN_TIMER is None:
            _SCAN_TIMER = QTimer()
            _SCAN_TIMER.setInterval(2000)
            _SCAN_TIMER.timeout.connect(lambda: _scan_for_tabwidgets())
            _SCAN_TIMER.start()
    except Exception:
        pass
    return True


def mustatil_plugin_init():
    return _install_hook()


def register_plugin(app=None, main_window=None):
    _install_hook(main_window or app)
    try:
        _scan_for_tabwidgets(main_window or app)
    except Exception:
        pass
    return True


def init_plugin(app=None, main_window=None):
    return register_plugin(app, main_window)


def load_plugin(app=None, main_window=None):
    return register_plugin(app, main_window)


try:
    _install_hook()
except Exception:
    pass
