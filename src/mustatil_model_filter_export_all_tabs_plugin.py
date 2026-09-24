#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mustatil plugin: LAE-DINO Existing Tab Patch V8 AutoConfig StrictPipeline

Safe LAE-DINO patch:
  - creates NO new tabs
  - replaces ONLY existing tabs whose title contains "LAE" and "DINO"
  - supports existing LAE-DINO tab under Detection and Satellite Detection
  - keeps right Detection preview / Satellite map untouched
  - no permanent scan timer
  - no global subprocess redirect
  - no old LAE-DINO/.venv path
  - avoids UI freeze by using Mustatil run_task or a background thread fallback

Important:
  This plugin intentionally uses:
    mustatil_model_runtimes/LAE-DINO-PATCH
  and never:
    mustatil_model_runtimes/LAE-DINO/.venv
"""
from __future__ import annotations

import concurrent.futures as _cf
import json
import os
import shutil
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

WEB_TILE_SIZE = 256

_PATCHED_QTAB = False
_ORIG_ADD = None
_ORIG_INSERT = None
_PATCHED_WIDGETS = set()
_PATCHING_NOW = False

PY310_URL = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
LAE_REPO_ZIP = "https://github.com/jaychempan/LAE-DINO/archive/refs/heads/main.zip"


def _log(msg: str):
    try:
        print("[Mustatil LAE-DINO ExistingTab V8 Strict] " + str(msg))
    except Exception:
        pass


def _plugin_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except Exception:
        return Path.cwd()


def _runtime_root() -> Path:
    return _plugin_dir() / "mustatil_model_runtimes" / "LAE-DINO-PATCH"


def _py_dir() -> Path:
    return _runtime_root() / "python310"


def _py() -> Path:
    return _py_dir() / "python.exe"


def _repo() -> Path:
    return _runtime_root() / "repo"


def _mmdet_dir(repo: Optional[Path] = None) -> Path:
    r = repo or _repo()
    return r / "mmdetection_lae" if (r / "mmdetection_lae").exists() else r


def _demo_path(repo: Optional[Path] = None) -> Path:
    return _mmdet_dir(repo) / "demo" / "image_demo.py"


def _config_path(repo: Optional[Path] = None) -> Path:
    r = repo or _repo()
    # Prefer a config that matches the largest/first checkpoint.
    try:
        w = _weights_path(r)
        if w:
            cfg = _best_config_for_weight(Path(w), r)
            if cfg is not None:
                return cfg
    except Exception:
        pass
    # Fallback to previous default if no checkpoint/config pairing is possible.
    return _mmdet_dir(r) / "configs" / "lae_dino" / "lae_dino_swin-t_pretrain_LAE-1M.py"

def _weights_path(repo: Optional[Path] = None) -> str:
    r = repo or _repo()
    candidates: List[Path] = []
    for base in [r / "weights", _runtime_root() / "weights", _repo() / "weights"]:
        try:
            candidates += list(base.rglob("*.pth"))
        except Exception:
            pass
    try:
        candidates = sorted(list(dict.fromkeys(candidates)), key=lambda p: p.stat().st_size, reverse=True)
    except Exception:
        pass
    return str(candidates[0]) if candidates else ""



def _all_checkpoints(repo: Optional[Path] = None) -> List[Path]:
    r = repo or _repo()
    bases = [r / "weights", _runtime_root() / "weights", r]
    out: List[Path] = []
    for base in bases:
        try:
            out += list(base.rglob("*.pth"))
            out += list(base.rglob("*.pt"))
        except Exception:
            pass
    # Keep unique paths, biggest first for stable default.
    uniq = list(dict.fromkeys(out))
    try:
        uniq.sort(key=lambda p: p.stat().st_size, reverse=True)
    except Exception:
        pass
    return uniq


def _all_configs(repo: Optional[Path] = None) -> List[Path]:
    r = repo or _repo()
    mmd = _mmdet_dir(r)
    out: List[Path] = []
    for base in [mmd / "configs", r / "configs", mmd, r]:
        try:
            out += list(base.rglob("*.py"))
        except Exception:
            pass
    out = [p for p in dict.fromkeys(out) if "checkpoint" not in str(p).lower() and "__pycache__" not in str(p).lower()]
    return out



def _is_base_config(path: Path) -> bool:
    s = str(path).replace("\\", "/").lower()
    return "/configs/_base_/" in s or "\\configs\\_base_\\" in s


def _is_lae_model_config_candidate(path: Path) -> bool:
    """Fast static filter: only top-level LAE-DINO model configs, never _base_ files."""
    s = str(path).replace("\\", "/").lower()
    if _is_base_config(path):
        return False
    if "/configs/lae_dino/" not in s and "\\configs\\lae_dino\\" not in s:
        return False
    if not s.endswith(".py"):
        return False
    bad = ["dataset", "schedule", "runtime", "readme", "__init__"]
    name = path.name.lower()
    if any(b in name for b in bad):
        return False
    return True


def _candidate_model_configs(repo: Optional[Path] = None) -> List[Path]:
    configs = [c for c in _all_configs(repo) if _is_lae_model_config_candidate(c)]
    configs.sort(key=lambda p: str(p).lower())
    return configs


def _runtime_config_has_model_and_test(py: Path, cfg: Path) -> Tuple[bool, str]:
    """Validate with the embedded runtime because MMEngine config inheritance matters."""
    if not py.exists() or not cfg.exists():
        return False, "missing python or config"
    code = (
        "import sys\n"
        "from mmengine.config import Config\n"
        "cfg=Config.fromfile(sys.argv[1])\n"
        "ok_model=hasattr(cfg,'model')\n"
        "ok_test=hasattr(cfg,'test_dataloader')\n"
        "print('model', ok_model, 'test_dataloader', ok_test)\n"
        "sys.exit(0 if (ok_model and ok_test) else 7)\n"
    )
    try:
        p = subprocess.run([str(py), "-c", code, str(cfg)], text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return p.returncode == 0, (p.stdout or "").strip()
    except Exception as exc:
        return False, str(exc)


def _choose_named_config_for_weight(weight: Path, repo: Optional[Path] = None) -> Optional[Path]:
    """Exact dataset-name matching before scoring."""
    w = str(weight).lower()
    candidates = _candidate_model_configs(repo)
    if not candidates:
        return None

    def find(*needles):
        vals = []
        for c in candidates:
            s = c.name.lower()
            full = str(c).lower()
            if all(n in s or n in full for n in needles):
                vals.append(c)
        if vals:
            vals.sort(key=lambda p: (0 if "swin-t" in p.name.lower() or "swint" in p.name.lower() else 1, len(str(p))))
            return vals[0]
        return None

    if "dior" in w:
        return find("dior") or find("finetune")
    if "dota" in w:
        return find("dota")
    if "lae-1m" in w or "lae_1m" in w or "lae1m" in w or "pretrain" in w:
        return find("pretrain") or find("lae")
    return None


def _make_inference_config(base_cfg: Path, temp_dir: Path, variant: str = "fixscale") -> Path:
    """Create a wrapper config that adds the pipeline DetInferencer expects."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    base = str(base_cfg).replace("\\", "/")

    if variant == "resize":
        resize_line = "dict(type='Resize', scale=(800, 800), keep_ratio=True),"
    elif variant == "resize1333":
        resize_line = "dict(type='Resize', scale=(1333, 800), keep_ratio=True),"
    else:
        resize_line = "dict(type='FixScaleResize', scale=(800, 800), keep_ratio=True),"

    out = temp_dir / f"{base_cfg.stem}_mustatil_infer_{variant}.py"
    content = f"""# Auto-generated by Mustatil LAE-DINO V8 strict patch.
# Original config:
# {base}

_base_ = r"{base}"

mustatil_test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    {resize_line}
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape',
            'scale_factor', 'text', 'custom_entities'
        )
    ),
]

test_dataloader = dict(
    dataset=dict(
        pipeline=mustatil_test_pipeline
    )
)

val_dataloader = test_dataloader
"""
    out.write_text(content, encoding="utf-8")
    return out



def _text_of(path: Path, max_chars: int = 250000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return ""


def _tokenize_name(path: Path) -> List[str]:
    import re
    s = str(path).replace("\\", "/").lower()
    raw = re.split(r"[^a-z0-9]+", s)
    toks = [t for t in raw if t]
    # normalize common spelling variants
    extra = []
    for t in toks:
        if t in {"swint", "swin"} or "swin" in t:
            extra += ["swin", "swint", "swin-t"]
        if t in {"fintune", "finetune", "fine"}:
            extra += ["finetune", "fintune", "fine-tune"]
        if t in {"pretrain", "pretrained"}:
            extra += ["pretrain", "pretrained"]
    return toks + extra


def _guess_dataset_from_checkpoint(weight: Path) -> str:
    s = str(weight).lower()
    known = [
        "dior", "dota", "hrsc", "nwpu", "xview", "visdrone", "coco",
        "voc", "rsod", "fair1m", "fair", "ucas", "aid", "lae-1m",
        "lae_1m", "lae1m", "pretrain"
    ]
    for k in known:
        if k in s:
            if k in {"lae_1m", "lae1m"}:
                return "lae-1m"
            return k
    return ""


def _config_has_pipeline(cfg: Path) -> bool:
    t = _text_of(cfg).lower()
    return ("test_dataloader" in t and "pipeline" in t) or "test_pipeline" in t


def _config_class_hint(cfg: Path) -> int:
    t = _text_of(cfg).lower()
    # Strong hints for known class-count configs
    if "num_classes=20" in t.replace(" ", "") or "num_classes = 20" in t or "dior" in t:
        return 20
    if "num_classes=1600" in t.replace(" ", "") or "num_classes = 1600" in t or "lae-1m" in t or "lae_1m" in t:
        return 1600
    return -1


def _score_config_for_weight(cfg: Path, weight: Path) -> int:
    w = str(weight).replace("\\", "/").lower()
    c = str(cfg).replace("\\", "/").lower()
    ct = _text_of(cfg).lower()
    score = 0

    w_tokens = set(_tokenize_name(weight))
    c_tokens = set(_tokenize_name(cfg))
    common = w_tokens & c_tokens
    score += len(common) * 8

    dataset = _guess_dataset_from_checkpoint(weight)
    if dataset:
        aliases = {
            "lae-1m": ["lae-1m", "lae_1m", "lae1m", "pretrain"],
            "pretrain": ["pretrain", "pretrained", "lae-1m", "lae_1m"],
            "dior": ["dior"],
            "dota": ["dota"],
            "hrsc": ["hrsc"],
            "nwpu": ["nwpu"],
            "xview": ["xview"],
        }.get(dataset, [dataset])
        if any(a in c or a in ct for a in aliases):
            score += 120
        else:
            score -= 120

    # prevent the exact mismatch from the current traceback:
    # DIOR checkpoint with LAE-1M/pretrain config.
    if "dior" in w and ("lae-1m" in c or "lae_1m" in c or "pretrain_lae" in c):
        score -= 500
    if ("lae-1m" in w or "lae_1m" in w or "pretrain" in w) and "dior" in c:
        score -= 300

    if "swin" in w and "swin" in c:
        score += 25
    if ("swint" in w or "swin-t" in w or "swin_t" in w) and ("swint" in c or "swin-t" in c or "swin_t" in c or "swin-t" in ct):
        score += 25
    if ("finetune" in w or "fintune" in w or "fine" in w) and ("finetune" in c or "fintune" in c or "fine" in c):
        score += 20

    # The image_demo inferencer needs a test pipeline. Prefer configs that define one.
    if _config_has_pipeline(cfg):
        score += 45
    else:
        score -= 80

    # Prefer LAE-DINO configs, not unrelated mmdet examples.
    if "lae_dino" in c or "lae-dino" in c or "lae dino" in ct:
        score += 60

    hint = _config_class_hint(cfg)
    if "dior" in w and hint == 20:
        score += 100
    if ("pretrain" in w or "lae-1m" in w or "lae_1m" in w) and hint == 1600:
        score += 100

    return score


def _best_config_for_weight(weight: Path, repo: Optional[Path] = None) -> Optional[Path]:
    exact = _choose_named_config_for_weight(weight, repo)
    if exact is not None:
        return exact

    configs = _candidate_model_configs(repo)
    if not configs:
        return None

    scored = [(_score_config_for_weight(c, weight), c) for c in configs]
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best = scored[0]
    if best_score < -50:
        return None
    return best

def _best_pair(repo: Optional[Path] = None, preferred_weight: Optional[Path] = None) -> Tuple[Optional[Path], Optional[Path], str]:
    weights = _all_checkpoints(repo)
    if preferred_weight and preferred_weight.exists():
        weights = [preferred_weight] + [w for w in weights if w != preferred_weight]
    if not weights:
        return None, None, "No .pth/.pt checkpoint found."

    candidates = []
    for w in weights:
        cfg = _best_config_for_weight(w, repo)
        if cfg is None:
            candidates.append((-9999, w, None))
        else:
            candidates.append((_score_config_for_weight(cfg, w), w, cfg))
    candidates.sort(key=lambda x: x[0], reverse=True)
    score, weight, cfg = candidates[0]
    if cfg is None:
        return weight, None, f"Found checkpoint but no matching config: {weight}"
    msg = f"Auto matched score={score}: checkpoint={weight.name} -> config={cfg.name}"
    return weight, cfg, msg


def _auto_match_lae_paths(ws, status_label=None, preferred_weight: Optional[Path] = None) -> Tuple[Path, Path]:
    """Choose matching checkpoint/config automatically and store them on workspace."""
    repo_s = str(getattr(ws, "mustatil_lae_existing_v8_repo", "") or _repo()).strip().strip('"')
    repo = Path(repo_s) if repo_s else _repo()

    if preferred_weight is None:
        cur_w = str(getattr(ws, "mustatil_lae_existing_v8_weights", "") or "").strip().strip('"')
        if cur_w:
            pw = Path(cur_w)
            if pw.exists():
                preferred_weight = pw

    weight, cfg, msg = _best_pair(repo, preferred_weight)
    try:
        ws.log("[LAE-DINO AutoMatch Strict] " + msg)
    except Exception:
        pass
    if status_label is not None:
        try:
            _progress(ws, status_label, 99.0, msg)
        except Exception:
            try:
                status_label.setText(msg)
            except Exception:
                pass

    if not weight or not cfg:
        raise RuntimeError(msg)

    if _is_base_config(Path(cfg)) or not _is_lae_model_config_candidate(Path(cfg)):
        raise RuntimeError(
            "Auto-match refused non-model/base config:\n"
            f"{cfg}\n"
            "Only configs/lae_dino/*.py model configs are allowed."
        )

    setattr(ws, "mustatil_lae_existing_v8_weights", str(weight))
    setattr(ws, "mustatil_lae_existing_v8_config", str(cfg))
    return Path(cfg), Path(weight)

def _download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Mustatil LAE-DINO ExistingTab V8"})
    with urllib.request.urlopen(req, context=ctx, timeout=120) as r:
        data = r.read()
    dest.write_bytes(data)


def _workspace_from_widget(widget: Any) -> Optional[Any]:
    cur = widget
    for _ in range(160):
        if cur is None:
            break
        try:
            if hasattr(cur, "tabs") and (hasattr(cur, "dets") or hasattr(cur, "satellite_detections") or hasattr(cur, "project")):
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


def _is_ancestor(ancestor, child) -> bool:
    cur = child
    for _ in range(160):
        if cur is None:
            return False
        if cur is ancestor:
            return True
        try:
            cur = cur.parentWidget()
        except Exception:
            try:
                cur = cur.parent()
            except Exception:
                return False
    return False


def _infer_tab_kind(widget) -> str:
    try:
        from PySide6.QtWidgets import QApplication, QTabWidget
        app = QApplication.instance()
        if app is None:
            return "unknown"
        for tw in app.allWidgets():
            try:
                if not isinstance(tw, QTabWidget):
                    continue
                for i in range(tw.count()):
                    label = str(tw.tabText(i) or "").strip().lower()
                    if label in {"detection", "satellite detection"}:
                        page = tw.widget(i)
                        if page is widget or _is_ancestor(page, widget):
                            return "satellite" if label == "satellite detection" else "detection"
            except Exception:
                pass
    except Exception:
        pass
    return "unknown"


def _get_var(v, default=""):
    try:
        if hasattr(v, "get"):
            return v.get()
    except Exception:
        pass
    return default


def _status(ws, label, msg: str):
    try:
        label.setText(str(msg))
    except Exception:
        pass
    try:
        ws.log(str(msg))
    except Exception:
        _log(str(msg))



def _progress(ws, label, percent: float, msg: str):
    """Log a visible percentage line to Mustatil's console and the tab status."""
    try:
        pct = max(0.0, min(100.0, float(percent)))
    except Exception:
        pct = 0.0
    line = f"[LAE-DINO {pct:5.1f}%] {msg}"
    try:
        label.setText(line)
    except Exception:
        pass
    try:
        ws.log(line)
    except Exception:
        _log(line)


def _progress_step(ws, label, index: int, total: int, msg: str):
    total = max(1, int(total))
    index = max(0, min(int(index), total))
    _progress(ws, label, 100.0 * index / total, f"{msg} ({index}/{total})")


def _run_async(ws, title, fn, allow_parallel=False):
    """Use Mustatil task runner when available; otherwise use a daemon thread.

    This avoids freezing the GUI if a task button launches a long pip/model command.
    """
    try:
        return ws.run_task(title, fn, allow_parallel=allow_parallel)
    except TypeError:
        try:
            return ws.run_task(title, fn)
        except Exception:
            pass
    except Exception:
        pass

    def runner():
        try:
            fn()
        except Exception as exc:
            try:
                ws.show_error(title, str(exc))
            except Exception:
                _log(title + " failed: " + str(exc))
            traceback.print_exc()

    t = threading.Thread(target=runner, name=title.replace(" ", "_")[:40], daemon=True)
    t.start()
    return t


def _run_cmd(ws, cmd: List[Any], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None, progress_label=None, progress_base: float = 0.0, progress_span: float = 0.0) -> str:
    """Run command with live console output.

    progress_base/progress_span are coarse progress values. Pip itself does not
    expose a reliable percent, so we show a phase percentage and live command lines.
    """
    cmd = [str(x) for x in cmd]
    try:
        ws.log("RUN: " + " ".join(cmd))
    except Exception:
        pass

    if progress_label is not None and progress_span:
        _progress(ws, progress_label, progress_base, "running: " + Path(cmd[0]).name + " " + " ".join(cmd[1:3]))

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
    except Exception:
        # Fallback to old behavior if Popen is unavailable for any reason.
        p = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        out = p.stdout or ""
        try:
            for line in out.splitlines()[-100:]:
                ws.log(line)
        except Exception:
            pass
        if p.returncode != 0:
            raise RuntimeError("Command failed with exit code %s:\n%s\n\n%s" % (p.returncode, " ".join(cmd), "\n".join(out.splitlines()[-120:])))
        if progress_label is not None and progress_span:
            _progress(ws, progress_label, progress_base + progress_span, "finished command")
        return out

    lines = []
    last_tick = time.time()
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\r\n")
        lines.append(line)
        try:
            ws.log(line)
        except Exception:
            pass
        if progress_label is not None and progress_span and time.time() - last_tick > 3.0:
            # Keep the UI/log alive during long downloads/build steps.
            _progress(ws, progress_label, min(progress_base + progress_span * 0.85, 99.0), "still running command...")
            last_tick = time.time()

    rc = proc.wait()
    out = "\n".join(lines)
    if rc != 0:
        raise RuntimeError("Command failed with exit code %s:\n%s\n\n%s" % (rc, " ".join(cmd), "\n".join(lines[-120:])))
    if progress_label is not None and progress_span:
        _progress(ws, progress_label, progress_base + progress_span, "finished command")
    return out

def _ensure_embedded_python(ws, status_label):
    if _py().exists():
        _status(ws, status_label, "Python 3.10 runtime already exists.")
        return _py()

    root = _runtime_root()
    root.mkdir(parents=True, exist_ok=True)
    z = root / "python-3.10.11-embed-amd64.zip"
    _status(ws, status_label, "Downloading Python 3.10.11 embedded...")
    _download(PY310_URL, z)

    if _py_dir().exists():
        shutil.rmtree(_py_dir(), ignore_errors=True)
    _py_dir().mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(z, "r") as zf:
        zf.extractall(_py_dir())

    for pth in _py_dir().glob("python*._pth"):
        lines = pth.read_text(encoding="utf-8", errors="ignore").splitlines()
        out = []
        has_site = False
        has_sp = False
        for line in lines:
            s = line.strip()
            if s in {"#import site", "import site"}:
                out.append("import site")
                has_site = True
            else:
                out.append(line)
            if "Lib\\site-packages" in line or "Lib/site-packages" in line:
                has_sp = True
        if not has_sp:
            out.insert(max(0, len(out) - 1), "Lib\\site-packages")
        if not has_site:
            out.append("import site")
        pth.write_text("\n".join(out) + "\n", encoding="utf-8")

    gp = root / "get-pip.py"
    _status(ws, status_label, "Bootstrapping pip...")
    _download(GET_PIP_URL, gp)
    _run_cmd(ws, [_py(), gp, "--no-warn-script-location"])
    return _py()


def _ensure_repo(ws, status_label):
    if (_repo() / "README.md").exists():
        _status(ws, status_label, "LAE-DINO repo already exists.")
        return _repo()

    _runtime_root().mkdir(parents=True, exist_ok=True)
    z = _runtime_root() / "LAE-DINO-main.zip"
    _status(ws, status_label, "Downloading LAE-DINO repo...")
    _download(LAE_REPO_ZIP, z)

    tmp = _runtime_root() / "_extract"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z, "r") as zf:
        zf.extractall(tmp)

    extracted = tmp / "LAE-DINO-main"
    if _repo().exists():
        shutil.rmtree(_repo(), ignore_errors=True)
    shutil.move(str(extracted), str(_repo()))
    shutil.rmtree(tmp, ignore_errors=True)
    return _repo()


def _site_packages(py: Path) -> Optional[Path]:
    try:
        p = subprocess.run(
            [str(py), "-c", "import site, json; print(json.dumps(site.getsitepackages()))"],
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        vals = json.loads((p.stdout or "[]").strip())
        for v in vals:
            path = Path(v)
            if path.exists():
                return path
    except Exception:
        pass
    return None


def _write_source_pth(ws, py: Path):
    sp = _site_packages(py)
    if sp is None:
        raise RuntimeError("Could not find Python site-packages for embedded runtime.")
    pth = sp / "mustatil_lae_dino_existingtab_v8_source.pth"
    mmd = _mmdet_dir()
    pth.write_text(str(mmd) + "\n" + str(mmd.parent) + "\n", encoding="utf-8")
    try:
        ws.log("Wrote source .pth: " + str(pth))
    except Exception:
        pass


def _has_nvidia() -> bool:
    try:
        p = subprocess.run(["nvidia-smi"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
        return p.returncode == 0
    except Exception:
        return False


def _runtime_env(repo: Optional[Path] = None) -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    old = env.get("PYTHONPATH", "")
    mmd = _mmdet_dir(repo)
    env["PYTHONPATH"] = os.pathsep.join([str(mmd), str(mmd.parent)] + ([old] if old else []))
    return env


def _install_runtime(ws, status_label, mode: str = "auto"):
    total_steps = 13
    step = 0

    def done(msg):
        nonlocal step
        step += 1
        _progress_step(ws, status_label, step, total_steps, msg)

    _progress(ws, status_label, 0.0, "starting runtime build")
    py = _ensure_embedded_python(ws, status_label)
    done("Python 3.10 runtime ready")

    _ensure_repo(ws, status_label)
    done("LAE-DINO repo ready")

    if mode == "auto":
        mode = "cu118" if _has_nvidia() else "cpu"
    _progress_step(ws, status_label, step, total_steps, "selected install mode: " + mode)

    _run_cmd(ws, [py, "-m", "pip", "install", "-U", "pip", "wheel"], progress_label=status_label, progress_base=100*step/total_steps, progress_span=100/total_steps)
    done("pip/wheel updated")

    _run_cmd(ws, [py, "-m", "pip", "install", "setuptools==69.5.1", "numpy<2", "packaging", "pyyaml"], progress_label=status_label, progress_base=100*step/total_steps, progress_span=100/total_steps)
    done("base packages installed")

    if mode == "cu118":
        _run_cmd(ws, [py, "-m", "pip", "install", "torch==2.0.0", "torchvision==0.15.1", "--index-url", "https://download.pytorch.org/whl/cu118"], progress_label=status_label, progress_base=100*step/total_steps, progress_span=100/total_steps)
        mmcv_index = "https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html"
    else:
        _run_cmd(ws, [py, "-m", "pip", "install", "torch==2.0.0", "torchvision==0.15.1", "--index-url", "https://download.pytorch.org/whl/cpu"], progress_label=status_label, progress_base=100*step/total_steps, progress_span=100/total_steps)
        mmcv_index = "https://download.openmmlab.com/mmcv/dist/cpu/torch2.0.0/index.html"
    done("torch/torchvision installed")

    subprocess.run([str(py), "-m", "pip", "uninstall", "-y", "mmcv", "mmcv-lite"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    done("old mmcv/mmcv-lite removed")

    _run_cmd(ws, [py, "-m", "pip", "install", "mmcv==2.0.0", "-f", mmcv_index], progress_label=status_label, progress_base=100*step/total_steps, progress_span=100/total_steps)
    done("full mmcv with compiled ops installed")

    _run_cmd(ws, [py, "-m", "pip", "install", "mmengine==0.10.4"], progress_label=status_label, progress_base=100*step/total_steps, progress_span=100/total_steps)
    done("mmengine installed")

    extras = [
        "transformers==4.42.3",
        "accelerate",
        "safetensors",
        "huggingface_hub",
        "opencv-python",
        "pillow",
        "matplotlib",
        "scipy",
        "shapely",
        "timm",
        "einops",
        "ftfy",
        "regex",
        "yapf",
        "addict",
        "termcolor",
        "emoji",
        "terminaltables",
        "rich",
        "pycocotools",
        "tokenizers",
        "sentencepiece",
        "protobuf",
        "tqdm",
        "scikit-learn",
        "scikit-image",
        "pandas",
        "cityscapesscripts",
        "model-index",
        "modelindex",
        "imagesize",
        "fairscale",
        "iopath",
        "yacs",
        "openai-clip",
        "open_clip_torch",
    ]
    _run_cmd(ws, [py, "-m", "pip", "install"] + extras, progress_label=status_label, progress_base=100*step/total_steps, progress_span=100/total_steps)
    done("extra packages installed")

    _ensure_clip(ws, status_label)
    try:
        ws.log("clip dependency ready")
    except Exception:
        pass

    req = _mmdet_dir() / "requirements" / "multimodal.txt"
    if req.exists():
        try:
            _run_cmd(ws, [py, "-m", "pip", "install", "-r", req], progress_label=status_label, progress_base=100*step/total_steps, progress_span=100/total_steps)
        except Exception as exc:
            try:
                ws.log("requirements/multimodal.txt warning: " + str(exc))
            except Exception:
                pass
    done("optional multimodal requirements handled")

    _write_source_pth(ws, py)
    done("LAE-DINO source path linked")

    try:
        _auto_match_lae_paths(ws, status_label)
    except Exception as exc:
        try:
            ws.log("LAE-DINO auto config/checkpoint match warning: " + str(exc))
        except Exception:
            pass

    weights_dir = _repo() / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    bert = weights_dir / "bert-base-uncased"
    if not (bert / "config.json").exists():
        try:
            _run_cmd(ws, [py, "-m", "huggingface_hub.commands.huggingface_cli", "download", "google-bert/bert-base-uncased", "--local-dir", bert], progress_label=status_label, progress_base=100*step/total_steps, progress_span=100/(2*total_steps))
        except Exception as exc:
            try:
                ws.log("BERT download warning: " + str(exc))
            except Exception:
                pass

    if not _weights_path():
        try:
            _run_cmd(ws, [py, "-m", "huggingface_hub.commands.huggingface_cli", "download", "jaychempan/LAE-DINO", "--local-dir", weights_dir], progress_label=status_label, progress_base=100*step/total_steps, progress_span=100/(2*total_steps))
        except Exception as exc:
            try:
                ws.log("LAE-DINO checkpoint auto-download warning: " + str(exc))
            except Exception:
                pass
    done("BERT/checkpoint download handled")

    try:
        _auto_match_lae_paths(ws, status_label)
    except Exception as exc:
        try:
            ws.log("LAE-DINO auto config/checkpoint match warning after download: " + str(exc))
        except Exception:
            pass

    meta = {
        "python": str(py),
        "repo": str(_repo()),
        "mmdet_dir": str(_mmdet_dir()),
        "demo": str(_demo_path()),
        "config": str(_config_path()),
        "weights": _weights_path(),
        "mode": mode,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (_runtime_root() / "mustatil_lae_dino_existingtab_v8_runtime.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    done("runtime metadata written")

    msg = _check_runtime(ws, status_label)
    _progress(ws, status_label, 100.0, "runtime build complete")
    _status(ws, status_label, "LAE-DINO runtime ready.\n" + msg)
    return meta


def _ensure_clip(ws, status_label=None):
    """Install/verify common LAE-DINO inference dependencies.

    Based on the official LAE-DINO install notes plus the actual runtime errors:
      - requirements/multimodal.txt
      - emoji
      - ddd-dataset
      - lvis API from GitHub
      - clip / open_clip backbones
      - common text/vision helper libs used by MMDetection-style OVD models

    Hard-required for current traceback:
      import clip
      import open_clip

    Optional packages are installed best-effort and warnings are logged rather
    than killing the runtime build.
    """
    py = _py()
    if not py.exists():
        return False

    def import_ok(module: str) -> bool:
        try:
            p = subprocess.run(
                [str(py), "-c", f"import {module}; print('{module} OK')"],
                text=True, encoding="utf-8", errors="replace",
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            return p.returncode == 0
        except Exception:
            return False

    def pip_install(args, label: str, base: float, span: float, required: bool = False):
        try:
            if status_label is not None:
                _progress(ws, status_label, base, "installing " + label)
            _run_cmd(ws, [py, "-m", "pip", "install"] + list(args), progress_label=status_label, progress_base=base, progress_span=span)
            return True
        except Exception as exc:
            try:
                ws.log(label + " install warning: " + str(exc))
            except Exception:
                pass
            if required:
                raise
            return False

    # Keep these names explicit, because they correspond to imports in LAE-DINO source.
    package_groups = [
        # Open vocabulary / CLIP backbones.
        (["openai-clip"], "openai-clip / module clip", "clip", True),
        (["open_clip_torch"], "open_clip_torch / module open_clip", "open_clip", True),

        # Text/tokenizer/model helpers often imported by GroundingDINO/MM-GroundingDINO-style configs.
        (["transformers==4.42.3", "tokenizers", "sentencepiece", "protobuf", "safetensors", "accelerate", "huggingface_hub"], "HF/text stack", "transformers", False),

        # Vision/model helper libs.
        (["timm", "einops", "ftfy", "regex", "tqdm", "scikit-learn", "scikit-image", "pandas", "opencv-python"], "vision/helper stack", "timm", False),

        # MMDetection datasets/eval utilities.
        (["pycocotools", "terminaltables", "cityscapesscripts", "model-index", "modelindex", "imagesize"], "mmdet eval/utils stack", "terminaltables", False),

        # Official LAE-DINO note includes emoji and ddd-dataset.
        (["emoji", "ddd-dataset"], "LAE official extras", "emoji", False),

        # Sometimes used by CLIP/large model utility code.
        (["fairscale", "iopath", "yacs"], "optional large-model helpers", "fairscale", False),
    ]

    total = len(package_groups) + 2
    idx = 0
    for pkgs, label, module, required in package_groups:
        idx += 1
        if import_ok(module):
            if status_label is not None:
                _progress_step(ws, status_label, idx, total, label + " already OK")
            continue
        pip_install(pkgs, label, 80.0 + min(18.0, idx * 2.0), 1.5, required=required)
        if required and not import_ok(module):
            raise RuntimeError(f"Required module '{module}' is still missing after installing {label}.")

    # Official LAE-DINO install notes mention lvis API via GitHub. Needs git, so best effort.
    idx += 1
    if not import_ok("lvis"):
        pip_install(["git+https://github.com/lvis-dataset/lvis-api.git"], "LVIS API", 96.0, 1.0, required=False)
    if status_label is not None:
        _progress_step(ws, status_label, idx, total, "LVIS API handled")

    # If PyPI openai-clip did not provide module clip for any reason, try official repo as fallback.
    idx += 1
    if not import_ok("clip"):
        pip_install(["git+https://github.com/openai/CLIP.git"], "official OpenAI CLIP git fallback", 97.0, 1.0, required=False)
    if not import_ok("clip"):
        raise RuntimeError("Python module 'clip' is still missing. Try: python.exe -m pip install openai-clip")
    if not import_ok("open_clip"):
        raise RuntimeError("Python module 'open_clip' is still missing. Try: python.exe -m pip install open_clip_torch")

    if status_label is not None:
        _progress(ws, status_label, 98.0, "LAE-DINO extra dependencies import OK")
    try:
        ws.log("LAE-DINO dependencies ready: clip OK, open_clip OK, extras handled")
    except Exception:
        pass
    return True


def _check_runtime(ws, status_label=None) -> str:
    if not _py().exists():
        msg = "Python runtime missing. Press 'Auto build runtime' first."
        if status_label is not None:
            _status(ws, status_label, msg)
        return msg

    checks = [
        ("python", "import sys; print(sys.version)"),
        ("torch", "import torch; print(torch.__version__, torch.version.cuda)"),
        ("mmengine", "import mmengine; print(mmengine.__version__)"),
        ("mmcv", "import mmcv; print(mmcv.__version__)"),
        ("mmcv._ext", "import mmcv._ext; print('OK')"),
        ("roi_align", "from mmcv.ops.roi_align import roi_align; print('OK')"),
        ("clip", "import clip; print('OK')"),
        ("open_clip", "import open_clip; print('OK')"),
        ("timm", "import timm; print('OK')"),
        ("einops", "import einops; print('OK')"),
        ("sentencepiece", "import sentencepiece; print('OK')"),
        ("pycocotools", "import pycocotools; print('OK')"),
        ("lvis optional", "import importlib.util; print('OK' if importlib.util.find_spec('lvis') else 'MISSING_OPTIONAL')"),
        ("mmdet", "import mmdet; print(getattr(mmdet, '__version__', 'source'))"),
        ("DetInferencer", "from mmdet.apis import DetInferencer; print('OK')"),
    ]
    env = _runtime_env()
    lines = ["LAE-DINO runtime check:"]
    total = len(checks)
    for idx, (name, code) in enumerate(checks, start=1):
        if status_label is not None:
            _progress_step(ws, status_label, idx - 1, total, "checking " + name)
        try:
            p = subprocess.run([str(_py()), "-c", code], env=env, text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            lines.append(f"{name}: {'OK' if p.returncode == 0 else 'FAIL'} {(p.stdout or '').strip().replace(chr(10), ' | ')}")
        except Exception as exc:
            lines.append(f"{name}: FAIL {exc}")
        if status_label is not None:
            _progress_step(ws, status_label, idx, total, "checked " + name)
    msg = "\n".join(lines)
    try:
        ws.log(msg)
    except Exception:
        pass
    if status_label is not None:
        _status(ws, status_label, msg)
    return msg

def _classes_from_ws(ws) -> List[str]:
    text = str(getattr(ws, "mustatil_lae_existing_v8_classes_text", "") or "")
    vals = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    return vals or ["mustatil", "burial mound", "tumulus", "stone enclosure", "rectangular structure"]


def _device_from_ws(ws) -> str:
    raw = str(getattr(ws, "mustatil_lae_existing_v8_device_text", "auto") or "auto").strip().lower()
    if raw == "cuda":
        return "cuda:0"
    if raw == "auto":
        return "cuda:0" if _has_nvidia() else "cpu"
    return raw if raw in {"cpu", "cuda:0", "cuda:1", "0"} else "cpu"


def _runtime_paths(ws) -> Tuple[Path, Path, Path, Path, Path]:
    py = Path(str(getattr(ws, "mustatil_lae_existing_v8_python", "") or _py()).strip().strip('"'))
    repo = Path(str(getattr(ws, "mustatil_lae_existing_v8_repo", "") or _repo()).strip().strip('"'))
    mmd = _mmdet_dir(repo)
    demo = _demo_path(repo)

    if "LAE-DINO\\.venv" in str(py) or "LAE-DINO/.venv" in str(py):
        raise RuntimeError("Old LAE-DINO .venv path detected. This V8 strict patch must use LAE-DINO-PATCH/python310.")

    if not py.exists():
        raise RuntimeError("LAE-DINO Python missing. Press 'Auto build runtime' first.")
    if not demo.exists():
        raise RuntimeError("LAE-DINO image_demo.py missing: " + str(demo))

    preferred = None
    cur_w = str(getattr(ws, "mustatil_lae_existing_v8_weights", "") or "").strip().strip('"')
    if cur_w:
        p = Path(cur_w)
        if p.exists():
            preferred = p

    cfg_auto, weights_auto = _auto_match_lae_paths(ws, None, preferred)
    cfg = Path(cfg_auto)
    weights = Path(weights_auto)

    if not cfg.exists():
        raise RuntimeError("LAE-DINO config missing: " + str(cfg))
    if not weights.exists():
        raise RuntimeError("LAE-DINO .pth weights missing. Press auto build again or select weights manually.")

    if _is_base_config(cfg) or not _is_lae_model_config_candidate(cfg):
        raise RuntimeError(
            "Invalid LAE-DINO config selected. Refusing to run helper/base config without model.\n"
            f"Config: {cfg}\n"
            "Expected something like configs\\lae_dino\\lae_dino_swin-t_finetune_DIOR.py"
        )

    ok, detail = _runtime_config_has_model_and_test(py, cfg)
    if not ok:
        raise RuntimeError(
            "Selected LAE-DINO config is not a valid model/test config.\n"
            f"Config: {cfg}\nRuntime check: {detail}"
        )

    wl = str(weights).lower()
    cl = str(cfg).lower()
    if "dior" in wl and ("lae-1m" in cl or "lae_1m" in cl or "pretrain_lae" in cl):
        better = _best_config_for_weight(weights, repo)
        raise RuntimeError(
            "Config/checkpoint mismatch blocked: DIOR checkpoint cannot use LAE-1M/pretrain config.\n"
            f"Checkpoint: {weights}\nConfig: {cfg}\n"
            + (f"Suggested config: {better}" if better else "No DIOR matching config was found in repo/configs.")
        )

    try:
        ws.log(f"[LAE-DINO V8 strict] Using config: {cfg}")
        ws.log(f"[LAE-DINO V8 strict] Using checkpoint: {weights}")
    except Exception:
        pass
    return py, mmd, demo, cfg, weights

def _find_lae_json(out_dir: Path, image_path: Path) -> Optional[Path]:
    candidates: List[Path] = []
    for sub in ("preds", "predictions", "vis", ""):
        d = out_dir / sub
        if d.exists():
            candidates += list(d.glob("*.json"))
    candidates += list(out_dir.rglob("*.json"))
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        return None
    stem = image_path.stem.lower()
    candidates.sort(key=lambda p: (0 if stem in p.stem.lower() else 1, -p.stat().st_mtime))
    return candidates[0]


def _parse_lae_json(json_path: Path, labels: List[str]) -> List[Dict[str, Any]]:
    data = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
    pred = data
    if isinstance(data, dict) and isinstance(data.get("predictions"), list) and data["predictions"]:
        pred = data["predictions"][0]
    elif isinstance(data, list) and data:
        pred = data[0]
    if not isinstance(pred, dict):
        return []

    bboxes = pred.get("bboxes") or pred.get("boxes") or []
    scores = pred.get("scores") or []
    labs = pred.get("labels") or pred.get("label_names") or []
    out = []
    for i, b in enumerate(bboxes):
        try:
            x1, y1, x2, y2 = map(float, list(b)[:4])
        except Exception:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        score = float(scores[i]) if i < len(scores) else 0.0
        raw = labs[i] if i < len(labs) else 0
        class_id = 0
        label = str(raw)
        try:
            class_id = int(raw)
            if 0 <= class_id < len(labels):
                label = labels[class_id]
        except Exception:
            low = str(raw).lower()
            for j, name in enumerate(labels):
                if low == name.lower() or low in name.lower() or name.lower() in low:
                    class_id = j
                    label = name
                    break
        out.append({
            "class_id": int(class_id),
            "class_name": label,
            "label": label,
            "confidence": score,
            "score": score,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "model": "LAE-DINO",
        })
    return out



def _parse_lae_stdout(stdout: str, labels: List[str]) -> List[Dict[str, Any]]:
    """Best-effort parser for --print-result output.

    Normal path is prediction JSON in out_dir. This fallback handles printed dicts
    from MMDetection if JSON saving changes between repo versions.
    """
    import ast
    if not stdout:
        return []
    candidates = []
    for line in stdout.splitlines():
        s = line.strip()
        if not s:
            continue
        if ("pred_instances" in s or "bboxes" in s or "scores" in s) and (s.startswith("{") or s.startswith("[") or "{" in s):
            try:
                s2 = s[s.find("{"):] if "{" in s else s
                candidates.append(ast.literal_eval(s2))
            except Exception:
                pass
    for obj in reversed(candidates):
        try:
            # Common MMDet print-result shape.
            pred = obj
            if isinstance(pred, dict) and "pred_instances" in pred:
                pred = pred["pred_instances"]
            if isinstance(pred, dict) and isinstance(pred.get("pred_instances"), dict):
                pred = pred["pred_instances"]

            bboxes = pred.get("bboxes") or pred.get("boxes") or []
            scores = pred.get("scores") or []
            labs = pred.get("labels") or pred.get("label_names") or []
            out = []
            for i, b in enumerate(bboxes):
                try:
                    x1, y1, x2, y2 = map(float, list(b)[:4])
                except Exception:
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue
                score = float(scores[i]) if i < len(scores) else 0.0
                raw = labs[i] if i < len(labs) else 0
                class_id = 0
                label = str(raw)
                try:
                    class_id = int(raw)
                    if 0 <= class_id < len(labels):
                        label = labels[class_id]
                except Exception:
                    low = str(raw).lower()
                    for j, name in enumerate(labels):
                        if low == name.lower() or low in name.lower() or name.lower() in low:
                            class_id = j
                            label = name
                            break
                out.append({
                    "class_id": int(class_id),
                    "class_name": label,
                    "label": label,
                    "confidence": score,
                    "score": score,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "model": "LAE-DINO",
                })
            if out:
                return out
        except Exception:
            pass
    return []


def _run_lae_on_pil(ws, pil_image, conf: float) -> List[Dict[str, Any]]:
    labels = _classes_from_ws(ws)
    py, mmd, demo, cfg, weights = _runtime_paths(ws)

    try:
        p_clip = subprocess.run([str(py), "-c", "import clip, open_clip; print('clip/open_clip OK')"], env=_runtime_env(Path(getattr(ws, "mustatil_lae_existing_v8_repo", "") or _repo())), text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if p_clip.returncode != 0:
            raise RuntimeError("clip/open_clip missing")
    except Exception:
        raise RuntimeError("Python modules 'clip' and/or 'open_clip' are missing in LAE-DINO-PATCH runtime. Press 'Auto build runtime' once with V8, or run: python.exe -m pip install openai-clip open_clip_torch")

    device = _device_from_ws(ws)
    prompt = " . ".join(labels)
    if prompt and not prompt.endswith("."):
        prompt += " ."

    with tempfile.TemporaryDirectory(prefix="mustatil_lae_existing_v8_strict_") as td:
        td = Path(td)
        img_path = td / "chunk.png"
        out_dir_base = td / "out"
        cfg_dir = td / "configs"
        pil_image.convert("RGB").save(img_path)

        cfg_variants = [
            ("fixscale", _make_inference_config(cfg, cfg_dir, "fixscale")),
            ("resize", _make_inference_config(cfg, cfg_dir, "resize")),
            ("resize1333", _make_inference_config(cfg, cfg_dir, "resize1333")),
        ]

        last_tail = ""
        for variant_name, run_cfg in cfg_variants:
            out_dir = out_dir_base / variant_name
            cmd = [
                str(py), str(demo),
                "--inputs", str(img_path),
                "--model", str(run_cfg),
                "--weights", str(weights),
                "--texts", prompt,
                "--custom-entities",
                "-c",
                "--pred-score-thr", str(float(conf)),
                "--device", device,
                "--out-dir", str(out_dir),
                "--palette", "random",
                "--print-result",
            ]
            env = _runtime_env(Path(getattr(ws, "mustatil_lae_existing_v8_repo", "") or _repo()))
            try:
                ws.log(f"[LAE-DINO V8 strict] Trying inference config variant: {variant_name}")
            except Exception:
                pass
            p = subprocess.run(cmd, cwd=str(mmd), env=env, text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            stdout = p.stdout or ""

            if p.returncode == 0:
                jp = _find_lae_json(out_dir, img_path)
                if jp is None:
                    recovered = _parse_lae_stdout(stdout, labels)
                    if recovered:
                        return recovered
                    tail = "\n".join(stdout.splitlines()[-120:])
                    last_tail = f"Variant {variant_name}: LAE-DINO did not write prediction JSON.\n{tail}"
                    continue
                return _parse_lae_json(jp, labels)

            tail = "\n".join(stdout.splitlines()[-120:])
            last_tail = f"Variant {variant_name} failed with exit code {p.returncode}\n{tail}"

            low = tail.lower()
            retry_markers = [
                "pipeline", "fixscaleresize", "resize is not", "not in the registry",
                "configdict", "transform", "packdetinputs", "loadimagefromfile",
            ]
            if not any(m in low for m in retry_markers):
                raise RuntimeError("LAE-DINO failed with exit code %s\n%s" % (p.returncode, tail))

        raise RuntimeError(
            "LAE-DINO failed after strict auto config + all inference-pipeline variants.\n"
            f"Base config: {cfg}\nCheckpoint: {weights}\n\n{last_tail}"
        )

def _refresh_ui_after_detections(ws):
    for call in (
        lambda: ws.loadprev(),
        lambda: ws.redraw(fit=False),
        lambda: ws.redraw(),
        lambda: ws.satellite_redraw_detection_overlay(),
        lambda: ws.refresh_layers(),
        lambda: ws._satellite_request_map_reload_from_worker(120),
    ):
        try:
            call()
        except Exception:
            pass
    try:
        sig = getattr(getattr(ws, "signals", None), "sat_overlay_redraw_requested", None)
        if sig is not None:
            sig.emit(0)
    except Exception:
        pass


def _run_detection_image(ws):
    try:
        from PIL import Image
        try:
            from mustatil_legacy_backend import Det
        except Exception:
            Det = None

        img_path = str(_get_var(getattr(ws, "image", None), "") or "").strip().strip('"')
        if not img_path or not Path(img_path).exists():
            raise RuntimeError("No Detection image selected.")

        conf = max(0.001, min(1.0, float(_get_var(getattr(ws, "conf", None), 0.15))))
        tile = max(128, int(_get_var(getattr(ws, "tile", None), 768) or 768))
        overlap = max(0, min(tile - 1, int(_get_var(getattr(ws, "overlap", None), 128) or 128)))
        step = max(1, tile - overlap)

        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        ws.log(f"LAE-DINO Detection started: {Path(img_path).name} {W}x{H}, tile={tile}, overlap={overlap}, conf={conf}")

        dets = []
        count = 0
        for y in range(0, H, step):
            for x in range(0, W, step):
                x2 = min(W, x + tile)
                y2 = min(H, y + tile)
                local = _run_lae_on_pil(ws, img.crop((x, y, x2, y2)), conf)
                for r in local:
                    if Det is not None:
                        dets.append(Det(0, "LAE-DINO", int(r.get("class_id", 0)), float(r.get("confidence", 0.0)),
                                        x + float(r["x1"]), y + float(r["y1"]), x + float(r["x2"]), y + float(r["y2"])))
                    else:
                        dets.append({"model": "LAE-DINO", "cls": int(r.get("class_id", 0)),
                                     "class_id": int(r.get("class_id", 0)), "confidence": float(r.get("confidence", 0.0)),
                                     "score": float(r.get("confidence", 0.0)),
                                     "x1": x + float(r["x1"]), "y1": y + float(r["y1"]),
                                     "x2": x + float(r["x2"]), "y2": y + float(r["y2"])})
                count += 1
                try:
                    total_tiles = max(1, ((W + step - 1) // step) * ((H + step - 1) // step))
                    pct = min(100.0, 100.0 * count / total_tiles)
                    if count == 1 or count % 5 == 0 or pct >= 100:
                        ws.log(f"[LAE-DINO {pct:5.1f}%] Detection tiles={count}/{total_tiles}, detections={len(dets)}")
                except Exception:
                    if count % 10 == 0:
                        ws.log(f"LAE-DINO: tiles={count}, detections={len(dets)}")
                if x2 >= W:
                    break
            if y + tile >= H:
                break

        try:
            dets = ws.nms(dets, 0.45)
        except Exception:
            pass

        ws.dets = list(dets)
        _refresh_ui_after_detections(ws)
        ws.log(f"LAE-DINO Detection finished: {len(dets)} detections")
    except Exception as exc:
        try:
            ws.show_error("LAE-DINO", str(exc))
        except Exception:
            _log("LAE-DINO Detection failed: " + str(exc))
        traceback.print_exc()


def _run_satellite(ws):
    try:
        from PIL import Image

        g = ws.__class__.satellite_detect_selected.__globals__
        sat_tile_bounds_for_bbox = g.get("sat_tile_bounds_for_bbox")
        sat_lonlat_from_world_px = g.get("sat_lonlat_from_world_px")
        cf_mod = g.get("cf") or _cf

        if sat_tile_bounds_for_bbox is None or sat_lonlat_from_world_px is None:
            raise RuntimeError("Satellite helper functions not found.")

        min_lat, min_lon, max_lat, max_lon = ws._satellite_bbox()
        z = int(_get_var(ws.sat_zoom, 0))
        x_min, y_min, x_max, y_max = sat_tile_bounds_for_bbox(min_lat, min_lon, max_lat, max_lon, z)
        cols = x_max - x_min + 1
        rows = y_max - y_min + 1
        width = cols * WEB_TILE_SIZE
        height = rows * WEB_TILE_SIZE
        chunk = max(64, int(_get_var(ws.tile, 768) or 768))
        conf = max(0.001, min(1.0, float(_get_var(ws.conf, 0.15))))
        run_stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1000000000:09d}"
        out_path = ws._satellite_unique_run_output_path(ws._satellite_output_gpkg_path(), run_stamp)
        temp_cache_root = ws._satellite_cache_dir() / "_detection_tmp" / f"lae_dino_existing_v8_z{z}_{run_stamp}_{threading.get_ident()}"
        temp_cache_root.mkdir(parents=True, exist_ok=True)

        old_temp = getattr(ws, "sat_detection_temp_cache_root", None)
        old_thread = getattr(getattr(ws, "_sat_detection_thread_local", None), "cache_root", None)
        ws.sat_detection_temp_cache_root = temp_cache_root
        try:
            ws._sat_detection_thread_local.cache_root = temp_cache_root
        except Exception:
            pass

        records = []
        try:
            jobs = []
            cid = 0
            for y in range(0, height, chunk):
                for x in range(0, width, chunk):
                    cid += 1
                    jobs.append((cid, x, y, min(chunk, width - x), min(chunk, height - y)))
            total = len(jobs)
            ws.log(f"LAE-DINO Satellite Detection started: z={z}, tiles={cols}x{rows}, chunks={total}, conf={conf}")
            workers = max(1, min(8, int(os.environ.get("MUSTATIL_SAT_CHUNK_PREFETCH", "3") or "3"), total or 1))

            def build(job):
                cid, x, y, cw, ch = job
                meta = ws._satellite_build_chunk_to_cache(x_min, y_min, z, x, y, cw, ch, cid, temp_cache_root)
                return job, meta

            with cf_mod.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mustatil-lae-sat") as ex:
                for job, meta in ex.map(build, jobs):
                    cid, x, y, cw, ch = job
                    chunk_path = Path(meta["chunk_path"])
                    used = list(meta.get("tile_paths") or [])
                    im = None
                    try:
                        im = Image.open(chunk_path).convert("RGB")
                        local = _run_lae_on_pil(ws, im, conf)
                        found = 0
                        for r in local:
                            bx1 = max(0.0, min(float(cw), float(r["x1"])))
                            bx2 = max(0.0, min(float(cw), float(r["x2"])))
                            by1 = max(0.0, min(float(ch), float(r["y1"])))
                            by2 = max(0.0, min(float(ch), float(r["y2"])))
                            try:
                                ok, _reason = ws._satellite_detection_box_is_valid(bx1, by1, bx2, by2, cw, ch)
                                if not ok:
                                    continue
                            except Exception:
                                pass
                            gx1 = x_min * WEB_TILE_SIZE + x + bx1
                            gy1 = y_min * WEB_TILE_SIZE + y + by1
                            gx2 = x_min * WEB_TILE_SIZE + x + bx2
                            gy2 = y_min * WEB_TILE_SIZE + y + by2
                            lon1, lat1 = sat_lonlat_from_world_px(gx1, gy1, z)
                            lon2, lat2 = sat_lonlat_from_world_px(gx2, gy2, z)
                            west, east = sorted((float(lon1), float(lon2)))
                            south, north = sorted((float(lat1), float(lat2)))
                            poly = [(west, north), (east, north), (east, south), (west, south), (west, north)]
                            records.append({
                                "class_id": int(r.get("class_id", 0)),
                                "class_name": str(r.get("class_name", r.get("label", "object"))),
                                "confidence": float(r.get("confidence", 0.0)),
                                "model": "LAE-DINO",
                                "model_slot": 1,
                                "zoom": int(z),
                                "tile_x_min": int(x_min),
                                "tile_y_min": int(y_min),
                                "chunk_id": int(cid),
                                "chunk_px_x": int(x),
                                "chunk_px_y": int(y),
                                "bbox_px_x1": float(x + bx1),
                                "bbox_px_y1": float(y + by1),
                                "bbox_px_x2": float(x + bx2),
                                "bbox_px_y2": float(y + by2),
                                "bbox_lon_min": west,
                                "bbox_lat_min": south,
                                "bbox_lon_max": east,
                                "bbox_lat_max": north,
                                "world_px_z": int(z),
                                "world_px_x1": float(gx1),
                                "world_px_y1": float(gy1),
                                "world_px_x2": float(gx2),
                                "world_px_y2": float(gy2),
                                "polygon_lonlat": poly,
                            })
                            found += 1
                        try:
                            pct = min(100.0, 100.0 * int(cid) / max(1, int(total)))
                            ws.log(f"[LAE-DINO {pct:5.1f}%] satellite chunk {cid}/{total}: detections={found}; total={len(records)}")
                        except Exception:
                            ws.log(f"LAE-DINO satellite chunk {cid}/{total}: detections={found}; total={len(records)}")
                    finally:
                        try:
                            if im is not None:
                                im.close()
                        except Exception:
                            pass
                        try:
                            chunk_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                        try:
                            ws._satellite_delete_tile_paths(used)
                        except Exception:
                            pass

            try:
                records = ws._satellite_deduplicate_records(records, 0.90)
            except Exception:
                pass

            ws.sat_last_x_min = int(x_min)
            ws.sat_last_y_min = int(y_min)
            ws.sat_last_z = int(z)
            ws.sat_last_records = [dict(r) for r in records]
            ws.satellite_detections = [dict(r) for r in records]

            if records:
                try:
                    ws._satellite_features_to_file([dict(r) for r in records], out_path)
                    ws.satellite_output_last = str(out_path)
                    ws.log(f"LAE-DINO Satellite GeoPackage written: {out_path}")
                except Exception as exc:
                    ws.log(f"LAE-DINO satellite export warning: {exc}")
            else:
                ws.log("LAE-DINO Satellite Detection finished: no detections.")

            _refresh_ui_after_detections(ws)
            ws.log(f"LAE-DINO Satellite Detection finished: {len(records)} detections")
        finally:
            shutil.rmtree(temp_cache_root, ignore_errors=True)
            ws.sat_detection_temp_cache_root = old_temp
            try:
                ws._sat_detection_thread_local.cache_root = old_thread
            except Exception:
                pass
    except Exception as exc:
        try:
            ws.show_error("LAE-DINO", str(exc))
        except Exception:
            _log("LAE-DINO Satellite failed: " + str(exc))
        traceback.print_exc()


def _browse_file(page, edit, title, filt):
    try:
        from PySide6.QtWidgets import QFileDialog
        start = str(Path(edit.text()).parent if edit.text() else Path.home())
        p, _ = QFileDialog.getOpenFileName(page, title, start, filt)
        if p:
            edit.setText(p)
    except Exception:
        pass


def _browse_dir(page, edit, title):
    try:
        from PySide6.QtWidgets import QFileDialog
        start = edit.text() or str(Path.home())
        p = QFileDialog.getExistingDirectory(page, title, start)
        if p:
            edit.setText(p)
    except Exception:
        pass


def _sync(ws, py_edit, repo_edit, cfg_edit, weights_edit, classes_edit, device_combo):
    try:
        ws.mustatil_lae_existing_v8_python = str(py_edit.text()).strip().strip('"')
        ws.mustatil_lae_existing_v8_repo = str(repo_edit.text()).strip().strip('"')
        ws.mustatil_lae_existing_v8_config = str(cfg_edit.text()).strip().strip('"')
        ws.mustatil_lae_existing_v8_weights = str(weights_edit.text()).strip().strip('"')
        ws.mustatil_lae_existing_v8_classes_text = str(classes_edit.text())
        ws.mustatil_lae_existing_v8_device_text = str(device_combo.currentText())
    except Exception:
        pass


def _build_existing_lae_tab(ws, tab_kind: str):
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QGridLayout, QGroupBox, QLabel, QLineEdit,
        QPushButton, QComboBox, QHBoxLayout
    )

    page = QWidget()
    page.setObjectName("MustatilLAEDINOExistingTabV8")
    lay = QVBoxLayout(page)
    lay.setContentsMargins(8, 8, 8, 8)

    if tab_kind == "satellite":
        header_text = "LAE-DINO für Satellite Detection. Nur der vorhandene LAE-DINO-Tab wurde ersetzt; Console zeigt Prozent; die rechte Karte bleibt original."
    elif tab_kind == "detection":
        header_text = "LAE-DINO für Detection. Nur der vorhandene LAE-DINO-Tab wurde ersetzt; Console zeigt Prozent; die rechte Preview bleibt original."
    else:
        header_text = "LAE-DINO. Nur der vorhandene LAE-DINO-Tab wurde ersetzt; Console zeigt Prozent; rechts bleibt Mustatil original."
    header = QLabel(header_text)
    header.setWordWrap(True)
    lay.addWidget(header)

    box = QGroupBox("Runtime - benutzt LAE-DINO-PATCH; Console zeigt Prozent; CLI gefixt; Auto Config/Checkpoint Match + strict model config + inference pipeline patch; nie die alte LAE-DINO/.venv")
    g = QGridLayout(box)
    py_edit = QLineEdit(str(getattr(ws, "mustatil_lae_existing_v8_python", "") or _py()))
    repo_edit = QLineEdit(str(getattr(ws, "mustatil_lae_existing_v8_repo", "") or _repo()))
    cfg_edit = QLineEdit(str(getattr(ws, "mustatil_lae_existing_v8_config", "") or _config_path()))
    weights_edit = QLineEdit(str(getattr(ws, "mustatil_lae_existing_v8_weights", "") or _weights_path()))

    rows = [
        ("Python", py_edit, "file", "Python (*.exe *.bat);;All files (*)"),
        ("Repo", repo_edit, "dir", ""),
        ("Config", cfg_edit, "file", "Config (*.py);;All files (*)"),
        ("Weights", weights_edit, "file", "Weights (*.pth *.pt);;All files (*)"),
    ]
    for row, (name, edit, kind, filt) in enumerate(rows):
        g.addWidget(QLabel(name), row, 0)
        h = QHBoxLayout()
        h.addWidget(edit, 1)
        b = QPushButton("...")
        if kind == "dir":
            b.clicked.connect(lambda _, e=edit, n=name: _browse_dir(page, e, "Select " + n))
        else:
            b.clicked.connect(lambda _, e=edit, n=name, f=filt: _browse_file(page, e, "Select " + n, f))
        h.addWidget(b)
        g.addLayout(h, row, 1)
    lay.addWidget(box)

    settings = QGroupBox("Detection settings")
    s = QGridLayout(settings)
    classes = QLineEdit(str(getattr(ws, "mustatil_lae_existing_v8_classes_text", "") or "mustatil, burial mound, tumulus, stone enclosure, rectangular structure"))
    device = QComboBox()
    device.setEditable(True)
    device.addItems(["auto", "cuda:0", "cuda", "cpu"])
    try:
        cur = str(getattr(ws, "mustatil_lae_existing_v8_device_text", "auto") or "auto")
        ix = device.findText(cur)
        if ix >= 0:
            device.setCurrentIndex(ix)
        else:
            device.setEditText(cur)
    except Exception:
        pass

    s.addWidget(QLabel("Classes / prompts"), 0, 0)
    s.addWidget(classes, 0, 1)
    s.addWidget(QLabel("Device"), 1, 0)
    s.addWidget(device, 1, 1)
    lay.addWidget(settings)

    status = QLabel("Idle")
    status.setWordWrap(True)
    lay.addWidget(status)

    actions = QGroupBox("Actions")
    a = QGridLayout(actions)
    b_auto = QPushButton("Auto build runtime")
    b_match = QPushButton("Auto match config/weights strict")
    b_check = QPushButton("Check runtime")
    if tab_kind == "satellite":
        b_run = QPushButton("Run LAE-DINO on selected satellite map")
    elif tab_kind == "detection":
        b_run = QPushButton("Run LAE-DINO on Detection image")
    else:
        b_run = QPushButton("Run LAE-DINO")
    a.addWidget(b_auto, 0, 0)
    a.addWidget(b_match, 0, 1)
    a.addWidget(b_check, 1, 0)
    a.addWidget(b_run, 1, 1)
    lay.addWidget(actions)
    lay.addStretch(1)

    def sync():
        _sync(ws, py_edit, repo_edit, cfg_edit, weights_edit, classes, device)

    for edit in [py_edit, repo_edit, cfg_edit, weights_edit, classes]:
        try:
            edit.textChanged.connect(lambda *_: sync())
        except Exception:
            pass
    try:
        device.currentTextChanged.connect(lambda *_: sync())
    except Exception:
        pass

    def build_runtime():
        sync()
        _status(ws, status, "Building LAE-DINO runtime in background...")
        _install_runtime(ws, status, "auto")
        try:
            py_edit.setText(str(_py()))
            repo_edit.setText(str(_repo()))
            cfg_edit.setText(str(_config_path()))
            if _weights_path():
                weights_edit.setText(_weights_path())
        except Exception:
            pass
        sync()

    def check_runtime():
        sync()
        _check_runtime(ws, status)

    def auto_match_paths():
        sync()
        try:
            cfg, weights = _auto_match_lae_paths(ws, status)
            cfg_edit.setText(str(cfg))
            weights_edit.setText(str(weights))
            sync()
        except Exception as exc:
            try:
                ws.show_error("LAE-DINO Auto Match", str(exc))
            except Exception:
                _status(ws, status, "Auto match failed: " + str(exc))

    def run_model():
        sync()
        try:
            cfg, weights = _auto_match_lae_paths(ws, status)
            cfg_edit.setText(str(cfg))
            weights_edit.setText(str(weights))
            sync()
        except Exception as exc:
            try:
                ws.log("Auto match before run failed: " + str(exc))
            except Exception:
                pass
        actual_kind = tab_kind
        if actual_kind == "unknown":
            actual_kind = _infer_tab_kind(page)
        if actual_kind == "satellite":
            _status(ws, status, "Starting LAE-DINO Satellite Detection...")
            _run_async(ws, "LAE-DINO Satellite Detection", lambda: _run_satellite(ws), allow_parallel=True)
        else:
            _status(ws, status, "Starting LAE-DINO Detection...")
            _run_async(ws, "LAE-DINO Detection", lambda: _run_detection_image(ws), allow_parallel=True)

    b_auto.clicked.connect(lambda: _run_async(ws, "Build LAE-DINO runtime", build_runtime, allow_parallel=True))
    b_match.clicked.connect(lambda: _run_async(ws, "Auto match LAE-DINO config/weights", auto_match_paths, allow_parallel=True))
    b_check.clicked.connect(lambda: _run_async(ws, "Check LAE-DINO runtime", check_runtime, allow_parallel=True))
    b_run.clicked.connect(run_model)
    sync()
    return page


def _label_is_lae_dino(label: str) -> bool:
    low = str(label or "").strip().lower()
    return "lae" in low and "dino" in low


def _patch_existing_lae_tabs(root=None):
    global _PATCHING_NOW
    if _PATCHING_NOW:
        return
    _PATCHING_NOW = True
    try:
        from PySide6.QtWidgets import QApplication, QTabWidget

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
        if app is not None:
            for w in app.allWidgets():
                try:
                    if isinstance(w, QTabWidget):
                        widgets.append(w)
                except Exception:
                    pass

        seen = set()
        for tw in widgets:
            if id(tw) in seen:
                continue
            seen.add(id(tw))

            for i in range(tw.count() - 1, -1, -1):
                try:
                    label = str(tw.tabText(i) or "").strip()
                    if not _label_is_lae_dino(label):
                        continue
                    old = tw.widget(i)
                    if old is not None and getattr(old, "objectName", lambda: "")() == "MustatilLAEDINOExistingTabV8":
                        continue
                    if old is not None and id(old) in _PATCHED_WIDGETS:
                        continue

                    ws = _workspace_from_widget(old or tw)
                    if ws is None:
                        continue
                    tab_kind = _infer_tab_kind(old or tw)

                    new_page = _build_existing_lae_tab(ws, tab_kind)
                    cur = tw.currentIndex()
                    old_label = tw.tabText(i)

                    tw.removeTab(i)
                    try:
                        if old is not None:
                            old.deleteLater()
                    except Exception:
                        pass
                    tw.insertTab(i, new_page, old_label or "LAE-DINO")
                    if cur == i:
                        tw.setCurrentIndex(i)
                    _PATCHED_WIDGETS.add(id(new_page))
                    try:
                        ws.log(f"Existing LAE-DINO tab patched V8 for {tab_kind}. No new tab created.")
                    except Exception:
                        pass
                except Exception as exc:
                    _log("patch LAE-DINO tab failed: " + str(exc))
                    traceback.print_exc()
    finally:
        _PATCHING_NOW = False


def _install_hook():
    global _PATCHED_QTAB, _ORIG_ADD, _ORIG_INSERT
    if _PATCHED_QTAB:
        return
    try:
        from PySide6.QtWidgets import QTabWidget
        from PySide6.QtCore import QTimer
    except Exception as exc:
        _log("Qt unavailable: " + str(exc))
        return

    _ORIG_ADD = QTabWidget.addTab
    _ORIG_INSERT = QTabWidget.insertTab

    def after(label):
        try:
            if _label_is_lae_dino(label):
                QTimer.singleShot(100, lambda: _patch_existing_lae_tabs())
                QTimer.singleShot(1000, lambda: _patch_existing_lae_tabs())
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
        after(label)
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
        after(label)
        return res

    QTabWidget.addTab = addTab_patched
    QTabWidget.insertTab = insertTab_patched
    _PATCHED_QTAB = True

    try:
        QTimer.singleShot(500, lambda: _patch_existing_lae_tabs())
        QTimer.singleShot(2000, lambda: _patch_existing_lae_tabs())
        QTimer.singleShot(5000, lambda: _patch_existing_lae_tabs())
        QTimer.singleShot(9000, lambda: _patch_existing_lae_tabs())
    except Exception:
        pass

    _log("LAE-DINO existing-tab V8 progress hook installed.")


def mustatil_plugin_init():
    _install_hook()
    _patch_existing_lae_tabs()


def register_plugin(app=None, main_window=None):
    _install_hook()
    _patch_existing_lae_tabs(main_window or app)
    return True


def init_plugin(app=None, main_window=None):
    return register_plugin(app, main_window)


def load_plugin(app=None, main_window=None):
    return register_plugin(app, main_window)


try:
    _install_hook()
except Exception:
    pass
