#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mustatil plugin: ADAF v7 - Detection/Satellite only, no Trainer tab, Open ADAF Notebook button.

Drop this single file into mustatil_plugins and restart Mustatil.

What this plugin does:
- Adds an "ADAF" inner tab directly to the RIGHT of "LAE-DINO" in the left model selector
  of Detection and Satellite Detection, without touching the right preview/map.
- Adds an "ADAF Trainer" tab directly to the RIGHT of "LAE-DINO Trainer" in training/model tabs.
- Uses the original ADAF model *.tar files from C:\ADAF\adaf\adaf\ml_models.
- Keeps ADAF isolated in its own conda environment.
- Adds download/install buttons and a simplified one-dropdown model selector.
"""
from __future__ import annotations

import os
import re
import sys
import json
import time
import math
import shutil
import tempfile
import traceback
import subprocess
import webbrowser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_PATCHED_QTAB = False
_ORIG_ADD = None
_ORIG_INSERT = None
_PATCHED_INNER_TABS = set()
_PATCHED_TRAINING_TABS = set()
_PATCHED_WS = set()

WEB_TILE_SIZE = 256

ADAF_GITHUB_URL = "https://github.com/EarthObservation/adaf.git"
ADAF_MODELS_ZENODO_URL = "https://zenodo.org/records/15848663"
DEFAULT_ADAF_REPO = r"C:\ADAF\adaf"
DEFAULT_ADAF_ENV = "adaf"
DEFAULT_CONDA = os.path.expanduser(r"~\miniconda3\Scripts\conda.exe")
DEFAULT_MODEL_DIR = r"C:\ADAF\adaf\adaf\ml_models"

ADAF_WORKER_CODE = '# mustatil_adaf_direct_worker.py\n# External worker for Mustatil ADAF tab.\n# Runs inside the ADAF conda env and tries to call the installed ADAF/AiTLAS stack.\n# It never fabricates detections. If no direct API is exposed by the installed ADAF version,\n# it writes a clear error/report JSON with zero detections.\n\nfrom __future__ import annotations\nimport os, sys, json, time, tarfile, tempfile, traceback, importlib, pkgutil, inspect\nfrom pathlib import Path\n\ndef safe_import(name):\n    try:\n        mod = importlib.import_module(name)\n        return mod, ""\n    except Exception as e:\n        return None, repr(e)\n\ndef classify_model(name):\n    low=name.lower()\n    task="object_detection" if any(k in low for k in ["object","detect","faster","rcnn","bbox","box","barrow","ringfort","enclosure","ao","all"]) else "unknown"\n    if any(k in low for k in ["semantic","seg","hrnet","unet","mask"]):\n        task="semantic_segmentation"\n    feature="archaeology"\n    if "barrow" in low: feature="barrow"\n    elif "ringfort" in low or "ring_fort" in low: feature="ringfort"\n    elif "enclosure" in low: feature="enclosure"\n    elif "ao" in low or "all" in low or "arch" in low: feature="all_archaeology"\n    return task,feature\n\ndef inspect_tar(path):\n    info={"exists": Path(path).exists(), "members": [], "pt_files": [], "json_files": [], "yaml_files": [], "py_files": []}\n    if not info["exists"]:\n        return info\n    try:\n        with tarfile.open(path, "r:*") as tf:\n            for m in tf.getmembers()[:300]:\n                n=m.name\n                info["members"].append(n)\n                low=n.lower()\n                if low.endswith((".pt",".pth",".ckpt")): info["pt_files"].append(n)\n                if low.endswith(".json"): info["json_files"].append(n)\n                if low.endswith((".yml",".yaml")): info["yaml_files"].append(n)\n                if low.endswith(".py"): info["py_files"].append(n)\n    except Exception as e:\n        info["error"]=repr(e)\n    return info\n\ndef normalize_detections(raw, labels=None):\n    labels=labels or []\n    out=[]\n    if raw is None:\n        return out\n    if isinstance(raw, dict):\n        if isinstance(raw.get("detections"), list): raw=raw["detections"]\n        elif isinstance(raw.get("predictions"), list): raw=raw["predictions"]\n        elif isinstance(raw.get("boxes"), (list,tuple)): raw=[raw]\n        else: raw=[raw]\n    if not isinstance(raw, (list,tuple)):\n        return out\n    for item in raw:\n        try:\n            if isinstance(item, dict):\n                box = item.get("bbox") or item.get("box") or item.get("boxes")\n                if box is None and all(k in item for k in ("x1","y1","x2","y2")):\n                    box=[item["x1"],item["y1"],item["x2"],item["y2"]]\n                score = item.get("score", item.get("confidence", item.get("probability", 0.0)))\n                lab = item.get("label", item.get("class_name", item.get("name", item.get("class_id", 0))))\n            elif isinstance(item, (list,tuple)) and len(item) >= 5:\n                box=item[:4]; score=item[4]; lab=item[5] if len(item)>5 else 0\n            else:\n                continue\n            if box is None:\n                continue\n            vals=[float(x) for x in list(box)[:4]]\n            x1,y1,x2,y2=vals\n            if x2 < x1: x1,x2=x2,x1\n            if y2 < y1: y1,y2=y2,y1\n            if x2 <= x1 or y2 <= y1:\n                continue\n            try:\n                cid=int(lab)\n                cname=labels[cid] if 0 <= cid < len(labels) else str(lab)\n            except Exception:\n                cname=str(lab)\n                cid=0\n                for i,l in enumerate(labels):\n                    if str(l).lower() in cname.lower() or cname.lower() in str(l).lower():\n                        cid=i; break\n            out.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"confidence":float(score or 0.0),"class_id":cid,"class_name":cname,"label":cname,"model":"ADAF"})\n        except Exception:\n            continue\n    return out\n\ndef try_direct_calls(repo, model_tar, image_path, conf, device, labels):\n    candidates = [\n        ("adaf", "predict"),\n        ("adaf", "detect"),\n        ("adaf", "inference"),\n        ("adaf.predict", "predict"),\n        ("adaf.inference", "predict"),\n        ("adaf.inference", "detect"),\n        ("adaf.object_detection", "predict"),\n        ("adaf.object_detection", "detect"),\n        ("adaf.ml", "predict"),\n        ("adaf.models", "predict"),\n    ]\n    errors=[]\n    for modname, funcname in candidates:\n        mod, err=safe_import(modname)\n        if mod is None:\n            errors.append(f"{modname}: {err}")\n            continue\n        fn=getattr(mod, funcname, None)\n        if not callable(fn):\n            errors.append(f"{modname}.{funcname}: not found")\n            continue\n        try:\n            sig=str(inspect.signature(fn))\n        except Exception:\n            sig=""\n        attempts=[\n            lambda: fn(str(image_path), str(model_tar), confidence=conf, device=device),\n            lambda: fn(str(image_path), str(model_tar), conf, device),\n            lambda: fn(image_path=str(image_path), model_path=str(model_tar), threshold=conf, device=device),\n            lambda: fn(input_path=str(image_path), model_path=str(model_tar), conf=conf, device=device),\n            lambda: fn(str(image_path), model=str(model_tar), threshold=conf),\n        ]\n        for a in attempts:\n            try:\n                raw=a()\n                dets=normalize_detections(raw, labels)\n                return dets, {"api": f"{modname}.{funcname}", "signature": sig, "raw_type": str(type(raw))}\n            except Exception as e:\n                errors.append(f"{modname}.{funcname}{sig}: {repr(e)}")\n    return [], {"api": "", "errors": errors[-50:]}\n\ndef try_torchscript_or_pickled_model(model_tar, image_path, conf, device, labels):\n    import torch\n    from PIL import Image\n    import torchvision.transforms.functional as F\n    tmp = tempfile.TemporaryDirectory(prefix="mustatil_adaf_model_")\n    root = Path(tmp.name)\n    with tarfile.open(model_tar, "r:*") as tf:\n        tf.extractall(root)\n    pts = []\n    for p in root.rglob("*"):\n        if p.suffix.lower() in {".pt",".pth",".ckpt"}:\n            pts.append(p)\n    if not pts:\n        return [], {"torch_attempt": "no .pt/.pth/.ckpt inside tar"}\n    pts.sort(key=lambda p: p.stat().st_size, reverse=True)\n    last_errors=[]\n    im = Image.open(image_path).convert("RGB")\n    tensor = F.to_tensor(im).to(device)\n    for pt in pts[:3]:\n        for mode in ("jit", "pickle"):\n            try:\n                if mode == "jit":\n                    model = torch.jit.load(str(pt), map_location=device)\n                else:\n                    obj = torch.load(str(pt), map_location=device)\n                    if hasattr(obj, "eval") and callable(obj):\n                        model = obj\n                    else:\n                        last_errors.append(f"{pt.name}: torch.load returned {type(obj)}, not callable model")\n                        continue\n                model.eval()\n                with torch.no_grad():\n                    try:\n                        raw = model([tensor])\n                    except Exception:\n                        raw = model(tensor.unsqueeze(0))\n                dets = normalize_detections(raw, labels)\n                return dets, {"torch_attempt": f"{mode}:{pt.name}", "raw_type": str(type(raw))}\n            except Exception as e:\n                last_errors.append(f"{mode}:{pt.name}: {repr(e)}")\n    return [], {"torch_attempt": "failed", "errors": last_errors[-20:]}\n\ndef main():\n    import argparse\n    ap=argparse.ArgumentParser()\n    ap.add_argument("--repo", required=True)\n    ap.add_argument("--model", required=True)\n    ap.add_argument("--input", required=True)\n    ap.add_argument("--output", required=True)\n    ap.add_argument("--conf", type=float, default=0.25)\n    ap.add_argument("--device", default="cpu")\n    ap.add_argument("--labels", default="")\n    args=ap.parse_args()\n\n    repo=Path(args.repo)\n    model_tar=Path(args.model)\n    image_path=Path(args.input)\n    out_path=Path(args.output)\n    out_path.parent.mkdir(parents=True, exist_ok=True)\n    labels=[p.strip() for p in args.labels.replace(";",",").split(",") if p.strip()]\n\n    if str(repo) not in sys.path:\n        sys.path.insert(0, str(repo))\n    if str(repo/"adaf") not in sys.path:\n        sys.path.insert(0, str(repo/"adaf"))\n\n    report={\n        "created": time.strftime("%Y-%m-%d %H:%M:%S"),\n        "repo": str(repo),\n        "model": str(model_tar),\n        "input": str(image_path),\n        "conf": args.conf,\n        "device": args.device,\n        "labels": labels,\n        "imports": {},\n        "model_tar": {},\n        "detections": [],\n        "status": "started",\n        "notes": [],\n    }\n\n    try:\n        for m in ["torch","torchvision","rasterio","fiona","rvt","rvt.default","aitlas","adaf"]:\n            mod,err=safe_import(m)\n            report["imports"][m]={"ok": mod is not None, "version": getattr(mod,"__version__","unknown") if mod else "", "error": err}\n        report["model_tar"]=inspect_tar(model_tar)\n        task,feature=classify_model(model_tar.name)\n        report["model_task_guess"]=task\n        report["model_feature_guess"]=feature\n\n        dets, api_info = try_direct_calls(repo, model_tar, image_path, args.conf, args.device, labels)\n        report["direct_api_info"]=api_info\n        if not dets:\n            try:\n                dets, torch_info = try_torchscript_or_pickled_model(model_tar, image_path, args.conf, args.device, labels)\n                report["torch_fallback_info"]=torch_info\n            except Exception as e:\n                report["torch_fallback_info"]={"error":repr(e), "traceback":traceback.format_exc()}\n\n        filtered=[]\n        for d in dets:\n            try:\n                if float(d.get("confidence",0.0)) < float(args.conf):\n                    continue\n            except Exception:\n                pass\n            if not d.get("class_name") and labels:\n                cid=int(d.get("class_id",0))\n                if 0 <= cid < len(labels):\n                    d["class_name"]=labels[cid]\n                    d["label"]=labels[cid]\n            filtered.append(d)\n\n        report["detections"]=filtered\n        report["status"]="ok" if filtered else "no_detections_or_api_missing"\n        if not filtered:\n            report["notes"].append("No detections were returned. If the model TAR is state-dict-only, the installed ADAF release must expose its exact inference API for headless use.")\n            report["notes"].append("Use the Open ADAF Notebook button for the official ADAF notebook/widget.")\n\n    except Exception:\n        report["status"]="error"\n        report["traceback"]=traceback.format_exc()\n\n    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")\n    print("MUSTATIL_ADAF_RESULT=" + str(out_path))\n    print("MUSTATIL_ADAF_STATUS=" + report["status"])\n    print("MUSTATIL_ADAF_DETECTIONS=" + str(len(report.get("detections",[]))))\n\nif __name__ == "__main__":\n    main()\n'


def _log(msg: str):
    try:
        print("[Mustatil ADAF Tabs] " + str(msg))
    except Exception:
        pass


def _get_var(v, default=""):
    try:
        if hasattr(v, "get"):
            return v.get()
    except Exception:
        pass
    return default


def _workspace_from_widget(widget: Any) -> Optional[Any]:
    cur = widget
    for _ in range(140):
        if cur is None:
            break
        try:
            if hasattr(cur, "tabs") and (hasattr(cur, "dets") or hasattr(cur, "satellite_detections") or hasattr(cur, "log")):
                return cur
        except Exception:
            pass
        try:
            for attr in ("ws", "workspace", "main_window"):
                obj = getattr(cur, attr, None)
                if obj is not None and hasattr(obj, "tabs"):
                    return obj
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


def _item_get(obj: Any, keys: Iterable[str], default=None):
    if isinstance(obj, dict):
        for k in keys:
            if k in obj:
                return obj.get(k)
        return default
    for k in keys:
        try:
            if hasattr(obj, k):
                return getattr(obj, k)
        except Exception:
            pass
    return default


def _item_cls(obj: Any) -> Optional[int]:
    val = _item_get(obj, ("cls", "class_id", "class", "category_id", "label_id"), None)
    try:
        if val is not None and str(val).strip() != "":
            return int(float(val))
    except Exception:
        pass
    lab = str(_item_get(obj, ("label", "class_name", "name", "status"), "") or "").lower().strip()
    if lab in {"positive", "mustatil", "true_positive", "true positive"}:
        return 0
    if lab in {"false_positive", "false positive", "false-positive", "fp"}:
        return 1
    return None


def _item_label(obj: Any) -> str:
    return str(_item_get(obj, ("label", "class_name", "name", "status"), "") or "")


def _item_score(obj: Any) -> float:
    v = _item_get(obj, ("confidence", "conf", "score", "probability"), 0.0)
    try:
        f = float(v)
        return 0.0 if math.isnan(f) else f
    except Exception:
        return 0.0


def _bbox(obj: Any) -> Optional[Tuple[float, float, float, float]]:
    candidates = [
        ("x1", "y1", "x2", "y2"),
        ("px_x1", "px_y1", "px_x2", "px_y2"),
        ("bbox_px_x1", "bbox_px_y1", "bbox_px_x2", "bbox_px_y2"),
        ("world_px_x1", "world_px_y1", "world_px_x2", "world_px_y2"),
        ("left", "top", "right", "bottom"),
    ]
    for keys in candidates:
        vals = [_item_get(obj, (k,), None) for k in keys]
        if all(v is not None for v in vals):
            try:
                x1, y1, x2, y2 = [float(v) for v in vals]
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1
                return x1, y1, x2, y2
            except Exception:
                pass
    raw = _item_get(obj, ("pixel_bbox", "bbox", "box"), None)
    if raw is not None:
        try:
            vals = [float(p) for p in (raw.strip("[]() ").replace(";", ",").split(",") if isinstance(raw, str) else list(raw))[:4]]
            x1, y1, x2, y2 = vals
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            return x1, y1, x2, y2
        except Exception:
            pass
    return None


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    bb = max(0, bx2 - bx1) * max(0, by2 - by1)
    den = aa + bb - inter
    return 0.0 if den <= 0 else inter / den


def _nms(items: Iterable[Any], thr: float) -> List[Any]:
    data = list(items or [])
    ordered = sorted(enumerate(data), key=lambda p: _item_score(p[1]), reverse=True)
    kept = []
    for idx, it in ordered:
        b = _bbox(it)
        if b is None:
            kept.append((idx, it, b))
            continue
        cls = _item_cls(it)
        suppress = False
        for _, kit, kb in kept:
            if kb is None:
                continue
            if _item_cls(kit) != cls:
                continue
            if _iou(b, kb) >= thr:
                suppress = True
                break
        if not suppress:
            kept.append((idx, it, b))
    kept.sort(key=lambda p: p[0])
    return [p[1] for p in kept]


def _selected_class(ws):
    try:
        v = getattr(ws, "mustatil_owl_selected_class_id", None)
        return None if v is None or v == "" else int(v)
    except Exception:
        return None


def _apply_filters(items, ws):
    out = []
    sel = _selected_class(ws)
    hide = bool(getattr(ws, "mustatil_owl_hide_fp", False))
    for it in list(items or []):
        cls = _item_cls(it)
        lab = _item_label(it).lower()
        if hide and (cls == 1 or lab in {"false_positive", "false positive", "false-positive", "fp"}):
            continue
        if sel is not None and cls != sel:
            continue
        out.append(it)
    if bool(getattr(ws, "mustatil_owl_geo_nms_enabled", False)):
        out = _nms(out, float(getattr(ws, "mustatil_owl_geo_nms_iou", 0.35)))
    return out


def _patch_workspace_filters(ws):
    if ws is None or id(ws) in _PATCHED_WS:
        return
    _PATCHED_WS.add(id(ws))
    try:
        old_visible = getattr(ws, "visible", None)
        if callable(old_visible) and not getattr(ws, "_adaf_safe_visible_patched", False):
            def visible_filtered(*a, **k):
                return _apply_filters(old_visible(*a, **k), ws)
            ws.visible = visible_filtered
            ws._adaf_safe_visible_patched = True
    except Exception as exc:
        _log("visible patch failed: " + str(exc))
    try:
        old_sat = getattr(ws, "_satellite_visible_records", None)
        if callable(old_sat) and not getattr(ws, "_adaf_safe_sat_visible_patched", False):
            def sat_filtered(records=None, *a, **k):
                return _apply_filters(old_sat(records, *a, **k), ws)
            ws._satellite_visible_records = sat_filtered
            ws._adaf_safe_sat_visible_patched = True
    except Exception as exc:
        _log("sat visible patch failed: " + str(exc))


def _redraw(ws):
    for call in (
        lambda: ws.redraw(fit=False),
        lambda: ws.redraw(),
        lambda: ws.satellite_redraw_detection_overlay(),
        lambda: ws.refresh_layers(),
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


def _refresh_combos(ws):
    combos = list(getattr(ws, "mustatil_owl_class_combos", []) or [])
    known = {0: "archaeology / ADAF", 1: "false_positive"}
    for it in list(getattr(ws, "dets", []) or []) + list(getattr(ws, "satellite_detections", []) or []) + list(getattr(ws, "sat_last_records", []) or []):
        cls = _item_cls(it)
        if cls is None:
            continue
        lab = _item_label(it).strip() or ("false_positive" if cls == 1 else f"class {cls}")
        known[int(cls)] = lab
    for combo in combos:
        try:
            cur = _selected_class(ws)
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("All classes", None)
            for cls in sorted(known):
                combo.addItem(f"Class {cls}: {known[cls]}", int(cls))
            ix = 0
            for i in range(combo.count()):
                if combo.itemData(i) == cur:
                    ix = i
                    break
            combo.setCurrentIndex(ix)
            combo.blockSignals(False)
        except Exception:
            pass


def _run_as_task(ws, title, fn, allow_parallel=False):
    try:
        return ws.run_task(title, fn, allow_parallel=allow_parallel)
    except TypeError:
        try:
            return ws.run_task(title, fn)
        except Exception:
            pass
    except Exception:
        pass
    return fn()


def _model_dir(ws=None) -> Path:
    raw = str(getattr(ws, "mustatil_adaf_model_dir", "") or os.environ.get("MUSTATIL_ADAF_MODEL_DIR", "") or DEFAULT_MODEL_DIR).strip().strip('"')
    return Path(raw)


def _repo_dir(ws=None) -> Path:
    raw = str(getattr(ws, "mustatil_adaf_repo", "") or os.environ.get("MUSTATIL_ADAF_REPO", "") or DEFAULT_ADAF_REPO).strip().strip('"')
    return Path(raw)


def _conda_exe(ws=None) -> Path:
    raw = str(getattr(ws, "mustatil_adaf_conda", "") or os.environ.get("MUSTATIL_ADAF_CONDA", "") or DEFAULT_CONDA).strip().strip('"')
    return Path(raw)


def _adaf_env(ws=None) -> str:
    return str(getattr(ws, "mustatil_adaf_env", "") or os.environ.get("MUSTATIL_ADAF_ENV", "") or DEFAULT_ADAF_ENV).strip() or DEFAULT_ADAF_ENV


def _classify_adaf_model_name(name: str) -> Dict[str, str]:
    low = name.lower()
    task = "unknown"
    feature = "archaeology"

    if any(k in low for k in ["object", "detect", "faster", "rcnn", "bbox", "box"]):
        task = "object detection"
    if any(k in low for k in ["semantic", "seg", "hrnet", "unet", "mask"]):
        task = "semantic segmentation"

    if "barrow" in low:
        feature = "barrow"
    elif "ringfort" in low or "ring_fort" in low:
        feature = "ringfort"
    elif "enclosure" in low:
        feature = "enclosure"
    elif "ao" in low or "all" in low or "arch" in low:
        feature = "all archaeology"

    if task == "unknown":
        task = "object detection" if any(k in low for k in ["barrow", "ringfort", "enclosure", "ao", "all"]) else "unknown"

    return {"task": task, "feature": feature}


def _candidate_adaf_model_dirs(ws=None) -> List[Path]:
    """Likely folders containing original ADAF *.tar model files."""
    out = []

    def add(p):
        try:
            p = Path(str(p).strip().strip('"')).expanduser()
            if p and p not in out:
                out.append(p)
        except Exception:
            pass

    add(_model_dir(ws))
    repo = _repo_dir(ws)
    add(repo / "adaf" / "ml_models")
    add(repo / "ml_models")
    add(repo / "models")
    add(repo / "weights")
    add(repo)

    home = Path.home()
    add(home / "Downloads")
    add(home / "Desktop")
    add(home / "Schreibtisch")
    add(home / "Documents")
    add(home / "Dokumente")

    add(Path(r"C:\ADAF"))
    add(Path(r"C:\ADAF\adaf"))
    add(Path(r"C:\ADAF\adaf\adaf"))
    add(Path(r"C:\ADAF\adaf\adaf\ml_models"))

    try:
        add(Path(__file__).resolve().parent)
        add(Path(__file__).resolve().parent / "models")
        add(Path(__file__).resolve().parent / "weights")
        add(Path(__file__).resolve().parent / "adaf_models")
    except Exception:
        pass

    for name in ("project_dir", "project_folder", "project_path", "last_project_dir", "current_project_dir"):
        try:
            v = getattr(ws, name, "")
            if v:
                add(v)
                add(Path(v) / "models")
                add(Path(v) / "weights")
                add(Path(v) / "adaf_models")
        except Exception:
            pass

    return out


def _scan_adaf_models(ws=None) -> List[Dict[str, Any]]:
    """Find original ADAF model TAR files robustly.

    The previous version only scanned exactly C:\ADAF\adaf\adaf\ml_models.
    This version also searches Downloads/Desktop and common ADAF/model folders
    recursively. Only *.tar files are accepted; original ADAF models should not
    be extracted or renamed.
    """
    out = []
    seen = set()
    scanned = []

    for folder in _candidate_adaf_model_dirs(ws):
        try:
            if not folder.exists() or not folder.is_dir():
                continue
            scanned.append(str(folder))

            candidates = list(folder.glob("*.tar"))
            root_s = str(folder).replace("\\", "/").lower()
            recursive_ok = any(k in root_s for k in [
                "/adaf", "/downloads", "/desktop", "/schreibtisch",
                "/documents", "/dokumente", "/models", "/weights", "/adaf_models"
            ])
            if recursive_ok:
                try:
                    candidates.extend(folder.rglob("*.tar"))
                except Exception:
                    pass

            for p in sorted(set(candidates)):
                try:
                    if not p.is_file():
                        continue
                    rp = str(p.resolve())
                    if rp in seen:
                        continue
                    seen.add(rp)
                    cls = _classify_adaf_model_name(p.name)
                    size_mb = p.stat().st_size / (1024 * 1024)
                    label = f"{cls['task']}: {cls['feature']}  —  {p.name} ({size_mb:.1f} MB)"
                    out.append({
                        "path": str(p),
                        "name": p.name,
                        "label": label,
                        "task": cls["task"],
                        "feature": cls["feature"],
                        "size_mb": size_mb,
                    })
                except Exception:
                    continue
        except Exception:
            continue

    try:
        if ws is not None:
            ws.mustatil_adaf_last_scanned_dirs = scanned
    except Exception:
        pass

    def sort_key(m):
        low = m["name"].lower()
        score = 0
        if m.get("task") == "object detection":
            score += 1000
        if "od_" in low:
            score += 500
        if "all" in low or "ao" in low or "arch" in low:
            score += 250
        if "barrow" in low:
            score += 150
        if "ringfort" in low:
            score += 120
        if "enclosure" in low:
            score += 100
        return (-score, low)

    out.sort(key=sort_key)
    return out


def _best_auto_model(ws=None) -> str:
    models = _scan_adaf_models(ws)
    if not models:
        return ""
    def score(m):
        s = 0
        low = m["name"].lower()
        if m["task"] == "object detection":
            s += 1000
        if "all" in low or "ao" in low or "arch" in low:
            s += 300
        if "barrow" in low:
            s += 150
        if "ringfort" in low:
            s += 120
        if "enclosure" in low:
            s += 100
        return -s, m["name"].lower()
    models.sort(key=score)
    return models[0]["path"]


def _fill_model_combo(combo, ws=None, prefer_current=True):
    try:
        cur = combo.currentData()
    except Exception:
        cur = None
    combo.blockSignals(True)
    combo.clear()
    models = _scan_adaf_models(ws)
    if models:
        combo.addItem("Auto: best ADAF object detection model", "__AUTO__")
        for m in models:
            combo.addItem(m["label"], m["path"])
    else:
        combo.addItem("No ADAF .tar models found — download/copy models, then Scan", "")
    if prefer_current and cur:
        ix = combo.findData(cur)
        if ix >= 0:
            combo.setCurrentIndex(ix)
    combo.blockSignals(False)


def _selected_model_from_combo(combo, ws=None) -> str:
    data = combo.currentData()
    if data == "__AUTO__" or data in (None, ""):
        return _best_auto_model(ws)
    return str(data)


def _open_model_folder(ws=None):
    folder = _model_dir(ws)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        os.startfile(str(folder))
    except Exception:
        webbrowser.open(str(folder))


def _write_adaf_download_script(ws=None) -> Path:
    """Create a non-interactive PowerShell fallback downloader for the three official ADAF OD models."""
    folder = _model_dir(ws)
    folder.mkdir(parents=True, exist_ok=True)
    ps1 = folder / "download_adaf_three_od_models_zenodo_15848663.ps1"

    lines = [
        "# Download official ADAF object detection model weights from Zenodo 15848663",
        "# Downloads exactly: OD_barrow.tar, OD_enclosure.tar, OD_ringfort.tar",
        "# Non-interactive: this file is safe to run from Mustatil.",
        "",
        "$ErrorActionPreference = \"Stop\"",
        "$Out = " + repr(str(folder)).replace("'", '"'),
        "New-Item -ItemType Directory -Force -Path $Out | Out-Null",
        "",
        "$files = @(",
        '  @{Name="OD_barrow.tar"; Url="https://zenodo.org/records/15848663/files/OD_barrow.tar?download=1"},',
        '  @{Name="OD_enclosure.tar"; Url="https://zenodo.org/records/15848663/files/OD_enclosure.tar?download=1"},',
        '  @{Name="OD_ringfort.tar"; Url="https://zenodo.org/records/15848663/files/OD_ringfort.tar?download=1"}',
        ")",
        "",
        "foreach ($f in $files) {",
        "    $dest = Join-Path $Out $f.Name",
        "    if ((Test-Path $dest) -and ((Get-Item $dest).Length -gt 1024)) {",
        '        Write-Host "[skip] already exists: $dest"',
        "        continue",
        "    }",
        '    Write-Host "[download] $($f.Name)"',
        "    if (Test-Path $dest) { Remove-Item -Force $dest }",
        "    try {",
        "        Start-BitsTransfer -Source $f.Url -Destination $dest -ErrorAction Stop",
        "    } catch {",
        '        Write-Host "[fallback] Invoke-WebRequest"',
        "        Invoke-WebRequest -Uri $f.Url -OutFile $dest -UseBasicParsing",
        "    }",
        "    if (-not (Test-Path $dest)) { throw \"Download failed: $dest\" }",
        "    if ((Get-Item $dest).Length -le 1024) { throw \"Downloaded file is too small: $dest\" }",
        "}",
        "",
        'Write-Host "Done. Models are in:"',
        "Write-Host $Out",
        'Write-Host "Do NOT extract or rename the .tar files."',
        "",
    ]
    ps1.write_text("\n".join(lines), encoding="utf-8")
    return ps1


def _download_adaf_models_with_powershell(ws=None):
    ps1 = _write_adaf_download_script(ws)
    try:
        subprocess.Popen([
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", str(ps1)
        ], cwd=str(ps1.parent))
    except Exception:
        webbrowser.open(ADAF_MODELS_ZENODO_URL)
    try:
        if ws is not None:
            ws.log("ADAF model downloader started: " + str(ps1))
    except Exception:
        pass
    return ps1


def _open_adaf_downloads(ws=None):
    try:
        webbrowser.open(ADAF_MODELS_ZENODO_URL)
    except Exception:
        pass
    try:
        _open_model_folder(ws)
    except Exception:
        pass


def _run_cmd_logged(ws, cmd, cwd=None, env=None, title="RUN"):
    try:
        ws.log(title + ": " + " ".join([str(x) for x in cmd]))
    except Exception:
        pass
    proc = subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    out = proc.stdout or ""
    try:
        for line in out.splitlines()[-60:]:
            ws.log(line)
    except Exception:
        pass
    if proc.returncode != 0:
        raise RuntimeError("Command failed with exit code %s:\n%s\n\n%s" % (proc.returncode, " ".join(map(str, cmd)), "\n".join(out.splitlines()[-80:])))
    return out


def _clone_or_update_adaf(ws):
    repo = _repo_dir(ws)
    repo.parent.mkdir(parents=True, exist_ok=True)
    if (repo / ".git").exists():
        _run_cmd_logged(ws, ["git", "pull"], cwd=repo, title="ADAF update")
    elif repo.exists() and any(repo.iterdir()):
        try:
            ws.log(f"ADAF repo folder exists and is not empty: {repo}")
        except Exception:
            pass
    else:
        _run_cmd_logged(ws, ["git", "clone", ADAF_GITHUB_URL, str(repo)], cwd=repo.parent, title="ADAF clone")
    (_model_dir(ws)).mkdir(parents=True, exist_ok=True)
    try:
        ws.log(f"ADAF repo ready: {repo}")
        ws.log(f"Put original ADAF model .tar files here: {_model_dir(ws)}")
    except Exception:
        pass


def _ensure_worker(ws=None) -> Path:
    repo = _repo_dir(ws)
    repo.mkdir(parents=True, exist_ok=True)
    worker = repo / "mustatil_adaf_direct_worker.py"
    worker.write_text(ADAF_WORKER_CODE, encoding="utf-8")
    return worker


def _run_adaf_worker_on_image(ws, image_path: Path, model_tar: str, conf: float, device: str, labels: List[str]) -> List[Dict[str, Any]]:
    repo = _repo_dir(ws)
    conda = _conda_exe(ws)
    env = _adaf_env(ws)
    worker = _ensure_worker(ws)
    if not conda.exists():
        raise RuntimeError(f"conda.exe not found: {conda}")
    if not repo.exists():
        raise RuntimeError(f"ADAF repo not found: {repo}. Press 'Clone/update ADAF repo'.")
    if not model_tar or not Path(model_tar).exists():
        raise RuntimeError(f"ADAF model .tar not found: {model_tar}. Press 'Download/Open ADAF models' and copy the TARs into ml_models.")
    if not image_path.exists():
        raise RuntimeError(f"Input image not found: {image_path}")

    out_dir = repo / "mustatil_adaf_runtime" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / ("adaf_pred_" + str(time.time_ns()) + ".json")
    cmd = [
        str(conda), "run", "--live-stream", "-n", env, "python", str(worker),
        "--repo", str(repo),
        "--model", str(model_tar),
        "--input", str(image_path),
        "--output", str(out_json),
        "--conf", str(float(conf)),
        "--device", str(device),
        "--labels", ",".join(labels),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(repo),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = proc.stdout or ""
    try:
        for line in output.splitlines()[-30:]:
            ws.log(line)
    except Exception:
        pass
    if proc.returncode != 0:
        raise RuntimeError("ADAF worker failed with exit code %s:\n%s" % (proc.returncode, "\n".join(output.splitlines()[-80:])))
    if not out_json.exists():
        raise RuntimeError("ADAF worker did not write result JSON.")
    data = json.loads(out_json.read_text(encoding="utf-8", errors="replace"))
    try:
        ws.mustatil_adaf_last_report = str(out_json)
    except Exception:
        pass
    status = data.get("status", "")
    if status not in {"ok", "no_detections_or_api_missing"}:
        raise RuntimeError("ADAF worker status: %s\n%s" % (status, data.get("traceback", "")))
    dets = data.get("detections", []) or []
    if not dets:
        try:
            ws.log("ADAF returned no detections. Report: " + str(out_json))
            for n in data.get("notes", []):
                ws.log("ADAF note: " + str(n))
        except Exception:
            pass
    return dets


def _adaf_labels_for_model(model_path: str) -> List[str]:
    name = Path(str(model_path or "")).name.lower()
    if "barrow" in name:
        return ["barrow"]
    if "ringfort" in name or "ring_fort" in name:
        return ["ringfort"]
    if "enclosure" in name:
        return ["enclosure"]
    return ["archaeological_feature", "barrow", "ringfort", "enclosure"]


def _adaf_detect_pil(ws, pil_image, conf: float, model_tar: str, device: str) -> List[Dict[str, Any]]:
    labels = _adaf_labels_for_model(model_tar)
    with tempfile.TemporaryDirectory(prefix="mustatil_adaf_chunk_") as td:
        p = Path(td) / "chunk.png"
        pil_image.convert("RGB").save(p)
        return _run_adaf_worker_on_image(ws, p, model_tar, conf, device, labels)


def _run_adaf_detection_image(ws):
    try:
        from PIL import Image
        from mustatil_legacy_backend import Det

        model_tar = str(getattr(ws, "mustatil_adaf_selected_model", "") or _best_auto_model(ws))
        device = str(getattr(ws, "mustatil_adaf_device", "cpu") or "cpu")
        img_path = str(_get_var(getattr(ws, "image", None), "") or "").strip().strip('"')
        if not img_path or not Path(img_path).exists():
            raise RuntimeError("No Detection image selected.")
        conf = max(0.001, min(1.0, float(_get_var(getattr(ws, "conf", None), 0.25))))
        tile = max(128, int(_get_var(getattr(ws, "tile", None), 768) or 768))
        overlap = max(0, min(tile - 1, int(_get_var(getattr(ws, "overlap", None), 128) or 128)))
        step = max(1, tile - overlap)

        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        ws.log(f"ADAF Detection started: {Path(img_path).name} {W}x{H}, model={Path(model_tar).name}, tile={tile}, overlap={overlap}, conf={conf}, device={device}")

        dets = []
        count = 0
        for y in range(0, H, step):
            for x in range(0, W, step):
                x2 = min(W, x + tile)
                y2 = min(H, y + tile)
                local = _adaf_detect_pil(ws, img.crop((x, y, x2, y2)), conf, model_tar, device)
                for r in local:
                    dets.append(Det(
                        0,
                        "ADAF " + Path(model_tar).stem,
                        int(r.get("class_id", 0)),
                        float(r.get("confidence", 0.0)),
                        x + float(r["x1"]),
                        y + float(r["y1"]),
                        x + float(r["x2"]),
                        y + float(r["y2"]),
                    ))
                count += 1
                if count % 5 == 0:
                    ws.log(f"ADAF Detection tiles processed: {count}; detections={len(dets)}")
                if x2 >= W:
                    break
            if y + tile >= H:
                break

        try:
            dets = ws.nms(dets, 0.45)
        except Exception:
            pass
        ws.dets = list(dets)
        try:
            ws.loadprev()
        except Exception:
            pass
        _refresh_combos(ws)
        _redraw(ws)
        ws.log(f"ADAF Detection finished: {len(dets)} detections")
    except Exception as exc:
        try:
            ws.show_error("ADAF Detection", str(exc))
        except Exception:
            _log("ADAF Detection failed: " + str(exc))
        traceback.print_exc()


def _run_adaf_satellite(ws):
    try:
        from PIL import Image
        g = ws.__class__.satellite_detect_selected.__globals__
        sat_tile_bounds_for_bbox = g.get("sat_tile_bounds_for_bbox")
        sat_lonlat_from_world_px = g.get("sat_lonlat_from_world_px")
        if sat_tile_bounds_for_bbox is None or sat_lonlat_from_world_px is None:
            raise RuntimeError("Satellite helper functions not found.")

        model_tar = str(getattr(ws, "mustatil_adaf_selected_model", "") or _best_auto_model(ws))
        device = str(getattr(ws, "mustatil_adaf_device", "cpu") or "cpu")
        min_lat, min_lon, max_lat, max_lon = ws._satellite_bbox()
        z = int(_get_var(ws.sat_zoom, 0))
        x_min, y_min, x_max, y_max = sat_tile_bounds_for_bbox(min_lat, min_lon, max_lat, max_lon, z)
        cols = x_max - x_min + 1
        rows = y_max - y_min + 1
        width = cols * WEB_TILE_SIZE
        height = rows * WEB_TILE_SIZE
        chunk = max(64, int(_get_var(ws.tile, 768) or 768))
        conf = max(0.001, min(1.0, float(_get_var(ws.conf, 0.25))))

        run_stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns()%1000000000:09d}"
        out_path = ws._satellite_unique_run_output_path(ws._satellite_output_gpkg_path(), run_stamp)
        temp_cache_root = ws._satellite_cache_dir() / "_detection_tmp" / f"adaf_z{z}_{run_stamp}_{os.getpid()}"
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
            ws.log(f"ADAF Satellite Detection started: z={z}, chunks={total}, model={Path(model_tar).name}, conf={conf}, device={device}")

            for job in jobs:
                cid, x, y, cw, ch = job
                meta = ws._satellite_build_chunk_to_cache(x_min, y_min, z, x, y, cw, ch, cid, temp_cache_root)
                chunk_path = Path(meta["chunk_path"])
                used = list(meta.get("tile_paths") or [])
                im = None
                try:
                    im = Image.open(chunk_path).convert("RGB")
                    local = _adaf_detect_pil(ws, im, conf, model_tar, device)
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
                            "class_name": str(r.get("class_name", r.get("label", "archaeological_feature"))),
                            "confidence": float(r.get("confidence", 0.0)),
                            "model": "ADAF " + Path(model_tar).stem,
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
                    ws.log(f"ADAF satellite chunk {cid}/{total}: detections={found}; total={len(records)}")
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
                    ws.log(f"ADAF Satellite GeoPackage written: {out_path}")
                except Exception as exc:
                    ws.log(f"ADAF satellite export warning: {exc}")
            else:
                ws.log("ADAF Satellite Detection finished: no detections.")

            try:
                ws._satellite_request_map_reload_from_worker(120)
            except Exception:
                pass
            _refresh_combos(ws)
            _redraw(ws)
            ws.log(f"ADAF Satellite Detection finished: {len(records)} detections")
        finally:
            shutil.rmtree(temp_cache_root, ignore_errors=True)
            ws.sat_detection_temp_cache_root = old_temp
            try:
                ws._sat_detection_thread_local.cache_root = old_thread
            except Exception:
                pass
    except Exception as exc:
        try:
            ws.show_error("ADAF Satellite Detection", str(exc))
        except Exception:
            _log("ADAF Satellite Detection failed: " + str(exc))
        traceback.print_exc()



# ============================================================================
# v4 simplified ADAF UI helpers: three official models + install + notebook WebView
# ============================================================================

ADAF_THREE_MODELS = [
    ("OD_barrow", "OD_barrow.tar", "https://zenodo.org/records/15848663/files/OD_barrow.tar?download=1"),
    ("OD_enclosure", "OD_enclosure.tar", "https://zenodo.org/records/15848663/files/OD_enclosure.tar?download=1"),
    ("OD_ringfort", "OD_ringfort.tar", "https://zenodo.org/records/15848663/files/OD_ringfort.tar?download=1"),
]


def _adaf_three_model_path(ws, filename: str) -> Path:
    return _model_dir(ws) / filename


def _adaf_fill_three_model_combo(combo, ws=None, include_custom=True):
    try:
        cur = combo.currentData()
    except Exception:
        cur = None
    combo.blockSignals(True)
    combo.clear()
    for label, filename, _url in ADAF_THREE_MODELS:
        p = _adaf_three_model_path(ws, filename)
        suffix = "" if p.exists() else "  (not installed yet)"
        combo.addItem(label + suffix, str(p))
    if include_custom:
        combo.addItem("Custom .tar…", "__CUSTOM__")
    if cur:
        ix = combo.findData(cur)
        if ix >= 0:
            combo.setCurrentIndex(ix)
    combo.blockSignals(False)


def _adaf_model_from_simple_combo(combo, ws=None) -> str:
    data = combo.currentData()
    if data == "__CUSTOM__":
        return str(getattr(ws, "mustatil_adaf_custom_model", "") or "")
    return str(data or "")


def _download_url_streaming(url: str, dest: Path, ws=None, label: str = ""):
    """Robust streaming downloader with progress logging."""
    import urllib.request
    import urllib.error

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    headers = {
        "User-Agent": "Mustatil-ADAF-Model-Installer/1.0",
        "Accept": "*/*",
    }
    req = urllib.request.Request(url, headers=headers)

    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass

    try:
        if ws is not None:
            ws.log("Downloading " + (label or dest.name) + " from Zenodo...")
    except Exception:
        pass

    with urllib.request.urlopen(req, timeout=120) as r:
        total = 0
        try:
            total = int(r.headers.get("Content-Length") or 0)
        except Exception:
            total = 0
        done = 0
        last_log = 0
        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if ws is not None and done - last_log >= 25 * 1024 * 1024:
                    last_log = done
                    try:
                        if total:
                            ws.log("%s: %.1f / %.1f MB" % (dest.name, done / 1048576, total / 1048576))
                        else:
                            ws.log("%s: %.1f MB" % (dest.name, done / 1048576))
                    except Exception:
                        pass

    if not tmp.exists() or tmp.stat().st_size <= 1024:
        raise RuntimeError("Downloaded file is missing or too small: " + str(tmp))

    if dest.exists():
        try:
            dest.unlink()
        except Exception:
            pass
    tmp.rename(dest)

    try:
        if ws is not None:
            ws.log("Downloaded: %s (%.1f MB)" % (dest, dest.stat().st_size / 1048576))
    except Exception:
        pass


def _adaf_download_three_models_direct(ws=None):
    """Download the three requested official ADAF OD models directly from Zenodo.

    v5 change:
    - no interactive Read-Host
    - no blocking BITS-only dependency
    - Python streaming download first
    - PowerShell fallback only if Python HTTP fails
    - verifies that all three .tar files exist and are non-empty
    """
    folder = _model_dir(ws)
    folder.mkdir(parents=True, exist_ok=True)

    try:
        if ws is not None:
            ws.log("ADAF install: downloading three official Zenodo models to " + str(folder))
    except Exception:
        pass

    errors = []
    for label, filename, url in ADAF_THREE_MODELS:
        dest = folder / filename
        if dest.exists() and dest.stat().st_size > 1024:
            try:
                if ws is not None:
                    ws.log("ADAF model already exists: " + str(dest))
            except Exception:
                pass
            continue
        try:
            _download_url_streaming(url, dest, ws=ws, label=label)
        except Exception as exc:
            errors.append("%s: %s" % (filename, exc))

    missing_after_python = [
        filename for _label, filename, _url in ADAF_THREE_MODELS
        if not ((_model_dir(ws) / filename).exists() and (_model_dir(ws) / filename).stat().st_size > 1024)
    ]

    if missing_after_python and os.name == "nt":
        try:
            if ws is not None:
                ws.log("Python download did not finish all files; trying non-interactive PowerShell fallback.")
        except Exception:
            pass
        ps1 = _write_adaf_download_script(ws)
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1)],
            cwd=str(folder),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            if ws is not None:
                for line in (proc.stdout or "").splitlines()[-80:]:
                    ws.log(line)
        except Exception:
            pass
        if proc.returncode != 0:
            errors.append("PowerShell fallback exit code %s: %s" % (proc.returncode, "\n".join((proc.stdout or "").splitlines()[-40:])))

    missing = [
        filename for _label, filename, _url in ADAF_THREE_MODELS
        if not ((_model_dir(ws) / filename).exists() and (_model_dir(ws) / filename).stat().st_size > 1024)
    ]
    if missing:
        raise RuntimeError(
            "ADAF model download incomplete. Missing: "
            + ", ".join(missing)
            + "\nTarget folder: "
            + str(_model_dir(ws))
            + ("\nErrors:\n" + "\n".join(errors[-10:]) if errors else "")
        )

    try:
        if ws is not None:
            ws.log("All three ADAF models are installed in: " + str(folder))
    except Exception:
        pass


def _install_adaf_repo_and_three_models(ws=None):
    """One install button: clone/update ADAF GitHub repo and download the three OD models."""
    repo = _repo_dir(ws)
    model_dir = _model_dir(ws)
    repo.parent.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    if (repo / ".git").exists():
        _run_cmd_logged(ws, ["git", "pull"], cwd=repo, title="ADAF update")
    elif repo.exists() and any(repo.iterdir()):
        try:
            if ws is not None:
                ws.log("ADAF repo folder already exists and is not empty: " + str(repo))
        except Exception:
            pass
    else:
        _run_cmd_logged(ws, ["git", "clone", ADAF_GITHUB_URL, str(repo)], cwd=repo.parent, title="ADAF clone")

    _adaf_download_three_models_direct(ws)

    missing = []
    for _label, filename, _url in ADAF_THREE_MODELS:
        if not (_model_dir(ws) / filename).exists():
            missing.append(filename)
    if missing:
        raise RuntimeError("ADAF install finished but these models are still missing: " + ", ".join(missing))

    try:
        if ws is not None:
            ws.log("ADAF install complete. Models ready in: " + str(_model_dir(ws)))
    except Exception:
        pass


def _find_adaf_notebook(repo: Path) -> Path:
    candidates = [
        repo / "ADAF_widget.ipynb",
        repo / "ADAF_main.ipynb",
        repo / "ADAF_training.ipynb",
    ]
    try:
        candidates += list(repo.glob("*training*.ipynb"))
        candidates += list(repo.glob("*Training*.ipynb"))
        candidates += list(repo.rglob("*object*detection*train*.ipynb"))
        candidates += list(repo.rglob("*semantic*segmentation*train*.ipynb"))
        candidates += list(repo.rglob("*.ipynb"))
    except Exception:
        pass
    for p in candidates:
        try:
            if p.exists() and p.suffix.lower() == ".ipynb":
                return p
        except Exception:
            pass
    return repo / "ADAF_main.ipynb"


def _open_adaf_notebook_webview(ws=None):
    """Open ADAF notebook in a new Qt WebView window. Falls back to browser if QtWebEngine is missing."""
    try:
        from PySide6.QtCore import QTimer, QUrl, QProcess
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView
            has_web = True
        except Exception:
            QWebEngineView = None
            has_web = False
    except Exception:
        webbrowser.open("http://127.0.0.1:8899/lab")
        return

    repo = _repo_dir(ws)
    conda = _conda_exe(ws)
    env = _adaf_env(ws)
    notebook = _find_adaf_notebook(repo)
    port = int(getattr(ws, "mustatil_adaf_jupyter_port", 8899) or 8899)
    url = "http://127.0.0.1:%d/lab/tree/%s" % (port, notebook.name)

    if not conda.exists():
        try:
            if ws is not None:
                ws.log("conda.exe not found for ADAF notebook: " + str(conda))
        except Exception:
            pass
        return
    if not repo.exists():
        try:
            if ws is not None:
                ws.log("ADAF repo not found. Press Install ADAF + Models first: " + str(repo))
        except Exception:
            pass
        return

    launcher = repo / "mustatil_start_adaf_notebook_webview.py"
    launcher_lines = [
        "# Auto-created by Mustatil ADAF Trainer",
        "import os, sys, ssl",
        "try:",
        "    import certifi",
        "    _orig = ssl.create_default_context",
        "    def _safe(purpose=ssl.Purpose.SERVER_AUTH, *, cafile=None, capath=None, cadata=None):",
        "        if cafile is None and capath is None and cadata is None:",
        "            cafile = certifi.where()",
        "        return _orig(purpose=purpose, cafile=cafile, capath=capath, cadata=cadata)",
        "    ssl.create_default_context = _safe",
        "except Exception:",
        "    pass",
        "os.chdir(r" + repr(str(repo)) + ")",
        "sys.argv = [",
        '    "jupyter-lab",',
        '    "--no-browser",',
        '    "--ip=127.0.0.1",',
        '    "--port=%d",' % port,
        '    "--ServerApp.open_browser=False",',
        '    "--ServerApp.token=",',
        '    "--ServerApp.password=",',
        '    "--ServerApp.disable_check_xsrf=True",',
        "    r" + repr(str(notebook)) + ",",
        "]",
        "print(" + repr("MUSTATIL_ADAF_NOTEBOOK_URL=" + url) + ")",
        "from jupyterlab.labapp import main",
        "main()",
        "",
    ]
    try:
        launcher.write_text("\n".join(launcher_lines), encoding="utf-8")
    except Exception:
        pass

    proc = getattr(ws, "_mustatil_adaf_jupyter_process", None)
    try:
        running = proc is not None and proc.state() != QProcess.NotRunning
    except Exception:
        running = False

    if not running:
        proc = QProcess()
        proc.setWorkingDirectory(str(repo))
        proc.setProgram(str(conda))
        proc.setArguments(["run", "--live-stream", "-n", env, "python", str(launcher)])
        proc.setProcessChannelMode(QProcess.MergedChannels)

        def read_out():
            try:
                data = bytes(proc.readAllStandardOutput()).decode(errors="replace")
                if ws is not None:
                    for line in data.splitlines()[-20:]:
                        ws.log(line)
            except Exception:
                pass

        proc.readyReadStandardOutput.connect(read_out)
        proc.start()
        try:
            ws._mustatil_adaf_jupyter_process = proc
        except Exception:
            pass

    if has_web:
        dlg = QDialog()
        dlg.setWindowTitle("ADAF Notebook")
        dlg.resize(1300, 850)
        lay = QVBoxLayout(dlg)
        lbl = QLabel("ADAF notebook running locally: " + url)
        lbl.setWordWrap(True)
        lay.addWidget(lbl)
        view = QWebEngineView()
        lay.addWidget(view, 1)

        buttons = QHBoxLayout()
        open_browser_btn = QPushButton("Open in Browser")
        reload_btn = QPushButton("Reload")
        close_btn = QPushButton("Close Window")
        buttons.addWidget(open_browser_btn)
        buttons.addWidget(reload_btn)
        buttons.addStretch(1)
        buttons.addWidget(close_btn)
        lay.addLayout(buttons)

        open_browser_btn.clicked.connect(lambda: webbrowser.open(url))
        reload_btn.clicked.connect(lambda: view.setUrl(QUrl(url)))
        close_btn.clicked.connect(dlg.close)

        try:
            if not hasattr(ws, "_mustatil_adaf_webview_windows"):
                ws._mustatil_adaf_webview_windows = []
            ws._mustatil_adaf_webview_windows.append(dlg)
        except Exception:
            pass

        QTimer.singleShot(4500, lambda: view.setUrl(QUrl(url)))
        dlg.show()
    else:
        try:
            if ws is not None:
                ws.log("QtWebEngine not available. Opening ADAF notebook in browser.")
        except Exception:
            pass
        QTimer.singleShot(4500, lambda: webbrowser.open(url))


def _make_filter_box(ws, title="Class filter / Geo NMS (ADAF)"):
    from PySide6.QtWidgets import QGroupBox, QGridLayout, QLabel, QComboBox, QCheckBox, QDoubleSpinBox

    box = QGroupBox(title)
    g = QGridLayout(box)
    combo = QComboBox()
    combo.addItem("All classes", None)
    combo.addItem("Class 0: archaeology / ADAF", 0)
    combo.addItem("Class 1: false_positive", 1)
    hide_fp = QCheckBox("Hide false_positive")
    nms_enabled = QCheckBox("Geo NMS")
    nms_iou = QDoubleSpinBox()
    nms_iou.setRange(0.05, 0.95)
    nms_iou.setSingleStep(0.05)
    nms_iou.setValue(float(getattr(ws, "mustatil_owl_geo_nms_iou", 0.35) or 0.35))

    try:
        ws.mustatil_owl_class_combos = list(getattr(ws, "mustatil_owl_class_combos", []) or []) + [combo]
    except Exception:
        pass

    def sync():
        try:
            ws.mustatil_owl_selected_class_id = combo.currentData()
            ws.mustatil_owl_hide_fp = bool(hide_fp.isChecked())
            ws.mustatil_owl_geo_nms_enabled = bool(nms_enabled.isChecked())
            ws.mustatil_owl_geo_nms_iou = float(nms_iou.value())
        except Exception:
            pass
        _redraw(ws)

    combo.currentIndexChanged.connect(lambda *_: sync())
    hide_fp.toggled.connect(lambda *_: sync())
    nms_enabled.toggled.connect(lambda *_: sync())
    nms_iou.valueChanged.connect(lambda *_: sync())

    g.addWidget(QLabel("Class"), 0, 0)
    g.addWidget(combo, 0, 1)
    g.addWidget(hide_fp, 1, 0, 1, 2)
    g.addWidget(nms_enabled, 2, 0)
    g.addWidget(nms_iou, 2, 1)
    return box


def _build_adaf_left_controls(ws, tab_kind: str):
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QGroupBox, QLabel, QPushButton, QComboBox, QFileDialog

    page = QWidget()
    page.setObjectName("MustatilADAFLeftControls")
    root = QVBoxLayout(page)
    root.setContentsMargins(8, 8, 8, 8)
    root.setSpacing(8)

    title = QLabel("ADAF")
    title.setStyleSheet("font-size: 16px; font-weight: bold;")
    root.addWidget(title)

    setup_box = QGroupBox("ADAF setup")
    sg = QGridLayout(setup_box)
    install_btn = QPushButton("Install ADAF + Models")
    install_btn.setMinimumHeight(42)
    install_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 8px; }")
    open_notebook_btn = QPushButton("Open ADAF Notebook")
    open_notebook_btn.setMinimumHeight(42)
    open_notebook_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 8px; }")
    setup_status = QLabel("Install clones/updates ADAF and downloads OD_barrow, OD_enclosure, OD_ringfort. Open ADAF Notebook opens the official notebook in a new WebView window.")
    setup_status.setWordWrap(True)
    sg.addWidget(install_btn, 0, 0)
    sg.addWidget(open_notebook_btn, 0, 1)
    sg.addWidget(setup_status, 1, 0, 1, 2)
    root.addWidget(setup_box)

    model_box = QGroupBox("Detection model")
    g = QGridLayout(model_box)
    model_combo = QComboBox()
    _adaf_fill_three_model_combo(model_combo, ws)
    custom_btn = QPushButton("Choose custom .tar…")
    run_btn = QPushButton("Run ADAF on Detection image" if tab_kind == "detection" else "Run ADAF on selected satellite map")
    run_btn.setMinimumHeight(36)
    status = QLabel("Select OD_barrow / OD_enclosure / OD_ringfort or Custom.")
    status.setWordWrap(True)

    g.addWidget(QLabel("ADAF model"), 0, 0)
    g.addWidget(model_combo, 0, 1)
    g.addWidget(custom_btn, 1, 0, 1, 2)
    g.addWidget(run_btn, 2, 0, 1, 2)
    g.addWidget(status, 3, 0, 1, 2)
    root.addWidget(model_box)

    root.addWidget(_make_filter_box(ws, "Class filter / Geo NMS (ADAF)"))
    root.addStretch(1)

    def append(msg):
        try:
            status.setText(str(msg))
        except Exception:
            pass
        try:
            ws.log(str(msg))
        except Exception:
            pass

    def setup_append(msg):
        try:
            setup_status.setText(str(msg))
        except Exception:
            pass
        try:
            ws.log(str(msg))
        except Exception:
            pass

    def sync():
        try:
            ws.mustatil_adaf_repo = str(_repo_dir(ws))
            ws.mustatil_adaf_model_dir = str(_model_dir(ws))
            ws.mustatil_adaf_conda = str(_conda_exe(ws))
            ws.mustatil_adaf_env = DEFAULT_ADAF_ENV
            ws.mustatil_adaf_device = "cpu"
            ws.mustatil_adaf_selected_model = _adaf_model_from_simple_combo(model_combo, ws)
        except Exception:
            pass

    def choose_custom():
        p, _ = QFileDialog.getOpenFileName(page, "Choose custom ADAF .tar model", str(_model_dir(ws)), "ADAF model TAR (*.tar);;All files (*)")
        if p:
            try:
                ws.mustatil_adaf_custom_model = p
                model_combo.blockSignals(True)
                ix = model_combo.findData("__CUSTOM__")
                if ix < 0:
                    model_combo.addItem("Custom .tar…", "__CUSTOM__")
                    ix = model_combo.count() - 1
                model_combo.setCurrentIndex(ix)
                model_combo.blockSignals(False)
                append("Custom ADAF model selected: " + p)
            except Exception:
                pass
            sync()

    def install_clicked():
        def task():
            _install_adaf_repo_and_three_models(ws)
            _adaf_fill_three_model_combo(model_combo, ws)
            sync()
            setup_append("ADAF installed. Three models are ready.")
        _run_as_task(ws, "Install ADAF + Models", task, allow_parallel=False)

    def open_notebook_clicked():
        setup_append("Opening ADAF Notebook in WebView...")
        _open_adaf_notebook_webview(ws)

    def run_clicked():
        sync()
        if not getattr(ws, "mustatil_adaf_selected_model", ""):
            append("No ADAF model selected. Choose OD_barrow / OD_enclosure / OD_ringfort or Custom.")
            return
        if not Path(ws.mustatil_adaf_selected_model).exists():
            append("Model file not found. Press Install ADAF + Models or choose Custom .tar.")
            return
        if tab_kind == "detection":
            _run_as_task(ws, "ADAF Detection", lambda: _run_adaf_detection_image(ws), allow_parallel=False)
        else:
            _run_as_task(ws, "ADAF Satellite Detection", lambda: _run_adaf_satellite(ws), allow_parallel=True)

    install_btn.clicked.connect(install_clicked)
    open_notebook_btn.clicked.connect(open_notebook_clicked)
    model_combo.currentIndexChanged.connect(lambda *_: sync())
    custom_btn.clicked.connect(choose_custom)
    run_btn.clicked.connect(run_clicked)

    sync()
    return page


ADAF_TRAINER_WORKER_CODE = "# mustatil_adaf_trainer_worker.py\n# Trains ADAF-style object-detection models from existing Mustatil/YOLO projects.\n# Output is a .tar that the Mustatil ADAF Detection tab can list and try to run.\n\nfrom __future__ import annotations\nimport os, sys, json, time, tarfile, random, shutil, traceback, argparse\nfrom pathlib import Path\n\nIMG_EXTS = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}\n\n\ndef parse_classes(raw):\n    vals=[p.strip() for p in str(raw or '').replace(';',',').split(',') if p.strip()]\n    return vals or ['archaeological_feature']\n\n\ndef find_images(root):\n    root=Path(root)\n    out=[]\n    for p in root.rglob('*'):\n        if p.is_file() and p.suffix.lower() in IMG_EXTS:\n            # avoid cache/output junk\n            low=str(p).replace('\\\\','/').lower()\n            if any(x in low for x in ['/runs/', '/outputs/', '/output/', '/.venv/', '/venv/', '/__pycache__/']):\n                continue\n            out.append(p)\n    return sorted(out)\n\n\ndef find_label_for_image(img, root):\n    root=Path(root)\n    stem=img.stem\n    candidates=[]\n    # Common YOLO layout: images/train/a.jpg -> labels/train/a.txt\n    parts=list(img.parts)\n    try:\n        if 'images' in parts:\n            idx=parts.index('images')\n            rel=Path(*parts[idx+1:]).with_suffix('.txt')\n            candidates.append(root/'labels'/rel)\n    except Exception:\n        pass\n    # Same folder / sibling labels dirs\n    candidates += [img.with_suffix('.txt'), root/'labels'/(stem+'.txt'), root/'labels'/'train'/(stem+'.txt'), root/'labels'/'val'/(stem+'.txt')]\n    for p in candidates:\n        if p.exists(): return p\n    # Slow fallback by stem\n    try:\n        for p in (root/'labels').rglob(stem+'.txt'):\n            return p\n    except Exception:\n        pass\n    return None\n\n\ndef parse_yolo_label(label_path, w, h, class_filter=None):\n    boxes=[]; labels=[]\n    if not label_path or not Path(label_path).exists():\n        return boxes, labels\n    class_filter=set(class_filter or [])\n    for line in Path(label_path).read_text(encoding='utf-8', errors='ignore').splitlines():\n        line=line.strip()\n        if not line or line.startswith('#'): continue\n        parts=line.replace(',', ' ').split()\n        if len(parts) < 5: continue\n        try:\n            cls=int(float(parts[0])); xc=float(parts[1]); yc=float(parts[2]); bw=float(parts[3]); bh=float(parts[4])\n            if class_filter and cls not in class_filter:\n                continue\n            x1=(xc-bw/2)*w; y1=(yc-bh/2)*h; x2=(xc+bw/2)*w; y2=(yc+bh/2)*h\n            x1=max(0,min(w-1,x1)); y1=max(0,min(h-1,y1)); x2=max(0,min(w,x2)); y2=max(0,min(h,y2))\n            if x2>x1+1 and y2>y1+1:\n                boxes.append([x1,y1,x2,y2]); labels.append(cls+1)  # +1 for FasterRCNN background\n        except Exception:\n            continue\n    return boxes, labels\n\n\ndef build_manifest(project, classes, class_filter=None):\n    from PIL import Image\n    project=Path(project)\n    imgs=find_images(project)\n    items=[]\n    for img in imgs:\n        lab=find_label_for_image(img, project)\n        if lab is None: continue\n        try:\n            with Image.open(img) as im:\n                w,h=im.size\n            boxes, labels=parse_yolo_label(lab,w,h,class_filter=class_filter)\n            if boxes:\n                items.append({'image':str(img), 'label':str(lab), 'width':w, 'height':h, 'boxes':boxes, 'labels':labels})\n        except Exception:\n            continue\n    return {'project':str(project), 'classes':classes, 'items':items, 'created':time.strftime('%Y-%m-%d %H:%M:%S')}\n\n\ndef train_object_detection(project, out_dir, output_name, classes, epochs=10, batch=1, imgsz=768, device='cpu', class_filter=None, pretrained=True):\n    import torch\n    from torch.utils.data import Dataset, DataLoader\n    from PIL import Image\n    import torchvision.transforms.functional as F\n    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn\n    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor\n\n    out_dir=Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)\n    manifest=build_manifest(project, classes, class_filter=class_filter)\n    items=manifest['items']\n    if not items:\n        raise RuntimeError('No YOLO labels with boxes found in project/dataset folder. Expected images plus matching labels/*.txt files.')\n    random.seed(42); random.shuffle(items)\n    split=max(1, int(len(items)*0.9))\n    train_items=items[:split]\n    val_items=items[split:] or items[:1]\n\n    class YoloBoxDataset(Dataset):\n        def __init__(self, data): self.data=data\n        def __len__(self): return len(self.data)\n        def __getitem__(self, idx):\n            it=self.data[idx]\n            im=Image.open(it['image']).convert('RGB')\n            w,h=im.size\n            boxes=torch.tensor(it['boxes'], dtype=torch.float32)\n            labels=torch.tensor(it['labels'], dtype=torch.int64)\n            # Resize to max side imgsz to reduce memory but keep aspect ratio\n            scale=1.0\n            if imgsz and max(w,h) > int(imgsz):\n                scale=float(imgsz)/float(max(w,h))\n                nw=max(1,int(round(w*scale))); nh=max(1,int(round(h*scale)))\n                im=im.resize((nw,nh))\n                boxes=boxes*scale\n            image=F.to_tensor(im)\n            target={'boxes':boxes, 'labels':labels, 'image_id':torch.tensor([idx]), 'area':(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]), 'iscrowd':torch.zeros((boxes.shape[0],), dtype=torch.int64)}\n            return image, target\n\n    def collate(batch): return tuple(zip(*batch))\n\n    num_classes=max(len(classes)+1, max([max(it['labels']) for it in items if it['labels']] or [1])+1)\n    weights='DEFAULT' if pretrained else None\n    model=fasterrcnn_mobilenet_v3_large_fpn(weights=weights)\n    in_features=model.roi_heads.box_predictor.cls_score.in_features\n    model.roi_heads.box_predictor=FastRCNNPredictor(in_features, num_classes)\n    device=torch.device('cuda' if str(device).lower().startswith('cuda') and torch.cuda.is_available() else 'cpu')\n    model.to(device)\n\n    loader=DataLoader(YoloBoxDataset(train_items), batch_size=max(1,int(batch)), shuffle=True, num_workers=0, collate_fn=collate)\n    params=[p for p in model.parameters() if p.requires_grad]\n    optimizer=torch.optim.AdamW(params, lr=0.0002, weight_decay=0.0001)\n\n    log=[]\n    for ep in range(1, int(epochs)+1):\n        model.train(); total=0.0; n=0\n        for images, targets in loader:\n            images=[im.to(device) for im in images]\n            targets=[{k:v.to(device) for k,v in t.items()} for t in targets]\n            losses=model(images, targets)\n            loss=sum(v for v in losses.values())\n            optimizer.zero_grad(set_to_none=True)\n            loss.backward()\n            optimizer.step()\n            total += float(loss.detach().cpu().item()); n += 1\n        msg=f'epoch {ep}/{epochs} loss={total/max(1,n):.5f}'\n        print(msg, flush=True); log.append(msg)\n\n    model.cpu(); model.eval()\n    stamp=time.strftime('%Y%m%d_%H%M%S')\n    safe=''.join(c if c.isalnum() or c in '-_.' else '_' for c in (output_name or 'ADAF_custom_OD'))\n    work=out_dir/(safe+'_'+stamp+'_work'); work.mkdir(parents=True, exist_ok=True)\n    model_pt=work/'model.pt'\n    state_pth=work/'checkpoint_state_dict.pth'\n    meta={\n        'format':'mustatil_adaf_trained_tar', 'task':'object_detection', 'architecture':'torchvision.fasterrcnn_mobilenet_v3_large_fpn',\n        'classes':classes, 'num_classes':num_classes, 'created':time.strftime('%Y-%m-%d %H:%M:%S'),\n        'project':str(project), 'epochs':int(epochs), 'batch':int(batch), 'imgsz':int(imgsz),\n        'train_images':len(train_items), 'val_images':len(val_items), 'labels_are_1_based_for_fasterrcnn':True,\n        'log':log,\n    }\n    torch.save(model, model_pt)\n    torch.save({'state_dict':model.state_dict(), 'metadata':meta, 'classes':classes}, state_pth)\n    (work/'metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')\n    (work/'dataset_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')\n    tar_path=out_dir/(safe+'_'+stamp+'.tar')\n    with tarfile.open(tar_path, 'w') as tf:\n        for p in [model_pt, state_pth, work/'metadata.json', work/'dataset_manifest.json']:\n            tf.add(p, arcname=p.name)\n    return {'status':'ok', 'tar':str(tar_path), 'metadata':meta, 'work_dir':str(work)}\n\n\ndef detect_masks(project):\n    project=Path(project)\n    masks=[]\n    for name in ['masks','mask','segmentation','segmentations','labels_masks']:\n        d=project/name\n        if d.exists():\n            for p in d.rglob('*'):\n                if p.is_file() and p.suffix.lower() in IMG_EXTS:\n                    masks.append(str(p))\n    return masks\n\n\ndef main():\n    ap=argparse.ArgumentParser()\n    ap.add_argument('--project', required=True)\n    ap.add_argument('--out-dir', required=True)\n    ap.add_argument('--model-kind', default='od_ao')\n    ap.add_argument('--classes', default='archaeological_feature')\n    ap.add_argument('--epochs', type=int, default=10)\n    ap.add_argument('--batch', type=int, default=1)\n    ap.add_argument('--imgsz', type=int, default=768)\n    ap.add_argument('--device', default='cpu')\n    ap.add_argument('--output-name', default='ADAF_custom_OD')\n    args=ap.parse_args()\n    report={'created':time.strftime('%Y-%m-%d %H:%M:%S'), 'args':vars(args), 'status':'started'}\n    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)\n    try:\n        classes=parse_classes(args.classes)\n        kind=str(args.model_kind).lower()\n        if kind.startswith('seg') or 'hrnet' in kind or 'semantic' in kind:\n            masks=detect_masks(args.project)\n            report['status']='segmentation_dataset_prepared_only'\n            report['masks_found']=len(masks)\n            report['note']='Semantic/HRNet training needs pixel masks. This plugin detects mask datasets and prepares the job, but object-detection training is the implemented direct trainer for Mustatil Detection/Satellite.'\n            print(report['note'], flush=True)\n        else:\n            report.update(train_object_detection(args.project, out_dir, args.output_name, classes, args.epochs, args.batch, args.imgsz, args.device))\n            report['status']='ok'\n    except Exception:\n        report['status']='error'; report['traceback']=traceback.format_exc(); print(report['traceback'], flush=True)\n    rp=out_dir/('adaf_training_report_'+str(time.time_ns())+'.json')\n    rp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')\n    print('MUSTATIL_ADAF_TRAIN_REPORT='+str(rp), flush=True)\n    if report.get('tar'):\n        print('MUSTATIL_ADAF_TRAINED_MODEL='+str(report['tar']), flush=True)\n    if report.get('status')=='error':\n        sys.exit(1)\n\nif __name__ == '__main__':\n    main()\n"


def _write_adaf_trainer_worker(ws=None) -> Path:
    repo = _repo_dir(ws)
    repo.mkdir(parents=True, exist_ok=True)
    p = repo / "mustatil_adaf_trainer_worker.py"
    p.write_text(ADAF_TRAINER_WORKER_CODE, encoding="utf-8")
    return p


def _guess_current_project_folder(ws=None) -> str:
    candidates = []
    for name in ("project_dir", "project_folder", "project_path", "mustatil_project_dir", "current_project_dir", "last_project_dir"):
        try:
            v = getattr(ws, name, "")
            if hasattr(v, "get"):
                v = v.get()
            if v:
                candidates.append(str(v))
        except Exception:
            pass
    try:
        img = str(_get_var(getattr(ws, "image", None), "") or "").strip().strip('"')
        if img:
            p = Path(img)
            candidates += [str(p.parent), str(p.parent.parent)]
    except Exception:
        pass
    for c in candidates:
        try:
            p = Path(c).expanduser()
            if (p / "labels").exists() or (p / "images").exists() or (p / "project.json").exists() or (p / "mustatil_project.json").exists():
                return str(p)
        except Exception:
            pass
    return candidates[0] if candidates else ""


def _adaf_trained_output_dir(ws=None) -> Path:
    return _model_dir(ws) / "trained"


def _run_adaf_trainer_worker(ws, project: str, model_kind: str, classes: str, epochs: int, batch: int, imgsz: int, device: str, output_name: str):
    conda = _conda_exe(ws)
    repo = _repo_dir(ws)
    env = _adaf_env(ws)
    worker = _write_adaf_trainer_worker(ws)
    out_dir = _adaf_trained_output_dir(ws)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not conda.exists():
        raise RuntimeError(f"conda.exe not found: {conda}")
    if not project or not Path(project).exists():
        raise RuntimeError(f"Project/dataset folder not found: {project}")
    cmd = [
        str(conda), "run", "--live-stream", "-n", env, "python", str(worker),
        "--project", str(project),
        "--out-dir", str(out_dir),
        "--model-kind", str(model_kind),
        "--classes", str(classes),
        "--epochs", str(int(epochs)),
        "--batch", str(int(batch)),
        "--imgsz", str(int(imgsz)),
        "--device", str(device),
        "--output-name", str(output_name or "ADAF_custom_OD"),
    ]
    _run_cmd_logged(ws, cmd, cwd=repo if repo.exists() else None, title="ADAF trainer")
    try:
        ws.log("ADAF training output folder: " + str(out_dir))
    except Exception:
        pass


def _open_adaf_training_notebooks(ws=None):
    repo = _repo_dir(ws)
    names = ["train_and_evaluate_object_detection.ipynb", "train_and_evaluate_semantic_segmentation.ipynb", "ADAF_main.ipynb"]
    for n in names:
        p = repo / n
        if p.exists():
            try:
                os.startfile(str(p))
            except Exception:
                webbrowser.open(str(p))
            return
    try:
        os.startfile(str(repo))
    except Exception:
        webbrowser.open(str(repo))



def _build_adaf_models_training_tab(ws):
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QGridLayout, QGroupBox, QLabel, QPushButton,
        QComboBox, QFileDialog, QLineEdit, QSpinBox
    )

    page = QWidget()
    page.setObjectName("MustatilADAFTrainerTab")
    root = QVBoxLayout(page)
    root.setContentsMargins(8, 8, 8, 8)
    root.setSpacing(8)

    title = QLabel("ADAF Trainer")
    title.setStyleSheet("font-size: 18px; font-weight: bold;")
    root.addWidget(title)

    top = QGroupBox("ADAF setup")
    tg = QGridLayout(top)
    install_btn = QPushButton("Install ADAF + Models")
    install_btn.setMinimumHeight(44)
    install_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 8px; }")
    open_adaf_btn = QPushButton("Open ADAF Notebook")
    open_adaf_btn.setMinimumHeight(44)
    open_adaf_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 8px; }")
    setup_status = QLabel("Install clones/updates ADAF and downloads OD_barrow, OD_enclosure, OD_ringfort. Open ADAF starts the notebook in a new WebView window. Simple training creates custom Faster R-CNN .tar models from YOLO/Mustatil boxes.")
    setup_status.setWordWrap(True)
    tg.addWidget(install_btn, 0, 0)
    tg.addWidget(open_adaf_btn, 0, 1)
    tg.addWidget(setup_status, 1, 0, 1, 2)
    root.addWidget(top)

    def append(msg):
        try:
            setup_status.setText(str(msg))
        except Exception:
            pass
        try:
            ws.log(str(msg))
        except Exception:
            pass

    train_box = QGroupBox("Simple ADAF training from existing Mustatil / YOLO project")
    g = QGridLayout(train_box)

    project_edit = QLineEdit(_guess_current_project_folder(ws))
    kind_combo = QComboBox()
    kind_combo.addItems(["OD_barrow", "OD_enclosure", "OD_ringfort", "Custom OD"])
    out_name_edit = QLineEdit("ADAF_custom_OD")
    class_edit = QLineEdit("archaeological_feature,barrow,ringfort,enclosure")
    epochs_spin = QSpinBox()
    epochs_spin.setRange(1, 1000)
    epochs_spin.setValue(20)
    batch_spin = QSpinBox()
    batch_spin.setRange(1, 32)
    batch_spin.setValue(1)
    imgsz_spin = QSpinBox()
    imgsz_spin.setRange(128, 4096)
    imgsz_spin.setSingleStep(64)
    imgsz_spin.setValue(768)
    device_combo = QComboBox()
    device_combo.setEditable(True)
    device_combo.addItems(["cpu", "cuda"])
    train_status = QLabel("Training idle.")
    train_status.setWordWrap(True)

    def train_append(msg):
        try:
            train_status.setText(str(msg))
        except Exception:
            pass
        try:
            ws.log(str(msg))
        except Exception:
            pass

    def choose_project():
        p = QFileDialog.getExistingDirectory(page, "Select Mustatil project / YOLO dataset folder", project_edit.text() or str(Path.home()))
        if p:
            project_edit.setText(p)

    def open_training_output():
        out = _adaf_trained_output_dir(ws)
        out.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(out))
        except Exception:
            webbrowser.open(str(out))

    def train_job(all_three=False):
        project = project_edit.text().strip().strip('"')
        classes = class_edit.text().strip()
        epochs = int(epochs_spin.value())
        batch = int(batch_spin.value())
        imgsz = int(imgsz_spin.value())
        device = device_combo.currentText().strip() or "cpu"

        if all_three:
            jobs = [
                ("od_barrow", "ADAF_custom_OD_barrow"),
                ("od_enclosure", "ADAF_custom_OD_enclosure"),
                ("od_ringfort", "ADAF_custom_OD_ringfort"),
            ]
        else:
            text = kind_combo.currentText().lower()
            if "barrow" in text:
                jobs = [("od_barrow", out_name_edit.text().strip() or "ADAF_custom_OD_barrow")]
            elif "enclosure" in text:
                jobs = [("od_enclosure", out_name_edit.text().strip() or "ADAF_custom_OD_enclosure")]
            elif "ringfort" in text:
                jobs = [("od_ringfort", out_name_edit.text().strip() or "ADAF_custom_OD_ringfort")]
            else:
                jobs = [("od_ao", out_name_edit.text().strip() or "ADAF_custom_OD")]

        def task():
            for mk, name in jobs:
                train_append("Starting ADAF training: " + name)
                _run_adaf_trainer_worker(ws, project, mk, classes, epochs, batch, imgsz, device, name)
            train_append("Training finished. Output: " + str(_adaf_trained_output_dir(ws)))
        _run_as_task(ws, "ADAF Trainer", task, allow_parallel=False)

    r = 0
    g.addWidget(QLabel("Project / dataset"), r, 0)
    g.addWidget(project_edit, r, 1, 1, 3)
    browse_btn = QPushButton("…")
    browse_btn.clicked.connect(choose_project)
    g.addWidget(browse_btn, r, 4)
    r += 1

    g.addWidget(QLabel("Model type"), r, 0)
    g.addWidget(kind_combo, r, 1, 1, 4)
    r += 1

    g.addWidget(QLabel("Output name"), r, 0)
    g.addWidget(out_name_edit, r, 1, 1, 4)
    r += 1

    g.addWidget(QLabel("Classes"), r, 0)
    g.addWidget(class_edit, r, 1, 1, 4)
    r += 1

    g.addWidget(QLabel("Epochs"), r, 0)
    g.addWidget(epochs_spin, r, 1)
    g.addWidget(QLabel("Batch"), r, 2)
    g.addWidget(batch_spin, r, 3)
    r += 1

    g.addWidget(QLabel("Image size"), r, 0)
    g.addWidget(imgsz_spin, r, 1)
    g.addWidget(QLabel("Device"), r, 2)
    g.addWidget(device_combo, r, 3)
    r += 1

    train_btn = QPushButton("Train selected")
    train_all_btn = QPushButton("Train all 3")
    out_btn = QPushButton("Open trained output")
    g.addWidget(train_btn, r, 0, 1, 2)
    g.addWidget(train_all_btn, r, 2, 1, 2)
    g.addWidget(out_btn, r, 4)
    r += 1

    g.addWidget(train_status, r, 0, 1, 5)

    root.addWidget(train_box)
    root.addStretch(1)

    def install_clicked():
        def task():
            _install_adaf_repo_and_three_models(ws)
            append("ADAF installed. Three OD models are ready.")
        _run_as_task(ws, "Install ADAF + Models", task, allow_parallel=False)

    install_btn.clicked.connect(install_clicked)
    open_adaf_btn.clicked.connect(lambda: _open_adaf_notebook_webview(ws))
    train_btn.clicked.connect(lambda: train_job(False))
    train_all_btn.clicked.connect(lambda: train_job(True))
    out_btn.clicked.connect(open_training_output)

    return page


def _tab_texts(tw):
    try:
        return [str(tw.tabText(i) or "") for i in range(tw.count())]
    except Exception:
        return []


def _has_tab_label(tw, *needles):
    lows = [t.lower() for t in _tab_texts(tw)]
    for low in lows:
        for n in needles:
            if n.lower() in low:
                return True
    return False


def _inner_tab_kind_from_widget(tw):
    texts = [t.lower() for t in _tab_texts(tw)]
    if not texts:
        return ""
    has_lae = any("lae" in t and "dino" in t for t in texts)
    has_model = any("owl" in t or "grounding" in t or "yolo" in t for t in texts)
    if not (has_lae and has_model):
        return ""
    cur = tw
    parent_text = " ".join(texts)
    try:
        for _ in range(40):
            if cur is None:
                break
            parent_text += " " + str(cur.objectName() or "")
            cur = cur.parentWidget()
    except Exception:
        pass
    low = parent_text.lower()
    if "sat" in low:
        return "satellite"
    return "detection"


def _ensure_adaf_inner_tab(tw):
    try:
        if id(tw) in _PATCHED_INNER_TABS and _has_tab_label(tw, "adaf"):
            return True
        texts = _tab_texts(tw)
        lae_idx = -1
        for i, t in enumerate(texts):
            low = t.lower()
            if "lae" in low and "dino" in low:
                lae_idx = i
                break
        if lae_idx < 0:
            return False
        if not (_has_tab_label(tw, "yolo") or _has_tab_label(tw, "owl") or _has_tab_label(tw, "grounding")):
            return False
        if any("trainer" in t.lower() or "training" in t.lower() for t in texts):
            return False

        for i in range(tw.count() - 1, -1, -1):
            try:
                if str(tw.tabText(i) or "").strip().lower() in {"adaf", "adaf models", "ad af"} or str(tw.widget(i).objectName() or "") == "MustatilADAFLeftControls":
                    old = tw.widget(i)
                    tw.removeTab(i)
                    try:
                        old.deleteLater()
                    except Exception:
                        pass
                    if i < lae_idx:
                        lae_idx -= 1
            except Exception:
                pass

        ws = _workspace_from_widget(tw)
        if ws is None:
            return False
        _patch_workspace_filters(ws)
        kind = _inner_tab_kind_from_widget(tw)
        if kind not in {"detection", "satellite"}:
            kind = "detection"
        page = _build_adaf_left_controls(ws, kind)
        tw.insertTab(min(tw.count(), lae_idx + 1), page, "ADAF")
        _PATCHED_INNER_TABS.add(id(tw))
        try:
            ws.log(f"ADAF tab inserted right next to LAE-DINO in {kind}.")
        except Exception:
            pass
        _log(f"ADAF inner tab inserted next to LAE-DINO ({kind})")
        return True
    except Exception as exc:
        _log("ensure inner tab failed: " + str(exc))
        traceback.print_exc()
        return False


def _is_lae_trainer_label(text):
    low = str(text or "").strip().lower()
    return ("lae" in low and "dino" in low and ("trainer" in low or "training" in low))


def _is_adaf_training_label(text):
    low = str(text or "").strip().lower()
    return low in {"adaf trainer", "adaf models", "adaf model", "adaf"} or ("adaf" in low and ("model" in low or "trainer" in low))


def _ensure_adaf_training_tab(tw):
    """v7: do not add any ADAF Trainer tab.

    Also removes old ADAF Trainer / ADAF Models tabs from previous ADAF plugin
    versions if they are in the same training tab widget.
    """
    try:
        texts = _tab_texts(tw)
        has_lae_trainer = any(_is_lae_trainer_label(t) for t in texts)
        if not has_lae_trainer:
            return False

        removed = False
        for i in range(tw.count() - 1, -1, -1):
            try:
                label = str(tw.tabText(i) or "").strip().lower()
                obj = str(tw.widget(i).objectName() or "")
                if (
                    label in {"adaf trainer", "adaf models", "adaf model", "adaf"}
                    or ("adaf" in label and ("trainer" in label or "model" in label))
                    or obj in {"MustatilADAFTrainerTab", "MustatilADAFModelsTrainingTab"}
                ):
                    old = tw.widget(i)
                    tw.removeTab(i)
                    try:
                        old.deleteLater()
                    except Exception:
                        pass
                    removed = True
            except Exception:
                pass

        if removed:
            ws = _workspace_from_widget(tw)
            try:
                if ws is not None:
                    ws.log("Removed old ADAF Trainer/Models tab. ADAF is now Detection/Satellite only.")
            except Exception:
                pass
            _log("removed old ADAF Trainer/Models tab")
        return removed
    except Exception as exc:
        _log("remove training tab failed: " + str(exc))
        traceback.print_exc()
        return False


def _scan_all_tabwidgets(root=None):
    try:
        from PySide6.QtWidgets import QApplication, QTabWidget
    except Exception as exc:
        _log("Qt unavailable: " + str(exc))
        return False

    widgets = []
    try:
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
    except Exception:
        pass

    try:
        app = QApplication.instance()
        if app:
            for w in app.allWidgets():
                try:
                    if isinstance(w, QTabWidget):
                        widgets.append(w)
                except Exception:
                    pass
    except Exception:
        pass

    seen = set()
    ok = False
    for tw in widgets:
        if id(tw) in seen:
            continue
        seen.add(id(tw))
        try:
            if _ensure_adaf_inner_tab(tw):
                ok = True
        except Exception:
            pass
        try:
            if _ensure_adaf_training_tab(tw):
                ok = True
        except Exception:
            pass
    return ok


def _install_qtab_hook(root=None):
    global _PATCHED_QTAB, _ORIG_ADD, _ORIG_INSERT
    try:
        from PySide6.QtWidgets import QTabWidget
        from PySide6.QtCore import QTimer
    except Exception as exc:
        _log("Qt unavailable: " + str(exc))
        return

    if not _PATCHED_QTAB:
        _ORIG_ADD = QTabWidget.addTab
        _ORIG_INSERT = QTabWidget.insertTab

        def _after(tw, label):
            try:
                low = str(label or "").lower()
                if any(k in low for k in ["lae", "dino", "trainer", "training", "owl", "grounding", "yolo", "satellite", "detection"]):
                    QTimer.singleShot(30, lambda tw=tw: (_ensure_adaf_inner_tab(tw), _ensure_adaf_training_tab(tw)))
                    QTimer.singleShot(300, lambda tw=tw: (_ensure_adaf_inner_tab(tw), _ensure_adaf_training_tab(tw)))
                    QTimer.singleShot(1200, lambda tw=tw: (_ensure_adaf_inner_tab(tw), _ensure_adaf_training_tab(tw)))
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
            _after(self, label)
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
            _after(self, label)
            return res

        QTabWidget.addTab = addTab_patched
        QTabWidget.insertTab = insertTab_patched
        _PATCHED_QTAB = True
        _log("QTabWidget hook installed")

    try:
        for ms in (100, 300, 800, 1500, 3000, 6000, 10000, 15000):
            QTimer.singleShot(ms, lambda root=root: _scan_all_tabwidgets(root))
    except Exception:
        pass


def mustatil_plugin_init():
    _install_qtab_hook()
    return True


def register_plugin(app=None, main_window=None):
    _install_qtab_hook(main_window or app)
    try:
        _scan_all_tabwidgets(main_window or app)
    except Exception:
        pass
    return True


def init_plugin(app=None, main_window=None):
    return register_plugin(app, main_window)


def load_plugin(app=None, main_window=None):
    return register_plugin(app, main_window)


try:
    _install_qtab_hook()
except Exception:
    traceback.print_exc()
