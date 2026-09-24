#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mustatil plugin: RF-DETR v15 infer_kind fix + ADAF-neighbor Detection placement.

- Detection/Satellite: adds RF-DETR tab next to LAE-DINO.
- Training: adds RF-DETR Trainer tab next to LAE-DINO Trainer with R-CNN-style image/label folders.
- Installs rfdetr into an isolated external RF-DETR runtime venv, not into the Mustatil venv.
- Supports Nano/Small/Medium/Large/Base and optional XL/2XL.
- Trainer calls RF-DETR's own .train(dataset_dir=..., epochs=..., batch_size=..., output_dir=...).
"""
from __future__ import annotations

import os, sys, json, time, tempfile, shutil, subprocess, traceback
from pathlib import Path

_PATCHED = False
_ORIG_ADD = None
_ORIG_INSERT = None
_PATCHED_INNER = set()
_PATCHED_TRAINER = set()
WEB_TILE_SIZE = 256

RFDETR_WORKER_CODE = '\nfrom __future__ import annotations\nimport os, sys, json, time, shutil, random, argparse, traceback, importlib, inspect, subprocess\nfrom pathlib import Path\n\nIMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}\n\nMODEL_MAP = {\n    "RFDETRNano": ("rfdetr", "RFDETRNano"),\n    "RFDETRSmall": ("rfdetr", "RFDETRSmall"),\n    "RFDETRMedium": ("rfdetr", "RFDETRMedium"),\n    "RFDETRLarge": ("rfdetr", "RFDETRLarge"),\n    "RFDETRBase": ("rfdetr", "RFDETRBase"),\n    "RFDETRSegNano": ("rfdetr", "RFDETRSegNano"),\n    "RFDETRSegSmall": ("rfdetr", "RFDETRSegSmall"),\n    "RFDETRSegMedium": ("rfdetr", "RFDETRSegMedium"),\n    "RFDETRSegLarge": ("rfdetr", "RFDETRSegLarge"),\n    "RFDETRSegXLarge": ("rfdetr", "RFDETRSegXLarge"),\n    "RFDETRSeg2XLarge": ("rfdetr", "RFDETRSeg2XLarge"),\n    "RFDETRXLarge": ("rfdetr_plus", "RFDETRXLarge"),\n    "RFDETR2XLarge": ("rfdetr_plus", "RFDETR2XLarge"),\n}\n\ndef log(msg):\n    print(str(msg), flush=True)\n\ndef parse_classes(raw):\n    vals = [p.strip() for p in str(raw or "").replace(";", ",").split(",") if p.strip()]\n    return vals or ["object"]\n\ndef get_model_class(name):\n    if name not in MODEL_MAP:\n        raise RuntimeError("Unknown RF-DETR model class: " + str(name))\n    mod_name, attr = MODEL_MAP[name]\n    mod = importlib.import_module(mod_name)\n    return getattr(mod, attr), mod_name\n\ndef make_model(name, checkpoint="", accept_plus=False):\n    cls, mod_name = get_model_class(name)\n    kwargs = {}\n    if mod_name == "rfdetr_plus":\n        kwargs["accept_platform_model_license"] = bool(accept_plus)\n    if checkpoint:\n        return cls(pretrain_weights=checkpoint, **kwargs)\n    return cls(**kwargs)\n\ndef detections_to_rows(detections, chunk):\n    out = []\n    xyxy = getattr(detections, "xyxy", None)\n    conf = getattr(detections, "confidence", None)\n    class_id = getattr(detections, "class_id", None)\n    if xyxy is None:\n        return out\n\n    boxes = xyxy.tolist() if hasattr(xyxy, "tolist") else list(xyxy)\n    scores = conf.tolist() if hasattr(conf, "tolist") else list(conf or [])\n    classes = class_id.tolist() if hasattr(class_id, "tolist") else list(class_id or [])\n\n    for i, box in enumerate(boxes):\n        try:\n            x1, y1, x2, y2 = [float(v) for v in list(box)[:4]]\n            score = float(scores[i]) if i < len(scores) and scores[i] is not None else 0.0\n            cid = int(classes[i]) if i < len(classes) and classes[i] is not None else 0\n            out.append({\n                "chunk_id": int(chunk.get("id", 0)),\n                "offset_x": float(chunk.get("x", 0)),\n                "offset_y": float(chunk.get("y", 0)),\n                "x1": x1, "y1": y1, "x2": x2, "y2": y2,\n                "confidence": score,\n                "class_id": cid,\n                "class_name": str(cid),\n            })\n        except Exception:\n            continue\n    return out\n\ndef mode_check(args):\n    report = {"python": sys.executable, "imports": {}, "status": "ok", "requested_device": getattr(args, "device", "cuda")}\n    for name in ["torch", "rfdetr", "supervision", "PIL"]:\n        try:\n            mod = importlib.import_module(name)\n            report["imports"][name] = {"ok": True, "version": getattr(mod, "__version__", "unknown")}\n        except Exception as exc:\n            report["imports"][name] = {"ok": False, "error": repr(exc)}\n    try:\n        import torch\n        report["cuda_available"] = bool(torch.cuda.is_available())\n        report["cuda_version"] = str(getattr(torch.version, "cuda", ""))\n    except Exception:\n        pass\n    log(json.dumps(report, indent=2))\n\ndef mode_predict(args):\n    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))\n    chunks = manifest.get("chunks", [])\n    model = make_model(args.model_class, checkpoint=args.checkpoint or "", accept_plus=args.accept_plus)\n    all_rows = []\n    for i, ch in enumerate(chunks, 1):\n        try:\n            det = model.predict(ch["path"], threshold=float(args.conf))\n            rows = detections_to_rows(det, ch)\n            all_rows.extend(rows)\n            log(f"RF-DETR chunk {i}/{len(chunks)}: {len(rows)} detections")\n        except Exception as exc:\n            log(f"RF-DETR chunk {i}/{len(chunks)} failed: {exc}")\n            if args.fail_fast:\n                raise\n    result = {"status": "ok", "model_class": args.model_class, "checkpoint": args.checkpoint, "detections": all_rows}\n    Path(args.output).parent.mkdir(parents=True, exist_ok=True)\n    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")\n    log("MUSTATIL_RFDETR_RESULT=" + str(args.output))\n    log("MUSTATIL_RFDETR_DETECTIONS=" + str(len(all_rows)))\n\ndef looks_like_dataset(root):\n    root = Path(root)\n    if (root / "data.yaml").exists():\n        return True\n    for s in ["train", "valid", "val", "test"]:\n        if (root / s / "images").exists() and (root / s / "labels").exists():\n            return True\n    return False\n\ndef write_data_yaml(root, classes):\n    lines = [\n        "path: " + str(root).replace("\\\\", "/"),\n        "train: train/images",\n        "val: valid/images",\n        "test: test/images",\n        "nc: " + str(len(classes)),\n        "names:",\n    ]\n    for i, c in enumerate(classes):\n        lines.append(f"  {i}: {c}")\n    (root / "data.yaml").write_text("\\n".join(lines) + "\\n", encoding="utf-8")\n\ndef find_label(img, label_dirs):\n    for d in label_dirs:\n        p = d / (img.stem + ".txt")\n        if p.exists():\n            return p\n    p = img.with_suffix(".txt")\n    if p.exists():\n        return p\n    return None\n\ndef prepare_yolo_dataset(project, output_dir, classes):\n    project = Path(project)\n    output_dir = Path(output_dir)\n    if looks_like_dataset(project):\n        if not (project / "data.yaml").exists():\n            try:\n                write_data_yaml(project, classes)\n            except Exception:\n                pass\n        return str(project)\n\n    imgs = []\n    for p in project.rglob("*"):\n        if p.is_file() and p.suffix.lower() in IMG_EXTS:\n            low = str(p).replace("\\\\", "/").lower()\n            if not any(x in low for x in ["/runs/", "/output/", "/outputs/", "/.venv/", "/venv/", "/__pycache__/"]):\n                imgs.append(p)\n\n    label_dirs = []\n    for name in ["labels", "label", "yolo_labels"]:\n        label_dirs += [d for d in project.rglob(name) if d.is_dir()]\n    label_dirs.append(project)\n\n    pairs = []\n    for img in imgs:\n        lab = find_label(img, label_dirs)\n        if lab and lab.exists() and lab.read_text(encoding="utf-8", errors="ignore").strip():\n            pairs.append((img, lab))\n    if not pairs:\n        raise RuntimeError("No images with matching YOLO labels found. Expected images and labels/*.txt files.")\n\n    random.seed(42)\n    random.shuffle(pairs)\n    split = max(1, int(len(pairs) * 0.9))\n    train = pairs[:split]\n    valid = pairs[split:] or pairs[:1]\n\n    dst = output_dir / "mustatil_rfdetr_prepared_dataset"\n    if dst.exists():\n        shutil.rmtree(dst, ignore_errors=True)\n    for s in ["train", "valid", "test"]:\n        (dst / s / "images").mkdir(parents=True, exist_ok=True)\n        (dst / s / "labels").mkdir(parents=True, exist_ok=True)\n\n    def copy_split(items, s):\n        for img, lab in items:\n            shutil.copy2(img, dst / s / "images" / img.name)\n            shutil.copy2(lab, dst / s / "labels" / (img.stem + ".txt"))\n\n    copy_split(train, "train")\n    copy_split(valid, "valid")\n    copy_split(valid, "test")\n    write_data_yaml(dst, classes)\n    return str(dst)\n\n\n\ndef ensure_training_deps():\n    """Install minimal RF-DETR training dependencies inside the isolated runtime if missing.\n\n    Deliberately avoids rfdetr[loggers], mlflow and opentelemetry because these\n    large logger packages caused WinError 32 file locks on Windows while Mustatil\n    was running.\n    """\n    missing = []\n    checks = [\n        ("pytorch_lightning", "pytorch-lightning"),\n        ("tensorboard", "tensorboard"),\n        ("torchmetrics", "torchmetrics"),\n        ("lightning_utilities", "lightning-utilities"),\n        ("albumentations", "albumentations"),\n        ("peft", "peft"),\n        ("accelerate", "accelerate"),\n    ]\n    for mod, pkg in checks:\n        try:\n            importlib.import_module(mod)\n        except Exception:\n            missing.append(pkg)\n    if not missing:\n        return\n    # Keep install list stable; pip will skip already satisfied packages.\n    pkgs = ["pytorch-lightning", "tensorboard", "torchmetrics", "lightning-utilities", "albumentations", "peft", "accelerate"]\n    log("RF-DETR light training deps missing: " + ", ".join(missing))\n    log("Installing light RF-DETR training deps only: " + " ".join(pkgs))\n    cmd = [sys.executable, "-m", "pip", "install", "-U", "--no-cache-dir"] + pkgs\n    p = subprocess.run(cmd, text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)\n    log(p.stdout or "")\n    if p.returncode != 0:\n        raise RuntimeError(\n            "Could not install RF-DETR light training dependencies. pip exit code="\n            + str(p.returncode)\n            + ". Close Mustatil completely and kill leftover python.exe processes, then press Install / Repair again."\n        )\n\n\ndef mode_train(args):\n    raw_device = str(getattr(args, "device", "cuda") or "cuda").strip().lower()\n    ensure_training_deps()\n    output_dir = Path(args.output_dir)\n    output_dir.mkdir(parents=True, exist_ok=True)\n    classes = parse_classes(args.classes)\n    dataset_dir = prepare_yolo_dataset(args.dataset_dir, output_dir, classes)\n    log("RF-DETR dataset_dir=" + str(dataset_dir))\n    model = make_model(args.model_class, checkpoint=args.resume or "", accept_plus=args.accept_plus)\n\n    kwargs = {\n        "dataset_dir": dataset_dir,\n        "epochs": int(args.epochs),\n        "batch_size": int(args.batch_size),\n        "grad_accum_steps": int(args.grad_accum_steps),\n        "lr": float(args.lr),\n        "output_dir": str(output_dir),\n    }\n\n    strategy = {}\n    if args.backbone_strategy:\n        try:\n            strategy = json.loads(args.backbone_strategy)\n            log("RF-DETR backbone strategy=" + json.dumps(strategy, ensure_ascii=False))\n        except Exception as exc:\n            log("Could not parse backbone strategy JSON: " + repr(exc))\n            strategy = {}\n\n    optional = {}\n    if args.early_stopping:\n        optional["early_stopping"] = True\n    if args.tensorboard:\n        optional["tensorboard"] = True\n    if args.freeze_backbone:\n        optional["freeze_backbone"] = True\n        optional["freeze_encoder"] = True\n    if args.backbone_lr and float(args.backbone_lr) > 0:\n        optional["backbone_lr"] = float(args.backbone_lr)\n        optional["lr_backbone"] = float(args.backbone_lr)\n    if args.encoder_lr and float(args.encoder_lr) > 0:\n        optional["encoder_lr"] = float(args.encoder_lr)\n        optional["lr_encoder"] = float(args.encoder_lr)\n    if args.resolution and int(args.resolution) > 0:\n        optional["resolution"] = int(args.resolution)\n        optional["image_size"] = int(args.resolution)\n        optional["imgsz"] = int(args.resolution)\n    if args.num_queries and int(args.num_queries) > 0:\n        optional["num_queries"] = int(args.num_queries)\n    if str(raw_device) in {"cpu", "cuda"} or str(raw_device).startswith("cuda:"):\n        optional["device"] = "cuda" if raw_device.startswith("cuda:") else raw_device\n    if str(raw_device) in {"multi-gpu", "multi_gpu", "cuda:all", "all"}:\n        optional["device"] = "cuda"\n        optional["multi_gpu"] = True\n        optional["distributed"] = True\n\n    for k, v in dict(strategy.get("train_kwargs", {}) or {}).items():\n        optional[k] = v\n\n    try:\n        sig = inspect.signature(model.train)\n        params = sig.parameters\n        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())\n        accepted_optional = {}\n        skipped_optional = {}\n        for k, v in optional.items():\n            if accepts_kwargs or k in params:\n                accepted_optional[k] = v\n            else:\n                skipped_optional[k] = v\n        if skipped_optional:\n            log("RF-DETR optional train args not supported by installed package and skipped: " + ", ".join(sorted(skipped_optional.keys())))\n        kwargs.update(accepted_optional)\n    except Exception as exc:\n        log("Could not inspect RF-DETR train signature; using base train kwargs only: " + repr(exc))\n\n    log("RF-DETR train kwargs=" + json.dumps({k: str(v) for k, v in kwargs.items()}))\n    try:\n        model.train(**kwargs)\n    except ImportError as exc:\n        msg = str(exc)\n        if "training dependencies are missing" in msg or "pytorch_lightning" in msg or "lightning" in msg:\n            log("RF-DETR train failed because training extras are missing; repairing runtime and retrying once.")\n            ensure_training_deps()\n            model.train(**kwargs)\n        else:\n            raise\n    result = {"status": "ok", "model_class": args.model_class, "output_dir": str(output_dir), "dataset_dir": dataset_dir}\n    rp = output_dir / ("mustatil_rfdetr_train_result_" + str(time.time_ns()) + ".json")\n    rp.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")\n    log("MUSTATIL_RFDETR_TRAIN_RESULT=" + str(rp))\n    log("MUSTATIL_RFDETR_TRAIN_OUTPUT=" + str(output_dir))\n\ndef main():\n    ap = argparse.ArgumentParser()\n    ap.add_argument("--mode", required=True, choices=["check", "predict", "train"])\n    ap.add_argument("--model-class", default="RFDETRSmall")\n    ap.add_argument("--checkpoint", default="")\n    ap.add_argument("--accept-plus", action="store_true")\n    ap.add_argument("--device", default="cuda")\n    ap.add_argument("--conf", type=float, default=0.35)\n    ap.add_argument("--manifest", default="")\n    ap.add_argument("--output", default="")\n    ap.add_argument("--fail-fast", action="store_true")\n    ap.add_argument("--dataset-dir", default="")\n    ap.add_argument("--output-dir", default="")\n    ap.add_argument("--classes", default="object")\n    ap.add_argument("--epochs", type=int, default=20)\n    ap.add_argument("--batch-size", type=int, default=1)\n    ap.add_argument("--grad-accum-steps", type=int, default=4)\n    ap.add_argument("--lr", type=float, default=1e-4)\n    ap.add_argument("--resume", default="")\n    ap.add_argument("--early-stopping", action="store_true")\n    ap.add_argument("--tensorboard", action="store_true")\n    ap.add_argument("--backbone-strategy", default="")\n    ap.add_argument("--freeze-backbone", action="store_true")\n    ap.add_argument("--backbone-lr", type=float, default=0.0)\n    ap.add_argument("--encoder-lr", type=float, default=0.0)\n    ap.add_argument("--resolution", type=int, default=0)\n    ap.add_argument("--num-queries", type=int, default=0)\n    args = ap.parse_args()\n\n    raw_device = str(args.device or "cuda").strip().lower()\n    os.environ["MUSTATIL_RFDETR_DEVICE"] = raw_device\n    if raw_device == "cpu":\n        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"\n        os.environ["ACCELERATE_USE_CPU"] = "true"\n    elif raw_device.startswith("cuda:"):\n        # Run this worker on exactly one visible physical GPU.\n        gpu_id = raw_device.split(":", 1)[1].strip()\n        if gpu_id.isdigit():\n            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id\n            os.environ.pop("ACCELERATE_USE_CPU", None)\n    elif raw_device in {"cuda", "auto", ""}:\n        os.environ.pop("CUDA_VISIBLE_DEVICES", None)\n        os.environ.pop("ACCELERATE_USE_CPU", None)\n    elif str(raw_device) in {"multi-gpu", "multi_gpu", "cuda:all", "all"}:\n        # The main Mustatil plugin splits detection chunks across devices.\n        # For a single worker, leave all GPUs visible.\n        os.environ.pop("CUDA_VISIBLE_DEVICES", None)\n        os.environ.pop("ACCELERATE_USE_CPU", None)\n\n    try:\n        if args.mode == "check":\n            mode_check(args)\n        elif args.mode == "predict":\n            mode_predict(args)\n        elif args.mode == "train":\n            mode_train(args)\n    except Exception:\n        traceback.print_exc()\n        sys.exit(1)\n\nif __name__ == "__main__":\n    main()\n'

CORE_MODELS = [
    ("RF-DETR Nano", "RFDETRNano"),
    ("RF-DETR Small", "RFDETRSmall"),
    ("RF-DETR Medium", "RFDETRMedium"),
    ("RF-DETR Large", "RFDETRLarge"),
    ("RF-DETR Base", "RFDETRBase"),
]
PLUS_MODELS = [
    ("RF-DETR XLarge", "RFDETRXLarge"),
    ("RF-DETR 2XLarge", "RFDETR2XLarge"),
]
SEG_MODELS = [
    ("RF-DETR Seg Nano", "RFDETRSegNano"),
    ("RF-DETR Seg Small", "RFDETRSegSmall"),
    ("RF-DETR Seg Medium", "RFDETRSegMedium"),
    ("RF-DETR Seg Large", "RFDETRSegLarge"),
    ("RF-DETR Seg XLarge", "RFDETRSegXLarge"),
    ("RF-DETR Seg 2XLarge", "RFDETRSeg2XLarge"),
]


def _log(msg):
    try:
        print("[Mustatil RF-DETR] " + str(msg))
    except Exception:
        pass


def _get_var(v, default=""):
    try:
        if hasattr(v, "get"):
            return v.get()
    except Exception:
        pass
    return default


def _ws(widget):
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
            cur = cur.parentWidget()
        except Exception:
            try:
                cur = cur.parent()
            except Exception:
                break
    return None


def _runtime() -> Path:
    try:
        p = Path(__file__).resolve().parent / "mustatil_model_runtimes" / "RF-DETR"
    except Exception:
        p = Path.home() / "mustatil_model_runtimes" / "RF-DETR"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _runs() -> Path:
    p = _runtime() / "runs"
    p.mkdir(parents=True, exist_ok=True)
    return p



def _runtime_python() -> Path:
    if os.name == "nt":
        return _runtime() / ".venv" / "Scripts" / "python.exe"
    return _runtime() / ".venv" / "bin" / "python"


def _find_base_python_for_runtime(ws=None) -> str:
    """Prefer Python 3.11/3.10 for RF-DETR. Fall back to current Mustatil Python."""
    if os.name == "nt":
        for ver in ("3.11", "3.10"):
            try:
                p = subprocess.run(
                    ["py", "-" + ver, "-c", "import sys; print(sys.executable)"],
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                if p.returncode == 0:
                    exe = (p.stdout or "").strip().splitlines()[-1].strip()
                    if exe and Path(exe).exists():
                        try:
                            if ws is not None:
                                ws.log("RF-DETR runtime base Python: " + exe)
                        except Exception:
                            pass
                        return exe
            except Exception:
                pass
    return sys.executable


def _ensure_runtime_created(ws=None, force_recreate=False) -> Path:
    rt = _runtime()
    py = _runtime_python()
    if force_recreate:
        try:
            shutil.rmtree(rt / ".venv", ignore_errors=True)
        except Exception:
            pass
    if py.exists():
        return py
    base = _find_base_python_for_runtime(ws)
    rt.mkdir(parents=True, exist_ok=True)
    _run_logged(ws, [base, "-m", "venv", str(rt / ".venv")], "RF-DETR create external runtime", cwd=rt)
    if not py.exists():
        raise RuntimeError("RF-DETR runtime python was not created: " + str(py))
    return py


def _install_external_runtime(ws=None, plus=False, metrics=False, force_recreate=False) -> Path:
    py = _ensure_runtime_created(ws, force_recreate=force_recreate)
    _run_logged(ws, [py, "-m", "pip", "install", "-U", "pip", "wheel", "setuptools"], "RF-DETR runtime pip base", cwd=_runtime())

    try:
        _run_logged(
            ws,
            [py, "-m", "pip", "install", "-U", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu121"],
            "RF-DETR runtime torch cu121",
            cwd=_runtime(),
        )
    except Exception as exc:
        try:
            if ws is not None:
                ws.log("CUDA torch install failed, trying default torch wheels: " + str(exc))
        except Exception:
            pass
        _run_logged(ws, [py, "-m", "pip", "install", "-U", "torch", "torchvision"], "RF-DETR runtime torch default", cwd=_runtime())

    pkgs = ["rfdetr", "supervision", "pillow", "pyyaml", "pytorch-lightning", "tensorboard", "torchmetrics", "lightning-utilities", "albumentations", "peft", "accelerate"]
    if metrics:
        pkgs.append("rfdetr[metrics]")
    if plus:
        pkgs.append("rfdetr-plus")
    _run_logged(ws, [py, "-m", "pip", "install", "-U"] + pkgs, "RF-DETR runtime package install", cwd=_runtime())
    _run_logged(ws, [py, str(_worker()), "--mode", "check", "--device", "cuda"], "RF-DETR runtime check", cwd=_runtime())
    return py


def _worker() -> Path:
    p = _runtime() / "mustatil_rfdetr_worker.py"
    p.write_text(RFDETR_WORKER_CODE, encoding="utf-8")
    return p


def _run_logged(ws, cmd, title="RUN", cwd=None):
    try:
        ws.log(title + ": " + " ".join(str(x) for x in cmd))
    except Exception:
        pass
    proc = subprocess.run([str(x) for x in cmd], cwd=str(cwd) if cwd else None, text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = proc.stdout or ""
    try:
        for line in out.splitlines()[-120:]:
            ws.log(line)
    except Exception:
        pass
    if proc.returncode != 0:
        raise RuntimeError(title + " failed with exit code " + str(proc.returncode) + "\n" + "\n".join(out.splitlines()[-120:]))
    return out


def _run_task(ws, title, fn, allow_parallel=False):
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


def _install(ws, plus=False, metrics=False):
    """Install RF-DETR into an isolated external runtime instead of Mustatil's venv."""
    try:
        if ws is not None:
            ws.log("RF-DETR v2: external runtime: " + str(_runtime()))
    except Exception:
        pass
    _install_external_runtime(ws, plus=plus, metrics=metrics, force_recreate=False)



def _fill(combo, include_seg=False, include_plus=True):
    combo.blockSignals(True)
    combo.clear()
    for label, cls in CORE_MODELS:
        combo.addItem(label, cls)
    if include_plus:
        for label, cls in PLUS_MODELS:
            combo.addItem(label + " (Plus/PML)", cls)
    if include_seg:
        for label, cls in SEG_MODELS:
            combo.addItem(label + " (Seg)", cls)
    combo.blockSignals(False)


def _model(combo):
    try:
        return str(combo.currentData() or "RFDETRSmall")
    except Exception:
        return "RFDETRSmall"


def _is_plus(cls):
    return cls in {"RFDETRXLarge", "RFDETR2XLarge"}



def _available_cuda_device_ids(ws=None) -> list[int]:
    """Return available CUDA device IDs for RF-DETR multi-GPU splitting."""
    ids = []
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if p.returncode == 0:
            for line in (p.stdout or "").splitlines():
                line = line.strip()
                if line.isdigit():
                    ids.append(int(line))
    except Exception:
        pass

    if not ids and _runtime_python().exists():
        try:
            p = subprocess.run(
                [str(_runtime_python()), "-c", "import torch; print(torch.cuda.device_count())"],
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if p.returncode == 0:
                n = int((p.stdout or "0").strip().splitlines()[-1])
                ids = list(range(max(0, n)))
        except Exception:
            pass

    if not ids:
        ids = [0]
    try:
        if ws is not None:
            ws.log("RF-DETR CUDA devices visible: " + ", ".join("cuda:%d" % i for i in ids))
    except Exception:
        pass
    return ids


def _is_multigpu_device(device: str) -> bool:
    low = str(device or "").strip().lower()
    return low in {"multi-gpu", "multi_gpu", "cuda:all", "all-gpu", "all_gpus", "all"}


def _device_combo_items():
    return ["cuda", "cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3", "multi-gpu"]


def _worker_predict_once(ws, chunks, model_class, checkpoint, conf, accept_plus, device, title_suffix=""):
    temp = Path(tempfile.mkdtemp(prefix="mustatil_rfdetr_pred_"))
    manifest = temp / "manifest.json"
    output = temp / "predictions.json"
    manifest.write_text(json.dumps({"chunks": chunks}, indent=2), encoding="utf-8")
    cmd = [str(_runtime_python()), str(_worker()), "--mode", "predict", "--model-class", model_class, "--manifest", str(manifest), "--output", str(output), "--conf", str(float(conf)), "--device", str(device)]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    if accept_plus:
        cmd += ["--accept-plus"]
    _run_logged(ws, cmd, "RF-DETR predict" + str(title_suffix), cwd=_runtime())
    data = json.loads(output.read_text(encoding="utf-8", errors="replace"))
    try:
        ws.mustatil_rfdetr_last_report = str(output)
    except Exception:
        pass
    return data.get("detections", []) or []


def _predict_multigpu(ws, chunks, model_class, checkpoint, conf, accept_plus):
    """Split Detection/Satellite chunks across GPUs and run one worker per GPU."""
    if not chunks:
        return []
    devices = _available_cuda_device_ids(ws)
    if len(devices) <= 1:
        try:
            if ws is not None:
                ws.log("RF-DETR multi-gpu requested but only one CUDA device was found. Using cuda:0.")
        except Exception:
            pass
        return _worker_predict_once(ws, chunks, model_class, checkpoint, conf, accept_plus, "cuda:0", " cuda:0")

    groups = {i: [] for i in devices}
    for idx, chunk in enumerate(chunks):
        gpu = devices[idx % len(devices)]
        groups[gpu].append(chunk)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    all_rows = []
    futures = []
    try:
        if ws is not None:
            ws.log("RF-DETR multi-gpu detection: " + ", ".join("cuda:%d=%d chunks" % (g, len(groups[g])) for g in devices))
    except Exception:
        pass

    with ThreadPoolExecutor(max_workers=len(devices)) as ex:
        for gpu in devices:
            part = groups[gpu]
            if not part:
                continue
            futures.append(ex.submit(_worker_predict_once, ws, part, model_class, checkpoint, conf, accept_plus, "cuda:%d" % gpu, " cuda:%d" % gpu))
        for f in as_completed(futures):
            all_rows.extend(f.result() or [])
    try:
        if ws is not None:
            ws.log("RF-DETR multi-gpu merged detections: " + str(len(all_rows)))
    except Exception:
        pass
    return all_rows



def _predict(ws, chunks, model_class, checkpoint, conf, accept_plus, device='cuda'):
    if not _runtime_python().exists():
        raise RuntimeError('RF-DETR external runtime not found. Press Install / Repair RF-DETR first.')
    if _is_multigpu_device(device):
        return _predict_multigpu(ws, chunks, model_class, checkpoint, conf, accept_plus)
    return _worker_predict_once(ws, chunks, model_class, checkpoint, conf, accept_plus, device)


def _run_detection(ws):
    from PIL import Image
    from mustatil_legacy_backend import Det
    img_path = str(_get_var(getattr(ws, "image", None), "") or "").strip().strip('"')
    if not img_path or not Path(img_path).exists():
        raise RuntimeError("No Detection image selected.")
    conf = max(0.001, min(1.0, float(_get_var(getattr(ws, "conf", None), 0.35))))
    tile = max(128, int(_get_var(getattr(ws, "tile", None), 1024) or 1024))
    overlap = max(0, min(tile - 1, int(_get_var(getattr(ws, "overlap", None), 160) or 160)))
    step = max(1, tile - overlap)
    model_class = str(getattr(ws, "mustatil_rfdetr_model_class", "RFDETRSmall") or "RFDETRSmall")
    checkpoint = str(getattr(ws, "mustatil_rfdetr_checkpoint", "") or "").strip().strip('"')
    accept_plus = bool(getattr(ws, "mustatil_rfdetr_accept_plus", False))
    device = str(getattr(ws, "mustatil_rfdetr_device", "cuda") or "cuda")
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    temp = Path(tempfile.mkdtemp(prefix="mustatil_rfdetr_chunks_"))
    chunks, cid = [], 0
    for y in range(0, H, step):
        for x in range(0, W, step):
            x2, y2 = min(W, x + tile), min(H, y + tile)
            cid += 1
            p = temp / f"chunk_{cid:05d}.png"
            im.crop((x, y, x2, y2)).save(p)
            chunks.append({"id": cid, "path": str(p), "x": x, "y": y, "w": x2 - x, "h": y2 - y})
            if x2 >= W:
                break
        if y + tile >= H:
            break
    ws.log(f"RF-DETR Detection started: chunks={len(chunks)}, model={model_class}, checkpoint={Path(checkpoint).name if checkpoint else 'pretrained'}")
    rows = _predict(ws, chunks, model_class, checkpoint, conf, accept_plus, device=device)
    dets = []
    for r in rows:
        ox, oy = float(r.get("offset_x", 0)), float(r.get("offset_y", 0))
        dets.append(Det(0, "RF-DETR " + model_class, int(r.get("class_id", 0)), float(r.get("confidence", 0.0)), ox+float(r["x1"]), oy+float(r["y1"]), ox+float(r["x2"]), oy+float(r["y2"])))
    try:
        dets = ws.nms(dets, 0.45)
    except Exception:
        pass
    ws.dets = list(dets)
    try:
        ws.loadprev()
    except Exception:
        pass
    try:
        ws.redraw(fit=False)
    except Exception:
        pass
    shutil.rmtree(temp, ignore_errors=True)
    ws.log(f"RF-DETR Detection finished: {len(dets)} detections")


def _run_satellite(ws):
    g = ws.__class__.satellite_detect_selected.__globals__
    sat_tile_bounds_for_bbox = g.get("sat_tile_bounds_for_bbox")
    sat_lonlat_from_world_px = g.get("sat_lonlat_from_world_px")
    if sat_tile_bounds_for_bbox is None or sat_lonlat_from_world_px is None:
        raise RuntimeError("Satellite helper functions not found.")
    min_lat, min_lon, max_lat, max_lon = ws._satellite_bbox()
    z = int(_get_var(ws.sat_zoom, 0))
    x_min, y_min, x_max, y_max = sat_tile_bounds_for_bbox(min_lat, min_lon, max_lat, max_lon, z)
    cols, rows = x_max - x_min + 1, y_max - y_min + 1
    width, height = cols * WEB_TILE_SIZE, rows * WEB_TILE_SIZE
    chunk = max(64, int(_get_var(ws.tile, 1024) or 1024))
    conf = max(0.001, min(1.0, float(_get_var(ws.conf, 0.35))))
    model_class = str(getattr(ws, "mustatil_rfdetr_model_class", "RFDETRSmall") or "RFDETRSmall")
    checkpoint = str(getattr(ws, "mustatil_rfdetr_checkpoint", "") or "").strip().strip('"')
    accept_plus = bool(getattr(ws, "mustatil_rfdetr_accept_plus", False))
    device = str(getattr(ws, "mustatil_rfdetr_device", "cuda") or "cuda")
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns()%1000000000:09d}"
    out_path = ws._satellite_unique_run_output_path(ws._satellite_output_gpkg_path(), stamp)
    temp_root = ws._satellite_cache_dir() / "_detection_tmp" / f"rfdetr_z{z}_{stamp}_{os.getpid()}"
    temp_root.mkdir(parents=True, exist_ok=True)
    old_temp = getattr(ws, "sat_detection_temp_cache_root", None)
    old_thread = getattr(getattr(ws, "_sat_detection_thread_local", None), "cache_root", None)
    ws.sat_detection_temp_cache_root = temp_root
    try:
        ws._sat_detection_thread_local.cache_root = temp_root
    except Exception:
        pass
    try:
        chunks, used_paths, cid = [], [], 0
        for y in range(0, height, chunk):
            for x in range(0, width, chunk):
                cid += 1
                cw, ch = min(chunk, width - x), min(chunk, height - y)
                meta = ws._satellite_build_chunk_to_cache(x_min, y_min, z, x, y, cw, ch, cid, temp_root)
                chunks.append({"id": cid, "path": str(meta["chunk_path"]), "x": x, "y": y, "w": cw, "h": ch})
                used_paths.append((Path(meta["chunk_path"]), list(meta.get("tile_paths") or [])))
        ws.log(f"RF-DETR Satellite started: chunks={len(chunks)}, model={model_class}")
        rows = _predict(ws, chunks, model_class, checkpoint, conf, accept_plus, device=device)
        by_id = {int(c["id"]): c for c in chunks}
        records = []
        for r in rows:
            c = by_id.get(int(r.get("chunk_id", 0)))
            if not c:
                continue
            x, y, cw, ch = float(c["x"]), float(c["y"]), float(c["w"]), float(c["h"])
            bx1, by1 = max(0.0, min(cw, float(r["x1"]))), max(0.0, min(ch, float(r["y1"])))
            bx2, by2 = max(0.0, min(cw, float(r["x2"]))), max(0.0, min(ch, float(r["y2"])))
            gx1, gy1 = x_min*WEB_TILE_SIZE+x+bx1, y_min*WEB_TILE_SIZE+y+by1
            gx2, gy2 = x_min*WEB_TILE_SIZE+x+bx2, y_min*WEB_TILE_SIZE+y+by2
            lon1, lat1 = sat_lonlat_from_world_px(gx1, gy1, z)
            lon2, lat2 = sat_lonlat_from_world_px(gx2, gy2, z)
            west, east = sorted((float(lon1), float(lon2)))
            south, north = sorted((float(lat1), float(lat2)))
            poly = [(west,north),(east,north),(east,south),(west,south),(west,north)]
            records.append({"class_id": int(r.get("class_id", 0)), "class_name": str(r.get("class_name", "object")), "confidence": float(r.get("confidence", 0.0)), "model": "RF-DETR " + model_class, "model_slot": 1, "zoom": int(z), "tile_x_min": int(x_min), "tile_y_min": int(y_min), "chunk_id": int(c["id"]), "chunk_px_x": int(x), "chunk_px_y": int(y), "bbox_px_x1": float(x+bx1), "bbox_px_y1": float(y+by1), "bbox_px_x2": float(x+bx2), "bbox_px_y2": float(y+by2), "bbox_lon_min": west, "bbox_lat_min": south, "bbox_lon_max": east, "bbox_lat_max": north, "world_px_z": int(z), "world_px_x1": float(gx1), "world_px_y1": float(gy1), "world_px_x2": float(gx2), "world_px_y2": float(gy2), "polygon_lonlat": poly})
        try:
            records = ws._satellite_deduplicate_records(records, 0.90)
        except Exception:
            pass
        ws.sat_last_x_min, ws.sat_last_y_min, ws.sat_last_z = int(x_min), int(y_min), int(z)
        ws.sat_last_records = [dict(r) for r in records]
        ws.satellite_detections = [dict(r) for r in records]
        if records:
            try:
                ws._satellite_features_to_file([dict(r) for r in records], out_path)
                ws.satellite_output_last = str(out_path)
            except Exception as exc:
                ws.log("RF-DETR satellite export warning: " + str(exc))
        try:
            ws._satellite_request_map_reload_from_worker(120)
        except Exception:
            pass
        try:
            ws.satellite_redraw_detection_overlay()
        except Exception:
            pass
        for p, used in used_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                ws._satellite_delete_tile_paths(used)
            except Exception:
                pass
        ws.log(f"RF-DETR Satellite Detection finished: {len(records)} detections")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
        ws.sat_detection_temp_cache_root = old_temp
        try:
            ws._sat_detection_thread_local.cache_root = old_thread
        except Exception:
            pass



IMG_EXTS_RFDETR = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _parse_classes_text(raw: str):
    vals = [p.strip() for p in str(raw or "").replace(";", ",").split(",") if p.strip()]
    return vals or ["object"]


def _guess_project_root(ws=None) -> Path:
    candidates = []
    for name in ("project", "project_create_dir", "project_dir", "project_folder", "project_path", "current_project_dir", "last_project_dir"):
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
            p = Path(str(c).strip().strip('"')).expanduser()
            if p.exists():
                return p
        except Exception:
            pass
    return Path.cwd()


def _guess_image_dir_from_project(project: Path) -> Path:
    candidates = [
        project / "images",
        project / "image",
        project / "dataset" / "images",
        project / "train" / "images",
        project,
    ]
    for p in candidates:
        try:
            if p.exists() and any(x.is_file() and x.suffix.lower() in IMG_EXTS_RFDETR for x in p.rglob("*")):
                return p
        except Exception:
            pass
    return project / "images"


def _guess_label_dir_from_project(project: Path) -> Path:
    candidates = [
        project / "labels",
        project / "label",
        project / "dataset" / "labels",
        project / "train" / "labels",
        project,
    ]
    for p in candidates:
        try:
            if p.exists() and any(x.is_file() and x.suffix.lower() == ".txt" for x in p.rglob("*")):
                return p
        except Exception:
            pass
    return project / "labels"


def _collect_rfdetr_images(image_dir: Path, max_images: int = 0):
    out = []
    try:
        for p in Path(image_dir).rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS_RFDETR:
                low = str(p).replace("\\", "/").lower()
                if any(skip in low for skip in ["/runs/", "/outputs/", "/output/", "/.venv/", "/venv/", "/__pycache__/", "/mustatil_rfdetr_prepared/"]):
                    continue
                out.append(p)
    except Exception:
        pass
    out = sorted(out, key=lambda p: str(p).lower())
    if max_images and int(max_images) > 0:
        out = out[: int(max_images)]
    return out


def _find_rfdetr_label_for_image(img: Path, image_dir: Path, label_dir: Path) -> Path:
    try:
        rel = img.relative_to(image_dir).with_suffix(".txt")
        cand = label_dir / rel
        if cand.exists():
            return cand
    except Exception:
        pass
    cand = label_dir / (img.stem + ".txt")
    if cand.exists():
        return cand
    try:
        matches = sorted(label_dir.rglob(img.stem + ".txt"), key=lambda p: len(str(p)))
        if matches:
            return matches[0]
    except Exception:
        pass
    return cand


def _write_rfdetr_data_yaml(root: Path, classes):
    lines = [
        "path: " + str(root).replace("\\", "/"),
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "nc: " + str(len(classes)),
        "names:",
    ]
    for i, c in enumerate(classes):
        lines.append(f"  {i}: {c}")
    (root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_rfdetr_folders(image_dir: str, label_dir: str, classes, max_images: int = 0, include_empty: bool = True, logger=None):
    def log(msg):
        if logger:
            logger(str(msg))
        else:
            _log(str(msg))
    image_dir = Path(str(image_dir).strip().strip('"')).expanduser()
    label_dir = Path(str(label_dir).strip().strip('"')).expanduser()
    images = _collect_rfdetr_images(image_dir, max_images=max_images)
    log("Image folder: " + str(image_dir))
    log("Label folder: " + str(label_dir))
    log("Images found: " + str(len(images)))
    log("Classes: " + ", ".join(classes))
    if not images:
        log("No images found.")
        return {"images": 0, "boxes": 0, "nonempty": 0, "missing_labels": 0}

    total_boxes = 0
    nonempty = 0
    missing_labels = 0
    bad_lines = 0
    checked = min(len(images), 500)
    for img in images[:checked]:
        lab = _find_rfdetr_label_for_image(img, image_dir, label_dir)
        if not lab.exists():
            missing_labels += 1
            continue
        try:
            for ln in lab.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = ln.replace(",", " ").split()
                if not parts:
                    continue
                if len(parts) < 5:
                    bad_lines += 1
                    continue
                try:
                    int(float(parts[0]))
                    [float(x) for x in parts[1:5]]
                    total_boxes += 1
                except Exception:
                    bad_lines += 1
        except Exception:
            missing_labels += 1
    # nonempty count more strictly
    for img in images[:checked]:
        lab = _find_rfdetr_label_for_image(img, image_dir, label_dir)
        try:
            if lab.exists() and lab.read_text(encoding="utf-8", errors="ignore").strip():
                nonempty += 1
        except Exception:
            pass
    log(f"Checked images: {checked}")
    log(f"Images with labels/objects: {nonempty}")
    log(f"Boxes/YOLO rows: {total_boxes}")
    log(f"Missing label txt: {missing_labels}")
    log(f"Bad label rows: {bad_lines}")
    if missing_labels and include_empty:
        log("Missing labels are allowed because Include empty/negative images is enabled.")
    return {"images": len(images), "boxes": total_boxes, "nonempty": nonempty, "missing_labels": missing_labels, "bad_lines": bad_lines}


def _prepare_rfdetr_dataset_from_folders(image_dir: str, label_dir: str, output_dir: str, classes, max_images: int = 0, include_empty: bool = True, logger=None) -> str:
    """Prepare a YOLO-style train/valid/test folder from separate image/label folders.

    This mirrors the R-CNN Trainer idea: image folder + label folder are explicit.
    RF-DETR then receives the prepared folder as dataset_dir.
    """
    def log(msg):
        if logger:
            logger(str(msg))
        else:
            _log(str(msg))
    image_dir = Path(str(image_dir).strip().strip('"')).expanduser()
    label_dir = Path(str(label_dir).strip().strip('"')).expanduser()
    out_root = Path(str(output_dir).strip().strip('"')).expanduser()
    prep = out_root / "mustatil_rfdetr_prepared"
    if prep.exists():
        shutil.rmtree(prep, ignore_errors=True)
    for split in ("train", "valid", "test"):
        (prep / split / "images").mkdir(parents=True, exist_ok=True)
        (prep / split / "labels").mkdir(parents=True, exist_ok=True)

    images = _collect_rfdetr_images(image_dir, max_images=max_images)
    pairs = []
    for img in images:
        lab = _find_rfdetr_label_for_image(img, image_dir, label_dir)
        has_label = False
        try:
            has_label = lab.exists() and bool(lab.read_text(encoding="utf-8", errors="ignore").strip())
        except Exception:
            has_label = False
        if has_label or include_empty:
            pairs.append((img, lab if lab.exists() else None))

    if not pairs:
        raise RuntimeError("No training images found after filtering. Check image folder, label folder, and Include empty/negative images.")

    # stable deterministic split
    import random
    random.seed(42)
    random.shuffle(pairs)
    split_idx = max(1, int(len(pairs) * 0.9))
    train_pairs = pairs[:split_idx]
    valid_pairs = pairs[split_idx:] or pairs[:1]

    def copy_pairs(items, split):
        for img, lab in items:
            dst_img = prep / split / "images" / img.name
            dst_lab = prep / split / "labels" / (img.stem + ".txt")
            shutil.copy2(img, dst_img)
            if lab is not None and Path(lab).exists():
                shutil.copy2(lab, dst_lab)
            else:
                dst_lab.write_text("", encoding="utf-8")

    copy_pairs(train_pairs, "train")
    copy_pairs(valid_pairs, "valid")
    copy_pairs(valid_pairs, "test")
    _write_rfdetr_data_yaml(prep, classes)
    log(f"RF-DETR prepared dataset: {prep}")
    log(f"Train images: {len(train_pairs)} | Valid images: {len(valid_pairs)} | Classes: {classes}")
    return str(prep)


def _find_recent_rfdetr_checkpoints() -> list[str]:
    roots = [_runs()]
    out = []
    for root in roots:
        try:
            for pat in ("*.pth", "*.pt", "*.ckpt"):
                out.extend(root.rglob(pat))
        except Exception:
            pass
    out = sorted({str(p) for p in out if Path(p).is_file()}, key=lambda s: Path(s).stat().st_mtime if Path(s).exists() else 0, reverse=True)
    return out[:80]


def _set_custom_model_from_checkpoint(ws, checkpoint: str):
    try:
        ws.mustatil_rfdetr_checkpoint = str(checkpoint or "").strip().strip('"')
        ws.mustatil_rfdetr_use_custom = bool(ws.mustatil_rfdetr_checkpoint)
    except Exception:
        pass


def _guess_dataset(ws=None):
    candidates = []
    for name in ("project_dir", "project_folder", "project_path", "current_project_dir", "last_project_dir"):
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
            p = Path(c)
            if (p / "images").exists() or (p / "labels").exists() or (p / "data.yaml").exists():
                return str(p)
        except Exception:
            pass
    return candidates[0] if candidates else ""


def _build_left(ws, kind):
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QGroupBox, QLabel, QPushButton, QComboBox, QFileDialog, QCheckBox, QLineEdit
    page = QWidget(); page.setObjectName("MustatilRFDETRLeftControls")
    root = QVBoxLayout(page); root.setContentsMargins(8,8,8,8)
    title = QLabel("RF-DETR"); title.setStyleSheet("font-size:16px;font-weight:bold;"); root.addWidget(title)

    setup = QGroupBox("Setup"); sg = QGridLayout(setup)
    install_btn = QPushButton("Install / Repair RF-DETR")
    plus_install = QCheckBox("install Plus / XL")
    setup_status = QLabel("RF-DETR runs in an isolated external runtime. Use cuda/cuda:0..3/multi-gpu or cpu fallback.")
    setup_status.setWordWrap(True)
    sg.addWidget(install_btn,0,0); sg.addWidget(plus_install,0,1); sg.addWidget(setup_status,1,0,1,2)
    root.addWidget(setup)

    box = QGroupBox("Detection model")
    g = QGridLayout(box)
    source_combo = QComboBox()
    source_combo.addItems(["Pretrained RF-DETR model", "Custom trained checkpoint"])
    model_combo = QComboBox(); _fill(model_combo, include_seg=False, include_plus=True)
    accept = QCheckBox("Accept PML for XL/2XL")
    device_combo = QComboBox(); device_combo.addItems(_device_combo_items())
    ckpt = QLineEdit(""); ckpt.setPlaceholderText("checkpoint_best_total.pth / .pth / .pt / .ckpt from RF-DETR Trainer")
    ckpt_btn = QPushButton("Choose custom model…")
    recent_btn = QPushButton("Latest trained")
    run_btn = QPushButton("Run RF-DETR on Detection image" if kind=="detection" else "Run RF-DETR on selected satellite map")
    open_report_btn = QPushButton("Open last report")
    status = QLabel("For your own model: choose 'Custom trained checkpoint' and select the checkpoint from RF-DETR Trainer output.")
    status.setWordWrap(True)

    g.addWidget(QLabel("Model source"),0,0); g.addWidget(source_combo,0,1,1,2)
    g.addWidget(QLabel("Architecture"),1,0); g.addWidget(model_combo,1,1,1,2)
    g.addWidget(QLabel("Device"),2,0); g.addWidget(device_combo,2,1,1,2)
    g.addWidget(accept,3,0,1,3)
    g.addWidget(QLabel("Custom model"),4,0); g.addWidget(ckpt,4,1); g.addWidget(ckpt_btn,4,2)
    g.addWidget(recent_btn,5,0,1,3)
    g.addWidget(run_btn,6,0,1,3)
    g.addWidget(open_report_btn,7,0,1,3)
    g.addWidget(status,8,0,1,3)
    root.addWidget(box)
    root.addStretch(1)

    def log(msg, target=status):
        try: target.setText(str(msg))
        except Exception: pass
        try: ws.log(str(msg))
        except Exception: pass

    def sync():
        ws.mustatil_rfdetr_model_class = _model(model_combo)
        use_custom = source_combo.currentIndex() == 1
        ws.mustatil_rfdetr_use_custom = bool(use_custom)
        ws.mustatil_rfdetr_checkpoint = ckpt.text().strip().strip('"') if use_custom else ""
        ws.mustatil_rfdetr_accept_plus = bool(accept.isChecked())
        ws.mustatil_rfdetr_device = device_combo.currentText().strip() or "cuda"
        try:
            ckpt.setEnabled(use_custom); ckpt_btn.setEnabled(use_custom); recent_btn.setEnabled(use_custom)
        except Exception:
            pass

    def install():
        def task():
            _install(ws, plus=plus_install.isChecked(), metrics=False)
            log("RF-DETR installed.", setup_status)
        _run_task(ws, "Install RF-DETR", task, allow_parallel=False)


    def browse():
        start = str(_runs())
        cps = _find_recent_rfdetr_checkpoints()
        if cps:
            start = str(Path(cps[0]).parent)
        p, _ = QFileDialog.getOpenFileName(page, "Select self-trained RF-DETR checkpoint", start, "RF-DETR checkpoints (*.pth *.pt *.ckpt);;All files (*)")
        if p:
            source_combo.setCurrentIndex(1)
            ckpt.setText(p)
            sync()
            log("Custom RF-DETR model selected: " + p)

    def latest():
        cps = _find_recent_rfdetr_checkpoints()
        if not cps:
            log("No trained RF-DETR checkpoints found yet in: " + str(_runs()))
            return
        source_combo.setCurrentIndex(1)
        ckpt.setText(cps[0])
        sync()
        log("Selected latest trained checkpoint: " + cps[0])

    def run():
        sync()
        if _is_plus(ws.mustatil_rfdetr_model_class) and not ws.mustatil_rfdetr_accept_plus:
            log("For XL/2XL: check PML acceptance."); return
        if getattr(ws, "mustatil_rfdetr_use_custom", False):
            cp = str(getattr(ws, "mustatil_rfdetr_checkpoint", "") or "")
            if not cp or not Path(cp).exists():
                log("Custom model selected, but checkpoint file does not exist.")
                return
        if kind == "detection":
            _run_task(ws, "RF-DETR Detection", lambda: _run_detection(ws), allow_parallel=False)
        else:
            _run_task(ws, "RF-DETR Satellite Detection", lambda: _run_satellite(ws), allow_parallel=True)

    def open_report():
        p = str(getattr(ws, "mustatil_rfdetr_last_report", "") or "")
        if p and Path(p).exists():
            try: os.startfile(p)
            except Exception:
                import webbrowser; webbrowser.open(p)
        else:
            log("No RF-DETR report yet.")

    install_btn.clicked.connect(install)
    ckpt_btn.clicked.connect(browse); recent_btn.clicked.connect(latest); run_btn.clicked.connect(run); open_report_btn.clicked.connect(open_report)
    source_combo.currentIndexChanged.connect(lambda *_: sync())
    model_combo.currentIndexChanged.connect(lambda *_: sync())
    device_combo.currentTextChanged.connect(lambda *_: sync())
    ckpt.textChanged.connect(lambda *_: sync())
    accept.toggled.connect(lambda *_: sync())
    sync()
    return page



def _open_rfdetr_backbone_builder(parent, ws, controls: dict):
    """AI-Pipeline-style graphical RF-DETR Backbone Strategy Builder.

    This intentionally mirrors the R-CNN visual backbone creator UI: draggable
    blocks, visible graph connections, soft logic gates, fusion gates, code view
    and auto-layout. RF-DETR does not accept a TorchVision custom backbone .py,
    so Apply writes an RF-DETR strategy JSON into the RF-DETR Trainer instead.
    """
    import json, ast, pprint, uuid
    from PySide6.QtCore import Qt, QRectF, QPointF
    from PySide6.QtGui import QColor, QBrush, QPen, QPainter, QPainterPath, QFont
    from PySide6.QtWidgets import (
        QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
        QGroupBox, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
        QCheckBox, QTextEdit, QPlainTextEdit, QSplitter, QStackedWidget,
        QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsPathItem,
        QMessageBox, QLineEdit, QScrollArea
    )

    BLOCK_TYPES = [
        "Stem Conv", "Conv Block", "Residual Block", "Bottleneck Block",
        "Depthwise Separable", "Downsample Conv", "SE Attention", "SPP Block",
        "Style Adapter", "Dilated Context", "Feature Gate", "Residual Add Gate",
        "Concat Fusion Gate", "Weighted Sum Gate", "Soft AND Gate", "Soft OR Gate",
        "Soft NOT Gate", "Soft XOR Gate", "Mixture-of-Experts Gate", "Style Gate",
        "Scale Mix Gate", "Skip Bridge", "Iterative Refinement",
    ]
    block_categories = {
        "Core CNN": ["Stem Conv", "Conv Block", "Residual Block", "Bottleneck Block", "Depthwise Separable", "Downsample Conv"],
        "Attention / Context": ["SE Attention", "SPP Block", "Style Adapter", "Dilated Context", "Feature Gate", "Style Gate", "Scale Mix Gate"],
        "Fusion Gates": ["Residual Add Gate", "Concat Fusion Gate", "Weighted Sum Gate", "Mixture-of-Experts Gate", "Skip Bridge"],
        "Soft Logic Gates": ["Soft AND Gate", "Soft OR Gate", "Soft NOT Gate", "Soft XOR Gate"],
        "Experimental": ["Iterative Refinement", "Mixture-of-Experts Gate", "Scale Mix Gate", "Style Gate"],
    }
    DEFAULT_BLOCKS = [
        {"type": "Stem Conv", "out": 32, "stride": 2, "repeats": 1, "feature": True, "x": 80.0, "y": 120.0},
        {"type": "Residual Block", "out": 64, "stride": 2, "repeats": 2, "feature": True, "x": 330.0, "y": 120.0},
        {"type": "Depthwise Separable", "out": 128, "stride": 2, "repeats": 2, "feature": True, "x": 580.0, "y": 120.0},
        {"type": "Soft AND Gate", "out": 128, "stride": 1, "repeats": 1, "feature": False, "x": 580.0, "y": 310.0},
        {"type": "Mixture-of-Experts Gate", "out": 256, "stride": 2, "repeats": 1, "feature": True, "x": 850.0, "y": 170.0},
        {"type": "Bottleneck Block", "out": 512, "stride": 2, "repeats": 2, "feature": True, "x": 1120.0, "y": 170.0},
    ]
    block_descriptions = {
        "Stem Conv": "First image-entry block: early edges, textures and relief cues.",
        "Conv Block": "Standard local feature block: stable and predictable.",
        "Residual Block": "Skip-connected block for deeper training stability.",
        "Bottleneck Block": "Higher-capacity ResNet-style bottleneck.",
        "Depthwise Separable": "Lightweight MobileNet-style block.",
        "Downsample Conv": "Forces stride-2 resolution reduction.",
        "SE Attention": "Channel attention for suppressing irrelevant texture.",
        "SPP Block": "Multi-scale context pooling for large structures.",
        "Style Adapter": "Style/contrast adaptation for satellite source changes.",
        "Dilated Context": "Wide contextual field without extra downsampling.",
        "Feature Gate": "Trainable channel gate.",
        "Residual Add Gate": "Fusion gate: residual addition after projection.",
        "Concat Fusion Gate": "Fusion gate: concatenate detail/context branches.",
        "Weighted Sum Gate": "Fusion gate: trainable weighted sum.",
        "Soft AND Gate": "Logic gate: emphasizes agreement between branches.",
        "Soft OR Gate": "Logic gate: passes features supported by any branch.",
        "Soft NOT Gate": "Logic gate: suppresses the first branch using later branches.",
        "Soft XOR Gate": "Logic gate: emphasizes disagreement/ambiguity.",
        "Mixture-of-Experts Gate": "Logic/fusion gate: learns which expert branch to trust.",
        "Style Gate": "Fusion/refinement gate for color/contrast variation.",
        "Scale Mix Gate": "Fusion gate for multi-scale branches.",
        "Skip Bridge": "Light skip connection from early detail to later blocks.",
        "Iterative Refinement": "Repeated refinement without a recurrent cycle.",
    }
    logic_types = {"Soft AND Gate", "Soft OR Gate", "Soft NOT Gate", "Soft XOR Gate", "Mixture-of-Experts Gate"}
    fusion_types = {"Residual Add Gate", "Concat Fusion Gate", "Weighted Sum Gate", "Mixture-of-Experts Gate", "Style Gate", "Scale Mix Gate", "Skip Bridge"}

    def new_id():
        return "rb_" + uuid.uuid4().hex[:8]

    def ensure_ids(blocks):
        for i, b in enumerate(blocks):
            b.setdefault("id", new_id())
            b.setdefault("name", f"{i+1}. {b.get('type','Block')}")
            b.setdefault("x", 80.0 + i * 245.0)
            b.setdefault("y", 120.0)
            b.setdefault("out", 64)
            b.setdefault("stride", 1)
            b.setdefault("repeats", 1)
            b.setdefault("feature", False)
            b.setdefault("enabled", True)
            b.setdefault("inputs", [] if i == 0 else [blocks[i-1]["id"]])
        return blocks

    blocks = ensure_ids([dict(b) for b in DEFAULT_BLOCKS])

    dlg = QDialog(parent)
    dlg.setWindowTitle("RF-DETR Backbone Creator — Visual Block Pipeline")
    dlg.resize(1500, 900)
    dlg.setMinimumSize(980, 620)

    root = QVBoxLayout(dlg)
    root.setContentsMargins(8, 8, 8, 8)
    root.setSpacing(6)

    toolbar = QHBoxLayout()
    root.addLayout(toolbar)

    category_combo = QComboBox()
    category_combo.addItems(["Core CNN", "Attention / Context", "Fusion Gates", "Soft Logic Gates", "Experimental"])
    type_add_combo = QComboBox()
    type_add_combo.setMinimumWidth(230)

    def refresh_type_combo():
        cat = str(category_combo.currentText() or "Core CNN")
        type_add_combo.clear()
        for typ in block_categories.get(cat, BLOCK_TYPES):
            type_add_combo.addItem(typ)
        type_add_combo.setToolTip(block_descriptions.get(str(type_add_combo.currentText()), "Add block"))

    category_combo.currentTextChanged.connect(lambda *_: refresh_type_combo())
    type_add_combo.currentTextChanged.connect(lambda t: type_add_combo.setToolTip(block_descriptions.get(str(t), "Add block")))
    refresh_type_combo()

    add_btn = QPushButton("+ Add block")
    dup_btn = QPushButton("Duplicate")
    del_btn = QPushButton("Delete")
    left_btn = QPushButton("←")
    right_btn = QPushButton("→")
    layout_btn = QPushButton("Auto layout")
    view_btn = QPushButton("View: Code")
    apply_btn_top = QPushButton("Apply to RF-DETR Trainer")

    toolbar.addWidget(QLabel("Block category")); toolbar.addWidget(category_combo)
    toolbar.addWidget(QLabel("Block")); toolbar.addWidget(type_add_combo); toolbar.addWidget(add_btn)
    toolbar.addWidget(dup_btn); toolbar.addWidget(del_btn); toolbar.addWidget(left_btn); toolbar.addWidget(right_btn); toolbar.addWidget(layout_btn); toolbar.addWidget(view_btn)
    toolbar.addStretch(1); toolbar.addWidget(apply_btn_top)

    main_stack = QStackedWidget()
    root.addWidget(main_stack, 1)

    canvas_page = QWidget()
    canvas_layout = QVBoxLayout(canvas_page)
    canvas_layout.setContentsMargins(0, 0, 0, 0)
    split = QSplitter()
    canvas_layout.addWidget(split, 1)

    scene = QGraphicsScene()
    scene.setSceneRect(-200, -200, 2400, 1300)

    selected_id = {"id": None}
    block_items = {}
    arrow_items = []

    class BlockItem(QGraphicsItem):
        WIDTH = 215
        HEIGHT = 128
        TYPE_COLORS = {
            "Stem Conv": QColor(75, 130, 255), "Conv Block": QColor(70, 155, 210),
            "Residual Block": QColor(60, 170, 120), "Bottleneck Block": QColor(180, 105, 220),
            "Depthwise Separable": QColor(70, 170, 170), "Downsample Conv": QColor(235, 150, 60),
            "SE Attention": QColor(225, 105, 140), "SPP Block": QColor(150, 120, 60),
            "Style Adapter": QColor(90, 120, 210), "Dilated Context": QColor(135, 95, 200),
            "Feature Gate": QColor(215, 95, 55), "Residual Add Gate": QColor(205, 120, 55),
            "Concat Fusion Gate": QColor(205, 85, 135), "Weighted Sum Gate": QColor(95, 135, 215),
            "Soft AND Gate": QColor(70, 150, 90), "Soft OR Gate": QColor(80, 165, 165),
            "Soft NOT Gate": QColor(190, 85, 85), "Soft XOR Gate": QColor(150, 90, 220),
            "Mixture-of-Experts Gate": QColor(85, 105, 225), "Style Gate": QColor(70, 120, 190),
            "Scale Mix Gate": QColor(120, 90, 210), "Skip Bridge": QColor(90, 150, 95),
            "Iterative Refinement": QColor(110, 110, 180),
        }
        def __init__(self, block):
            super().__init__()
            self.block = block
            self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemSendsGeometryChanges)
            self.setAcceptHoverEvents(True)
            self.setPos(float(block.get("x", 0)), float(block.get("y", 0)))
            self.setToolTip(block_descriptions.get(str(block.get("type")), "RF-DETR block"))

        def boundingRect(self):
            return QRectF(0, 0, self.WIDTH, self.HEIGHT)

        def input_anchor(self):
            return self.scenePos() + QPointF(0, self.HEIGHT / 2)

        def output_anchor(self):
            return self.scenePos() + QPointF(self.WIDTH, self.HEIGHT / 2)

        def paint(self, painter, option, widget=None):
            b = self.block
            typ = str(b.get("type", "Conv Block"))
            color = QColor(self.TYPE_COLORS.get(typ, QColor(120, 120, 120)))
            if not bool(b.get("enabled", True)):
                color = QColor(145, 145, 145)
            header = QColor(max(0, color.red() - 35), max(0, color.green() - 35), max(0, color.blue() - 35))
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setPen(QPen(QColor(20, 20, 20) if self.isSelected() else QColor(80, 80, 80), 3 if self.isSelected() else 1.4))
            painter.setBrush(QBrush(color))
            painter.drawRoundedRect(self.boundingRect(), 12, 12)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(header))
            path = QPainterPath()
            path.addRoundedRect(QRectF(0, 0, self.WIDTH, 30), 12, 12)
            path.addRect(QRectF(0, 14, self.WIDTH, 18))
            painter.drawPath(path)
            painter.setPen(QPen(QColor(45, 45, 45), 1))
            painter.setBrush(QBrush(QColor(245, 245, 245)))
            painter.drawEllipse(QPointF(0, self.HEIGHT / 2), 7, 7)
            painter.drawEllipse(QPointF(self.WIDTH, self.HEIGHT / 2), 7, 7)
            painter.setPen(QPen(QColor("white")))
            painter.setFont(QFont("Arial", 8, QFont.Bold))
            painter.drawText(QRectF(10, 4, self.WIDTH - 20, 22), Qt.AlignLeft | Qt.AlignVCenter, typ.upper())
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            painter.drawText(QRectF(12, 36, self.WIDTH - 24, 22), Qt.AlignLeft | Qt.AlignVCenter, str(b.get("name", typ))[:26])
            painter.setFont(QFont("Arial", 8))
            painter.setPen(QPen(QColor(30, 30, 30)))
            painter.drawText(QRectF(12, 65, self.WIDTH - 24, 18), Qt.AlignLeft | Qt.AlignVCenter, f"channels {int(b.get('out',64))} | stride {int(b.get('stride',1))} | x{int(b.get('repeats',1))}")
            if typ in logic_types:
                line2 = "SOFT LOGIC FUNCTION"
            elif typ in fusion_types:
                line2 = "fusion / gate block"
            elif bool(b.get("feature", False)):
                line2 = "RF-DETR feature hint"
            else:
                line2 = "internal strategy block"
            painter.drawText(QRectF(12, 86, self.WIDTH - 24, 18), Qt.AlignLeft | Qt.AlignVCenter, line2)
            if bool(b.get("feature", False)):
                painter.setBrush(QBrush(QColor(255, 255, 255, 220)))
                painter.setPen(QPen(QColor(40, 100, 40), 1))
                painter.drawRoundedRect(QRectF(self.WIDTH - 60, self.HEIGHT - 27, 49, 18), 5, 5)
                painter.setPen(QPen(QColor(20, 100, 20)))
                painter.drawText(QRectF(self.WIDTH - 60, self.HEIGHT - 27, 49, 18), Qt.AlignCenter, "FEAT")

        def mouseDoubleClickEvent(self, event):
            select_block(str(self.block.get("id")))
            super().mouseDoubleClickEvent(event)

        def itemChange(self, change, value):
            if change == QGraphicsItem.ItemPositionHasChanged:
                p = self.pos()
                self.block["x"] = float(p.x())
                self.block["y"] = float(p.y())
                refresh_arrows()
            if change == QGraphicsItem.ItemSelectedHasChanged and bool(value):
                select_block(str(self.block.get("id")))
            return super().itemChange(change, value)

    class ArrowItem(QGraphicsPathItem):
        def __init__(self, source_id, target_id):
            super().__init__()
            self.source_id = source_id
            self.target_id = target_id
            self.setAcceptHoverEvents(True)
            self.setZValue(-10)
            self.setPen(QPen(QColor(50, 50, 50), 2.4))

        def update_path(self):
            s = block_items.get(self.source_id)
            t = block_items.get(self.target_id)
            if not s or not t:
                return
            a = s.output_anchor()
            b = t.input_anchor()
            dx = max(70.0, abs(b.x() - a.x()) * 0.45)
            path = QPainterPath(a)
            path.cubicTo(a + QPointF(dx, 0), b - QPointF(dx, 0), b)
            self.setPath(path)

        def hoverEnterEvent(self, event):
            self.setPen(QPen(QColor(200, 60, 40), 3.4))
            super().hoverEnterEvent(event)

        def hoverLeaveEvent(self, event):
            self.setPen(QPen(QColor(50, 50, 50), 2.4))
            super().hoverLeaveEvent(event)

        def mousePressEvent(self, event):
            if event.button() == Qt.RightButton:
                remove_connection(self.source_id, self.target_id)
                event.accept()
                return
            super().mousePressEvent(event)

    class GraphView(QGraphicsView):
        def __init__(self, scene):
            super().__init__(scene)
            self._connect_source = None
            self._temp_path = None
            self._panning = False
            self._pan_last = None
            self.setRenderHint(QPainter.Antialiasing, True)
            self.setDragMode(QGraphicsView.RubberBandDrag)
            self.setMouseTracking(True)

        def _scene_pos(self, event):
            try:
                return self.mapToScene(event.position().toPoint())
            except Exception:
                return self.mapToScene(event.pos())

        def _block_at_socket(self, pos, socket):
            best = None
            best_dist = 999999.0
            for item in block_items.values():
                anchor = item.output_anchor() if socket == "output" else item.input_anchor()
                d = ((anchor.x() - pos.x()) ** 2 + (anchor.y() - pos.y()) ** 2) ** 0.5
                if d < 18 and d < best_dist:
                    best = item
                    best_dist = d
            return best

        def mousePressEvent(self, event):
            pos = self._scene_pos(event)
            if event.button() == Qt.LeftButton:
                src = self._block_at_socket(pos, "output")
                if src is not None:
                    self._connect_source = src
                    self._temp_path = QGraphicsPathItem()
                    self._temp_path.setPen(QPen(QColor(30, 120, 220), 2, Qt.DashLine))
                    scene.addItem(self._temp_path)
                    event.accept()
                    return
                if self.itemAt(event.position().toPoint() if hasattr(event, "position") else event.pos()) is None:
                    self._panning = True
                    try: self._pan_last = event.position().toPoint()
                    except Exception: self._pan_last = event.pos()
                    self.setCursor(Qt.ClosedHandCursor)
                    event.accept()
                    return
            if event.button() == Qt.MiddleButton:
                self._panning = True
                try: self._pan_last = event.position().toPoint()
                except Exception: self._pan_last = event.pos()
                self.setCursor(Qt.ClosedHandCursor)
                event.accept()
                return
            super().mousePressEvent(event)

        def mouseMoveEvent(self, event):
            if self._connect_source is not None and self._temp_path is not None:
                a = self._connect_source.output_anchor()
                b = self._scene_pos(event)
                dx = max(70.0, abs(b.x() - a.x()) * 0.45)
                path = QPainterPath(a)
                path.cubicTo(a + QPointF(dx, 0), b - QPointF(dx, 0), b)
                self._temp_path.setPath(path)
                event.accept()
                return
            if self._panning and self._pan_last is not None:
                try: p = event.position().toPoint()
                except Exception: p = event.pos()
                dx = p.x() - self._pan_last.x()
                dy = p.y() - self._pan_last.y()
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - dx)
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() - dy)
                self._pan_last = p
                event.accept()
                return
            super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event):
            if self._connect_source is not None:
                pos = self._scene_pos(event)
                target = self._block_at_socket(pos, "input")
                if self._temp_path is not None:
                    try: scene.removeItem(self._temp_path)
                    except Exception: pass
                src = self._connect_source
                self._connect_source = None
                self._temp_path = None
                if target is not None and target.block.get("id") != src.block.get("id"):
                    add_connection(str(src.block.get("id")), str(target.block.get("id")))
                event.accept()
                return
            if self._panning:
                self._panning = False
                self._pan_last = None
                self.unsetCursor()
                event.accept()
                return
            super().mouseReleaseEvent(event)

    view = GraphView(scene)
    view.setMinimumWidth(860)

    right = QWidget()
    right_lay = QVBoxLayout(right)
    right_lay.setContentsMargins(8, 0, 0, 0)

    prop_box = QGroupBox("Selected block properties")
    form = QFormLayout(prop_box)
    name_edit = QLineEdit()
    type_combo = QComboBox(); type_combo.addItems(BLOCK_TYPES)
    out_spin = QSpinBox(); out_spin.setRange(4, 4096); out_spin.setValue(64)
    stride_combo = QComboBox(); stride_combo.addItems(["1", "2"])
    repeats_spin = QSpinBox(); repeats_spin.setRange(1, 32); repeats_spin.setValue(1)
    feature_check = QCheckBox("Use as RF-DETR feature hint")
    enabled_check = QCheckBox("Enabled")
    enabled_check.setChecked(True)
    input_label = QLabel("No inputs")
    input_label.setWordWrap(True)
    desc_label = QLabel("Select a block.")
    desc_label.setWordWrap(True)
    form.addRow("Name", name_edit)
    form.addRow("Type", type_combo)
    form.addRow("Channels", out_spin)
    form.addRow("Stride", stride_combo)
    form.addRow("Repeats", repeats_spin)
    form.addRow("Feature", feature_check)
    form.addRow("Enabled", enabled_check)
    form.addRow("Inputs", input_label)
    form.addRow("Description", desc_label)
    prop_scroll = QScrollArea()
    prop_scroll.setWidgetResizable(True)
    prop_scroll.setWidget(prop_box)
    right_lay.addWidget(prop_scroll, 2)

    actions = QGroupBox("Block actions")
    ag = QGridLayout(actions)
    fit_btn = QPushButton("Fit canvas")
    clear_inputs_btn = QPushButton("Clear selected inputs")
    export_json_btn = QPushButton("Apply strategy")
    ag.addWidget(fit_btn, 0, 0)
    ag.addWidget(clear_inputs_btn, 0, 1)
    ag.addWidget(export_json_btn, 1, 0, 1, 2)
    right_lay.addWidget(actions)

    help_label = QLabel(
        "Connect blocks by dragging from the right socket of one block to the left socket of another. "
        "Right-click a connection to delete it. Logic blocks work like in the R-CNN builder: "
        "Soft AND / OR / NOT / XOR and Mixture-of-Experts are preserved in the RF-DETR strategy JSON."
    )
    help_label.setWordWrap(True)
    right_lay.addWidget(help_label)

    summary = QTextEdit()
    summary.setReadOnly(True)
    summary.setMinimumHeight(180)
    right_lay.addWidget(QLabel("Builder log / strategy summary"))
    right_lay.addWidget(summary, 1)

    split.addWidget(view)
    split.addWidget(right)
    split.setSizes([950, 520])
    main_stack.addWidget(canvas_page)

    code_page = QWidget()
    code_lay = QVBoxLayout(code_page)
    code_help = QLabel(
        "Code view: edit the RF-DETR visual graph as a Python literal. "
        "Click 'Apply code to blocks' to update the canvas."
    )
    code_help.setWordWrap(True)
    code_lay.addWidget(code_help)
    code_editor = QPlainTextEdit()
    code_lay.addWidget(code_editor, 1)
    code_buttons = QHBoxLayout()
    apply_code_btn = QPushButton("Apply code to blocks")
    refresh_code_btn = QPushButton("Refresh code from blocks")
    code_buttons.addWidget(apply_code_btn)
    code_buttons.addWidget(refresh_code_btn)
    code_buttons.addStretch(1)
    code_lay.addLayout(code_buttons)
    main_stack.addWidget(code_page)

    # RF-DETR global settings row below canvas/code
    bottom = QGroupBox("RF-DETR training settings generated by the graph")
    bg = QGridLayout(bottom)
    model_combo = QComboBox(); _fill(model_combo, include_seg=True, include_plus=True)
    device_combo = QComboBox(); device_combo.addItems(_device_combo_items())
    freeze = QCheckBox("Freeze backbone / encoder where supported")
    epochs = QSpinBox(); epochs.setRange(1, 1000); epochs.setValue(int(controls.get("epochs").value() if controls.get("epochs") else 20))
    batch = QSpinBox(); batch.setRange(1, 64); batch.setValue(int(controls.get("batch").value() if controls.get("batch") else 1))
    grad = QSpinBox(); grad.setRange(1, 64); grad.setValue(int(controls.get("grad").value() if controls.get("grad") else 4))
    lr = QDoubleSpinBox(); lr.setDecimals(7); lr.setRange(0.0000001, 1.0); lr.setSingleStep(0.00001); lr.setValue(float(controls.get("lr").value() if controls.get("lr") else 0.0001))
    backbone_lr = QDoubleSpinBox(); backbone_lr.setDecimals(8); backbone_lr.setRange(0.0, 1.0); backbone_lr.setSingleStep(0.000001); backbone_lr.setValue(float(controls.get("backbone_lr").value() if controls.get("backbone_lr") else 0.00001))
    encoder_lr = QDoubleSpinBox(); encoder_lr.setDecimals(8); encoder_lr.setRange(0.0, 1.0); encoder_lr.setSingleStep(0.000001); encoder_lr.setValue(float(controls.get("encoder_lr").value() if controls.get("encoder_lr") else 0.00005))
    resolution = QSpinBox(); resolution.setRange(0, 4096); resolution.setSingleStep(64); resolution.setValue(int(controls.get("resolution").value() if controls.get("resolution") else 768))
    queries = QSpinBox(); queries.setRange(0, 5000); queries.setSingleStep(50); queries.setValue(int(controls.get("queries").value() if controls.get("queries") else 300))

    try: device_combo.setCurrentText(str(controls.get("device_combo").currentText()))
    except Exception: pass
    try:
        want = str(controls.get("model_combo").currentData())
        for i in range(model_combo.count()):
            if str(model_combo.itemData(i)) == want:
                model_combo.setCurrentIndex(i); break
    except Exception: pass

    r = 0
    bg.addWidget(QLabel("RF-DETR model"), r, 0); bg.addWidget(model_combo, r, 1)
    bg.addWidget(QLabel("Device"), r, 2); bg.addWidget(device_combo, r, 3)
    bg.addWidget(freeze, r, 4); r += 1
    bg.addWidget(QLabel("Epochs"), r, 0); bg.addWidget(epochs, r, 1)
    bg.addWidget(QLabel("Batch"), r, 2); bg.addWidget(batch, r, 3)
    bg.addWidget(QLabel("Grad accum"), r, 4); bg.addWidget(grad, r, 5); r += 1
    bg.addWidget(QLabel("Main LR"), r, 0); bg.addWidget(lr, r, 1)
    bg.addWidget(QLabel("Backbone LR"), r, 2); bg.addWidget(backbone_lr, r, 3)
    bg.addWidget(QLabel("Encoder LR"), r, 4); bg.addWidget(encoder_lr, r, 5); r += 1
    bg.addWidget(QLabel("Resolution"), r, 0); bg.addWidget(resolution, r, 1)
    bg.addWidget(QLabel("Object queries"), r, 2); bg.addWidget(queries, r, 3)
    root.addWidget(bottom)

    # ---------------- data/model helpers ----------------
    def current_block():
        bid = selected_id.get("id")
        for b in blocks:
            if b.get("id") == bid:
                return b
        return None

    def normalize_blocks():
        # keep only existing input refs and avoid self-cycles directly
        ids = {b.get("id") for b in blocks}
        for b in blocks:
            cleaned = []
            for ref in list(b.get("inputs", []) or []):
                if ref in ids and ref != b.get("id") and ref not in cleaned:
                    cleaned.append(ref)
            b["inputs"] = cleaned

    def strategy_dict():
        normalize_blocks()
        logic_blocks = [dict(b) for b in blocks if b.get("type") in logic_types and bool(b.get("enabled", True))]
        fusion_blocks = [dict(b) for b in blocks if b.get("type") in fusion_types and bool(b.get("enabled", True))]
        feature_blocks = [dict(b) for b in blocks if b.get("feature") and bool(b.get("enabled", True))]
        gate_count = len(logic_blocks) + len(fusion_blocks)
        max_channels = max([int(b.get("out", 64)) for b in blocks if bool(b.get("enabled", True))] or [64])
        graph_complexity = len([b for b in blocks if bool(b.get("enabled", True))]) + gate_count * 2
        data = {
            "version": 7,
            "kind": "rfdetr_visual_backbone_graph",
            "note": "Graphical RF-DETR strategy from Mustatil. Blocks/logic gates are preserved as RF-DETR training hints; unsupported RF-DETR train kwargs are skipped safely by the worker.",
            "model": str(model_combo.currentData() or "RFDETRSmall"),
            "device": str(device_combo.currentText() or "cuda"),
            "blocks": [dict(b) for b in blocks],
            "logic_gates": logic_blocks,
            "fusion_gates": fusion_blocks,
            "feature_blocks": feature_blocks,
            "graph_stats": {
                "enabled_blocks": len([b for b in blocks if bool(b.get("enabled", True))]),
                "logic_blocks": len(logic_blocks),
                "fusion_blocks": len(fusion_blocks),
                "feature_blocks": len(feature_blocks),
                "max_channels": max_channels,
                "complexity": graph_complexity,
            },
            "train_kwargs": {
                "backbone_lr": float(backbone_lr.value()),
                "encoder_lr": float(encoder_lr.value()),
                "resolution": int(resolution.value()),
                "num_queries": int(queries.value()),
                "multi_gpu": str(device_combo.currentText()).lower() == "multi-gpu",
                "freeze_backbone": bool(freeze.isChecked()),
                "freeze_encoder": bool(freeze.isChecked()),
                "visual_graph_complexity": int(graph_complexity),
                "visual_max_channels": int(max_channels),
                "visual_logic_gate_count": int(len(logic_blocks)),
                "visual_fusion_gate_count": int(len(fusion_blocks)),
            },
        }
        return data

    def strategy_to_python():
        body = pprint.pformat(strategy_dict(), width=120, sort_dicts=False)
        return (
            "# Mustatil RF-DETR Backbone Creator - Python code view\n"
            "# Edit values below, then click 'Apply code to blocks'.\n"
            "# This is parsed safely as Python data, not executed.\n\n"
            f"rfdetr_backbone_strategy = {body}\n"
        )

    def log(msg):
        summary.append(str(msg))
        try: ws.log("RF-DETR Backbone Builder: " + str(msg).replace("\n", " | "))
        except Exception: pass

    def refresh_code():
        code_editor.setPlainText(strategy_to_python())

    def update_summary():
        data = strategy_dict()
        lines = [
            f"Blocks: {data['graph_stats']['enabled_blocks']}",
            f"Logic gates: {data['graph_stats']['logic_blocks']}",
            f"Fusion gates: {data['graph_stats']['fusion_blocks']}",
            f"Feature hints: {data['graph_stats']['feature_blocks']}",
            f"Model: {data['model']}",
            f"Device: {data['device']}",
            f"Resolution: {data['train_kwargs']['resolution']}",
            f"Object queries: {data['train_kwargs']['num_queries']}",
        ]
        summary.setPlainText("\n".join(lines))
        refresh_code()

    def refresh_arrows():
        for a in list(arrow_items):
            try: scene.removeItem(a)
            except Exception: pass
        arrow_items.clear()
        normalize_blocks()
        for tgt in blocks:
            for src in list(tgt.get("inputs", []) or []):
                if src in block_items and tgt.get("id") in block_items:
                    item = ArrowItem(src, tgt.get("id"))
                    scene.addItem(item)
                    item.update_path()
                    arrow_items.append(item)

    def refresh_scene():
        for item in list(scene.items()):
            scene.removeItem(item)
        block_items.clear()
        for b in blocks:
            item = BlockItem(b)
            scene.addItem(item)
            block_items[b["id"]] = item
        refresh_arrows()
        update_summary()

    def select_block(bid):
        selected_id["id"] = bid
        b = current_block()
        if not b:
            return
        try:
            name_edit.blockSignals(True); type_combo.blockSignals(True); out_spin.blockSignals(True)
            stride_combo.blockSignals(True); repeats_spin.blockSignals(True); feature_check.blockSignals(True); enabled_check.blockSignals(True)
            name_edit.setText(str(b.get("name", "")))
            type_combo.setCurrentText(str(b.get("type", "Conv Block")))
            out_spin.setValue(int(b.get("out", 64)))
            stride_combo.setCurrentText(str(int(b.get("stride", 1))))
            repeats_spin.setValue(int(b.get("repeats", 1)))
            feature_check.setChecked(bool(b.get("feature", False)))
            enabled_check.setChecked(bool(b.get("enabled", True)))
            input_names = []
            for ref in b.get("inputs", []) or []:
                rb = next((x for x in blocks if x.get("id") == ref), None)
                input_names.append(str(rb.get("name", ref)) if rb else str(ref))
            input_label.setText(", ".join(input_names) if input_names else "No inputs")
            typ = str(b.get("type", "Conv Block"))
            desc_label.setText(block_descriptions.get(typ, "RF-DETR block"))
        finally:
            name_edit.blockSignals(False); type_combo.blockSignals(False); out_spin.blockSignals(False)
            stride_combo.blockSignals(False); repeats_spin.blockSignals(False); feature_check.blockSignals(False); enabled_check.blockSignals(False)

    def save_editor():
        b = current_block()
        if not b:
            return
        b["name"] = name_edit.text().strip() or b.get("type", "Block")
        b["type"] = str(type_combo.currentText() or "Conv Block")
        b["out"] = int(out_spin.value())
        b["stride"] = int(stride_combo.currentText() or "1")
        b["repeats"] = int(repeats_spin.value())
        b["feature"] = bool(feature_check.isChecked())
        b["enabled"] = bool(enabled_check.isChecked())
        try:
            block_items[b["id"]].update()
        except Exception:
            pass
        select_block(b["id"])
        update_summary()

    def add_connection(src, tgt):
        tb = next((b for b in blocks if b.get("id") == tgt), None)
        if not tb:
            return
        ins = list(tb.get("inputs", []) or [])
        if src not in ins:
            ins.append(src)
        tb["inputs"] = ins
        select_block(tgt)
        refresh_arrows()
        update_summary()

    def remove_connection(src, tgt):
        tb = next((b for b in blocks if b.get("id") == tgt), None)
        if tb:
            tb["inputs"] = [x for x in list(tb.get("inputs", []) or []) if x != src]
        refresh_arrows()
        update_summary()

    def add_block(typ):
        idx = len(blocks)
        b = {
            "id": new_id(),
            "name": f"{idx+1}. {typ}",
            "type": typ,
            "out": 128 if "Gate" in typ else 64,
            "stride": 1 if "Gate" in typ or "Attention" in typ else 2,
            "repeats": 1,
            "feature": typ in {"Residual Block", "Bottleneck Block", "SPP Block", "Mixture-of-Experts Gate"},
            "enabled": True,
            "inputs": [blocks[-1]["id"]] if blocks else [],
            "x": 100.0 + idx * 235.0,
            "y": 130.0 + (idx % 3) * 170.0,
        }
        blocks.append(b)
        refresh_scene()
        try: block_items[b["id"]].setSelected(True)
        except Exception: pass
        select_block(b["id"])

    def duplicate_selected():
        b = current_block()
        if not b:
            return
        nb = dict(b)
        nb["id"] = new_id()
        nb["name"] = str(b.get("name", "Block")) + " copy"
        nb["x"] = float(b.get("x", 0)) + 40
        nb["y"] = float(b.get("y", 0)) + 40
        blocks.append(nb)
        refresh_scene()
        select_block(nb["id"])

    def delete_selected():
        b = current_block()
        if not b:
            return
        bid = b["id"]
        blocks[:] = [x for x in blocks if x.get("id") != bid]
        for x in blocks:
            x["inputs"] = [i for i in list(x.get("inputs", []) or []) if i != bid]
        selected_id["id"] = None
        refresh_scene()

    def move_selected(delta):
        b = current_block()
        if not b:
            return
        i = blocks.index(b)
        j = max(0, min(len(blocks)-1, i + int(delta)))
        if i != j:
            blocks[i], blocks[j] = blocks[j], blocks[i]
            refresh_scene()
            select_block(b["id"])

    def auto_layout():
        for i, b in enumerate(blocks):
            b["x"] = 80.0 + (i % 5) * 255.0
            b["y"] = 120.0 + (i // 5) * 190.0
        refresh_scene()

    def clear_selected_inputs():
        b = current_block()
        if b:
            b["inputs"] = []
            refresh_arrows()
            select_block(b["id"])
            update_summary()

    def apply_code_to_blocks():
        text = code_editor.toPlainText().strip()
        if not text:
            return
        try:
            tree = ast.parse(text, mode="exec")
            node = None
            for n in tree.body:
                if isinstance(n, ast.Assign):
                    for t in n.targets:
                        if isinstance(t, ast.Name) and t.id in {"rfdetr_backbone_strategy", "strategy", "pipeline"}:
                            node = n.value
                            break
                if node is not None:
                    break
            if node is None and len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
                node = tree.body[0].value
            if node is None:
                raise ValueError("Expected: rfdetr_backbone_strategy = {...}")
            data = ast.literal_eval(node)
            new_blocks = data.get("blocks", data if isinstance(data, list) else [])
            if not isinstance(new_blocks, list):
                raise ValueError("No blocks list found.")
            blocks[:] = ensure_ids([dict(b) for b in new_blocks])
            # Apply high-level settings too.
            if isinstance(data, dict):
                if data.get("model"):
                    for i in range(model_combo.count()):
                        if str(model_combo.itemData(i)) == str(data.get("model")):
                            model_combo.setCurrentIndex(i); break
                if data.get("device"):
                    device_combo.setCurrentText(str(data.get("device")))
                tk = data.get("train_kwargs", {}) or {}
                if "resolution" in tk: resolution.setValue(int(tk.get("resolution") or 0))
                if "num_queries" in tk: queries.setValue(int(tk.get("num_queries") or 0))
                if "backbone_lr" in tk: backbone_lr.setValue(float(tk.get("backbone_lr") or 0))
                if "encoder_lr" in tk: encoder_lr.setValue(float(tk.get("encoder_lr") or 0))
                if "freeze_backbone" in tk: freeze.setChecked(bool(tk.get("freeze_backbone")))
            refresh_scene()
            log("Code applied to visual RF-DETR blocks.")
        except Exception as exc:
            QMessageBox.warning(dlg, "RF-DETR Backbone Builder", "Could not apply code:\n" + str(exc))

    def toggle_view():
        if main_stack.currentIndex() == 0:
            refresh_code()
            main_stack.setCurrentIndex(1)
            view_btn.setText("View: Blocks")
        else:
            main_stack.setCurrentIndex(0)
            view_btn.setText("View: Code")

    def apply_to_trainer():
        data = strategy_dict()
        try:
            controls["model_combo"].setCurrentText(model_combo.currentText())
        except Exception:
            try:
                for i in range(controls["model_combo"].count()):
                    if str(controls["model_combo"].itemData(i)) == str(data["model"]):
                        controls["model_combo"].setCurrentIndex(i); break
            except Exception:
                pass
        try: controls["device_combo"].setCurrentText(str(data["device"]))
        except Exception: pass
        for key, widget, value in [
            ("epochs", controls.get("epochs"), int(epochs.value())),
            ("batch", controls.get("batch"), int(batch.value())),
            ("grad", controls.get("grad"), int(grad.value())),
            ("lr", controls.get("lr"), float(lr.value())),
            ("backbone_lr", controls.get("backbone_lr"), float(backbone_lr.value())),
            ("encoder_lr", controls.get("encoder_lr"), float(encoder_lr.value())),
            ("resolution", controls.get("resolution"), int(resolution.value())),
            ("queries", controls.get("queries"), int(queries.value())),
        ]:
            try:
                widget.setValue(value)
            except Exception:
                pass
        try: controls["freeze"].setChecked(bool(freeze.isChecked()))
        except Exception: pass
        try: controls["strategy_json"].setPlainText(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception: pass
        try: ws.log("RF-DETR graphical backbone strategy applied.")
        except Exception: pass
        dlg.accept()

    # signal wiring
    add_btn.clicked.connect(lambda: add_block(str(type_add_combo.currentText() or "Conv Block")))
    dup_btn.clicked.connect(duplicate_selected)
    del_btn.clicked.connect(delete_selected)
    left_btn.clicked.connect(lambda: move_selected(-1))
    right_btn.clicked.connect(lambda: move_selected(1))
    layout_btn.clicked.connect(auto_layout)
    view_btn.clicked.connect(toggle_view)
    apply_btn_top.clicked.connect(apply_to_trainer)
    export_json_btn.clicked.connect(apply_to_trainer)
    fit_btn.clicked.connect(lambda: view.fitInView(scene.itemsBoundingRect().adjusted(-80, -80, 80, 80), Qt.KeepAspectRatio))
    clear_inputs_btn.clicked.connect(clear_selected_inputs)
    apply_code_btn.clicked.connect(apply_code_to_blocks)
    refresh_code_btn.clicked.connect(refresh_code)

    for w in [name_edit, type_combo, out_spin, stride_combo, repeats_spin, feature_check, enabled_check]:
        try:
            if hasattr(w, "textChanged"):
                w.textChanged.connect(lambda *_: save_editor())
            if hasattr(w, "currentIndexChanged"):
                w.currentIndexChanged.connect(lambda *_: save_editor())
            if hasattr(w, "valueChanged"):
                w.valueChanged.connect(lambda *_: save_editor())
            if hasattr(w, "toggled"):
                w.toggled.connect(lambda *_: save_editor())
        except Exception:
            pass
    for w in [model_combo, device_combo, freeze, epochs, batch, grad, lr, backbone_lr, encoder_lr, resolution, queries]:
        try:
            if hasattr(w, "currentIndexChanged"):
                w.currentIndexChanged.connect(lambda *_: update_summary())
            if hasattr(w, "valueChanged"):
                w.valueChanged.connect(lambda *_: update_summary())
            if hasattr(w, "toggled"):
                w.toggled.connect(lambda *_: update_summary())
        except Exception:
            pass

    refresh_scene()
    if blocks:
        select_block(blocks[0]["id"])
    log(
        "Visual RF-DETR Backbone Creator ready. Drag blocks, connect sockets, use Soft AND/OR/NOT/XOR and fusion gates, then Apply strategy."
    )
    dlg.exec()


def _build_trainer(ws):
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QGroupBox, QLabel, QPushButton, QComboBox, QFileDialog, QCheckBox, QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QSplitter
    page = QWidget(); page.setObjectName("MustatilRFDETRTrainerTab")
    root = QVBoxLayout(page); root.setContentsMargins(8,8,8,8)
    title = QLabel("RF-DETR Trainer"); title.setStyleSheet("font-size:18px;font-weight:bold;"); root.addWidget(title)

    splitter = QSplitter()
    root.addWidget(splitter, 1)
    controls = QWidget(); left = QVBoxLayout(controls); left.setContentsMargins(0,0,8,0)
    splitter.addWidget(controls)
    log_edit = QTextEdit(); log_edit.setReadOnly(True); log_edit.setMinimumWidth(420)
    splitter.addWidget(log_edit)
    try: splitter.setSizes([560, 720])
    except Exception: pass

    project = _guess_project_root(ws)
    image_guess = _guess_image_dir_from_project(project)
    label_guess = _guess_label_dir_from_project(project)
    out_guess = project / "runs" / "rfdetr_training"

    def append(msg):
        try:
            log_edit.append(str(msg))
            log_edit.ensureCursorVisible()
        except Exception:
            pass
        try: ws.log(str(msg))
        except Exception: pass

    setup = QGroupBox("Setup")
    sg = QGridLayout(setup)
    install_btn = QPushButton("Install / Repair RF-DETR")
    plus_install = QCheckBox("install Plus / XL")
    metrics = QCheckBox("metrics / TensorBoard")
    setup_status = QLabel("RF-DETR uses an isolated external runtime. Install / Repair installs light training deps only, avoiding heavy MLflow/logger packages. Trainer is R-CNN-style: image folder + label folder + classes.")
    setup_status.setWordWrap(True)
    sg.addWidget(install_btn,0,0); sg.addWidget(plus_install,0,1); sg.addWidget(metrics,0,2)
    sg.addWidget(setup_status,1,0,1,3)
    left.addWidget(setup)

    data_box = QGroupBox("Dataset")
    dg = QGridLayout(data_box)
    image_line = QLineEdit(str(image_guess))
    label_line = QLineEdit(str(label_guess))
    out_line = QLineEdit(str(out_guess))
    classes = QLineEdit("mustatil,false_positive")
    include_empty = QCheckBox("Include empty / negative images")
    include_empty.setChecked(True)
    max_images = QSpinBox(); max_images.setRange(0, 1000000); max_images.setValue(0); max_images.setToolTip("0 = all images")
    img_btn = QPushButton("…"); lab_btn = QPushButton("…"); out_btn = QPushButton("…")
    validate_btn = QPushButton("Validate Dataset")

    r=0
    dg.addWidget(QLabel("Image folder"),r,0); dg.addWidget(image_line,r,1); dg.addWidget(img_btn,r,2); r+=1
    dg.addWidget(QLabel("Label folder"),r,0); dg.addWidget(label_line,r,1); dg.addWidget(lab_btn,r,2); r+=1
    dg.addWidget(QLabel("Output folder"),r,0); dg.addWidget(out_line,r,1); dg.addWidget(out_btn,r,2); r+=1
    dg.addWidget(QLabel("Classes"),r,0); dg.addWidget(classes,r,1,1,2); r+=1
    dg.addWidget(include_empty,r,0,1,2); dg.addWidget(QLabel("Max images"),r,2); dg.addWidget(max_images,r,3); r+=1
    dg.addWidget(validate_btn,r,0,1,4)
    left.addWidget(data_box)

    model_box = QGroupBox("Model / Training")
    g = QGridLayout(model_box)
    model_combo = QComboBox(); _fill(model_combo, include_seg=True, include_plus=True)
    weight_source = QComboBox(); weight_source.addItems(["Pretrained RF-DETR weights", "Custom / resume checkpoint"])
    accept = QCheckBox("Accept PML for XL/2XL")
    device_combo = QComboBox(); device_combo.addItems(_device_combo_items())
    resume = QLineEdit(""); resume.setPlaceholderText("Optional: resume/fine-tune from checkpoint_best_total.pth")
    resume_btn = QPushButton("Choose checkpoint…")
    latest_btn = QPushButton("Latest trained")
    epochs = QSpinBox(); epochs.setRange(1,1000); epochs.setValue(20)
    batch = QSpinBox(); batch.setRange(1,64); batch.setValue(1)
    grad = QSpinBox(); grad.setRange(1,64); grad.setValue(4)
    lr = QDoubleSpinBox(); lr.setDecimals(7); lr.setRange(0.0000001,1.0); lr.setSingleStep(0.0001); lr.setValue(0.0001)
    early = QCheckBox("early stopping")
    tensor = QCheckBox("TensorBoard")
    freeze_backbone = QCheckBox("Freeze backbone / encoder where supported")
    backbone_lr = QDoubleSpinBox(); backbone_lr.setDecimals(8); backbone_lr.setRange(0.0, 1.0); backbone_lr.setSingleStep(0.000001); backbone_lr.setValue(0.00001)
    encoder_lr = QDoubleSpinBox(); encoder_lr.setDecimals(8); encoder_lr.setRange(0.0, 1.0); encoder_lr.setSingleStep(0.000001); encoder_lr.setValue(0.00005)
    resolution = QSpinBox(); resolution.setRange(0, 4096); resolution.setSingleStep(64); resolution.setValue(768)
    queries = QSpinBox(); queries.setRange(0, 5000); queries.setSingleStep(50); queries.setValue(300)
    strategy_json = QTextEdit(); strategy_json.setMaximumHeight(96); strategy_json.setPlaceholderText("RF-DETR backbone strategy JSON from Backbone Builder")
    builder_btn = QPushButton("Backbone Builder")
    train_btn = QPushButton("Start Training")
    all_btn = QPushButton("Train all core detection models")
    open_btn = QPushButton("Open output")
    use_last_btn = QPushButton("Use latest trained model in Detection")

    r=0
    g.addWidget(QLabel("Architecture"),r,0); g.addWidget(model_combo,r,1,1,5); r+=1
    g.addWidget(QLabel("Weights"),r,0); g.addWidget(weight_source,r,1,1,5); r+=1
    g.addWidget(accept,r,1,1,5); r+=1
    g.addWidget(QLabel("Device"),r,0); g.addWidget(device_combo,r,1,1,5); r+=1
    g.addWidget(QLabel("Checkpoint"),r,0); g.addWidget(resume,r,1,1,3); g.addWidget(resume_btn,r,4); g.addWidget(latest_btn,r,5); r+=1
    g.addWidget(QLabel("Epochs"),r,0); g.addWidget(epochs,r,1); g.addWidget(QLabel("Batch"),r,2); g.addWidget(batch,r,3); g.addWidget(QLabel("Grad accum"),r,4); g.addWidget(grad,r,5); r+=1
    g.addWidget(QLabel("LR"),r,0); g.addWidget(lr,r,1); g.addWidget(early,r,2); g.addWidget(tensor,r,3); r+=1
    g.addWidget(freeze_backbone,r,0,1,3); g.addWidget(builder_btn,r,3,1,3); r+=1
    g.addWidget(QLabel("Backbone LR"),r,0); g.addWidget(backbone_lr,r,1); g.addWidget(QLabel("Encoder LR"),r,2); g.addWidget(encoder_lr,r,3); r+=1
    g.addWidget(QLabel("Resolution"),r,0); g.addWidget(resolution,r,1); g.addWidget(QLabel("Object queries"),r,2); g.addWidget(queries,r,3); r+=1
    g.addWidget(QLabel("Strategy JSON"),r,0); g.addWidget(strategy_json,r,1,1,5); r+=1
    g.addWidget(train_btn,r,0,1,2); g.addWidget(all_btn,r,2,1,2); g.addWidget(open_btn,r,4,1,2); r+=1
    g.addWidget(use_last_btn,r,0,1,6)
    left.addWidget(model_box)
    left.addStretch(1)

    def browse_dir(line, title):
        p = QFileDialog.getExistingDirectory(page, title, line.text() or str(project))
        if p: line.setText(p)

    def install():
        def task():
            _install(ws, plus=plus_install.isChecked(), metrics=metrics.isChecked())
            try: setup_status.setText("RF-DETR installed.")
            except Exception: pass
            append("RF-DETR installed/repaired.")
        _run_task(ws, "Install RF-DETR", task, allow_parallel=False)


    def validate():
        append("Validating RF-DETR dataset...")
        _validate_rfdetr_folders(
            image_line.text(), label_line.text(), _parse_classes_text(classes.text()),
            max_images=max_images.value(), include_empty=include_empty.isChecked(), logger=append
        )

    def choose_resume():
        start = str(_runs())
        cps = _find_recent_rfdetr_checkpoints()
        if cps:
            start = str(Path(cps[0]).parent)
        p, _ = QFileDialog.getOpenFileName(page, "Select RF-DETR checkpoint", start, "RF-DETR checkpoints (*.pth *.pt *.ckpt);;All files (*)")
        if p:
            weight_source.setCurrentIndex(1)
            resume.setText(p)

    def latest():
        cps = _find_recent_rfdetr_checkpoints()
        if not cps:
            append("No trained RF-DETR checkpoints found yet in: " + str(_runs()))
            return
        weight_source.setCurrentIndex(1)
        resume.setText(cps[0])
        append("Selected latest trained checkpoint: " + cps[0])

    def current_model_class():
        return _model(model_combo)

    def run_one(cls, suffix):
        if _is_plus(cls) and not accept.isChecked():
            raise RuntimeError("XL/2XL selected: check PML acceptance.")
        base_out = Path(out_line.text().strip().strip('"') or str(out_guess))
        out = base_out / suffix
        out.mkdir(parents=True, exist_ok=True)
        class_list = _parse_classes_text(classes.text())
        prepared = _prepare_rfdetr_dataset_from_folders(
            image_line.text(), label_line.text(), str(out), class_list,
            max_images=max_images.value(), include_empty=include_empty.isChecked(), logger=append
        )
        cmd = [
            str(_runtime_python()), str(_worker()), "--mode", "train",
            "--model-class", cls,
            "--dataset-dir", prepared,
            "--output-dir", str(out),
            "--classes", ",".join(class_list),
            "--epochs", str(epochs.value()),
            "--batch-size", str(batch.value()),
            "--grad-accum-steps", str(grad.value()),
            "--lr", str(float(lr.value())),
            "--device", device_combo.currentText().strip() or "cuda",
        ]
        if strategy_json.toPlainText().strip():
            cmd += ["--backbone-strategy", strategy_json.toPlainText().strip()]
        if freeze_backbone.isChecked():
            cmd += ["--freeze-backbone"]
        if float(backbone_lr.value()) > 0:
            cmd += ["--backbone-lr", str(float(backbone_lr.value()))]
        if float(encoder_lr.value()) > 0:
            cmd += ["--encoder-lr", str(float(encoder_lr.value()))]
        if int(resolution.value()) > 0:
            cmd += ["--resolution", str(int(resolution.value()))]
        if int(queries.value()) > 0:
            cmd += ["--num-queries", str(int(queries.value()))]
        if weight_source.currentIndex() == 1 and resume.text().strip():
            cmd += ["--resume", resume.text().strip().strip('"')]
        if accept.isChecked(): cmd += ["--accept-plus"]
        if early.isChecked(): cmd += ["--early-stopping"]
        if tensor.isChecked(): cmd += ["--tensorboard"]
        append("Training command: " + " ".join(map(str, cmd)))
        _run_logged(ws, cmd, "RF-DETR train", cwd=_runtime())

    def train_selected():
        cls = current_model_class()
        def task():
            try:
                train_btn.setEnabled(False); train_btn.setText("Training running...")
            except Exception:
                pass
            try:
                append("Training RF-DETR: " + cls)
                run_one(cls, cls)
                append("Training finished: " + cls)
            finally:
                try:
                    train_btn.setEnabled(True); train_btn.setText("Start Training")
                except Exception:
                    pass
        _run_task(ws, "RF-DETR Train " + cls, task, allow_parallel=False)

    def train_all():
        core = ["RFDETRNano","RFDETRSmall","RFDETRMedium","RFDETRLarge","RFDETRBase"]
        def task():
            for cls in core:
                append("Training " + cls)
                run_one(cls, cls)
            append("All core detection models finished.")
        _run_task(ws, "RF-DETR Train All Core", task, allow_parallel=False)

    def open_out():
        p = Path(out_line.text().strip().strip('"') or str(out_guess))
        p.mkdir(parents=True, exist_ok=True)
        try: os.startfile(str(p))
        except Exception:
            import webbrowser; webbrowser.open(str(p))

    def use_latest_in_detection():
        cps = _find_recent_rfdetr_checkpoints()
        if not cps:
            append("No trained RF-DETR checkpoint found.")
            return
        _set_custom_model_from_checkpoint(ws, cps[0])
        append("Latest trained checkpoint stored for Detection: " + cps[0])

    def open_backbone_builder():
        controls = {
            "model_combo": model_combo,
            "device_combo": device_combo,
            "epochs": epochs,
            "batch": batch,
            "grad": grad,
            "lr": lr,
            "freeze": freeze_backbone,
            "backbone_lr": backbone_lr,
            "encoder_lr": encoder_lr,
            "resolution": resolution,
            "queries": queries,
            "strategy_json": strategy_json,
        }
        _open_rfdetr_backbone_builder(page, ws, controls)

    install_btn.clicked.connect(install)
    img_btn.clicked.connect(lambda: browse_dir(image_line, "Select image folder"))
    lab_btn.clicked.connect(lambda: browse_dir(label_line, "Select label folder"))
    out_btn.clicked.connect(lambda: browse_dir(out_line, "Select output folder"))
    validate_btn.clicked.connect(validate)
    resume_btn.clicked.connect(choose_resume); latest_btn.clicked.connect(latest)
    train_btn.clicked.connect(train_selected); all_btn.clicked.connect(train_all); open_btn.clicked.connect(open_out); use_last_btn.clicked.connect(use_latest_in_detection); builder_btn.clicked.connect(open_backbone_builder)

    return page


def _tab_texts(tw):
    try: return [str(tw.tabText(i) or "") for i in range(tw.count())]
    except Exception: return []


def _has_label(tw, *needles):
    lows = [t.lower() for t in _tab_texts(tw)]
    return any(n.lower() in low for low in lows for n in needles)


def _kind(tw):
    text = " ".join(_tab_texts(tw)).lower()
    cur = tw
    try:
        for _ in range(40):
            if cur is None: break
            text += " " + str(cur.objectName() or "").lower()
            cur = cur.parentWidget()
    except Exception: pass
    return "satellite" if "sat" in text else "detection"



def _tabwidget_has_training_labels(tw):
    try:
        texts = _tab_texts(tw)
    except Exception:
        texts = []
    for t in texts:
        low = str(t or "").strip().lower()
        if "trainer" in low or "training" in low:
            return True
    return False


def _tabwidget_has_detection_context(tw):
    """Detect Detection/Satellite inner model tab rows.

    RF-DETR should appear in the model row with ADAF/LAE-DINO/YOLO, not in the
    trainer row. Parent containers may also contain trainer tabs, so we do not
    reject purely because another tab elsewhere says Trainer.
    """
    try:
        texts = _tab_texts(tw)
    except Exception:
        texts = []
    if not texts:
        return False

    low = [str(t or "").strip().lower() for t in texts]
    low_all = " | ".join(low)

    # Never treat the actual RF-DETR Trainer tab row as Detection.
    if any(("rf-detr trainer" in t or "rfdetr trainer" in t) for t in low):
        return False

    # Strong Detection row signals.
    if any(_is_adaf_tab_label(t) for t in texts):
        return True
    if any("lae-dino" in t for t in low):
        return True

    # Fallback model-row signal.
    has_yolo_or_original = any(("yolo" in t or "original" in t) for t in low)
    has_other_model = any(k in low_all for k in ["owl", "grounding", "sam", "dino", "adaf"])
    return bool(has_yolo_or_original and has_other_model)


def _remove_rfdetr_trainer_tabs_outside_training_group(tw):
    """Remove only RF-DETR Trainer tabs from non-training model-tab rows."""
    try:
        texts = _tab_texts(tw)
        is_training_container = any(
            ("u-net" in str(t).lower() or "unet" in str(t).lower() or "r-cnn" in str(t).lower() or "rcnn" in str(t).lower() or "lae-dino trainer" in str(t).lower() or "trainer" in str(t).lower())
            for t in texts
        )
        # If this is the inner Detection/Satellite model row, it may have LAE-DINO but must not have RF-DETR Trainer.
        if _tabwidget_has_detection_context(tw):
            is_training_container = False
        if not is_training_container:
            for i in range(tw.count() - 1, -1, -1):
                label = str(tw.tabText(i) or "").strip().lower()
                obj = str(tw.widget(i).objectName() or "")
                if ("rf-detr trainer" in label or obj == "MustatilRFDETRTrainerTab"):
                    old = tw.widget(i)
                    tw.removeTab(i)
                    try:
                        old.deleteLater()
                    except Exception:
                        pass
            return True
    except Exception:
        pass
    return False



def _is_rfdetr_detection_widget(tw, idx):
    try:
        label = str(tw.tabText(idx) or "").strip().lower()
        obj = str(tw.widget(idx).objectName() or "")
        return ("rf-detr" in label or "rfdetr" in label) and obj != "MustatilRFDETRTrainerTab"
    except Exception:
        return False


def _is_adaf_tab_label(text):
    low = str(text or "").strip().lower()
    return "adaf" in low or "automatic detection of archaeological features" in low


def _find_rfdetr_detection_target_index(tw):
    """Prefer direct placement next to ADAF in Detection, fallback to LAE-DINO."""
    try:
        texts = _tab_texts(tw)
    except Exception:
        texts = []
    # User request: Detection point beside ADAF.
    for i, t in enumerate(texts):
        if _is_adaf_tab_label(t):
            return i
    # Fallback: beside LAE-DINO.
    for i, t in enumerate(texts):
        if _is_lae_tab(t):
            return i
    # Fallback: beside YOLO/Original.
    for i, t in enumerate(texts):
        low = str(t or "").lower()
        if "yolo" in low or "original" in low:
            return i
    return None




def _infer_kind(tw):
    """Infer whether this RF-DETR model tab belongs to Detection or Satellite.

    Safe fallback: detection. This prevents startup crashes if Mustatil or another
    plugin changes object names/tab labels.
    """
    try:
        texts = " | ".join(_tab_texts(tw)).lower()
    except Exception:
        texts = ""
    try:
        names = []
        cur = tw
        for _ in range(8):
            if cur is None:
                break
            try:
                names.append(str(cur.objectName() or ""))
            except Exception:
                pass
            try:
                par = cur.parent()
            except Exception:
                par = None
            cur = par
        names = " | ".join(names).lower()
    except Exception:
        names = ""
    blob = texts + " | " + names
    if "satellite" in blob or "webmap" in blob or "web map" in blob:
        return "satellite"
    return "detection"


def _ensure_inner(tw):
    """Insert/move RF-DETR Detection only into Detection/Satellite model tab rows.

    Placement rule:
      1. directly right of ADAF when ADAF exists
      2. otherwise right of LAE-DINO
      3. otherwise right of YOLO/Original
    """
    try:
        if not _tabwidget_has_detection_context(tw):
            return False

        _remove_rfdetr_trainer_tabs_outside_training_group(tw)

        ws = _ws(tw)
        target = _find_rfdetr_detection_target_index(tw)
        if target is None:
            return False

        # If RF-DETR exists, ensure it is directly right of ADAF/LAE target.
        existing = None
        for i in range(tw.count()):
            if _is_rfdetr_detection_widget(tw, i):
                existing = i
                break

        desired = min(tw.count(), target + 1)

        if existing is not None:
            # If already exactly in the right place, done.
            if existing == desired:
                _PATCHED_INNER.add(id(tw))
                return True

            # Move existing RF-DETR tab without rebuilding the page.
            widget = tw.widget(existing)
            label = tw.tabText(existing)
            try:
                icon = tw.tabIcon(existing)
            except Exception:
                icon = None
            tw.removeTab(existing)
            if existing < desired:
                desired -= 1
            desired = min(tw.count(), max(0, desired))
            if icon is not None:
                try:
                    tw.insertTab(desired, widget, icon, label)
                except Exception:
                    tw.insertTab(desired, widget, label)
            else:
                tw.insertTab(desired, widget, label)
            _PATCHED_INNER.add(id(tw))
            try:
                if ws:
                    ws.log("RF-DETR Detection tab moved next to ADAF/LAE-DINO.")
            except Exception:
                pass
            return True

        try:
            kind = _infer_kind(tw)
        except Exception:
            kind = "detection"
        insert_at = min(tw.count(), target + 1)
        tw.insertTab(insert_at, _build_left(ws, kind), "RF-DETR")
        _PATCHED_INNER.add(id(tw))
        try:
            if ws:
                if any(_is_adaf_tab_label(t) for t in _tab_texts(tw)):
                    ws.log("RF-DETR Detection tab inserted directly next to ADAF.")
                elif kind == "satellite":
                    ws.log("RF-DETR Detection tab inserted next to LAE-DINO in satellite.")
                else:
                    ws.log("RF-DETR Detection tab inserted next to LAE-DINO in detection.")
        except Exception:
            pass
        return True
    except Exception as exc:
        _log("ensure inner failed: "+str(exc)); traceback.print_exc(); return False


def _is_unet_trainer_label(text):
    low = str(text or "").strip().lower().replace("_", "-")
    return ("u-net" in low or "unet" in low or "u net" in low) and ("trainer" in low or "training" in low)


def _find_rfdetr_training_anchor_index(tw):
    """Place RF-DETR in the same training-tab cluster as R-CNN/U-Net.

    Preferred order:
      U-Net Trainer -> RF-DETR Trainer
      fallback: R-CNN Trainer -> RF-DETR Trainer
      fallback: LAE-DINO Trainer -> RF-DETR Trainer
      fallback: any trainer tab
    """
    try:
        texts = _tab_texts(tw)
    except Exception:
        texts = []
    for i, t in enumerate(texts):
        if _is_unet_trainer_label(t):
            return i
    for i, t in enumerate(texts):
        low = str(t or "").strip().lower().replace("_", "-")
        if ("r-cnn" in low or "rcnn" in low or "mask r-cnn" in low or "faster r-cnn" in low) and ("trainer" in low or "training" in low):
            return i
    for i, t in enumerate(texts):
        if _is_lae_trainer(t):
            return i
    for i, t in enumerate(texts):
        low = str(t or "").lower().replace("_", "-")
        if "u-net" in low or "unet" in low or "u net" in low or "r-cnn" in low or "rcnn" in low:
            return i
    for i, t in enumerate(texts):
        low = str(t or "").lower()
        if "trainer" in low or "training" in low:
            return i
    return None


def _ensure_trainer(tw):
    try:
        anchor = _find_rfdetr_training_anchor_index(tw)
        if anchor is None:
            return False

        # Fast path: RF-DETR Trainer already exists directly right of the chosen training anchor.
        # This prevents expensive delete/rebuild loops during late plugin scans.
        for i in range(tw.count()):
            try:
                label = str(tw.tabText(i) or "").strip().lower()
                obj = str(tw.widget(i).objectName() or "")
                is_rfd = ("rf-detr" in label or "rfdetr" in label or obj == "MustatilRFDETRTrainerTab")
                if is_rfd:
                    if i == anchor + 1:
                        _PATCHED_TRAINER.add(id(tw))
                        return True
                    break
            except Exception:
                pass

        # Remove RF-DETR Trainer duplicates only once, then insert exactly once.
        for i in range(tw.count()-1, -1, -1):
            try:
                label = str(tw.tabText(i) or "").strip().lower()
                obj = str(tw.widget(i).objectName() or "")
                if "rf-detr" in label or "rfdetr" in label or obj == "MustatilRFDETRTrainerTab":
                    old = tw.widget(i)
                    tw.removeTab(i)
                    try:
                        old.deleteLater()
                    except Exception:
                        pass
                    if i < anchor:
                        anchor -= 1
            except Exception:
                pass

        ws = _ws(tw)
        insert_at = min(tw.count(), anchor + 1)
        tw.insertTab(insert_at, _build_trainer(ws), "RF-DETR Trainer")
        _PATCHED_TRAINER.add(id(tw))
        try:
            if ws:
                ws.log("RF-DETR Trainer tab inserted in the training tab group next to U-Net/R-CNN Trainer.")
        except Exception:
            pass
        return True
    except Exception as exc:
        _log("ensure trainer failed: "+str(exc)); traceback.print_exc(); return False


def _scan(root=None):
    try:
        from PySide6.QtWidgets import QApplication, QTabWidget
    except Exception:
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

    if not widgets:
        try:
            app = QApplication.instance()
            if app:
                for w in app.topLevelWidgets():
                    try:
                        widgets += w.findChildren(QTabWidget)
                        if isinstance(w, QTabWidget):
                            widgets.append(w)
                    except Exception:
                        pass
        except Exception:
            pass

    seen=set(); ok=False
    for tw in widgets:
        if id(tw) in seen:
            continue
        seen.add(id(tw))
        try:
            if _tabwidget_has_detection_context(tw):
                if _ensure_inner(tw): ok=True
            else:
                _remove_rfdetr_trainer_tabs_outside_training_group(tw)
        except Exception:
            pass
        try:
            # Trainer placement only on tab rows that actually contain trainer tabs.
            if _tabwidget_has_training_labels(tw):
                if _ensure_trainer(tw): ok=True
        except Exception:
            pass
    return ok


def _install_hook(root=None):
    global _PATCHED, _ORIG_ADD, _ORIG_INSERT
    try:
        from PySide6.QtWidgets import QTabWidget
        from PySide6.QtCore import QTimer
    except Exception as exc:
        _log("Qt unavailable: "+str(exc)); return
    if not _PATCHED:
        _ORIG_ADD = QTabWidget.addTab
        _ORIG_INSERT = QTabWidget.insertTab
        def after(tw, label):
            try:
                low=str(label or "").lower()
                if any(k in low for k in ["lae","dino","trainer","training","owl","grounding","yolo","detection","satellite","adaf"]):
                    for ms in (20, 350, 1400):
                        QTimer.singleShot(ms, lambda tw=tw: (_ensure_inner(tw), _ensure_trainer(tw) if _tabwidget_has_training_labels(tw) else False))
            except Exception: pass
        def add(self, page, *args, **kwargs):
            res = _ORIG_ADD(self, page, *args, **kwargs)
            label = next((a for a in reversed(args) if isinstance(a,str)), "")
            if not label:
                try: label = self.tabText(int(res))
                except Exception: pass
            after(self, label); return res
        def insert(self, index, page, *args, **kwargs):
            res = _ORIG_INSERT(self, index, page, *args, **kwargs)
            label = next((a for a in reversed(args) if isinstance(a,str)), "")
            if not label:
                try: label = self.tabText(int(res))
                except Exception: pass
            after(self, label); return res
        QTabWidget.addTab = add
        QTabWidget.insertTab = insert
        _PATCHED = True
        _log("QTabWidget hook installed")
    for ms in (80, 350, 1200, 2500, 5000, 9000):
        QTimer.singleShot(ms, lambda root=root: _scan(root))


def mustatil_plugin_init():
    _install_hook(); return True

def register_plugin(app=None, main_window=None):
    _install_hook(main_window or app)
    try: _scan(main_window or app)
    except Exception: pass
    return True

def init_plugin(app=None, main_window=None): return register_plugin(app, main_window)
def load_plugin(app=None, main_window=None): return register_plugin(app, main_window)

try:
    _install_hook()
except Exception:
    traceback.print_exc()
