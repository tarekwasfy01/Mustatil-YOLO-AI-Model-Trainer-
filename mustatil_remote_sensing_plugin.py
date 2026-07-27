#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mustatil plugin: Google OWLv2 tab integration for existing Detection + Satellite Detection tabs.

Single-file drop-in plugin. It patches only the left controls area of the existing Detection and Satellite Detection pages with inner tabs:
  - YOLO / Original
  - Google OWL (OWLv2 only)
  - Grounding DINO
  - LAE-DINO
Class filter / Geo NMS controls are shown on all model pages.

Designed for the current Mustatil Qt workspace layout where Detection and Satellite Detection
are normal QTabWidget pages.
"""
from __future__ import annotations

import json, math, os, shutil, threading, time, traceback, subprocess, tempfile, sys, urllib.request, zipfile, re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_PATCHED_QTAB = False
_ORIG_ADD = None
_ORIG_INSERT = None
_PATCHED_PAGES = set()
_PATCHED_WS = set()
_SCAN_TIMER = None

WEB_TILE_SIZE = 256


def _log(msg: str):
    try: print("[Mustatil Google OWLv2 Tabs] " + str(msg))
    except Exception: pass


def _workspace_from_widget(widget: Any) -> Optional[Any]:
    cur = widget
    for _ in range(120):
        if cur is None: break
        try:
            if hasattr(cur, "tabs") and (hasattr(cur, "dets") or hasattr(cur, "satellite_detections")):
                return cur
        except Exception: pass
        try: cur = cur.parentWidget()
        except Exception:
            try: cur = cur.parent()
            except Exception: break
    return None


def _get_var(v, default=""):
    try:
        if hasattr(v, "get"): return v.get()
    except Exception: pass
    return default


def _set_var(v, value):
    try:
        if hasattr(v, "set"):
            v.set(value); return True
    except Exception: pass
    return False


def _find_left_layout(page: Any):
    try:
        from PySide6.QtWidgets import QSplitter, QScrollArea
        for sp in page.findChildren(QSplitter):
            if sp.count() >= 1:
                left = sp.widget(0)
                if isinstance(left, QScrollArea):
                    w = left.widget()
                    if w is not None and w.layout() is not None: return w.layout()
                if left is not None and left.layout() is not None: return left.layout()
    except Exception: pass
    try: return page.layout()
    except Exception: return None


def _insert_after_text(layout: Any, box: Any, needles: Iterable[str]) -> None:
    try:
        best = -1
        for i in range(layout.count()):
            item = layout.itemAt(i); w = item.widget() if item else None
            txt = ""
            if w is not None:
                try:
                    if hasattr(w, "title"): txt += " " + str(w.title())
                    for ch in w.findChildren(object):
                        try:
                            if hasattr(ch, "text") and callable(ch.text): txt += " " + str(ch.text())
                            if hasattr(ch, "title") and callable(ch.title): txt += " " + str(ch.title())
                        except Exception: pass
                except Exception: pass
            low = txt.lower()
            if any(n.lower() in low for n in needles): best = i
        if best >= 0:
            layout.insertWidget(best + 1, box); return
    except Exception: pass
    try: layout.insertWidget(max(0, layout.count()-1), box)
    except Exception: layout.addWidget(box)


def _item_get(obj: Any, keys: Iterable[str], default=None):
    if isinstance(obj, dict):
        for k in keys:
            if k in obj: return obj.get(k)
        return default
    for k in keys:
        try:
            if hasattr(obj, k): return getattr(obj, k)
        except Exception: pass
    return default


def _item_cls(obj: Any) -> Optional[int]:
    val = _item_get(obj, ("cls", "class_id", "class", "category_id", "label_id"), None)
    try:
        if val is not None and str(val).strip() != "": return int(float(val))
    except Exception: pass
    lab = str(_item_get(obj, ("label", "class_name", "name", "status"), "") or "").lower().strip()
    if lab in {"positive", "mustatil", "true_positive", "true positive"}: return 0
    if lab in {"false_positive", "false positive", "false-positive", "fp"}: return 1
    return None


def _item_label(obj: Any) -> str:
    return str(_item_get(obj, ("label", "class_name", "name", "status"), "") or "")


def _item_score(obj: Any) -> float:
    v = _item_get(obj, ("confidence", "conf", "score", "probability"), 0.0)
    try:
        f = float(v); return 0.0 if math.isnan(f) else f
    except Exception: return 0.0


def _bbox(obj: Any) -> Optional[Tuple[float,float,float,float]]:
    candidates = [
        ("x1","y1","x2","y2"), ("px_x1","px_y1","px_x2","px_y2"),
        ("bbox_px_x1","bbox_px_y1","bbox_px_x2","bbox_px_y2"),
        ("world_px_x1","world_px_y1","world_px_x2","world_px_y2"),
        ("left","top","right","bottom"),
    ]
    for keys in candidates:
        vals = [_item_get(obj, (k,), None) for k in keys]
        if all(v is not None for v in vals):
            try:
                x1,y1,x2,y2 = [float(v) for v in vals]
                if x2 < x1: x1,x2 = x2,x1
                if y2 < y1: y1,y2 = y2,y1
                return x1,y1,x2,y2
            except Exception: pass
    raw = _item_get(obj, ("pixel_bbox", "bbox", "box"), None)
    if raw is not None:
        try:
            vals = [float(p) for p in (raw.strip("[]() ").replace(";",",").split(",") if isinstance(raw,str) else list(raw))[:4]]
            x1,y1,x2,y2 = vals
            if x2 < x1: x1,x2=x2,x1
            if y2 < y1: y1,y2=y2,y1
            return x1,y1,x2,y2
        except Exception: pass
    return None


def _iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1,ix2,iy2=max(ax1,bx1),max(ay1,by1),min(ax2,bx2),min(ay2,by2)
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    aa=max(0,ax2-ax1)*max(0,ay2-ay1); bb=max(0,bx2-bx1)*max(0,by2-by1)
    den=aa+bb-inter
    return 0.0 if den <= 0 else inter/den


def _nms(items: Iterable[Any], thr: float) -> List[Any]:
    data=list(items or [])
    ordered=sorted(enumerate(data), key=lambda p: _item_score(p[1]), reverse=True)
    kept=[]
    for idx,it in ordered:
        b=_bbox(it)
        if b is None:
            kept.append((idx,it,b)); continue
        cls=_item_cls(it); suppress=False
        for _,kit,kb in kept:
            if kb is None: continue
            if _item_cls(kit) != cls: continue
            if _iou(b,kb) >= thr:
                suppress=True; break
        if not suppress: kept.append((idx,it,b))
    kept.sort(key=lambda p:p[0])
    return [p[1] for p in kept]


def _selected_class(ws):
    try:
        v=getattr(ws,"mustatil_owl_selected_class_id",None)
        return None if v is None or v == "" else int(v)
    except Exception: return None


def _apply_filters(items, ws):
    out=[]; sel=_selected_class(ws); hide=bool(getattr(ws,"mustatil_owl_hide_fp",False))
    for it in list(items or []):
        cls=_item_cls(it); lab=_item_label(it).lower()
        if hide and (cls == 1 or lab in {"false_positive","false positive","false-positive","fp"}): continue
        if sel is not None and cls != sel: continue
        out.append(it)
    if bool(getattr(ws,"mustatil_owl_geo_nms_enabled",False)):
        out=_nms(out, float(getattr(ws,"mustatil_owl_geo_nms_iou",0.35)))
    return out


def _patch_workspace_filters(ws):
    if ws is None or id(ws) in _PATCHED_WS: return
    _PATCHED_WS.add(id(ws))
    try:
        old_visible = getattr(ws,"visible",None)
        if callable(old_visible) and not getattr(ws,"_owl_safe_visible_patched",False):
            def visible_filtered(*a, **k): return _apply_filters(old_visible(*a, **k), ws)
            ws.visible = visible_filtered
            ws._owl_safe_visible_patched = True
    except Exception as exc: _log("visible patch failed: "+str(exc))
    try:
        old_sat = getattr(ws,"_satellite_visible_records",None)
        if callable(old_sat) and not getattr(ws,"_owl_safe_sat_visible_patched",False):
            def sat_filtered(records=None, *a, **k): return _apply_filters(old_sat(records, *a, **k), ws)
            ws._satellite_visible_records = sat_filtered
            ws._owl_safe_sat_visible_patched = True
    except Exception as exc: _log("sat visible patch failed: "+str(exc))


def _redraw(ws):
    for call in (lambda: ws.redraw(fit=False), lambda: ws.redraw(), lambda: ws.satellite_redraw_detection_overlay(), lambda: ws.refresh_layers()):
        try: call()
        except Exception: pass
    try:
        sig=getattr(getattr(ws,"signals",None),"sat_overlay_redraw_requested",None)
        if sig is not None: sig.emit(0)
    except Exception: pass


def _refresh_combos(ws):
    combos=list(getattr(ws,"mustatil_owl_class_combos",[]) or [])
    known={0:"positive / mustatil", 1:"false_positive"}
    for it in list(getattr(ws,"dets",[]) or []) + list(getattr(ws,"satellite_detections",[]) or []) + list(getattr(ws,"sat_last_records",[]) or []):
        cls=_item_cls(it)
        if cls is None: continue
        lab=_item_label(it).strip() or ("false_positive" if cls==1 else "positive / mustatil" if cls==0 else f"class {cls}")
        known[int(cls)] = lab
    for combo in combos:
        try:
            cur=_selected_class(ws)
            combo.blockSignals(True); combo.clear(); combo.addItem("All classes", None)
            for cls in sorted(known): combo.addItem(f"Class {cls}: {known[cls]}", int(cls))
            ix=0
            for i in range(combo.count()):
                if combo.itemData(i) == cur: ix=i; break
            combo.setCurrentIndex(ix); combo.blockSignals(False)
        except Exception: pass


def _owl_labels(ws) -> List[str]:
    text=""
    try: text=str(ws.mustatil_owl_classes_edit.text())
    except Exception: pass
    if not text: text=str(getattr(ws,"mustatil_owl_classes_text","") or "")
    vals=[p.strip() for p in text.replace(";",",").split(",") if p.strip()]
    return vals or ["house","car","airplane","mustatil","mound"]


def _owl_backend_name(ws) -> str:
    return "owlv2"


def _owl_device(ws) -> str:
    try: raw=str(ws.mustatil_owl_device_combo.currentText()).strip().lower()
    except Exception: raw="cpu"
    return raw if raw in {"cpu","cuda","auto"} else "cpu"


def _load_owl_model(ws):
    device_req=_owl_device(ws)
    import torch
    device = "cuda" if (device_req in {"cuda","auto"} and torch.cuda.is_available()) else "cpu"
    key=("owlv2",device)
    cache=getattr(ws,"_mustatil_owl_model_cache",{}) or {}
    if key in cache: return cache[key]
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    mid="google/owlv2-base-patch16-ensemble"
    proc=Owlv2Processor.from_pretrained(mid)
    model=Owlv2ForObjectDetection.from_pretrained(mid).to(device)
    model.eval()
    cache[key]=(proc,model,device,mid)
    ws._mustatil_owl_model_cache=cache
    try: ws.log(f"Google OWLv2 loaded: {mid} on {device}")
    except Exception: pass
    return cache[key]


def _owl_detect_pil(ws, pil_image, conf: float) -> List[Dict[str,Any]]:
    labels=_owl_labels(ws)
    proc, model, device, mid = _load_owl_model(ws)
    import torch
    inputs=proc(text=[labels], images=pil_image, return_tensors="pt")
    inputs={k:(v.to(device) if hasattr(v,"to") else v) for k,v in inputs.items()}
    with torch.no_grad(): outputs=model(**inputs)
    target_sizes=torch.tensor([pil_image.size[::-1]], device=device)
    processed=proc.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=float(conf))
    out=[]
    if not processed: return out
    r=processed[0]
    for b,s,lid in zip(r.get("boxes",[]), r.get("scores",[]), r.get("labels",[])):
        li=int(lid.detach().cpu().item() if hasattr(lid,"detach") else lid)
        lab=labels[li] if li < len(labels) else str(li)
        box=b.detach().cpu().tolist() if hasattr(b,"detach") else list(b)
        score=float(s.detach().cpu().item() if hasattr(s,"detach") else s)
        if len(box)>=4:
            x1,y1,x2,y2=map(float,box[:4])
            if x2>x1 and y2>y1:
                out.append({"class_id":li,"class_name":lab,"label":lab,"confidence":score,"score":score,"x1":x1,"y1":y1,"x2":x2,"y2":y2,"model":mid})
    return out


def _run_owl_detection_image(ws):
    try:
        from PIL import Image
        from mustatil_legacy_backend import Det
        img_path=str(_get_var(getattr(ws,"image",None),"") or "").strip().strip('"')
        if not img_path or not Path(img_path).exists(): raise RuntimeError("No Detection image selected.")
        conf=max(0.001,min(1.0,float(_get_var(getattr(ws,"conf",None),0.15))))
        tile=max(128,int(_get_var(getattr(ws,"tile",None),768) or 768))
        overlap=max(0,min(tile-1,int(_get_var(getattr(ws,"overlap",None),128) or 128)))
        step=max(1,tile-overlap)
        img=Image.open(img_path).convert("RGB")
        W,H=img.size
        ws.log(f"Google OWL Detection started: {Path(img_path).name} {W}x{H}, tile={tile}, overlap={overlap}, conf={conf}")
        dets=[]; count=0
        for y in range(0,H,step):
            for x in range(0,W,step):
                x2=min(W,x+tile); y2=min(H,y+tile)
                crop=img.crop((x,y,x2,y2))
                local=_owl_detect_pil(ws,crop,conf)
                for r in local:
                    dets.append(Det(0, str(r.get("model","Google OWL")), int(r.get("class_id",0)), float(r.get("confidence",0.0)), x+float(r["x1"]), y+float(r["y1"]), x+float(r["x2"]), y+float(r["y2"])))
                count += 1
                if count % 10 == 0: ws.log(f"Google OWL Detection tiles processed: {count}; detections={len(dets)}")
                if x2>=W: break
            if y+tile>=H: break
        ws.dets=dets
        try: ws.loadprev()
        except Exception: pass
        _refresh_combos(ws); _redraw(ws)
        ws.log(f"Google OWL Detection finished: {len(dets)} detections")
    except Exception as exc:
        try: ws.show_error("Google OWL Detection", str(exc))
        except Exception: _log("OWL image detection failed: "+str(exc))
        traceback.print_exc()


def _run_owl_satellite(ws):
    try:
        from PIL import Image
        g=ws.__class__.satellite_detect_selected.__globals__
        sat_tile_bounds_for_bbox=g.get("sat_tile_bounds_for_bbox")
        sat_lonlat_from_world_px=g.get("sat_lonlat_from_world_px")
        cf_mod=g.get("cf")
        if sat_tile_bounds_for_bbox is None or sat_lonlat_from_world_px is None: raise RuntimeError("Satellite helper functions not found.")
        min_lat,min_lon,max_lat,max_lon=ws._satellite_bbox()
        z=int(_get_var(ws.sat_zoom,0))
        x_min,y_min,x_max,y_max=sat_tile_bounds_for_bbox(min_lat,min_lon,max_lat,max_lon,z)
        cols=x_max-x_min+1; rows=y_max-y_min+1
        width=cols*WEB_TILE_SIZE; height=rows*WEB_TILE_SIZE
        chunk=max(64,int(_get_var(ws.tile,768) or 768)); conf=max(0.001,min(1.0,float(_get_var(ws.conf,0.15))))
        run_stamp=time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns()%1000000000:09d}"
        out_path=ws._satellite_unique_run_output_path(ws._satellite_output_gpkg_path(), run_stamp)
        temp_cache_root=ws._satellite_cache_dir()/"_detection_tmp"/f"owl_z{z}_{run_stamp}_{threading.get_ident()}"
        temp_cache_root.mkdir(parents=True,exist_ok=True)
        old_temp=getattr(ws,"sat_detection_temp_cache_root",None)
        old_thread=getattr(getattr(ws,"_sat_detection_thread_local",None),"cache_root",None)
        ws.sat_detection_temp_cache_root=temp_cache_root
        try: ws._sat_detection_thread_local.cache_root=temp_cache_root
        except Exception: pass
        records=[]
        try:
            jobs=[]; cid=0
            for y in range(0,height,chunk):
                for x in range(0,width,chunk):
                    cid+=1; jobs.append((cid,x,y,min(chunk,width-x),min(chunk,height-y)))
            total=len(jobs)
            ws.log(f"Google OWL Satellite Detection started: z={z}, tiles={cols}x{rows}, chunks={total}, conf={conf}")
            workers=max(1,min(8,int(os.environ.get("MUSTATIL_SAT_CHUNK_PREFETCH","3") or "3"),total or 1))
            def build(job):
                cid,x,y,cw,ch=job
                meta=ws._satellite_build_chunk_to_cache(x_min,y_min,z,x,y,cw,ch,cid,temp_cache_root)
                return job,meta
            with cf_mod.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="owl-sat") as ex:
                for job,meta in ex.map(build,jobs):
                    cid,x,y,cw,ch=job
                    chunk_path=Path(meta["chunk_path"]); used=list(meta.get("tile_paths") or [])
                    im=None
                    try:
                        im=Image.open(chunk_path).convert("RGB")
                        local=_owl_detect_pil(ws,im,conf)
                        found=0
                        for r in local:
                            bx1=max(0.0,min(float(cw),float(r["x1"]))); bx2=max(0.0,min(float(cw),float(r["x2"])))
                            by1=max(0.0,min(float(ch),float(r["y1"]))); by2=max(0.0,min(float(ch),float(r["y2"])))
                            try:
                                ok,_reason=ws._satellite_detection_box_is_valid(bx1,by1,bx2,by2,cw,ch)
                                if not ok: continue
                            except Exception: pass
                            gx1=x_min*WEB_TILE_SIZE+x+bx1; gy1=y_min*WEB_TILE_SIZE+y+by1
                            gx2=x_min*WEB_TILE_SIZE+x+bx2; gy2=y_min*WEB_TILE_SIZE+y+by2
                            lon1,lat1=sat_lonlat_from_world_px(gx1,gy1,z); lon2,lat2=sat_lonlat_from_world_px(gx2,gy2,z)
                            west,east=sorted((float(lon1),float(lon2))); south,north=sorted((float(lat1),float(lat2)))
                            poly=[(west,north),(east,north),(east,south),(west,south),(west,north)]
                            records.append({"class_id":int(r.get("class_id",0)),"class_name":str(r.get("class_name",r.get("label","object"))),"confidence":float(r.get("confidence",0.0)),"model":str(r.get("model","Google OWL")),"model_slot":1,"zoom":int(z),"tile_x_min":int(x_min),"tile_y_min":int(y_min),"chunk_id":int(cid),"chunk_px_x":int(x),"chunk_px_y":int(y),"bbox_px_x1":float(x+bx1),"bbox_px_y1":float(y+by1),"bbox_px_x2":float(x+bx2),"bbox_px_y2":float(y+by2),"bbox_lon_min":west,"bbox_lat_min":south,"bbox_lon_max":east,"bbox_lat_max":north,"world_px_z":int(z),"world_px_x1":float(gx1),"world_px_y1":float(gy1),"world_px_x2":float(gx2),"world_px_y2":float(gy2),"polygon_lonlat":poly})
                            found+=1
                        ws.log(f"Google OWL satellite chunk {cid}/{total}: detections={found}; total={len(records)}")
                    finally:
                        try:
                            if im is not None: im.close()
                        except Exception: pass
                        try: chunk_path.unlink(missing_ok=True)
                        except Exception: pass
                        try: ws._satellite_delete_tile_paths(used)
                        except Exception: pass
            try:
                records=ws._satellite_deduplicate_records(records,0.90)
            except Exception: pass
            ws.sat_last_x_min=int(x_min); ws.sat_last_y_min=int(y_min); ws.sat_last_z=int(z)
            ws.sat_last_records=[dict(r) for r in records]; ws.satellite_detections=[dict(r) for r in records]
            if records:
                try: ws._satellite_features_to_file([dict(r) for r in records], out_path); ws.satellite_output_last=str(out_path); ws.log(f"Google OWL Satellite GeoPackage written: {out_path}")
                except Exception as exc: ws.log(f"Google OWL satellite export warning: {exc}")
            else: ws.log("Google OWL Satellite Detection finished: no detections.")
            try: ws._satellite_request_map_reload_from_worker(120)
            except Exception: pass
            _refresh_combos(ws); _redraw(ws)
            ws.log(f"Google OWL Satellite Detection finished: {len(records)} detections")
        finally:
            shutil.rmtree(temp_cache_root, ignore_errors=True)
            ws.sat_detection_temp_cache_root=old_temp
            try: ws._sat_detection_thread_local.cache_root=old_thread
            except Exception: pass
    except Exception as exc:
        try: ws.show_error("Google OWL Satellite Detection", str(exc))
        except Exception: _log("OWL satellite detection failed: "+str(exc))
        traceback.print_exc()


def _run_as_task(ws, title, fn, allow_parallel=False):
    try: return ws.run_task(title, fn, allow_parallel=allow_parallel)
    except TypeError:
        try: return ws.run_task(title, fn)
        except Exception: pass
    except Exception: pass
    return fn()




# ============================================================
# Additional left-panel model tabs: Grounding DINO + LAE-DINO
# ============================================================

def _model_labels_from_widget(ws, attr_edit, attr_text, default_text) -> List[str]:
    text = ""
    try:
        text = str(getattr(ws, attr_edit).text())
    except Exception:
        pass
    if not text:
        text = str(getattr(ws, attr_text, "") or default_text)
    vals = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    return vals or [p.strip() for p in default_text.split(",") if p.strip()] or ["object"]


def _gdino_labels(ws) -> List[str]:
    return _model_labels_from_widget(
        ws,
        "mustatil_gdino_classes_edit",
        "mustatil_gdino_classes_text",
        "burial mound, tumulus, stone enclosure, rectangular structure, mustatil"
    )


def _gdino_device(ws) -> str:
    try:
        raw = str(ws.mustatil_gdino_device_combo.currentText()).strip().lower()
    except Exception:
        raw = "cpu"
    return raw if raw in {"cpu", "cuda", "auto"} else "cpu"


def _load_gdino_model(ws):
    device_req = _gdino_device(ws)
    import torch
    device = "cuda" if (device_req in {"cuda", "auto"} and torch.cuda.is_available()) else "cpu"
    key = ("grounding-dino", device)
    cache = getattr(ws, "_mustatil_gdino_model_cache", {}) or {}
    if key in cache:
        return cache[key]
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    mid = "IDEA-Research/grounding-dino-base"
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(device)
    model.eval()
    cache[key] = (proc, model, device, mid)
    ws._mustatil_gdino_model_cache = cache
    try:
        ws.log(f"Grounding DINO loaded: {mid} on {device}")
    except Exception:
        pass
    return cache[key]


def _gdino_detect_pil(ws, pil_image, conf: float) -> List[Dict[str,Any]]:
    labels = _gdino_labels(ws)
    proc, model, device, mid = _load_gdino_model(ws)
    import torch, inspect
    prompt = ". ".join(labels)
    if prompt and not prompt.endswith("."):
        prompt += "."
    inputs = proc(images=pil_image, text=prompt, return_tensors="pt")
    inputs = {k:(v.to(device) if hasattr(v,"to") else v) for k,v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([pil_image.size[::-1]], device=device)

    # transformers changed this method several times:
    # - threshold=
    # - box_threshold= / text_threshold=
    # - positional input_ids vs keyword input_ids
    # Try all known compatible call shapes.
    processed = None
    post = getattr(proc, "post_process_grounded_object_detection", None)
    if post is None:
        raise RuntimeError("This transformers version has no post_process_grounded_object_detection(). Please press 'Install/repair Grounding DINO deps'.")
    input_ids = inputs.get("input_ids")
    attempts = [
        lambda: post(outputs=outputs, input_ids=input_ids, threshold=float(conf), target_sizes=target_sizes),
        lambda: post(outputs, input_ids, threshold=float(conf), target_sizes=target_sizes),
        lambda: post(outputs=outputs, input_ids=input_ids, box_threshold=float(conf), text_threshold=0.25, target_sizes=target_sizes),
        lambda: post(outputs, input_ids, box_threshold=float(conf), text_threshold=0.25, target_sizes=target_sizes),
        lambda: post(outputs=outputs, box_threshold=float(conf), text_threshold=0.25, target_sizes=target_sizes),
        lambda: post(outputs, target_sizes=target_sizes, threshold=float(conf)),
    ]
    last_exc = None
    for fn in attempts:
        try:
            processed = fn()
            break
        except TypeError as exc:
            last_exc = exc
            continue
    if processed is None:
        raise RuntimeError(
            "Grounding DINO postprocess threshold API mismatch. "
            "Bitte 'Install/repair Grounding DINO deps' drücken. Letzter Fehler: " + str(last_exc)
        )

    out = []
    if not processed:
        return out
    r = processed[0]
    boxes = r.get("boxes", [])
    scores = r.get("scores", [])
    text_labels = r.get("text_labels", r.get("labels", []))
    for b, s, lab in zip(boxes, scores, text_labels):
        label = str(lab)
        li = 0
        low = label.lower()
        for i, c in enumerate(labels):
            if c.lower() in low or low in c.lower():
                li = i
                label = c
                break
        box = b.detach().cpu().tolist() if hasattr(b, "detach") else list(b)
        score = float(s.detach().cpu().item() if hasattr(s, "detach") else s)
        if len(box) >= 4:
            x1,y1,x2,y2 = map(float, box[:4])
            if x2 > x1 and y2 > y1:
                out.append({
                    "class_id":li, "class_name":label, "label":label,
                    "confidence":score, "score":score,
                    "x1":x1, "y1":y1, "x2":x2, "y2":y2,
                    "model":"Grounding DINO"
                })
    return out


def _lae_labels(ws) -> List[str]:
    return _model_labels_from_widget(
        ws,
        "mustatil_lae_classes_edit",
        "mustatil_lae_classes_text",
        "burial mound, tumulus, stone enclosure, rectangular structure, mustatil"
    )


def _lae_device(ws) -> str:
    try:
        raw = str(ws.mustatil_lae_device_combo.currentText()).strip().lower()
    except Exception:
        raw = "cpu"
    if raw == "auto":
        try:
            import torch
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if raw == "cuda":
        return "cuda:0"
    return raw if raw in {"cpu", "cuda:0", "cuda:1", "0"} else "cpu"


def _plugin_runtime_root() -> Path:
    try:
        return Path(__file__).resolve().parent / "mustatil_model_runtimes"
    except Exception:
        return Path.home() / "mustatil_model_runtimes"


def _run_cmd_logged(ws, cmd, cwd=None, env=None):
    try:
        ws.log("RUN: " + " ".join([str(x) for x in cmd]))
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
        stderr=subprocess.STDOUT
    )
    out = proc.stdout or ""
    try:
        for line in out.splitlines()[-30:]:
            ws.log(line)
    except Exception:
        pass
    if proc.returncode != 0:
        raise RuntimeError("Command failed with exit code %s:\n%s\n\n%s" % (proc.returncode, " ".join(map(str, cmd)), "\n".join(out.splitlines()[-60:])))
    return out


def _download_file_logged(ws, url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        ws.log(f"Downloading: {url} -> {dest}")
    except Exception:
        pass
    urllib.request.urlretrieve(url, str(dest))
    return dest


def _install_gdino_deps(ws):
    py = Path(sys.executable)
    # Keep this light: do not reinstall torch; Mustatil controls its own torch build.
    _run_cmd_logged(ws, [py, "-m", "pip", "install", "-U", "transformers", "accelerate", "safetensors", "huggingface_hub"])
    try:
        ws.log("Grounding DINO deps installed/repaired. Restart Mustatil if the model was already loaded.")
    except Exception:
        pass


def _lae_patch_root() -> Path:
    return _plugin_runtime_root() / "LAE-DINO-PATCH"


def _lae_old_root() -> Path:
    return _plugin_runtime_root() / "LAE-DINO"


def _lae_patch_python() -> Path:
    root = _lae_patch_root()
    if os.name == "nt":
        return root / "python310" / "python.exe"
    return root / "python310" / "python"


def _lae_patch_repo() -> Path:
    return _lae_patch_root() / "repo"


def _lae_patch_available() -> bool:
    root = _lae_patch_root()
    return (
        _lae_patch_python().exists()
        and (_lae_patch_repo() / "mmdetection_lae" / "demo" / "image_demo.py").exists()
    )


def _lae_default_root() -> Path:
    # Prefer the known-good training runtime if present.
    # The older LAE-DINO/.venv can miss mmcv._ext on Windows.
    if _lae_patch_available():
        return _lae_patch_root()
    return _lae_old_root()


def _lae_default_venv_python() -> Path:
    root = _lae_default_root()
    if root.name == "LAE-DINO-PATCH":
        return _lae_patch_python()
    if os.name == "nt":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"



def _lae_run_python_imports(py: Path, modules: List[str], env=None) -> Dict[str, str]:
    """Return import status for LAE runtime modules."""
    result = {}
    for mod in modules:
        try:
            proc = subprocess.run(
                [str(py), "-c", f"import {mod}; print('OK')"],
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env
            )
            result[mod] = "OK" if proc.returncode == 0 else (proc.stdout or "").strip().splitlines()[-1] if (proc.stdout or "").strip() else "FAILED"
        except Exception as exc:
            result[mod] = str(exc)
    return result


def _lae_make_env(mmdet_dir: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    old_pp = env.get("PYTHONPATH", "")
    pp = os.pathsep.join([str(mmdet_dir), str(mmdet_dir.parent)] + ([old_pp] if old_pp else []))
    env["PYTHONPATH"] = pp
    return env


def _lae_install_package_best_effort(ws, py: Path, args: List[str], title: str = "") -> bool:
    try:
        _run_cmd_logged(ws, [py, "-m", "pip", "install"] + list(args))
        return True
    except Exception as exc:
        try:
            ws.log((title or "pip install") + " warning: " + str(exc))
        except Exception:
            pass
        return False


def _lae_install_minimal_stack(ws, py: Path, mmdet_dir: Path) -> None:
    """Install only runtime deps. Never build mmdetection_lae editable.

    The official LAE-DINO instructions use Python 3.8, Torch 1.10/cu113,
    mmengine, mmcv and pip install -e . for mmdetection_lae. In Mustatil's
    Windows/Python 3.12 runtime that editable build is the unstable part, so
    this plugin uses PYTHONPATH/.pth source loading instead.
    """
    _run_cmd_logged(ws, [py, "-m", "pip", "install", "-U", "pip", "wheel"])
    _lae_install_package_best_effort(ws, py, ["setuptools==69.5.1", "openmim"], "setuptools/openmim")

    # Keep NumPy below 2 for older OpenMMLab / Torch extension expectations.
    _lae_install_package_best_effort(ws, py, ["numpy<2", "packaging", "pyyaml"], "base scientific deps")

    # Torch first. mmdetection_lae setup/import paths expect it.
    try:
        _ensure_torch_in_lae_venv(ws, py)
    except Exception as exc:
        try: ws.log("Torch install warning: " + str(exc))
        except Exception: pass

    if not _python_has_module(py, "mmengine"):
        _lae_install_package_best_effort(ws, py, ["mmengine"], "mmengine")

    if not _python_has_module(py, "mmcv"):
        # Full mmcv is ideal. On Windows/Python 3.12 it may fail; mmcv-lite can be enough until mmcv.ops are needed.
        ok = False
        try:
            _run_cmd_logged(ws, [py, "-m", "mim", "install", "mmcv>=2.0.0"])
            ok = True
        except Exception as exc:
            try: ws.log("mim mmcv install warning: " + str(exc))
            except Exception: pass
        if not ok:
            _lae_install_package_best_effort(ws, py, ["mmcv-lite"], "mmcv-lite fallback")

    # Never run pip install -e . here. Use source path fallback only.
    _add_mmdet_dir_to_venv_path(ws, py, mmdet_dir)

    # Runtime extras commonly used by demo/image_demo.py and mmdet multimodal configs.
    base_pkgs = [
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
    ]
    _lae_install_package_best_effort(ws, py, base_pkgs, "LAE runtime extras")

    if not _python_has_module(py, "pycocotools"):
        if not _lae_install_package_best_effort(ws, py, ["pycocotools"], "pycocotools"):
            _lae_install_package_best_effort(ws, py, ["pycocotools-windows"], "pycocotools-windows fallback")

    # LAE-specific requirements if present. Do it after base packages, but ignore conflicts.
    for req in [mmdet_dir / "requirements" / "multimodal.txt", mmdet_dir.parent / "lae_requirements.txt"]:
        if req.exists():
            _lae_install_package_best_effort(ws, py, ["-r", str(req)], f"requirements {req.name}")

    # Re-add source path after potential pip operations.
    _add_mmdet_dir_to_venv_path(ws, py, mmdet_dir)


def _lae_check_runtime(ws, py: Path, mmdet_dir: Path) -> str:
    env = _lae_make_env(mmdet_dir)
    mods = ["torch", "mmengine", "mmcv", "mmdet", "transformers", "cv2", "pycocotools"]
    res = _lae_run_python_imports(py, mods, env=env)
    lines = ["LAE-DINO runtime check:"]
    for k in mods:
        lines.append(f"{k}: {res.get(k, 'UNKNOWN')}")
    try:
        proc = subprocess.run(
            [str(py), "-c", "import sys; print(sys.version)"],
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env
        )
        lines.append("python: " + (proc.stdout or "").strip().replace("\\n", " "))
    except Exception:
        pass
    msg = "\\n".join(lines)
    try:
        ws.log(msg)
    except Exception:
        pass
    return msg


def _lae_repair_and_check(ws, py: Path, mmdet_dir: Path) -> str:
    _lae_install_minimal_stack(ws, py, mmdet_dir)
    return _lae_check_runtime(ws, py, mmdet_dir)


def _lae_autodownload_runtime(ws, py_edit=None, repo_edit=None, cfg_edit=None, weights_edit=None, status_cb=None):
    """Prepare LAE-DINO runtime without editable mmdetection_lae build.

    Only LAE-DINO path is touched. OWL/Grounding/layout remain unchanged.
    """
    def status(msg):
        try:
            if status_cb: status_cb(msg)
        except Exception:
            pass
        try:
            ws.log(msg)
        except Exception:
            pass

    root = _lae_default_root()
    root.mkdir(parents=True, exist_ok=True)
    repo = root / "repo"
    mmdet_dir = repo / "mmdetection_lae"

    if not (repo / "README.md").exists():
        status("Downloading LAE-DINO repository zip...")
        zip_path = root / "LAE-DINO-main.zip"
        _download_file_logged(ws, "https://github.com/jaychempan/LAE-DINO/archive/refs/heads/main.zip", zip_path)
        tmp_extract = root / "_extract"
        shutil.rmtree(tmp_extract, ignore_errors=True)
        tmp_extract.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_extract)
        extracted = tmp_extract / "LAE-DINO-main"
        if repo.exists():
            shutil.rmtree(repo, ignore_errors=True)
        shutil.move(str(extracted), str(repo))
        shutil.rmtree(tmp_extract, ignore_errors=True)
    else:
        status("LAE-DINO repository already exists.")

    if not mmdet_dir.exists():
        mmdet_dir = repo

    vpy = _lae_default_venv_python()
    if not vpy.exists():
        status("Creating LAE-DINO private venv...")
        _run_cmd_logged(ws, [sys.executable, "-m", "venv", str(root / ".venv")])
    else:
        status("LAE-DINO venv already exists.")

    status("Installing/repairing LAE-DINO runtime without editable build...")
    _lae_install_minimal_stack(ws, vpy, mmdet_dir)

    bert_dir = repo / "weights" / "bert-base-uncased"
    if not (bert_dir / "config.json").exists():
        try:
            status("Downloading BERT weights for LAE-DINO...")
            _run_cmd_logged(ws, [vpy, "-m", "huggingface_hub.commands.huggingface_cli", "download", "google-bert/bert-base-uncased", "--local-dir", str(bert_dir)])
        except Exception as exc:
            status("BERT download warning: " + str(exc))

    weights_dir = repo / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    candidate_weight = None
    try:
        pths = sorted(weights_dir.rglob("*.pth"), key=lambda q: q.stat().st_size, reverse=True)
        if pths:
            candidate_weight = pths[0]
    except Exception:
        pass
    if candidate_weight is None:
        try:
            status("Trying to download LAE-DINO checkpoint from Hugging Face...")
            _run_cmd_logged(ws, [vpy, "-m", "huggingface_hub.commands.huggingface_cli", "download", "jaychempan/LAE-DINO", "--local-dir", str(weights_dir)])
            pths = sorted(weights_dir.rglob("*.pth"), key=lambda q: q.stat().st_size, reverse=True)
            if pths:
                candidate_weight = pths[0]
        except Exception as exc:
            status("LAE-DINO checkpoint auto-download warning: " + str(exc))

    cfg = mmdet_dir / "configs" / "lae_dino" / "lae_dino_swin-t_pretrain_LAE-1M.py"

    try:
        if py_edit: py_edit.setText(str(vpy))
        if repo_edit: repo_edit.setText(str(repo))
        if cfg_edit: cfg_edit.setText(str(cfg))
        if weights_edit and candidate_weight: weights_edit.setText(str(candidate_weight))
        ws.mustatil_lae_python = str(vpy)
        ws.mustatil_lae_repo = str(repo)
        ws.mustatil_lae_config = str(cfg)
        if candidate_weight:
            ws.mustatil_lae_weights = str(candidate_weight)
    except Exception:
        pass

    check = _lae_check_runtime(ws, vpy, mmdet_dir)
    status(check)
    if candidate_weight:
        status("LAE-DINO runtime ready. Checkpoint: " + str(candidate_weight))
    else:
        status("LAE-DINO runtime repaired, but checkpoint was not found. Select the .pth weights manually.")
    return root




def _lae_project_classes(project: Path) -> List[str]:
    for p in [project / "project.json", project / "mustatil_project.json", project / "classes.json"]:
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
                                name = item.get("name") or item.get("label") or item.get("class")
                                if name:
                                    out.append(str(name))
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


def _lae_epoch_number(path: Path) -> int:
    m = re.search(r"epoch[_\- ]?(\d+)", path.name.lower())
    if not m:
        return -1
    try:
        return int(m.group(1))
    except Exception:
        return -1


def _lae_score_checkpoint(path: Path):
    s = str(path).replace("\\", "/").lower()
    name = path.name.lower()
    score = 0
    if "/work_dirs/" in s:
        score += 1000
    if "lae_dino_mustatil" in s:
        score += 300
    if name.startswith("best") or "best_" in name:
        score += 900
    if name == "latest.pth":
        score += 800
    if "epoch" in name:
        score += 400 + max(0, min(999, _lae_epoch_number(path)))
    if "sanitized" in name or "dior" in name or "pretrain" in name:
        score -= 700
    try:
        mtime = path.stat().st_mtime
    except Exception:
        mtime = 0.0
    return (-score, -mtime, str(path).lower())


def _lae_find_project_checkpoints(project: Path) -> List[Path]:
    roots = [
        project / "lae_dino_dataset" / "work_dirs" / "lae_dino_mustatil",
        project / "lae_dino_dataset" / "work_dirs",
        project / "lae_dino_dataset",
        project,
    ]
    out = []
    for root in roots:
        try:
            if not root.exists():
                continue
            for pat in ("best*.pth", "latest.pth", "epoch_*.pth", "epoch*.pth", "*.pth", "*.pt"):
                out.extend([p for p in root.rglob(pat) if p.is_file()])
        except Exception:
            pass
    seen = set(); unique = []
    for p in out:
        try:
            sp = str(p.resolve())
        except Exception:
            sp = str(p)
        if sp in seen:
            continue
        seen.add(sp); unique.append(p)
    unique.sort(key=_lae_score_checkpoint)
    return unique


def _lae_score_config(path: Path, project: Path):
    s = str(path).replace("\\", "/").lower()
    n = path.name.lower()
    score = 0
    if str(project).replace("\\", "/").lower() in s:
        score += 1000
    for tag, val in (("v14", 1200), ("v13", 1100), ("v12", 800)):
        if tag in n and "final" in n:
            score += val
            break
    if "mmengine_final" in n:
        score += 500
    if "mustatil_train" in n:
        score += 300
    if "train_from_project" in n:
        score += 200
    if "verify_config" in n or "build_final_config" in n or "__pycache__" in s:
        score -= 2000
    try:
        mtime = path.stat().st_mtime
    except Exception:
        mtime = 0.0
    return (-score, -mtime, str(path).lower())


def _lae_find_project_configs(project: Path) -> List[Path]:
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
                n = p.name.lower()
                if "__pycache__" in str(p).lower() or "verify_config" in n or "build_final_config" in n:
                    continue
                if ("lae" in n and "dino" in n) or "mustatil_train" in n:
                    out.append(p)
        except Exception:
            pass
    seen = set(); unique = []
    for p in out:
        try:
            sp = str(p.resolve())
        except Exception:
            sp = str(p)
        if sp in seen:
            continue
        seen.add(sp); unique.append(p)
    unique.sort(key=lambda p: _lae_score_config(p, project))
    return unique


def _lae_guess_project(ws=None) -> str:
    candidates = []
    for obj in [ws]:
        if obj is None:
            continue
        for name in ("project_dir", "project_folder", "project_path", "mustatil_project_dir", "mustatil_project_folder", "last_project_dir", "current_project_dir"):
            try:
                v = getattr(obj, name, "")
                if v:
                    candidates.append(str(v))
            except Exception:
                pass
    candidates.append("G:/Khirigsuurs")
    for c in candidates:
        try:
            p = Path(str(c).strip().strip('"'))
            if (p / "project.json").exists() or (p / "lae_dino_dataset").exists():
                return str(p)
        except Exception:
            pass
    return candidates[0] if candidates else ""


def _lae_select_patch_runtime(py_edit=None, repo_edit=None) -> Tuple[str, str]:
    if _lae_patch_available():
        py = str(_lae_patch_python())
        repo = str(_lae_patch_repo())
    else:
        py = str(_lae_default_venv_python())
        repo = str(_lae_default_root() / "repo")
    try:
        if py_edit is not None:
            py_edit.setText(py)
        if repo_edit is not None:
            repo_edit.setText(repo)
    except Exception:
        pass
    return py, repo

def _lae_runtime_paths(ws):
    # Auto-fill defaults from the private runtime if fields are empty.
    default_repo = _lae_default_root() / "repo"
    default_py = _lae_default_venv_python()
    default_cfg = default_repo / "mmdetection_lae" / "configs" / "lae_dino" / "lae_dino_swin-t_pretrain_LAE-1M.py"
    default_weights = ""
    try:
        pths = sorted((default_repo / "weights").glob("*.pth"), key=lambda q: q.stat().st_size, reverse=True)
        if pths:
            default_weights = str(pths[0])
    except Exception:
        pass

    repo = Path(str(getattr(ws, "mustatil_lae_repo", "") or os.environ.get("MUSTATIL_LAE_DINO_REPO", "") or default_repo).strip().strip('"'))
    if not str(repo):
        raise RuntimeError("LAE-DINO repo not set. Press 'Auto install/download LAE-DINO runtime'.")
    mmdet_dir = repo / "mmdetection_lae" if (repo / "mmdetection_lae").exists() else repo
    demo = mmdet_dir / "demo" / "image_demo.py"
    if not demo.exists():
        raise RuntimeError(f"LAE-DINO demo not found: {demo}\nPress 'Auto install/download LAE-DINO runtime'.")

    cfg = str(getattr(ws, "mustatil_lae_config", "") or os.environ.get("MUSTATIL_LAE_DINO_CONFIG", "") or default_cfg).strip().strip('"')
    cfg_path = Path(cfg)
    if not cfg_path.is_absolute():
        cfg_path = mmdet_dir / cfg
    if not cfg_path.exists():
        raise RuntimeError(f"LAE-DINO config not found: {cfg_path}\nPress 'Auto install/download LAE-DINO runtime'.")

    weights = Path(str(getattr(ws, "mustatil_lae_weights", "") or os.environ.get("MUSTATIL_LAE_DINO_WEIGHTS", "") or default_weights).strip().strip('"'))
    if not str(weights) or not weights.exists():
        raise RuntimeError("LAE-DINO weights not found. Press 'Auto install/download LAE-DINO runtime' or select the .pth checkpoint manually.")

    py = str(getattr(ws, "mustatil_lae_python", "") or os.environ.get("MUSTATIL_LAE_DINO_PYTHON", "") or default_py or sys.executable).strip().strip('"')
    return Path(py), mmdet_dir, demo, cfg_path, weights


def _parse_lae_prediction_json(json_path: Path, labels: List[str]) -> List[Dict[str,Any]]:
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
            x1,y1,x2,y2 = map(float, list(b)[:4])
        except Exception:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        score = float(scores[i]) if i < len(scores) else 0.0
        labraw = labs[i] if i < len(labs) else 0
        li = 0
        lab = str(labraw)
        try:
            li = int(labraw)
            if 0 <= li < len(labels):
                lab = labels[li]
        except Exception:
            low = str(labraw).lower()
            for j, c in enumerate(labels):
                if c.lower() in low or low in c.lower():
                    li = j
                    lab = c
                    break
        out.append({
            "class_id":li, "class_name":lab, "label":lab,
            "confidence":score, "score":score,
            "x1":x1, "y1":y1, "x2":x2, "y2":y2,
            "model":"LAE-DINO"
        })
    return out


def _find_lae_json(out_dir: Path, image_path: Path) -> Optional[Path]:
    candidates = []
    for sub in ("preds", "predictions", ""):
        d = out_dir / sub
        if d.exists():
            candidates += list(d.glob("*.json"))
    if not candidates:
        candidates = list(out_dir.rglob("*.json"))
    stem = image_path.stem.lower()
    candidates.sort(key=lambda p: (0 if stem in p.stem.lower() else 1, -p.stat().st_mtime))
    return candidates[0] if candidates else None


def _python_has_module(py: Path, module: str) -> bool:
    try:
        proc = subprocess.run(
            [str(py), "-c", f"import {module}"],
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        return proc.returncode == 0
    except Exception:
        return False


def _torch_cuda_tag_for_current_runtime() -> str:
    """Best-effort CUDA wheel index selection.

    Mustatil usually already has torch. For the private LAE-DINO venv we install
    a compatible wheel. Default to CPU if CUDA is unknown.
    """
    try:
        import torch
        cu = getattr(torch.version, "cuda", None)
        if cu:
            s = str(cu).replace(".", "")
            if s.startswith("12"):
                return "cu121"
            if s.startswith("11"):
                return "cu118"
    except Exception:
        pass
    return "cpu"


def _ensure_torch_in_lae_venv(ws, py: Path) -> None:
    if _python_has_module(py, "torch"):
        return
    tag = _torch_cuda_tag_for_current_runtime()
    try:
        ws.log(f"LAE-DINO runtime repair: torch missing, installing torch ({tag}) before mmdetection_lae setup.")
    except Exception:
        pass
    if tag == "cpu":
        _run_cmd_logged(ws, [py, "-m", "pip", "install", "-U", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"])
    else:
        _run_cmd_logged(ws, [py, "-m", "pip", "install", "-U", "torch", "torchvision", "--index-url", f"https://download.pytorch.org/whl/{tag}"])



def _venv_site_packages(py: Path) -> Optional[Path]:
    try:
        proc = subprocess.run(
            [str(py), "-c", "import site, json; print(json.dumps(site.getsitepackages()))"],
            text=True, encoding="utf-8", errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        if proc.returncode == 0:
            vals = json.loads((proc.stdout or "[]").strip())
            for v in vals:
                p = Path(v)
                if p.exists():
                    return p
    except Exception:
        pass
    return None


def _add_mmdet_dir_to_venv_path(ws, py: Path, mmdet_dir: Path) -> None:
    """Fallback when pip install -e . fails.

    mmdetection_lae is mostly loaded as source code. On Windows/Python 3.12 the
    editable install can fail during setuptools build hooks. A .pth file is
    enough for image_demo.py/tools/train.py to import the local mmdet package.
    """
    sp = _venv_site_packages(py)
    if sp is None:
        try: ws.log("Could not find LAE-DINO venv site-packages for .pth fallback.")
        except Exception: pass
        return
    pth = sp / "mustatil_lae_dino_repo.pth"
    lines = [str(mmdet_dir), str(mmdet_dir.parent)]
    pth.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
    try: ws.log(f"Added LAE-DINO source path fallback: {pth}")
    except Exception: pass


def _local_mmdet_source_exists(mmdet_dir: Path) -> bool:
    return (mmdet_dir / "mmdet").exists() and (mmdet_dir / "demo" / "image_demo.py").exists()



def _ensure_lae_runtime_deps(ws, py: Path, mmdet_dir: Path) -> None:
    """Repair the private LAE-DINO venv before running image_demo.py.

    Important: this function intentionally does NOT run pip install -e .
    On this Windows/Python-3.12 setup the editable build is the failing part.
    """
    _lae_install_minimal_stack(ws, py, mmdet_dir)

    env = _lae_make_env(mmdet_dir)
    status = _lae_run_python_imports(py, ["torch", "mmengine", "mmcv", "mmdet"], env=env)
    missing = [k for k, v in status.items() if v != "OK"]
    if missing:
        raise RuntimeError(
            "LAE-DINO runtime still misses modules after repair: "
            + ", ".join([f"{m}: {status[m]}" for m in missing])
            + "\nHinweis: Wenn mmcv unter Windows/Python 3.12 nicht lädt, ist wahrscheinlich eine Python-3.8/3.10-LAE-Runtime oder ein passendes OpenMMLab-Wheel nötig."
        )


def _lae_detect_pil(ws, pil_image, conf: float) -> List[Dict[str,Any]]:
    labels = _lae_labels(ws)
    py, mmdet_dir, demo, cfg, weights = _lae_runtime_paths(ws)

    _ensure_lae_runtime_deps(ws, py, mmdet_dir)

    device = _lae_device(ws)
    prompt = " . ".join(labels)
    if prompt and not prompt.endswith("."):
        prompt += " ."
    with tempfile.TemporaryDirectory(prefix="mustatil_lae_") as td:
        td = Path(td)
        img_path = td / "chunk.png"
        out_dir = td / "out"
        pil_image.convert("RGB").save(img_path)
        cmd = [
            str(py), str(demo), str(img_path), str(cfg),
            "--weights", str(weights),
            "--texts", prompt,
            "-c",
            "--pred-score-thr", str(float(conf)),
            "--device", device,
            "--out-dir", str(out_dir),
            "--palette", "random"
        ]
        env = _lae_make_env(mmdet_dir)
        proc = subprocess.run(
            cmd, cwd=str(mmdet_dir), text=True, encoding="utf-8", errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
        )
        if proc.returncode != 0 and (
            "No module named" in (proc.stdout or "")
            or "ModuleNotFoundError" in (proc.stdout or "")
            or "ImportError" in (proc.stdout or "")
        ):
            _ensure_lae_runtime_deps(ws, py, mmdet_dir)
            env = _lae_make_env(mmdet_dir)
            proc = subprocess.run(
                cmd, cwd=str(mmdet_dir), text=True, encoding="utf-8", errors="replace",
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
            )
        if proc.returncode != 0:
            tail = "\n".join((proc.stdout or "").splitlines()[-80:])
            raise RuntimeError("LAE-DINO failed with exit code %s\n%s" % (proc.returncode, tail))
        jp = _find_lae_json(out_dir, img_path)
        if jp is None:
            tail = "\n".join((proc.stdout or "").splitlines()[-80:])
            raise RuntimeError("LAE-DINO did not write prediction JSON. Output tail:\n" + tail)
        return _parse_lae_prediction_json(jp, labels)


def _run_generic_detection_image(ws, model_name: str, detect_fn):
    try:
        from PIL import Image
        from mustatil_legacy_backend import Det
        img_path = str(_get_var(getattr(ws, "image", None), "") or "").strip().strip('"')
        if not img_path or not Path(img_path).exists():
            raise RuntimeError("No Detection image selected.")
        conf = max(0.001, min(1.0, float(_get_var(getattr(ws, "conf", None), 0.15))))
        tile = max(128, int(_get_var(getattr(ws, "tile", None), 768) or 768))
        overlap = max(0, min(tile-1, int(_get_var(getattr(ws, "overlap", None), 128) or 128)))
        step = max(1, tile-overlap)
        img = Image.open(img_path).convert("RGB")
        W,H = img.size
        ws.log(f"{model_name} Detection started: {Path(img_path).name} {W}x{H}, tile={tile}, overlap={overlap}, conf={conf}")
        dets = []
        count = 0
        for y in range(0, H, step):
            for x in range(0, W, step):
                x2 = min(W, x+tile)
                y2 = min(H, y+tile)
                local = detect_fn(ws, img.crop((x,y,x2,y2)), conf)
                for r in local:
                    dets.append(Det(
                        0, str(r.get("model", model_name)),
                        int(r.get("class_id", 0)), float(r.get("confidence", 0.0)),
                        x+float(r["x1"]), y+float(r["y1"]), x+float(r["x2"]), y+float(r["y2"])
                    ))
                count += 1
                if count % 10 == 0:
                    ws.log(f"{model_name}: tiles={count}, detections={len(dets)}")
                if x2 >= W:
                    break
            if y+tile >= H:
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
        ws.log(f"{model_name} Detection finished: {len(dets)} detections")
    except Exception as exc:
        try:
            ws.show_error(model_name, str(exc))
        except Exception:
            _log(f"{model_name} failed: " + str(exc))
        traceback.print_exc()


def _run_generic_satellite(ws, model_name: str, detect_fn):
    try:
        from PIL import Image
        g = ws.__class__.satellite_detect_selected.__globals__
        sat_tile_bounds_for_bbox = g.get("sat_tile_bounds_for_bbox")
        sat_lonlat_from_world_px = g.get("sat_lonlat_from_world_px")
        cf_mod = g.get("cf")
        if sat_tile_bounds_for_bbox is None or sat_lonlat_from_world_px is None:
            raise RuntimeError("Satellite helper functions not found.")
        min_lat,min_lon,max_lat,max_lon = ws._satellite_bbox()
        z = int(_get_var(ws.sat_zoom, 0))
        x_min,y_min,x_max,y_max = sat_tile_bounds_for_bbox(min_lat,min_lon,max_lat,max_lon,z)
        cols = x_max-x_min+1
        rows = y_max-y_min+1
        width = cols*WEB_TILE_SIZE
        height = rows*WEB_TILE_SIZE
        chunk = max(64, int(_get_var(ws.tile,768) or 768))
        conf = max(0.001, min(1.0, float(_get_var(ws.conf,0.15))))
        run_stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns()%1000000000:09d}"
        out_path = ws._satellite_unique_run_output_path(ws._satellite_output_gpkg_path(), run_stamp)
        temp_cache_root = ws._satellite_cache_dir() / "_detection_tmp" / f"{model_name.lower().replace(' ','_')}_z{z}_{run_stamp}_{threading.get_ident()}"
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
                    jobs.append((cid,x,y,min(chunk,width-x),min(chunk,height-y)))
            total = len(jobs)
            ws.log(f"{model_name} Satellite Detection started: z={z}, tiles={cols}x{rows}, chunks={total}, conf={conf}")
            workers = max(1, min(8, int(os.environ.get("MUSTATIL_SAT_CHUNK_PREFETCH", "3") or "3"), total or 1))
            def build(job):
                cid,x,y,cw,ch = job
                meta = ws._satellite_build_chunk_to_cache(x_min,y_min,z,x,y,cw,ch,cid,temp_cache_root)
                return job, meta
            with cf_mod.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mustatil-model-sat") as ex:
                for job, meta in ex.map(build, jobs):
                    cid,x,y,cw,ch = job
                    chunk_path = Path(meta["chunk_path"])
                    used = list(meta.get("tile_paths") or [])
                    im = None
                    try:
                        im = Image.open(chunk_path).convert("RGB")
                        local = detect_fn(ws, im, conf)
                        found = 0
                        for r in local:
                            bx1 = max(0.0, min(float(cw), float(r["x1"])))
                            bx2 = max(0.0, min(float(cw), float(r["x2"])))
                            by1 = max(0.0, min(float(ch), float(r["y1"])))
                            by2 = max(0.0, min(float(ch), float(r["y2"])))
                            try:
                                ok,_reason = ws._satellite_detection_box_is_valid(bx1,by1,bx2,by2,cw,ch)
                                if not ok:
                                    continue
                            except Exception:
                                pass
                            gx1 = x_min*WEB_TILE_SIZE+x+bx1
                            gy1 = y_min*WEB_TILE_SIZE+y+by1
                            gx2 = x_min*WEB_TILE_SIZE+x+bx2
                            gy2 = y_min*WEB_TILE_SIZE+y+by2
                            lon1,lat1 = sat_lonlat_from_world_px(gx1,gy1,z)
                            lon2,lat2 = sat_lonlat_from_world_px(gx2,gy2,z)
                            west,east = sorted((float(lon1),float(lon2)))
                            south,north = sorted((float(lat1),float(lat2)))
                            poly = [(west,north),(east,north),(east,south),(west,south),(west,north)]
                            records.append({
                                "class_id":int(r.get("class_id",0)),
                                "class_name":str(r.get("class_name",r.get("label","object"))),
                                "confidence":float(r.get("confidence",0.0)),
                                "model":str(r.get("model",model_name)),
                                "model_slot":1,
                                "zoom":int(z),
                                "tile_x_min":int(x_min),
                                "tile_y_min":int(y_min),
                                "chunk_id":int(cid),
                                "chunk_px_x":int(x),
                                "chunk_px_y":int(y),
                                "bbox_px_x1":float(x+bx1),
                                "bbox_px_y1":float(y+by1),
                                "bbox_px_x2":float(x+bx2),
                                "bbox_px_y2":float(y+by2),
                                "bbox_lon_min":west,
                                "bbox_lat_min":south,
                                "bbox_lon_max":east,
                                "bbox_lat_max":north,
                                "world_px_z":int(z),
                                "world_px_x1":float(gx1),
                                "world_px_y1":float(gy1),
                                "world_px_x2":float(gx2),
                                "world_px_y2":float(gy2),
                                "polygon_lonlat":poly
                            })
                            found += 1
                        ws.log(f"{model_name} satellite chunk {cid}/{total}: detections={found}; total={len(records)}")
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
                    ws.log(f"{model_name} Satellite GeoPackage written: {out_path}")
                except Exception as exc:
                    ws.log(f"{model_name} satellite export warning: {exc}")
            else:
                ws.log(f"{model_name} Satellite Detection finished: no detections.")
            try:
                ws._satellite_request_map_reload_from_worker(120)
            except Exception:
                pass
            _refresh_combos(ws)
            _redraw(ws)
            ws.log(f"{model_name} Satellite Detection finished: {len(records)} detections")
        finally:
            shutil.rmtree(temp_cache_root, ignore_errors=True)
            ws.sat_detection_temp_cache_root = old_temp
            try:
                ws._sat_detection_thread_local.cache_root = old_thread
            except Exception:
                pass
    except Exception as exc:
        try:
            ws.show_error(model_name, str(exc))
        except Exception:
            _log(f"{model_name} satellite failed: " + str(exc))
        traceback.print_exc()


def _run_gdino_detection_image(ws):
    return _run_generic_detection_image(ws, "Grounding DINO", _gdino_detect_pil)


def _run_gdino_satellite(ws):
    return _run_generic_satellite(ws, "Grounding DINO", _gdino_detect_pil)


def _run_lae_detection_image(ws):
    return _run_generic_detection_image(ws, "LAE-DINO", _lae_detect_pil)


def _run_lae_satellite(ws):
    return _run_generic_satellite(ws, "LAE-DINO", _lae_detect_pil)


def _build_gdino_left_controls(ws, tab_kind: str):
    from PySide6.QtWidgets import QWidget,QVBoxLayout,QGridLayout,QGroupBox,QLabel,QLineEdit,QPushButton,QComboBox
    page = QWidget()
    left_lay = QVBoxLayout(page)
    left_lay.setContentsMargins(8,8,8,8)
    header = QLabel("Grounding DINO: nur linkes Bedienfeld. Rechts bleibt Mustatils Original-Preview/Karte.")
    header.setWordWrap(True)
    left_lay.addWidget(header)
    box = QGroupBox("Grounding DINO detection")
    g = QGridLayout(box)
    device = QComboBox()
    device.setEditable(True)
    device.addItems(["cpu","cuda","auto"])
    classes = QLineEdit(str(getattr(ws,"mustatil_gdino_classes_text","burial mound, tumulus, stone enclosure, rectangular structure, mustatil") or "burial mound, tumulus, stone enclosure, rectangular structure, mustatil"))
    status = QLabel("Idle")
    status.setWordWrap(True)
    try:
        if _lae_patch_available() and ("LAE-DINO\\.venv" in py_edit.text() or "LAE-DINO/.venv" in py_edit.text().replace("\\", "/")):
            _lae_select_patch_runtime(py_edit, repo_edit)
    except Exception:
        pass
    run_btn = QPushButton("Run Grounding DINO on Detection image" if tab_kind=="detection" else "Run Grounding DINO on selected satellite map")
    install_btn = QPushButton("Install/repair Grounding DINO deps")
    g.addWidget(QLabel("Model"),0,0)
    g.addWidget(QLabel("IDEA-Research/grounding-dino-base"),0,1)
    g.addWidget(QLabel("Device"),1,0)
    g.addWidget(device,1,1)
    g.addWidget(QLabel("Classes / prompt"),2,0)
    g.addWidget(classes,2,1)
    g.addWidget(run_btn,3,0,1,2)
    g.addWidget(install_btn,4,0,1,2)
    g.addWidget(status,5,0,1,2)
    left_lay.addWidget(box)
    left_lay.addWidget(_make_filter_box(ws,"Class filter / Geo NMS (Grounding DINO)"))
    left_lay.addStretch(1)
    def sync():
        ws.mustatil_gdino_classes_text = classes.text()
        ws.mustatil_gdino_classes_edit = classes
        ws.mustatil_gdino_device_combo = device
    def append(msg):
        try: status.setText(str(msg))
        except Exception: pass
        try: ws.log(str(msg))
        except Exception: pass
    classes.textChanged.connect(lambda *_: sync())
    device.currentTextChanged.connect(lambda *_: sync())
    install_btn.clicked.connect(lambda: (
        sync(),
        append("Installing/repairing Grounding DINO deps..."),
        _run_as_task(ws, "Install Grounding DINO deps", lambda: _install_gdino_deps(ws), allow_parallel=True)
    ))
    run_btn.clicked.connect(lambda: (
        sync(),
        append("Starting Grounding DINO..."),
        _run_as_task(
            ws,
            "Grounding DINO Detection" if tab_kind=="detection" else "Grounding DINO Satellite Detection",
            lambda: _run_gdino_detection_image(ws) if tab_kind=="detection" else _run_gdino_satellite(ws),
            allow_parallel=(tab_kind!="detection")
        )
    ))
    sync()
    return page


def _build_lae_left_controls(ws, tab_kind: str):
    from PySide6.QtWidgets import QWidget,QVBoxLayout,QGridLayout,QGroupBox,QLabel,QLineEdit,QPushButton,QComboBox,QFileDialog,QHBoxLayout
    page = QWidget()
    left_lay = QVBoxLayout(page)
    left_lay.setContentsMargins(8,8,8,8)
    header = QLabel("LAE-DINO Bridge: nur linkes Bedienfeld. Rechts bleibt Mustatils Original-Preview/Karte. Diese Version nutzt Source/PYTHONPATH statt fehlerhaftem pip install -e .")
    header.setWordWrap(True)
    left_lay.addWidget(header)
    box = QGroupBox("LAE-DINO detection")
    g = QGridLayout(box)
    device = QComboBox()
    device.setEditable(True)
    device.addItems(["cpu","cuda","cuda:0","auto"])
    classes = QLineEdit(str(getattr(ws,"mustatil_lae_classes_text","burial mound, tumulus, stone enclosure, rectangular structure, mustatil") or "burial mound, tumulus, stone enclosure, rectangular structure, mustatil"))

    default_repo = _lae_default_root() / "repo"
    default_py = _lae_default_venv_python()
    default_cfg = default_repo / "mmdetection_lae" / "configs" / "lae_dino" / "lae_dino_swin-t_pretrain_LAE-1M.py"
    default_weights = ""
    try:
        pths = sorted((default_repo / "weights").glob("*.pth"), key=lambda q: q.stat().st_size, reverse=True)
        if pths: default_weights = str(pths[0])
    except Exception: pass

    py_edit = QLineEdit(str(getattr(ws,"mustatil_lae_python", os.environ.get("MUSTATIL_LAE_DINO_PYTHON", str(default_py))) or str(default_py)))
    repo_edit = QLineEdit(str(getattr(ws,"mustatil_lae_repo", os.environ.get("MUSTATIL_LAE_DINO_REPO", str(default_repo))) or str(default_repo)))
    cfg_edit = QLineEdit(str(getattr(ws,"mustatil_lae_config", os.environ.get("MUSTATIL_LAE_DINO_CONFIG", str(default_cfg))) or str(default_cfg)))
    weights_edit = QLineEdit(str(getattr(ws,"mustatil_lae_weights", os.environ.get("MUSTATIL_LAE_DINO_WEIGHTS", default_weights)) or default_weights))
    project_edit = QLineEdit(str(getattr(ws,"mustatil_lae_project", _lae_guess_project(ws)) or _lae_guess_project(ws)))
    status = QLabel("Idle")
    status.setWordWrap(True)
    run_btn = QPushButton("Run LAE-DINO on Detection image" if tab_kind=="detection" else "Run LAE-DINO on selected satellite map")
    install_btn = QPushButton("Auto install/repair LAE-DINO runtime")
    check_btn = QPushButton("Check LAE-DINO runtime")
    patch_runtime_btn = QPushButton("Use LAE-DINO-PATCH runtime")
    import_trained_btn = QPushButton("Import model from training project")
    def browse_dir(edit):
        p = QFileDialog.getExistingDirectory(page, "Select LAE-DINO repository", edit.text() or str(Path.home()))
        if p: edit.setText(p)
    def browse_file(edit, title, filt):
        p,_ = QFileDialog.getOpenFileName(page, title, str(Path(edit.text()).parent if edit.text() else Path.home()), filt)
        if p: edit.setText(p)
    row = 0
    g.addWidget(QLabel("Backend"),row,0)
    g.addWidget(QLabel("LAE-DINO / MMDetection"),row,1)
    row += 1
    g.addWidget(QLabel("Device"),row,0)
    g.addWidget(device,row,1)
    row += 1
    g.addWidget(QLabel("Classes / prompt"),row,0)
    g.addWidget(classes,row,1)
    row += 1
    for label, edit, kind in [
        ("Python", py_edit, "file"),
        ("LAE repo", repo_edit, "dir"),
        ("Config", cfg_edit, "file"),
        ("Weights", weights_edit, "weights"),
    ]:
        g.addWidget(QLabel(label), row, 0)
        h = QHBoxLayout()
        h.addWidget(edit, 1)
        b = QPushButton("...")
        if kind == "dir":
            b.clicked.connect(lambda _, e=edit: browse_dir(e))
        else:
            b.clicked.connect(lambda _, e=edit: browse_file(e, "Select file", "Python/Config/Weights (*.exe *.py *.pth *.pt);;All files (*)"))
        h.addWidget(b)
        g.addLayout(h, row, 1)
        row += 1
    g.addWidget(QLabel("Project"), row, 0)
    h = QHBoxLayout()
    h.addWidget(project_edit, 1)
    b = QPushButton("...")
    b.clicked.connect(lambda _, e=project_edit: browse_dir(e))
    h.addWidget(b)
    g.addLayout(h, row, 1)
    row += 1
    g.addWidget(import_trained_btn,row,0,1,2)
    row += 1
    g.addWidget(patch_runtime_btn,row,0,1,2)
    row += 1
    g.addWidget(install_btn,row,0,1,2)
    row += 1
    g.addWidget(check_btn,row,0,1,2)
    row += 1
    g.addWidget(run_btn,row,0,1,2)
    row += 1
    g.addWidget(status,row,0,1,2)
    left_lay.addWidget(box)
    left_lay.addWidget(_make_filter_box(ws,"Class filter / Geo NMS (LAE-DINO)"))
    left_lay.addStretch(1)
    def sync():
        ws.mustatil_lae_classes_text = classes.text()
        ws.mustatil_lae_classes_edit = classes
        ws.mustatil_lae_device_combo = device
        ws.mustatil_lae_python = py_edit.text().strip().strip('"')
        ws.mustatil_lae_repo = repo_edit.text().strip().strip('"')
        ws.mustatil_lae_config = cfg_edit.text().strip().strip('"')
        ws.mustatil_lae_weights = weights_edit.text().strip().strip('"')
        ws.mustatil_lae_project = project_edit.text().strip().strip('"')
        try:
            os.environ["MUSTATIL_LAE_DINO_PYTHON"] = ws.mustatil_lae_python
            os.environ["MUSTATIL_LAE_DINO_REPO"] = ws.mustatil_lae_repo
            os.environ["MUSTATIL_LAE_DINO_CONFIG"] = ws.mustatil_lae_config
            os.environ["MUSTATIL_LAE_DINO_WEIGHTS"] = ws.mustatil_lae_weights
        except Exception:
            pass
    def append(msg):
        try: status.setText(str(msg))
        except Exception: pass
        try: ws.log(str(msg))
        except Exception: pass
    for w in (classes, py_edit, repo_edit, cfg_edit, weights_edit, project_edit):
        w.textChanged.connect(lambda *_: sync())
    device.currentTextChanged.connect(lambda *_: sync())
    def use_patch_runtime_clicked():
        py, repo = _lae_select_patch_runtime(py_edit, repo_edit)
        sync()
        append("LAE-DINO runtime set: " + py)

    def import_trained_clicked():
        project = Path(project_edit.text().strip().strip('"'))
        if not str(project) or not project.exists():
            append("Project folder not found: " + str(project))
            return
        configs = _lae_find_project_configs(project)
        checkpoints = _lae_find_project_checkpoints(project)
        classes_found = _lae_project_classes(project)
        py, repo = _lae_select_patch_runtime(py_edit, repo_edit)
        if configs:
            cfg_edit.setText(str(configs[0]))
        if checkpoints:
            weights_edit.setText(str(checkpoints[0]))
        if classes_found:
            classes.setText(", ".join(classes_found))
        sync()
        parts = ["project=" + str(project), "runtime=" + py]
        parts.append("config=" + (str(configs[0]) if configs else "NOT FOUND"))
        parts.append("checkpoint=" + (str(checkpoints[0]) if checkpoints else "NOT FOUND"))
        if classes_found:
            parts.append("classes=" + ", ".join(classes_found))
        append("Imported trained LAE-DINO model: " + " | ".join(parts))

    patch_runtime_btn.clicked.connect(use_patch_runtime_clicked)
    import_trained_btn.clicked.connect(import_trained_clicked)
    install_btn.clicked.connect(lambda: (
        sync(),
        append("Preparing/repairing LAE-DINO runtime without editable build..."),
        _run_as_task(ws, "Auto install/repair LAE-DINO", lambda: _lae_autodownload_runtime(ws, py_edit, repo_edit, cfg_edit, weights_edit, append), allow_parallel=True)
    ))
    def check_clicked():
        sync()
        append("Checking LAE-DINO runtime...")
        def run_check():
            py, mmdet_dir, demo, cfg, weights = _lae_runtime_paths(ws)
            msg = _lae_check_runtime(ws, py, mmdet_dir)
            append(msg)
        _run_as_task(ws, "Check LAE-DINO runtime", run_check, allow_parallel=True)
    check_btn.clicked.connect(check_clicked)
    run_btn.clicked.connect(lambda: (
        sync(),
        append("Starting LAE-DINO..."),
        _run_as_task(
            ws,
            "LAE-DINO Detection" if tab_kind=="detection" else "LAE-DINO Satellite Detection",
            lambda: _run_lae_detection_image(ws) if tab_kind=="detection" else _run_lae_satellite(ws),
            allow_parallel=(tab_kind!="detection")
        )
    ))
    sync()
    return page


def _make_filter_box(ws, title="Class filter / Geo NMS"):
    from PySide6.QtWidgets import QGroupBox,QVBoxLayout,QHBoxLayout,QLabel,QComboBox,QPushButton,QCheckBox,QDoubleSpinBox
    box=QGroupBox(title)
    box.setObjectName("MustatilOwlV2ClassFilterBox")
    bl=QVBoxLayout(box); bl.setContentsMargins(8,6,8,6)
    row=QHBoxLayout(); row.addWidget(QLabel("Class filter:")); cls_combo=QComboBox(); row.addWidget(cls_combo,1); bl.addLayout(row)
    hide_fp=QCheckBox("Hide class 1 / false_positive"); hide_fp.setChecked(bool(getattr(ws,"mustatil_owl_hide_fp",False))); bl.addWidget(hide_fp)
    geo=QCheckBox("Apply Geo NMS to preview/export"); geo.setChecked(bool(getattr(ws,"mustatil_owl_geo_nms_enabled",False))); bl.addWidget(geo)
    row=QHBoxLayout(); row.addWidget(QLabel("Geo NMS IoU:")); iou=QDoubleSpinBox(); iou.setRange(0.01,0.99); iou.setSingleStep(0.05); iou.setDecimals(2); iou.setValue(float(getattr(ws,"mustatil_owl_geo_nms_iou",0.35))); row.addWidget(iou,1); bl.addLayout(row)
    row=QHBoxLayout(); refresh=QPushButton("Refresh classes"); redraw=QPushButton("Redraw"); row.addWidget(refresh); row.addWidget(redraw); bl.addLayout(row)
    if not hasattr(ws,"mustatil_owl_class_combos"): ws.mustatil_owl_class_combos=[]
    ws.mustatil_owl_class_combos.append(cls_combo)
    def on_cls(*_):
        try:
            data=cls_combo.currentData(); ws.mustatil_owl_selected_class_id = None if data is None else int(data)
            for other in list(getattr(ws,"mustatil_owl_class_combos",[]) or []):
                if other is cls_combo: continue
                other.blockSignals(True)
                for i in range(other.count()):
                    if other.itemData(i)==data: other.setCurrentIndex(i); break
                other.blockSignals(False)
        except Exception: ws.mustatil_owl_selected_class_id=None
        _redraw(ws)
    cls_combo.currentIndexChanged.connect(on_cls)
    hide_fp.toggled.connect(lambda state: (setattr(ws,"mustatil_owl_hide_fp",bool(state)), _redraw(ws)))
    geo.toggled.connect(lambda state: (setattr(ws,"mustatil_owl_geo_nms_enabled",bool(state)), _redraw(ws)))
    iou.valueChanged.connect(lambda *_: (setattr(ws,"mustatil_owl_geo_nms_iou",float(iou.value())), _redraw(ws)))
    refresh.clicked.connect(lambda: _refresh_combos(ws))
    redraw.clicked.connect(lambda: _redraw(ws))
    _refresh_combos(ws)
    return box


def _append_filter_to_original_page(original_page, ws):
    try:
        layout=_find_left_layout(original_page)
        if layout is None: return False
        # Avoid duplicates on rescans.
        try:
            if original_page.findChild(object, "MustatilOwlV2ClassFilterBox") is not None:
                return True
        except Exception: pass
        _insert_after_text(layout, _make_filter_box(ws, "Class filter / Geo NMS (YOLO + OWL)"), ["post-detection filter", "filter sliders", "yolo models", "detection settings"])
        return True
    except Exception as exc:
        _log("append filter failed: "+str(exc)); traceback.print_exc(); return False



def _render_view_scene_to_pixmap(view, max_side=2600):
    """Render a QGraphicsView scene to a QPixmap even if the widget is hidden.
    This is used for the Google OWL pages: right side must show the same preview/map
    as the original YOLO pages, not a blank console or empty canvas.
    """
    try:
        from PySide6.QtGui import QPixmap, QPainter
        from PySide6.QtCore import Qt, QRectF
        if view is None or not hasattr(view, "scene") or view.scene() is None:
            return None
        scene=view.scene()
        rect=scene.sceneRect()
        try:
            if (rect.width() <= 1 or rect.height() <= 1) and scene.items():
                rect=scene.itemsBoundingRect()
        except Exception:
            pass
        if rect.width() <= 1 or rect.height() <= 1:
            return None
        scale=min(1.0, float(max_side)/max(1.0, float(max(rect.width(), rect.height()))))
        w=max(1, int(float(rect.width())*scale)); h=max(1, int(float(rect.height())*scale))
        pm=QPixmap(w,h)
        pm.fill(Qt.transparent)
        painter=QPainter(pm)
        try:
            scene.render(painter, QRectF(0,0,w,h), rect)
        finally:
            painter.end()
        return pm
    except Exception as exc:
        _log('render scene to pixmap failed: '+str(exc))
        return None


def _set_label_pixmap(label, pm):
    try:
        from PySide6.QtCore import Qt
        if label is None or pm is None:
            return False
        label.setPixmap(pm)
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(200, 200)
        return True
    except Exception:
        return False


def _refresh_owl_detection_preview_label(ws, fit=False, log=False):
    """Mirror the ORIGINAL Detection preview into the Google OWL Detection tab.
    It uses Mustatil's real image_view scene. If that scene is empty it calls loadprev().
    """
    label=getattr(ws, 'mustatil_owl_detection_preview_label', None)
    if label is None:
        return False
    try:
        # Ensure original Detection preview exists.
        view=getattr(ws, 'image_view', None)
        pm=_render_view_scene_to_pixmap(view)
        if pm is None:
            try:
                lp=getattr(ws, 'loadprev', None)
                if callable(lp): lp()
            except Exception as exc:
                try: ws.log('Google OWL preview loadprev warning: '+str(exc))
                except Exception: pass
            view=getattr(ws, 'image_view', None)
            pm=_render_view_scene_to_pixmap(view)
        if pm is not None:
            _set_label_pixmap(label, pm)
            if log:
                try: ws.log('Google OWL Detection preview mirrored from Detection tab.')
                except Exception: pass
            return True
        # Direct PIL fallback if original scene is still empty.
        try:
            from PIL import Image
            from PySide6.QtGui import QImage, QPixmap
            path=str(_get_var(getattr(ws,'image',None),'') or '').strip().strip('"')
            if path and Path(path).exists():
                img=Image.open(path).convert('RGB')
                scale=min(1.0, 2600.0/max(1, max(img.size)))
                if scale < 1.0:
                    img=img.resize((max(1,int(img.width*scale)), max(1,int(img.height*scale))))
                im=img.convert('RGBA')
                qimg=QImage(im.tobytes('raw','RGBA'), im.width, im.height, QImage.Format_RGBA8888)
                pm=QPixmap.fromImage(qimg.copy())
                _set_label_pixmap(label, pm)
                return True
        except Exception as exc:
            try: ws.log('Google OWL direct preview fallback failed: '+str(exc))
            except Exception: pass
        return False
    except Exception as exc:
        try: ws.log('Google OWL Detection preview mirror error: '+str(exc))
        except Exception: _log('Google OWL Detection preview mirror error: '+str(exc))
        return False


def _refresh_owl_satellite_preview_label(ws, fit=False, log=False):
    """Mirror the ORIGINAL Satellite Detection map scene into the Google OWL Satellite tab."""
    label=getattr(ws, 'mustatil_owl_satellite_preview_label', None)
    if label is None:
        return False
    try:
        view=getattr(ws, 'satellite_view', None)
        pm=_render_view_scene_to_pixmap(view)
        if pm is None:
            try:
                rf=getattr(ws, 'satellite_refresh_map', None)
                if callable(rf): rf()
            except Exception as exc:
                try: ws.log('Google OWL satellite refresh warning: '+str(exc))
                except Exception: pass
            view=getattr(ws, 'satellite_view', None)
            pm=_render_view_scene_to_pixmap(view)
        if pm is not None:
            _set_label_pixmap(label, pm)
            if log:
                try: ws.log('Google OWL Satellite preview mirrored from Satellite Detection tab.')
                except Exception: pass
            return True
        return False
    except Exception as exc:
        try: ws.log('Google OWL Satellite preview mirror error: '+str(exc))
        except Exception: _log('Google OWL Satellite preview mirror error: '+str(exc))
        return False


def _refresh_current_owl_preview_labels(ws):
    try: _refresh_owl_detection_preview_label(ws, fit=False, log=False)
    except Exception: pass
    try: _refresh_owl_satellite_preview_label(ws, fit=False, log=False)
    except Exception: pass


def _schedule_owl_preview_label_refresh(ws, delay_ms=120):
    try:
        from PySide6.QtCore import QTimer
        if not hasattr(ws, '_owl_preview_label_timer') or ws._owl_preview_label_timer is None:
            ws._owl_preview_label_timer=QTimer(ws)
            ws._owl_preview_label_timer.setSingleShot(True)
            ws._owl_preview_label_timer.timeout.connect(lambda: _refresh_current_owl_preview_labels(ws))
        ws._owl_preview_label_timer.start(max(20, int(delay_ms)))
    except Exception:
        _refresh_current_owl_preview_labels(ws)


def _build_owl_left_controls(ws, tab_kind: str):
    from PySide6.QtWidgets import QWidget,QVBoxLayout,QGridLayout,QGroupBox,QLabel,QLineEdit,QPushButton,QComboBox

    page = QWidget()
    left_lay = QVBoxLayout(page)
    left_lay.setContentsMargins(8,8,8,8)

    header = QLabel(
        "Google OWL = OWLv2 only: google/owlv2-base-patch16-ensemble. "
        "Nur das linke Bedienfeld ist geändert; die rechte Preview/Karte bleibt Mustatils Original."
    )
    header.setWordWrap(True)
    left_lay.addWidget(header)

    box = QGroupBox("Google OWLv2 detection")
    g = QGridLayout(box)

    device = QComboBox()
    device.setEditable(True)
    device.addItems(["cpu", "cuda", "auto"])

    classes = QLineEdit(str(getattr(ws, "mustatil_owl_classes_text", "house, car, airplane, mustatil, mound") or "house, car, airplane, mustatil, mound"))
    classes.setPlaceholderText("house, car, airplane, mustatil, mound")

    run_btn = QPushButton("Run Google OWLv2 on Detection image" if tab_kind == "detection" else "Run Google OWLv2 on selected satellite map")
    status = QLabel("Idle")
    status.setWordWrap(True)

    g.addWidget(QLabel("Model"), 0, 0)
    g.addWidget(QLabel("OWLv2 / Google"), 0, 1)
    g.addWidget(QLabel("Device"), 1, 0)
    g.addWidget(device, 1, 1)
    g.addWidget(QLabel("Classes / prompt"), 2, 0)
    g.addWidget(classes, 2, 1)
    g.addWidget(run_btn, 3, 0, 1, 2)
    g.addWidget(status, 4, 0, 1, 2)
    left_lay.addWidget(box)

    left_lay.addWidget(_make_filter_box(ws, "Class filter / Geo NMS (Google OWL)"))
    left_lay.addStretch(1)

    def append(msg):
        try:
            status.setText(str(msg))
        except Exception:
            pass
        try:
            ws.log(str(msg))
        except Exception:
            pass

    def sync():
        try:
            ws.mustatil_owl_classes_text = classes.text()
            ws.mustatil_owl_classes_edit = classes
        except Exception:
            pass
        try:
            ws.mustatil_owl_device_combo = device
        except Exception:
            pass

    classes.textChanged.connect(lambda *_: sync())
    device.currentTextChanged.connect(lambda *_: sync())

    def run_clicked():
        sync()
        append("Starting Google OWLv2...")
        if tab_kind == "detection":
            _run_as_task(ws, "Google OWLv2 Detection", lambda: _run_owl_detection_image(ws))
        else:
            _run_as_task(ws, "Google OWLv2 Satellite Detection", lambda: _run_owl_satellite(ws), allow_parallel=True)

    run_btn.clicked.connect(run_clicked)
    sync()
    return page


# Backward-compatible alias; this plugin now intentionally builds only left controls.
def _build_owl_tab(ws, tab_kind: str):
    return _build_owl_left_controls(ws, tab_kind)


def _move_layout_items(src_layout, dst_layout):
    while src_layout.count():
        item=src_layout.takeAt(0)
        if item is None: continue
        w=item.widget()
        lay=item.layout()
        sp=item.spacerItem()
        if w is not None:
            dst_layout.addWidget(w)
        elif lay is not None:
            dst_layout.addLayout(lay)
        elif sp is not None:
            dst_layout.addItem(sp)


def _wrap_page_with_inner_tabs(page: Any, tab_kind: str) -> bool:
    """Patch only the LEFT controls panel.

    This is the same working logic as the saved OWLv2 plugin, extended only by
    adding Grounding DINO and LAE-DINO tabs next to Google OWL.

    The right preview/map widget is never moved, copied, mirrored, replaced, or touched.
    """
    if id(page) in _PATCHED_PAGES:
        return False
    ws = _workspace_from_widget(page)
    if ws is None:
        return False
    _patch_workspace_filters(ws)

    try:
        from PySide6.QtWidgets import QTabWidget, QSplitter

        splitter = None
        try:
            for sp in page.findChildren(QSplitter):
                if sp.count() >= 2:
                    splitter = sp
                    break
        except Exception:
            splitter = None

        if splitter is None:
            _log(f"No QSplitter found for {tab_kind}; left-only patch skipped.")
            return False

        try:
            left_now = splitter.widget(0)
            if left_now is not None and left_now.objectName() == "MustatilOwlV2LeftTabs":
                # Already patched by this plugin. Make sure extra tabs exist.
                if isinstance(left_now, QTabWidget):
                    names=[str(left_now.tabText(i)).strip().lower() for i in range(left_now.count())]
                    if "grounding dino" not in names:
                        left_now.addTab(_build_gdino_left_controls(ws, tab_kind), "Grounding DINO")
                    names=[str(left_now.tabText(i)).strip().lower() for i in range(left_now.count())]
                    if "lae-dino" not in names:
                        left_now.addTab(_build_lae_left_controls(ws, tab_kind), "LAE-DINO")
                _PATCHED_PAGES.add(id(page))
                return True
        except Exception:
            pass

        original_left = splitter.widget(0)
        if original_left is None:
            return False

        sizes = []
        try:
            sizes = splitter.sizes()
        except Exception:
            sizes = []

        inner = QTabWidget()
        inner.setObjectName("MustatilOwlV2LeftTabs")

        # Replace the left widget only. The right preview/map remains the original splitter.widget(1).
        returned = None
        try:
            returned = splitter.replaceWidget(0, inner)
        except Exception:
            try:
                original_left.setParent(None)
                splitter.insertWidget(0, inner)
                returned = original_left
            except Exception:
                returned = original_left

        if returned is None:
            returned = original_left

        inner.addTab(returned, "YOLO / Original")
        inner.addTab(_build_owl_left_controls(ws, tab_kind), "Google OWL")
        inner.addTab(_build_gdino_left_controls(ws, tab_kind), "Grounding DINO")
        inner.addTab(_build_lae_left_controls(ws, tab_kind), "LAE-DINO")

        # Add the same filter to original YOLO controls, but do not touch the right preview.
        try:
            _append_filter_to_original_page(returned, ws)
        except Exception:
            pass

        try:
            if sizes and len(sizes) >= 2:
                splitter.setSizes(sizes)
        except Exception:
            pass

        def _left_tab_changed(ix, _inner=inner, _ws=ws, _kind=tab_kind):
            try:
                # No preview swapping. Only refresh overlays/classes when switching.
                _refresh_combos(_ws)
                _redraw(_ws)
            except Exception:
                pass

        try:
            inner.currentChanged.connect(_left_tab_changed)
        except Exception:
            pass

        _PATCHED_PAGES.add(id(page))
        try:
            ws.log(f"Model left-panel tabs added to {tab_kind}. Right preview unchanged.")
        except Exception:
            pass
        _log(f"Patched only left controls of {tab_kind}; right preview untouched.")
        return True

    except Exception as exc:
        _log("left-panel patch failed: " + str(exc))
        traceback.print_exc()
        return False


def _scan_tabs(root=None):
    try:
        from PySide6.QtWidgets import QApplication,QTabWidget
        widgets=[]
        if root is not None:
            try: widgets += root.findChildren(QTabWidget)
            except Exception: pass
            try:
                if isinstance(root,QTabWidget): widgets.append(root)
            except Exception: pass
        app=QApplication.instance()
        if app is not None:
            for w in app.allWidgets():
                try:
                    if isinstance(w,QTabWidget): widgets.append(w)
                except Exception: pass
        seen=[]
        for tw in widgets:
            if id(tw) not in seen:
                seen.append(id(tw))
            else:
                continue
            for i in range(tw.count()):
                label=str(tw.tabText(i) or "").strip().lower()
                page=tw.widget(i)
                if label == "detection": _wrap_page_with_inner_tabs(page,"detection")
                elif label == "satellite detection": _wrap_page_with_inner_tabs(page,"satellite")
    except Exception as exc: _log("scan tabs failed: "+str(exc))


def _install_hook():
    global _PATCHED_QTAB,_ORIG_ADD,_ORIG_INSERT,_SCAN_TIMER
    if _PATCHED_QTAB: return
    try:
        from PySide6.QtWidgets import QTabWidget
        from PySide6.QtCore import QTimer
    except Exception as exc:
        _log("Qt unavailable: "+str(exc)); return
    _ORIG_ADD=QTabWidget.addTab; _ORIG_INSERT=QTabWidget.insertTab
    def after(page,label):
        try:
            norm=str(label or "").strip().lower()
            if norm == "detection":
                QTimer.singleShot(200,lambda p=page:_wrap_page_with_inner_tabs(p,"detection")); QTimer.singleShot(1200,lambda p=page:_wrap_page_with_inner_tabs(p,"detection"))
            elif norm == "satellite detection":
                QTimer.singleShot(200,lambda p=page:_wrap_page_with_inner_tabs(p,"satellite")); QTimer.singleShot(1200,lambda p=page:_wrap_page_with_inner_tabs(p,"satellite"))
        except Exception as exc: _log("after hook warning: "+str(exc))
    def addTab_patched(self,page,*args,**kwargs):
        res=_ORIG_ADD(self,page,*args,**kwargs)
        label=""
        for a in reversed(args):
            if isinstance(a,str): label=a; break
        if not label:
            try: label=self.tabText(int(res))
            except Exception: pass
        after(page,label); return res
    def insertTab_patched(self,index,page,*args,**kwargs):
        res=_ORIG_INSERT(self,index,page,*args,**kwargs)
        label=""
        for a in reversed(args):
            if isinstance(a,str): label=a; break
        if not label:
            try: label=self.tabText(int(res))
            except Exception: pass
        after(page,label); return res
    QTabWidget.addTab=addTab_patched; QTabWidget.insertTab=insertTab_patched; _PATCHED_QTAB=True
    try:
        _SCAN_TIMER=QTimer(); _SCAN_TIMER.setInterval(1500); _SCAN_TIMER.timeout.connect(lambda: _scan_tabs()) ; _SCAN_TIMER.start()
        QTimer.singleShot(300,lambda:_scan_tabs()); QTimer.singleShot(2500,lambda:_scan_tabs())
    except Exception: pass
    _log("OWLv2 tab hook installed")


def mustatil_plugin_init(): _install_hook(); _scan_tabs()
def register_plugin(app=None, main_window=None): _install_hook(); _scan_tabs(main_window or app); return True
def init_plugin(app=None, main_window=None): return register_plugin(app,main_window)
def load_plugin(app=None, main_window=None): return register_plugin(app,main_window)
try: _install_hook()
except Exception: pass
