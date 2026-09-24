#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mustatil AI Pipeline Video Input Patch v1
----------------------------------------
Drop-in plugin for Mustatil's existing AI Pipeline tab.

Goal:
- Add a compact video input bar directly below the existing image/original-image selection.
- Treat video frames as normal image inputs for the existing AI Pipeline blocks.
- Run the already existing pipeline frame-by-frame without changing the pipeline core.

Install:
- Put this file into mustatil_plugins.
- Keep the existing AI Pipeline plugin active.
- Restart Mustatil.

This plugin is deliberately additive: it does not replace AI Pipeline, Detection,
Satellite Detection, or the video detection page. It only patches tabs that expose:
  pipeline_image_path + run_pipeline()
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import (
        QApplication, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel,
        QLineEdit, QPushButton, QFileDialog, QMessageBox, QGroupBox, QSpinBox,
        QCheckBox, QProgressBar, QSizePolicy, QTabWidget
    )
except Exception:  # Mustatil currently uses PySide6, but keep import failure harmless.
    Qt = None
    QTimer = None
    QApplication = None
    QWidget = None
    QHBoxLayout = None
    QVBoxLayout = None
    QGridLayout = None
    QLabel = None
    QLineEdit = None
    QPushButton = None
    QFileDialog = None
    QMessageBox = None
    QGroupBox = None
    QSpinBox = None
    QCheckBox = None
    QProgressBar = None
    QSizePolicy = None
    QTabWidget = None
    QColor = None


_PATCHED_TABS = set()
_QTAB_PATCHED = False
_ORIG_ADD_TAB = None
_ORIG_INSERT_TAB = None
_SCAN_COUNTER = 0


def _log(msg: str):
    try:
        print("[Mustatil AI Pipeline Video Input] " + str(msg))
    except Exception:
        pass


def _safe_title(widget: Any) -> str:
    try:
        if isinstance(widget, QGroupBox):
            return str(widget.title() or "")
    except Exception:
        pass
    return ""


def _is_ai_pipeline_tab(w: Any) -> bool:
    """Detect Mustatil's AI Pipeline/YoloPipelineTab instance without importing it."""
    if w is None:
        return False
    try:
        if str(getattr(w, "objectName", lambda: "")() or "") == "MustatilAIPipelineNextToLAEDINOTrainer":
            return True
    except Exception:
        pass
    # Functional signature of the existing AI Pipeline tab.
    try:
        if hasattr(w, "pipeline_image_path") and hasattr(w, "run_pipeline") and hasattr(w, "blocks"):
            return True
    except Exception:
        pass
    return False


def _find_workspace(tab: Any) -> Any:
    try:
        ws = getattr(tab, "ws", None)
        if ws is not None:
            return ws
    except Exception:
        pass
    cur = tab
    for _ in range(80):
        if cur is None:
            break
        try:
            for attr in ("ws", "workspace", "main_window"):
                obj = getattr(cur, attr, None)
                if obj is not None:
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


def _tab_log(tab: Any, msg: str):
    try:
        if hasattr(tab, "_log"):
            tab._log("[Video input] " + str(msg))
            return
    except Exception:
        pass
    try:
        ws = _find_workspace(tab)
        if ws is not None and hasattr(ws, "log"):
            ws.log("[AI Pipeline Video] " + str(msg))
            return
    except Exception:
        pass
    _log(msg)


def _require_cv2():
    try:
        import cv2  # noqa
        return cv2
    except Exception as exc:
        raise RuntimeError(
            "OpenCV/cv2 is missing. Install opencv-python in the Mustatil environment, "
            "or keep a Mustatil version/plugin active that already includes cv2. Original error: " + str(exc)
        )


def _records_by_frame(records: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    by: Dict[int, List[Dict[str, Any]]] = {}
    for r in records or []:
        try:
            fi = int(r.get("video_frame", r.get("frame_index", -1)))
        except Exception:
            fi = -1
        if fi >= 0:
            by.setdefault(fi, []).append(r)
    return by


class VideoPipelineController:
    """Small state machine attached to one AI Pipeline tab."""

    def __init__(self, tab: Any):
        self.tab = tab
        self.ws = _find_workspace(tab)
        self.video_path = ""
        self.output_dir = ""
        self.total_frames = 0
        self.fps = 0.0
        self.width = 0
        self.height = 0
        self.current_queue: List[int] = []
        self.queue_pos = 0
        self.running = False
        self.paused = False
        self.stop_requested = False
        self.results: List[Dict[str, Any]] = []
        self.results_by_block: Dict[str, List[Dict[str, Any]]] = {}
        self.last_frame_path = ""
        self.started_at = 0.0

    # ---------- paths / video ----------
    def default_output_dir(self) -> Path:
        if self.output_dir:
            return Path(self.output_dir)
        base = None
        try:
            if self.video_path:
                base = Path(self.video_path).parent
        except Exception:
            base = None
        if base is None:
            base = Path.home() / "Desktop"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return base / f"mustatil_ai_pipeline_video_frames_{stamp}"

    def set_video(self, path: str):
        self.video_path = str(path or "")
        self.probe_video()

    def probe_video(self):
        self.total_frames = 0
        self.fps = 0.0
        self.width = 0
        self.height = 0
        if not self.video_path:
            return
        cv2 = _require_cv2()
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError("Could not open video: " + str(self.video_path))
        try:
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        finally:
            cap.release()

    def frame_time(self, frame_index: int) -> float:
        if self.fps and self.fps > 0:
            return float(frame_index) / float(self.fps)
        return 0.0

    def _save_frame(self, frame_index: int, target_dir: Path, prefix: str = "frame") -> Path:
        cv2 = _require_cv2()
        if not self.video_path:
            raise RuntimeError("Choose a video first.")
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError("Could not open video: " + str(self.video_path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Could not read frame {frame_index}.")
            target_dir.mkdir(parents=True, exist_ok=True)
            out = target_dir / f"{prefix}_{int(frame_index):08d}.jpg"
            if not cv2.imwrite(str(out), frame):
                raise RuntimeError("Could not write frame image: " + str(out))
            self.last_frame_path = str(out)
            return out
        finally:
            cap.release()

    # ---------- pipeline calls ----------
    def _set_pipeline_image(self, frame_path: Path):
        try:
            self.tab.pipeline_image_path.setText(str(frame_path))
        except Exception:
            pass
        try:
            if hasattr(self.tab, "_sync_pipeline_image_to_workspace"):
                self.tab._sync_pipeline_image_to_workspace(str(frame_path))
        except Exception:
            pass
        try:
            if self.ws is not None and hasattr(self.ws, "image") and hasattr(self.ws.image, "set"):
                self.ws.image.set(str(frame_path))
        except Exception:
            pass

    def run_pipeline_on_frame(self, frame_index: int, output_dir: Optional[Path] = None, collect: bool = True) -> List[Dict[str, Any]]:
        if output_dir is None:
            output_dir = self.default_output_dir()
        frames_dir = output_dir / "frames"
        frame_path = self._save_frame(int(frame_index), frames_dir)
        self._set_pipeline_image(frame_path)
        _tab_log(self.tab, f"Frame {frame_index}: sent to AI Pipeline as original image: {frame_path.name}")
        recs = self.tab.run_pipeline()
        recs = list(recs or getattr(self.tab, "results", []) or [])
        enriched = []
        for r in recs:
            rr = dict(r)
            rr["video_path"] = str(self.video_path)
            rr["video_frame"] = int(frame_index)
            rr["frame_index"] = int(frame_index)
            rr["time_seconds"] = float(self.frame_time(int(frame_index)))
            rr["frame_image"] = str(frame_path)
            rr.setdefault("source_image", str(frame_path))
            enriched.append(rr)
        if collect:
            self.results.extend(enriched)
            # Merge per-block too. This lets the AI Pipeline's normal result tools still see something useful.
            try:
                for block_id, block_recs in dict(getattr(self.tab, "results_by_block", {}) or {}).items():
                    bucket = self.results_by_block.setdefault(block_id, [])
                    for br in block_recs or []:
                        rr = dict(br)
                        rr["video_path"] = str(self.video_path)
                        rr["video_frame"] = int(frame_index)
                        rr["frame_index"] = int(frame_index)
                        rr["time_seconds"] = float(self.frame_time(int(frame_index)))
                        rr["frame_image"] = str(frame_path)
                        rr.setdefault("source_image", str(frame_path))
                        bucket.append(rr)
            except Exception:
                pass
        return enriched

    def finalize_results_on_tab(self):
        try:
            self.tab.video_pipeline_results = list(self.results)
            self.tab.results = list(self.results)
            if self.results_by_block:
                self.tab.results_by_block = {k: list(v) for k, v in self.results_by_block.items()}
        except Exception:
            pass
        try:
            if hasattr(self.tab, "_write_summary"):
                self.tab._write_summary()
        except Exception:
            pass

    # ---------- export ----------
    def export_json(self, path: str):
        data = {
            "kind": "mustatil_ai_pipeline_video_results",
            "video_path": self.video_path,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "records": self.results or list(getattr(self.tab, "video_pipeline_results", []) or []),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def export_csv(self, path: str):
        recs = self.results or list(getattr(self.tab, "video_pipeline_results", []) or [])
        cols = [
            "video_frame", "time_seconds", "block_name", "block_type", "label", "class_id", "conf",
            "x1", "y1", "x2", "y2", "model", "source_image", "frame_image", "video_path", "id",
        ]
        extra = sorted({str(k) for r in recs for k in r.keys() if str(k) not in cols and isinstance(k, str)})
        fieldnames = cols + extra
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in recs:
                writer.writerow(r)

    def export_annotated_video(self, path: str):
        cv2 = _require_cv2()
        recs = self.results or list(getattr(self.tab, "video_pipeline_results", []) or [])
        if not recs:
            raise RuntimeError("No video pipeline records are available. Run video pipeline first.")
        if not self.video_path:
            raise RuntimeError("No source video selected.")
        by_frame = _records_by_frame(recs)
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError("Could not open source video: " + str(self.video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or self.fps or 25.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.width or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.height or 0)
        if w <= 0 or h <= 0:
            raise RuntimeError("Could not determine video size.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Could not create annotated video: " + str(path))
        try:
            frame_i = 0
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                for r in by_frame.get(frame_i, []):
                    try:
                        x1 = int(round(float(r.get("x1", 0))))
                        y1 = int(round(float(r.get("y1", 0))))
                        x2 = int(round(float(r.get("x2", 0))))
                        y2 = int(round(float(r.get("y2", 0))))
                        label = str(r.get("label", r.get("class_id", "object")))
                        conf = r.get("conf", r.get("score", ""))
                        txt = f"{label} {float(conf):.2f}" if isinstance(conf, (float, int)) or str(conf).replace('.', '', 1).isdigit() else label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, txt, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
                    except Exception:
                        pass
                writer.write(frame)
                frame_i += 1
        finally:
            cap.release()
            writer.release()


def _ui_status(tab: Any, text: str):
    try:
        st = getattr(tab, "_video_pipeline_status", None)
        if st is not None:
            st.setText(str(text))
    except Exception:
        pass
    _tab_log(tab, text)


def _controller(tab: Any) -> VideoPipelineController:
    c = getattr(tab, "_video_pipeline_controller", None)
    if c is None:
        c = VideoPipelineController(tab)
        tab._video_pipeline_controller = c
    return c


def _choose_video(tab: Any):
    try:
        current = ""
        try:
            current = str(getattr(tab, "_video_pipeline_video_path").text() or "")
        except Exception:
            pass
        start_dir = str(Path(current).parent) if current and Path(current).exists() else str(Path.home() / "Desktop")
        path, _ = QFileDialog.getOpenFileName(
            tab,
            "Choose video for AI Pipeline",
            start_dir,
            "Video files (*.mp4 *.avi *.mov *.mkv *.m4v *.wmv);;All files (*)",
        )
        if not path:
            return
        c = _controller(tab)
        c.set_video(path)
        try:
            tab._video_pipeline_video_path.setText(path)
            tab._video_pipeline_start_spin.setMaximum(max(0, c.total_frames - 1))
            tab._video_pipeline_end_spin.setMaximum(max(0, c.total_frames - 1))
            tab._video_pipeline_current_spin.setMaximum(max(0, c.total_frames - 1))
            tab._video_pipeline_end_spin.setValue(max(0, c.total_frames - 1))
            tab._video_pipeline_progress.setMaximum(max(1, c.total_frames))
            tab._video_pipeline_meta.setText(f"{c.total_frames} frames | {c.fps:.3f} fps | {c.width}×{c.height}")
        except Exception:
            pass
        _ui_status(tab, "Video loaded: " + path)
    except Exception as exc:
        QMessageBox.critical(tab, "AI Pipeline video", str(exc))


def _choose_output_dir(tab: Any):
    try:
        c = _controller(tab)
        start = c.output_dir or str(Path.home() / "Desktop")
        path = QFileDialog.getExistingDirectory(tab, "Choose output folder for extracted video frames", start)
        if not path:
            return
        c.output_dir = path
        try:
            tab._video_pipeline_output_dir.setText(path)
        except Exception:
            pass
    except Exception as exc:
        QMessageBox.critical(tab, "AI Pipeline video", str(exc))


def _sync_video_path(tab: Any):
    try:
        c = _controller(tab)
        path = str(tab._video_pipeline_video_path.text() or "").strip().strip('"')
        if path and path != c.video_path:
            c.set_video(path)
    except Exception:
        pass


def _sync_output_dir(tab: Any):
    try:
        c = _controller(tab)
        c.output_dir = str(tab._video_pipeline_output_dir.text() or "").strip().strip('"')
    except Exception:
        pass


def _make_frame_queue(tab: Any, full_video: bool = True) -> List[int]:
    c = _controller(tab)
    _sync_video_path(tab)
    _sync_output_dir(tab)
    if not c.video_path:
        raise RuntimeError("Choose a video first.")
    if c.total_frames <= 0:
        c.probe_video()
    every = max(1, int(tab._video_pipeline_every_spin.value()))
    max_frames = max(0, int(tab._video_pipeline_max_spin.value()))
    if full_video:
        start = 0
        end = max(0, c.total_frames - 1)
    else:
        start = max(0, int(tab._video_pipeline_start_spin.value()))
        end = int(tab._video_pipeline_end_spin.value())
        if end <= 0 or end >= c.total_frames:
            end = max(0, c.total_frames - 1)
        if start > end:
            start, end = end, start
    frames = list(range(start, end + 1, every))
    if max_frames > 0:
        frames = frames[:max_frames]
    if not frames:
        raise RuntimeError("No frames selected.")
    return frames


def _extract_current_frame(tab: Any, run_pipeline: bool = False):
    try:
        c = _controller(tab)
        _sync_video_path(tab)
        _sync_output_dir(tab)
        output_dir = c.default_output_dir()
        frame_i = int(tab._video_pipeline_current_spin.value())
        frame_path = c._save_frame(frame_i, output_dir / "frames", prefix="manual_frame")
        c._set_pipeline_image(frame_path)
        if run_pipeline:
            c.results = []
            c.results_by_block = {}
            recs = c.run_pipeline_on_frame(frame_i, output_dir=output_dir, collect=True)
            c.finalize_results_on_tab()
            _ui_status(tab, f"Current frame detection finished: frame {frame_i}, {len(recs)} record(s).")
        else:
            _ui_status(tab, f"Frame extracted and set as pipeline image: {frame_path}")
    except Exception as exc:
        QMessageBox.critical(tab, "AI Pipeline video", str(exc))


def _start_video_pipeline(tab: Any, full_video: bool):
    try:
        c = _controller(tab)
        if c.running:
            QMessageBox.information(tab, "AI Pipeline video", "Video pipeline is already running.")
            return
        frames = _make_frame_queue(tab, full_video=full_video)
        out_dir = c.default_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        c.output_dir = str(out_dir)
        try:
            tab._video_pipeline_output_dir.setText(str(out_dir))
        except Exception:
            pass
        c.current_queue = frames
        c.queue_pos = 0
        c.running = True
        c.paused = False
        c.stop_requested = False
        c.results = []
        c.results_by_block = {}
        c.started_at = time.time()
        try:
            tab._video_pipeline_progress.setMaximum(len(frames))
            tab._video_pipeline_progress.setValue(0)
            tab._video_pipeline_pause_btn.setText("Pause")
        except Exception:
            pass
        mode = "full video" if full_video else "selected range"
        _ui_status(tab, f"Starting AI Pipeline on {mode}: {len(frames)} frame(s). Output: {out_dir}")
        QTimer.singleShot(0, lambda tab=tab: _process_next_frame(tab))
    except Exception as exc:
        QMessageBox.critical(tab, "AI Pipeline video", str(exc))


def _process_next_frame(tab: Any):
    c = _controller(tab)
    if not c.running:
        return
    if c.stop_requested:
        c.running = False
        c.paused = False
        c.finalize_results_on_tab()
        _ui_status(tab, f"Stopped. Collected {len(c.results)} result record(s).")
        return
    if c.paused:
        QTimer.singleShot(250, lambda tab=tab: _process_next_frame(tab))
        return
    if c.queue_pos >= len(c.current_queue):
        c.running = False
        c.paused = False
        c.finalize_results_on_tab()
        elapsed = time.time() - c.started_at if c.started_at else 0.0
        _ui_status(tab, f"Video AI Pipeline finished: {len(c.current_queue)} frame(s), {len(c.results)} record(s), {elapsed:.1f}s.")
        return
    frame_i = int(c.current_queue[c.queue_pos])
    try:
        output_dir = Path(c.output_dir) if c.output_dir else c.default_output_dir()
        recs = c.run_pipeline_on_frame(frame_i, output_dir=output_dir, collect=True)
        c.queue_pos += 1
        try:
            tab._video_pipeline_current_spin.setValue(frame_i)
            tab._video_pipeline_progress.setValue(c.queue_pos)
        except Exception:
            pass
        _ui_status(tab, f"Frame {frame_i}: {len(recs)} record(s). Progress {c.queue_pos}/{len(c.current_queue)}")
    except Exception as exc:
        # Stop on errors because pipeline errors usually mean wrong model/config.
        c.running = False
        c.paused = False
        c.finalize_results_on_tab()
        tb = traceback.format_exc()
        _tab_log(tab, "Video pipeline error:\n" + tb)
        QMessageBox.critical(tab, "AI Pipeline video", f"Frame {frame_i} failed:\n{exc}")
        return
    QTimer.singleShot(10, lambda tab=tab: _process_next_frame(tab))


def _pause_resume(tab: Any):
    c = _controller(tab)
    if not c.running:
        return
    c.paused = not c.paused
    try:
        tab._video_pipeline_pause_btn.setText("Resume" if c.paused else "Pause")
    except Exception:
        pass
    _ui_status(tab, "Paused." if c.paused else "Resumed.")


def _stop(tab: Any):
    c = _controller(tab)
    c.stop_requested = True
    _ui_status(tab, "Stop requested. Finishing current frame…")


def _export_json_clicked(tab: Any):
    try:
        c = _controller(tab)
        default = str((Path(c.output_dir) if c.output_dir else c.default_output_dir()) / "ai_pipeline_video_results.json")
        path, _ = QFileDialog.getSaveFileName(tab, "Export AI Pipeline video JSON", default, "JSON (*.json);;All files (*)")
        if not path:
            return
        c.export_json(str(Path(path).with_suffix(".json")))
        _ui_status(tab, "Exported JSON: " + str(Path(path).with_suffix(".json")))
    except Exception as exc:
        QMessageBox.critical(tab, "AI Pipeline video export", str(exc))


def _export_csv_clicked(tab: Any):
    try:
        c = _controller(tab)
        default = str((Path(c.output_dir) if c.output_dir else c.default_output_dir()) / "ai_pipeline_video_results.csv")
        path, _ = QFileDialog.getSaveFileName(tab, "Export AI Pipeline video CSV", default, "CSV (*.csv);;All files (*)")
        if not path:
            return
        c.export_csv(str(Path(path).with_suffix(".csv")))
        _ui_status(tab, "Exported CSV: " + str(Path(path).with_suffix(".csv")))
    except Exception as exc:
        QMessageBox.critical(tab, "AI Pipeline video export", str(exc))


def _export_annotated_video_clicked(tab: Any):
    try:
        c = _controller(tab)
        default = str((Path(c.output_dir) if c.output_dir else c.default_output_dir()) / "ai_pipeline_annotated_video.mp4")
        path, _ = QFileDialog.getSaveFileName(tab, "Export annotated AI Pipeline video", default, "MP4 video (*.mp4);;All files (*)")
        if not path:
            return
        c.export_annotated_video(str(Path(path).with_suffix(".mp4")))
        _ui_status(tab, "Exported annotated video: " + str(Path(path).with_suffix(".mp4")))
    except Exception as exc:
        QMessageBox.critical(tab, "AI Pipeline video export", str(exc))


def _open_output_folder(tab: Any):
    try:
        c = _controller(tab)
        folder = Path(c.output_dir) if c.output_dir else c.default_output_dir()
        folder.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(str(folder))  # type: ignore[attr-defined]
        else:
            import subprocess
            subprocess.Popen(["xdg-open", str(folder)])
    except Exception as exc:
        QMessageBox.critical(tab, "AI Pipeline video", str(exc))


def _make_video_bar(tab: Any) -> QGroupBox:
    """Create a compact, low-height video input bar for the existing AI Pipeline tab.

    v2 layout change:
    - Action/export buttons are arranged horizontally in compact rows.
    - Path fields are shorter so the bar does not dominate the AI Pipeline page.
    - Margins, spacing and widget heights are reduced.
    """
    box = QGroupBox("Pipeline video input / frames as images")
    try:
        box.setObjectName("MustatilAIPipelineVideoInputBar")
        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        box.setStyleSheet(
            "QGroupBox#MustatilAIPipelineVideoInputBar {"
            " font-weight: bold; border: 1px solid #666; border-radius: 5px; margin-top: 6px; }"
            "QGroupBox#MustatilAIPipelineVideoInputBar::title {"
            " subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
            "QPushButton { min-height: 21px; max-height: 24px; padding: 1px 7px; }"
            "QLineEdit { min-height: 21px; max-height: 24px; }"
            "QSpinBox { min-height: 21px; max-height: 24px; }"
            "QLabel { font-weight: normal; }"
        )
    except Exception:
        pass

    lay = QVBoxLayout(box)
    lay.setContentsMargins(7, 8, 7, 6)
    lay.setSpacing(4)

    def _btn(text: str, width: int = 92) -> QPushButton:
        b = QPushButton(text)
        try:
            b.setMaximumWidth(width)
            b.setMinimumWidth(min(width, 66))
            b.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        except Exception:
            pass
        return b

    def _lbl(text: str) -> QLabel:
        l = QLabel(text)
        try:
            l.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        except Exception:
            pass
        return l

    def _spin(default: int = 0, minv: int = 0, maxv: int = 999999999, width: int = 72) -> QSpinBox:
        s = QSpinBox()
        s.setRange(minv, maxv)
        s.setValue(default)
        try:
            s.setMaximumWidth(width)
            s.setMinimumWidth(58)
            s.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        except Exception:
            pass
        return s

    # --- Row 1: video + output paths, compact and side by side ---
    tab._video_pipeline_video_path = QLineEdit()
    tab._video_pipeline_video_path.setPlaceholderText("Video path")
    tab._video_pipeline_video_path.textChanged.connect(lambda *_: _sync_video_path(tab))
    try:
        tab._video_pipeline_video_path.setMaximumWidth(390)
    except Exception:
        pass
    btn_video = _btn("Video…", 76)
    btn_video.setToolTip("Choose video file")
    btn_video.clicked.connect(lambda *_: _choose_video(tab))

    tab._video_pipeline_output_dir = QLineEdit()
    tab._video_pipeline_output_dir.setPlaceholderText("Output folder")
    tab._video_pipeline_output_dir.textChanged.connect(lambda *_: _sync_output_dir(tab))
    try:
        tab._video_pipeline_output_dir.setMaximumWidth(330)
    except Exception:
        pass
    btn_out = _btn("Output…", 82)
    btn_out.setToolTip("Choose output folder")
    btn_out.clicked.connect(lambda *_: _choose_output_dir(tab))

    row_paths = QHBoxLayout()
    row_paths.setSpacing(5)
    row_paths.addWidget(_lbl("Video"))
    row_paths.addWidget(tab._video_pipeline_video_path, 1)
    row_paths.addWidget(btn_video)
    row_paths.addSpacing(8)
    row_paths.addWidget(_lbl("Output"))
    row_paths.addWidget(tab._video_pipeline_output_dir, 1)
    row_paths.addWidget(btn_out)
    lay.addLayout(row_paths)

    # --- Row 2: frame/range controls + main run buttons ---
    tab._video_pipeline_every_spin = _spin(default=10, minv=1, maxv=999999, width=64)
    tab._video_pipeline_max_spin = _spin(default=0, minv=0, maxv=999999, width=64)
    try:
        tab._video_pipeline_max_spin.setSpecialValueText("all")
    except Exception:
        pass
    tab._video_pipeline_start_spin = _spin(default=0, minv=0, maxv=999999999, width=76)
    tab._video_pipeline_end_spin = _spin(default=0, minv=0, maxv=999999999, width=76)
    tab._video_pipeline_current_spin = _spin(default=0, minv=0, maxv=999999999, width=76)

    btn_full = _btn("Full video", 88)
    btn_full.setToolTip("Run the existing AI Pipeline on the whole video. Every N frames controls sampling.")
    btn_full.clicked.connect(lambda *_: _start_video_pipeline(tab, full_video=True))
    btn_range = _btn("Range", 72)
    btn_range.setToolTip("Run the existing AI Pipeline only from Start frame to End frame.")
    btn_range.clicked.connect(lambda *_: _start_video_pipeline(tab, full_video=False))
    btn_extract = _btn("Set frame", 82)
    btn_extract.setToolTip("Extract current frame and set it as the AI Pipeline image without running blocks.")
    btn_extract.clicked.connect(lambda *_: _extract_current_frame(tab, run_pipeline=False))
    btn_frame_detect = _btn("Frame detect", 96)
    btn_frame_detect.setToolTip("Run the existing AI Pipeline only on the current frame.")
    btn_frame_detect.clicked.connect(lambda *_: _extract_current_frame(tab, run_pipeline=True))
    tab._video_pipeline_pause_btn = _btn("Pause", 64)
    tab._video_pipeline_pause_btn.clicked.connect(lambda *_: _pause_resume(tab))
    btn_stop = _btn("Stop", 58)
    btn_stop.clicked.connect(lambda *_: _stop(tab))

    row_controls = QHBoxLayout()
    row_controls.setSpacing(5)
    for label, spin in (
        ("Every", tab._video_pipeline_every_spin),
        ("Max", tab._video_pipeline_max_spin),
        ("Start", tab._video_pipeline_start_spin),
        ("End", tab._video_pipeline_end_spin),
        ("Current", tab._video_pipeline_current_spin),
    ):
        row_controls.addWidget(_lbl(label))
        row_controls.addWidget(spin)
    row_controls.addSpacing(8)
    for b in (btn_full, btn_range, btn_extract, btn_frame_detect, tab._video_pipeline_pause_btn, btn_stop):
        row_controls.addWidget(b)
    row_controls.addStretch(1)
    lay.addLayout(row_controls)

    # --- Row 3: exports + progress/status ---
    btn_csv = _btn("CSV", 58)
    btn_csv.clicked.connect(lambda *_: _export_csv_clicked(tab))
    btn_json = _btn("JSON", 62)
    btn_json.clicked.connect(lambda *_: _export_json_clicked(tab))
    btn_annot = _btn("MP4", 58)
    btn_annot.setToolTip("Export annotated MP4")
    btn_annot.clicked.connect(lambda *_: _export_annotated_video_clicked(tab))
    btn_open = _btn("Folder", 70)
    btn_open.clicked.connect(lambda *_: _open_output_folder(tab))

    tab._video_pipeline_progress = QProgressBar()
    tab._video_pipeline_progress.setRange(0, 1)
    tab._video_pipeline_progress.setValue(0)
    try:
        tab._video_pipeline_progress.setMaximumHeight(18)
        tab._video_pipeline_progress.setMinimumWidth(160)
    except Exception:
        pass

    tab._video_pipeline_status = QLabel("Ready. Video frames are passed to the existing AI Pipeline blocks.")
    tab._video_pipeline_status.setWordWrap(False)
    try:
        tab._video_pipeline_status.setMinimumWidth(220)
        tab._video_pipeline_status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    except Exception:
        pass
    tab._video_pipeline_meta = QLabel("No video")
    try:
        tab._video_pipeline_meta.setStyleSheet("color: #666;")
        tab._video_pipeline_meta.setMaximumWidth(260)
    except Exception:
        pass

    row_exports = QHBoxLayout()
    row_exports.setSpacing(5)
    row_exports.addWidget(_lbl("Export"))
    for b in (btn_csv, btn_json, btn_annot, btn_open):
        row_exports.addWidget(b)
    row_exports.addSpacing(8)
    row_exports.addWidget(tab._video_pipeline_meta)
    row_exports.addWidget(tab._video_pipeline_progress, 1)
    row_exports.addWidget(tab._video_pipeline_status, 2)
    lay.addLayout(row_exports)

    return box

def _insert_video_bar(tab: Any) -> bool:
    if tab is None or id(tab) in _PATCHED_TABS:
        return False
    if not _is_ai_pipeline_tab(tab):
        return False
    try:
        if getattr(tab, "_mustatil_video_input_patch_installed", False):
            _PATCHED_TABS.add(id(tab))
            return True
    except Exception:
        pass
    try:
        root = tab.layout()
    except Exception:
        root = None
    if root is None:
        return False

    # Do not add it twice if Qt keeps the old Python object id unavailable.
    try:
        existing = tab.findChildren(QGroupBox, "MustatilAIPipelineVideoInputBar")
        if existing:
            tab._mustatil_video_input_patch_installed = True
            _PATCHED_TABS.add(id(tab))
            return True
    except Exception:
        pass

    bar = _make_video_bar(tab)

    insert_at = 1
    try:
        for i in range(root.count()):
            item = root.itemAt(i)
            w = item.widget() if item is not None else None
            title = _safe_title(w).strip().lower()
            if "pipeline image" in title or "original image" in title:
                insert_at = i + 1
                break
        root.insertWidget(insert_at, bar)
    except Exception:
        try:
            root.addWidget(bar)
        except Exception:
            return False
    try:
        tab._mustatil_video_input_patch_installed = True
    except Exception:
        pass
    _PATCHED_TABS.add(id(tab))
    _tab_log(tab, "Video input bar inserted below image selection. Video frames are now valid AI Pipeline inputs.")
    return True


def _scan_for_ai_pipeline_tabs(root: Any = None) -> bool:
    if QApplication is None:
        return False
    ok = False
    widgets = []
    try:
        if root is not None:
            if _is_ai_pipeline_tab(root):
                widgets.append(root)
            try:
                widgets.extend(root.findChildren(QWidget))
            except Exception:
                pass
    except Exception:
        pass
    try:
        app = QApplication.instance()
        if app is not None:
            widgets.extend(app.allWidgets())
    except Exception:
        pass
    seen = set()
    for w in widgets:
        try:
            if id(w) in seen:
                continue
            seen.add(id(w))
            if _is_ai_pipeline_tab(w):
                if _insert_video_bar(w):
                    ok = True
        except Exception:
            pass
    return ok


def _install_qtab_hook(root: Any = None):
    global _QTAB_PATCHED, _ORIG_ADD_TAB, _ORIG_INSERT_TAB
    if QTabWidget is None or QTimer is None:
        return False
    if not _QTAB_PATCHED:
        _ORIG_ADD_TAB = QTabWidget.addTab
        _ORIG_INSERT_TAB = QTabWidget.insertTab

        def after_tab_change(tw: Any, label: str = ""):
            try:
                txt = str(label or "").lower()
                if "pipeline" in txt or "trainer" in txt or "ai" in txt:
                    QTimer.singleShot(50, lambda tw=tw: _scan_for_ai_pipeline_tabs(tw))
                    QTimer.singleShot(300, lambda tw=tw: _scan_for_ai_pipeline_tabs(tw))
                    QTimer.singleShot(1000, lambda tw=tw: _scan_for_ai_pipeline_tabs(tw))
            except Exception:
                pass

        def addTab_patched(self, page, *args, **kwargs):
            res = _ORIG_ADD_TAB(self, page, *args, **kwargs)
            label = ""
            try:
                for a in reversed(args):
                    if isinstance(a, str):
                        label = a
                        break
                if not label:
                    label = self.tabText(int(res))
            except Exception:
                pass
            try:
                if _is_ai_pipeline_tab(page):
                    QTimer.singleShot(0, lambda page=page: _insert_video_bar(page))
            except Exception:
                pass
            after_tab_change(self, label)
            return res

        def insertTab_patched(self, index, page, *args, **kwargs):
            res = _ORIG_INSERT_TAB(self, index, page, *args, **kwargs)
            label = ""
            try:
                for a in reversed(args):
                    if isinstance(a, str):
                        label = a
                        break
                if not label:
                    label = self.tabText(int(res))
            except Exception:
                pass
            try:
                if _is_ai_pipeline_tab(page):
                    QTimer.singleShot(0, lambda page=page: _insert_video_bar(page))
            except Exception:
                pass
            after_tab_change(self, label)
            return res

        QTabWidget.addTab = addTab_patched
        QTabWidget.insertTab = insertTab_patched
        _QTAB_PATCHED = True
        _log("QTabWidget hook installed")

    try:
        for ms in (0, 100, 300, 800, 1500, 3000, 6000, 10000, 15000, 25000):
            QTimer.singleShot(ms, lambda root=root: _scan_for_ai_pipeline_tabs(root))
    except Exception:
        pass
    return True


def install_ai_pipeline_video_input_patch(root: Any = None):
    return _install_qtab_hook(root)


def mustatil_plugin_init():
    install_ai_pipeline_video_input_patch()
    return True


def register_plugin(app=None, main_window=None):
    install_ai_pipeline_video_input_patch(main_window or app)
    try:
        _scan_for_ai_pipeline_tabs(main_window or app)
    except Exception:
        pass
    return True


def init_plugin(app=None, main_window=None):
    return register_plugin(app, main_window)


def load_plugin(app=None, main_window=None):
    return register_plugin(app, main_window)


try:
    install_ai_pipeline_video_input_patch()
except Exception:
    pass
