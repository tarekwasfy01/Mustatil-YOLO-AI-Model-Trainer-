#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""External visual YOLO Pipeline tab for Mustatil Qt Workspace.

Put this file into mustatil_plugins next to mustatil_qt_workspace.py.
The main workspace only imports this module and calls install_yolo_pipeline_tab(workspace).

This version uses a visual block canvas inspired by educational block builders:
- draggable blocks on a canvas
- arrows between blocks
- selected block properties on the right with a wide scrollable editor
- dependency-based execution order
- IF/logic-gate rule blocks: AND, OR, NAND, NOR, XOR and NOT
- rule blocks contain an embedded gate dropdown, dependency/operator/count controls and a THEN action/value selector on the block
"""
from __future__ import annotations

import ast
import json
import math
import pprint
import time
import uuid
import os
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

try:
    import numpy as np
except Exception:
    np = None

from PIL import Image
from PySide6.QtCore import Qt, QRectF, QPointF, QLineF, QTimer
from PySide6.QtGui import QColor, QBrush, QPen, QPainter, QPainterPath, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QMessageBox, QSplitter, QGroupBox,
    QTextEdit, QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox,
    QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsPathItem, QGraphicsProxyWidget,
    QDialog, QListWidget, QListWidgetItem, QDialogButtonBox, QInputDialog,
    QStackedWidget, QPlainTextEdit, QScrollArea, QSizePolicy
)


@dataclass
class PipelineBlock:
    id: str = field(default_factory=lambda: "b_" + uuid.uuid4().hex[:8])
    name: str = "YOLO block"
    type: str = "yolo"          # yolo | formlearner | rule
    enabled: bool = True
    input_ref: str = "original" # original or another block id
    x: float = 80.0
    y: float = 80.0
    model_path: str = ""
    confidence: float = 0.05
    imgsz: int = 640
    device: str = "auto cuda"
    use_chunked_yolo: bool = True
    tile_size: int = 1024
    overlap: int = 384
    shifted_tiles: bool = True
    crop_padding: int = 0
    classes_filter: str = ""    # comma separated class ids/names; blank = all
    formlearner_model: str = ""
    form_threshold: float = 0.50
    rule_json: str = '{\n  "label": "classified_object",\n  "logic": "AND",\n  "within_parent": "",\n  "conditions": [\n    {"source": "", "count_min": 1, "count_max": null}\n  ],\n  "class_id": 0\n}'


class VisualBlockItem(QGraphicsItem):
    WIDTH = 315
    HEIGHT = 178
    YOLO_HEIGHT = 178
    FORM_HEIGHT = 154
    RULE_HEIGHT = 244

    TYPE_COLORS = {
        "yolo": QColor(78, 132, 255),
        "owlv2": QColor(130, 90, 220),
        "grounding_dino": QColor(60, 150, 190),
        "lae_dino": QColor(190, 80, 170),
        "sam2": QColor(50, 170, 170),
        "formlearner": QColor(44, 170, 120),
        "rule": QColor(238, 155, 45),
    }

    def __init__(self, block: PipelineBlock, tab: "YoloPipelineTab"):
        super().__init__()
        self.block = block
        self.tab = tab
        self.setFlags(
            QGraphicsItem.ItemIsMovable |
            QGraphicsItem.ItemIsSelectable |
            QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.rule_proxy: Optional[QGraphicsProxyWidget] = None
        self._proxy_type: str = ""
        self._updating_rule_controls = False
        self._updating_inline_controls = False
        self.setPos(float(block.x), float(block.y))
        self.refresh_embedded_controls()

    def block_height(self) -> float:
        if self.block.type == "rule":
            return float(self.RULE_HEIGHT)
        if self.block.type == "formlearner":
            return float(self.FORM_HEIGHT)
        return float(self.YOLO_HEIGHT)

    def block_width(self) -> float:
        # Grow the visual block when the visible content is long. This keeps
        # inline dropdowns, model paths and long block names from being clipped.
        content = [self.block.name or ""]
        if self.block.type == "yolo" or self.block.type in {"owlv2", "grounding_dino", "lae_dino", "sam2"}:
            content += [self.block.model_path or "", self.block.classes_filter or "", self.block.device or ""]
        elif self.block.type == "formlearner":
            content += [self.block.formlearner_model or self.block.model_path or ""]
        elif self.block.type == "rule":
            try:
                rule = json.loads(self.block.rule_json or "{}")
                content += [str(rule.get("label", "")), str(rule.get("then_value", "")), str(rule.get("logic", ""))]
            except Exception:
                pass
        longest = max([len(str(x)) for x in content] + [0])
        return float(max(type(self).WIDTH, min(460, 245 + longest * 5)))

    def boundingRect(self):
        return QRectF(0, 0, self.block_width(), self.block_height())

    def input_anchor(self) -> QPointF:
        return self.scenePos() + QPointF(0, self.block_height() / 2)

    def output_anchor(self) -> QPointF:
        return self.scenePos() + QPointF(self.block_width(), self.block_height() / 2)

    def paint(self, painter: QPainter, option, widget=None):
        b = self.block
        color = self.TYPE_COLORS.get(b.type, QColor(110, 110, 110))
        if not b.enabled:
            color = QColor(145, 145, 145)
        body = QColor(color)
        header = QColor(max(0, color.red() - 30), max(0, color.green() - 30), max(0, color.blue() - 30))
        outline = QColor(20, 20, 20) if self.isSelected() else QColor(90, 90, 90)

        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(outline, 3 if self.isSelected() else 1.4))
        painter.setBrush(QBrush(body))
        painter.drawRoundedRect(self.boundingRect(), 12, 12)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(header))
        header_path = QPainterPath()
        header_path.addRoundedRect(QRectF(0, 0, self.block_width(), 28), 12, 12)
        header_path.addRect(QRectF(0, 14, self.block_width(), 16))
        painter.drawPath(header_path)

        # small visual sockets, like block connectors
        painter.setPen(QPen(QColor(45, 45, 45), 1))
        painter.setBrush(QBrush(QColor(245, 245, 245)))
        painter.drawEllipse(QPointF(0, self.block_height() / 2), 7, 7)
        painter.drawEllipse(QPointF(self.block_width(), self.block_height() / 2), 7, 7)

        painter.setPen(QPen(QColor("white")))
        painter.setFont(QFont("Arial", 9, QFont.Bold))
        painter.drawText(QRectF(10, 4, self.block_width() - 20, 20), Qt.AlignLeft | Qt.AlignVCenter, b.type.upper())

        painter.setFont(QFont("Arial", 10, QFont.Bold))
        painter.drawText(QRectF(12, 32, self.block_width() - 24, 22), Qt.AlignLeft | Qt.AlignVCenter, b.name[:28])

        painter.setFont(QFont("Arial", 8))
        detail = "Original image" if b.input_ref == "original" else self.tab.block_label(b.input_ref)
        if b.type == "yolo" or b.type in {"owlv2", "grounding_dino", "lae_dino", "sam2"}:
            line = f"conf {b.confidence:.3f} | {b.device}"
        elif b.type == "formlearner":
            line = f"FormScore ≥ {b.form_threshold:.2f}"
        else:
            try:
                rule = json.loads(b.rule_json or "{}")
                ncond = len(rule.get("conditions") or [])
                gate = str(rule.get("logic", "AND")).upper()
                line = f"{gate} | {ncond or 1} input(s)"
            except Exception:
                line = "count / classify rule"
        if b.type == "rule":
            painter.drawText(QRectF(12, 55, self.block_width() - 24, 18), Qt.AlignLeft | Qt.AlignVCenter, "Logic/IF settings are inside this block")
            painter.setPen(QPen(QColor(120, 90, 30), 1))
            painter.drawLine(10, 78, self.block_width() - 10, 78)
        else:
            painter.drawText(QRectF(12, 55, self.block_width() - 24, 18), Qt.AlignLeft | Qt.AlignVCenter, f"in: {detail[:24]}")
            painter.setPen(QPen(QColor(80, 80, 80), 1))
            painter.drawLine(10, 76, self.block_width() - 10, 76)

    def refresh_embedded_controls(self):
        """Create/remove inline controls that sit directly inside the visual block.

        YOLO and FormLearner blocks now expose their most important settings inside
        the block, not only in the right properties panel. The block is deliberately
        taller than before so the controls fit instead of being clipped.
        """
        if self.rule_proxy is not None and self._proxy_type != self.block.type:
            try:
                self.scene().removeItem(self.rule_proxy)
            except Exception:
                pass
            self.rule_proxy = None
            self._proxy_type = ""

        if self.block.type == "rule":
            if self.rule_proxy is None:
                w = QWidget()
                w.setObjectName("RuleBlockInlineEditor")
                w.setStyleSheet(
                    "QWidget#RuleBlockInlineEditor { background: rgba(255,255,255,218); border-radius: 6px; }"
                    "QComboBox, QSpinBox, QLineEdit { font-size: 8pt; min-height: 20px; max-height: 24px; }"
                    "QLabel { font-size: 8pt; font-weight: bold; }"
                )
                lay = QGridLayout(w)
                lay.setContentsMargins(5, 3, 5, 3)
                lay.setHorizontalSpacing(4)
                lay.setVerticalSpacing(3)

                self._rule_gate_combo = QComboBox()
                self._rule_gate_combo.addItems(["AND", "OR", "NAND", "NOR", "XOR", "NOT"])
                lay.addWidget(QLabel("IF"), 0, 0)
                lay.addWidget(self._rule_gate_combo, 0, 1, 1, 3)

                self._rule_source_combos = []
                self._rule_op_combos = []
                self._rule_min_spins = []
                for row in range(2):
                    src = QComboBox()
                    src.setMinimumWidth(145)
                    op = QComboBox()
                    op.addItems([">=", "<=", "=="])
                    op.setMinimumWidth(48)
                    spin = QSpinBox()
                    spin.setRange(0, 9999)
                    spin.setValue(1 if row == 0 else 0)
                    self._rule_source_combos.append(src)
                    self._rule_op_combos.append(op)
                    self._rule_min_spins.append(spin)
                    lay.addWidget(src, row + 1, 0, 1, 2)
                    lay.addWidget(op, row + 1, 2)
                    lay.addWidget(spin, row + 1, 3)

                self._rule_then_action_combo = QComboBox()
                self._rule_then_action_combo.addItems(["classify", "number", "flag true", "label", "score", "tag", "keep parent", "discard/filter"])
                self._rule_label_edit = QLineEdit()
                self._rule_label_edit.setPlaceholderText("value, e.g. airplane / 2 / true")
                lay.addWidget(QLabel("THEN"), 3, 0)
                lay.addWidget(self._rule_then_action_combo, 3, 1, 1, 3)
                lay.addWidget(QLabel("VALUE"), 4, 0)
                lay.addWidget(self._rule_label_edit, 4, 1, 1, 3)

                self._rule_gate_combo.currentTextChanged.connect(lambda *_: self._inline_rule_changed())
                self._rule_then_action_combo.currentTextChanged.connect(lambda *_: self._inline_rule_changed())
                self._rule_label_edit.textChanged.connect(lambda *_: self._inline_rule_changed())
                for src in self._rule_source_combos:
                    src.currentIndexChanged.connect(lambda *_: self._inline_rule_changed())
                for op in self._rule_op_combos:
                    op.currentTextChanged.connect(lambda *_: self._inline_rule_changed())
                for spin in self._rule_min_spins:
                    spin.valueChanged.connect(lambda *_: self._inline_rule_changed())

                self.rule_proxy = QGraphicsProxyWidget(self)
                self.rule_proxy.setWidget(w)
                self.rule_proxy.setPos(9, 82)
                self.rule_proxy.resize(self.block_width() - 18, 154)
                self._proxy_type = "rule"
            self.update_rule_controls_values()
            return

        if self.block.type == "yolo" or self.block.type in {"owlv2", "grounding_dino", "lae_dino", "sam2"}:
            if self.rule_proxy is None:
                w = QWidget()
                w.setObjectName("YoloBlockInlineEditor")
                w.setStyleSheet(
                    "QWidget#YoloBlockInlineEditor { background: rgba(255,255,255,218); border-radius: 6px; }"
                    "QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { font-size: 8pt; min-height: 20px; max-height: 24px; }"
                    "QLabel { font-size: 8pt; font-weight: bold; }"
                )
                lay = QGridLayout(w)
                lay.setContentsMargins(5, 3, 5, 3)
                lay.setHorizontalSpacing(4)
                lay.setVerticalSpacing(3)
                self._yolo_model_combo = QComboBox(); self._yolo_model_combo.setEditable(True); self._yolo_model_combo.setMinimumWidth(185)
                self._yolo_conf_spin = QDoubleSpinBox(); self._yolo_conf_spin.setRange(0.001, 1.0); self._yolo_conf_spin.setDecimals(3); self._yolo_conf_spin.setSingleStep(0.01)
                self._yolo_imgsz_spin = QSpinBox(); self._yolo_imgsz_spin.setRange(64, 8192); self._yolo_imgsz_spin.setSingleStep(32)
                self._yolo_device_combo = QComboBox(); self._yolo_device_combo.setEditable(True); self._yolo_device_combo.addItems(["auto cuda", "cpu", "cuda", "0", "rocm", "directml", "openml", "openvino"])
                self._yolo_class_edit = QLineEdit(); self._yolo_class_edit.setPlaceholderText("classes optional")
                lay.addWidget(QLabel("MODEL"), 0, 0); lay.addWidget(self._yolo_model_combo, 0, 1, 1, 3)
                lay.addWidget(QLabel("CONF"), 1, 0); lay.addWidget(self._yolo_conf_spin, 1, 1)
                lay.addWidget(QLabel("SIZE"), 1, 2); lay.addWidget(self._yolo_imgsz_spin, 1, 3)
                lay.addWidget(QLabel("DEVICE"), 2, 0); lay.addWidget(self._yolo_device_combo, 2, 1)
                lay.addWidget(QLabel("CLASS"), 2, 2); lay.addWidget(self._yolo_class_edit, 2, 3)
                self._yolo_model_combo.currentTextChanged.connect(lambda *_: self._inline_yolo_changed())
                self._yolo_conf_spin.valueChanged.connect(lambda *_: self._inline_yolo_changed())
                self._yolo_imgsz_spin.valueChanged.connect(lambda *_: self._inline_yolo_changed())
                self._yolo_device_combo.currentTextChanged.connect(lambda *_: self._inline_yolo_changed())
                self._yolo_class_edit.textChanged.connect(lambda *_: self._inline_yolo_changed())
                self.rule_proxy = QGraphicsProxyWidget(self)
                self.rule_proxy.setWidget(w)
                self.rule_proxy.setPos(9, 82)
                self.rule_proxy.resize(self.block_width() - 18, 88)
                self._proxy_type = "yolo"
            self.update_yolo_controls_values()
            return

        if self.block.type == "formlearner":
            if self.rule_proxy is None:
                w = QWidget()
                w.setObjectName("FormBlockInlineEditor")
                w.setStyleSheet(
                    "QWidget#FormBlockInlineEditor { background: rgba(255,255,255,218); border-radius: 6px; }"
                    "QComboBox, QDoubleSpinBox, QLineEdit { font-size: 8pt; min-height: 20px; max-height: 24px; }"
                    "QLabel { font-size: 8pt; font-weight: bold; }"
                )
                lay = QGridLayout(w)
                lay.setContentsMargins(5, 3, 5, 3)
                lay.setHorizontalSpacing(4)
                lay.setVerticalSpacing(3)
                self._form_model_combo = QComboBox(); self._form_model_combo.setEditable(True); self._form_model_combo.setMinimumWidth(185)
                self._form_threshold_spin = QDoubleSpinBox(); self._form_threshold_spin.setRange(0.0, 1.0); self._form_threshold_spin.setDecimals(3); self._form_threshold_spin.setSingleStep(0.01)
                lay.addWidget(QLabel("MODEL"), 0, 0); lay.addWidget(self._form_model_combo, 0, 1, 1, 3)
                lay.addWidget(QLabel("FORM ≥"), 1, 0); lay.addWidget(self._form_threshold_spin, 1, 1, 1, 3)
                self._form_model_combo.currentTextChanged.connect(lambda *_: self._inline_form_changed())
                self._form_threshold_spin.valueChanged.connect(lambda *_: self._inline_form_changed())
                self.rule_proxy = QGraphicsProxyWidget(self)
                self.rule_proxy.setWidget(w)
                self.rule_proxy.setPos(9, 82)
                self.rule_proxy.resize(self.block_width() - 18, 64)
                self._proxy_type = "formlearner"
            self.update_form_controls_values()
            return

        if self.rule_proxy is not None:
            try:
                self.scene().removeItem(self.rule_proxy)
            except Exception:
                pass
            self.rule_proxy = None
            self._proxy_type = ""

    def update_rule_controls_values(self):
        if self.rule_proxy is None or self.block.type != "rule":
            return
        self._updating_rule_controls = True
        try:
            rule = self.tab._current_rule_dict(self.block)
            gate = self.tab._normalize_logic_gate(rule.get("logic", "AND"))
            self._rule_gate_combo.setCurrentText(gate)
            action = self.tab._normalize_then_action(rule.get("then_action", rule.get("action", "classify")))
            if hasattr(self, "_rule_then_action_combo"):
                self._rule_then_action_combo.setCurrentText(action)
            self._rule_label_edit.setText(str(rule.get("then_value", rule.get("label", "")) or ""))
            conditions = list(rule.get("conditions") or [])
            for idx, combo in enumerate(self._rule_source_combos):
                combo.clear()
                combo.addItem("choose input", "")
                for other in self.tab.blocks:
                    if other.id == self.block.id:
                        continue
                    combo.addItem(other.name, other.id)
                src_ref = ""
                min_val = 0
                op_val = ">="
                if idx < len(conditions):
                    src_ref = self.tab._resolve_block_ref(conditions[idx].get("source")) or ""
                    op_val = self.tab._normalize_rule_operator(conditions[idx].get("operator", conditions[idx].get("op", ">=")))
                    try:
                        min_val = int(conditions[idx].get("count_min", conditions[idx].get("value", 1)))
                    except Exception:
                        min_val = 1
                ix = combo.findData(src_ref)
                combo.setCurrentIndex(ix if ix >= 0 else 0)
                if idx < len(getattr(self, "_rule_op_combos", [])):
                    self._rule_op_combos[idx].setCurrentText(op_val)
                self._rule_min_spins[idx].setValue(max(0, min(9999, min_val)))
        finally:
            self._updating_rule_controls = False

    def update_yolo_controls_values(self):
        if self.rule_proxy is None or self.block.type != "yolo":
            return
        self._updating_inline_controls = True
        try:
            combo = self._yolo_model_combo
            current = str(self.block.model_path or "")
            old = combo.blockSignals(True)
            combo.clear()
            for label, value in self.tab._model_choices("yolo"):
                combo.addItem(label, value)
            ix = combo.findData(current)
            if ix >= 0:
                combo.setCurrentIndex(ix)
            else:
                combo.setEditText(current)
            combo.blockSignals(old)
            self._yolo_conf_spin.setValue(float(self.block.confidence or 0.05))
            self._yolo_imgsz_spin.setValue(int(self.block.imgsz or 640))
            self._yolo_device_combo.setCurrentText(str(self.block.device or "cpu"))
            self._yolo_class_edit.setText(str(self.block.classes_filter or ""))
        finally:
            self._updating_inline_controls = False

    def update_form_controls_values(self):
        if self.rule_proxy is None or self.block.type != "formlearner":
            return
        self._updating_inline_controls = True
        try:
            combo = self._form_model_combo
            current = str(self.block.formlearner_model or self.block.model_path or "")
            old = combo.blockSignals(True)
            combo.clear()
            for label, value in self.tab._model_choices("formlearner"):
                combo.addItem(label, value)
            ix = combo.findData(current)
            if ix >= 0:
                combo.setCurrentIndex(ix)
            else:
                combo.setEditText(current)
            combo.blockSignals(old)
            self._form_threshold_spin.setValue(float(self.block.form_threshold or 0.5))
        finally:
            self._updating_inline_controls = False

    def _inline_yolo_changed(self):
        if self._updating_inline_controls:
            return
        try:
            data = self._yolo_model_combo.currentData()
            text = str(data if data not in (None, "") else self._yolo_model_combo.currentText()).strip()
            self.block.model_path = text
            self.block.confidence = float(self._yolo_conf_spin.value())
            self.block.imgsz = int(self._yolo_imgsz_spin.value())
            self.block.device = str(self._yolo_device_combo.currentText() or "cpu")
            self.block.classes_filter = str(self._yolo_class_edit.text() or "")
            self.tab._update_editor_from_inline(self.block)
            self.tab.refresh_code_from_blocks()
            self.update()
        except Exception:
            pass

    def _inline_form_changed(self):
        if self._updating_inline_controls:
            return
        try:
            data = self._form_model_combo.currentData()
            text = str(data if data not in (None, "") else self._form_model_combo.currentText()).strip()
            self.block.formlearner_model = text
            self.block.model_path = text
            self.block.form_threshold = float(self._form_threshold_spin.value())
            self.tab._update_editor_from_inline(self.block)
            self.tab.refresh_code_from_blocks()
            self.update()
        except Exception:
            pass

    def _inline_rule_changed(self):
        if self._updating_rule_controls:
            return
        self.tab._rule_block_inline_controls_changed(self.block, self)

    def mouseDoubleClickEvent(self, event):
        self.tab.select_block(self.block.id)
        super().mouseDoubleClickEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            p = self.pos()
            self.block.x = float(p.x())
            self.block.y = float(p.y())
            self.tab.refresh_arrows()
        if change == QGraphicsItem.ItemSelectedHasChanged and bool(value):
            self.tab.select_block(self.block.id, from_scene=True)
        return super().itemChange(change, value)


class ArrowItem(QGraphicsPathItem):
    def __init__(self, start_item: VisualBlockItem, end_item: VisualBlockItem, tab: "YoloPipelineTab"):
        super().__init__()
        self.start_item = start_item
        self.end_item = end_item
        self.tab = tab
        self.source_id = start_item.block.id
        self.target_id = end_item.block.id
        self.setZValue(-10)
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
        self._normal_pen = QPen(QColor(55, 55, 55), 2.2)
        self._hover_pen = QPen(QColor(210, 60, 45), 3.0)
        self.setPen(self._normal_pen)
        self.update_path()

    def shape(self):
        # Wider clickable/hover shape so right-clicking a curved line is easy.
        stroker_path = self.path()
        from PySide6.QtGui import QPainterPathStroker
        stroker = QPainterPathStroker()
        stroker.setWidth(14)
        return stroker.createStroke(stroker_path)

    def hoverEnterEvent(self, event):
        self.setPen(self._hover_pen)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setPen(self._normal_pen)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.tab.remove_connection(self.source_id, self.target_id)
            event.accept()
            return
        super().mousePressEvent(event)

    def update_path(self):
        s = self.start_item.output_anchor()
        e = self.end_item.input_anchor()
        dx = max(70.0, abs(e.x() - s.x()) * 0.45)
        path = QPainterPath(s)
        path.cubicTo(QPointF(s.x() + dx, s.y()), QPointF(e.x() - dx, e.y()), e)
        self.setPath(path)

        # arrow head
        line = QLineF(path.pointAtPercent(0.985), e)
        angle = math.atan2(-line.dy(), line.dx())
        size = 10
        p1 = e - QPointF(math.sin(angle + math.pi / 3) * size, math.cos(angle + math.pi / 3) * size)
        p2 = e - QPointF(math.sin(angle + math.pi - math.pi / 3) * size, math.cos(angle + math.pi - math.pi / 3) * size)
        head = QPainterPath(e)
        head.lineTo(p1)
        head.lineTo(p2)
        head.closeSubpath()
        path.addPath(head)
        self.setPath(path)


class PipelineGraphicsView(QGraphicsView):
    """Canvas view with zoom, block dragging and socket-to-socket line connections."""

    def __init__(self, scene: QGraphicsScene, tab: "YoloPipelineTab", parent=None):
        super().__init__(scene, parent or tab)
        self.tab = tab
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor(248, 248, 248)))
        self._connect_source: Optional[VisualBlockItem] = None
        self._temp_connection: Optional[QGraphicsPathItem] = None

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
        event.accept()

    def _scene_pos(self, event) -> QPointF:
        try:
            return self.mapToScene(event.position().toPoint())
        except Exception:
            return self.mapToScene(event.pos())

    def _socket_item_at(self, scene_pos: QPointF, socket: str) -> Optional[VisualBlockItem]:
        best = None
        best_dist = 999999.0
        # Scale-aware hit radius: still usable when zoomed out.
        hit_radius = max(14.0, 14.0 / max(0.25, abs(float(self.transform().m11())) or 1.0))
        for item in self.tab.block_items.values():
            anchor = item.output_anchor() if socket == "output" else item.input_anchor()
            dist = QLineF(scene_pos, anchor).length()
            if dist < hit_radius and dist < best_dist:
                best = item
                best_dist = dist
        return best

    def _connection_path(self, start: QPointF, end: QPointF) -> QPainterPath:
        dx = max(70.0, abs(end.x() - start.x()) * 0.45)
        path = QPainterPath(start)
        path.cubicTo(QPointF(start.x() + dx, start.y()), QPointF(end.x() - dx, end.y()), end)
        return path

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            scene_pos = self._scene_pos(event)
            source = self._socket_item_at(scene_pos, "output")
            if source is not None:
                self._connect_source = source
                self._temp_connection = QGraphicsPathItem()
                self._temp_connection.setZValue(-5)
                pen = QPen(QColor(25, 115, 255), 2.8)
                pen.setStyle(Qt.DashLine)
                self._temp_connection.setPen(pen)
                self._temp_connection.setPath(self._connection_path(source.output_anchor(), scene_pos))
                self.scene().addItem(self._temp_connection)
                self.setDragMode(QGraphicsView.NoDrag)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._connect_source is not None and self._temp_connection is not None:
            scene_pos = self._scene_pos(event)
            self._temp_connection.setPath(self._connection_path(self._connect_source.output_anchor(), scene_pos))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._connect_source is not None:
            scene_pos = self._scene_pos(event)
            target = self._socket_item_at(scene_pos, "input")
            if self._temp_connection is not None:
                try:
                    self.scene().removeItem(self._temp_connection)
                except Exception:
                    pass
            source = self._connect_source
            self._connect_source = None
            self._temp_connection = None
            self.setDragMode(QGraphicsView.RubberBandDrag)
            if target is not None and target.block.id != source.block.id:
                if target.block.type == "rule":
                    self.tab.add_rule_condition_from_source(target.block, source.block.id, interactive=False)
                    self.tab.select_block(target.block.id)
                    self.tab._log(f"Added rule input condition: {source.block.name} → {target.block.name}")
                else:
                    target.block.input_ref = source.block.id
                    self.tab.select_block(target.block.id)
                    self.tab._log(f"Connected: {source.block.name} → {target.block.name}")
                self.tab.refresh_arrows()
                self.tab._refresh_input_refs()
                self.tab.refresh_code_from_blocks()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class YoloPipelineTab(QWidget):
    def __init__(self, workspace):
        super().__init__(workspace)
        self.ws = workspace
        self.blocks: List[PipelineBlock] = []
        self.block_items: Dict[str, VisualBlockItem] = {}
        self.arrows: List[ArrowItem] = []
        self.results: List[Dict[str, Any]] = []
        self.results_by_block: Dict[str, List[Dict[str, Any]]] = {}
        self.selected_block_id: Optional[str] = None
        self._updating = False
        self._build_ui()
        self._load_default_airplane_example()
        self.select_block(self.blocks[0].id if self.blocks else None)
        self.refresh_code_from_blocks()
        self._write_intro()
        # Sort the initial/example blocks after the Qt widgets have finished
        # calculating their real dynamic sizes. A zero-delay timer is important
        # here because embedded combo boxes/spin boxes can change the final block
        # dimensions after construction. Auto layout then uses those measured
        # dimensions and its collision pass, so the first view opens cleanly.
        try:
            QTimer.singleShot(0, self.auto_layout)
        except Exception:
            try:
                self.auto_layout()
            except Exception:
                pass

    def _load_default_airplane_example(self):
        """Load the default airplane/engine/wing example pipeline.

        This matches the Python code-view example requested by the user:
        YOLO 1 detects airplanes, YOLO 2 detects engines inside YOLO 1,
        YOLO 3 detects wings inside YOLO 1, and the IF/Rule block classifies
        an airplane when engine and wing counts match the rule.
        """
        default_rule = '{\n  "label": "classified_object",\n  "logic": "AND",\n  "within_parent": "",\n  "conditions": [\n    {"source": "", "count_min": 1, "count_max": null}\n  ],\n  "class_id": 0\n}'
        airplane_rule = (
            '{\n'
            '  "label": "airplane",\n'
            '  "logic": "AND",\n'
            '  "within_parent": "YOLO 2: engines",\n'
            '  "conditions": [\n'
            '    {\n'
            '      "source": "YOLO 2: engines",\n'
            '      "operator": ">=",\n'
            '      "count_min": 1,\n'
            '      "count_max": null\n'
            '    },\n'
            '    {\n'
            '      "source": "YOLO 3: wings",\n'
            '      "operator": ">=",\n'
            '      "count_min": 2,\n'
            '      "count_max": null\n'
            '    }\n'
            '  ],\n'
            '  "class_id": 0,\n'
            '  "then_action": "classify",\n'
            '  "then_value": "airplane"\n'
            '}'
        )
        self.blocks = [
            PipelineBlock(
                id="b_591f1006", name="YOLO 1: airplanes", type="yolo", enabled=True,
                input_ref="original", x=90.0, y=80.0, model_path="", confidence=0.05,
                imgsz=640, device="auto cuda", crop_padding=0, classes_filter="",
                formlearner_model="", form_threshold=0.5, rule_json=default_rule,
            ),
            PipelineBlock(
                id="b_4aef203d", name="YOLO 2: engines", type="yolo", enabled=True,
                input_ref="b_591f1006", x=540.0, y=80.0, model_path="", confidence=0.05,
                imgsz=640, device="auto cuda", crop_padding=0, classes_filter="",
                formlearner_model="", form_threshold=0.5, rule_json=default_rule,
            ),
            PipelineBlock(
                id="b_b050f65b", name="YOLO 3: wings", type="yolo", enabled=True,
                input_ref="b_591f1006", x=542.5, y=416.0, model_path="", confidence=0.05,
                imgsz=640, device="auto cuda", crop_padding=0, classes_filter="",
                formlearner_model="", form_threshold=0.5, rule_json=default_rule,
            ),
            PipelineBlock(
                id="b_577d05c6", name="Rule: airplane", type="rule", enabled=True,
                input_ref="b_4aef203d", x=980.0, y=80.0, model_path="", confidence=0.05,
                imgsz=640, device="auto cuda", crop_padding=0, classes_filter="",
                formlearner_model="", form_threshold=0.5, rule_json=airplane_rule,
            ),
        ]
        self._rebuild_scene()

    # ---------- UI ----------
    def _build_ui(self):
        root = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        root.addLayout(toolbar)

        for text, typ in [
            ("+ YOLO block", "yolo"),
            ("+ OWLv2 block", "owlv2"),
            ("+ Grounding DINO block", "grounding_dino"),
            ("+ LAE-DINO block", "lae_dino"),
            ("+ SAM2 block", "sam2"),
            ("+ FormLearner block", "formlearner"),
            ("+ IF / Logic block", "rule"),
        ]:
            btn = QPushButton(text)
            btn.clicked.connect(lambda _=False, t=typ: self._add_block(t))
            toolbar.addWidget(btn)

        btn = QPushButton("Duplicate selected"); btn.clicked.connect(self._duplicate_current); toolbar.addWidget(btn)
        btn = QPushButton("Delete selected"); btn.clicked.connect(self._delete_current); toolbar.addWidget(btn)
        btn = QPushButton("Auto layout"); btn.clicked.connect(self.auto_layout); toolbar.addWidget(btn)
        self.view_mode_btn = QPushButton("View: Code")
        self.view_mode_btn.clicked.connect(self.toggle_block_code_view)
        toolbar.addWidget(self.view_mode_btn)
        toolbar.addStretch(1)
        btn = QPushButton("Load pipeline"); btn.clicked.connect(self.load_pipeline); toolbar.addWidget(btn)
        btn = QPushButton("Save pipeline"); btn.clicked.connect(self.save_pipeline); toolbar.addWidget(btn)

        image_bar = QGroupBox("Pipeline image / Original image")
        image_lay = QHBoxLayout(image_bar)
        image_lay.setContentsMargins(8, 6, 8, 6)
        self.pipeline_image_path = QLineEdit()
        try:
            self.pipeline_image_path.setText(str(self.ws.image.get() or ""))
        except Exception:
            pass
        self.pipeline_image_path.setPlaceholderText("Choose the image/GeoTIFF that should be used as Original image for the pipeline")
        self.pipeline_image_path.textChanged.connect(self._sync_pipeline_image_to_workspace)
        image_lay.addWidget(QLabel("Image"))
        image_lay.addWidget(self.pipeline_image_path, 1)
        btn = QPushButton("Choose image…")
        btn.clicked.connect(self.choose_pipeline_image)
        image_lay.addWidget(btn)
        root.addWidget(image_bar)

        self.mode_stack = QStackedWidget()
        root.addWidget(self.mode_stack, 1)

        self.run_pipeline_big_btn = QPushButton("▶ RUN PIPELINE")
        self.run_pipeline_big_btn.setMinimumHeight(68)
        self.run_pipeline_big_btn.setStyleSheet("QPushButton { font-size: 22px; font-weight: bold; padding: 16px; border-radius: 10px; }")
        self.run_pipeline_big_btn.clicked.connect(self.run_pipeline_clicked)
        root.addWidget(self.run_pipeline_big_btn)

        split = QSplitter(Qt.Horizontal)

        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(-200, -150, 1800, 1000)
        self.view = PipelineGraphicsView(self.scene, self)
        split.addWidget(self.view)

        right = QWidget()
        right.setMinimumWidth(540)
        right.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(8, 8, 8, 8)
        right_lay.setSpacing(8)

        # The properties editor can become tall/wide because YOLO/FormLearner/IF blocks
        # now expose many controls. Keep it in its own scroll area so fields never get
        # squeezed into each other on smaller screens.
        edit_box = QGroupBox("Selected block properties")
        edit_box.setMinimumWidth(510)
        edit_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        form = QFormLayout(edit_box)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.DontWrapRows)
        form.setFormAlignment(Qt.AlignTop)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(8)
        self.enabled = QCheckBox("enabled"); self.enabled.toggled.connect(self._save_editor_to_block)
        self.name = QLineEdit(); self.name.textChanged.connect(self._save_editor_to_block)
        self.typ = QComboBox(); self.typ.addItems(["yolo", "owlv2", "grounding_dino", "lae_dino", "sam2", "formlearner", "rule"]); self.typ.currentTextChanged.connect(self._save_editor_to_block)
        self.input_ref = QComboBox(); self.input_ref.currentTextChanged.connect(self._save_editor_to_block)
        self.model_path = self._path_row("YOLO model (*.pt *.onnx *.engine *.xml);;All files (*)")
        self.conf = QDoubleSpinBox(); self.conf.setRange(0.001, 1.0); self.conf.setSingleStep(0.01); self.conf.setDecimals(3); self.conf.valueChanged.connect(self._save_editor_to_block)
        self.imgsz = QSpinBox(); self.imgsz.setRange(64, 8192); self.imgsz.setSingleStep(32); self.imgsz.valueChanged.connect(self._save_editor_to_block)
        self.device = QComboBox(); self.device.setEditable(True); self.device.addItems(["auto cuda", "cpu", "cuda", "0", "rocm", "directml", "openml", "openvino"]); self.device.currentTextChanged.connect(self._save_editor_to_block)
        self.use_chunked_yolo = QCheckBox("chunk large images / parent crops"); self.use_chunked_yolo.toggled.connect(self._save_editor_to_block)
        self.tile_size = QSpinBox(); self.tile_size.setRange(128, 8192); self.tile_size.setSingleStep(128); self.tile_size.valueChanged.connect(self._save_editor_to_block)
        self.tile_overlap = QSpinBox(); self.tile_overlap.setRange(0, 4096); self.tile_overlap.setSingleStep(32); self.tile_overlap.valueChanged.connect(self._save_editor_to_block)
        self.shifted_tiles = QCheckBox("shifted second pass"); self.shifted_tiles.toggled.connect(self._save_editor_to_block)
        self.crop_padding = QSpinBox(); self.crop_padding.setRange(0, 4096); self.crop_padding.setSingleStep(8); self.crop_padding.valueChanged.connect(self._save_editor_to_block)
        self.classes_filter = QLineEdit(); self.classes_filter.setPlaceholderText("optional: airplane,0,engine"); self.classes_filter.textChanged.connect(self._save_editor_to_block)
        self.form_model = self._path_row("FormLearner JSON (*.json);;All files (*)")
        self.form_threshold = QDoubleSpinBox(); self.form_threshold.setRange(0.0, 1.0); self.form_threshold.setSingleStep(0.01); self.form_threshold.setDecimals(3); self.form_threshold.valueChanged.connect(self._save_editor_to_block)
        self.rule_json = QTextEdit(); self.rule_json.setMinimumHeight(185); self.rule_json.setMinimumWidth(330); self.rule_json.textChanged.connect(self._save_editor_to_block)

        for _w in [self.name, self.input_ref, self.classes_filter, self.pipeline_image_path]:
            try:
                _w.setMinimumWidth(310)
            except Exception:
                pass
        for _w in [self.typ, self.device]:
            try:
                _w.setMinimumWidth(220)
            except Exception:
                pass
        for _w in [self.model_path, self.form_model]:
            try:
                _w.setMinimumWidth(330)
                if hasattr(_w, "_edit"):
                    _w._edit.setMinimumWidth(285)
            except Exception:
                pass

        form.addRow("", self.enabled)
        form.addRow("Name", self.name)
        form.addRow("Block type", self.typ)
        form.addRow("Input socket", self.input_ref)
        form.addRow("YOLO model", self.model_path)
        form.addRow("YOLO confidence", self.conf)
        form.addRow("YOLO imgsz", self.imgsz)
        form.addRow("YOLO device", self.device)
        form.addRow("Chunked YOLO", self.use_chunked_yolo)
        form.addRow("Tile size", self.tile_size)
        form.addRow("Tile overlap", self.tile_overlap)
        form.addRow("Shifted tiles", self.shifted_tiles)
        form.addRow("Parent crop padding", self.crop_padding)
        form.addRow("Class filter", self.classes_filter)
        form.addRow("FormLearner model", self.form_model)
        form.addRow("Form threshold", self.form_threshold)
        form.addRow("Rule JSON", self.rule_json)

        prop_scroll = QScrollArea()
        prop_scroll.setWidgetResizable(True)
        prop_scroll.setMinimumHeight(430)
        prop_scroll.setMinimumWidth(530)
        prop_scroll.setWidget(edit_box)
        right_lay.addWidget(prop_scroll, 2)

        connect_box = QGroupBox("Block actions")
        cgrid = QGridLayout(connect_box)
        btn = QPushButton("Use selected input")
        btn.clicked.connect(self._save_editor_to_block)
        cgrid.addWidget(btn, 0, 0)
        btn = QPushButton("Fit canvas")
        btn.clicked.connect(lambda: self.view.fitInView(self.scene.itemsBoundingRect().adjusted(-80, -80, 80, 80), Qt.KeepAspectRatio))
        cgrid.addWidget(btn, 0, 1)
        btn = QPushButton("Add input to rule")
        btn.clicked.connect(self.add_selected_input_as_rule_condition)
        cgrid.addWidget(btn, 1, 0)
        btn = QPushButton("Edit rule inputs")
        btn.clicked.connect(self.edit_rule_conditions_dialog)
        cgrid.addWidget(btn, 1, 1)
        btn = QPushButton("Run Pipeline")
        btn.clicked.connect(self.run_pipeline_clicked)
        cgrid.addWidget(btn, 2, 0, 1, 2)
        btn = QPushButton("Send to Detection tab")
        btn.clicked.connect(self.send_results_to_detection_tab)
        cgrid.addWidget(btn, 3, 0)
        btn = QPushButton("Export")
        btn.clicked.connect(self.export_results_gpkg)
        cgrid.addWidget(btn, 3, 1)
        connect_box.setMinimumWidth(510)
        right_lay.addWidget(connect_box, 0)

        help_label = QLabel(
            "How to connect: drag a line from the right socket of one block to the left socket of another block, "
            "or select a block and choose its Input socket. "
            "YOLO blocks after another YOLO run inside the parent detections. "
            "IF / Logic blocks work like Lego Mindstorms/Yenka blocks: choose AND/OR/NAND/NOR/XOR/NOT directly inside the block, then choose dependency blocks, comparison operators and counts in the block, e.g. IF YOLO 2 >= 1 AND YOLO 3 == 2 THEN true; the next rule can use that true-rule output to classify as airplane."
        )
        help_label.setWordWrap(True)
        right_lay.addWidget(help_label)

        self.summary = QTextEdit(); self.summary.setReadOnly(True); self.summary.setMinimumHeight(190); self.summary.setMinimumWidth(510)
        right_lay.addWidget(QLabel("Pipeline log / result summary"))
        right_lay.addWidget(self.summary, 1)
        split.addWidget(right)
        try:
            split.setStretchFactor(0, 1)
            split.setStretchFactor(1, 0)
            split.setCollapsible(1, False)
        except Exception:
            pass
        split.setSizes([900, 560])
        self.mode_stack.addWidget(split)

        code_page = QWidget()
        code_lay = QVBoxLayout(code_page)
        code_help = QLabel(
            "Code view: edit the pipeline as readable Python. You can change block names, types, input_ref, models, thresholds, rules and x/y positions. "
            "Click Apply code to update the visual blocks. JSON is still accepted when loading older code."
        )
        code_help.setWordWrap(True)
        code_lay.addWidget(code_help)
        code_btns = QHBoxLayout()
        btn = QPushButton("Refresh code from blocks")
        btn.clicked.connect(self.refresh_code_from_blocks)
        code_btns.addWidget(btn)
        btn = QPushButton("Apply code to blocks")
        btn.clicked.connect(lambda: self.apply_code_to_blocks(silent=False))
        code_btns.addWidget(btn)
        code_btns.addStretch(1)
        code_lay.addLayout(code_btns)
        self.pipeline_code = QPlainTextEdit()
        self.pipeline_code.setMinimumHeight(520)
        self.pipeline_code.setPlaceholderText("pipeline = {\n    'version': 3,\n    'blocks': []\n}")
        code_lay.addWidget(self.pipeline_code, 1)
        self.mode_stack.addWidget(code_page)

    def choose_pipeline_image(self):
        start = ""
        try:
            current = str(self.pipeline_image_path.text() or self.ws.image.get() or "").strip().strip('"')
            if current:
                start = str(Path(current).expanduser().parent if Path(current).expanduser().suffix else Path(current).expanduser())
        except Exception:
            start = ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose pipeline image",
            start,
            "Images / GeoTIFF (*.tif *.tiff *.jpg *.jpeg *.png *.bmp *.webp);;All files (*)"
        )
        if path:
            self.pipeline_image_path.setText(path)
            self._sync_pipeline_image_to_workspace(path)

    def _sync_pipeline_image_to_workspace(self, path: str):
        try:
            if hasattr(self.ws, "image") and hasattr(self.ws.image, "set"):
                self.ws.image.set(str(path or ""))
        except Exception:
            pass

    def _model_choices(self, kind: str) -> List[Tuple[str, str]]:
        """Return model dropdown choices for inline block editors.

        Choices are collected from the current workspace variables, the project
        weights folder and common local model files. The combo remains editable,
        so custom paths can still be typed/pasted. For OWLv2/Grounding DINO/
        LAE-DINO/SAM2 blocks, the same existing YOLO/FormLearner editor widgets
        are reused so the original layout stays intact.
        """
        choices: List[Tuple[str, str]] = [("choose model / custom path", "")]
        seen: Set[str] = set()

        def add(raw: Any, label: Optional[str] = None):
            val = str(raw or "").strip().strip('"')
            if not val or val in seen:
                return
            seen.add(val)
            shown = label or Path(val).name or val
            choices.append((shown, val))

        try:
            current_type = ""
            try:
                cur = self._current_block()
                current_type = str(getattr(cur, "type", "") or "").lower() if cur else ""
            except Exception:
                current_type = ""

            if kind == "yolo":
                # Open-vocabulary defaults. These are editable text values, not files.
                if current_type == "owlv2":
                    add("google/owlv2-base-patch16-ensemble", "google/owlv2-base-patch16-ensemble")
                    add("google/owlv2-large-patch14-ensemble", "google/owlv2-large-patch14-ensemble")
                elif current_type == "grounding_dino":
                    add("IDEA-Research/grounding-dino-base", "IDEA-Research/grounding-dino-base")
                    add("IDEA-Research/grounding-dino-tiny", "IDEA-Research/grounding-dino-tiny")
                elif current_type == "lae_dino":
                    for attr in ("mustatil_lae_existing_v9_config", "mustatil_lae_existing_v8_config", "mustatil_lae_config"):
                        v = str(getattr(self.ws, attr, "") or "").strip()
                        if v:
                            add(v, "LAE config: " + Path(v).name)
                elif current_type == "sam2":
                    # SAM2 checkpoint goes into the existing model_path field.
                    pass

                # Keep the original YOLO choices exactly available.
                for v in getattr(self.ws, "models", []) or []:
                    try:
                        add(v.get())
                    except Exception:
                        pass
                for attr in ("trainmodel", "auto_annotate_yolo_model"):
                    obj = getattr(self.ws, attr, None)
                    if hasattr(obj, "get"):
                        add(obj.get())
                for preset_name, vals in (getattr(self.ws, "YOLO_MODEL_PRESETS", {}) or {}).items():
                    for val in vals or []:
                        add(val, f"{preset_name}: {Path(str(val)).name}")
            else:
                # FormLearner field is reused as checkpoint/config companion field for LAE/SAM2.
                if current_type == "lae_dino":
                    for attr in ("mustatil_lae_existing_v9_weights", "mustatil_lae_existing_v8_weights", "mustatil_lae_weights", "mustatil_lae_checkpoint"):
                        v = str(getattr(self.ws, attr, "") or "").strip()
                        if v:
                            add(v, "LAE checkpoint: " + Path(v).name)
                for attr in ("fl_model_path", "form_model_path", "formlearner_model"):
                    obj = getattr(self.ws, attr, None)
                    if hasattr(obj, "get"):
                        add(obj.get())

            search_dirs = []
            try:
                app_dir = Path(getattr(self.ws, "MUSTATIL_APP_DIR", Path.cwd()))
                search_dirs.append(app_dir)
                search_dirs.append(app_dir / "weights")
                search_dirs.append(app_dir / "models")
                search_dirs.append(Path(__file__).resolve().parent / "weights")
                search_dirs.append(Path(__file__).resolve().parent / "models")
            except Exception:
                pass
            try:
                project = str(getattr(self.ws, "project", None).get() or "").strip()
                if project:
                    search_dirs.extend([Path(project), Path(project) / "weights", Path(project) / "models", Path(project) / "lae_dino_dataset" / "work_dirs" / "lae_dino_mustatil"])
            except Exception:
                pass
            if kind == "yolo":
                if current_type == "lae_dino":
                    patterns = ("*.py", "*.yaml", "*.yml", "*.pth", "*.pt")
                elif current_type == "sam2":
                    patterns = ("*.pt", "*.pth")
                else:
                    patterns = ("*.pt", "*.onnx", "*.engine", "*.xml")
            else:
                if current_type in {"lae_dino", "sam2"}:
                    patterns = ("*.pth", "*.pt", "*.yaml", "*.yml", "*.py", "*.json")
                else:
                    patterns = ("*.json",)
            for folder in search_dirs:
                try:
                    if not folder.exists():
                        continue
                    for pat in patterns:
                        for file in sorted(folder.glob(pat))[:120]:
                            add(str(file))
                except Exception:
                    continue
        except Exception:
            pass
        return choices

    def _update_editor_from_inline(self, block: PipelineBlock):
        if self.selected_block_id != block.id or self._updating:
            return
        self._updating = True
        try:
            self.model_path._edit.setText(block.model_path)
            self.conf.setValue(float(block.confidence or 0.05))
            self.imgsz.setValue(int(block.imgsz or 640))
            self.device.setCurrentText(str(block.device or "cpu"))
            self.classes_filter.setText(str(block.classes_filter or ""))
            self.form_model._edit.setText(block.formlearner_model or block.model_path)
            self.form_threshold.setValue(float(block.form_threshold or 0.5))
        finally:
            self._updating = False

    def _current_pipeline_image_path(self) -> Path:
        raw = ""
        try:
            raw = str(self.pipeline_image_path.text() or "").strip().strip('"')
        except Exception:
            raw = ""
        if not raw:
            try:
                raw = str(self.ws.image.get() or "").strip().strip('"')
            except Exception:
                raw = ""
        if raw:
            self._sync_pipeline_image_to_workspace(raw)
        return Path(raw).expanduser() if raw else Path("")

    def _path_row(self, filt):
        w = QWidget(); lay = QHBoxLayout(w); lay.setContentsMargins(0,0,0,0); lay.setSpacing(4)
        w.setMinimumWidth(330)
        edit = QLineEdit(); edit.setMinimumWidth(285); lay.addWidget(edit, 1)
        btn = QPushButton("…"); btn.setMaximumWidth(34); lay.addWidget(btn)
        btn.clicked.connect(lambda: self._browse_into(edit, filt))
        edit.textChanged.connect(self._save_editor_to_block)
        w._edit = edit
        return w

    def _browse_into(self, edit: QLineEdit, filt: str):
        path, _ = QFileDialog.getOpenFileName(self, "Choose file", "", filt)
        if path:
            edit.setText(path)


    # ---------- Block/code view ----------
    def pipeline_to_dict(self) -> Dict[str, Any]:
        self._sync_positions()
        return {"version": 3, "blocks": [asdict(b) for b in self.blocks]}

    def _pipeline_to_python_code(self) -> str:
        """Return a readable Python representation for the code view.

        The code view deliberately uses a plain Python literal assignment instead
        of a custom DSL, so users can understand and edit it without learning a
        special pipeline syntax. It is parsed with ast.literal_eval, not exec().
        """
        data = self.pipeline_to_dict()
        body = pprint.pformat(data, width=120, sort_dicts=False)
        return (
            "# Mustatil YOLO Pipeline - Python code view\n"
            "# Edit values below, then click 'Apply code to blocks'.\n"
            "# This is parsed safely as Python data, not executed.\n\n"
            f"pipeline = {body}\n"
        )

    def _parse_pipeline_code(self, text: str) -> Dict[str, Any]:
        """Parse Python code-view text, with JSON fallback for old saved code."""
        text = str(text or "").strip()
        if not text:
            return {"version": 3, "blocks": []}
        # Backward compatibility: older plugin versions used raw JSON in code view.
        if text.startswith("{") or text.startswith("["):
            return json.loads(text)

        tree = ast.parse(text, mode="exec")
        pipeline_node = None
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "pipeline":
                        pipeline_node = node.value
                        break
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "pipeline":
                pipeline_node = node.value
            if pipeline_node is not None:
                break
        if pipeline_node is None:
            # Also allow a file that contains only a literal dict/list expression.
            if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
                pipeline_node = tree.body[0].value
            else:
                raise ValueError("Expected Python code like: pipeline = {'version': 3, 'blocks': [...]}.")
        return ast.literal_eval(pipeline_node)

    def refresh_code_from_blocks(self):
        if not hasattr(self, "pipeline_code"):
            return
        try:
            self.pipeline_code.setPlainText(self._pipeline_to_python_code())
        except Exception as exc:
            self._log(f"Could not refresh code view: {exc}")

    def apply_code_to_blocks(self, silent: bool = False) -> bool:
        if not hasattr(self, "pipeline_code"):
            return True
        try:
            data = self._parse_pipeline_code(self.pipeline_code.toPlainText() or "")
            raw_blocks = data.get("blocks", data if isinstance(data, list) else [])
            if not isinstance(raw_blocks, list):
                raise ValueError("Expected Python pipeline with a 'blocks' list, or a raw list of blocks.")
            new_blocks: List[PipelineBlock] = []
            known = set(PipelineBlock.__dataclass_fields__.keys())
            for i, raw in enumerate(raw_blocks):
                if not isinstance(raw, dict):
                    raise ValueError(f"Block #{i + 1} is not an object.")
                clean = {k: v for k, v in raw.items() if k in known}
                if not clean.get("id"):
                    clean["id"] = "b_" + uuid.uuid4().hex[:8]
                if not clean.get("name"):
                    clean["name"] = f"Block {i + 1}"
                if not clean.get("type"):
                    clean["type"] = "yolo"
                if "x" not in clean:
                    clean["x"] = 80 + i * 260
                if "y" not in clean:
                    clean["y"] = 100
                new_blocks.append(PipelineBlock(**clean))
            ids = {b.id for b in new_blocks}
            for b in new_blocks:
                if b.input_ref not in ids and b.input_ref != "original":
                    b.input_ref = "original"
            self.blocks = new_blocks
            self._rebuild_scene()
            if self.blocks:
                self.select_block(self.blocks[0].id)
            if not silent:
                self._log(f"Applied code view: {len(self.blocks)} block(s).")
            return True
        except Exception as exc:
            if not silent:
                QMessageBox.critical(self, "YOLO Pipeline Python code", f"Could not apply code view:\n{exc}")
            return False

    def toggle_block_code_view(self):
        if not hasattr(self, "mode_stack"):
            return
        if self.mode_stack.currentIndex() == 0:
            self.refresh_code_from_blocks()
            self.mode_stack.setCurrentIndex(1)
            self.view_mode_btn.setText("View: Blocks")
        else:
            if self.apply_code_to_blocks(silent=False):
                self.mode_stack.setCurrentIndex(0)
                self.view_mode_btn.setText("View: Code")

    # ---------- Visual model ----------
    def block_label(self, block_id: str) -> str:
        if block_id == "original":
            return "Original image"
        b = self._block_by_id(block_id)
        return b.name if b else str(block_id)

    def _block_by_id(self, block_id: str) -> Optional[PipelineBlock]:
        for b in self.blocks:
            if b.id == block_id:
                return b
        return None

    def _current_block(self) -> Optional[PipelineBlock]:
        return self._block_by_id(self.selected_block_id or "")

    def _add_block(self, typ="yolo", name=None, x=None, y=None, input_ref=None):
        idx = len(self.blocks) + 1
        if x is None:
            x = 100 + (idx - 1) * 55
        if y is None:
            y = 100 + (idx - 1) * 55
        default_names = {
            "yolo": "YOLO block",
            "owlv2": "OWLv2 block",
            "grounding_dino": "Grounding DINO block",
            "lae_dino": "LAE-DINO block",
            "sam2": "SAM2 block",
            "formlearner": "FormLearner block",
            "rule": "IF / Logic block",
        }
        default_name = default_names.get(str(typ), "Pipeline block")
        b = PipelineBlock(type=typ, name=name or default_name, x=float(x), y=float(y))
        if typ == "owlv2":
            b.classes_filter = "mustatil, burial mound, mound, house"
            b.confidence = 0.05
            b.tile_size = 768
            b.overlap = 128
            b.use_chunked_yolo = True
        elif typ == "grounding_dino":
            b.classes_filter = "mustatil, burial mound, tumulus, stone enclosure"
            b.confidence = 0.10
            b.tile_size = 768
            b.overlap = 128
            b.use_chunked_yolo = True
        elif typ == "lae_dino":
            b.classes_filter = "mustatil, false_positive"
            b.confidence = 0.05
            b.tile_size = 768
            b.overlap = 128
            b.use_chunked_yolo = True
            try:
                b.model_path = str(getattr(self.ws, "mustatil_lae_existing_v9_config", "") or getattr(self.ws, "mustatil_lae_config", "") or "")
                b.formlearner_model = str(getattr(self.ws, "mustatil_lae_existing_v9_weights", "") or getattr(self.ws, "mustatil_lae_weights", "") or "")
            except Exception:
                pass
        elif typ == "sam2":
            b.classes_filter = "sam2_mask"
            b.confidence = 0.05
            b.use_chunked_yolo = False
            b.crop_padding = 16
        if input_ref is not None:
            b.input_ref = input_ref
        elif typ != "yolo" and self.blocks:
            b.input_ref = self.blocks[-1].id
        self.blocks.append(b)
        self._create_item_for_block(b)
        self.refresh_arrows()
        self.select_block(b.id)
        self.refresh_code_from_blocks()
        return b

    def _create_item_for_block(self, b: PipelineBlock):
        item = VisualBlockItem(b, self)
        self.block_items[b.id] = item
        self.scene.addItem(item)

    def _rebuild_scene(self):
        self.scene.clear()
        self.block_items = {}
        self.arrows = []
        for b in self.blocks:
            self._create_item_for_block(b)
        self.refresh_arrows()
        if self.blocks:
            self.select_block(self.blocks[0].id)
        self.refresh_code_from_blocks()

    def refresh_arrows(self):
        for a in list(self.arrows):
            try:
                self.scene.removeItem(a)
            except Exception:
                pass
        self.arrows = []
        drawn = set()
        for b in self.blocks:
            refs = self._block_dependency_refs(b)
            for ref in refs:
                key = (ref, b.id)
                if key in drawn:
                    continue
                if ref and ref != "original" and ref in self.block_items and b.id in self.block_items:
                    a = ArrowItem(self.block_items[ref], self.block_items[b.id], self)
                    self.arrows.append(a)
                    self.scene.addItem(a)
                    drawn.add(key)
        for item in self.block_items.values():
            try:
                item.refresh_embedded_controls()
            except Exception:
                pass
            item.update()

    def remove_connection(self, source_id: str, target_id: str):
        """Remove a visual connection by right-clicking the line.

        For normal blocks this resets target.input_ref to Original image.
        For IF/Logic blocks this removes the matching condition source and also
        clears within_parent if that was the clicked source.
        """
        source_id = self._resolve_block_ref(source_id) or source_id
        target = self._block_by_id(target_id)
        if not target or not source_id:
            return
        removed = False
        if target.type == "rule":
            try:
                rule = self._current_rule_dict(target)
                conds = []
                for c in list(rule.get("conditions") or []):
                    if self._resolve_block_ref(c.get("source")) == source_id:
                        removed = True
                        continue
                    conds.append(c)
                rule["conditions"] = conds
                if self._resolve_block_ref(rule.get("within_parent") or rule.get("parent")) == source_id:
                    rule["within_parent"] = ""
                    removed = True
                if target.input_ref == source_id:
                    target.input_ref = "original"
                    removed = True
                target.rule_json = self._rule_to_text(rule)
            except Exception as exc:
                self._log(f"Could not remove IF/Logic connection: {exc}")
        else:
            if target.input_ref == source_id:
                target.input_ref = "original"
                removed = True

        if removed:
            if target.id in self.block_items:
                self.block_items[target.id].refresh_embedded_controls()
                self.block_items[target.id].update()
            if self.selected_block_id == target.id:
                self._load_block_to_editor(target)
            self.refresh_arrows()
            self._refresh_input_refs()
            self.refresh_code_from_blocks()
            self._log(f"Removed connection: {self.block_label(source_id)} → {target.name}")

    def select_block(self, block_id: str, from_scene: bool = False):
        b = self._block_by_id(block_id)
        if not b:
            return
        self.selected_block_id = b.id
        if not from_scene:
            for bid, item in self.block_items.items():
                item.setSelected(bid == b.id)
        self._load_block_to_editor(b)

    def _refresh_input_refs(self):
        current = self.input_ref.currentData() if hasattr(self, 'input_ref') else 'original'
        current_block = self._current_block()
        self.input_ref.blockSignals(True)
        self.input_ref.clear()
        self.input_ref.addItem("Original image", "original")
        for b in self.blocks:
            if current_block and b.id == current_block.id:
                continue
            self.input_ref.addItem(f"{b.name} ({b.type})", b.id)
        ix = self.input_ref.findData(current)
        self.input_ref.setCurrentIndex(ix if ix >= 0 else 0)
        self.input_ref.blockSignals(False)

    def _load_block_to_editor(self, b: PipelineBlock):
        self._updating = True
        self._refresh_input_refs()
        self.enabled.setChecked(bool(b.enabled))
        self.name.setText(b.name)
        self.typ.setCurrentText(b.type)
        ix = self.input_ref.findData(b.input_ref)
        self.input_ref.setCurrentIndex(ix if ix >= 0 else 0)
        self.model_path._edit.setText(b.model_path)
        self.conf.setValue(float(b.confidence))
        self.imgsz.setValue(int(b.imgsz))
        self.device.setCurrentText(b.device)
        self.use_chunked_yolo.setChecked(bool(getattr(b, "use_chunked_yolo", True)))
        self.tile_size.setValue(int(getattr(b, "tile_size", 1024) or 1024))
        self.tile_overlap.setValue(int(getattr(b, "overlap", 384) or 0))
        self.shifted_tiles.setChecked(bool(getattr(b, "shifted_tiles", True)))
        self.crop_padding.setValue(int(b.crop_padding))
        self.classes_filter.setText(b.classes_filter)
        self.form_model._edit.setText(b.formlearner_model)
        self.form_threshold.setValue(float(b.form_threshold))
        self.rule_json.setPlainText(b.rule_json)
        self._updating = False

    def _save_editor_to_block(self, *_):
        if self._updating:
            return
        b = self._current_block()
        if not b:
            return
        b.enabled = self.enabled.isChecked()
        b.name = self.name.text().strip() or b.name
        b.type = self.typ.currentText().strip() or "yolo"
        new_input = self.input_ref.currentData() or "original"
        if new_input == b.id:
            new_input = "original"
        b.input_ref = new_input
        b.model_path = self.model_path._edit.text().strip()
        b.confidence = float(self.conf.value())
        b.imgsz = int(self.imgsz.value())
        b.device = self.device.currentText().strip() or "auto cuda"
        b.use_chunked_yolo = bool(self.use_chunked_yolo.isChecked())
        b.tile_size = int(self.tile_size.value())
        b.overlap = int(self.tile_overlap.value())
        b.shifted_tiles = bool(self.shifted_tiles.isChecked())
        b.crop_padding = int(self.crop_padding.value())
        b.classes_filter = self.classes_filter.text().strip()
        b.formlearner_model = self.form_model._edit.text().strip()
        b.form_threshold = float(self.form_threshold.value())
        b.rule_json = self.rule_json.toPlainText()
        if b.id in self.block_items:
            self.block_items[b.id].refresh_embedded_controls()
            self.block_items[b.id].update()
        self.refresh_arrows()
        self._refresh_input_refs()
        if not hasattr(self, "mode_stack") or self.mode_stack.currentIndex() == 0:
            self.refresh_code_from_blocks()

    def _duplicate_current(self):
        b = self._current_block()
        if not b:
            return
        d = asdict(b)
        d["id"] = "b_" + uuid.uuid4().hex[:8]
        d["name"] = b.name + " copy"
        d["x"] = float(b.x) + 35
        d["y"] = float(b.y) + 35
        new_b = PipelineBlock(**d)
        self.blocks.append(new_b)
        self._create_item_for_block(new_b)
        self.refresh_arrows()
        self.select_block(new_b.id)
        self.refresh_code_from_blocks()

    def _delete_current(self):
        b = self._current_block()
        if not b:
            return
        if b.id in self.block_items:
            self.scene.removeItem(self.block_items[b.id])
            del self.block_items[b.id]
        self.blocks = [x for x in self.blocks if x.id != b.id]
        for x in self.blocks:
            if x.input_ref == b.id:
                x.input_ref = "original"
        self.selected_block_id = None
        self.refresh_arrows()
        if self.blocks:
            self.select_block(self.blocks[0].id)
        self.refresh_code_from_blocks()

    def auto_layout(self):
        """Lay out blocks with dynamic sizes, collision avoidance and padding.

        The block widgets can now grow with their embedded controls. Therefore
        the layout must use the real rendered block dimensions and then run a
        small collision-resolution pass so no two padded block rectangles touch.
        """
        if not self.blocks:
            return

        # Make sure every block has its latest size before measuring. Embedded
        # controls can change width/height after model paths, logic rows, etc.
        for b in self.blocks:
            item = self.block_items.get(b.id)
            if item:
                try:
                    item.refresh_embedded_controls()
                    item.update()
                except Exception:
                    pass

        levels: Dict[str, int] = {}

        def level(b: PipelineBlock, seen: Set[str]) -> int:
            if b.id in levels:
                return levels[b.id]
            refs = [r for r in self._block_dependency_refs(b) if r and r != "original"]
            if not refs or b.id in seen:
                levels[b.id] = 0
            else:
                parent_levels = []
                for ref in refs:
                    parent = self._block_by_id(ref)
                    if parent:
                        parent_levels.append(level(parent, seen | {b.id}))
                levels[b.id] = 1 + (max(parent_levels) if parent_levels else 0)
            return levels[b.id]

        for b in self.blocks:
            level(b, set())

        buckets: Dict[int, List[PipelineBlock]] = {}
        for b in self.blocks:
            buckets.setdefault(levels.get(b.id, 0), []).append(b)

        # Stable visual order inside each dependency level.
        for lvl in buckets:
            buckets[lvl].sort(key=lambda b: (float(b.y), float(b.x), b.name))

        margin_x = 90.0
        margin_y = 80.0
        min_gap_x = 120.0
        min_gap_y = 70.0
        collision_padding = 44.0

        def size_of(block: PipelineBlock):
            item = self.block_items.get(block.id)
            if item:
                try:
                    return float(item.block_width()), float(item.block_height())
                except Exception:
                    pass
            return 360.0, 210.0

        # Determine real column widths first.
        level_widths: Dict[int, float] = {}
        for lvl, items in buckets.items():
            level_widths[lvl] = max([size_of(b)[0] for b in items] or [360.0])

        x_by_level: Dict[int, float] = {}
        x_cursor = margin_x
        for lvl in sorted(buckets):
            x_by_level[lvl] = x_cursor
            x_cursor += level_widths.get(lvl, 360.0) + min_gap_x

        # Initial dependency-column layout. Blocks in the same column are stacked
        # using their own measured height and a real minimum gap.
        for lvl in sorted(buckets):
            y_cursor = margin_y
            col_width = level_widths.get(lvl, 360.0)
            for b in buckets[lvl]:
                w, h = size_of(b)
                b.x = x_by_level[lvl] + max(0.0, (col_width - w) / 2.0)
                b.y = y_cursor
                y_cursor += h + min_gap_y

        # Collision pass: treat every block as a padded rectangle. If two padded
        # rectangles overlap, push the visually lower/right block downward until
        # the required spacing is restored. A few passes are enough for the small
        # block counts used in the builder and avoid ugly overlaps after very
        # large embedded dropdown content.
        def padded_rect(block: PipelineBlock):
            w, h = size_of(block)
            return (
                float(block.x) - collision_padding,
                float(block.y) - collision_padding,
                float(block.x) + w + collision_padding,
                float(block.y) + h + collision_padding,
            )

        def intersects(a, b) -> bool:
            return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

        ordered = sorted(self.blocks, key=lambda b: (levels.get(b.id, 0), float(b.y), float(b.x), b.name))
        for _pass in range(max(4, len(ordered) * 2)):
            moved = False
            ordered = sorted(ordered, key=lambda b: (levels.get(b.id, 0), float(b.y), float(b.x), b.name))
            for i, a in enumerate(ordered):
                ra = padded_rect(a)
                for b in ordered[i + 1:]:
                    rb = padded_rect(b)
                    if not intersects(ra, rb):
                        continue
                    # Prefer moving the block with the greater dependency level;
                    # otherwise move the lower block. This preserves the left-to-
                    # right data-flow shape while removing overlaps.
                    move = b
                    if levels.get(a.id, 0) > levels.get(b.id, 0):
                        move = a
                        base = b
                    else:
                        base = a
                    base_rect = padded_rect(base)
                    _, mh = size_of(move)
                    new_y = base_rect[3] + collision_padding + min_gap_y
                    if new_y <= move.y:
                        new_y = move.y + mh + min_gap_y
                    move.y = float(new_y)
                    moved = True
            if not moved:
                break

        # Apply final positions to QGraphicsItems.
        for b in self.blocks:
            item = self.block_items.get(b.id)
            if item:
                try:
                    item.prepareGeometryChange()
                except Exception:
                    pass
                item.setPos(float(b.x), float(b.y))
                try:
                    item.refresh_embedded_controls()
                    item.update()
                except Exception:
                    pass

        self.refresh_arrows()
        self.refresh_code_from_blocks()
        rect = self.scene.itemsBoundingRect().adjusted(-120, -120, 120, 120)
        self.scene.setSceneRect(rect)
        self.view.fitInView(rect, Qt.KeepAspectRatio)

    # ---------- Persistence ----------
    def save_pipeline(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save YOLO pipeline", "yolo_pipeline_visual.json", "JSON (*.json);;All files (*)")
        if not path:
            return
        Path(path).write_text(json.dumps(self.pipeline_to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        self._log(f"Saved visual pipeline: {path}")

    def load_pipeline(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load YOLO pipeline", "", "JSON (*.json);;All files (*)")
        if not path:
            return
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        blocks = []
        for x in data.get("blocks", []):
            if "x" not in x: x["x"] = 80 + len(blocks) * 250
            if "y" not in x: x["y"] = 100
            blocks.append(PipelineBlock(**x))
        self.blocks = blocks
        self._rebuild_scene()
        self._log(f"Loaded visual pipeline: {path}")

    def _sync_positions(self):
        for bid, item in self.block_items.items():
            b = self._block_by_id(bid)
            if b:
                p = item.pos()
                b.x = float(p.x()); b.y = float(p.y())

    # ---------- Pipeline execution ----------
    def run_pipeline_clicked(self):
        if hasattr(self.ws, "run_task"):
            self.ws.run_task("YOLO Visual Pipeline", self.run_pipeline)
        else:
            self.run_pipeline()

    def _execution_order(self) -> List[PipelineBlock]:
        by_id = {b.id: b for b in self.blocks if b.enabled}
        order: List[PipelineBlock] = []
        visiting: Set[str] = set()
        visited: Set[str] = set()
        def visit(b: PipelineBlock):
            if b.id in visited:
                return
            if b.id in visiting:
                raise RuntimeError(f"Pipeline has a connection cycle at block: {b.name}")
            visiting.add(b.id)
            if b.input_ref in by_id:
                visit(by_id[b.input_ref])
            # Rule JSON can reference multiple blocks through conditions.
            for rid in self._block_dependency_refs(b):
                if rid in by_id:
                    visit(by_id[rid])
            visiting.remove(b.id)
            visited.add(b.id)
            order.append(b)
        for b in list(by_id.values()):
            visit(b)
        return order

    def run_pipeline(self):
        image_path = self._current_pipeline_image_path()
        if not image_path.is_file():
            raise RuntimeError("Choose a valid image at the top of the YOLO Pipeline tab first. This image is used as the Original image block.")
        self.results = []
        self.results_by_block = {}
        self._log(f"Visual pipeline started: {image_path}")
        full = Image.open(image_path)  # lazy open; crops are converted to RGB only per parent/tile
        order = self._execution_order()
        self._log("Execution order: " + " → ".join(b.name for b in order))
        for b in order:
            if b.type == "yolo":
                out = self._run_yolo_block(b, full, image_path)
            elif b.type == "owlv2":
                out = self._run_owlv2_block(b, full, image_path)
            elif b.type == "grounding_dino":
                out = self._run_grounding_dino_block(b, full, image_path)
            elif b.type == "lae_dino":
                out = self._run_lae_dino_block(b, full, image_path)
            elif b.type == "sam2":
                out = self._run_sam2_block(b, full, image_path)
            elif b.type == "formlearner":
                out = self._run_formlearner_block(b, full)
            elif b.type == "rule":
                out = self._run_rule_block(b)
            else:
                self._log(f"Unknown block type skipped: {b.name} / {b.type}")
                out = []
            self.results_by_block[b.id] = out
            self.results.extend(out)
            self._log(f"{b.name}: {len(out)} result(s)")
        self._log(f"Visual pipeline finished. Total records: {len(self.results)}")
        self._write_summary()
        return self.results

    def _input_records(self, b: PipelineBlock, full: Image.Image) -> List[Dict[str, Any]]:
        if b.input_ref == "original":
            return [{"id": "original", "x1": 0.0, "y1": 0.0, "x2": float(full.width), "y2": float(full.height), "label": "original", "block_id": "original"}]
        return list(self.results_by_block.get(b.input_ref, []))

    def _resolve_yolo_device(self, requested: str) -> str:
        """Resolve pipeline device names against the already installed runtime.

        - "auto cuda" chooses CUDA device 0 when PyTorch reports CUDA support.
        - "cuda" is kept as CUDA when available, otherwise CPU fallback is used.
        - "0" is passed through for Ultralytics CUDA device 0.
        """
        req = str(requested or "auto cuda").strip().lower()
        if req in {"auto", "auto cuda", "cuda auto", "gpu"}:
            try:
                import torch
                if torch.cuda.is_available():
                    return "0"
            except Exception:
                pass
            return "cpu"
        if req in {"cuda", "0", "cuda:0"}:
            try:
                import torch
                if torch.cuda.is_available():
                    return "0" if req in {"cuda", "cuda:0"} else req
            except Exception:
                pass
            self._log(f"CUDA requested for a YOLO pipeline block, but torch.cuda.is_available() is false. Falling back to CPU.")
            return "cpu"
        return req

    def _tile_positions(self, width: int, height: int, tile: int, overlap: int, shifted: bool = True) -> List[Tuple[int, int]]:
        tile = max(32, int(tile or 1024))
        overlap = max(0, min(int(overlap or 0), tile - 1))
        stride = max(1, tile - overlap)
        offsets = [0]
        if shifted and stride // 2 > 0:
            offsets.append(stride // 2)
        out: List[Tuple[int, int]] = []
        seen = set()
        for off in offsets:
            y = off
            while y < height:
                x = off
                while x < width:
                    px = min(x, max(0, width - 1))
                    py = min(y, max(0, height - 1))
                    key = (px, py)
                    if key not in seen:
                        seen.add(key); out.append(key)
                    if x + tile >= width:
                        break
                    x += stride
                if y + tile >= height:
                    break
                y += stride
        return out

    def _predict_yolo_array(self, model, arr, b: PipelineBlock, device: str):
        # Prefer the workspace helper because it already contains Mustatil's CUDA /
        # DirectML / provider compatibility logic.
        if hasattr(self.ws, "_yolo_predict_selected_device"):
            return self.ws._yolo_predict_selected_device(
                model, arr, conf=float(b.confidence), imgsz=int(b.imgsz), verbose=False
            )
        kwargs = dict(conf=float(b.confidence), imgsz=int(b.imgsz), verbose=False)
        if device not in {"directml", "openml"}:
            kwargs["device"] = device
        return model.predict(arr, **kwargs)

    def _bbox_iou(self, a: Dict[str, Any], b: Dict[str, Any]) -> float:
        ax1, ay1, ax2, ay2 = float(a["x1"]), float(a["y1"]), float(a["x2"]), float(a["y2"])
        bx1, by1, bx2, by2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        return inter / max(1e-9, aa + bb - inter)

    def _deduplicate_yolo_records(self, recs: List[Dict[str, Any]], iou_thr: float = 0.55) -> List[Dict[str, Any]]:
        # Simple class-aware NMS for duplicated overlap/shifted-tile detections.
        ordered = sorted(recs, key=lambda r: float(r.get("conf", 0.0) or 0.0), reverse=True)
        kept: List[Dict[str, Any]] = []
        for r in ordered:
            cls = int(r.get("class_id", -1) or -1)
            parent = str(r.get("parent_id", ""))
            duplicate = False
            for k in kept:
                if int(k.get("class_id", -2) or -2) != cls:
                    continue
                if str(k.get("parent_id", "")) != parent:
                    continue
                if self._bbox_iou(r, k) >= float(iou_thr):
                    duplicate = True
                    break
            if not duplicate:
                kept.append(r)
        return kept

    def _run_yolo_block(self, b: PipelineBlock, full: Image.Image, image_path: Path) -> List[Dict[str, Any]]:
        model_path = Path(b.model_path.strip().strip('"')).expanduser()
        if not model_path.is_file():
            raise RuntimeError(f"YOLO block '{b.name}' needs a valid model file.")
        if np is None:
            raise RuntimeError("NumPy is missing. Install numpy for YOLO pipeline execution.")
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        class_filter = self._parse_class_filter(b.classes_filter)
        parents = self._input_records(b, full)
        out: List[Dict[str, Any]] = []
        resolved_device = self._resolve_yolo_device(getattr(b, "device", "auto cuda"))
        old_device = None
        if hasattr(self.ws, "device"):
            try:
                old_device = self.ws.device.get()
                self.ws.device.set(resolved_device)
            except Exception:
                old_device = None
        try:
            tile_size = max(128, int(getattr(b, "tile_size", 1024) or 1024))
            overlap = max(0, min(int(getattr(b, "overlap", 384) or 0), tile_size - 1))
            shifted = bool(getattr(b, "shifted_tiles", True))
            use_chunked = bool(getattr(b, "use_chunked_yolo", True))
            self._log(f"{b.name}: device={resolved_device}, chunked={use_chunked}, tile={tile_size}, overlap={overlap}, shifted={shifted}")
            for parent in parents:
                crop, offset = self._crop_parent(full, parent, int(b.crop_padding))
                crop = crop.convert("RGB")
                crop_w, crop_h = int(crop.width), int(crop.height)
                # Chunk original maps and oversized parent crops exactly to avoid feeding
                # enormous images into YOLO or the GPU. Small parent crops are predicted once.
                do_tiles = use_chunked and (crop_w > tile_size or crop_h > tile_size)
                positions = self._tile_positions(crop_w, crop_h, tile_size, overlap, shifted) if do_tiles else [(0, 0)]
                if do_tiles:
                    self._log(f"{b.name}: parent {parent.get('id')} split into {len(positions)} YOLO tile(s).")
                for tx, ty in positions:
                    tw = min(tile_size, crop_w - tx)
                    th = min(tile_size, crop_h - ty)
                    if tw <= 0 or th <= 0:
                        continue
                    tile_img = crop.crop((tx, ty, tx + tw, ty + th)).convert("RGB") if do_tiles else crop
                    arr = np.asarray(tile_img)
                    res = self._predict_yolo_array(model, arr, b, resolved_device)
                    if not res or getattr(res[0], "boxes", None) is None:
                        continue
                    names = getattr(model, "names", {}) or {}
                    boxes = res[0].boxes
                    xy = boxes.xyxy.cpu().numpy(); cf = boxes.conf.cpu().numpy(); cl = boxes.cls.cpu().numpy()
                    for bb, conf, cls_id in zip(xy, cf, cl):
                        cls_i = int(cls_id)
                        label = str(names.get(cls_i, cls_i)) if isinstance(names, dict) else str(cls_i)
                        if class_filter and str(cls_i).lower() not in class_filter and label.lower() not in class_filter:
                            continue
                        x1, y1, x2, y2 = map(float, bb[:4])
                        x1 += offset[0] + tx; x2 += offset[0] + tx; y1 += offset[1] + ty; y2 += offset[1] + ty
                        x1 = max(0.0, min(float(full.width), x1)); x2 = max(0.0, min(float(full.width), x2))
                        y1 = max(0.0, min(float(full.height), y1)); y2 = max(0.0, min(float(full.height), y2))
                        if x2 <= x1 or y2 <= y1:
                            continue
                        out.append({
                            "id": "r_" + uuid.uuid4().hex[:10],
                            "block_id": b.id, "block_name": b.name, "block_type": b.type,
                            "parent_id": parent.get("id"), "parent_block_id": parent.get("block_id"),
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "conf": float(conf), "class_id": cls_i, "label": label,
                            "model": str(model_path), "source_image": str(image_path),
                            "tile_x": int(tx), "tile_y": int(ty), "tile_size": int(tile_size), "device": str(resolved_device),
                        })
        finally:
            if old_device is not None:
                try:
                    self.ws.device.set(old_device)
                except Exception:
                    pass
        before = len(out)
        out = self._deduplicate_yolo_records(out, iou_thr=0.55)
        if len(out) != before:
            self._log(f"{b.name}: removed {before - len(out)} duplicate overlap detection(s) by NMS.")
        return out

    # ---------- Open-vocabulary / segmentation pipeline blocks ----------
    def _mm_labels(self, b: PipelineBlock, default: List[str]) -> List[str]:
        vals = [p.strip() for p in str(getattr(b, "classes_filter", "") or "").replace(";", ",").replace("\n", ",").split(",") if p.strip()]
        return vals or list(default)

    def _mm_torch_device(self, requested: str) -> str:
        resolved = self._resolve_yolo_device(requested)
        if str(resolved).strip() in {"0", "cuda"}:
            return "cuda:0"
        return str(resolved or "cpu")

    def _mm_cache(self) -> Dict[Any, Any]:
        cache = getattr(self, "_multimodel_pipeline_cache", None)
        if cache is None:
            cache = {}
            self._multimodel_pipeline_cache = cache
        return cache

    def _mm_make_record(self, b: PipelineBlock, parent: Dict[str, Any], full: Image.Image,
                        image_path: Path, x1: float, y1: float, x2: float, y2: float,
                        conf: float, class_id: int, label: str, model: str,
                        extra: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        x1 = max(0.0, min(float(full.width), float(x1)))
        x2 = max(0.0, min(float(full.width), float(x2)))
        y1 = max(0.0, min(float(full.height), float(y1)))
        y2 = max(0.0, min(float(full.height), float(y2)))
        if x2 <= x1 or y2 <= y1:
            return None
        rec = {
            "id": "r_" + uuid.uuid4().hex[:10],
            "block_id": b.id, "block_name": b.name, "block_type": b.type,
            "parent_id": parent.get("id"), "parent_block_id": parent.get("block_id"),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "conf": float(conf), "class_id": int(class_id), "label": str(label),
            "model": str(model), "source_image": str(image_path),
            "device": str(getattr(b, "device", "auto cuda")),
        }
        if extra:
            rec.update(extra)
        return rec

    def _mm_open_vocab_block(self, b: PipelineBlock, full: Image.Image, image_path: Path, backend: str) -> List[Dict[str, Any]]:
        parents = self._input_records(b, full)
        out: List[Dict[str, Any]] = []
        tile_size = max(128, int(getattr(b, "tile_size", 768) or 768))
        overlap = max(0, min(int(getattr(b, "overlap", 128) or 0), tile_size - 1))
        shifted = bool(getattr(b, "shifted_tiles", False))
        use_chunked = bool(getattr(b, "use_chunked_yolo", True))
        conf = max(0.001, min(1.0, float(getattr(b, "confidence", 0.05) or 0.05)))
        self._log(f"{b.name}: {backend} block started; conf={conf}, tile={tile_size}, overlap={overlap}, prompts={getattr(b, 'classes_filter', '')}")
        for parent in parents:
            crop, offset = self._crop_parent(full, parent, int(getattr(b, "crop_padding", 0) or 0))
            crop = crop.convert("RGB")
            crop_w, crop_h = int(crop.width), int(crop.height)
            do_tiles = use_chunked and (crop_w > tile_size or crop_h > tile_size)
            positions = self._tile_positions(crop_w, crop_h, tile_size, overlap, shifted) if do_tiles else [(0, 0)]
            if do_tiles:
                self._log(f"{b.name}: parent {parent.get('id')} split into {len(positions)} tile(s).")
            for tx, ty in positions:
                tw = min(tile_size, crop_w - tx)
                th = min(tile_size, crop_h - ty)
                if tw <= 0 or th <= 0:
                    continue
                tile_img = crop.crop((tx, ty, tx + tw, ty + th)).convert("RGB") if do_tiles else crop
                if backend == "OWLv2":
                    local = self._mm_owlv2_predict(b, tile_img, conf)
                elif backend == "Grounding DINO":
                    local = self._mm_grounding_dino_predict(b, tile_img, conf)
                elif backend == "LAE-DINO":
                    local = self._mm_lae_dino_predict(b, tile_img, conf)
                else:
                    local = []
                for r in local:
                    try:
                        x1 = float(r["x1"]) + offset[0] + tx
                        y1 = float(r["y1"]) + offset[1] + ty
                        x2 = float(r["x2"]) + offset[0] + tx
                        y2 = float(r["y2"]) + offset[1] + ty
                        rec = self._mm_make_record(
                            b, parent, full, image_path,
                            x1, y1, x2, y2,
                            float(r.get("conf", r.get("confidence", r.get("score", 0.0))) or 0.0),
                            int(r.get("class_id", r.get("cls", 0)) or 0),
                            str(r.get("label", r.get("class_name", "object"))),
                            str(r.get("model", backend)),
                            {"tile_x": int(tx), "tile_y": int(ty), "tile_size": int(tile_size), "backend": backend},
                        )
                        if rec:
                            out.append(rec)
                    except Exception:
                        continue
        before = len(out)
        out = self._deduplicate_yolo_records(out, iou_thr=0.55)
        if len(out) != before:
            self._log(f"{b.name}: removed {before - len(out)} duplicate overlap detection(s) by NMS.")
        return out

    def _mm_owlv2_predict(self, b: PipelineBlock, pil_image: Image.Image, conf: float) -> List[Dict[str, Any]]:
        labels = self._mm_labels(b, ["mustatil", "burial mound", "mound", "house"])
        model_id = str(getattr(b, "model_path", "") or "").strip() or "google/owlv2-base-patch16-ensemble"
        device = self._mm_torch_device(getattr(b, "device", "auto cuda"))
        key = ("owlv2", model_id, device)
        cache = self._mm_cache()
        if key not in cache:
            import torch
            from transformers import Owlv2Processor, Owlv2ForObjectDetection
            proc = Owlv2Processor.from_pretrained(model_id)
            model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device)
            model.eval()
            cache[key] = (proc, model, device)
            self._log(f"OWLv2 loaded: {model_id} on {device}")
        proc, model, device = cache[key]
        import torch
        inputs = proc(text=[labels], images=pil_image, return_tensors="pt")
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([pil_image.size[::-1]], device=device)
        processed = proc.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=float(conf))
        out = []
        if not processed:
            return out
        r = processed[0]
        for box, score, label_id in zip(r.get("boxes", []), r.get("scores", []), r.get("labels", [])):
            li = int(label_id.detach().cpu().item() if hasattr(label_id, "detach") else label_id)
            label = labels[li] if 0 <= li < len(labels) else str(li)
            xy = box.detach().cpu().tolist() if hasattr(box, "detach") else list(box)
            sc = float(score.detach().cpu().item() if hasattr(score, "detach") else score)
            if len(xy) >= 4:
                out.append({"x1": xy[0], "y1": xy[1], "x2": xy[2], "y2": xy[3], "conf": sc, "class_id": li, "label": label, "model": model_id})
        return out

    def _mm_grounding_dino_predict(self, b: PipelineBlock, pil_image: Image.Image, conf: float) -> List[Dict[str, Any]]:
        labels = self._mm_labels(b, ["mustatil", "burial mound", "tumulus", "stone enclosure"])
        model_id = str(getattr(b, "model_path", "") or "").strip() or "IDEA-Research/grounding-dino-base"
        device = self._mm_torch_device(getattr(b, "device", "auto cuda"))
        key = ("grounding_dino", model_id, device)
        cache = self._mm_cache()
        if key not in cache:
            import torch
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            proc = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
            model.eval()
            cache[key] = (proc, model, device)
            self._log(f"Grounding DINO loaded: {model_id} on {device}")
        proc, model, device = cache[key]
        import torch
        prompt = ". ".join(labels)
        if prompt and not prompt.endswith("."):
            prompt += "."
        inputs = proc(images=pil_image, text=prompt, return_tensors="pt")
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([pil_image.size[::-1]], device=device)
        post = getattr(proc, "post_process_grounded_object_detection", None)
        if post is None:
            raise RuntimeError("Grounding DINO processor has no post_process_grounded_object_detection().")
        input_ids = inputs.get("input_ids")
        attempts = [
            lambda: post(outputs=outputs, input_ids=input_ids, threshold=float(conf), target_sizes=target_sizes),
            lambda: post(outputs, input_ids, threshold=float(conf), target_sizes=target_sizes),
            lambda: post(outputs=outputs, input_ids=input_ids, box_threshold=float(conf), text_threshold=0.25, target_sizes=target_sizes),
            lambda: post(outputs, input_ids, box_threshold=float(conf), text_threshold=0.25, target_sizes=target_sizes),
            lambda: post(outputs=outputs, box_threshold=float(conf), text_threshold=0.25, target_sizes=target_sizes),
            lambda: post(outputs, target_sizes=target_sizes, threshold=float(conf)),
        ]
        processed = None
        last_exc = None
        for fn in attempts:
            try:
                processed = fn()
                break
            except TypeError as exc:
                last_exc = exc
        if processed is None:
            raise RuntimeError("Grounding DINO postprocess API mismatch: " + str(last_exc))
        out = []
        if not processed:
            return out
        r = processed[0]
        text_labels = r.get("text_labels", r.get("labels", []))
        for box, score, raw_label in zip(r.get("boxes", []), r.get("scores", []), text_labels):
            label = str(raw_label)
            li = 0
            low = label.lower()
            for j, name in enumerate(labels):
                if name.lower() in low or low in name.lower():
                    li = j
                    label = name
                    break
            xy = box.detach().cpu().tolist() if hasattr(box, "detach") else list(box)
            sc = float(score.detach().cpu().item() if hasattr(score, "detach") else score)
            if len(xy) >= 4:
                out.append({"x1": xy[0], "y1": xy[1], "x2": xy[2], "y2": xy[3], "conf": sc, "class_id": li, "label": label, "model": model_id})
        return out

    def _mm_find_lae_runner(self):
        for mod in list(sys.modules.values()):
            try:
                fn = getattr(mod, "_run_lae_on_pil", None)
                if callable(fn):
                    return mod, fn
            except Exception:
                pass
        return None, None

    def _mm_lae_runtime_paths(self) -> Tuple[Path, Path]:
        plugin_dir = Path(__file__).resolve().parent
        runtime = plugin_dir / "mustatil_model_runtimes" / "LAE-DINO-PATCH"
        py = runtime / "python310" / ("python.exe" if os.name == "nt" else "python")
        repo = runtime / "repo"
        return py, repo

    def _mm_lae_dino_predict(self, b: PipelineBlock, pil_image: Image.Image, conf: float) -> List[Dict[str, Any]]:
        _mod, fn = self._mm_find_lae_runner()
        if fn is None:
            raise RuntimeError("LAE-DINO runtime _run_lae_on_pil was not found. Keep your LAE-DINO V8/V9 ExistingTab plugin active.")
        labels = self._mm_labels(b, ["mustatil", "false_positive"])
        cfg = str(getattr(b, "model_path", "") or "").strip().strip('"')
        weights = str(getattr(b, "formlearner_model", "") or "").strip().strip('"')
        py, repo = self._mm_lae_runtime_paths()
        for prefix in ("mustatil_lae_existing_v9", "mustatil_lae_existing_v8"):
            try:
                setattr(self.ws, prefix + "_python", str(py))
                setattr(self.ws, prefix + "_repo", str(repo))
                setattr(self.ws, prefix + "_classes_text", ", ".join(labels))
                if cfg:
                    setattr(self.ws, prefix + "_config", cfg)
                if weights:
                    setattr(self.ws, prefix + "_weights", weights)
            except Exception:
                pass
        try:
            self.ws.mustatil_lae_python = str(py)
            self.ws.mustatil_lae_repo = str(repo)
            self.ws.mustatil_lae_classes_text = ", ".join(labels)
            if cfg:
                self.ws.mustatil_lae_config = cfg
            if weights:
                self.ws.mustatil_lae_weights = weights
                self.ws.mustatil_lae_checkpoint = weights
            if cfg and weights:
                self.ws.mustatil_lae_force_custom_model = True
                self.ws.mustatil_lae_custom_model_locked = True
        except Exception:
            pass
        local = fn(self.ws, pil_image.convert("RGB"), float(conf))
        out = []
        for r in list(local or []):
            try:
                out.append({
                    "x1": float(r["x1"]), "y1": float(r["y1"]), "x2": float(r["x2"]), "y2": float(r["y2"]),
                    "conf": float(r.get("conf", r.get("confidence", r.get("score", 0.0))) or 0.0),
                    "class_id": int(r.get("class_id", r.get("cls", 0)) or 0),
                    "label": str(r.get("label", r.get("class_name", "lae_dino"))),
                    "model": str(weights or getattr(self.ws, "mustatil_lae_existing_v9_weights", "LAE-DINO")),
                })
            except Exception:
                pass
        return out

    def _run_owlv2_block(self, b: PipelineBlock, full: Image.Image, image_path: Path) -> List[Dict[str, Any]]:
        try:
            return self._mm_open_vocab_block(b, full, image_path, "OWLv2")
        except Exception as exc:
            self._log(f"{b.name}: OWLv2 failed: {exc}")
            return []

    def _run_grounding_dino_block(self, b: PipelineBlock, full: Image.Image, image_path: Path) -> List[Dict[str, Any]]:
        try:
            return self._mm_open_vocab_block(b, full, image_path, "Grounding DINO")
        except Exception as exc:
            self._log(f"{b.name}: Grounding DINO failed: {exc}")
            return []

    def _run_lae_dino_block(self, b: PipelineBlock, full: Image.Image, image_path: Path) -> List[Dict[str, Any]]:
        try:
            return self._mm_open_vocab_block(b, full, image_path, "LAE-DINO")
        except Exception as exc:
            self._log(f"{b.name}: LAE-DINO failed: {exc}")
            return []

    def _run_sam2_block(self, b: PipelineBlock, full: Image.Image, image_path: Path) -> List[Dict[str, Any]]:
        parents = self._input_records(b, full)
        if not parents or b.input_ref == "original":
            self._log(f"{b.name}: SAM2 needs upstream boxes. Connect YOLO/OWLv2/Grounding DINO/LAE-DINO into SAM2.")
            return []
        checkpoint = str(getattr(b, "model_path", "") or "").strip().strip('"')
        config = str(getattr(b, "formlearner_model", "") or "").strip().strip('"')
        device = self._mm_torch_device(getattr(b, "device", "auto cuda"))
        conf = float(getattr(b, "confidence", 0.05) or 0.05)
        try:
            import numpy as _np
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            if not checkpoint or not Path(checkpoint).exists():
                raise RuntimeError("SAM2 checkpoint missing in YOLO model field.")
            if not config or not Path(config).exists():
                raise RuntimeError("SAM2 config missing in FormLearner model field.")
            key = ("sam2", checkpoint, config, device)
            cache = self._mm_cache()
            if key not in cache:
                model = build_sam2(config, checkpoint, device=device)
                predictor = SAM2ImagePredictor(model)
                cache[key] = predictor
                self._log(f"SAM2 loaded: {Path(checkpoint).name} on {device}")
            predictor = cache[key]
            out: List[Dict[str, Any]] = []
            for parent in parents:
                crop, offset = self._crop_parent(full, parent, int(getattr(b, "crop_padding", 0) or 0))
                crop = crop.convert("RGB")
                predictor.set_image(_np.asarray(crop))
                box = _np.array([0, 0, max(1, crop.width - 1), max(1, crop.height - 1)], dtype=_np.float32)
                masks, scores, _logits = predictor.predict(box=box, multimask_output=False)
                if masks is None or len(masks) == 0:
                    continue
                mask = masks[0]
                ys, xs = _np.where(mask > 0)
                if len(xs) == 0 or len(ys) == 0:
                    continue
                x1, x2 = float(xs.min() + offset[0]), float(xs.max() + 1 + offset[0])
                y1, y2 = float(ys.min() + offset[1]), float(ys.max() + 1 + offset[1])
                score = float(scores[0]) if scores is not None and len(scores) else float(parent.get("conf", conf) or conf)
                rec = self._mm_make_record(
                    b, parent, full, image_path, x1, y1, x2, y2, score,
                    int(parent.get("class_id", 0) or 0),
                    str(parent.get("label", "sam2_mask")),
                    str(checkpoint),
                    {"sam2_status": "mask_refined", "mask_area_px": int(mask.sum())},
                )
                if rec:
                    out.append(rec)
            return out
        except Exception as exc:
            self._log(f"{b.name}: SAM2 unavailable or not configured; passing parent boxes through. Reason: {exc}")
            out = []
            for parent in parents:
                rec = dict(parent)
                rec.update({
                    "id": "r_" + uuid.uuid4().hex[:10],
                    "block_id": b.id, "block_name": b.name, "block_type": b.type,
                    "parent_id": parent.get("id"),
                    "sam2_status": "fallback_parent_box",
                    "conf": float(parent.get("conf", conf) or conf),
                    "label": str(parent.get("label", "sam2_parent")),
                    "model": str(checkpoint or "SAM2 fallback"),
                    "source_image": str(image_path),
                })
                out.append(rec)
            return out

    def _run_formlearner_block(self, b: PipelineBlock, full: Image.Image) -> List[Dict[str, Any]]:
        parents = self._input_records(b, full)
        model_path = Path((b.formlearner_model or b.model_path).strip().strip('"')).expanduser()
        if not model_path.is_file():
            raise RuntimeError(f"FormLearner block '{b.name}' needs a valid .json model.")
        import mustatil_legacy_backend as backend
        form_model = backend.SimpleFormLearner.load(str(model_path))
        out = []
        for parent in parents:
            bbox = (float(parent["x1"]), float(parent["y1"]), float(parent["x2"]), float(parent["y2"]))
            score = float(form_model.predict(backend.crop_features(full, bbox)))
            rec = dict(parent)
            rec.update({
                "id": "r_" + uuid.uuid4().hex[:10],
                "block_id": b.id, "block_name": b.name, "block_type": b.type,
                "parent_id": parent.get("id"), "form_score": score,
                "form_threshold": float(b.form_threshold),
                "form_status": "positive" if score >= float(b.form_threshold) else "false_positive",
                "label": parent.get("label", "object") if score >= float(b.form_threshold) else "false_positive",
                "class_id": parent.get("class_id", 0) if score >= float(b.form_threshold) else 1,
            })
            out.append(rec)
        return out

    def _normalize_rule_conditions(self, rule: Dict[str, Any], b: Optional[PipelineBlock] = None) -> List[Dict[str, Any]]:
        """Return rule conditions, accepting both new multi-input and old single-input syntax."""
        raw = rule.get("conditions")
        conditions: List[Dict[str, Any]] = []
        if isinstance(raw, list):
            for c in raw:
                if isinstance(c, dict):
                    cc = dict(c)
                    if cc.get("source"):
                        conditions.append(cc)
        elif isinstance(raw, dict):
            if raw.get("source"):
                conditions.append(dict(raw))
        # Backwards compatibility with the first plugin version.
        if not conditions and (rule.get("source") or (b and b.input_ref != "original")):
            conditions.append({
                "source": rule.get("source") or (b.input_ref if b else ""),
                "count_min": rule.get("count_min", 1),
                "count_max": rule.get("count_max", None),
            })
        return conditions

    def _block_dependency_refs(self, b: PipelineBlock) -> List[str]:
        """All upstream block ids used by a block, including multi-input rule conditions."""
        refs: List[str] = []
        def add(ref: Any):
            rid = self._resolve_block_ref(ref)
            if rid and rid != "original" and rid != b.id and rid not in refs:
                refs.append(rid)
        add(b.input_ref)
        if b.type == "rule":
            try:
                rule = json.loads(b.rule_json or "{}")
            except Exception:
                rule = {}
            add(rule.get("within_parent") or rule.get("parent"))
            for cond in self._normalize_rule_conditions(rule, b):
                add(cond.get("source"))
                add(cond.get("within_parent") or cond.get("parent"))
        return refs

    def _rule_to_text(self, rule: Dict[str, Any]) -> str:
        return json.dumps(rule, indent=2, ensure_ascii=False)

    def _current_rule_dict(self, b: PipelineBlock) -> Dict[str, Any]:
        try:
            rule = json.loads(b.rule_json or "{}")
            if not isinstance(rule, dict):
                rule = {}
        except Exception:
            rule = {}
        rule.setdefault("label", self._safe_label(b.name.replace("Rule:", "").strip() or "classification"))
        rule.setdefault("logic", "AND")
        rule["then_action"] = self._normalize_then_action(rule.get("then_action", rule.get("action", "classify")))
        rule.setdefault("then_value", rule.get("label", "classification"))
        if not rule.get("within_parent") and b.input_ref and b.input_ref != "original":
            rule["within_parent"] = self.block_label(b.input_ref)
        rule["conditions"] = self._normalize_rule_conditions(rule, b)
        # Remove old single-condition fields to avoid ambiguity once the new editor is used.
        for k in ("source", "count_min", "count_max"):
            rule.pop(k, None)
        return rule

    def _rule_block_inline_controls_changed(self, rule_block: PipelineBlock, item: VisualBlockItem):
        """Update rule JSON from the dropdowns embedded directly in the visual rule block."""
        if not rule_block or rule_block.type != "rule":
            return
        try:
            rule = self._current_rule_dict(rule_block)
            gate = item._rule_gate_combo.currentText().strip().upper() if hasattr(item, "_rule_gate_combo") else "AND"
            rule["logic"] = self._normalize_logic_gate(gate)
            action = self._normalize_then_action(item._rule_then_action_combo.currentText() if hasattr(item, "_rule_then_action_combo") else "classify")
            rule["then_action"] = action
            label = item._rule_label_edit.text().strip() if hasattr(item, "_rule_label_edit") else ""
            if label:
                rule["then_value"] = label
                if action in {"classify", "label", "flag true", "number", "tag"}:
                    rule["label"] = self._safe_label(label if action != "number" else f"number_{label}")
            conditions = []
            source_combos = list(getattr(item, "_rule_source_combos", []))
            op_combos = list(getattr(item, "_rule_op_combos", []))
            min_spins = list(getattr(item, "_rule_min_spins", []))
            for idx, combo in enumerate(source_combos):
                src_id = combo.currentData() or ""
                if not src_id:
                    continue
                spin = min_spins[idx] if idx < len(min_spins) else None
                op_combo = op_combos[idx] if idx < len(op_combos) else None
                op = self._normalize_rule_operator(op_combo.currentText() if op_combo is not None else ">=")
                value = int(spin.value()) if spin is not None else 1
                cond = {
                    "source": self.block_label(src_id),
                    "operator": op,
                    "count_min": value,
                    "count_max": None,
                }
                if op == "<=":
                    cond["count_max"] = value
                conditions.append(cond)
            if conditions:
                rule["conditions"] = conditions
            else:
                rule["conditions"] = []
            rule_block.rule_json = self._rule_to_text(rule)
            rule_block.name = "Rule: " + str(rule.get("label", rule.get("then_value", "classification")))
            if self._current_block() and self._current_block().id == rule_block.id:
                old = self._updating
                self._updating = True
                try:
                    self.name.setText(rule_block.name)
                    self.rule_json.setPlainText(rule_block.rule_json)
                finally:
                    self._updating = old
            item.update()
            self.refresh_arrows()
            self.refresh_code_from_blocks()
        except Exception as exc:
            self._log(f"Could not update inline rule block controls: {exc}")

    def add_rule_condition_from_source(self, rule_block: PipelineBlock, source_id: str, interactive: bool = True):
        """Append a source/count condition to a rule block."""
        if rule_block.type != "rule":
            return
        source_id = self._resolve_block_ref(source_id) or source_id
        if not source_id or source_id == rule_block.id:
            return
        count_min = 1
        count_max = None
        if interactive:
            count_min, ok = QInputDialog.getInt(self, "Add rule input", "Minimum count for this input:", 1, 0, 9999, 1)
            if not ok:
                return
        rule = self._current_rule_dict(rule_block)
        conds = list(rule.get("conditions") or [])
        # If the same source already exists, update its minimum instead of duplicating it.
        for c in conds:
            if self._resolve_block_ref(c.get("source")) == source_id:
                c["operator"] = ">="
                c["count_min"] = int(count_min)
                c["count_max"] = count_max
                break
        else:
            conds.append({"source": self.block_label(source_id), "operator": ">=", "count_min": int(count_min), "count_max": count_max})
        rule["conditions"] = conds
        if not rule.get("within_parent") and rule_block.input_ref and rule_block.input_ref != "original":
            rule["within_parent"] = self.block_label(rule_block.input_ref)
        rule_block.rule_json = self._rule_to_text(rule)
        if self._current_block() and self._current_block().id == rule_block.id:
            old = self._updating
            self._updating = True
            self.rule_json.setPlainText(rule_block.rule_json)
            self._updating = old
        if rule_block.id in self.block_items:
            self.block_items[rule_block.id].refresh_embedded_controls()
        self.refresh_arrows()
        self.refresh_code_from_blocks()

    def add_selected_input_as_rule_condition(self):
        b = self._current_block()
        if not b or b.type != "rule":
            QMessageBox.information(self, "YOLO Pipeline", "Select a Rule block first.")
            return
        source_id = self.input_ref.currentData() or ""
        if not source_id or source_id == "original":
            QMessageBox.information(self, "YOLO Pipeline", "Choose another block in the Input socket dropdown first.")
            return
        self.add_rule_condition_from_source(b, source_id, interactive=True)
        self._load_block_to_editor(b)
        self._log(f"Added logic-gate input to {b.name}: {self.block_label(source_id)}")

    def edit_rule_conditions_dialog(self):
        b = self._current_block()
        if not b or b.type != "rule":
            QMessageBox.information(self, "YOLO Pipeline", "Select a Rule block first.")
            return
        rule = self._current_rule_dict(b)
        dlg = QDialog(self)
        dlg.setWindowTitle("Edit IF / logic-gate rule")
        lay = QVBoxLayout(dlg)
        form = QFormLayout()
        then_action_combo = QComboBox(); then_action_combo.addItems(["classify", "number", "flag true", "label", "score", "tag", "keep parent", "discard/filter"])
        then_action_combo.setCurrentText(self._normalize_then_action(rule.get("then_action", rule.get("action", "classify"))))
        label_edit = QLineEdit(str(rule.get("then_value", rule.get("label", "classification"))))
        logic_combo = QComboBox(); logic_combo.addItems(["AND", "OR", "NAND", "NOR", "XOR", "NOT"])
        _logic_raw = str(rule.get("logic", "AND")).strip().upper()
        _logic_alias = {"ALL": "AND", "ANY": "OR"}.get(_logic_raw, _logic_raw)
        logic_combo.setCurrentText(_logic_alias if _logic_alias in {"AND", "OR", "NAND", "NOR", "XOR", "NOT"} else "AND")
        parent_combo = QComboBox(); parent_combo.addItem("Use block input / none", "")
        for x in self.blocks:
            if x.id != b.id:
                parent_combo.addItem(f"{x.name} ({x.type})", x.id)
        parent_id = self._resolve_block_ref(rule.get("within_parent") or rule.get("parent") or b.input_ref)
        ix = parent_combo.findData(parent_id or "")
        parent_combo.setCurrentIndex(ix if ix >= 0 else 0)
        form.addRow("THEN action", then_action_combo)
        form.addRow("THEN value", label_edit)
        form.addRow("IF logic gate", logic_combo)
        form.addRow("Classify parent", parent_combo)
        lay.addLayout(form)

        cond_list = QListWidget()
        lay.addWidget(QLabel("IF inputs / conditions"))
        lay.addWidget(cond_list, 1)

        def refresh_list():
            cond_list.clear()
            for c in rule.get("conditions", []):
                src = self._resolve_block_ref(c.get("source")) or str(c.get("source", ""))
                name = self.block_label(src) if src else str(c.get("source", ""))
                cmin = int(c.get("count_min", c.get("value", c.get("min", 1))) or 0)
                op = self._normalize_rule_operator(c.get("operator", c.get("op", ">=")))
                cmax = c.get("count_max", c.get("max", None))
                if op == "<=":
                    txt = f"{name}: count <= {cmin}"
                elif op == "==":
                    txt = f"{name}: count == {cmin}"
                else:
                    max_txt = "∞" if cmax in (None, "", "null") else str(cmax)
                    txt = f"{name}: count >= {cmin}" + ("" if max_txt == "∞" else f", <= {max_txt}")
                cond_list.addItem(txt)

        refresh_list()

        add_row = QHBoxLayout()
        source_combo = QComboBox()
        for x in self.blocks:
            if x.id != b.id:
                source_combo.addItem(f"{x.name} ({x.type})", x.id)
        op_combo = QComboBox(); op_combo.addItems([">=", "<=", "=="])
        min_spin = QSpinBox(); min_spin.setRange(0, 9999); min_spin.setValue(1)
        max_spin = QSpinBox(); max_spin.setRange(-1, 9999); max_spin.setValue(-1); max_spin.setSpecialValueText("none")
        add_btn = QPushButton("Add / update")
        rem_btn = QPushButton("Remove selected")
        add_row.addWidget(QLabel("Input")); add_row.addWidget(source_combo, 1)
        add_row.addWidget(QLabel("op")); add_row.addWidget(op_combo)
        add_row.addWidget(QLabel("value")); add_row.addWidget(min_spin)
        add_row.addWidget(QLabel("max")); add_row.addWidget(max_spin)
        add_row.addWidget(add_btn); add_row.addWidget(rem_btn)
        lay.addLayout(add_row)

        def add_update():
            sid = source_combo.currentData()
            if not sid:
                return
            conds = list(rule.get("conditions") or [])
            for c in conds:
                if self._resolve_block_ref(c.get("source")) == sid:
                    op = self._normalize_rule_operator(op_combo.currentText())
                    c["operator"] = op
                    c["count_min"] = int(min_spin.value())
                    c["count_max"] = int(min_spin.value()) if op == "<=" else (None if int(max_spin.value()) < 0 else int(max_spin.value()))
                    break
            else:
                op = self._normalize_rule_operator(op_combo.currentText())
                conds.append({"source": self.block_label(sid), "operator": op, "count_min": int(min_spin.value()), "count_max": int(min_spin.value()) if op == "<=" else (None if int(max_spin.value()) < 0 else int(max_spin.value()))})
            rule["conditions"] = conds
            refresh_list()

        def remove_selected():
            row = cond_list.currentRow()
            if row >= 0:
                conds = list(rule.get("conditions") or [])
                if row < len(conds):
                    del conds[row]
                    rule["conditions"] = conds
                    refresh_list()

        add_btn.clicked.connect(add_update)
        rem_btn.clicked.connect(remove_selected)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept); buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        if dlg.exec() != QDialog.Accepted:
            return
        rule["then_action"] = self._normalize_then_action(then_action_combo.currentText())
        rule["then_value"] = label_edit.text().strip() or "classification"
        if rule["then_action"] in {"classify", "label", "flag true", "number", "tag"}:
            rule["label"] = self._safe_label(rule["then_value"] if rule["then_action"] != "number" else f"number_{rule['then_value']}")
        else:
            rule["label"] = self._safe_label(label_edit.text().strip() or "classification")
        rule["logic"] = logic_combo.currentText().upper()
        pid = parent_combo.currentData()
        if pid:
            rule["within_parent"] = self.block_label(pid)
            b.input_ref = pid
        else:
            rule.pop("within_parent", None)
        b.name = "Rule: " + rule["label"]
        b.rule_json = self._rule_to_text(rule)
        self._load_block_to_editor(b)
        self.refresh_arrows()
        self.refresh_code_from_blocks()
        self._log("Updated IF / logic-gate rule block: " + b.name)


    def _normalize_then_action(self, raw: Any) -> str:
        """Normalize the THEN action selected in IF/logic blocks."""
        text = str(raw or "classify").strip().lower().replace("_", " ").replace("-", " ")
        aliases = {
            "class": "classify", "classification": "classify", "klassifizieren": "classify",
            "num": "number", "count": "number", "zahl": "number",
            "true": "flag true", "flag": "flag true", "boolean": "flag true", "bool": "flag true",
            "set label": "label", "rename": "label",
            "set score": "score", "confidence": "score",
            "mark": "tag",
            "keep": "keep parent", "keep_parent": "keep parent",
            "discard": "discard/filter", "filter": "discard/filter", "remove": "discard/filter",
        }
        text = aliases.get(text, text)
        valid = {"classify", "number", "flag true", "label", "score", "tag", "keep parent", "discard/filter"}
        return text if text in valid else "classify"

    def _apply_then_action_to_record(self, rec: Dict[str, Any], rule: Dict[str, Any], parent: Dict[str, Any], label: str) -> Optional[Dict[str, Any]]:
        """Apply the IF block THEN action to the output record.

        Actions:
        - classify: sets label/class_name and keeps class_id if provided
        - number: stores numeric/text value in rule_value and labels it as number_<value>
        - flag true: stores a boolean true marker and uses the value as label
        - label: renames the output label without forcing class semantics
        - score: writes rule_score when the value is numeric
        - tag: appends/sets a tag while preserving the parent label unless no tag exists
        - keep parent: emits the parent unchanged apart from rule metadata
        - discard/filter: emits no detection even when the IF condition is true
        """
        action = self._normalize_then_action(rule.get("then_action", rule.get("action", "classify")))
        value = str(rule.get("then_value", rule.get("value", label)) or label).strip()
        if action == "discard/filter":
            return None
        if action == "keep parent":
            rec["label"] = parent.get("label", rec.get("label", label))
            rec["then_action"] = action
            rec["then_value"] = value
            return rec
        if action == "number":
            rec["then_action"] = action
            rec["then_value"] = value
            try:
                rec["rule_value"] = float(value.replace(",", "."))
            except Exception:
                rec["rule_value"] = value
            rec["label"] = self._safe_label(f"number_{value}" if value else "number")
            return rec
        if action == "flag true":
            rec["then_action"] = action
            rec["then_value"] = value or "true"
            rec["rule_flag"] = True
            rec["label"] = self._safe_label(value or "true")
            return rec
        if action == "score":
            rec["then_action"] = action
            rec["then_value"] = value
            try:
                rec["rule_score"] = float(value.replace(",", "."))
            except Exception:
                rec["rule_score"] = value
            rec["label"] = self._safe_label(label)
            return rec
        if action == "tag":
            rec["then_action"] = action
            rec["then_value"] = value
            rec["tag"] = value or label
            rec["label"] = parent.get("label", rec.get("label", label))
            return rec
        # classify and label both produce a visible class/label output.
        rec["then_action"] = action
        rec["then_value"] = value or label
        rec["label"] = self._safe_label(value or label)
        if action == "classify":
            rec["class_name"] = rec["label"]
        return rec

    def _normalize_rule_operator(self, raw: Any) -> str:
        """Normalize comparison operator for an IF condition count."""
        text = str(raw or ">=").strip().lower()
        aliases = {
            "≥": ">=", "gte": ">=", "ge": ">=", "greater_equal": ">=", "greater_or_equal": ">=", "min": ">=", "at_least": ">=",
            "<=": "<=", "≤": "<=", "lte": "<=", "le": "<=", "less_equal": "<=", "less_or_equal": "<=", "max": "<=", "at_most": "<=",
            "=": "==", "==": "==", "eq": "==", "equal": "==", "equals": "==", "exactly": "==",
        }
        return aliases.get(text, ">=")

    def _evaluate_count_condition(self, count: int, cond: Dict[str, Any]) -> bool:
        """Evaluate one IF-condition count using >=, <= or ==."""
        n = int(count)
        op = self._normalize_rule_operator(cond.get("operator", cond.get("op", ">=")))
        value = int(cond.get("count_min", cond.get("value", cond.get("min", 1))) or 0)
        if op == "<=":
            return n <= value
        if op == "==":
            return n == value
        # Backward compatible >= with optional max range.
        cmax = cond.get("count_max", cond.get("max", None))
        cmax = None if cmax in (None, "", "null") else int(cmax)
        return n >= value and (cmax is None or n <= cmax)

    def _normalize_logic_gate(self, raw: Any) -> str:
        """Normalize user-facing rule logic to one gate name."""
        text = str(raw or "AND").strip().upper().replace(" ", "_")
        aliases = {
            "ALL": "AND",
            "ANY": "OR",
            "&&": "AND",
            "&": "AND",
            "UND": "AND",
            "ODER": "OR",
            "||": "OR",
            "!AND": "NAND",
            "NOT_AND": "NAND",
            "!OR": "NOR",
            "NOT_OR": "NOR",
            "EXCLUSIVE_OR": "XOR",
            "EITHER_OR": "XOR",
        }
        text = aliases.get(text, text)
        return text if text in {"AND", "OR", "NAND", "NOR", "XOR", "NOT"} else "AND"

    def _evaluate_logic_gate(self, logic: str, values: List[bool]) -> bool:
        """Evaluate IF/logic-gate rule inputs. Empty inputs always fail."""
        vals = [bool(v) for v in values]
        if not vals:
            return False
        logic = self._normalize_logic_gate(logic)
        if logic == "AND":
            return all(vals)
        if logic == "OR":
            return any(vals)
        if logic == "NAND":
            return not all(vals)
        if logic == "NOR":
            return not any(vals)
        if logic == "XOR":
            return sum(1 for v in vals if v) == 1
        if logic == "NOT":
            # For NOT, use the first input as the condition to invert.
            return not vals[0]
        return all(vals)

    def _run_rule_block(self, b: PipelineBlock) -> List[Dict[str, Any]]:
        """Run a classification/count rule.

        Supported schemas:
        Legacy single-input rule:
        {
          "label": "two_engine_plane",
          "source": "YOLO 2: engines",
          "within_parent": "YOLO 1: airplanes",
          "count_min": 2,
          "count_max": null
        }

        IF / logic-gate rule:
        {
          "label": "complex_plane",
          "logic": "AND",   // AND, OR, NAND, NOR, XOR, NOT
          "within_parent": "YOLO 1: airplanes",
          "conditions": [
            {"source": "Rule 1", "operator": ">=", "count_min": 1},
            {"source": "Rule 2", "operator": "==", "count_min": 2}
          ]
        }
        """
        try:
            rule = json.loads(b.rule_json or "{}")
        except Exception as exc:
            raise RuntimeError(f"Rule block '{b.name}' has invalid JSON: {exc}")

        label = str(rule.get("then_value", rule.get("label") or b.name))
        logic = self._normalize_logic_gate(rule.get("logic") or "AND")

        parent_ref = rule.get("within_parent") or rule.get("parent") or (b.input_ref if b.input_ref != "original" else "")
        parent_id = self._resolve_block_ref(parent_ref)
        conditions = self._normalize_rule_conditions(rule, b)
        if not conditions:
            raise RuntimeError(f"Rule block '{b.name}' needs at least one condition/source.")

        # If no explicit parent was given, use the first condition's within_parent,
        # otherwise fall back to the first enabled block. This keeps old/simple rules usable.
        if not parent_id:
            for c in conditions:
                parent_id = self._resolve_block_ref(c.get("within_parent") or c.get("parent"))
                if parent_id:
                    break
        if not parent_id and self.blocks:
            parent_id = self.blocks[0].id
        if not parent_id:
            raise RuntimeError(f"Rule block '{b.name}' needs within_parent/parent or an input block.")

        parents = self.results_by_block.get(parent_id, [])
        if not parents:
            return []

        out = []
        for parent in parents:
            condition_details = []
            all_child_ids: List[Any] = []
            ok_values: List[bool] = []
            for cond in conditions:
                source_id = self._resolve_block_ref(cond.get("source"))
                if not source_id:
                    ok_values.append(False)
                    condition_details.append({"source": "", "count": 0, "ok": False})
                    continue
                children = self.results_by_block.get(source_id, [])
                # Count records whose center is inside the current parent. Rule-output
                # records copy the parent bbox, so combined rules can count earlier rule results too.
                inside = [c for c in children if self._center_inside(c, parent)]
                n = len(inside)
                op = self._normalize_rule_operator(cond.get("operator", cond.get("op", ">=")))
                cmin = int(cond.get("count_min", cond.get("value", cond.get("min", 1))) or 0)
                cmax = cond.get("count_max", cond.get("max", None))
                cmax = None if cmax in (None, "", "null") else int(cmax)
                ok = self._evaluate_count_condition(n, cond)
                ok_values.append(ok)
                ids = [c.get("id") for c in inside]
                all_child_ids.extend(ids)
                condition_details.append({
                    "source": self.block_label(source_id),
                    "source_id": source_id,
                    "count": int(n),
                    "operator": op,
                    "count_min": int(cmin),
                    "count_max": cmax,
                    "ok": bool(ok),
                    "child_ids": ids,
                })

            passed = self._evaluate_logic_gate(logic, ok_values)
            if passed:
                rec = dict(parent)
                rec.update({
                    "id": "r_" + uuid.uuid4().hex[:10],
                    "block_id": b.id, "block_name": b.name, "block_type": b.type,
                    "parent_id": parent.get("id"), "label": label,
                    "class_id": int(rule.get("class_id", parent.get("class_id", 0) or 0)),
                    "rule_count": int(sum(int(d.get("count", 0)) for d in condition_details)),
                    "rule_logic": logic,
                    "rule_parent_block_id": parent_id,
                    "rule_conditions": condition_details,
                    "child_ids": all_child_ids,
                })
                rec = self._apply_then_action_to_record(rec, rule, parent, label)
                if rec is not None:
                    out.append(rec)
        return out

    def _resolve_block_ref(self, ref: Any) -> Optional[str]:
        ref = str(ref or "").strip()
        if not ref:
            return None
        for b in self.blocks:
            if ref == b.id or ref.lower() == b.name.lower():
                return b.id
        return ref

    def _crop_parent(self, full: Image.Image, rec: Dict[str, Any], pad: int) -> Tuple[Image.Image, Tuple[float, float]]:
        x1 = max(0, int(float(rec.get("x1", 0)) - pad)); y1 = max(0, int(float(rec.get("y1", 0)) - pad))
        x2 = min(full.width, int(float(rec.get("x2", full.width)) + pad)); y2 = min(full.height, int(float(rec.get("y2", full.height)) + pad))
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, full.width, full.height
        return full.crop((x1, y1, x2, y2)), (float(x1), float(y1))

    def _center_inside(self, child: Dict[str, Any], parent: Dict[str, Any]) -> bool:
        cx = (float(child.get("x1", 0)) + float(child.get("x2", 0))) / 2.0
        cy = (float(child.get("y1", 0)) + float(child.get("y2", 0))) / 2.0
        return float(parent.get("x1", 0)) <= cx <= float(parent.get("x2", 0)) and float(parent.get("y1", 0)) <= cy <= float(parent.get("y2", 0))

    def _parse_class_filter(self, raw: str) -> set:
        return {x.strip().lower() for x in str(raw or "").split(",") if x.strip()}

    # ---------- Rule helper / output / integration ----------
    def _safe_label(self, text: str) -> str:
        import re
        text = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(text or "classification").strip())
        text = re.sub(r"_+", "_", text).strip("_")
        return text or "classification"

    def _guess_block_by_words(self, words: List[str]) -> Optional[str]:
        words = [w.lower() for w in words]
        for b in self.blocks:
            hay = (b.name + " " + b.classes_filter + " " + b.rule_json).lower()
            if any(w in hay for w in words):
                return b.id
        return None

    def _guess_block_from_text(self, low_prompt: str, kind: str) -> Optional[str]:
        # Explicit block references by name/id always win.
        for b in self.blocks:
            if b.id.lower() in low_prompt or b.name.lower() in low_prompt:
                return b.id
        if kind == "source":
            return self._guess_block_by_words(["engine", "engines", "motor", "motoren", "triebwerk", "triebwerke", "wing", "wings", "flügel", "fluegel"])
        return self._guess_block_by_words(["airplane", "airplanes", "plane", "planes", "aircraft", "flugzeug", "flugzeuge"])

    def _is_labeled_or_classified_record(self, rec: Dict[str, Any]) -> bool:
        """Return True for records that represent a named/classified output."""
        label = str(rec.get("label", rec.get("class_name", "")) or "").strip()
        if not label:
            return False
        if label.lower() in {"object", "detection", "unknown"}:
            # Keep plain object detections only when they also carry explicit class metadata.
            return bool(rec.get("class_name") or rec.get("then_action") == "classify")
        return True

    def _choose_result_subset(self, title: str, include_full_gpkg_option: bool = False) -> Optional[List[Dict[str, Any]]]:
        if not self.results:
            QMessageBox.information(self, "YOLO Pipeline", "No pipeline results available yet. Run the pipeline first.")
            return None
        self._export_all_information = False
        options: List[Tuple[str, List[Dict[str, Any]], bool]] = []
        options.append((f"All pipeline results ({len(self.results)})", list(self.results), False))

        if include_full_gpkg_option:
            labeled = [r for r in self.results if self._is_labeled_or_classified_record(r)]
            if labeled:
                options.append((
                    f"All labelled/classified with classes + all information ({len(labeled)})",
                    labeled,
                    True,
                ))

        # Block/output choices.
        for b in self.blocks:
            recs = list(self.results_by_block.get(b.id, []))
            if recs:
                options.append((f"Output: {b.name} ({len(recs)})", recs, False))

        # Class/label choices.
        labels = sorted({str(r.get("label", "object")) for r in self.results})
        for lab in labels:
            recs = [r for r in self.results if str(r.get("label", "object")) == lab]
            options.append((f"Class/label: {lab} ({len(recs)})", recs, False))

        labels_only = [x[0] for x in options]
        choice, ok = QInputDialog.getItem(self, title, "Export/send which result set?", labels_only, 0, False)
        if not ok:
            return None
        for label, recs, full_info in options:
            if label == choice:
                self._export_all_information = bool(full_info)
                return recs
        return None

    def send_results_to_detection_tab(self):
        recs = self._choose_result_subset("Send to Detection tab")
        if recs is None:
            return
        dets = self._records_to_dets(recs)
        self.ws.dets = dets
        try:
            self.ws.redraw()
            self.ws.refresh_layers()
        except Exception:
            pass
        self._log(f"Sent {len(dets)} selected pipeline record(s) to the Detection tab overlay.")

    def _records_to_dets(self, recs: List[Dict[str, Any]]):
        dets = []
        try:
            from mustatil_legacy_backend import Det
            for r in recs:
                dets.append(Det(
                    0,
                    r.get("block_name", "pipeline"),
                    int(r.get("class_id", 0) or 0),
                    float(r.get("conf", r.get("form_score", 1.0)) or 0),
                    float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]),
                ))
        except Exception:
            dets = list(recs)
        return dets

    def export_results_gpkg(self):
        recs = self._choose_result_subset("Export YOLO pipeline results", include_full_gpkg_option=True)
        if recs is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export selected YOLO pipeline results",
            "yolo_pipeline_results.gpkg",
            "GeoPackage (*.gpkg);;All files (*)",
        )
        if not path:
            return
        path = str(Path(path).with_suffix(".gpkg"))
        try:
            self._export_records_to_gpkg(recs, path, include_all_fields=bool(getattr(self, "_export_all_information", False)))
        except Exception as exc:
            QMessageBox.critical(self, "YOLO Pipeline export", f"GeoPackage export failed:\n{exc}")
            return
        self._log(f"Exported {len(recs)} selected pipeline record(s) to GeoPackage: {path}")

    def _export_records_to_gpkg(self, recs: List[Dict[str, Any]], path: str, include_all_fields: bool = False):
        """Write selected pipeline records as a real GeoPackage without GDAL/OGR.

        This mirrors the Detection-tab goal: write one self-contained .gpkg that
        QGIS can open directly. It uses only Python's sqlite3 and writes standard
        GeoPackage polygon geometries. When the current image has georeferencing,
        pixel boxes are converted through the workspace _px_to_map() transform;
        otherwise the layer is still valid in local pixel coordinates.
        """
        if not recs:
            raise RuntimeError("No records selected for export.")

        import sqlite3
        import struct
        import datetime
        import re

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            out_path.unlink()

        use_map = False
        srs_id = 0
        srs_name = "Undefined Cartesian SRS"
        srs_org = "NONE"
        srs_org_id = 0
        srs_definition = "undefined"
        try:
            if hasattr(self.ws, "_ensure_geo_from_current_image"):
                self.ws._ensure_geo_from_current_image()
            crs_name = str(self.ws._feature_crs_name() if hasattr(self.ws, "_feature_crs_name") else "")
            crs_wkt = str(self.ws._feature_crs_wkt() if hasattr(self.ws, "_feature_crs_wkt") else "")
            if "3857" in crs_name or "Pseudo-Mercator" in crs_wkt or "Pseudo_Mercator" in crs_wkt:
                srs_id = 3857
                srs_name = "WGS 84 / Pseudo-Mercator"
                srs_org = "EPSG"
                srs_org_id = 3857
                srs_definition = crs_wkt or 'PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Mercator_1SP"],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1],AUTHORITY["EPSG","3857"]]'
            elif "4326" in crs_name or "WGS 84" in crs_wkt:
                srs_id = 4326
                srs_name = "WGS 84 geodetic"
                srs_org = "EPSG"
                srs_org_id = 4326
                srs_definition = crs_wkt or 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]]'
            elif crs_wkt:
                # Keep unknown WKT as a user-defined SRS. QGIS reads this from GPKG.
                srs_id = 100000
                srs_name = crs_name or "User-defined image CRS"
                srs_org = "USER"
                srs_org_id = 100000
                srs_definition = crs_wkt
            use_map = hasattr(self.ws, "_px_to_map")
        except Exception:
            use_map = False

        base_fields: List[Tuple[str, str]] = [
            ("fid", "INTEGER PRIMARY KEY AUTOINCREMENT"),
            ("geom", "BLOB NOT NULL"),
            ("label", "TEXT"),
            ("class_id", "INTEGER"),
            ("score", "REAL"),
            ("block", "TEXT"),
            ("block_type", "TEXT"),
            ("model", "TEXT"),
            ("form_score", "REAL"),
            ("rule_count", "INTEGER"),
            ("parent_id", "TEXT"),
            ("child_ids", "TEXT"),
            ("rule_conditions", "TEXT"),
        ]
        used = {name for name, _typ in base_fields}
        dynamic_field_map: Dict[str, Tuple[str, str]] = {}

        if include_all_fields:
            extra_names = [
                "class_name", "then_action", "then_value", "tag", "rule_value", "rule_flag",
                "rule_score", "block_id", "id", "input_ref", "source", "classes_filter", "all_info",
            ]
            for rec in recs:
                for key, val in rec.items():
                    if isinstance(val, (str, int, float, bool)) or val is None:
                        extra_names.append(str(key))
            for raw_name in extra_names:
                raw_name = str(raw_name or "field")
                safe = re.sub(r"[^A-Za-z0-9_]+", "_", raw_name).strip("_") or "field"
                if safe[0].isdigit():
                    safe = "f_" + safe
                safe = safe[:55]
                base = safe
                i = 2
                while safe in used:
                    safe = f"{base[:50]}_{i}"
                    i += 1
                used.add(safe)
                vals = [r.get(raw_name) for r in recs if raw_name in r and r.get(raw_name) is not None]
                if vals and all(isinstance(v, bool) for v in vals):
                    sql_typ = "INTEGER"
                elif vals and all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
                    sql_typ = "INTEGER"
                elif vals and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
                    sql_typ = "REAL"
                else:
                    sql_typ = "TEXT"
                dynamic_field_map[raw_name] = (safe, sql_typ)

        all_fields: List[Tuple[str, str]] = list(base_fields)
        for safe, sql_typ in dynamic_field_map.values():
            if safe not in {n for n, _t in all_fields}:
                all_fields.append((safe, sql_typ))

        def quote_ident(name: str) -> str:
            return '"' + str(name).replace('"', '""') + '"'

        def px_to_xy(x, y):
            if use_map:
                try:
                    return self.ws._px_to_map(float(x), float(y))
                except Exception:
                    pass
            return float(x), float(y)

        def gpkg_polygon_blob(points, gpkg_srs_id: int) -> bytes:
            # GeoPackageBinaryHeader: magic GP, version 0, flags=1 (little endian, no envelope), srs_id.
            header = b"GP" + bytes([0, 1]) + struct.pack("<i", int(gpkg_srs_id))
            # Standard little-endian WKB Polygon: byte order, type, ring count, point count, xy pairs.
            wkb = bytearray()
            wkb.extend(struct.pack("<BII", 1, 3, 1))
            wkb.extend(struct.pack("<I", len(points)))
            for x, y in points:
                wkb.extend(struct.pack("<dd", float(x), float(y)))
            return header + bytes(wkb)

        rows = []
        minx = miny = maxx = maxy = None
        for r in recs:
            if not all(k in r for k in ("x1", "y1", "x2", "y2")):
                continue
            x1, y1, x2, y2 = float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])
            pts = [px_to_xy(x1, y1), px_to_xy(x2, y1), px_to_xy(x2, y2), px_to_xy(x1, y2), px_to_xy(x1, y1)]
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            minx = min(xs) if minx is None else min(minx, min(xs))
            miny = min(ys) if miny is None else min(miny, min(ys))
            maxx = max(xs) if maxx is None else max(maxx, max(xs))
            maxy = max(ys) if maxy is None else max(maxy, max(ys))
            row = {
                "geom": gpkg_polygon_blob(pts, srs_id),
                "label": str(r.get("label", "object")),
                "class_id": int(r.get("class_id", 0) or 0),
                "score": float(r.get("conf", r.get("form_score", 1.0)) or 0.0),
                "block": str(r.get("block_name", "")),
                "block_type": str(r.get("block_type", "")),
                "model": str(r.get("model", "")),
                "form_score": None if r.get("form_score") is None else float(r.get("form_score") or 0.0),
                "rule_count": None if r.get("rule_count") is None else int(r.get("rule_count") or 0),
                "parent_id": str(r.get("parent_id", "")),
                "child_ids": json.dumps(r.get("child_ids", []), ensure_ascii=False, default=str),
                "rule_conditions": json.dumps(r.get("rule_conditions", []), ensure_ascii=False, default=str),
            }
            if include_all_fields:
                full_info = dict(r)
                full_info.setdefault("label", row["label"])
                full_info.setdefault("class_id", row["class_id"])
                full_info.setdefault("score", row["score"])
                for raw_name, (safe, sql_typ) in dynamic_field_map.items():
                    if raw_name == "all_info":
                        row[safe] = json.dumps(full_info, ensure_ascii=False, default=str)
                        continue
                    val = r.get(raw_name)
                    if val is None:
                        row[safe] = None
                    elif sql_typ == "INTEGER":
                        row[safe] = 1 if isinstance(val, bool) and val else 0 if isinstance(val, bool) else int(val)
                    elif sql_typ == "REAL":
                        row[safe] = float(val)
                    elif isinstance(val, (str, int, float, bool)):
                        row[safe] = str(val) if sql_typ == "TEXT" else val
                    else:
                        row[safe] = json.dumps(val, ensure_ascii=False, default=str)
            rows.append(row)

        if not rows:
            raise RuntimeError("Selected records do not contain exportable bounding boxes.")

        now = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        conn = sqlite3.connect(str(out_path))
        try:
            cur = conn.cursor()
            cur.execute("PRAGMA application_id = 1196437808")  # 'GPKG'
            cur.execute("PRAGMA user_version = 10400")
            cur.execute("CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT NOT NULL, srs_id INTEGER NOT NULL PRIMARY KEY, organization TEXT NOT NULL, organization_coordsys_id INTEGER NOT NULL, definition TEXT NOT NULL, description TEXT)")
            cur.execute("INSERT INTO gpkg_spatial_ref_sys VALUES (?, ?, ?, ?, ?, ?)", ("Undefined Cartesian SRS", -1, "NONE", -1, "undefined", "undefined cartesian coordinate reference system"))
            cur.execute("INSERT INTO gpkg_spatial_ref_sys VALUES (?, ?, ?, ?, ?, ?)", ("Undefined Geographic SRS", 0, "NONE", 0, "undefined", "undefined geographic coordinate reference system"))
            if srs_id not in (-1, 0):
                cur.execute("INSERT OR REPLACE INTO gpkg_spatial_ref_sys VALUES (?, ?, ?, ?, ?, ?)", (srs_name, srs_id, srs_org, srs_org_id, srs_definition, srs_name))
            cur.execute("CREATE TABLE gpkg_contents (table_name TEXT NOT NULL PRIMARY KEY, data_type TEXT NOT NULL, identifier TEXT UNIQUE, description TEXT DEFAULT '', last_change DATETIME NOT NULL, min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE, srs_id INTEGER, CONSTRAINT fk_gc_r_srs_id FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id))")
            cur.execute("CREATE TABLE gpkg_geometry_columns (table_name TEXT NOT NULL, column_name TEXT NOT NULL, geometry_type_name TEXT NOT NULL, srs_id INTEGER NOT NULL, z TINYINT NOT NULL, m TINYINT NOT NULL, PRIMARY KEY (table_name, column_name), CONSTRAINT fk_gc_tn FOREIGN KEY (table_name) REFERENCES gpkg_contents(table_name), CONSTRAINT fk_gc_srs FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id))")
            col_defs = ", ".join(f"{quote_ident(name)} {typ}" for name, typ in all_fields)
            cur.execute(f"CREATE TABLE pipeline_results ({col_defs})")
            cur.execute("INSERT INTO gpkg_contents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ("pipeline_results", "features", "pipeline_results", "Mustatil YOLO Pipeline results", now, minx, miny, maxx, maxy, srs_id))
            cur.execute("INSERT INTO gpkg_geometry_columns VALUES (?, ?, ?, ?, ?, ?)", ("pipeline_results", "geom", "POLYGON", srs_id, 0, 0))

            insert_cols = [name for name, _typ in all_fields if name != "fid"]
            placeholders = ", ".join("?" for _ in insert_cols)
            sql = f"INSERT INTO pipeline_results ({', '.join(quote_ident(c) for c in insert_cols)}) VALUES ({placeholders})"
            for row in rows:
                cur.execute(sql, [row.get(c) for c in insert_cols])

            # RTree tables are optional for GeoPackage, but QGIS uses them for fast display.
            try:
                cur.execute("CREATE VIRTUAL TABLE rtree_pipeline_results_geom USING rtree(id, minx, maxx, miny, maxy)")
                cur.execute("CREATE TABLE gpkg_extensions (table_name TEXT, column_name TEXT, extension_name TEXT NOT NULL, definition TEXT NOT NULL, scope TEXT NOT NULL, CONSTRAINT ge_tce UNIQUE (table_name, column_name, extension_name))")
                cur.execute("INSERT INTO gpkg_extensions VALUES (?, ?, ?, ?, ?)", ("pipeline_results", "geom", "gpkg_rtree_index", "http://www.geopackage.org/spec/#extension_rtree", "write-only"))
                for fid, row in cur.execute("SELECT fid, geom FROM pipeline_results").fetchall():
                    # Use already computed row bounds from WKB coordinates for exactness by reading bounding points from blob.
                    blob = bytes(row)
                    # Header is 8 bytes, WKB starts at 8. We wrote 5 points after 8+1+4+4+4 = 21 bytes.
                    coords = []
                    off = 8 + 1 + 4 + 4 + 4
                    for _i in range(5):
                        x, y = struct.unpack_from("<dd", blob, off)
                        coords.append((x, y))
                        off += 16
                    xs = [c[0] for c in coords]; ys = [c[1] for c in coords]
                    cur.execute("INSERT INTO rtree_pipeline_results_geom VALUES (?, ?, ?, ?, ?)", (fid, min(xs), max(xs), min(ys), max(ys)))
            except Exception:
                # A valid GPKG does not require the RTree extension.
                pass

            conn.commit()
        finally:
            conn.close()

    def export_results_json(self):
        # Kept for backwards compatibility with saved UI callbacks from older plugin versions.
        self.export_results_gpkg()

    def _write_summary(self):
        counts = []
        for b in self.blocks:
            if b.id in self.results_by_block:
                counts.append(f"{b.name} [{b.type}]: {len(self.results_by_block[b.id])}")
        by_label: Dict[str, int] = {}
        for r in self.results:
            by_label[str(r.get("label", "object"))] = by_label.get(str(r.get("label", "object")), 0) + 1
        self._log("\nBlock counts:\n" + "\n".join(counts))
        self._log("\nLabel counts:\n" + "\n".join(f"{k}: {v}" for k, v in sorted(by_label.items())))

    def _write_intro(self):
        self._log(
            "Visual YOLO Pipeline ready. Drag blocks like building bricks. "
            "Connect blocks by dragging a line from a right socket to a left socket, or use the Input socket dropdown. "
            "Use the View: Code button to edit the same pipeline as readable Python. "
            "Example: YOLO 1 airplanes → YOLO 2 engines and YOLO 3 wings → Rule classifies the parent airplane. "
            "Multi-input rule example: Rule 1 minimum 1 AND Rule 2 minimum 2 → new class."
        )

    def _log(self, msg: str):
        self.summary.append(str(msg))
        try:
            self.ws.log("YOLO Pipeline: " + str(msg).replace("\n", " | "))
        except Exception:
            pass


def _unused_original_install_yolo_pipeline_tab(workspace):
    """Unused in this hook plugin. Kept only as source-compatible fallback."""
    tab = YoloPipelineTab(workspace)
    workspace.yolo_pipeline_tab = tab
    try:
        workspace.log("WARNING: unused original YOLO Pipeline installer was called from AI Pipeline hook plugin.")
    except Exception:
        pass
    # Place YOLO Pipeline next to the YOLO Trainer tab when possible.
    try:
        insert_at = workspace.tabs.count()
        for i in range(workspace.tabs.count()):
            tab_name = workspace.tabs.tabText(i).strip().lower()
            if "yolo" in tab_name and ("trainer" in tab_name or "training" in tab_name):
                insert_at = i + 1
                break
        workspace.tabs.insertTab(insert_at, tab, "YOLO Pipeline")
    except Exception:
        workspace.tabs.addTab(tab, "YOLO Pipeline")
    return tab


# ============================================================================
# Mustatil AI Pipeline tab hook
# Places this functional multi-model pipeline RIGHT NEXT TO "LAE-DINO Trainer".
# This does NOT replace yolo_pipeline_plugin.py and does NOT touch the original
# YOLO Pipeline tab. It creates a separate tab named "AI Pipeline".
# ============================================================================

_AI_PIPELINE_INSERTED_WIDGETS = set()
_AI_PIPELINE_PATCHED_QTAB = False
_AI_PIPELINE_ORIG_ADD = None
_AI_PIPELINE_ORIG_INSERT = None


def _ai_pipe_log(msg):
    try:
        print("[Mustatil AI Pipeline Hook] " + str(msg))
    except Exception:
        pass


def _ai_pipe_workspace_from_widget(w):
    cur = w
    for _ in range(80):
        if cur is None:
            break
        try:
            for attr in ("ws", "workspace", "main_window"):
                obj = getattr(cur, attr, None)
                if obj is not None and hasattr(obj, "tabs"):
                    return obj
        except Exception:
            pass
        try:
            if hasattr(cur, "tabs") and hasattr(cur, "log"):
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


def _ai_pipe_is_lae_trainer_label(text):
    low = str(text or "").strip().lower()
    return low in {"lae-dino trainer", "lae dino trainer", "laedino trainer"} or ("lae" in low and "dino" in low and "train" in low)


def _ai_pipe_is_own_label(text):
    low = str(text or "").strip().lower()
    return low in {"ai pipeline", "multimodel pipeline", "multi-model pipeline", "owl/gdino/lae/sam2 pipeline"}


def _ai_pipe_has_own_tab(tw):
    try:
        for i in range(tw.count()):
            if _ai_pipe_is_own_label(tw.tabText(i)):
                return True
            try:
                w = tw.widget(i)
                if str(w.objectName() or "") == "MustatilAIPipelineNextToLAEDINOTrainer":
                    return True
            except Exception:
                pass
    except Exception:
        pass
    return False


def _ai_pipe_build_tab(ws):
    tab = YoloPipelineTab(ws)
    try:
        tab.setObjectName("MustatilAIPipelineNextToLAEDINOTrainer")
    except Exception:
        pass
    try:
        # Do not overwrite the original workspace.yolo_pipeline_tab.
        if ws is not None:
            ws.ai_pipeline_tab = tab
            ws.mustatil_ai_pipeline_tab = tab
    except Exception:
        pass
    try:
        tab._log(
            "AI Pipeline hook tab ready. This tab is independent from the original YOLO Pipeline tab. "
            "Available blocks: YOLO, OWLv2, Grounding DINO, LAE-DINO, SAM2, FormLearner, Rule."
        )
    except Exception:
        pass
    return tab


def _ai_pipe_ensure_tab_on_widget(tw):
    try:
        if tw is None:
            return False
        if id(tw) in _AI_PIPELINE_INSERTED_WIDGETS and _ai_pipe_has_own_tab(tw):
            return True

        texts = [str(tw.tabText(i) or "").strip() for i in range(tw.count())]
        lower_texts = [t.lower() for t in texts]

        # Only target the tab widget that actually contains LAE-DINO Trainer.
        lae_index = -1
        for i, t in enumerate(texts):
            if _ai_pipe_is_lae_trainer_label(t):
                lae_index = i
                break

        if lae_index < 0:
            return False

        # If an older AI Pipeline tab exists elsewhere in same widget, remove it
        # and reinsert directly after LAE-DINO Trainer.
        for i in range(tw.count() - 1, -1, -1):
            try:
                if _ai_pipe_is_own_label(tw.tabText(i)) or str(tw.widget(i).objectName() or "") == "MustatilAIPipelineNextToLAEDINOTrainer":
                    old = tw.widget(i)
                    tw.removeTab(i)
                    try:
                        old.deleteLater()
                    except Exception:
                        pass
                    if i < lae_index:
                        lae_index -= 1
            except Exception:
                pass

        ws = _ai_pipe_workspace_from_widget(tw)
        page = _ai_pipe_build_tab(ws)
        insert_at = min(tw.count(), lae_index + 1)
        tw.insertTab(insert_at, page, "AI Pipeline")
        _AI_PIPELINE_INSERTED_WIDGETS.add(id(tw))

        try:
            if ws is not None and hasattr(ws, "log"):
                ws.log("AI Pipeline tab inserted right next to LAE-DINO Trainer.")
        except Exception:
            pass
        _ai_pipe_log("inserted next to LAE-DINO Trainer")
        return True
    except Exception as exc:
        _ai_pipe_log("ensure tab failed: " + str(exc))
        try:
            import traceback
            traceback.print_exc()
        except Exception:
            pass
        return False


def _ai_pipe_scan_for_tabwidgets(root=None):
    try:
        from PySide6.QtWidgets import QApplication, QTabWidget
    except Exception as exc:
        _ai_pipe_log("Qt unavailable: " + str(exc))
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
            if _ai_pipe_ensure_tab_on_widget(tw):
                ok = True
        except Exception:
            pass
    return ok


def _ai_pipe_install_hook(root=None):
    global _AI_PIPELINE_PATCHED_QTAB, _AI_PIPELINE_ORIG_ADD, _AI_PIPELINE_ORIG_INSERT
    try:
        from PySide6.QtWidgets import QTabWidget
        from PySide6.QtCore import QTimer
    except Exception as exc:
        _ai_pipe_log("Qt unavailable: " + str(exc))
        return

    if not _AI_PIPELINE_PATCHED_QTAB:
        # Wrap the current QTabWidget methods, even if another plugin already patched them.
        _AI_PIPELINE_ORIG_ADD = QTabWidget.addTab
        _AI_PIPELINE_ORIG_INSERT = QTabWidget.insertTab

        def _after(tw, label):
            try:
                if _ai_pipe_is_lae_trainer_label(label) or _ai_pipe_is_own_label(label) or "trainer" in str(label or "").lower():
                    QTimer.singleShot(30, lambda tw=tw: _ai_pipe_ensure_tab_on_widget(tw))
                    QTimer.singleShot(250, lambda tw=tw: _ai_pipe_ensure_tab_on_widget(tw))
                    QTimer.singleShot(1000, lambda tw=tw: _ai_pipe_ensure_tab_on_widget(tw))
            except Exception:
                pass

        def addTab_patched(self, page, *args, **kwargs):
            res = _AI_PIPELINE_ORIG_ADD(self, page, *args, **kwargs)
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
            res = _AI_PIPELINE_ORIG_INSERT(self, index, page, *args, **kwargs)
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
        _AI_PIPELINE_PATCHED_QTAB = True
        _ai_pipe_log("QTabWidget hook installed")

    try:
        # Several delayed scans: LAE-DINO Trainer may be inserted by another plugin later.
        for ms in (100, 300, 800, 1500, 3000, 6000, 10000, 15000):
            QTimer.singleShot(ms, lambda root=root: _ai_pipe_scan_for_tabwidgets(root))
    except Exception:
        pass


def mustatil_plugin_init():
    _ai_pipe_install_hook()
    return True


def register_plugin(app=None, main_window=None):
    _ai_pipe_install_hook(main_window or app)
    try:
        _ai_pipe_scan_for_tabwidgets(main_window or app)
    except Exception:
        pass
    return True


def init_plugin(app=None, main_window=None):
    return register_plugin(app, main_window)


def load_plugin(app=None, main_window=None):
    return register_plugin(app, main_window)


try:
    _ai_pipe_install_hook()
except Exception:
    pass
