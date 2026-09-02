#!/usr/bin/env python3
"""Viser annotator for video instances in an HO3D dataset directory."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import viser

VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv")
HAND_OPTIONS = ("left", "right", "both")
INT_FIELDS = ("frame_n", "source_reference_frame", "grasp_frame", "release_frame")


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    label: int


@dataclass(frozen=True)
class VideoInfo:
    frame_count: int
    fps: float
    width: int
    height: int


@dataclass(frozen=True)
class InstanceRecord:
    directory: Path
    config_path: Path
    video_path: Path


@dataclass
class AnnotationState:
    config: dict[str, Any]
    frame_map: dict[int, dict[str, Any]]
    frame_n: int | None
    source_reference_frame: int | None
    object_name: str
    object_id: str | None
    anchor_hand: str
    grasp_frame: int | None
    release_frame: int | None
    points: list[Point] = field(default_factory=list)
    bbox: list[float] | None = None
    bbox_center: list[float] | None = None
    bbox_points: list[Point] = field(default_factory=list)
    dirty_fields: set[str] = field(default_factory=set)
    points_dirty: bool = False
    bbox_dirty: bool = False
    point_load_error: str | None = None


class VideoSource:
    def __init__(self, path: Path, cache_size: int = 24):
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        self.lock = threading.RLock()
        if not self.capture.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_count <= 0 or width <= 0 or height <= 0:
            self.close()
            raise RuntimeError(f"invalid video metadata: {path}")
        self.info = VideoInfo(frame_count, fps if fps > 0 else 30.0, width, height)
        self.cache_size = cache_size
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def get_rgb(self, frame_index: int) -> np.ndarray:
        frame_index = int(frame_index)
        if not 0 <= frame_index < self.info.frame_count:
            raise IndexError(f"frame {frame_index} outside [0, {self.info.frame_count})")
        with self.lock:
            cached = self.cache.get(frame_index)
            if cached is not None:
                self.cache.move_to_end(frame_index)
                return cached.copy()
            if not self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                raise RuntimeError(f"failed to seek video frame {frame_index}")
            ok, bgr = self.capture.read()
            if not ok or bgr is None:
                raise RuntimeError(f"failed to read video frame {frame_index}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.cache[frame_index] = rgb
            self.cache.move_to_end(frame_index)
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
            return rgb.copy()

    def close(self) -> None:
        with getattr(self, "lock", threading.RLock()):
            capture = getattr(self, "capture", None)
            if capture is not None:
                capture.release()
                self.capture = None


def parse_optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if not re.fullmatch(r"[+-]?\d+", text):
        raise ValueError(f"{field_name} must be an integer or blank")
    return int(text)


def parse_points(points_text: Any, labels_text: Any) -> list[Point]:
    points_text = "" if points_text is None else str(points_text).strip()
    labels_text = "" if labels_text is None else str(labels_text).strip()
    if not points_text and not labels_text:
        return []
    point_tokens = [token for token in points_text.split(";") if token.strip()]
    label_tokens = [token for token in labels_text.split(";") if token.strip()]
    if len(point_tokens) != len(label_tokens):
        raise ValueError(
            f"point count {len(point_tokens)} does not match label count {len(label_tokens)}"
        )
    points = []
    for point_token, label_token in zip(point_tokens, label_tokens):
        xy = point_token.split(",")
        if len(xy) != 2:
            raise ValueError(f"invalid point: {point_token!r}")
        try:
            x, y = float(xy[0].strip()), float(xy[1].strip())
            label = int(label_token.strip())
        except ValueError as error:
            raise ValueError(f"invalid point or label: {point_token!r}") from error
        if label not in (0, 1):
            raise ValueError(f"point label must be 0 or 1, got {label}")
        points.append(Point(x, y, label))
    return points


def serialize_points(points: list[Point]) -> tuple[str, str]:
    point_text = ";".join(f"{int(round(p.x))},{int(round(p.y))}" for p in points)
    label_text = ";".join(str(p.label) for p in points)
    return point_text, label_text


def parse_bbox(value: Any) -> tuple[list[float] | None, list[Point]]:
    if not isinstance(value, list) or len(value) != 4:
        return None, []
    if not all(isinstance(item, (int, float)) for item in value):
        return None, []
    x1, y1, x2, y2 = [float(item) for item in value]
    return [x1, y1, x2, y2], [Point(x1, y1, 1), Point(x2, y2, 1)]


def bbox_from_points(points: list[Point]) -> list[float] | None:
    if len(points) != 2:
        return None
    x1, x2 = sorted((float(points[0].x), float(points[1].x)))
    y1, y2 = sorted((float(points[0].y), float(points[1].y)))
    return [x1, y1, x2, y2]


def bbox_center(bbox: list[float]) -> list[float]:
    return [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]


def file_signature(path: Path) -> tuple[int, int, str]:
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return stat.st_mtime_ns, stat.st_size, digest


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def find_video(instance_dir: Path) -> Path:
    for suffix in VIDEO_SUFFIXES:
        candidate = instance_dir / f"{instance_dir.name}{suffix}"
        if candidate.is_file():
            return candidate
    for suffix in VIDEO_SUFFIXES:
        candidate = instance_dir / f"video{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no convention video found in {instance_dir}")


def discover_instances(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    records = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        if not (path / "daid_config.json").is_file():
            continue
        try:
            find_video(path)
        except FileNotFoundError:
            continue
        records.append(path)
    return records


def load_frame_map(instance_dir: Path, config: dict[str, Any]) -> dict[int, dict[str, Any]]:
    map_path = instance_dir / str(config.get("ho3d_frame_map", "ho3d_frame_map.json"))
    if not map_path.is_file():
        return {}
    raw = json.loads(map_path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict) and isinstance(raw.get("mapping"), dict):
        rows = []
        for key, value in raw["mapping"].items():
            item = dict(value)
            item.setdefault("local_frame", int(key))
            rows.append(item)
    else:
        raise ValueError(f"unsupported frame map format: {map_path}")
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"frame map contains a non-object entry: {map_path}")
        local_frame = parse_optional_int(row.get("local_frame"), "local_frame")
        if local_frame is None or local_frame in result:
            raise ValueError(f"invalid or duplicate local_frame: {map_path}")
        result[local_frame] = row
    expected = set(range(len(result)))
    if set(result) != expected:
        raise ValueError(f"local_frame must be contiguous from 0: {map_path}")
    return result


def source_frame_for(frame_map: dict[int, dict[str, Any]], local_frame: int) -> int | None:
    row = frame_map.get(int(local_frame))
    if row is None:
        return None
    return parse_optional_int(row.get("source_frame"), "source_frame")


def load_state(record: InstanceRecord) -> tuple[AnnotationState, tuple[int, int, str]]:
    config = json.loads(record.config_path.read_text(encoding="utf-8"))
    frame_map = load_frame_map(record.directory, config)
    nested_objects = config.get("objects")
    nested = (
        nested_objects[0]
        if isinstance(nested_objects, list)
        and len(nested_objects) == 1
        and isinstance(nested_objects[0], dict)
        else {}
    )
    object_name = str(config.get("object") or nested.get("object_name") or "")
    raw_object_id = config.get("object_id")
    object_id = None if raw_object_id in (None, "") else str(raw_object_id)
    raw_hand = config.get("anchor_hand", "right")
    anchor_hand = "" if raw_hand is None else str(raw_hand)
    point_load_error = None
    try:
        points = parse_points(
            config.get("sam3_object_points", ""),
            config.get("sam3_object_point_labels", ""),
        )
    except ValueError as error:
        points = []
        point_load_error = str(error)
    bbox, bbox_points = parse_bbox(nested.get("bbox"))
    center = nested.get("bbox_center")
    bbox_center_value = (
        [float(center[0]), float(center[1])]
        if isinstance(center, list)
        and len(center) == 2
        and all(isinstance(item, (int, float)) for item in center)
        else None
    )
    state = AnnotationState(
        config=config,
        frame_map=frame_map,
        frame_n=parse_optional_int(config.get("frame_n"), "frame_n"),
        source_reference_frame=parse_optional_int(
            config.get("source_reference_frame"), "source_reference_frame"
        ),
        object_name=object_name,
        object_id=object_id,
        anchor_hand=anchor_hand,
        grasp_frame=parse_optional_int(config.get("grasp_frame"), "grasp_frame"),
        release_frame=parse_optional_int(config.get("release_frame"), "release_frame"),
        points=points,
        bbox=bbox,
        bbox_center=bbox_center_value,
        bbox_points=bbox_points,
        point_load_error=point_load_error,
    )
    return state, file_signature(record.config_path)


def decorate_frame(
    frame_rgb: np.ndarray,
    state: AnnotationState,
    current_frame: int,
    video_info: VideoInfo,
    bbox_mode: str,
) -> np.ndarray:
    image = frame_rgb.copy()
    is_anchor = state.frame_n is not None and current_frame == state.frame_n
    if is_anchor:
        for index, point in enumerate(state.points, start=1):
            color = (60, 220, 80) if point.label == 1 else (240, 70, 70)
            center = (int(round(point.x)), int(round(point.y)))
            cv2.circle(image, center, 6, color, -1)
            cv2.circle(image, center, 9, (255, 255, 255), 1)
            cv2.putText(
                image,
                str(index),
                (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        if len(state.bbox_points) == 2:
            box = bbox_from_points(state.bbox_points)
            if box is not None:
                x1, y1, x2, y2 = [int(round(value)) for value in box]
                cv2.rectangle(image, (x1, y1), (x2, y2), (60, 200, 240), 2)
    lines = [
        f"frame {current_frame}/{video_info.frame_count - 1}",
        f"frame_n={state.frame_n if state.frame_n is not None else '-'}",
        f"object={state.object_name or '-'} hand={state.anchor_hand or '-'}",
    ]
    if state.grasp_frame is not None or state.release_frame is not None:
        lines.append(
            f"grasp={state.grasp_frame if state.grasp_frame is not None else '-'} "
            f"release={state.release_frame if state.release_frame is not None else '-'}"
        )
    if is_anchor and bbox_mode == "bbox":
        lines.append(f"bbox points={len(state.bbox_points)}/2")
    overlay_height = 24 * len(lines) + 8
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (min(image.shape[1], 600), overlay_height),
        (0, 0, 0),
        -1,
    )
    image = cv2.addWeighted(overlay, 0.65, image, 0.35, 0)
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (8, 20 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return image


class Ho3dAnnotator:
    def __init__(self, root: Path, host: str, port: int):
        self.root = root.resolve()
        self.host = host
        self.port = port
        self.records = [
            InstanceRecord(path, path / "daid_config.json", find_video(path))
            for path in discover_instances(self.root)
        ]
        self.server: viser.ViserServer | None = None
        self.current: InstanceRecord | None = None
        self.state: AnnotationState | None = None
        self.baseline_signature: tuple[int, int, str] | None = None
        self.video: VideoSource | None = None
        self.current_frame = 0
        self.playing = False
        self.playback_fps = 30.0
        self.last_tick = 0.0
        self.point_label = 1
        self.bbox_mode = "points"
        self.loading = False
        self.lock = threading.RLock()
        self.gui: dict[str, Any] = {}

    def run(self) -> None:
        if not self.records:
            raise RuntimeError(f"no instances found under {self.root}")
        self.server = viser.ViserServer(host=self.host, port=self.port)
        self._build_gui()

        @self.server.scene.on_click()
        def _(event: Any) -> None:
            self._on_scene_click(event)

        self._select_instance(self.records[0].directory.name)
        try:
            while True:
                self._tick()
                time.sleep(0.03)
        except KeyboardInterrupt:
            pass
        finally:
            if self.video is not None:
                self.video.close()
            self.server.stop()

    def _build_gui(self) -> None:
        assert self.server is not None
        names = tuple(record.directory.name for record in self.records)
        with self.server.gui.add_folder("Instance"):
            self.gui["instance"] = self.server.gui.add_dropdown(
                "Instance", options=names, initial_value=names[0]
            )
            self.gui["instance"].on_update(
                lambda event: self._select_instance(str(event.target.value))
            )
            self.gui["instance_info"] = self.server.gui.add_markdown("-")

        with self.server.gui.add_folder("Playback"):
            self.gui["frame"] = self.server.gui.add_slider(
                "Current frame", min=0, max=1, step=1, initial_value=0
            )
            self.gui["frame"].on_update(
                lambda event: self._set_current_frame(int(event.target.value))
            )
            for key, label, delta in (
                ("previous10", "-10", -10),
                ("previous", "-1", -1),
                ("next", "+1", 1),
                ("next10", "+10", 10),
            ):
                self.gui[key] = self.server.gui.add_button(label)
                self.gui[key].on_click(lambda _event, d=delta: self._step_frame(d))
            self.gui["play"] = self.server.gui.add_button("Play / Pause")
            self.gui["play"].on_click(lambda _event: self._toggle_play())
            self.gui["fps"] = self.server.gui.add_number(
                "Playback FPS", initial_value=30.0, min=1.0, max=120.0, step=1.0
            )
            self.gui["fps"].on_update(
                lambda event: self._set_fps(float(event.target.value))
            )

        with self.server.gui.add_folder("Annotation"):
            self.gui["object"] = self.server.gui.add_text("Object", initial_value="")
            self.gui["object"].on_update(
                lambda event: self._on_text_field("object_name", event.target.value)
            )
            self.gui["object_id"] = self.server.gui.add_text(
                "Object ID", initial_value=""
            )
            self.gui["object_id"].on_update(
                lambda event: self._on_text_field("object_id", event.target.value)
            )
            self.gui["hand"] = self.server.gui.add_dropdown(
                "Anchor hand", options=HAND_OPTIONS, initial_value="right"
            )
            self.gui["hand"].on_update(
                lambda event: self._on_text_field("anchor_hand", event.target.value)
            )
            self.gui["frame_n"] = self.server.gui.add_text(
                "frame_n (local)", initial_value=""
            )
            self.gui["frame_n"].on_update(
                lambda event: self._on_int_field("frame_n", event.target.value)
            )
            self.gui["source_reference"] = self.server.gui.add_text(
                "source_reference_frame", initial_value=""
            )
            self.gui["source_reference"].on_update(
                lambda event: self._on_int_field(
                    "source_reference_frame", event.target.value
                )
            )
            self.gui["set_frame"] = self.server.gui.add_button(
                "Set frame_n = current"
            )
            self.gui["set_frame"].on_click(
                lambda _event: self._set_frame_n_current()
            )
            self.gui["set_source"] = self.server.gui.add_button(
                "Set source reference = current"
            )
            self.gui["set_source"].on_click(
                lambda _event: self._set_source_reference_current()
            )
            self.gui["grasp"] = self.server.gui.add_text(
                "grasp_frame", initial_value=""
            )
            self.gui["grasp"].on_update(
                lambda event: self._on_int_field("grasp_frame", event.target.value)
            )
            self.gui["release"] = self.server.gui.add_text(
                "release_frame", initial_value=""
            )
            self.gui["release"].on_update(
                lambda event: self._on_int_field("release_frame", event.target.value)
            )
            self.gui["set_grasp"] = self.server.gui.add_button(
                "Set grasp = current"
            )
            self.gui["set_grasp"].on_click(
                lambda _event: self._set_grasp_current()
            )
            self.gui["set_release"] = self.server.gui.add_button(
                "Set release = current"
            )
            self.gui["set_release"].on_click(
                lambda _event: self._set_release_current()
            )

        with self.server.gui.add_folder("SAM3 points and bbox"):
            self.gui["label"] = self.server.gui.add_button_group(
                "Point label",
                options=("positive", "negative"),
            )
            self.gui["label"].on_click(
                lambda event: self._set_point_label(str(event.target.value))
            )
            self.gui["mode"] = self.server.gui.add_dropdown(
                "Click mode", options=("points", "bbox"), initial_value="points"
            )
            self.gui["mode"].on_update(
                lambda event: self._set_bbox_mode(str(event.target.value))
            )
            self.gui["undo"] = self.server.gui.add_button("Undo last point")
            self.gui["undo"].on_click(lambda _event: self._undo_point())
            self.gui["clear"] = self.server.gui.add_button("Clear points")
            self.gui["clear"].on_click(lambda _event: self._clear_points())
            self.gui["clear_bbox"] = self.server.gui.add_button("Clear bbox")
            self.gui["clear_bbox"].on_click(lambda _event: self._clear_bbox())
            self.gui["point_info"] = self.server.gui.add_markdown(
                "Set frame_n before clicking; bbox uses two explicit clicks."
            )

        with self.server.gui.add_folder("Save"):
            self.gui["save_draft"] = self.server.gui.add_button("Save draft")
            self.gui["save_draft"].on_click(
                lambda _event: self._save(approve=False)
            )
            self.gui["save_approve"] = self.server.gui.add_button(
                "Save / Approve"
            )
            self.gui["save_approve"].on_click(
                lambda _event: self._save(approve=True)
            )
            self.gui["status"] = self.server.gui.add_markdown("status: ready")

    def _set_gui_text(self, key: str, value: str) -> None:
        self.loading = True
        try:
            self.gui[key].value = value
        finally:
            self.loading = False

    def _on_text_field(self, field_name: str, value: Any) -> None:
        with self.lock:
            if self.loading or self.state is None:
                return
            if field_name == "object_name":
                self.state.object_name = str(value).strip()
            elif field_name == "object_id":
                text = str(value).strip()
                self.state.object_id = text or None
            elif field_name == "anchor_hand":
                self.state.anchor_hand = str(value).strip()
            self.state.dirty_fields.add(field_name)
            self._render()

    def _on_int_field(self, field_name: str, value: Any) -> None:
        with self.lock:
            if self.loading or self.state is None:
                return
            try:
                parsed = parse_optional_int(value, field_name)
            except ValueError as error:
                self._set_status(str(error))
                return
            if field_name == "frame_n":
                if parsed != self.state.frame_n:
                    self.state.points = []
                    self.state.points_dirty = True
                    self.state.bbox_points = []
                    self.state.bbox_dirty = True
                self.state.source_reference_frame = (
                    source_frame_for(self.state.frame_map, parsed)
                    if parsed is not None
                    else None
                )
                self.state.dirty_fields.add("source_reference_frame")
                self._set_gui_text(
                    "source_reference",
                    ""
                    if self.state.source_reference_frame is None
                    else str(self.state.source_reference_frame),
                )
            setattr(self.state, field_name, parsed)
            self.state.dirty_fields.add(field_name)
            self._render()

    def _select_instance(self, name: str) -> None:
        with self.lock:
            if self.loading:
                return
            record = next(
                (item for item in self.records if item.directory.name == name), None
            )
            if record is None:
                self._set_status(f"unknown instance: {name}")
                return
            if self.video is not None:
                self.video.close()
                self.video = None
            try:
                state, signature = load_state(record)
                video = VideoSource(record.video_path)
            except Exception as error:
                self.current = record
                self.state = None
                self.baseline_signature = None
                self._set_status(f"load failed: {error}")
                return
            self.current = record
            self.state = state
            self.baseline_signature = signature
            self.video = video
            self.current_frame = state.frame_n
            if self.current_frame is None:
                self.current_frame = 0
            self.current_frame = min(self.current_frame, video.info.frame_count - 1)
            self.playing = False
            self.playback_fps = video.info.fps
            self.loading = True
            try:
                self.gui["instance"].value = record.directory.name
                self.gui["frame"].max = video.info.frame_count - 1
                self.gui["frame"].value = self.current_frame
                self.gui["fps"].value = self.playback_fps
                self.gui["object"].value = state.object_name
                self.gui["object_id"].value = state.object_id or ""
                self.gui["hand"].value = (
                    state.anchor_hand if state.anchor_hand in HAND_OPTIONS else "right"
                )
                self.gui["frame_n"].value = (
                    "" if state.frame_n is None else str(state.frame_n)
                )
                self.gui["source_reference"].value = (
                    ""
                    if state.source_reference_frame is None
                    else str(state.source_reference_frame)
                )
                self.gui["grasp"].value = (
                    "" if state.grasp_frame is None else str(state.grasp_frame)
                )
                self.gui["release"].value = (
                    "" if state.release_frame is None else str(state.release_frame)
                )
            finally:
                self.loading = False
            self._set_status(
                f"loaded {record.directory.name}: {video.info.frame_count} frames; "
                f"status={state.config.get('status', 'unknown')}"
            )
            if state.point_load_error:
                self._set_status(f"warning: {state.point_load_error}")
            self._render()

    def _set_current_frame(self, index: int) -> None:
        with self.lock:
            if self.video is None:
                return
            self.current_frame = max(
                0, min(int(index), self.video.info.frame_count - 1)
            )
            if not self.loading:
                self.gui["frame"].value = self.current_frame
            self._render()

    def _step_frame(self, delta: int) -> None:
        self._set_current_frame(self.current_frame + delta)

    def _toggle_play(self) -> None:
        with self.lock:
            self.playing = not self.playing
            self.last_tick = time.monotonic()
            self._set_status("playing" if self.playing else "paused")

    def _set_fps(self, value: float) -> None:
        with self.lock:
            self.playback_fps = max(1.0, float(value))

    def _set_frame_n_current(self) -> None:
        with self.lock:
            if self.state is None:
                return
            changed = self.state.frame_n != self.current_frame
            self.state.frame_n = self.current_frame
            self.state.dirty_fields.add("frame_n")
            if changed:
                self.state.points = []
                self.state.points_dirty = True
                self.state.bbox_points = []
                self.state.bbox_dirty = True
            source_frame = source_frame_for(self.state.frame_map, self.current_frame)
            self.state.source_reference_frame = source_frame
            self.state.dirty_fields.add("source_reference_frame")
            self._set_gui_text(
                "source_reference",
                "" if source_frame is None else str(source_frame),
            )
            self._set_gui_text("frame_n", str(self.current_frame))
            self._set_status(
                "frame_n set; previous points/bbox cleared"
                if changed
                else "frame_n set to current frame"
            )
            self._render()

    def _set_source_reference_current(self) -> None:
        with self.lock:
            if self.state is None or self.state.frame_n is None:
                self._set_status("set frame_n first")
                return
            source_frame = source_frame_for(self.state.frame_map, self.state.frame_n)
            if source_frame is None:
                self._set_status("no exact source_frame mapping; nothing was filled")
                return
            self.state.source_reference_frame = source_frame
            self.state.dirty_fields.add("source_reference_frame")
            self._set_gui_text("source_reference", str(source_frame))
            self._set_status(f"source reference set to {source_frame} from frame map")
            self._render()

    def _set_grasp_current(self) -> None:
        with self.lock:
            if self.state is not None:
                self.state.grasp_frame = self.current_frame
                self.state.dirty_fields.add("grasp_frame")
                self._set_gui_text("grasp", str(self.current_frame))
                self._set_status(f"grasp_frame set to {self.current_frame}")
                self._render()

    def _set_release_current(self) -> None:
        with self.lock:
            if self.state is not None:
                self.state.release_frame = self.current_frame
                self.state.dirty_fields.add("release_frame")
                self._set_gui_text("release", str(self.current_frame))
                self._set_status(f"release_frame set to {self.current_frame}")
                self._render()

    def _set_point_label(self, value: str) -> None:
        with self.lock:
            self.point_label = 1 if value == "positive" else 0
            self._set_status(
                f"point label = {'positive' if self.point_label else 'negative'}"
            )

    def _set_bbox_mode(self, value: str) -> None:
        with self.lock:
            self.bbox_mode = value if value in ("points", "bbox") else "points"
            self._set_status(
                "bbox mode: click two corners"
                if self.bbox_mode == "bbox"
                else "point mode: click SAM3 points"
            )
            self._render()

    def _undo_point(self) -> None:
        with self.lock:
            if self.state is None:
                return
            if self.bbox_mode == "bbox" and self.state.bbox_points:
                self.state.bbox_points.pop()
                self.state.bbox_dirty = True
            elif self.state.points:
                self.state.points.pop()
                self.state.points_dirty = True
            self._render()
            self._set_status("removed last annotation click")

    def _clear_points(self) -> None:
        with self.lock:
            if self.state is not None:
                self.state.points = []
                self.state.points_dirty = True
                self._render()
                self._set_status("SAM3 points cleared")

    def _clear_bbox(self) -> None:
        with self.lock:
            if self.state is not None:
                self.state.bbox = None
                self.state.bbox_center = None
                self.state.bbox_points = []
                self.state.bbox_dirty = True
                self._render()
                self._set_status("bbox cleared")

    def _on_scene_click(self, event: Any) -> None:
        with self.lock:
            if self.video is None or self.state is None:
                return
            if self.state.frame_n is None:
                self._set_status("click ignored: set frame_n before annotating")
                return
            if self.current_frame != self.state.frame_n:
                self._set_status(
                    f"click ignored: current frame {self.current_frame} != frame_n {self.state.frame_n}"
                )
                return
            u, v = event.screen_pos
            x = int(round(float(u) * (self.video.info.width - 1)))
            y = int(round(float(v) * (self.video.info.height - 1)))
            if not (0 <= x < self.video.info.width and 0 <= y < self.video.info.height):
                self._set_status("click ignored: outside video bounds")
                return
            if self.bbox_mode == "bbox":
                if len(self.state.bbox_points) >= 2:
                    self.state.bbox_points = []
                self.state.bbox_points.append(Point(float(x), float(y), 1))
                self.state.bbox_dirty = True
                if len(self.state.bbox_points) == 2:
                    box = bbox_from_points(self.state.bbox_points)
                    assert box is not None
                    self.state.bbox = box
                    self.state.bbox_center = bbox_center(box)
                    self._set_status(f"bbox set to {[int(value) for value in box]}")
                else:
                    self._set_status("bbox first corner set; click the second corner")
            else:
                self.state.points.append(Point(float(x), float(y), self.point_label))
                self.state.points_dirty = True
                self._set_status(
                    f"added {'positive' if self.point_label else 'negative'} point ({x},{y})"
                )
            self._render()

    def _candidate_values(self) -> tuple[dict[str, Any], str | None]:
        assert self.state is not None and self.video is not None
        try:
            values = {
                "frame_n": parse_optional_int(self.gui["frame_n"].value, "frame_n"),
                "source_reference_frame": parse_optional_int(
                    self.gui["source_reference"].value, "source_reference_frame"
                ),
                "grasp_frame": parse_optional_int(
                    self.gui["grasp"].value, "grasp_frame"
                ),
                "release_frame": parse_optional_int(
                    self.gui["release"].value, "release_frame"
                ),
            }
        except ValueError as error:
            return {}, str(error)
        for field_name in ("frame_n", "grasp_frame", "release_frame"):
            value = values[field_name]
            if value is not None and not 0 <= value < self.video.info.frame_count:
                return {}, f"{field_name} must be in [0, {self.video.info.frame_count})"
        source_reference = values["source_reference_frame"]
        if source_reference is not None and source_reference < 0:
            return {}, "source_reference_frame must be non-negative"
        grasp, release = values["grasp_frame"], values["release_frame"]
        if (grasp is None) != (release is None):
            return {}, "grasp_frame and release_frame must both be set or both blank"
        if grasp is not None and release is not None and grasp >= release:
            return {}, "grasp_frame must be before release_frame"
        anchor_hand = str(self.gui["hand"].value).strip()
        if anchor_hand not in HAND_OPTIONS:
            return {}, "anchor_hand must be left, right, or both"
        object_name = str(self.gui["object"].value).strip()
        if not object_name:
            return {}, "object must not be blank"
        values.update(
            {
                "object": object_name,
                "object_id": str(self.gui["object_id"].value).strip() or None,
                "anchor_hand": anchor_hand,
            }
        )
        if len(self.state.bbox_points) == 1:
            return {}, "bbox has one corner; add the second corner or clear bbox"
        if self.state.points_dirty and values["frame_n"] is None and self.state.points:
            return {}, "set frame_n before saving points"
        return values, None

    def _apply_updates(self, payload: dict[str, Any], values: dict[str, Any], approve: bool) -> None:
        state = self.state
        assert state is not None
        fields = state.dirty_fields
        for key in (
            "frame_n",
            "source_reference_frame",
            "object",
            "object_id",
            "anchor_hand",
            "grasp_frame",
            "release_frame",
        ):
            if key in fields:
                payload[key] = values[key]
        if state.points_dirty:
            points_text, labels_text = serialize_points(state.points)
            payload["sam3_object_points"] = points_text
            payload["sam3_object_point_labels"] = labels_text
        if approve:
            payload["status"] = "approved"

        objects = payload.get("objects")
        if not isinstance(objects, list) or len(objects) != 1 or not isinstance(objects[0], dict):
            return
        nested = objects[0]
        old_object = state.config.get("object")
        nested_name = nested.get("object_name")
        matches_single_object = nested_name in (None, old_object, values["object"])
        if not matches_single_object:
            return
        if "object" in fields and "object_name" in nested:
            nested["object_name"] = values["object"]
        if "object_id" in fields and "object_id" in nested:
            nested["object_id"] = values["object_id"]
        if state.points_dirty:
            if "sam3_object_points" in nested:
                nested["sam3_object_points"] = payload["sam3_object_points"]
            if "sam3_object_point_labels" in nested:
                nested["sam3_object_point_labels"] = payload["sam3_object_point_labels"]
        if state.bbox_dirty:
            nested["bbox"] = state.bbox
            nested["bbox_center"] = state.bbox_center

    def _save(self, approve: bool) -> None:
        with self.lock:
            if self.current is None or self.state is None or self.baseline_signature is None:
                self._set_status("nothing loaded")
                return
            values, error = self._candidate_values()
            if error:
                self._set_status(f"save rejected: {error}")
                return
            if approve:
                if values["frame_n"] is None:
                    self._set_status("approval requires frame_n")
                    return
                if not any(point.label == 1 for point in self.state.points):
                    self._set_status("approval requires at least one positive point")
                    return
            try:
                if file_signature(self.current.config_path) != self.baseline_signature:
                    self._set_status(
                        "save rejected: config changed on disk; reload the instance"
                    )
                    return
                payload = copy.deepcopy(self.state.config)
                self._apply_updates(payload, values, approve)
                write_json_atomic(self.current.config_path, payload)
                self.state.config = payload
                self.state.frame_n = values["frame_n"]
                self.state.source_reference_frame = values["source_reference_frame"]
                self.state.object_name = values["object"]
                self.state.object_id = values["object_id"]
                self.state.anchor_hand = values["anchor_hand"]
                self.state.grasp_frame = values["grasp_frame"]
                self.state.release_frame = values["release_frame"]
                self.state.points_dirty = False
                self.state.bbox_dirty = False
                self.state.dirty_fields.clear()
                self.baseline_signature = file_signature(self.current.config_path)
                self._set_status(
                    f"saved {self.current.config_path.name} "
                    f"({'approved' if approve else 'draft'}); unknown fields preserved"
                )
                self._render()
            except Exception as save_error:
                self._set_status(f"save failed: {save_error}")

    def _render(self) -> None:
        if self.server is None or self.video is None or self.state is None or self.current is None:
            return
        try:
            frame = self.video.get_rgb(self.current_frame)
            image = decorate_frame(
                frame,
                self.state,
                self.current_frame,
                self.video.info,
                self.bbox_mode,
            )
            source_ref = (
                source_frame_for(self.state.frame_map, self.state.frame_n)
                if self.state.frame_n is not None
                else None
            )
            info = (
                f"**{self.current.directory.name}** · frame {self.current_frame}/"
                f"{self.video.info.frame_count - 1} · frame_n="
                f"{self.state.frame_n if self.state.frame_n is not None else '-'} · "
                f"source={source_ref if source_ref is not None else '-'}"
            )
            point_info = (
                f"SAM3 points={len(self.state.points)} · positive="
                f"{sum(point.label == 1 for point in self.state.points)} · "
                f"bbox={'set' if self.state.bbox is not None else 'blank'}"
            )
            with self.server.atomic():
                self.server.scene.set_background_image(
                    np.asarray(image), format="jpeg", jpeg_quality=85
                )
                self.gui["instance_info"].content = info
                self.gui["point_info"].content = point_info
        except Exception as error:
            self._set_status(f"render failed: {error}")

    def _set_status(self, message: str) -> None:
        if "status" in self.gui:
            self.gui["status"].content = f"status: {message}"

    def _tick(self) -> None:
        with self.lock:
            if not self.playing or self.video is None:
                return
            now = time.monotonic()
            if now - self.last_tick < 1.0 / max(self.playback_fps, 1.0):
                return
            self.last_tick = now
            next_frame = self.current_frame + 1
            if next_frame >= self.video.info.frame_count:
                self.playing = False
                self._set_status("playback reached end")
                return
            self.current_frame = next_frame
            self.gui["frame"].value = next_frame
            self._render()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument(
        "--list", action="store_true", help="list instances without starting Viser"
    )
    return parser.parse_args()


def print_instances(root: Path) -> None:
    for path in discover_instances(root):
        try:
            video = VideoSource(find_video(path))
            try:
                info = video.info
            finally:
                video.close()
            print(
                f"{path.name}\tframes={info.frame_count}\t"
                f"fps={info.fps:g}\tsize={info.width}x{info.height}"
            )
        except Exception as error:
            print(f"{path.name}\tERROR\t{error}")


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if args.list:
        print_instances(root)
        return 0
    Ho3dAnnotator(root, args.host, args.port).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
