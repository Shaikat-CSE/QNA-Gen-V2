from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]


class DocLayoutFigureDetector:
    def __init__(
        self,
        model_path: Path,
        auto_download: bool,
        repo_id: str,
        filename: str,
        image_size: int,
        confidence: float,
        iou: float | None,
        device: str,
        figure_labels: list[str],
    ) -> None:
        self.image_size = image_size
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.figure_labels = {normalize_label(label) for label in figure_labels}

        resolved_model_path = self._resolve_model_path(model_path, auto_download, repo_id, filename)
        self.model = self._load_model(resolved_model_path)

    def detect(self, image_path: Path) -> list[Detection]:
        kwargs: dict[str, Any] = {
            "imgsz": self.image_size,
            "conf": self.confidence,
            "device": self.device,
            "verbose": False,
        }
        if self.iou is not None:
            kwargs["iou"] = self.iou

        try:
            results = self.model.predict(str(image_path), **kwargs)
        except TypeError:
            kwargs.pop("iou", None)
            results = self.model.predict(str(image_path), **kwargs)

        if not results:
            return []

        detections = self._parse_result(results[0])
        LOGGER.debug("Detected %s figure candidates in %s", len(detections), image_path)
        return detections

    def _resolve_model_path(self, model_path: Path, auto_download: bool, repo_id: str, filename: str) -> Path:
        if model_path.exists():
            return model_path

        if not auto_download:
            raise FileNotFoundError(
                f"DocLayout-YOLO model not found at {model_path}. "
                "Place the .pt file there or enable model.auto_download in config.yaml."
            )

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface-hub is required for model auto-download. Run: pip install -r requirements.txt"
            ) from exc

        LOGGER.info("Model not found at %s; downloading %s/%s", model_path, repo_id, filename)
        downloaded = hf_hub_download(repo_id=repo_id, filename=filename)
        return Path(downloaded)

    def _load_model(self, model_path: Path) -> Any:
        try:
            from doclayout_yolo import YOLOv10
        except ImportError as exc:
            raise RuntimeError(
                "doclayout-yolo is required for figure detection. Run: pip install -r requirements.txt"
            ) from exc

        LOGGER.info("Loading DocLayout-YOLO model: %s", model_path)
        return YOLOv10(str(model_path))

    def _parse_result(self, result: Any) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        names = getattr(result, "names", None) or getattr(self.model, "names", {})
        detections: list[Detection] = []

        for index in range(len(boxes)):
            cls_id = int(to_scalar(boxes.cls[index])) if getattr(boxes, "cls", None) is not None else -1
            label = class_name(names, cls_id)
            if normalize_label(label) not in self.figure_labels:
                continue

            confidence = float(to_scalar(boxes.conf[index])) if getattr(boxes, "conf", None) is not None else 0.0
            if confidence < self.confidence:
                continue

            xyxy = to_list(boxes.xyxy[index])
            bbox = tuple(max(0, int(round(value))) for value in xyxy[:4])
            detections.append(Detection(label=label, confidence=confidence, bbox=bbox))

        detections.sort(key=lambda detection: (detection.bbox[1], detection.bbox[0]))
        return detections


def normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", label.lower())


def class_name(names: Any, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, names.get(str(cls_id), cls_id)))

    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return str(names[cls_id])

    return str(cls_id)


def to_scalar(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def to_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return [float(item) for item in value.tolist()]
    return [float(item) for item in value]
