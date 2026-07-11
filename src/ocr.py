from __future__ import annotations

from pathlib import Path
from typing import Any


class PaddleTextExtractor:
    def __init__(self, language: str = "en", min_confidence: float = 0.5) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is required when ocr.enabled is true. Run: pip install -r requirements-ocr.txt"
            ) from exc

        self.min_confidence = min_confidence
        self.engine = PaddleOCR(use_angle_cls=True, lang=language)

    def extract(self, image_path: Path) -> dict[str, Any]:
        raw_result = self.engine.ocr(str(image_path))
        labels: list[dict[str, Any]] = []

        for page_result in raw_result or []:
            for line in page_result or []:
                if len(line) < 2:
                    continue

                text_data = line[1]
                if not isinstance(text_data, (list, tuple)) or len(text_data) < 2:
                    continue

                text = str(text_data[0]).strip()
                confidence = float(text_data[1])
                if text and confidence >= self.min_confidence:
                    labels.append({"text": text, "confidence": confidence})

        return {
            "labels": labels,
            "text": " ".join(label["text"] for label in labels),
        }
