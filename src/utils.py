from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CONFIG: dict[str, Any] = {
    "input_dir": "input",
    "output_dir": "output",
    "render": {
        "dpi": 200,
        "workers": 4,
        "keep_rendered_pages": False,
    },
    "model": {
        "path": "models/doclayout_yolo.pt",
        "auto_download": True,
        "repo_id": "juliozhao/DocLayout-YOLO-DocStructBench",
        "filename": "doclayout_yolo_docstructbench_imgsz1024.pt",
    },
    "detection": {
        "image_size": 1024,
        "confidence": 0.2,
        "iou": 0.5,
        "device": "cpu",
        "figure_labels": ["figure"],
    },
    "crop": {
        "padding": 20,
        "refine": True,
        "whitespace_threshold": 245,
    },
    "quality": {
        "enabled": False,
        "min_width": 64,
        "min_height": 64,
        "min_area": 4096,
        "max_blank_ratio": 0.995,
    },
    "ocr": {
        "enabled": False,
        "language": "en",
        "min_confidence": 0.5,
    },
    "analysis": {
        "enabled": False,
        "mode": "auto",
        "qp_pdf": None,
        "ms_pdf": None,
        "output_json": None,
        "dpi": 200,
        "image_size": 1000,
        "page_plan_enabled": True,
        "workers": 6,
        "qp_workers": None,
        "ms_workers": 6,
        "cleanup_workers": 8,
        "keep_pages": False,
        "cleanup_with_llm": True,
        "page_start": None,
        "page_end": None,
        "ms_page_start": None,
        "ms_page_end": None,
        "llm": {
            "enabled": True,
            "api_key": None,
            "api_key_env": None,
            "base_url": None,
            "base_url_env": None,
            "model": None,
            "model_env": None,
            "dotenv_path": ".env",
            "cache_dir": ".cache/question_analysis",
            "temperature": 0.1,
            "max_retries": 2,
        },
        "ocr": {
            "enabled": False,
            "language": "en",
            "min_confidence": 0.45,
        },
    },
    "html": {
        "enabled": False,
        "output_dir": None,
        "qna_json": None,
        "subject": None,
        "year": None,
        "paper_key": None,
        "group_by_parent": True,
        "copy_images": True,
    },
}


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config and merge it with defaults."""
    config = deepcopy(DEFAULT_CONFIG)

    if not config_path.exists():
        return config

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read config.yaml. Run: pip install -r requirements.txt") from exc

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")

    return deep_merge(config, loaded)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def list_pdfs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Input file is not a PDF: {input_path}")
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    return sorted(path for path in input_path.rglob("*.pdf") if path.is_file())


def make_pdf_output_dir(pdf_path: Path, input_root: Path, output_root: Path) -> Path:
    try:
        relative = pdf_path.relative_to(input_root)
    except ValueError:
        relative = Path(pdf_path.name)

    parts = [sanitize_name(part) for part in relative.with_suffix("").parts]
    return output_root.joinpath(*parts)


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^\w.-]+", "_", value.strip(), flags=re.ASCII)
    value = value.strip("._")
    return value or "untitled"


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")


def relative_to_or_absolute(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def flatten(items: Iterable[Iterable[Any]]) -> list[Any]:
    return [value for group in items for value in group]


class ProgressBar:
    def __init__(self, label: str, total: int, enabled: bool | None = None) -> None:
        self.label = label
        self.total = max(0, int(total))
        self.current = 0
        self.width = 28
        self._lock = threading.Lock()
        self.enabled = (
            enabled
            if enabled is not None
            else self.total > 0 and sys.stderr.isatty() and not os.environ.get("NO_PROGRESS")
        )

    def __enter__(self) -> "ProgressBar":
        self.render()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def update(self, amount: int = 1) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.current = min(self.total, self.current + amount)
            self.render()

    def render(self) -> None:
        if not self.enabled:
            return
        ratio = self.current / self.total if self.total else 1.0
        filled = min(self.width, int(round(self.width * ratio)))
        bar = "#" * filled + "-" * (self.width - filled)
        percent = int(round(ratio * 100))
        sys.stderr.write(f"\r{self.label} [{bar}] {self.current}/{self.total} {percent:3d}%")
        sys.stderr.flush()

    def close(self) -> None:
        if self.enabled:
            sys.stderr.write("\n")
            sys.stderr.flush()
