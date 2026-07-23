from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .utils import ensure_dir

LOGGER = logging.getLogger(__name__)


@dataclass
class CropResult:
    image: Any
    bbox: tuple[int, int, int, int]
    width: int
    height: int
    blank_ratio: float


def extract_figure_crop(
    page_image_path: Path,
    bbox: tuple[int, int, int, int],
    padding: int = 20,
    refine: bool = True,
    whitespace_threshold: int = 245,
) -> CropResult:
    page_image = cv2.imread(str(page_image_path), cv2.IMREAD_COLOR)
    if page_image is None:
        raise RuntimeError(f"Could not read rendered page image: {page_image_path}")

    page_height, page_width = page_image.shape[:2]
    padded_bbox = expand_bbox(bbox, padding, page_width, page_height)
    x1, y1, x2, y2 = padded_bbox

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid detection bbox after padding: {bbox}")

    final_bbox = padded_bbox

    if refine:
        crop = page_image[y1:y2, x1:x2]
        refined_bbox = whitespace_trim_bbox(crop, whitespace_threshold)
        if refined_bbox is not None:
            rx1, ry1, rx2, ry2 = refined_bbox
            final_bbox = expand_bbox(
                (x1 + rx1, y1 + ry1, x1 + rx2, y1 + ry2),
                padding,
                page_width,
                page_height,
            )

    fx1, fy1, fx2, fy2 = final_bbox
    final_crop = page_image[fy1:fy2, fx1:fx2]
    blank_ratio = compute_blank_ratio(final_crop, whitespace_threshold)

    return CropResult(
        image=final_crop,
        bbox=final_bbox,
        width=fx2 - fx1,
        height=fy2 - fy1,
        blank_ratio=blank_ratio,
    )


def save_crop(crop: CropResult, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    ok = cv2.imwrite(str(output_path), crop.image)
    if not ok:
        raise RuntimeError(f"Could not save crop: {output_path}")


def has_structural_content(image: Any, threshold: int = 240) -> bool:
    """Check if image has meaningful structural content (lines, shapes, text)."""
    if image is None or image.size == 0:
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection to find lines and shapes
    edges = cv2.Canny(gray, 30, 100)
    edge_density = np.sum(edges > 0) / edges.size

    # If we have substantial edge content, it's likely a real diagram
    if edge_density > 0.01:  # More than 1% edges indicates structure
        return True

    # Check for text-like regions using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Real diagrams have multiple distinct contours
    if len(contours) >= 3:
        return True

    return False


def is_dotted_writing_space(image: Any, threshold: int = 240) -> bool:
    if image is None or image.size == 0:
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = (gray < threshold).astype(np.uint8)

    height, width = binary.shape
    if width < 50 or height < 15:
        return False

    # Check if this has structural content first
    # If it does, it's NOT a dotted line regardless of dots
    if has_structural_content(image, threshold):
        return False

    # Perform connected components analysis to find dot patterns
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    total_components = 0
    dot_components = 0
    medium_components = 0
    large_components = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Ignore sub-pixel noise
        if area <= 1:
            continue

        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        total_components += 1

        # Check if the component is dot-like (small bounding box and area)
        if w <= 15 and h <= 15 and area <= 60:
            dot_components += 1
        # Track medium-sized features (text labels, diagram elements)
        elif w <= 50 and h <= 50 and area <= 500:
            medium_components += 1
        # Track large features (diagram structures)
        else:
            large_components += 1

    if total_components == 0:
        return False

    dot_ratio = dot_components / total_components

    # Reject if we have substantial non-dot features (diagrams have labels, structures)
    if medium_components >= 5 or large_components >= 3:
        return False

    # Stricter thresholds: require MORE dots (30+) and HIGHER ratio (95%)
    # Real dotted lines have hundreds of uniform dots with no large features
    if total_components >= 30 and dot_ratio >= 0.95:
        return True

    return False


def passes_quality(
    crop: CropResult,
    quality_config: dict[str, object],
    confidence: float = 1.0,
    page_width: int | None = None,
    page_height: int | None = None,
) -> bool:
    if not quality_config.get("enabled", False):
        return True

    # Filter out dotted answer spaces
    if is_dotted_writing_space(crop.image):
        LOGGER.info(
            "Rejecting crop: detected as dotted writing space (dimensions: %dx%d)",
            crop.width,
            crop.height,
        )
        return False

    # Reject if both width and height coverage ratios exceed the max page ratios (e.g. 85%),
    # indicating a full page grid or dotted writing space false positive.
    if page_width is not None and page_height is not None:
        max_w_ratio = float(quality_config.get("max_width_ratio", 1.0))
        max_h_ratio = float(quality_config.get("max_height_ratio", 1.0))
        if (crop.width / page_width > max_w_ratio) and (crop.height / page_height > max_h_ratio):
            return False

    area = crop.width * crop.height
    aspect_ratio = max(crop.width, crop.height) / max(min(crop.width, crop.height), 1)
    return (
        crop.width >= int(quality_config.get("min_width", 1))
        and crop.height >= int(quality_config.get("min_height", 1))
        and area >= int(quality_config.get("min_area", 1))
        and crop.blank_ratio <= float(quality_config.get("max_blank_ratio", 1.0))
        and aspect_ratio <= float(quality_config.get("max_aspect_ratio", 999))
        and confidence >= float(quality_config.get("min_confidence", 0.0))
    )


def expand_bbox(
    bbox: tuple[int, int, int, int],
    padding: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(image_width, x2 + padding),
        min(image_height, y2 + padding),
    )


def whitespace_trim_bbox(image: Any, threshold: int) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (gray < threshold).astype(np.uint8) * 255
    points = cv2.findNonZero(mask)

    if points is None:
        return None

    x, y, width, height = cv2.boundingRect(points)
    return x, y, x + width, y + height


def compute_blank_ratio(image: Any, threshold: int) -> float:
    if image.size == 0:
        return 1.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray >= threshold))
