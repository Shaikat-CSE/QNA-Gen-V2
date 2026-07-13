from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .utils import ensure_dir


@dataclass(frozen=True)
class RenderedPage:
    pdf_path: Path
    page_number: int
    image_path: Path
    width: int
    height: int
    dpi: int


def render_pdf(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 200,
    page_start: int | None = None,
    page_end: int | None = None,
    target_size: tuple[int, int] | None = None,
) -> Iterator[RenderedPage]:
    """Render PDF pages to PNG files and yield page metadata."""
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for rendering. Run: pip install -r requirements.txt") from exc

    ensure_dir(output_dir)
    document = fitz.open(pdf_path)
    try:
        first_page = max((page_start or 1) - 1, 0)
        last_page = min(page_end or document.page_count, document.page_count)

        for page_index in range(first_page, last_page):
            page = document.load_page(page_index)

            if target_size is not None:
                max_dim_pts = max(page.rect.width, page.rect.height)
                target_max_dim = max(target_size)
                dynamic_dpi = (target_max_dim / max_dim_pts) * 72.0
                zoom = dynamic_dpi / 72.0
            else:
                zoom = dpi / 72.0

            matrix = fitz.Matrix(zoom, zoom)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image_path = output_dir / f"page{page_index + 1:03d}.png"

            if target_size is not None:
                import cv2
                import numpy as np
                # Convert PyMuPDF pixmap to numpy array directly (RGB)
                img = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape((pixmap.height, pixmap.width, 3))
                # Convert RGB to BGR for cv2.imwrite
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                resized = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(image_path), resized)
                w, h = target_size
            else:
                pixmap.save(image_path)
                w, h = pixmap.width, pixmap.height

            yield RenderedPage(
                pdf_path=pdf_path,
                page_number=page_index + 1,
                image_path=image_path,
                width=w,
                height=h,
                dpi=dpi if target_size is None else int(dynamic_dpi),
            )
    finally:
        document.close()


def count_rendered_pages(
    pdf_path: Path,
    page_start: int | None = None,
    page_end: int | None = None,
) -> int:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for rendering. Run: pip install -r requirements.txt") from exc

    document = fitz.open(pdf_path)
    try:
        first_page = max((page_start or 1) - 1, 0)
        last_page = min(page_end or document.page_count, document.page_count)
        return max(0, last_page - first_page)
    finally:
        document.close()
