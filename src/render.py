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
    dpi: int = 400,
    page_start: int | None = None,
    page_end: int | None = None,
) -> Iterator[RenderedPage]:
    """Render PDF pages to PNG files and yield page metadata."""
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for rendering. Run: pip install -r requirements.txt") from exc

    ensure_dir(output_dir)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    document = fitz.open(pdf_path)
    try:
        first_page = max((page_start or 1) - 1, 0)
        last_page = min(page_end or document.page_count, document.page_count)

        for page_index in range(first_page, last_page):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image_path = output_dir / f"page{page_index + 1:03d}.png"
            pixmap.save(image_path)

            yield RenderedPage(
                pdf_path=pdf_path,
                page_number=page_index + 1,
                image_path=image_path,
                width=pixmap.width,
                height=pixmap.height,
                dpi=dpi,
            )
    finally:
        document.close()
