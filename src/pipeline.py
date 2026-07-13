from __future__ import annotations

# Set environment variables to restrict internal PyTorch/OpenMP thread pools to 1 thread.
# This prevents extreme CPU thread contention when executing YOLOv10 inference in parallel threads on CPU.
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .crop import extract_figure_crop, passes_quality, save_crop
from .detect import Detection, DocLayoutFigureDetector
from .ocr import PaddleTextExtractor
from .render import RenderedPage, count_rendered_pages, render_pdf
from .utils import (
    ensure_dir,
    list_pdfs,
    load_config,
    make_pdf_output_dir,
    ProgressBar,
    relative_to_or_absolute,
    resolve_path,
    sanitize_name,
    setup_logging,
    write_json,
)

LOGGER = logging.getLogger(__name__)


def print_pipeline_banner(config: dict[str, Any], args: Any) -> None:
    import sys
    import os
    use_color = (
        not os.environ.get("NO_COLOR")
        and os.environ.get("TERM") != "dumb"
        and sys.stdout.isatty()
    )
    def c(code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if use_color else text

    banner_lines = [
        " ██████╗ ███╗   ██╗ █████╗      ██████╗ ███████╗███╗   ██╗",
        "██╔═══██╗████╗  ██║██╔══██╗    ██╔════╝ ██╔════╝████╗  ██║",
        "██║   ██║██╔██╗ ██║███████║    ██║  ███╗█████╗  ██╔██╗ ██║",
        "██║▄▄ ██║██║╚██╗██║██╔══██║    ██║   ██║██╔══╝  ██║╚██╗██║",
        "╚██████╔╝██║ ╚████║██║  ██║    ╚██████╔╝███████╗██║ ╚████║",
        " ╚══▀▀═╝ ╚═╝  ╚═══╝╚═╝  ╚═╝     ╚═════╝ ╚══════╝╚═╝  ╚═══╝",
    ]
    
    print()
    try:
        for line in banner_lines:
            print(f"  {c('38;5;214', line)}")
        print(f"  {c('1;37', '═══ Exam Diagram Extraction & Q&A Workbook Generator ═══')}")
    except UnicodeEncodeError:
        ascii_banner = [
            "=========================================================",
            "  __  _  _   __      ___  ___  _  _ ",
            " /  \ |\ |  /  \\    /  _  |__  |\\ | ",
            " \\__\\ | \\| /____\\\\   \\\\____ |___ | \\| ",
            "=========================================================",
        ]
        for line in ascii_banner:
            print(f"  {c('38;5;214', line)}")
        print(f"  {c('1;37', '--- Exam Diagram Extraction & Q&A Workbook Generator ---')}")
    print()

    def row(label: str, val: str) -> None:
        print(f"  {c('36', label.ljust(22))} : {c('0', val)}")

    row("Configuration", str(args.config))
    row("Mode", "AUTO (Incremental batch processing)" if getattr(args, "auto", False) else "STANDARD (Single run)")
    row("Input Path", str(config.get("input_dir")))
    row("Output Path", str(config.get("output_dir")))
    row("Batch Size", str((config.get("analysis") or {}).get("batch_size", 1)))
    row("Analysis Backend", str((config.get("analysis") or {}).get("mode", "auto")).upper())
    row("LLM Model", str(((config.get("analysis") or {}).get("llm") or {}).get("model") or "Not configured"))
    row("Device", str((config.get("detection") or {}).get("device", "cpu")).upper())
    row("DPI", f"Render: {config.get('render', {}).get('dpi')} DPI | Analysis: {config.get('analysis', {}).get('dpi')} DPI")
    
    try:
        print(f"  {c('1;37', '═════════════════════════════════════════════════════════')}")
    except UnicodeEncodeError:
        print(f"  {c('1;37', '---------------------------------------------------------')}")
    print()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    config_path = Path(args.config).resolve()
    base_dir = config_path.parent
    config = load_config(config_path)
    apply_cli_overrides(config, args)
    print_pipeline_banner(config, args)

    input_path = resolve_path(config["input_dir"], base_dir)
    output_root = ensure_dir(resolve_path(config["output_dir"], base_dir))
    model_path = resolve_path(config["model"]["path"], base_dir)

    pdfs = list_pdfs(input_path)
    if not pdfs:
        LOGGER.warning("No PDFs found in %s", input_path)
        write_json(output_root / "metadata.json", {"count": 0, "diagrams": [], "errors": []})
        return 0

    if getattr(args, "auto", False):
        from .question_analysis import find_unprocessed_pairs
        from .html_builder import infer_names_from_path
        import json
        from copy import deepcopy

        assets_root = ensure_dir(output_root / "assets")
        unprocessed = find_unprocessed_pairs(pdfs, input_path, output_root, base_dir)
        if not unprocessed:
            LOGGER.info("All paper pairs are already processed.")
            return 0

        LOGGER.info("Found %s unprocessed paper pair(s). Starting auto mode...", len(unprocessed))

        detector = DocLayoutFigureDetector(
            model_path=model_path,
            auto_download=bool(config["model"]["auto_download"]),
            repo_id=str(config["model"]["repo_id"]),
            filename=str(config["model"]["filename"]),
            image_size=int(config["detection"]["image_size"]),
            confidence=float(config["detection"]["confidence"]),
            iou=float(config["detection"]["iou"]) if config["detection"].get("iou") is not None else None,
            device=str(config["detection"]["device"]),
            figure_labels=list(config["detection"]["figure_labels"]),
        )

        ocr_extractor = None
        if config["ocr"].get("enabled", False):
            ocr_extractor = PaddleTextExtractor(
                language=str(config["ocr"].get("language", "en")),
                min_confidence=float(config["ocr"].get("min_confidence", 0.5)),
            )

        all_diagrams: list[dict[str, Any]] = []
        all_errors: list[dict[str, Any]] = []
        metadata_path = assets_root / "metadata.json"

        if metadata_path.exists():
            try:
                with metadata_path.open("r", encoding="utf-8") as file:
                    existing_meta = json.load(file)
                all_diagrams.extend(existing_meta.get("diagrams", []))
                all_errors.extend(existing_meta.get("errors", []))
            except Exception:
                pass

        input_root = input_path if input_path.is_dir() else input_path.parent

        for pair in unprocessed:
            LOGGER.info("Auto-processing pair: %s", pair.question_pdf.name)
            try:
                pair_config = deepcopy(config)
                pair_config["analysis"]["enabled"] = True
                pair_config["analysis"]["mode"] = "llm"
                pair_config["analysis"].setdefault("llm", {})["enabled"] = True
                pair_config["analysis"]["cleanup_with_llm"] = True
                pair_config["analysis"]["qp_pdf"] = str(pair.question_pdf)
                pair_config["analysis"]["ms_pdf"] = str(pair.mark_scheme_pdf) if pair.mark_scheme_pdf else None
                pair_config["html"]["enabled"] = True
                
                subject, year, paper_key = infer_names_from_path(pair.question_pdf)
                pair_config["html"]["subject"] = subject
                pair_config["html"]["year"] = year
                pair_config["html"]["paper_key"] = paper_key

                all_diagrams = [d for d in all_diagrams if d.get("pdf") != pair.question_pdf.name]

                qp_diagrams, qp_errors = process_pdf(pair.question_pdf, input_root, assets_root, pair_config, detector, ocr_extractor)
                all_diagrams.extend(qp_diagrams)
                all_errors.extend(qp_errors)

                write_json(
                    metadata_path,
                    {
                        "count": len(all_diagrams),
                        "diagrams": all_diagrams,
                        "errors": all_errors,
                    },
                )

                qna_json_path = maybe_analyze_questions([pair.question_pdf], input_root, assets_root, metadata_path, pair_config, base_dir)
                if qna_json_path is not None:
                    pair_config["html"]["qna_json"] = str(qna_json_path)

                maybe_build_html(pair_config, base_dir, output_root, [pair.question_pdf], input_root)
                LOGGER.info("Completed auto-processing for: %s", pair.question_pdf.name)

            except Exception as exc:
                LOGGER.exception("Failed to auto-process pair %s", pair.question_pdf.name)
                all_errors.append({"pdf": str(pair.question_pdf), "error": str(exc)})
                write_json(
                    metadata_path,
                    {
                        "count": len(all_diagrams),
                        "diagrams": all_diagrams,
                        "errors": all_errors,
                    },
                )

        LOGGER.info("Auto-processing of all unprocessed papers finished.")
        return 0

    detector = DocLayoutFigureDetector(
        model_path=model_path,
        auto_download=bool(config["model"]["auto_download"]),
        repo_id=str(config["model"]["repo_id"]),
        filename=str(config["model"]["filename"]),
        image_size=int(config["detection"]["image_size"]),
        confidence=float(config["detection"]["confidence"]),
        iou=float(config["detection"]["iou"]) if config["detection"].get("iou") is not None else None,
        device=str(config["detection"]["device"]),
        figure_labels=list(config["detection"]["figure_labels"]),
    )

    ocr_extractor = None
    if config["ocr"].get("enabled", False):
        ocr_extractor = PaddleTextExtractor(
            language=str(config["ocr"].get("language", "en")),
            min_confidence=float(config["ocr"].get("min_confidence", 0.5)),
        )

    all_diagrams: list[dict[str, Any]] = []
    all_errors: list[dict[str, Any]] = []
    input_root = input_path if input_path.is_dir() else input_path.parent

    # Assets folder under output root
    assets_root = ensure_dir(output_root / "assets")

    for pdf_path in pdfs:
        try:
            diagrams, errors = process_pdf(pdf_path, input_root, assets_root, config, detector, ocr_extractor)
            all_diagrams.extend(diagrams)
            all_errors.extend(errors)
        except Exception as exc:
            LOGGER.exception("Failed to process %s", pdf_path)
            all_errors.append({"pdf": str(pdf_path), "error": str(exc)})

    metadata_path = assets_root / "metadata.json"
    write_json(
        metadata_path,
        {
            "count": len(all_diagrams),
            "diagrams": all_diagrams,
            "errors": all_errors,
        },
    )

    qna_json_path = maybe_analyze_questions(pdfs, input_root, assets_root, metadata_path, config, base_dir)
    if qna_json_path is not None and not (config.get("html") or {}).get("qna_json"):
        config.setdefault("html", {})["qna_json"] = str(qna_json_path)

    maybe_build_html(config, base_dir, output_root, pdfs, input_root)
    LOGGER.info("Done. Extracted %s diagrams from %s PDFs", len(all_diagrams), len(pdfs))
    return 0 if not all_errors else 1


def process_pdf(
    pdf_path: Path,
    input_root: Path,
    output_root: Path,
    config: dict[str, Any],
    detector: DocLayoutFigureDetector,
    ocr_extractor: PaddleTextExtractor | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    LOGGER.info("Processing %s", pdf_path)

    pdf_output_dir = ensure_dir(make_pdf_output_dir(pdf_path, input_root, output_root))
    pages_dir = ensure_dir(pdf_output_dir / "pages")
    keep_pages = bool(config["render"].get("keep_rendered_pages", False))
    workers = normalize_worker_count(config["render"].get("workers"), default=1)
    total_pages = count_rendered_pages(
        pdf_path,
        page_start=config.get("page_start"),
        page_end=config.get("page_end"),
    )
    diagrams: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    rendered_pages = render_pdf(
        pdf_path,
        pages_dir,
        dpi=int(config["render"]["dpi"]),
        page_start=config.get("page_start"),
        page_end=config.get("page_end"),
    )

    with ProgressBar(f"Extract {pdf_path.name}", total_pages) as progress:
        if workers <= 1:
            for rendered_page in rendered_pages:
                try:
                    page_diagrams = process_page(
                        rendered_page,
                        pdf_output_dir,
                        pdf_path,
                        input_root,
                        output_root,
                        config,
                        detector,
                        ocr_extractor,
                    )
                    diagrams.extend(page_diagrams)
                except Exception as exc:
                    LOGGER.exception("Failed page %s of %s", rendered_page.page_number, pdf_path)
                    errors.append({"pdf": str(pdf_path), "page": rendered_page.page_number, "error": str(exc)})
                finally:
                    if not keep_pages:
                        rendered_page.image_path.unlink(missing_ok=True)
                    progress.update()
        else:
            LOGGER.info("Processing %s with %s page worker(s)", pdf_path, workers)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        process_page,
                        rendered_page,
                        pdf_output_dir,
                        pdf_path,
                        input_root,
                        output_root,
                        config,
                        detector,
                        ocr_extractor,
                    ): rendered_page
                    for rendered_page in rendered_pages
                }
                for future in as_completed(futures):
                    rendered_page = futures[future]
                    try:
                        diagrams.extend(future.result())
                    except Exception as exc:
                        LOGGER.exception("Failed page %s of %s", rendered_page.page_number, pdf_path)
                        errors.append({"pdf": str(pdf_path), "page": rendered_page.page_number, "error": str(exc)})
                    finally:
                        if not keep_pages:
                            rendered_page.image_path.unlink(missing_ok=True)
                        progress.update()

    diagrams.sort(key=lambda item: (int(item.get("page") or 0), int(item.get("figure") or 0)))

    if not keep_pages:
        try:
            pages_dir.rmdir()
        except OSError:
            pass

    write_json(
        pdf_output_dir / "metadata.json",
        {
            "pdf": pdf_path.name,
            "source_pdf": str(pdf_path),
            "count": len(diagrams),
            "diagrams": diagrams,
            "errors": errors,
        },
    )
    return diagrams, errors


def process_page(
    rendered_page: RenderedPage,
    pdf_output_dir: Path,
    pdf_path: Path,
    input_root: Path,
    output_root: Path,
    config: dict[str, Any],
    detector: DocLayoutFigureDetector,
    ocr_extractor: PaddleTextExtractor | None,
) -> list[dict[str, Any]]:
    detections = detector.detect(rendered_page.image_path)
    page_output_dir = pdf_output_dir / f"page{rendered_page.page_number:03d}"
    diagrams: list[dict[str, Any]] = []

    for detection_index, detection in enumerate(detections, start=1):
        record = crop_detection(
            rendered_page,
            pdf_path,
            input_root,
            output_root,
            page_output_dir,
            detection,
            detection_index,
            config,
            ocr_extractor,
        )
        if record is not None:
            diagrams.append(record)

    return diagrams


def crop_detection(
    rendered_page: RenderedPage,
    pdf_path: Path,
    input_root: Path,
    output_root: Path,
    page_output_dir: Path,
    detection: Detection,
    detection_index: int,
    config: dict[str, Any],
    ocr_extractor: PaddleTextExtractor | None,
) -> dict[str, Any] | None:
    crop = extract_figure_crop(
        rendered_page.image_path,
        detection.bbox,
        padding=int(config["crop"]["padding"]),
        refine=bool(config["crop"].get("refine", True)),
        whitespace_threshold=int(config["crop"].get("whitespace_threshold", 245)),
    )

    if not passes_quality(
        crop,
        config["quality"],
        confidence=detection.confidence,
        page_width=rendered_page.width,
        page_height=rendered_page.height,
    ):
        LOGGER.debug(
            "Rejected low-quality crop on page %s figure %s",
            rendered_page.page_number,
            detection_index,
        )
        return None

    output_path = page_output_dir / f"page{rendered_page.page_number:03d}_fig{detection_index:02d}.png"
    save_crop(crop, output_path)

    record: dict[str, Any] = {
        "pdf": pdf_path.name,
        "source_pdf": relative_to_or_absolute(pdf_path, input_root),
        "page": rendered_page.page_number,
        "figure": detection_index,
        "file": relative_to_or_absolute(output_path, output_root),
        "bbox": list(detection.bbox),
        "crop_bbox": list(crop.bbox),
        "label": detection.label,
        "confidence": detection.confidence,
        "render_dpi": rendered_page.dpi,
        "page_width": rendered_page.width,
        "page_height": rendered_page.height,
        "crop_width": crop.width,
        "crop_height": crop.height,
        "blank_ratio": crop.blank_ratio,
    }

    if ocr_extractor is not None:
        record["ocr"] = ocr_extractor.extract(output_path)

    return record


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.input is not None:
        config["input_dir"] = args.input
    if args.output is not None:
        config["output_dir"] = args.output
    if args.model is not None:
        config["model"]["path"] = args.model
    if args.device is not None:
        config["detection"]["device"] = args.device
    if args.dpi is not None:
        config["render"]["dpi"] = args.dpi
    if args.workers is not None:
        config["render"]["workers"] = args.workers
    if args.confidence is not None:
        config["detection"]["confidence"] = args.confidence
    if args.image_size is not None:
        config["detection"]["image_size"] = args.image_size
    if args.iou is not None:
        config["detection"]["iou"] = args.iou
    if args.keep_pages is not None:
        config["render"]["keep_rendered_pages"] = args.keep_pages
    if args.refine is not None:
        config["crop"]["refine"] = args.refine
    if args.quality is not None:
        config["quality"]["enabled"] = args.quality
    if args.ocr is not None:
        config["ocr"]["enabled"] = args.ocr
    analysis_config = config.setdefault("analysis", {})
    if args.analyze is not None:
        analysis_config["enabled"] = args.analyze
    if args.analysis_mode is not None:
        analysis_config["mode"] = args.analysis_mode
    if args.qp_pdf is not None:
        analysis_config["qp_pdf"] = args.qp_pdf
    if args.ms_pdf is not None:
        analysis_config["ms_pdf"] = args.ms_pdf
    if args.analysis_output is not None:
        analysis_config["output_json"] = args.analysis_output
    if args.analysis_dpi is not None:
        analysis_config["dpi"] = args.analysis_dpi
    if args.analysis_image_size is not None:
        analysis_config["image_size"] = args.analysis_image_size
    if args.page_plan is not None:
        analysis_config["page_plan_enabled"] = args.page_plan
    if args.analysis_workers is not None:
        analysis_config["workers"] = args.analysis_workers
    if args.qp_workers is not None:
        analysis_config["qp_workers"] = args.qp_workers
    if args.ms_workers is not None:
        analysis_config["ms_workers"] = args.ms_workers
    if args.cleanup_workers is not None:
        analysis_config["cleanup_workers"] = args.cleanup_workers
    if args.keep_analysis_pages is not None:
        analysis_config["keep_pages"] = args.keep_analysis_pages
    if args.cleanup_with_llm is not None:
        analysis_config["cleanup_with_llm"] = args.cleanup_with_llm
    if args.analysis_page_start is not None:
        analysis_config["page_start"] = args.analysis_page_start
    if args.analysis_page_end is not None:
        analysis_config["page_end"] = args.analysis_page_end
    if args.ms_page_start is not None:
        analysis_config["ms_page_start"] = args.ms_page_start
    if args.ms_page_end is not None:
        analysis_config["ms_page_end"] = args.ms_page_end
    llm_config = analysis_config.setdefault("llm", {})
    if args.llm_model is not None:
        llm_config["model"] = args.llm_model
    if args.llm_base_url is not None:
        llm_config["base_url"] = args.llm_base_url
    if args.llm_api_key_env is not None:
        llm_config["api_key_env"] = args.llm_api_key_env
    analysis_ocr_config = analysis_config.setdefault("ocr", {})
    if args.analysis_ocr is not None:
        analysis_ocr_config["enabled"] = args.analysis_ocr
    html_config = config.setdefault("html", {})
    if args.html is not None:
        html_config["enabled"] = args.html
    if args.html_output is not None:
        html_config["output_dir"] = args.html_output
    if args.qna_json is not None:
        html_config["qna_json"] = args.qna_json
    if args.subject is not None:
        html_config["subject"] = args.subject
    if args.year is not None:
        html_config["year"] = args.year
    if args.paper_key is not None:
        html_config["paper_key"] = args.paper_key
    if args.html_group_by_parent is not None:
        html_config["group_by_parent"] = args.html_group_by_parent
    if args.html_copy_images is not None:
        html_config["copy_images"] = args.html_copy_images
    if args.page_start is not None:
        config["page_start"] = args.page_start
    if args.page_end is not None:
        config["page_end"] = args.page_end


def maybe_analyze_questions(
    pdfs: list[Path],
    input_root: Path,
    output_root: Path,
    metadata_path: Path,
    config: dict[str, Any],
    base_dir: Path,
) -> Path | None:
    analysis_config = config.get("analysis") or {}
    if not analysis_config.get("enabled", False):
        return None

    from .question_analysis import run_question_analysis

    return run_question_analysis(
        pdfs=pdfs,
        input_root=input_root,
        output_root=output_root,
        metadata_path=metadata_path,
        config=config,
        base_dir=base_dir,
    )


def maybe_build_html(config: dict[str, Any], base_dir: Path, output_root: Path, pdfs: list[Path], input_root: Path) -> None:
    html_config = config.get("html") or {}
    if not html_config.get("enabled", False):
        return

    from .html_builder import build_from_metadata, build_from_qna_json

    # Determine paper name from processed PDFs
    if pdfs:
        pdf_path = pdfs[0]
        try:
            relative = pdf_path.relative_to(input_root)
            paper_name = sanitize_name(relative.parts[0])
        except ValueError:
            paper_name = sanitize_name(pdf_path.stem)
    else:
        paper_name = "default"

    output_value = html_config.get("output_dir")
    if output_value:
        html_output_dir = ensure_dir(resolve_path(output_value, base_dir))
    else:
        html_output_dir = ensure_dir(output_root / "htmls" / paper_name)

    qna_json = html_config.get("qna_json")

    common_options = {
        "output_dir": html_output_dir,
        "subject_name": optional_str(html_config.get("subject")),
        "year": optional_str(html_config.get("year")),
        "paper_key": optional_str(html_config.get("paper_key")),
        "copy_images": bool(html_config.get("copy_images", True)),
    }

    if qna_json:
        summary = build_from_qna_json(
            resolve_path(qna_json, base_dir),
            group_by_parent=bool(html_config.get("group_by_parent", True)),
            **common_options,
        )
    else:
        assets_root = output_root / "assets"
        summary = build_from_metadata(assets_root / "metadata.json", **common_options)

    LOGGER.info("HTML dashboard generated: %s", summary["dashboard"])


def optional_str(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None


def normalize_worker_count(value: Any, default: int = 1) -> int:
    try:
        workers = int(value)
    except (TypeError, ValueError):
        workers = default
    return max(1, workers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract diagram crops from educational PDFs.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--input", help="PDF file or folder of PDFs")
    parser.add_argument("--output", help="Output folder")
    parser.add_argument("--model", help="Path to DocLayout-YOLO .pt model")
    parser.add_argument("--device", help="Inference device, e.g. cpu or cuda:0")
    parser.add_argument("--dpi", type=int, help="PDF render DPI")
    parser.add_argument("--workers", type=int, help="Parallel page workers for diagram extraction")
    parser.add_argument("--confidence", type=float, help="Detection confidence threshold")
    parser.add_argument("--image-size", type=int, help="DocLayout-YOLO inference image size")
    parser.add_argument("--iou", type=float, help="Detection IoU threshold")
    parser.add_argument("--page-start", type=int, help="First 1-based page to process")
    parser.add_argument("--page-end", type=int, help="Last 1-based page to process")
    parser.add_argument("--keep-pages", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--refine", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--quality", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ocr", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--analyze", action=argparse.BooleanOptionalAction, default=None, help="Extract question/answer structure")
    parser.add_argument("--analysis-mode", choices=["auto", "llm", "ocr"], help="Question analysis backend")
    parser.add_argument("--qp-pdf", help="Question paper PDF for analysis")
    parser.add_argument("--ms-pdf", help="Mark scheme PDF for analysis")
    parser.add_argument("--analysis-output", help="Path for extracted_qna.json")
    parser.add_argument("--analysis-dpi", type=int, help="DPI for analysis page images")
    parser.add_argument("--analysis-image-size", type=int, help="Square pixel size for analysis page images")
    parser.add_argument("--page-plan", action=argparse.BooleanOptionalAction, default=None, help="Plan QP page numbering before extraction")
    parser.add_argument("--analysis-workers", type=int, help="Default parallel workers for analysis page LLM/OCR calls")
    parser.add_argument("--qp-workers", type=int, help="Question-paper page workers; keep at 1 for best numbering")
    parser.add_argument("--ms-workers", type=int, help="Mark-scheme page workers")
    parser.add_argument("--cleanup-workers", type=int, help="Parallel workers for optional LLM cleanup")
    parser.add_argument("--keep-analysis-pages", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--cleanup-with-llm", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--analysis-page-start", type=int, help="First QP page to analyze")
    parser.add_argument("--analysis-page-end", type=int, help="Last QP page to analyze")
    parser.add_argument("--ms-page-start", type=int, help="First MS page to analyze")
    parser.add_argument("--ms-page-end", type=int, help="Last MS page to analyze")
    parser.add_argument("--llm-model", help="OpenAI-compatible model for LLM analysis")
    parser.add_argument("--llm-base-url", help="OpenAI-compatible API base URL")
    parser.add_argument("--llm-api-key-env", help="Environment variable containing the LLM API key")
    parser.add_argument("--analysis-ocr", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--html", action=argparse.BooleanOptionalAction, default=None, help="Generate static HTML output")
    parser.add_argument("--html-output", help="HTML output folder; defaults to <output>/html")
    parser.add_argument("--qna-json", help="Structured QNA JSON to render instead of diagram metadata")
    parser.add_argument("--subject", help="Subject/title for generated HTML")
    parser.add_argument("--year", help="Year label for generated HTML")
    parser.add_argument("--paper-key", help="Paper key label for generated HTML")
    parser.add_argument("--html-group-by-parent", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--html-copy-images", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--auto", action="store_true", help="Auto-detect and process all unprocessed QP/MS pairs")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n\nExecution cancelled by user.")
        sys.exit(130)
