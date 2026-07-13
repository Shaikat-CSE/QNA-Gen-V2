from __future__ import annotations

import base64
import hashlib
import html
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .render import count_rendered_pages, render_pdf
from .utils import ProgressBar, ensure_dir, make_pdf_output_dir, relative_to_or_absolute, resolve_path, sanitize_name, write_json

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExamPair:
    question_pdf: Path
    mark_scheme_pdf: Path | None


def run_question_analysis(
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

    pairs = find_exam_pairs(pdfs, analysis_config, base_dir)
    if not pairs:
        raise RuntimeError(
            "Question analysis is enabled, but no QP/MS pair was found. "
            "Set analysis.qp_pdf and analysis.ms_pdf, or use filenames containing qp/ms."
        )

    metadata = load_json(metadata_path) if metadata_path.exists() else {"diagrams": []}
    all_outputs: list[dict[str, Any]] = []
    last_qna_path: Path | None = None

    for pair in pairs:
        result = analyze_exam_pair(pair, input_root, output_root, metadata, config, base_dir)
        all_outputs.append(result)
        last_qna_path = Path(result["output_json"])

    manifest_path = output_root / "analysis_manifest.json"
    write_json(
        manifest_path,
        {
            "count": len(all_outputs),
            "outputs": all_outputs,
        },
    )
    LOGGER.info("Question analysis manifest written: %s", manifest_path)
    return last_qna_path


def analyze_exam_pair(
    pair: ExamPair,
    input_root: Path,
    output_root: Path,
    metadata: dict[str, Any],
    config: dict[str, Any],
    base_dir: Path,
) -> dict[str, Any]:
    analysis_config = config.get("analysis") or {}
    qp_output_dir = ensure_dir(make_pdf_output_dir(pair.question_pdf, input_root, output_root))
    analysis_output_dir = ensure_dir(qp_output_dir / "analysis")
    qna_json_path = resolve_analysis_output_path(analysis_config, analysis_output_dir, base_dir)

    diagrams = collect_pdf_diagrams(metadata, pair.question_pdf, output_root)
    diagrams_by_page = group_diagrams_by_page(diagrams)
    mode = str(analysis_config.get("mode", "auto")).lower()
    dpi = int(analysis_config.get("dpi", 200))
    keep_pages = bool(analysis_config.get("keep_pages", False))
    image_size = normalize_int_or_none(analysis_config.get("image_size")) or 1000
    page_plan_enabled = bool(analysis_config.get("page_plan_enabled", True))
    workers = normalize_worker_count(analysis_config.get("workers"), default=1)
    default_qp_workers = workers if page_plan_enabled else 1
    qp_workers = normalize_worker_count(analysis_config.get("qp_workers"), default=default_qp_workers)
    ms_workers = normalize_worker_count(analysis_config.get("ms_workers"), default=workers)
    cleanup_workers = normalize_worker_count(
        analysis_config.get("cleanup_workers"),
        default=workers,
    )

    llm_client = make_llm_client(analysis_config, base_dir)
    ocr_engine = make_page_ocr(analysis_config)

    LOGGER.info("Analyzing question paper: %s", pair.question_pdf)
    questions = analyze_question_pdf(
        pair.question_pdf,
        analysis_output_dir,
        diagrams_by_page,
        mode=mode,
        dpi=dpi,
        keep_pages=keep_pages,
        llm_client=llm_client,
        ocr_engine=ocr_engine,
        config=analysis_config,
        workers=qp_workers,
        image_size=image_size,
        page_plan_enabled=page_plan_enabled,
    )

    answers: list[dict[str, Any]] = []
    if pair.mark_scheme_pdf is not None:
        LOGGER.info("Analyzing mark scheme: %s", pair.mark_scheme_pdf)
        answers = analyze_mark_scheme_pdf(
            pair.mark_scheme_pdf,
            analysis_output_dir,
            mode=mode,
            dpi=dpi,
            keep_pages=keep_pages,
            llm_client=llm_client,
            ocr_engine=ocr_engine,
            config=analysis_config,
            workers=ms_workers,
            image_size=image_size,
        )
    else:
        LOGGER.warning("No mark scheme PDF available for %s", pair.question_pdf.name)

    questions = bind_diagrams_to_questions(questions, diagrams_by_page)
    qnas = match_questions_to_answers(questions, answers)

    if llm_client and bool(analysis_config.get("cleanup_with_llm", True)):
        qnas = cleanup_qnas_with_llm(qnas, llm_client, workers=cleanup_workers)

    payload = {
        "source": {
            "question_pdf": str(pair.question_pdf),
            "mark_scheme_pdf": str(pair.mark_scheme_pdf) if pair.mark_scheme_pdf else None,
        },
        "questions": questions,
        "answers": answers,
        "qnas": qnas,
    }
    write_json(qna_json_path, payload)
    LOGGER.info("Structured QNA JSON written: %s", qna_json_path)

    return {
        "question_pdf": str(pair.question_pdf),
        "mark_scheme_pdf": str(pair.mark_scheme_pdf) if pair.mark_scheme_pdf else None,
        "output_json": str(qna_json_path),
        "questions": len(questions),
        "answers": len(answers),
        "qnas": len(qnas),
    }


def analyze_question_pdf(
    pdf_path: Path,
    output_dir: Path,
    diagrams_by_page: dict[int, list[dict[str, Any]]],
    mode: str,
    dpi: int,
    keep_pages: bool,
    llm_client: "OpenAICompatibleVisionClient | None",
    ocr_engine: "PageOCREngine | None",
    config: dict[str, Any],
    workers: int = 1,
    image_size: int = 1000,
    page_plan_enabled: bool = True,
) -> list[dict[str, Any]]:
    pages_dir = ensure_dir(output_dir / "qp_pages")
    questions: list[dict[str, Any]] = []
    active_question = "1"
    total_pages = count_rendered_pages(
        pdf_path,
        page_start=config.get("page_start"),
        page_end=config.get("page_end"),
    )
    rendered_pages = list(render_pdf(
        pdf_path,
        pages_dir,
        dpi=dpi,
        page_start=config.get("page_start"),
        page_end=config.get("page_end"),
        target_size=(image_size, image_size) if image_size > 0 else None,
    ))
    rendered_pages = normalize_rendered_pages_to_square(rendered_pages, image_size)
    page_plans = (
        plan_question_pages(rendered_pages, llm_client, workers=workers)
        if page_plan_enabled and llm_client is not None and choose_backend(mode, llm_client, ocr_engine) == "llm"
        else {}
    )

    batch_size = max(1, int(config.get("batch_size", 1)))
    backend = choose_backend(mode, llm_client, ocr_engine)

    if batch_size > 1 and backend == "llm" and llm_client is not None:
        batches = [
            rendered_pages[i : i + batch_size]
            for i in range(0, len(rendered_pages), batch_size)
        ]

        def process_batch(batch_pages: list[Any], active_q_start: str) -> tuple[list[dict[str, Any]], str]:
            image_paths = [p.image_path for p in batch_pages]
            page_numbers = [p.page_number for p in batch_pages]
            plans = {p.page_number: page_plans.get(p.page_number, {}) for p in batch_pages}
            
            batch_questions = llm_client.analyze_question_pages_batch(
                image_paths=image_paths,
                page_numbers=page_numbers,
                active_question=active_q_start,
                diagrams_by_page=diagrams_by_page,
                page_plans=plans,
            )
            
            normalized_list = []
            for question in batch_questions:
                p_num = normalize_int_or_none(question.get("page"))
                if p_num not in page_numbers:
                    p_num = page_numbers[0]
                
                p_diagrams = diagrams_by_page.get(p_num, [])
                norm_q = normalize_question_block(question, p_num, p_diagrams)
                normalized_list.append(norm_q)
                
            next_active = active_q_start
            for q in normalized_list:
                q_num = str(q.get("question_number", ""))
                match = re.match(r"^(\d+)", q_num)
                if match:
                    next_active = match.group(1)
            
            normalized_list = fix_orphan_subparts(normalized_list, next_active)
            return normalized_list, next_active

        with ProgressBar(f"Analyze QP {pdf_path.name}", total_pages) as progress:
            if workers <= 1:
                for batch in batches:
                    first_page_plan = page_plans.get(batch[0].page_number)
                    active_question = planned_active_question(first_page_plan, active_question)
                    try:
                        normalized, active_question = process_batch(batch, active_question)
                        questions.extend(normalized)
                        LOGGER.debug(
                            "QP pages %s analyzed in batch with LLM: %s question block(s)",
                            [p.page_number for p in batch],
                            len(normalized),
                        )
                    except Exception as exc:
                        LOGGER.exception("Failed QP batch %s", [p.page_number for p in batch])
                    finally:
                        if not keep_pages:
                            for rp in batch:
                                rp.image_path.unlink(missing_ok=True)
                        progress.update(len(batch))
            else:
                LOGGER.info("Analyzing QP pages in batches with %s worker(s)", workers)
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(
                            process_batch,
                            batch,
                            planned_active_question(page_plans.get(batch[0].page_number), "unknown")
                        ): batch
                        for batch in batches
                    }
                    for future in as_completed(futures):
                        batch = futures[future]
                        try:
                            normalized, _ = future.result()
                            questions.extend(normalized)
                            LOGGER.debug(
                                "QP pages %s analyzed in batch with LLM: %s question block(s)",
                                [p.page_number for p in batch],
                                len(normalized),
                            )
                        except Exception as exc:
                            LOGGER.exception("Failed QP batch %s", [p.page_number for p in batch])
                        finally:
                            if not keep_pages:
                                for rp in batch:
                                    rp.image_path.unlink(missing_ok=True)
                            progress.update(len(batch))
    else:
        with ProgressBar(f"Analyze QP {pdf_path.name}", total_pages) as progress:
            if workers <= 1:
                for rendered_page in rendered_pages:
                    page_plan = page_plans.get(rendered_page.page_number)
                    active_question = planned_active_question(page_plan, active_question)
                    normalized, active_question, backend = analyze_question_page_render(
                        rendered_page,
                        diagrams_by_page,
                        mode,
                        llm_client,
                        ocr_engine,
                        active_question,
                        keep_pages,
                        page_plan=page_plan,
                    )
                    questions.extend(normalized)
                    LOGGER.debug(
                        "QP page %s analyzed with %s backend: %s question block(s)",
                        rendered_page.page_number,
                        backend,
                        len(normalized),
                    )
                    progress.update()
            else:
                LOGGER.info("Analyzing QP pages with %s worker(s)", workers)
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(
                            analyze_question_page_render,
                            rendered_page,
                            diagrams_by_page,
                            mode,
                            llm_client,
                            ocr_engine,
                            planned_active_question(page_plans.get(rendered_page.page_number), "unknown"),
                            keep_pages,
                            page_plan=page_plans.get(rendered_page.page_number),
                        ): rendered_page
                        for rendered_page in rendered_pages
                    }
                    for future in as_completed(futures):
                        rendered_page = futures[future]
                        normalized, _active_question, backend = future.result()
                        questions.extend(normalized)
                        LOGGER.debug(
                            "QP page %s analyzed with %s backend: %s question block(s)",
                            rendered_page.page_number,
                            backend,
                            len(normalized),
                        )
                        progress.update()

    if not keep_pages:
        try_remove_dir(pages_dir)

    return sort_blocks(questions)


def normalize_rendered_pages_to_square(rendered_pages: list[Any], image_size: int) -> list[Any]:
    if image_size <= 0:
        return rendered_pages

    normalized_pages = []
    for rendered_page in rendered_pages:
        if rendered_page.width != image_size or rendered_page.height != image_size:
            normalize_page_image_to_square(rendered_page.image_path, image_size)
            normalized_pages.append(replace(rendered_page, width=image_size, height=image_size))
        else:
            normalized_pages.append(rendered_page)
    return normalized_pages


def normalize_page_image_to_square(image_path: Path, image_size: int) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for analysis image resizing") from exc

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read analysis page image: {image_path}")

    # Directly resize (stretch) to image_size x image_size to match coordinate mapping
    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)

    if not cv2.imwrite(str(image_path), resized):
        raise RuntimeError(f"Could not save normalized analysis page image: {image_path}")


def plan_question_pages(
    rendered_pages: list[Any],
    llm_client: "OpenAICompatibleVisionClient",
    workers: int = 1,
) -> dict[int, dict[str, Any]]:
    # Try batched sequence planning first
    try:
        LOGGER.info("Attempting batched sequence planning for %s pages...", len(rendered_pages))
        image_paths = [rp.image_path for rp in rendered_pages]
        page_numbers = [rp.page_number for rp in rendered_pages]
        raw_plans = llm_client.analyze_question_pages_plan(image_paths, page_numbers)
    except Exception as exc:
        LOGGER.warning("Batched sequence planning failed: %s. Falling back to page-by-page planning.", exc)
        raw_plans = {}

    # If some pages are missing from the batched response, fetch them individually
    missing_pages = [rp for rp in rendered_pages if rp.page_number not in raw_plans]
    if missing_pages:
        LOGGER.info("Fetching plans for %s missing pages individually...", len(missing_pages))
        def plan_single_page(rendered_page: Any) -> tuple[int, dict[str, Any]]:
            plan = llm_client.analyze_question_page_plan(
                rendered_page.image_path,
                rendered_page.page_number,
                "previous_active",
            )
            return rendered_page.page_number, plan

        with ProgressBar("Plan QP sequence", len(missing_pages)) as progress:
            if workers <= 1 or len(missing_pages) == 1:
                for rendered_page in missing_pages:
                    try:
                        page_number, plan = plan_single_page(rendered_page)
                        raw_plans[page_number] = plan
                    except Exception as exc:
                        LOGGER.warning("QP page planning failed for page %s: %s", rendered_page.page_number, exc)
                    finally:
                        progress.update()
            else:
                LOGGER.info("Planning remaining QP sequence with %s worker(s)", workers)
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(plan_single_page, rendered_page): rendered_page
                        for rendered_page in missing_pages
                    }
                    for future in as_completed(futures):
                        rendered_page = futures[future]
                        try:
                            page_number, plan = future.result()
                            raw_plans[page_number] = plan
                        except Exception as exc:
                            LOGGER.warning("QP page planning failed for page %s: %s", rendered_page.page_number, exc)
                        finally:
                            progress.update()

    page_plans: dict[int, dict[str, Any]] = {}
    active_question = "1"
    for rendered_page in rendered_pages:
        plan = raw_plans.get(rendered_page.page_number)
        plan = normalize_page_plan(plan, rendered_page.page_number, active_question)
        active_question = planned_active_question(plan, active_question)
        page_plans[rendered_page.page_number] = plan

    return page_plans


ROMAN_RE = re.compile(r"^[ivx]+$", re.I)

def is_roman_numeral(value: str) -> bool:
    return bool(ROMAN_RE.match(value))


def parse_label_components(label: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", label)


def format_label_components(components: list[str]) -> str:
    if not components:
        return ""
    result = components[0]
    for comp in components[1:]:
        result += f"({comp})"
    return result


def resolve_relative_question_label(relative: str, base: str) -> str:
    relative_clean = normalize_question_label(relative)
    if not relative_clean or relative_clean in {"previousactive", "previous_active"}:
        return base

    rel_comps = parse_label_components(relative_clean)
    base_comps = parse_label_components(base)

    if not rel_comps:
        return base
    if not base_comps:
        return relative_clean

    first_rel = rel_comps[0]

    # If it is a placeholder word like 'previous' or 'active', ignore and return base
    if first_rel in {"previous", "active", "previousactive"}:
        # Check if there are other comps appended
        if len(rel_comps) > 2:
            rel_comps = rel_comps[2:]
            first_rel = rel_comps[0]
        else:
            return base

    if first_rel.isdigit():
        return format_label_components(rel_comps)

    if first_rel.isalpha() and not is_roman_numeral(first_rel):
        main_num = base_comps[0] if base_comps[0].isdigit() else "1"
        return format_label_components([main_num] + rel_comps)

    if is_roman_numeral(first_rel):
        prefix_comps = []
        if base_comps[0].isdigit():
            prefix_comps.append(base_comps[0])
            if len(base_comps) > 1 and base_comps[1].isalpha() and not is_roman_numeral(base_comps[1]):
                prefix_comps.append(base_comps[1])
        if not prefix_comps:
            prefix_comps = ["1", "a"]
        return format_label_components(prefix_comps + rel_comps)

    return relative_clean


def normalize_page_plan(plan: dict[str, Any] | None, page_number: int, fallback_active_question: str) -> dict[str, Any]:
    visible = plan.get("visible_question_numbers") if isinstance(plan, dict) else []
    if not isinstance(visible, list):
        visible = []
    visible = [normalize_question_label(str(item)) for item in visible if str(item).strip()]

    raw_active = str(plan.get("active_parent_question") or "") if isinstance(plan, dict) else ""
    active = resolve_relative_question_label(raw_active, fallback_active_question)
    
    if not active or not active[0].isdigit():
        active = infer_active_question_from_numbers(visible) or fallback_active_question

    return {
        "page_number": page_number,
        "active_parent_question": active,
        "visible_question_numbers": visible,
        "continues_previous_question": bool(plan.get("continues_previous_question", False)) if isinstance(plan, dict) else False,
        "notes": str(plan.get("notes") or "") if isinstance(plan, dict) else "",
    }


def planned_active_question(page_plan: dict[str, Any] | None, fallback: str) -> str:
    if not page_plan:
        return fallback
    active = normalize_question_label(str(page_plan.get("active_parent_question") or ""))
    return active or fallback


def infer_active_question_from_numbers(question_numbers: list[str]) -> str | None:
    for value in question_numbers:
        match = re.match(r"^(\d+)", value)
        if match:
            return match.group(1)
    return None


def analyze_question_page_render(
    rendered_page: Any,
    diagrams_by_page: dict[int, list[dict[str, Any]]],
    mode: str,
    llm_client: "OpenAICompatibleVisionClient | None",
    ocr_engine: "PageOCREngine | None",
    active_question: str,
    keep_pages: bool,
    page_plan: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str, str]:
    backend = choose_backend(mode, llm_client, ocr_engine)
    page_diagrams = diagrams_by_page.get(rendered_page.page_number, [])

    try:
        if backend == "llm" and llm_client is not None:
            page_questions = llm_client.analyze_question_page(
                rendered_page.image_path,
                rendered_page.page_number,
                active_question,
                page_diagrams,
                page_plan=page_plan,
            )
        elif backend == "ocr" and ocr_engine is not None:
            ocr_lines = ocr_engine.extract_lines(rendered_page.image_path)
            page_questions = heuristic_questions_from_lines(
                ocr_lines,
                rendered_page.page_number,
                page_diagrams,
                source="ocr",
            )
        else:
            page_questions = []

        normalized = [
            normalize_question_block(question, rendered_page.page_number, page_diagrams)
            for question in page_questions
        ]
        next_active_question = active_question
        for question in normalized:
            q_num = str(question.get("question_number", ""))
            match = re.match(r"^(\d+)", q_num)
            if match:
                next_active_question = match.group(1)
        normalized = fix_orphan_subparts(normalized, next_active_question)
        return normalized, next_active_question, backend
    finally:
        if not keep_pages:
            rendered_page.image_path.unlink(missing_ok=True)


def analyze_mark_scheme_pdf(
    pdf_path: Path,
    output_dir: Path,
    mode: str,
    dpi: int,
    keep_pages: bool,
    llm_client: "OpenAICompatibleVisionClient | None",
    ocr_engine: "PageOCREngine | None",
    config: dict[str, Any],
    workers: int = 1,
    image_size: int = 1000,
) -> list[dict[str, Any]]:
    pages_dir = ensure_dir(output_dir / "ms_pages")
    answers: list[dict[str, Any]] = []
    total_pages = count_rendered_pages(
        pdf_path,
        page_start=config.get("ms_page_start"),
        page_end=config.get("ms_page_end"),
    )
    rendered_pages = list(render_pdf(
        pdf_path,
        pages_dir,
        dpi=dpi,
        page_start=config.get("ms_page_start"),
        page_end=config.get("ms_page_end"),
        target_size=(image_size, image_size) if image_size > 0 else None,
    ))
    rendered_pages = normalize_rendered_pages_to_square(rendered_pages, image_size)

    batch_size = max(1, int(config.get("batch_size", 1)))
    backend = choose_backend(mode, llm_client, ocr_engine)

    if batch_size > 1 and backend == "llm" and llm_client is not None:
        batches = [
            rendered_pages[i : i + batch_size]
            for i in range(0, len(rendered_pages), batch_size)
        ]

        def process_ms_batch(batch_pages: list[Any]) -> list[dict[str, Any]]:
            image_paths = [p.image_path for p in batch_pages]
            page_numbers = [p.page_number for p in batch_pages]
            batch_answers = llm_client.analyze_mark_scheme_pages_batch(
                image_paths=image_paths,
                page_numbers=page_numbers,
            )
            
            normalized_list = []
            for answer in batch_answers:
                p_num = normalize_int_or_none(answer.get("page"))
                if p_num not in page_numbers:
                    p_num = page_numbers[0]
                norm_a = normalize_answer_block(answer, p_num)
                normalized_list.append(norm_a)
            return normalized_list

        with ProgressBar(f"Analyze MS {pdf_path.name}", total_pages) as progress:
            if workers <= 1:
                for batch in batches:
                    try:
                        normalized = process_ms_batch(batch)
                        answers.extend(normalized)
                        LOGGER.debug(
                            "MS pages %s analyzed in batch with LLM: %s answer block(s)",
                            [p.page_number for p in batch],
                            len(normalized),
                        )
                    except Exception as exc:
                        LOGGER.exception("Failed MS batch %s", [p.page_number for p in batch])
                    finally:
                        if not keep_pages:
                            for rp in batch:
                                rp.image_path.unlink(missing_ok=True)
                        progress.update(len(batch))
            else:
                LOGGER.info("Analyzing MS pages in batches with %s worker(s)", workers)
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(process_ms_batch, batch): batch
                        for batch in batches
                    }
                    for future in as_completed(futures):
                        batch = futures[future]
                        try:
                            normalized = future.result()
                            answers.extend(normalized)
                            LOGGER.debug(
                                "MS pages %s analyzed in batch with LLM: %s answer block(s)",
                                [p.page_number for p in batch],
                                len(normalized),
                            )
                        except Exception as exc:
                            LOGGER.exception("Failed MS batch %s", [p.page_number for p in batch])
                        finally:
                            if not keep_pages:
                                for rp in batch:
                                    rp.image_path.unlink(missing_ok=True)
                            progress.update(len(batch))
    else:
        with ProgressBar(f"Analyze MS {pdf_path.name}", total_pages) as progress:
            if workers <= 1:
                for rendered_page in rendered_pages:
                    normalized, backend = analyze_mark_scheme_page_render(
                        rendered_page,
                        mode,
                        llm_client,
                        ocr_engine,
                        keep_pages,
                    )
                    answers.extend(normalized)
                    LOGGER.debug(
                        "MS page %s analyzed with %s backend: %s answer block(s)",
                        rendered_page.page_number,
                        backend,
                        len(normalized),
                    )
                    progress.update()
            else:
                LOGGER.info("Analyzing MS pages with %s worker(s)", workers)
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(
                            analyze_mark_scheme_page_render,
                            rendered_page,
                            mode,
                            llm_client,
                            ocr_engine,
                            keep_pages,
                        ): rendered_page
                        for rendered_page in rendered_pages
                    }
                    for future in as_completed(futures):
                        rendered_page = futures[future]
                        normalized, backend = future.result()
                        answers.extend(normalized)
                        LOGGER.debug(
                            "MS page %s analyzed with %s backend: %s answer block(s)",
                            rendered_page.page_number,
                            backend,
                            len(normalized),
                        )
                        progress.update()

    if not keep_pages:
        try_remove_dir(pages_dir)

    return dedupe_answer_blocks(sort_blocks(answers))


def analyze_mark_scheme_page_render(
    rendered_page: Any,
    mode: str,
    llm_client: "OpenAICompatibleVisionClient | None",
    ocr_engine: "PageOCREngine | None",
    keep_pages: bool,
) -> tuple[list[dict[str, Any]], str]:
    backend = choose_backend(mode, llm_client, ocr_engine)

    try:
        if backend == "llm" and llm_client is not None:
            page_answers = llm_client.analyze_mark_scheme_page(rendered_page.image_path, rendered_page.page_number)
        elif backend == "ocr" and ocr_engine is not None:
            ocr_lines = ocr_engine.extract_lines(rendered_page.image_path)
            page_answers = heuristic_answers_from_lines(ocr_lines, rendered_page.page_number, source="ocr")
        else:
            page_answers = []

        normalized = [normalize_answer_block(answer, rendered_page.page_number) for answer in page_answers]
        return normalized, backend
    finally:
        if not keep_pages:
            rendered_page.image_path.unlink(missing_ok=True)


class OpenAICompatibleVisionClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None,
        cache_dir: Path,
        temperature: float = 0.1,
        max_retries: int = 2,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai is required for analysis.mode=llm. Run: pip install -r requirements-llm.txt") from exc

        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.cache_dir = ensure_dir(cache_dir)
        self._cache_lock = threading.Lock()
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = normalize_base_url(base_url)
        self.client = OpenAI(**kwargs)

    def analyze_question_page(
        self,
        image_path: Path,
        page_number: int,
        active_question: str,
        diagrams: list[dict[str, Any]],
        page_plan: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        diagrams_payload = [
            {
                "diagram_id": diagram["diagram_id"],
                "box_1000": diagram["box_1000"],
                "file": diagram.get("file"),
            }
            for diagram in diagrams
        ]
        page_plan_payload = page_plan or {}
        prompt = f"""You are analyzing one page of an exam question paper.

Return only valid JSON. Do not include markdown fences.

Task:
1. Extract every real question/sub-question visible on this page in reading order.
2. Preserve the question numbering exactly, e.g. "1", "1(a)", "1(b)(ii)".
3. If a question continues from the previous page and the parent number is not shown, use the page plan and active parent "{active_question}".
4. CRITICAL: Always extract the intro/context text that appears before the first sub-question as a separate question with the parent number only (e.g. "1", "2", "3"). This intro text often describes a diagram, experiment, or scenario. Do NOT merge it into the first sub-question.
5. Clean the question text for HTML presentation: remove page headers/footers, page numbers, candidate instructions that are not part of a question, answer dotted lines, repeated underscores, and handwriting space.
6. Preserve science/math notation using LaTeX delimiters \\( ... \\) or \\[ ... \\].
7. For MCQs, put options in mcq_options and do not duplicate them in text.
8. For tables, output clean table_html and do not duplicate raw table text.
9. Each question must include a tight box_1000 as [ymin, xmin, ymax, xmax].
10. Do not detect or crop diagrams. Use only the supplied diagrams list below for association.
11. If a supplied diagram belongs inside a question, include its id in associated_diagram_ids and insert [DIAGRAM:diagram_id] in the text at the visual position where it belongs.

Supplied diagrams on this page:
{json.dumps(diagrams_payload, ensure_ascii=False)}

Precomputed page sequence plan:
{json.dumps(page_plan_payload, ensure_ascii=False)}

Required JSON shape:
{{
  "questions": [
    {{
      "question_number": "1",
      "text": "The diagram shows a plant cell. This is the intro/context for the whole question.",
      "marks": null,
      "table_html": null,
      "is_mcq": false,
      "mcq_options": null,
      "box_1000": [100, 80, 240, 900],
      "associated_diagram_ids": ["p2f1"]
    }},
    {{
      "question_number": "1(a)",
      "text": "Name the structures labelled A, B, C, and D.",
      "marks": 4,
      "table_html": null,
      "is_mcq": false,
      "mcq_options": null,
      "box_1000": [100, 80, 240, 900],
      "associated_diagram_ids": ["p2f1"]
    }}
  ]
}}
"""
        payload = self._image_json_request(image_path, prompt, cache_prefix=f"qp_p{page_number}")
        questions = payload.get("questions", []) if isinstance(payload, dict) else []
        return questions if isinstance(questions, list) else []

    def analyze_question_page_plan(
        self,
        image_path: Path,
        page_number: int,
        previous_active_question: str,
    ) -> dict[str, Any]:
        prompt = f"""You are planning the sequence of an exam question paper before detailed extraction.

Return only valid JSON. Do not include markdown fences.

Look at this single 1000x1000 page image. Do not extract full question text.

Task:
1. Identify the active parent label for this page. This may be a main number like "4" or a letter parent like "4(b)" when the visible labels are only "(i)", "(ii)", etc.
2. List all visible question/sub-question labels in reading order.
3. If the page continues from a previous page and only labels like (i), (ii), (b), or (c)(i) are visible, attach them to the most specific likely parent. Use previous active parent "{previous_active_question}" if the page itself does not show a better parent.
4. Preserve labels exactly in normalized form such as "4", "4(a)", "4(b)(ii)".
5. If uncertain, use the most likely parent from the page context and previous active parent.

Required JSON shape:
{{
  "page_number": {page_number},
  "active_parent_question": "4(b)",
  "visible_question_numbers": ["4(b)(i)", "4(b)(ii)"],
  "continues_previous_question": true,
  "notes": "brief reason"
}}
"""
        payload = self._image_json_request(image_path, prompt, cache_prefix=f"qp_plan_p{page_number}")
        return payload if isinstance(payload, dict) else {}

    def analyze_mark_scheme_page(self, image_path: Path, page_number: int) -> list[dict[str, Any]]:
        prompt = """You are analyzing one page of an exam mark scheme.

Return only valid JSON. Do not include markdown fences.

Task:
1. Extract answer/marking guidance blocks for each question number visible on this page.
2. Preserve question numbering exactly, e.g. "1(a)", "1(b)(ii)".
3. Clean answer content for HTML: remove generic boilerplate, wrong-option explanations unless needed for the correct answer, repeated headers, footers, page numbers, and empty table artifacts.
4. Keep actual marking points, accepted alternatives, equations, mark counts, and important additional guidance.
5. Use concise HTML in answer_html: paragraphs, ul/li, table/tr/th/td, br, strong are allowed.
6. Each answer should include box_1000 as [ymin, xmin, ymax, xmax] when possible.

Required JSON shape:
{
  "answers": [
    {
      "question_number": "1(a)",
      "answer_html": "<ul><li>marking point</li></ul>",
      "marks": 2,
      "box_1000": [100, 80, 240, 900]
    }
  ]
}
"""
        payload = self._image_json_request(image_path, prompt, cache_prefix=f"ms_p{page_number}")
        answers = payload.get("answers", []) if isinstance(payload, dict) else []
        return answers if isinstance(answers, list) else []

    def analyze_question_pages_batch(
        self,
        image_paths: list[Path],
        page_numbers: list[int],
        active_question: str,
        diagrams_by_page: dict[int, list[dict[str, Any]]],
        page_plans: dict[int, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        diagrams_payload = {}
        for p_num in page_numbers:
            p_diagrams = diagrams_by_page.get(p_num, [])
            diagrams_payload[p_num] = [
                {
                    "diagram_id": diagram["diagram_id"],
                    "box_1000": diagram["box_1000"],
                    "file": diagram.get("file"),
                }
                for diagram in p_diagrams
            ]

        prompt = f"""You are analyzing a sequence of {len(image_paths)} pages of an exam question paper in reading order.

Return only valid JSON. Do not include markdown fences.

Task:
1. Extract every real question/sub-question visible on these pages in reading order.
2. For each question, specify the "page" number (an integer from the list {page_numbers}).
3. Preserve the question numbering exactly, e.g. "1", "1(a)", "1(b)(ii)".
4. If a question continues from the previous page/context and the parent number is not shown, use the page plans and the active parent question "{active_question}" at the start of the batch.
5. CRITICAL: Always extract the intro/context text that appears before the first sub-question as a separate question with the parent number only (e.g. "1", "2", "3"). This intro text often describes a diagram, experiment, or scenario. Do NOT merge it into the first sub-question.
6. Clean the question text for HTML presentation: remove page headers/footers, page numbers, candidate instructions that are not part of a question, answer dotted lines, repeated underscores, and handwriting space.
7. Preserve science/math notation using LaTeX delimiters \\( ... \\) or \\[ ... \\].
8. For MCQs, put options in mcq_options and do not duplicate them in text.
9. For tables, output clean table_html and do not duplicate raw table text.
10. Each question must include a tight box_1000 as [ymin, xmin, ymax, xmax] relative to that page image.
11. Do not detect or crop diagrams. Use only the supplied diagrams list for association.
12. If a supplied diagram belongs inside a question, include its id in associated_diagram_ids and insert [DIAGRAM:diagram_id] in the text at the visual position where it belongs.

Supplied diagrams on these pages:
{json.dumps(diagrams_payload, ensure_ascii=False)}

Precomputed page sequence plans:
{json.dumps(page_plans, ensure_ascii=False)}

Required JSON shape:
{{
  "questions": [
    {{
      "page": {page_numbers[0]},
      "question_number": "1",
      "text": "The diagram shows a plant cell. This is the intro/context for the whole question.",
      "marks": null,
      "table_html": null,
      "is_mcq": false,
      "mcq_options": null,
      "box_1000": [100, 80, 240, 900],
      "associated_diagram_ids": ["p2f1"]
    }},
    {{
      "page": {page_numbers[0]},
      "question_number": "1(a)",
      "text": "Name the structures labelled A, B, C, and D.",
      "marks": 4,
      "table_html": null,
      "is_mcq": false,
      "mcq_options": null,
      "box_1000": [100, 80, 240, 900],
      "associated_diagram_ids": ["p2f1"]
    }}
  ]
}}
"""
        payload = self._multi_image_json_request(
            image_paths,
            prompt,
            cache_prefix=f"qp_batch_p{page_numbers[0]}_to_p{page_numbers[-1]}",
        )
        questions = payload.get("questions", []) if isinstance(payload, dict) else []
        return questions if isinstance(questions, list) else []

    def analyze_mark_scheme_pages_batch(
        self,
        image_paths: list[Path],
        page_numbers: list[int],
    ) -> list[dict[str, Any]]:
        prompt = f"""You are analyzing a sequence of {len(image_paths)} pages of an exam mark scheme in reading order.

Return only valid JSON. Do not include markdown fences.

Task:
1. Extract answer/marking guidance blocks for each question number visible on these pages.
2. For each answer, specify the "page" number (an integer from the list {page_numbers}).
3. Preserve question numbering exactly, e.g. "1(a)", "1(b)(ii)".
4. Clean answer content for HTML: remove generic boilerplate, wrong-option explanations unless needed for the correct answer, repeated headers, footers, page numbers, and empty table artifacts.
5. Keep actual marking points, accepted alternatives, equations, mark counts, and important additional guidance.
6. Use concise HTML in answer_html: paragraphs, ul/li, table/tr/th/td, br, strong are allowed.
7. Each answer should include box_1000 as [ymin, xmin, ymax, xmax] relative to that page image when possible.

Required JSON shape:
{{
  "answers": [
    {{
      "page": {page_numbers[0]},
      "question_number": "1(a)",
      "answer_html": "<ul><li>marking point</li></ul>",
      "marks": 2,
      "box_1000": [100, 80, 240, 900]
    }}
  ]
}}
"""
        payload = self._multi_image_json_request(
            image_paths,
            prompt,
            cache_prefix=f"ms_batch_p{page_numbers[0]}_to_p{page_numbers[-1]}",
        )
        answers = payload.get("answers", []) if isinstance(payload, dict) else []
        return answers if isinstance(answers, list) else []

    def clean_question_html(self, question_html: str) -> str:
        prompt = f"""Clean this exam question HTML fragment.

Rules:
- Preserve existing HTML tags, image tags, diagram wrappers, and LaTeX exactly.
- Remove OCR noise, repeated blank lines, dotted answer lines, underscores, page numbers, and irrelevant headers.
- Convert markdown-style tables to valid HTML tables.
- Return only the cleaned HTML fragment.

Fragment:
{question_html}
"""
        return self._text_request(prompt, cache_prefix="clean_q")

    def clean_answer_html(self, answer_html: str) -> str:
        prompt = f"""Clean this mark scheme HTML fragment.

Rules:
- Preserve existing section wrappers and useful HTML tags.
- Keep only actual marking points, accepted answers, equations, and essential guidance.
- Remove generic boilerplate and wrong-option explanations that do not help answer the question.
- Convert literal newline markers into <br> only where needed.
- Return only the cleaned HTML fragment.

Fragment:
{answer_html}
"""
        return self._text_request(prompt, cache_prefix="clean_a")

    def clean_qna_html(self, question_html: str, answer_html: str) -> tuple[str, str]:
        if not question_html.strip() and not answer_html.strip():
            return "", ""

        prompt = f"""You are cleaning an exam question and its corresponding answer/mark scheme HTML fragment.

Return only valid JSON. Do not include markdown fences.

Question Rules:
- Preserve existing HTML tags, image tags, diagram wrappers, and LaTeX exactly.
- Remove OCR noise, repeated blank lines, dotted answer lines, underscores, page numbers, and irrelevant headers.
- Convert markdown-style tables to valid HTML tables.

Answer Rules:
- Preserve existing section wrappers and useful HTML tags.
- Keep only actual marking points, accepted answers, equations, and essential guidance.
- Remove generic boilerplate and wrong-option explanations that do not help answer the question.
- Convert literal newline markers into <br> only where needed.

Required JSON shape:
{{
  "cleaned_question": "...",
  "cleaned_answer": "..."
}}

Question Fragment to Clean:
{question_html}

Answer Fragment to Clean:
{answer_html}
"""
        cache_key = self._cache_key(prompt, None)
        cached = self._read_cache(cache_key)
        if isinstance(cached, dict) and "cleaned_question" in cached and "cleaned_answer" in cached:
            return str(cached["cleaned_question"]), str(cached["cleaned_answer"])

        response_text = self._request_with_retries([{"role": "user", "content": prompt}]).strip()
        response_text = strip_markdown_fence(response_text)
        try:
            parsed = parse_json_response(response_text)
        except Exception:
            parsed = {}

        cleaned_q = parsed.get("cleaned_question", question_html)
        cleaned_a = parsed.get("cleaned_answer", answer_html)

        cleaned_q = str(cleaned_q) if cleaned_q is not None else ""
        cleaned_a = str(cleaned_a) if cleaned_a is not None else ""

        self._write_cache(
            cache_key,
            {"cleaned_question": cleaned_q, "cleaned_answer": cleaned_a},
            "clean_qna"
        )
        return cleaned_q, cleaned_a

    def analyze_question_pages_plan(
        self,
        image_paths: list[Path],
        page_numbers: list[int],
    ) -> dict[int, dict[str, Any]]:
        if not image_paths:
            return {}

        prompt = f"""You are planning the sequence of an exam question paper before detailed extraction.

Return only valid JSON. Do not include markdown fences.

You are given a sequence of {len(image_paths)} page images in reading order.
For each page, identify the active parent label and all visible question/sub-question labels.

Task:
1. Identify the active parent label for each page. If the page continues from a previous page and only labels like (i), (ii), (b), or (c)(i) are visible, attach them to the most specific likely parent.
2. List all visible question/sub-question labels in reading order for each page.
3. Preserve labels exactly in normalized form such as "4", "4(a)", "4(b)(ii)".
4. If uncertain, use the most likely parent from the surrounding page contexts.

Required JSON shape:
{{
  "plans": [
    {{
      "page_number": 1,
      "active_parent_question": "1",
      "visible_question_numbers": ["1(a)", "1(b)"],
      "continues_previous_question": false,
      "notes": "starts question 1"
    }},
    ...
  ]
}}
"""
        payload = self._multi_image_json_request(
            image_paths,
            prompt,
            cache_prefix=f"qp_plan_batch_p{page_numbers[0]}_to_p{page_numbers[-1]}",
        )
        plans_list = payload.get("plans", []) if isinstance(payload, dict) else []
        plans_dict: dict[int, dict[str, Any]] = {}
        for item in plans_list if isinstance(plans_list, list) else []:
            if isinstance(item, dict) and "page_number" in item:
                try:
                    p_num = int(item["page_number"])
                    plans_dict[p_num] = item
                except (ValueError, TypeError):
                    pass
        return plans_dict

    def _multi_image_json_request(self, image_paths: list[Path], prompt: str, cache_prefix: str) -> dict[str, Any]:
        hasher = hashlib.sha256()
        hasher.update(self.model.encode("utf-8"))
        hasher.update(prompt.encode("utf-8"))
        for path in image_paths:
            hasher.update(path.read_bytes())
        cache_key = hasher.hexdigest()

        cached = self._read_cache(cache_key)
        if isinstance(cached, dict):
            return cached

        content_list: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in image_paths:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content_list.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
            )

        response_text = self._request_with_retries(
            [
                {
                    "role": "user",
                    "content": content_list,
                }
            ]
        )
        parsed = parse_json_response(response_text)
        self._write_cache(cache_key, parsed, cache_prefix)
        return parsed

    def _image_json_request(self, image_path: Path, prompt: str, cache_prefix: str) -> dict[str, Any]:
        cache_key = self._cache_key(prompt, image_path)
        cached = self._read_cache(cache_key)
        if isinstance(cached, dict):
            return cached

        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        response_text = self._request_with_retries(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                    ],
                }
            ]
        )
        parsed = parse_json_response(response_text)
        self._write_cache(cache_key, parsed, cache_prefix)
        return parsed

    def _text_request(self, prompt: str, cache_prefix: str) -> str:
        cache_key = self._cache_key(prompt, None)
        cached = self._read_cache(cache_key)
        if isinstance(cached, dict) and isinstance(cached.get("text"), str):
            return cached["text"]

        response_text = self._request_with_retries([{"role": "user", "content": prompt}]).strip()
        response_text = strip_markdown_fence(response_text)
        self._write_cache(cache_key, {"text": response_text}, cache_prefix)
        return response_text

    def _request_with_retries(self, messages: list[dict[str, Any]]) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
                content = response.choices[0].message.content
                return content or ""
            except Exception as exc:  # pragma: no cover - network/API behavior
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM request failed: {last_error}") from last_error

    def _cache_key(self, prompt: str, image_path: Path | None) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.model.encode("utf-8"))
        hasher.update(prompt.encode("utf-8"))
        if image_path:
            hasher.update(image_path.read_bytes())
        return hasher.hexdigest()

    def _read_cache(self, cache_key: str) -> Any:
        path = self.cache_dir / f"{cache_key}.json"
        if not path.exists():
            return None
        try:
            return load_json(path)
        except Exception:
            LOGGER.warning("Ignoring invalid LLM cache file: %s", path)
            return None

    def _write_cache(self, cache_key: str, payload: Any, cache_prefix: str) -> None:
        with self._cache_lock:
            path = self.cache_dir / f"{cache_key}.json"
            write_json(path, payload)
            index_path = self.cache_dir / f"{cache_prefix}_{cache_key[:12]}.json"
            if not index_path.exists():
                write_json(index_path, {"cache_key": cache_key})


class PageOCREngine:
    def __init__(self, language: str = "en", min_confidence: float = 0.45) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is required for analysis OCR fallback. Run: pip install -r requirements-ocr.txt"
            ) from exc

        self.min_confidence = min_confidence
        self.engine = PaddleOCR(use_angle_cls=True, lang=language)
        self._lock = threading.Lock()

    def extract_lines(self, image_path: Path) -> list[dict[str, Any]]:
        with self._lock:
            result = self.engine.ocr(str(image_path))
        lines: list[dict[str, Any]] = []
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for OCR coordinate normalization") from exc

        image = cv2.imread(str(image_path))
        if image is None:
            return lines
        height, width = image.shape[:2]

        for page_result in result or []:
            for item in page_result or []:
                if len(item) < 2:
                    continue
                points = item[0]
                text_data = item[1]
                if not isinstance(text_data, (list, tuple)) or len(text_data) < 2:
                    continue
                text = str(text_data[0]).strip()
                confidence = float(text_data[1])
                if not text or confidence < self.min_confidence:
                    continue
                xs = [float(point[0]) for point in points]
                ys = [float(point[1]) for point in points]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
                lines.append(
                    {
                        "text": text,
                        "confidence": confidence,
                        "bbox": bbox,
                        "box_1000": pixel_bbox_to_box_1000(bbox, width, height),
                    }
                )

        lines.sort(key=lambda line: (line["box_1000"][0], line["box_1000"][1]))
        return lines


def cleanup_qnas_with_llm(
    qnas: list[dict[str, Any]],
    llm_client: OpenAICompatibleVisionClient,
    workers: int = 1,
) -> list[dict[str, Any]]:
    workers = min(normalize_worker_count(workers), max(len(qnas), 1))
    if workers > 1:
        LOGGER.info("Cleaning QNA HTML with %s LLM worker(s)", workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(lambda qna: cleanup_single_qna_with_llm(qna, llm_client), qnas))

    return [cleanup_single_qna_with_llm(qna, llm_client) for qna in qnas]


def cleanup_single_qna_with_llm(qna: dict[str, Any], llm_client: OpenAICompatibleVisionClient) -> dict[str, Any]:
    item = dict(qna)
    try:
        text = str(item.get("text", "")).strip()
        answer_text = str(item.get("answer_text", "")).strip()

        if text or answer_text:
            cleaned_q, cleaned_a = llm_client.clean_qna_html(text, answer_text)
            item["text"] = cleaned_q
            item["answer_text"] = cleaned_a
        else:
            item["text"] = ""
            item["answer_text"] = ""
    except Exception as exc:
        LOGGER.warning("LLM cleanup failed for question %s: %s", item.get("question_number"), exc)
    return item


def make_llm_client(config: dict[str, Any], base_dir: Path) -> OpenAICompatibleVisionClient | None:
    llm_config = config.get("llm") or {}
    if not bool(llm_config.get("enabled", True)):
        return None

    load_dotenv_file(resolve_path(llm_config.get("dotenv_path", ".env"), base_dir))

    api_key = optional_config_or_env(llm_config, "api_key", "api_key_env", ["ANALYSIS_LLM_API_KEY", "KIMI_API_KEY", "OPENAI_API_KEY"])
    model = optional_config_or_env(llm_config, "model", "model_env", ["ANALYSIS_LLM_MODEL", "KIMI_MODEL", "OPENAI_MODEL"])
    base_url = optional_config_or_env(llm_config, "base_url", "base_url_env", ["ANALYSIS_LLM_BASE_URL", "KIMI_BASE_URL", "OPENAI_BASE_URL"])

    mode = str(config.get("mode", "auto")).lower()
    if not api_key or not model:
        if mode == "llm":
            raise RuntimeError(
                "analysis.mode=llm requires an API key and model. Set analysis.llm.* or env vars "
                "ANALYSIS_LLM_API_KEY/ANALYSIS_LLM_MODEL, KIMI_API_KEY/KIMI_MODEL, or OPENAI_API_KEY/OPENAI_MODEL."
            )
        LOGGER.info("No LLM analysis credentials found; falling back to OCR analysis when possible")
        return None

    cache_dir_value = llm_config.get("cache_dir") or ".cache/question_analysis"
    cache_dir = resolve_path(cache_dir_value, base_dir)
    return OpenAICompatibleVisionClient(
        api_key=api_key,
        model=model,
        base_url=base_url,
        cache_dir=cache_dir,
        temperature=float(llm_config.get("temperature", 0.1)),
        max_retries=int(llm_config.get("max_retries", 2)),
    )


def make_page_ocr(config: dict[str, Any]) -> PageOCREngine | None:
    ocr_config = config.get("ocr") or {}
    mode = str(config.get("mode", "auto")).lower()
    if mode not in {"auto", "ocr"}:
        return None
    if mode != "ocr" and not bool(ocr_config.get("enabled", False)):
        return None
    return PageOCREngine(
        language=str(ocr_config.get("language", "en")),
        min_confidence=float(ocr_config.get("min_confidence", 0.45)),
    )


def choose_backend(
    mode: str,
    llm_client: OpenAICompatibleVisionClient | None,
    ocr_engine: PageOCREngine | None,
) -> str:
    if mode in {"llm", "ocr"}:
        return mode
    if llm_client is not None:
        return "llm"
    if ocr_engine is not None:
        return "ocr"
    return "llm"


def normalize_worker_count(value: Any, default: int = 1) -> int:
    try:
        workers = int(value)
    except (TypeError, ValueError):
        workers = default
    return max(1, workers)


def find_exam_pairs(pdfs: list[Path], config: dict[str, Any], base_dir: Path) -> list[ExamPair]:
    qp_value = config.get("qp_pdf")
    ms_value = config.get("ms_pdf")
    if qp_value:
        qp_path = resolve_path(qp_value, base_dir)
        ms_path = resolve_path(ms_value, base_dir) if ms_value else None
        return [ExamPair(qp_path, ms_path)]

    grouped: dict[str, dict[str, Path]] = {}
    for pdf in pdfs:
        kind = classify_exam_pdf(pdf)
        if kind is None:
            continue
        key = exam_pair_key(pdf)
        grouped.setdefault(key, {})[kind] = pdf

    pairs = [
        ExamPair(group["qp"], group.get("ms"))
        for group in grouped.values()
        if "qp" in group
    ]
    return sorted(pairs, key=lambda pair: pair.question_pdf.name)


def classify_exam_pdf(path: Path) -> str | None:
    name = path.stem.lower()
    if re.search(r"(^|[-_\s])(qp|question[-_\s]*paper)($|[-_\s])", name):
        return "qp"
    if re.search(r"(^|[-_\s])(ms|mark[-_\s]*scheme)($|[-_\s])", name):
        return "ms"
    return None


def exam_pair_key(path: Path) -> str:
    name = path.stem.lower()
    name = re.sub(r"(^|[-_\s])(qp|ms|question[-_\s]*paper|mark[-_\s]*scheme)($|[-_\s])", " ", name)
    name = re.sub(r"[^a-z0-9]+", "", name)
    return name



def heuristic_questions_from_lines(
    lines: list[dict[str, Any]],
    page_number: int,
    diagrams: list[dict[str, Any]],
    source: str,
) -> list[dict[str, Any]]:
    return heuristic_questions_from_text_units(lines, page_number, diagrams, source)


def heuristic_questions_from_text_units(
    units: list[dict[str, Any]],
    page_number: int,
    diagrams: list[dict[str, Any]],
    source: str,
) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for unit in sorted(units, key=lambda item: (item.get("box_1000", [0, 0, 0, 0])[0], item.get("box_1000", [0, 0, 0, 0])[1])):
        text = normalize_question_text(str(unit.get("text", "")))
        if not text or is_page_artifact(text):
            continue

        match = QUESTION_START_RE.match(text)
        if match:
            if current:
                questions.append(finalize_question_candidate(current, page_number, source))
            question_number = normalize_question_label(match.group("number"))
            body = text[match.end():].strip(" .:-")
            current = {
                "question_number": question_number,
                "parts": [body] if body else [],
                "boxes": [unit.get("box_1000")],
            }
        elif current:
            current["parts"].append(text)
            current["boxes"].append(unit.get("box_1000"))

    if current:
        questions.append(finalize_question_candidate(current, page_number, source))

    return associate_diagrams_heuristically(questions, diagrams)


def heuristic_answers_from_lines(lines: list[dict[str, Any]], page_number: int, source: str) -> list[dict[str, Any]]:
    text = "\n".join(str(line.get("text", "")) for line in lines)
    return heuristic_answers_from_text(text, page_number, source=source)


def heuristic_answers_from_text(text: str, page_number: int, source: str = "ocr") -> list[dict[str, Any]]:
    if "Question" not in text and "Answer" not in text:
        return []

    normalized = normalize_whitespace(text)
    normalized = re.sub(r"\bQuestion\s+Number\b", " Question Number ", normalized, flags=re.I)
    matches = list(ANSWER_LABEL_RE.finditer(normalized))
    answers: list[dict[str, Any]] = []

    for index, match in enumerate(matches):
        label = normalize_question_label(match.group("number"))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        body = normalized[start:end]
        body = clean_mark_scheme_text(body)
        if not body or len(body) < 3:
            continue
        marks = extract_marks(body)
        answers.append(
            {
                "question_number": label,
                "answer_html": plain_text_to_answer_html(body),
                "marks": marks,
                "page": page_number,
                "box_1000": None,
                "analysis_source": source,
            }
        )

    return answers


def finalize_question_candidate(candidate: dict[str, Any], page_number: int, source: str) -> dict[str, Any]:
    text = normalize_question_text(" ".join(part for part in candidate.get("parts", []) if part))
    marks = extract_marks(text)
    if marks is not None:
        text = strip_marks(text)
    box_1000 = union_boxes([box for box in candidate.get("boxes", []) if box])
    is_mcq, options, text_without_options = extract_mcq_options(text)
    return {
        "question_number": candidate["question_number"],
        "text": text_without_options,
        "marks": marks,
        "table_html": None,
        "is_mcq": is_mcq,
        "mcq_options": options if is_mcq else None,
        "box_1000": box_1000,
        "associated_diagram_ids": [],
        "page": page_number,
        "analysis_source": source,
    }


def associate_diagrams_heuristically(
    questions: list[dict[str, Any]],
    diagrams: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not diagrams or not questions:
        return questions

    sorted_questions = sorted(questions, key=lambda item: (item.get("box_1000") or [0, 0, 0, 0])[0])
    for question in sorted_questions:
        question_box = question.get("box_1000")
        if not question_box:
            continue
        ids = []
        for diagram in diagrams:
            diagram_box = diagram.get("box_1000")
            if diagram_box and boxes_related(question_box, diagram_box):
                ids.append(diagram["diagram_id"])
        question["associated_diagram_ids"] = ids
    return sorted_questions


def bind_diagrams_to_questions(
    questions: list[dict[str, Any]],
    diagrams_by_page: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    diagram_lookup = {
        diagram["diagram_id"]: diagram
        for diagrams in diagrams_by_page.values()
        for diagram in diagrams
    }
    output: list[dict[str, Any]] = []

    for question in questions:
        page = int(question.get("page") or 0)
        page_diagrams = diagrams_by_page.get(page, [])
        ids = [str(value) for value in question.get("associated_diagram_ids") or [] if str(value) in diagram_lookup]

        if not ids and page_diagrams and question.get("box_1000"):
            ids = [
                diagram["diagram_id"]
                for diagram in page_diagrams
                if boxes_related(question["box_1000"], diagram.get("box_1000"))
            ]

        images = []
        text = str(question.get("text", ""))
        for diagram_id in ids:
            diagram = diagram_lookup[diagram_id]
            image_path = str(diagram.get("absolute_file") or diagram.get("file") or "")
            if image_path:
                images.append(image_path)
                text = inject_diagram_html(text, diagram_id, image_path)

        item = dict(question)
        item["associated_diagram_ids"] = ids
        item["associated_images"] = images
        item["text"] = text
        output.append(item)

    return output


def match_questions_to_answers(questions: list[dict[str, Any]], answers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    answer_lookup: dict[str, list[dict[str, Any]]] = {}
    for answer in answers:
        key = normalize_match_key(answer.get("question_number"))
        if key:
            answer_lookup.setdefault(key, []).append(answer)

    qnas: list[dict[str, Any]] = []
    seen: set[str] = set()
    for question in questions:
        q_num = str(question.get("question_number") or "")
        q_key = normalize_match_key(q_num)
        unique_key = f"{q_key}:{question.get('page')}:{normalize_whitespace(str(question.get('text')))[:80]}"
        if unique_key in seen:
            continue
        seen.add(unique_key)

        answer_blocks = find_matching_answers(q_key, answer_lookup)
        answer_html = merge_answer_html(answer_blocks, q_num)
        qnas.append(
            {
                "question_number": q_num,
                "text": question.get("text") or "",
                "marks": question.get("marks"),
                "table_html": question.get("table_html"),
                "is_mcq": bool(question.get("is_mcq", False)),
                "mcq_options": question.get("mcq_options"),
                "associated_images": question.get("associated_images", []),
                "box_1000": question.get("box_1000"),
                "page": question.get("page"),
                "answer_text": answer_html,
                "answer_blocks": answer_blocks,
            }
        )

    return qnas


def find_matching_answers(q_key: str, answer_lookup: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if q_key in answer_lookup:
        return answer_lookup[q_key]

    answer_keys_sorted = sorted(answer_lookup.keys(), key=lambda k: (len(k), k))

    for ak in answer_keys_sorted:
        if q_key == ak:
            return answer_lookup[ak]
        if ak.startswith(q_key):
            return answer_lookup[ak]

    for ak in sorted(answer_lookup.keys(), key=len, reverse=True):
        if q_key.startswith(ak):
            return answer_lookup[ak]

    q_norm = q_key.replace("(", "").replace(")", "")
    for ak in sorted(answer_lookup.keys(), key=lambda k: (len(k), k)):
        ak_norm = ak.replace("(", "").replace(")", "")
        if ak_norm == q_norm:
            return answer_lookup[ak]
        if ak_norm.startswith(q_norm):
            return answer_lookup[ak]
        if q_norm.startswith(ak_norm):
            return answer_lookup[ak]

    q_digits = re.sub(r"[^0-9]", "", q_key)
    q_letters = re.sub(r"[0-9]", "", q_key)
    if q_digits:
        candidates = []
        for ak in answer_lookup:
            ak_digits = re.sub(r"[^0-9]", "", ak)
            ak_letters = re.sub(r"[0-9]", "", ak)
            if ak_digits == q_digits:
                if not q_letters or not ak_letters or q_letters in ak_letters or ak_letters in q_letters:
                    candidates.append((len(ak), ak))
        if candidates:
            candidates.sort()
            return answer_lookup[candidates[0][1]]

    return []


def merge_answer_html(answer_blocks: list[dict[str, Any]], question_number: str) -> str:
    if not answer_blocks:
        return "<p class='ms-para'>No official mark scheme answer found for this question block.</p>"

    sections = []
    seen = set()
    for answer in answer_blocks:
        body = str(answer.get("answer_html") or answer.get("answer_text") or "").strip()
        key = normalize_whitespace(body)[:120]
        if not body or key in seen:
            continue
        seen.add(key)
        sections.append(body)
    return "\n".join(sections) if sections else "<p class='ms-para'>No official mark scheme answer found for this question block.</p>"


def collect_pdf_diagrams(metadata: dict[str, Any], pdf_path: Path, output_root: Path) -> list[dict[str, Any]]:
    diagrams = metadata.get("diagrams", []) if isinstance(metadata, dict) else []
    records: list[dict[str, Any]] = []
    for record in diagrams:
        if not isinstance(record, dict):
            continue
        pdf_name = str(record.get("pdf") or Path(str(record.get("source_pdf", ""))).name)
        source_pdf = str(record.get("source_pdf") or "")
        if pdf_name != pdf_path.name and not source_pdf.endswith(pdf_path.name):
            continue

        page_width = int(record.get("page_width") or 0)
        page_height = int(record.get("page_height") or 0)
        bbox = record.get("crop_bbox") or record.get("bbox")
        box_1000 = pixel_bbox_to_box_1000(bbox, page_width, page_height) if bbox and page_width and page_height else None
        page = int(record.get("page") or 0)
        figure = int(record.get("figure") or len(records) + 1)
        diagram_id = f"p{page}f{figure}"
        file_value = str(record.get("file") or "")
        absolute_file = output_root / file_value if file_value and not Path(file_value).is_absolute() else Path(file_value)

        item = dict(record)
        item.update(
            {
                "diagram_id": diagram_id,
                "box_1000": box_1000,
                "absolute_file": str(absolute_file.resolve()) if absolute_file else "",
            }
        )
        records.append(item)
    return records


def group_diagrams_by_page(diagrams: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for diagram in diagrams:
        page = int(diagram.get("page") or 0)
        grouped.setdefault(page, []).append(diagram)
    for page_diagrams in grouped.values():
        page_diagrams.sort(key=lambda item: (item.get("box_1000") or [0, 0, 0, 0])[0])
    return grouped


def fix_orphan_subparts(questions: list[dict[str, Any]], active_question: str) -> list[dict[str, Any]]:
    fixed = []
    for question in questions:
        q_num = str(question.get("question_number", ""))
        text = str(question.get("text", ""))
        if q_num == active_question:
            sub_match = re.match(r"^\s*\(([a-z])\)", text)
            if sub_match:
                sub = sub_match.group(1)
                question["question_number"] = f"{q_num}({sub})"
        fixed.append(question)
    return fixed


def normalize_question_block(
    question: dict[str, Any],
    page_number: int,
    diagrams: list[dict[str, Any]],
) -> dict[str, Any]:
    q_num = normalize_question_label(str(question.get("question_number") or ""))
    text = normalize_question_text(str(question.get("text") or ""))
    marks = question.get("marks")
    if marks is None:
        marks = extract_marks(text)
    if marks is not None:
        text = strip_marks(text)

    ids = [str(value) for value in question.get("associated_diagram_ids") or []]
    ids = [diagram_id for diagram_id in ids if any(diagram.get("diagram_id") == diagram_id for diagram in diagrams)]

    return {
        "question_number": q_num,
        "text": text,
        "marks": normalize_int_or_none(marks),
        "table_html": question.get("table_html"),
        "is_mcq": bool(question.get("is_mcq", False)),
        "mcq_options": question.get("mcq_options"),
        "box_1000": normalize_box_1000(question.get("box_1000")),
        "associated_diagram_ids": ids,
        "page": page_number,
        "analysis_source": question.get("analysis_source", "llm"),
    }


def normalize_answer_block(answer: dict[str, Any], page_number: int) -> dict[str, Any]:
    answer_html = str(answer.get("answer_html") or answer.get("answer_text") or "").strip()
    if answer_html and "<" not in answer_html:
        answer_html = plain_text_to_answer_html(answer_html)
    return {
        "question_number": normalize_question_label(str(answer.get("question_number") or "")),
        "answer_html": answer_html,
        "marks": normalize_int_or_none(answer.get("marks")),
        "box_1000": normalize_box_1000(answer.get("box_1000")),
        "page": page_number,
        "analysis_source": answer.get("analysis_source", "llm"),
    }


def dedupe_answer_blocks(answers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for answer in answers:
        key = (normalize_match_key(answer.get("question_number")), normalize_whitespace(str(answer.get("answer_html")))[:180])
        if key in seen:
            continue
        seen.add(key)
        output.append(answer)
    return output


def sort_blocks(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        items,
        key=lambda item: (
            int(item.get("page") or 0),
            (item.get("box_1000") or [9999, 9999, 9999, 9999])[0],
            (item.get("box_1000") or [9999, 9999, 9999, 9999])[1],
            normalize_match_key(item.get("question_number")),
        ),
    )


QUESTION_START_RE = re.compile(
    r"^\s*(?P<number>\d+(?:\s*\([a-z]\))*(?:\s*\([ivxlcdm]+\))?)\s*(?![a-z])",
    flags=re.I,
)

ANSWER_LABEL_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<number>\d+(?:\s*\([a-z]\))*(?:\s*\([ivxlcdm]+\))?)(?![A-Za-z0-9(])",
    flags=re.I,
)


def normalize_question_label(value: str) -> str:
    value = normalize_whitespace(value)
    value = re.sub(r"^(question|q)\s*", "", value, flags=re.I)
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"\)\(", ")(", value)
    return value


def normalize_match_key(value: Any) -> str:
    value = str(value or "").lower()
    value = re.sub(r"^(question|q)\s*", "", value)
    return re.sub(r"[^a-z0-9]+", "", value)


def normalize_question_text(value: str) -> str:
    value = value.replace("\u00a0", " ")
    value = value.replace("\uf0b7", "-")
    value = re.sub(r"\.{4,}", "", value)
    value = re.sub(r"_{3,}", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\u00a0", " ")).strip()


def is_page_artifact(text: str) -> bool:
    low = text.lower().strip()
    if len(low) < 2:
        return True
    exact_artifacts = {
        "turn over",
        "blank page",
        "answer all questions",
        "do not write outside the box",
        "centre number",
        "candidate number",
        "candidate name",
    }
    if low in exact_artifacts:
        return True
    header_prefixes = [
        "total for question",
        "international gcse",
        "pearson edexcel",
    ]
    return any(low.startswith(prefix) for prefix in header_prefixes)


def extract_marks(text: str) -> int | None:
    candidates = re.findall(r"\[(\d{1,2})\]|\((\d{1,2})\)\s*(?:marks?)?", text, flags=re.I)
    if not candidates:
        return None
    for left, right in reversed(candidates):
        value = left or right
        if value:
            return int(value)
    return None


def strip_marks(text: str) -> str:
    text = re.sub(r"\s*\[\d{1,2}\]\s*$", "", text)
    text = re.sub(r"\s*\(\d{1,2}\)\s*marks?\s*$", "", text, flags=re.I)
    return text.strip()


def extract_mcq_options(text: str) -> tuple[bool, list[str] | None, str]:
    matches = list(re.finditer(r"(?<!\w)([A-D])[\s.)-]+", text))
    if len(matches) < 3:
        return False, None, text

    options: list[str] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            options.append(f"{match.group(1)} {body}")
    question_text = text[: matches[0].start()].strip()
    return bool(options), options, question_text


def clean_mark_scheme_text(text: str) -> str:
    text = re.sub(r"\bQuestion\s+Number\b", " ", text, flags=re.I)
    text = re.sub(r"\bAdditional\s+guidance\b", " Additional guidance: ", text, flags=re.I)
    text = re.sub(r"^\s*(Answer|Mark)\s+", " ", text, flags=re.I)
    text = re.sub(r"\b\d{1,2}\s+Answer\b", " ", text, flags=re.I)
    text = re.sub(r"\bAnswer\s+Additional guidance:", " Additional guidance:", text, flags=re.I)
    text = re.sub(
        r"\b[B-D]\s+is\s+not\s+correct\b.*?(?=\b[A-D]\s+is\s+not\s+correct\b|\bAdditional guidance:|\bIgnore\b|\bAllow\b|$)",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(r"\b\d{1,2}\s+Additional guidance:\s*Mark\b", " ", text, flags=re.I)
    text = re.sub(r"\bAdditional guidance:\s*Mark\b", " ", text, flags=re.I)
    text = re.sub(r"\bTotal\s+\d+\s+marks?\b.*$", "", text, flags=re.I)
    text = text.replace("\uf0b7", "-")
    return normalize_whitespace(text)


def plain_text_to_answer_html(text: str) -> str:
    text = html.escape(normalize_whitespace(text))
    if " - " in text or "; " in text:
        parts = [part.strip(" -;") for part in re.split(r"\s+-\s+|;\s+", text) if part.strip(" -;")]
        if len(parts) > 1:
            return "<ul class=\"ms-list\">" + "".join(f"<li>{part}</li>" for part in parts) + "</ul>"
    return f"<p class=\"ms-para\">{text}</p>"


def inject_diagram_html(text: str, diagram_id: str, image_path: str) -> str:
    safe_path = html.escape(image_path, quote=True)
    figure = (
        f'<div class="figure-wrapper">'
        f'<img src="{safe_path}" alt="Diagram {html.escape(diagram_id)}">'
        f'<div class="figure-caption">Figure: {html.escape(diagram_id)}</div>'
        "</div>"
    )
    placeholder = f"[DIAGRAM:{diagram_id}]"
    if placeholder in text:
        return text.replace(placeholder, figure)
    bracket_placeholder = f"[DIAGRAM_{diagram_id}]"
    if bracket_placeholder in text:
        return text.replace(bracket_placeholder, figure)
    return text


def boxes_related(question_box: list[int], diagram_box: list[int] | None) -> bool:
    if not diagram_box:
        return False
    qy0, qx0, qy1, qx1 = question_box
    dy0, dx0, dy1, dx1 = diagram_box
    vertical_overlap = max(0, min(qy1, dy1) - max(qy0, dy0))
    diagram_height = max(1, dy1 - dy0)
    x_overlap = max(0, min(qx1, dx1) - max(qx0, dx0))
    diagram_width = max(1, dx1 - dx0)
    center_y = (dy0 + dy1) / 2
    is_inside_vertical_span = qy0 - 80 <= center_y <= qy1 + 140
    return (vertical_overlap / diagram_height > 0.2 and x_overlap / diagram_width > 0.2) or is_inside_vertical_span


def union_boxes(boxes: list[list[int]]) -> list[int] | None:
    boxes = [box for box in boxes if box and len(box) == 4]
    if not boxes:
        return None
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def normalize_box_1000(value: Any) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        ymin, xmin, ymax, xmax = [int(round(float(item))) for item in value]
    except (TypeError, ValueError):
        return None
    ymin = max(0, min(1000, ymin))
    xmin = max(0, min(1000, xmin))
    ymax = max(0, min(1000, ymax))
    xmax = max(0, min(1000, xmax))
    if ymax <= ymin or xmax <= xmin:
        return None
    return [ymin, xmin, ymax, xmax]


def pixel_bbox_to_box_1000(bbox: Any, width: int | float, height: int | float) -> list[int]:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return [
        int(round(y0 * 1000 / height)),
        int(round(x0 * 1000 / width)),
        int(round(y1 * 1000 / height)),
        int(round(x1 * 1000 / width)),
    ]


def point_bbox_to_box_1000(bbox: Any, width: int | float, height: int | float) -> list[int]:
    return pixel_bbox_to_box_1000(bbox, width, height)


def normalize_int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_json_response(text: str) -> dict[str, Any]:
    cleaned = strip_markdown_fence(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        fixed = re.sub(r"(?<!\\)\\(?![\"\\/bfnrtu])", r"\\\\", cleaned)
        return json.loads(fixed)


def strip_markdown_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def optional_config_or_env(config: dict[str, Any], key: str, env_key: str, fallback_envs: list[str]) -> str | None:
    value = config.get(key)
    if value:
        return str(value)
    configured_env = config.get(env_key)
    if configured_env and os.getenv(str(configured_env)):
        return os.getenv(str(configured_env))
    for name in fallback_envs:
        env_value = os.getenv(name)
        if env_value:
            return env_value
    return None


def normalize_base_url(value: str) -> str:
    return value[:-17] if value.endswith("/chat/completions") else value


def resolve_analysis_output_path(config: dict[str, Any], output_dir: Path, base_dir: Path) -> Path:
    value = config.get("output_json")
    if value:
        path = resolve_path(value, base_dir)
        return path if path.suffix else path / "extracted_qna.json"
    return output_dir / "extracted_qna.json"


def try_remove_dir(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        pass


def find_unprocessed_pairs(
    pdfs: list[Path],
    input_root: Path,
    output_root: Path,
    base_dir: Path,
) -> list[ExamPair]:
    """Find QP/MS pairs that haven't been processed yet."""
    all_pairs = find_exam_pairs(pdfs, {}, base_dir)
    unprocessed: list[ExamPair] = []

    for pair in all_pairs:
        qp_stem = pair.question_pdf.stem
        qna_json = output_root / "assets" / qp_stem / "analysis" / "extracted_qna.json"

        if qna_json.exists():
            continue

        html_marker = output_root / "htmls"
        if html_marker.exists():
            for html_dir in html_marker.iterdir():
                if html_dir.is_dir() and qp_stem.lower() in html_dir.name.lower():
                    qna_json_alt = html_dir / "qna.json"
                    if qna_json_alt.exists():
                        break
            else:
                unprocessed.append(pair)
                continue
        else:
            unprocessed.append(pair)

    return unprocessed
