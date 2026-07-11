from __future__ import annotations

import base64
import hashlib
import html
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .render import render_pdf
from .utils import ensure_dir, make_pdf_output_dir, relative_to_or_absolute, resolve_path, sanitize_name, write_json

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExamPair:
    question_pdf: Path
    mark_scheme_pdf: Path | None


@dataclass(frozen=True)
class PageText:
    page: int
    text: str
    blocks: list[dict[str, Any]]
    width: float
    height: float


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
        )
    else:
        LOGGER.warning("No mark scheme PDF available for %s", pair.question_pdf.name)

    questions = bind_diagrams_to_questions(questions, diagrams_by_page)
    qnas = match_questions_to_answers(questions, answers)

    if llm_client and bool(analysis_config.get("cleanup_with_llm", True)):
        qnas = cleanup_qnas_with_llm(qnas, llm_client)

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
) -> list[dict[str, Any]]:
    page_texts = extract_pdf_text_pages(pdf_path)
    pages_dir = ensure_dir(output_dir / "qp_pages")
    questions: list[dict[str, Any]] = []
    active_question = "1"

    for rendered_page in render_pdf(
        pdf_path,
        pages_dir,
        dpi=dpi,
        page_start=config.get("page_start"),
        page_end=config.get("page_end"),
    ):
        page_text = page_texts.get(rendered_page.page_number)
        backend = choose_backend(mode, page_text, llm_client, ocr_engine)
        page_diagrams = diagrams_by_page.get(rendered_page.page_number, [])

        try:
            if backend == "llm" and llm_client is not None:
                page_questions = llm_client.analyze_question_page(
                    rendered_page.image_path,
                    rendered_page.page_number,
                    active_question,
                    page_diagrams,
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
                page_questions = heuristic_questions_from_blocks(
                    page_text.blocks if page_text else [],
                    rendered_page.page_number,
                    page_diagrams,
                    source="digital",
                )

            normalized = [
                normalize_question_block(question, rendered_page.page_number, page_diagrams)
                for question in page_questions
            ]
            for question in normalized:
                q_num = str(question.get("question_number", ""))
                match = re.match(r"^(\d+)", q_num)
                if match:
                    active_question = match.group(1)
            normalized = fix_orphan_subparts(normalized, active_question)
            questions.extend(normalized)
            LOGGER.info(
                "QP page %s analyzed with %s backend: %s question block(s)",
                rendered_page.page_number,
                backend,
                len(normalized),
            )
        finally:
            if not keep_pages:
                rendered_page.image_path.unlink(missing_ok=True)

    if not keep_pages:
        try_remove_dir(pages_dir)

    return sort_blocks(questions)


def analyze_mark_scheme_pdf(
    pdf_path: Path,
    output_dir: Path,
    mode: str,
    dpi: int,
    keep_pages: bool,
    llm_client: "OpenAICompatibleVisionClient | None",
    ocr_engine: "PageOCREngine | None",
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    page_texts = extract_pdf_text_pages(pdf_path)
    pages_dir = ensure_dir(output_dir / "ms_pages")
    answers: list[dict[str, Any]] = []

    for rendered_page in render_pdf(
        pdf_path,
        pages_dir,
        dpi=dpi,
        page_start=config.get("ms_page_start"),
        page_end=config.get("ms_page_end"),
    ):
        page_text = page_texts.get(rendered_page.page_number)
        backend = choose_backend(mode, page_text, llm_client, ocr_engine)

        try:
            if backend == "llm" and llm_client is not None:
                page_answers = llm_client.analyze_mark_scheme_page(rendered_page.image_path, rendered_page.page_number)
            elif backend == "ocr" and ocr_engine is not None:
                ocr_lines = ocr_engine.extract_lines(rendered_page.image_path)
                page_answers = heuristic_answers_from_lines(ocr_lines, rendered_page.page_number, source="ocr")
            else:
                page_answers = heuristic_answers_from_text(page_text.text if page_text else "", rendered_page.page_number)

            normalized = [normalize_answer_block(answer, rendered_page.page_number) for answer in page_answers]
            answers.extend(normalized)
            LOGGER.info(
                "MS page %s analyzed with %s backend: %s answer block(s)",
                rendered_page.page_number,
                backend,
                len(normalized),
            )
        finally:
            if not keep_pages:
                rendered_page.image_path.unlink(missing_ok=True)

    if not keep_pages:
        try_remove_dir(pages_dir)

    return dedupe_answer_blocks(sort_blocks(answers))


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
    ) -> list[dict[str, Any]]:
        diagrams_payload = [
            {
                "diagram_id": diagram["diagram_id"],
                "box_1000": diagram["box_1000"],
                "file": diagram.get("file"),
            }
            for diagram in diagrams
        ]
        prompt = f"""You are analyzing one page of an exam question paper.

Return only valid JSON. Do not include markdown fences.

Task:
1. Extract every real question/sub-question visible on this page in reading order.
2. Preserve the question numbering exactly, e.g. "1", "1(a)", "1(b)(ii)".
3. If a question continues from the previous page and the parent number is not shown, use parent "{active_question}".
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

    def extract_lines(self, image_path: Path) -> list[dict[str, Any]]:
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


def cleanup_qnas_with_llm(qnas: list[dict[str, Any]], llm_client: OpenAICompatibleVisionClient) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for qna in qnas:
        item = dict(qna)
        try:
            item["text"] = llm_client.clean_question_html(str(item.get("text", "")))
            item["answer_text"] = llm_client.clean_answer_html(str(item.get("answer_text", "")))
        except Exception as exc:
            LOGGER.warning("LLM cleanup failed for question %s: %s", item.get("question_number"), exc)
        cleaned.append(item)
    return cleaned


def make_llm_client(config: dict[str, Any], base_dir: Path) -> OpenAICompatibleVisionClient | None:
    llm_config = config.get("llm") or {}
    mode = str(config.get("mode", "auto")).lower()
    if mode == "digital":
        return None
    if not bool(llm_config.get("enabled", True)):
        return None

    load_dotenv_file(resolve_path(llm_config.get("dotenv_path", ".env"), base_dir))

    api_key = optional_config_or_env(llm_config, "api_key", "api_key_env", ["ANALYSIS_LLM_API_KEY", "KIMI_API_KEY", "OPENAI_API_KEY"])
    model = optional_config_or_env(llm_config, "model", "model_env", ["ANALYSIS_LLM_MODEL", "KIMI_MODEL", "OPENAI_MODEL"])
    base_url = optional_config_or_env(llm_config, "base_url", "base_url_env", ["ANALYSIS_LLM_BASE_URL", "KIMI_BASE_URL", "OPENAI_BASE_URL"])

    if not api_key or not model:
        if mode == "llm":
            raise RuntimeError(
                "analysis.mode=llm requires an API key and model. Set analysis.llm.* or env vars "
                "ANALYSIS_LLM_API_KEY/ANALYSIS_LLM_MODEL, KIMI_API_KEY/KIMI_MODEL, or OPENAI_API_KEY/OPENAI_MODEL."
            )
        LOGGER.info("No LLM analysis credentials found; falling back to digital/OCR analysis when possible")
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
    if not bool(ocr_config.get("enabled", False)):
        return None
    return PageOCREngine(
        language=str(ocr_config.get("language", "en")),
        min_confidence=float(ocr_config.get("min_confidence", 0.45)),
    )


def choose_backend(
    mode: str,
    page_text: PageText | None,
    llm_client: OpenAICompatibleVisionClient | None,
    ocr_engine: PageOCREngine | None,
) -> str:
    if mode in {"llm", "ocr", "digital"}:
        return mode
    if page_text and len(page_text.text.strip()) >= 80:
        return "digital"
    if llm_client is not None:
        return "llm"
    if ocr_engine is not None:
        return "ocr"
    return "digital"


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


def extract_pdf_text_pages(pdf_path: Path) -> dict[int, PageText]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for question analysis") from exc

    pages: dict[int, PageText] = {}
    document = fitz.open(pdf_path)
    try:
        for index, page in enumerate(document, start=1):
            blocks = []
            for block in page.get_text("blocks"):
                if len(block) < 5:
                    continue
                x0, y0, x1, y1, text, *_ = block
                clean = normalize_whitespace(str(text))
                if clean:
                    blocks.append(
                        {
                            "text": clean,
                            "bbox": [float(x0), float(y0), float(x1), float(y1)],
                            "box_1000": point_bbox_to_box_1000([x0, y0, x1, y1], page.rect.width, page.rect.height),
                        }
                    )
            pages[index] = PageText(
                page=index,
                text=page.get_text("text"),
                blocks=blocks,
                width=float(page.rect.width),
                height=float(page.rect.height),
            )
    finally:
        document.close()
    return pages


def heuristic_questions_from_blocks(
    blocks: list[dict[str, Any]],
    page_number: int,
    diagrams: list[dict[str, Any]],
    source: str,
) -> list[dict[str, Any]]:
    return heuristic_questions_from_text_units(blocks, page_number, diagrams, source)


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


def heuristic_answers_from_text(text: str, page_number: int, source: str = "digital") -> list[dict[str, Any]]:
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
        label = html.escape(str(answer.get("question_number") or question_number))
        sections.append(
            f"""
            <div class="answer-section">
              <h4 class="answer-part-header">Part {label}</h4>
              <div class="answer-part-body">
                {body}
              </div>
            </div>"""
        )
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
    r"^\s*(?P<number>\d+\s*(?:\([a-z]\))?(?:\([ivxlcdm]+\))?)\s*(?=[A-Z(]|$)",
    flags=re.I,
)

ANSWER_LABEL_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<number>\d+\s*(?:\([a-z]\)\s*)+(?:\([ivxlcdm]+\)\s*)?)(?![A-Za-z0-9])",
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
    low = text.lower()
    if len(text) < 2:
        return True
    patterns = [
        "turn over",
        "blank page",
        "answer all questions",
        "total for question",
        "international gcse",
        "pearson edexcel",
        "candidate",
        "centre number",
    ]
    return any(pattern in low for pattern in patterns)


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
