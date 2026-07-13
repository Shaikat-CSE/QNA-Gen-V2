from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from pathlib import Path
from typing import Any

from .html_generator import HTMLGenerator
from .utils import ensure_dir, sanitize_name, write_json

NO_ANSWER_TEXT = "No official mark scheme answer is available for this item."


def build_from_qna_json(
    qna_json_path: Path,
    output_dir: Path,
    subject_name: str | None = None,
    year: str | None = None,
    paper_key: str | None = None,
    group_by_parent: bool = True,
    copy_images: bool = True,
) -> dict[str, Any]:
    payload = load_json(qna_json_path)
    raw_qnas = extract_qna_list(payload)
    inferred_subject, inferred_year, inferred_paper = infer_names_from_path(qna_json_path)

    return build_workbook(
        raw_qnas,
        output_dir=output_dir,
        source_dir=qna_json_path.parent,
        subject_name=subject_name or inferred_subject,
        year=year or inferred_year,
        paper_key=paper_key or inferred_paper,
        group_by_parent=group_by_parent,
        copy_images=copy_images,
    )


def build_from_metadata(
    metadata_path: Path,
    output_dir: Path,
    subject_name: str | None = None,
    year: str | None = None,
    paper_key: str | None = None,
    copy_images: bool = True,
) -> dict[str, Any]:
    payload = load_json(metadata_path)
    inferred_subject, inferred_year, inferred_paper = infer_names_from_metadata(payload, metadata_path)
    qnas = metadata_to_review_qnas(payload, metadata_path.parent)

    return build_workbook(
        qnas,
        output_dir=output_dir,
        source_dir=metadata_path.parent,
        subject_name=subject_name or inferred_subject,
        year=year or inferred_year,
        paper_key=paper_key or inferred_paper,
        group_by_parent=False,
        copy_images=copy_images,
    )


def build_workbook(
    raw_qnas: list[dict[str, Any]],
    output_dir: Path,
    source_dir: Path | None,
    subject_name: str,
    year: str,
    paper_key: str,
    group_by_parent: bool = True,
    copy_images: bool = True,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    prepared_qnas = prepare_qnas(
        raw_qnas,
        output_dir=output_dir,
        source_dir=source_dir,
        group_by_parent=group_by_parent,
        copy_images=copy_images,
    )

    generator = HTMLGenerator(str(output_dir), subject_name=subject_name, year=year, paper_key=paper_key)
    remove_stale_question_pages(output_dir)
    generated_pages = [
        generator.generate_qna_page(qna, prepared_qnas, index)
        for index, qna in enumerate(prepared_qnas)
    ]
    dashboard_path = Path(generator.generate_dashboard(prepared_qnas))

    manifest = {
        "dashboard": str(dashboard_path),
        "count": len(prepared_qnas),
        "pages": generated_pages,
        "subject": subject_name,
        "year": year,
        "paper_key": paper_key,
    }
    write_json(output_dir / "html_manifest.json", manifest)
    return manifest


def remove_stale_question_pages(output_dir: Path) -> None:
    for path in output_dir.glob("q_*.html"):
        if path.is_file():
            path.unlink()


def prepare_qnas(
    raw_qnas: list[dict[str, Any]],
    output_dir: Path,
    source_dir: Path | None,
    group_by_parent: bool,
    copy_images: bool,
) -> list[dict[str, Any]]:
    if group_by_parent and looks_like_flat_qnas(raw_qnas):
        render_qnas = group_qnas_by_parent([normalize_flat_qna(item, index) for index, item in enumerate(raw_qnas)])
    else:
        render_qnas = [normalize_render_qna(item, index) for index, item in enumerate(raw_qnas)]

    for qna in render_qnas:
        stage_associated_images(qna, output_dir=output_dir, source_dir=source_dir, copy_images=copy_images)

    return render_qnas


def group_qnas_by_parent(flat_qnas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Correct missing letter prefixes for roman numeral subparts (e.g. 5(i) -> 5(b)(i), 7(iii) -> 7(a)(iii))
    parent_to_items: dict[str, list[dict[str, Any]]] = {}
    for qna in flat_qnas:
        q_num = str(qna["question_number"])
        match = re.match(r"^(\d+)", q_num)
        parent_num = match.group(1) if match else q_num
        parent_to_items.setdefault(parent_num, []).append(qna)

    corrected_flat_qnas: list[dict[str, Any]] = []
    for parent_num, items in parent_to_items.items():
        latest_letter = None
        for item in items:
            q_num = str(item["question_number"])
            letter_match = re.search(r"^\d+\(([a-zA-Z])\)", q_num)
            if letter_match:
                matched_letter = letter_match.group(1).lower()
                if matched_letter not in ("i", "v", "x"):
                    latest_letter = matched_letter
            
            roman_match = re.match(r"^(\d+)\(([ivxIVX]+)\)$", q_num)
            if roman_match and latest_letter:
                roman_part = roman_match.group(2)
                item["question_number"] = f"{parent_num}({latest_letter})({roman_part})"
            corrected_flat_qnas.append(item)

    # 1.5 Inject virtual subpart intros for any letters that have nested children but no explicit intro item
    parent_to_corrected: dict[str, list[dict[str, Any]]] = {}
    for qna in corrected_flat_qnas:
        q_num = str(qna["question_number"])
        match = re.match(r"^(\d+)", q_num)
        parent_num = match.group(1) if match else q_num
        parent_to_corrected.setdefault(parent_num, []).append(qna)

    final_flat_qnas: list[dict[str, Any]] = []
    for parent_num, items in parent_to_corrected.items():
        existing_numbers = {str(item["question_number"]) for item in items}
        active_letters = set()
        for item in items:
            q_num = str(item["question_number"])
            # check if it is a nested item like parent(letter)(child)
            match = re.match(r"^\d+\(([a-zA-Z])\)\([^)]+\)", q_num)
            if match:
                active_letters.add(match.group(1).lower())

        # For each active letter prefix, if there is no explicit intro item, insert a virtual placeholder intro
        for letter in sorted(active_letters):
            intro_num = f"{parent_num}({letter})"
            if intro_num not in existing_numbers:
                virtual_intro = {
                    "question_number": intro_num,
                    "text": "",
                    "marks": None,
                    "table_html": None,
                    "is_mcq": False,
                    "mcq_options": [],
                    "associated_images": [],
                    "answer_text": NO_ANSWER_TEXT,
                    "answer_blocks": [],
                }
                items.append(virtual_intro)
                existing_numbers.add(intro_num)

        final_flat_qnas.extend(items)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for qna in final_flat_qnas:
        q_num = str(qna["question_number"])
        match = re.match(r"^(\d+)", q_num)
        parent_num = match.group(1) if match else q_num
        grouped.setdefault(parent_num, []).append(qna)

    parent_qnas: list[dict[str, Any]] = []
    for parent_num in sorted(grouped.keys(), key=parent_sort_key):
        subparts = sorted(grouped[parent_num], key=lambda item: str(item.get("question_number", "")))

        parent_text_blocks: list[str] = []
        all_associated_images: list[str] = []
        all_answers: list[tuple[str, str]] = []
        total_marks = 0
        has_mcq = False
        seen_images: set[str] = set()

        for subpart in subparts:
            sub_q_num = str(subpart["question_number"])
            sub_text = str(subpart.get("text") or "")
            sub_marks = subpart.get("marks")

            # Intro block detection logic:
            is_parent_intro = (sub_q_num == parent_num)
            is_sub_intro = False
            for other in subparts:
                other_num = str(other["question_number"])
                if other_num != sub_q_num and other_num.startswith(sub_q_num):
                    is_sub_intro = True
                    break

            # Find matching intro prefixes within the parent group (excluding the root parent number)
            matching_intros = []
            for other in subparts:
                other_num = str(other["question_number"])
                # check if other is a sub-intro and a prefix of sub_q_num (ignoring parent_num root)
                if other_num != parent_num and other_num != sub_q_num and sub_q_num.startswith(other_num):
                    # verify it is actually classified as an intro
                    other_is_intro = False
                    for check in subparts:
                        check_num = str(check["question_number"])
                        if check_num != other_num and check_num.startswith(other_num):
                            other_is_intro = True
                            break
                    if other_is_intro:
                        matching_intros.append(other_num)

            is_nested = False
            if matching_intros:
                # Strip the longest matching subpart intro prefix to get the relative nested label
                longest_intro = max(matching_intros, key=len)
                label_display = sub_q_num[len(longest_intro):]
                is_nested = True
            else:
                # Fallback to stripping only the parent number
                label_display = sub_q_num[len(parent_num):] if sub_q_num.startswith(parent_num) else sub_q_num

            table_html = ""
            if subpart.get("table_html"):
                table_html = f'<div class="table-wrapper">{subpart["table_html"]}</div>'
                # If there's also an inline table in sub_text, remove it to keep only table_html
                if "<table" in sub_text.lower():
                    sub_text = re.sub(r"<table.*?>.*?</table>", "", sub_text, flags=re.DOTALL | re.IGNORECASE).strip()

            mcq_html = ""
            if subpart.get("is_mcq") and subpart.get("mcq_options"):
                has_mcq = True
                mcq_html = render_mcq_options(sub_q_num, subpart["mcq_options"])

            nested_class = " nested" if is_nested else ""

            if is_parent_intro:
                parent_text_blocks.append(f'<div class="question-part-parent">{sub_text}{table_html}{mcq_html}</div>')
            elif is_sub_intro:
                # Sub-question intro (e.g. 1(b), 2(c)): Format with intro wrapper, no borders/checkboxes
                parent_text_blocks.append(
                    f"""
                    <div class="question-part-intro{nested_class}" id="part-{sanitize_name(sub_q_num)}">
                      <div class="part-label-intro">{html.escape(label_display)}</div>
                      <div class="part-body-intro">
                        <p>{sub_text}</p>
                        {table_html}
                        {mcq_html}
                      </div>
                    </div>"""
                )
            else:
                # Standard question part (leaf node)
                marks_html = f' <span class="part-marks">[{sub_marks} marks]</span>' if sub_marks else ""
                parent_text_blocks.append(
                    f"""
                    <div class="question-part{nested_class}" id="part-{sanitize_name(sub_q_num)}">
                      <div class="part-label">{html.escape(label_display)}</div>
                      <div class="part-body">
                        <p>{sub_text}{marks_html}</p>
                        {table_html}
                        {mcq_html}
                      </div>
                    </div>"""
                )

            if sub_marks:
                try:
                    total_marks += int(sub_marks)
                except (TypeError, ValueError):
                    pass

            for image_path in normalize_image_list(subpart.get("associated_images")):
                if image_path not in seen_images:
                    seen_images.add(image_path)
                    all_associated_images.append(image_path)

            # Only collect answers for non-intro parts
            if not is_sub_intro:
                answer_text = str(subpart.get("answer_text") or "")
                if answer_text and answer_text != NO_ANSWER_TEXT:
                    all_answers.append((sub_q_num, answer_text))

        answer_html = render_answer_sections(all_answers)
        parent_qnas.append(
            {
                "question_number": parent_num,
                "text_html": "\n".join(parent_text_blocks),
                "marks": total_marks if total_marks > 0 else None,
                "is_mcq": has_mcq,
                "associated_images": all_associated_images,
                "answer_html": answer_html,
            }
        )

    return parent_qnas


def render_mcq_options(question_number: str, options: list[Any]) -> str:
    mcq_id = f"mcq-{sanitize_name(question_number)}"
    items: list[str] = []
    for option in options:
        if isinstance(option, dict):
            label = str(option.get("option") or option.get("label") or "").strip()
            body = str(option.get("text") or option.get("body") or "").strip()
        else:
            option_text = str(option).strip()
            match = re.match(r"^(?P<label>[A-H])[\s.)-]+(?P<body>.+)$", option_text)
            if match:
                label = match.group("label")
                body = match.group("body")
            else:
                label = option_text[:1].upper() if option_text else ""
                body = option_text[1:].strip() if len(option_text) > 1 else option_text

        item_id = sanitize_name(f"option-{mcq_id}-{label}")
        items.append(
            f"""
            <li class="mcq-option-item">
              <input type="radio" id="{item_id}" name="{mcq_id}" class="mcq-radio-input" value="{html.escape(label)}">
              <label for="{item_id}" class="mcq-option-label">
                <span class="option-letter">{html.escape(label)}.</span>
                <span class="option-text">{body}</span>
              </label>
            </li>"""
        )

    return '<ul class="mcq-options-container">' + "\n".join(items) + "</ul>"


def render_answer_sections(answers: list[tuple[str, str]]) -> str:
    if not answers:
        return "<p class='ms-para'>No official mark scheme answer found for this question block.</p>"

    sections = []
    for question_number, answer_text in answers:
        if 'class="answer-section"' in answer_text or "class='answer-section'" in answer_text:
            sections.append(answer_text)
            continue
        sections.append(
            f"""
            <div class="answer-section">
              <h4 class="answer-part-header">Part {html.escape(question_number)}</h4>
              <div class="answer-part-body">
                {answer_text}
              </div>
            </div>"""
        )
    return "\n".join(sections)


def remove_figure_captions(html_content: str) -> str:
    # Matches and removes `<div class="figure-caption">...</div>`
    pattern = r'<div class="figure-caption">.*?</div>'
    return re.sub(pattern, '', html_content, flags=re.DOTALL | re.IGNORECASE)


def stage_associated_images(
    qna: dict[str, Any],
    output_dir: Path,
    source_dir: Path | None,
    copy_images: bool,
) -> None:
    images = normalize_image_list(qna.get("associated_images"))
    if not images:
        return

    text_html = str(qna.get("text_html") or "")
    staged_sources: list[str] = []

    import base64
    import mimetypes

    for image_value in images:
        source_image = resolve_existing_image(image_value, source_dir)
        if not source_image or not source_image.exists():
            continue

        # Convert to Base64 Data URI
        mime_type, _ = mimetypes.guess_type(source_image)
        if not mime_type:
            mime_type = "image/png"
        try:
            with source_image.open("rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            base64_src = f"data:{mime_type};base64,{encoded}"
        except Exception:
            continue

        text_html = replace_image_references(text_html, image_value, source_image, base64_src)
        if not html_references_image(text_html, base64_src, image_value):
            text_html += "\n" + figure_html(base64_src, Path(image_value).name)

        staged_sources.append(base64_src)

    text_html = remove_figure_captions(text_html)
    qna["text_html"] = text_html
    qna["associated_images"] = staged_sources


def metadata_to_review_qnas(payload: Any, metadata_dir: Path) -> list[dict[str, Any]]:
    diagrams = extract_diagrams(payload)
    qnas: list[dict[str, Any]] = []

    for index, diagram in enumerate(diagrams, start=1):
        page = diagram.get("page", "?")
        figure = diagram.get("figure", index)
        image_path = resolve_metadata_image(diagram.get("file"), metadata_dir)
        question_number = f"p{page}f{figure}"
        source_pdf = diagram.get("pdf") or diagram.get("source_pdf") or "Unknown PDF"
        confidence = diagram.get("confidence")
        crop_size = format_crop_size(diagram)

        qnas.append(
            {
                "question_number": question_number,
                "text_html": (
                    '<div class="question-part-parent">'
                    f"Diagram extracted from page {html.escape(str(page))}, figure {html.escape(str(figure))}."
                    "</div>"
                ),
                "marks": None,
                "is_mcq": False,
                "associated_images": [str(image_path)] if image_path else [],
                "answer_html": metadata_answer_html(source_pdf, page, figure, confidence, crop_size),
            }
        )

    return qnas


def metadata_answer_html(
    source_pdf: Any,
    page: Any,
    figure: Any,
    confidence: Any,
    crop_size: str,
) -> str:
    confidence_text = "n/a" if confidence is None else f"{float(confidence):.3f}"
    return f"""
    <div class="answer-section">
      <h4 class="answer-part-header">Metadata</h4>
      <div class="answer-part-body">
        <p class="ms-para"><strong>Source:</strong> {html.escape(str(source_pdf))}</p>
        <p class="ms-para"><strong>Page:</strong> {html.escape(str(page))}</p>
        <p class="ms-para"><strong>Figure:</strong> {html.escape(str(figure))}</p>
        <p class="ms-para"><strong>Detector confidence:</strong> {html.escape(confidence_text)}</p>
        <p class="ms-para"><strong>Crop size:</strong> {html.escape(crop_size)}</p>
      </div>
    </div>"""


def normalize_flat_qna(item: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "question_number": str(item.get("question_number") or item.get("question") or item.get("id") or index + 1),
        "text": str(item.get("text") or item.get("text_html") or ""),
        "marks": item.get("marks"),
        "table_html": item.get("table_html"),
        "is_mcq": bool(item.get("is_mcq", False)),
        "mcq_options": item.get("mcq_options") or [],
        "associated_images": normalize_image_list(item.get("associated_images") or item.get("images")),
        "answer_text": str(item.get("answer_text") or item.get("answer_html") or NO_ANSWER_TEXT),
    }


def normalize_render_qna(item: dict[str, Any], index: int) -> dict[str, Any]:
    answer_html = item.get("answer_html")
    if not answer_html:
        answer_text = str(item.get("answer_text") or NO_ANSWER_TEXT)
        answer_html = render_answer_sections([(str(item.get("question_number") or index + 1), answer_text)])

    return {
        "question_number": str(item.get("question_number") or item.get("question") or item.get("id") or index + 1),
        "text_html": str(item.get("text_html") or item.get("text") or ""),
        "marks": item.get("marks"),
        "is_mcq": bool(item.get("is_mcq", False)),
        "associated_images": normalize_image_list(item.get("associated_images") or item.get("images")),
        "answer_html": str(answer_html),
    }


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_qna_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if isinstance(payload, dict):
        for key in ("qnas", "questions", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]

    raise ValueError("QNA JSON must be a list or contain one of: qnas, questions, items, data")


def extract_diagrams(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("diagrams"), list):
        return [item for item in payload["diagrams"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise ValueError("Metadata JSON must be a list or contain a diagrams list")


def looks_like_flat_qnas(items: list[dict[str, Any]]) -> bool:
    return any("answer_text" in item or ("text" in item and "text_html" not in item) for item in items)


def parent_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def normalize_image_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, (str, Path)):
        return [str(value)]
    if isinstance(value, list):
        images: list[str] = []
        for item in value:
            if isinstance(item, dict):
                path = item.get("filepath") or item.get("file") or item.get("path")
                if path:
                    images.append(str(path))
            elif item:
                images.append(str(item))
        return images
    return []


def resolve_existing_image(image_value: str, source_dir: Path | None) -> Path | None:
    image_path = Path(image_value)
    candidates = [image_path] if image_path.is_absolute() else []
    
    # If absolute path doesn't exist, try resolving under assets/
    if image_path.is_absolute():
        parts = image_path.parts
        if "output" in parts:
            idx = parts.index("output")
            # Try inserting 'assets' into path
            new_parts = list(parts[:idx+1]) + ["assets"] + list(parts[idx+1:])
            candidates.append(Path(*new_parts))
            
    if source_dir and not image_path.is_absolute():
        candidates.append(source_dir / image_path)
    if not image_path.is_absolute():
        candidates.append(Path.cwd() / image_path)

    # Search relative to source_dir even if absolute path is stored
    if source_dir:
        candidates.append(source_dir.parent / image_path.name)
        candidates.append(source_dir.parent.parent / image_path.name)
        try:
            if len(image_path.parts) >= 2:
                sub_path = Path(image_path.parts[-2]) / image_path.parts[-1]
                candidates.append(source_dir.parent / sub_path)
                candidates.append(source_dir.parent.parent / sub_path)
        except Exception:
            pass

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_metadata_image(file_value: Any, metadata_dir: Path) -> Path | None:
    if not file_value:
        return None

    file_path = Path(str(file_value))
    candidates = [file_path] if file_path.is_absolute() else []
    if not file_path.is_absolute():
        candidates.extend([metadata_dir / file_path, metadata_dir.parent / file_path])

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0] if candidates else None


def unique_image_destination(source_image: Path, output_images_dir: Path) -> Path:
    suffix = source_image.suffix or ".png"
    stem = sanitize_name(source_image.stem)
    destination = output_images_dir / f"{stem}{suffix}"

    if not destination.exists():
        return destination

    try:
        if destination.resolve() == source_image.resolve():
            return destination
    except FileNotFoundError:
        pass

    counter = 2
    while True:
        candidate = output_images_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def normalize_html_image_src(image_value: str) -> str:
    value = str(image_value).replace("\\", "/")
    if value.startswith("images/") or value.startswith("http://") or value.startswith("https://"):
        return value
    return value


def replace_image_references(
    text_html: str,
    original_value: str,
    source_image: Path | None,
    relative_src: str,
) -> str:
    replacements = {str(original_value), str(original_value).replace("\\", "/")}
    if source_image:
        replacements.add(str(source_image))
        replacements.add(source_image.as_posix())

    for old_value in replacements:
        if old_value:
            text_html = text_html.replace(old_value, relative_src)
    return text_html


def html_references_image(text_html: str, relative_src: str, original_value: str) -> bool:
    image_name = Path(str(original_value)).name
    return relative_src in text_html or (bool(image_name) and image_name in text_html and "<img" in text_html)


def figure_html(relative_src: str, caption: str) -> str:
    safe_src = html.escape(relative_src, quote=True)
    return f"""
    <div class="figure-wrapper">
      <img src="{safe_src}" alt="Diagram">
    </div>"""


def format_crop_size(diagram: dict[str, Any]) -> str:
    width = diagram.get("crop_width")
    height = diagram.get("crop_height")
    if width and height:
        return f"{width} x {height}px"
    return "n/a"


def infer_names_from_path(path: Path) -> tuple[str, str, str]:
    base = path.stem.replace("-", " ").replace("_", " ").title()
    year_match = re.search(r"\b(19|20)\d{2}\b", path.stem)
    year = year_match.group(0) if year_match else "Unknown Year"
    return base, year, path.stem.upper()


def infer_names_from_metadata(payload: Any, metadata_path: Path) -> tuple[str, str, str]:
    source_name = metadata_path.parent.name
    if isinstance(payload, dict):
        source_name = str(payload.get("pdf") or payload.get("source_pdf") or source_name)
    stem = Path(source_name).stem
    subject = stem.replace("-", " ").replace("_", " ").title()
    year_match = re.search(r"\b(19|20)\d{2}\b", stem)
    year = year_match.group(0) if year_match else "Unknown Year"
    return subject or "Diagram Extraction Review", year, stem.upper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HTML workbook output from QNA JSON or extraction metadata.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--qna-json", type=Path, help="Path to structured QNA JSON")
    source.add_argument("--metadata", type=Path, help="Path to extraction metadata.json")
    parser.add_argument("--output", type=Path, required=True, help="HTML output directory")
    parser.add_argument("--subject", help="Workbook subject/title")
    parser.add_argument("--year", help="Exam year label")
    parser.add_argument("--paper-key", help="Paper key label")
    parser.add_argument("--group-by-parent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--copy-images", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.qna_json:
        build_from_qna_json(
            args.qna_json.resolve(),
            args.output.resolve(),
            subject_name=args.subject,
            year=args.year,
            paper_key=args.paper_key,
            group_by_parent=args.group_by_parent,
            copy_images=args.copy_images,
        )
    else:
        build_from_metadata(
            args.metadata.resolve(),
            args.output.resolve(),
            subject_name=args.subject,
            year=args.year,
            paper_key=args.paper_key,
            copy_images=args.copy_images,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
