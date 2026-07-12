from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from .question_analysis import ExamPair, find_exam_pairs
from .utils import list_pdfs, load_config, relative_to_or_absolute, resolve_path

USE_COLOR = (
    not os.environ.get("NO_COLOR")
    and os.environ.get("TERM") != "dumb"
    and sys.stdout.isatty()
)


def _color(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def _bold(text: str) -> str:
    return _color("1", text)


def _dim(text: str) -> str:
    return _color("2", text)


def _cyan(text: str) -> str:
    return _color("36", text)


def _green(text: str) -> str:
    return _color("32", text)


def _yellow(text: str) -> str:
    return _color("33", text)


def _magenta(text: str) -> str:
    return _color("35", text)


def _blue(text: str) -> str:
    return _color("34", text)


def _section(title: str) -> None:
    width = 60
    line = "\u2500" * width
    print(f"\n{_bold(_cyan(line))}")
    print(f"{_bold(_cyan(f'  {title}'))}")
    print(f"{_bold(_cyan(line))}")


CHECK = "\u2714"
WARN_SYM = "\u26a0"
OK_SYM = "\u2713"
HLINE = "\u2550"
EMDASH = "\u2014"


def _banner() -> None:
    tl = "\u2554" + "\u2550" * 52 + "\u2557"
    mid_l = "\u2551"
    mid_r = "\u2551"
    bl = "\u255a" + "\u2550" * 52 + "\u255d"
    title = "QNA Pipeline Interactive Launcher"
    pad = (52 - len(title)) // 2
    print()
    print(f"  {_bold(_cyan(tl))}")
    print(f"  {_bold(_cyan(mid_l))} {' ' * pad}{_bold(title)}{' ' * (52 - pad - len(title))} {_bold(_cyan(mid_r))}")
    print(f"  {_bold(_cyan(bl))}")


def _info(label: str, value: str) -> None:
    print(f"  {_bold(_blue(label))}: {_dim(value)}")


def _summary(label: str, value: str) -> None:
    print(f"  {_bold(_green(CHECK))} {_bold(label)}: {_green(value)}")


def _warn(msg: str) -> None:
    print(f"  {_bold(_yellow(WARN_SYM))} {msg}")


def _ok(msg: str) -> None:
    print(f"  {_bold(_green(OK_SYM))} {msg}")


def main() -> int:
    args = parse_args()
    last_return_code = 0

    while True:
        config_path = Path(args.config).resolve()
        base_dir = config_path.parent
        config = load_config(config_path)

        _banner()
        _info("Config", str(config_path))

        input_path = resolve_path(config["input_dir"], base_dir)
        output_path = resolve_path(config["output_dir"], base_dir)
        pdfs = list_pdfs(input_path)
        input_root = input_path if input_path.is_dir() else input_path.parent

        command = [sys.executable, "-m", "src.pipeline", "--config", str(config_path)]

        _section("Choose Papers")
        pair = choose_pair(pdfs, input_root, base_dir)
        if pair is not None:
            command.extend(["--input", display_path(pair.question_pdf, base_dir)])
            command.extend(["--qp-pdf", display_path(pair.question_pdf, base_dir)])
            _summary("QP", str(pair.question_pdf.name))
            if pair.mark_scheme_pdf is not None:
                command.extend(["--ms-pdf", display_path(pair.mark_scheme_pdf, base_dir)])
                _summary("MS", str(pair.mark_scheme_pdf.name))
            else:
                _warn("No mark scheme selected")
        else:
            _ok("Will process all PDFs in input folder")

        _section("Pipeline Stage")
        action = prompt_choice(
            "What do you want to run?",
            [
                ("Full pipeline: extract + analyze + HTML", "full"),
                ("Extract + analyze only", "analyze"),
                ("Extract diagrams only", "extract"),
            ],
            default="full",
        )
        action_labels = {
            "full": "Full pipeline",
            "analyze": "Extract + analyze",
            "extract": "Extract diagrams only",
        }
        _summary("Action", action_labels[action])

        if action in {"full", "analyze"}:
            command.append("--analyze")
            mode = prompt_choice(
                "Analysis mode?",
                [
                    ("LLM vision", "llm"),
                    ("OCR extraction + LLM cleanup", "ocr"),
                ],
                default=str(config.get("analysis", {}).get("mode") or "llm"),
            )
            command.extend(["--analysis-mode", mode])
            _summary("Analysis mode", mode)

            if mode == "ocr":
                cleanup = prompt_yes_no(
                    "Use LLM cleanup after OCR?",
                    default=bool(
                        config.get("analysis", {}).get("cleanup_with_llm", True)
                    ),
                )
                command.append(
                    "--cleanup-with-llm" if cleanup else "--no-cleanup-with-llm"
                )
                _summary("LLM cleanup", "yes" if cleanup else "no")

        if action == "full":
            command.append("--html")
            group = prompt_yes_no(
                "Group sub-parts under parent questions in HTML?",
                default=bool(config.get("html", {}).get("group_by_parent", True)),
            )
            command.append(
                "--html-group-by-parent" if group else "--no-html-group-by-parent"
            )
            _summary("Group by parent", "yes" if group else "no")

            defaults = infer_labels(pair)
            subject = prompt_text(
                "Subject label",
                default=config_value(config, "html", "subject") or defaults["subject"],
            )
            year = prompt_text(
                "Year label",
                default=config_value(config, "html", "year") or defaults["year"],
            )
            paper_key = prompt_text(
                "Paper key label",
                default=config_value(config, "html", "paper_key")
                or defaults["paper_key"],
            )
            add_option(command, "--subject", subject)
            add_option(command, "--year", year)
            add_option(command, "--paper-key", paper_key)
            _summary("Subject", subject)
            _summary("Year", year)
            _summary("Paper key", paper_key)

        _section("Output Options")
        if not args.quick and prompt_yes_no("Set custom output folder?", default=False):
            output_override = prompt_text(
                "Output folder", default=relative_to_or_absolute(output_path, base_dir)
            )
            add_option(command, "--output", output_override)
            _summary("Output folder", output_override)

        if not args.quick and prompt_yes_no("Set page ranges or advanced options?", default=False):
            _section("Advanced Options")
            add_optional_int(command, "--page-start", "First extraction page")
            add_optional_int(command, "--page-end", "Last extraction page")
            if action in {"full", "analyze"}:
                add_optional_int(
                    command, "--analysis-page-start", "First QP analysis page"
                )
                add_optional_int(command, "--analysis-page-end", "Last QP analysis page")
                add_optional_int(command, "--ms-page-start", "First MS analysis page")
                add_optional_int(command, "--ms-page-end", "Last MS analysis page")
            add_optional_int(command, "--dpi", "Extraction render DPI")
            add_optional_int(command, "--analysis-dpi", "Analysis render DPI")
            add_optional_float(command, "--confidence", "Detection confidence threshold")
            add_optional_int(
                command, "--image-size", "DocLayout-YOLO inference image size"
            )
            add_optional_float(command, "--iou", "Detection IoU threshold")
            device = prompt_text(
                "Detection device",
                default=str(config.get("detection", {}).get("device") or "cpu"),
            )
            add_option(command, "--device", device)
            _summary("Device", device)
            if prompt_yes_no("Keep rendered page images?", default=False):
                command.append("--keep-pages")
                _summary("Keep pages", "yes")
            if action in {"full", "analyze"} and prompt_yes_no(
                "Keep analysis page images?", default=False
            ):
                command.append("--keep-analysis-pages")
                _summary("Keep analysis pages", "yes")
            if prompt_yes_no(
                "Enable OCR for diagram labels?",
                default=bool(config.get("ocr", {}).get("enabled", False)),
            ):
                command.append("--ocr")
                _summary("OCR", "enabled")
            if prompt_yes_no("Apply quality filter?", default=True):
                command.append("--quality")
                _summary("Quality filter", "yes")
            else:
                command.append("--no-quality")
                _summary("Quality filter", "no")
            if prompt_yes_no(
                "Apply whitespace refinement?",
                default=bool(config.get("crop", {}).get("refine", True)),
            ):
                command.append("--refine")
                _summary("Refinement", "yes")
            else:
                command.append("--no-refine")
                _summary("Refinement", "no")

        if not args.quick and prompt_yes_no("Configure LLM provider?", default=False):
            _section("LLM Provider")
            llm_model = prompt_text(
                "LLM model",
                default=str(
                    config.get("analysis", {}).get("llm", {}).get("model") or ""
                ),
            )
            add_option(command, "--llm-model", llm_model)
            llm_base = prompt_text(
                "LLM base URL",
                default=str(
                    config.get("analysis", {}).get("llm", {}).get("base_url") or ""
                ),
            )
            add_option(command, "--llm-base-url", llm_base)
            llm_key_env = prompt_text("LLM API key env var", default="KIMI_API_KEY")
            add_option(command, "--llm-api-key-env", llm_key_env)
            if llm_model:
                _summary("LLM model", llm_model)
            if llm_base:
                _summary("LLM base URL", llm_base)
            _summary("LLM API key env", llm_key_env)

        if not args.quick and prompt_yes_no("Enable verbose logging?", default=False):
            command.append("--verbose")
            _summary("Verbose", "yes")

        print()
        print(f"  {_bold(_cyan(HLINE * 60))}")
        print(f"  {_bold('Generated Command')}")
        print(f"  {_bold(_cyan(HLINE * 60))}")
        print()
        display = format_command(command)
        if USE_COLOR:
            parts = display.split(" ")
            current = ""
            for part in parts:
                if current and len(current) + len(part) + 1 > 80:
                    print(f"    {_dim(current)}")
                    current = part
                else:
                    current = f"{current} {part}" if current else part
            if current:
                print(f"    {_dim(current)}")
        else:
            print(f"  {display}")
        print()

        if args.dry_run:
            print(f"  {_bold(_yellow(f'Dry-run mode {EMDASH} command not executed.'))}")
            last_return_code = 0
        else:
            if prompt_yes_no("Run this command now?", default=True):
                print(f"  {_bold(_green('Running pipeline...'))}")
                env = os.environ.copy()
                python_path = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = str(base_dir) + (os.pathsep + python_path if python_path else "")
                completed = subprocess.run(command, cwd=base_dir, env=env)
                last_return_code = int(completed.returncode)
                print()
                if last_return_code == 0:
                    _ok("Pipeline completed successfully.")
                else:
                    _warn(f"Pipeline exited with code {last_return_code}.")
            else:
                print(f"  {_bold(_yellow('Cancelled by user.'))}")

        print()
        if prompt_yes_no("Run another pipeline?", default=False):
            pdfs = list_pdfs(input_path)
            print()
            continue

        break

    return last_return_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive launcher for the QNA extraction pipeline."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and print a command without running it",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip optional prompts (output folder, advanced options, LLM config, verbose)",
    )
    return parser.parse_args()


def choose_pair(
    pdfs: list[Path], input_root: Path, base_dir: Path
) -> ExamPair | None:
    pairs = find_exam_pairs(pdfs, {}, base_dir)
    choices = [(pair_label(pair, input_root), pair) for pair in pairs]
    choices.append(("Process every PDF in the input folder", None))
    choices.append(("Enter custom QP/MS paths", "custom"))

    selected = prompt_choice(
        "Choose a QP/MS pair", choices, default=choices[0][1] if choices else "custom"
    )
    if selected == "custom":
        qp_pdf = prompt_existing_pdf("Question paper PDF", base_dir)
        ms_value = prompt_text("Mark scheme PDF (blank if none)", default="")
        ms_pdf = resolve_path(ms_value, base_dir) if ms_value else None
        return ExamPair(qp_pdf, ms_pdf)

    return selected


def pair_label(pair: ExamPair, input_root: Path) -> str:
    qp = relative_to_or_absolute(pair.question_pdf, input_root)
    ms = (
        relative_to_or_absolute(pair.mark_scheme_pdf, input_root)
        if pair.mark_scheme_pdf
        else "no MS"
    )
    year = infer_year(pair.question_pdf.name)
    prefix = f"{year} - " if year else ""
    return f"{prefix}{qp} -> {ms}"


def infer_labels(pair: ExamPair | None) -> dict[str, str]:
    if pair is None:
        return {
            "subject": "Exam Workbook",
            "year": "Unknown Year",
            "paper_key": "Paper",
        }

    stem = pair.question_pdf.stem
    year = infer_year(stem) or "Unknown Year"
    subject = "Exam Workbook"
    subject_match = re.search(r"igcse[-_\s]+([a-z]+)", stem, flags=re.I)
    if subject_match:
        subject = f"IGCSE {subject_match.group(1).title()}"
    paper_key = stem.upper()
    return {"subject": subject, "year": year, "paper_key": paper_key}


def infer_year(value: str) -> str | None:
    match = re.search(r"\b(19|20)\d{2}\b", value)
    return match.group(0) if match else None


def prompt_choice(
    prompt: str, choices: list[tuple[str, Any]], default: Any | None = None
) -> Any:
    if not choices:
        raise ValueError(f"No choices available for prompt: {prompt}")

    default_index = 1
    for index, (_, value) in enumerate(choices, start=1):
        if value == default:
            default_index = index
            break

    while True:
        print(f"\n{prompt}")
        for index, (label, _) in enumerate(choices, start=1):
            marker = " [default]" if index == default_index else ""
            print(f"  {index}. {label}{marker}")
        value = read_input(f"Select 1-{len(choices)} [{default_index}]: ").strip()
        if not value:
            return choices[default_index - 1][1]
        if value.isdigit() and 1 <= int(value) <= len(choices):
            return choices[int(value) - 1][1]
        print("Enter a valid number.")


def prompt_text(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = read_input(f"{prompt}{suffix}: ").strip()
    return value or default


def prompt_yes_no(prompt: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        value = read_input(f"{prompt} [{suffix}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Enter y or n.")


def read_input(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        print()
        return ""


def prompt_existing_pdf(prompt: str, base_dir: Path) -> Path:
    while True:
        value = prompt_text(prompt)
        path = resolve_path(value, base_dir)
        if path.is_file() and path.suffix.lower() == ".pdf":
            return path
        print(f"PDF not found: {path}")


def add_optional_int(command: list[str], flag: str, prompt: str) -> None:
    value = prompt_text(f"{prompt} (blank to skip)")
    if value:
        try:
            int(value)
        except ValueError:
            print(f"Skipped {flag}: expected an integer.")
            return
        command.extend([flag, value])


def add_optional_float(command: list[str], flag: str, prompt: str) -> None:
    value = prompt_text(f"{prompt} (blank to skip)")
    if value:
        try:
            float(value)
        except ValueError:
            print(f"Skipped {flag}: expected a number.")
            return
        command.extend([flag, value])


def add_option(command: list[str], flag: str, value: str | None) -> None:
    if value:
        command.extend([flag, value])


def config_value(config: dict[str, Any], section: str, key: str) -> str | None:
    value = (config.get(section) or {}).get(key)
    return str(value) if value not in (None, "") else None


def display_path(path: Path, base_dir: Path) -> str:
    return relative_to_or_absolute(path.resolve(), base_dir)


def format_command(command: list[str]) -> str:
    display = ["python" if index == 0 else part for index, part in enumerate(command)]
    return " ".join(shlex.quote(part) for part in display)


if __name__ == "__main__":
    raise SystemExit(main())