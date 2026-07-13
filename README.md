# Exam Diagram Extraction & Q&A Workbook Generator

> **From scanned exam PDFs to interactive HTML workbooks — automatically.**
> Renders pages -> detects figures -> extracts questions via LLM vision -> matches mark scheme answers -> generates a premium 3-column study workbook.

---

## Pipeline Overview

```text
PDFs (QP + MS)
    |
    v
+--------------------------------------------------------------+
|  1. Render -- PyMuPDF @ 200 DPI                              |
|     -> Page images (PNG)                                      |
+--------------------------------------------------------------+
|  2. Detect -- DocLayout-YOLO figure detection                 |
|     -> Bounding boxes for figures only                        |
|     -> Quality filter: min_height, aspect_ratio, confidence   |
+--------------------------------------------------------------+
|  3. Crop -- OpenCV + whitespace trim + padding               |
|     -> Clean diagram crops (PNG)                              |
+--------------------------------------------------------------+
|  4. Analyze -- LLM Vision or OCR                             |
|     -> Extract questions from QP page images                  |
|     -> Extract answers from MS page images                    |
|     -> Match Q&A via smart number normalization               |
+--------------------------------------------------------------+
|  5. Generate -- Static HTML workbook                          |
|     -> 3-column layout: sidebar nav | question | answer       |
|     -> Dashboard with search, stats, MCQ radios               |
|     -> MathJax, diagrams inline, mark scheme panels           |
+--------------------------------------------------------------+
```

---

## Features

| Feature | Detail |
|---|---|
| **Figure Detection** | DocLayout-YOLO pretrained on document layouts |
| **False Positive Filter** | Aspect ratio + min height + confidence threshold kills dotted answer lines |
| **LLM Vision Analysis** | Extracts questions/answers from scanned PDFs via OpenAI-compatible API |
| **OCR Fallback** | PaddleOCR + heuristic splitting when no API key available |
| **Answer Matching** | Smart number normalization handles QP/MS numbering mismatches (`5(i)` -> `5(b)(i)`) |
| **Parent Intro Texts** | Extracted as separate `Q1`, `Q2`... entries with diagram placement |
| **HTML Workbook** | 3-column layout, search, MCQ radio buttons, MathJax, answer reveal toggle |
| **Batch Processing** | Thousands of PDFs, automatic QP/MS pairing by filename |
| **Interactive CLI** | Iterative launcher with guided prompts, loops for multiple runs |
| **Fast Defaults** | 200 DPI rendering plus parallel workers for extraction, analysis, and LLM cleanup |

---

## Quick Start

### Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For OCR (optional):
```powershell
pip install -r requirements-ocr.txt
```

For LLM analysis (recommended):
```powershell
pip install -r requirements-llm.txt
```

### Model

Auto-downloads DocLayout-YOLO from Hugging Face on first run.
To go offline: place `doclayout_yolo.pt` in `models/` and set `model.auto_download: false`.

### Configure API Key

Create `.env` in the project root (or use existing):

```env
KIMI_API_KEY=sk-...
KIMI_MODEL=kimi-k2.6
KIMI_BASE_URL=https://api.sfkey.cn/v1
```

Or use any OpenAI-compatible provider. See `config.yaml` -> `analysis.llm.*`.

---

## Run

### Interactive CLI (Recommended)

```powershell
python src/interactive_cli.py
```

Guided prompts for paper selection, pipeline stage, analysis mode, output options.
After each run, asks "Run another pipeline?" and loops back.

### Full Pipeline -- Extract + Analyze + HTML

```powershell
python -m src.pipeline --config config.yaml --input input\2021-qp.pdf `
  --qp-pdf input\2021-qp.pdf --ms-pdf input\2021-ms.pdf `
  --analyze --analysis-mode llm `
  --workers 4 --analysis-workers 6 --cleanup-workers 8 `
  --html --html-group-by-parent `
  --subject "IGCSE Biology" --year 2021 --paper-key "4BI1 1.0"
```

**Flags explained:**
| Flag | Why |
|---|---|
| `--input` | Limits extraction to this specific PDF (not all in input/) |
| `--qp-pdf` / `--ms-pdf` | Specifies which files to analyze for Q&A extraction |
| `--analysis-mode llm` | Forces LLM vision on every page |
| `--workers` | Parallel page workers for diagram extraction |
| `--analysis-workers` | Parallel workers for page-level QP/MS analysis |
| `--cleanup-workers` | Parallel workers for optional LLM cleanup |
| `--html-group-by-parent` | Groups sub-parts (`1(a)`, `1(b)`) under parent question (`1`) |

### Extract Only (Diagrams + Metadata)

```powershell
python -m src.pipeline --config config.yaml
```

### Custom Input/Output

```powershell
python -m src.pipeline --input input\exam.pdf --output output\my_exam `
  --device cuda:0 --confidence 0.25 --dpi 200 --workers 4 `
  --analysis-workers 6 --cleanup-workers 8 --page-start 1 --page-end 10
```

### Specify QP/MS Pair Explicitly

```powershell
python -m src.pipeline --analyze `
  --qp-pdf input\2019-biology-qp.pdf --ms-pdf input\2019-biology-ms.pdf `
  --html --subject "IGCSE Biology" --year 2019 --paper-key "4BI1-1"
```

---

## Analysis Modes

| Mode | Description |
|---|---|
| `auto` (default) | LLM vision if credentials available, falls back to OCR |
| `llm` | Force OpenAI-compatible vision analysis (best for scanned PDFs) |
| `ocr` | PaddleOCR + heuristic splitting (no API key needed) |

---

## Performance Tuning

The default pipeline now renders at `200` DPI instead of `400` DPI. This reduces page image size substantially and speeds up rendering, detection, upload, and LLM vision calls while keeping enough detail for typical scanned exam papers.

Parallel workers are enabled by default:

| Setting | Default | What it controls |
|---|---:|---|
| `render.workers` | 4 | Diagram extraction page processing |
| `analysis.workers` | 6 | QP/MS page LLM or OCR analysis |
| `analysis.cleanup_workers` | 8 | Optional LLM HTML cleanup |

For maximum throughput, increase workers until your CPU/GPU, disk, memory, or LLM provider rate limit becomes the bottleneck:

```powershell
python -m src.pipeline --config config.yaml --analyze --analysis-mode llm `
  --dpi 200 --analysis-dpi 200 `
  --workers 6 --analysis-workers 8 --cleanup-workers 12
```

If the LLM provider returns rate-limit or timeout errors, lower `--analysis-workers` and `--cleanup-workers`. Use `--workers 1 --analysis-workers 1 --cleanup-workers 1` to restore mostly sequential behavior for debugging.

---

## Output Structure

```
output/
  biology_2019_full/
    page001/
      page001_fig01.png          # cropped diagram
      page001_fig02.png
    metadata.json                   # per-PDF diagram manifest
    analysis/
      extracted_qna.json          # structured Q&A data
    html/
      index.html                  # dashboard with stats + search
      q_1.html                    # individual question page
      q_2.html
      ...
      images/                     # staged diagram copies
      _site_assets/               # CSS + JS
  metadata.json                       # aggregate manifest
  analysis_manifest.json              # analysis summary
```

---

## Config Highlights (`config.yaml`)

| Section | Key Setting |
|---|---|
| `render.dpi` | 200 (default) |
| `render.workers` | 4 parallel page workers |
| `detection.confidence` | 0.2 |
| `detection.figure_labels` | `["figure"]` |
| `crop.padding` | 20px |
| `crop.refine` | whitespace trim + re-pad |
| `quality.min_height` | 400px (kills dotted answer lines) |
| `quality.max_aspect_ratio` | 4.0 (kills wide strips) |
| `quality.min_confidence` | 0.6 (kills low-confidence detections) |
| `analysis.mode` | `auto` / `llm` / `ocr` |
| `analysis.workers` | 6 parallel page-analysis workers |
| `analysis.cleanup_workers` | 8 parallel LLM cleanup workers |
| `analysis.llm` | API key, model, base URL, cache dir |
| `html.group_by_parent` | Groups sub-parts under parent question |

---

## Code Layout

```
src/
  interactive_cli.py   # iterative interactive launcher
  pipeline.py          # main entry point, orchestrates stages
  render.py            # PyMuPDF -> page images
  detect.py            # DocLayout-YOLO figure detection
  crop.py              # OpenCV crop + trim + quality filter
  ocr.py               # PaddleOCR label extraction
  question_analysis.py # LLM vision + heuristic Q&A extraction
  html_builder.py      # QNA/metadata -> render-ready data
  html_generator.py    # Static HTML workbook generator
  utils.py             # config, paths, logging, helpers
```

---

## Notes

- **Use `--analysis-mode llm`** with a vision-capable API for best results on scanned PDFs.
- **LLM responses are cached** in `.cache/question_analysis/` -- delete to force re-analysis.
- **False positive figures** (dotted answer lines) are filtered by `quality.min_height: 400` + `quality.max_aspect_ratio: 4.0` + `quality.min_confidence: 0.6`.
- **Interactive CLI loops** after each run. Say "n" to exit, "y" to process another paper.
- **`--input` limits extraction** to the specified PDF. Without it, all PDFs in `input/` are processed.

---

**Developed by Shaikat S. &middot; SugarClass Limited**
