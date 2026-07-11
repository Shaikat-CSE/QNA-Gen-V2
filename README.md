# 📄 Exam Diagram Extraction & Q&A Workbook Generator

> **From scanned exam PDFs to interactive HTML workbooks — automatically.**  
> Renders pages → detects figures → extracts questions via LLM vision → matches mark scheme answers → generates a premium 3-column study workbook.

---

## 🧠 Pipeline Overview

```text
PDFs (QP + MS)
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  1. Render — PyMuPDF @ 400 DPI                              │
│     → Page images (PNG)                                      │
├──────────────────────────────────────────────────────────────┤
│  2. Detect — DocLayout-YOLO figure detection                 │
│     → Bounding boxes for figures only                        │
│     → Quality filter: min_height, aspect_ratio, confidence   │
├──────────────────────────────────────────────────────────────┤
│  3. Crop — OpenCV + whitespace trim + padding               │
│     → Clean diagram crops (PNG)                              │
├──────────────────────────────────────────────────────────────┤
│  4. Analyze — LLM Vision (Kimi/GPT-4o)                      │
│     → Extract questions from scanned QP page images          │
│     → Extract answers from MS digital text                   │
│     → Match Q↔A via smart number normalization               │
├──────────────────────────────────────────────────────────────┤
│  5. Generate — Static HTML workbook                          │
│     → 3-column layout: sidebar nav | question | answer       │
│     → Dashboard with search, stats, MCQ radios               │
│     → MathJax, diagrams inline, mark scheme panels           │
└──────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

| Feature | Detail |
|---|---|
| **Figure Detection** | DocLayout-YOLO pretrained on document layouts |
| **False Positive Filter** | Aspect ratio + min height + confidence threshold kills dotted answer lines |
| **Scanned PDF Q&A** | LLM vision extracts questions from image-only PDFs |
| **Answer Matching** | Smart number normalization handles QP↔MS numbering mismatches (`5(i)` → `5(b)(i)`) |
| **Parent Intro Texts** | Extracted as separate `Q1`, `Q2`... entries with diagram placement |
| **HTML Workbook** | 3-column layout, search, MCQ radio buttons, MathJax, answer reveal toggle |
| **Batch Processing** | Thousands of PDFs, automatic QP/MS pairing by filename |

---

## 🚀 Quick Start

### Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For OCR (optional, heuristic fallback):
```powershell
pip install -r requirements-ocr.txt
```

For LLM analysis (recommended for scanned PDFs):
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

Or use any OpenAI-compatible provider. See `config.yaml` → `analysis.llm.*`.

---

## 🎯 Run

### Full Pipeline — Extract + Analyze + HTML

```powershell
python -m src.pipeline --config config.yaml --analyze --html `
  --subject "IGCSE Biology" --year 2019 --paper-key "4BI1-1"
```

### Extract Only (Diagrams + Metadata)

```powershell
python -m src.pipeline --config config.yaml
```

### Custom Input/Output

```powershell
python -m src.pipeline --input input\exam.pdf --output output\my_exam `
  --device cuda:0 --confidence 0.25 --page-start 1 --page-end 10
```

### Specify QP/MS Pair Explicitly

```powershell
python -m src.pipeline --analyze `
  --qp-pdf input\2019-biology-qp.pdf --ms-pdf input\2019-biology-ms.pdf `
  --html --subject "IGCSE Biology" --year 2019 --paper-key "4BI1-1"
```

---

## ⚙️ Analysis Modes

| Mode | Description |
|---|---|
| `auto` (default) | Prefers digital text, falls back to LLM vision, then OCR |
| `llm` | Force OpenAI-compatible vision analysis (best for scanned PDFs) |
| `digital` | PyMuPDF text extraction + heuristic splitting (fastest) |
| `ocr` | PaddleOCR + heuristic splitting (no API key needed) |

---

## 📁 Output Structure

```
output/
├── biology_2019_full/
│   ├── page001/
│   │   ├── page001_fig01.png          # cropped diagram
│   │   └── page001_fig02.png
│   ├── metadata.json                   # per-PDF diagram manifest
│   ├── analysis/
│   │   └── extracted_qna.json          # structured Q&A data
│   └── html/
│       ├── index.html                  # dashboard with stats + search
│       ├── q_1.html                    # individual question page
│       ├── q_2.html
│       ├── ...
│       ├── images/                     # staged diagram copies
│       └── _site_assets/               # CSS + JS
├── metadata.json                       # aggregate manifest
└── analysis_manifest.json              # analysis summary
```

---

## 🧩 Config Highlights (`config.yaml`)

| Section | Key Setting |
|---|---|
| `render.dpi` | 400 (default) |
| `detection.confidence` | 0.2 |
| `detection.figure_labels` | `["figure"]` |
| `crop.padding` | 20px |
| `crop.refine` | whitespace trim + re-pad |
| `quality.min_height` | 400px (kills dotted answer lines) |
| `quality.max_aspect_ratio` | 4.0 (kills wide strips) |
| `quality.min_confidence` | 0.6 (kills low-confidence detections) |
| `analysis.mode` | `auto` / `llm` / `ocr` / `digital` |
| `analysis.llm` | API key, model, base URL, cache dir |
| `html.group_by_parent` | Groups sub-parts under parent question |

---

## 🛠 Code Layout

```
src/
├── pipeline.py         # main entry point, orchestrates stages
├── render.py           # PyMuPDF → page images
├── detect.py           # DocLayout-YOLO figure detection
├── crop.py             # OpenCV crop + trim + quality filter
├── ocr.py              # PaddleOCR label extraction
├── question_analysis.py # LLM vision + heuristic Q&A extraction
├── html_builder.py     # QNA/metadata → render-ready data
├── html_generator.py   # Static HTML workbook generator
└── utils.py            # config, paths, logging, helpers
```

---

## 📝 Notes

- **QP must be scanned?** Use `--analysis-mode llm` with a vision-capable API.
- **MS must have digital text** for answer extraction (heuristic parsing).
- **LLM responses are cached** in `.cache/question_analysis/` — delete to force re-analysis.
- **False positive figures** (dotted answer lines) are filtered by `quality.min_height: 400` + `quality.max_aspect_ratio: 4.0` + `quality.min_confidence: 0.6`.