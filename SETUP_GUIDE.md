# RAG Ingestion Pipeline — Setup & Run Guide

---

## What You Need

### Azure Resources (3 total)

| # | Resource | Purpose | Cost |
|---|----------|---------|------|
| 1 | **Azure Document Intelligence** | Extract text, tables, and images from PDFs & PPTXs | Free tier: 500 pages/month |
| 2 | **Azure OpenAI** — `gpt-5.1` deployment | Summarize images/charts, generate table titles, generate chunk summaries | Pay-per-use |
| 3 | **Azure OpenAI** — `text-embedding-ada-002` deployment | Generate embeddings for vector search | Pay-per-use |

> Resources 2 and 3 are in the **same** Azure OpenAI resource — you deploy two models inside it.

### System Dependencies

| Dependency | Required | Purpose |
|-----------|----------|---------|
| Python 3.11+ | Yes | Runtime |
| Microsoft Office (PowerPoint) | Yes — Windows only | Converts PPTX → PDF before Azure DI extraction |
| Tesseract OCR | Optional | Image classification (filters logos vs diagrams) |

> PPTX files are converted to PDF via PowerPoint COM automation (`comtypes`) before being sent to Azure DI. This requires Microsoft Office to be installed on the machine running the pipeline.

---

## Step 1: Install Python 3.11+

**Windows:**
1. Download from https://python.org/downloads
2. Run the installer — **tick "Add python.exe to PATH"** before clicking Install
3. Verify in a new terminal:

```cmd
python --version
```

Expected: `Python 3.11.x` or higher.

> Use **Command Prompt** (`cmd`) or **PowerShell**, not Git Bash. Git Bash uses Unix paths that can cause issues with PyMuPDF and OpenCV.

**macOS:**
```bash
brew install python@3.11
```

**Ubuntu/Debian:**
```bash
sudo apt-get install python3.11 python3.11-venv
```

---

## Step 2: Install Tesseract OCR (Optional but Recommended)

Used to classify images — distinguishes charts and diagrams from logos and backgrounds. The pipeline works without it but image filtering will be less precise.

**Windows:**
1. Download from https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer (default path: `C:\Program Files\Tesseract-OCR\`)
3. Add to PATH:
   - Search **"Environment Variables"** → System variables → `Path` → Edit → New
   - Add: `C:\Program Files\Tesseract-OCR`
4. Restart your terminal and verify:

```cmd
tesseract --version
```

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install -y tesseract-ocr
```

---

## Step 3: Verify Microsoft Office (for PPTX files)

PPTX files are converted to PDF using PowerPoint COM automation — Microsoft PowerPoint must be installed on the same Windows machine that runs the pipeline.

- If Office is installed, no additional steps are needed.
- If processing only PDF files, this requirement does not apply.

> **Note:** The `comtypes` Python package (included in `requirements.txt`) handles the COM interface automatically. It is only installed on Windows (`sys_platform == "win32"`).

---

## Step 4: Create Azure Document Intelligence

1. Go to https://portal.azure.com → **Create a resource**
2. Search **"Document Intelligence"** → Create
3. Settings:
   - Region: East US (or nearest to you)
   - Pricing: **F0 (Free)** — 500 pages/month
4. After deployment → **Keys and Endpoint** → copy both values:

```
Endpoint : https://your-name.cognitiveservices.azure.com/
Key 1    : abc123...
```

---

## Step 5: Create Azure OpenAI + Deploy 2 Models

1. Azure Portal → **Create a resource** → **Azure OpenAI** → Create
2. After deployment → **Azure OpenAI Studio**: https://oai.azure.com
3. **Deploy model #1** (image summarization + table titles + chunk summaries):
   - Deployments → New deployment → Model: **gpt-5.1** → Deployment name: `gpt-5.1`
4. **Deploy model #2** (vector embeddings):
   - Deployments → New deployment → Model: **text-embedding-ada-002** → Deployment name: `text-embedding-ada-002`
5. Azure Portal → your OpenAI resource → **Keys and Endpoint** → copy:

```
Endpoint : https://your-name.openai.azure.com/
Key 1    : xyz789...
```

---

## Step 6: Install the Pipeline

```cmd
cd rag_ingestion

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install --upgrade pip
pip install -r requirements.txt
```

> Takes 1–3 minutes. No PyTorch download — embeddings use Azure OpenAI.

Your prompt should show `(venv)` at the start:
```
(venv) C:\Users\you\rag_ingestion>
```

### Common Install Errors

| Error | Fix |
|-------|-----|
| `faiss-cpu` build fails | `pip install faiss-cpu==1.7.4` |
| `PyMuPDF` fails | `pip install PyMuPDF --no-cache-dir` |
| `opencv-python-headless` fails | `pip install opencv-python-headless --no-cache-dir` |
| `Microsoft Visual C++ required` | Install **Build Tools for Visual Studio 2022**: https://visualstudio.microsoft.com/visual-cpp-build-tools/ — select "C++ build tools" workload |
| `pip` not recognized | Close and reopen terminal after Python install |

---

## Step 7: Configure

```cmd
copy .env.example .env        # Windows
# cp .env.example .env        # macOS / Linux
```

Open `.env` and fill in your values from Steps 4–5:

```env
# ── Azure Document Intelligence ──
AZURE_DI_ENDPOINT=https://your-doc-intel.cognitiveservices.azure.com/
AZURE_DI_KEY=abc123...

# ── Azure OpenAI ──
AZURE_OPENAI_API_KEY=xyz789...
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT=gpt-5.1
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
```

> No quotes around values. No spaces around `=`.

---

## Step 8: Run

Activate your virtual environment first, then use the `ingest` and `query` subcommands:

### Ingest a single file

```cmd
python test_local.py ingest --file report.pdf
python test_local.py ingest --file presentation.pptx
```

### Ingest multiple files

```cmd
python test_local.py ingest --file report.pdf deck.pptx summary.pptx
```

### Ingest an entire folder (recursive — all PDFs and PPTXs including subfolders)

```cmd
python test_local.py ingest --dir C:\path\to\documents
python test_local.py ingest --dir C:\path\to\documents --name "Q3 Reports"
```

### Check how a file is classified (before ingesting)

```cmd
python test_local.py classify --file report.pdf presentation.pptx
```

Shows whether each file is detected as `pdf`, `slide_pdf`, `pptx`, or `unknown`.

### Query the vector store

```cmd
python test_local.py query --q "What is the revenue trend?"
python test_local.py query --q "leave policy" --collection <collection-id>
python test_local.py query --q "topic1" --q "topic2"          # multiple queries
python test_local.py query --query-file questions.txt          # one query per line
```

### Ingest then immediately query

```cmd
python test_local.py ingest --file report.pdf
python test_local.py query --q "What are the key findings?"
```

> The collection ID printed after ingestion can be passed to `--collection` to search only that collection.

### Expected output

```
=================================================================
  📥 Ingestion
=================================================================
  ✓ report.pdf (349,981 bytes)

  ⏳ Processing 1 file(s)...

  ──────────────────────────────────────────────────
  ✓ Pipeline complete in 18.3s
    Files:    1
    Text:     8 rows
    Tables:   3 rows
    Images:   4 rows
    Errors:   0 rows
    Chunks:   16 embedded
    Collection: a1b2c3d4-...

  📊 Vector Store: 16 total vectors
     Types: {'text': 10, 'table': 3, 'chart': 2, 'image': 1}

=================================================================
  🔍 Query
=================================================================
  Collection: a1b2c3d4-...
  Queries: 1  |  Top-K: 5

  ── Query 1: "What are the key findings?" (312ms, 5 results) ──
    [1] 0.842 | report.pdf | p.2 | text
        "Revenue grew 18% year-over-year to $4.2 billion..."
    [2] 0.781 | report.pdf | p.5 | table
        "Table Title: Quarterly Results
         Row1: Metric=Revenue, Q3=3.6B, Q4=4.2B"
    [3] 0.734 | report.pdf | p.8 | chart
        "Graph Data Table:
         Row1: Year=2020, Revenue=5M..."

  💾 Results saved to:
     data/output/queries/session_xxx/
     ├── session.json
     ├── query_000.json
     └── summary.csv
```

---

## Step 9: (Optional) Run API Server

```cmd
python run.py
```

Open http://localhost:8000/docs → Swagger UI with all endpoints.

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/ingest` | Upload + process files (multipart) |
| POST | `/api/v1/ingest/local` | Ingest by local file path |
| POST | `/api/v1/ingest/directory` | Recursively ingest a folder |
| POST | `/api/v1/search` | Vector similarity search (no persistence) |
| POST | `/api/v1/query` | Search + save result to disk |
| POST | `/api/v1/query/batch` | Batch queries + save session |
| GET | `/api/v1/collections` | List all collections |
| GET | `/api/v1/collections/{id}` | Collection metadata |
| DELETE | `/api/v1/collections/{id}` | Delete collection + vectors |
| GET | `/api/v1/vector-store/stats` | Index statistics |

Stop the server: `Ctrl+C`.

---

## What Happens Under the Hood

```
report.pdf  /  presentation.pptx
    │
    ▼  Document Classifier
    │  ├── True PPTX (ZIP with ppt/ dir)      → PowerPoint COM conversion → temp PDF
    │  ├── PDF exported from PowerPoint       → Azure DI  (is_pptx_source=True)
    │  └── Regular PDF                        → Azure DI  (is_pptx_source=False)
    │
    ▼  [PPTX only] PowerPoint COM (comtypes)
    │  Converts PPTX → PDF in a temp directory, then feeds PDF to Azure DI
    │  (requires Microsoft Office installed on Windows)
    │
    ▼  Azure Document Intelligence (prebuilt-layout)
    │  Receives a PDF for all document types
    │  ├── Figures collected first → bounding-box index built per page
    │  ├── Paragraphs → clean_ocr_text()
    │  │               overlap check: skip paragraphs inside figure regions
    │  │               (avoids duplicate text appearing in both text + image chunks)
    │  ├── Tables → table_to_markdown()
    │  │           context_before / context_after captured by y-position
    │  └── Figures → PyMuPDF crop → base64 PNG
    │
    ▼  [PPTX-sourced images only] Image Filter (classify_pdf_image)
    │  ├── Skip: unreadable by Pillow
    │  ├── Skip: pixel area < 100×100  (10,000 px minimum)
    │  ├── Skip: covers ≥ 85% of page  (background slide image)
    │  ├── Skip: < 50 KB + no OCR text + no shapes  (logo / icon)
    │  └── Keep: everything else
    │
    ▼  Auto-save useful images to disk
    │  data/output/images/<file_stem>/page001_<uuid8>.png
    │
    ▼  Azure OpenAI GPT — Image Summarization
    │  ├── CHART/GRAPH → structured "Graph Data Table:" + "Graph Description:"
    │  ├── Other image → 150-300 word description
    │  ├── Logo/blank  → SKIP
    │  └── 3 retries with exponential backoff, Semaphore(5) concurrency
    │
    ▼  Chunking
    │  ├── PPTX files  → one chunk per slide
    │  │                 (title + bullets + table rows + chart data + image descriptions)
    │  ├── PDF TEXT    → token-based 300-500 tokens, 80-token overlap
    │  ├── TABLE       → never split; converted to structured Row format
    │  │                 "Row1: col=val, col=val, ..."
    │  ├── CHART/GRAPH → "Graph Data Table:\n...\n\nGraph Description:\n..."
    │  └── IMAGE       → description as single chunk
    │
    ▼  LLM Table Title Generation
    │  For each table chunk: use surrounding text (before/after by y-position)
    │  to ask GPT for a concise title → prepended as "Table Title: ..."
    │
    ▼  GPT Chunk Summaries
    │  1-2 sentence summary for every chunk  (Semaphore(10) async)
    │
    ▼  context_before / context_after filled with final GPT summaries
    │  (neighbours get quality summaries, not placeholder text)
    │
    ▼  Azure OpenAI text-embedding-ada-002 (1536 dims)
    │  Batch embed all chunks (16 per batch), L2-normalised for cosine similarity
    │
    ▼  FAISS Index  (persistent on disk — data/vector_index/)
    │  Inner product search on normalised vectors ≡ cosine similarity
    │
    ▼  JSON Metadata  (data/output/<collection-id>/)
       ├── .collection_metadata.json
       ├── {file_id}.metadata.json   status flags, extraction stats
       ├── {file_id}.rows.json       raw extracted rows
       └── {file_id}.chunks.json     final chunks with full metadata
```

---

## Chunk Metadata Schema

Every chunk stored in the vector index contains:

```json
{
  "id": "chunk_abc123def456",
  "content": "Retail banking revenue grew by 18% YoY driven by digital channels...",
  "summary": "Retail revenue growth driven by digital channel adoption",
  "keywords": ["retail", "banking", "revenue", "digital"],
  "folder_meta": "finance/q3/",
  "document_id": "file-uuid",
  "document_name": "Q3_Business_Review.pptx",
  "file_name": "Q3_Business_Review.pptx",
  "section": "Retail Banking",
  "subsection": "Performance Overview",
  "page_or_slide_number": 12,
  "chunk_type": "text",
  "context_before": "Customer acquisition costs declined 12% across all segments...",
  "context_after": "Cost drivers and efficiency ratios discussed in the next section...",
  "image_path": null,
  "confidence_score": 0.92,
  "created_at": "2026-04-10T10:30:00+00:00"
}
```

`chunk_type` values: `text` · `table` · `chart` · `image`

---

## Output Files & Directories

```
data/
├── uploads/                          uploaded files (API mode)
├── vector_index/
│   ├── faiss.index                   FAISS vector index (persists across restarts)
│   └── metadata.pkl                  chunk metadata keyed by FAISS ID
└── output/
    ├── images/
    │   └── <file_stem>/
    │       ├── page001_a1b2c3d4.png  useful images saved automatically during ingestion
    │       └── page003_e5f6g7h8.png
    ├── queries/
    │   └── session_<id>/
    │       ├── session.json          full session (all queries + results)
    │       ├── query_000.json        per-query results
    │       └── summary.csv           quick overview (query, top score, file, page)
    └── <collection-id>/
        ├── .collection_metadata.json collection name, file count, timestamps
        ├── {file_id}.metadata.json   status flags, extraction stats, errors
        ├── {file_id}.rows.json       raw extracted rows (text/table/image per page)
        └── {file_id}.chunks.json     final chunks with full metadata + embeddings
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `'python' is not recognized` | Reinstall Python and tick "Add to PATH" |
| `ModuleNotFoundError: No module named 'src'` | `cd rag_ingestion` first, then run commands |
| `venv\Scripts\activate` fails in PowerShell | Run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` then retry |
| `AZURE_DI_ENDPOINT must be set` | Edit `.env` — add your keys |
| `401 Unauthorized` | Double-check your Azure API keys in `.env` |
| `DeploymentNotFound` | Verify `AZURE_OPENAI_DEPLOYMENT` exactly matches your model deployment name |
| `tesseract not found` | Install Tesseract (Step 2) or leave uninstalled — pipeline continues without OCR |
| `PPTX conversion failed` / `PowerPoint.Application` COM error | Microsoft Office must be installed on the same machine; pipeline only runs on Windows for PPTX files |
| `SSL certificate verify failed` | Ensure `certifi` is installed (`pip install certifi`) |
| `Microsoft Visual C++ required` | Install Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/ |
| `WinError 5 Access Denied` | Run Command Prompt as Administrator |
| Empty search results | Check `{file_id}.metadata.json` — `status` should be `"done"` |
| Slow first run | Normal — Azure DI + OpenAI API calls take time per page |
| Antivirus blocking `.env` | Add the project folder as an exclusion in Windows Defender |
| Image chunks missing | Ensure `IMAGE_MODE=base64` in `.env` (default); `bbox` mode skips image saving |

---

## Quick Reference

```cmd
:: ── One-time setup ──────────────────────────────────────────
cd rag_ingestion
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
:: Edit .env with your Azure keys

:: ── Daily use (always activate venv first) ─────────────────
venv\Scripts\activate

:: Check file classification
python test_local.py classify --file report.pdf deck.pptx

:: Ingest files
python test_local.py ingest --file report.pdf
python test_local.py ingest --file report.pdf deck.pptx
python test_local.py ingest --dir C:\Users\you\documents\
python test_local.py ingest --dir C:\Users\you\documents\ --name "Q3 Reports"

:: Query
python test_local.py query --q "What is the revenue trend?"
python test_local.py query --q "leave policy" --collection <collection-id>
python test_local.py query --query-file questions.txt

:: API server
python run.py
:: Open http://localhost:8000/docs
```

**macOS / Linux equivalent:**
```bash
source venv/bin/activate
python test_local.py ingest --file report.pdf
python test_local.py ingest --dir /path/to/documents --name "Q3 Reports"
python test_local.py query --q "What is the revenue trend?"
python run.py
```
