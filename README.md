# 🍁 Maple Lens
### AI-Powered Topic Trend Analysis for r/Canada

Maple Lens is a full-stack AI system that identifies **what topics are trending on r/Canada** — not just which posts are popular. It clusters Reddit discussions into semantic topics, tracks how they evolve over time, and lets you ask natural language questions about what Canadians are talking about.

Built for the **AI Hackathon Thunder Bay 2026**.

![Next.js](https://img.shields.io/badge/Next.js-16-black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)
![Gemini](https://img.shields.io/badge/Gemini-Flash-4285F4)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB)

---

## What It Does

Traditional Reddit feeds rank individual posts. Maple Lens answers higher-level questions:

- **What issues are Canadians actively discussing right now?**
- **Which topics are gaining momentum across multiple threads?**
- **What's the overall sentiment around a topic?**

Ask the chatbot a question like *"What are people saying about housing?"* and get an AI summary with relevant threads, sentiment analysis, and data visualizations.

---

## Features

- **Semantic Topic Discovery** — UMAP + HDBSCAN clustering on thread embeddings (no predefined categories)
- **Trend Detection** — Time-decayed scoring based on upvotes, comments, and recency
- **AI Chat Interface** — Ask natural language questions, get Gemini-powered summaries with source threads
- **Sentiment Analysis** — VADER sentiment on comments, per-topic and per-thread breakdowns
- **Interactive Charts** — Comment activity, sentiment distribution, topic trends, sentiment over time (Recharts)
- **Semantic Search** — Cosine similarity over precomputed embeddings for fast query matching

---

## Architecture

```
┌─────────────────────┐     ┌──────────────────────┐
│   Next.js Frontend  │────▶│   FastAPI Backend     │
│   (React + Recharts)│◀────│   (Python)            │
└─────────────────────┘     └──────────┬───────────┘
                                       │
                     ┌─────────────────┼─────────────────┐
                     │                 │                  │
              ┌──────▼──────┐  ┌───────▼───────┐  ┌──────▼──────┐
              │  Embeddings │  │  Gemini API   │  │   VADER     │
              │  (MiniLM)   │  │  (Summaries)  │  │ (Sentiment) │
              └─────────────┘  └───────────────┘  └─────────────┘
```

### Pipeline

1. **Data Processing** (`data_processing.py`) — Clean and process raw Reddit threads + comments
2. **Embedding** (`save_thread_embeddings.py`) — Encode threads using `all-MiniLM-L6-v2`, save as float16 memmap
3. **Clustering & Trending** (`summarize_recent_data.py`) — UMAP dimensionality reduction → HDBSCAN clustering → trend scoring → Gemini topic labeling → JSON output
4. **API Server** (`app.py`) — FastAPI with `/topics` (trending feed) and `/analyze` (chat queries)
5. **Query Pipeline** (`process_query.py`) — Semantic search → top-K retrieval → dedup → per-thread sentiment + timeline → Gemini summary
6. **Web App** (`maple-lens-web/`) — Next.js 16 with topic cards, chat panel, thread cards, and Recharts visualizations

---

## Project Structure

```
├── app.py                      # FastAPI server (endpoints: /topics, /analyze)
├── config.py                   # All configuration (paths, model params, thresholds)
├── data_processing.py          # Raw data → processed CSVs
├── save_thread_embeddings.py   # Thread → embedding vectors (memmap)
├── summarize_recent_data.py    # Clustering + trending + Gemini summaries
├── process_query.py            # Semantic search + query summarization
├── get_gemini_results.py       # Standalone Gemini result fetcher
├── gemma_embedder/             # Custom embedder module
│
├── maple-lens-web/             # Next.js frontend
│   ├── app/
│   │   ├── page.tsx            # Main page (state management)
│   │   ├── layout.tsx          # App layout with header/footer
│   │   └── globals.css         # Global styles
│   └── components/
│       ├── MainFeed.tsx        # Trending topics + query results + charts
│       ├── ChatPanel.tsx       # Chat interface for queries
│       ├── ThreadCard.tsx      # Rich thread display (sentiment, timeline)
│       ├── TopicCharts.tsx     # 4 Recharts visualizations (2x2 grid)
│       ├── MiniChart.tsx       # Sparkline chart for thread cards
│       └── RobotBuddy.tsx      # Animated mascot
│
├── r_canada_dataset/           # Raw data (not tracked)
└── embeddings_out/             # Generated embeddings + JSON (not tracked)
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- A Gemini API key (free at [Google AI Studio](https://aistudio.google.com/)) — only needed for production

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/maple-lens.git
cd maple-lens

# Python dependencies
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt

# Frontend dependencies
cd maple-lens-web
npm install
cd ..
```

### 2. Prepare Data

Place your Reddit dataset CSVs in `r_canada_dataset/`:
- `canada_subreddit_threads_processed.csv`
- `canada_subreddit_comments_processed.csv`

### 3. Generate Embeddings (one-time)

```bash
python save_thread_embeddings.py
```

### 4. Generate Trending Topics

```bash
# Without Gemini (mock summaries for testing):
MOCK_GEMINI=1 python summarize_recent_data.py

# With Gemini (production):
export GEMINI_API_KEY="your-key-here"
python summarize_recent_data.py
```

### 5. Run the App

**Terminal 1 — Backend:**
```bash
# Mock mode (no API key needed):
MOCK_GEMINI=1 uvicorn app:app --reload --port 8000

# Production mode:
export GEMINI_API_KEY="your-key-here"
uvicorn app:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd maple-lens-web
npm run dev
```

Open **http://localhost:3000**

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Production only | Google Gemini API key |
| `MOCK_GEMINI` | No | Set to `1` to skip Gemini calls (uses placeholder summaries) |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/topics` | Returns trending topics with sentiment, timelines, and thread data |
| `POST` | `/analyze` | Accepts `{"message": "..."}`, returns AI summary + matched threads |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 16, React 19, Tailwind CSS 4, Recharts |
| Backend | FastAPI, Uvicorn |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Clustering | UMAP + HDBSCAN |
| Summarization | Google Gemini Flash |
| Sentiment | VADER |
| Data | Pandas, NumPy (float16 memmap for efficiency) |

---

## License

MIT
