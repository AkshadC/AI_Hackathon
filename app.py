import json
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from config import CFG
from summarize_recent_data import load_all_thread_embeddings
from process_query import process_query

# ── shared resources loaded once at startup ──────────────────────────
resources = {}


@asynccontextmanager
async def lifespan(app):
    # Load embedding model
    model = SentenceTransformer(CFG.embedding.model_name)
    model.max_seq_length = CFG.embedding.max_seq_length
    resources["model"] = model

    # Load memmap embeddings + id mapping
    X_all, ids, id_to_idx, meta = load_all_thread_embeddings()
    resources["X_all"] = X_all
    resources["thread_ids"] = ids
    resources["id_to_idx"] = id_to_idx

    # Load dataframes (pre-parse dates once)
    threads_df = pd.read_csv(CFG.paths.threads_csv)
    threads_df["thread_id"] = threads_df["thread_id"].astype(str)
    if "created_utc" in threads_df.columns:
        threads_df["created_utc"] = pd.to_datetime(threads_df["created_utc"], utc=True, errors="coerce")

    comments_df = pd.read_csv(CFG.paths.comments_csv)
    comments_df["thread_id"] = comments_df["thread_id"].astype(str)
    if "created_utc" in comments_df.columns:
        comments_df["created_utc"] = pd.to_datetime(comments_df["created_utc"], utc=True, errors="coerce")

    # Build thread_url column if not present
    if "thread_url" not in threads_df.columns:
        if "external_url" in threads_df.columns:
            threads_df["thread_url"] = threads_df["external_url"].astype(str)
        elif "permalink" in threads_df.columns:
            p = threads_df["permalink"].fillna("").astype(str)
            threads_df["thread_url"] = p.where(p.str.startswith("http"), "https://www.reddit.com" + p)
        else:
            threads_df["thread_url"] = ""

    resources["threads_df"] = threads_df
    resources["comments_df"] = comments_df

    print(f"[startup] Loaded {len(threads_df)} threads, {len(comments_df)} comments, {X_all.shape[0]} embeddings")
    yield
    resources.clear()


# ── app ──────────────────────────────────────────────────────────────
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    message: str


@app.post("/analyze")
def analyze(query: Query):
    result = process_query(
        query.message,
        resources["model"],
        resources["X_all"],
        resources["thread_ids"],
        resources["id_to_idx"],
        resources["threads_df"],
        resources["comments_df"],
    )
    return result


@app.get("/topics")
def get_topics():
    path = CFG.paths.homepage_json
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []
