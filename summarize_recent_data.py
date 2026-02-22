import json
import os
import re
import time
import random
from collections import Counter
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import umap
import hdbscan
from google import genai
from google.genai.errors import ServerError
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import CFG

_vader = SentimentIntensityAnalyzer()


def _sentiment_bucket(score: float) -> str:
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    return "neutral"


pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)
pd.set_option("display.max_columns", None)

JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


# -----------------------------
# Utilities: JSON extraction + retries
# -----------------------------
def extract_json_obj(text: str) -> dict:
    m = JSON_OBJ_RE.search(text)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return json.loads(m.group(0))


def call_with_retry(fn, max_retries: int = 6, base_sleep: float = 1.0, max_sleep: float = 20.0):
    for attempt in range(max_retries):
        try:
            return fn()
        except ServerError as e:
            msg = str(e)
            if "503" not in msg and "UNAVAILABLE" not in msg:
                raise
            sleep = min(max_sleep, base_sleep * (2 ** attempt))
            sleep = sleep * (0.7 + 0.6 * random.random())
            print(f"[Gemini] 503 overloaded. Retry {attempt+1}/{max_retries} in {sleep:.1f}s")
            time.sleep(sleep)
    raise RuntimeError("Gemini still overloaded after retries.")


# -----------------------------
# Thread URL helper
# -----------------------------
def ensure_thread_url(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Prefer external_url (your schema)
    if "external_url" in df.columns:
        df["thread_url"] = df["external_url"].astype(str)
        return df

    # else try 'url'
    if "url" in df.columns:
        df["thread_url"] = df["url"].astype(str)
        return df

    # else try permalink
    if "permalink" in df.columns:
        p = df["permalink"].fillna("").astype(str)
        df["thread_url"] = np.where(p.str.startswith("http"), p, "https://www.reddit.com" + p)
        return df

    # fallback: empty
    df["thread_url"] = ""
    return df


# -----------------------------
# Load precomputed ALL embeddings (memmap) + mapping id->row
# -----------------------------
def load_all_thread_embeddings():
    """
    Returns:
      X_all: np.memmap (N, D) float16
      thread_ids_all: list[str] length N
      id_to_idx: dict[str,int]
      meta: dict
    """
    with open(CFG.paths.all_threads_ids_json, "r") as f:
        thread_ids_all = json.load(f)
    with open(CFG.paths.all_threads_meta_json, "r") as f:
        meta = json.load(f)

    N = int(meta["n"])
    D = int(meta["dim"])
    dtype = np.float16 if meta.get("dtype", "float16") == "float16" else np.float32

    X_all = np.memmap(
        CFG.paths.all_threads_emb_mmap,
        dtype=dtype,
        mode="r",
        shape=(N, D),
    )

    # Build mapping (fast enough for typical sizes)
    id_to_idx = {tid: i for i, tid in enumerate(thread_ids_all)}
    return X_all, thread_ids_all, id_to_idx, meta


# -----------------------------
# Step 1: Build docs_df (14-day window from latest comment timestamp)
# -----------------------------
def process_for_trending_topics(threads_df: pd.DataFrame, comments_df: pd.DataFrame) -> pd.DataFrame:
    threads_df = threads_df.copy()
    comments_df = comments_df.copy()

    threads_df["created_utc"] = pd.to_datetime(threads_df["created_utc"], utc=True, errors="coerce")
    comments_df["created_utc"] = pd.to_datetime(comments_df["created_utc"], utc=True, errors="coerce")

    # keep only comments for known threads
    comments_df = comments_df.merge(threads_df[["thread_id"]], on="thread_id", how="inner")
    comments_df = comments_df[comments_df["created_utc"].notna()].copy()

    end_ts = comments_df["created_utc"].max()
    start_ts = end_ts - pd.Timedelta(days=CFG.trending.days)

    comments_df = comments_df[
        (comments_df["created_utc"] >= start_ts) & (comments_df["created_utc"] <= end_ts)
    ].copy()

    active_thread_ids = comments_df["thread_id"].astype(str).unique()
    threads_df["thread_id"] = threads_df["thread_id"].astype(str)
    threads_df = threads_df[threads_df["thread_id"].isin(active_thread_ids)].copy()
    threads_df = threads_df[threads_df["created_utc"].notna()].copy()

    # comment ranking to build "thread_doc" context (optional but useful for summarization)
    now = pd.Timestamp.now(tz="UTC")
    hours_ago = (now - comments_df["created_utc"]).dt.total_seconds() / 3600.0
    comments_df["comment_priority"] = comments_df["score"].clip(lower=0) / (1.0 + hours_ago)

    comments_ranked = comments_df.sort_values(
        ["thread_id", "comment_priority"],
        ascending=[True, False],
        kind="mergesort",
    )
    top_k = comments_ranked.groupby("thread_id", sort=False).head(CFG.trending.k_comments_per_thread)

    # small safety: cap comment body to reduce huge joins
    top_k["body"] = top_k["body"].fillna("").astype(str).str.replace("\n", " ").str.slice(0, 250)

    agg_comments = (
        top_k.groupby("thread_id", sort=False)["body"]
        .agg(" ".join)
        .reset_index(name="top_comments_text")
    )

    threads_df = ensure_thread_url(threads_df)

    docs_df = threads_df.merge(agg_comments, on="thread_id", how="left")
    docs_df["top_comments_text"] = docs_df["top_comments_text"].fillna("")
    docs_df["thread_doc"] = (
        docs_df["title_and_selftext"].fillna("").astype(str)
        + " "
        + docs_df["top_comments_text"].fillna("").astype(str)
    ).str.strip()

    return docs_df


# -----------------------------
# Step 2: Build active embedding matrix from memmap
# -----------------------------
def build_active_embeddings(
    docs_df: pd.DataFrame,
    id_to_idx: Dict[str, int],
    X_all: np.memmap,
) -> Tuple[np.ndarray, List[str]]:
    """
    Returns:
      X_active: np.ndarray float32 (M, D)
      active_ids: list[str] length M aligned to X_active
    """
    active_ids = docs_df["thread_id"].astype(str).tolist()

    idxs = []
    kept_ids = []
    for tid in active_ids:
        j = id_to_idx.get(tid)
        if j is not None:
            idxs.append(j)
            kept_ids.append(tid)

    if not idxs:
        raise ValueError("No active threads found in precomputed embedding store.")

    # pull only active rows from memmap; cast to float32 for UMAP/HDBSCAN
    X_active = np.asarray(X_all[idxs], dtype=np.float32)
    return X_active, kept_ids


# -----------------------------
# Step 3: Cluster active threads
# -----------------------------
def cluster_active_threads(X_active: np.ndarray):
    # Cache based on active-set size (simple; good enough)
    prefix = CFG.paths.trending_prefix
    red_path = CFG.paths.emb_dir / f"{prefix}_active_umap{CFG.trending.umap_components}.npy"
    lab_path = CFG.paths.emb_dir / f"{prefix}_active_labels.npy"

    # If cached and matches shape, reuse
    if red_path.exists() and lab_path.exists():
        x_red = np.load(red_path)
        labels = np.load(lab_path)
        if x_red.shape[0] == X_active.shape[0] and labels.shape[0] == X_active.shape[0]:
            return x_red, labels

    umap_model = umap.UMAP(
        n_neighbors=CFG.trending.umap_neighbors,
        n_components=CFG.trending.umap_components,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        low_memory=True,
        n_jobs=-1,
    )
    x_red = umap_model.fit_transform(X_active)
    np.save(red_path, x_red)

    # generalized HDBSCAN scaling
    N = X_active.shape[0]
    min_cluster_size = max(8, int(0.02 * N))
    min_samples = max(4, min_cluster_size // 2)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(x_red)
    np.save(lab_path, labels)

    return x_red, labels


# -----------------------------
# Step 4: Trending scores
# -----------------------------
def compute_trends(docs_df: pd.DataFrame, active_ids: List[str], labels: np.ndarray):
    topics_df = pd.DataFrame({"thread_id": active_ids, "topic_id": labels})
    df = docs_df.merge(topics_df, on="thread_id", how="inner")
    df = df[df["topic_id"] != -1].copy()

    now = pd.Timestamp.now(tz="UTC")
    hours_ago = (now - df["created_utc"]).dt.total_seconds() / 3600.0

    df["trend_score"] = (
        np.log1p(df["score"].clip(lower=0) + df["num_comments"].clip(lower=0))
        / (1.0 + hours_ago)
    )

    topic_trends = (
        df.groupby("topic_id", sort=False)
        .agg(
            topic_trend_score=("trend_score", "sum"),
            thread_count=("thread_id", "count"),
            last_activity=("created_utc", "max"),
        )
        .reset_index()
    )
    topic_trends["topic_trend_score"] /= np.sqrt(topic_trends["thread_count"])
    topic_trends = topic_trends.sort_values("topic_trend_score", ascending=False)

    return topic_trends, df


# -----------------------------
# Gemini summarization
# -----------------------------
def build_topic_context(topic_df: pd.DataFrame, max_threads: int = 5) -> str:
    # pick highest trend_score threads for better representativeness
    topic_df = topic_df.sort_values("trend_score", ascending=False).head(max_threads)

    lines = []
    for _, r in topic_df.iterrows():
        title = str(r.get("title_and_selftext", ""))[:220].replace("\n", " ").strip()
        comments = str(r.get("top_comments_text", ""))[:350].replace("\n", " ").strip()
        lines.append(f"- {title}\n  {comments}")

    ctx = "\n".join(lines)
    return ctx[:CFG.gemini.max_chars_per_topic_context]


def gemini_summarize_topic(topic_id: int, topic_context: str) -> dict:
    if CFG.gemini.mock:
        # Extract a rough headline from the first line of context
        first_line = topic_context.split("\n")[0][:60].strip("- ").strip()
        return {
            "headline": first_line[:40] if first_line else f"Topic {topic_id}",
            "summary": f"Mock summary for topic {topic_id} (MOCK_GEMINI=1).",
        }

    prompt = f"""
Return STRICT JSON only:
{{"headline":"...","summary":"..."}}

Rules:
- headline: 3–6 words, Title Case
- summary: one sentence, <= 20 words
- avoid generic words like Canada/New/Says and dates/years

CONTENT:
{topic_context}
""".strip()

    client = genai.Client(api_key=os.environ.get(CFG.gemini.api_key_env))
    def _call():
        resp = client.models.generate_content(
            model=CFG.gemini.model_name,
            contents=prompt,
        )
        return resp.text

    text = call_with_retry(_call, max_retries=CFG.gemini.max_retries)
    data = extract_json_obj(text)
    return {
        "headline": str(data.get("headline", f"Topic {topic_id}")).strip(),
        "summary": str(data.get("summary", "")).strip(),
    }


def summarize_top_topics(topic_trends: pd.DataFrame, df_full: pd.DataFrame) -> pd.DataFrame:
    top_topic_ids = topic_trends["topic_id"].head(CFG.trending.top_topics).tolist()

    rows = []
    for tid in top_topic_ids:
        topic_df = df_full[df_full["topic_id"] == tid]
        ctx = build_topic_context(topic_df, max_threads=CFG.trending.top_threads_per_topic)

        info = gemini_summarize_topic(tid, ctx)

        tr = topic_trends.loc[topic_trends["topic_id"] == tid].iloc[0]
        rows.append({
            "topic_id": int(tid),
            "topic_trend_score": float(tr["topic_trend_score"]),
            "thread_count": int(tr["thread_count"]),
            "last_activity": tr["last_activity"],
            **info
        })

    return pd.DataFrame(rows)


# -----------------------------
# Step 6: threads under each topic (latest 5) + build cards
# -----------------------------
def get_latest_threads_per_topic(df_full: pd.DataFrame, topic_ids: List[int], k: int = 5) -> pd.DataFrame:
    df = df_full[df_full["topic_id"].isin(topic_ids)].copy()
    df = df.sort_values(["topic_id", "created_utc"], ascending=[True, False])

    return (
        df.groupby("topic_id", sort=False)
          .head(k)
          .loc[:, ["topic_id", "thread_id", "title_and_selftext", "thread_url", "created_utc", "score", "num_comments"]]
    )


def build_topic_cards(
    top_topics_df: pd.DataFrame,
    top_threads_df: pd.DataFrame,
    comments_df: pd.DataFrame | None = None,
) -> List[dict]:
    cards = []
    for _, trow in top_topics_df.iterrows():
        tid = int(trow["topic_id"])

        threads = top_threads_df[top_threads_df["topic_id"] == tid]
        thread_ids_in_topic = threads["thread_id"].astype(str).tolist()

        # --- per-topic enrichment from comments ---
        sentiment_dist = {"positive": 0, "neutral": 0, "negative": 0}
        comment_timeline: list[dict] = []
        total_comments = 0
        last_updated: str | None = None

        if comments_df is not None and not comments_df.empty:
            topic_coms = comments_df[comments_df["thread_id"].astype(str).isin(thread_ids_in_topic)].copy()
            total_comments = len(topic_coms)

            if not topic_coms.empty:
                # sentiment distribution
                sentiments = topic_coms["body"].fillna("").apply(
                    lambda t: _vader.polarity_scores(t)["compound"]
                )
                buckets = Counter(_sentiment_bucket(s) for s in sentiments)
                sentiment_dist = {
                    "positive": buckets.get("positive", 0),
                    "neutral": buckets.get("neutral", 0),
                    "negative": buckets.get("negative", 0),
                }

                # comment timeline (daily counts)
                if "created_utc" in topic_coms.columns:
                    coms_dated = topic_coms.dropna(subset=["created_utc"]).copy()
                    if not coms_dated.empty:
                        coms_dated["date"] = coms_dated["created_utc"].dt.strftime("%Y-%m-%d")
                        daily = coms_dated.groupby("date").size().reset_index(name="count")
                        daily = daily.sort_values("date")
                        comment_timeline = daily.to_dict(orient="records")

                        # last_updated = most recent comment timestamp
                        last_updated = str(topic_coms["created_utc"].max())

        thread_items = []
        for _, r in threads.iterrows():
            thread_items.append({
                "thread_id": str(r["thread_id"]),
                "title": str(r["title_and_selftext"]),
                "url": str(r["thread_url"]),
                "created_utc": str(r["created_utc"]),
                "score": int(r["score"]) if pd.notna(r["score"]) else None,
                "num_comments": int(r["num_comments"]) if pd.notna(r["num_comments"]) else None,
            })

        cards.append({
            "topic_id": tid,
            "headline": str(trow.get("headline", "")),
            "summary": str(trow.get("summary", "")),
            "topic_trend_score": float(trow.get("topic_trend_score", 0.0)),
            "threads": thread_items,
            "sentiment_dist": sentiment_dist,
            "comment_timeline": comment_timeline,
            "total_comments": total_comments,
            "last_updated": last_updated,
        })
    return cards


# -----------------------------
# Main
# -----------------------------
def main():
    # Load processed datasets using CFG paths
    # If pyarrow gives newline issues, switch to engine="c"
    threads_df = pd.read_csv(CFG.paths.threads_csv, engine="pyarrow")
    comments_df = pd.read_csv(CFG.paths.comments_csv, engine="pyarrow")

    # Load cached all-thread embeddings store (memmap)
    X_all, thread_ids_all, id_to_idx, meta = load_all_thread_embeddings()

    # Build docs for active 14-day window
    docs_df = process_for_trending_topics(threads_df, comments_df)

    # Pull active embeddings from memmap
    X_active, active_ids = build_active_embeddings(docs_df, id_to_idx, X_all)

    # Cluster active set
    x_red, labels = cluster_active_threads(X_active)

    # Attach topic_id to docs_df
    topics_df = pd.DataFrame({"thread_id": active_ids, "topic_id": labels})
    df_full = docs_df.merge(topics_df, on="thread_id", how="inner")
    df_full = df_full[df_full["topic_id"] != -1].copy()

    # Compute trends
    topic_trends, df_full = compute_trends(df_full, df_full["thread_id"].tolist(), df_full["topic_id"].to_numpy())

    # Summarize top topics (or load from file if you already saved them)
    # If you already saved 'topic_summaries.csv', you can skip this call.
    top_topics_df = summarize_top_topics(topic_trends, df_full)

    # Latest 5 thread links per topic
    topic_ids = top_topics_df["topic_id"].tolist()
    top_threads = get_latest_threads_per_topic(df_full, topic_ids, k=CFG.trending.top_threads_per_topic)

    # Build cards + save to homepage JSON (pass comments_df for enrichment)
    comments_df["created_utc"] = pd.to_datetime(comments_df["created_utc"], utc=True, errors="coerce")
    comments_df["thread_id"] = comments_df["thread_id"].astype(str)
    cards = build_topic_cards(top_topics_df, top_threads, comments_df)

    with open(CFG.paths.homepage_json, "w") as f:
        json.dump(cards, f, indent=2)

    print("Saved homepage cards:", CFG.paths.homepage_json)


