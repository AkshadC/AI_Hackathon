import os
import re
import json
import time
import random
from collections import Counter

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from google import genai
from google.genai.errors import ServerError

from config import CFG

# ── module-level singletons ──────────────────────────────────────────
_vader = SentimentIntensityAnalyzer()
JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


# ── helpers ──────────────────────────────────────────────────────────
def get_sentiment(text: str) -> float:
    return _vader.polarity_scores(text)["compound"]


def _call_with_retry(fn, max_retries: int = 6, base_sleep: float = 1.0, max_sleep: float = 20.0):
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


def _extract_json_obj(text: str) -> dict:
    m = JSON_OBJ_RE.search(text)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return json.loads(m.group(0))


# ── 1. chunked cosine similarity over float16 memmap ────────────────
def topk_cosine_memmap(
    X_all: np.memmap,
    q_vec: np.ndarray,
    k: int = 5,
    chunk_size: int = 4096,
):
    """Returns (scores, indices) for the top-k most similar rows."""
    q = q_vec.astype(np.float32).ravel()
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm

    N = X_all.shape[0]
    k = min(k, N)
    all_scores = np.empty(N, dtype=np.float32)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = np.asarray(X_all[start:end], dtype=np.float32)
        all_scores[start:end] = chunk @ q

    top_k_idx = np.argpartition(all_scores, -k)[-k:]
    top_k_idx = top_k_idx[np.argsort(all_scores[top_k_idx])[::-1]]
    return all_scores[top_k_idx], top_k_idx


# ── 2. lightweight dedup among top-K results ────────────────────────
def _dedup_topk(X_all, indices, scores, thread_ids, threshold: float = 0.92):
    """Collapse near-duplicate threads (O(K^2) where K is tiny)."""
    k = len(indices)
    vecs = np.asarray(X_all[indices], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vecs = vecs / norms

    keep = []
    for i in range(k):
        is_dup = False
        for j in keep:
            sim = float(vecs[i] @ vecs[j])
            if sim > threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append(i)

    kept_indices = indices[keep]
    kept_scores = scores[keep]
    return kept_scores, kept_indices


# ── 3. build Gemini prompt for query summarization ──────────────────
def build_query_context(query: str, matched_threads: pd.DataFrame, top_comments: pd.DataFrame) -> str:
    lines = [f"USER QUERY: {query}\n"]
    lines.append("RELEVANT REDDIT THREADS:\n")

    for _, row in matched_threads.iterrows():
        tid = str(row["thread_id"])
        title = str(row.get("title_and_selftext", ""))[:300].replace("\n", " ").strip()
        lines.append(f"Thread: {title}")

        thread_comments = top_comments[top_comments["thread_id"] == tid]
        if not thread_comments.empty:
            for _, c in thread_comments.iterrows():
                body = str(c.get("body", ""))[:200].replace("\n", " ").strip()
                if body:
                    lines.append(f"  - {body}")
        lines.append("")

    return "\n".join(lines)[:3000]


def call_gemini_for_query(context: str) -> str:
    prompt = f"""You are Maple Lens, an AI assistant that summarizes Reddit discussions from r/Canada.

Based on the following Reddit threads and comments, provide a helpful, conversational summary that answers the user's query.
Be concise (3-5 sentences). Mention specific viewpoints from the comments when relevant.
Do NOT use markdown formatting — just plain text.

{context}"""

    client = genai.Client(api_key=os.environ.get(CFG.gemini.api_key_env))

    def _call():
        resp = client.models.generate_content(
            model=CFG.gemini.model_name,
            contents=prompt,
        )
        return resp.text

    try:
        return _call_with_retry(_call, max_retries=CFG.gemini.max_retries)
    except Exception as e:
        return f"Sorry, I couldn't generate a summary right now. ({e})"


# ── 4. per-thread stats: sentiment + comment timeline ───────────────
def _sentiment_bucket(score: float) -> str:
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    return "neutral"


def build_thread_cards(
    matched_threads: pd.DataFrame,
    matched_comments: pd.DataFrame,
    scores: np.ndarray,
    indices: np.ndarray,
    thread_ids: list,
) -> list:
    # Build a mapping from thread_id -> similarity score
    sim_map = {}
    for i, idx in enumerate(indices):
        sim_map[thread_ids[idx]] = float(scores[i])

    cards = []
    for _, row in matched_threads.iterrows():
        tid = str(row["thread_id"])
        thread_coms = matched_comments[matched_comments["thread_id"] == tid]

        # sentiment on comments
        sentiments = thread_coms["body"].fillna("").apply(get_sentiment).tolist() if not thread_coms.empty else []
        sentiment_avg = float(np.mean(sentiments)) if sentiments else 0.0
        buckets = Counter(_sentiment_bucket(s) for s in sentiments)
        sentiment_dist = {
            "positive": buckets.get("positive", 0),
            "neutral": buckets.get("neutral", 0),
            "negative": buckets.get("negative", 0),
        }

        # comment timeline (bucket by day)
        comment_timeline = []
        if not thread_coms.empty and "created_utc" in thread_coms.columns:
            coms_with_date = thread_coms.dropna(subset=["created_utc"]).copy()
            if not coms_with_date.empty:
                coms_with_date["date"] = coms_with_date["created_utc"].dt.strftime("%Y-%m-%d")
                daily = coms_with_date.groupby("date").size().reset_index(name="count")
                daily = daily.sort_values("date")
                comment_timeline = daily.to_dict(orient="records")

        # thread URL
        url = ""
        if "thread_url" in row.index:
            url = str(row["thread_url"])
        elif "external_url" in row.index:
            url = str(row["external_url"])
        elif "permalink" in row.index:
            p = str(row.get("permalink", ""))
            url = p if p.startswith("http") else f"https://www.reddit.com{p}"

        cards.append({
            "thread_id": tid,
            "title": str(row.get("title_and_selftext", ""))[:200].strip(),
            "url": url,
            "score": int(row["score"]) if pd.notna(row.get("score")) else 0,
            "num_comments": int(row["num_comments"]) if pd.notna(row.get("num_comments")) else 0,
            "similarity": round(sim_map.get(tid, 0.0), 3),
            "sentiment_avg": round(sentiment_avg, 3),
            "sentiment_dist": sentiment_dist,
            "comment_timeline": comment_timeline,
        })

    # Sort by similarity descending
    cards.sort(key=lambda c: c["similarity"], reverse=True)
    return cards


# ── 5. main query pipeline ──────────────────────────────────────────
def process_query(
    query: str,
    model,
    X_all,
    thread_ids,
    id_to_idx,
    threads_df,
    comments_df,
) -> dict:
    # 1. Embed query
    q_vec = model.encode(query, normalize_embeddings=True)

    # 2. Cosine similarity -> top K threads
    k = CFG.query.top_k_threads
    scores, indices = topk_cosine_memmap(X_all, q_vec, k=k, chunk_size=CFG.query.cosine_chunk_size)

    # 3. Dedup near-identical threads
    scores, indices = _dedup_topk(X_all, indices, scores, thread_ids, threshold=CFG.query.dedup_threshold)
    matched_ids = [thread_ids[i] for i in indices]

    # 4. Gather thread metadata + comments
    matched_threads = threads_df[threads_df["thread_id"].isin(matched_ids)].copy()
    matched_comments = comments_df[comments_df["thread_id"].isin(matched_ids)].copy()

    # 5. Rank comments per thread (top N by score)
    if not matched_comments.empty and "score" in matched_comments.columns:
        top_comments = (
            matched_comments
            .sort_values(["thread_id", "score"], ascending=[True, False])
            .groupby("thread_id").head(CFG.query.top_k_comments_per_thread)
        )
    else:
        top_comments = matched_comments.head(0)

    # 6. Build Gemini context & summarize
    context = build_query_context(query, matched_threads, top_comments)
    summary = call_gemini_for_query(context)

    # 7. Compute per-thread stats
    thread_cards = build_thread_cards(matched_threads, matched_comments, scores, indices, thread_ids)

    return {"summary": summary, "threads": thread_cards}
