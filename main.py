import os
import gc
import json
import re
import numpy as np
import pandas as pd
import torch

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from gemma_embedder.gemma_embedder import save_threads_single_tensor, encode_user_query
from get_gemini_results import build_community_prompt, call_gemini, extract_text, fetch_url_text, build_urls_prompt



PROCESSED_R_CANADA_THREADS_CSV = "r_canada_dataset/canada_subreddit_threads_processed.csv"
PROCESSED_R_CANADA_COMMENTS_CSV = "r_canada_dataset/canada_subreddit_comments_processed.csv"

EMB_DIR = "embeddings_out"
THREAD_EMB_PATH = os.path.join(EMB_DIR, "threads_emb.pt")
THREAD_IDS_PATH = os.path.join(EMB_DIR, "threads_ids.json")

def strip_prefix(x: str) -> str:
    x = str(x)
    return x.split("_", 1)[1] if "_" in x else x

def load_thread_embeddings():
    emb = torch.load(THREAD_EMB_PATH, map_location="cpu")  # [N, D], normalized
    with open(THREAD_IDS_PATH, "r") as f:
        # you wrote ids as newline-joined; support both json list and newline file
        raw = f.read().strip()
        if raw.startswith("["):
            ids = json.loads(raw)
        else:
            ids = raw.splitlines()
    return ids, emb

def build_id_to_index(ids):
    return {str(tid): i for i, tid in enumerate(ids)}

def topk_threads_by_similarity(query_emb, thread_emb, k=10):
    """
    query_emb: np array (D,) or torch tensor (D,) normalized
    thread_emb: torch tensor (N, D) on CPU normalized
    returns: indices, scores
    """
    if isinstance(query_emb, np.ndarray):
        q = torch.from_numpy(query_emb).float()
    else:
        q = query_emb.detach().cpu().float()

    if q.ndim == 2:
        q = q.squeeze(0)

    # ensure normalized (dot == cosine)
    q = torch.nn.functional.normalize(q, p=2, dim=0)
    scores = thread_emb @ q  # (N,)

    top_scores, top_idx = torch.topk(scores, k=min(k, scores.shape[0]))
    return top_idx.numpy().tolist(), top_scores.numpy().tolist()

def prepare_comments_for_threads(comments_df, thread_ids_set):
    c = comments_df.copy()

    c["comment_id_clean"] = c["comment_id"].astype(str).map(strip_prefix)
    c["parent_id_clean"]  = c["parent_id"].astype(str).map(strip_prefix)

    if "thread_id" in c.columns:
        c["thread_id_clean"] = c["thread_id"].astype(str).map(strip_prefix)
    elif "link_id" in c.columns:
        c["thread_id_clean"] = c["link_id"].astype(str).map(strip_prefix)
    else:
        raise ValueError("comments_df must have either 'thread_id' or 'link_id'")

    c = c[c["thread_id_clean"].isin(thread_ids_set)].copy()

    # reply_count within selected threads
    reply_counts = c.groupby("parent_id_clean").size()
    c["reply_count"] = c["comment_id_clean"].map(reply_counts).fillna(0).astype(int)

    # top-level marker (optional)
    c["is_top_level"] = c["parent_id"].astype(str).str.startswith("t3_") | (c["parent_id_clean"] == c["thread_id_clean"])

    return c

def select_top_comments_per_thread(c, top_n=30):
    sort_cols = ["reply_count"]
    asc = [False]
    if "score" in c.columns:
        sort_cols.append("score")
        asc.append(False)

    top = (
        c.sort_values(sort_cols, ascending=asc)
         .groupby("thread_id_clean", as_index=False)
         .head(top_n)
         .copy()
    )
    return top

def encode_texts_in_batches(model, texts, batch_size=64, normalize=True):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        e = model.encode(
            batch,
            convert_to_tensor=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        ).cpu()
        embs.append(e)
        del e, batch
        gc.collect()
    return torch.cat(embs, dim=0)

def pick_best_k(emb_np, k_min=3, k_max=8):
    """
    emb_np: (n, d) float32 numpy
    returns best_k
    """
    n = emb_np.shape[0]
    if n < (k_min + 1):
        return 1  # not enough points for clustering

    best_k = k_min
    best_score = -1.0

    k_max = min(k_max, n-1)
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(emb_np)
        # silhouette needs >= 2 clusters and each cluster > 1 sample
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(emb_np, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k

def get_representatives(emb_np, labels, texts, comment_ids, per_cluster=3):
    """
    picks nearest to centroid reps for each cluster
    returns list of clusters with reps
    """
    clusters = []
    k = len(set(labels))
    for cid in range(k):
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            continue
        cluster_vecs = emb_np[idx]
        centroid = cluster_vecs.mean(axis=0, keepdims=True)
        # cosine similarity because normalized -> dot
        sims = (cluster_vecs @ centroid.T).reshape(-1)
        top_local = idx[np.argsort(-sims)[:min(per_cluster, len(idx))]]

        reps = []
        for j in top_local:
            reps.append({
                "comment_id": str(comment_ids[j]),
                "text": str(texts[j])
            })

        clusters.append({
            "cluster_id": int(cid),
            "size": int(len(idx)),
            "rep_comments": reps
        })
    # sort by size desc
    clusters.sort(key=lambda x: x["size"], reverse=True)
    return clusters

# ----------------------------
# Main “query → retrieve → cluster” flow
# ----------------------------
def run_query_pipeline(query: str, top_threads_k=10, top_comments_per_thread=30, comment_embed_batch=64):
    # load processed data
    thread_df = pd.read_csv(PROCESSED_R_CANADA_THREADS_CSV)
    comments_df = pd.read_csv(PROCESSED_R_CANADA_COMMENTS_CSV)

    # ensure thread embeddings exist
    if not (os.path.exists(THREAD_EMB_PATH) and os.path.exists(THREAD_IDS_PATH)):
        save_threads_single_tensor(thread_df, batch_size=256)

    thread_ids_ordered, thread_emb = load_thread_embeddings()
    id_to_idx = build_id_to_index(thread_ids_ordered)

    # encode query (Gemma)
    q_emb = encode_user_query(query)  # returns numpy or torch depending on your helper
    if isinstance(q_emb, np.ndarray):
        q = torch.from_numpy(q_emb).float()
    else:
        q = torch.tensor(q_emb).float() if not torch.is_tensor(q_emb) else q_emb
    q = q.squeeze()
    q = torch.nn.functional.normalize(q.cpu().float(), p=2, dim=0)

    # retrieve top threads by similarity
    top_idx, top_scores = topk_threads_by_similarity(q, thread_emb, k=top_threads_k)
    top_thread_ids = [thread_ids_ordered[i] for i in top_idx]

    top_threads = thread_df[thread_df["thread_id"].astype(str).isin(set(top_thread_ids))].copy()
    # attach similarity score
    score_map = {thread_ids_ordered[i]: float(s) for i, s in zip(top_idx, top_scores)}
    top_threads["sim_score"] = top_threads["thread_id"].astype(str).map(score_map)
    top_threads = top_threads.sort_values("sim_score", ascending=False)

    # load comments for these threads
    c = prepare_comments_for_threads(comments_df, set(top_thread_ids))
    top_comments = select_top_comments_per_thread(c, top_n=top_comments_per_thread)

    # encode only selected comments
    from gemma_embedder.gemma_embedder import model as GEMMA_MODEL  # reuse same model instance
    comment_texts = top_comments["body"].astype(str).tolist()
    comment_ids = top_comments["comment_id_clean"].astype(str).tolist()

    comment_emb = encode_texts_in_batches(GEMMA_MODEL, comment_texts, batch_size=comment_embed_batch, normalize=True)
    emb_np = comment_emb.numpy().astype(np.float32)

    # cluster with best k
    best_k = pick_best_k(emb_np, k_min=3, k_max=8)
    if best_k == 1:
        labels = np.zeros((emb_np.shape[0],), dtype=int)
    else:
        km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
        labels = km.fit_predict(emb_np)

    # representative comments per cluster
    clusters = get_representatives(emb_np, labels, comment_texts, comment_ids, per_cluster=3)

    return {
        "query": query,
        "top_threads": top_threads[["thread_id", "title", "external_url", "score", "num_comments", "sim_score"]].to_dict("records"),
        "comment_clusters": clusters,
        "selected_comments_df": top_comments  # for debugging / UI
    }
def get_two_summaries_for_ui(query: str, result: dict):
    """
    result is your pipeline output:
      result["top_threads"]      -> list[dict] with url/title/thread_id...
      result["comment_clusters"] -> list[dict] with reps
    Returns:
      (community_summary_str, url_facts_summary_str)
    """
    top_threads = result["top_threads"]
    clusters = result["comment_clusters"]

    # 1) community summary from clusters
    community_prompt = build_community_prompt(query, clusters)
    community_resp = call_gemini(community_prompt)
    community_summary = extract_text(community_resp).strip()

    # 2) fetch URL texts for the 10 threads, then summarize
    articles = []
    for t in top_threads:
        tid = str(t.get("thread_id"))
        url = t.get("url", "")
        txt = fetch_url_text(url)
        articles.append({"thread_id": tid, "url": url, "text": txt})

    urls_prompt = build_urls_prompt(query, top_threads, articles)
    urls_resp = call_gemini(urls_prompt)
    url_facts_summary = extract_text(urls_resp).strip()

    return community_summary, url_facts_summary



UUID_PAREN_REGEX = re.compile(
    r"\s*\([0-9a-fA-F\-]{32,}\)"
)

def remove_comment_ids(text: str) -> str:
    if not text:
        return text
    return UUID_PAREN_REGEX.sub("", text).strip()

def process_query(query):

    result = run_query_pipeline(
        query=query,
        top_threads_k=10,
        top_comments_per_thread=30,
        comment_embed_batch=32
    )
    community_summary, url_facts_summary = get_two_summaries_for_ui(result["query"], result)
    community_summary = remove_comment_ids(community_summary)
    url_facts_summary = remove_comment_ids(url_facts_summary)

    return {
            "community_view": community_summary,
            "facts_view": url_facts_summary,
        }

