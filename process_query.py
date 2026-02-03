import json
import os

import numpy as np
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import CFG

def topk_cosine_memmap(all_embeddings, embedded_query, k):
    q = embedded_query.astype(np.float32)
    N = all_embeddings.shape[0]

    top_scores = np.full(k, -1.0, dtype=np.float32)
    top_idx = np.full(k, -1, dtype=np.int64)

    for start in range(0, N):
        X = np.asarray(all_embeddings, np.float32)
        scores = X @ q
        if len(scores) > k:
            idx_local = np.argpartition(scores, -k)[-k:]
        else:
            idx_local = np.arange(len(scores))




def process_query(query: str):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = CFG.embedding.batch_size_gpu if device == "cuda" else CFG.embedding.batch_size_cpu

    model = SentenceTransformer(CFG.embedding.model_name, device=device)
    model.max_seq_length = CFG.embedding.max_seq_length

    with torch.inference_mode():
            q_vec = model.encode(
                query,
                convert_to_tensor=True,
                normalize_embeddings=CFG.embedding.normalize,
                show_progress_bar=False
            )
    all_embedded_threads = np.load(CFG.paths.all_threads_emb_mmap)
    idx = json.load(CFG.paths.all_threads_ids_json)
    scores, idx = topk_cosine_memmap(all_embedded_threads, q_vec, k=200)
