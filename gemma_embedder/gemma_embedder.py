import os, gc
import torch
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer

MODEL_NAME = "google/embeddinggemma-300m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = "embeddings_out"
os.makedirs(OUT_DIR, exist_ok=True)

model = SentenceTransformer(MODEL_NAME, device=DEVICE)
model.max_seq_length = 256


def encode_user_query(query):
    query_embeddings = model.encode_query(query)
    return query_embeddings

@torch.inference_mode()
def save_threads_single_tensor(
    threads_df,
    batch_size=128,
    out_dir=OUT_DIR,
    prefix="threads",
    normalize=True
):
    ids = threads_df["thread_id"].astype(str).tolist()
    texts = threads_df["thread_doc"].astype(str).tolist()

    # figure out embedding dim once
    sample = model.encode(
        [texts[0]],
        convert_to_tensor=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )
    emb_dim = sample.shape[-1]
    del sample
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    N = len(texts)
    all_embs = torch.empty((N, emb_dim), dtype=torch.float32)  # CPU prealloc

    write_idx = 0
    for i in tqdm(range(0, N, batch_size), desc="Embedding threads"):
        batch_txt = texts[i:i+batch_size]

        emb = model.encode(
            batch_txt,
            convert_to_tensor=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        ).cpu()

        bsz = emb.shape[0]
        all_embs[write_idx:write_idx+bsz] = emb
        write_idx += bsz

        del emb, batch_txt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    torch.save(all_embs, os.path.join(out_dir, f"{prefix}_emb.pt"))

    # real JSON
    import json
    with open(os.path.join(out_dir, f"{prefix}_ids.json"), "w") as f:
        json.dump(ids, f)

    print("Saved single tensor:", all_embs.shape)
    return all_embs