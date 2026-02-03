import os
import json
import numpy as np
import pandas as  pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from config import CFG
import pyarrow.csv as pv

def read_csv_arrow_allow_newlines(path: str) -> pd.DataFrame:
    read_opts = pv.ReadOptions(use_threads=True)
    parse_opts = pv.ParseOptions(newlines_in_values=True)  # ✅ key fix
    convert_opts = pv.ConvertOptions()

    table = pv.read_csv(path, read_options=read_opts, parse_options=parse_opts, convert_options=convert_opts)
    return table.to_pandas()

def embed_all_threads():
    f"""
    Embeds all the reddit threads 
    from the reddit canada dataset using the {CFG.embedding.model_name}
    & saves them on disk
    """
    if (os.path.exists(CFG.paths.all_threads_emb_mmap)
            and os.path.exists(CFG.paths.all_threads_ids_json)
            and os.path.exists(CFG.paths.all_threads_meta_json)):
        print("Skipped thread embedding as embeddings already exists")
        return

    threads_df = read_csv_arrow_allow_newlines(CFG.paths.threads_csv)

    texts = threads_df['title_and_selftext'].fillna("").astype(str).tolist()
    ids = threads_df['thread_id'].astype(str).tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = CFG.embedding.batch_size_gpu if device == "cuda" else CFG.embedding.batch_size_cpu

    model = SentenceTransformer(CFG.embedding.model_name, device=device)
    model.max_seq_length = CFG.embedding.max_seq_length

    # infer dim
    sample = model.encode(["hello"], convert_to_tensor=True, normalize_embeddings=CFG.embedding.normalize)
    emb_dim = int(sample.shape[-1])

    N = len(texts)

    emb_mm = np.memmap(
        CFG.paths.all_threads_emb_mmap,
        dtype=np.float16,
        mode="w+",
        shape=(N, emb_dim)
    )

    write_idx = 0
    for i in tqdm(range(0, N, batch_size), desc="Embedding all threads"):
        batch_txt = texts[i:i + batch_size]

        with torch.inference_mode():
            emb = model.encode(
                batch_txt,
                convert_to_tensor=True,
                normalize_embeddings=CFG.embedding.normalize,
                show_progress_bar=False
            )

        emb = emb.detach().cpu().numpy().astype(np.float16, copy=False)
        bsz = emb.shape[0]
        emb_mm[write_idx:write_idx + bsz] = emb
        write_idx += bsz

        if (i // batch_size) % 20 == 0:
            emb_mm.flush()

    emb_mm.flush()

    with open(CFG.paths.all_threads_ids_json, "w") as f:
        json.dump(ids, f)

    with open(CFG.paths.all_threads_meta_json, "w") as f:
        json.dump({"n": N, "dim": emb_dim, "dtype": "float16", "model": CFG.embedding.model_name}, f)

    print("Saved:", CFG.paths.all_threads_emb_mmap)
    print("Saved:", CFG.paths.all_threads_ids_json)
    print("Saved:", CFG.paths.all_threads_meta_json)