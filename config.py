from dataclasses import dataclass
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent

@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / 'r_canada_dataset'
    web_dir: Path = PROJECT_ROOT / 'maple-lens-web'
    emb_dir: Path = PROJECT_ROOT / 'embeddings_out'

    # files
    threads_csv: Path = data_dir / "canada_subreddit_threads_processed.csv"
    comments_csv: Path = data_dir / "canada_subreddit_comments_processed.csv"

    # global embeddings (all threads)
    all_threads_emb_mmap: Path = emb_dir / "all_threads_emb.f16.mmap"
    all_threads_ids_json: Path = emb_dir / "all_threads_ids.json"
    all_threads_meta_json: Path = emb_dir / "all_threads_meta.json"

    # trending embeddings (windowed)
    trending_prefix: str = "latest_topics"
    trending_emb_pt: Path = emb_dir / f"{trending_prefix}_emb.pt"
    trending_ids_json: Path = emb_dir / f"{trending_prefix}_ids.json"
    trending_umap_npy: Path = emb_dir / f"{trending_prefix}_umap10.npy"
    trending_labels_npy: Path = emb_dir / f"{trending_prefix}_labels.npy"

    # homepage payload for webapp
    homepage_json: Path = emb_dir / "homepage_topics.json"

@dataclass(frozen=True)
class Embedding:
    # lightweight, fast, good enough for clustering/search
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_seq_length: int = 256
    normalize: bool = True
    batch_size_gpu: int = 256
    batch_size_cpu: int = 64
    save_dtype: str = "float16"


@dataclass(frozen=True)
class Trending:
    days: int = 14
    k_comments_per_thread: int = 5
    top_topics: int = 3
    top_threads_per_topic: int = 5

    # UMAP
    umap_neighbors: int = 15
    umap_components: int = 10

    # HDBSCAN (base; you can also auto-scale)
    hdb_min_cluster_size: int = 12
    hdb_min_samples: int = 6


@dataclass(frozen=True)
class Query:
    top_k_threads: int = 5
    top_k_comments_per_thread: int = 5
    cosine_chunk_size: int = 4096
    dedup_threshold: float = 0.92


@dataclass(frozen=True)
class Gemini:
    api_key_env: str = "GEMINI_API_KEY"
    # choose one:
    model_name: str = "gemini-3-flash"  # or "gemma-3-2b"
    max_retries: int = 6
    max_chars_per_topic_context: int = 2500
    # set MOCK_GEMINI=1 to skip API calls during local testing
    mock: bool = os.getenv("MOCK_GEMINI", "0") == "1"


@dataclass(frozen=True)
class Config:
    paths: Paths = Paths()
    embedding: Embedding = Embedding()
    trending: Trending = Trending()
    query: Query = Query()
    gemini: Gemini = Gemini()


CFG = Config()


# Create output dirs automatically when imported
CFG.paths.emb_dir.mkdir(parents=True, exist_ok=True)