import glob
import os
import re
from html import unescape

import pandas as pd
from bs4 import BeautifulSoup
import swifter

from config import CFG
import pyarrow.csv as pv

def read_csv_arrow_allow_newlines(path: str) -> pd.DataFrame:
    read_opts = pv.ReadOptions(use_threads=True)
    parse_opts = pv.ParseOptions(newlines_in_values=True)  # ✅ key fix
    convert_opts = pv.ConvertOptions()

    table = pv.read_csv(path, read_options=read_opts, parse_options=parse_opts, convert_options=convert_opts)
    return table.to_pandas()

# -----------------------------
# Text cleaning helpers
# -----------------------------
def unify_multiple_whitespaces(x: str) -> str:
    return re.sub(r"\s{2,}", " ", str(x))


def clean_html(x: str) -> str:
    # Fast + safe HTML text extraction
    soup = BeautifulSoup(unescape(str(x)), "lxml")
    return soup.get_text(" ", strip=True)


def remove_urls(x: str) -> str:
    return re.sub(r"(https|http)?:\/\/\S+\b", "", str(x))


_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\u2600-\u2B55"
    "\u200d"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\ufe0f"
    "\u3030"
    "]+",
    flags=re.UNICODE,
)


def remove_emojis(x: str) -> str:
    return re.sub(_EMOJI_RE, "", str(x))


def remove_quotes(x: str) -> str:
    # Remove Reddit blockquotes (lines starting with >)
    lines = []
    for ln in str(x).splitlines():
        if not ln.strip().startswith(">"):
            lines.append(ln)
    return "\n".join(lines)


def normalize_reddit_text(x: str) -> str:
    x = str(x)
    if x.lower() in {"[deleted]", "[removed]", "nan", "none"}:
        return ""
    x = x.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    x = clean_html(x)
    x = remove_urls(x)
    x = remove_emojis(x)
    x = unify_multiple_whitespaces(x).strip()
    return x


# -----------------------------
# Main processing
# -----------------------------
def process_data():
    # Ensure output dir exists
    CFG.paths.data_dir.mkdir(parents=True, exist_ok=True)

    # Load raw CSVs (adjust these if your raw filenames differ)
    # If you already have raw paths in CFG, use them instead.
    threads_df = read_csv_arrow_allow_newlines(CFG.paths.threads_csv)
    comments_df = read_csv_arrow_allow_newlines(CFG.paths.comments_csv)

    if os.path.exists(CFG.paths.threads_csv) and os.path.exists(CFG.paths.comments_csv):
        print("Skipped threads and comments processing as files already exists")
        return

    # Keep best row per thread_id (highest score)
    threads_filtered = (
        threads_df.sort_values(["thread_id", "score"], ascending=[True, False])
        .drop_duplicates("thread_id", keep="first")
        .copy()
    )

    # Detect images (folders: r_canada_dataset/canada_subreddit_images_*)
    image_dirs = glob.glob(str(CFG.paths.data_dir / "canada_subreddit_images_*"))
    image_thread_ids = set()

    for img_dir in image_dirs:
        if not os.path.isdir(img_dir):
            continue
        for fname in os.listdir(img_dir):
            if "!!" in fname:
                thread_id = fname.split("!!", 1)[0]
                image_thread_ids.add(thread_id)

    threads_filtered["has_image"] = threads_filtered["thread_id"].astype(str).isin(image_thread_ids)

    # Flair normalization
    if "link_flair_text" in threads_filtered.columns:
        threads_filtered["link_flair_text"] = (
            threads_filtered["link_flair_text"]
            .fillna("[No Flair]")
            .astype(str)
            .str.strip()
            .str.lower()
        )
    else:
        threads_filtered["link_flair_text"] = "[no flair]"

    # Combine title + selftext
    threads_filtered["title_and_selftext"] = (
        threads_filtered["title"].fillna("").astype(str).str.strip()
        + "\n\n"
        + threads_filtered["selftext"].fillna("").astype(str).str.strip()
    )

    # Clean threads text (swifter is fine)
    threads_filtered["title_and_selftext"] = threads_filtered["title_and_selftext"].swifter.apply(normalize_reddit_text)

    # Clean comments
    comments_df["body"] = comments_df["body"].fillna("").astype(str)
    comments_df["body"] = comments_df["body"].apply(remove_quotes)
    comments_df["body"] = comments_df["body"].swifter.apply(normalize_reddit_text)

    # Filter very short comments
    comments_df["body_len"] = comments_df["body"].str.split().str.len()
    comments_df = comments_df[comments_df["body_len"] >= 3].copy()

    # Save processed CSVs (to the processed paths from CFG)
    threads_filtered.to_csv(CFG.paths.threads_csv, index=False)
    comments_df.to_csv(CFG.paths.comments_csv, index=False)

    print("Saved processed threads:", CFG.paths.threads_csv)
    print("Saved processed comments:", CFG.paths.comments_csv)



