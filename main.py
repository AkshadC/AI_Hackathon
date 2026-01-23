import glob

import torch
import pandas as pd

from data_processing import *
from pandarallel import pandarallel

from gemma_embedder.gemma_embedder import *

pandarallel.initialize(progress_bar=True, nb_workers=6)

R_CANADA_THREADS_CSV = "r_canada_dataset/canada_subreddit_threads.csv"
R_CANADA_COMMENTS_CSV = "r_canada_dataset/canada_subreddit_comments.csv"
PROCESSED_R_CANADA_THREADS_CSV = "r_canada_dataset/canada_subreddit_threads_processed.csv"
PROCESSED_R_CANADA_COMMENTS_CSV= "r_canada_dataset/canada_subreddit_comments_processed.csv"

def process_data(thread_df, comments_df):
    threads_filtered = thread_df[
        (thread_df["num_comments"].between(10, 100)) &
        (thread_df["score"] > 0) & (thread_df['upvote_ratio'] > 0.25)
        ].copy()
    threads_filtered = threads_filtered.sort_values(["thread_id","score"], ascending=[True, False]).drop_duplicates("thread_id", keep="first")
    image_dirs = glob.glob("r_canada_dataset/canada_subreddit_images_*")

    # --- 3) Collect thread_ids that have images ---
    image_thread_ids = set()

    for img_dir in image_dirs:
        if not os.path.isdir(img_dir):
            continue
        for fname in os.listdir(img_dir):
            # filename format: <thread_id>!!something.ext
            if "!!" in fname:
                thread_id = fname.split("!!", 1)[0]
                image_thread_ids.add(thread_id)

    # --- 4) Add has_image column ---
    threads_filtered["has_image"] = (
        threads_filtered["thread_id"].astype(str).isin(image_thread_ids)
    )
    threads_filtered["link_flair_text"] = threads_filtered["link_flair_text"].fillna("[No Flair]")
    threads_filtered["link_flair_text"] = threads_filtered["link_flair_text"].str.strip()
    threads_filtered["link_flair_text"] = threads_filtered["link_flair_text"].str.lower()
    threads_filtered["title_and_selftext"] = (
            threads_filtered["title"].fillna("").astype(str).str.strip()
            + "\n\n"
            + threads_filtered["selftext"].fillna("").astype(str).str.strip()
    )
    threads_filtered["title_and_selftext"] = threads_filtered["title_and_selftext"].parallel_apply(normalize_reddit_text)


    #comments_df['body'] = comments_df['body'].apply(remove_quotes).parallel_apply(normalize_reddit_text)
    #comments_df["body_len"] = comments_df["body"].str.split().str.len()
    #comments_df = comments_df[comments_df["body_len"] >= 3]
    threads_filtered.to_csv(PROCESSED_R_CANADA_THREADS_CSV, index=False)
    #comments_df.to_csv(PROCESSED_R_CANADA_COMMENTS_CSV)


def main():
    thread_df = pd.read_csv(R_CANADA_THREADS_CSV)
    comments_df = pd.read_csv(R_CANADA_COMMENTS_CSV)

    process_data(thread_df, comments_df)
    #save_threads_sharded(pd.read_csv(PROCESSED_R_CANADA_THREADS_CSV), batch_size=256)


if __name__ == "__main__":
    main()