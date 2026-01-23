import glob
import os

import pandas as pd
from bs4 import BeautifulSoup
import re
from html import unescape

from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)
pd.set_option("display.max_columns", None)




def remove_carriage_returns(x):
    return x.replace('\n', ' ').replace('\r\n', ' ')


def unify_multiple_whitespaces(x):
    """replace multiple whitespaces with just one"""
    return re.sub(' {2,}', ' ', str(x))


def clean_html(x):
    """Unescape string then remove html parts"""
    soup = BeautifulSoup(unescape(str(x)), 'lxml')
    return soup.text


def remove_urls(x):
    """remove urls from string"""
    cleaned_string = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', str(x))
    return cleaned_string


def remove_emojis(x):
    """remove known emojis and smileys"""
    emoj = re.compile("["
                      u"\U0001F600-\U0001F64F"  # emoticons
                      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                      u"\U0001F680-\U0001F6FF"  # transport & map symbols
                      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                      u"\U00002500-\U00002BEF"  # chinese char
                      u"\U00002702-\U000027B0"
                      u"\U000024C2-\U0001F251"
                      u"\U0001f926-\U0001f937"
                      u"\U00010000-\U0010ffff"
                      u"\u2640-\u2642"
                      u"\u2600-\u2B55"
                      u"\u200d"
                      u"\u23cf"
                      u"\u23e9"
                      u"\u231a"
                      u"\ufe0f"  # dingbats
                      u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', str(x))


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

def remove_quotes(x: str) -> str:
    return "\n".join([ln for ln in str(x).splitlines() if not ln.strip().startswith(">")])
def process_data(thread_df, comments_df, PROCESSED_R_CANADA_THREADS_CSV, PROCESSED_R_CANADA_COMMENTS_CSV):
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
