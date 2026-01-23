import torch
import pandas as pd


thread_df = pd.read_csv("r_canada_dataset/canada_subreddit_threads.csv")
comments_df = pd.read_csv("r_canada_dataset/canada_subreddit_comments.csv")

print(f"Comments: ")
print(len(comments_df))
print("Threads")
print(len(thread_df))