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



def main():

    #save_threads_sharded(pd.read_csv(PROCESSED_R_CANADA_THREADS_CSV), batch_size=256)
    query = "Information on tariffs in Canada"
    encoded_query = encode_user_query(query)
    print(encoded_query.shape)

if __name__ == "__main__":
    main()