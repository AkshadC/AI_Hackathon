from data_processing import process_data
from save_thread_embeddings import embed_all_threads

def main():
    process_data()
    embed_all_threads()

if __name__ == "__main__":
    main()