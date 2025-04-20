import os
import sys
import time
import json

import pandas as pd
import numpy as np
import chromadb
from chromadb.utils import embedding_functions

from colorama import init, Fore, Style
from dotenv import load_dotenv

import tiktoken
from langchain.embeddings import OpenAIEmbeddings

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import agent/utility functions (direct calls)
from agents.data_loader_agent import load_csv_data
from agents.column_selection_agent import run_column_selection
from agents.vector_store_agent import add_to_vector_store, search_vector_store
from agents.output_agent import format_and_print_results

from utils.embedding_utils import compute_embedding
from utils.chunking_utils import create_overlapping_chunks

# Import config
import config

# --- Configuration ---
load_dotenv()
init(autoreset=True)  # Initialize colorama

DATA_FILE_PATH = "dataset/#1-mini.csv"

def get_user_input():
    """Prompts the user for the time series array and annotation label."""
    print(f"\n{Fore.YELLOW}--- User Input ---{Style.RESET_ALL}")
    while True:
        try:
            ts_input_str = input(f"{Fore.YELLOW}Enter time series array: {Style.RESET_ALL}")
            ts_list = json.loads(ts_input_str)
            if not isinstance(ts_list, list) or not all(isinstance(x, (int, float)) for x in ts_list):
                raise ValueError("Input must be a list of numbers.")
            ts_array = np.array(ts_list, dtype=float)
            if ts_array.size == 0:
                raise ValueError("Input array cannot be empty.")
            break
        except json.JSONDecodeError:
            print(f"{Fore.RED}Invalid format. Enter a JSON list (e.g., [1.0,2.5,3.0]).{Style.RESET_ALL}")
        except ValueError as e:
            print(f"{Fore.RED}Invalid input: {e}.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Unexpected error: {e}{Style.RESET_ALL}")

    while True:
        annotation_label = input(f"{Fore.YELLOW}Enter annotation label (e.g., 'peak-pattern'): {Style.RESET_ALL}")
        if annotation_label.strip():
            break
        else:
            print(f"{Fore.RED}Annotation label cannot be empty.{Style.RESET_ALL}")

    print(f"{Fore.YELLOW}------------------{Style.RESET_ALL}")
    return ts_array, annotation_label.strip()

def main():
    print(f"{Fore.BLUE}--- Time Series Similarity Agent Initializing ---{Style.RESET_ALL}")

    # 1. Initialize ChromaDB
    print(f"{Fore.CYAN}Initializing ChromaDB client at: {config.CHROMA_DB_PATH}{Style.RESET_ALL}")
    try:
        client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        collection = client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"{Fore.GREEN}ChromaDB collection '{config.COLLECTION_NAME}' ready.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Fatal: Could not initialize ChromaDB: {e}{Style.RESET_ALL}")
        return

    # 2. Process data if store is empty
    if collection.count() == 0:
        print(f"{Fore.YELLOW}Vector store empty. Processing data...{Style.RESET_ALL}")

        print(f"\n{Fore.BLUE}--- Step 2: Data Loading ---{Style.RESET_ALL}")
        df = load_csv_data(DATA_FILE_PATH)
        if df.empty:
            print(f"{Fore.RED}Fatal: Failed to load data.{Style.RESET_ALL}")
            return

        print(f"\n{Fore.BLUE}--- Step 2b: Column Selection ---{Style.RESET_ALL}")
        columns = run_column_selection(df)
        if not columns:
            print(f"{Fore.RED}Fatal: No columns selected.{Style.RESET_ALL}")
            return

        print(f"\n{Fore.BLUE}--- Step 3: Chunk → Embed → Store ---{Style.RESET_ALL}")
        total = 0
        for col in columns:
            if col not in df.columns:
                print(f"{Fore.YELLOW}Skipping missing column: {col}{Style.RESET_ALL}")
                continue
            arr = df[col].dropna().to_numpy()
            if arr.size < config.CHUNK_SIZE:
                print(f"{Fore.YELLOW}Not enough data in '{col}'.{Style.RESET_ALL}")
                continue
            chunks = list(create_overlapping_chunks(arr, config.CHUNK_SIZE, config.OVERLAP))
            for i, chunk in enumerate(chunks):
                emb = compute_embedding(chunk)
                if not emb:
                    continue
                doc_id = f"{col}_chunk_{i}"
                meta = {
                    "column": col,
                    "chunk_index": i,
                    "start_row": i * (config.CHUNK_SIZE - config.OVERLAP),
                    "end_row": i * (config.CHUNK_SIZE - config.OVERLAP) + config.CHUNK_SIZE
                }
                if add_to_vector_store(collection, emb, chunk.tolist(), meta, doc_id):
                    total += 1
        print(f"{Fore.GREEN}Added {total} embeddings.{Style.RESET_ALL}")
    else:
        print(f"{Fore.GREEN}Store has {collection.count()} items. Skipping load.{Style.RESET_ALL}")

    # Prepare frameworks for comparison
    lc_ef   = OpenAIEmbeddings(model="text-embedding-ada-002")
    encoder = tiktoken.encoding_for_model("text-embedding-ada-002")

    # 4. User Loop with comparison
    print(f"\n{Fore.BLUE}--- Step 4: User Interaction ---{Style.RESET_ALL}")
    while True:
        query_array, annotation_label = get_user_input()

        # Pad/truncate
        padded = query_array
        if len(query_array) < config.CHUNK_SIZE:
            padded = np.pad(query_array, (0, config.CHUNK_SIZE - len(query_array)), mode='constant')
        elif len(query_array) > config.CHUNK_SIZE:
            padded = query_array[:config.CHUNK_SIZE]

        # Token count
        query_text  = json.dumps(padded.tolist())
        token_count = len(encoder.encode(query_text))
        print(f"{Fore.CYAN}Tokens in query: {token_count}{Style.RESET_ALL}")

        # Framework A: LangChain/OpenAI
        t0      = time.time()
        emb_lc  = lc_ef.embed_documents([query_text])[0]
        t_lc    = time.time() - t0
        print(f"{Fore.BLUE}LangChain embed time: {t_lc:.3f}s, dim={len(emb_lc)}{Style.RESET_ALL}")

        # Framework B: Custom FFT/time-domain
        t1           = time.time()
        emb_custom   = compute_embedding(padded)
        t_custom     = time.time() - t1
        print(f"{Fore.GREEN}Custom embed time: {t_custom:.3f}s, dim={len(emb_custom)}{Style.RESET_ALL}")

        if not emb_lc or not emb_custom:
            print(f"{Fore.RED}Embedding failed for one framework. Retry.{Style.RESET_ALL}")
            continue

        # Similarity search LangChain
        t2        = time.time()
        res_lc    = search_vector_store(collection, emb_lc, config.TOP_K)
        t_search_lc = time.time() - t2
        print(f"{Fore.BLUE}LangChain search time: {t_search_lc:.3f}s{Style.RESET_ALL}")

        # Similarity search Custom
        t3           = time.time()
        res_custom   = search_vector_store(collection, emb_custom, config.TOP_K)
        t_search_custom = time.time() - t3
        print(f"{Fore.GREEN}Custom search time: {t_search_custom:.3f}s{Style.RESET_ALL}")

        # Output results
        print(f"\n{Fore.MAGENTA}--- Results: LangChain ---{Style.RESET_ALL}")
        format_and_print_results(annotation_label, res_lc, config.TOP_K)
        print(f"\n{Fore.MAGENTA}--- Results: Custom ---{Style.RESET_ALL}")
        format_and_print_results(annotation_label, res_custom, config.TOP_K)

        if input(f"\n{Fore.YELLOW}Perform another search? (y/n): {Style.RESET_ALL}").lower() != 'y':
            break

    print(f"\n{Fore.BLUE}--- Exiting ---{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
