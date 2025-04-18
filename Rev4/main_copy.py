import os
import pandas as pd
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
import json
from colorama import init, Fore, Style
from dotenv import load_dotenv
import sys

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import agent/utility functions (using the direct call approach for robustness)
from agents.data_loader_agent import run_data_loader_agent, load_csv_data
from agents.chunking_agent import run_chunking_agent, perform_chunking
from agents.embedding_agent import run_embedding_agent, generate_chunk_embedding
from agents.vector_store_agent import add_to_vector_store, search_vector_store
from agents.output_agent import format_and_print_results
from agents.column_selection_agent import run_column_selection # Import the new function
# Utilities might be imported directly if agents aren't strictly run
from utils.embedding_utils import compute_embedding
from utils.chunking_utils import create_overlapping_chunks
# Import config
import config

# --- Configuration ---
load_dotenv()
init(autoreset=True) # Initialize colorama

# File path for the CLI application (can be different from API usage)
# Updated path to use the local 'dataset' folder
DATA_FILE_PATH = "dataset/#1-mini.csv"

# Constants are now imported from config
# CHUNK_SIZE = 50
# OVERLAP = 10
# TOP_K = 3
# CHROMA_DB_PATH = "./chroma_db"
# COLLECTION_NAME = "timeseries_chunks"

# --- F1 Score Calculation Function ---
def calculate_f1_score(predicted_chunks, ground_truth_chunks, threshold=0.8):
    """
    Calculate Precision, Recall and F1 score for time series similarity results.
    
    Args:
        predicted_chunks: List of time series chunks returned by the model
        ground_truth_chunks: List of ground truth chunks that should have been returned
        threshold: Similarity threshold to consider a match (0.0 to 1.0)
        
    Returns:
        Dictionary with precision, recall, and f1 metrics
    """
    if not predicted_chunks or not ground_truth_chunks:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matches": 0}
    
    # Count true positives (matches between predicted and ground truth)
    true_positives = 0
    
    # For each predicted chunk, check if it's similar enough to any ground truth chunk
    for pred_chunk in predicted_chunks:
        # Convert to numpy array for calculations
        pred_array = np.array(pred_chunk)
        
        for gt_chunk in ground_truth_chunks:
            gt_array = np.array(gt_chunk)
            
            # Make sure arrays are the same length for comparison
            min_len = min(len(pred_array), len(gt_array))
            pred_array_trimmed = pred_array[:min_len]
            gt_array_trimmed = gt_array[:min_len]
            
            # Calculate cosine similarity
            # Normalize arrays
            norm_pred = np.linalg.norm(pred_array_trimmed)
            norm_gt = np.linalg.norm(gt_array_trimmed)
            
            if norm_pred > 0 and norm_gt > 0:  # Avoid division by zero
                similarity = np.dot(pred_array_trimmed, gt_array_trimmed) / (norm_pred * norm_gt)
                
                # If similarity exceeds threshold, count as a match
                if similarity >= threshold:
                    true_positives += 1
                    break  # Once a match is found, move to next prediction
    
    # Calculate metrics
    precision = true_positives / len(predicted_chunks) if predicted_chunks else 0
    recall = true_positives / len(ground_truth_chunks) if ground_truth_chunks else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": true_positives
    }

# --- Helper Functions ---
def get_user_input():
    """Prompts the user for the time series array, annotation label, and optional ground truth."""
    print(f"\n{Fore.YELLOW}--- User Input ---{Style.RESET_ALL}")
    while True:
        try:
            ts_input_str = input(f"{Fore.YELLOW}Enter time series array : {Style.RESET_ALL}")
            # Basic parsing - convert string list to Python list of floats
            # Using json.loads is safer than eval()
            ts_list = json.loads(ts_input_str)
            if not isinstance(ts_list, list) or not all(isinstance(x, (int, float)) for x in ts_list):
                raise ValueError("Input must be a list of numbers.")
            # Convert to numpy array for embedding consistency
            ts_array = np.array(ts_list, dtype=float)
            if ts_array.size == 0:
                 raise ValueError("Input array cannot be empty.")
            break
        except json.JSONDecodeError:
            print(f"{Fore.RED}Invalid input format. Please enter a valid JSON list (e.g., [1.0, 2.5, 3.0]).{Style.RESET_ALL}")
        except ValueError as e:
            print(f"{Fore.RED}Invalid input: {e}. Please enter a non-empty list of numbers.{Style.RESET_ALL}")
        except Exception as e:
             print(f"{Fore.RED}An unexpected error occurred during input: {e}{Style.RESET_ALL}")

    while True:
        annotation_label = input(f"{Fore.YELLOW}Enter annotation label (e.g., 'peak-pattern'): {Style.RESET_ALL}")
        if annotation_label.strip():
            break
        else:
            print(f"{Fore.RED}Annotation label cannot be empty.{Style.RESET_ALL}")
    
    # Add option to input ground truth for F1 score calculation
    print(f"{Fore.CYAN}F1 Score Calculation: To evaluate model performance, you can provide known correct patterns.{Style.RESET_ALL}")
    use_ground_truth = input(f"{Fore.YELLOW}Would you like to provide ground truth for F1 score calculation? (y/n): {Style.RESET_ALL}").lower() == 'y'
    ground_truth = []
    
    if use_ground_truth:
        print(f"{Fore.CYAN}Enter ground truth as a JSON array of arrays. Example: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]{Style.RESET_ALL}")
        print(f"{Fore.CYAN}This represents multiple time series patterns that should be found by the model.{Style.RESET_ALL}")
        while True:
            try:
                gt_input_str = input(f"{Fore.YELLOW}Enter ground truth time series array(s): {Style.RESET_ALL}")
                # Parse as list of lists (multiple ground truth chunks)
                gt_list = json.loads(gt_input_str)
                
                # Validate input format (should be list of lists of numbers)
                if not isinstance(gt_list, list):
                    raise ValueError("Ground truth must be a list of arrays.")
                
                # If user enters a single array instead of list of arrays, wrap it
                if gt_list and all(isinstance(x, (int, float)) for x in gt_list):
                    gt_list = [gt_list]
                    print(f"{Fore.CYAN}Interpreted as a single ground truth pattern.{Style.RESET_ALL}")
                
                # Validate each array
                for arr in gt_list:
                    if not isinstance(arr, list) or not all(isinstance(x, (int, float)) for x in arr):
                        raise ValueError("Each ground truth array must be a list of numbers.")
                
                ground_truth = gt_list
                print(f"{Fore.GREEN}Successfully loaded {len(ground_truth)} ground truth pattern(s).{Style.RESET_ALL}")
                break
            except json.JSONDecodeError:
                print(f"{Fore.RED}Invalid input format. Please enter valid JSON (e.g., [[1.0, 2.0], [3.0, 4.0]]).{Style.RESET_ALL}")
            except ValueError as e:
                print(f"{Fore.RED}Invalid input: {e}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}An unexpected error occurred during input: {e}{Style.RESET_ALL}")
    else:
        print(f"{Fore.CYAN}Skipping F1 score calculation (no ground truth provided).{Style.RESET_ALL}")
    
    print(f"{Fore.YELLOW}------------------{Style.RESET_ALL}")
    return ts_array, annotation_label.strip(), ground_truth

# --- Main Application Logic ---
def main():
    print(f"{Fore.BLUE}--- Time Series Similarity Agent Initializing ---{Style.RESET_ALL}")

    # 1. Initialize ChromaDB Client and Collection using config values
    print(f"{Fore.CYAN}Initializing ChromaDB client at: {config.CHROMA_DB_PATH}{Style.RESET_ALL}")
    try:
        client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        # Using default embedding function for the collection setup, though we provide embeddings manually
        # ef = embedding_functions.DefaultEmbeddingFunction() # Not strictly needed if providing embeddings
        collection = client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            # embedding_function=ef # Not needed when adding embeddings directly
            metadata={"hnsw:space": "cosine"} # Specify cosine distance
        )
        print(f"{Fore.GREEN}ChromaDB collection '{config.COLLECTION_NAME}' ready.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Fatal Error: Could not initialize ChromaDB: {e}{Style.RESET_ALL}")
        return # Exit if DB fails

    # --- Check if data needs processing ---
    # Simple check: if collection is empty, process data. Otherwise, assume processed.
    # For robustness, could add a flag file or check metadata.
    should_process_data = collection.count() == 0
    if should_process_data:
        print(f"{Fore.YELLOW}Vector store appears empty. Processing data file...{Style.RESET_ALL}")

        # 2. Load Data
        print(f"\n{Fore.BLUE}--- Step 2: Data Loading ---{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}Invoking DataLoaderAgent (via direct function call)...{Style.RESET_ALL}")
        # df = run_data_loader_agent(DATA_FILE_PATH) # Agent invocation example
        df = load_csv_data(DATA_FILE_PATH) # Direct call
        if df.empty:
            print(f"{Fore.RED}Fatal Error: Failed to load data. Exiting.{Style.RESET_ALL}")
            return
        print(f"{Fore.BLUE}--- Data Loading Complete ---{Style.RESET_ALL}")

        # Dynamically identify columns to process
        print(f"\n{Fore.BLUE}--- Step 2b: Column Selection ---{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}Invoking ColumnSelectionAgent (via direct function call)...{Style.RESET_ALL}")
        columns_to_process_dynamic = run_column_selection(df)

        if not columns_to_process_dynamic:
            print(f"{Fore.RED}Fatal Error: No suitable columns identified for processing. Exiting.{Style.RESET_ALL}")
            return
        print(f"{Fore.BLUE}--- Column Selection Complete ---{Style.RESET_ALL}")


        # 3. Process Each Identified Column: Chunk -> Embed -> Store
        print(f"\n{Fore.BLUE}--- Step 3: Data Processing (Chunk -> Embed -> Store) ---{Style.RESET_ALL}")
        total_embeddings_added = 0
        # Use the dynamic list here:
        for col_name in columns_to_process_dynamic:
            print(f"\n{Fore.MAGENTA}--> Processing column: {col_name}{Style.RESET_ALL}")
            # Ensure column exists (should be guaranteed by selection, but good practice)
            if col_name not in df.columns:
                 print(f"{Fore.YELLOW}Warning: Column '{col_name}' unexpectedly missing after initial selection. Skipping.{Style.RESET_ALL}")
                 continue

            column_data = df[col_name].dropna().to_numpy() # Get data as numpy array, drop NaNs

            if column_data.size < config.CHUNK_SIZE:
                print(f"{Fore.YELLOW}Skipping column '{col_name}': Not enough data points ({column_data.size}) for chunk size ({config.CHUNK_SIZE}).{Style.RESET_ALL}")
                continue

            # 3a. Chunking using config values
            print(f"{Fore.MAGENTA}  Invoking ChunkingAgent for '{col_name}' (via direct function call)...{Style.RESET_ALL}")
            # chunks = run_chunking_agent(df[col_name].dropna(), config.CHUNK_SIZE, config.OVERLAP) # Agent example
            chunks = list(create_overlapping_chunks(column_data, config.CHUNK_SIZE, config.OVERLAP)) # Direct call

            if not chunks:
                print(f"{Fore.YELLOW}  No chunks generated for column '{col_name}'.{Style.RESET_ALL}")
                continue

            # 3b. Embedding & Storing
            print(f"{Fore.MAGENTA}  Invoking EmbeddingAgent & VectorStoreAgent for {len(chunks)} chunks in '{col_name}' (via direct calls)...{Style.RESET_ALL}")
            # print(f"{Fore.CYAN}Generating and storing embeddings for {len(chunks)} chunks in '{col_name}'...{Style.RESET_ALL}") # Old message
            for i, chunk in enumerate(chunks):
                # Embedding
                embedding = compute_embedding(chunk) # Direct call

                if embedding:
                    # Prepare metadata and ID
                    doc_id = f"{col_name}_chunk_{i}"
                    metadata = {
                        "column": col_name,
                        "chunk_index": i,
                        "start_row": i * (config.CHUNK_SIZE - config.OVERLAP), # Approximate original start row
                        "end_row": i * (config.CHUNK_SIZE - config.OVERLAP) + config.CHUNK_SIZE # Approximate original end row
                    }
                    # Add to Vector Store
                    success = add_to_vector_store(collection, embedding, chunk, metadata, doc_id)
                    if success:
                        total_embeddings_added += 1
                else:
                    print(f"{Fore.YELLOW}  Skipping storage for chunk {i} in column '{col_name}' due to embedding failure.{Style.RESET_ALL}")

        print(f"\n{Fore.BLUE}--- Step 3 Complete ---{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Total embeddings added/processed: {total_embeddings_added}{Style.RESET_ALL}")
        if total_embeddings_added == 0:
             print(f"{Fore.RED}Warning: No embeddings were successfully added. Similarity search may yield no results.{Style.RESET_ALL}")

    else:
        print(f"{Fore.GREEN}Vector store already contains data ({collection.count()} items). Skipping data processing.{Style.RESET_ALL}")


    # 4. User Interaction Loop
    print(f"\n{Fore.BLUE}--- Step 4: User Interaction ---{Style.RESET_ALL}")
    while True:
        query_array, annotation_label, ground_truth = get_user_input()

        # 5. Embed User Query
        print(f"\n{Fore.BLUE}--- Step 5: Query Embedding ---{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}Invoking EmbeddingAgent for user query (via direct function call)...{Style.RESET_ALL}")
        # print(f"{Fore.CYAN}Embedding user query...{Style.RESET_ALL}") # Old message
        # Pad the query array if it's shorter than config.CHUNK_SIZE
        padded_query_array = query_array
        if len(query_array) < config.CHUNK_SIZE:
            print(f"{Fore.YELLOW}Padding query array (length {len(query_array)}) to match chunk size ({config.CHUNK_SIZE})...{Style.RESET_ALL}")
            # Pad with zeros at the end
            padding_size = config.CHUNK_SIZE - len(query_array)
            # Use 'constant' mode with default constant_values=0
            padded_query_array = np.pad(query_array, (0, padding_size), mode='constant')
            print(f"{Fore.YELLOW}Padded query array length: {len(padded_query_array)}{Style.RESET_ALL}")
        elif len(query_array) > config.CHUNK_SIZE:
             print(f"{Fore.YELLOW}Warning: User query length ({len(query_array)}) is greater than chunk size ({config.CHUNK_SIZE}). Truncating query for embedding.{Style.RESET_ALL}")
             padded_query_array = query_array[:config.CHUNK_SIZE] # Truncate if longer

        # query_embedding = run_embedding_agent(padded_query_array) # Agent example
        query_embedding = compute_embedding(padded_query_array) # Direct call using padded/truncated array

        if not query_embedding:
            print(f"{Fore.RED}Failed to generate embedding for the user query. Please try again.{Style.RESET_ALL}")
            continue # Ask for input again
        print(f"{Fore.BLUE}--- Query Embedding Complete ---{Style.RESET_ALL}")

        # 6. Similarity Search using config value for TOP_K
        print(f"\n{Fore.BLUE}--- Step 6: Similarity Search ---{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}Invoking VectorStoreAgent for search (via direct function call)...{Style.RESET_ALL}")
        similar_chunks_data = search_vector_store(collection, query_embedding, config.TOP_K)
        print(f"{Fore.BLUE}--- Similarity Search Complete ---{Style.RESET_ALL}")

        # 7. Format and Print Output using config value for TOP_K
        print(f"\n{Fore.BLUE}--- Step 7: Output Formatting ---{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}Invoking OutputAgent (via direct function call)...{Style.RESET_ALL}")
        format_and_print_results(annotation_label, similar_chunks_data, config.TOP_K)
        print(f"{Fore.BLUE}--- Output Formatting Complete ---{Style.RESET_ALL}")
        
        # 8. Calculate and display F1 score if ground truth was provided
        if ground_truth:
            print(f"\n{Fore.BLUE}--- Step 8: Performance Evaluation ---{Style.RESET_ALL}")
            metrics = calculate_f1_score(similar_chunks_data, ground_truth)
            print(f"{Fore.MAGENTA}--- F1 Score Metrics ---{Style.RESET_ALL}")
            print(f"{Fore.WHITE}Precision: {metrics['precision']:.4f}{Style.RESET_ALL}")
            print(f"{Fore.WHITE}Recall: {metrics['recall']:.4f}{Style.RESET_ALL}")
            print(f"{Fore.WHITE}F1 Score: {metrics['f1']:.4f}{Style.RESET_ALL}")
            print(f"{Fore.WHITE}Matches found: {metrics['matches']} out of {len(ground_truth)} ground truth items{Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}------------------------{Style.RESET_ALL}")
            print(f"{Fore.BLUE}--- Performance Evaluation Complete ---{Style.RESET_ALL}")

        # Ask user if they want to perform another search
        another = input(f"\n{Fore.YELLOW}Perform another search? (y/n): {Style.RESET_ALL}").lower()
        if another != 'y':
            break

    print(f"\n{Fore.BLUE}--- Time Series Similarity Agent Exiting ---{Style.RESET_ALL}")

if __name__ == "__main__":
    main()