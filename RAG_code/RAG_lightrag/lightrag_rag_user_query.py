#!/usr/bin/env python3

import os
import asyncio
import sys
import argparse # Added for CLI argument parsing

try:
    from termcolor import colored
except ImportError:
    print("Warning: termcolor not installed. Install using: pip install termcolor", file=sys.stderr)
    # Define a fallback colored function
    def colored(text, color=None, on_color=None, attrs=None):
        return text

try:
    from pypdf import PdfReader
except ImportError:
    print(colored("ERROR: pypdf not installed. Please install using: pip install pypdf", 'red'), file=sys.stderr)
    sys.exit(1)

# Ensure lightrag and openai are installed and configured
try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed # Keep model names as is (Instruction #4)
    from lightrag.kg.shared_storage import initialize_pipeline_status
    from lightrag.utils import setup_logger
except ImportError as e:
     print(colored(f"ERROR: Required LightRAG components not installed or found: {e}. Please install lightrag and openai.", 'red'), file=sys.stderr)
     sys.exit(1)

# --- Major Variables (Instruction #3) ---
WORKING_DIR = "lightrag_data_cli" # Use a potentially different dir for CLI version if needed
LOGGER_NAME = "lightrag_cli"
LOGGER_LEVEL = "INFO"
DEFAULT_MODE = "mix" # Default query mode
PDF_FILE_PATH = "/mnt/nas/gpt/RAG/rag_all/31496.pdf" # Path to the PDF document

# Setup logger
setup_logger(LOGGER_NAME, level=LOGGER_LEVEL)

# --- Helper Functions (Copied & adapted from lightrag_rag.py) ---

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text content from a PDF file."""
    print(colored(f"Reading PDF: {pdf_path}", 'cyan')) # Termcolor (Instruction #1)
    if not os.path.exists(pdf_path):
        print(colored(f"ERROR: PDF file not found at {pdf_path}", 'red'), file=sys.stderr) # Error Handling (Instruction #6)
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        print(colored("PDF read successfully.", 'green'))
        if not text:
             print(colored("Warning: No text extracted from PDF.", 'yellow'), file=sys.stderr)
        return text
    except Exception as e:
        print(colored(f"ERROR reading PDF {pdf_path}: {e}", 'red'), file=sys.stderr) # Error Handling (Instruction #6)
        raise

async def initialize_rag():
    """Initializes the LightRAG instance and its dependencies."""
    api_key = os.getenv("OPENAI_API_KEY") # API Keys (Instruction #8)
    if not api_key:
        print(colored("ERROR: OPENAI_API_KEY environment variable not set.", 'red'), file=sys.stderr) # Error Handling (Instruction #6)
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    api_base = os.getenv("OPENAI_API_BASE") # API Keys (Instruction #8)
    if not api_base:
        default_base_url = "https://api.openai.com/v1"
        print(colored(f"WARN: OPENAI_API_BASE not found. Setting default for LightRAG: {default_base_url}", 'yellow'), file=sys.stderr)
        os.environ['OPENAI_API_BASE'] = default_base_url
        api_base = default_base_url

    print(colored("Initializing LightRAG...", 'cyan')) # Termcolor (Instruction #1)
    try:
        rag = LightRAG(
            working_dir=WORKING_DIR,
            embedding_func=openai_embed,
            llm_model_func=gpt_4o_mini_complete # Keep model name (Instruction #4)
        )
        print(colored("Initializing storages...", 'cyan'))
        await rag.initialize_storages()
        print(colored("Initializing pipeline status...", 'cyan'))
        await initialize_pipeline_status()
        print(colored("LightRAG initialized successfully.", 'green'))
        return rag
    except Exception as e:
        print(colored(f"ERROR during LightRAG initialization: {e}", 'red'), file=sys.stderr) # Error Handling (Instruction #6)
        if isinstance(e, KeyError) and 'OPENAI_API_BASE' in str(e):
            print(colored("Hint: Ensure OPENAI_API_BASE env var is set correctly for custom endpoints or unset for default OpenAI.", 'yellow'), file=sys.stderr)
        raise

async def main():
    """Main asynchronous function to run the RAG process with interactive query."""
    # Initialize RAG instance (Separation of Concerns - Instruction #7)
    rag = await initialize_rag()

    # Extract text from PDF (Separation of Concerns - Instruction #7)
    # Consider optimizing this: check if data exists in WORKING_DIR before re-extracting/re-inserting
    print(colored(f"Checking for existing data in {WORKING_DIR}...", 'cyan'))
    # A simple check (can be made more robust)
    if not os.path.exists(os.path.join(WORKING_DIR, "storage", "doc_store")):
         print(colored(f"No existing data found. Processing PDF: {PDF_FILE_PATH}", 'cyan'))
         document_text = extract_text_from_pdf(PDF_FILE_PATH)

         if not document_text:
             print(colored("Skipping insertion and query as no text was extracted from PDF.", 'yellow'), file=sys.stderr)
             return

         # Insert text
         text_snippet = document_text[:100].replace('\n', ' ')
         print(colored(f"Inserting text from PDF (first 100 chars): '{text_snippet}...'", 'cyan'))
         try:
             await rag.ainsert(document_text) # Async Task (Instruction #5 inferred via lightrag)
             print(colored("Text inserted successfully.", 'green'))
         except Exception as e:
             print(colored(f"ERROR during text insertion: {e}", 'red'), file=sys.stderr) # Error Handling (Instruction #6)
             # Decide if execution should stop or continue based on the error
             return # Stop if insertion fails
    else:
         print(colored(f"Existing data found in {WORKING_DIR}. Skipping PDF processing and insertion.", 'green'))

    # --- Start Interactive Query Loop ---
    while True:
        print(colored("\nEnter your query (or type 'exit'/'stop' to quit):", 'yellow'))
        user_query = input("> ")
        if not user_query:
            print(colored("No query entered. Please try again.", 'yellow'))
            continue # Ask for input again

        if user_query.lower() in ['exit', 'stop']:
            print(colored("Exiting interactive query loop.", 'blue'))
            break # Exit the loop

        allowed_modes = ['naive', 'local', 'global', 'hybrid', 'mix']
        print(colored(f"Enter query mode {allowed_modes} (default: {DEFAULT_MODE}):", 'yellow'))
        query_mode_input = input(f"> ")
        if query_mode_input.strip().lower() in allowed_modes:
            query_mode = query_mode_input.strip().lower()
        else:
            if query_mode_input.strip() == "": # User just pressed Enter
                print(colored(f"Using default mode: {DEFAULT_MODE}", 'cyan'))
                query_mode = DEFAULT_MODE
            else:
                print(colored(f"Invalid mode entered. Using default: {DEFAULT_MODE}", 'yellow'))
                query_mode = DEFAULT_MODE

        # Perform query using user input (Separation of Concerns - Instruction #7)
        print(colored(f"\nPerforming query with mode='{query_mode}'...", 'cyan')) # Termcolor (Instruction #1)
        print(colored(f"Query: '{user_query}'", 'cyan'))
        try:
            result = await rag.aquery(
                user_query,
                param=QueryParam(mode=query_mode) # Use the specified mode
            )
            print(colored("Query completed successfully.", 'green'))
            # Nicer result printing
            print(colored("\n--- Query Result ---", 'magenta'))
            print(result) # TODO: Potentially format result for better readability
            print(colored("--- End Result ---", 'magenta'))

        except Exception as e:
            print(colored(f"ERROR during query: {e}", 'red'), file=sys.stderr) # Error Handling (Instruction #6)
            # Add specific error hints if known (like the OPENAI_API_BASE one)
            if isinstance(e, KeyError) and 'OPENAI_API_BASE' in str(e):
                 print(colored("Hint: Check OPENAI_API_BASE environment variable setting.", 'yellow'), file=sys.stderr)
    # --- End Interactive Query Loop ---

if __name__ == "__main__":
    # Removed argparse setup

    # Update PDF_FILE_PATH if needed (e.g., from env var or config), but keep it fixed for now
    # PDF_FILE_PATH remains as defined globally or could be prompted if needed

    # Async Task Runner & Error Handling (Instructions #5 & #6)
    try:
        asyncio.run(main()) # Call main without args
    except (ValueError, FileNotFoundError) as config_err: # Catch specific configuration/setup errors
         print(colored(f"Setup error: {config_err}", 'red'), file=sys.stderr)
         sys.exit(1)
    except Exception as e:
         print(colored(f"An unexpected error occurred in the main execution: {e}", 'red', attrs=['bold']), file=sys.stderr)
         # Optional: Log full traceback for debugging
         # import traceback
         # traceback.print_exc()
         sys.exit(1) 