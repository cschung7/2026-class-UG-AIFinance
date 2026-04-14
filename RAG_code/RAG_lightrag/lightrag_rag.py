import os
import asyncio
import sys
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

from lightrag import LightRAG, QueryParam
# Assuming these functions correctly use os.getenv() internally or are configured by environment variables
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger
#from dotenv import load_dotenv # Removed as per instruction #8

#load_dotenv("/mnt/nas/gpt/.env") # Removed as per instruction #8

# Ensure OPENAI_API_KEY is set as an environment variable before running
# If using a custom base URL, also set OPENAI_API_BASE

# --- Major Variables (Instruction #3) ---
WORKING_DIR = "lightrag_data"
LOGGER_NAME = "lightrag"
LOGGER_LEVEL = "INFO"
DEFAULT_MODE = "mix" # Example, adjust as needed. Modes: naive, local, global, hybrid, mix
DEFAULT_QUERY = "What are the main themes of the paper?" # Example query
# SAMPLE_TEXT = "This is some sample text about AI and large language models." # Removed
PDF_FILE_PATH = "/mnt/nas/gpt/RAG/rag_all/31496.pdf" # Added PDF path

setup_logger(LOGGER_NAME, level=LOGGER_LEVEL)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text content from a PDF file."""
    print(colored(f"Reading PDF: {pdf_path}", 'cyan'))
    if not os.path.exists(pdf_path):
        print(colored(f"ERROR: PDF file not found at {pdf_path}", 'red'), file=sys.stderr)
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n" # Add newline between pages
        print(colored("PDF read successfully.", 'green'))
        if not text:
             print(colored("Warning: No text extracted from PDF.", 'yellow'), file=sys.stderr)
        return text
    except Exception as e:
        print(colored(f"ERROR reading PDF {pdf_path}: {e}", 'red'), file=sys.stderr)
        raise

async def initialize_rag():
    """Initializes the LightRAG instance and its dependencies."""
    # Retrieve API key from environment variable (Instruction #8)
    # from dotenv import load_dotenv # REMOVED
    # load_dotenv("/mnt/nas/gpt/.env") # REMOVED
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Error Handling (Instruction #6) with Termcolor (Instruction #1)
        print(colored("ERROR: OPENAI_API_KEY environment variable not set.", 'red'), file=sys.stderr)
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    # Retrieve optional base URL (Instruction #8)
    # Use os.getenv to avoid KeyError if not set. Lightrag/OpenAI library should handle None.
    api_base = os.getenv("OPENAI_API_BASE")

    # WORKAROUND: Lightrag's embedder seems to require OPENAI_API_BASE to be set.
    # Set default only if not provided by the user environment.
    if not api_base:
        default_base_url = "https://api.openai.com/v1"
        print(colored(f"WARN: OPENAI_API_BASE not found in environment. Setting default for LightRAG: {default_base_url}", 'yellow'), file=sys.stderr)
        os.environ['OPENAI_API_BASE'] = default_base_url
        # Update our local variable too, though the os.environ change is what likely matters here
        api_base = default_base_url

    print(colored("Initializing LightRAG...", 'cyan')) # Termcolor status update
    try:
        rag = LightRAG(
            working_dir=WORKING_DIR,
            embedding_func=openai_embed, # Assumes openai_embed uses env vars
            llm_model_func=gpt_4o_mini_complete # Assumes gpt_4o_mini_complete uses env vars
        )

        print(colored("Initializing storages...", 'cyan'))
        await rag.initialize_storages()
        print(colored("Initializing pipeline status...", 'cyan'))
        await initialize_pipeline_status()
        print(colored("LightRAG initialized successfully.", 'green'))
        return rag
    except Exception as e:
        print(colored(f"ERROR during LightRAG initialization: {e}", 'red'), file=sys.stderr)
        # Check specifically for the original error to provide a hint
        if isinstance(e, KeyError) and 'OPENAI_API_BASE' in str(e):
            print(colored("Hint: The underlying library raised KeyError for OPENAI_API_BASE.", 'yellow'), file=sys.stderr)
            print(colored("Ensure OPENAI_API_BASE is set if using a custom endpoint, or unset if using default OpenAI.", 'yellow'), file=sys.stderr)
        raise

async def main():
    """Main asynchronous function to run the RAG process."""
    # Initialize RAG instance
    rag = await initialize_rag()

    # Extract text from PDF
    document_text = extract_text_from_pdf(PDF_FILE_PATH)

    if not document_text:
        print(colored("Skipping insertion and query as no text was extracted from PDF.", 'yellow'), file=sys.stderr)
        return

    # Insert text (Instruction #7 - Separation of Concerns: Data Handling)
    # Prepare snippet for printing (fix for f-string backslash error)
    text_snippet = document_text[:100].replace('\n', ' ')
    print(colored(f"Inserting text from PDF (first 100 chars): '{text_snippet}...", 'cyan'))
    try:
        # Use ainsert for the async version
        # Consider chunking for very large PDFs if necessary/supported by LightRAG
        await rag.ainsert(document_text)
        print(colored("Text inserted successfully.", 'green'))
    except Exception as e:
        print(colored(f"ERROR during text insertion: {e}", 'red'), file=sys.stderr)
        # Decide if execution should stop or continue

    # Perform query (Instruction #7 - Separation of Concerns: Business Logic)
    print(colored(f"Performing query with mode='{DEFAULT_MODE}'...", 'cyan'))
    print(colored(f"Query: '{DEFAULT_QUERY}'", 'cyan'))
    try:
        # Assuming query is async
        # Use aquery for the async version
        result = await rag.aquery(
            DEFAULT_QUERY,
            param=QueryParam(mode=DEFAULT_MODE)
        )
        print(colored("Query completed successfully.", 'green'))
        print(f"Query Result: {result}") # Consider formatting result nicely
    except Exception as e:
        print(colored(f"ERROR during query: {e}", 'red'), file=sys.stderr)
        if isinstance(e, KeyError) and 'OPENAI_API_BASE' in str(e):
             print(colored("Hint: This KeyError suggests 'OPENAI_API_BASE' is accessed directly (e.g., os.environ['OPENAI_API_BASE']).", 'yellow'), file=sys.stderr)
             print(colored("1. If you use a custom OpenAI endpoint, ensure OPENAI_API_BASE is set as a system environment variable.", 'yellow'), file=sys.stderr)
             print(colored("2. If you use the default OpenAI endpoint, ensure OPENAI_API_BASE is *not* set.", 'yellow'), file=sys.stderr)
             print(colored("3. Check if 'lightrag' or 'openai' libraries require explicit base_url=None or are misinterpreting settings.", 'yellow'), file=sys.stderr)
        # Optionally re-raise or handle more specifically
        # raise e


if __name__ == "__main__":
    # Async Task Runner (Instruction #5) & Error Handling (Instruction #6)
    try:
        asyncio.run(main())
    except ValueError as ve: # Catch specific configuration errors like missing API key
         print(colored(f"Configuration error: {ve}", 'red'), file=sys.stderr)
         sys.exit(1) # Exit with a non-zero code for errors
    except Exception as e:
         print(colored(f"An unexpected error occurred in the main execution: {e}", 'red', attrs=['bold']), file=sys.stderr)
         # Optional: Log full traceback for debugging
         # import traceback
         # traceback.print_exc()
         sys.exit(1) # Exit with a non-zero code for errors