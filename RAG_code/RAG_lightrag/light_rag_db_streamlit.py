#!/usr/bin/env python3

import os
import sys
import asyncio
import streamlit as st
from typing import Optional

try:
    from termcolor import colored
except ImportError:
    print("Warning: termcolor not installed. Install using: pip install termcolor", file=sys.stderr)
    def colored(text, color=None, on_color=None, attrs=None):
        return text

# Ensure lightrag and openai are installed and configured
try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
    from lightrag.kg.shared_storage import initialize_pipeline_status
    from lightrag.utils import setup_logger
except ImportError as e:
    st.error(f"ERROR: Required LightRAG components not installed or found: {e}")
    sys.exit(1)

# --- Major Variables ---
WORKING_DIR = "/mnt/nas/gpt/RAG/rag_all/lightrag_storage"  # CHANGE HERE
LOGGER_NAME = "lightrag_streamlit"
LOGGER_LEVEL = "INFO"
DEFAULT_MODE = "mix"
ALLOWED_MODES = ['naive', 'local', 'global', 'hybrid', 'mix']

# Setup logger
setup_logger(LOGGER_NAME, level=LOGGER_LEVEL)

def verify_storage_exists():
    """Verify that the required storage files exist."""
    try:
        required_files = [
            "kv_store_llm_response_cache.json",
            "graph_chunk_entity_relation.graphml",
            "vdb_chunks.json",
            "vdb_relationships.json",
            "vdb_entities.json",
            "kv_store_text_chunks.json",
            "kv_store_full_docs.json",
            "kv_store_doc_status.json"
        ]
        
        if not os.path.exists(WORKING_DIR):
            error_msg = f"Database directory not found at: {WORKING_DIR}"
            print(colored(error_msg, 'red'))
            return False
            
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(WORKING_DIR, f))]
        if missing_files:
            error_msg = f"Missing required database files: {', '.join(missing_files)}"
            print(colored(error_msg, 'red'))
            return False
            
        print(colored(f"Database verified at: {WORKING_DIR}", 'green'))
        return True
    except Exception as e:
        print(colored(f"Error verifying database: {e}", 'red'))
        return False

async def initialize_rag():
    """Initializes the LightRAG instance using existing data."""
    if 'rag' in st.session_state:
        return st.session_state.rag

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        error_msg = "ERROR: OPENAI_API_KEY environment variable not set."
        st.error(error_msg)
        print(colored(error_msg, 'red'))
        raise ValueError(error_msg)

    api_base = os.getenv("OPENAI_API_BASE")
    if not api_base:
        default_base_url = "https://api.openai.com/v1"
        warning_msg = (
            "Note: OPENAI_API_BASE not set. Using default OpenAI API endpoint. "
            f"Default endpoint: {default_base_url}"
        )
        st.info(warning_msg)
        print(colored(warning_msg, 'yellow'))
        os.environ['OPENAI_API_BASE'] = default_base_url

    print(colored("Initializing LightRAG...", 'cyan'))
    try:
        if not verify_storage_exists():
            raise RuntimeError("Storage directories not found. Please ensure the database is properly processed.")
        
        rag = LightRAG(
            working_dir=WORKING_DIR,
            embedding_func=openai_embed,
            llm_model_func=gpt_4o_mini_complete
        )
        print(colored("Initializing storages...", 'cyan'))
        await rag.initialize_storages()
        print(colored("Initializing pipeline status...", 'cyan'))
        await initialize_pipeline_status()
        print(colored("LightRAG initialized successfully.", 'green'))
        
        st.session_state.rag = rag
        return rag
    except Exception as e:
        error_msg = f"ERROR during LightRAG initialization: {e}"
        st.error(error_msg)
        print(colored(error_msg, 'red'))
        raise

async def process_query(rag: LightRAG, query: str, mode: str):
    """Process a single query using the RAG system."""
    try:
        print(colored(f"Processing query in {mode} mode: {query}", 'cyan'))
        result = await rag.aquery(
            query,
            param=QueryParam(mode=mode)
        )
        print(colored("Query completed successfully.", 'green'))
        if not result or result.strip() == "":
            error_msg = "No response generated. Please try rephrasing your question or using a different query mode."
            st.warning(error_msg)
            print(colored(error_msg, 'yellow'))
            return None
        return result
    except Exception as e:
        error_msg = f"ERROR during query: {e}"
        st.error(error_msg)
        print(colored(error_msg, 'red'))
        if "context_window_exceeded" in str(e).lower():
            st.info("💡 Tip: Try breaking your question into smaller parts or use a different query mode.")
        elif "rate_limit" in str(e).lower():
            st.info("💡 Tip: Please wait a moment and try again.")
        elif "invalid_request_error" in str(e).lower():
            st.info("💡 Tip: Try rephrasing your question or using simpler language.")
        return None

async def main():
    st.title("🔍 LightRAG Query Interface")
    st.markdown("Query your processed documents using LightRAG")
    
    # Configuration status section
    with st.expander("⚙️ Configuration Status", expanded=True):
        st.write("API Configuration:")
        api_key = "✓ Set" if os.getenv("OPENAI_API_KEY") else "❌ Not Set"
        api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        api_base_status = "Custom Endpoint" if os.getenv("OPENAI_API_BASE") else "Default OpenAI Endpoint"
        
        st.markdown(f"""
        - **OpenAI API Key**: {api_key}
        - **API Endpoint**: {api_base_status}
        - **Current Endpoint**: `{api_base}`
        - **Database Location**: `{WORKING_DIR}`
        """)

    # Initialize RAG
    try:
        rag = await initialize_rag()
    except Exception as e:
        st.error("Failed to initialize LightRAG. Please check your configuration and database location.")
        return

    # Query interface
    if verify_storage_exists():
        st.subheader("🤔 Ask Questions")
        
        # Query mode selection with tooltips
        mode_descriptions = {
            'naive': 'Basic vector search without refinement',
            'local': 'Focuses on context-dependent information',
            'global': 'Utilizes broader knowledge from the entire corpus',
            'hybrid': 'Combines local and global retrieval',
            'mix': 'Integrates knowledge graph structures with vector retrieval'
        }
        
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("Enter your question:")
        with col2:
            mode = st.selectbox(
                "Query mode:",
                ALLOWED_MODES,
                index=ALLOWED_MODES.index(DEFAULT_MODE),
                help="\n".join([f"• {mode}: {desc}" for mode, desc in mode_descriptions.items()])
            )
        
        if st.button("🔍 Submit Query", type="primary"):
            if query:
                with st.spinner("🤖 Processing your query..."):
                    result = await process_query(rag, query, mode)
                    if result:
                        st.markdown("### 📝 Answer")
                        st.markdown(result)
            else:
                st.warning("⚠️ Please enter a question")
    else:
        st.error("❌ Database not found. Please ensure the correct database path is specified and contains processed documents.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        print(colored(f"An unexpected error occurred: {e}", 'red', attrs=['bold']))
        sys.exit(1) 