"""
Reranking RAG Implementation using Cohere's Rerank model
https://docs.cohere.com/reference/rerank

This implementation enhances RAG by:
1. Using vector search to retrieve candidate documents
2. Applying Cohere's Rerank to improve retrieval quality
3. Passing the reranked documents to an LLM for answer generation

References:
- https://docs.cohere.com/docs/rerank
"""

import os
import time
from typing import List, Dict, Any, Tuple
import numpy as np
from termcolor import colored
from tqdm import tqdm

# Langchain & Vector store
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.manager import CallbackManager

# For document storage and metadata
from langchain.schema import Document

# ChromaDB for vector search
import chromadb
from chromadb.utils import embedding_functions

# Cohere for reranking
import cohere

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Major variables defined in ALL CAPS
PDF_PATH = "/mnt/nas/gpt/nber/nber_2023/31496.pdf"
PERSIST_DIRECTORY = os.path.abspath("../chroma_db_rerank")
LLM_MODEL_NAME = "llama3"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
INITIAL_RETRIEVAL_K = 20  # Retrieve more documents initially
FINAL_K = 5  # Number of documents after reranking
COLLECTION_NAME = "oil_market_research_reranked"
COHERE_RERANK_MODEL = "rerank-english-v3.0"

# Document parsing
NBER_NUM = 31496 
TITLE = "Pandemic, war, inflation: Oil market at a crossroads"
AUTHOR = "C. Baumeister"

# Define prompt template for RAG
RAG_PROMPT_TEMPLATE = """
You are an economic analyst specializing in the oil market.
You are given the following chunks of a research paper to help answer a question.

CONTEXT:
{context}

QUESTION:
{question}

Using only the information from the provided context, please provide a detailed and well-structured answer.
If the information needed to answer the question is not present in the context, state that you don't have enough information.
"""

def add_metadata_to_chunk(
    chunk: Document, 
    title: str, 
    author: str, 
    doc_id: str
) -> Document:
    """Add metadata to chunk for better retrieval and reranking."""
    
    # Extract page number from metadata
    page_num = chunk.metadata.get("page", 0) + 1  # 0-indexed to 1-indexed
    
    # Add metadata to the document
    enhanced_doc = Document(
        page_content=chunk.page_content,
        metadata={
            **chunk.metadata,
            "title": title,
            "author": author,
            "doc_id": doc_id,
            "page_num": page_num
        }
    )
    
    return enhanced_doc

def rerank_documents(
    query: str,
    documents: List[Document],
    cohere_client,
    top_k: int = 5
) -> List[Document]:
    """
    Rerank documents using Cohere's reranking model.
    
    Args:
        query: User query
        documents: List of candidate documents to rerank
        cohere_client: Initialized Cohere client
        top_k: Number of documents to return after reranking
        
    Returns:
        List of reranked documents
    """
    print(colored("🔄 Reranking documents with Cohere...", "cyan"))
    
    try:
        # Extract document texts for reranking
        doc_texts = [doc.page_content for doc in documents]
        
        # Call Cohere Rerank API
        rerank_results = cohere_client.rerank(
            model=COHERE_RERANK_MODEL,
            query=query,
            documents=doc_texts,
            top_n=top_k
        )
        
        # Create reranked document list
        reranked_docs = []
        for result in rerank_results.results:
            idx = result.index
            relevance_score = result.relevance_score
            
            # Create new document with relevance score in metadata
            reranked_doc = Document(
                page_content=documents[idx].page_content,
                metadata={
                    **documents[idx].metadata,
                    "relevance_score": relevance_score
                }
            )
            reranked_docs.append(reranked_doc)
        
        print(colored(f"✅ Successfully reranked documents", "green"))
        return reranked_docs
        
    except Exception as e:
        print(colored(f"❌ Error during reranking: {str(e)}", "red"))
        # Fall back to original documents if reranking fails
        return documents[:top_k]

def main():
    try:
        print(colored("📚 RERANK RAG SYSTEM", "green", attrs=["bold"]))
        print(colored("===========================", "green"))
        print(colored("Enhanced RAG with Cohere Reranking", "green"))
        print("")
        
        # Get the Cohere API key
        COHERE_API_KEY = os.getenv("COHERE_API_KEY")
        if not COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY not found in environment variables")
        
        # Initialize Cohere client
        print(colored("🔄 Initializing Cohere client for reranking...", "cyan"))
        co = cohere.Client(COHERE_API_KEY)
        print(colored("✅ Cohere client initialized successfully", "green"))
        
        print(colored("📄 Loading PDF document...", "cyan"))
        loader = PyPDFLoader(PDF_PATH)
        pages = loader.load()
        print(colored(f"✅ Successfully loaded {len(pages)} pages from PDF", "green"))
        
        # Split the documents into smaller chunks
        print(colored("🔪 Splitting document into smaller chunks...", "cyan"))
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(pages)
        print(colored(f"✅ Split into {len(chunks)} text chunks", "green"))
        
        # Add metadata to chunks
        print(colored("📋 Adding metadata to chunks...", "cyan"))
        enhanced_chunks = []
        for chunk in tqdm(chunks):
            enhanced_chunk = add_metadata_to_chunk(
                chunk=chunk,
                title=TITLE,
                author=AUTHOR,
                doc_id=f"NBER-{NBER_NUM}"
            )
            enhanced_chunks.append(enhanced_chunk)
        print(colored("✅ Added metadata to all chunks", "green"))
        
        # Create directory if it doesn't exist
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        
        # Initialize ChromaDB client
        print(colored(f"💾 Creating vector database at {PERSIST_DIRECTORY}...", "cyan"))
        chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        
        # Create or get collection with default embedding function
        default_ef = embedding_functions.DefaultEmbeddingFunction()
        
        # Clear existing collection if it exists and recreate it
        try:
            print(colored(f"🧹 Checking for existing collection: {COLLECTION_NAME}...", "cyan"))
            chroma_client.delete_collection(COLLECTION_NAME)
            print(colored(f"🗑️ Deleted existing collection to prevent duplicate IDs", "yellow"))
        except:
            print(colored("✅ No existing collection to delete", "green"))
        
        # Create new collection
        collection = chroma_client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=default_ef
        )
        print(colored(f"✅ Created new collection: {COLLECTION_NAME}", "green"))
        
        # Convert LangChain documents to a format compatible with ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for i, doc in enumerate(enhanced_chunks):
            documents.append(doc.page_content)
            metadatas.append(doc.metadata)
            ids.append(f"doc_{i}")
        
        # Add documents to the collection
        print(colored(f"🔖 Adding {len(documents)} documents to ChromaDB collection...", "cyan"))
        # Add in batches to prevent memory issues
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            collection.add(
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
        
        print(colored("✅ Vector database created successfully", "green"))
        
        # Initialize the LLM
        print(colored(f"🧠 Initializing Ollama LLM model: {LLM_MODEL_NAME}", "cyan"))
        llm = Ollama(
            model=LLM_MODEL_NAME,
            temperature=0.2,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )
        print(colored("✅ LLM initialized", "green"))
        
        # Main interaction loop
        while True:
            print(colored("\n==== RERANK RAG Question Answering System ====", "green"))
            user_query = input(colored("Enter your question (or 'exit' to quit): ", "yellow"))
            
            if user_query.lower() == 'exit':
                print(colored("Goodbye!", "green"))
                break
            
            start_time = time.time()
            print(colored("\n🔍 Retrieving candidate documents...", "cyan"))
            
            # Initial retrieval with ChromaDB
            vector_results = collection.query(
                query_texts=[user_query],
                n_results=INITIAL_RETRIEVAL_K,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert ChromaDB results to Document objects
            retrieved_docs = []
            for i in range(len(vector_results['documents'][0])):
                doc_content = vector_results['documents'][0][i]
                metadata = vector_results['metadatas'][0][i]
                distance = vector_results['distances'][0][i]
                
                # Create Document with vector similarity info
                doc = Document(
                    page_content=doc_content,
                    metadata={
                        **metadata,
                        "vector_distance": distance,
                        "vector_similarity": 1.0 / (1.0 + distance)
                    }
                )
                retrieved_docs.append(doc)
            
            retrieval_time = time.time() - start_time
            print(colored(f"✅ Retrieved {len(retrieved_docs)} candidate documents in {retrieval_time:.2f}s", "green"))
            
            # Rerank documents with Cohere
            rerank_start = time.time()
            reranked_docs = rerank_documents(
                query=user_query,
                documents=retrieved_docs,
                cohere_client=co,
                top_k=FINAL_K
            )
            rerank_time = time.time() - rerank_start
            print(colored(f"✅ Reranked documents in {rerank_time:.2f}s", "green"))
            
            # Construct context from reranked documents
            context_parts = []
            for i, doc in enumerate(reranked_docs):
                relevance_score = doc.metadata.get("relevance_score", 0.0)
                page_num = doc.metadata.get("page_num", "unknown")
                doc_content = doc.page_content
                
                # Add document number and relevance score for reference
                context_parts.append(f"[Document {i+1} - Page {page_num} - Relevance: {relevance_score:.4f}]\n{doc_content}\n")
            
            # Construct context from reranked documents
            context = "\n\n".join(context_parts)
            
            print(colored("📝 Constructing answer using LLM...", "cyan"))
            
            # Format the prompt
            prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=user_query)
            
            print(colored("\n🤖 Answer:", "green"))
            # Generate the answer
            llm_start = time.time()
            llm.invoke(prompt)
            llm_time = time.time() - llm_start
            
            total_time = retrieval_time + rerank_time + llm_time
            print(colored(f"\n⏱️ Time: Retrieval={retrieval_time:.2f}s, Rerank={rerank_time:.2f}s, LLM={llm_time:.2f}s, Total={total_time:.2f}s", "blue"))
            print("\n")

    except FileNotFoundError as e:
        print(colored(f"❌ Error: PDF file not found at {PDF_PATH}", "red"))
        print(colored(f"Original error: {str(e)}", "red"))
    except ValueError as e:
        print(colored(f"❌ Value Error: {str(e)}", "red"))
    except Exception as e:
        print(colored("❌ An error occurred:", "red"))
        print(colored(str(e), "red"))

if __name__ == "__main__":
    main() 