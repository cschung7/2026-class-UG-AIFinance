"""
Contextual RAG Implementation based on Anthropic's approach
https://www.anthropic.com/news/contextual-retrieval

This implementation enhances RAG by:
1. Using contextual embeddings - adding metadata and document context to chunks
2. Adding hybrid search with BM25 
3. Implementing a simple reranking approach

References:
- https://www.anthropic.com/news/contextual-retrieval
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

# For BM25
import rank_bm25

# For document storage and metadata
from langchain.schema import Document

# ChromaDB for direct access
import chromadb
from chromadb.utils import embedding_functions

# Load environment variables
from dotenv import load_dotenv
load_dotenv("/mnt/nas/gpt/.env")

# Major variables defined in ALL CAPS
PDF_PATH = "/mnt/nas/gpt/nber/nber_2023/31496.pdf"
PERSIST_DIRECTORY = os.path.abspath("../chroma_db_contextual")
LLM_MODEL_NAME = "llama3"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 5
RERANKING_WEIGHT = 0.7  # Weight for vector similarity (vs BM25)
COLLECTION_NAME = "oil_market_research_contextual"

# Document parsing
NBER_NUM = 31496 
TITLE = "Pandemic, war, inflation: Oil market at a crossroads"
AUTHOR = "C. Baumeister"

# Define prompt template for contextual RAG
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

# Define a prompt template for chunk enrichment
CHUNK_ENRICHMENT_TEMPLATE = """
Title: {title}
Author: {author}
Document ID: {doc_id}
Page: {page_num}
Section: This chunk appears to be from {section_context}.

CONTENT:
{content}
"""

def enrich_chunk_with_context(
    chunk: Document, 
    title: str, 
    author: str, 
    doc_id: str,
    preceding_text: str = "",
    following_text: str = ""
) -> Document:
    """Add contextual information to the chunk to improve embedding quality."""
    
    # Extract page number from metadata
    page_num = chunk.metadata.get("page", 0) + 1  # 0-indexed to 1-indexed
    
    # Determine section context from content (simplified approach)
    section_context = "unknown section"
    content_lower = chunk.page_content.lower()
    
    if "abstract" in content_lower[:100]:
        section_context = "abstract"
    elif "introduction" in content_lower[:100]:
        section_context = "introduction"
    elif "conclusion" in content_lower[:100]:
        section_context = "conclusion"
    elif "methodology" in content_lower[:100] or "method" in content_lower[:100]:
        section_context = "methodology"
    elif "result" in content_lower[:100]:
        section_context = "results"
    elif "discussion" in content_lower[:100]:
        section_context = "discussion"
    elif "reference" in content_lower[:100] or "bibliography" in content_lower[:100]:
        section_context = "references"
    
    # Create enriched content
    enriched_content = CHUNK_ENRICHMENT_TEMPLATE.format(
        title=title,
        author=author,
        doc_id=doc_id,
        page_num=page_num,
        section_context=section_context,
        content=chunk.page_content
    )
    
    # Create a new Document with enriched content and original metadata
    enriched_doc = Document(
        page_content=enriched_content,
        metadata={
            **chunk.metadata,
            "enriched": True,
            "section_context": section_context,
            "original_content": chunk.page_content,
        }
    )
    
    return enriched_doc

def create_bm25_index(documents: List[Document]) -> Tuple[rank_bm25.BM25Okapi, List[str]]:
    """Create a BM25 index from the documents."""
    print(colored("🔍 Creating BM25 index...", "cyan"))
    
    # Extract original content from enriched documents
    tokenized_corpus = []
    original_contents = []
    
    for doc in documents:
        if "original_content" in doc.metadata:
            text = doc.metadata["original_content"]
        else:
            text = doc.page_content
            
        original_contents.append(text)
        # Simple tokenization by splitting on whitespace
        tokenized_doc = text.lower().split()
        tokenized_corpus.append(tokenized_doc)
    
    # Create BM25 index
    bm25 = rank_bm25.BM25Okapi(tokenized_corpus)
    print(colored("✅ BM25 index created successfully", "green"))
    
    return bm25, original_contents

def hybrid_search(
    query: str,
    chroma_collection,
    bm25_index,
    documents: List[str],
    top_k: int = 5,
    vector_weight: float = 0.7
) -> List[Document]:
    """
    Perform hybrid search using both vector similarity and BM25.
    
    Args:
        query: The user query
        chroma_collection: The ChromaDB collection for semantic search
        bm25_index: The BM25 index
        documents: The list of document contents
        top_k: Number of results to return
        vector_weight: Weight for vector search (1-vector_weight = BM25 weight)
        
    Returns:
        List of retrieved documents
    """
    print(colored(f"🔍 Performing hybrid search with vector_weight={vector_weight}...", "cyan"))
    
    # Get vector search results
    vector_results = chroma_collection.query(
        query_texts=[query],
        n_results=top_k*2,
        include=["documents", "metadatas", "distances"]
    )
    
    # Get BM25 results
    tokenized_query = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    
    # Normalize BM25 scores to [0, 1] range
    if max(bm25_scores) > 0:
        bm25_scores = bm25_scores / max(bm25_scores)
    
    # Create a dictionary of document_id -> vector_score
    vector_scores = {}
    
    # ChromaDB distances
    docs = vector_results['documents'][0]
    distances = vector_results['distances'][0]
    metadatas = vector_results['metadatas'][0]
    
    for i, (doc, distance) in enumerate(zip(docs, distances)):
        # Convert distance to similarity (lower distance = higher similarity)
        similarity = 1.0 / (1.0 + distance)
        
        # Get original content
        original_content = metadatas[i].get("original_content", doc)
        vector_scores[original_content] = similarity
    
    # Combine scores
    combined_scores = []
    for i, doc_content in enumerate(documents):
        vector_score = vector_scores.get(doc_content, 0.0)
        bm25_score = bm25_scores[i]
        
        # Weighted combination
        combined_score = (vector_weight * vector_score) + ((1 - vector_weight) * bm25_score)
        combined_scores.append((i, combined_score))
    
    # Sort by combined score (descending)
    combined_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Get top_k documents
    top_indices = [idx for idx, _ in combined_scores[:top_k]]
    
    # Create Document objects with the original content
    result_docs = []
    for idx in top_indices:
        doc_content = documents[idx]
        result_docs.append(Document(
            page_content=doc_content,
            metadata={"score": combined_scores[top_indices.index(idx)][1]}
        ))
    
    print(colored(f"✅ Retrieved {len(result_docs)} documents using hybrid search", "green"))
    return result_docs

def main():
    try:
        print(colored("📚 CONTEXTUAL RAG SYSTEM", "green", attrs=["bold"]))
        print(colored("===========================", "green"))
        print(colored("Based on Anthropic's Contextual Retrieval approach", "green"))
        print("")
        
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
        
        # Enrich chunks with context
        print(colored("🔍 Enriching chunks with contextual information...", "cyan"))
        enriched_chunks = []
        for i, chunk in enumerate(tqdm(chunks)):
            enriched_chunk = enrich_chunk_with_context(
                chunk=chunk,
                title=TITLE,
                author=AUTHOR,
                doc_id=f"NBER-{NBER_NUM}"
            )
            enriched_chunks.append(enriched_chunk)
        print(colored("✅ Enriched all chunks with contextual information", "green"))
        
        # Create directory if it doesn't exist
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        
        # Initialize ChromaDB client with default embeddings
        print(colored(f"💾 Creating vector database at {PERSIST_DIRECTORY}...", "cyan"))
        print(colored("🤖 Using ChromaDB's default embedding function", "cyan"))
        
        # Initialize ChromaDB client
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
        
        for i, doc in enumerate(enriched_chunks):
            documents.append(doc.page_content)
            metadatas.append({
                **doc.metadata,
                "original_content": doc.metadata.get("original_content", doc.page_content)
            })
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
        
        # Create BM25 index
        bm25_index, doc_contents = create_bm25_index(enriched_chunks)
        
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
            print(colored("\n==== CONTEXTUAL RAG Question Answering System ====", "green"))
            user_query = input(colored("Enter your question (or 'exit' to quit): ", "yellow"))
            
            if user_query.lower() == 'exit':
                print(colored("Goodbye!", "green"))
                break
            
            start_time = time.time()
            print(colored("\n🔍 Retrieving relevant documents with hybrid search...", "cyan"))
            
            # Perform hybrid search
            retrieved_docs = hybrid_search(
                query=user_query,
                chroma_collection=collection,
                bm25_index=bm25_index,
                documents=doc_contents,
                top_k=TOP_K_RETRIEVAL,
                vector_weight=RERANKING_WEIGHT
            )
            
            # Extract original content from metadata for cleaner context
            context_parts = []
            for i, doc in enumerate(retrieved_docs):
                doc_content = doc.page_content
                score = doc.metadata.get("score", 0.0)
                
                # Add document number and relevance score for reference
                context_parts.append(f"[Document {i+1} - Relevance: {score:.2f}]\n{doc_content}\n")
            
            # Construct context from retrieved documents
            context = "\n\n".join(context_parts)
            
            retrieval_time = time.time() - start_time
            print(colored(f"✅ Retrieved {len(retrieved_docs)} documents in {retrieval_time:.2f}s", "green"))
            
            print(colored("📝 Constructing answer using LLM...", "cyan"))
            
            # Format the prompt
            prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=user_query)
            
            print(colored("\n🤖 Answer:", "green"))
            # Generate the answer
            llm_start = time.time()
            llm.invoke(prompt)
            llm_time = time.time() - llm_start
            
            print(colored(f"\n⏱️ Time: Retrieval={retrieval_time:.2f}s, LLM={llm_time:.2f}s, Total={(retrieval_time+llm_time):.2f}s", "blue"))
            print("\n")

    except FileNotFoundError as e:
        print(colored(f"❌ Error: PDF file not found at {PDF_PATH}", "red"))
        print(colored(f"Original error: {str(e)}", "red"))
    except Exception as e:
        print(colored("❌ An error occurred:", "red"))
        print(colored(str(e), "red"))

if __name__ == "__main__":
    main() 