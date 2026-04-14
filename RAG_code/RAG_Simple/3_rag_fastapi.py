from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, RedirectResponse
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
import os
import time
import chromadb
from termcolor import colored
import tempfile
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
import uvicorn
from pydantic import BaseModel
import asyncio
import shutil
from fastapi.staticfiles import StaticFiles
from fastapi import Request

# Load environment variables
load_dotenv("/mnt/nas/gpt/.env")

# Major variables defined in ALL CAPS
DEFAULT_PDF_PATH = "/mnt/nas/gpt/nber/nber_2023/31496.pdf"
PERSIST_DIRECTORY = os.path.abspath("../chroma_db")
LLM_MODEL_NAME = "llama3"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
UPLOAD_DIR = "uploads"

# Document info
NBER_NUM = 31496
TITLE = "Pandemic, war, inflation: Oil market at a crossroads"
AUTHOR = "C. Baumeister"

# Define prompt template
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

# Custom streaming handler
class AsyncCallbackHandler(StreamingStdOutCallbackHandler):
    def __init__(self):
        super().__init__()
        self.text = ""
        self.queue = asyncio.Queue()
        
    def on_llm_new_token(self, token, **kwargs):
        self.text += token
        self.queue.put_nowait(token)

# Global state to store application state
global_state = {
    "initialized": False,
    "collection": None,
    "llm": None,
    "pdf_path": DEFAULT_PDF_PATH,
    "llm_model": LLM_MODEL_NAME,
    "temperature": 0.2,
    "n_results": 4,
    "status": []
}

# Pydantic models for request and response validation
class ProcessRequest(BaseModel):
    pdf_option: str
    llm_model: str
    temperature: float
    n_results: int

class QueryRequest(BaseModel):
    question: str

class StatusResponse(BaseModel):
    status: List[str]
    initialized: bool

class AnswerResponse(BaseModel):
    answer: str

# Create FastAPI app
app = FastAPI(
    title="Oil Market RAG Analysis API",
    description="API for querying oil market research using RAG (Retrieval Augmented Generation)",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads and static directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Function to process document and initialize the system
async def process_document_task(pdf_option: str, pdf_path: Optional[str], llm_model: str, temperature: float, n_results: int):
    global global_state
    
    # Update state with current settings
    global_state["llm_model"] = llm_model
    global_state["temperature"] = temperature
    global_state["n_results"] = n_results
    global_state["status"] = []
    
    try:
        # Handle PDF source
        if pdf_option == "Upload custom PDF" and pdf_path:
            global_state["status"].append(colored("📄 Loading custom PDF document...", "blue"))
            global_state["pdf_path"] = pdf_path
        else:
            global_state["status"].append(colored("📄 Loading default PDF document...", "blue"))
            global_state["pdf_path"] = DEFAULT_PDF_PATH
        
        # Load PDF
        loader = PyPDFLoader(global_state["pdf_path"])
        pages = loader.load()
        global_state["status"].append(colored(f"✅ Successfully loaded {len(pages)} pages from PDF", "green"))
        
        # Split documents
        global_state["status"].append(colored("🔪 Splitting document into smaller chunks...", "blue"))
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        texts = text_splitter.split_documents(pages)
        global_state["status"].append(colored(f"✅ Split into {len(texts)} text chunks", "green"))
        
        # Create directory if it doesn't exist
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        
        # Vector DB setup
        global_state["status"].append(colored(f"💾 Creating vector database at {PERSIST_DIRECTORY}...", "blue"))
        global_state["status"].append(colored("🤖 Using ChromaDB's default embedding function", "blue"))
        
        # Initialize ChromaDB client with default embeddings
        chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        
        # Convert LangChain documents to a format compatible with ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for i, doc in enumerate(texts):
            documents.append(doc.page_content)
            metadatas.append(doc.metadata)
            ids.append(f"doc_{i}")
        
        # Create or get collection
        collection_name = "oil_market_research"
        if pdf_option == "Upload custom PDF" and pdf_path:
            collection_name = f"custom_pdf_{int(time.time())}"
            
        collection = chroma_client.get_or_create_collection(name=collection_name)
        
        # Add documents to the collection
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        global_state["status"].append(colored("✅ Vector database created successfully", "green"))
        
        # Initialize the LLM
        global_state["status"].append(colored(f"🧠 Initializing Ollama LLM model: {llm_model}", "blue"))
        llm = Ollama(
            model=llm_model,
            temperature=temperature
        )
        global_state["status"].append(colored("✅ LLM initialized", "green"))
        
        # Save to global state
        global_state["collection"] = collection
        global_state["llm"] = llm
        global_state["initialized"] = True
        global_state["status"].append(colored("✅ System ready for queries", "green"))
        
        return True
    
    except Exception as e:
        error_msg = f"❌ An error occurred: {str(e)}"
        global_state["status"].append(colored(error_msg, "red"))
        return False

# Async generator for streaming responses
async def generate_response(question: str):
    global global_state
    
    # Create a custom handler for streaming
    callback_handler = AsyncCallbackHandler()
    
    # Query the collection
    results = global_state["collection"].query(
        query_texts=[question],
        n_results=global_state["n_results"]
    )
    
    retrieved_docs = results['documents'][0]
    
    # Construct context from retrieved documents
    context = "\n\n".join(retrieved_docs)
    
    # Format the prompt
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    
    # Generate the answer with streaming output
    llm_with_cb = Ollama(
        model=global_state["llm_model"],
        temperature=global_state["temperature"],
        callback_manager=CallbackManager([callback_handler])
    )
    
    # Start generation in a separate task
    asyncio.create_task(llm_with_cb.ainvoke(prompt))
    
    # Stream tokens as they are generated
    while True:
        try:
            token = await callback_handler.queue.get()
            yield token
        except asyncio.CancelledError:
            break

# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Redirect to UI"""
    return RedirectResponse(url="/ui")

@app.get("/api", tags=["Root"])
async def api_root():
    """API information"""
    return {
        "message": "🛢️ Oil Market RAG Analysis API",
        "description": "API for querying oil market research using RAG",
        "status": "available"
    }

@app.get("/status", response_model=StatusResponse, tags=["System"])
async def get_status():
    return {
        "status": global_state["status"],
        "initialized": global_state["initialized"]
    }

@app.post("/process", tags=["Document Processing"])
async def process_document(
    background_tasks: BackgroundTasks,
    pdf_option: str = Form("Use default PDF"),
    pdf_file: Optional[UploadFile] = File(None),
    llm_model: str = Form(LLM_MODEL_NAME),
    temperature: float = Form(0.2),
    n_results: int = Form(4)
):
    global global_state
    
    # Reset status
    global_state["status"] = []
    global_state["status"].append(colored("🔄 Starting document processing...", "blue"))
    
    # Handle uploaded file
    pdf_path = None
    if pdf_option == "Upload custom PDF" and pdf_file:
        pdf_path = os.path.join(UPLOAD_DIR, f"{int(time.time())}_{pdf_file.filename}")
        
        try:
            # Save uploaded file
            with open(pdf_path, "wb") as f:
                shutil.copyfileobj(pdf_file.file, f)
            global_state["status"].append(colored(f"✅ File uploaded successfully: {pdf_file.filename}", "green"))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")
    
    # Start processing in background
    background_tasks.add_task(
        process_document_task, pdf_option, pdf_path, llm_model, temperature, n_results
    )
    
    return {"message": "Document processing started", "status": "processing"}

@app.post("/query", tags=["Query"])
async def query(request: QueryRequest):
    global global_state
    
    if not global_state["initialized"]:
        raise HTTPException(status_code=400, detail="System not initialized. Please process a document first.")
    
    try:
        # Query the collection
        results = global_state["collection"].query(
            query_texts=[request.question],
            n_results=global_state["n_results"]
        )
        
        retrieved_docs = results['documents'][0]
        
        # Construct context from retrieved documents
        context = "\n\n".join(retrieved_docs)
        
        # Format the prompt
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=request.question)
        
        # Generate the answer
        response = global_state["llm"].invoke(prompt)
        
        return {"answer": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/query/stream", tags=["Query"])
async def query_stream(request: QueryRequest):
    global global_state
    
    if not global_state["initialized"]:
        raise HTTPException(status_code=400, detail="System not initialized. Please process a document first.")
    
    return StreamingResponse(
        generate_response(request.question),
        media_type="text/event-stream"
    )

@app.get("/examples", tags=["Query"])
async def get_example_questions():
    return {
        "examples": [
            "What were the main factors affecting oil prices during the pandemic?",
            "How did the Russia-Ukraine war impact the oil market?",
            "What are the projections for future oil demand?",
            "What supply shocks affected the oil market?"
        ]
    }

@app.get("/ui", response_class=HTMLResponse, tags=["UI"])
async def get_ui():
    """Serve the web UI"""
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        return HTMLResponse(content=f"<html><body><h1>Error loading UI</h1><p>{str(e)}</p></body></html>")

# Entry point
if __name__ == "__main__":
    print(colored("🚀 Starting the Oil Market RAG Analysis System with FastAPI", "green"))
    uvicorn.run("3_rag_fastapi:app", host="127.0.0.1", port=8080, reload=True) 