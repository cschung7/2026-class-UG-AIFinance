from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama 
from langchain_community.vectorstores import Chroma 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate 
from langchain.callbacks.manager import CallbackManager
from langchain.memory import ConversationBufferMemory 
import streamlit as st 
import os 
import time 
from termcolor import colored
import chromadb

from dotenv import load_dotenv
load_dotenv("/mnt/nas/gpt/.env")

# Major variables defined in ALL CAPS
PDF_PATH = "/mnt/nas/gpt/nber/nber_2023/31496.pdf"
PERSIST_DIRECTORY = os.path.abspath("../chroma_db")
LLM_MODEL_NAME = "llama3"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Document parsing
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

try:
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
    texts = text_splitter.split_documents(pages)
    print(colored(f"✅ Split into {len(texts)} text chunks", "green"))
    
    # Create directory if it doesn't exist
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    
    print(colored(f"💾 Creating vector database at {PERSIST_DIRECTORY}...", "cyan"))
    print(colored("🤖 Using ChromaDB's default embedding function", "cyan"))
    
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
    collection = chroma_client.get_or_create_collection(name="oil_market_research")
    
    # Add documents to the collection
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
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
    
    while True:
        print(colored("\n==== RAG Question Answering System ====", "green"))
        user_query = input(colored("Enter your question (or 'exit' to quit): ", "yellow"))
        
        if user_query.lower() == 'exit':
            print(colored("Goodbye!", "green"))
            break
        
        print(colored("\n🔍 Retrieving relevant documents...", "cyan"))
        
        # Query the collection
        results = collection.query(
            query_texts=[user_query],
            n_results=4
        )
        
        retrieved_docs = results['documents'][0]
        
        # Construct context from retrieved documents
        context = "\n\n".join(retrieved_docs)
        
        print(colored("📝 Constructing answer using LLM...", "cyan"))
        
        # Format the prompt
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=user_query)
        
        print(colored("\n🤖 Answer:", "green"))
        # Generate the answer
        llm.invoke(prompt)
        print("\n")

except FileNotFoundError as e:
    print(colored(f"❌ Error: PDF file not found at {PDF_PATH}", "red"))
    print(colored(f"Original error: {str(e)}", "red"))
except Exception as e:
    print(colored("❌ An error occurred:", "red"))
    print(colored(str(e), "red"))
