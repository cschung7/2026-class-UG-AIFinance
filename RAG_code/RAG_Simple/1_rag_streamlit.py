from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
import streamlit as st
import os
import time
import chromadb
from termcolor import colored
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv("/mnt/nas/gpt/.env")

# Major variables defined in ALL CAPS
DEFAULT_PDF_PATH = "/mnt/nas/gpt/nber/nber_2023/31496.pdf"
PERSIST_DIRECTORY = os.path.abspath("../chroma_db")
LLM_MODEL_NAME = "llama3"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

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

# Custom callback handler to redirect Ollama's output to Streamlit
class StreamlitCallbackHandler(StreamingStdOutCallbackHandler):
    def __init__(self, container):
        super().__init__()
        self.container = container
        self.text = ""
        
    def on_llm_new_token(self, token, **kwargs):
        self.text += token
        self.container.markdown(self.text)

# Set page config
st.set_page_config(
    page_title="RAG Oil Market Analysis",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for dark mode
st.markdown("""
<style>
    .stApp {
        background-color: #121212;
        color: white;
    }
    .stMarkdown {
        color: white;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
    }
    .success-message {
        color: #4CAF50;
        font-weight: bold;
    }
    .info-message {
        color: #2196F3;
        font-weight: bold;
    }
    .warning-message {
        color: #FFC107;
        font-weight: bold;
    }
    .error-message {
        color: #F44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🛢️ Oil Market RAG Analysis System")
st.markdown("An interactive system to query research on oil markets using RAG (Retrieval Augmented Generation)")

# Sidebar
with st.sidebar:
    st.header("📋 Configuration")
    
    # PDF upload option
    pdf_option = st.radio("Choose PDF source:", ["Use default PDF", "Upload custom PDF"])
    
    uploaded_file = None
    if pdf_option == "Upload custom PDF":
        uploaded_file = st.file_uploader("Upload a PDF document", type=['pdf'])
    
    # Model parameters
    st.subheader("Model Parameters")
    llm_model = st.selectbox("LLM Model", ["llama3", "mistral", "gemma"], index=0)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    
    n_results = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=4)
    
    # Process document button
    process_button = st.button("Process Document & Initialize System")

# Main content area
document_status = st.empty()
vector_db_status = st.empty()
model_status = st.empty()
query_container = st.container()
answer_container = st.container()

# Session state initialization
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'pdf_path' not in st.session_state:
    st.session_state.pdf_path = DEFAULT_PDF_PATH

# Process document and initialize system
if process_button or st.session_state.initialized:
    try:
        if not st.session_state.initialized:
            # Set up progress display
            document_status.info("📄 Loading PDF document...")
            
            # Handle PDF source
            temp_file = None
            if pdf_option == "Upload custom PDF" and uploaded_file is not None:
                # Save uploaded file to temp location
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                temp_file.write(uploaded_file.getvalue())
                st.session_state.pdf_path = temp_file.name
                temp_file.close()
            
            # Load PDF
            loader = PyPDFLoader(st.session_state.pdf_path)
            pages = loader.load()
            document_status.success(f"✅ Successfully loaded {len(pages)} pages from PDF")
            
            # Split documents
            document_status.info("🔪 Splitting document into smaller chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            texts = text_splitter.split_documents(pages)
            document_status.success(f"✅ Split into {len(texts)} text chunks")
            
            # Create directory if it doesn't exist
            os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
            
            # Vector DB setup
            vector_db_status.info(f"💾 Creating vector database at {PERSIST_DIRECTORY}...")
            vector_db_status.info("🤖 Using ChromaDB's default embedding function")
            
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
            if pdf_option == "Upload custom PDF" and uploaded_file is not None:
                collection_name = f"custom_pdf_{int(time.time())}"
                
            collection = chroma_client.get_or_create_collection(name=collection_name)
            
            # Add documents to the collection
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            vector_db_status.success("✅ Vector database created successfully")
            
            # Initialize the LLM
            model_status.info(f"🧠 Initializing Ollama LLM model: {llm_model}")
            llm = Ollama(
                model=llm_model,
                temperature=temperature
            )
            model_status.success("✅ LLM initialized")
            
            # Save to session state
            st.session_state.collection = collection
            st.session_state.llm = llm
            st.session_state.initialized = True
            
            # Clean up the temp file if we created one
            if temp_file is not None and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
        
        # User query section
        with query_container:
            st.subheader("🔍 Ask Questions About the Document")
            user_query = st.text_input("Enter your question:")
            submit_button = st.button("Submit Query")
        
        # Process user query
        if submit_button and user_query:
            with answer_container:
                # Status indicators during processing
                query_status = st.info("🔍 Retrieving relevant documents...")
                
                # Query the collection
                results = st.session_state.collection.query(
                    query_texts=[user_query],
                    n_results=n_results
                )
                
                retrieved_docs = results['documents'][0]
                
                # Construct context from retrieved documents
                context = "\n\n".join(retrieved_docs)
                
                query_status.info("📝 Constructing answer using LLM...")
                
                # Format the prompt
                prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=user_query)
                
                # Display answer
                st.subheader("🤖 Answer:")
                
                # Create a container for streaming output
                answer_text = st.empty()
                
                # Create a custom handler for streaming
                output_container = st.empty()
                callback_handler = StreamlitCallbackHandler(output_container)
                
                # Generate the answer with streaming output
                llm_with_cb = Ollama(
                    model=llm_model,
                    temperature=temperature,
                    callback_manager=CallbackManager([callback_handler])
                )
                
                # Generate response
                llm_with_cb.invoke(prompt)
                
                # Clear the status indicator
                query_status.empty()
                
    except FileNotFoundError as e:
        st.error(f"❌ Error: PDF file not found at {st.session_state.pdf_path}")
        st.error(f"Original error: {str(e)}")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")

else:
    # Initial instructions
    st.info("👈 Configure the system and click 'Process Document & Initialize System' to start")
