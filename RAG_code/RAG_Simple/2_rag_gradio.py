from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
import gradio as gr
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

# Custom callback handler for streaming to Gradio
class GradioCallbackHandler(StreamingStdOutCallbackHandler):
    def __init__(self):
        super().__init__()
        self.text = ""
        
    def on_llm_new_token(self, token, **kwargs):
        self.text += token
        return self.text

# Global variables to store state
global_state = {
    "initialized": False,
    "collection": None,
    "llm": None,
    "pdf_path": DEFAULT_PDF_PATH,
    "llm_model": LLM_MODEL_NAME,
    "temperature": 0.2,
    "n_results": 4
}

# Function to initialize the system and process the document
def process_document(pdf_option, pdf_file, llm_model, temperature, n_results, progress=gr.Progress()):
    global global_state
    
    # Update state with current settings
    global_state["llm_model"] = llm_model
    global_state["temperature"] = temperature
    global_state["n_results"] = n_results
    
    status_updates = []
    
    try:
        # Handle PDF source
        temp_file = None
        if pdf_option == "Upload custom PDF" and pdf_file is not None:
            status_updates.append("📄 Loading custom PDF document...")
            # Save uploaded file to temp location
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            with open(pdf_file.name, "rb") as f:
                temp_file.write(f.read())
            global_state["pdf_path"] = temp_file.name
            temp_file.close()
        else:
            status_updates.append("📄 Loading default PDF document...")
            global_state["pdf_path"] = DEFAULT_PDF_PATH
        
        progress(0.1, desc="Loading PDF")
        
        # Load PDF
        loader = PyPDFLoader(global_state["pdf_path"])
        pages = loader.load()
        status_updates.append(f"✅ Successfully loaded {len(pages)} pages from PDF")
        
        progress(0.2, desc="Splitting document")
        
        # Split documents
        status_updates.append("🔪 Splitting document into smaller chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        texts = text_splitter.split_documents(pages)
        status_updates.append(f"✅ Split into {len(texts)} text chunks")
        
        progress(0.4, desc="Creating vector database")
        
        # Create directory if it doesn't exist
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        
        # Vector DB setup
        status_updates.append(f"💾 Creating vector database at {PERSIST_DIRECTORY}...")
        status_updates.append("🤖 Using ChromaDB's default embedding function")
        
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
        
        progress(0.6, desc="Storing document chunks")
        
        # Create or get collection
        collection_name = "oil_market_research"
        if pdf_option == "Upload custom PDF" and pdf_file is not None:
            collection_name = f"custom_pdf_{int(time.time())}"
            
        collection = chroma_client.get_or_create_collection(name=collection_name)
        
        # Add documents to the collection
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        status_updates.append("✅ Vector database created successfully")
        
        progress(0.8, desc="Initializing LLM")
        
        # Initialize the LLM
        status_updates.append(f"🧠 Initializing Ollama LLM model: {llm_model}")
        llm = Ollama(
            model=llm_model,
            temperature=temperature
        )
        status_updates.append("✅ LLM initialized")
        
        # Save to global state
        global_state["collection"] = collection
        global_state["llm"] = llm
        global_state["initialized"] = True
        
        progress(1.0, desc="System ready")
        
        # Clean up the temp file if we created one
        if temp_file is not None and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        
        # Return combined status
        return "\n".join(status_updates), gr.update(interactive=True), gr.update(interactive=True), gr.update(visible=True)
    
    except FileNotFoundError as e:
        error_msg = f"❌ Error: PDF file not found at {global_state['pdf_path']}\nOriginal error: {str(e)}"
        return error_msg, gr.update(interactive=False), gr.update(interactive=False), gr.update(visible=False)
    except Exception as e:
        error_msg = f"❌ An error occurred: {str(e)}"
        return error_msg, gr.update(interactive=False), gr.update(interactive=False), gr.update(visible=False)

# Function to answer queries
def answer_query(question, progress=gr.Progress()):
    global global_state
    
    if not global_state["initialized"]:
        return "❌ System not initialized. Please process a document first."
    
    try:
        # Status update
        progress(0.2, desc="Retrieving documents")
        status = "🔍 Retrieving relevant documents..."
        
        # Query the collection
        results = global_state["collection"].query(
            query_texts=[question],
            n_results=global_state["n_results"]
        )
        
        retrieved_docs = results['documents'][0]
        
        # Construct context from retrieved documents
        context = "\n\n".join(retrieved_docs)
        
        progress(0.5, desc="Generating answer")
        status += "\n📝 Constructing answer using LLM..."
        
        # Format the prompt
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
        
        # Create a custom handler for streaming
        callback_handler = GradioCallbackHandler()
        
        # Generate the answer with streaming output
        llm_with_cb = Ollama(
            model=global_state["llm_model"],
            temperature=global_state["temperature"],
            callback_manager=CallbackManager([callback_handler])
        )
        
        progress(0.7, desc="Streaming response")
        
        # Generate response
        response = llm_with_cb.invoke(prompt)
        
        progress(1.0, desc="Complete")
        
        return callback_handler.text
    
    except Exception as e:
        return f"❌ An error occurred: {str(e)}"

# Define the Gradio interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", neutral_hue="zinc", text_size="lg")) as demo:
    gr.Markdown(
        """
        # 🛢️ Oil Market RAG Analysis System
        An interactive system to query research on oil markets using RAG (Retrieval Augmented Generation)
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # Configuration panel
            with gr.Group():
                gr.Markdown("## 📋 Configuration")
                
                # PDF selection
                pdf_option = gr.Radio(
                    ["Use default PDF", "Upload custom PDF"],
                    label="Choose PDF source",
                    value="Use default PDF"
                )
                
                pdf_file = gr.File(
                    label="Upload a PDF document",
                    file_types=[".pdf"],
                    visible=False
                )
                
                # Show/hide file uploader based on selection
                def update_file_visibility(choice):
                    return gr.update(visible=choice == "Upload custom PDF")
                
                pdf_option.change(
                    fn=update_file_visibility,
                    inputs=pdf_option,
                    outputs=pdf_file
                )
                
                # Model parameters
                gr.Markdown("### Model Parameters")
                llm_model = gr.Dropdown(
                    ["llama3", "mistral", "gemma"],
                    label="LLM Model",
                    value="llama3"
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.2,
                    step=0.1,
                    label="Temperature"
                )
                n_results = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=4,
                    step=1,
                    label="Number of chunks to retrieve"
                )
                
                # Process button
                process_button = gr.Button(
                    "Process Document & Initialize System",
                    variant="primary"
                )
            
            # Status display
            status_display = gr.Textbox(
                label="System Status",
                placeholder="System not initialized. Configure and process a document to start.",
                lines=10,
                interactive=False
            )
        
        with gr.Column(scale=2):
            # Query section
            query_container = gr.Group(visible=False)
            with query_container:
                gr.Markdown("## 🔍 Ask Questions About the Document")
                query_input = gr.Textbox(
                    label="Enter your question",
                    placeholder="What are the main factors affecting oil prices during the pandemic?",
                    lines=2
                )
                query_button = gr.Button("Submit Query", variant="primary")
                
                # Answer display
                answer_display = gr.Markdown(
                    label="Answer"
                )
    
    # Connect buttons to functions
    process_button.click(
        fn=process_document,
        inputs=[pdf_option, pdf_file, llm_model, temperature, n_results],
        outputs=[status_display, query_input, query_button, query_container]
    )
    
    query_button.click(
        fn=answer_query,
        inputs=query_input,
        outputs=answer_display
    )
    
    # Example queries
    gr.Examples(
        [
            ["What were the main factors affecting oil prices during the pandemic?"],
            ["How did the Russia-Ukraine war impact the oil market?"],
            ["What are the projections for future oil demand?"],
            ["What supply shocks affected the oil market?"]
        ],
        inputs=query_input
    )

# Launch the application
if __name__ == "__main__":
    print(colored("🚀 Starting the Oil Market RAG Analysis System with Gradio", "green"))
    demo.launch() 