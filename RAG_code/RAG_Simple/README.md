# Oil Market RAG Analysis System

A Retrieval Augmented Generation (RAG) application that analyzes oil market research papers, available in multiple UI variants.

## Features

- RAG-based query answering using local LLMs via Ollama
- Support for both default and custom PDF uploads
- Configurable model parameters and retrieval settings
- Real-time streaming responses
- Available in multiple UI frameworks:
  - Command-line interface
  - Streamlit web interface
  - Gradio web interface

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd RAG_Simple
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure you have Ollama installed and running locally with models like llama3, mistral, or gemma.
   - Install from: https://ollama.com/
   - Pull models with: `ollama pull llama3`

## Usage

### Streamlit Version

```bash
streamlit run 1_rag_streamlit.py
```

The Streamlit interface features:
- Interactive web interface with dark mode UI
- Sidebar configuration panel
- Status indicators during processing

### Gradio Version

```bash
python 2_rag_gradio.py
```

The Gradio interface features:
- Modern UI with a responsive layout
- Example queries to get started quickly
- Progress indicators during processing

### Command Line Version

```bash
python 0_rag_simple.py
```

The command-line version is available if you prefer a non-GUI interface.

## Default Document

By default, the system uses a research paper titled "Pandemic, war, inflation: Oil market at a crossroads" (NBER #31496) by C. Baumeister.

## Requirements

- Python 3.8+
- Ollama (with compatible models)
- See requirements.txt for Python package dependencies 