# 🩺 AI Medical Chatbot & Reference Assistant

An AI-powered medical reference assistant built with LangChain, FAISS vector embeddings, Hugging Face / Transformers, and Streamlit. It searches indexed medical encyclopedia PDFs and answers questions with grounded answers, resource document citations, and exact page numbers.

---

## 🌟 Features
- **PDF Knowledge Base Ingestion**: Automatically parses medical PDFs, extracts text, chunks documents, and creates FAISS vector embeddings.
- **Predicted AI Answers**: Generates answers grounded in reference literature.
- **Source Citations**: Displays source document names, exact PDF page numbers, and matching text excerpts.
- **Dual Interface**:
  - 🌐 **Web UI (`medibot.py`)**: Built with Streamlit with chat history and collapsible source drawers.
  - 💻 **Terminal CLI (`connect_memory_llm.py`)**: Fast interactive command-line session.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ai-medical-chatbot.git
cd ai-medical-chatbot
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and configure your settings:
```env
HF_TOKEN=your_huggingface_token_here
HUGGINGFACE_REPO_ID=mistralai/Mistral-7B-Instruct-v0.3
DB_FAISS_PATH=faiss_index
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

---

## 📖 Usage

### Step 1: Ingest Medical PDFs
Place your medical PDFs inside the `data/` folder, then run:
```bash
python create_memory_llm.py
```

### Step 2: Run the Web App (Streamlit)
```bash
streamlit run medibot.py
```

### Step 3: Run the Terminal Chatbot
```bash
python connect_memory_llm.py
```

---

## 📁 Project Structure
```text
ai-medical-chatbot/
│
├── data/                    # Folder containing raw medical PDFs
├── faiss_index/             # Local FAISS vector index (generated)
├── create_memory_llm.py     # PDF chunking and vector embedding builder
├── connect_memory_llm.py     # Interactive terminal chatbot
├── medibot.py               # Streamlit web application
├── .env.example             # Example environment variable template
├── .gitignore               # Ignored files for Git/GitHub
├── requirements.txt         # Project Python dependencies
└── README.md                # Project documentation
```
