import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "data/")
FAISS_PATH = os.getenv("DB_FAISS_PATH", "faiss_index")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")


def check_data_directory(directory_path):
    """Check if directory exists and has PDF files."""
    if not os.path.exists(directory_path):
        print(f"[ERROR] Directory '{directory_path}' not found. Please create it and add your PDF files.")
        sys.exit(1)

    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"[ERROR] No PDF files found in '{directory_path}'. Please upload/add at least one PDF.")
        sys.exit(1)

    print(f"[INFO] Found {len(pdf_files)} PDF file(s) in '{directory_path}': {', '.join(pdf_files)}")
    return pdf_files


def load_pdfs_from_directory(directory_path):
    """Load all PDF files from directory."""
    print("\n[STEP 1/5] Loading PDF documents...")
    loader = DirectoryLoader(
        directory_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f"[SUCCESS] Loaded {len(documents)} page(s) from PDF files.")
    return documents


def create_chunks(documents, chunk_size=500, chunk_overlap=50):
    """Split documents into smaller chunks for vector indexing."""
    print("\n[STEP 2/5] Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    print(f"[SUCCESS] Generated {len(chunks)} text chunks (chunk_size={chunk_size}, overlap={chunk_overlap}).")
    return chunks


def create_vector_store(chunks, embedding_model_name, output_path):
    """Create embeddings and store them in FAISS vector store."""
    print(f"\n[STEP 3/5] Loading embedding model: '{embedding_model_name}'...")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name
    )

    print("\n[STEP 4/5] Generating embeddings and building FAISS vector database...")
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    print(f"\n[STEP 5/5] Saving FAISS database locally to '{output_path}'...")
    os.makedirs(output_path, exist_ok=True)
    vectorstore.save_local(output_path)
    print(f"[SUCCESS] Vector database saved successfully to '{output_path}/'!")


def main():
    print("=" * 60)
    print("      MEDICAL AI CHATBOT - KNOWLEDGE BASE BUILDER")
    print("=" * 60)

    check_data_directory(DATA_PATH)
    documents = load_pdfs_from_directory(DATA_PATH)
    chunks = create_chunks(documents)
    create_vector_store(chunks, EMBEDDING_MODEL_NAME, FAISS_PATH)

    print("\n" + "=" * 60)
    print(" Knowledge Base build complete! You can now run connect_memory_llm.py to start querying.")
    print("=" * 60)


if __name__ == "__main__":
    main()
