import os
import sys
import torch
from dotenv import load_dotenv

# Ensure stdout and stderr support UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Optimize CPU threads for fast inference
try:
    num_cores = os.cpu_count() or 4
    torch.set_num_threads(num_cores)
except Exception:
    pass

# Step 1: Load environment variables
load_dotenv()


def load_vector_store(faiss_path, embedding_model_name):
    """Load FAISS vector database with error checks."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    if not os.path.exists(faiss_path):
        print(f"\n[ERROR] FAISS vector index not found at '{faiss_path}'.")
        print("Please run `python create_memory_llm.py` first to index your medical PDF.")
        sys.exit(1)

    print(f"\n[INFO] Loading embedding model '{embedding_model_name}'...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name
    )

    print(f"[INFO] Loading FAISS index from '{faiss_path}'...")
    db = FAISS.load_local(
        faiss_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    print("[SUCCESS] FAISS database loaded successfully!")
    return db


def get_llm_pipeline():
    """Load lightweight local LLM pipeline optimized for fast 2-5s CPU response."""
    from transformers import pipeline

    print("[INFO] Loading fast AI Prediction model (Qwen2.5-0.5B-Instruct)...")
    try:
        pipe = pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            max_new_tokens=100,      # Fast concise responses
            temperature=0.2,
            do_sample=False           # Greedy decoding = fastest CPU generation (< 3-5 seconds)
        )
        print("[SUCCESS] Fast AI Prediction Model loaded!")
        return pipe
    except Exception as e:
        print(f"[WARNING] Could not load local LLM: {e}")
        return None


def generate_predicted_answer(pipe, context, question):
    """Generate concise medical answer rapidly."""
    if pipe is None:
        return "Prediction model not loaded. Refer to the matching resource documents below."

    prompt = (
        f"Answer the medical question concisely based only on this context:\n"
        f"{context[:800]}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )

    try:
        output = pipe(prompt)
        gen_text = output[0]["generated_text"]
        if "Answer:" in gen_text:
            ans = gen_text.split("Answer:")[-1].strip()
        else:
            ans = gen_text.replace(prompt, "").strip()
        return ans if ans else "See matching resource excerpts below."
    except Exception as e:
        return "See matching resource excerpts below."


def display_sources_and_answer(answer, docs):
    """Format and display both the predicted answer and the resource documents."""
    print("\n" + "=" * 70)
    print(" [PREDICTED MEDICAL ANSWER / RESULT]")
    print("=" * 70)
    print(answer)
    print("=" * 70)

    if docs:
        print("\n" + "-" * 70)
        print(f" [RESOURCE DOCUMENTS & CITATIONS] ({len(docs)} Matched)")
        print("-" * 70)

        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "PDF Document")
            raw_page = doc.metadata.get("page")
            page_str = f"Page {raw_page + 1}" if isinstance(raw_page, int) else f"Page {raw_page}"

            print(f"\n[Resource #{i}] File: {os.path.basename(source)} | {page_str}")
            print("Content Excerpt:")
            print(doc.page_content.strip())
            print("-" * 50)


def start_chat_loop(db, pipe):
    """Start interactive search loop."""
    retriever = db.as_retriever(search_kwargs={"k": 2})

    print("\n" + "=" * 70)
    print("       MEDICAL AI ASSISTANT - READY FOR QUESTIONS (FAST MODE)")
    print("=" * 70)
    print("Type your medical query below.")
    print("Type 'exit', 'quit', or 'q' to end the session.\n")

    while True:
        try:
            user_query = input("\n[Question] Enter your medical query: ").strip()

            if not user_query:
                continue

            if user_query.lower() in ["exit", "quit", "q", "bye"]:
                print("\nThank you for using Medical AI Chatbot. Stay healthy!\n")
                break

            print("\n⚡ Searching medical encyclopedia & predicting answer in 2-5 seconds...\n")

            # Retrieve matching documents from FAISS
            matched_docs = retriever.invoke(user_query)

            # Combine context
            combined_context = "\n\n".join([doc.page_content for doc in matched_docs])

            # Generate predicted answer quickly
            answer = generate_predicted_answer(pipe, combined_context, user_query)

            # Display both result and source documents
            display_sources_and_answer(answer, matched_docs)

        except KeyboardInterrupt:
            print("\n\nSession terminated by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n[Error]: {e}")


def main():
    print("=" * 70)
    print("          AI MEDICAL CHATBOT & KNOWLEDGE RETRIEVER")
    print("=" * 70)

    faiss_path = os.getenv("DB_FAISS_PATH", "faiss_index")
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

    db = load_vector_store(faiss_path, embedding_model_name)
    pipe = get_llm_pipeline()

    start_chat_loop(db, pipe)


if __name__ == "__main__":
    main()