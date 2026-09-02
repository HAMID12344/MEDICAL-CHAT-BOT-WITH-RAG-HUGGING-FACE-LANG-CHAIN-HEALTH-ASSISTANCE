import os
import sys
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
    """Load lightweight local LLM pipeline for predicting medical answers."""
    from transformers import pipeline

    print("[INFO] Loading AI Prediction model (Qwen2.5-0.5B-Instruct)...")
    try:
        pipe = pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True
        )
        print("[SUCCESS] AI Prediction Model loaded!")
        return pipe
    except Exception as e:
        print(f"[WARNING] Could not load local LLM: {e}")
        return None


def generate_predicted_answer(pipe, context, question):
    """Generate answer from retrieved context."""
    if pipe is None:
        return "Prediction model not loaded. Refer to the matching resource documents below."

    prompt = (
        f"You are an AI medical assistant. Answer the medical question using the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    try:
        output = pipe(prompt)
        gen_text = output[0]["generated_text"]
        if "Answer:" in gen_text:
            ans = gen_text.split("Answer:")[-1].strip()
        else:
            ans = gen_text.replace(prompt, "").strip()
        return ans
    except Exception as e:
        return f"Error during generation: {e}"


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
    retriever = db.as_retriever(search_kwargs={"k": 3})

    print("\n" + "=" * 70)
    print("       MEDICAL AI ASSISTANT - READY FOR QUESTIONS")
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

            print("\nSearching medical encyclopedia & predicting answer...\n")

            # Retrieve matching documents from FAISS
            matched_docs = retriever.invoke(user_query)

            # Combine context
            combined_context = "\n\n".join([doc.page_content for doc in matched_docs])

            # Generate predicted answer
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