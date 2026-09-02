import os
import sys
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# Page Configuration
st.set_page_config(
    page_title="AI Medical Chatbot",
    page_icon="🩺",
    layout="wide"
)

# Load environment
load_dotenv()
FAISS_PATH = os.getenv("DB_FAISS_PATH", "faiss_index")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Loading medical database index...")
def load_vector_database():
    """Cache and load FAISS vector database."""
    if not os.path.exists(FAISS_PATH):
        return None
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = FAISS.load_local(FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


@st.cache_resource(show_spinner="Loading AI model (cached after first run)...")
def load_llm_pipeline():
    """Cache and load fast local text generation pipeline."""
    try:
        pipe = pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            max_new_tokens=120,      # Optimized for fast CPU response (< 3-5 seconds)
            temperature=0.2,
            do_sample=False           # Greedy decoding is 2x faster on CPU
        )
        return pipe
    except Exception as e:
        return None


def generate_medical_answer(pipe, context, question):
    """Generate concise medical answer quickly."""
    if pipe is None:
        return "Please refer to the matching resource documents below for the answer."

    # Keep prompt compact for faster processing
    prompt = (
        f"Answer the question concisely based only on this medical context:\n"
        f"{context[:1000]}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )

    try:
        output = pipe(prompt)
        gen_text = output[0]["generated_text"]
        if "Answer:" in gen_text:
            answer = gen_text.split("Answer:")[-1].strip()
        else:
            answer = gen_text.replace(prompt, "").strip()
        return answer if answer else "See matching resource excerpts below."
    except Exception as e:
        return f"See matching resource excerpts below."


# UI Header
st.title("🩺 AI Medical Chatbot & Reference Assistant")
st.caption("⚡ Fast query engine grounded in your medical encyclopedia PDF with exact page citations.")

# Sidebar Information
with st.sidebar:
    st.header("📚 Medical Knowledge Base")
    db = load_vector_database()
    if db is not None:
        st.success("✅ FAISS Vector Index Loaded")
        st.info("Source: Gale Encyclopedia of Medicine")
    else:
        st.error("❌ FAISS Vector Index not found. Please run `create_memory_llm.py` first.")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Number of source references to retrieve (k):", min_value=1, max_value=4, value=2)

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📖 View Matching Resource Documents"):
                for src in msg["sources"]:
                    st.markdown(f"**File:** `{src['file']}` | **{src['page']}**")
                    st.markdown(f"> {src['content']}")
                    st.markdown("---")

# User Input Box
user_query = st.chat_input("Ask a medical question (e.g., What are the symptoms of asthma?)...")

if user_query:
    if db is None:
        st.error("Please build the vector database first using `create_memory_llm.py`.")
    else:
        # Show user message
        st.chat_message("user").markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Process retrieval and answer
        with st.chat_message("assistant"):
            with st.spinner("⚡ Retrieving medical facts & generating answer..."):
                retriever = db.as_retriever(search_kwargs={"k": top_k})
                matched_docs = retriever.invoke(user_query)

                # Prepare context
                combined_context = "\n\n".join([doc.page_content for doc in matched_docs])
                pipe = load_llm_pipeline()
                predicted_answer = generate_medical_answer(pipe, combined_context, user_query)

                # Display predicted answer
                st.markdown(f"### 💡 Predicted Medical Answer\n{predicted_answer}")

                # Format source references
                sources_data = []
                if matched_docs:
                    with st.expander("📖 View Matching Resource Documents (PDF Sources & Pages)", expanded=True):
                        for i, doc in enumerate(matched_docs, 1):
                            source_file = os.path.basename(doc.metadata.get("source", "Medical Reference PDF"))
                            raw_page = doc.metadata.get("page")
                            page_label = f"Page {raw_page + 1}" if isinstance(raw_page, int) else f"Page {raw_page}"
                            content_snippet = doc.page_content.strip()

                            sources_data.append({
                                "file": source_file,
                                "page": page_label,
                                "content": content_snippet
                            })

                            st.markdown(f"**[Resource #{i}]** `{source_file}` — **{page_label}**")
                            st.info(content_snippet)

            # Save assistant message in session
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"### 💡 Predicted Medical Answer\n{predicted_answer}",
                "sources": sources_data
            })