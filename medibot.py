import os
import sys
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# Page Configuration
st.set_page_config(
    page_title="MediAssist AI — Medical Intelligence Chatbot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment
load_dotenv()
FAISS_PATH = os.getenv("DB_FAISS_PATH", "faiss_index")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Custom CSS for Premium Modern Healthcare Aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
    }

    /* Main Container & Gradient Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(14, 116, 144, 0.15) 0%, rgba(59, 130, 246, 0.08) 50%, rgba(16, 185, 129, 0.05) 100%);
        border: 1px solid rgba(14, 116, 144, 0.25);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(8px);
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(90deg, #0284c7 0%, #0d9488 50%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.98rem;
        margin-bottom: 12px;
        line-height: 1.5;
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(14, 165, 233, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        margin-right: 8px;
        margin-bottom: 4px;
    }

    .badge-pulse {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10b981;
    }

    /* Source Citation Cards */
    .source-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(51, 65, 85, 0.6);
        border-left: 4px solid #0284c7;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 10px 0;
        transition: all 0.2s ease;
    }

    .source-card:hover {
        border-color: rgba(56, 189, 248, 0.6);
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.12);
    }

    .source-meta {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
    }

    .source-tag {
        font-size: 0.78rem;
        font-weight: 600;
        background: #0369a1;
        color: #f0f9ff;
        padding: 2px 8px;
        border-radius: 6px;
    }

    .page-tag {
        font-size: 0.78rem;
        font-weight: 600;
        color: #38bdf8;
        background: rgba(56, 189, 248, 0.1);
        padding: 2px 8px;
        border-radius: 6px;
        border: 1px solid rgba(56, 189, 248, 0.2);
    }

    .source-text {
        font-size: 0.88rem;
        color: #cbd5e1;
        line-height: 1.55;
        font-style: italic;
    }

    /* Disclaimer Callout */
    .disclaimer-box {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.25);
        border-radius: 10px;
        padding: 10px 14px;
        font-size: 0.8rem;
        color: #fbbf24;
        margin-top: 15px;
    }

    /* Prompt Suggestion Chips */
    .suggestion-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Connecting to medical knowledge base...")
def load_vector_database():
    """Cache and load FAISS vector database."""
    if not os.path.exists(FAISS_PATH):
        return None
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = FAISS.load_local(FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


@st.cache_resource(show_spinner="Loading AI model into memory...")
def load_llm_pipeline(model_name: str):
    """Cache and load LLM pipeline for selected model."""
    try:
        pipe = pipeline(
            "text-generation",
            model=model_name,
            device_map="auto"
        )
        return pipe
    except Exception as e:
        try:
            # Fallback without device_map
            pipe = pipeline(
                "text-generation",
                model=model_name
            )
            return pipe
        except Exception as e2:
            st.warning(f"Could not load {model_name}: {e2}")
            return None


def generate_medical_answer(pipe, context, question, temperature: float, max_tokens: int):
    """Generate concise medical answer based on selected hyperparameters."""
    if pipe is None:
        return "Please refer to the matching textbook resource documents below for details."

    prompt = (
        f"You are a certified clinical assistant. Answer the medical question directly and concisely based strictly on the provided textbook context.\n\n"
        f"Context:\n{context[:1000]}\n\n"
        f"Question: {question}\n\n"
        f"Clinical Summary:"
    )

    try:
        is_greedy = (temperature <= 0.05)
        output = pipe(
            prompt,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 0.01),
            do_sample=not is_greedy
        )
        gen_text = output[0]["generated_text"]
        if "Clinical Summary:" in gen_text:
            answer = gen_text.split("Clinical Summary:")[-1].strip()
        else:
            answer = gen_text.replace(prompt, "").strip()
        return answer if answer else "Refer to the verified medical textbook excerpts below."
    except Exception as e:
        return "Refer to the verified medical textbook excerpts below."


# 🌟 Sidebar: Model Selection, Temperature & Hyperparameters
with st.sidebar:
    st.markdown("### 🤖 LLM & Model Settings")
    
    # 1. Model Selector
    model_options = {
        "Qwen 2.5 (0.5B Instruct) — Fast Local": "Qwen/Qwen2.5-0.5B-Instruct",
        "TinyLlama (1.1B Chat) — Local": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "GPT-2 (Base Medical Fallback)": "gpt2"
    }
    
    selected_model_label = st.selectbox(
        "🧠 Select Language Model (LLM):",
        options=list(model_options.keys()),
        index=0,
        help="Choose the neural network model used for medical reasoning and text synthesis."
    )
    selected_model_name = model_options[selected_model_label]

    # 2. Temperature Slider
    temperature = st.slider(
        "🌡️ Temperature (Creativity vs Determinism):",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="0.0 = Highly factual, deterministic, strict. Higher values (0.7+) = More creative and descriptive responses."
    )

    # 3. Max Response Tokens Slider
    max_tokens = st.slider(
        "📏 Max Response Length (Tokens):",
        min_value=50,
        max_value=300,
        value=140,
        step=10,
        help="Controls the maximum length of generated medical answers."
    )

    st.markdown("---")
    st.markdown("### 📚 Search & Citations")
    top_k = st.slider(
        "Citations per question (k):",
        min_value=1,
        max_value=5,
        value=2,
        help="Number of textbook excerpts to retrieve from the medical encyclopedia."
    )

    st.markdown("---")
    db = load_vector_database()
    if db is not None:
        st.success("🟢 **Knowledge Base Online**")
        st.caption("Indexed: **Gale Encyclopedia of Medicine**")
    else:
        st.error("🔴 **Index Missing**")
        st.caption("Run `create_memory_llm.py` to index data.")

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("""
    <div class="disclaimer-box">
        ⚠️ <b>Clinical Disclaimer:</b><br>
        For educational & research purposes only. Always consult a licensed physician for diagnosis and medical decisions.
    </div>
    """, unsafe_allow_html=True)


# 🌟 Hero Header Section with Dynamic Parameters Display
st.markdown(f"""
<div class="hero-container">
    <div class="hero-title">
        <span>🩺</span> MediAssist AI
    </div>
    <div class="hero-subtitle">
        Clinical Reference & Question Answering System grounded in the <b>Gale Encyclopedia of Medicine</b>.
    </div>
    <div>
        <span class="badge-pill"><span class="badge-pulse"></span> RAG Online</span>
        <span class="badge-pill">🧠 {selected_model_name.split('/')[-1]}</span>
        <span class="badge-pill">🌡️ Temp: {temperature:.2f}</span>
        <span class="badge-pill">📏 Tokens: {max_tokens}</span>
        <span class="badge-pill">📚 k={top_k}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Quick Suggestion Chips (Pre-populate queries)
if len(st.session_state.messages) == 0:
    st.markdown("<div class='suggestion-title'>💡 Suggested Medical Queries:</div>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🩺 Asthma Symptoms", use_container_width=True):
            st.session_state.prefilled_query = "What are the common symptoms and triggers of asthma?"
    with col2:
        if st.button("🩸 Diabetes Causes", use_container_width=True):
            st.session_state.prefilled_query = "What causes diabetes and how is it classified?"
    with col3:
        if st.button("🫀 Hypertension Info", use_container_width=True):
            st.session_state.prefilled_query = "What is hypertension and how does it affect the body?"
    with col4:
        if st.button("💊 Allergy Treatment", use_container_width=True):
            st.session_state.prefilled_query = "What causes allergic reactions and how are they managed?"


# Display Previous Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🩺" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Verified Textbook Citations & Excerpts", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-meta">
                            <span class="source-tag">📄 {src['file']}</span>
                            <span class="page-tag">{src['page']}</span>
                        </div>
                        <div class="source-text">"{src['content']}"</div>
                    </div>
                    """, unsafe_allow_html=True)


# User Input Handling
prompt_input = st.chat_input("Ask any medical question...")

# Support suggestion chip clicks
if "prefilled_query" in st.session_state and st.session_state.prefilled_query:
    prompt_input = st.session_state.prefilled_query
    st.session_state.prefilled_query = None

if prompt_input:
    if db is None:
        st.error("Please build the vector database first using `create_memory_llm.py`.")
    else:
        # Display User Message
        st.chat_message("user", avatar="👤").markdown(prompt_input)
        st.session_state.messages.append({"role": "user", "content": prompt_input})

        # Process and Display Assistant Response
        with st.chat_message("assistant", avatar="🩺"):
            with st.spinner("Analyzing medical literature & formulating clinical answer..."):
                retriever = db.as_retriever(search_kwargs={"k": top_k})
                matched_docs = retriever.invoke(prompt_input)

                combined_context = "\n\n".join([doc.page_content for doc in matched_docs])
                pipe = load_llm_pipeline(selected_model_name)
                predicted_answer = generate_medical_answer(pipe, combined_context, prompt_input, temperature, max_tokens)

                # Render Answer
                st.markdown(f"**Medical Summary:**\n\n{predicted_answer}")

                # Format and Render Citations
                sources_data = []
                if matched_docs:
                    with st.expander("📚 Verified Textbook Citations & Excerpts", expanded=True):
                        for i, doc in enumerate(matched_docs, 1):
                            source_file = os.path.basename(doc.metadata.get("source", "The_GALE_ENCYCLOPEDIA_of_MEDICINE.pdf"))
                            raw_page = doc.metadata.get("page")
                            page_label = f"Page {raw_page + 1}" if isinstance(raw_page, int) else f"Page {raw_page}"
                            content_snippet = doc.page_content.strip()

                            sources_data.append({
                                "file": source_file,
                                "page": page_label,
                                "content": content_snippet
                            })

                            st.markdown(f"""
                            <div class="source-card">
                                <div class="source-meta">
                                    <span class="source-tag">📄 {source_file}</span>
                                    <span class="page-tag">{page_label}</span>
                                </div>
                                <div class="source-text">"{content_snippet}"</div>
                            </div>
                            """, unsafe_allow_html=True)

            # Store in conversation state
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"**Medical Summary:**\n\n{predicted_answer}",
                "sources": sources_data
            })