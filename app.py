"""
🚀 Construction Marketplace RAG Chatbot — Enhanced Dark UI
Run: streamlit run app.py
"""

import html
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_answer import MiniRAG

# -------------------- CONFIG --------------------
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

st.set_page_config(
    page_title="Construction RAG Assistant",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- CUSTOM STYLES --------------------
def inject_styles():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #0E1117;
            color: #E6EDF3;
        }

        .main-header {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(90deg, #00E0FF, #7C3AED);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .sub-header {
            color: #9DA7B3;
            margin-bottom: 1.5rem;
        }

        .answer-box {
            background: linear-gradient(135deg, #1A1F2B, #111827);
            border-left: 4px solid #00E0FF;
            padding: 1rem 1.2rem;
            border-radius: 12px;
            margin: 1rem 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }

        .context-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            margin: 0.5rem 0;
            backdrop-filter: blur(8px);
        }

        .meta-badge {
            display: inline-block;
            background: linear-gradient(90deg, #00E0FF, #7C3AED);
            color: black;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 0.7rem;
            margin-bottom: 0.4rem;
            font-weight: 600;
        }

        .stButton button {
            background: linear-gradient(90deg, #00E0FF, #7C3AED);
            color: black;
            border-radius: 8px;
            font-weight: 600;
        }

    </style>
    """, unsafe_allow_html=True)


# -------------------- LOAD RAG --------------------
@st.cache_resource
def get_rag():
    return MiniRAG()


def apply_secrets():
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            os.environ["GEMINI_API_KEY"] = key
    except Exception:
        pass


# -------------------- MAIN --------------------
def main():
    apply_secrets()
    inject_styles()

    # Header
    st.markdown('<p class="main-header">🏗️ Construction RAG Assistant</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">⚡ AI-powered answers grounded in internal construction documents</p>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.subheader("⚙️ Settings")
        top_k = st.slider("Chunks to retrieve", 1, 10, 5)

        st.markdown("---")
        st.markdown("### 🧠 Tech Stack")
        st.markdown("""
        - MiniLM Embeddings  
        - FAISS Vector Search  
        - Gemini LLM  
        """)

        if st.button("🧹 Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # Chat memory
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.write(msg["content"])
            else:
                safe_ans = html.escape(msg["answer"])
                st.markdown(
                    f'<div class="answer-box"><pre style="white-space:pre-wrap;margin:0">{safe_ans}</pre></div>',
                    unsafe_allow_html=True,
                )

                with st.expander("📄 Retrieved Context"):
                    for i, ch in enumerate(msg.get("retrieved", []), 1):
                        st.markdown(
                            f'<div class="context-card">'
                            f'<span class="meta-badge">#{i} · {ch["source"]} · score {ch["score"]:.3f}</span>'
                            f"<pre>{html.escape(ch['text'])}</pre>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

    # Load RAG
    rag = get_rag()

    try:
        rag.ensure_index()
    except FileNotFoundError:
        st.error("⚠️ Vector store not found. Run `python ingest.py`")
        return

    # Input
    if prompt := st.chat_input("Ask about construction policies, delays, pricing..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                try:
                    result = rag.answer(prompt, k=top_k)
                except Exception as e:
                    st.error(f"Error: {e}")
                    return

            answer = result["answer"]
            retrieved = result["retrieved"]

            st.markdown(
                f'<div class="answer-box"><pre>{html.escape(answer)}</pre></div>',
                unsafe_allow_html=True,
            )

            with st.expander("📄 Retrieved Context", expanded=True):
                for i, ch in enumerate(retrieved, 1):
                    st.markdown(
                        f'<div class="context-card">'
                        f'<span class="meta-badge">#{i} · {ch["source"]} · score {ch["score"]:.3f}</span>'
                        f"<pre>{html.escape(ch['text'])}</pre>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        st.session_state.messages.append({
            "role": "assistant",
            "answer": answer,
            "retrieved": retrieved,
        })


if __name__ == "__main__":
    main()