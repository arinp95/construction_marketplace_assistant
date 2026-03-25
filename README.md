# Mini RAG — Construction Marketplace Assistant

A small **Retrieval-Augmented Generation (RAG)** demo: chunk internal documents, embed them locally, retrieve with **FAISS**, and answer with **Google Gemini** using **only** retrieved text. A **Streamlit** UI shows both the **retrieved chunks** and the **final answer** for transparency.

---

## 🚀 Live Demo

👉 https://construction-marketplace-assistant.streamlit.app/

---

## ❓ Why RAG?

Traditional LLMs rely on general knowledge and may generate **hallucinated or unsupported responses**.  
This system uses Retrieval-Augmented Generation (RAG) to ensure:

- Answers are grounded in internal documents  
- No unsupported claims are generated  
- Responses are explainable and verifiable  

---

## Models (free tier)

| Role | Choice | Why |
|------|--------|-----|
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (via **PyTorch** + `transformers`, same weights as the sentence-transformers package) | Runs locally, no API key, fast, good semantic search for short chunks. Avoids importing the full `sentence_transformers` stack (helps on broken TensorFlow installs). |
| **LLM** | `gemini-2.5-flash` (override with `GEMINI_MODEL` in `.env`) | Free quota on [Google AI Studio](https://aistudio.google.com/apikey); strong instruction-following for “answer only from context.” |

## How it works

1. **Document processing** — Files under `data/raw/` and `data/documents/` (`.txt`, `.md`, `.pdf`) are loaded. PDFs use `pypdf`; if a file is plain text with a `.pdf` extension, it still loads as UTF-8 (matching the provided assessment files).
2. **Chunking** — `chunking.py` merges paragraphs up to ~650 characters with ~90-character overlap so segments stay readable and contiguous.
3. **Embeddings & index** — Each chunk is embedded with MiniLM; vectors are **L2-normalized** and stored in **FAISS** `IndexFlatIP` so **inner product = cosine similarity**.
4. **Retrieval** — The query is embedded the same way; **top-k** chunks are returned by similarity score.
5. **Grounding** — `rag_answer.py` sends Gemini a **system instruction** and user prompt that require using **only** labeled `CONTEXT` blocks, and to reply with a fixed sentence if the context is insufficient—reducing unsupported claims (hallucinations are still possible in edge cases; the UI shows sources so users can verify).

---

## 🛡️ Grounding & Hallucination Control

The system enforces strict grounding using:

- A custom system prompt restricting responses to retrieved context  
- No external knowledge usage  
- A fallback response when information is unavailable  

> "The provided internal documents do not contain enough information to answer this question."

---

## Repository layout

```
├── app.py              # Streamlit chatbot (run this)
├── ingest.py           # Build vector_store from data/*
├── download_docs.py    # Optional: fetch assessment files
├── chunking.py
├── document_loader.py
├── vector_index.py     # FAISS logic
├── rag_answer.py       # Retrieval + generation
├── data/
│   ├── raw/
│   └── documents/
├── vector_store/       # FAISS index + metadata
├── eval_questions.md
├── requirements.txt
└── .env.example
```

---

## Quick start (local)

### 1. Create virtual environment

```bash
python -m venv .venv
```

Activate:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Add documents

- Run:

```bash
python download_docs.py
```

- OR manually place files in `data/raw/`

---

### 4. Build the index

```bash
python ingest.py
```

---

### 5. Configure Gemini

Create `.env` file:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
```

---

### 6. Run the app

```bash
streamlit run app.py
```

---

### Chat Interface
<img width="1226" height="911" alt="construction-marketplace-assistant streamlit app_" src="https://github.com/user-attachments/assets/60b4029a-b19d-493e-b6c2-3d418fea159e" />

---

## Example question

> What factors affect construction project delays?

The app shows:
- Retrieved chunks (with source + score)  
- A grounded answer based only on those chunks  

---

## Optional: evaluation

See **`eval_questions.md`** for sample queries. Evaluate based on:

- Retrieval relevance  
- Grounded answers  
- Absence of hallucinations  
- Clarity  

---

## Limitations

- Retrieval quality depends on chunking  
- Limited to provided documents  
- LLM may still generalize in edge cases  

---

## Future improvements

- Add reranking (cross-encoder)  
- Improve embeddings  
- Add inline citations  
- Scale backend (FastAPI)  

---

## License

Use and modify for learning and assessment purposes.
