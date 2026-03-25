# Mini RAG — Construction Marketplace Assistant

A small **Retrieval-Augmented Generation (RAG)** demo: chunk internal documents, embed them locally, retrieve with **FAISS**, and answer with **Google Gemini** using **only** retrieved text. A **Streamlit** UI shows both the **retrieved chunks** and the **final answer** for transparency.

## Models (free tier)

| Role | Choice | Why |
|------|--------|-----|
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (via **PyTorch** + `transformers`, same weights as the sentence-transformers package) | Runs locally, no API key, fast, good semantic search for short chunks. Avoids importing the full `sentence_transformers` stack (helps on broken TensorFlow installs). |
| **LLM** | `gemini-2.0-flash` (override with `GEMINI_MODEL` in `.env`) | Free quota on [Google AI Studio](https://aistudio.google.com/apikey); strong instruction-following for “answer only from context.” |

## How it works

1. **Document processing** — Files under `data/raw/` and `data/documents/` (`.txt`, `.md`, `.pdf`) are loaded. PDFs use `pypdf`; if a file is plain text with a `.pdf` extension, it still loads as UTF-8 (matching the provided assessment files).
2. **Chunking** — `chunking.py` merges paragraphs up to ~650 characters with ~90-character overlap so segments stay readable and contiguous.
3. **Embeddings & index** — Each chunk is embedded with MiniLM; vectors are **L2-normalized** and stored in **FAISS** `IndexFlatIP` so **inner product = cosine similarity**.
4. **Retrieval** — The query is embedded the same way; **top-k** chunks are returned by similarity score.
5. **Grounding** — `rag_answer.py` sends Gemini a **system instruction** and user prompt that require using **only** labeled `CONTEXT` blocks, and to reply with a fixed sentence if the context is insufficient—reducing unsupported claims (hallucinations are still possible in edge cases; the UI shows sources so users can verify).

## Repository layout

```
├── app.py              # Streamlit chatbot (run this)
├── ingest.py           # Build vector_store from data/*
├── download_docs.py    # Optional: fetch assessment files from Google Drive
├── chunking.py
├── document_loader.py
├── vector_index.py     # FAISS save/load + search
├── rag_answer.py       # retrieve + Gemini generation
├── data/
│   ├── raw/            # Assessment docs (after download or manual copy)
│   └── documents/      # Add more internal docs here
├── vector_store/       # Created by ingest (gitignored; see Deploy note)
├── eval_questions.md   # Sample questions for manual QA
├── run_app.ps1         # Windows helper to launch Streamlit with `.venv`
├── notebooks/
│   └── mini_rag_overview.ipynb
├── requirements.txt
└── .env.example
```

## Quick start (local)

**1. Clone / open this folder** and create a virtual environment (recommended, especially if Anaconda has conflicting TensorFlow/Keras packages):

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Documents**

- Either run `python download_docs.py` to pull the three assessment files into `data/raw/`, or copy your own files into `data/raw/` or `data/documents/`.

**4. Build the index**

```bash
python ingest.py
```

**5. Configure Gemini**

Copy `.env.example` to `.env` and set:

```env
GEMINI_API_KEY=your_key_here
```

(Optional) `GEMINI_MODEL=gemini-1.5-flash` if you need another model name for your account.

**6. Launch the UI**

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

On Windows you can also run `.\run_app.ps1` from the project root (uses `.venv` automatically).

## Deploying the chatbot (free)

**Streamlit Community Cloud**

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), connect the repo, set **Main file** to `app.py`.
3. Under **Secrets**, add:

   ```toml
   GEMINI_API_KEY = "your_key_here"
   ```

4. Deploy. The first load may take a minute while dependencies and the embedding model download.

**Note:** FAISS indices in `vector_store/` are not in git by default. Options:

- **A)** Commit the `vector_store/` folder (remove `vector_store/` from `.gitignore` if you want a ready-to-run deploy), or  
- **B)** Run `python ingest.py` in a **build step** / CI job that commits artifacts, or  
- **C)** Add a one-time “Rebuild index” admin path (not included by default)—for class projects, committing `vector_store/` after `ingest.py` is the simplest.

## Example question

> What factors affect construction project delays?

The app will show **retrieved chunks** (with source file name and similarity score) and an **answer** grounded in those chunks only.

## Optional: evaluation (bonus ideas)

See **`eval_questions.md`** for 10 sample questions aligned with the assessment documents. Manually check: chunk relevance, unsupported claims vs. context, and clarity. Summarize findings in this README or a short `EVAL.md` if you extend the project.

The code tries **`gemini-2.0-flash`** first, then falls back to **`gemini-1.5-flash`** if your API region/account does not expose the first model.

## Limitations

- Retrieval quality depends on chunking and embedding model size; very long or technical PDFs may need larger chunks or a bigger embedder.
- The LLM can still occasionally overgeneralize; **always** inspect retrieved chunks for high-stakes use.
- Gemini availability and model names depend on Google’s API; update `GEMINI_MODEL` if the default is unavailable in your region.

## License

Use and modify for learning and assessment; ensure compliance with your document sources and Google’s API terms.
