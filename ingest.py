"""
Build the FAISS vector store from documents in data/raw and data/documents.

Run: python ingest.py
"""

from pathlib import Path

import numpy as np

from chunking import chunk_text
from local_embedder import EMBED_MODEL_NAME, LocalSentenceTransformer
from document_loader import iter_document_paths, load_document
from vector_index import build_index, save_index

ROOT = Path(__file__).resolve().parent
DATA_DIRS = [ROOT / "data" / "raw", ROOT / "data" / "documents"]

def main() -> None:
    paths = iter_document_paths(DATA_DIRS)
    if not paths:
        print("No documents found. Add .txt, .md, or .pdf files under data/raw or data/documents.")
        return

    all_records: list[dict] = []
    for path in paths:
        text = load_document(path)
        chunks = chunk_text(text, source_name=path.name)
        for c in chunks:
            all_records.append(
                {
                    "text": c["text"],
                    "source": c["source"],
                    "chunk_index": c["chunk_index"],
                }
            )

    if not all_records:
        print("No chunks produced.")
        return

    print(f"Embedding {len(all_records)} chunks with {EMBED_MODEL_NAME} ...")
    model = LocalSentenceTransformer()
    texts = [r["text"] for r in all_records]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    index = build_index(embeddings)
    save_index(index, all_records)
    print(f"Saved index and metadata to vector_store/ ({len(all_records)} vectors).")


if __name__ == "__main__":
    main()
