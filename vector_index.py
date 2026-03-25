"""Build and query a FAISS inner-product index (cosine similarity on L2-normalized vectors)."""

import pickle
from pathlib import Path

import faiss
import numpy as np

STORE_DIR = Path(__file__).resolve().parent / "vector_store"
INDEX_PATH = STORE_DIR / "faiss.index"
META_PATH = STORE_DIR / "metadata.pkl"


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (vecs / norms).astype(np.float32)


def build_index(embeddings: np.ndarray) -> faiss.Index:
    embeddings = _l2_normalize(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, records: list[dict]) -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with META_PATH.open("wb") as f:
        pickle.dump(records, f)


def load_index() -> tuple[faiss.Index, list[dict]]:
    index = faiss.read_index(str(INDEX_PATH))
    with META_PATH.open("rb") as f:
        records = pickle.load(f)
    return index, records


def index_exists() -> bool:
    return INDEX_PATH.is_file() and META_PATH.is_file()


def search(
    index: faiss.Index,
    records: list[dict],
    query_embedding: np.ndarray,
    k: int,
) -> list[tuple[dict, float]]:
    q = _l2_normalize(query_embedding.reshape(1, -1))
    scores, indices = index.search(q, min(k, len(records)))
    hits: list[tuple[dict, float]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        hits.append((records[idx], float(score)))
    return hits
