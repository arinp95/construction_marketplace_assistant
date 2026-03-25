"""Retrieve top-k chunks and generate a grounded answer with Google Gemini (free tier)."""

import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from local_embedder import EMBED_MODEL_NAME, LocalSentenceTransformer
from vector_index import index_exists, load_index, search

load_dotenv(Path(__file__).resolve().parent / ".env")

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

def _configure_gemini():
    import google.generativeai as genai
    # from google import genai

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key from "
            "https://aistudio.google.com/apikey"
        )
    genai.configure(api_key=key)
    return genai


SYSTEM_INSTRUCTION = """You are an assistant for a construction marketplace (e.g. Indecimal).
Rules:
- Use ONLY the CONTEXT blocks below to answer. Do not use general world knowledge.
- If the context does not contain the answer, reply exactly: "The provided internal documents do not contain enough information to answer this question."
- Do not invent policies, numbers, or features not stated in the context.
- Write clearly in complete sentences. If helpful, briefly mention which theme from the context you used (without claiming unseen sources).
"""


def build_user_prompt(question: str, contexts: list[tuple[str, str]]) -> str:
    """contexts: list of (source_label, chunk_text)"""
    parts = []
    for i, (src, text) in enumerate(contexts, start=1):
        parts.append(f"--- CONTEXT {i} (source: {src}) ---\n{text.strip()}\n")
    body = "\n".join(parts)
    return f"{body}\nQUESTION:\n{question.strip()}\n"


class MiniRAG:
    def __init__(self) -> None:
        self._embedder: LocalSentenceTransformer | None = None
        self._index = None
        self._records: list[dict] | None = None

    def ensure_index(self) -> None:
        if not index_exists():
            raise FileNotFoundError(
                "Vector store not found. Run: python ingest.py"
            )

    def _get_embedder(self) -> LocalSentenceTransformer:
        if self._embedder is None:
            self._embedder = LocalSentenceTransformer(EMBED_MODEL_NAME)
        return self._embedder

    def _get_index(self) -> tuple:
        if self._index is None:
            self._index, self._records = load_index()
        return self._index, self._records

    def retrieve(self, query: str, k: int = 5) -> list[tuple[dict, float]]:
        self.ensure_index()
        model = self._get_embedder()
        index, records = self._get_index()
        qvec = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        qvec = np.asarray(qvec, dtype=np.float32)
        return search(index, records, qvec[0], k)

    def answer(self, question: str, k: int = 5) -> dict:
        hits = self.retrieve(question, k=k)
        contexts: list[tuple[str, str]] = [
            (h[0]["source"], h[0]["text"]) for h in hits
        ]
        retrieved = [
            {
                "source": h[0]["source"],
                "chunk_index": h[0]["chunk_index"],
                "score": h[1],
                "text": h[0]["text"],
            }
            for h in hits
        ]

        if not contexts:
            return {
                "answer": "No document chunks were retrieved.",
                "retrieved": [],
            }

        genai = _configure_gemini()
        user_prompt = build_user_prompt(question, contexts)

        fallback_models = [
            GEMINI_MODEL,
            "gemini-2.0-flash",
            "gemini-pro-latest",
        ]
        text = ""
        last_err: Exception | None = None
        for name in fallback_models:
            try:
                model = genai.GenerativeModel(
                    name,
                    system_instruction=SYSTEM_INSTRUCTION,
                )
                response = model.generate_content(user_prompt)
                text = (response.text or "").strip()
                if text:
                    break
            except Exception as e:
                last_err = e
        if not text and last_err:
            raise last_err

        return {"answer": text, "retrieved": retrieved}
