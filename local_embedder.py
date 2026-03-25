"""Local embeddings with the same MiniLM weights as sentence-transformers (torch + transformers only)."""

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = getattr(model_output, "last_hidden_state", model_output[0])
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class LocalSentenceTransformer:
    """Drop-in style encoder: encode(list[str]) -> ndarray float32."""

    def __init__(self, model_name: str = EMBED_MODEL_NAME, cache_dir: Path | None = None) -> None:
        kwargs = {}
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self.model = AutoModel.from_pretrained(model_name, **kwargs)
        self.model.eval()

    @torch.inference_mode()
    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = False,
    ) -> np.ndarray | torch.Tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        all_emb: list[torch.Tensor] = []
        n = len(texts)
        for start in range(0, n, batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=256,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            out = self.model(**encoded)
            emb = _mean_pooling(out, encoded["attention_mask"])
            if normalize_embeddings:
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_emb.append(emb.cpu())
            if show_progress_bar:
                print(f"  embedded {min(start + batch_size, n)}/{n}")

        result = torch.cat(all_emb, dim=0)
        if convert_to_numpy:
            return result.numpy().astype(np.float32)
        return result
