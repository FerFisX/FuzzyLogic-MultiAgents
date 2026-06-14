"""
orchestrator/similarity.py  ─  Semantic similarity for consensus fusion
========================================================================
Implements Eq. (23) of the paper: cosine similarity between the
vector embeddings of two agent responses,

    s_jk = ( E(R_j) · E(R_k) ) / ( ‖E(R_j)‖ · ‖E(R_k)‖ )

The embedding function E(·) is served locally by Ollama with the
``nomic-embed-text`` model (zero marginal cost, no cloud round-trip).

A token-level Jaccard similarity is kept as a degradation fallback so
the orchestrator never fails when the embedding model is unavailable.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "nomic-embed-text:latest"

# A similarity function takes a list of N texts and returns an N×N
# matrix with zeros on the diagonal (self-similarity is excluded
# from consensus support, Eq. 24).
SimilarityFn = Callable[[List[str]], np.ndarray]


# ---------------------------------------------------------------------------
# JACCARD FALLBACK  (token-level approximation)
# ---------------------------------------------------------------------------
def jaccard_similarity_matrix(texts: List[str]) -> np.ndarray:
    """Pairwise Jaccard similarity over lower-cased word tokens."""
    n = len(texts)
    token_sets = [set(t.lower().split()) for t in texts]
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            union = token_sets[i] | token_sets[j]
            s = len(token_sets[i] & token_sets[j]) / len(union) if union else 0.0
            sim[i, j] = sim[j, i] = s
    return sim


# ---------------------------------------------------------------------------
# EMBEDDING-BASED COSINE SIMILARITY  (paper Eq. 23)
# ---------------------------------------------------------------------------
class OllamaEmbeddingSimilarity:
    """
    Computes E(R) with a local Ollama embedding model and returns the
    full cosine similarity matrix.  Falls back to Jaccard on any error.
    """

    def __init__(
        self,
        model: str = DEFAULT_EMBED_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 60.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    # ------------------------------------------------------------------
    def _embed(self, texts: List[str]) -> Optional[np.ndarray]:
        """Batch-embed texts via Ollama /api/embed. Returns (N, D) or None."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": texts},
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            embeddings = resp.json().get("embeddings")
            if not embeddings or len(embeddings) != len(texts):
                return None
            return np.asarray(embeddings, dtype=float)
        except Exception as exc:                     # noqa: BLE001
            logger.warning("Embedding call failed (%s) — Jaccard fallback", exc)
            return None

    # ------------------------------------------------------------------
    def __call__(self, texts: List[str]) -> np.ndarray:
        n = len(texts)
        if n < 2:
            return np.zeros((n, n))

        emb = self._embed(texts)
        if emb is None:
            return jaccard_similarity_matrix(texts)

        # Cosine similarity: normalise rows, then S = Ê · Êᵀ  (Eq. 23)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        unit = emb / norms
        sim = unit @ unit.T

        # Clamp numeric noise into [0, 1] and zero the diagonal (j ≠ i in Eq. 24)
        sim = np.clip(sim, 0.0, 1.0)
        np.fill_diagonal(sim, 0.0)
        return sim
