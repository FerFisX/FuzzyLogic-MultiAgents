"""
orchestrator/complexity.py  ─  Semantic Complexity Estimator
=============================================================
Computes the Semantic Complexity (CS) input variable for the FIE
analytically from task text, without an LLM call.

Algorithm (Section 4.1 of the paper):
    CS = clip(
            w_kw  · keyword_density_score
          + w_dep · grammatical_depth_score
          + w_ctx · context_length_score,
          0, 100
         )

The three components are independently normalised to [0, 100] and
combined with configurable weights.  Default weights were calibrated
against the 150-task benchmark ground-truth labels.

This is intentionally heuristic — the paper acknowledges that more
sophisticated NLP-based complexity estimators are future work.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List

# ---------------------------------------------------------------------------
# TECHNICAL KEYWORD SETS  (domain-specific vocabulary triggers)
# ---------------------------------------------------------------------------
_KW_HIGH: List[str] = [
    # Database / transactions
    "deadlock", "race condition", "transaction", "rollback", "replication lag",
    "index corruption", "foreign key violation", "cascade delete",
    # Systems / infra
    "kernel panic", "oom killer", "segfault", "memory leak", "buffer overflow",
    "tcp retransmission", "ssl handshake", "certificate chain",
    # Multi-step reasoning
    "root cause", "dependency graph", "distributed trace", "audit trail",
    "idempotency", "eventual consistency", "two-phase commit",
]

_KW_MED: List[str] = [
    "error", "exception", "timeout", "latency", "throughput", "bottleneck",
    "query", "index", "cache", "deploy", "pipeline", "container", "service",
    "config", "authentication", "authorisation", "permission", "log",
]

_KW_LOW: List[str] = [
    "how", "what", "when", "list", "show", "get", "help",
    "status", "version", "name", "check",
]

# Default blending weights
_W_KW:  float = 0.50
_W_DEP: float = 0.25
_W_CTX: float = 0.25

# Context length normalisation reference (chars ≥ this = max score)
_MAX_CONTEXT_CHARS: int = 800


def _keyword_density_score(tokens: List[str]) -> float:
    """
    Score based on presence of high/medium technical vocabulary.
    Returns value in [0, 100].
    """
    if not tokens:
        return 0.0
    text_lower = " ".join(tokens).lower()
    high_hits  = sum(1 for kw in _KW_HIGH if kw in text_lower)
    med_hits   = sum(1 for kw in _KW_MED  if kw in text_lower)
    low_hits   = sum(1 for kw in _KW_LOW  if kw in text_lower)

    raw = (high_hits * 3.0 + med_hits * 1.0) / max(1, len(tokens) ** 0.5)
    # Normalise: empirically, raw ≈ 2.0 maps to CS=100 for high-complexity tasks
    return min(100.0, raw * 50.0)


def _grammatical_depth_score(text: str) -> float:
    """
    Proxy for parse-tree depth: count subordinate clause indicators
    and multi-clause connectors.  Returns value in [0, 100].
    """
    indicators = [
        "if", "when", "while", "because", "although", "however",
        "therefore", "which", "that", "whose", "wherein", "whereas",
        "provided that", "given that", "in order to", "such that",
    ]
    count = sum(text.lower().count(ind) for ind in indicators)
    return min(100.0, count * 15.0)


def _context_length_score(text: str) -> float:
    """
    Longer tasks generally require more contextual reasoning.
    Returns value in [0, 100] with a logarithmic curve.
    """
    chars = len(text)
    if chars == 0:
        return 0.0
    log_score = math.log1p(chars) / math.log1p(_MAX_CONTEXT_CHARS)
    return min(100.0, log_score * 100.0)


def compute_semantic_complexity(
    task_text: str,
    w_kw:  float = _W_KW,
    w_dep: float = _W_DEP,
    w_ctx: float = _W_CTX,
) -> float:
    """
    Estimate Semantic Complexity (CS) ∈ [0, 100] from raw task text.

    Parameters
    ----------
    task_text : str  The full text of the task/query.
    w_kw, w_dep, w_ctx : blending weights (must sum to 1.0).

    Returns
    -------
    float  CS score in [0, 100].
    """
    tokens = re.findall(r"\w+", task_text)
    kw_score  = _keyword_density_score(tokens)
    dep_score = _grammatical_depth_score(task_text)
    ctx_score = _context_length_score(task_text)

    cs = w_kw * kw_score + w_dep * dep_score + w_ctx * ctx_score
    return round(min(100.0, max(0.0, cs)), 2)
