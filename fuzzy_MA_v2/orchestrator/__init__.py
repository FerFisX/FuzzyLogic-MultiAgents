"""
orchestrator  ─  Fuzzy Orchestrator Core (FOC) package
=======================================================
Exports:
    FuzzyOrchestratorCore, OrchestratorResult
    compute_semantic_complexity (CS estimator)
"""

from .complexity import compute_semantic_complexity
from .foc import FuzzyOrchestratorCore, OrchestratorResult, ProposalEvaluation
from .similarity import OllamaEmbeddingSimilarity, jaccard_similarity_matrix

__all__ = [
    "FuzzyOrchestratorCore",
    "OrchestratorResult",
    "ProposalEvaluation",
    "compute_semantic_complexity",
    "OllamaEmbeddingSimilarity",
    "jaccard_similarity_matrix",
]
