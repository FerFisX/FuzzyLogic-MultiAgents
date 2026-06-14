"""
fie  ─  Fuzzy Inference Engine package
=======================================
Public API:
    evaluate(cs, ir, fh)          → FIEResult
    evaluar_propuesta(cs, ir, fh) → (nc, ien)   [v1 compatibility]
    fuzzify(cs, ir, fh)           → membership degrees dict
"""

from .engine import (
    FIEResult,
    IEN_ESCALATION_THRESHOLD,
    evaluate,
    evaluar_propuesta,
    fuzzify,
)

__all__ = [
    "FIEResult",
    "IEN_ESCALATION_THRESHOLD",
    "evaluate",
    "evaluar_propuesta",
    "fuzzify",
]
