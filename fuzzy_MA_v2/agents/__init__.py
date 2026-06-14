"""
agents  ─  Technical Specialty Agents (TSA) package
====================================================
Exports:
    BaseAgent, AgentProposal, TaskRequest   (contracts)
    StubEngineeringAgent, StubSupportAgent  (no-LLM testing)
    OllamaAgent, OllamaEngineeringAgent, OllamaSupportAgent  (real local LLMs)
"""

from .base import AgentProposal, BaseAgent, TaskRequest
from .stubs import StubEngineeringAgent, StubSupportAgent

__all__ = [
    "AgentProposal",
    "BaseAgent",
    "TaskRequest",
    "StubEngineeringAgent",
    "StubSupportAgent",
]

# Ollama agents require `requests`; import lazily so stub-only
# environments keep working without it.
try:
    from .ollama_agent import (  # noqa: F401
        OllamaAgent,
        OllamaEngineeringAgent,
        OllamaSupportAgent,
    )
    __all__ += ["OllamaAgent", "OllamaEngineeringAgent", "OllamaSupportAgent"]
except ImportError:  # pragma: no cover
    pass
