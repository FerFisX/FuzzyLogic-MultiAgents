"""
agents/base.py  ─  Agent base class and shared data structures
==============================================================
Defines the contract every Technical Specialty Agent (TSA) must
satisfy so the Fuzzy Orchestrator Core can interact with them
uniformly.

Design notes
------------
- Agents are intentionally *synchronous* here.  The async message
  bus pattern described in the paper is layered on top by the FOC.
- `response_uncertainty` is the IR input to the FIE.  Real agents
  should derive this from token log-probabilities (average Shannon
  entropy) + a self-consistency check.  The stub implementation
  returns a configurable constant so the system is testable without
  a running LLM.
"""

from __future__ import annotations

import abc
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# SHARED DATA STRUCTURES
# ---------------------------------------------------------------------------
@dataclass
class AgentProposal:
    """
    The structured response every TSA returns to the FOC.

    Fields
    ------
    agent_id         : unique agent identifier
    task_id          : correlation UUID assigned by the FOC
    response_text    : the natural-language or structured answer
    response_uncertainty : IR ∈ [0, 1]  (Shannon entropy / self-consistency)
    latency_ms       : wall-clock inference time
    metadata         : arbitrary extra data (token counts, model name, etc.)
    """
    agent_id: str
    task_id: str
    response_text: str
    response_uncertainty: float          # IR ∈ [0.0, 1.0]
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.response_uncertainty <= 1.0):
            raise ValueError(
                f"response_uncertainty={self.response_uncertainty!r} must be in [0, 1]"
            )


@dataclass
class TaskRequest:
    """Incoming task envelope dispatched by the FOC."""
    content: str
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BASE AGENT
# ---------------------------------------------------------------------------
class BaseAgent(abc.ABC):
    """
    Abstract base for all Technical Specialty Agents.

    Subclass and implement:
        handle(request) -> AgentProposal
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    @abc.abstractmethod
    def handle(self, request: TaskRequest) -> AgentProposal:
        """Process a task and return a structured proposal."""

    def _make_proposal(
        self,
        request: TaskRequest,
        response_text: str,
        response_uncertainty: float,
        latency_ms: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> AgentProposal:
        return AgentProposal(
            agent_id=self.agent_id,
            task_id=request.task_id,
            response_text=response_text,
            response_uncertainty=response_uncertainty,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )
