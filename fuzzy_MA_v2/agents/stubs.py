"""
agents/stubs.py  ─  Stub/mock agents for testing
==================================================
These stubs let you run the full orchestrator pipeline — including
FIE evaluation and consensus — without needing Ollama or cloud
credentials.

Two stub agents ship here:
  - StubEngineeringAgent   (EA — simulates Llama-3.2-7B behaviour)
  - StubSupportAgent       (SA — simulates Mistral-7B behaviour)

Each stub accepts an ``uncertainty_override`` parameter so unit
tests can drive specific FIE scenarios deterministically.
"""

from __future__ import annotations

import time

from .base import BaseAgent, AgentProposal, TaskRequest

# ---------------------------------------------------------------------------
# ENGINEERING AGENT STUB
# ---------------------------------------------------------------------------
class StubEngineeringAgent(BaseAgent):
    """
    Simulates the Engineering Agent (EA).

    In production this would call an Ollama Llama-3.2-7B-Instruct
    endpoint and parse token log-probabilities to compute IR.
    """

    def __init__(
        self,
        agent_id: str = "engineering_agent",
        uncertainty_override: float = 0.2,
        latency_ms: float = 350.0,
    ):
        super().__init__(agent_id)
        self._uncertainty = uncertainty_override
        self._latency_ms = latency_ms

    def handle(self, request: TaskRequest) -> AgentProposal:
        t0 = time.perf_counter()
        # Simulate inference latency
        time.sleep(self._latency_ms / 1000.0)

        response = (
            f"[Engineering] Analysed task '{request.content[:60]}...'. "
            "Recommended action: review system logs, apply patch v2.3.1, "
            "validate DB integrity via checksum comparison."
        )
        actual_latency = (time.perf_counter() - t0) * 1000.0

        return self._make_proposal(
            request,
            response_text=response,
            response_uncertainty=self._uncertainty,
            latency_ms=actual_latency,
            metadata={"model": "llama-3.2-7b-instruct", "stub": True},
        )


# ---------------------------------------------------------------------------
# SUPPORT AGENT STUB
# ---------------------------------------------------------------------------
class StubSupportAgent(BaseAgent):
    """
    Simulates the Technical Support Agent (SA).

    In production this would call an Ollama Mistral-7B-Instruct
    endpoint.
    """

    def __init__(
        self,
        agent_id: str = "support_agent",
        uncertainty_override: float = 0.35,
        latency_ms: float = 280.0,
    ):
        super().__init__(agent_id)
        self._uncertainty = uncertainty_override
        self._latency_ms = latency_ms

    def handle(self, request: TaskRequest) -> AgentProposal:
        t0 = time.perf_counter()
        time.sleep(self._latency_ms / 1000.0)

        response = (
            f"[Support] Assessed task '{request.content[:60]}...'. "
            "User guidance: restart the affected service, escalate to L2 "
            "if the issue recurs within 2 hours."
        )
        actual_latency = (time.perf_counter() - t0) * 1000.0

        return self._make_proposal(
            request,
            response_text=response,
            response_uncertainty=self._uncertainty,
            latency_ms=actual_latency,
            metadata={"model": "mistral-7b-instruct", "stub": True},
        )
