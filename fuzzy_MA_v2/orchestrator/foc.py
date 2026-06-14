"""
orchestrator/foc.py  ─  Fuzzy Orchestrator Core (FOC)
======================================================
Central control entity implementing Algorithm 1 from the paper.

Responsibilities
----------------
1. Receive a task request and assign a UUID correlation ID.
2. Broadcast the task *concurrently* to all registered agents via
   ThreadPoolExecutor (simulating the async message bus pattern).
3. Compute Semantic Complexity (CS) from task text.
4. For each proposal received within the timeout window (τ), invoke
   the FIE with (CS, agent.IR, agent.FH) → (NC, IEN).
5. Apply weighted semantic fusion to resolve conflicts.
6. Execute the routing decision: local consensus or cloud escalation.
7. Update the Metric Auditor Agent (MA) with outcomes.

Paper reference: Section 3.2 (Task Lifecycle) + Algorithm 1.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from agents.base import BaseAgent, AgentProposal, TaskRequest
from audit.auditor import MetricAuditorAgent
from fie import evaluate, FIEResult
from orchestrator.complexity import compute_semantic_complexity
from orchestrator.similarity import SimilarityFn, jaccard_similarity_matrix

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIGURATION DEFAULTS  (tunable at FOC instantiation)
# ---------------------------------------------------------------------------
DEFAULT_TIMEOUT_S:  float = 10.0   # τ — max wait for agent responses
DEFAULT_THETA:      float = 0.30   # min weight for proposal inclusion in fusion
DEFAULT_GAMMA:      float = 0.25   # min max(W) before irresolvable conflict
ESCALATION_IEN_THR: float = 75.0  # IEN ≥ this → cloud escalation


# ---------------------------------------------------------------------------
# RESULT DATACLASSES
# ---------------------------------------------------------------------------
@dataclass
class ProposalEvaluation:
    """FIE evaluation result bound to a specific proposal."""
    proposal: AgentProposal
    fie_result: FIEResult
    cs: float                    # Semantic Complexity used

    @property
    def nc(self) -> float:
        return self.fie_result.nc

    @property
    def ien(self) -> float:
        return self.fie_result.ien


@dataclass
class OrchestratorResult:
    """Final output of one complete orchestration cycle."""
    task_id: str
    final_response: str
    escalated: bool
    escalation_reason: Optional[str]
    evaluations: List[ProposalEvaluation]
    weights: Dict[str, float]         # agent_id → normalised weight
    wall_ms: float
    # Convenience flag
    resolved_locally: bool = field(init=False)

    def __post_init__(self) -> None:
        self.resolved_locally = not self.escalated

    def summary(self) -> str:
        lines = [
            f"Task {self.task_id[:8]}…",
            f"  Escalated   : {self.escalated}"
            + (f" ({self.escalation_reason})" if self.escalation_reason else ""),
            f"  Wall time   : {self.wall_ms:.1f} ms",
        ]
        for ev in self.evaluations:
            lines.append(
                f"  Agent {ev.proposal.agent_id:<24} "
                f"NC={ev.nc:.3f}  IEN={ev.ien:.2f}  "
                f"W={self.weights.get(ev.proposal.agent_id, 0.0):.3f}"
            )
        lines.append(f"  Response    : {self.final_response[:100]}…")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# FUZZY ORCHESTRATOR CORE
# ---------------------------------------------------------------------------
class FuzzyOrchestratorCore:
    """
    Implements the complete FIE-driven orchestration pipeline.

    Parameters
    ----------
    agents          : list of BaseAgent instances
    auditor         : MetricAuditorAgent (shared state)
    cloud_handler   : callable(task_text) → str  (cloud escalation endpoint)
    timeout_s       : τ — maximum wait for agent responses
    theta           : minimum weight for fusion inclusion
    gamma           : minimum max(W) before declaring irresolvable conflict
    similarity_fn   : callable(texts) → N×N similarity matrix (Eq. 23).
                      Defaults to token Jaccard; pass
                      ``orchestrator.similarity.OllamaEmbeddingSimilarity()``
                      for true embedding cosine similarity.
    """

    def __init__(
        self,
        agents: List[BaseAgent],
        auditor: MetricAuditorAgent,
        cloud_handler: Optional[Callable[[str], str]] = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        theta: float = DEFAULT_THETA,
        gamma: float = DEFAULT_GAMMA,
        similarity_fn: Optional[SimilarityFn] = None,
        ien_threshold: float = ESCALATION_IEN_THR,
    ):
        self._agents = {a.agent_id: a for a in agents}
        self._auditor = auditor
        self._cloud_handler = cloud_handler or self._default_cloud_handler
        self._timeout_s = timeout_s
        self._theta = theta
        self._gamma = gamma
        self._similarity_fn = similarity_fn or jaccard_similarity_matrix
        self._ien_threshold = ien_threshold   # set to float("inf") for Pure-Local
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def process(self, task_content: str, task_id: Optional[str] = None) -> OrchestratorResult:
        """
        Full pipeline: broadcast → evaluate → fuse or escalate.

        Parameters
        ----------
        task_content : str  Raw task text.
        task_id      : optional UUID; generated if not provided.
        """
        if task_id is None:
            task_id = str(uuid.uuid4())

        t0 = time.perf_counter()
        request = TaskRequest(content=task_content, task_id=task_id)

        logger.info("FOC | task=%s | content='%s'", task_id[:8], task_content[:80])

        # ── Step 1: Compute CS once (task-level, not per-agent) ──
        cs = compute_semantic_complexity(task_content)
        logger.debug("FOC | CS=%.1f", cs)

        # ── Step 2: Broadcast asynchronously ──
        proposals = self._collect_proposals(request)

        if not proposals:
            return self._escalate(
                task_id, request, evaluations=[], weights={},
                reason="timeout — no agent responses received",
                t0=t0,
            )

        # ── Steps 3-4: FIE evaluation for each proposal ──
        evaluations = self._evaluate_proposals(proposals, cs)

        # ── Step 5a: Check IEN escalation threshold ──
        max_ien = max(ev.ien for ev in evaluations)
        if max_ien >= self._ien_threshold:
            reason = f"cognitive insufficiency (max IEN={max_ien:.2f} ≥ {self._ien_threshold})"
            return self._escalate(task_id, request, evaluations, weights={}, reason=reason, t0=t0)

        # ── Step 6: Weighted fusion ──
        weights = self._compute_weights(evaluations, self._similarity_fn)
        max_w = max(weights.values()) if weights else 0.0

        if max_w < self._gamma:
            reason = f"irresolvable conflict (max W={max_w:.3f} < γ={self._gamma})"
            return self._escalate(task_id, request, evaluations, weights, reason=reason, t0=t0)

        # ── Step 7: Fuse responses ──
        final_response = self._fuse(evaluations, weights, self._theta)
        wall_ms = (time.perf_counter() - t0) * 1000.0

        # ── Step 8: Audit ──  (lock: auditor is not thread-safe and the
        # benchmark drives several process() calls concurrently)
        with self._lock:
            for ev in evaluations:
                fh_new = self._auditor.record_outcome(
                    task_id=task_id,
                    agent_id=ev.proposal.agent_id,
                    cs=cs, ir=ev.proposal.response_uncertainty,
                    nc=ev.nc, ien=ev.ien,
                    escalated=False,
                    success=True,            # assume success; caller can update
                    latency_ms=ev.proposal.latency_ms,
                )
                logger.debug("Audit | agent=%s FH→%.1f", ev.proposal.agent_id, fh_new)

        logger.info(
            "FOC | task=%s RESOLVED LOCALLY | wall=%.1f ms | max_w=%.3f",
            task_id[:8], wall_ms, max_w,
        )

        return OrchestratorResult(
            task_id=task_id,
            final_response=final_response,
            escalated=False,
            escalation_reason=None,
            evaluations=evaluations,
            weights=weights,
            wall_ms=wall_ms,
        )

    # ------------------------------------------------------------------
    # INTERNAL STEPS
    # ------------------------------------------------------------------
    def _collect_proposals(self, request: TaskRequest) -> List[AgentProposal]:
        """
        Broadcast task to all agents concurrently and collect proposals
        that arrive within the timeout window τ.
        """
        proposals: List[AgentProposal] = []
        with ThreadPoolExecutor(max_workers=len(self._agents)) as executor:
            futures = {
                executor.submit(agent.handle, request): agent_id
                for agent_id, agent in self._agents.items()
            }
            for future in as_completed(futures, timeout=self._timeout_s):
                try:
                    proposal = future.result()
                    proposals.append(proposal)
                    logger.debug(
                        "FOC | received proposal from %s (IR=%.2f, %.0f ms)",
                        proposal.agent_id, proposal.response_uncertainty, proposal.latency_ms,
                    )
                except Exception as exc:
                    agent_id = futures[future]
                    logger.warning("Agent %s raised: %s", agent_id, exc)

        return proposals

    def _evaluate_proposals(
        self, proposals: List[AgentProposal], cs: float
    ) -> List[ProposalEvaluation]:
        """Run the FIE for each proposal."""
        evaluations = []
        for prop in proposals:
            fh = self._auditor.get_fh(prop.agent_id)
            result = evaluate(cs=cs, ir=prop.response_uncertainty, fh=fh)
            evaluations.append(ProposalEvaluation(proposal=prop, fie_result=result, cs=cs))
            logger.debug(
                "FIE  | agent=%-24s CS=%.1f IR=%.2f FH=%.1f → NC=%.3f IEN=%.2f [%.2f ms]",
                prop.agent_id, cs, prop.response_uncertainty, fh,
                result.nc, result.ien, result.eval_ms,
            )
        return evaluations

    @staticmethod
    def _compute_weights(
        evaluations: List[ProposalEvaluation],
        similarity_fn: SimilarityFn = jaccard_similarity_matrix,
    ) -> Dict[str, float]:
        """
        Compute normalised consensus weights W_i per Eqs. (23)–(25)
        of the paper.

        W_i = NC_i · (1 + SP_i) / Σ_k NC_k · (1 + SP_k)        (Eq. 25)

        where SP_i = Σ_{j≠i} s_ij · NC_j                        (Eq. 24)
        and s_ij is the semantic similarity matrix (Eq. 23) —
        cosine over embeddings when ``OllamaEmbeddingSimilarity``
        is injected, token Jaccard otherwise.
        """
        n = len(evaluations)
        if n == 0:
            return {}
        if n == 1:
            return {evaluations[0].proposal.agent_id: 1.0}

        texts = [ev.proposal.response_text for ev in evaluations]
        sim = similarity_fn(texts)            # N×N, zero diagonal (Eq. 23)

        nc_arr = np.array([ev.nc for ev in evaluations])

        # Consensus support SP_i = Σ_{j≠i} sim_ij · NC_j
        sp = sim @ nc_arr   # matrix mult gives Σ_j sim_ij · NC_j; diagonal is 0

        raw_w = nc_arr * (1.0 + sp)
        total = raw_w.sum()

        if total == 0.0:
            uniform = 1.0 / n
            return {ev.proposal.agent_id: uniform for ev in evaluations}

        normalised = raw_w / total
        return {ev.proposal.agent_id: float(w) for ev, w in zip(evaluations, normalised)}

    @staticmethod
    def _fuse(
        evaluations: List[ProposalEvaluation],
        weights: Dict[str, float],
        theta: float,
    ) -> str:
        """
        Syntactic fusion Eq. (8): G({R_i | W_i > θ}).

        For text responses, this selects the highest-weight proposal
        above the θ threshold as the primary response, then appends
        supplementary context from secondary proposals.

        A production implementation would use a proper sentence-level
        fusion operator; this approach preserves the paper's semantics.
        """
        eligible = [
            (ev, weights.get(ev.proposal.agent_id, 0.0))
            for ev in evaluations
            if weights.get(ev.proposal.agent_id, 0.0) > theta
        ]

        if not eligible:
            # Fall back to best overall
            best = max(evaluations, key=lambda ev: weights.get(ev.proposal.agent_id, 0.0))
            return best.proposal.response_text

        eligible.sort(key=lambda x: x[1], reverse=True)
        primary = eligible[0][0].proposal.response_text

        # Append unique supplementary sentences from secondary proposals
        supplementary = []
        for ev, w in eligible[1:]:
            note = f"[{ev.proposal.agent_id} | W={w:.2f}] {ev.proposal.response_text}"
            supplementary.append(note)

        if supplementary:
            return primary + "\n\n--- Additional perspectives ---\n" + "\n".join(supplementary)
        return primary

    def _escalate(
        self,
        task_id: str,
        request: TaskRequest,
        evaluations: List[ProposalEvaluation],
        weights: Dict[str, float],
        reason: str,
        t0: float,
    ) -> OrchestratorResult:
        """Route task to the cloud handler and audit the escalation."""
        logger.warning("FOC | ESCALATING task=%s | reason: %s", task_id[:8], reason)

        cloud_response = self._cloud_handler(request.content)
        wall_ms = (time.perf_counter() - t0) * 1000.0

        # Audit all agents involved (if any) as escalated
        with self._lock:
            for ev in evaluations:
                self._auditor.record_outcome(
                    task_id=task_id,
                    agent_id=ev.proposal.agent_id,
                    cs=ev.cs, ir=ev.proposal.response_uncertainty,
                    nc=ev.nc, ien=ev.ien,
                    escalated=True,
                    success=False,
                    latency_ms=ev.proposal.latency_ms,
                )

        return OrchestratorResult(
            task_id=task_id,
            final_response=cloud_response,
            escalated=True,
            escalation_reason=reason,
            evaluations=evaluations,
            weights=weights,
            wall_ms=wall_ms,
        )

    @staticmethod
    def _default_cloud_handler(task_text: str) -> str:
        """
        Placeholder cloud handler.
        In production: call AWS Bedrock / Anthropic API here.
        Replace this method or pass a real callable to __init__.
        """
        return (
            f"[CLOUD STUB] Task escalated for frontier-model processing.\n"
            f"Task: {task_text[:200]}"
        )
