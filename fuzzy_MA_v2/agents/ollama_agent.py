"""
agents/ollama_agent.py  ─  Real local-LLM agents backed by Ollama
==================================================================
Implements the two Technical Specialty Agents (TSA) from the paper
(Section 3.1) on top of a local Ollama server:

  - OllamaEngineeringAgent (EA) : low-level engineering analysis
  - OllamaSupportAgent     (SA) : high-level semantic support

Response Uncertainty (IR)
-------------------------
The paper (Section 4.1) defines IR as the *average Shannon entropy
over the token probability distribution* of the local LLM output.
Ollama ≥ 0.12 exposes per-token log-probabilities through its
OpenAI-compatible endpoint (``/v1/chat/completions`` with
``logprobs=true`` + ``top_logprobs=k``), so IR is computed exactly:

    For each generated token t with top-k alternatives:
        p_i   = exp(logprob_i) renormalised over the top-k
        H_t   = -Σ p_i · log(p_i)            (Shannon entropy, nats)
        Ĥ_t   = H_t / log(k)                 (normalised to [0, 1])
    IR = mean over all generated tokens of Ĥ_t

If the server does not return logprobs (older Ollama), a conservative
fallback IR of 0.5 (maximum ignorance midpoint) is used and flagged
in the proposal metadata.
"""

from __future__ import annotations

import logging
import math
import time
from typing import List, Optional

import requests

from .base import AgentProposal, BaseAgent, TaskRequest

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"
_FALLBACK_IR = 0.5      # neutral midpoint when logprobs are unavailable


# ---------------------------------------------------------------------------
# IR — AVERAGE NORMALISED SHANNON ENTROPY  (paper Section 4.1, Eq. 14-16 input)
# ---------------------------------------------------------------------------
def shannon_entropy_ir(logprob_content: List[dict]) -> Optional[float]:
    """
    Compute IR ∈ [0, 1] from OpenAI-style ``logprobs.content`` entries.

    Each entry must carry ``top_logprobs``: a list of {token, logprob}.
    Returns None when no usable entries exist.
    """
    if not logprob_content:
        return None

    entropies: List[float] = []
    for entry in logprob_content:
        top = entry.get("top_logprobs") or []
        if len(top) < 2:
            continue
        # Renormalise the top-k probabilities
        probs = [math.exp(alt["logprob"]) for alt in top]
        total = sum(probs)
        if total <= 0.0:
            continue
        probs = [p / total for p in probs]
        h = -sum(p * math.log(p) for p in probs if p > 0.0)
        h_norm = h / math.log(len(probs))           # → [0, 1]
        entropies.append(h_norm)

    if not entropies:
        return None
    return min(1.0, max(0.0, sum(entropies) / len(entropies)))


# ---------------------------------------------------------------------------
# GENERIC OLLAMA AGENT
# ---------------------------------------------------------------------------
class OllamaAgent(BaseAgent):
    """
    A Technical Specialty Agent backed by a local Ollama model.

    Parameters
    ----------
    agent_id      : unique agent identifier
    model         : Ollama model tag (e.g. "llama3.2:latest")
    system_prompt : role specialisation prompt (EA vs SA behaviour)
    base_url      : Ollama server URL
    temperature   : sampling temperature (low = deterministic engineering)
    max_tokens    : response length cap (keeps benchmark latency bounded)
    top_logprobs  : k alternatives per token for the entropy computation
    timeout_s     : HTTP timeout for one inference call
    """

    def __init__(
        self,
        agent_id: str,
        model: str,
        system_prompt: str,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.3,
        max_tokens: int = 256,
        top_logprobs: int = 5,
        timeout_s: float = 240.0,
    ):
        super().__init__(agent_id)
        self.model = model
        self.system_prompt = system_prompt
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_logprobs = top_logprobs
        self.timeout_s = timeout_s

    # ------------------------------------------------------------------
    def handle(self, request: TaskRequest) -> AgentProposal:
        t0 = time.perf_counter()

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": request.content},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
        }

        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        text = (choice["message"]["content"] or "").strip()
        usage = data.get("usage", {})

        # ── IR from real token log-probabilities ──
        logprob_content = (choice.get("logprobs") or {}).get("content") or []
        ir = shannon_entropy_ir(logprob_content)
        ir_source = "shannon_entropy"
        if ir is None:
            ir = _FALLBACK_IR
            ir_source = "fallback_no_logprobs"
            logger.warning(
                "Agent %s: no logprobs from Ollama — IR fallback %.2f",
                self.agent_id, ir,
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        logger.debug(
            "OllamaAgent %s | model=%s | IR=%.3f (%s) | %.0f ms | %d tokens",
            self.agent_id, self.model, ir, ir_source, latency_ms,
            usage.get("completion_tokens", -1),
        )

        return self._make_proposal(
            request,
            response_text=text,
            response_uncertainty=round(ir, 4),
            latency_ms=latency_ms,
            metadata={
                "model": self.model,
                "ir_source": ir_source,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "stub": False,
            },
        )


# ---------------------------------------------------------------------------
# SPECIALISED TSA SUBCLASSES  (paper Section 3.1)
# ---------------------------------------------------------------------------
_EA_SYSTEM_PROMPT = (
    "Eres el Agente de Ingeniería (EA) de un sistema multi-agente de soporte "
    "técnico. Tu especialidad es el análisis de bajo nivel: código, logs de "
    "sistemas, bases de datos, consultas SQL, redes e infraestructura. "
    "Responde a la incidencia con un diagnóstico técnico preciso y los pasos "
    "concretos de resolución. Sé directo y conciso (máximo 6 líneas)."
)

_SA_SYSTEM_PROMPT = (
    "Eres el Agente de Soporte Técnico (SA) de un sistema multi-agente. "
    "Tu especialidad es la interpretación semántica de alto nivel: entender "
    "requerimientos vagos del usuario, gestionar flujos procedimentales y "
    "dar diagnóstico conceptual con guía paso a paso para el usuario final. "
    "Responde de forma clara y accionable (máximo 6 líneas)."
)


class OllamaEngineeringAgent(OllamaAgent):
    """EA — low-level engineering analysis (default model: llama3.2)."""

    def __init__(
        self,
        agent_id: str = "ollama_engineering_agent",
        model: str = "llama3.2:latest",
        **kwargs,
    ):
        super().__init__(
            agent_id=agent_id,
            model=model,
            system_prompt=_EA_SYSTEM_PROMPT,
            **kwargs,
        )


class OllamaSupportAgent(OllamaAgent):
    """SA — high-level semantic support (default model: gemma3:12b)."""

    def __init__(
        self,
        agent_id: str = "ollama_support_agent",
        model: str = "gemma3:12b",
        **kwargs,
    ):
        super().__init__(
            agent_id=agent_id,
            model=model,
            system_prompt=_SA_SYSTEM_PROMPT,
            **kwargs,
        )
