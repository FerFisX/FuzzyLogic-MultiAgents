"""
cloud/bedrock.py  ─  AWS Bedrock cloud escalation handler
==========================================================
Implements the "Enrutamiento Adaptativo y Escalado" step of the paper
(Section 3.2, step 5): when the FIE determines that local confidence
is insufficient (IEN ≥ 75) or the consensus conflict is irresolvable,
the task is dispatched to a frontier model on AWS Bedrock.

Features
--------
- Uses the Bedrock ``converse`` API (model-agnostic message format).
- Model fallback chain: tries each candidate until one responds
  (handles missing on-demand access / inference-profile requirements).
- Cost tracking: accumulates input/output tokens × published prices,
  feeding the "Costo Financiero por 1k Tareas" KPI of Table 2.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

DEFAULT_REGION = "us-east-1"

# Candidate models, tried in order.  Newer Anthropic models on Bedrock
# require cross-region inference profiles (the "us." prefix).
DEFAULT_MODEL_CHAIN: List[str] = [
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "anthropic.claude-3-5-haiku-20241022-v1:0",
    "us.amazon.nova-pro-v1:0",
    "amazon.nova-pro-v1:0",
]

# Published on-demand prices (USD per 1M tokens) — us-east-1.
# Used only for the financial KPI; update if AWS pricing changes.
_PRICES_PER_MTOK = {
    "claude-haiku-4-5":  (1.00,  5.00),
    "claude-3-5-haiku":  (0.80,  4.00),
    "claude-sonnet-4":   (3.00, 15.00),
    "nova-pro":          (0.80,  3.20),
    "nova-lite":         (0.06,  0.24),
}
_DEFAULT_PRICE = (3.00, 15.00)   # conservative fallback


def _price_for(model_id: str) -> tuple:
    for key, price in _PRICES_PER_MTOK.items():
        if key in model_id:
            return price
    return _DEFAULT_PRICE


_SYSTEM_PROMPT = (
    "Eres el modelo fundacional de escalamiento en la nube de un sistema "
    "multi-agente de soporte técnico de ingeniería. Las tareas que recibes "
    "fueron clasificadas como de alta complejidad o conflicto irresoluble "
    "por el orquestador difuso local. Entrega un diagnóstico experto y un "
    "plan de resolución preciso y accionable. Sé conciso (máximo 8 líneas)."
)


@dataclass
class CloudCallRecord:
    """Audit record for one Bedrock invocation."""
    model_id: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    timestamp: float = field(default_factory=time.time)


class BedrockCloudHandler:
    """
    Callable cloud handler for ``FuzzyOrchestratorCore``.

        handler = BedrockCloudHandler()
        foc = FuzzyOrchestratorCore(..., cloud_handler=handler)

    The instance is thread-safe and accumulates cost/latency stats.
    """

    def __init__(
        self,
        model_chain: Optional[List[str]] = None,
        region: str = DEFAULT_REGION,
        max_tokens: int = 500,
        temperature: float = 0.2,
        system_prompt: str = _SYSTEM_PROMPT,
        max_attempts: int = 8,
        read_timeout: int = 90,
    ):
        self.model_chain = list(model_chain or DEFAULT_MODEL_CHAIN)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(
                # High attempts for real task dispatch (Bedrock on-demand
                # throttles under burst load; adaptive mode backs off and
                # retries). The judge overrides this with low attempts +
                # short timeout so a dead model can't stall the run for
                # 8*read_timeout*len(chain) seconds per evaluation.
                retries={"max_attempts": max_attempts, "mode": "adaptive"},
                connect_timeout=15,     # fail fast on degraded networks
                read_timeout=read_timeout,
            ),
        )
        self._records: List[CloudCallRecord] = []
        # RLock (re-entrant): stats() holds the lock and calls mean_latency_ms,
        # which re-acquires it. A plain Lock self-deadlocks there.
        self._lock = threading.RLock()
        self._active_model: Optional[str] = None   # last model that worked

    # ------------------------------------------------------------------
    # STATS  (consumed by the benchmark for Table 2)
    # ------------------------------------------------------------------
    @property
    def total_cost_usd(self) -> float:
        with self._lock:
            return sum(r.cost_usd for r in self._records)

    @property
    def total_calls(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def mean_latency_ms(self) -> float:
        with self._lock:
            if not self._records:
                return 0.0
            return sum(r.latency_ms for r in self._records) / len(self._records)

    def stats(self) -> dict:
        with self._lock:
            return {
                "calls": len(self._records),
                "total_cost_usd": round(sum(r.cost_usd for r in self._records), 6),
                "mean_latency_ms": round(self.mean_latency_ms, 1),
                "active_model": self._active_model,
            }

    def reset_stats(self) -> None:
        with self._lock:
            self._records.clear()

    # ------------------------------------------------------------------
    # MAIN ENTRY POINT  (signature required by FuzzyOrchestratorCore)
    # ------------------------------------------------------------------
    def __call__(self, task_text: str) -> str:
        # Try the last-known-good model first, then the rest of the chain
        chain = self.model_chain
        if self._active_model and self._active_model in chain:
            chain = [self._active_model] + [m for m in chain if m != self._active_model]

        last_error: Optional[Exception] = None
        for model_id in chain:
            try:
                return self._converse(model_id, task_text)
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                logger.warning("Bedrock %s failed (%s) — trying next model", model_id, code)
                last_error = exc
            except Exception as exc:                 # noqa: BLE001
                logger.warning("Bedrock %s failed (%s) — trying next model", model_id, exc)
                last_error = exc

        raise RuntimeError(
            f"All Bedrock models in the fallback chain failed. Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    def _converse(self, model_id: str, task_text: str) -> str:
        t0 = time.perf_counter()
        response = self._client.converse(
            modelId=model_id,
            system=[{"text": self.system_prompt}],
            messages=[{"role": "user", "content": [{"text": task_text}]}],
            inferenceConfig={
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
            },
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        usage = response.get("usage", {})
        in_tok = int(usage.get("inputTokens", 0))
        out_tok = int(usage.get("outputTokens", 0))
        p_in, p_out = _price_for(model_id)
        cost = (in_tok * p_in + out_tok * p_out) / 1_000_000.0

        with self._lock:
            self._records.append(CloudCallRecord(
                model_id=model_id,
                input_tokens=in_tok,
                output_tokens=out_tok,
                latency_ms=latency_ms,
                cost_usd=cost,
            ))
            self._active_model = model_id

        parts = response["output"]["message"]["content"]
        text = "\n".join(p["text"] for p in parts if "text" in p).strip()

        logger.debug(
            "Bedrock | %s | %d→%d tokens | $%.6f | %.0f ms",
            model_id, in_tok, out_tok, cost, latency_ms,
        )
        return text
