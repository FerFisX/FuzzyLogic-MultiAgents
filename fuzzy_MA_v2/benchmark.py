"""
benchmark.py  ─  Real three-configuration benchmark (paper Section 6)
======================================================================
Runs the synthetic task dataset through the three architectures the
paper compares (Table 2):

  1. pure-local   : every task resolved by the local Ollama agents only
                    (FIE escalation disabled — IEN threshold = ∞).
  2. pure-cloud   : every task sent directly to AWS Bedrock.
  3. fuzzy-hybrid : full FIE + consensus + adaptive routing
                    (local first, Bedrock escalation when IEN ≥ 75).

Measured KPIs (Table 2 of the paper)
------------------------------------
  - Mean response latency (s)
  - Financial cost per 1 000 tasks (USD)      [Bedrock token accounting]
  - Global accuracy (%)                        [LLM-judge: Amazon Nova Lite]
  - Escalation rate + false-positive escalations (fuzzy-hybrid only)

Usage
-----
  python benchmark.py --tasks 150 --configs pure-local pure-cloud fuzzy-hybrid
  python benchmark.py --tasks 30 --configs fuzzy-hybrid --no-judge   # quick run
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(__file__))

# Force UTF-8 stdout regardless of the Windows console code page or whether
# output is piped (e.g. through PowerShell Tee-Object, which defaults to
# cp1252 and would crash on the box-drawing/accented characters below).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # pragma: no cover
    pass

from agents.ollama_agent import OllamaEngineeringAgent, OllamaSupportAgent
from audit.auditor import MetricAuditorAgent
from cloud.bedrock import BedrockCloudHandler
from orchestrator.foc import FuzzyOrchestratorCore
from orchestrator.similarity import OllamaEmbeddingSimilarity

logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s %(name)s | %(message)s")
logger = logging.getLogger("benchmark")

_RANDOM_SEED = 42          # reproducible dataset across runs


# ───────────────────────────────────────────────────────────────────────────
# 1. SYNTHETIC DATASET  (paper Section 6.2 — 3 complexity strata)
# ───────────────────────────────────────────────────────────────────────────
_TEMPLATES: Dict[str, List[str]] = {
    "alta": [
        "Deadlock detectado en transacciones concurrentes con index corruption en la tabla de pagos. Realiza un root cause analysis usando el distributed trace y el audit trail.",
        "Spike crítico en tcp retransmission rate causando fallos intermitentes de ssl handshake en el balanceador. Diagnostica la causa raíz considerando la certificate chain.",
        "Memory leak severo identificado tras un cascade delete con foreign key violation en la base de datos de auditoría. Propón un plan de rollback y verificación de idempotency.",
        "Replication lag superior a 5000 ms con violaciones de eventual consistency en los logs transaccionales. Analiza el two-phase commit y propón mitigación.",
        "Segfault intermitente con oom killer activado en el nodo de ejecución. Correlaciona el kernel panic con el dependency graph del despliegue.",
        "Race condition en el pipeline de facturación provoca transacciones duplicadas y rollback parcial. Reconstruye la secuencia con el distributed trace.",
    ],
    "media": [
        "Excepción de timeout en la API de authentication del contenedor de servicios. Revisa la configuración y propone un fix.",
        "Optimizar una query SQL lenta que genera un bottleneck en los logs de auditoría del servicio.",
        "Error intermitente en el pipeline de deploy debido a permisos mal configurados en el contenedor.",
        "La caché del servicio reporta una baja tasa de aciertos y sube la latency de las consultas. ¿Qué revisar?",
        "Fallo de conexión por certificados vencidos en el entorno de pruebas. Indica los pasos de renovación.",
        "El throughput del servicio de mensajería cayó tras el último deploy. Identifica qué configuración revisar.",
    ],
    "baja": [
        "¿Cómo puedo verificar el estado actual del servicio local?",
        "Listar las versiones activas del software en el clúster.",
        "Mostrar los logs generados en los últimos 10 minutos.",
        "Ayuda para reiniciar de forma segura el contenedor de logs.",
        "Chequeo básico de salud del sistema mediante ping.",
        "¿Qué comando muestra el uso de disco del servidor?",
    ],
}


def build_dataset(n_tasks: int) -> List[dict]:
    """Balanced, seeded dataset of n_tasks across the three strata."""
    rng = random.Random(_RANDOM_SEED)
    dataset = []
    levels = ["alta", "media", "baja"]
    i = 0
    while len(dataset) < n_tasks:
        level = levels[i % 3]
        template = rng.choice(_TEMPLATES[level])
        dataset.append({
            "task_id": f"TASK-{i + 1:03d}",
            "content": f"[Incidencia #{i + 1:03d}] {template}",
            "categoria_esperada": level,
        })
        i += 1
    return dataset


# ───────────────────────────────────────────────────────────────────────────
# 2. LLM JUDGE  (automated accuracy proxy — Amazon Nova Lite, ~$0.06/MTok)
# ───────────────────────────────────────────────────────────────────────────
_JUDGE_SYSTEM = (
    "Eres un evaluador experto de soporte técnico. Recibirás una INCIDENCIA y "
    "una RESPUESTA. Evalúa si la respuesta es técnicamente razonable, "
    "específica y aborda directamente la incidencia. Responde EXACTAMENTE con "
    "una sola palabra: CORRECTA o INCORRECTA."
)

_JUDGE_CHAIN = [
    "us.amazon.nova-2-lite-v1:0",
    "amazon.nova-2-lite-v1:0",
    "us.amazon.nova-lite-v1:0",
    "amazon.nova-lite-v1:0",
]


class LLMJudge:
    """Binary correctness judge over (task, response) pairs."""

    def __init__(self):
        self._handler = BedrockCloudHandler(
            model_chain=_JUDGE_CHAIN,
            max_tokens=5,
            temperature=0.0,
            system_prompt=_JUDGE_SYSTEM,
        )

    def is_correct(self, task: str, response: str) -> bool:
        if not response or response.startswith("[LOCAL-FAIL]"):
            return False
        prompt = f"INCIDENCIA:\n{task}\n\nRESPUESTA:\n{response[:1500]}"
        try:
            verdict = self._handler(prompt).strip().upper()
            return verdict.startswith("CORRECTA")
        except Exception as exc:                     # noqa: BLE001
            logger.warning("Judge failed (%s) — counting as incorrect", exc)
            return False

    @property
    def cost_usd(self) -> float:
        return self._handler.total_cost_usd


# ───────────────────────────────────────────────────────────────────────────
# 3. CONFIGURATION RUNNERS
# ───────────────────────────────────────────────────────────────────────────
def _sa_model() -> str:
    """SA model: gemma3:12b — the locally installed Gemma (user decision:
    use the existing model, no extra downloads)."""
    return "gemma3:12b"


def _make_local_agents():
    return [
        OllamaEngineeringAgent(),                    # llama3.2 (EA)
        OllamaSupportAgent(model=_sa_model()),       # gemma3 (SA)
    ]


def run_pure_local(dataset: List[dict], workers: int) -> dict:
    """All tasks resolved locally — FIE consensus active, escalation off."""
    auditor = MetricAuditorAgent()
    foc = FuzzyOrchestratorCore(
        agents=_make_local_agents(),
        auditor=auditor,
        cloud_handler=lambda t: "[LOCAL-FAIL] no local agent responded",
        similarity_fn=OllamaEmbeddingSimilarity(),
        ien_threshold=float("inf"),                  # never escalate
        gamma=0.0,                                   # never declare conflict
        timeout_s=300.0,
    )
    records = _run_through_foc(foc, dataset, workers)
    return {"records": records, "cloud_cost_usd": 0.0, "auditor": auditor.summary()}


def run_pure_cloud(dataset: List[dict], workers: int) -> dict:
    """Every task sent directly to Bedrock."""
    handler = BedrockCloudHandler()
    records = []

    def _one(task):
        t0 = time.perf_counter()
        try:
            response = handler(task["content"])
            ok = True
        except Exception as exc:                     # noqa: BLE001
            response, ok = f"[CLOUD-FAIL] {exc}", False
        wall_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "task_id": task["task_id"],
            "categoria_esperada": task["categoria_esperada"],
            "content": task["content"],
            "response": response,
            "escalated": True,
            "escalation_reason": "pure-cloud (direct dispatch)",
            "wall_ms": round(wall_ms, 1),
            "ok": ok,
        }

    n = len(dataset)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, t) for t in dataset]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            records.append(r)
            flag = "ok " if r["ok"] else "FAIL"
            print(f"      [pure-cloud {i:>3}/{n}] {r['task_id']} "
                  f"({r['categoria_esperada']:<5}) -> NUBE  {flag} "
                  f"{r['wall_ms']/1000:5.1f}s  ${handler.total_cost_usd:.4f}",
                  flush=True)

    return {"records": records, "cloud_cost_usd": handler.total_cost_usd,
            "cloud_stats": handler.stats()}


def run_fuzzy_hybrid(dataset: List[dict], workers: int) -> dict:
    """Full paper architecture: FIE + consensus + adaptive Bedrock routing."""
    auditor = MetricAuditorAgent()
    handler = BedrockCloudHandler()
    foc = FuzzyOrchestratorCore(
        agents=_make_local_agents(),
        auditor=auditor,
        cloud_handler=handler,
        similarity_fn=OllamaEmbeddingSimilarity(),
        timeout_s=300.0,
    )
    records = _run_through_foc(foc, dataset, workers)
    return {"records": records, "cloud_cost_usd": handler.total_cost_usd,
            "cloud_stats": handler.stats(), "auditor": auditor.summary()}


def _run_through_foc(foc: FuzzyOrchestratorCore, dataset: List[dict], workers: int) -> List[dict]:
    records = []

    def _one(task):
        result = foc.process(task["content"], task_id=task["task_id"])
        return {
            "task_id": task["task_id"],
            "categoria_esperada": task["categoria_esperada"],
            "content": task["content"],
            "response": result.final_response,
            "escalated": result.escalated,
            "escalation_reason": result.escalation_reason,
            "wall_ms": round(result.wall_ms, 1),
            "evaluations": [
                {
                    "agent_id": ev.proposal.agent_id,
                    "cs": round(ev.cs, 2),
                    "ir": round(ev.proposal.response_uncertainty, 4),
                    "nc": round(ev.nc, 4),
                    "ien": round(ev.ien, 2),
                    "latency_ms": round(ev.proposal.latency_ms, 1),
                }
                for ev in result.evaluations
            ],
            "weights": {k: round(v, 4) for k, v in result.weights.items()},
            "ok": not result.final_response.startswith(("[LOCAL-FAIL]", "[CLOUD-FAIL]")),
        }

    # Local GPU serialises Ollama calls anyway — keep concurrency modest
    n = len(dataset)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, t) for t in dataset]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                r = fut.result()
                records.append(r)
            except Exception as exc:                 # noqa: BLE001
                logger.error("Task failed: %s", exc)
                print(f"      [foc {i:>3}/{n}] ERROR: {exc}", flush=True)
                continue
            route = "NUBE " if r["escalated"] else "LOCAL"
            # show the local agents' fuzzy verdict that drove the routing
            nc = max((e["nc"] for e in r["evaluations"]), default=0.0)
            ien = max((e["ien"] for e in r["evaluations"]), default=0.0)
            print(f"      [foc {i:>3}/{n}] {r['task_id']} "
                  f"({r['categoria_esperada']:<5}) -> {route} "
                  f"NC={nc:.2f} IEN={ien:4.1f} {r['wall_ms']/1000:5.1f}s",
                  flush=True)

    return records


# ───────────────────────────────────────────────────────────────────────────
# 4. METRICS  (Table 2 KPIs)
# ───────────────────────────────────────────────────────────────────────────
def compute_metrics(config_name: str, run: dict, n_tasks: int,
                    judge: Optional[LLMJudge]) -> dict:
    records = run["records"]
    latencies_s = [r["wall_ms"] / 1000.0 for r in records]
    escalated = [r for r in records if r["escalated"]]
    baja = [r for r in records if r["categoria_esperada"] == "baja"]
    alta = [r for r in records if r["categoria_esperada"] == "alta"]

    # Accuracy via LLM judge (sequential — judge calls are fast + cheap)
    accuracy = None
    if judge is not None:
        correct = 0
        n = len(records)
        print(f"      -- evaluando precision con juez LLM ({config_name}) --", flush=True)
        for i, r in enumerate(records, 1):
            r["judge_correct"] = judge.is_correct(r["content"], r["response"])
            correct += int(r["judge_correct"])
            mark = "CORRECTA  " if r["judge_correct"] else "INCORRECTA"
            print(f"      [judge {i:>3}/{n}] {r['task_id']} -> {mark} "
                  f"(acum {correct}/{i})", flush=True)
        accuracy = 100.0 * correct / len(records) if records else 0.0

    # False-positive escalations: low-complexity tasks sent to the cloud
    fp_esc = None
    if config_name == "fuzzy-hybrid" and baja:
        fp = sum(1 for r in baja if r["escalated"])
        fp_esc = 100.0 * fp / len(baja)

    # Detection rate: high-complexity tasks correctly escalated
    alta_esc_rate = None
    if config_name == "fuzzy-hybrid" and alta:
        alta_esc_rate = 100.0 * sum(1 for r in alta if r["escalated"]) / len(alta)

    cost_per_1k = (run["cloud_cost_usd"] / max(1, len(records))) * 1000.0

    return {
        "config": config_name,
        "tasks": len(records),
        "latencia_media_s": round(statistics.mean(latencies_s), 3) if latencies_s else None,
        "latencia_p95_s": round(
            statistics.quantiles(latencies_s, n=20)[18], 3
        ) if len(latencies_s) >= 20 else None,
        "costo_usd_total": round(run["cloud_cost_usd"], 4),
        "costo_usd_por_1k_tareas": round(cost_per_1k, 2),
        "precision_global_pct": round(accuracy, 1) if accuracy is not None else None,
        "tasa_escalamiento_pct": round(100.0 * len(escalated) / len(records), 1) if records else None,
        "falsos_positivos_escalado_pct": round(fp_esc, 1) if fp_esc is not None else None,
        "deteccion_alta_complejidad_pct": round(alta_esc_rate, 1) if alta_esc_rate is not None else None,
    }


# ───────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ───────────────────────────────────────────────────────────────────────────
RUNNERS = {
    "pure-local": run_pure_local,
    "pure-cloud": run_pure_cloud,
    "fuzzy-hybrid": run_fuzzy_hybrid,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-configuration benchmark (paper Table 2)")
    parser.add_argument("--tasks", type=int, default=60)
    parser.add_argument("--configs", nargs="+", default=list(RUNNERS),
                        choices=list(RUNNERS))
    parser.add_argument("--workers", type=int, default=1,
                        help="concurrent tasks for Ollama configs. KEEP AT 1: "
                             "a 12b model split CPU/GPU saturates and crashes "
                             "Ollama under 6-way (3 tasks x 2 agents) contention.")
    parser.add_argument("--cloud-workers", type=int, default=2,
                        help="concurrent tasks for pure-cloud. Keep low (2): "
                             "Bedrock on-demand throttles under burst load — "
                             "6 produced a 19%% FAIL rate.")
    parser.add_argument("--no-judge", action="store_true",
                        help="skip LLM-judge accuracy evaluation")
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/benchmark_results.json"))
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore per-config checkpoints and rerun everything")
    args = parser.parse_args()

    dataset = build_dataset(args.tasks)
    print(f"\n=== Benchmark MICAI 2026 — {args.tasks} tareas | configs: {args.configs} ===\n",
          flush=True)

    judge = None if args.no_judge else LLMJudge()
    all_metrics, all_runs = [], {}
    ckpt_dir = args.out.parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for config in args.configs:
        ckpt_path = ckpt_dir / f"benchmark_ckpt_{config}_{args.tasks}.json"

        # ── Resume: reuse a completed config from its checkpoint ──
        if ckpt_path.exists() and not args.no_resume:
            ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
            print(f"  ── Config: {config} — REANUDADA desde checkpoint "
                  f"({ckpt_path.name}) ──", flush=True)
            all_metrics.append(ckpt["metrics"])
            all_runs[config] = ckpt["run"]
            continue

        print(f"  ── Config: {config} ──", flush=True)
        workers = args.cloud_workers if config == "pure-cloud" else args.workers
        t0 = time.perf_counter()
        run = RUNNERS[config](dataset, workers)
        elapsed = time.perf_counter() - t0
        print(f"      completado en {elapsed:.1f} s", flush=True)

        metrics = compute_metrics(config, run, args.tasks, judge)
        metrics["tiempo_total_config_s"] = round(elapsed, 1)
        all_metrics.append(metrics)
        all_runs[config] = run
        print(f"      {json.dumps(metrics, ensure_ascii=False)}\n", flush=True)

        # ── Checkpoint: a crash in a later config never loses this one ──
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "run": run}, f, ensure_ascii=False)
        print(f"      checkpoint → {ckpt_path.name}", flush=True)

    # ── Export ──
    output = {
        "benchmark": "MICAI 2026 — Fuzzy-Hybrid vs Pure-Local vs Pure-Cloud",
        "n_tasks": args.tasks,
        "seed": _RANDOM_SEED,
        "judge_cost_usd": round(judge.cost_usd, 4) if judge else None,
        "table2_metrics": all_metrics,
        "runs": {
            cfg: {k: v for k, v in run.items() if k != "auditor"} | {
                "auditor": run.get("auditor")
            }
            for cfg, run in all_runs.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Resultados → {args.out}")

    # ── Table 2 summary ──
    print("\n  ══ TABLA 2 (datos reales) ══")
    header = f"  {'KPI':<38}" + "".join(f"{m['config']:>16}" for m in all_metrics)
    print(header)
    rows = [
        ("Latencia media (s)", "latencia_media_s"),
        ("Costo USD / 1k tareas", "costo_usd_por_1k_tareas"),
        ("Precisión global (%)", "precision_global_pct"),
        ("Tasa de escalamiento (%)", "tasa_escalamiento_pct"),
        ("Falsos positivos escalado (%)", "falsos_positivos_escalado_pct"),
        ("Detección alta complejidad (%)", "deteccion_alta_complejidad_pct"),
    ]
    for label, key in rows:
        vals = "".join(
            f"{(m.get(key) if m.get(key) is not None else '—'):>16}"
            for m in all_metrics
        )
        print(f"  {label:<38}{vals}")
    print()


if __name__ == "__main__":
    main()
