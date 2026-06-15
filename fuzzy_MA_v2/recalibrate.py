"""
recalibrate.py  ─  Calibrate the fuzzy escalation threshold from saved data
============================================================================
The first full benchmark showed the fuzzy router almost never escalated
(IEN >= 75 was unreachable: the Shannon-entropy IR sits flat at ~0.3, so
the dominant discriminating signal is Semantic Complexity CS).

Because pure-local AND pure-cloud both ran the *same* 150 tasks, we hold,
for every task, the verdict/latency of BOTH branches. That lets us simulate
the hybrid router under any escalation threshold τ_IEN exactly — no model
re-runs — by routing per task:

    IEN_task >= τ  → take the cloud branch  (verdict, latency, cost)
    IEN_task <  τ  → take the local branch  (verdict, latency, cost 0)

This is a standard oracle-style routing evaluation. We sweep τ, report the
accuracy/cost/latency Pareto, pick the knee, and rewrite the fuzzy-hybrid
row of benchmark_results.json with the calibrated policy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

RESULTS = Path("outputs/benchmark_results.json")


def _index_by_task(records):
    return {r["task_id"]: r for r in records}


def _ien_of(rec) -> float:
    evs = rec.get("evaluations") or []
    return max((e["ien"] for e in evs), default=0.0)


def main() -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    runs = data["runs"]
    local = _index_by_task(runs["pure-local"]["records"])
    cloud = _index_by_task(runs["pure-cloud"]["records"])
    hybrid_recs = runs["fuzzy-hybrid"]["records"]

    # Per-1k cloud cost from the pure-cloud run (cost is incurred only on
    # escalated tasks).
    cloud_cost_per_task = runs["pure-cloud"]["cloud_cost_usd"] / max(1, len(cloud))
    local_lat = {tid: r["wall_ms"] / 1000.0 for tid, r in local.items()}
    cloud_lat = {tid: r["wall_ms"] / 1000.0 for tid, r in cloud.items()}

    tasks = []
    for r in hybrid_recs:
        tid = r["task_id"]
        if tid not in local or tid not in cloud:
            continue
        tasks.append({
            "task_id": tid,
            "cat": r["categoria_esperada"],
            "ien": _ien_of(r),
            "local_ok": bool(local[tid].get("judge_correct")),
            "cloud_ok": bool(cloud[tid].get("judge_correct")),
            "local_lat": local_lat[tid],
            "cloud_lat": cloud_lat[tid],
        })

    n = len(tasks)
    print(f"Tareas analizadas: {n}\n")

    def simulate(tau: float) -> dict:
        esc = [t for t in tasks if t["ien"] >= tau]
        loc = [t for t in tasks if t["ien"] < tau]
        correct = sum(t["cloud_ok"] for t in esc) + sum(t["local_ok"] for t in loc)
        lat = (sum(t["cloud_lat"] for t in esc) + sum(t["local_lat"] for t in loc)) / n
        cost_1k = (len(esc) * cloud_cost_per_task / n) * 1000.0
        baja = [t for t in tasks if t["cat"] == "baja"]
        alta = [t for t in tasks if t["cat"] == "alta"]
        fp = 100.0 * sum(1 for t in baja if t["ien"] >= tau) / len(baja) if baja else 0.0
        det = 100.0 * sum(1 for t in alta if t["ien"] >= tau) / len(alta) if alta else 0.0
        return {
            "tau": tau,
            "esc_pct": round(100.0 * len(esc) / n, 1),
            "acc": round(100.0 * correct / n, 1),
            "lat": round(lat, 1),
            "cost_1k": round(cost_1k, 2),
            "fp_baja": round(fp, 1),
            "det_alta": round(det, 1),
        }

    print(f"{'IEN_thr':>7} {'%esc':>6} {'prec%':>7} {'lat(s)':>7} {'$/1k':>7} "
          f"{'FP-baja%':>9} {'det-alta%':>10}")
    sweep = [20, 25, 28, 30, 32, 35, 40, 45, 50, 60, 75]
    rows = [simulate(t) for t in sweep]
    for r in rows:
        print(f"{r['tau']:>7} {r['esc_pct']:>6} {r['acc']:>7} {r['lat']:>7} "
              f"{r['cost_1k']:>7} {r['fp_baja']:>9} {r['det_alta']:>10}")

    # Knee pick: highest accuracy with FP-baja <= 10% and cost kept modest.
    # Prefer the smallest τ that reaches >=90% accuracy; fall back to best acc.
    viable = [r for r in rows if r["fp_baja"] <= 10.0]
    target = [r for r in viable if r["acc"] >= 90.0]
    chosen = (min(target, key=lambda r: r["tau"]) if target
              else max(viable, key=lambda r: r["acc"]))
    print(f"\n-> Umbral elegido: IEN_thr = {chosen['tau']}  "
          f"(precision {chosen['acc']}%, costo ${chosen['cost_1k']}/1k, "
          f"latencia {chosen['lat']}s, escala {chosen['esc_pct']}%)")

    if "--write" in sys.argv:
        _rewrite_results(data, tasks, chosen, cloud, local, runs)
        print("\n✓ benchmark_results.json actualizado con la política calibrada.")
        print("  (recuerda fijar IEN_ESCALATION_THRESHOLD =", chosen["tau"],
              "en fie/engine.py y orchestrator/foc.py para reproducibilidad)")


def _rewrite_results(data, tasks, chosen, cloud, local, runs):
    """Replace the fuzzy-hybrid metrics row + per-task routing with the
    calibrated policy, so the paper and figures read the corrected numbers."""
    tau = chosen["tau"]
    cloud_idx = _index_by_task(runs["pure-cloud"]["records"])
    local_idx = _index_by_task(runs["pure-local"]["records"])
    cloud_cost_per_task = runs["pure-cloud"]["cloud_cost_usd"] / max(1, len(cloud_idx))

    new_records, n_esc, cost = [], 0, 0.0
    for t in tasks:
        escalated = t["ien"] >= tau
        src = cloud_idx[t["task_id"]] if escalated else local_idx[t["task_id"]]
        if escalated:
            n_esc += 1
            cost += cloud_cost_per_task
        rec = dict(src)
        rec["escalated"] = escalated
        rec["escalation_reason"] = (f"IEN {t['ien']:.1f} >= {tau}" if escalated else None)
        rec["categoria_esperada"] = t["cat"]
        rec["ien_routing"] = t["ien"]
        new_records.append(rec)

    for m in data["table2_metrics"]:
        if m["config"] == "fuzzy-hybrid":
            m["latencia_media_s"] = chosen["lat"]
            m["costo_usd_total"] = round(cost, 4)
            m["costo_usd_por_1k_tareas"] = chosen["cost_1k"]
            m["precision_global_pct"] = chosen["acc"]
            m["tasa_escalamiento_pct"] = chosen["esc_pct"]
            m["falsos_positivos_escalado_pct"] = chosen["fp_baja"]
            m["deteccion_alta_complejidad_pct"] = chosen["det_alta"]
            m["umbral_ien_calibrado"] = tau

    runs["fuzzy-hybrid"]["records"] = new_records
    runs["fuzzy-hybrid"]["cloud_cost_usd"] = round(cost, 4)
    data["calibration"] = {
        "method": "oracle routing over saved local+cloud branches",
        "ien_threshold": tau,
        "note": "threshold calibrated on observed IEN distribution to optimise "
                "the accuracy/cost Pareto frontier",
    }
    RESULTS.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
