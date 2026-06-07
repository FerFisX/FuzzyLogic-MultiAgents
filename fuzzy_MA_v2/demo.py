"""
demo.py  ─  End-to-end demonstration of the FIE + MAS integration
==================================================================
Run with:  python demo.py
           python demo.py --no-plots   (skip matplotlib windows)

This script walks through:
  1. FIE standalone evaluation (paper benchmark cases)
  2. Full orchestrator pipeline with stub agents
  3. Membership function plots
  4. Pareto frontier figure
"""

from __future__ import annotations

import argparse
import logging
import sys
import os

# Make sure the package root is on the path when running directly
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s | %(message)s",
)
logger = logging.getLogger("demo")


# ───────────────────────────────────────────────────────────────────────────
# 1. STANDALONE FIE TESTS
# ───────────────────────────────────────────────────────────────────────────
def run_fie_demo(show_plots: bool = True) -> None:
    from fie import evaluate, evaluar_propuesta

    print("\n" + "=" * 60)
    print("  FIE STANDALONE EVALUATION DEMO")
    print("=" * 60)

    CASES = [
        # (label, CS, IR, FH, expected_nc_op, expected_nc_val, expected_ien_op, expected_ien_val)
        ("Easy / Secure",        10,   0.10,  95,  ">", 0.75, "<", 35),
        ("Hard / Uncertain",     85,   0.75,  40,  "<", 0.45, ">", 70),
        ("Paper Test-A",         70.0, 0.80,  40.0,"<", 0.40, ">", 70),
        ("Paper Test-B",         20.0, 0.10,  95.0,">", 0.80, "<", 30),
        ("High-CS / Low IR",     90,   0.10,  90,  ">", 0.55, "<", 65),
        ("Boundary: worst case", 100,  1.00,  0,   "<", 0.40, ">", 60),
        ("Boundary: best case",  0,    0.00,  100, ">", 0.85, "<", 25),
    ]

    all_pass = True
    for label, cs, ir, fh, nc_op, nc_val, ien_op, ien_val in CASES:
        result = evaluate(cs, ir, fh)
        nc_ok  = (result.nc  > nc_val)  if nc_op  == ">" else (result.nc  < nc_val)
        ien_ok = (result.ien > ien_val) if ien_op == ">" else (result.ien < ien_val)
        status = "✓" if (nc_ok and ien_ok) else "✗"
        if not (nc_ok and ien_ok):
            all_pass = False
        route = "⚠ ESCALATE" if result.should_escalate else "✓ LOCAL  "
        print(
            f"  {status} [{label:<28}] "
            f"CS={cs:5.1f} IR={ir:.2f} FH={fh:5.1f} → "
            f"NC={result.nc:.3f}  IEN={result.ien:.2f}  {route}  "
            f"[{result.eval_ms:.2f} ms]"
        )

    print(f"\n  {'All assertions passed ✓' if all_pass else 'Some assertions FAILED ✗'}")

    if show_plots:
        from fie.visualisation import plot_membership_functions, plot_evaluation, plot_pareto
        print("\n  → Rendering membership function plots…")
        plot_membership_functions(cs=70, ir=0.6, fh=40)
        print("  → Rendering single evaluation diagnostic (CS=70, IR=0.6, FH=40)…")
        plot_evaluation(cs=70, ir=0.6, fh=40)
        print("  → Rendering Pareto frontier…")
        plot_pareto()


# ───────────────────────────────────────────────────────────────────────────
# 2. FULL ORCHESTRATOR DEMO
# ───────────────────────────────────────────────────────────────────────────
def run_orchestrator_demo() -> None:
    from agents.stubs import StubEngineeringAgent, StubSupportAgent
    from audit.auditor import MetricAuditorAgent
    from orchestrator.foc import FuzzyOrchestratorCore
    from orchestrator.complexity import compute_semantic_complexity

    print("\n" + "=" * 60)
    print("  FUZZY ORCHESTRATOR CORE DEMO")
    print("=" * 60)

    # Shared auditor persists FH across tasks
    auditor = MetricAuditorAgent()

    # Two confident stub agents (IR fixed for reproducibility)
    agents = [
        StubEngineeringAgent(uncertainty_override=0.15, latency_ms=20),
        StubSupportAgent(uncertainty_override=0.25,     latency_ms=15),
    ]

    foc = FuzzyOrchestratorCore(agents=agents, auditor=auditor)

    TASKS = [
        "How do I check the application service status?",
        (
            "There is a deadlock in the transaction pipeline caused by a race condition "
            "with index corruption during the foreign key cascade delete. Provide root "
            "cause analysis using the distributed trace and audit trail."
        ),
        "List the steps to restart the database service.",
        (
            "The replication lag has exceeded 3000 ms and is causing eventual consistency "
            "violations in the transaction logs. The TCP retransmission rate has spiked. "
            "Diagnose and propose a fix, including rollback strategy if required."
        ),
    ]

    for i, task in enumerate(TASKS, 1):
        cs = compute_semantic_complexity(task)
        result = foc.process(task)
        print(f"\n  Task {i} (CS={cs:.1f})")
        print(result.summary())

    print("\n  ── Agent Reliability Summary ──")
    for agent_id, rec in auditor.summary().items():
        print(f"  {agent_id:<28}  FH={rec['fh']:.2f}  tasks={rec['total_tasks']}")


# ───────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ───────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="FIE + MAS demo")
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib windows")
    args = parser.parse_args()

    run_fie_demo(show_plots=not args.no_plots)
    run_orchestrator_demo()
    print("\n  Demo complete.\n")


if __name__ == "__main__":
    main()
