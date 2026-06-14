"""
merge_results.py  ─  Merge per-config benchmark JSONs into one file
====================================================================
Allows running configurations at different times (e.g. pure-cloud while
the local models download) and consolidating them for the paper:

    python merge_results.py outputs/bench_pure_cloud.json \
                            outputs/bench_local_hybrid.json \
                            -o outputs/benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("-o", "--out", type=Path,
                        default=Path("outputs/benchmark_results.json"))
    args = parser.parse_args()

    merged = None
    judge_cost = 0.0
    for path in args.inputs:
        data = json.loads(path.read_text(encoding="utf-8"))
        judge_cost += data.get("judge_cost_usd") or 0.0
        if merged is None:
            merged = data
            continue
        # Later files override/extend earlier ones per config
        existing = {m["config"]: m for m in merged["table2_metrics"]}
        for m in data["table2_metrics"]:
            existing[m["config"]] = m
        merged["table2_metrics"] = list(existing.values())
        merged["runs"].update(data["runs"])

    merged["judge_cost_usd"] = round(judge_cost, 4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    configs = [m["config"] for m in merged["table2_metrics"]]
    print(f"Fusionado → {args.out}  (configs: {configs})")


if __name__ == "__main__":
    main()
