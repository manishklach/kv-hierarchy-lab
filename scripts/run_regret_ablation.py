"""Benchmark runner for regret-aware policy ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kv_hierarchy_lab.bench.report import write_csv, write_json
from kv_hierarchy_lab.bench.runner import run_benchmark
from kv_hierarchy_lab.bench.scenarios import Scenario, default_tier_configs
from kv_hierarchy_lab.config import TierConfig
from kv_hierarchy_lab.policies import CostAwarePolicy, LRUPolicy, RegretAwarePolicy
from kv_hierarchy_lab.utils.io import ensure_dir
from kv_hierarchy_lab.workloads import (
    generate_adversarial_burst,
    generate_chat_continuation,
    generate_periodic_reuse,
)

def create_scenarios() -> list[Scenario]:
    base = default_tier_configs()
    medium = [
        TierConfig(cfg.name, max(1, cfg.capacity_bytes // 2), cfg.read_latency_ms, cfg.write_latency_ms, cfg.bandwidth_bytes_per_ms, cfg.concurrency_limit)
        for cfg in base
    ]
    constrained = [
        TierConfig(cfg.name, max(1, cfg.capacity_bytes // 4), cfg.read_latency_ms, cfg.write_latency_ms, cfg.bandwidth_bytes_per_ms, cfg.concurrency_limit)
        for cfg in base
    ]
    return [
        Scenario("medium", medium, "fp16", False, notes="Medium capacity, prefetch off"),
        Scenario("constrained", constrained, "fp16", False, notes="Constrained capacity, prefetch off"),
    ]

def plot_latency_vs_horizon(frame: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(7, 5))
    regret_frame = frame[frame["policy"].str.startswith("regret_aware_w6.0")].copy()
    if regret_frame.empty:
        return
    regret_frame["horizon"] = regret_frame["policy"].apply(lambda p: int(p.split("_h")[1]))
    
    for (workload, scenario), subset in regret_frame.groupby(["workload", "scenario"]):
        agg = subset.groupby("horizon", as_index=False)["avg_latency_ms"].mean()
        plt.plot(agg["horizon"], agg["avg_latency_ms"], marker='o', label=f"{workload} ({scenario})")
    
    plt.xlabel("Regret Horizon (steps)")
    plt.ylabel("Avg Latency (ms)")
    plt.title("Latency vs Regret Horizon")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 8})
    plt.tight_layout()
    plt.savefig(out_dir / "latency_vs_horizon.png")
    plt.close()

def plot_hit_rate_vs_weight(frame: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(7, 5))
    regret_frame = frame[frame["policy"].str.contains("regret_aware_w.*_h24")].copy()
    if regret_frame.empty:
        return
    regret_frame["weight"] = regret_frame["policy"].apply(lambda p: float(p.split("_w")[1].split("_")[0]))
    
    for (workload, scenario), subset in regret_frame.groupby(["workload", "scenario"]):
        agg = subset.groupby("weight", as_index=False)["overall_hit_rate"].mean()
        plt.plot(agg["weight"], agg["overall_hit_rate"], marker='s', label=f"{workload} ({scenario})")

    plt.xlabel("Regret Weight")
    plt.ylabel("Overall Hit Rate")
    plt.title("Hit Rate vs Regret Weight")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 8})
    plt.tight_layout()
    plt.savefig(out_dir / "hit_rate_vs_weight.png")
    plt.close()

def plot_tradeoff(frame: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(7, 5))
    # Filter to 1 scenario and workload to keep it readable, e.g. run_adversarial_burst under constrained
    subset = frame[(frame["workload"] == "adversarial_burst") & (frame["scenario"] == "constrained")]
    if subset.empty:
        return
    for policy, group in subset.groupby("policy"):
        plt.scatter(group["avg_latency_ms"], group["bytes_moved"], label=policy, s=100, alpha=0.7)
    
    plt.xlabel("Avg Latency (ms)")
    plt.ylabel("Bytes Moved")
    plt.title("Latency vs Transfer Cost\n(Adversarial Burst, Constrained Capacity)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 8})
    plt.tight_layout()
    plt.savefig(out_dir / "tradeoff.png")
    plt.close()

def plot_failure_case(frame: pd.DataFrame, out_dir: Path) -> None:
    # A potential failure/neutral case might be purely recency-heavy (chat_continuation)
    # where LRU might perform just as well or better than heavily weighted regret logic.
    plt.figure(figsize=(8, 5))
    subset = frame[(frame["workload"] == "chat_continuation") & (frame["scenario"] == "medium")]
    if subset.empty:
        return
    subset = subset.groupby("policy", as_index=False)["avg_latency_ms"].mean().sort_values("avg_latency_ms")
    
    plt.bar(subset["policy"], subset["avg_latency_ms"], color="salmon")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Avg Latency (ms)")
    plt.title("Failure/Neutral Case: Chat Continuation (Medium Cap.)\nRegret logic offers limited/no advantage over Recency")
    plt.tight_layout()
    plt.savefig(out_dir / "failure_case.png")
    plt.close()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/regret_ablation"))
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    ensure_dir(args.output_dir / "plots")

    workload_factories = [
        generate_chat_continuation,
        generate_periodic_reuse,
        generate_adversarial_burst,
    ]
    
    policies = [
        LRUPolicy(),
        CostAwarePolicy(),
    ]
    
    weights = [1.0, 6.0, 12.0]
    horizons = [12, 24, 96]
    
    for weight in weights:
        for horizon in horizons:
            policies.append(
                RegretAwarePolicy(
                    name=f"regret_aware_w{weight}_h{horizon}",
                    regret_weight=weight,
                    regret_horizon=horizon,
                )
            )

    scenarios = create_scenarios()
    
    results = []
    for scenario in scenarios:
        for w_factory in workload_factories:
            workload = w_factory(quantization_scheme=scenario.quantization_scheme)
            for policy in policies:
                results.append(run_benchmark(scenario, workload, policy))

    json_path = args.output_dir / "raw_results.json"
    csv_path = args.output_dir / "summary.csv"
    write_json(results, json_path)
    write_csv(results, csv_path)
    
    frame = pd.read_json(json_path)
    metrics = pd.json_normalize(frame["metrics"])
    frame = pd.concat([frame.drop(columns=["metrics"]), metrics], axis=1)
    if "overall_hit_rate" not in frame.columns:
        frame["overall_hit_rate"] = 1.0 - frame["miss_count"] / frame["accesses"]

    plot_latency_vs_horizon(frame, args.output_dir / "plots")
    plot_hit_rate_vs_weight(frame, args.output_dir / "plots")
    plot_tradeoff(frame, args.output_dir / "plots")
    plot_failure_case(frame, args.output_dir / "plots")

    print(f"Ablation complete. Wrote {len(results)} outputs to {args.output_dir.as_posix()}")

if __name__ == "__main__":
    main()
