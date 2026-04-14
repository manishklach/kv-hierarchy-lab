"""Regret-aware policy ablation study.

Sweeps regret_horizon and regret_weight under pressure-inducing scenarios
with multi-seed variance reporting. Each benchmark run uses a fresh policy
instance to avoid state leakage between independent runs.

Regenerate artifacts:
    python scripts/run_regret_ablation.py --output-dir artifacts/regret_ablation
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

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

# ---------------------------------------------------------------------------
# Ablation configuration
# ---------------------------------------------------------------------------

SEEDS = [7, 42, 91]
WEIGHTS = [1.0, 6.0, 12.0]
HORIZONS = [8, 24, 64]

# Workload parameters are tuned to create meaningful eviction pressure:
# - larger working sets relative to fast-tier capacity
# - longer traces so policies accumulate enough history to diverge
WORKLOAD_CONFIGS: dict[str, dict[str, Any]] = {
    "chat_continuation": {"num_pages": 80, "length": 512},
    "periodic_reuse": {"num_pages": 96, "length": 640, "period": 11},
    "adversarial_burst": {"num_pages": 112, "length": 640, "burst_size": 14},
}

WORKLOAD_FACTORIES: dict[str, Callable[..., Any]] = {
    "chat_continuation": generate_chat_continuation,
    "periodic_reuse": generate_periodic_reuse,
    "adversarial_burst": generate_adversarial_burst,
}


PolicyFactory = Callable[[], Any]


def _policy_factories() -> list[tuple[str, PolicyFactory]]:
    """Returns (name, factory) pairs so every run gets a fresh instance."""
    factories: list[tuple[str, PolicyFactory]] = [
        ("lru", LRUPolicy),
        ("cost_aware", CostAwarePolicy),
    ]
    for weight in WEIGHTS:
        for horizon in HORIZONS:
            name = f"regret_aware_w{weight}_h{horizon}"
            factories.append((
                name,
                lambda w=weight, h=horizon, n=name: RegretAwarePolicy(
                    name=n, regret_weight=w, regret_horizon=h,
                ),
            ))
    return factories


def _create_scenarios() -> list[Scenario]:
    """Scenarios with tight fast-tier budgets to force eviction pressure."""
    base = default_tier_configs()
    # Medium: fast tier at 1/3 of base (was 1/2 previously — too loose)
    medium = [
        TierConfig(cfg.name, max(1, cfg.capacity_bytes // 3),
                   cfg.read_latency_ms, cfg.write_latency_ms,
                   cfg.bandwidth_bytes_per_ms)
        for cfg in base
    ]
    # Constrained: fast tier at 1/6 of base (was 1/4 — tighter now)
    constrained = [
        TierConfig(cfg.name, max(1, cfg.capacity_bytes // 6),
                   cfg.read_latency_ms, cfg.write_latency_ms,
                   cfg.bandwidth_bytes_per_ms)
        for cfg in base
    ]
    return [
        Scenario("medium", medium, "fp16", False,
                 notes="Moderate pressure, prefetch off."),
        Scenario("constrained", constrained, "fp16", False,
                 notes="Aggressive fast-tier pressure, prefetch off."),
    ]


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def _run_sweep(scenarios: list[Scenario]) -> list[Any]:
    """Runs the full ablation matrix with multi-seed variance."""
    factories = _policy_factories()
    results = []
    for seed in SEEDS:
        for scenario in scenarios:
            for wl_name, wl_factory in WORKLOAD_FACTORIES.items():
                wl_cfg = {**WORKLOAD_CONFIGS[wl_name], "seed": seed,
                          "quantization_scheme": scenario.quantization_scheme}
                workload = wl_factory(**wl_cfg)
                for policy_name, policy_factory in factories:
                    # Fresh policy instance per run — no state leakage.
                    result = run_benchmark(scenario, workload, policy_factory())
                    results.append(result)
    return results


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def _load_frame(json_path: Path) -> pd.DataFrame:
    frame = pd.read_json(json_path)
    metrics = pd.json_normalize(frame["metrics"])
    frame = pd.concat([frame.drop(columns=["metrics"]), metrics], axis=1)
    if "overall_hit_rate" not in frame.columns:
        frame["overall_hit_rate"] = 1.0 - frame["miss_count"] / frame["accesses"]
    return frame


def _aggregate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregates across seeds to produce mean ± std for key metrics."""
    group_cols = ["scenario", "workload", "policy"]
    metric_cols = ["avg_latency_ms", "overall_hit_rate", "miss_count",
                   "bytes_moved", "eviction_count"]
    agg = frame.groupby(group_cols, as_index=False)[metric_cols].agg(["mean", "std"])
    # Flatten multi-level columns
    agg.columns = [
        col[0] if col[1] == "" else f"{col[0]}_{col[1]}"
        for col in agg.columns
    ]
    return agg


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _extract_horizon(policy: str) -> int | None:
    if "_h" not in policy:
        return None
    try:
        return int(policy.split("_h")[-1])
    except ValueError:
        return None


def _extract_weight(policy: str) -> float | None:
    if "_w" not in policy:
        return None
    try:
        return float(policy.split("_w")[1].split("_")[0])
    except (ValueError, IndexError):
        return None


def plot_latency_vs_horizon(agg: pd.DataFrame, out_dir: Path) -> None:
    """Latency as a function of horizon at fixed weight=6.0, with error bars."""
    plt.figure(figsize=(7, 5))
    subset = agg[agg["policy"].str.startswith("regret_aware_w6.0")].copy()
    if subset.empty:
        return
    subset["horizon"] = subset["policy"].apply(_extract_horizon)

    for (workload, scenario), grp in subset.groupby(["workload", "scenario"]):
        grp = grp.sort_values("horizon")
        plt.errorbar(grp["horizon"], grp["avg_latency_ms_mean"],
                     yerr=grp["avg_latency_ms_std"], marker="o", capsize=3,
                     label=f"{workload} ({scenario})")

    plt.xlabel("Regret Horizon (steps)")
    plt.ylabel("Avg Latency (ms)  [mean ± std]")
    plt.title("Latency vs Regret Horizon  (weight = 6.0)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", prop={"size": 8})
    plt.tight_layout()
    plt.savefig(out_dir / "latency_vs_horizon.png", dpi=150)
    plt.close()


def plot_hit_rate_vs_weight(agg: pd.DataFrame, out_dir: Path) -> None:
    """Hit rate as a function of weight at fixed horizon=24, with error bars."""
    plt.figure(figsize=(7, 5))
    subset = agg[agg["policy"].str.contains("_h24")].copy()
    if subset.empty:
        return
    subset["weight"] = subset["policy"].apply(_extract_weight)

    for (workload, scenario), grp in subset.groupby(["workload", "scenario"]):
        grp = grp.sort_values("weight")
        plt.errorbar(grp["weight"], grp["overall_hit_rate_mean"],
                     yerr=grp["overall_hit_rate_std"], marker="s", capsize=3,
                     label=f"{workload} ({scenario})")

    plt.xlabel("Regret Weight")
    plt.ylabel("Overall Hit Rate  [mean ± std]")
    plt.title("Hit Rate vs Regret Weight  (horizon = 24)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", prop={"size": 8})
    plt.tight_layout()
    plt.savefig(out_dir / "hit_rate_vs_weight.png", dpi=150)
    plt.close()


def plot_tradeoff(agg: pd.DataFrame, out_dir: Path) -> None:
    """Latency vs bytes-moved scatter for adversarial_burst × constrained."""
    plt.figure(figsize=(7, 5))
    subset = agg[(agg["workload"] == "adversarial_burst")
                 & (agg["scenario"] == "constrained")]
    if subset.empty:
        return
    for _, row in subset.iterrows():
        plt.scatter(row["avg_latency_ms_mean"], row["bytes_moved_mean"],
                    s=100, alpha=0.75)
        plt.annotate(row["policy"],
                     (row["avg_latency_ms_mean"], row["bytes_moved_mean"]),
                     fontsize=7, ha="center", va="bottom")

    plt.xlabel("Avg Latency (ms)")
    plt.ylabel("Bytes Moved")
    plt.title("Latency vs Transfer Cost\n(adversarial_burst, constrained)")
    plt.tight_layout()
    plt.savefig(out_dir / "tradeoff.png", dpi=150)
    plt.close()


def plot_failure_case(agg: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart for chat_continuation × medium where regret adds no value."""
    plt.figure(figsize=(8, 5))
    subset = agg[(agg["workload"] == "chat_continuation")
                 & (agg["scenario"] == "medium")].copy()
    if subset.empty:
        return
    subset = subset.sort_values("avg_latency_ms_mean")

    plt.barh(subset["policy"], subset["avg_latency_ms_mean"],
             xerr=subset["avg_latency_ms_std"], color="salmon", capsize=3)
    plt.xlabel("Avg Latency (ms)  [mean ± std]")
    plt.title("Neutral Case: chat_continuation (medium)\n"
              "Regret logic offers limited advantage over plain recency")
    plt.tight_layout()
    plt.savefig(out_dir / "failure_case.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/regret_ablation"))
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    ensure_dir(args.output_dir / "plots")

    scenarios = _create_scenarios()
    results = _run_sweep(scenarios)

    # ---- raw results ----
    json_path = args.output_dir / "raw_results.json"
    csv_path = args.output_dir / "summary.csv"
    write_json(results, json_path)
    write_csv(results, csv_path)

    # ---- aggregated summary ----
    frame = _load_frame(json_path)
    agg = _aggregate_frame(frame)
    agg_path = args.output_dir / "aggregated_summary.csv"
    agg.to_csv(agg_path, index=False)

    # ---- metadata ----
    meta = {
        "command": f"python scripts/run_regret_ablation.py --output-dir {args.output_dir.as_posix()}",
        "seeds": SEEDS,
        "weights": WEIGHTS,
        "horizons": HORIZONS,
        "workloads": list(WORKLOAD_CONFIGS.keys()),
        "workload_configs": WORKLOAD_CONFIGS,
        "scenarios": [s.name for s in scenarios],
        "total_runs": len(results),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    # ---- plots ----
    plot_latency_vs_horizon(agg, args.output_dir / "plots")
    plot_hit_rate_vs_weight(agg, args.output_dir / "plots")
    plot_tradeoff(agg, args.output_dir / "plots")
    plot_failure_case(agg, args.output_dir / "plots")

    print(f"Ablation complete: {len(results)} runs across {len(SEEDS)} seeds.")
    print(f"Artifacts written to {args.output_dir.as_posix()}/")


if __name__ == "__main__":
    main()
