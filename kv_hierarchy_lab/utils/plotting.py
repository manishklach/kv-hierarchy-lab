"""Plotting utilities for benchmark results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from kv_hierarchy_lab.utils.io import ensure_dir


def load_results_frame(results_path: Path) -> pd.DataFrame:
    """Loads result JSON into a tidy DataFrame."""
    frame = pd.read_json(results_path)
    metrics = pd.json_normalize(frame["metrics"])
    return pd.concat([frame.drop(columns=["metrics"]), metrics], axis=1)


def plot_bar(frame: pd.DataFrame, metric: str, output_path: Path) -> None:
    """Creates a policy comparison bar chart for one metric."""
    summary = frame.groupby("policy", as_index=False)[metric].mean().sort_values(metric)
    plt.figure(figsize=(8, 4.5))
    plt.bar(summary["policy"], summary[metric], color="#315c8f")
    plt.ylabel(metric)
    plt.title(f"{metric} by policy")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_tradeoff(frame: pd.DataFrame, output_path: Path) -> None:
    """Creates a latency-vs-bytes-moved scatter plot."""
    plt.figure(figsize=(6.5, 5))
    for policy, subset in frame.groupby("policy"):
        plt.scatter(subset["avg_latency_ms"], subset["bytes_moved"], label=policy)
    plt.xlabel("avg_latency_ms")
    plt.ylabel("bytes_moved")
    plt.title("Latency vs bytes moved")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_standard_plots(results_path: Path, out_dir: Path) -> list[Path]:
    """Generates a basic plot set from JSON results."""
    ensure_dir(out_dir)
    frame = load_results_frame(results_path)
    outputs = []
    for metric in ["avg_latency_ms", "bytes_moved", "prefetch_usefulness", "miss_count"]:
        path = out_dir / f"{metric}.png"
        plot_bar(frame, metric, path)
        outputs.append(path)
    tradeoff = out_dir / "latency_vs_bytes_moved.png"
    plot_tradeoff(frame, tradeoff)
    outputs.append(tradeoff)
    return outputs
