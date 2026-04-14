"""Result export helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from kv_hierarchy_lab.bench.runner import BenchmarkResult


def _flatten_result(result: BenchmarkResult) -> dict[str, str | int | float]:
    flat: dict[str, str | int | float] = {
        "scenario": result.scenario,
        "workload": result.workload,
        "policy": result.policy,
        "quantization_scheme": result.quantization_scheme,
        "prefetch_enabled": result.prefetch_enabled,
    }
    for key, value in result.metrics.items():
        flat[key] = json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
    return flat


def write_json(results: list[BenchmarkResult], output_path: Path) -> None:
    """Writes benchmark results to JSON."""
    payload = [
        {
            "scenario": result.scenario,
            "workload": result.workload,
            "policy": result.policy,
            "quantization_scheme": result.quantization_scheme,
            "prefetch_enabled": result.prefetch_enabled,
            "metrics": result.metrics,
        }
        for result in results
    ]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(results: list[BenchmarkResult], output_path: Path) -> None:
    """Writes benchmark results to CSV."""
    rows = [_flatten_result(result) for result in results]
    if not rows:
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_summary_table(results: list[BenchmarkResult]) -> str:
    """Returns a human-readable summary table."""
    headers = ["scenario", "workload", "policy", "avg_latency_ms", "miss_count", "bytes_moved"]
    rows = []
    for result in results:
        rows.append(
            [
                result.scenario,
                result.workload,
                result.policy,
                f"{result.metrics['avg_latency_ms']:.3f}",
                str(result.metrics["miss_count"]),
                str(result.metrics["bytes_moved"]),
            ]
        )
    widths = [max(len(header), *(len(row[idx]) for row in rows)) for idx, header in enumerate(headers)]
    lines = [
        " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)),
        "-+-".join("-" * width for width in widths),
    ]
    for row in rows:
        lines.append(" | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)))
    return "\n".join(lines)
