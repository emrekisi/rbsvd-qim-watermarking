"""Measure one serial in-memory embed-and-extract run for each pair and method."""

from __future__ import annotations

import argparse
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))

from controlled_methods import LABELS, METHODS, embed, extract
from experiment_inputs import load_hosts, load_watermarks
from experiment_utils import sample_std, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/runtime"))
    args = parser.parse_args()

    hosts = load_hosts()
    watermarks = load_watermarks()
    if not args.full:
        hosts = OrderedDict([next(iter(hosts.items()))])
        watermarks = OrderedDict([next(iter(watermarks.items()))])

    rows = []
    for host_id, host in hosts.items():
        for watermark_id, payload in watermarks.items():
            for method_id in METHODS:
                embed_started = time.perf_counter_ns()
                marked, _ = embed(method_id, host, payload.bits)
                embed_ms = (time.perf_counter_ns() - embed_started) / 1e6

                extract_started = time.perf_counter_ns()
                recovered = extract(method_id, marked, payload.bits.size)
                extract_ms = (time.perf_counter_ns() - extract_started) / 1e6
                rows.append({
                    "method_id": method_id,
                    "host": host_id,
                    "watermark": watermark_id,
                    "embed_ms": embed_ms,
                    "extract_ms": extract_ms,
                    "total_ms": embed_ms + extract_ms,
                    "attack_free_errors": int(np.count_nonzero(recovered != payload.bits)),
                })

    summary = []
    for method_id in METHODS:
        method_rows = [row for row in rows if row["method_id"] == method_id]
        embed_values = [row["embed_ms"] for row in method_rows]
        extract_values = [row["extract_ms"] for row in method_rows]
        total_values = [row["total_ms"] for row in method_rows]
        summary.append({
            "method_id": method_id,
            "label": LABELS[method_id],
            "pair_count": len(method_rows),
            "mean_embed_ms": float(np.mean(embed_values)),
            "sample_std_embed_ms": sample_std(embed_values),
            "mean_extract_ms": float(np.mean(extract_values)),
            "sample_std_extract_ms": sample_std(extract_values),
            "mean_total_ms": float(np.mean(total_values)),
            "sample_std_total_ms": sample_std(total_values),
            "attack_free_errors": sum(row["attack_free_errors"] for row in method_rows),
        })

    output = args.output.resolve()
    write_csv(output / "runtime_pair_results.csv", rows)
    write_csv(output / "runtime_summary.csv", summary)
    print(f"Results written to {output}")


if __name__ == "__main__":
    main()
