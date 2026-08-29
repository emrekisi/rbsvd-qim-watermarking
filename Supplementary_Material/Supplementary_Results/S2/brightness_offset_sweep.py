"""Evaluate brightness offsets from -20 to +20 at the retained reference point."""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core"))

import rbsvd_qim as method
from experiment_inputs import WATERMARK_LABELS, load_hosts, load_watermarks
from experiment_settings import selected_parameters
from experiment_utils import write_csv, write_json


OFFSETS = tuple(range(-20, 21))


def evaluate_pair(task: tuple) -> dict:
    host_id, host, watermark_id, payload = task
    point = selected_parameters()
    marked, _ = method.rbq_embed(host, payload.bits, **point)
    values = []
    for offset in OFFSETS:
        attacked = method.atk_bright(marked, offset)
        recovered = method.rbq_extract(
            attacked, payload.bits.size, d=point["d"], Delta=point["Delta"],
            r=point["r"], seed=point["seed"],
        )
        errors = int(np.count_nonzero(recovered != payload.bits))
        values.append({"offset": offset, "errors": errors, "ber_percent": 100.0 * errors / payload.bits.size})
    return {"host": host_id, "watermark": watermark_id, "offsets": values}


def run_experiment(output: Path, full: bool, workers: int) -> list[dict]:
    hosts = load_hosts()
    watermarks = load_watermarks()
    if not full:
        hosts = OrderedDict([next(iter(hosts.items()))])
        watermarks = OrderedDict([next(iter(watermarks.items()))])
    tasks = [
        (host_id, host, watermark_id, payload)
        for host_id, host in hosts.items()
        for watermark_id, payload in watermarks.items()
    ]
    if workers == 1:
        rows = [evaluate_pair(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(evaluate_pair, tasks))
    write_json(output / "brightness_pair_results.json", rows)
    return rows


def summarize(rows: list[dict], output: Path) -> None:
    summary = []
    for offset in OFFSETS:
        values = np.asarray([
            item["ber_percent"]
            for row in rows for item in row["offsets"]
            if int(item["offset"]) == offset
        ], dtype=float)
        summary.append({
            "offset": offset,
            "mean_ber_percent": float(np.mean(values)),
            "q1_ber_percent": float(np.quantile(values, 0.25)),
            "q3_ber_percent": float(np.quantile(values, 0.75)),
            "maximum_pair_ber_percent": float(np.max(values)),
        })
    write_csv(output / "brightness_offset_summary.csv", summary)

    offsets = np.asarray([row["offset"] for row in summary])
    mean = np.asarray([row["mean_ber_percent"] for row in summary])
    q1 = np.asarray([row["q1_ber_percent"] for row in summary])
    q3 = np.asarray([row["q3_ber_percent"] for row in summary])
    maximum = np.asarray([row["maximum_pair_ber_percent"] for row in summary])
    figure, axis = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    axis.plot(offsets, mean, color="#276FBF", marker="o", markersize=3, label="Mean BER")
    axis.fill_between(offsets, q1, q3, color="#276FBF", alpha=0.18, label="Interquartile range")
    axis.plot(offsets, maximum, color="#C23B22", linestyle="--", label="Maximum pair BER")
    zero = int(np.where(offsets == 0)[0][0])
    axis.plot([0], [mean[zero]], marker="o", markerfacecolor="white", markeredgecolor="#222222", markersize=7, linestyle="none", label="No-offset reference")
    axis.axvline(10, color="#333333", linestyle=":", linewidth=1.0, label="Brightness +10")
    axis.set_xlabel("Brightness offset")
    axis.set_ylabel("BER (%)")
    axis.grid(alpha=0.25)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=2, frameon=False)
    figure.savefig(
        output / "brightness_offset_overall.pdf", dpi=500, bbox_inches="tight"
    )
    figure.savefig(output / "brightness_offset_overall.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    pair_values = {
        (row["host"], row["watermark"]): {
            int(item["offset"]): float(item["ber_percent"])
            for item in row["offsets"]
        }
        for row in rows
    }
    host_order = list(dict.fromkeys(row["host"] for row in rows))
    watermark_order = list(dict.fromkeys(row["watermark"] for row in rows))
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 4.3), sharey=True, constrained_layout=True)
    colors = list(plt.get_cmap("tab10").colors)
    for index, host_id in enumerate(host_order):
        values = [
            np.mean([pair_values[(host_id, watermark_id)][offset] for watermark_id in watermark_order])
            for offset in offsets
        ]
        axes[0].plot(offsets, values, linewidth=1.0, color=colors[index % len(colors)], label=host_id.capitalize())
    for index, watermark_id in enumerate(watermark_order):
        values = [
            np.mean([pair_values[(host_id, watermark_id)][offset] for host_id in host_order])
            for offset in offsets
        ]
        label = "SAÜ" if watermark_id == "reference_logo" else WATERMARK_LABELS[watermark_id]
        axes[1].plot(offsets, values, linewidth=1.2, color=colors[index % len(colors)], label=label)
    for axis, title in zip(axes, ("Host means (four watermarks)", "Watermark means (eight hosts)")):
        axis.axvline(10, color="#333333", linestyle=":", linewidth=0.9)
        axis.set_title(title)
        axis.set_xlabel("Brightness offset")
        axis.set_xlim(-20.8, 20.8)
        axis.set_ylim(-2, 102)
        axis.set_xticks(np.arange(-20, 21, 10))
        axis.grid(alpha=0.25)
        axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.16), ncol=2, frameon=False)
    axes[0].set_ylabel("Mean BER (%)")
    figure.savefig(
        output / "brightness_offset_heterogeneity.pdf", dpi=500, bbox_inches="tight"
    )
    figure.savefig(output / "brightness_offset_heterogeneity.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("results/brightness_offset"))
    parser.add_argument("--summarize", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = (
        json.loads(args.summarize.read_text(encoding="utf-8"))
        if args.summarize else run_experiment(output, args.full, args.workers)
    )
    summarize(rows, output)
    print(f"Results written to {output}")


if __name__ == "__main__":
    main()
