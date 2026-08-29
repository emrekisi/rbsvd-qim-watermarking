"""Compare RBSVD-QIM with quality-matched Channel-SVD-QIM and QSVD-QIM."""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))

import rbsvd_qim as metric
from controlled_methods import LABELS, METHODS, embed, extract, quantization_step
from experiment_inputs import load_hosts, load_watermarks
from experiment_settings import (
    ATTACKS,
    ATTACK_SEEDS,
    SELECTED_DELTA,
    attack_seed_values,
    family_balanced_ber,
)
from experiment_utils import sample_std, write_csv, write_json


def evaluate_pair(task: tuple) -> dict:
    host_id, host, watermark_id, payload, delta = task
    methods = {}
    for method_id in METHODS:
        marked, diagnostics = embed(method_id, host, payload.bits, delta)
        attacks = []
        for attack in ATTACKS:
            for attack_seed in attack_seed_values(attack):
                attacked = attack.apply(marked, attack_seed)
                recovered = extract(method_id, attacked, payload.bits.size, delta)
                errors = int(np.count_nonzero(recovered != payload.bits))
                attacks.append({
                    "attack": attack.name,
                    "family": attack.family or "excluded",
                    "attack_seed": attack_seed,
                    "errors": errors,
                    "ber_percent": 100.0 * errors / payload.bits.size,
                })
        methods[method_id] = {
            "Delta": quantization_step(method_id, delta),
            "psnr_db": float(metric.psnr(host, marked)),
            "ssim": float(metric.ssim_color(host, marked)),
            "diagnostics": diagnostics,
            "attacks": attacks,
        }
    return {"host": host_id, "watermark": watermark_id, "methods": methods}


def run_experiment(
    output: Path,
    full: bool,
    workers: int,
    delta: float,
) -> list[dict]:
    hosts = load_hosts()
    watermarks = load_watermarks()
    if not full:
        hosts = OrderedDict([next(iter(hosts.items()))])
        watermarks = OrderedDict([next(iter(watermarks.items()))])
    tasks = [
        (host_id, host, watermark_id, payload, delta)
        for host_id, host in hosts.items()
        for watermark_id, payload in watermarks.items()
    ]
    if workers == 1:
        rows = [evaluate_pair(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(evaluate_pair, tasks))
    write_json(output / "controlled_comparison_pair_results.json", rows)
    return rows


def summarize(rows: list[dict], output: Path) -> None:
    quality_rows = []
    attack_table = []
    watermark_table = []
    watermark_kinds = {
        watermark_id: payload.kind
        for watermark_id, payload in load_watermarks().items()
    }
    available_watermarks = list(OrderedDict(
        (row["watermark"], None) for row in rows
    ))
    for method_id in METHODS:
        method_rows = [row["methods"][method_id] for row in rows]
        all_attacks = [attack for row in method_rows for attack in row["attacks"]]
        attack_names = list(OrderedDict((attack["attack"], None) for attack in all_attacks))
        attack_means = OrderedDict()
        attack_sds = OrderedDict()
        for name in attack_names:
            selected = [attack for attack in all_attacks if attack["attack"] == name]
            attack_means[name] = float(np.mean([attack["ber_percent"] for attack in selected]))
            stochastic_seed_means = []
            if len({attack["attack_seed"] for attack in selected}) > 1:
                for seed in ATTACK_SEEDS:
                    seed_rows = [attack["ber_percent"] for attack in selected if attack["attack_seed"] == seed]
                    stochastic_seed_means.append(float(np.mean(seed_rows)))
            attack_sds[name] = sample_std(stochastic_seed_means) if stochastic_seed_means else float("nan")

        family_score, _ = family_balanced_ber(dict(attack_means))
        quality_rows.append({
            "method_id": method_id,
            "label": LABELS[method_id],
            "Delta": float(method_rows[0]["Delta"]),
            "psnr_db_mean": float(np.mean([row["psnr_db"] for row in method_rows])),
            "psnr_db_sample_std_across_pairs": sample_std(row["psnr_db"] for row in method_rows),
            "ssim_mean": float(np.mean([row["ssim"] for row in method_rows])),
            "ssim_sample_std_across_pairs": sample_std(row["ssim"] for row in method_rows),
            "family_balanced_ber_percent": family_score,
        })
        for name in attack_names:
            attack_table.append({
                "attack": name,
                "method_id": method_id,
                "ber_percent": attack_means[name],
                "ber_percent_seed_sd": attack_sds[name],
            })

        for watermark_id in available_watermarks:
            watermark_method_rows = [
                row["methods"][method_id]
                for row in rows
                if row["watermark"] == watermark_id
            ]
            watermark_attack_means = {}
            for name in attack_names:
                selected = [
                    attack["ber_percent"]
                    for row in watermark_method_rows
                    for attack in row["attacks"]
                    if attack["attack"] == name
                ]
                watermark_attack_means[name] = float(np.mean(selected))
            watermark_score, _ = family_balanced_ber(watermark_attack_means)
            watermark_table.append({
                "method_id": method_id,
                "label": LABELS[method_id],
                "watermark_id": watermark_id,
                "watermark_kind": watermark_kinds[watermark_id],
                "family_balanced_ber_percent": watermark_score,
            })
    write_csv(output / "controlled_quality_summary.csv", quality_rows)
    write_csv(output / "controlled_attack_summary.csv", attack_table)
    write_csv(output / "controlled_watermark_summary.csv", watermark_table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--delta",
        type=float,
        default=SELECTED_DELTA,
        help="Reference QIM step; the Channel-SVD step remains Delta/sqrt(3).",
    )
    parser.add_argument("--output", type=Path, default=Path("results/controlled_comparison"))
    parser.add_argument("--summarize", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not np.isfinite(args.delta) or args.delta <= 0:
        raise ValueError("--delta must be a finite positive number")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = (
        json.loads(args.summarize.read_text(encoding="utf-8"))
        if args.summarize
        else run_experiment(output, args.full, args.workers, args.delta)
    )
    summarize(rows, output)
    print(f"Results written to {output}")


if __name__ == "__main__":
    main()
