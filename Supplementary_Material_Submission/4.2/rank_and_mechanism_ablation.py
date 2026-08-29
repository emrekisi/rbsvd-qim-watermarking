"""Evaluate singular-value rank and the order-preservation/fallback mechanisms."""

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

import rbsvd_qim as method
from experiment_inputs import load_hosts, load_watermarks
from experiment_settings import (
    RANKS,
    SAFETY,
    SELECTED_D,
    SELECTED_DELTA,
    SELECTED_KMAX,
)
from experiment_utils import attack_results, sample_std, summarize_attacks, write_csv, write_json


BLOCK_SEEDS = tuple(range(2026, 2036))
MECHANISM_STRESS_RANK = RANKS[1]


def configurations() -> list[dict]:
    rows = [
        {"panel": "rank", "rank": rank, "preserve_order": True, "fallback": True}
        for rank in RANKS
    ]
    rows.extend([
        {
            "panel": "mechanism", "rank": MECHANISM_STRESS_RANK,
            "preserve_order": False, "fallback": False,
        },
        {
            "panel": "mechanism", "rank": MECHANISM_STRESS_RANK,
            "preserve_order": True, "fallback": False,
        },
        {
            "panel": "mechanism", "rank": MECHANISM_STRESS_RANK,
            "preserve_order": False, "fallback": True,
        },
    ])
    return rows


def evaluate_pair(task: tuple) -> dict:
    config, block_seed, host_id, host, watermark_id, payload = task
    marked, diagnostics = method.rbq_embed(
        host,
        payload.bits,
        d=SELECTED_D,
        Delta=SELECTED_DELTA,
        r=config["rank"],
        seed=block_seed,
        K_max=SELECTED_KMAX,
        safety=SAFETY,
        preserve_order=config["preserve_order"],
        alternative_targets=config["fallback"],
    )
    attacks = attack_results(
        marked,
        payload.bits,
        d=SELECTED_D,
        delta=SELECTED_DELTA,
        rank=config["rank"],
        seed=block_seed,
    )
    return {
        **config,
        "block_seed": block_seed,
        "host": host_id,
        "watermark": watermark_id,
        "embedded_blocks": int(payload.bits.size),
        "psnr_db": float(method.psnr(host, marked)),
        "ssim": float(method.ssim_color(host, marked)),
        "n_nonconv": int(diagnostics["n_nonconv"]),
        "n_order_clipped": int(diagnostics["n_order_clipped"]),
        "n_fallback_triggered": int(diagnostics["n_fallback_triggered"]),
        "n_adjacent_target_selected": int(diagnostics["n_adjacent_target_selected"]),
        "distortion_bound_violations": int(
            diagnostics["n_distortion_bound_violations"]
        ),
        "attacks": attacks,
    }


def run_experiment(output: Path, full: bool, workers: int) -> list[dict]:
    hosts = load_hosts()
    watermarks = load_watermarks()
    configs = configurations()
    seeds = BLOCK_SEEDS
    if not full:
        hosts = OrderedDict([next(iter(hosts.items()))])
        watermarks = OrderedDict([next(iter(watermarks.items()))])
        configs = configs[:1]
        seeds = seeds[:1]
    tasks = [
        (config, seed, host_id, host, watermark_id, payload)
        for config in configs
        for seed in seeds
        for host_id, host in hosts.items()
        for watermark_id, payload in watermarks.items()
    ]
    print(f"Evaluating {len(tasks)} host-watermark/configuration/seed combinations.")
    if workers == 1:
        rows = [evaluate_pair(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(evaluate_pair, tasks))
    write_json(output / "rank_mechanism_pair_results.json", rows)
    return rows


def summarize(rows: list[dict], output: Path) -> list[dict]:
    configurations_by_key = OrderedDict()
    for row in rows:
        key = (
            row["panel"], int(row["rank"]),
            bool(row["preserve_order"]), bool(row["fallback"]),
        )
        configurations_by_key.setdefault(key, []).append(row)

    table: list[dict] = []
    for key, config_rows in configurations_by_key.items():
        panel, rank, preserve_order, fallback = key
        per_seed: list[dict] = []
        for seed in sorted({int(row["block_seed"]) for row in config_rows}):
            seed_rows = [row for row in config_rows if int(row["block_seed"]) == seed]
            attack_summary = summarize_attacks(
                attack for row in seed_rows for attack in row["attacks"]
            )
            attack_free = attack_summary["attack_mean_ber_percent"]["No attack"]
            embedded_blocks = sum(int(row["embedded_blocks"]) for row in seed_rows)
            nonconvergence_count = sum(int(row["n_nonconv"]) for row in seed_rows)
            bound_violations = sum(
                int(row["distortion_bound_violations"]) for row in seed_rows
            )
            per_seed.append({
                "attack_free": attack_free,
                "nonconv": 100.0 * nonconvergence_count / embedded_blocks,
                "clipping": 100.0 * sum(
                    int(row["n_order_clipped"]) for row in seed_rows
                ) / embedded_blocks,
                "fallback_trigger": 100.0 * sum(
                    int(row["n_fallback_triggered"]) for row in seed_rows
                ) / embedded_blocks,
                "adjacent_target_selection": 100.0 * sum(
                    int(row["n_adjacent_target_selected"]) for row in seed_rows
                ) / embedded_blocks,
                "psnr": float(np.mean([row["psnr_db"] for row in seed_rows])),
                "ssim": float(np.mean([row["ssim"] for row in seed_rows])),
                "family": attack_summary["family_balanced_ber_percent"],
                "worst": attack_summary["worst_attack_ber_percent"],
                "bound_violations": bound_violations,
                "admissible": bool(
                    attack_free == 0.0
                    and bound_violations == 0
                ),
            })

        record = {
            "panel": panel,
            "rank": rank,
            "preserve_order": preserve_order,
            "fallback": fallback,
            "block_seed_count": len(per_seed),
            "block_seeds": ",".join(str(seed) for seed in sorted({int(row["block_seed"]) for row in config_rows})),
            "pair_count_per_seed": len(config_rows) // len(per_seed),
            "distortion_bound_violations": sum(
                int(seed_row["bound_violations"]) for seed_row in per_seed
            ),
            "admissible_for_all_block_seeds": all(
                bool(seed_row["admissible"]) for seed_row in per_seed
            ),
        }
        for label, key_name in (
            ("attack_free_ber_percent", "attack_free"),
            ("nonconvergence_rate_percent", "nonconv"),
            ("order_clipping_rate_percent", "clipping"),
            ("fallback_trigger_rate_percent", "fallback_trigger"),
            (
                "adjacent_target_selection_rate_percent",
                "adjacent_target_selection",
            ),
            ("mean_psnr_db", "psnr"),
            ("mean_ssim", "ssim"),
            ("family_balanced_ber_percent", "family"),
            ("worst_attack_ber_percent", "worst"),
        ):
            values = [seed_row[key_name] for seed_row in per_seed]
            record[f"{label}_mean"] = float(np.mean(values))
            record[f"{label}_sample_std"] = sample_std(values)
        table.append(record)

    write_csv(output / "rank_mechanism_summary.csv", table)
    return table


def select_rank(table: list[dict]) -> dict:
    eligible = [
        row for row in table
        if row["panel"] == "rank"
        and float(row["attack_free_ber_percent_mean"]) == 0.0
        and int(row["distortion_bound_violations"]) == 0
    ]
    if not eligible:
        raise ValueError("No eligible singular-value rank was found")
    selected = min(eligible, key=lambda row: int(row["rank"]))
    return {
        "selected_rank": int(selected["rank"]),
        "eligible_ranks": sorted(int(row["rank"]) for row in eligible),
        "selection_rule": (
            "Among full-mechanism rank rows with zero attack-free BER and zero "
            "distortion-bound violations, select the smallest rank. "
            "Non-convergence remains diagnostic."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Run all configurations, pairs, and block seeds.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("results/rank_mechanism"))
    parser.add_argument("--summarize", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        rows = json.loads(args.summarize.read_text(encoding="utf-8"))
    else:
        rows = run_experiment(output, args.full, args.workers)
    summary = summarize(rows, output)
    write_json(output / "rank_selection.json", select_rank(summary))
    print(f"Results written to {output}")


if __name__ == "__main__":
    main()
