"""Evaluate the correction budget K_max on the complete d x Delta grid."""

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
    BLOCK_SELECTION_SEED,
    BLOCK_SIZES,
    CORRECTION_BUDGETS,
    QUANTIZATION_STEPS,
    SAFETY,
    SELECTED_D,
    SELECTED_DELTA,
    SELECTED_RANK,
)
from experiment_utils import sample_std, write_csv, write_json


VALIDATION_BUDGETS = (2, 3, 4)
VALIDATION_SEEDS = tuple(range(2026, 2036))


def evaluate_pair(task: tuple) -> dict:
    phase, kmax, d, delta, block_seed, host_id, host, watermark_id, payload = task
    marked, diagnostics = method.rbq_embed(
        host,
        payload.bits,
        d=d,
        Delta=delta,
        r=SELECTED_RANK,
        seed=block_seed,
        K_max=kmax,
        safety=SAFETY,
        preserve_order=True,
        alternative_targets=True,
    )
    recovered = method.rbq_extract(
        marked, payload.bits.size, d=d, Delta=delta,
        r=SELECTED_RANK, seed=block_seed,
    )
    return {
        "phase": phase,
        "K_max": kmax,
        "d": d,
        "Delta": delta,
        "block_seed": block_seed,
        "host": host_id,
        "watermark": watermark_id,
        "embedded_blocks": int(payload.bits.size),
        "n_nonconv": int(diagnostics["n_nonconv"]),
        "attack_free_errors": int(np.count_nonzero(recovered != payload.bits)),
        "fixed_target_update_iterations": int(
            diagnostics["target_selection_evaluation_iterations"]
        ),
        "distortion_bound_violations": int(diagnostics["n_distortion_bound_violations"]),
    }


def tasks(full: bool) -> list[tuple]:
    hosts = load_hosts()
    watermarks = load_watermarks()
    if not full:
        hosts = OrderedDict([next(iter(hosts.items()))])
        watermarks = OrderedDict([next(iter(watermarks.items()))])
        budgets, ds, deltas = CORRECTION_BUDGETS[:2], BLOCK_SIZES[:1], QUANTIZATION_STEPS[:1]
        validation_budgets, validation_seeds = VALIDATION_BUDGETS[:1], VALIDATION_SEEDS[:1]
    else:
        budgets, ds, deltas = CORRECTION_BUDGETS, BLOCK_SIZES, QUANTIZATION_STEPS
        validation_budgets, validation_seeds = VALIDATION_BUDGETS, VALIDATION_SEEDS

    rows = [
        ("factorial", kmax, d, delta, BLOCK_SELECTION_SEED, host_id, host, watermark_id, payload)
        for kmax in budgets
        for d in ds
        for delta in deltas
        for host_id, host in hosts.items()
        for watermark_id, payload in watermarks.items()
    ]
    rows.extend([
        ("multiseed", kmax, SELECTED_D, SELECTED_DELTA, seed, host_id, host, watermark_id, payload)
        for kmax in validation_budgets
        for seed in validation_seeds
        for host_id, host in hosts.items()
        for watermark_id, payload in watermarks.items()
    ])
    return rows


def run_experiment(output: Path, full: bool, workers: int) -> list[dict]:
    work = tasks(full)
    print(f"Evaluating {len(work)} host-watermark/budget combinations.")
    if workers == 1:
        rows = [evaluate_pair(task) for task in work]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(evaluate_pair, work))
    write_json(output / "correction_budget_pair_results.json", rows)
    return rows


def aggregate_unit(rows: list[dict]) -> dict:
    return {
        "embedded_blocks": sum(int(row["embedded_blocks"]) for row in rows),
        "n_nonconv": sum(row["n_nonconv"] for row in rows),
        "fixed_target_update_iterations": sum(
            row["fixed_target_update_iterations"] for row in rows
        ),
        "attack_free_errors": sum(row["attack_free_errors"] for row in rows),
        "distortion_bound_violations": sum(row["distortion_bound_violations"] for row in rows),
    }


def select_budget(table: list[dict]) -> dict:
    eligible = [
        row for row in table
        if int(row["attack_free_errors"]) == 0
        and int(row["distortion_bound_violations"]) == 0
    ]
    eligible.sort(key=lambda row: int(row["K_max"]))
    for current, following in zip(eligible, eligible[1:]):
        current_affected = int(str(current["affected_cells"]).split("/", 1)[0])
        following_affected = int(str(following["affected_cells"]).split("/", 1)[0])
        if current_affected == following_affected:
            return {
                "selected_K_max": int(current["K_max"]),
                "next_tested_K_max": int(following["K_max"]),
                "affected_cells_at_plateau": current_affected,
                "selection_rule": (
                    "Among budgets with zero attack-free errors and zero "
                    "distortion-bound violations, select the first tested K_max "
                    "whose affected-cell count equals that of the next budget."
                ),
            }
    raise ValueError("No correction-budget plateau was found in the tested grid")


def summarize(rows: list[dict], output: Path) -> None:
    factorial = [row for row in rows if row["phase"] == "factorial"]
    units: "OrderedDict[tuple, list[dict]]" = OrderedDict()
    for row in factorial:
        units.setdefault((row["K_max"], row["d"], row["Delta"]), []).append(row)
    aggregated = {key: aggregate_unit(value) for key, value in units.items()}

    table: list[dict] = []
    for kmax in sorted({int(row["K_max"]) for row in factorial}):
        cells = [value for key, value in aggregated.items() if int(key[0]) == kmax]
        blocks = sum(cell["embedded_blocks"] for cell in cells)
        table.append({
            "K_max": kmax,
            "affected_cells": f"{sum(cell['n_nonconv'] > 0 for cell in cells)}/{len(cells)}",
            "nonconvergence_rate_percent": 100.0 * sum(cell["n_nonconv"] for cell in cells) / blocks,
            "mean_fixed_target_update_iterations_per_embedded_block": sum(
                cell["fixed_target_update_iterations"] for cell in cells
            ) / blocks,
            "attack_free_errors": sum(cell["attack_free_errors"] for cell in cells),
            "distortion_bound_violations": sum(cell["distortion_bound_violations"] for cell in cells),
            "admissible_cells": (
                f"{sum(cell['attack_free_errors'] == 0 and cell['n_nonconv'] == 0 and cell['distortion_bound_violations'] == 0 for cell in cells)}"
                f"/{len(cells)}"
            ),
        })
    write_csv(output / "correction_budget_summary.csv", table)
    write_json(output / "correction_budget_selection.json", select_budget(table))

    validation = [row for row in rows if row["phase"] == "multiseed"]
    validation_units: "OrderedDict[tuple, list[dict]]" = OrderedDict()
    for row in validation:
        validation_units.setdefault((row["K_max"], row["block_seed"]), []).append(row)
    validation_rows: list[dict] = []
    for kmax in sorted({int(row["K_max"]) for row in validation}):
        seed_values = []
        for (budget, seed), pair_rows in validation_units.items():
            if int(budget) != kmax:
                continue
            unit = aggregate_unit(pair_rows)
            seed_values.append((seed, 100.0 * unit["n_nonconv"] / unit["embedded_blocks"], unit))
        rates = [value[1] for value in seed_values]
        validation_rows.append({
            "K_max": kmax,
            "block_seed_count": len(seed_values),
            "block_seeds": ",".join(str(value[0]) for value in seed_values),
            "nonconvergence_rate_percent_mean": float(np.mean(rates)),
            "nonconvergence_rate_percent_sample_std": sample_std(rates),
            "nonconvergence_rate_percent_min": min(rates),
            "nonconvergence_rate_percent_max": max(rates),
            "attack_free_errors": sum(value[2]["attack_free_errors"] for value in seed_values),
            "distortion_bound_violations": sum(value[2]["distortion_bound_violations"] for value in seed_values),
            "admissible_for_all_block_seeds": all(
                value[2]["attack_free_errors"] == 0
                and value[2]["n_nonconv"] == 0
                and value[2]["distortion_bound_violations"] == 0
                for value in seed_values
            ),
        })
    write_csv(output / "correction_budget_multiseed_check.csv", validation_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("results/correction_budget"))
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
