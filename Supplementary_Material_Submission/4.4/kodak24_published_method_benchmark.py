"""Run the quality-matched Kodak24 benchmark with nine published methods.

Use ``--retune`` to repeat the PSNR-only global-strength selection before the
benchmark.  Without it, the strengths selected for the reported experiment
are used directly.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))

import rbsvd_qim as metric
from experiment_inputs import load_kodak24, load_watermarks
from experiment_settings import ATTACKS, SELECTED_DELTA, attack_seed_values
from experiment_utils import sample_std, write_csv, write_json
from published_methods import Method, methods


def pairs(full: bool):
    images = load_kodak24()
    watermarks = load_watermarks()
    if not full:
        images = OrderedDict([next(iter(images.items()))])
        watermarks = OrderedDict([next(iter(watermarks.items()))])
    return [
        (image_id, image, watermark_id, payload)
        for image_id, image in images.items()
        for watermark_id, payload in watermarks.items()
    ]


def adapter_by_id(method_id: str, delta: float | None = None) -> Method:
    for adapter in methods(delta):
        if adapter.id == method_id:
            return adapter
    raise ValueError(f"Unknown method id: {method_id}")


def evaluate_psnr(task: tuple) -> float:
    method_id, strength, pair, delta = task
    _, image, _, payload = pair
    adapter = adapter_by_id(method_id, delta)
    marked, _ = adapter.embed(image, payload.bits, strength)
    return float(metric.psnr(image, marked))


def mean_psnr(
    adapter: Method,
    strength: float,
    work: list[tuple],
    workers: int,
    delta: float,
    executor: ProcessPoolExecutor | None = None,
) -> float:
    tasks = [(adapter.id, strength, pair, delta) for pair in work]

    if workers == 1:
        values = [evaluate_psnr(task) for task in tasks]
    else:
        if executor is None:
            with ProcessPoolExecutor(max_workers=workers) as local_executor:
                values = list(local_executor.map(evaluate_psnr, tasks))
        else:
            values = list(executor.map(evaluate_psnr, tasks))
    return float(np.mean(values))


def tune(
    adapter: Method,
    target_psnr: float,
    work: list[tuple],
    workers: int,
    max_evals: int,
    delta: float,
    executor: ProcessPoolExecutor | None = None,
) -> dict:
    if adapter.fixed_strength:
        psnr = mean_psnr(
            adapter, adapter.selected_strength, work, workers, delta, executor
        )
        return {"strength": adapter.selected_strength, "mean_psnr_db": psnr, "target_gap_db": abs(psnr - target_psnr)}

    evaluated: dict[float, float] = {}

    def evaluate(value: float) -> tuple[float, float]:
        clipped = float(min(adapter.tuning_max, max(adapter.tuning_min, value)))
        clipped = float(f"{clipped:.15g}")
        if clipped not in evaluated:
            evaluated[clipped] = mean_psnr(
                adapter, clipped, work, workers, delta, executor
            )
        return clipped, evaluated[clipped]

    anchor, anchor_psnr = evaluate(adapter.tuning_anchor)
    if anchor_psnr > target_psnr:
        low = (anchor, anchor_psnr)
        high = evaluate(anchor * 1.6)
        while high[1] > target_psnr and len(evaluated) < max_evals and high[0] < adapter.tuning_max:
            low, high = high, evaluate(high[0] * 1.6)
    else:
        high = (anchor, anchor_psnr)
        low = evaluate(anchor / 1.6)
        while low[1] < target_psnr and len(evaluated) < max_evals and low[0] > adapter.tuning_min:
            high, low = low, evaluate(low[0] / 1.6)

    while len(evaluated) < max_evals:
        lo, hi = min(low[0], high[0]), max(low[0], high[0])
        if math.isclose(lo, hi, rel_tol=1e-10, abs_tol=1e-12):
            break
        middle = math.sqrt(lo * hi)
        before = len(evaluated)
        row = evaluate(middle)
        if len(evaluated) == before:
            break
        if row[1] > target_psnr:
            low = row
        else:
            high = row

    strength, psnr = min(evaluated.items(), key=lambda item: abs(item[1] - target_psnr))
    return {
        "strength": strength,
        "mean_psnr_db": psnr,
        "target_gap_db": abs(psnr - target_psnr),
        "candidates": [{"strength": key, "mean_psnr_db": value} for key, value in sorted(evaluated.items())],
    }


def select_strengths(
    work: list[tuple],
    workers: int,
    max_evals: int,
    output: Path,
    delta: float,
) -> dict[str, float]:
    adapters = methods(delta)
    rows = {}

    def select(executor: ProcessPoolExecutor | None) -> float:
        target_psnr = mean_psnr(
            adapters[0], adapters[0].selected_strength, work, workers, delta, executor
        )
        for adapter in adapters:
            print(f"Selecting {adapter.parameter} for {adapter.label}.")
            rows[adapter.id] = tune(
                adapter, target_psnr, work, workers, max_evals, delta, executor
            )
        return target_psnr

    if workers == 1:
        target = select(None)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            target = select(executor)
    write_json(output / "global_strength_selection.json", {
        "selection_metric": "absolute difference from the RBSVD mean PSNR over all Kodak24-watermark pairs",
        "requested_delta": delta,
        "target_psnr_db": target,
        "methods": rows,
    })
    return {method_id: row["strength"] for method_id, row in rows.items()}


def evaluate_pair(task: tuple) -> dict:
    method_id, strength, image_id, image, watermark_id, payload, delta = task
    adapter = adapter_by_id(method_id, delta)
    marked, diagnostics = adapter.embed(image, payload.bits, strength)
    attacks = []
    for attack in ATTACKS:
        if attack.name == "Brightness (+10)":
            continue
        for attack_seed in attack_seed_values(attack):
            attacked = attack.apply(marked, attack_seed)
            recovered = adapter.extract(attacked, payload.bits.size, strength)
            errors = int(np.count_nonzero(recovered != payload.bits))
            attacks.append({
                "attack": attack.name,
                "attack_seed": attack_seed,
                "errors": errors,
                "ber_percent": 100.0 * errors / payload.bits.size,
            })
    return {
        "method_id": adapter.id,
        "image": image_id,
        "watermark": watermark_id,
        "strength": strength,
        "psnr_db": float(metric.psnr(image, marked)),
        "ssim": float(metric.ssim_color(image, marked)),
        "diagnostics": diagnostics,
        "attacks": attacks,
    }


def run_benchmark(
    work: list[tuple],
    strengths: dict[str, float],
    workers: int,
    output: Path,
    delta: float,
) -> list[dict]:
    adapters = methods(delta)
    tasks = [
        (
            adapter.id,
            strengths[adapter.id],
            image_id,
            image,
            watermark_id,
            payload,
            delta,
        )
        for adapter in adapters
        for image_id, image, watermark_id, payload in work
    ]
    print(f"Evaluating {len(tasks)} method/image/watermark combinations.")
    if workers == 1:
        rows = [evaluate_pair(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(evaluate_pair, tasks))
    write_json(output / "kodak_benchmark_pair_results.json", rows)
    return rows


def summarize(rows: list[dict], output: Path) -> None:
    rbsvd_rows = [row for row in rows if row["method_id"] == "rbsvd"]
    if not rbsvd_rows:
        raise ValueError("No benchmark rows were supplied for method 'rbsvd'")
    reported_delta = float(rbsvd_rows[0]["strength"])
    adapters = {adapter.id: adapter for adapter in methods(reported_delta)}
    quality = []
    attack_table = []
    feasibility = []
    for method_id in adapters:
        method_rows = [row for row in rows if row["method_id"] == method_id]
        if not method_rows:
            raise ValueError(
                f"No benchmark rows were supplied for method '{method_id}'"
            )
        quality.append({
            "method_id": method_id,
            "label": adapters[method_id].label,
            "global_strength_parameter": adapters[method_id].parameter,
            "global_strength_value": method_rows[0]["strength"],
            "mean_psnr_db": float(np.mean([row["psnr_db"] for row in method_rows])),
            "psnr_sample_std": sample_std(row["psnr_db"] for row in method_rows),
            "mean_ssim": float(np.mean([row["ssim"] for row in method_rows])),
            "ssim_sample_std": sample_std(row["ssim"] for row in method_rows),
        })

        if any(
            "n_negative_target_shift" in row.get("diagnostics", {})
            for row in method_rows
        ):
            carrier_count = sum(
                int(row["diagnostics"].get("n_bits", 0))
                for row in method_rows
            )
            shifted_count = sum(
                int(row["diagnostics"].get("n_negative_target_shift", 0))
                for row in method_rows
            )
            feasibility.append({
                "method_id": method_id,
                "label": adapters[method_id].label,
                "carrier_count": carrier_count,
                "negative_target_shift_count": shifted_count,
                "negative_target_shift_rate_percent": (
                    100.0 * shifted_count / carrier_count
                    if carrier_count else 0.0
                ),
            })

        all_attacks = [attack for row in method_rows for attack in row["attacks"]]
        names = list(OrderedDict((attack["attack"], None) for attack in all_attacks))
        for name in names:
            selected = [attack for attack in all_attacks if attack["attack"] == name]
            attack_table.append({
                "attack": name,
                "method_id": method_id,
                "ber_percent": float(np.mean([attack["ber_percent"] for attack in selected])),
            })
    write_csv(output / "kodak_quality_and_strengths.csv", quality)
    write_csv(output / "kodak_attack_ber.csv", attack_table)
    if feasibility:
        write_csv(output / "nonnegative_target_feasibility.csv", feasibility)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Use all 24 x 4 pairs.")
    parser.add_argument("--retune", action="store_true", help="Repeat global strength selection before testing.")
    parser.add_argument("--max-evals", type=int, default=18)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--delta",
        type=float,
        default=SELECTED_DELTA,
        help="RBSVD reference QIM step. A non-default value requires --retune.",
    )
    parser.add_argument("--output", type=Path, default=Path("results/kodak_benchmark"))
    parser.add_argument("--summarize", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.delta) or args.delta <= 0:
        parser.error("--delta must be a finite positive number")
    if (
        not args.summarize
        and not math.isclose(args.delta, SELECTED_DELTA)
        and not args.retune
    ):
        parser.error("a non-default --delta requires --retune")
    return args


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        rows = json.loads(args.summarize.read_text(encoding="utf-8"))
    else:
        work = pairs(args.full)
        strengths = {
            adapter.id: adapter.selected_strength for adapter in methods(args.delta)
        }
        if args.retune:
            strengths = select_strengths(
                work, args.workers, args.max_evals, output, args.delta
            )
        rows = run_benchmark(
            work, strengths, args.workers, output, args.delta
        )
    summarize(rows, output)
    print(f"Results written to {output}")


if __name__ == "__main__":
    main()
