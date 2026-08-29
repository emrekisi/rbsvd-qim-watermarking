"""Run the d x Delta sensitivity experiment and create its summary heatmaps.

Examples
--------
python operating_point_sensitivity.py --full --workers 8
python operating_point_sensitivity.py --block-sizes 8 --deltas 140 160 --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))

import rbsvd_qim as method
from experiment_inputs import load_hosts, load_watermarks
from experiment_settings import (
    BLOCK_SELECTION_SEED,
    BLOCK_SIZES,
    QUANTIZATION_STEPS,
    SAFETY,
    SELECTED_D,
    SELECTED_DELTA,
    SELECTED_KMAX,
    SELECTED_RANK,
)
from experiment_utils import attack_results, sample_std, summarize_attacks, write_csv, write_json


def evaluate_pair(task: tuple) -> dict:
    d, delta, host_id, host, watermark_id, payload = task
    marked, diagnostics = method.rbq_embed(
        host,
        payload.bits,
        d=d,
        Delta=delta,
        r=SELECTED_RANK,
        seed=BLOCK_SELECTION_SEED,
        K_max=SELECTED_KMAX,
        safety=SAFETY,
        preserve_order=True,
        alternative_targets=True,
    )
    attacks = attack_results(
        marked,
        payload.bits,
        d=d,
        delta=delta,
        rank=SELECTED_RANK,
        seed=BLOCK_SELECTION_SEED,
    )
    return {
        "d": d,
        "Delta": delta,
        "host": host_id,
        "watermark": watermark_id,
        "embedded_blocks": int(payload.bits.size),
        "psnr_db": method.psnr(host, marked),
        "ssim": method.ssim_color(host, marked),
        "n_nonconv": int(diagnostics["n_nonconv"]),
        "n_order_clipped": int(diagnostics["n_order_clipped"]),
        "n_fallback_triggered": int(diagnostics["n_fallback_triggered"]),
        "n_adjacent_target_selected": int(diagnostics["n_adjacent_target_selected"]),
        "fixed_target_update_iterations": int(
            diagnostics["target_selection_evaluation_iterations"]
        ),
        "distortion_bound_violations": int(
            diagnostics["n_distortion_bound_violations"]
        ),
        "attacks": attacks,
    }


def run_experiment(
    output: Path,
    full: bool,
    workers: int,
    requested_block_sizes: tuple[int, ...] | None = None,
    requested_deltas: tuple[float, ...] | None = None,
) -> list[dict]:
    hosts = load_hosts()
    watermarks = load_watermarks()
    custom_grid = requested_block_sizes is not None or requested_deltas is not None
    block_sizes = (
        requested_block_sizes
        if requested_block_sizes is not None
        else (BLOCK_SIZES if full else (SELECTED_D,))
    )
    deltas = (
        requested_deltas
        if requested_deltas is not None
        else (QUANTIZATION_STEPS if full else (SELECTED_DELTA,))
    )
    if any(d <= 0 for d in block_sizes):
        raise ValueError("Block sizes must be positive integers")
    if any(delta <= 0 for delta in deltas):
        raise ValueError("Quantization steps must be positive")
    if not full and not custom_grid:
        hosts = OrderedDict([next(iter(hosts.items()))])
        watermarks = OrderedDict([next(iter(watermarks.items()))])

    tasks = [
        (d, delta, host_id, host, watermark_id, payload)
        for d in block_sizes
        for delta in deltas
        for host_id, host in hosts.items()
        for watermark_id, payload in watermarks.items()
    ]
    print(
        "Purpose: measure the imperceptibility and attack response of the "
        "requested block-size and QIM-step configurations."
    )
    print(f"Evaluating {len(tasks)} host-watermark/configuration combinations.")
    if workers == 1:
        rows = [evaluate_pair(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(evaluate_pair, tasks))
    write_json(output / "operating_point_pair_results.json", rows)
    return rows


def summarize(rows: list[dict], output: Path) -> list[dict]:
    grouped: "OrderedDict[tuple[int, float], list[dict]]" = OrderedDict()
    for row in rows:
        grouped.setdefault((int(row["d"]), float(row["Delta"])), []).append(row)

    summary: list[dict] = []
    attack_rows: list[dict] = []
    for (d, delta), pairs in grouped.items():
        embedded_blocks = sum(int(pair["embedded_blocks"]) for pair in pairs)
        nonconvergence_count = sum(int(pair["n_nonconv"]) for pair in pairs)
        order_clipping_count = sum(int(pair["n_order_clipped"]) for pair in pairs)
        fallback_trigger_count = sum(
            int(pair["n_fallback_triggered"]) for pair in pairs
        )
        adjacent_target_count = sum(
            int(pair["n_adjacent_target_selected"]) for pair in pairs
        )
        fixed_target_updates = sum(
            int(pair["fixed_target_update_iterations"]) for pair in pairs
        )
        bound_violations = sum(
            int(pair["distortion_bound_violations"]) for pair in pairs
        )
        all_attacks = [attack for pair in pairs for attack in pair["attacks"]]
        attack_summary = summarize_attacks(all_attacks)
        attack_means = attack_summary["attack_mean_ber_percent"]
        for attack_name, value in attack_means.items():
            attack_rows.append({
                "d": d,
                "Delta": delta,
                "attack": attack_name,
                "ber_percent": value,
            })
        attack_free_ber = attack_means["No attack"]
        summary.append({
            "d": d,
            "Delta": delta,
            "pairs": len(pairs),
            "embedded_blocks": embedded_blocks,
            "psnr_db_mean": float(np.mean([pair["psnr_db"] for pair in pairs])),
            "psnr_db_pair_sample_std": sample_std(pair["psnr_db"] for pair in pairs),
            "ssim_mean": float(np.mean([pair["ssim"] for pair in pairs])),
            "ssim_pair_sample_std": sample_std(pair["ssim"] for pair in pairs),
            "attack_free_ber_percent": attack_free_ber,
            "family_balanced_ber_percent": attack_summary["family_balanced_ber_percent"],
            "worst_attack": attack_summary["worst_attack"],
            "worst_attack_ber_percent": attack_summary["worst_attack_ber_percent"],
            "brightness_plus10_ber_percent": attack_means["Brightness (+10)"],
            "nonconvergence_count": nonconvergence_count,
            "nonconvergence_rate_percent": 100.0 * nonconvergence_count / embedded_blocks,
            "order_clipping_count": order_clipping_count,
            "order_clipping_rate_percent": 100.0 * order_clipping_count / embedded_blocks,
            "fallback_trigger_count": fallback_trigger_count,
            "fallback_trigger_rate_percent": 100.0 * fallback_trigger_count / embedded_blocks,
            "adjacent_target_selection_count": adjacent_target_count,
            "adjacent_target_selection_rate_percent": 100.0 * adjacent_target_count / embedded_blocks,
            "mean_fixed_target_update_iterations_per_embedded_block": (
                fixed_target_updates / embedded_blocks
            ),
            "distortion_bound_violations": bound_violations,
            "admissible": bool(
                attack_free_ber == 0.0
                and nonconvergence_count == 0
                and bound_violations == 0
            ),
        })

    write_csv(output / "operating_point_summary.csv", summary)
    write_csv(output / "operating_point_attack_summary.csv", attack_rows)
    return summary


def reference_operating_point(summary: list[dict]) -> dict:
    """Record admissibility and the author-retained reference point.

    This finite-grid study is descriptive.  It does not impose a universal
    perceptual threshold or select an optimizer from the tested cells.
    """
    admissible = [row for row in summary if bool(row["admissible"])]
    reference = next(
        (
            row
            for row in summary
            if int(row["d"]) == SELECTED_D
            and float(row["Delta"]) == SELECTED_DELTA
        ),
        None,
    )
    return {
        "record_type": "descriptive finite-grid summary with an author-retained reference point",
        "tested_cell_count": len(summary),
        "admissible_cell_count": len(admissible),
        "admissibility_rule": (
            "Zero attack-free BER, zero non-convergence, and zero "
            "distortion-bound violations."
        ),
        "reference_point": {
            "d": SELECTED_D,
            "Delta": SELECTED_DELTA,
            "present_in_tested_grid": reference is not None,
            "admissible": bool(reference["admissible"]) if reference else None,
            "metrics": reference,
        },
        "interpretation": (
            "The reference point is retained by the authors as an intermediate "
            "imperceptibility-robustness compromise. This record does not "
            "identify a panel-wise, global, or automatically selected optimum."
        ),
    }


def heatmap(
    summary: list[dict], output: Path, selected_d: int, selected_delta: float
) -> None:
    ds = sorted({int(row["d"]) for row in summary})
    deltas = sorted({float(row["Delta"]) for row in summary})
    lookup = {(int(row["d"]), float(row["Delta"])): row for row in summary}
    if len(ds) < 2 or len(deltas) < 2:
        return

    figure, axes = plt.subplots(2, 2, figsize=(9.6, 7.6), constrained_layout=True)
    for axis, key, title, cmap, digits, low_is_dark in (
        (axes[0, 0], "psnr_db_mean", "Mean PSNR (dB)", "cividis", 2, True),
        (axes[0, 1], "ssim_mean", "Mean SSIM", "cividis", 4, True),
        (
            axes[1, 0], "family_balanced_ber_percent",
            "Family-balanced BER (%)", "magma_r", 2, False,
        ),
        (
            axes[1, 1], "worst_attack_ber_percent",
            "Worst-attack BER (%)", "magma_r", 2, False,
        ),
    ):
        values = np.array([[lookup[(d, delta)][key] for delta in deltas] for d in ds])
        image = axis.imshow(values, aspect="auto", cmap=cmap)
        axis.set_xticks(range(len(deltas)), [f"{value:g}" for value in deltas])
        axis.set_yticks(range(len(ds)), [str(value) for value in ds])
        axis.set_xlabel(r"QIM step $\Delta$")
        axis.set_ylabel(r"Block size $d$")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.85, label=title)
        middle = 0.5 * (values.min() + values.max())
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                dark_cell = value < middle if low_is_dark else value > middle
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.{digits}f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if dark_cell else "black",
                )
        if selected_d in ds and selected_delta in deltas:
            axis.add_patch(plt.Rectangle(
                (
                    deltas.index(selected_delta) - 0.5,
                    ds.index(selected_d) - 0.5,
                ),
                1,
                1,
                fill=False,
                edgecolor="#00BFC4",
                linestyle="--",
                linewidth=1.8,
            ))
    figure.savefig(
        output / "Figure_d_delta_sensitivity.pdf", dpi=500, bbox_inches="tight"
    )
    figure.savefig(
        output / "Figure_d_delta_sensitivity.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the complete configured grid on all 32 pairs.",
    )
    parser.add_argument(
        "--block-sizes",
        type=int,
        nargs="+",
        help="Override the tested block sizes and use all 32 pairs.",
    )
    parser.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        help="Override the tested QIM steps and use all 32 pairs.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("results/operating_point"))
    parser.add_argument("--summarize", type=Path, help="Summarize an existing pair-results JSON file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        rows = json.loads(args.summarize.read_text(encoding="utf-8"))
    else:
        rows = run_experiment(
            output,
            args.full,
            args.workers,
            tuple(args.block_sizes) if args.block_sizes else None,
            tuple(args.deltas) if args.deltas else None,
        )
    summary = summarize(rows, output)
    reference = reference_operating_point(summary)
    write_json(output / "operating_point_reference.json", reference)
    heatmap(
        summary,
        output,
        SELECTED_D,
        SELECTED_DELTA,
    )
    print(f"Results written to {output}")


if __name__ == "__main__":
    main()
