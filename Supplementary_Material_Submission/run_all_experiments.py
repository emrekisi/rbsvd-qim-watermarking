"""Run the section-specific experiment entry files in manuscript order."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

EXPERIMENTS = (
    ("4.2", "4.2/correction_budget_elbow.py", "correction_budget", True),
    ("4.2", "4.2/operating_point_sensitivity.py", "operating_point", True),
    ("4.2", "4.2/rank_and_mechanism_ablation.py", "rank_mechanism", True),
    ("4.3", "4.3/controlled_robustness_comparison.py", "controlled_comparison", True),
    ("4.3", "4.3/end_to_end_runtime.py", "runtime", False),
    ("4.4", "4.4/kodak24_published_method_benchmark.py", "kodak_benchmark", True),
    (
        "Supplementary_Results",
        "Supplementary_Results/S2/brightness_offset_sweep.py",
        "brightness_offset",
        True,
    ),
)


def create_attack_maps(output_root: Path) -> None:
    operating_summary = (
        output_root / "operating_point" / "operating_point_attack_summary.csv"
    )
    if not operating_summary.is_file():
        raise FileNotFoundError(
            "Section S1 requires "
            f"{operating_summary}. Run the full Section 4.2 operating-point "
            "experiment first or use the same --output-root."
        )
    subprocess.run([
        sys.executable,
        str(ROOT / "Supplementary_Results/S1/attack_wise_heatmaps.py"),
        str(operating_summary),
        "--output",
        str(output_root / "attack_maps"),
    ], cwd=ROOT, check=True)


def run_alternative_delta(
    output_root: Path,
    delta: float,
    workers: int,
) -> None:
    delta_name = f"{delta:g}"
    s4_root = output_root / f"S4_delta{delta_name}"
    commands = (
        (
            "4.3/controlled_robustness_comparison.py",
            [
                "--full",
                "--delta",
                delta_name,
                "--workers",
                str(workers),
                "--output",
                str(s4_root / "controlled_comparison"),
            ],
        ),
        (
            "4.4/kodak24_published_method_benchmark.py",
            [
                "--full",
                "--delta",
                delta_name,
                "--retune",
                "--max-evals",
                "18",
                "--workers",
                str(workers),
                "--output",
                str(s4_root / "kodak_benchmark"),
            ],
        ),
    )
    for relative_path, options in commands:
        print(f"Running {relative_path} for Supplementary Results S4", flush=True)
        subprocess.run(
            [sys.executable, str(ROOT / relative_path), *options],
            cwd=ROOT,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--retune-kodak",
        action="store_true",
        help="Repeat the Kodak24 PSNR-only strength selection before benchmarking.",
    )
    parser.add_argument(
        "--section",
        choices=("all", "4.2", "4.3", "4.4", "Supplementary_Results", "S4"),
        default="all",
    )
    parser.add_argument(
        "--s4-deltas",
        type=float,
        nargs="+",
        default=(120.0, 160.0),
        help=(
            "Auxiliary QIM steps used by Supplementary Results S4 "
            "(default: 120 160)."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.retune_kodak and not args.full:
        parser.error("--retune-kodak requires --full")
    if any(not math.isfinite(delta) or delta <= 0 for delta in args.s4_deltas):
        parser.error("every --s4-deltas value must be finite and positive")
    if args.section == "S4" and not args.full:
        parser.error("--section S4 requires --full")
    output_root = args.output_root.resolve()

    if args.section == "S4":
        for delta in args.s4_deltas:
            run_alternative_delta(output_root, delta, args.workers)
        return

    if args.section == "Supplementary_Results":
        create_attack_maps(output_root)

    for section, relative_path, output_name, parallel in EXPERIMENTS:
        if args.section != "all" and section != args.section:
            continue
        script = ROOT / relative_path
        output = output_root / output_name
        command = [sys.executable, str(script), "--output", str(output)]
        if args.full:
            command.append("--full")
        if section == "4.4" and args.retune_kodak:
            command.extend(["--retune", "--max-evals", "18"])
        if parallel:
            command.extend(["--workers", str(args.workers)])
        print(f"Running {relative_path}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        if args.section == "all" and relative_path == "4.2/operating_point_sensitivity.py":
            create_attack_maps(output_root)

    if args.section == "all" and args.full:
        for delta in args.s4_deltas:
            run_alternative_delta(output_root, delta, args.workers)


if __name__ == "__main__":
    main()
