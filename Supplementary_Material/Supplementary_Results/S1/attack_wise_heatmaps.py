"""Create the 19 supplementary attack-wise maps from the Section 4.2 CSV file.

Each attack is written as a separate PDF and PNG with an attack-specific
colour scale and annotated cell values. A combined overview is also written
for visual checking; the manuscript uses the 19 separately named files.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core"))

from experiment_settings import ATTACKS


def slug(name: str) -> str:
    """Return the filename stem used by the Supplementary Results figures."""
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")


def validate_grid(
    rows: list[dict[str, str]],
) -> tuple[list[str], list[int], list[float]]:
    if not rows:
        raise ValueError("The attack-summary CSV is empty")
    present = set(row["attack"] for row in rows)
    expected = [attack.name for attack in ATTACKS]
    missing = [name for name in expected if name not in present]
    extra = sorted(present.difference(expected))
    if missing or extra:
        raise ValueError(
            "Attack-summary mismatch: "
            f"missing={missing or 'none'}, extra={extra or 'none'}"
        )

    ds = sorted({int(row["d"]) for row in rows})
    deltas = sorted({float(row["Delta"]) for row in rows})
    keys = {
        (row["attack"], int(row["d"]), float(row["Delta"]))
        for row in rows
    }
    absent_cells = [
        (name, d, delta)
        for name in expected
        for d in ds
        for delta in deltas
        if (name, d, delta) not in keys
    ]
    if absent_cells:
        preview = ", ".join(map(str, absent_cells[:5]))
        raise ValueError(
            f"The attack-summary grid is incomplete ({len(absent_cells)} missing cells): "
            f"{preview}"
        )
    return expected, ds, deltas


def draw(
    axis: plt.Axes,
    values: np.ndarray,
    ds: list[int],
    deltas: list[float],
    title: str,
    *,
    compact: bool = False,
):
    minimum = float(values.min())
    maximum = float(values.max())
    if minimum == maximum:
        lower = 0.0 if minimum >= 0.0 else minimum - 1.0
        upper = 1.0 if maximum == 0.0 else maximum * 1.05
    else:
        lower, upper = minimum, maximum
    image = axis.imshow(
        values, aspect="auto", cmap="magma_r", vmin=lower, vmax=upper
    )
    axis.set_title(title, fontsize=8 if compact else 11)
    axis.set_xticks(range(len(deltas)), [f"{value:g}" for value in deltas])
    axis.set_yticks(range(len(ds)), [str(value) for value in ds])
    axis.tick_params(labelsize=7 if compact else 10.5)
    axis.set_xlabel(r"QIM step $\Delta$", fontsize=8 if compact else 11)
    axis.set_ylabel(r"Block size $d$", fontsize=8 if compact else 11)
    middle = 0.5 * (lower + upper)
    annotation_size = 6.5 if compact else 10.5
    if title == "Brightness (+10)":
        annotation_size = 5.7 if compact else 9.0
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=annotation_size,
                color="white" if value > middle else "black",
            )
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summary_csv", type=Path, help="operating_point_attack_summary.csv"
    )
    parser.add_argument("--output", type=Path, default=Path("results/attack_maps"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.summary_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    attack_names, ds, deltas = validate_grid(rows)
    lookup = {
        (row["attack"], int(row["d"]), float(row["Delta"])): float(
            row["ber_percent"]
        )
        for row in rows
    }
    grids = OrderedDict(
        (
            name,
            np.array(
                [[lookup[(name, d, delta)] for delta in deltas] for d in ds],
                dtype=float,
            ),
        )
        for name in attack_names
    )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for number, name in enumerate(attack_names, start=1):
        figure, axis = plt.subplots(figsize=(4.2, 3.4), constrained_layout=True)
        image = draw(axis, grids[name], ds, deltas, name)
        figure.colorbar(image, ax=axis, shrink=0.85, label="BER (%)")
        stem = f"{number:02d}_{slug(name)}"
        figure.savefig(output / f"{stem}.pdf", dpi=500, bbox_inches="tight")
        figure.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
        plt.close(figure)

    columns = 3
    row_count = math.ceil(len(attack_names) / columns)
    figure, axes = plt.subplots(
        row_count,
        columns,
        figsize=(10.5, 2.9 * row_count),
        constrained_layout=True,
    )
    flat_axes = np.asarray(axes).ravel()
    for axis, name in zip(flat_axes, attack_names):
        image = draw(axis, grids[name], ds, deltas, name, compact=True)
        figure.colorbar(image, ax=axis, shrink=0.78, label="BER (%)")
    for axis in flat_axes[len(attack_names):]:
        axis.set_visible(False)
    figure.savefig(
        output / "attack_wise_heatmaps.pdf", dpi=500, bbox_inches="tight"
    )
    figure.savefig(output / "attack_wise_heatmaps.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {len(attack_names)} individual attack maps and one overview to {output}")


if __name__ == "__main__":
    main()
