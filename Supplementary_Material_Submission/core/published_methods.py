"""Adapters for the proposed method and nine disclosed author reimplementations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable

import numpy as np

import rbsvd_qim as proposed
from experiment_settings import selected_parameters


BASELINE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "Supplementary_Results"
    / "S3"
)
sys.path.insert(0, str(BASELINE_ROOT))

import chen_2021_qqrd as chen21
import chen_2023_qsvd as chen23
import chen_2026_qqr as chen26
import hu_2020_mixed_modulation as hu20
import li_2020_qdft_qqr as li20
import su_2022_spatial_qim as su22
import sun_2022_qsvd as sun22
import wqsd_wqhd_2026 as eswa26


@dataclass(frozen=True)
class Method:
    id: str
    label: str
    parameter: str
    selected_strength: float
    tuning_anchor: float
    tuning_min: float
    tuning_max: float
    embed_function: Callable
    extract_function: Callable
    fixed_strength: bool = False

    def embed(self, image: np.ndarray, bits: np.ndarray, strength: float | None = None):
        value = self.selected_strength if strength is None else float(strength)
        return self.embed_function(image, bits, value)

    def extract(self, image: np.ndarray, n_bits: int, strength: float | None = None):
        value = self.selected_strength if strength is None else float(strength)
        return self.extract_function(image, n_bits, value)


def methods(delta: float | None = None) -> tuple[Method, ...]:
    point = selected_parameters(delta)
    seed = int(point["seed"])

    def rbsvd_embed(image, bits, strength):
        if not np.isclose(strength, point["Delta"]):
            raise ValueError("The proposed-method strength is fixed at the requested RBSVD setting")
        return proposed.rbq_embed(
            image, bits, d=point["d"], Delta=point["Delta"], r=point["r"],
            seed=seed, K_max=point["K_max"], safety=point["safety"],
            preserve_order=True, alternative_targets=True,
        )

    def rbsvd_extract(image, n_bits, strength):
        return proposed.rbq_extract(
            image, n_bits, d=point["d"], Delta=point["Delta"],
            r=point["r"], seed=seed,
        )

    return (
        Method(
            "rbsvd", "Proposed RBSVD-QIM", "(d, Delta, r, K_max)",
            float(point["Delta"]), float(point["Delta"]),
            float(point["Delta"]), float(point["Delta"]),
            rbsvd_embed, rbsvd_extract, True,
        ),
        Method("sun22", "Sun et al. (2022)", "T", 128.95015683255, 54.0, 5.0, 220.0,
               lambda image, bits, value: sun22.embed(image, bits, step=value, seed=seed),
               lambda image, n, value: sun22.extract(image, n, step=value, seed=seed)),
        Method("chen21", "Chen et al. (2021)", "T_rel", 0.0623958185897081, 0.03, 0.002, 0.20,
               lambda image, bits, value: chen21.embed(image, bits, threshold=value, seed=seed),
               lambda image, n, value: chen21.extract(image, n, threshold=value, seed=seed)),
        Method("chen23", "Chen et al. (2023)", "T", 0.0656149080152585, 0.035, 0.003, 0.15,
               lambda image, bits, value: chen23.embed(image, bits, strength=value, seed=seed, selection_policy="minimum_distance_to_strength_conditional"),
               lambda image, n, value: chen23.extract(image, n, strength=value, seed=seed)),
        Method("chen26", "Chen et al. (2026)", "T", 0.55573182257849, 0.6, 0.03, 1.50,
               lambda image, bits, value: chen26.embed(image, bits, step=value, seed=seed),
               lambda image, n, value: chen26.extract(image, n, step=value, seed=seed)),
        Method("su22", "Su et al. (2022)", "QT_blue", 153.395738829222, 130.0, 8.0, 300.0,
               lambda image, bits, value: su22.embed(image, bits, qt_blue=value, seed=seed),
               lambda image, n, value: su22.extract(image, n, qt_blue=value, seed=seed)),
        Method("li20", "Li et al. (2020)", "phi", 106.645843227148, 89.0, 4.0, 1500.0,
               lambda image, bits, value: li20.embed(image, bits, step=value, seed=seed, entropy_mode="minmax", position_mode="top_left"),
               lambda image, n, value: li20.extract(image, n, step=value, seed=seed, entropy_mode="minmax", position_mode="top_left")),
        Method("hu20", "Hu et al. (2020)", "D", 0.060350664248951, 0.032, 0.001, 0.20,
               lambda image, bits, value: hu20.embed(image, bits, step=value, seed=seed),
               lambda image, n, value: hu20.extract(image, n, step=value, seed=seed)),
        Method("wqsd", "WQSD", "T", 140.266050553951, 50.0, 8.0, 300.0,
               lambda image, bits, value: eswa26.embed(image, bits, step=value, seed=seed, variant="wqsd"),
               lambda image, n, value: eswa26.extract(image, n, step=value, seed=seed, variant="wqsd")),
        Method("wqhd", "WQHD", "T", 142.329178531074, 50.0, 8.0, 300.0,
               lambda image, bits, value: eswa26.embed(image, bits, step=value, seed=seed, variant="wqhd"),
               lambda image, n, value: eswa26.extract(image, n, step=value, seed=seed, variant="wqhd")),
    )
