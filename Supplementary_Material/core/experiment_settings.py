"""Shared settings and attack definitions used by the experiment scripts."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import numpy as np

import rbsvd_qim as method


BLOCK_SIZES = (4, 6, 8, 10, 12, 16)
QUANTIZATION_STEPS = (40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0)
RANKS = (1, 2, 3, 4, 5)
CORRECTION_BUDGETS = (1, 2, 3, 4, 5, 10)
ATTACK_SEEDS = tuple(range(7, 17))

SELECTED_D = 8
SELECTED_DELTA = 140.0
SELECTED_RANK = 1
SELECTED_KMAX = 2
BLOCK_SELECTION_SEED = 2026
SAFETY = 0.5


@dataclass(frozen=True)
class Attack:
    name: str
    family: str | None
    apply: Callable[[np.ndarray, int], np.ndarray]
    stochastic: bool = False


ATTACKS = (
    Attack("No attack", None, lambda image, seed: image.copy()),
    Attack("JPEG (Q=90)", "compression", lambda image, seed: method.atk_jpeg(image, 90)),
    Attack("JPEG (Q=70)", "compression", lambda image, seed: method.atk_jpeg(image, 70)),
    Attack("JPEG (Q=50)", "compression", lambda image, seed: method.atk_jpeg(image, 50)),
    Attack("JPEG (Q=30)", "compression", lambda image, seed: method.atk_jpeg(image, 30)),
    Attack("JPEG2000 (CR=2)", "compression", lambda image, seed: method.atk_jpeg2000(image, 2)),
    Attack("JPEG2000 (CR=4)", "compression", lambda image, seed: method.atk_jpeg2000(image, 4)),
    Attack("JPEG2000 (CR=8)", "compression", lambda image, seed: method.atk_jpeg2000(image, 8)),
    Attack("Gaussian noise (σ=5)", "gaussian", lambda image, seed: method.atk_gauss(image, 5, seed), True),
    Attack("Gaussian noise (σ=10)", "gaussian", lambda image, seed: method.atk_gauss(image, 10, seed), True),
    Attack("Salt & pepper (1%)", "salt-and-pepper", lambda image, seed: method.atk_sp(image, 0.01, seed), True),
    Attack("Salt & pepper (2%)", "salt-and-pepper", lambda image, seed: method.atk_sp(image, 0.02, seed), True),
    Attack("Median filter (3×3)", "filtering", lambda image, seed: method.atk_median(image, 3)),
    Attack("Average filter (3×3)", "filtering", lambda image, seed: method.atk_average(image, 3)),
    Attack("Zero-masking (25%)", "geometry/masking", lambda image, seed: method.atk_crop(image, 0.25)),
    Attack("Resize-back (0.5×)", "geometry/masking", lambda image, seed: method.atk_scale(image, 0.5)),
    Attack("Inverse-registered rotation (2°)", "geometry/masking", lambda image, seed: method.atk_rotate(image, 2.0)),
    Attack("Brightness (+10)", None, lambda image, seed: method.atk_bright(image, 10)),
    Attack("Sharpening", "filtering", lambda image, seed: method.atk_sharpen(image)),
)

FAMILY_ORDER = (
    "compression",
    "gaussian",
    "salt-and-pepper",
    "filtering",
    "geometry/masking",
)


def selected_parameters(delta: float | None = None) -> dict[str, float | int | bool]:
    selected_delta = SELECTED_DELTA if delta is None else float(delta)
    if not np.isfinite(selected_delta) or selected_delta <= 0:
        raise ValueError("Delta must be a finite positive number")
    return {
        "d": SELECTED_D,
        "Delta": selected_delta,
        "r": SELECTED_RANK,
        "K_max": SELECTED_KMAX,
        "seed": BLOCK_SELECTION_SEED,
        "safety": SAFETY,
        "preserve_order": True,
        "alternative_targets": True,
    }


def attack_seed_values(attack: Attack) -> tuple[int, ...]:
    return ATTACK_SEEDS if attack.stochastic else (ATTACK_SEEDS[0],)


def family_balanced_ber(attack_means: dict[str, float]) -> tuple[float, dict[str, float]]:
    family_values: "OrderedDict[str, list[float]]" = OrderedDict(
        (family, []) for family in FAMILY_ORDER
    )
    for attack in ATTACKS:
        if attack.family in family_values and attack.name in attack_means:
            family_values[attack.family].append(float(attack_means[attack.name]))
    missing = [family for family, values in family_values.items() if not values]
    if missing:
        raise ValueError(f"Missing attack values for families: {missing}")
    means = OrderedDict(
        (family, float(np.mean(values))) for family, values in family_values.items()
    )
    return float(np.mean(list(means.values()))), dict(means)
