"""Small reusable helpers for the experiment entry files."""

from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

import rbsvd_qim as method
from experiment_settings import ATTACKS, attack_seed_values, family_balanced_ber


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    records = list(rows)
    if not records:
        raise ValueError(f"No rows supplied for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(item: Any) -> Any:
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(f"Cannot serialize {type(item).__name__}")

    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=convert) + "\n",
        encoding="utf-8",
    )


def extract(
    image: np.ndarray,
    bits: np.ndarray,
    *,
    d: int,
    delta: float,
    rank: int,
    seed: int,
    extractor: Callable[..., np.ndarray] = method.rbq_extract,
) -> dict[str, float | int]:
    recovered = extractor(image, bits.size, d=d, Delta=delta, r=rank, seed=seed)
    errors = int(np.count_nonzero(recovered != bits))
    return {
        "errors": errors,
        "ber_percent": float(100.0 * errors / bits.size),
    }


def attack_results(
    marked: np.ndarray,
    bits: np.ndarray,
    *,
    d: int,
    delta: float,
    rank: int,
    seed: int,
    extractor: Callable[..., np.ndarray] = method.rbq_extract,
    include_brightness: bool = True,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for attack in ATTACKS:
        if not include_brightness and attack.name == "Brightness (+10)":
            continue
        for attack_seed in attack_seed_values(attack):
            attacked = attack.apply(marked, attack_seed)
            result = extract(
                attacked, bits, d=d, delta=delta, rank=rank, seed=seed,
                extractor=extractor,
            )
            rows.append({
                "attack": attack.name,
                "family": attack.family or "excluded",
                "attack_seed": attack_seed,
                **result,
            })
    return rows


def mean_by_attack(rows: Iterable[Mapping[str, Any]]) -> "OrderedDict[str, float]":
    values: "OrderedDict[str, list[float]]" = OrderedDict()
    for row in rows:
        values.setdefault(str(row["attack"]), []).append(float(row["ber_percent"]))
    return OrderedDict((name, float(np.mean(data))) for name, data in values.items())


def summarize_attacks(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    attacks = mean_by_attack(rows)
    family_score, family_means = family_balanced_ber(dict(attacks))
    retained = OrderedDict(
        (name, value)
        for name, value in attacks.items()
        if name not in {"No attack", "Brightness (+10)"}
    )
    worst_name = max(retained, key=retained.get)
    return {
        "attack_mean_ber_percent": dict(attacks),
        "family_mean_ber_percent": family_means,
        "family_balanced_ber_percent": family_score,
        "worst_attack": worst_name,
        "worst_attack_ber_percent": retained[worst_name],
    }


def sample_std(values: Iterable[float]) -> float:
    data = np.asarray(list(values), dtype=float)
    return float(np.std(data, ddof=1)) if data.size > 1 else float("nan")
