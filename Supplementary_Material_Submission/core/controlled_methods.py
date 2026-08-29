"""The three methods used in the controlled mechanism comparison."""

from __future__ import annotations

import math

import numpy as np

import rbsvd_qim as method
from experiment_settings import selected_parameters


METHODS = ("rbsvd", "channel_svd", "qsvd")
LABELS = {
    "rbsvd": "Proposed RBSVD-QIM",
    "channel_svd": "Channel-SVD-QIM",
    "qsvd": "Dense-adjoint QSVD-QIM",
}


def quantization_step(method_id: str, delta: float | None = None) -> float:
    delta = float(selected_parameters(delta)["Delta"])
    return delta / math.sqrt(3.0) if method_id == "channel_svd" else delta


def embed(
    method_id: str,
    image: np.ndarray,
    bits: np.ndarray,
    delta: float | None = None,
):
    point = selected_parameters(delta)
    common = dict(
        d=point["d"], Delta=quantization_step(method_id, delta), r=point["r"],
        seed=point["seed"], K_max=point["K_max"], safety=point["safety"],
        preserve_order=True,
    )
    if method_id == "rbsvd":
        return method.rbq_embed(image, bits, alternative_targets=True, **common)
    if method_id == "channel_svd":
        return method.chan_embed_corrected(image, bits, **common)
    if method_id == "qsvd":
        return method.qsvd_embed_corrected(image, bits, **common)
    raise KeyError(method_id)


def extract(
    method_id: str,
    image: np.ndarray,
    n_bits: int,
    delta: float | None = None,
) -> np.ndarray:
    point = selected_parameters(delta)
    common = dict(
        d=point["d"], Delta=quantization_step(method_id, delta), r=point["r"],
        seed=point["seed"],
    )
    if method_id == "rbsvd":
        return method.rbq_extract(image, n_bits, **common)
    if method_id == "channel_svd":
        return method.chan_extract(image, n_bits, **common)
    if method_id == "qsvd":
        return method.qsvd_extract(image, n_bits, **common)
    raise KeyError(method_id)
