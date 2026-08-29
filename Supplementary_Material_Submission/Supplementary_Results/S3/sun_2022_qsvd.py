"""Author reimplementation of Sun et al. (2022), DOI 10.1007/s11042-021-11815-x.

The paper embeds one bit in the maximum singular value of each selected 2x2
pure-Hamilton-quaternion RGB block.  Its QIM representatives are T/4 (bit 0)
and 3T/4 (bit 1), and extraction thresholds ``sigma_max mod T`` at T/2.

Protocol adaptations used by our comparison are deliberately kept outside the
atomic embedding rule: a keyed PCG64 permutation replaces MATLAB ``randperm``
and the paper's 24,576-bit dual colour watermark is replaced by the same keyed
1,024-bit binary payload used by every method in our experiment.  As a
common-protocol feasibility adaptation for the nonnegative singular-value
carrier, a negative nearest QIM representative is shifted by one period to the
first nonnegative representative on the same bit lattice.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_CORE = _Path(__file__).resolve().parents[2] / "core"
if str(_CORE) not in _sys.path:
    _sys.path.insert(0, str(_CORE))

import numpy as np

import rbsvd_qim as R


BLOCK_SIZE = 2
TL = 0.25
TR = 0.75
TP = 1
TM = 255.0
TU = 235.0
TD = 1.0


def _maximum_quaternion_singular_value(block: np.ndarray) -> float:
    adjoint = R.quat_adjoint(
        block[..., 0].astype(np.float64),
        block[..., 1].astype(np.float64),
        block[..., 2].astype(np.float64),
    )
    singular_values = np.linalg.svd(adjoint, compute_uv=False)
    # The complex-adjoint representation repeats every quaternion singular
    # value twice. Averaging the leading pair suppresses numerical splitting.
    return float((singular_values[0] + singular_values[1]) / 2.0)


def qim_target(value: float, bit: int, step: float) -> float:
    """Nearest periodic representative of Eq. (7)."""
    if step <= 0:
        raise ValueError("step must be positive")
    if bit not in (0, 1):
        raise ValueError("bit must be 0 or 1")
    offset = (TL if bit == 0 else TR) * step
    target = float(step * np.round((value - offset) / step) + offset)
    if target < 0.0:
        target += step
    return target


def extract_bit(value: float, step: float) -> np.uint8:
    """Equation (9): lower half-period is 0, upper half-period is 1."""
    return np.uint8(np.mod(value, step) >= 0.5 * step)


def _reconstruct(U: np.ndarray, s: np.ndarray, Vh: np.ndarray,
                 target: float, d: int) -> tuple[np.ndarray, np.ndarray]:
    changed = s.copy()
    changed[0] = target
    changed[1] = target
    adjoint = U @ np.diag(changed) @ Vh
    red, green, blue, _ = R.adjoint_to_quat(adjoint, d)
    raw = np.stack([red, green, blue], axis=-1)
    pixels = np.clip(np.round(raw), 0, 255).astype(np.uint8)
    return raw, pixels


def embed(
    image: np.ndarray,
    bits: np.ndarray,
    step: float = 54.0,
    seed: int = 2026,
    block_size: int = BLOCK_SIZE,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Embed one bit per selected block using Sun et al.'s QSVD-QIM rule.

    Equation (8) is applied only when the unrounded inverse-QSVD block exceeds
    the legal RGB range: subtract one period for overflow, or add one period
    for underflow when the paper's upper guard TU leaves sufficient headroom.
    Both moves preserve the embedded bit because they change the representative
    by exactly one quantisation period.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape H x W x 3")
    if block_size != 2:
        raise ValueError("Sun et al. (2022) specifies 2x2 blocks")
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    coords = R.block_views(image.shape[0], image.shape[1], block_size)
    if bits.size > len(coords):
        raise ValueError("payload exceeds the number of non-overlapping blocks")
    selected = R.select_blocks(len(coords), bits.size, seed)
    output = image.copy()
    diagnostics = {
        "n_bits": int(bits.size),
        "n_negative_target_shift": 0,
        "n_extreme_blue_adjustments": 0,
        "n_overflow_minus_period": 0,
        "n_underflow_plus_period": 0,
        "n_range_clipped": 0,
        "n_decode_fail": 0,
    }

    for coordinate_index, bit_value in zip(selected, bits):
        row, column = coords[int(coordinate_index)]
        block = output[row:row + block_size, column:column + block_size].copy()
        # Step 3: the paper changes the first blue sample from 0 to TP.
        if block[0, 0, 2] == 0:
            block[0, 0, 2] = TP
            diagnostics["n_extreme_blue_adjustments"] += 1

        adjoint = R.quat_adjoint(
            block[..., 0].astype(np.float64),
            block[..., 1].astype(np.float64),
            block[..., 2].astype(np.float64),
        )
        U, singular_values, Vh = np.linalg.svd(adjoint)
        maximum = float((singular_values[0] + singular_values[1]) / 2.0)
        offset = (TL if int(bit_value) == 0 else TR) * step
        unconstrained_target = float(
            step * np.round((maximum - offset) / step) + offset
        )
        diagnostics["n_negative_target_shift"] += int(
            unconstrained_target < 0.0
        )
        target = qim_target(maximum, int(bit_value), step)
        raw, pixels = _reconstruct(U, singular_values, Vh, target, block_size)

        # Eq. (8): a +/-T representative is bit-equivalent.  TU is the
        # paper's guard against creating a new upper overflow while repairing
        # an underflow; TD is checked after reconstruction.
        if np.max(raw) > TM:
            target -= step
            raw, pixels = _reconstruct(U, singular_values, Vh, target, block_size)
            diagnostics["n_overflow_minus_period"] += 1
        elif np.min(raw) < 0.0 and np.max(raw) <= TU:
            candidate_raw, candidate_pixels = _reconstruct(
                U, singular_values, Vh, target + step, block_size
            )
            if np.max(candidate_raw) <= TM and np.min(candidate_raw) >= -TD:
                target += step
                raw, pixels = candidate_raw, candidate_pixels
                diagnostics["n_underflow_plus_period"] += 1

        if np.min(raw) < 0.0 or np.max(raw) > TM:
            diagnostics["n_range_clipped"] += 1
        output[row:row + block_size, column:column + block_size] = pixels
        decoded = extract_bit(_maximum_quaternion_singular_value(pixels), step)
        diagnostics["n_decode_fail"] += int(decoded != bit_value)

    return output, diagnostics


def extract(
    image: np.ndarray,
    n_bits: int,
    step: float = 54.0,
    seed: int = 2026,
    block_size: int = BLOCK_SIZE,
) -> np.ndarray:
    """Blind extraction using the same keyed block ordering."""
    coords = R.block_views(image.shape[0], image.shape[1], block_size)
    if n_bits > len(coords):
        raise ValueError("payload exceeds the number of non-overlapping blocks")
    selected = R.select_blocks(len(coords), n_bits, seed)
    bits = np.zeros(n_bits, dtype=np.uint8)
    for output_index, coordinate_index in enumerate(selected):
        row, column = coords[int(coordinate_index)]
        block = image[row:row + block_size, column:column + block_size]
        bits[output_index] = extract_bit(
            _maximum_quaternion_singular_value(block), step
        )
    return bits
