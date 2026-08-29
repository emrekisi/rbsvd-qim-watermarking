"""Protocol-adapted Chen et al. QQRD relative-modulation baseline.

Paper: "A new structure-preserving quaternion QR decomposition method for
color image blind watermarking", Signal Processing 185 (2021) 108088,
DOI 10.1016/j.sigpro.2021.108088.

Each selected 4x4 pure-quaternion RGB block is factorized as H=QR.  Equations
(4)-(6) are applied to the i, j, and k components of the strongly correlated
(q21, q31) pair, carrying three bits per block.  The positive-real-diagonal
Givens QQR kernel is algebraically equivalent to the paper's Householder
factorization for full-rank blocks and reproduces the paper's printed
numerical example to the stated precision; it does not reproduce the
structure-preserving runtime optimization.  The shared Kodak24 protocol
uses the same 1,024-bit binary payload and PCG64 block permutation as the
other methods instead of the paper's color-logo/Arnold-transform layout.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_CORE = _Path(__file__).resolve().parents[2] / "core"
if str(_CORE) not in _sys.path:
    _sys.path.insert(0, str(_CORE))

import numpy as np

import rbsvd_qim as R
from chen_2026_qqr import _qmatmul, qqr


BLOCK_SIZE = 4
BITS_PER_BLOCK = 3


def _embed_component(unitary: np.ndarray, component: int, bit: int,
                     threshold: float) -> None:
    first = float(unitary[1, 0, component])
    second = float(unitary[2, 0, component])
    difference = first - second
    average = 0.5 * (first + second)
    if bit == 1 and difference < threshold:
        unitary[1, 0, component] = average + 0.5 * threshold
        unitary[2, 0, component] = average - 0.5 * threshold
    elif bit == 0 and difference > -threshold:
        unitary[1, 0, component] = average - 0.5 * threshold
        unitary[2, 0, component] = average + 0.5 * threshold


def _extract_components(unitary: np.ndarray) -> np.ndarray:
    return np.asarray([
        int(unitary[1, 0, component] >= unitary[2, 0, component])
        for component in (1, 2, 3)
    ], dtype=np.uint8)


def _selected_blocks(image: np.ndarray, n_bits: int, seed: int):
    coordinates = R.block_views(image.shape[0], image.shape[1], BLOCK_SIZE)
    count = (n_bits + BITS_PER_BLOCK - 1) // BITS_PER_BLOCK
    if count > len(coordinates):
        raise ValueError("payload exceeds available blocks")
    return coordinates, R.select_blocks(len(coordinates), count, seed)


def embed(
    image: np.ndarray,
    bits: np.ndarray,
    threshold: float = 0.035,
    seed: int = 2026,
) -> tuple[np.ndarray, dict[str, int | float]]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape H x W x 3")
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    coordinates, selected = _selected_blocks(image, bits.size, seed)
    output = image.copy()
    diagnostics = {
        "n_bits": int(bits.size),
        "n_blocks": int(selected.size),
        "n_decode_fail": 0,
        "n_range_clipped_components": 0,
        "max_real_part_residual": 0.0,
        "max_lower_triangular_residual": 0.0,
        "threshold": float(threshold),
    }

    for block_number, coordinate_index in enumerate(selected):
        row, column = coordinates[int(coordinate_index)]
        block = output[row:row + BLOCK_SIZE, column:column + BLOCK_SIZE]
        unitary, upper, qr_diagnostics = qqr(block.astype(np.float64) / 255.0)
        diagnostics["max_lower_triangular_residual"] = max(
            float(diagnostics["max_lower_triangular_residual"]),
            qr_diagnostics["lower_triangular_residual"],
        )
        start = block_number * BITS_PER_BLOCK
        stop = min(start + BITS_PER_BLOCK, bits.size)
        for local_index, bit in enumerate(bits[start:stop], start=1):
            _embed_component(unitary, local_index, int(bit), float(threshold))

        reconstructed = _qmatmul(unitary, upper)
        raw = 255.0 * reconstructed[..., 1:4]
        diagnostics["max_real_part_residual"] = max(
            float(diagnostics["max_real_part_residual"]),
            float(255.0 * np.max(np.abs(reconstructed[..., 0]))),
        )
        diagnostics["n_range_clipped_components"] += int(
            np.count_nonzero((raw < 0.0) | (raw > 255.0))
        )
        pixels = np.clip(np.round(raw), 0, 255).astype(np.uint8)
        output[row:row + BLOCK_SIZE, column:column + BLOCK_SIZE] = pixels
        checked_unitary, _, _ = qqr(pixels.astype(np.float64) / 255.0)
        decoded = _extract_components(checked_unitary)[:stop - start]
        diagnostics["n_decode_fail"] += int(
            np.count_nonzero(decoded != bits[start:stop])
        )
    return output, diagnostics


def extract(
    image: np.ndarray,
    n_bits: int,
    threshold: float = 0.035,
    seed: int = 2026,
) -> np.ndarray:
    del threshold  # Blind decision in Eq. (6) depends only on relative order.
    coordinates, selected = _selected_blocks(image, n_bits, seed)
    recovered = np.empty(selected.size * BITS_PER_BLOCK, dtype=np.uint8)
    for block_number, coordinate_index in enumerate(selected):
        row, column = coordinates[int(coordinate_index)]
        unitary, _, _ = qqr(
            image[row:row + BLOCK_SIZE, column:column + BLOCK_SIZE].astype(
                np.float64
            ) / 255.0
        )
        start = block_number * BITS_PER_BLOCK
        recovered[start:start + BITS_PER_BLOCK] = _extract_components(unitary)
    return recovered[:n_bits]
