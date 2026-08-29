"""Author reimplementation of Chen et al., Signal Processing 238 (2026).

Paper: "Fast quaternion QR algorithm: Advancing watermarking with
multifaceted capabilities", DOI 10.1016/j.sigpro.2025.110215.

One bit is embedded in the real component R0(1,4) of the upper-triangular
factor of a 4x4 pure-quaternion RGB block.  Equations (12)-(15) and the
mathematical quaternion-Givens QQR are retained.  The rotations are evaluated
directly in quaternion arithmetic rather than with the paper's runtime-
optimized fast real-representation kernel; this implementation therefore does
not reproduce that runtime optimization.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_CORE = _Path(__file__).resolve().parents[2] / "core"
if str(_CORE) not in _sys.path:
    _sys.path.insert(0, str(_CORE))

import numpy as np

try:
    from numba import njit
except ModuleNotFoundError:
    def njit(*args, **kwargs):
        def decorate(function):
            return function
        return decorate

import rbsvd_qim as R


BLOCK_SIZE = 4


@njit(cache=True)
def _qmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    output = np.empty(4, dtype=np.float64)
    a, b, c, d = left[0], left[1], left[2], left[3]
    e, f, g, h = right[0], right[1], right[2], right[3]
    output[0] = a * e - b * f - c * g - d * h
    output[1] = a * f + b * e + c * h - d * g
    output[2] = a * g - b * h + c * e + d * f
    output[3] = a * h + b * g - c * f + d * e
    return output


@njit(cache=True)
def _qconj(value: np.ndarray) -> np.ndarray:
    output = value.copy()
    output[1:] *= -1.0
    return output


@njit(cache=True)
def _qidentity(size: int) -> np.ndarray:
    output = np.zeros((size, size, 4), dtype=np.float64)
    for index in range(size):
        output[index, index, 0] = 1.0
    return output


@njit(cache=True)
def _qmatmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    output = np.zeros((left.shape[0], right.shape[1], 4), dtype=np.float64)
    for row in range(left.shape[0]):
        for column in range(right.shape[1]):
            for inner in range(left.shape[1]):
                output[row, column] += _qmul(left[row, inner], right[inner, column])
    return output


@njit(cache=True)
def _givens(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Quaternion Givens matrix G such that G* [first,second]^T=[r,0]."""
    first_norm = np.sqrt(np.sum(first * first))
    second_norm = np.sqrt(np.sum(second * second))
    combined = np.sqrt(first_norm * first_norm + second_norm * second_norm)
    matrix = np.zeros((2, 2, 4), dtype=np.float64)
    if combined <= 2.220446049250313e-16:
        matrix[0, 0, 0] = 1.0
        matrix[1, 1, 0] = 1.0
        return matrix

    g1 = first / combined
    g2 = second / combined
    g1_norm = first_norm / combined
    g2_norm = second_norm / combined
    matrix[0, 0] = g1
    matrix[1, 0] = g2
    if first_norm >= second_norm:
        if g1_norm <= 2.220446049250313e-16:
            matrix[0, 1, 0] = 1.0
            matrix[1, 1, 0] = 0.0
        else:
            matrix[0, 1] = -_qmul(g1, _qconj(g2)) / g1_norm
            matrix[1, 1, 0] = g1_norm
    else:
        matrix[0, 1, 0] = g2_norm
        matrix[1, 1] = -_qmul(g2, _qconj(g1)) / g2_norm
    return matrix


@njit(cache=True)
def _qqr_core(block: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    size = block.shape[0]
    upper = np.zeros((size, size, 4), dtype=np.float64)
    upper[:, :, 1:] = block
    unitary = _qidentity(size)

    for column in range(size - 1):
        for lower_row in range(column + 1, size):
            givens = _givens(upper[column, column], upper[lower_row, column])
            # R <- G^* R on the two active rows.
            for active_column in range(column, size):
                top = upper[column, active_column].copy()
                bottom = upper[lower_row, active_column].copy()
                upper[column, active_column] = (
                    _qmul(_qconj(givens[0, 0]), top)
                    + _qmul(_qconj(givens[1, 0]), bottom)
                )
                upper[lower_row, active_column] = (
                    _qmul(_qconj(givens[0, 1]), top)
                    + _qmul(_qconj(givens[1, 1]), bottom)
                )
            # Q <- Q G, preserving B = Q R.
            for row in range(size):
                left = unitary[row, column].copy()
                right = unitary[row, lower_row].copy()
                unitary[row, column] = (
                    _qmul(left, givens[0, 0]) + _qmul(right, givens[1, 0])
                )
                unitary[row, lower_row] = (
                    _qmul(left, givens[0, 1]) + _qmul(right, givens[1, 1])
                )

    # The published Eq. (10) fixes every diagonal element as positive real.
    # The final 1x1 block receives no two-row Givens rotation in the printed
    # loop, so apply the standard QR diagonal phase normalisation explicitly:
    # Q[:,k] <- Q[:,k] p and R[k,:] <- conj(p) R[k,:].
    for diagonal_index in range(size):
        diagonal = upper[diagonal_index, diagonal_index].copy()
        diagonal_norm = np.sqrt(np.sum(diagonal * diagonal))
        if diagonal_norm > 2.220446049250313e-16:
            phase = diagonal / diagonal_norm
            conjugate_phase = _qconj(phase)
            for active_column in range(diagonal_index, size):
                upper[diagonal_index, active_column] = _qmul(
                    conjugate_phase, upper[diagonal_index, active_column]
                )
            for row in range(size):
                unitary[row, diagonal_index] = _qmul(
                    unitary[row, diagonal_index], phase
                )

    lower_residual = 0.0
    diagonal_imaginary_residual = 0.0
    for row in range(size):
        diagonal_imaginary_residual = max(
            diagonal_imaginary_residual,
            np.max(np.abs(upper[row, row, 1:])),
        )
        for column in range(row):
            lower_residual = max(lower_residual, np.max(np.abs(upper[row, column])))
    return unitary, upper, lower_residual, diagonal_imaginary_residual


def qqr(block: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    unitary, upper, lower_residual, diagonal_residual = _qqr_core(
        block.astype(np.float64)
    )
    return unitary, upper, {
        "lower_triangular_residual": float(lower_residual),
        "diagonal_imaginary_residual": float(diagonal_residual),
    }


def qim_target(value: float, bit: int, step: float) -> float:
    """Nearest endpoint from Eqs. (12)-(14)."""
    if step <= 0.0:
        raise ValueError("step must be positive")
    if bit not in (0, 1):
        raise ValueError("bit must be 0 or 1")
    remainder = float(np.mod(value, step))
    base = value - remainder
    if bit == 1:
        lower, upper = base + 0.25 * step, base + 1.25 * step
    else:
        lower, upper = base - 0.25 * step, base + 0.75 * step
    return float(lower if abs(lower - value) <= abs(upper - value) else upper)


def extract_bit(value: float, step: float) -> np.uint8:
    """Equation (15): lower half-period is bit 1, upper half is bit 0."""
    return np.uint8(1 if np.mod(value, step) < 0.5 * step else 0)


def embed(
    image: np.ndarray,
    bits: np.ndarray,
    step: float = 0.6,
    seed: int = 2026,
    block_size: int = BLOCK_SIZE,
) -> tuple[np.ndarray, dict[str, int | float]]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape H x W x 3")
    if block_size != BLOCK_SIZE:
        raise ValueError("Chen et al. (2026) specifies 4x4 blocks")
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    coordinates = R.block_views(image.shape[0], image.shape[1], block_size)
    if bits.size > len(coordinates):
        raise ValueError("payload exceeds available blocks")
    selected = R.select_blocks(len(coordinates), bits.size, seed)
    output = image.copy()
    diagnostics = {
        "n_bits": int(bits.size),
        "n_decode_fail": 0,
        "n_range_clipped": 0,
        "max_lower_triangular_residual": 0.0,
        "max_diagonal_imaginary_residual": 0.0,
        "max_real_part_residual": 0.0,
    }

    for coordinate_index, bit_value in zip(selected, bits):
        row, column = coordinates[int(coordinate_index)]
        block = output[row:row + block_size, column:column + block_size]
        # The paper's QQR examples and T<=0.6 sweep use RGB intensities in
        # [0,1]; convert back to the project's uint8 protocol after inverse QR.
        unitary, upper, qr_diagnostics = qqr(block.astype(np.float64) / 255.0)
        diagnostics["max_lower_triangular_residual"] = max(
            float(diagnostics["max_lower_triangular_residual"]),
            qr_diagnostics["lower_triangular_residual"],
        )
        diagnostics["max_diagonal_imaginary_residual"] = max(
            float(diagnostics["max_diagonal_imaginary_residual"]),
            qr_diagnostics["diagonal_imaginary_residual"],
        )
        upper[0, 3, 0] = qim_target(
            float(upper[0, 3, 0]), int(bit_value), step
        )
        reconstructed = _qmatmul(unitary, upper)
        raw = 255.0 * reconstructed[..., 1:4]
        diagnostics["max_real_part_residual"] = max(
            float(diagnostics["max_real_part_residual"]),
            float(255.0 * np.max(np.abs(reconstructed[..., 0]))),
        )
        diagnostics["n_range_clipped"] += int(
            np.min(raw) < 0.0 or np.max(raw) > 255.0
        )
        pixels = np.clip(np.round(raw), 0, 255).astype(np.uint8)
        output[row:row + block_size, column:column + block_size] = pixels
        _, checked_upper, _ = qqr(pixels.astype(np.float64) / 255.0)
        diagnostics["n_decode_fail"] += int(
            extract_bit(float(checked_upper[0, 3, 0]), step) != bit_value
        )
    return output, diagnostics


def extract(
    image: np.ndarray,
    n_bits: int,
    step: float = 0.6,
    seed: int = 2026,
    block_size: int = BLOCK_SIZE,
) -> np.ndarray:
    coordinates = R.block_views(image.shape[0], image.shape[1], block_size)
    if n_bits > len(coordinates):
        raise ValueError("payload exceeds available blocks")
    selected = R.select_blocks(len(coordinates), n_bits, seed)
    bits = np.zeros(n_bits, dtype=np.uint8)
    for output_index, coordinate_index in enumerate(selected):
        row, column = coordinates[int(coordinate_index)]
        _, upper, _ = qqr(
            image[row:row + block_size, column:column + block_size].astype(
                np.float64
            ) / 255.0
        )
        bits[output_index] = extract_bit(float(upper[0, 3, 0]), step)
    return bits


def reconstruction_error(block: np.ndarray) -> float:
    unitary, upper, _ = qqr(block)
    reconstructed = _qmatmul(unitary, upper)
    expected = np.zeros_like(reconstructed)
    expected[..., 1:4] = block.astype(np.float64)
    return float(np.linalg.norm(reconstructed - expected) / np.linalg.norm(expected))
