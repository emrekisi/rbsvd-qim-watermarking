"""Author reimplementation of Chen et al., Expert Syst. Appl. 295 (2026)
128901, "Interpretable feature modeling for robust color watermarking in
the quaternion framework": the two QIM-mechanism variants.

- WQSD: QIM on t11, the maximal-modulus (leading) diagonal element of the
  quaternion Schur form T.  Blindly, |t11| equals the largest standard
  eigenvalue modulus of the 4x4 pure quaternion block, computed here via
  the 8x8 complex adjoint; embedding rescales the leading Schur diagonal
  pair and reconstructs Z S* Z^H (Table 6, mechanism Eqs. (4)-(5)).
- WQHD: QIM on h21, the subdiagonal element produced by Hessenberg
  reduction.  Because the bilateral reduction leaves the first column's
  tail norm invariant, h21 equals the quaternion norm of [a21,a31,a41]
  and the reconstruction of the paper reduces exactly to rescaling that
  subcolumn; both are implemented in this closed form.

Shared-protocol adaptations, disclosed: 1,024-bit keyed PCG64 payload
(one bit per selected 4x4 block) replaces the logistic-map keys.  This module
implements only the paper's WQSD and WQHD variants; the earlier Chen et al.
(2021) WQQRD method is implemented separately in
``chen_2021_qqrd.py``.  Strength T is selected by Kodak24 PSNR
quality matching only.  As a common-protocol feasibility adaptation for the
nonnegative magnitude carriers, a negative nearest QIM representative is
shifted by one period to the first nonnegative representative on the same bit
lattice.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_CORE = _Path(__file__).resolve().parents[2] / "core"
if str(_CORE) not in _sys.path:
    _sys.path.insert(0, str(_CORE))

import numpy as np
from scipy.linalg import schur

import rbsvd_qim as R


BLOCK_SIZE = 4


# ---------------- shared QIM mechanism, Eqs. (4)-(5) ----------------

def _unconstrained_qim_target(value: float, bit: int, step: float) -> float:
    """Nearest representative before the common nonnegative-feasibility guard."""
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
    target = lower if abs(lower - value) <= abs(upper - value) else upper
    return float(target)


def qim_target(value: float, bit: int, step: float) -> float:
    """Nearest of the two representatives; w=1 -> 0.25T, w=0 -> 0.75T."""
    target = _unconstrained_qim_target(value, bit, step)
    if target < 0.0:
        target += step
    return float(target)


def extract_bit(value: float, step: float) -> np.uint8:
    """Equation (5): remainder closer to 0.25T decodes as bit 1."""
    remainder = float(np.mod(value, step))
    return np.uint8(1 if abs(remainder - 0.25 * step)
                    <= abs(remainder - 0.75 * step) else 0)


# ---------------- WQSD ----------------

def _leading_eigen_modulus(adjoint: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(adjoint))))


def _wqsd_feature(block: np.ndarray) -> float:
    adjoint = R.quat_adjoint(block[..., 0].astype(np.float64),
                             block[..., 1].astype(np.float64),
                             block[..., 2].astype(np.float64))
    return _leading_eigen_modulus(adjoint)


def _wqsd_mark_block(
    block: np.ndarray, bit: int, step: float
) -> tuple[np.ndarray, bool]:
    adjoint = R.quat_adjoint(block[..., 0].astype(np.float64),
                             block[..., 1].astype(np.float64),
                             block[..., 2].astype(np.float64))
    S, Z = schur(adjoint, output="complex")
    moduli = np.abs(np.diag(S))
    t11 = float(np.max(moduli))
    unconstrained_target = _unconstrained_qim_target(t11, bit, step)
    target = (
        unconstrained_target + step
        if unconstrained_target < 0.0
        else unconstrained_target
    )
    if t11 <= np.finfo(np.float64).eps:
        return block, bool(unconstrained_target < 0.0)
    factor = target / t11
    S_marked = S.copy()
    # The adjoint repeats every quaternion eigenvalue as a conjugate pair;
    # rescale every diagonal entry attaining the leading modulus so the
    # quaternion structure of the modified matrix is preserved.
    leading = np.isclose(moduli, t11, rtol=1e-9, atol=1e-12)
    S_marked[np.diag_indices_from(S_marked)] = np.where(
        leading, np.diag(S) * factor, np.diag(S))
    marked = Z @ S_marked @ Z.conj().T
    red, green, blue, _ = R.adjoint_to_quat(marked, BLOCK_SIZE)
    raw = np.stack([red, green, blue], axis=-1)
    return (
        np.clip(np.round(raw), 0, 255).astype(np.uint8),
        bool(unconstrained_target < 0.0),
    )


# ---------------- WQHD ----------------

def _wqhd_feature(block: np.ndarray) -> float:
    tail = block[1:, 0, :].astype(np.float64)  # rows 2..4 of column 1, RGB
    return float(np.sqrt(np.sum(tail ** 2)))


def _wqhd_mark_block(
    block: np.ndarray, bit: int, step: float
) -> tuple[np.ndarray, bool]:
    h21 = _wqhd_feature(block)
    unconstrained_target = _unconstrained_qim_target(h21, bit, step)
    target = (
        unconstrained_target + step
        if unconstrained_target < 0.0
        else unconstrained_target
    )
    output = block.astype(np.float64).copy()
    if h21 > np.finfo(np.float64).eps:
        output[1:, 0, :] *= target / h21
    return (
        np.clip(np.round(output), 0, 255).astype(np.uint8),
        bool(unconstrained_target < 0.0),
    )


# ---------------- shared driver ----------------

_VARIANTS = {
    "wqsd": (_wqsd_feature, _wqsd_mark_block),
    "wqhd": (_wqhd_feature, _wqhd_mark_block),
}


def embed(
    image: np.ndarray,
    bits: np.ndarray,
    step: float,
    seed: int = 2026,
    variant: str = "wqsd",
    block_size: int = BLOCK_SIZE,
) -> tuple[np.ndarray, dict[str, int | float]]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape H x W x 3")
    if block_size != BLOCK_SIZE:
        raise ValueError("the paper uses 4x4 blocks")
    feature, mark = _VARIANTS[variant]
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    coords = R.block_views(image.shape[0], image.shape[1], block_size)
    if bits.size > len(coords):
        raise ValueError("payload exceeds available blocks")
    selected = R.select_blocks(len(coords), bits.size, seed)
    output = image.copy()
    diagnostics = {
        "n_bits": int(bits.size),
        "n_decode_fail": 0,
        "n_negative_target_shift": 0,
    }
    for coordinate_index, bit_value in zip(selected, bits):
        row, column = coords[int(coordinate_index)]
        block = output[row:row + block_size, column:column + block_size]
        pixels, shifted = mark(block, int(bit_value), float(step))
        output[row:row + block_size, column:column + block_size] = pixels
        diagnostics["n_negative_target_shift"] += int(shifted)
        decoded = extract_bit(feature(pixels), float(step))
        diagnostics["n_decode_fail"] += int(decoded != bit_value)
    return output, diagnostics


def extract(
    image: np.ndarray,
    n_bits: int,
    step: float,
    seed: int = 2026,
    variant: str = "wqsd",
    block_size: int = BLOCK_SIZE,
) -> np.ndarray:
    feature, _ = _VARIANTS[variant]
    coords = R.block_views(image.shape[0], image.shape[1], block_size)
    if n_bits > len(coords):
        raise ValueError("payload exceeds available blocks")
    selected = R.select_blocks(len(coords), n_bits, seed)
    bits = np.zeros(n_bits, dtype=np.uint8)
    for output_index, coordinate_index in enumerate(selected):
        row, column = coords[int(coordinate_index)]
        block = image[row:row + block_size, column:column + block_size]
        bits[output_index] = extract_bit(feature(block), float(step))
    return bits
