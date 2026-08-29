"""Author reimplementation of Li et al. (2020) QDFT-QQR-QIM watermarking.

The paper processes 512x512 images, partitions the scalar part of quaternion
QR's R factor into 128x128 blocks, and embeds Eq. (8) QIM representatives in
the highest-entropy block.  For rectangular Kodak images, a centered 512x512
crop is processed and restored into the unchanged image canvas.  Following
the source extraction procedure, the highest-entropy block is selected again
from the received image.  Embedding can change the entropy ordering, so this
source-faithful reselection can choose a different block at extraction.
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
except ModuleNotFoundError:  # Correctness-preserving fallback; slower only.
    def njit(*args, **kwargs):
        def decorate(function):
            return function
        return decorate


PROCESS_SIZE = 512
ENTROPY_BLOCK = 128
MU = np.array([0.0, 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0),
               1.0 / np.sqrt(3.0)], dtype=np.float64)


def _qmul_left_mu(values: np.ndarray) -> np.ndarray:
    """Return MU * values for an array whose last dimension is quaternion."""
    a, b, c, d = [values[..., index] for index in range(4)]
    x, y, z = MU[1:]
    out = np.empty_like(values, dtype=np.float64)
    out[..., 0] = -x * b - y * c - z * d
    out[..., 1] = x * a + y * d - z * c
    out[..., 2] = -x * d + y * a + z * b
    out[..., 3] = x * c - y * b + z * a
    return out


def qdft2(values: np.ndarray) -> np.ndarray:
    """Unitary two-dimensional left-sided QDFT (paper Eqs. 4-5)."""
    source = np.asarray(values, dtype=np.float64)
    spectra = np.stack(
        [np.fft.fft2(source[..., index], norm="ortho") for index in range(4)],
        axis=-1,
    )
    return spectra.real + _qmul_left_mu(spectra.imag)


def iqdft2(values: np.ndarray) -> np.ndarray:
    """Inverse of :func:`qdft2` under the same left-sided axis."""
    source = np.asarray(values, dtype=np.float64)
    spatial = np.stack(
        [np.fft.ifft2(source[..., index], norm="ortho") for index in range(4)],
        axis=-1,
    )
    return spatial.real + _qmul_left_mu(spatial.imag)


@njit(cache=True)
def _partial_qqr_core(matrix: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Modified Gram-Schmidt quaternion QR through the requested R rows."""
    n_rows, n_cols, _ = matrix.shape
    work = matrix.copy()
    q = np.zeros((n_rows, rank, 4), dtype=np.float64)
    r = np.zeros((rank, n_cols, 4), dtype=np.float64)
    for k in range(rank):
        norm_sq = 0.0
        for i in range(n_rows):
            for component in range(4):
                norm_sq += work[i, k, component] * work[i, k, component]
        norm = np.sqrt(norm_sq)
        r[k, k, 0] = norm
        if norm <= 1e-14:
            continue
        for i in range(n_rows):
            for component in range(4):
                q[i, k, component] = work[i, k, component] / norm

        for j in range(k + 1, n_cols):
            r0 = r1 = r2 = r3 = 0.0
            for i in range(n_rows):
                a = q[i, k, 0]
                b = q[i, k, 1]
                c = q[i, k, 2]
                d = q[i, k, 3]
                e = work[i, j, 0]
                f = work[i, j, 1]
                g = work[i, j, 2]
                h = work[i, j, 3]
                r0 += a * e + b * f + c * g + d * h
                r1 += a * f - b * e - c * h + d * g
                r2 += a * g + b * h - c * e - d * f
                r3 += a * h - b * g + c * f - d * e
            r[k, j, 0] = r0
            r[k, j, 1] = r1
            r[k, j, 2] = r2
            r[k, j, 3] = r3
            for i in range(n_rows):
                a = q[i, k, 0]
                b = q[i, k, 1]
                c = q[i, k, 2]
                d = q[i, k, 3]
                work[i, j, 0] -= a*r0 - b*r1 - c*r2 - d*r3
                work[i, j, 1] -= a*r1 + b*r0 + c*r3 - d*r2
                work[i, j, 2] -= a*r2 - b*r3 + c*r0 + d*r1
                work[i, j, 3] -= a*r3 + b*r2 - c*r1 + d*r0
    return q, r


def partial_qqr(matrix: np.ndarray, rank: int = ENTROPY_BLOCK) -> tuple[np.ndarray, np.ndarray]:
    source = np.ascontiguousarray(matrix, dtype=np.float64)
    return _partial_qqr_core(source, int(rank))


def _entropy(block: np.ndarray, mode: str = "minmax") -> float:
    """Return 256-bin Shannon entropy under a disclosed scaling convention."""
    if mode == "minmax":
        low, high = float(np.min(block)), float(np.max(block))
        if high <= low:
            return 0.0
        indices = np.floor((block - low) * (255.0 / (high - low))).astype(np.int32)
    elif mode == "matlab_literal":
        # MATLAB image functions expect double images in [0,1].  This branch
        # deliberately applies that literal saturation without mat2gray.
        indices = np.rint(np.clip(block, 0.0, 1.0) * 255.0).astype(np.int32)
    else:
        raise ValueError("entropy mode must be 'minmax' or 'matlab_literal'")
    counts = np.bincount(indices.ravel(), minlength=256).astype(np.float64)
    probabilities = counts[counts > 0.0] / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _select_block(
    r_scalar: np.ndarray, entropy_mode: str = "minmax"
) -> tuple[tuple[int, int], list[list[float]]]:
    """Search all sixteen paper-defined 128x128 blocks and take the argmax."""
    entropy_grid: list[list[float]] = []
    for block_row in range(4):
        row: list[float] = []
        for block_col in range(4):
            y = block_row * ENTROPY_BLOCK
            x = block_col * ENTROPY_BLOCK
            row.append(_entropy(
                r_scalar[y:y + ENTROPY_BLOCK, x:x + ENTROPY_BLOCK], entropy_mode
            ))
        entropy_grid.append(row)
    selected_flat = int(np.argmax(np.asarray(entropy_grid)))
    return (selected_flat // 4, selected_flat % 4), entropy_grid


def _positions(
    block_index: tuple[int, int], n_bits: int, seed: int,
    mode: str = "top_left",
) -> np.ndarray:
    block_row, block_col = block_index
    if mode == "top_left":
        side = int(round(np.sqrt(n_bits)))
        if side * side != n_bits or side > ENTROPY_BLOCK:
            raise ValueError("top-left mapping requires a square payload <= 128x128")
        local_rows, local_cols = np.indices((side, side))
        positions = np.column_stack((
            block_row * ENTROPY_BLOCK + local_rows.ravel(),
            block_col * ENTROPY_BLOCK + local_cols.ravel(),
        ))
        if np.any(positions[:, 0] > positions[:, 1]):
            raise ValueError("selected top-left payload region is below R's diagonal")
        return positions.astype(np.int32)
    if mode != "pcg_distributed":
        raise ValueError("position mode must be 'top_left' or 'pcg_distributed'")
    local_rows, local_cols = np.indices((ENTROPY_BLOCK, ENTROPY_BLOCK))
    global_rows = block_row * ENTROPY_BLOCK + local_rows.ravel()
    global_cols = block_col * ENTROPY_BLOCK + local_cols.ravel()
    eligible = np.column_stack((global_rows, global_cols))
    eligible = eligible[eligible[:, 0] <= eligible[:, 1]]
    if n_bits > len(eligible):
        raise ValueError("payload exceeds selected 128x128 block")
    order = np.random.Generator(np.random.PCG64(seed)).permutation(len(eligible))[:n_bits]
    return eligible[order].astype(np.int32)


def _matlab_round(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def _qim_targets(values: np.ndarray, bits: np.ndarray, step: float) -> np.ndarray:
    centers = 2.0 * step * _matlab_round(values / (2.0 * step))
    return centers + np.where(bits.astype(bool), step / 2.0, -step / 2.0)


def _center_crop(image: np.ndarray) -> tuple[np.ndarray, int, int]:
    height, width = image.shape[:2]
    if height < PROCESS_SIZE or width < PROCESS_SIZE:
        raise ValueError("Li QDFT-QQR baseline requires both dimensions >= 512")
    y = (height - PROCESS_SIZE) // 2
    x = (width - PROCESS_SIZE) // 2
    return image[y:y + PROCESS_SIZE, x:x + PROCESS_SIZE].copy(), y, x


def _to_quaternion(image: np.ndarray) -> np.ndarray:
    output = np.zeros((PROCESS_SIZE, PROCESS_SIZE, 4), dtype=np.float64)
    output[..., 1:] = image.astype(np.float64)
    return output


def _analyse(
    image: np.ndarray, entropy_mode: str = "minmax"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int], list[list[float]]]:
    spectrum = qdft2(_to_quaternion(image))
    q, r = partial_qqr(spectrum, PROCESS_SIZE)
    block_index, entropies = _select_block(r[..., 0], entropy_mode)
    return spectrum, q, r, block_index, entropies


def embed(
    image: np.ndarray,
    bits: np.ndarray,
    *,
    step: float,
    seed: int = 2026,
    entropy_mode: str = "minmax",
    position_mode: str = "top_left",
) -> tuple[np.ndarray, dict[str, int | float | list[float]]]:
    source = np.asarray(image, dtype=np.uint8)
    payload = np.asarray(bits, dtype=np.uint8).ravel()
    crop, y0, x0 = _center_crop(source)
    spectrum, q, r, block_index, entropies = _analyse(crop, entropy_mode)
    positions = _positions(block_index, payload.size, seed, position_mode)
    original = r[positions[:, 0], positions[:, 1], 0]
    targets = _qim_targets(original, payload, float(step))
    deltas = targets - original

    watermarked_spectrum = spectrum.copy()
    for (row, col), delta in zip(positions, deltas):
        watermarked_spectrum[:, int(col), :] += q[:, int(row), :] * float(delta)
    reconstructed = iqdft2(watermarked_spectrum)
    watermarked_crop = np.clip(np.rint(reconstructed[..., 1:]), 0, 255).astype(np.uint8)
    output = source.copy()
    output[y0:y0 + PROCESS_SIZE, x0:x0 + PROCESS_SIZE] = watermarked_crop
    recovered = extract(
        output, payload.size, step=step, seed=seed, entropy_mode=entropy_mode,
        position_mode=position_mode,
    )
    return output, {
        "n_bits": int(payload.size),
        "n_decode_fail": int(np.count_nonzero(recovered != payload)),
        "selected_entropy_block": [int(value) for value in block_index],
        "entropy_values": [[float(value) for value in row] for row in entropies],
        "entropy_mode": entropy_mode,
        "position_mode": position_mode,
        "mean_abs_qim_delta": float(np.mean(np.abs(deltas))),
        "max_abs_qim_delta": float(np.max(np.abs(deltas))),
        "max_reconstructed_scalar_abs": float(np.max(np.abs(reconstructed[..., 0]))),
        "n_clipped_components": int(np.count_nonzero(
            (reconstructed[..., 1:] < 0.0) | (reconstructed[..., 1:] > 255.0)
        )),
    }


def extract(
    image: np.ndarray,
    n_bits: int,
    *,
    step: float,
    seed: int = 2026,
    entropy_mode: str = "minmax",
    position_mode: str = "top_left",
) -> np.ndarray:
    source = np.asarray(image, dtype=np.uint8)
    crop, _, _ = _center_crop(source)
    _, _, r, block_index, _ = _analyse(crop, entropy_mode)
    positions = _positions(block_index, int(n_bits), seed, position_mode)
    values = r[positions[:, 0], positions[:, 1], 0]
    residual = values - 2.0 * step * _matlab_round(values / (2.0 * step))
    return (residual > 0.0).astype(np.uint8)
