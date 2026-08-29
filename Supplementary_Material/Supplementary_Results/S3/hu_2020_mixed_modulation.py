"""Disclosed reimplementation of Hu, Hsu and Chou (2020).

The implementation follows Eqs. (4)--(32) and Figs. 1--3 of
"An improved SVD-based blind color image watermarking algorithm with mixed
modulation incorporated", Information Sciences 519 (2020), 161--182.

The paper processes each colour channel independently and embeds one bit in
the gap ``u21-u31`` of the first left singular vector.  Its published 8x8
variant is used here.  The benchmark's fixed 1024-bit payload is assigned to a
PCG64-keyed subset of the paper-eligible channel/block positions; this is the
only payload-layout adaptation.  The six method modules are retained: level
shifting, sign correction, mixed RM/QIM modulation, orthonormal restoration,
distortion compensation, and output-pixel iterative regulation.
"""

from __future__ import annotations

import numpy as np


BLOCK_SIZE = 8
SHIFT_LEVEL = 216.0
MAX_ITERATIONS = 20
REGULATION_ALPHA = 0.5


def _validate_image(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("Hu2020 expects a uint8 RGB image")
    return np.ascontiguousarray(value)


def _validate_bits(bits: np.ndarray) -> np.ndarray:
    value = np.asarray(bits, dtype=np.uint8).ravel()
    if value.size == 0 or not np.all(np.isin(value, (0, 1))):
        raise ValueError("payload must be a nonempty binary vector")
    return value


def _locations(
    shape: tuple[int, int, int], n_bits: int, seed: int
) -> np.ndarray:
    """Return keyed paper-eligible ``(channel, block_row, block_col)`` rows."""
    height, width, channels = shape
    rows, columns = height // BLOCK_SIZE, width // BLOCK_SIZE
    eligible = np.asarray(
        [
            (channel, row, column)
            for channel in range(channels)
            for row in range(rows)
            for column in range(columns)
        ],
        dtype=np.int32,
    )
    if int(n_bits) > eligible.shape[0]:
        raise ValueError(
            f"payload {n_bits} exceeds Hu2020 d=8 capacity {eligible.shape[0]}"
        )
    order = np.random.Generator(np.random.PCG64(int(seed))).permutation(
        eligible.shape[0]
    )
    return eligible[order[: int(n_bits)]]


def _dither(size: int, seed: int, location_index: int, iteration: int) -> np.ndarray:
    """Deterministic bounded zero-mean realization of Eq. (4)'s dither."""
    sequence = np.random.SeedSequence(
        [int(seed), int(location_index), int(iteration), 2020]
    )
    rng = np.random.Generator(np.random.PCG64(sequence))
    count = int(size) * int(size)
    pair_count = count // 2

    # Dyadic PCG64 draws keep the shuffled u,-u construction exactly
    # zero-sum in float64 while preserving the source's [-0.5, 0.5] bounds.
    resolution = 1 << 40
    numerators = rng.integers(
        -resolution, resolution, size=pair_count, dtype=np.int64
    )
    draws = numerators.astype(np.float64) / float(2 * resolution)
    noise = np.empty(count, dtype=np.float64)
    noise[:pair_count] = draws
    noise[pair_count:2 * pair_count] = -draws
    if count % 2:
        noise[-1] = 0.0
    rng.shuffle(noise)
    return noise.reshape((int(size), int(size)))


def _level_shift(
    block: np.ndarray,
    *,
    dither: np.ndarray | None,
    shift_level: float = SHIFT_LEVEL,
) -> np.ndarray:
    source = block.astype(np.float64)
    shifted = source + (float(shift_level) - float(np.mean(source)))
    if dither is not None:
        shifted = shifted + dither
    return shifted


def _signed_svd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD followed by the correlation-index sign rule in Eq. (5)."""
    left, singular_values, right_h = np.linalg.svd(matrix.astype(np.float64))
    ones = np.ones(matrix.shape[0], dtype=np.float64)
    first_left = left[:, 0]
    first_right = right_h[0, :]
    correlation = float(
        first_left @ matrix @ ones + ones @ matrix @ first_right
    )
    if correlation < 0.0:
        left[:, 0] *= -1.0
        right_h[0, :] *= -1.0
    return left, singular_values, right_h


def _gap(left: np.ndarray) -> float:
    return float(left[1, 0] - left[2, 0])


def _mixed_target(gap: float, bit: int, step: float) -> tuple[float, str]:
    """Mixed-modulation target from Eqs. (6)--(9)."""
    D = float(step)
    if not np.isfinite(D) or D <= 0.0:
        raise ValueError("D must be positive and finite")
    bit = int(bit)
    delta, psi, rho, eta = 0.5 * D, D, 1.5 * D, 2.0 * D
    if bit == 1:
        gamma_rm = min(psi, max(gap, delta))
    else:
        gamma_rm = max(-psi, min(gap, -delta))

    if gap > rho:
        if bit == 1:
            gamma_qim = max(
                0.0,
                np.floor((gap - eta) / (2.0 * D)) * 2.0 * D + D + eta,
            )
        else:
            gamma_qim = (
                np.floor((gap - eta) / (2.0 * D) + 0.5) * 2.0 * D
                + eta
            )
        if abs(gamma_qim - gap) < abs(gamma_rm - gap):
            return float(gamma_qim), "qim_positive"
    elif gap < -rho:
        if bit == 1:
            gamma_qim = (
                np.ceil((gap + eta) / (2.0 * D) - 0.5) * 2.0 * D
                - eta
            )
        else:
            gamma_qim = min(
                0.0,
                np.ceil((gap + eta) / (2.0 * D)) * 2.0 * D - D - eta,
            )
        if abs(gamma_qim - gap) < abs(gamma_rm - gap):
            return float(gamma_qim), "qim_negative"
    return float(gamma_rm), "rm"


def _decode_gap(gap: float, step: float) -> int:
    """Blind mixed-modulation decision rule in Eq. (10)."""
    D = float(step)
    rho, eta = 1.5 * D, 2.0 * D
    if abs(gap) <= rho:
        return 1 if gap >= 0.0 else 0
    if gap > rho:
        index = int(np.floor((gap - eta) / D + 0.5))
        return index % 2
    index = int(np.ceil((gap + eta) / D - 0.5)) - 1
    return index % 2


def _zeta_values(u21: float, u31: float, target_gap: float) -> tuple[float, float]:
    """The two least-alteration criteria in Eqs. (14)--(17)."""
    linear = 4.0 * u21 - 2.0 * target_gap
    constant = (u21 - target_gap) ** 2 - u31**2
    discriminant = linear**2 - 8.0 * constant
    if discriminant >= 0.0:
        root = np.sqrt(max(0.0, discriminant))
        roots = ((-linear + root) / 4.0, (-linear - root) / 4.0)
        zeta1 = min(roots, key=lambda value: (abs(value), value))
    else:
        zeta1 = 0.5 * target_gap - u21
    zeta2 = 0.5 * (target_gap + u31 - u21)
    return float(zeta1), float(zeta2)


def _modified_first_vector(
    first: np.ndarray, target_gap: float, mode: str, step: float
) -> tuple[np.ndarray, float, int]:
    """Apply Eqs. (11)--(19), generalized to the paper's d=8 variant."""
    source = np.asarray(first, dtype=np.float64)
    u21, u31 = float(source[1]), float(source[2])
    candidate_gap = float(target_gap)
    reductions = 0
    while True:
        zeta1, zeta2 = _zeta_values(u21, u31, candidate_gap)
        weight = 0.5 if mode == "rm" else 0.25
        zeta = weight * zeta1 + (1.0 - weight) * zeta2
        pair = np.array(
            [u21 + zeta, u21 - candidate_gap + zeta], dtype=np.float64
        )
        if mode == "rm" or float(pair @ pair) <= 1.0 + 1e-14:
            break
        candidate_gap += -2.0 * step if candidate_gap > 0.0 else 2.0 * step
        reductions += 1
        if reductions > 100:
            raise RuntimeError("Hu2020 orthonormal-restoration loop did not terminate")

    modified = source.copy()
    old_pair_energy = u21**2 + u31**2
    new_pair_energy = float(pair @ pair)
    if mode == "rm":
        if new_pair_energy > np.finfo(np.float64).eps:
            pair *= np.sqrt(max(0.0, old_pair_energy) / new_pair_energy)
        modified[1:3] = pair
    else:
        modified[1:3] = pair
        other = np.ones(source.size, dtype=bool)
        other[1:3] = False
        old_other_energy = max(0.0, 1.0 - old_pair_energy)
        new_other_energy = max(0.0, 1.0 - float(pair @ pair))
        if old_other_energy > np.finfo(np.float64).eps:
            modified[other] *= np.sqrt(new_other_energy / old_other_energy)
        else:
            modified[other] = 0.0
    norm = float(np.linalg.norm(modified))
    if norm <= np.finfo(np.float64).eps:
        raise RuntimeError("Hu2020 produced a zero first singular vector")
    return modified / norm, candidate_gap, reductions


def _gram_schmidt(first: np.ndarray, original_left: np.ndarray) -> np.ndarray:
    """Standard modified Gram--Schmidt realization of Eq. (20)."""
    size = original_left.shape[0]
    basis = np.empty_like(original_left, dtype=np.float64)
    basis[:, 0] = first / np.linalg.norm(first)
    for column in range(1, size):
        vector = original_left[:, column].astype(np.float64).copy()
        for previous in range(column):
            vector -= np.dot(basis[:, previous], vector) * basis[:, previous]
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            vector = np.zeros(size, dtype=np.float64)
            vector[column] = 1.0
            for previous in range(column):
                vector -= np.dot(basis[:, previous], vector) * basis[:, previous]
            norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise RuntimeError("Hu2020 Gram--Schmidt restoration became singular")
        basis[:, column] = vector / norm
    return basis


def _distortion_compensated_reconstruction(
    shifted: np.ndarray,
    restored_left: np.ndarray,
    right_h: np.ndarray,
    original_mean: float,
) -> np.ndarray:
    """Least-squares diagonal compensation in Eqs. (21)--(24)."""
    size = shifted.shape[0]
    compensated = np.empty(size, dtype=np.float64)
    for index in range(size):
        plane = np.outer(restored_left[:, index], right_h[index, :])
        compensated[index] = float(np.sum(shifted * plane))
    recomposed = restored_left @ np.diag(compensated) @ right_h
    return recomposed + (float(original_mean) - SHIFT_LEVEL)


def _uint8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.floor(values + 0.5), 0.0, 255.0).astype(np.uint8)


def _decode_block(block: np.ndarray, step: float) -> int:
    shifted = _level_shift(block, dither=None)
    left, _, _ = _signed_svd(shifted)
    return _decode_gap(_gap(left), step)


def _regulate(
    current: np.ndarray,
    reconstructed: np.ndarray,
    target_gap: float,
    step: float,
) -> tuple[np.ndarray, str]:
    """Generalized d=8 form of Eqs. (27)--(32)."""
    size = current.shape[0]
    carrier_rows = np.zeros(size, dtype=bool)
    carrier_rows[1:3] = True
    if abs(target_gap) - 2.0 * step >= 1.5 * step:
        adjusted = current.astype(np.float64).copy()
        adjusted[carrier_rows, :] = reconstructed[carrier_rows, :]
        adjusted[~carrier_rows, :] = (
            REGULATION_ALPHA * reconstructed[~carrier_rows, :]
            + (1.0 - REGULATION_ALPHA) * current[~carrier_rows, :]
        )
        return _uint8(adjusted), "readjust"

    minimum, maximum = float(np.min(reconstructed)), float(np.max(reconstructed))
    if 0.0 <= minimum and maximum <= 255.0:
        shrunk = reconstructed
    elif maximum > minimum:
        upper = min(255.0, maximum)
        lower = max(0.0, minimum)
        shrunk = (reconstructed - minimum) * (upper - lower) / (
            maximum - minimum
        ) + lower
    else:
        shrunk = np.clip(reconstructed, 0.0, 255.0)
    adjusted = current.astype(np.float64).copy()
    adjusted[carrier_rows, :] = shrunk[carrier_rows, :]
    return _uint8(adjusted), "shrink"


def _embed_block(
    block: np.ndarray,
    bit: int,
    step: float,
    *,
    seed: int,
    location_index: int,
    max_iterations: int,
) -> tuple[np.ndarray, dict[str, int | float | str]]:
    current = np.asarray(block, dtype=np.uint8).copy()
    last_output = current.copy()
    last_mode = "none"
    last_gap = 0.0
    total_gap_reductions = 0
    regulation_branch = "none"
    for iteration in range(int(max_iterations)):
        shifted = _level_shift(
            current,
            dither=_dither(current.shape[0], seed, location_index, iteration),
        )
        left, _, right_h = _signed_svd(shifted)
        gap = _gap(left)
        target_gap, mode = _mixed_target(gap, int(bit), float(step))
        last_mode, last_gap = mode, target_gap

        if np.isclose(target_gap, gap, rtol=0.0, atol=1e-15):
            last_output = current.copy()
        else:
            first, realized_gap, reductions = _modified_first_vector(
                left[:, 0], target_gap, mode, float(step)
            )
            total_gap_reductions += reductions
            restored = _gram_schmidt(first, left)
            reconstructed = _distortion_compensated_reconstruction(
                shifted, restored, right_h, float(np.mean(current))
            )
            last_output = _uint8(reconstructed)
            last_gap = realized_gap

        if _decode_block(last_output, float(step)) == int(bit):
            return last_output, {
                "iterations": iteration + 1,
                "decode_fail": 0,
                "mixed_mode": last_mode,
                "gap_reductions": total_gap_reductions,
                "regulation_branch": regulation_branch,
            }
        if np.isclose(target_gap, gap, rtol=0.0, atol=1e-15):
            reconstructed = current.astype(np.float64)
        current, regulation_branch = _regulate(
            current, reconstructed, last_gap, float(step)
        )

    return last_output, {
        "iterations": int(max_iterations),
        "decode_fail": int(_decode_block(last_output, float(step)) != int(bit)),
        "mixed_mode": last_mode,
        "gap_reductions": total_gap_reductions,
        "regulation_branch": regulation_branch,
    }


def embed(
    image: np.ndarray,
    bits: np.ndarray,
    *,
    step: float,
    seed: int = 2026,
    max_iterations: int = MAX_ITERATIONS,
) -> tuple[np.ndarray, dict[str, int | float | dict[str, int]]]:
    """Embed the common payload with the published d=8 Hu2020 variant."""
    source = _validate_image(image)
    payload = _validate_bits(bits)
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be positive")
    locations = _locations(source.shape, payload.size, int(seed))
    output = source.copy()
    mode_counts = {"rm": 0, "qim_positive": 0, "qim_negative": 0}
    branch_counts = {"none": 0, "readjust": 0, "shrink": 0}
    total_iterations = 0
    decode_failures = 0
    gap_reductions = 0
    for index, (bit, location) in enumerate(zip(payload, locations)):
        channel, block_row, block_column = map(int, location)
        row, column = block_row * BLOCK_SIZE, block_column * BLOCK_SIZE
        embedded, diagnostics = _embed_block(
            output[row:row + BLOCK_SIZE, column:column + BLOCK_SIZE, channel],
            int(bit),
            float(step),
            seed=int(seed),
            location_index=index,
            max_iterations=int(max_iterations),
        )
        output[row:row + BLOCK_SIZE, column:column + BLOCK_SIZE, channel] = embedded
        mode_counts[str(diagnostics["mixed_mode"])] += 1
        branch_counts[str(diagnostics["regulation_branch"])] += 1
        total_iterations += int(diagnostics["iterations"])
        decode_failures += int(diagnostics["decode_fail"])
        gap_reductions += int(diagnostics["gap_reductions"])
    recovered = extract(output, payload.size, step=float(step), seed=int(seed))
    verified_failures = int(np.count_nonzero(recovered != payload))
    if verified_failures != decode_failures:
        decode_failures = verified_failures
    return output, {
        "n_bits": int(payload.size),
        "n_decode_fail": decode_failures,
        "mean_iterations_per_bit": float(total_iterations / payload.size),
        "max_iterations": int(max_iterations),
        "mixed_mode_counts": mode_counts,
        "final_regulation_branch_counts": branch_counts,
        "orthonormal_gap_reductions": int(gap_reductions),
        "block_size": BLOCK_SIZE,
        "shift_level": SHIFT_LEVEL,
    }


def extract(
    image: np.ndarray,
    n_bits: int,
    *,
    step: float,
    seed: int = 2026,
) -> np.ndarray:
    """Blind extraction by level shifting, sign correction and Eq. (10)."""
    source = _validate_image(image)
    locations = _locations(source.shape, int(n_bits), int(seed))
    recovered = np.empty(int(n_bits), dtype=np.uint8)
    for index, location in enumerate(locations):
        channel, block_row, block_column = map(int, location)
        row, column = block_row * BLOCK_SIZE, block_column * BLOCK_SIZE
        block = source[
            row:row + BLOCK_SIZE, column:column + BLOCK_SIZE, channel
        ]
        recovered[index] = np.uint8(_decode_block(block, float(step)))
    return recovered
