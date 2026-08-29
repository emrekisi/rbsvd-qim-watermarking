"""Author reimplementation of Su, Zhang and Wang (2022) spatial MSV-QIM.

The implementation follows Eqs. (7), (11)--(17) of the paper.  Although the
paper calls Eq. (7) a maximum singular value (MSV), the printed expression is
the Frobenius norm of a 4x4 block; that printed expression is used here.  The
paper's monotonicity statement for the two candidate representatives conflicts
with its Figure 1 numerical example.  Following Eqs. (12)--(14) and that
example, this implementation selects the nearest representative and imposes
no one-directional update rule.  As a common-protocol feasibility adaptation
for this nonnegative carrier, a negative nearest QIM representative is shifted
by one period to the first nonnegative representative on the same-bit lattice.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_CORE = _Path(__file__).resolve().parents[2] / "core"
if str(_CORE) not in _sys.path:
    _sys.path.insert(0, str(_CORE))

import numpy as np


BLOCK_SIZE = 4
CHANNEL_STEP_RATIOS = np.array([0.78, 0.94, 1.0], dtype=np.float64)


def _locations(shape: tuple[int, int, int], n_bits: int, seed: int) -> np.ndarray:
    """Return dispersed paper-eligible (channel, block-row, block-col) tuples.

    K=2 in the paper gives columns 1,3,... in odd block rows and 2,4,... in
    even block rows (one-based indexing).  The paper fills all such positions
    with a 24,576-bit colour watermark.  For the shared 1,024-bit protocol we
    select a keyed subset of the same eligible positions.
    """
    height, width, channels = shape
    if channels != 3:
        raise ValueError("Su MSV-QIM expects an RGB image")
    rows, cols = height // BLOCK_SIZE, width // BLOCK_SIZE
    eligible: list[tuple[int, int, int]] = []
    for channel in range(3):
        for block_row in range(rows):
            start = 0 if block_row % 2 == 0 else 1
            for block_col in range(start, cols, 2):
                eligible.append((channel, block_row, block_col))
    if n_bits > len(eligible):
        raise ValueError(f"payload {n_bits} exceeds eligible capacity {len(eligible)}")
    order = np.random.Generator(np.random.PCG64(seed)).permutation(len(eligible))
    return np.asarray(eligible, dtype=np.int32)[order[:n_bits]]


def _feature(block: np.ndarray) -> float:
    """Equation (7), i.e. sqrt(sum(pixel**2)), as printed in the paper."""
    return float(np.sqrt(np.sum(block.astype(np.float64) ** 2)))


def _unconstrained_target(feature: float, bit: int, step: float) -> float:
    """Nearest candidate before the common nonnegative-feasibility guard."""
    base = np.floor(feature / step)
    if int(bit) == 0:
        low, high = (base + 0.25) * step, (base + 1.25) * step
    else:
        low, high = (base - 0.25) * step, (base + 0.75) * step
    target = low if abs(low - feature) <= abs(high - feature) else high
    return float(target)


def _target(feature: float, bit: int, step: float) -> float:
    """Nearest of the two candidates in Eqs. (12)--(14)."""
    target = _unconstrained_target(feature, bit, step)
    if target < 0.0:
        target += step
    return float(target)


def _embed_block(block: np.ndarray, bit: int, step: float) -> tuple[np.ndarray, dict]:
    feature = _feature(block)
    unconstrained_target = _unconstrained_target(feature, bit, step)
    target = (
        unconstrained_target + step
        if unconstrained_target < 0.0
        else unconstrained_target
    )
    delta = target - feature
    per_pixel = delta / BLOCK_SIZE
    source = block.astype(np.float64)
    altered = source + per_pixel

    # Equation (16): wrap only pixels that would underflow or overflow by one
    # QIM period distributed over the block.  The source does not prescribe a
    # unique float-to-uint rule; this reimplementation retains floor conversion.
    period_per_pixel = step / (BLOCK_SIZE * BLOCK_SIZE)
    if per_pixel < 0:
        altered[source < abs(per_pixel)] += period_per_pixel
    elif per_pixel > 0:
        altered[source > 255.0 - per_pixel] -= period_per_pixel
    output = np.floor(np.clip(altered, 0.0, 255.0)).astype(np.uint8)
    return output, {
        "feature": feature,
        "target": target,
        "negative_target_shifted": bool(unconstrained_target < 0.0),
        "delta": delta,
        "output_feature": _feature(output),
    }


def embed(
    image: np.ndarray,
    bits: np.ndarray,
    *,
    qt_blue: float,
    seed: int = 2026,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Embed a binary payload using paper K=2 eligibility and keyed subsampling."""
    source = np.asarray(image, dtype=np.uint8)
    payload = np.asarray(bits, dtype=np.uint8).ravel()
    locations = _locations(source.shape, payload.size, seed)
    output = source.copy()
    steps = CHANNEL_STEP_RATIOS * float(qt_blue)
    n_wrapped = 0
    n_negative_target_shift = 0
    target_abs_error: list[float] = []
    for bit, (channel, block_row, block_col) in zip(payload, locations):
        y = int(block_row) * BLOCK_SIZE
        x = int(block_col) * BLOCK_SIZE
        block = output[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE, int(channel)]
        embedded, row = _embed_block(block, int(bit), float(steps[int(channel)]))
        output[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE, int(channel)] = embedded
        n_wrapped += int(np.any((embedded == 0) | (embedded == 255)))
        n_negative_target_shift += int(row["negative_target_shifted"])
        target_abs_error.append(abs(row["output_feature"] - row["target"]))
    recovered = extract(output, payload.size, qt_blue=qt_blue, seed=seed)
    return output, {
        "n_bits": int(payload.size),
        "n_decode_fail": int(np.count_nonzero(recovered != payload)),
        "n_negative_target_shift": int(n_negative_target_shift),
        "n_blocks_with_endpoint_pixels": int(n_wrapped),
        "mean_post_integer_target_abs_error": float(np.mean(target_abs_error)),
        "max_post_integer_target_abs_error": float(np.max(target_abs_error)),
    }


def extract(
    image: np.ndarray,
    n_bits: int,
    *,
    qt_blue: float,
    seed: int = 2026,
) -> np.ndarray:
    """Blind extraction using Eq. (17)."""
    source = np.asarray(image, dtype=np.uint8)
    locations = _locations(source.shape, int(n_bits), seed)
    steps = CHANNEL_STEP_RATIOS * float(qt_blue)
    recovered = np.empty(int(n_bits), dtype=np.uint8)
    for index, (channel, block_row, block_col) in enumerate(locations):
        y = int(block_row) * BLOCK_SIZE
        x = int(block_col) * BLOCK_SIZE
        block = source[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE, int(channel)]
        step = float(steps[int(channel)])
        residue = np.mod(np.round(_feature(block)), step)
        recovered[index] = np.uint8(0 if residue < 0.5 * step else 1)
    return recovered
