"""Author reimplementation of Chen et al., IEEE TIP 2023.

Paper: "Efficient Robust Watermarking Based on Structure-Preserving
Quaternion Singular Value Decomposition", DOI 10.1109/TIP.2023.3293773.

The watermark bit is carried by one imaginary component of the correlated
coefficient pair (u21, u31) in the first left quaternion singular vector of a
4x4 RGB quaternion block.  Equations (6)-(8) are used with the disclosed
conditional minimum-distance component-selection rule described below.

The paper's custom structure-preserving QSVD is implemented directly with its
H3 quaternion Householder bidiagonalisation.  The final length-two H3 steps are
mathematically equivalent to the paper's generalized-Givens speed shortcut;
using H3 throughout retains the published phase/sign convention.  The result
reproduces the paper's numerical U-column example to its four printed decimals.
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

import rbsvd_qim as R


BLOCK_SIZE = 4
PAPER_STRENGTH = 0.035


# --- Direct quaternion H3 bidiagonalisation used by the paper's Algorithm 3 ---


def _qconj(values: np.ndarray) -> np.ndarray:
    result = values.copy()
    result[..., 1:] *= -1.0
    return result


def _qmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a, b, c, d = np.moveaxis(left, -1, 0)
    e, f, g, h = np.moveaxis(right, -1, 0)
    return np.stack([
        a * e - b * f - c * g - d * h,
        a * f + b * e + c * h - d * g,
        a * g - b * h + c * e + d * f,
        a * h + b * g - c * f + d * e,
    ], axis=-1)


def _qmatmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape[1] != right.shape[0]:
        raise ValueError("incompatible quaternion matrix shapes")
    output = np.zeros((left.shape[0], right.shape[1], 4), dtype=np.float64)
    for index in range(left.shape[1]):
        output += _qmul(left[:, index, None, :], right[None, index, :, :])
    return output


def _qhermitian(values: np.ndarray) -> np.ndarray:
    return np.swapaxes(_qconj(values), 0, 1)


def _qidentity(size: int) -> np.ndarray:
    result = np.zeros((size, size, 4), dtype=np.float64)
    result[np.arange(size), np.arange(size), 0] = 1.0
    return result


def _real_quaternion_matrix(values: np.ndarray) -> np.ndarray:
    result = np.zeros(values.shape + (4,), dtype=np.float64)
    result[..., 0] = values
    return result


def _h3(vector: np.ndarray) -> np.ndarray:
    """Paper Algorithm 1/2 H3: map a quaternion vector to ||y||e1."""
    length = vector.shape[0]
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(np.float64).eps:
        return _qidentity(length)
    first_norm = float(np.linalg.norm(vector[0]))
    if first_norm > np.finfo(np.float64).eps:
        alpha = -norm * vector[0] / first_norm
    else:
        alpha = np.array([-norm, 0.0, 0.0, 0.0])
    target = np.zeros_like(vector)
    target[0] = alpha
    difference = vector - target
    difference_norm = float(np.linalg.norm(difference))
    if difference_norm <= np.finfo(np.float64).eps:
        h1 = _qidentity(length)
    else:
        unit = difference / difference_norm
        outer = _qmul(unit[:, None, :], _qconj(unit)[None, :, :])
        h1 = _qidentity(length) - 2.0 * outer
    transformed = _qmatmul(h1, vector[:, None, :])[:, 0, :]
    phases = np.zeros_like(transformed)
    for index, value in enumerate(transformed):
        value_norm = float(np.linalg.norm(value))
        phases[index] = (_qconj(value) / value_norm if value_norm >
                         np.finfo(np.float64).eps else
                         np.array([1.0, 0.0, 0.0, 0.0]))
    diagonal = np.zeros((length, length, 4), dtype=np.float64)
    diagonal[np.arange(length), np.arange(length)] = phases
    return _qmatmul(diagonal, h1)


@njit(cache=True)
def _qmul4(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    output = np.empty(4, dtype=np.float64)
    a, b, c, d = left[0], left[1], left[2], left[3]
    e, f, g, h = right[0], right[1], right[2], right[3]
    output[0] = a * e - b * f - c * g - d * h
    output[1] = a * f + b * e + c * h - d * g
    output[2] = a * g - b * h + c * e + d * f
    output[3] = a * h + b * g - c * f + d * e
    return output


@njit(cache=True)
def _qmatmul_jit(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    output = np.zeros((left.shape[0], right.shape[1], 4), dtype=np.float64)
    for row in range(left.shape[0]):
        for column in range(right.shape[1]):
            for inner in range(left.shape[1]):
                output[row, column] += _qmul4(
                    left[row, inner], right[inner, column]
                )
    return output


@njit(cache=True)
def _qidentity_jit(size: int) -> np.ndarray:
    output = np.zeros((size, size, 4), dtype=np.float64)
    for index in range(size):
        output[index, index, 0] = 1.0
    return output


@njit(cache=True)
def _qhermitian_jit(values: np.ndarray) -> np.ndarray:
    output = np.empty((values.shape[1], values.shape[0], 4), dtype=np.float64)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            output[column, row, 0] = values[row, column, 0]
            output[column, row, 1] = -values[row, column, 1]
            output[column, row, 2] = -values[row, column, 2]
            output[column, row, 3] = -values[row, column, 3]
    return output


@njit(cache=True)
def _h3_jit(vector: np.ndarray) -> np.ndarray:
    length = vector.shape[0]
    norm = np.sqrt(np.sum(vector * vector))
    if norm <= 2.220446049250313e-16:
        return _qidentity_jit(length)
    first_norm = np.sqrt(np.sum(vector[0] * vector[0]))
    alpha = np.zeros(4, dtype=np.float64)
    if first_norm > 2.220446049250313e-16:
        alpha = -norm * vector[0] / first_norm
    else:
        alpha[0] = -norm
    difference = vector.copy()
    difference[0] -= alpha
    difference_norm = np.sqrt(np.sum(difference * difference))
    h1 = _qidentity_jit(length)
    if difference_norm > 2.220446049250313e-16:
        unit = difference / difference_norm
        for row in range(length):
            for column in range(length):
                conjugate = unit[column].copy()
                conjugate[1:] *= -1.0
                h1[row, column] -= 2.0 * _qmul4(unit[row], conjugate)
    transformed = np.zeros((length, 4), dtype=np.float64)
    for row in range(length):
        for inner in range(length):
            transformed[row] += _qmul4(h1[row, inner], vector[inner])
    output = np.empty_like(h1)
    for row in range(length):
        value_norm = np.sqrt(np.sum(transformed[row] * transformed[row]))
        phase = np.zeros(4, dtype=np.float64)
        if value_norm > 2.220446049250313e-16:
            phase[0] = transformed[row, 0] / value_norm
            phase[1:] = -transformed[row, 1:] / value_norm
        else:
            phase[0] = 1.0
        for column in range(length):
            output[row, column] = _qmul4(phase, h1[row, column])
    return output


@njit(cache=True)
def _h3_qsvd_core(block: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rows, columns = block.shape[0], block.shape[1]
    matrix = np.zeros((rows, columns, 4), dtype=np.float64)
    matrix[:, :, 1:] = block
    left_product = _qidentity_jit(rows)
    right_product = _qidentity_jit(columns)
    for index in range(columns):
        left_small = _h3_jit(matrix[index:, index])
        left_full = _qidentity_jit(rows)
        left_full[index:, index:] = left_small
        matrix = _qmatmul_jit(left_full, matrix)
        left_product = _qmatmul_jit(left_full, left_product)
        if index + 1 < columns:
            row_conjugate = matrix[index, index + 1:].copy()
            row_conjugate[:, 1:] *= -1.0
            right_left = _h3_jit(row_conjugate)
            right_small = _qhermitian_jit(right_left)
            right_full = _qidentity_jit(columns)
            right_full[index + 1:, index + 1:] = right_small
            matrix = _qmatmul_jit(matrix, right_full)
            right_product = _qmatmul_jit(right_product, right_full)
    bidiagonal = matrix[:, :, 0]
    imaginary_residual = np.max(np.abs(matrix[:, :, 1:]))
    real_left, singular_values, real_right_h = np.linalg.svd(bidiagonal)
    real_left_q = np.zeros((rows, rows, 4), dtype=np.float64)
    real_left_q[:, :, 0] = real_left
    real_right_q = np.zeros((columns, columns, 4), dtype=np.float64)
    real_right_q[:, :, 0] = real_right_h
    left_vectors = _qmatmul_jit(_qhermitian_jit(left_product), real_left_q)
    right_vectors_h = _qmatmul_jit(real_right_q, _qhermitian_jit(right_product))
    # MATLAB and LAPACK may choose opposite real signs for a singular-vector
    # pair.  Eqs. (6)-(7) assume Algorithm 3's published negative first-column
    # convention.  Flip U[:,1] and the matching V^H row together, preserving Q.
    sign_score = np.sum(left_vectors[:, 0, 1:])
    if sign_score > 0.0:
        left_vectors[:, 0, :] *= -1.0
        right_vectors_h[0, :, :] *= -1.0
    return left_vectors, singular_values, right_vectors_h, imaginary_residual


def _h3_qsvd(block: np.ndarray) -> dict[str, np.ndarray | float]:
    """Quaternion bidiagonal QSVD following the H3 path of Algorithm 3.

    Algorithm 3 replaces the final length-two H3 transformations by equivalent
    generalized Givens rotations for speed.  Using H3 throughout preserves its
    mathematical output convention while avoiding an implementation-only fast
    path that is irrelevant for this baseline's robustness comparison.
    """
    left_vectors, singular_values, right_vectors_h, imaginary_residual = (
        _h3_qsvd_core(block.astype(np.float64))
    )
    return {
        "left_q": left_vectors,
        "singular_values_q": singular_values,
        "right_h_q": right_vectors_h,
        "bidiagonal_imaginary_residual": imaginary_residual,
    }


def _qsvd_reconstruction_error(block: np.ndarray) -> float:
    decomposition = _h3_qsvd(block)
    diagonal = _real_quaternion_matrix(np.diag(decomposition["singular_values_q"]))
    reconstructed = _qmatmul(
        _qmatmul(decomposition["left_q"], diagonal), decomposition["right_h_q"]
    )
    expected = np.zeros_like(reconstructed)
    expected[..., 1:] = block.astype(np.float64)
    return float(np.linalg.norm(reconstructed - expected) / np.linalg.norm(expected))


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a, b, c, d = left
    e, f, g, h = right
    return np.array([
        a * e - b * f - c * g - d * h,
        a * f + b * e + c * h - d * g,
        a * g - b * h + c * e + d * f,
        a * h + b * g - c * f + d * e,
    ], dtype=np.float64)


def _quat_inverse(value: np.ndarray) -> np.ndarray:
    squared_norm = float(np.dot(value, value))
    if squared_norm <= np.finfo(np.float64).eps:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return np.array([value[0], -value[1], -value[2], -value[3]]) / squared_norm


def _scalar_adjoint(value: np.ndarray) -> np.ndarray:
    c1 = value[0] + 1j * value[1]
    c2 = value[2] + 1j * value[3]
    return np.array([[c1, c2], [-np.conj(c2), np.conj(c1)]],
                    dtype=np.complex128)


def _partner(column: np.ndarray, d: int) -> np.ndarray:
    """Symplectic partner forming the second adjoint column."""
    return np.concatenate([-np.conj(column[d:]), np.conj(column[:d])])


def _column_to_quaternions(column: np.ndarray, d: int) -> np.ndarray:
    c1 = column[:d]
    c2 = -np.conj(column[d:])
    return np.column_stack([c1.real, c1.imag, c2.real, c2.imag])


def _quaternions_to_column(values: np.ndarray) -> np.ndarray:
    c1 = values[:, 0] + 1j * values[:, 1]
    c2 = values[:, 2] + 1j * values[:, 3]
    return np.concatenate([c1, -np.conj(c2)])


def _canonical_first_pair(
    left_vectors: np.ndarray,
    singular_values: np.ndarray,
    right_vectors_h: np.ndarray,
    d: int,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Legacy complex-adjoint diagnostic, not used by the H3 implementation.

    The first two complex singular values encode one repeated quaternion
    singular value.  Rotations inside this two-dimensional degenerate subspace
    are transferred to V^H, so canonicalisation does not change the block.
    """
    column = left_vectors[:, 0]
    quaternion_pair = np.column_stack([column, _partner(column, d)])
    mixing = quaternion_pair.conj().T @ left_vectors[:, :2]
    adjusted_right = mixing @ right_vectors_h[:2, :]

    quaternion_column = _column_to_quaternions(column, d)
    # Algorithm 3 begins its left Householder reduction at q11.  Its published
    # numerical example fixes the leading quaternion u11 near the negative
    # equal-imaginary direction (real part approximately zero).  Anchoring on
    # u11, rather than on a whole-column statistic, reproduces that convention.
    anchor = quaternion_column[0]
    anchor_norm = float(np.linalg.norm(anchor))
    if anchor_norm > np.finfo(np.float64).eps:
        target = np.array(
            [0.0, -anchor_norm / np.sqrt(3.0),
             -anchor_norm / np.sqrt(3.0),
             -anchor_norm / np.sqrt(3.0)],
            dtype=np.float64,
        )
        phase = _quat_multiply(_quat_inverse(anchor), target)
        phase /= np.linalg.norm(phase)
        phase_adjoint = _scalar_adjoint(phase)
        quaternion_pair = quaternion_pair @ phase_adjoint
        adjusted_right = phase_adjoint.conj().T @ adjusted_right

    repeated_value = float(np.mean(singular_values[:2]))
    return quaternion_pair, repeated_value, adjusted_right, mixing


def _decompose(block: np.ndarray) -> dict[str, np.ndarray | float]:
    return _h3_qsvd(block)


def _feature_components(decomposition: dict[str, np.ndarray | float],
                        d: int) -> np.ndarray:
    quaternions = decomposition["left_q"]
    assert isinstance(quaternions, np.ndarray)
    # Rows 2 and 3 in one-based paper notation; columns i, j, k.
    return quaternions[1:3, 0, 1:4].copy()


def _modulated_pair(values: np.ndarray, bit: int, strength: float,
                    selection_policy: str = "minimum_max_change") -> tuple[np.ndarray, int, float]:
    """Apply Eqs. (6)-(7) under an explicit adaptive-selection reading.

    The paper states that the most correlated local pair is chosen so as to
    reduce the maximum coefficient modification, but does not give a scalar
    selection equation.  The benchmark uses
    ``minimum_distance_to_strength_conditional``: a component that already
    satisfies the bit inequality is left unchanged, the three components are
    ranked by their distance from the extraction magnitude T, and exact ties
    are resolved by modification cost and then component order.  The other
    policies keep the source ambiguity explicit for direct sensitivity checks.
    """
    if bit not in (0, 1):
        raise ValueError("bit must be 0 or 1")
    allowed = {
        "minimum_max_change",
        "always_minimum_max_change",
        "minimum_abs_difference",
        "minimum_distance_to_strength",
        "minimum_abs_difference_conditional",
        "minimum_distance_to_strength_conditional",
    }
    if selection_policy not in allowed:
        raise ValueError(f"unknown selection_policy: {selection_policy}")
    candidates = []
    for component in range(3):
        first, second = values[:, component]
        difference = float(first - second)
        condition_met = difference < -strength if bit == 1 else difference >= strength
        proposed = values.copy()
        always_modulate = selection_policy in {
            "always_minimum_max_change",
            "minimum_abs_difference",
            "minimum_distance_to_strength",
        }
        if always_modulate or not condition_met:
            average = 0.5 * (abs(first) + abs(second))
            if bit == 1:  # Eq. (6): difference is -T.
                proposed[0, component] = -average - strength / 2.0
                proposed[1, component] = -average + strength / 2.0
            else:  # Eq. (7): difference is +T.
                proposed[0, component] = -average + strength / 2.0
                proposed[1, component] = -average - strength / 2.0
        change = proposed[:, component] - values[:, component]
        # The paper describes decreasing the maximum coefficient change.
        cost = float(np.max(np.abs(change)))
        if selection_policy in {"minimum_max_change", "always_minimum_max_change"}:
            selection_score = cost
        elif selection_policy in {
            "minimum_abs_difference", "minimum_abs_difference_conditional"
        }:
            selection_score = abs(difference)
        else:
            selection_score = abs(abs(difference) - strength)
        candidates.append((selection_score, cost, component, proposed))
    _, cost, component, proposed = min(
        candidates, key=lambda row: (row[0], row[1], row[2])
    )
    return proposed, int(component), cost


def _reconstruct(
    decomposition: dict[str, np.ndarray | float],
    modified_components: np.ndarray,
    d: int,
) -> tuple[np.ndarray, float]:
    left = decomposition["left_q"].copy()
    singular_values = decomposition["singular_values_q"]
    right_h = decomposition["right_h_q"]
    assert isinstance(singular_values, np.ndarray)
    assert isinstance(left, np.ndarray)
    assert isinstance(right_h, np.ndarray)
    left[1:3, 0, 1:4] = modified_components
    diagonal = _real_quaternion_matrix(np.diag(singular_values))
    reconstructed = _qmatmul(_qmatmul(left, diagonal), right_h)
    raw = reconstructed[..., 1:4]
    real_residual = float(np.max(np.abs(reconstructed[..., 0])))
    return raw, real_residual


def extract_bit_from_components(values: np.ndarray, strength: float) -> tuple[np.uint8, int, float]:
    """Equation (8), including adaptive component recovery."""
    differences = values[0] - values[1]
    distances = np.abs(np.abs(differences) - strength)
    component = int(np.argmin(distances))
    difference = float(differences[component])
    # Exact zero is unspecified in the paper; deterministic >=0 resolves the
    # measure-zero numerical tie as bit 0.
    bit = np.uint8(0 if difference >= 0.0 else 1)
    return bit, component, difference


def embed(
    image: np.ndarray,
    bits: np.ndarray,
    strength: float = PAPER_STRENGTH,
    seed: int = 2026,
    block_size: int = BLOCK_SIZE,
    selection_policy: str = "minimum_max_change",
) -> tuple[np.ndarray, dict[str, int | float]]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape H x W x 3")
    if block_size != BLOCK_SIZE:
        raise ValueError("Chen et al. (2023) specifies 4x4 blocks")
    if strength <= 0.0:
        raise ValueError("strength must be positive")
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    coordinates = R.block_views(image.shape[0], image.shape[1], block_size)
    if bits.size > len(coordinates):
        raise ValueError("payload exceeds available blocks")
    selected = R.select_blocks(len(coordinates), bits.size, seed)
    output = image.copy()
    diagnostics = {
        "n_bits": int(bits.size),
        "n_modified": 0,
        "n_selected_i": 0,
        "n_selected_j": 0,
        "n_selected_k": 0,
        "n_decode_fail": 0,
        "n_extraction_component_mismatch": 0,
        "n_range_clipped": 0,
        "max_real_part_residual": 0.0,
        "max_adjoint_reconstruction_error": 0.0,
    }

    for coordinate_index, bit_value in zip(selected, bits):
        row, column = coordinates[int(coordinate_index)]
        block = output[row:row + block_size, column:column + block_size]
        decomposition = _decompose(block)
        components = _feature_components(decomposition, block_size)
        changed, chosen_component, cost = _modulated_pair(
            components, int(bit_value), strength, selection_policy
        )
        diagnostics[("n_selected_i", "n_selected_j", "n_selected_k")[chosen_component]] += 1
        diagnostics["n_modified"] += int(cost > 0.0)
        raw, real_residual = _reconstruct(decomposition, changed, block_size)
        diagnostics["max_real_part_residual"] = max(
            float(diagnostics["max_real_part_residual"]), real_residual
        )
        diagnostics["n_range_clipped"] += int(np.min(raw) < 0.0 or np.max(raw) > 255.0)
        pixels = np.clip(np.round(raw), 0, 255).astype(np.uint8)
        output[row:row + block_size, column:column + block_size] = pixels

        check = _decompose(pixels)
        recovered_components = _feature_components(check, block_size)
        recovered_bit, recovered_component, _ = extract_bit_from_components(
            recovered_components, strength
        )
        diagnostics["n_decode_fail"] += int(recovered_bit != bit_value)
        diagnostics["n_extraction_component_mismatch"] += int(
            recovered_component != chosen_component
        )

    return output, diagnostics


def extract(
    image: np.ndarray,
    n_bits: int,
    strength: float = PAPER_STRENGTH,
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
        block = image[row:row + block_size, column:column + block_size]
        decomposition = _decompose(block)
        components = _feature_components(decomposition, block_size)
        bits[output_index] = extract_bit_from_components(components, strength)[0]
    return bits


def decomposition_reconstruction_error(block: np.ndarray) -> float:
    """Relative quaternion reconstruction error of the H3 QSVD."""
    return _qsvd_reconstruction_error(block)
