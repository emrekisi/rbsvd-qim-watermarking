"""Reference implementation of the blind RBQ/RBSVD-QIM scheme.

For a reduced biquaternion ``q = q0 + q1*i + q2*j + q3*k``, the
idempotents ``e1 = (1+j)/2`` and ``e2 = (1-j)/2`` give
``q = z1*e1 + z2*e2``. A pure RBQ color pixel ``R*i + G*j + B*k`` maps
to ``A1 = G + (R+B)*i`` and ``A2 = -G + (R-B)*i``. The functions below
implement this mapping, QIM embedding/extraction, matched baselines,
attacks, and the metrics used in the paper.
"""
import io
import numpy as np
from PIL import Image
from scipy import ndimage


# RBQ transforms

def rgb_to_idem(img):
    """img: float array (H,W,3) -> (A1, A2) complex."""
    R = img[..., 0].astype(np.float64)
    G = img[..., 1].astype(np.float64)
    B = img[..., 2].astype(np.float64)
    A1 = G + 1j * (R + B)
    A2 = -G + 1j * (R - B)
    return A1, A2


def idem_to_rgb(A1, A2):
    """Map idempotent components to ``(R, G, B, q0)``."""
    q0 = (A1.real + A2.real) / 2.0
    q2 = (A1.real - A2.real) / 2.0   # G
    q1 = (A1.imag + A2.imag) / 2.0   # R
    q3 = (A1.imag - A2.imag) / 2.0   # B
    return q1, q2, q3, q0


# ---------------- QIM ----------------

def qim_embed_value(s, w, Delta):
    """Move value s to the nearest quantization point of bit w.
    bit 0 -> {k*Delta + Delta/4},  bit 1 -> {k*Delta + 3*Delta/4}"""
    target = Delta / 4.0 + w * (Delta / 2.0)
    return Delta * np.round((s - target) / Delta) + target


def _nearest_nonnegative_qim_target(s, w, Delta):
    """Nearest admissible same-bit QIM target for a nonnegative feature."""
    target = float(qim_embed_value(s, w, Delta))
    if target < 0.0:
        target += np.ceil(-target / Delta) * Delta
    if target < 0.0 or qim_extract_value(target, Delta) != int(w):
        raise AssertionError("failed to construct a nonnegative same-bit target")
    return target


def qim_extract_value(s, Delta):
    """Relative position p = s mod Delta; p < Delta/2 -> 0, otherwise 1."""
    p = np.mod(s, Delta)
    return 0 if p < Delta / 2.0 else 1


# Block-selection helpers

def _validate_image_and_params(img_u8, d, Delta, r):
    """Validate the shared public inputs used by all embedding/extraction paths."""
    if not isinstance(img_u8, np.ndarray):
        raise TypeError("img_u8 must be a NumPy array")
    if img_u8.ndim != 3 or img_u8.shape[2] != 3:
        raise ValueError("img_u8 must have shape (height, width, 3) for RGB data")
    if not isinstance(d, (int, np.integer)) or isinstance(d, (bool, np.bool_)) or d < 1:
        raise ValueError("d must be a positive integer")
    if not np.isfinite(Delta) or Delta <= 0:
        raise ValueError("Delta must be a finite positive number")
    if not isinstance(r, (int, np.integer)) or isinstance(r, (bool, np.bool_)):
        raise ValueError("r must be an integer")
    if not 1 <= r <= d:
        raise ValueError("r must satisfy 1 <= r <= d")


def _validate_watermark_bits(wm_bits):
    """Return a one-dimensional binary watermark array or raise clearly."""
    bits = np.asarray(wm_bits)
    if bits.ndim != 1:
        raise ValueError("wm_bits must be a one-dimensional array; flatten it before embedding")
    if bits.size == 0:
        raise ValueError("wm_bits must contain at least one bit")
    if not np.all(np.isin(bits, (0, 1))):
        raise ValueError("wm_bits may contain only binary values 0 and 1")
    return bits


def _validate_n_bits(n_bits):
    if (not isinstance(n_bits, (int, np.integer))
            or isinstance(n_bits, (bool, np.bool_)) or n_bits < 0):
        raise ValueError("n_bits must be a non-negative integer")


def block_views(H, W, d):
    coords = [(bi, bj) for bi in range(0, H - H % d, d) for bj in range(0, W - W % d, d)]
    return coords


def select_blocks(n_total, n_needed, seed):
    if (not isinstance(n_total, (int, np.integer))
            or isinstance(n_total, (bool, np.bool_)) or n_total < 0):
        raise ValueError("n_total must be a non-negative integer")
    if (not isinstance(n_needed, (int, np.integer))
            or isinstance(n_needed, (bool, np.bool_)) or n_needed < 0):
        raise ValueError("n_needed must be a non-negative integer")
    if n_needed > n_total:
        raise ValueError(
            f"watermark capacity exceeded: requested {n_needed} blocks, "
            f"but only {n_total} non-overlapping blocks are available")
    # Pin the bit generator instead of relying on the evolving default alias.
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.permutation(n_total)[:n_needed]


def _gather_selected_blocks(img_u8, coords, idx, d):
    """Stack selected RGB blocks as an ``(n, d, d, 3)`` array."""
    rows = np.fromiter(
        (coords[t][0] for t in idx), dtype=np.intp, count=len(idx))
    cols = np.fromiter(
        (coords[t][1] for t in idx), dtype=np.intp, count=len(idx))
    offset = np.arange(d, dtype=np.intp)
    return img_u8[
        rows[:, None, None] + offset[None, :, None],
        cols[:, None, None] + offset[None, None, :],
    ]


# Proposed RBSVD-QIM method

def feasible_delta_range(s1, s2, r, d):
    """Order-preservation range (Lemma [Order-preservation condition]):
    sigma_{r-1}^(k) >= sigma_r^(k)+delta >= sigma_{r+1}^(k), k=1,2 (sigma_0=+inf, sigma_{d+1}=0).
    Returns the common delta range that is valid for both components."""
    lo1 = s1[r] if r < d else 0.0
    hi1 = s1[r - 2] if r - 2 >= 0 else np.inf
    lo2 = s2[r] if r < d else 0.0
    hi2 = s2[r - 2] if r - 2 >= 0 else np.inf
    lo = max(lo1 - s1[r - 1], lo2 - s2[r - 1])
    hi = min(hi1 - s1[r - 1], hi2 - s2[r - 1])
    return lo, hi


def _rbq_run_fixed_target(U1, s1, V1h, U2, s2, V2h, target,
                          d, r, K_max, tau, preserve_order=True):
    """Run the output-pixel correction loop for one fixed target.

    ``preserve_order=False`` is exposed only for the controlled ablation.  The
    paper algorithm and the public :func:`rbq_embed` default keep the
    two-component singular-value ordering constraint active.
    """
    U1 = U1.copy(); s1 = s1.copy(); V1h = V1h.copy()
    U2 = U2.copy(); s2 = s2.copy(); V2h = V2h.copy()
    sbar = float((s1[r - 1] + s2[r - 1]) / 2.0)
    block_pix = None
    engaged = False
    order_clipped = False
    converged = False
    delta_l1 = 0.0
    n_iterations = 0
    for k in range(K_max):
        requested_delta = target - sbar
        if preserve_order:
            lo, hi = feasible_delta_range(s1, s2, r, d)
            delta = float(min(max(requested_delta, lo), hi))
        else:
            delta = float(requested_delta)
        if abs(delta - requested_delta) > 1e-10:
            order_clipped = True
        delta_l1 += abs(delta)
        n_iterations = k + 1
        s1w = s1.copy(); s2w = s2.copy()
        s1w[r - 1] += delta
        s2w[r - 1] += delta
        A1p = U1 @ np.diag(s1w) @ V1h
        A2p = U2 @ np.diag(s2w) @ V2h
        red, green, blue, _ = idem_to_rgb(A1p, A2p)
        block_pix = np.clip(
            np.round(np.stack([red, green, blue], axis=-1)), 0, 255
        ).astype(np.uint8)
        A1n, A2n = rgb_to_idem(block_pix.astype(np.float64))
        U1, s1, V1h = np.linalg.svd(A1n)
        U2, s2, V2h = np.linalg.svd(A2n)
        sbar = float((s1[r - 1] + s2[r - 1]) / 2.0)
        if abs(sbar - target) < tau:
            converged = True
            break
        engaged = True
    return {
        'block_pix': block_pix,
        'sbar': sbar,
        'target': float(target),
        'rho': float(abs(sbar - target)),
        'engaged': engaged,
        'order_clipped': order_clipped,
        'converged': converged,
        'delta_l1': float(delta_l1),
        'n_iterations': n_iterations,
    }


def rbq_embed(img_u8, wm_bits, d=8, Delta=40.0, r=1, seed=2026,
              K_max=3, safety=0.5, preserve_order=True,
              alternative_targets=True):
    """Embed with correction and a residual/read-back triggered fallback.

    The nearest QIM target is evaluated first with the original correction
    loop.  When its final residual does not satisfy ``rho < tau`` or its final
    rounded RGB block decodes incorrectly, the *nonnegative* adjacent same-bit
    targets ``{q-Delta, q+Delta} ∩ [0, ∞)`` are evaluated from the original
    host block.  A target is skipped when it is negative (see the
    ``target < 0.0`` guard below), so when ``q-Delta < 0`` only the single
    admissible target ``q+Delta`` is evaluated.  The selected tier is:
    converged, otherwise correctly decoded, otherwise failed; candidates
    within a tier are ordered by realized RGB distortion and final residual.
    Extraction is unchanged.

    The robustness guarantee continues to apply only to the finally selected
    blocks with ``rho < safety*Delta/4``.  A correctly decoded but
    non-converged block remains outside that guarantee.
    """
    _validate_image_and_params(img_u8, d, Delta, r)
    wm_bits = _validate_watermark_bits(wm_bits)
    if K_max < 1:
        raise ValueError("K_max must be at least 1")
    if not 0.0 < safety <= 1.0:
        raise ValueError("safety must lie in (0, 1]")
    img = img_u8.astype(np.float64)
    H, W, _ = img.shape
    A1, A2 = rgb_to_idem(img)
    coords = block_views(H, W, d)
    idx = select_blocks(len(coords), len(wm_bits), seed)
    out = img_u8.copy()
    n_nonconv = 0
    n_decode_fail = 0
    n_corrected = 0
    n_order_clipped = 0
    n_fallback_triggered = 0
    n_adjacent_target_selected = 0
    n_target_minus_delta = 0
    n_target_plus_delta = 0
    target_selection_evaluation_iterations = 0
    sum_actual_distortion_sq = 0.0
    sum_algebraic_l1_sq = 0.0
    sum_rounding_bound_sq = 0.0
    n_distortion_bound_violations = 0
    max_distortion_bound_ratio = 0.0
    tau = safety * Delta / 4.0

    for t, bit_value in zip(idx, wm_bits):
        bit = int(bit_value)
        bi, bj = coords[t]
        host_block = img_u8[bi:bi + d, bj:bj + d].astype(np.float64)
        U1, s1, V1h = np.linalg.svd(A1[bi:bi + d, bj:bj + d])
        U2, s2, V2h = np.linalg.svd(A2[bi:bi + d, bj:bj + d])
        initial_sbar = float((s1[r - 1] + s2[r - 1]) / 2.0)
        # A low-rank stress case can place the formal nearest lattice point
        # below zero. The trajectory remains valid because every increment is
        # clipped to the nonnegative, order-feasible interval; such a target
        # is unreachable/non-converged, and the fallback evaluates only
        # nonnegative adjacent alternatives.
        nearest_target = float(qim_embed_value(initial_sbar, bit, Delta))
        selected = _rbq_run_fixed_target(
            U1, s1, V1h, U2, s2, V2h, nearest_target,
            d, r, K_max, tau, preserve_order=preserve_order)
        selected['target_offset'] = 0
        target_selection_evaluation_iterations += selected['n_iterations']

        primary_decodes_correctly = (
            qim_extract_value(selected['sbar'], Delta) == bit)
        if (alternative_targets and
                (not selected['converged'] or
                 not primary_decodes_correctly)):
            n_fallback_triggered += 1
            candidates = [selected]
            for target_offset in (-1, 1):
                target = nearest_target + target_offset * Delta
                if target < 0.0:
                    continue
                if qim_extract_value(target, Delta) != bit:
                    raise AssertionError('same-bit target left its QIM lattice')
                candidate = _rbq_run_fixed_target(
                    U1, s1, V1h, U2, s2, V2h, target,
                    d, r, K_max, tau,
                    preserve_order=preserve_order)
                candidate['target_offset'] = target_offset
                candidates.append(candidate)
                target_selection_evaluation_iterations += candidate['n_iterations']

            def candidate_score(candidate):
                decoded_correctly = (
                    qim_extract_value(candidate['sbar'], Delta) == bit)
                tier = (0 if candidate['converged'] and decoded_correctly else
                        1 if decoded_correctly else 2)
                actual_sq = float(np.linalg.norm(
                    candidate['block_pix'].astype(np.float64) - host_block
                ) ** 2)
                primary = actual_sq if tier < 2 else candidate['rho']
                return (tier, primary, candidate['rho'],
                        abs(candidate['target_offset']))

            selected = min(candidates, key=candidate_score)

        block_pix = selected['block_pix']
        sbar = selected['sbar']
        delta_l1 = selected['delta_l1']
        n_iterations = selected['n_iterations']
        if selected['engaged']:
            n_corrected += 1
        if selected['order_clipped']:
            n_order_clipped += 1
        if not selected['converged']:
            n_nonconv += 1
        if qim_extract_value(sbar, Delta) != bit:
            n_decode_fail += 1
        target_offset = selected['target_offset']
        if target_offset != 0:
            n_adjacent_target_selected += 1
            if target_offset == -1:
                n_target_minus_delta += 1
            else:
                n_target_plus_delta += 1

        actual_norm = float(np.linalg.norm(
            block_pix.astype(np.float64) - host_block))
        rounding_bound = delta_l1 + n_iterations * np.sqrt(3.0) * d / 2.0
        sum_actual_distortion_sq += actual_norm ** 2
        sum_algebraic_l1_sq += delta_l1 ** 2
        sum_rounding_bound_sq += rounding_bound ** 2
        if actual_norm > rounding_bound + 1e-9:
            n_distortion_bound_violations += 1
        if rounding_bound > 0.0:
            max_distortion_bound_ratio = max(
                max_distortion_bound_ratio, actual_norm / rounding_bound)
        out[bi:bi + d, bj:bj + d] = block_pix

    return out, {
        'n_corrected': n_corrected,
        'n_nonconv': n_nonconv,
        'n_decode_fail': n_decode_fail,
        'n_order_clipped': n_order_clipped,
        'n_fallback_triggered': n_fallback_triggered,
        'n_adjacent_target_selected': n_adjacent_target_selected,
        'n_target_minus_delta': n_target_minus_delta,
        'n_target_plus_delta': n_target_plus_delta,
        'target_selection_evaluation_iterations':
            target_selection_evaluation_iterations,
        'mean_target_selection_evaluation_iterations_per_block':
            target_selection_evaluation_iterations / len(wm_bits),
        'sum_actual_distortion_sq': sum_actual_distortion_sq,
        'sum_algebraic_l1_sq': sum_algebraic_l1_sq,
        'sum_rounding_bound_sq': sum_rounding_bound_sq,
        'n_distortion_bound_violations': n_distortion_bound_violations,
        'max_distortion_bound_ratio': max_distortion_bound_ratio,
        'preserve_order': bool(preserve_order),
        'alternative_targets': bool(alternative_targets),
    }


def rbq_extract(img_u8, n_bits, d=8, Delta=40.0, r=1, seed=2026):
    _validate_image_and_params(img_u8, d, Delta, r)
    _validate_n_bits(n_bits)
    H, W, _ = img_u8.shape
    coords = block_views(H, W, d)
    idx = select_blocks(len(coords), n_bits, seed)
    if n_bits == 0:
        return np.zeros(0, dtype=np.uint8)
    blocks = _gather_selected_blocks(img_u8, coords, idx, d)
    A1, A2 = rgb_to_idem(blocks)
    s1 = np.linalg.svd(A1, compute_uv=False)
    s2 = np.linalg.svd(A2, compute_uv=False)
    sbar = (s1[:, r - 1] + s2[:, r - 1]) / 2.0
    return (np.mod(sbar, Delta) >= Delta / 2.0).astype(np.uint8)


# Baseline 1: channel-wise real SVD-QIM

def chan_embed(img_u8, wm_bits, d=8, Delta=40.0, r=1, seed=2026):
    _validate_image_and_params(img_u8, d, Delta, r)
    wm_bits = _validate_watermark_bits(wm_bits)
    img = img_u8.astype(np.float64)
    H, W, _ = img.shape
    coords = block_views(H, W, d)
    idx = select_blocks(len(coords), len(wm_bits), seed)
    for t, bit in zip(idx, wm_bits):
        bi, bj = coords[t]
        svs, Us, Vhs, ss = [], [], [], []
        for c in range(3):
            U, s, Vh = np.linalg.svd(img[bi:bi + d, bj:bj + d, c])
            Us.append(U); Vhs.append(Vh); ss.append(s)
        sbar = np.mean([s[r - 1] for s in ss])
        delta = _nearest_nonnegative_qim_target(sbar, bit, Delta) - sbar
        for c in range(3):
            s = ss[c].copy(); s[r - 1] += delta
            img[bi:bi + d, bj:bj + d, c] = Us[c] @ np.diag(s) @ Vhs[c]
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def chan_extract(img_u8, n_bits, d=8, Delta=40.0, r=1, seed=2026):
    _validate_image_and_params(img_u8, d, Delta, r)
    _validate_n_bits(n_bits)
    H, W, _ = img_u8.shape
    coords = block_views(H, W, d)
    idx = select_blocks(len(coords), n_bits, seed)
    if n_bits == 0:
        return np.zeros(0, dtype=np.uint8)
    blocks = _gather_selected_blocks(img_u8, coords, idx, d)
    channels = np.moveaxis(
        blocks.astype(np.float64), -1, 1)  # (n, 3, d, d)
    spectra = np.linalg.svd(channels, compute_uv=False)
    sbar = spectra[:, :, r - 1].mean(axis=1)
    return (np.mod(sbar, Delta) >= Delta / 2.0).astype(np.uint8)


def _common_feasible_delta_range(spectra, r):
    """Common order-preserving delta range for several SVD spectra."""
    lows, highs = [], []
    for s in spectra:
        d = len(s)
        lower_neighbor = s[r] if r < d else 0.0
        upper_neighbor = s[r - 2] if r - 2 >= 0 else np.inf
        lows.append(lower_neighbor - s[r - 1])
        highs.append(upper_neighbor - s[r - 1])
    return max(lows), min(highs)


def chan_embed_corrected(img_u8, wm_bits, d=8, Delta=40.0, r=1,
                         seed=2026, K_max=3, safety=0.5,
                         preserve_order=False):
    """Channel-SVD baseline: same output-pixel correction loop.

    ``preserve_order`` is configurable for legacy diagnostic runs.  The final
    controlled comparison enables it so that all three methods use the same
    order-preserving output-pixel correction protocol.
    """
    _validate_image_and_params(img_u8, d, Delta, r)
    wm_bits = _validate_watermark_bits(wm_bits)
    if K_max < 1:
        raise ValueError("K_max must be at least 1")
    if not 0.0 < safety <= 1.0:
        raise ValueError("safety must lie in (0, 1]")
    H, W, _ = img_u8.shape
    coords = block_views(H, W, d)
    idx = select_blocks(len(coords), len(wm_bits), seed)
    out = img_u8.copy()
    tau = safety * Delta / 4.0
    info = {'n_corrected': 0, 'n_nonconv': 0,
            'n_decode_fail': 0, 'n_order_clipped': 0}

    for t, bit in zip(idx, wm_bits):
        bi, bj = coords[t]
        block = out[bi:bi + d, bj:bj + d].astype(np.float64)
        decomps = [np.linalg.svd(block[..., c]) for c in range(3)]
        sbar = float(np.mean([dec[1][r - 1] for dec in decomps]))
        target = _nearest_nonnegative_qim_target(sbar, bit, Delta)
        engaged = False
        order_clipped = False
        converged = False

        for _ in range(K_max):
            spectra = [dec[1] for dec in decomps]
            desired = target - sbar
            if preserve_order:
                lo, hi = _common_feasible_delta_range(spectra, r)
                delta = min(max(desired, lo), hi)
                if delta != desired:
                    order_clipped = True
            else:
                delta = desired

            channels = []
            for U, s, Vh in decomps:
                sw = s.copy()
                sw[r - 1] += delta
                channels.append(U @ np.diag(sw) @ Vh)
            block_pix = np.clip(
                np.round(np.stack(channels, axis=-1)), 0, 255).astype(np.uint8)
            decomps = [np.linalg.svd(block_pix[..., c].astype(np.float64))
                       for c in range(3)]
            sbar = float(np.mean([dec[1][r - 1] for dec in decomps]))
            if abs(sbar - target) < tau:
                converged = True
                break
            engaged = True

        if engaged:
            info['n_corrected'] += 1
        if order_clipped:
            info['n_order_clipped'] += 1
        if not converged:
            info['n_nonconv'] += 1
        if qim_extract_value(sbar, Delta) != bit:
            info['n_decode_fail'] += 1
        out[bi:bi + d, bj:bj + d] = block_pix
    return out, info


# Complementary fairness control: single-G SVD-QIM

def green_extract(img_u8, n_bits, d=8, Delta=40.0, r=1, seed=2026):
    """Blind QIM extraction from the r-th singular value of the green channel only."""
    _validate_image_and_params(img_u8, d, Delta, r)
    _validate_n_bits(n_bits)
    H, W, _ = img_u8.shape
    coords = block_views(H, W, d)
    idx = select_blocks(len(coords), n_bits, seed)
    bits = np.zeros(n_bits, dtype=np.uint8)
    for out_i, t in enumerate(idx):
        bi, bj = coords[t]
        s = np.linalg.svd(
            img_u8[bi:bi + d, bj:bj + d, 1].astype(np.float64),
            compute_uv=False)
        bits[out_i] = qim_extract_value(s[r - 1], Delta)
    return bits


def green_embed_corrected(img_u8, wm_bits, d=8, Delta=40.0, r=1,
                          seed=2026, K_max=3, safety=0.5,
                          preserve_order=False):
    """Single-G SVD-QIM; same output-correction loop as the other methods."""
    _validate_image_and_params(img_u8, d, Delta, r)
    wm_bits = _validate_watermark_bits(wm_bits)
    if K_max < 1:
        raise ValueError("K_max must be at least 1")
    if not 0.0 < safety <= 1.0:
        raise ValueError("safety must lie in (0, 1]")
    H, W, _ = img_u8.shape
    coords = block_views(H, W, d)
    idx = select_blocks(len(coords), len(wm_bits), seed)
    out = img_u8.copy()
    tau = safety * Delta / 4.0
    info = {'n_corrected': 0, 'n_nonconv': 0,
            'n_decode_fail': 0, 'n_order_clipped': 0}

    for t, bit in zip(idx, wm_bits):
        bi, bj = coords[t]
        block = out[bi:bi + d, bj:bj + d].copy()
        U, s, Vh = np.linalg.svd(block[..., 1].astype(np.float64))
        feature = float(s[r - 1])
        target = _nearest_nonnegative_qim_target(feature, bit, Delta)
        engaged = False
        order_clipped = False
        converged = False

        for _ in range(K_max):
            desired = target - feature
            if preserve_order:
                lo, hi = _common_feasible_delta_range([s], r)
                delta = min(max(desired, lo), hi)
                if delta != desired:
                    order_clipped = True
            else:
                delta = desired
            sw = s.copy()
            sw[r - 1] += delta
            green = np.clip(np.round(U @ np.diag(sw) @ Vh), 0, 255).astype(np.uint8)
            block_pix = block.copy()
            block_pix[..., 1] = green
            U, s, Vh = np.linalg.svd(green.astype(np.float64))
            feature = float(s[r - 1])
            if abs(feature - target) < tau:
                converged = True
                break
            engaged = True

        if engaged:
            info['n_corrected'] += 1
        if order_clipped:
            info['n_order_clipped'] += 1
        if not converged:
            info['n_nonconv'] += 1
        if qim_extract_value(feature, Delta) != bit:
            info['n_decode_fail'] += 1
        out[bi:bi + d, bj:bj + d] = block_pix
    return out, info


# Baseline 2: Hamilton QSVD-QIM through the complex adjoint

def quat_adjoint(Rb, Gb, Bb):
    """Complex adjoint of one or more pure Hamilton quaternion blocks.

    The final two axes contain the square block dimensions; any leading axes
    are retained as batch axes.  For ``q = R i + G j + B k``, write
    ``q = C1 + C2 j`` with ``C1 = R i`` and ``C2 = G + B i``.
    """
    C1 = 1j * Rb
    C2 = Gb + 1j * Bb
    top = np.concatenate([C1, C2], axis=-1)
    bot = np.concatenate([-np.conj(C2), np.conj(C1)], axis=-1)
    return np.concatenate([top, bot], axis=-2)


def adjoint_to_quat(Aq, d):
    """Projection from the 2d x 2d adjoint back to quaternion structure (symmetrization)."""
    C1 = (Aq[:d, :d] + np.conj(Aq[d:, d:])) / 2.0
    C2 = (Aq[:d, d:] - np.conj(Aq[d:, :d])) / 2.0
    q0 = C1.real
    R = C1.imag
    G = C2.real
    B = C2.imag
    return R, G, B, q0


def qsvd_embed(img_u8, wm_bits, d=8, Delta=40.0, r=1, seed=2026):
    _validate_image_and_params(img_u8, d, Delta, r)
    wm_bits = _validate_watermark_bits(wm_bits)
    img = img_u8.astype(np.float64)
    H, W, _ = img.shape
    coords = block_views(H, W, d)
    idx = select_blocks(len(coords), len(wm_bits), seed)
    for t, bit in zip(idx, wm_bits):
        bi, bj = coords[t]
        Aq = quat_adjoint(img[bi:bi + d, bj:bj + d, 0],
                          img[bi:bi + d, bj:bj + d, 1],
                          img[bi:bi + d, bj:bj + d, 2])
        U, s, Vh = np.linalg.svd(Aq)
        # quaternion singular values come in pairs: (s[0],s[1]), (s[2],s[3]), ...
        i0, i1 = 2 * (r - 1), 2 * (r - 1) + 1
        sbar = (s[i0] + s[i1]) / 2.0
        delta = _nearest_nonnegative_qim_target(sbar, bit, Delta) - sbar
        sw = s.copy(); sw[i0] += delta; sw[i1] += delta
        Aw = U @ np.diag(sw) @ Vh
        R, G, B, _ = adjoint_to_quat(Aw, d)
        img[bi:bi + d, bj:bj + d, 0] = R
        img[bi:bi + d, bj:bj + d, 1] = G
        img[bi:bi + d, bj:bj + d, 2] = B
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def qsvd_extract(img_u8, n_bits, d=8, Delta=40.0, r=1, seed=2026):
    _validate_image_and_params(img_u8, d, Delta, r)
    _validate_n_bits(n_bits)
    H, W, _ = img_u8.shape
    coords = block_views(H, W, d)
    idx = select_blocks(len(coords), n_bits, seed)
    if n_bits == 0:
        return np.zeros(0, dtype=np.uint8)
    blocks = _gather_selected_blocks(img_u8, coords, idx, d).astype(
        np.float64)
    Aq = quat_adjoint(
        blocks[..., 0], blocks[..., 1], blocks[..., 2])
    spectra = np.linalg.svd(Aq, compute_uv=False)
    i0, i1 = 2 * (r - 1), 2 * (r - 1) + 1
    sbar = (spectra[:, i0] + spectra[:, i1]) / 2.0
    return (np.mod(sbar, Delta) >= Delta / 2.0).astype(np.uint8)


def _qsvd_feasible_delta_range(s, r):
    """Order-preserving common delta range for paired adjoint singular values."""
    i0, i1 = 2 * (r - 1), 2 * (r - 1) + 1
    lo = (s[i1 + 1] - s[i1]) if i1 + 1 < len(s) else -s[i1]
    hi = (s[i0 - 1] - s[i0]) if i0 > 0 else np.inf
    return lo, hi


def qsvd_embed_corrected(img_u8, wm_bits, d=8, Delta=40.0, r=1,
                         seed=2026, K_max=3, safety=0.5,
                         preserve_order=False):
    """QSVD baseline: same output-pixel correction loop."""
    _validate_image_and_params(img_u8, d, Delta, r)
    wm_bits = _validate_watermark_bits(wm_bits)
    if K_max < 1:
        raise ValueError("K_max must be at least 1")
    if not 0.0 < safety <= 1.0:
        raise ValueError("safety must lie in (0, 1]")
    H, W, _ = img_u8.shape
    coords = block_views(H, W, d)
    idx = select_blocks(len(coords), len(wm_bits), seed)
    out = img_u8.copy()
    tau = safety * Delta / 4.0
    info = {'n_corrected': 0, 'n_nonconv': 0,
            'n_decode_fail': 0, 'n_order_clipped': 0}
    i0, i1 = 2 * (r - 1), 2 * (r - 1) + 1

    for t, bit in zip(idx, wm_bits):
        bi, bj = coords[t]
        block = out[bi:bi + d, bj:bj + d].astype(np.float64)
        Aq = quat_adjoint(block[..., 0], block[..., 1], block[..., 2])
        U, s, Vh = np.linalg.svd(Aq)
        sbar = float((s[i0] + s[i1]) / 2.0)
        target = _nearest_nonnegative_qim_target(sbar, bit, Delta)
        engaged = False
        order_clipped = False
        converged = False

        for _ in range(K_max):
            desired = target - sbar
            if preserve_order:
                lo, hi = _qsvd_feasible_delta_range(s, r)
                delta = min(max(desired, lo), hi)
                if delta != desired:
                    order_clipped = True
            else:
                delta = desired

            sw = s.copy()
            sw[i0] += delta
            sw[i1] += delta
            Aw = U @ np.diag(sw) @ Vh
            Rb, Gb, Bb, _ = adjoint_to_quat(Aw, d)
            block_pix = np.clip(
                np.round(np.stack([Rb, Gb, Bb], axis=-1)), 0, 255).astype(np.uint8)
            Aq = quat_adjoint(block_pix[..., 0].astype(np.float64),
                              block_pix[..., 1].astype(np.float64),
                              block_pix[..., 2].astype(np.float64))
            U, s, Vh = np.linalg.svd(Aq)
            sbar = float((s[i0] + s[i1]) / 2.0)
            if abs(sbar - target) < tau:
                converged = True
                break
            engaged = True

        if engaged:
            info['n_corrected'] += 1
        if order_clipped:
            info['n_order_clipped'] += 1
        if not converged:
            info['n_nonconv'] += 1
        if qim_extract_value(sbar, Delta) != bit:
            info['n_decode_fail'] += 1
        out[bi:bi + d, bj:bj + d] = block_pix
    return out, info


# Attacks. Labels remain stable because they are keys in the reference JSON.

def atk_jpeg(img_u8, q):
    buf = io.BytesIO()
    Image.fromarray(img_u8).save(
        buf, format='JPEG', quality=int(q), subsampling=2,
        optimize=False, progressive=False)
    buf.seek(0)
    return np.array(Image.open(buf).convert('RGB'))


def atk_jpeg2000(img_u8, compression_ratio):
    """JPEG2000 round trip at a nominal OpenJPEG compression ratio.

    Pillow's ``quality_mode='rates'`` passes the requested compression ratio
    to OpenJPEG.  A single irreversible quality layer and the reversible RGB
    multiple-component transform are fixed for every experiment.  The JP2
    container is decoded back to an 8-bit RGB array before extraction.
    """
    ratio = float(compression_ratio)
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("compression_ratio must be positive and finite")
    buf = io.BytesIO()
    Image.fromarray(np.asarray(img_u8, dtype=np.uint8), mode='RGB').save(
        buf,
        format='JPEG2000',
        quality_mode='rates',
        quality_layers=[ratio],
        irreversible=True,
        mct=1,
        no_jp2=False,
    )
    buf.seek(0)
    with Image.open(buf) as decoded:
        return np.asarray(decoded.convert('RGB'), dtype=np.uint8)


def atk_gauss(img_u8, std, seed=7):
    rng = np.random.Generator(np.random.PCG64(seed))
    out = img_u8.astype(np.float64) + rng.normal(0, std, img_u8.shape)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def atk_sp(img_u8, dens, seed=7):
    rng = np.random.Generator(np.random.PCG64(seed))
    out = img_u8.copy()
    m = rng.random(img_u8.shape[:2])
    out[m < dens / 2] = 0
    out[m > 1 - dens / 2] = 255
    return out


def atk_median(img_u8, k=3):
    out = np.stack([ndimage.median_filter(img_u8[..., c], size=k) for c in range(3)], axis=-1)
    return out


def atk_average(img_u8, k=3):
    out = np.stack([ndimage.uniform_filter(img_u8[..., c].astype(np.float64), size=k)
                    for c in range(3)], axis=-1)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def atk_crop(img_u8, frac=0.25):
    """Zero a top-left square covering ``frac`` of the area; preserve size."""
    out = img_u8.copy()
    H, W, _ = out.shape
    h = int(round(H * np.sqrt(frac)))
    w = int(round(W * np.sqrt(frac)))
    out[:h, :w] = 0
    return out


def atk_scale(img_u8, f=0.5):
    H, W, _ = img_u8.shape
    small = Image.fromarray(img_u8).resize((int(W * f), int(H * f)), Image.BICUBIC)
    back = small.resize((W, H), Image.BICUBIC)
    return np.array(back)


def atk_rotate(img_u8, ang=2.0):
    im = Image.fromarray(img_u8).rotate(ang, resample=Image.BICUBIC, expand=False)
    back = im.rotate(-ang, resample=Image.BICUBIC, expand=False)
    return np.array(back)


def atk_bright(img_u8, add=10):
    return np.clip(img_u8.astype(np.int16) + add, 0, 255).astype(np.uint8)


def atk_sharpen(img_u8):
    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float)
    out = np.stack([ndimage.convolve(img_u8[..., c].astype(np.float64), k, mode='reflect')
                    for c in range(3)], axis=-1)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


ATTACKS = [
    ("No attack",                          lambda x: x.copy()),
    ("JPEG (Q=90)",                        lambda x: atk_jpeg(x, 90)),
    ("JPEG (Q=70)",                        lambda x: atk_jpeg(x, 70)),
    ("JPEG (Q=50)",                        lambda x: atk_jpeg(x, 50)),
    ("JPEG (Q=30)",                        lambda x: atk_jpeg(x, 30)),
    ("JPEG2000 (CR=2)",                    lambda x: atk_jpeg2000(x, 2)),
    ("JPEG2000 (CR=4)",                    lambda x: atk_jpeg2000(x, 4)),
    ("JPEG2000 (CR=8)",                    lambda x: atk_jpeg2000(x, 8)),
    ("Gaussian noise (σ=5)",               lambda x: atk_gauss(x, 5)),
    ("Gaussian noise (σ=10)",              lambda x: atk_gauss(x, 10)),
    ("Salt & pepper (1%)",                 lambda x: atk_sp(x, 0.01)),
    ("Salt & pepper (2%)",                 lambda x: atk_sp(x, 0.02)),
    ("Median filter (3×3)",                lambda x: atk_median(x, 3)),
    ("Average filter (3×3)",               lambda x: atk_average(x, 3)),
    ("Zero-masking (25%)",                 lambda x: atk_crop(x, 0.25)),
    ("Resize-back (0.5×)",                 lambda x: atk_scale(x, 0.5)),
    ("Inverse-registered rotation (2°)",   lambda x: atk_rotate(x, 2.0)),
    ("Brightness (+10)",                   lambda x: atk_bright(x, 10)),
    ("Sharpening",                         lambda x: atk_sharpen(x)),
]


# Metrics

def psnr(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    return 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else np.inf


def ssim_color(a, b):
    try:
        from skimage.metrics import structural_similarity
        return structural_similarity(a, b, channel_axis=-1, data_range=255)
    except ModuleNotFoundError:
        # Dependency-light fallback matching the scikit-image defaults used by
        # this project: 7x7 uniform window, sample covariance, K1=.01, K2=.03,
        # and the mean of the three independently evaluated RGB channels.
        x = a.astype(np.float64)
        y = b.astype(np.float64)
        win_size = 7
        npix = win_size ** 2
        cov_norm = npix / (npix - 1.0)
        c1 = (0.01 * 255.0) ** 2
        c2 = (0.03 * 255.0) ** 2
        pad = (win_size - 1) // 2
        scores = []
        for channel in range(x.shape[-1]):
            xc = x[..., channel]
            yc = y[..., channel]
            ux = ndimage.uniform_filter(xc, size=win_size)
            uy = ndimage.uniform_filter(yc, size=win_size)
            uxx = ndimage.uniform_filter(xc * xc, size=win_size)
            uyy = ndimage.uniform_filter(yc * yc, size=win_size)
            uxy = ndimage.uniform_filter(xc * yc, size=win_size)
            vx = cov_norm * (uxx - ux * ux)
            vy = cov_norm * (uyy - uy * uy)
            vxy = cov_norm * (uxy - ux * uy)
            numerator = (2.0 * ux * uy + c1) * (2.0 * vxy + c2)
            denominator = (ux * ux + uy * uy + c1) * (vx + vy + c2)
            ssim_map = numerator / denominator
            scores.append(float(ssim_map[pad:-pad, pad:-pad].mean()))
        return float(np.mean(scores))


def ber(w, w2):
    return float(np.mean(w.astype(np.uint8) != w2.astype(np.uint8)))
