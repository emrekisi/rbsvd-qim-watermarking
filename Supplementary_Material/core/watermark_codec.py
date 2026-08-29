"""Deterministic 1024-bit watermark codecs for the final protocol.

Binary sources are centre-cropped to a square, resized to 32 x 32 with
Pillow's LANCZOS filter, converted to integer BT.601 luma, and thresholded at
128.  Dark foreground is encoded as bit one.

Colour sources that already are paletteless 16 x 16 RGB images on the exact
RGB121 reconstruction lattice are used pixel-for-pixel, without crop, resize,
or quantization.  Other colour sources retain the deterministic centre-crop,
16 x 16 LANCZOS resize, and RGB121 quantization path.  Each pixel is represented
in row-major order as ``[R, G_MSB, G_LSB, B]``.  R and B use threshold 128.  G
is assigned to the nearest of 0, 85, 170, and 255 (natural-binary indices 00,
01, 10, and 11).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


BINARY_SIZE = (32, 32)
RGB121_SIZE = (16, 16)
PAYLOAD_BITS = 1024
RGB121_GREEN_LEVELS = np.array([0, 85, 170, 255], dtype=np.uint8)
RGB121_RED_BLUE_LEVELS = np.array([0, 255], dtype=np.uint8)


def _as_uint8_rgb(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("expected an RGB array with shape (height, width, 3)")
    if value.dtype != np.uint8:
        raise ValueError("expected uint8 RGB data")
    return value


def _validate_bits(bits: np.ndarray, expected: int = PAYLOAD_BITS) -> np.ndarray:
    value = np.asarray(bits)
    if value.ndim != 1 or value.size != expected:
        raise ValueError(f"expected a one-dimensional {expected}-bit payload")
    if value.dtype != np.uint8:
        value = value.astype(np.uint8)
    if not np.all(np.isin(value, (0, 1))):
        raise ValueError("payload contains values other than 0 and 1")
    return np.ascontiguousarray(value)


def centre_crop_square(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Return the deterministic centred square crop and its Pillow crop box."""
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    box = (left, top, left + side, top + side)
    return image.crop(box), box


def resize_source_rgb(
    source_path: str | Path,
    size: tuple[int, int],
) -> tuple[np.ndarray, dict[str, object]]:
    """Load RGB, centre-crop, and resize with the pinned LANCZOS rule."""
    path = Path(source_path)
    with Image.open(path) as opened:
        source_mode = opened.mode
        source_size = opened.size
        rgb = opened.convert("RGB")
        cropped, crop_box = centre_crop_square(rgb)
        resized = cropped.resize(size, resample=Image.Resampling.LANCZOS)
        array = np.asarray(resized, dtype=np.uint8).copy()
    return array, {
        "source_mode": source_mode,
        "source_width": int(source_size[0]),
        "source_height": int(source_size[1]),
        "centre_crop_box_left_top_right_bottom": [int(v) for v in crop_box],
        "resize_width": int(size[0]),
        "resize_height": int(size[1]),
        "resampling": "Pillow Image.Resampling.LANCZOS",
    }


def encode_binary(binary_image: np.ndarray) -> np.ndarray:
    value = np.asarray(binary_image)
    if value.shape != (BINARY_SIZE[1], BINARY_SIZE[0]):
        raise ValueError("binary watermark must have shape (32, 32)")
    if not np.all(np.isin(value, (0, 1))):
        raise ValueError("binary watermark may contain only 0 and 1")
    return np.ascontiguousarray(value.astype(np.uint8).reshape(-1, order="C"))


def decode_binary(bits: np.ndarray) -> np.ndarray:
    value = _validate_bits(bits)
    return value.reshape((BINARY_SIZE[1], BINARY_SIZE[0]), order="C").copy()


def prepare_binary_source(source_path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    resized_rgb, metadata = resize_source_rgb(source_path, BINARY_SIZE)
    rgb32 = resized_rgb.astype(np.uint32)
    # Rounded integer BT.601 luma; explicit arithmetic avoids an implicit
    # library-specific RGB-to-L conversion.
    luma = (
        299 * rgb32[..., 0]
        + 587 * rgb32[..., 1]
        + 114 * rgb32[..., 2]
        + 500
    ) // 1000
    binary = (luma < 128).astype(np.uint8)
    bits = encode_binary(binary)
    metadata.update({
        "luma": "round((299*R + 587*G + 114*B)/1000)",
        "threshold": 128,
        "polarity": "luma < 128 maps to bit 1 (dark foreground)",
        "flatten_order": "row-major (C)",
    })
    return binary, bits, metadata


def encode_rgb121(rgb_image: np.ndarray) -> np.ndarray:
    rgb = _as_uint8_rgb(rgb_image)
    if rgb.shape != (RGB121_SIZE[1], RGB121_SIZE[0], 3):
        raise ValueError("RGB121 watermark must have shape (16, 16, 3)")
    r_bit = (rgb[..., 0] >= 128).astype(np.uint8)
    # For integer samples there are no midpoint ties: boundaries are
    # 42.5, 127.5, and 212.5.  The integer rule below is exact nearest-level
    # assignment for the ordered levels 0, 85, 170, 255.
    g_index = np.minimum(
        (rgb[..., 1].astype(np.uint16) + 42) // 85,
        3,
    ).astype(np.uint8)
    g_msb = (g_index >> 1) & 1
    g_lsb = g_index & 1
    b_bit = (rgb[..., 2] >= 128).astype(np.uint8)
    symbols = np.stack((r_bit, g_msb, g_lsb, b_bit), axis=-1)
    return np.ascontiguousarray(symbols.reshape(-1, order="C"))


def decode_rgb121(bits: np.ndarray) -> np.ndarray:
    value = _validate_bits(bits)
    symbols = value.reshape((RGB121_SIZE[1], RGB121_SIZE[0], 4), order="C")
    r = symbols[..., 0] * np.uint8(255)
    g_index = (symbols[..., 1] << 1) | symbols[..., 2]
    g = RGB121_GREEN_LEVELS[g_index]
    b = symbols[..., 3] * np.uint8(255)
    return np.ascontiguousarray(np.stack((r, g, b), axis=-1).astype(np.uint8))


def _rgb121_level_summary(rgb: np.ndarray) -> dict[str, object]:
    value = _as_uint8_rgb(rgb)
    colors = np.unique(value.reshape(-1, 3), axis=0)
    return {
        "R_levels": [int(item) for item in np.unique(value[..., 0])],
        "G_levels": [int(item) for item in np.unique(value[..., 1])],
        "B_levels": [int(item) for item in np.unique(value[..., 2])],
        "unique_color_count": int(colors.shape[0]),
        "unique_colors": [[int(channel) for channel in color] for color in colors],
    }


def _is_exact_rgb121(rgb: np.ndarray) -> bool:
    value = _as_uint8_rgb(rgb)
    return bool(
        value.shape == (RGB121_SIZE[1], RGB121_SIZE[0], 3)
        and np.all(np.isin(value[..., 0], RGB121_RED_BLUE_LEVELS))
        and np.all(np.isin(value[..., 1], RGB121_GREEN_LEVELS))
        and np.all(np.isin(value[..., 2], RGB121_RED_BLUE_LEVELS))
    )


def prepare_rgb121_source(
    source_path: str | Path,
    *,
    require_exact: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    path = Path(source_path)
    with Image.open(path) as opened:
        source_mode = opened.mode
        source_size = opened.size
        source_bands = list(opened.getbands())
        source_has_palette = opened.palette is not None
        exact_source = (
            source_mode == "RGB"
            and source_size == RGB121_SIZE
            and not source_has_palette
        )
        source_rgb = (
            np.asarray(opened, dtype=np.uint8).copy() if exact_source else None
        )
    exact_source = bool(
        exact_source
        and source_rgb is not None
        and _is_exact_rgb121(source_rgb)
    )
    if require_exact and not exact_source:
        raise ValueError(
            "required exact RGB121 source must be a paletteless 16 x 16 RGB "
            "image with R,B in {0,255} and G in {0,85,170,255}"
        )
    if exact_source:
        prepared_rgb = np.ascontiguousarray(source_rgb)
        metadata: dict[str, object] = {
            "source_mode": source_mode,
            "source_bands": source_bands,
            "source_has_palette": source_has_palette,
            "source_width": int(source_size[0]),
            "source_height": int(source_size[1]),
            "centre_crop_box_left_top_right_bottom": [0, 0, 16, 16],
            "resize_width": 16,
            "resize_height": 16,
            "resampling": "none",
            "rgb_conversion": "none",
            "source_used_without_resize_or_quantization": True,
            "exact_rgb121_level_compliance": True,
            **_rgb121_level_summary(prepared_rgb),
        }
    else:
        prepared_rgb, metadata = resize_source_rgb(path, RGB121_SIZE)
        metadata.update({
            "source_bands": source_bands,
            "source_has_palette": source_has_palette,
            "rgb_conversion": "Pillow convert('RGB')",
            "source_used_without_resize_or_quantization": False,
            "exact_rgb121_level_compliance": False,
        })
    bits = encode_rgb121(prepared_rgb)
    reconstructed = decode_rgb121(bits)
    metadata.update({
        "codec": "RGB121",
        "pixel_bit_order": ["R", "G_MSB", "G_LSB", "B"],
        "pixel_traversal": "row-major (C)",
        "R_threshold": 128,
        "B_threshold": 128,
        "G_nearest_levels": [0, 85, 170, 255],
        "G_index_codes": ["00", "01", "10", "11"],
    })
    return reconstructed, bits, metadata


def pack_bits(bits: np.ndarray) -> bytes:
    """Pack 1024 bits into 128 bytes, first stream bit in the byte MSB."""
    value = _validate_bits(bits)
    return np.packbits(value, bitorder="big").tobytes()


def unpack_bits(payload: bytes) -> np.ndarray:
    if len(payload) != PAYLOAD_BITS // 8:
        raise ValueError("packed 1024-bit payload must contain exactly 128 bytes")
    array = np.frombuffer(payload, dtype=np.uint8)
    return np.unpackbits(array, bitorder="big")[:PAYLOAD_BITS].astype(np.uint8)
