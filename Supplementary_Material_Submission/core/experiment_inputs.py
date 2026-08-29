"""Load the images used in the experiments.

All paths are resolved relative to this Supplementary Material package.  The
module prepares the two binary and two RGB121 watermarks as 1024-bit payloads
using the same deterministic rules as the experiments reported in the paper.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from watermark_codec import decode_rgb121, encode_binary, prepare_binary_source, prepare_rgb121_source


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = PACKAGE_ROOT / "images"
PAYLOAD_BITS = 1024


HOST_FILES = OrderedDict([
    ("airplane", "airplane.bmp"),
    ("barbara", "barbara.bmp"),
    ("boats", "boats.bmp"),
    ("fruits", "fruits.png"),
    ("goldhill", "goldhill.bmp"),
    ("mandrill", "mandrill.tif"),
    ("peppers", "peppers.png"),
    ("sailboat", "sailboat.bmp"),
])

WATERMARK_FILES = OrderedDict([
    ("reference_logo", "sau_32x32_binary.npy"),
    ("published_peugeot", "peugeot_binary_source.png"),
    ("published_number", "number_16x16_rgb121.png"),
    ("published_penguin", "penguin_16x16_rgb121.png"),
])

WATERMARK_LABELS = {
    "reference_logo": "SAU",
    "published_peugeot": "Peugeot",
    "published_number": "Number",
    "published_penguin": "Penguin",
}


@dataclass(frozen=True)
class WatermarkPayload:
    id: str
    label: str
    kind: str
    bits: np.ndarray
    preview_rgb: np.ndarray


def _load_rgb(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as opened:
        image = np.asarray(opened.convert("RGB"), dtype=np.uint8).copy()
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an RGB image at {path}; got {image.shape}")
    return np.ascontiguousarray(image)


def load_hosts() -> "OrderedDict[str, np.ndarray]":
    hosts: "OrderedDict[str, np.ndarray]" = OrderedDict()
    for host_id, filename in HOST_FILES.items():
        hosts[host_id] = _load_rgb(IMAGE_ROOT / "hosts" / filename)
    return hosts


def _binary_preview(binary: np.ndarray) -> np.ndarray:
    gray = np.where(binary == 1, 0, 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def load_watermarks() -> "OrderedDict[str, WatermarkPayload]":
    root = IMAGE_ROOT / "watermarks"
    result: "OrderedDict[str, WatermarkPayload]" = OrderedDict()

    sau = np.load(root / WATERMARK_FILES["reference_logo"], allow_pickle=False)
    sau = np.asarray(sau, dtype=np.uint8)
    if sau.shape != (32, 32) or not np.all(np.isin(sau, (0, 1))):
        raise ValueError("SAU must be a 32 x 32 binary array")
    result["reference_logo"] = WatermarkPayload(
        "reference_logo", WATERMARK_LABELS["reference_logo"], "binary",
        encode_binary(sau), _binary_preview(sau),
    )

    peugeot, bits, _ = prepare_binary_source(
        root / WATERMARK_FILES["published_peugeot"]
    )
    result["published_peugeot"] = WatermarkPayload(
        "published_peugeot", WATERMARK_LABELS["published_peugeot"], "binary",
        bits, _binary_preview(peugeot),
    )

    for watermark_id in ("published_number", "published_penguin"):
        preview, bits, _ = prepare_rgb121_source(
            root / WATERMARK_FILES[watermark_id], require_exact=True
        )
        if not np.array_equal(preview, decode_rgb121(bits)):
            raise AssertionError(f"RGB121 round trip failed for {watermark_id}")
        result[watermark_id] = WatermarkPayload(
            watermark_id, WATERMARK_LABELS[watermark_id], "RGB121", bits, preview
        )

    for payload in result.values():
        if payload.bits.dtype != np.uint8 or payload.bits.shape != (PAYLOAD_BITS,):
            raise ValueError(f"Invalid payload for {payload.id}: {payload.bits.shape}")
    return result


def load_kodak24() -> "OrderedDict[str, np.ndarray]":
    images: "OrderedDict[str, np.ndarray]" = OrderedDict()
    for index in range(1, 25):
        image_id = f"kodim{index:02d}"
        images[image_id] = _load_rgb(IMAGE_ROOT / "kodak24" / f"{image_id}.png")
    return images


def iter_host_watermark_pairs(
) -> Iterator[tuple[str, np.ndarray, str, WatermarkPayload]]:
    for host_id, host in load_hosts().items():
        for watermark_id, payload in load_watermarks().items():
            yield host_id, host, watermark_id, payload


def iter_kodak_watermark_pairs(
) -> Iterator[tuple[str, np.ndarray, str, WatermarkPayload]]:
    for host_id, host in load_kodak24().items():
        for watermark_id, payload in load_watermarks().items():
            yield host_id, host, watermark_id, payload
