"""Dataset location, integrity verification and read-only loading."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src import config


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream a SHA-256 digest over ``path`` without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_dataset(path: Path) -> dict:
    """Check the file against the pinned identity. Read-only; never modifies.

    Verification rests on the SHA-256 digest, because a digest match establishes
    byte-identity and therefore identity of shape as well. The declared shape is
    confirmed separately, after parsing, by :func:`load_raw_dataset`.

    Returns a dict of observed and expected values plus a ``verified`` flag.
    Raises ``FileNotFoundError`` if the file is absent.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    size = path.stat().st_size
    observed_sha256 = sha256_of(path)

    return {
        "path": str(path),
        "filename": path.name,
        "size_bytes": size,
        "expected_size_bytes": config.EXPECTED_SIZE_BYTES,
        "sha256": observed_sha256,
        "expected_sha256": config.EXPECTED_SHA256,
        "expected_shape": config.EXPECTED_SHAPE,
        "size_matches": size == config.EXPECTED_SIZE_BYTES,
        "sha256_matches": observed_sha256 == config.EXPECTED_SHA256,
        "verified": (
            size == config.EXPECTED_SIZE_BYTES
            and observed_sha256 == config.EXPECTED_SHA256
        ),
    }


def load_raw_dataset(
    path: Path | None = None,
    verify: bool = True,
    expect_shape: bool = True,
) -> pd.DataFrame:
    """Load the full official file, read-only, without modifying it.

    ``convert_categoricals=False`` is mandatory, not stylistic: pandas' default
    converter raises ``ValueError`` on this official file because the value-label
    set for ``TEMPF`` contains a duplicate label. The flag preserves the original
    numeric codes, which this project needs in order to apply the codebook's
    sentinel rules explicitly rather than after an opaque label conversion.
    """
    dataset_path = config.resolve_dataset_path(path)

    if verify:
        report = verify_dataset(dataset_path)
        if not report["verified"]:
            raise ValueError(
                "Dataset integrity check FAILED.\n"
                f"  file    : {report['path']}\n"
                f"  size    : {report['size_bytes']:,} bytes "
                f"(expected {report['expected_size_bytes']:,})\n"
                f"  sha256  : {report['sha256']}\n"
                f"  expected: {report['expected_sha256']}\n"
                "Refusing to proceed. Either the wrong file was supplied, or the "
                "audit no longer applies to this file."
            )

    with pd.io.stata.StataReader(dataset_path) as reader:
        frame = reader.read(**config.PANDAS_READ_KWARGS)

    if expect_shape and tuple(frame.shape) != tuple(config.EXPECTED_SHAPE):
        raise ValueError(
            f"Unexpected dataset shape {tuple(frame.shape)}, "
            f"expected {tuple(config.EXPECTED_SHAPE)}."
        )
    return frame
