"""Download the UCI Polish Companies Bankruptcy dataset into data/raw/.

Usage (from the project root):
    python -m src.download_data

The script is idempotent: if the files are already present it does nothing.
The zip's SHA-256 checksum is verified so we know we are working with exactly
the same bytes every time (and that the download was not truncated).
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "polish"

UCI_URL = (
    "https://archive.ics.uci.edu/static/public/365/"
    "polish+companies+bankruptcy+data.zip"
)
# Checksum of the zip as downloaded on 2026-11-17. If UCI ever changes the
# file, this check fails loudly instead of silently changing our results.
EXPECTED_SHA256 = "17377929aa0b204bbf957e56462cf827c19fe4e2ce89f27dfbc77f9ea2bb16c9"
EXPECTED_FILES = [f"{n}year.arff" for n in range(1, 6)]


def _sha256(data: bytes) -> str:
    """Return the hex SHA-256 digest of a bytes object."""
    return hashlib.sha256(data).hexdigest()


def download_polish(dest: Path = RAW_DIR, force: bool = False) -> Path:
    """Download and extract the five ARFF files into ``dest``.

    Args:
        dest: Folder to extract into.
        force: Re-download even if all files already exist.

    Returns:
        The folder containing ``1year.arff`` ... ``5year.arff``.
    """
    if not force and all((dest / name).exists() for name in EXPECTED_FILES):
        print(f"Dataset already present in {dest}")
        return dest

    print(f"Downloading {UCI_URL} ...")
    response = requests.get(UCI_URL, timeout=60)
    response.raise_for_status()
    content = response.content

    digest = _sha256(content)
    if digest != EXPECTED_SHA256:
        raise ValueError(
            f"Checksum mismatch: expected {EXPECTED_SHA256}, got {digest}. "
            "The file on UCI may have changed; inspect it before updating."
        )

    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        # Extract only the files we expect (never trust paths inside a zip).
        for name in EXPECTED_FILES:
            (dest / name).write_bytes(archive.read(name))

    print(f"Extracted {len(EXPECTED_FILES)} files to {dest}")
    return dest


if __name__ == "__main__":
    download_polish()
