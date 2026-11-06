"""Load PyTorch checkpoints from .pt, .pt.gz, .pt.xz, or multipart .pt.xz.

Multipart format:
    checkpoint_best.pt.xz.manifest.json
    checkpoint_best.pt.xz.part000
    checkpoint_best.pt.xz.part001
    ...

Only the project loader needs to use this function; original .pt files work unchanged.
"""

import gzip
import hashlib
import json
import lzma
import shutil
import tempfile
from pathlib import Path

import torch

_BUFFER_SIZE = 4 * 1024 * 1024
_MAX_OUTPUT_CHUNK = 8 * 1024 * 1024
_LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"


def _base_path(path: Path) -> Path:
    """Return base .pt path for any supported single-file requested path."""
    if path.name.endswith(".pt.gz") or path.name.endswith(".pt.xz"):
        return path.with_suffix("")
    if path.suffix != ".pt":
        raise ValueError(f"Expected a .pt, .pt.gz, or .pt.xz path, got: {path}")
    return path


def _load_compressed(source: Path, **torch_kwargs):
    opener = gzip.open if source.suffix == ".gz" else lzma.open
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        with opener(source, "rb") as stream:
            shutil.copyfileobj(stream, tmp, length=_BUFFER_SIZE)
        tmp.seek(0)
        return torch.load(tmp, **torch_kwargs)


def _load_xz_parts(xz_path: Path, **torch_kwargs):
    manifest_path = Path(str(xz_path) + ".manifest.json")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)

    if manifest.get("format") != "qec-xz-parts-v1":
        raise ValueError(f"Invalid multipart checkpoint manifest: {manifest_path}")

    count = manifest.get("parts")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1 or count > 999:
        raise ValueError(f"Invalid number of checkpoint parts in {manifest_path}")

    expected_size = manifest.get("original_size")
    expected_sha = manifest.get("original_sha256")

    if (not isinstance(expected_size, int) or expected_size < 0
            or not isinstance(expected_sha, str) or len(expected_sha) != 64):
        raise ValueError(f"Invalid checksum or size in {manifest_path}")

    parts = [
        Path(f"{xz_path}.part{index:03d}")
        for index in range(count)
    ]

    for part in parts:
        if not part.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint part {part}. Ensure Git LFS is installed "
                "and run `git lfs pull`."
            )

    unexpected = Path(f"{xz_path}.part{count:03d}")
    if unexpected.exists():
        raise ValueError(f"Unexpected extra checkpoint part: {unexpected}")

    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
    digest = hashlib.sha256()
    total_size = 0

    with tempfile.TemporaryFile(mode="w+b") as tmp:
        for part in parts:
            with part.open("rb") as source:
                while compressed := source.read(_BUFFER_SIZE):
                    if compressed.startswith(_LFS_POINTER):
                        raise RuntimeError(
                            f"{part} is a Git LFS pointer, not model data. "
                            "Install Git LFS and run `git lfs pull`."
                        )

                    if decoder.eof:
                        raise ValueError("Unexpected bytes after end of XZ stream")

                    input_data = compressed

                    while True:
                        decoded = decoder.decompress(
                            input_data,
                            max_length=_MAX_OUTPUT_CHUNK
                        )

                        if decoded:
                            tmp.write(decoded)
                            digest.update(decoded)
                            total_size += len(decoded)

                        input_data = b""

                        if decoder.eof:
                            if decoder.unused_data:
                                raise ValueError(
                                    "Unexpected trailing data in XZ stream"
                                )
                            break

                        if decoder.needs_input:
                            break

        if not decoder.eof:
            raise ValueError("Truncated or incomplete multipart XZ checkpoint")

        if total_size != expected_size or digest.hexdigest() != expected_sha:
            raise ValueError(
                f"Multipart checkpoint failed SHA-256 verification: {xz_path}"
            )

        tmp.seek(0)
        return torch.load(tmp, **torch_kwargs)


def load_torch_checkpoint(path, **torch_kwargs):
    """Load a checkpoint transparently, preserving requested-path priority.

    Search order:
        1. Exact requested file
        2. Base .pt file
        3. .pt.gz file
        4. .pt.xz file
        5. Multipart .pt.xz files

    Only the first existing format is loaded.
    """
    requested = Path(path)
    base = _base_path(requested)
    xz_path = Path(str(base) + ".xz")

    candidates = list(dict.fromkeys([
        requested,
        base,
        Path(str(base) + ".gz"),
        xz_path,
    ]))

    for candidate in candidates:
        if candidate.is_file():
            if candidate.suffix == ".pt":
                return torch.load(candidate, **torch_kwargs)

            return _load_compressed(candidate, **torch_kwargs)

    manifest_path = Path(str(xz_path) + ".manifest.json")

    if manifest_path.is_file():
        return _load_xz_parts(xz_path, **torch_kwargs)

    raise FileNotFoundError(
        "Checkpoint not found. Tried: "
        + ", ".join(str(p) for p in candidates)
        + f", or multipart manifest {manifest_path}"
    )
