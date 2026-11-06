"""Split an existing losslessly compressed PyTorch .pt.xz into Git LFS parts.

Usage:
    python QEC_model_zoo/scripts/pack_checkpoint_xz.py PATH/TO/checkpoint_best.pt

Requires both checkpoint_best.pt and checkpoint_best.pt.xz to exist.
"""

import argparse
import hashlib
import json
from pathlib import Path

CHUNK_SIZE = 1800 * 1024 * 1024
COPY_SIZE = 8 * 1024 * 1024


def checksum(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        while data := stream.read(COPY_SIZE):
            digest.update(data)

    return digest.hexdigest()


def pack(original: Path) -> None:
    if original.suffix != ".pt" or not original.is_file():
        raise FileNotFoundError(
            f"Original .pt checkpoint not found: {original}"
        )

    archive = Path(str(original) + ".xz")

    if not archive.is_file():
        raise FileNotFoundError(
            f"Existing XZ-compressed checkpoint not found: {archive}"
        )

    # Do not overwrite previous multipart outputs.
    prior = list(
        archive.parent.glob(
            archive.name + ".part[0-9][0-9][0-9]"
        )
    )

    manifest_path = Path(str(archive) + ".manifest.json")

    if prior or manifest_path.exists():
        raise FileExistsError(
            f"Multipart outputs already exist for {archive}. "
            "Remove or move old .partNNN files and manifest before repacking."
        )

    print("Calculating original SHA-256...")

    original_hash = checksum(original)
    original_size = original.stat().st_size

    print(f"Original: {original_size / (1024**3):.3f} GiB")
    print(f"Compressed: {archive.stat().st_size / (1024**3):.3f} GiB")

    count = 0

    with archive.open("rb") as source:
        while True:
            part_path = Path(f"{archive}.part{count:03d}")
            bytes_written = 0

            with part_path.open("xb") as target:
                while bytes_written < CHUNK_SIZE:
                    data = source.read(
                        min(COPY_SIZE, CHUNK_SIZE - bytes_written)
                    )

                    if not data:
                        break

                    target.write(data)
                    bytes_written += len(data)

            if bytes_written == 0:
                part_path.unlink()
                break

            print(
                f"Created {part_path.name}: "
                f"{bytes_written / (1024**3):.3f} GiB"
            )

            count += 1

            if count >= 999:
                raise ValueError(
                    "Too many chunks for .partNNN naming"
                )

            if bytes_written < CHUNK_SIZE:
                break

    manifest = {
        "format": "qec-xz-parts-v1",
        "parts": count,
        "original_size": original_size,
        "original_sha256": original_hash,
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8"
    )

    print(f"Wrote {manifest_path} with original checkpoint hash")
    print("The original .pt and .pt.xz are untouched.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Original .pt checkpoint path"
    )

    args = parser.parse_args()
    pack(args.checkpoint)
