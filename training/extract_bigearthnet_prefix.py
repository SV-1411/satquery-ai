from __future__ import annotations

import argparse
import gzip
import json
import tarfile
from pathlib import Path, PurePosixPath


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely extract a bounded prefix of the paired BigEarthNet S1/S2 tar stream.")
    parser.add_argument("prefix", type=Path)
    parser.add_argument("--output", type=Path, default=Path(r"D:\satquery-data\bigearthnet-subset"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = 0
    bytes_written = 0
    try:
        with args.prefix.open("rb") as source, gzip.GzipFile(fileobj=source) as compressed:
            with tarfile.open(fileobj=compressed, mode="r|") as archive:
                for member in archive:
                    relative = PurePosixPath(member.name)
                    if not member.isfile() or relative.is_absolute() or ".." in relative.parts:
                        continue
                    target = args.output.joinpath(*relative.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    handle = archive.extractfile(member)
                    if handle is None:
                        continue
                    with target.open("wb") as destination:
                        while chunk := handle.read(1024 * 1024):
                            destination.write(chunk)
                            bytes_written += len(chunk)
                    files += 1
    except (EOFError, OSError, tarfile.ReadError):
        # A bounded HTTP range ends in the middle of the archive by design.
        pass
    metadata = args.output / "BigEarthNet" / "metadata.csv"
    s1 = list((args.output / "BigEarthNet").glob("**/S1*.tif")) if (args.output / "BigEarthNet").exists() else []
    s2 = list((args.output / "BigEarthNet").glob("**/S2*.tif")) if (args.output / "BigEarthNet").exists() else []
    print(json.dumps({"output": str(args.output.resolve()), "files": files, "bytes": bytes_written, "metadata": metadata.exists(), "s1_files": len(s1), "s2_files": len(s2)}))


if __name__ == "__main__":
    main()
