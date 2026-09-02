from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


IMAGE_EXTENSIONS = (".tif", ".tiff", ".jp2", ".png", ".jpg", ".jpeg")


def load_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def index_images(root: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    for path in root.rglob("*"):
        if path.is_dir() and any(item.suffix.lower() in IMAGE_EXTENSIONS for item in path.iterdir()):
            index.setdefault(path.name, str(path.resolve()))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(path.stem, str(path.resolve()))
    return index


def resolve(index: dict[str, str], value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return str(path.resolve())
    return index.get(path.stem)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert BigEarthNet.txt/HF annotation exports into a local JSONL manifest.")
    parser.add_argument("--annotations", type=Path, required=True, help="CSV or JSONL export with input/output/s1_name/s2_name fields.")
    parser.add_argument("--image-root", type=Path, required=True, help="Root containing local Sentinel-1/Sentinel-2 patches.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    image_index = index_images(args.image_root)
    rows = load_rows(args.annotations)
    if args.limit > 0:
        rows = rows[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            s1 = resolve(image_index, row.get("s1_path") or row.get("s1_name"))
            s2 = resolve(image_index, row.get("s2_path") or row.get("s2_name") or row.get("patch_id"))
            record = {
                "id": row.get("id") or row.get("patch_id") or str(written),
                "s1_path": s1,
                "s2_path": s2,
                "input": row.get("input") or row.get("question"),
                "output": row.get("output") or row.get("answer") or row.get("caption"),
                "type": row.get("type") or row.get("category") or "unknown",
                "split": row.get("split") or "train",
                "latitude": row.get("latitude"),
                "longitude": row.get("longitude"),
                "country": row.get("country"),
                "season": row.get("season"),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    print(json.dumps({"output": str(args.output.resolve()), "records": written, "indexed_images": len(image_index), "records_with_s1": sum(1 for row in rows if resolve(image_index, row.get("s1_path") or row.get("s1_name"))), "records_with_s2": sum(1 for row in rows if resolve(image_index, row.get("s2_path") or row.get("s2_name") or row.get("patch_id")))}))


if __name__ == "__main__":
    main()
