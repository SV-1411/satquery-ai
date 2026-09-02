from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


API = "https://datasets-server.huggingface.co/rows"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a small public BigEarthNet.txt annotation slice; image patches remain a separate download.")
    parser.add_argument("--length", type=int, default=1000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path(r"D:\satquery-data\manifests\bigearthnet_annotations.jsonl"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        downloaded = 0
        while downloaded < args.length:
            page_length = min(100, args.length - downloaded)
            query = urlencode({"dataset": "BIFOLD-BigEarthNetv2-0/BigEarthNet.txt", "config": "default", "split": "all_data", "offset": args.offset + downloaded, "length": page_length})
            with urlopen(f"{API}?{query}", timeout=60) as response:
                payload = json.load(response)
            rows = payload.get("rows", [])
            for row in rows:
                handle.write(json.dumps(row.get("row", {}), ensure_ascii=False) + "\n")
            downloaded += len(rows)
            if len(rows) < page_length:
                break
    print(json.dumps({"output": str(args.output.resolve()), "records": downloaded, "offset": args.offset}))


if __name__ == "__main__":
    main()
