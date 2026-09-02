from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a SatQuery JSONL manifest without loading the full dataset into memory.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--sample", type=int, default=100)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    sample = records[: args.sample]
    missing = []
    for record in sample:
        for key in ("s1_path", "s2_path"):
            value = record.get(key)
            if value and not Path(value).exists():
                missing.append({"id": record.get("id"), "field": key, "path": value})
    splits: dict[str, int] = {}
    types: dict[str, int] = {}
    for record in records:
        splits[record.get("split", "unknown")] = splits.get(record.get("split", "unknown"), 0) + 1
        types[record.get("type", "unknown")] = types.get(record.get("type", "unknown"), 0) + 1
    with_s1 = sum(1 for record in records if record.get("s1_path"))
    with_s2 = sum(1 for record in records if record.get("s2_path"))
    result = {"records": len(records), "checked": len(sample), "records_with_s1": with_s1, "records_with_s2": with_s2, "missing_sample_paths": missing[:20], "missing_count": len(missing), "splits": splits, "types": types, "status": "passed" if not missing and with_s1 > 0 and with_s2 > 0 else "failed"}
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if not missing else 2)


if __name__ == "__main__":
    main()
