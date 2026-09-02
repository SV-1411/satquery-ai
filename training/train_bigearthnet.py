from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from inference.models.pixel_scene import SatQueryPixelModel
from inference.preprocessing.raster_loader import _robust_normalise, model_channels


def broad_label(labels: list[str]) -> int:
    text = " ".join(labels).lower()
    if any(token in text for token in ("water", "wetland", "marsh", "peatbog", "salt marsh", "intertidal", "lagoon", "estuaries", "sea and ocean")):
        return 1
    if any(token in text for token in ("urban", "industrial", "road", "port", "airport", "construction", "mineral extraction", "dump", "sport")):
        return 2
    return 0


class PairedBigEarthNet(Dataset):
    def __init__(self, root: Path, limit: int):
        self.root = root
        self.rows = []
        with (root / "metadata.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if all((root / row[key]).exists() for key in ("optical_path", "radar_path", "label_path")):
                    labels = json.loads((root / row["label_path"]).read_text(encoding="utf-8"))["labels"]
                    self.rows.append((row["optical_path"], row["radar_path"], broad_label(labels)))
                    if limit > 0 and len(self.rows) >= limit:
                        break

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        optical_path, radar_path, label = self.rows[index]
        optical = np.asarray(tifffile.imread(self.root / optical_path))
        radar = np.asarray(tifffile.imread(self.root / radar_path))
        optical = np.moveaxis(optical, -1, 0) if optical.ndim == 3 else optical[None, ...]
        radar = np.moveaxis(radar, -1, 0) if radar.ndim == 3 else radar[None, ...]
        optical = _robust_normalise(optical)
        radar = _robust_normalise(10.0 * np.log10(np.maximum(radar.astype(np.float32), 1e-6)))
        image = torch.from_numpy(model_channels(np.concatenate([optical[:4], radar[:2]], axis=0), size=128)).float()
        mask = torch.full((128, 128), label, dtype=torch.long)
        return image, mask, torch.tensor(label, dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune SatQuery on a bounded paired BigEarthNet S1/S2 subset.")
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\satquery-data\bigearthnet-subset-1gb\BigEarthNet"))
    parser.add_argument("--checkpoint", type=Path, default=Path(r"D:\satquery-checkpoints\satquery-pixel-v0.1.pt"))
    parser.add_argument("--output", type=Path, default=Path(r"D:\satquery-checkpoints\satquery-pixel-bigearthnet-s1s2.pt"))
    parser.add_argument("--limit", type=int, default=76)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PairedBigEarthNet(args.data_root, args.limit)
    if len(dataset) < 10:
        raise SystemExit(f"Need at least 10 complete paired samples; found {len(dataset)}")
    validation_size = max(2, len(dataset) // 5)
    train_set, validation_set = random_split(dataset, [len(dataset) - validation_size, validation_size], generator=torch.Generator().manual_seed(7))
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = SatQueryPixelModel().to(device)
    if args.checkpoint.exists():
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; seen = 0
        for images, masks, labels in loader:
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                pixel_logits, scene_logits = model(images)
                loss = criterion(pixel_logits, masks) + 0.75 * criterion(scene_logits, labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total += loss.item() * images.size(0); seen += images.size(0)
        model.eval(); scene_correct = 0; total_val = 0
        with torch.inference_mode():
            for images, _, labels in val_loader:
                _, scene_logits = model(images.to(device))
                scene_correct += (scene_logits.argmax(1).cpu() == labels).sum().item(); total_val += labels.numel()
        metrics = {"epoch": epoch, "train_loss": total / max(seen, 1), "validation_scene_accuracy": scene_correct / max(total_val, 1), "device": str(device), "paired_records": len(dataset), "train_records": len(train_set), "validation_records": len(validation_set)}
        history.append(metrics); print(json.dumps(metrics), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_version": "satquery-bigearthnet-s1s2-v0.3", "model": model.state_dict(), "history": history, "dataset": "BigEarthNet paired Sentinel-1/Sentinel-2 bounded prefix", "limit": len(dataset)}, args.output)
    print(json.dumps({"checkpoint": str(args.output), "device": str(device), "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"}))


if __name__ == "__main__":
    main()
