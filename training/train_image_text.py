from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from inference.models.pixel_scene import SatQueryPixelModel
from inference.preprocessing.raster_loader import load_raster_path, model_channels


def text_vector(text: str, width: int = 512) -> torch.Tensor:
    vector = np.zeros(width, dtype=np.float32)
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        vector[hash(token) % width] += 1.0
    norm = np.linalg.norm(vector)
    return torch.from_numpy(vector / max(norm, 1.0))


class PairedTextDataset(Dataset):
    def __init__(self, manifest: Path, limit: int = 0):
        records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.records = records[:limit] if limit else records
        self.records = [r for r in self.records if r.get("s1_path") and (r.get("output") or r.get("input"))]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        first = load_raster_path(record["s1_path"])
        second_path = record.get("s2_path") or record["s1_path"]
        second = load_raster_path(second_path)
        image = torch.from_numpy(np.concatenate([model_channels(first, 128)[:3], model_channels(second, 128)[:3]], axis=0))
        return image, text_vector(str(record.get("output") or record.get("input")))


class ImageTextAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = SatQueryPixelModel().encoder
        self.image_projection = nn.Linear(64, 128)
        self.text_projection = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 128))

    def forward(self, images: torch.Tensor, text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_features = self.image(images).mean(dim=(-2, -1))
        return F.normalize(self.image_projection(image_features), dim=-1), F.normalize(self.text_projection(text), dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt a compact image encoder to BigEarthNet.txt image-text pairs.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--checkpoint-dir", default=r"D:\satquery-checkpoints")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PairedTextDataset(args.manifest, args.limit)
    if not dataset:
        raise SystemExit("No records with local s1_path and output/input text. Download image patches and prepare the manifest first.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    model = ImageTextAdapter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0
        for images, text in loader:
            images, text = images.to(device), text.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                image_features, text_features = model(images, text)
                logits = image_features @ text_features.T / 0.07
                targets = torch.arange(images.size(0), device=device)
                loss = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total += loss.item() * images.size(0)
        print(json.dumps({"epoch": epoch, "loss": total / len(dataset), "records": len(dataset), "device": str(device)}), flush=True)
    out = Path(args.checkpoint_dir); out.mkdir(parents=True, exist_ok=True)
    target = out / "satquery-image-text-v0.1.pt"
    torch.save({"model_version": "satquery-image-text-v0.1", "model": model.state_dict()}, target)
    print(json.dumps({"checkpoint": str(target), "device": str(device)}))


if __name__ == "__main__":
    main()
