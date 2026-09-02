from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from inference.models.pixel_scene import SatQueryPixelModel


class SyntheticRemoteSensing(Dataset):
    """Deterministic smoke dataset; replace with BigEarthNet.txt manifests for real adaptation."""

    def __init__(self, count: int, size: int, seed: int):
        self.count, self.size, self.seed = count, size, seed

    def __len__(self):
        return self.count

    def __getitem__(self, index: int):
        rng = np.random.default_rng(self.seed + index)
        size = self.size
        image = rng.normal(0.35, 0.08, (6, size, size)).astype(np.float32)
        mask = np.zeros((size, size), dtype=np.int64)
        for _ in range(1 + index % 3):
            x0, y0 = rng.integers(8, size // 2, 2)
            w, h = rng.integers(size // 8, size // 2, 2)
            yy, xx = np.ogrid[:size, :size]
            ellipse = ((xx - x0) / max(w, 1)) ** 2 + ((yy - y0) / max(h, 1)) ** 2 < 1
            label = 1 if index % 2 == 0 else 2
            mask[ellipse] = label
            if label == 1:
                image[0, ellipse] *= 0.35; image[1, ellipse] *= 0.55; image[2, ellipse] *= 1.35
            else:
                image[0, ellipse] *= 1.35; image[1, ellipse] *= 1.15; image[2, ellipse] *= 0.75
        return torch.from_numpy(np.clip(image, 0, 1)), torch.from_numpy(mask)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SatQuery's compact pixel baseline; replace the smoke dataset with BigEarthNet manifests for adaptation.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--checkpoint-dir", default=r"D:\satquery-checkpoints")
    args = parser.parse_args()
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SyntheticRemoteSensing(args.samples, args.size, 7)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    model = SatQueryPixelModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; correct = 0; pixels = 0
        for images, masks in loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits, _ = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total += loss.item() * images.size(0)
            correct += (logits.argmax(1) == masks).sum().item(); pixels += masks.numel()
        metrics = {"epoch": epoch, "loss": total / len(dataset), "pixel_accuracy": correct / pixels, "device": str(device)}
        history.append(metrics); print(json.dumps(metrics), flush=True)
    out = Path(args.checkpoint_dir); out.mkdir(parents=True, exist_ok=True)
    target = out / "satquery-pixel-v0.1.pt"
    torch.save({"model_version": "satquery-pixel-v0.1-smoke", "model": model.state_dict(), "history": history}, target)
    print(json.dumps({"checkpoint": str(target), "device": str(device), "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"}))


if __name__ == "__main__":
    main()

