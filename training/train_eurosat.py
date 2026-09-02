from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import EuroSAT

from inference.models.pixel_scene import SatQueryPixelModel


class EuroSATWeakMask(Dataset):
    """Uses EuroSAT scene labels as weak full-patch masks for compact adaptation.

    This is deliberately a transparent bridge: it provides real EO adaptation
    today, while BigEarthNet pixel/reference-map training remains the stronger
    next stage when S1/S2 patches are available.
    """

    BUILT = {"Highway", "Industrial", "Residential"}
    WATER = {"River", "SeaLake"}

    def __init__(self, root: Path, limit: int, size: int):
        self.base = EuroSAT(root=str(root), download=True, transform=transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()]))
        self.size = size
        indices = list(range(len(self.base)))
        if limit > 0:
            indices = indices[:limit]
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int):
        source_index = self.indices[index]
        image, target = self.base[source_index]
        class_name = self.base.classes[target]
        label = 1 if class_name in self.WATER else 2 if class_name in self.BUILT else 0
        mask = torch.full((self.size, self.size), label, dtype=torch.long)
        return torch.cat([image, image], dim=0), mask, torch.tensor(label, dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt SatQuery on a capped real EO dataset under 10 GB.")
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\satquery-data\eurosat"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path(r"D:\satquery-checkpoints"))
    parser.add_argument("--limit", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-gb", type=float, default=9.0, help="Hard local dataset budget; abort if the downloaded dataset exceeds it.")
    args = parser.parse_args()
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = EuroSATWeakMask(args.data_root, args.limit, args.size)
    bytes_on_disk = sum(item.stat().st_size for item in args.data_root.rglob("*") if item.is_file())
    if bytes_on_disk > args.max_gb * 1024**3:
        raise SystemExit(f"Dataset budget exceeded: {bytes_on_disk / 1024**3:.3f} GiB > {args.max_gb:.3f} GiB")
    validation_size = max(1, min(len(dataset) // 5, 500))
    train_size = len(dataset) - validation_size
    train_set, validation_set = random_split(dataset, [train_size, validation_size], generator=torch.Generator().manual_seed(7))
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = SatQueryPixelModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; seen = 0
        for images, masks, scene_labels in loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            scene_labels = scene_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits, scene_logits = model(images)
                loss = criterion(logits, masks) + 0.75 * criterion(scene_logits, scene_labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total += loss.item() * images.size(0); seen += images.size(0)
        model.eval(); correct = 0; pixels = 0; scene_correct = 0
        with torch.inference_mode():
            for images, masks, scene_labels in val_loader:
                logits, scene_logits = model(images.to(device, non_blocking=True))
                correct += (logits.argmax(1).cpu() == masks).sum().item(); pixels += masks.numel()
                scene_correct += (scene_logits.argmax(1).cpu() == scene_labels).sum().item()
        metrics = {"epoch": epoch, "train_loss": total / max(seen, 1), "validation_pixel_accuracy": correct / max(pixels, 1), "validation_scene_accuracy": scene_correct / max(validation_size, 1), "device": str(device), "train_records": train_size, "validation_records": validation_size}
        history.append(metrics); print(json.dumps(metrics), flush=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target = args.checkpoint_dir / "satquery-pixel-v0.1.pt"
    torch.save({"model_version": "satquery-eurosat-weak-v0.2", "model": model.state_dict(), "history": history, "dataset": "EuroSAT", "limit": args.limit}, target)
    print(json.dumps({"checkpoint": str(target), "dataset_root": str(args.data_root), "dataset_gb": round(bytes_on_disk / 1024**3, 4), "budget_gb": args.max_gb, "device": str(device), "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"}))


if __name__ == "__main__":
    main()
