from __future__ import annotations

import os
import sys
from pathlib import Path

# Python 3.13 on Windows does not always search the bundled CUDA DLL folder
# when launched through uvicorn. Keep the directory handle alive for import.
_TORCH_DLL_DIR = None
if sys.platform == "win32":
    _torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    if _torch_lib.exists():
        _TORCH_DLL_DIR = os.add_dll_directory(str(_torch_lib))

import torch

from .pixel_scene import SatQueryPixelModel


class ModelRegistry:
    def __init__(self, checkpoint_dir: str | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = Path(checkpoint_dir or os.getenv("SATQUERY_CHECKPOINT_DIR", r"D:\satquery-checkpoints")) / "satquery-pixel-v0.1.pt"
        self.model: SatQueryPixelModel | None = None
        self.version = "satquery-pixel-v0.1-unloaded"
        self.load_error: str | None = None
        self._load()

    def _load(self) -> None:
        if not self.checkpoint.exists():
            self.load_error = f"Checkpoint not found: {self.checkpoint}"
            return
        try:
            payload = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
            model = SatQueryPixelModel()
            model.load_state_dict(payload["model"] if isinstance(payload, dict) and "model" in payload else payload)
            model.eval().to(self.device)
            self.model = model
            self.version = str(payload.get("model_version", "satquery-pixel-v0.1")) if isinstance(payload, dict) else "satquery-pixel-v0.1"
        except Exception as exc:
            self.load_error = str(exc)

    @property
    def ready(self) -> bool:
        return self.model is not None
