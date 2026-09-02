from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .modality import infer_modality


@dataclass
class RasterData:
    name: str
    data: np.ndarray  # C x H x W, float32 in [0, 1]
    width: int
    height: int
    bands: int
    dtype: str
    crs: str | None
    transform: Any | None
    is_geospatial: bool
    modality: str
    modality_confidence: float
    modality_note: str


def _robust_normalise(data: np.ndarray) -> np.ndarray:
    data = np.nan_to_num(data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    output = np.empty_like(data, dtype=np.float32)
    for channel in range(data.shape[0]):
        band = data[channel]
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            output[channel] = 0.0
            continue
        low, high = np.percentile(finite, [2, 98])
        if high <= low:
            high = low + 1.0
        output[channel] = np.clip((band - low) / (high - low), 0.0, 1.0)
    return output


def _read_multiframe_tiff(payload: bytes) -> np.ndarray | None:
    """Read BigEarthNet's multi-page TIFF layout as C x H x W."""
    try:
        import tifffile
        raw = np.asarray(tifffile.imread(BytesIO(payload)))
    except Exception:
        return None
    if raw.ndim == 2:
        return raw[None, ...]
    if raw.ndim != 3:
        return None
    if raw.shape[-1] <= 32:
        return np.moveaxis(raw, -1, 0)
    if raw.shape[0] <= 32:
        return raw
    return None


def load_raster(name: str, payload: bytes, modality_hint: str = "AUTO") -> RasterData:
    """Load GeoTIFF/TIFF through Rasterio and benchmark images through Pillow."""
    suffix = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    if suffix in {"tif", "tiff"}:
        try:
            import rasterio
            from rasterio.io import MemoryFile

            with MemoryFile(payload) as memory_file:
                with memory_file.open() as dataset:
                    raw = dataset.read()
                    width, height, bands = dataset.width, dataset.height, dataset.count
                    if dataset.count == 1 and min(dataset.width, dataset.height) <= 16:
                        alternate = _read_multiframe_tiff(payload)
                        if alternate is not None and alternate.shape[1] >= 32 and alternate.shape[2] >= 32:
                            raw = alternate
                            height, width, bands = raw.shape[1], raw.shape[2], raw.shape[0]
                    modality, confidence, note = infer_modality(name, bands, modality_hint)
                    if modality == "SAR":
                        raw = 10.0 * np.log10(np.maximum(raw.astype(np.float32), 1e-6))
                    data = _robust_normalise(raw)
                    return RasterData(
                        name=name,
                        data=data,
                        width=width,
                        height=height,
                        bands=bands,
                        dtype=str(raw.dtype),
                        crs=str(dataset.crs) if dataset.crs else None,
                        transform=dataset.transform,
                        is_geospatial=dataset.crs is not None,
                        modality=modality,
                        modality_confidence=confidence,
                        modality_note=note,
                    )
        except Exception as exc:
            raise ValueError(f"Unable to read TIFF {name}: {exc}") from exc

    try:
        with Image.open(BytesIO(payload)) as image:
            image.load()
            modality, confidence, note = infer_modality(name, 3, modality_hint)
            raw = np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
            return RasterData(
                name=name,
                data=raw,
                width=image.width,
                height=image.height,
                bands=raw.shape[0],
                dtype="uint8",
                crs=None,
                transform=None,
                is_geospatial=False,
                modality=modality,
                modality_confidence=confidence,
                modality_note=note,
            )
    except Exception as exc:
        raise ValueError(f"Unable to read image {name}: {exc}") from exc


def load_raster_path(path: str | Path, modality_hint: str = "AUTO") -> RasterData:
    """Load a file or a BigEarthNet patch directory containing one GeoTIFF per band."""
    target = Path(path)
    if not target.is_dir():
        return load_raster(target.name, target.read_bytes(), modality_hint)
    band_files = sorted(item for item in target.iterdir() if item.suffix.lower() in {".tif", ".tiff", ".jp2"} and "label" not in item.name.lower())
    if not band_files:
        raise ValueError(f"No raster bands found in patch directory {target}")
    try:
        import rasterio
        arrays = []
        profile = None
        with rasterio.open(band_files[0]) as first:
            profile = first.profile
            height, width = first.height, first.width
        for band_file in band_files:
            with rasterio.open(band_file) as dataset:
                if dataset.width != width or dataset.height != height:
                    raise ValueError(f"Band dimensions differ inside {target}")
                arrays.append(dataset.read(1))
        raw = np.stack(arrays, axis=0)
        modality, confidence, note = infer_modality(target.name, raw.shape[0], modality_hint)
        if modality == "SAR":
            raw = 10.0 * np.log10(np.maximum(raw.astype(np.float32), 1e-6))
        return RasterData(name=target.name, data=_robust_normalise(raw), width=width, height=height, bands=raw.shape[0], dtype=str(raw.dtype), crs=str(profile.get("crs")) if profile and profile.get("crs") else None, transform=profile.get("transform") if profile else None, is_geospatial=True, modality=modality, modality_confidence=confidence, modality_note=note)
    except Exception as exc:
        raise ValueError(f"Unable to read patch directory {target}: {exc}") from exc


def model_channels(data: np.ndarray, size: int = 256) -> np.ndarray:
    """Resize/pad a raster into the six-channel contract used by the compact model."""
    import torch
    import torch.nn.functional as F

    tensor = torch.from_numpy(data).unsqueeze(0).float()
    tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    tensor = tensor.squeeze(0)
    if tensor.shape[0] >= 6:
        return tensor[:6].numpy()
    repeats = (6 + tensor.shape[0] - 1) // tensor.shape[0]
    return tensor.repeat(repeats, 1, 1)[:6].numpy()
