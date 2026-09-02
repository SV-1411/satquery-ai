from __future__ import annotations

import re


VALID_MODALITIES = {"AUTO", "OPTICAL", "SAR"}


def infer_modality(name: str, bands: int | None = None, hint: str = "AUTO") -> tuple[str, float, str]:
    """Infer sensor type conservatively; never silently labels ambiguous data SAR/optical."""
    requested = hint.strip().upper()
    if requested in {"OPTICAL", "SAR"}:
        return requested, 1.0, "User-selected modality hint."
    lower = name.lower()
    if re.search(r"(^|[_-])(sar|s1|sentinel[-_ ]?1|risat|radar|vv|vh)([_-]|\.|$)", lower):
        return "SAR", 0.98, "Filename contains a SAR/Sentinel-1/RISAT polarization token."
    if re.search(r"(^|[_-])(optical|s2|sentinel[-_ ]?2|multispectral|cartosat)([_-]|\.|$)", lower):
        return "OPTICAL", 0.98, "Filename contains an optical/Sentinel-2/multispectral token."
    if lower.endswith((".png", ".jpg", ".jpeg")):
        return "OPTICAL", 0.80, "Benchmark raster image assumed optical; confirm if this is a rendered SAR image."
    if bands is not None and bands >= 4:
        return "OPTICAL", 0.82, "Four or more bands are consistent with multispectral optical imagery."
    if bands in {1, 2}:
        return "AMBIGUOUS", 0.45, "One or two bands can be SAR or single-band optical; select the modality explicitly."
    return "AMBIGUOUS", 0.30, "Sensor could not be determined from the filename and band structure."

