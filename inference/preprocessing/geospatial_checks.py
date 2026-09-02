from __future__ import annotations

from .raster_loader import RasterData


def pair_check(first: RasterData, second: RasterData) -> tuple[bool, str]:
    if first.width != second.width or first.height != second.height:
        return False, "Pair dimensions differ; reproject/resample to a common grid before analysis."
    if first.is_geospatial and second.is_geospatial and first.crs != second.crs:
        return False, "Pair CRS values differ; reprojection is required before analysis."
    if first.is_geospatial != second.is_geospatial:
        return False, "One pair member has geospatial metadata and the other does not."
    if first.modality == "AMBIGUOUS" or second.modality == "AMBIGUOUS":
        return False, "Could not distinguish one pair member. Select OPTICAL or SAR for each upload."
    return True, "Dimensions and available CRS metadata are compatible."
