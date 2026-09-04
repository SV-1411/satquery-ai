# SatQuery AI test data

This is the separate `test` image pack for exercising the upload interface end to end. The files were pulled with the public repository and are deliberately small enough for normal testing. The app analyses the uploaded pixels, draws the uploaded image as the map base, overlays a fresh evidence mask, and asks the local Qwen2.5-VL model for a plain-language summary.

For a single optical image, the UI also shows the original image plus water-consistent, vegetation, built-up-like, and surface-brightness visual layers. Surface temperature and air/atmosphere are deliberately marked **not available** unless a source provides the necessary thermal or atmospheric measurements; they are never fabricated from ordinary RGB/SAR imagery.

## Calibrated thermal and atmosphere reference data

`calibrated_reference` contains real satellite science products for future calibrated-product integration tests. They are deliberately named by product and measurement so they cannot be mistaken for RGB image tests:

* `LANDSAT_C2_L2SP_SURFACE_TEMPERATURE_ST_B10.tif`
  - Official Landsat 9 Collection 2 Level-2 Surface Temperature band over the Phoenix, Arizona region (2025-12-20). This is a one-band `uint16` product in **scaled Kelvin**, not a visible image. Convert valid digital numbers using `Kelvin = DN * 0.00341802 + 149.0`; Celsius is Kelvin minus 273.15.
* `SENTINEL5P_TROPOMI_TROPOSPHERIC_NO2.tif`
  - Openly licensed Digital Earth Africa Sentinel-5P/TROPOMI tropospheric nitrogen-dioxide retrieval tile (2018-04-30). This is an atmospheric-column retrieval, **not** a ground-level AQI or a direct concentration a person breathes.
* `SENTINEL5P_TROPOMI_TROPOSPHERIC_NO2.metadata.json`
  - Acquisition date, projection, provenance, and licence for the NO2 tile.

The current UI correctly keeps the **SURFACE TEMPERATURE** and **AIR / ATMOSPHERE** layers unavailable when these files are uploaded: calibrated numerical processing and metadata validation have not yet been implemented. Do not use them as a normal UI pass/fail test. They are the correct, labelled input files for that next capability; the present RGB/SAR tests above remain the end-to-end runnable suite.

## Quick tests

### Judge-ready mixed coastal scene

Use `sentinel2_sample.png`, select `OPTICAL`, and ask:

```text
What does this image show? Highlight water, vegetation, and built-up evidence, and explain any disagreement in simple words.
```

This sample is intentionally a mixed coastal/urban scene. A healthy result should describe it as mixed coastal water and urban land and show separate water, built-up-like, and vegetation layers. The report should expose any disagreement from the learned scene head and show an **evidence score**, not imply that the score is validated classification accuracy.

### Single-image VQA

Upload one of these files:

* `sentinel2_sample.tif` (3-band Sentinel-2 sample GeoTIFF)
* `sentinel2_sample.png` (visual version of the same sample)
* `landsat_style_rgb.byte.tif`
  - Single-image sample only; do not pair it with `landsat_style_rgb2.byte.tif` for change detection because their map grids differ.
* `goes_satellite_sample.tif`
* `world_rgb_sample.tif`

Use `AUTO DETECT` (or `OPTICAL` if the format has no useful metadata) and ask:

```text
Describe this uploaded observation in simple words. Highlight water-consistent areas and state the main caution.
```

### Optical-SAR pair

The `change_detection_demo` folder comes from a small registered/aligned remote-sensing change-detection sample released under CC BY 4.0:

* `before_optical_A.tif` = Gaofen-2 pre-event optical image.
* `after_sar_B.tif` = Gaofen-3 post-event SAR image.

Upload both, select `OPTICAL` for `before_optical_A.tif` and `SAR` for `after_sar_B.tif`, then ask:

```text
Are the optical and SAR signals in agreement? Explain the measured result simply and state what needs human review.
```

### Bi-temporal change test

Upload:

* `before_optical_A.tif`
* `after_optical_corrected_D.tif`

Choose `OPTICAL` for both, then ask:

```text
What changed between these two uploaded observations? Show the changed pixels on the map and explain the result in simple words.
```

`change_mask_E.png` is the reference mask for visual inspection. `annotations_A.json` contains the source annotation.

### Map-update check

1. Upload `sentinel2_sample.png` and run the single-image query above.
2. Upload `world_rgb_sample.tif` and run the same query.
3. The old map clears as soon as the new file is selected. After each run, the map must show that newly uploaded image with a new mask; it must not retain the previous image or the old static `A/B` boxes.

### Multiple-upload summary

Upload `sentinel2_sample.tif`, `landsat_style_rgb.byte.tif`, `world_rgb_sample.tif`, and `goes_satellite_sample.tif`, then ask:

```text
Summarize the differences between these uploaded observations in simple words. Do not infer a real-world change unless the images are a matched pair.
```

## Sources and attribution

* Sentinel-2 sample: https://github.com/mommermi/geotiff_sample (Copernicus Sentinel data; see its license terms).
* Rasterio geospatial samples: https://github.com/rasterio/rasterio/tree/main/tests/data.
* Change-detection sample: https://huggingface.co/datasets/Mercyiris/remote-sensing-change-detection (CC BY 4.0; cite Tingxuan Yan if used in research).
* Surface-temperature reference: [USGS Landsat Collection 2 Surface Temperature](https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-temperature) (Landsat Level-2 science product; use the stated scale and offset).
* Atmospheric reference: [Digital Earth Africa Sentinel-5P TROPOMI Level-2 NO2](https://registry.opendata.aws/deafrica-sentinel5p/) (CC BY 4.0).

## Important limitations

* A real change pair must show the same geographic footprint at different times.
* A real optical-SAR pair must be co-registered and cover the same area.
* A JPEG/PNG may not contain CRS or band metadata.
* Surface temperature requires a calibrated thermal product plus its scale/offset and valid-pixel rules; it cannot be derived from an RGB display.
* TROPOMI NO2 is a satellite column retrieval at kilometre-scale resolution; it is not a local ground-station AQI reading.
* A change result is an image-difference signal, not proof of the cause of that difference.
* The baseline must be verified against aligned, dated source data before operational use.
