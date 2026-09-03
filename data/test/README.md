# SatQuery AI test data

This is the separate `test` image pack for exercising the upload interface end to end. The files were pulled with the public repository and are deliberately small enough for normal testing. The app analyses the uploaded pixels, draws the uploaded image as the map base, overlays a fresh evidence mask, and asks the local Qwen2.5-VL model for a plain-language summary.

For a single optical image, the UI also shows the original image plus water-consistent, vegetation, built-up-like, and surface-brightness visual layers. Surface temperature and air/atmosphere are deliberately marked **not available** unless a source provides the necessary thermal or atmospheric measurements; they are never fabricated from ordinary RGB/SAR imagery.

## Quick tests

### Single-image VQA

Upload one of these files:

* `sentinel2_sample.tif` (3-band Sentinel-2 sample GeoTIFF)
* `sentinel2_sample.png` (visual version of the same sample)
* `landsat_style_rgb.byte.tif`
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

## Important limitations

* A real change pair must show the same geographic footprint at different times.
* A real optical-SAR pair must be co-registered and cover the same area.
* A JPEG/PNG may not contain CRS or band metadata.
* A change result is an image-difference signal, not proof of the cause of that difference.
* The baseline must be verified against aligned, dated source data before operational use.
