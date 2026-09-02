# SatQuery AI test data

This is a small public test pack for the current SatQuery AI upload interface. It is intended for UI/API testing only. The current deployed app still returns deterministic demo claims; it does not yet run a trained model on these pixels.

## Quick tests

### Single-image VQA

Upload one of these files:

* `sentinel2_sample.tif` (3-band Sentinel-2 sample GeoTIFF)
* `sentinel2_sample.png` (visual version of the same sample)
* `landsat_style_rgb.byte.tif`
* `goes_satellite_sample.tif`
* `world_rgb_sample.tif`

Use mode `OPTICAL` and ask:

```text
Describe the land-cover and major objects visible in this image.
```

### Optical-SAR pair

The `change_detection_demo` folder comes from a small registered/aligned remote-sensing change-detection sample released under CC BY 4.0:

* `before_optical_A.tif` = Gaofen-2 pre-event optical image.
* `after_sar_B.tif` = Gaofen-3 post-event SAR image.

Upload both, choose `FUSED`, and ask:

```text
Use the optical and SAR images together to identify built-up and water-covered regions.
```

### Bi-temporal change test

Upload:

* `before_optical_A.tif`
* `after_optical_corrected_D.tif`

Choose `OPTICAL` and ask:

```text
What changed between these two dates, and where did the change occur?
```

`change_mask_E.png` is the reference mask for this sample, but the current UI does not compare against it yet. `annotations_A.json` contains the source annotation.

## Sources and attribution

* Sentinel-2 sample: https://github.com/mommermi/geotiff_sample (Copernicus Sentinel data; see its license terms).
* Rasterio geospatial samples: https://github.com/rasterio/rasterio/tree/main/tests/data.
* Change-detection sample: https://huggingface.co/datasets/Mercyiris/remote-sensing-change-detection (CC BY 4.0; cite Tingxuan Yan if used in research).

## Important limitations

* A real change pair must show the same geographic footprint at different times.
* A real optical-SAR pair must be co-registered and cover the same area.
* A JPEG/PNG may not contain CRS or band metadata.
* The current Vercel map is a static evidence backdrop, not a map of the uploaded scene.
* Current claims and confidence values are illustrative until the GPU inference backend is connected.
