# Calibrated reference products

This folder is intentionally separate from the upload-ready RGB/SAR test pack.

* `LANDSAT_C2_L2SP_SURFACE_TEMPERATURE_ST_B10.tif` is a real USGS Landsat Collection 2 Level-2 Surface Temperature B10 band. It stores scaled integer values. For valid non-zero pixels: `temperature_kelvin = DN * 0.00341802 + 149.0`; `temperature_celsius = temperature_kelvin - 273.15`.
* `SENTINEL5P_TROPOMI_TROPOSPHERIC_NO2.tif` is a real Sentinel-5P/TROPOMI NO2 column-retrieval tile. Consult its adjacent `metadata.json` before analysis. It is not surface-level air-quality or AQI data.

These files document the correct inputs for future calibrated temperature/atmosphere functionality. The current SatQuery UI will show their pixels but intentionally does not calculate temperature or atmospheric claims from them yet.
