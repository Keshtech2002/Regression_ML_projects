# Urban temperature downscaling — synthetic ML demonstration

**Educational prototype only, not validated on observed urban air temperature or operational weather forecasts.**

This standalone Python example generates an illustrative fine-resolution temperature surface and land-cover/elevation features. It aggregates to a coarse grid, trains `HistGradientBoostingRegressor`, and evaluates predictions against the coarse-grid baseline using a **held-out spatial strip** (not randomly shuffled adjacent pixels).

## Run

```bash
pip install -r requirements.txt
python downscale.py --output outputs
pytest -q
```

Outputs include `outputs/metrics.json` (MAE and RMSE) and compressed example surfaces. The default model performs **worse** than the coarse-grid baseline on the spatial holdout; this is a legitimate negative result, not evidence of successful real-world downscaling.

## Future work
- Replace synthetic data with appropriately licensed ERA5-Land, land-cover, topography, and *observed air temperature* data. Satellite surface temperature is not a substitute for near-surface air temperature.
- Align time stamps, grid cells, coordinate systems, and units with xarray and GIS tools.
- Use spatial and time-block splits, region/hour breakdowns, and comparisons against interpolation and simpler regressors.
- Evaluate extreme temperatures, missing data, domain shift, uncertainty, and calibration.

## Credit
This is independently written starter code, **not a copied research result**. Relevant open-source references for study: [ECMWF tesserugged](https://github.com/ECMWFCode4Earth/tesserugged) and [urban temperature project](https://github.com/rabotnev/DsLab26S_infrared.city-2). Credit their authors if reusing their code.
