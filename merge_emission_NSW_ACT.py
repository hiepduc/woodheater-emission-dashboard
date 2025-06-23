import xarray as xr

# Load datasets
nsw_ds = xr.open_dataset("gridded_emissions_weekday_July_NSW_hourly_trimmed.nc")
act_ds = xr.open_dataset("gridded_emissions_weekday_July_ACT_hourly.nc")

# Interpolate ACT data to NSW grid
act_interp = act_ds.interp(
    latitude=nsw_ds.latitude,
    longitude=nsw_ds.longitude,
    method="nearest"
)

# Add ACT data to NSW
combined = nsw_ds.copy()
for var in nsw_ds.data_vars:
    combined[var] = nsw_ds[var].fillna(0) + act_interp[var].fillna(0)

# Save to NetCDF
combined.to_netcdf("gridded_emissions_weekday_July_NSW_ACT_combined.nc")

