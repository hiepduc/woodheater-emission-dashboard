import xarray as xr
import pandas as pd
import numpy as np

# Load daily emissions NetCDF (already spatially distributed)
daily_ds = xr.open_dataset("gridded_emissions_ACT_weekday_July.nc")

# Load diurnal pattern
diurnal_df = pd.read_csv("diurnals/diurnal_weekday.csv")
diurnal_profile = diurnal_df["emission.rate"].values
diurnal_profile = diurnal_profile / diurnal_profile.sum()  # Normalize to sum to 1

# Create hourly time coordinates
time_coords = pd.date_range("2023-07-01", periods=24, freq="H")

# Apply diurnal scaling to each species
hourly_data = {}
for var in daily_ds.data_vars:
    daily_emission = daily_ds[var].isel(time=0)  # get 2D (lat, lon)
    hourly_stack = [daily_emission * rate for rate in diurnal_profile]
    hourly_data[var] = xr.concat(hourly_stack, dim="time")

# Create new dataset with hourly emissions
hourly_ds = xr.Dataset(hourly_data)
hourly_ds = hourly_ds.assign_coords(
    latitude=daily_ds.latitude,
    longitude=daily_ds.longitude,
    time=time_coords
)

# Save to NetCDF
hourly_ds.to_netcdf("gridded_emissions_weekday_July_ACT_hourly.nc")

print("✅ Hourly emission file created: gridded_emissions_weekday_July_ACT_hourly.nc")

