import xarray as xr
import numpy as np

file_list = [
    "gridded_emissions_weekday_July_NSW_ACT_combined.nc",
    "gridded_emissions_weekday_July_VIC_hourly.nc",
]

# Load all datasets
ds_list = [xr.open_dataset(f) for f in file_list]

# Build union of latitude and longitude
all_lats = np.unique(np.concatenate([ds.latitude.values for ds in ds_list]))
all_lons = np.unique(np.concatenate([ds.longitude.values for ds in ds_list]))

# Reindex to common grid (fill with NaN where outside)
reindexed = [
    ds.reindex(latitude=all_lats, longitude=all_lons, method=None)
    for ds in ds_list
]

# Collect all unique variable names
all_vars = set()
for ds in reindexed:
    all_vars.update(ds.data_vars)

# Initialize combined dataset
combined = xr.Dataset()

# Loop over each variable and combine
for var in all_vars:
    var_summed = 0
    for ds in reindexed:
        if var in ds:
            var_summed += ds[var].fillna(0)
        else:
            # Add 0 array with same dims as expected
            shape = (len(ds["time"]), len(all_lats), len(all_lons))
            var_summed += xr.DataArray(
                np.zeros(shape, dtype=np.float32),
                dims=("time", "latitude", "longitude"),
                coords={"time": ds["time"], "latitude": all_lats, "longitude": all_lons},
            )
    combined[var] = var_summed

# Add global attrs from first dataset
combined.attrs = ds_list[0].attrs

# Save to NetCDF
combined.to_netcdf("merged_emission_allstates.nc")
print("✅ Saved merged NetCDF: merged_emission_allstates.nc")


