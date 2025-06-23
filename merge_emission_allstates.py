import xarray as xr
import glob

# === List your files ===
#file_list = sorted(glob.glob("path/to/emissions/*.nc"))  # adjust path
# === Manually list the files ===
file_list = [
    "/home/duch/whe_dashboard/gridded_emissions_weekday_July_NSW_ACT_combined.nc",
    "/home/duch/whe_dashboard/gridded_emissions_weekday_July_VIC_hourly.nc"
]
#file_list = [
#    "/home/duch/whe_dashboard/gridded_emissions_weekday_July_NSW_ACT_combined.nc",
#    "/home/duch/whe_dashboard/gridded_emissions_weekday_July_VIC_hourly.nc",
#    "/path/to/emissions/emission_mar.nc"
#]

# === Open them as a list of Datasets ===
datasets = [xr.open_dataset(f) for f in file_list]

# === Merge them by spatial concat ===
# First, check that time and variable dimensions are aligned
for ds in datasets:
    assert 'time' in ds.dims
    # Add more checks as needed

# === Use xarray.merge() and combine by coordinates ===
#combined = xr.combine_by_coords(datasets, compat='no_conflicts')
combined = xr.combine_by_coords(datasets, compat='override')

# === Save to a new NetCDF ===
combined.to_netcdf("gridded_emissions_weekday_July_NSW_ACT_VIC_hourly.nc")

