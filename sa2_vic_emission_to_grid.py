import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np
import rioxarray
from shapely.geometry import mapping
from glob import glob
import os

# Load VIC SA2 shapefile
sa2_gdf = gpd.read_file("/mnt/scratch_lustre/duch/whe_dashboard/sa2shape/VIC_ShapeFile_SA2/VIC_Shapefile_SA2.shp")

# Load population grid (Australia-wide)
pop_ds = xr.open_dataset("/mnt/scratch_lustre/duch/whe_dashboard/population/populationraster2016_Australia.nc")
pop_grid = pop_ds['apg16e_1_0_0']
pop_grid = pop_grid.rio.write_crs("EPSG:4326")

# Clip population to VIC bounding box
act_bounds = sa2_gdf.total_bounds  # [minx, miny, maxx, maxy]
pop_grid_act = pop_grid.sel(
    longitude=slice(act_bounds[0], act_bounds[2]),
    latitude=slice(act_bounds[3], act_bounds[1])  # note: decreasing latitude
)

# Prepare output structure
emission_arrays = {}

# Process NSW weekday July CSV files
csv_files = glob("sa2_vic_emission/*_weekday_July_sa2_VIC_kg.csv")

for csv_file in csv_files:
    species = os.path.basename(csv_file).split('_')[0]
    df = pd.read_csv(csv_file)

    # Normalize names
    df["sa2_name_2016"] = df["sa2_name_2016"].str.strip().str.upper()
    sa2_gdf["SA2_NAME16"] = sa2_gdf["SA2_NAME16"].str.strip().str.upper()

    gridded_emission = xr.zeros_like(pop_grid_act)

    for _, row in df.iterrows():
        sa2_name = row["sa2_name_2016"]
        total_emission = row["emission_sa2"]

        if total_emission == 0 or pd.isna(total_emission):
            continue

        poly = sa2_gdf[sa2_gdf['SA2_NAME16'] == sa2_name]
        if poly.empty:
            print(f"SA2 not found: {sa2_name}")
            continue

        masked_pop = pop_grid_act.rio.clip([mapping(geom) for geom in poly.geometry], sa2_gdf.crs, drop=False)

        if masked_pop.isnull().all():
            print(f"All masked pop is NaN for {sa2_name}")
            continue

        total_pop = masked_pop.sum().item()
        if total_pop == 0:
            continue

        weighted_emission = masked_pop * (total_emission / total_pop)
        gridded_emission += weighted_emission.fillna(0)

    emission_arrays[species] = gridded_emission

# Combine into dataset
emission_ds = xr.Dataset({spec: arr.expand_dims(time=[np.datetime64("2023-07-01")]) for spec, arr in emission_arrays.items()})
emission_ds = emission_ds.assign_coords(latitude=pop_grid_act.latitude, longitude=pop_grid_act.longitude)

# Save NSW emissions NetCDF
emission_ds.to_netcdf("gridded_emissions_weekday_July_VIC.nc")

