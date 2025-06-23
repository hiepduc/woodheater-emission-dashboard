import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np
import rioxarray
from shapely.geometry import mapping
from glob import glob
import os

# Load SA2 shapefile
sa2_gdf = gpd.read_file("/home/duch/whe_dashboard/sa2shape/NSW_Shapefile_SA2/NSW_shapefile_sa2.shp")

# Reproject to match population grid
sa2_gdf = sa2_gdf.to_crs("EPSG:4326")

# Load population grid
pop_ds = xr.open_dataset("/home/duch/whe_dashboard/population/populationraster2016_Australia.nc")
pop_grid = pop_ds['apg16e_1_0_0']
pop_grid = pop_grid.rio.write_crs("EPSG:4326")  # Set CRS if needed

#print("Population CRS:", pop_grid.rio.crs)
#print("SA2 shapefile CRS:", sa2_gdf.crs)

# Prepare output structure
emission_arrays = {}

# Process all weekday July CSV files
csv_files = glob("sa2_nsw_emission/*_weekday_July*_sa2_NSW_KgHr_24Hsum.csv")

print("csv files", csv_files)

for csv_file in csv_files:
    species = os.path.basename(csv_file).split('_')[0]
    df = pd.read_csv(csv_file)
    # Normalize names
    df["sa2_name_2016"] = df["sa2_name_2016"].str.strip().str.upper()
    sa2_gdf["SA2_NAME16"] = sa2_gdf["SA2_NAME16"].str.strip().str.upper()

    csv_sa2_names = set(df["sa2_name_2016"].unique())
    shape_sa2_names = set(sa2_gdf["SA2_NAME16"].unique())

    # Find SA2 names in CSV not found in shapefile
    unmatched = csv_sa2_names - shape_sa2_names
    print("Unmatched SA2 names from CSV:", unmatched)
    print(f"{len(unmatched)} unmatched out of {len(csv_sa2_names)} in total")
#    print("csv_sa2_names", csv_sa2_names)

    # Initialize empty grid for species
    gridded_emission = xr.zeros_like(pop_grid)

    for _, row in df.iterrows():
        sa2_name = row["sa2_name_2016"]
        total_emission = row["emission_kg_sa2"]

        if total_emission == 0 or pd.isna(total_emission):
            continue

        poly = sa2_gdf[sa2_gdf['SA2_NAME16'] == sa2_name]
        if poly.empty:
            print(f"SA2 not found: {sa2_name}")
            continue

        masked_pop = pop_grid.rio.clip([mapping(geom) for geom in poly.geometry], sa2_gdf.crs, drop=False)

        if masked_pop.isnull().all():
            print(f"All masked pop is NaN for {sa2_name}")
            continue

        total_pop = masked_pop.sum().item()
        if total_pop == 0:
            continue

        weighted_emission = masked_pop * (total_emission / total_pop)
        gridded_emission += weighted_emission.fillna(0)


    emission_arrays[species] = gridded_emission

# Combine all into one dataset
emission_ds = xr.Dataset({spec: arr.expand_dims(time=[np.datetime64("2023-07-01")]) for spec, arr in emission_arrays.items()})

# Add coordinates from population dataset
emission_ds = emission_ds.assign_coords(latitude=pop_ds.latitude, longitude=pop_ds.longitude)

# Save to NetCDF
emission_ds.to_netcdf("gridded_emissions_weekday_July.nc")

