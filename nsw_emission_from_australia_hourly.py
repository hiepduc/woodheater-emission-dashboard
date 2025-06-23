import xarray as xr
import geopandas as gpd
import numpy as np
import rasterio.features
from affine import Affine

# --- File paths ---
input_file = "gridded_emissions_weekday_July_hourly.nc"
output_file = "gridded_emissions_weekday_July_NSW_hourly.nc"
nsw_shapefile = "/home/duch/whe_dashboard/lgashape/NSW_LGA_POLYGON_shp/NSW_LGA_POLYGON_shp.shp"

# --- Load emission data ---
ds = xr.open_dataset(input_file)

# --- Load NSW LGA shapefile ---
gdf = gpd.read_file(nsw_shapefile)
gdf = gdf.to_crs("EPSG:4326")  # Ensure same CRS

# --- Create mask ---
transform = Affine.translation(ds.longitude[0], ds.latitude[0]) * Affine.scale(
    float(ds.longitude[1] - ds.longitude[0]), float(ds.latitude[1] - ds.latitude[0])
)

# Combine all LGA polygons into one NSW geometry
nsw_geometry = gdf.unary_union

# Create 2D mask (True = inside NSW)
mask = rasterio.features.geometry_mask(
    [nsw_geometry],
    out_shape=(len(ds.latitude), len(ds.longitude)),
    transform=transform,
    invert=True
)

# Convert to xarray DataArray aligned with ds
mask_da = xr.DataArray(mask, dims=("latitude", "longitude"))

# --- Apply mask to all data variables ---
ds_masked = ds.where(mask_da)

# --- Save to new NetCDF ---
ds_masked.to_netcdf(output_file)
print(f"✅ NSW-only emission file saved to: {output_file}")

