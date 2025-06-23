import xarray as xr
import geopandas as gpd
import numpy as np
import rasterio.features
from affine import Affine

# --- File paths ---
input_file = "gridded_emissions_weekday_July.nc"
output_file = "gridded_emissions_weekday_July_NSW_trimmed.nc"
nsw_shapefile = "/home/duch/whe_dashboard/lgashape/NSW_LGA_POLYGON_shp/NSW_LGA_POLYGON_shp.shp"

# --- Load emission data ---
ds = xr.open_dataset(input_file)

# --- Load NSW shapefile and reproject ---
gdf = gpd.read_file(nsw_shapefile).to_crs("EPSG:4326")

# --- Get bounding box of NSW (minx, miny, maxx, maxy) ---
minx, miny, maxx, maxy = gdf.total_bounds

# --- Subset dataset spatially to NSW bounding box ---
ds_nsw = ds.sel(
    longitude=slice(minx, maxx),
    latitude=slice(maxy, miny)  # latitude decreasing
)

# --- Create mask for NSW geometry on the trimmed domain ---
transform = Affine.translation(ds_nsw.longitude[0], ds_nsw.latitude[0]) * Affine.scale(
    float(ds_nsw.longitude[1] - ds_nsw.longitude[0]),
    float(ds_nsw.latitude[1] - ds_nsw.latitude[0])
)

nsw_geom = gdf.unary_union
mask = rasterio.features.geometry_mask(
    [nsw_geom],
    out_shape=(len(ds_nsw.latitude), len(ds_nsw.longitude)),
    transform=transform,
    invert=True
)

mask_da = xr.DataArray(mask, dims=("latitude", "longitude"))

# --- Apply the mask ---
ds_nsw_masked = ds_nsw.where(mask_da)

# --- Save trimmed NSW domain ---
ds_nsw_masked.to_netcdf(output_file)
print(f"✅ NSW-trimmed emission file saved: {output_file}")

