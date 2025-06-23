import streamlit as st
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import numpy as np
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- CONFIGURATION ---
NC_FILE = "/mnt/scratch_lustre/duch/whe_dashboard/emission_nc/emission_whe_model_wkdayjul_debug_kgperhourpm25.nc"
SHAPEFILE_PATH = "/home/duch/whe_dashboard/sa3shape/sa3_2016_aust_shape/SA3_2016_AUST.shp"

# --- Load data ---
st.title("NSW Emission Diurnal Pattern Dashboard")

@st.cache_data
def load_dataset():
    return xr.open_dataset(NC_FILE)

@st.cache_data
def load_shapefile():
    return gpd.read_file(SHAPEFILE_PATH)

ds = load_dataset()
gdf_sa3 = load_shapefile()

# Extract and sort unique state names
state_names = sorted(gdf_sa3["STE_NAME16"].dropna().unique())
selected_state = st.sidebar.selectbox("Select State", state_names)
# Filter regions by selected state
filtered_sa3 = gdf_sa3[gdf_sa3["STE_NAME16"] == selected_state]
sa3_names = sorted(filtered_sa3["SA3_NAME16"].dropna().unique())
selected_region = st.sidebar.selectbox("Select SA3 Region", sa3_names)
# Define selected_polygon from filtered SA3s
selected_polygon = filtered_sa3[filtered_sa3["SA3_NAME16"] == selected_region].geometry.unary_union

plot_region_only = st.sidebar.checkbox("Plot SA3 Region Only", value=False)
# --- Select region and species ---
#sa3_names = sorted(gdf_sa3["SA3_NAME16"].dropna().unique())
#selected_region = st.sidebar.selectbox("Select SA3 Region", sa3_names)

species = [v for v in ds.data_vars if ds[v].dims == ("Time", "lat", "lon")]
selected_species = st.sidebar.selectbox("Select Species", species)

# Create meshgrid for plotting
lon = ds["lon"].values
lat = ds["lat"].values
lon2d, lat2d = np.meshgrid(lon, lat)

selected_hour = st.sidebar.slider("Select Hour", 0, 23, 12)
emission = ds[selected_species][selected_hour, :, :].values  # extract hourly emission

if plot_region_only:
    # Mask emissions outside the selected region
    flat_points = [Point(x, y) for x, y in zip(lon2d.ravel(), lat2d.ravel())]
    mask = np.array([selected_polygon.contains(p) for p in flat_points])
    mask_2d = mask.reshape(lat2d.shape)
    emission = np.where(mask_2d, emission, np.nan)

# --- PLOTTING ---
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_title(f"{selected_species} Emissions at Hour {selected_hour}")
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='lightgray')

import cartopy.io.shapereader as shpreader
from cartopy.feature import ShapelyFeature

if plot_region_only:
    region_shape = ShapelyFeature([selected_polygon], ccrs.PlateCarree(), edgecolor='black', facecolor='none')
    ax.add_feature(region_shape, linewidth=1)

# Normalize and plot
# Assume 'emission' is your 2D array to plot
masked_data = np.ma.masked_invalid(emission)

masked_data_np = masked_data.filled(np.nan)  # Convert MaskedArray to regular array
masked_data_np = np.array(masked_data_np, copy=True)  # Make it writeable
vmax = np.nanpercentile(masked_data_np, 99)

plot = ax.pcolormesh(lon2d, lat2d, masked_data_np, cmap="YlOrRd", shading="auto", transform=ccrs.PlateCarree(), vmin=0, vmax=vmax)
#plt.colorbar(plot, ax=ax, label=f"{selected_species} (mol/s)")
plt.colorbar(plot, ax=ax, label=f"{selected_species} (kg/hour)")

st.pyplot(fig)

# --- Subset shape ---
#selected_polygon = gdf_sa3[gdf_sa3["SA3_NAME16"] == selected_region].geometry.unary_union
selected_polygon = filtered_sa3[filtered_sa3["SA3_NAME16"] == selected_region].geometry.unary_union

# --- Create lat/lon grid ---
lon2d, lat2d = np.meshgrid(ds["lon"].values, ds["lat"].values)
flat_points = [Point(x, y) for x, y in zip(lon2d.ravel(), lat2d.ravel())]

# Mask for selected region
mask = np.array([selected_polygon.contains(p) for p in flat_points], dtype=bool)
mask_2d = mask.reshape(lat2d.shape)

# --- Apply mask and calculate time series ---
species_data = ds[selected_species].values  # shape: (time, lat, lon)
masked_data = np.where(mask_2d[None, :, :], species_data, np.nan)  # shape: (time, lat, lon)

mean_timeseries = np.nanmean(masked_data, axis=(1, 2))  # mean over lat/lon

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ds["Time"].values, mean_timeseries, marker='o')
ax.set_title(f"Diurnal Emission Pattern of {selected_species} in {selected_region}")
ax.set_xlabel("Hour of Day")
ax.set_ylabel(f"{selected_species} (kg/hour)")
ax.grid(True)

st.pyplot(fig)

