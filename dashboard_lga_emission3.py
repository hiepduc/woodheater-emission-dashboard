import streamlit as st
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
from shapely.geometry import Point
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- CONFIGURATION ---
NC_FILE = "/mnt/scratch_lustre/duch/whe_dashboard/emission_nc/emission_whe_model_wkdayjul_debug_kgperhourpm25.nc"
SHAPEFILE_PATH = "/home/duch/whe_dashboard/sa3shape/sa3_2016_aust_shape/SA3_2016_AUST.shp"

# --- Load data ---
st.title("Australia Wood Heater Emission Dashboard")

@st.cache_data
def load_dataset():
    return xr.open_dataset(NC_FILE)

@st.cache_data
def load_shapefile():
    return gpd.read_file(SHAPEFILE_PATH)

ds = load_dataset()
gdf_sa3 = load_shapefile()

# --- Sidebar options ---
states = sorted(gdf_sa3["STE_NAME16"].unique())
selected_state = st.sidebar.selectbox("Select State", states)

sa3_in_state = gdf_sa3[gdf_sa3["STE_NAME16"] == selected_state]["SA3_NAME16"].dropna().unique()
selected_region = st.sidebar.selectbox("Select SA3 Region", sorted(sa3_in_state))

species = [v for v in ds.data_vars if ds[v].dims == ("Time", "lat", "lon")]
selected_species = st.sidebar.selectbox("Select Species", species)

selected_hour = st.sidebar.slider("Select Hour", 0, ds["Time"].size - 1, 0)

# New option: plot mode
plot_mode = st.sidebar.radio("Plot Mode", ["Whole NSW", "Selected SA3 Only"])

# --- Prepare emission data ---
emission = ds[selected_species][selected_hour, :, :].values
masked_data = np.ma.masked_invalid(emission)
masked_data_np = np.array(masked_data.filled(np.nan), copy=True)  # writable array
vmax = np.nanpercentile(masked_data_np, 99)

# --- Prepare coordinates ---
lon = ds["lon"].values
lat = ds["lat"].values
lon2d, lat2d = np.meshgrid(lon, lat)

# --- Setup plot ---
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_title(f"{selected_species} Emissions at Hour {selected_hour}")

ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='lightgray')

# Plot emissions
plot = ax.pcolormesh(
    lon2d, lat2d, masked_data_np,
    cmap="YlOrRd", shading="auto",
    transform=ccrs.PlateCarree(),
    vmin=0, vmax=vmax
)
#plt.colorbar(plot, ax=ax, label=f"{selected_species} (mol/s)")
plt.colorbar(plot, ax=ax, label=f"{selected_species} (kg/hour)")

# Plot SA3 outlines
gdf_sa3.boundary.plot(ax=ax, edgecolor="black", linewidth=0.5, transform=ccrs.PlateCarree())

# Highlight selected SA3
selected_polygon = gdf_sa3[gdf_sa3["SA3_NAME16"] == selected_region].geometry.unary_union
gdf_sa3[gdf_sa3["SA3_NAME16"] == selected_region].boundary.plot(
    ax=ax, edgecolor="red", linewidth=2, transform=ccrs.PlateCarree()
)

# Zoom if needed
if plot_mode == "Selected SA3 Only":
    minx, miny, maxx, maxy = selected_polygon.bounds
    ax.set_extent([minx - 0.2, maxx + 0.2, miny - 0.2, maxy + 0.2], crs=ccrs.PlateCarree())
else:
    ax.set_extent([140, 154, -39, -27], crs=ccrs.PlateCarree())  # NSW extent

st.pyplot(fig)

# --- Diurnal profile over selected SA3 ---
# Grid points
flat_points = [Point(x, y) for x, y in zip(lon2d.ravel(), lat2d.ravel())]
mask = np.array([selected_polygon.contains(p) for p in flat_points], dtype=bool).reshape(lat2d.shape)

# Mask all time steps
species_data = ds[selected_species].values  # (time, lat, lon)
masked_region_data = np.where(mask[None, :, :], species_data, np.nan)
mean_timeseries = np.nanmean(masked_region_data, axis=(1, 2))

# --- Time series plot ---
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(ds["Time"].values, mean_timeseries, marker='o')
ax2.set_title(f"Diurnal Emission Pattern of {selected_species} in {selected_region}")
ax2.set_xlabel("Hour of Day")
ax2.set_ylabel(f"{selected_species} (kg/hour)")
st.pyplot(fig2)

