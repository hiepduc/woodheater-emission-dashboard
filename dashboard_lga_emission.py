import streamlit as st
import xarray as xr
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import os

# --- CONFIGURATION ---
NC_FILE = "/mnt/scratch_lustre/duch/whe_dashboard/emission_whe_model_wkdayjul_debug_kgperhourpm25.nc"  # Update this path
SHAPEFILE_PATH = "/home/duch/whe_dashboard/sa3shape/sa3_2016_aust_shape/SA3_2016_AUST.shp"

# Load shapefile and inspect columns
gdf_lga = gpd.read_file(SHAPEFILE_PATH)
st.write("Available columns in shapefile:", gdf_lga.columns.tolist())
st.write(gdf_lga[["SA3_CODE16", "SA3_NAME16", "STE_CODE16",  "STE_NAME16"]].head(100))

st.subheader("Available LGA Name Columns")

for col in ["SA3_CODE16", "SA3_NAME16", "STE_CODE16",  "STE_NAME16"]:
    st.markdown(f"**Column:** `{col}`")
    unique_vals = gdf_lga[col].dropna().unique()
    st.text(f"Unique values ({len(unique_vals)}):")
    st.write(sorted(unique_vals.tolist()))
# --- LOAD SHAPEFILE ---
@st.cache_data
def load_lga_shapefile(path):
    return gpd.read_file(path)

gdf_lga = load_lga_shapefile(SHAPEFILE_PATH)
lga_names = gdf_lga["LGA_NAME"].sort_values().unique()

# --- LOAD NETCDF DATA ---
@st.cache_data
def load_netcdf(filepath):
    return xr.open_dataset(filepath)

ds = load_netcdf(NC_FILE)

# --- STREAMLIT SIDEBAR ---
st.sidebar.title("Emission Dashboard")
selected_lga = st.sidebar.selectbox("Select LGA", lga_names)
species_options = list(ds.data_vars)
selected_species = st.sidebar.selectbox("Select Chemical Species", species_options)

# --- GET LGA POLYGON ---
lga_poly = gdf_lga[gdf_lga["LGA_NAME"] == selected_lga].geometry.values[0]

# --- EXTRACT LAT/LON FROM DATASET ---
lats = ds["lat"].values
lons = ds["lon"].values

lon_grid, lat_grid = np.meshgrid(lons, lats)
points = [Point(lon, lat) for lon, lat in zip(lon_grid.flatten(), lat_grid.flatten())]

# --- CREATE MASK OF POINTS INSIDE LGA ---
inside = [lga_poly.contains(pt) for pt in points]
mask = np.array(inside).reshape(lat_grid.shape)

# --- APPLY MASK TO DATA ---
species_data = ds[selected_species][:, :, :]
masked_data = species_data.where(mask)

# --- AGGREGATE BY TIME ---
mean_per_hour = masked_data.mean(dim=["lat", "lon"]).to_series()

# --- PLOTTING ---
st.title(f"Diurnal Pattern for {selected_species} in {selected_lga}")
fig, ax = plt.subplots(figsize=(10, 5))
mean_per_hour.plot(ax=ax, marker='o')
ax.set_xlabel("Hour of Day")
ax.set_ylabel(f"{selected_species} ({species_data.units})")
ax.set_title(f"Hourly Emissions in {selected_lga}")
ax.grid(True)
st.pyplot(fig)

