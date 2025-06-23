import streamlit as st
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
from shapely.geometry import Point
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from io import BytesIO, StringIO

# --- CONFIGURATION ---
#NC_FILE = "/mnt/scratch_lustre/duch/whe_dashboard/emission_nc/emission_whe_model_wkdayjul_debug_kgperhourpm25.nc"
NC_FILE = "/mnt/scratch_lustre/duch/whe_dashboard/emission_nc/gridded_emissions_weekday_July_NSW_hourly_trimmed.nc"
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

species = [v for v in ds.data_vars if ds[v].dims == ("time", "latitude", "longitude")]
selected_species = st.sidebar.selectbox("Select Species", species)

selected_hour = st.sidebar.slider("Select Hour", 0, ds["time"].size - 1, 0)
plot_mode = st.sidebar.radio("Plot Mode", ["Whole State", "Selected SA3 Only"])

# --- Coordinates and emission ---
lon = ds["longitude"].values
lat = ds["latitude"].values
lon2d, lat2d = np.meshgrid(lon, lat)

emission = ds[selected_species][selected_hour, :, :].values
masked_data = np.ma.masked_invalid(emission)
masked_data_np = np.array(masked_data.filled(np.nan), copy=True)
vmax = np.nanpercentile(masked_data_np, 99)

# --- Plot emission map ---
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_title(f"{selected_species} Emissions at Hour {selected_hour}")
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='lightgray')

# Emission map
plot = ax.pcolormesh(
    lon2d, lat2d, masked_data_np,
    cmap="YlOrRd", shading="auto",
    transform=ccrs.PlateCarree(),
    vmin=0, vmax=vmax
)
#plt.colorbar(plot, ax=ax, label=f"{selected_species} (mol/s)")
plt.colorbar(plot, ax=ax, label=f"{selected_species} (kg/hour)")

# Add SA3 boundary
gdf_sa3.boundary.plot(ax=ax, edgecolor="black", linewidth=0.5, transform=ccrs.PlateCarree())
selected_polygon = gdf_sa3[gdf_sa3["SA3_NAME16"] == selected_region].geometry.unary_union
gdf_sa3[gdf_sa3["SA3_NAME16"] == selected_region].boundary.plot(
    ax=ax, edgecolor="red", linewidth=2, transform=ccrs.PlateCarree()
)

if plot_mode == "Selected SA3 Only":
    minx, miny, maxx, maxy = selected_polygon.bounds
    ax.set_extent([minx - 0.2, maxx + 0.2, miny - 0.2, maxy + 0.2], crs=ccrs.PlateCarree())
else:
    ax.set_extent([140, 154, -39, -27], crs=ccrs.PlateCarree())  # NSW

st.pyplot(fig)

# --- Download Map as PNG ---
img_buf = BytesIO()
fig.savefig(img_buf, format="png")
st.download_button(
    label="Download Emission Map (PNG)",
    data=img_buf.getvalue(),
    file_name=f"{selected_species}_emission_map_hour{selected_hour}.png",
    mime="image/png"
)

# --- Region masking ---
flat_points = [Point(x, y) for x, y in zip(lon2d.ravel(), lat2d.ravel())]
mask = np.array([selected_polygon.contains(p) for p in flat_points], dtype=bool).reshape(lat2d.shape)
species_data = ds[selected_species].values
masked_region_data = np.where(mask[None, :, :], species_data, np.nan)
mean_timeseries = np.nanmean(masked_region_data, axis=(1, 2))

# --- Emission Summary ---
st.subheader("Emission Summary for Selected Region")
total_emission = np.nansum(masked_region_data)
mean_emission = np.nanmean(masked_region_data)
max_emission = np.nanmax(masked_region_data)

st.write(f"**Total Emission (All Hours)**: {total_emission:.2f} kg/hour")
st.write(f"**Mean Emission per Hour**: {mean_emission:.2f} kg/hour")
st.write(f"**Max Emission**: {max_emission:.2f} kg/hour")

# --- Diurnal Plot ---
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(ds["time"].values, mean_timeseries, marker='o')
ax2.set_title(f"Diurnal Emission Pattern of {selected_species} in {selected_region}")
ax2.set_xlabel("Hour of Day")
ax2.set_ylabel(f"{selected_species} (kg/hour)")
st.pyplot(fig2)

# --- Download Time Series as CSV ---
time_vals = ds["time"].values
df_timeseries = pd.DataFrame({
    "Hour": range(len(time_vals)),
    f"{selected_species}": mean_timeseries
})
csv_data = df_timeseries.to_csv(index=False)
st.download_button(
    label="Download Time Series (CSV)",
    data=csv_data,
    file_name=f"{selected_species}_diurnal_{selected_region.replace(' ', '_')}.csv",
    mime="text/csv"
)

# --- Export Per-hour Spatial Data for Selected Region ---
# Save as CSV (flattened for export)
if st.checkbox("Enable SA3 hourly spatial export"):
    hour_list = []
    lat_list = []
    lon_list = []
    value_list = []

    for t in range(species_data.shape[0]):
        values = np.where(mask, species_data[t], np.nan)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if not np.isnan(values[i, j]):
                    hour_list.append(t)
                    lat_list.append(lat[i])
                    lon_list.append(lon[j])
                    value_list.append(values[i, j])

    export_df = pd.DataFrame({
        "Hour": hour_list,
        "Lat": lat_list,
        "Lon": lon_list,
        f"{selected_species}": value_list
    })

    export_csv = export_df.to_csv(index=False)
    st.download_button(
        label="Download SA3 Hourly Spatial Emission (CSV)",
        data=export_csv,
        file_name=f"{selected_species}_sa3_hourly_{selected_region.replace(' ', '_')}.csv",
        mime="text/csv"
    )

