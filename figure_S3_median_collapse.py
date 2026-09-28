#!/usr/bin/env python3
"""
Figure S3 (rebuilt; the original plotting script was lost): median probability of complete damage per district
for each of the 30 scenario earthquakes, with PGA contours.

Data:
  - data/aggregated_stats/stats_eqimpact_2024-06-24_physiog.csv (column collapse_mid_median,
    i.e. median over 90 m METEOR cells in each district, middle-case fragility)
  - data/shp/hermes_NPL_new_wgs/hermes_NPL_new_wgs_2.shp (district outlines)
  - data/tif/robinson_ensemble_expanded/Format__UTM45_<event>[_IDW].tif (PGA, g)

Output:
  FIGURES/FigS3_median_collapse_probability.png  (included by main_updt.tex)
  FIGURES/formats/FigS3_median_collapse_probability.{png,pdf}
"""

# %% Setup
import os
import glob
import importlib.util

# Use rasterio's bundled PROJ database (the conda env's proj.db is too old for rasterio)
_rio_spec = importlib.util.find_spec("rasterio")
if _rio_spec:
    _rio_proj = os.path.join(os.path.dirname(_rio_spec.origin), "proj_data")
    if os.path.exists(os.path.join(_rio_proj, "proj.db")):
        os.environ["PROJ_DATA"] = os.environ["PROJ_LIB"] = _rio_proj

import matplotlib.pyplot as plt  # import first to avoid a libstdc++ (CXXABI) clash
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from pyproj import Transformer

base_dir = "/mnt/CEPH_PROJECTS/Proslide/Alex/SajagN"
data_dir = os.path.join(base_dir, "data")
fig_dir = os.path.join(base_dir, "Sajag-Nepal---HyperedgeMH-ensemble", "FIGURES")
output_path = os.path.join(fig_dir, "FigS3_median_collapse_probability.png")

plt.rcParams.update({"text.usetex": False, "font.size": 11})

# %% Load data
stats = pd.read_csv(os.path.join(data_dir, "aggregated_stats", "stats_eqimpact_2024-06-24_physiog.csv"))
districts = gpd.read_file(os.path.join(data_dir, "shp", "hermes_NPL_new_wgs", "hermes_NPL_new_wgs_2.shp"))
districts = districts.to_crs("EPSG:4326")
events = sorted(stats["event"].unique())
pga_dir = os.path.join(data_dir, "tif", "robinson_ensemble_expanded")


def pga_file(event: str) -> str:
    """Return the PGA raster for an event (some files carry an _IDW suffix)."""
    matches = sorted(glob.glob(os.path.join(pga_dir, f"Format__UTM45_{event}*.tif")))
    if not matches:
        raise FileNotFoundError(f"No PGA raster for {event}")
    return matches[0]


to_lonlat = Transformer.from_crs("EPSG:32645", "EPSG:4326", always_xy=True)

# %% Plot
n_cols, n_rows = 3, 10
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 28))
norm = colors.Normalize(vmin=0, vmax=1)
cmap = "viridis"
pga_levels = np.arange(0.1, 1.51, 0.1)
aspect = 1 / np.cos(np.deg2rad(28.2))  # correct lon/lat distortion at Nepal's latitude

for ax, event in zip(axes.flat, events):
    ev = stats.loc[stats["event"] == event, ["DISTRICT", "collapse_mid_median"]]
    gdf = districts.merge(ev, on="DISTRICT", how="left")
    gdf.plot(column="collapse_mid_median", ax=ax, cmap=cmap, norm=norm,
             edgecolor="white", linewidth=0.25, missing_kwds={"color": "lightgrey"})

    with rasterio.open(pga_file(event)) as src:
        pga = src.read(1).astype(float)
        if src.nodata is not None:
            pga[pga == src.nodata] = np.nan
        cols, rows = np.meshgrid(np.arange(src.width) + 0.5, np.arange(src.height) + 0.5)
        xs, ys = rasterio.transform.xy(src.transform, rows.ravel(), cols.ravel())
        lon, lat = to_lonlat.transform(np.asarray(xs), np.asarray(ys))
        lon = lon.reshape(pga.shape)
        lat = lat.reshape(pga.shape)
    cs = ax.contour(lon, lat, pga, levels=pga_levels, colors="red", linewidths=0.5)
    ax.clabel(cs, cs.levels[::2], fontsize=6, fmt="%.1f")

    ax.set_xlim(79.9, 88.5)
    ax.set_ylim(26.2, 30.6)
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax.set_aspect(aspect)
    ax.set_title(event, fontsize=12)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}°E"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}°N"))

for ax in axes.flat[len(events):]:
    ax.set_visible(False)

fig.tight_layout(rect=[0, 0.03, 1, 1])
cax = fig.add_axes([0.25, 0.012, 0.5, 0.008])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
cbar.set_label("Median probability of complete damage (district)", fontsize=12)

fig.savefig(output_path, dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(fig_dir, "formats", "FigS3_median_collapse_probability.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(fig_dir, "formats", "FigS3_median_collapse_probability.pdf"), bbox_inches="tight")
print(f"Figure S3 saved to: {output_path}")
