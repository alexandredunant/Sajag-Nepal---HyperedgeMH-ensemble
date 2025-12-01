"""
Figure 3: District Impacts Ratio
Creates a combined two-panel figure showing:
  Panel A: Stacked bar chart of impact counts by district with landslide/shaking ratio overlay
  Panel B: Choropleth map of landslide/shaking damage ratio by district

This script generates:
  - Fig3_district_impacts_ratio.png
  - Fig3_district_impacts_ratio.pdf

Dependencies:
  - data_loader: Module for loading impact and administrative data
  - utils: Utility functions for styling and visualization
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
from utils import apply_science_style


warnings.filterwarnings('ignore')
apply_science_style()


def load_data(data_dir):
    """
    Load impact and administrative boundary data.

    Parameters
    ----------
    data_dir : str
        Path to the data directory containing aggregated_stats and shapefiles

    Returns
    -------
    dict
        Dictionary containing impact_data and nepal_admin GeoDataFrame
    """
    stats_dir = os.path.join(data_dir, "aggregated_stats")

    # Load impact statistics
    stats_buildings_lsimpact = pd.read_csv(
        os.path.join(stats_dir, "stats_buildings_lsimpact_2024-06-24_physiog.csv")
    )
    stats_eqimpact = pd.read_csv(
        os.path.join(stats_dir, "stats_eqimpact_2024-06-24_physiog.csv")
    )
    stats_roads_lsimpact = pd.read_csv(
        os.path.join(stats_dir, "stats_roads_lsimpact_2024-06-24_physiog.csv")
    )

    # Combine all impact types into one dataframe
    impact_data = pd.DataFrame({
        'District': stats_eqimpact['DISTRICT'],
        'Event': stats_eqimpact['event'],
        'Earthquake': stats_eqimpact['collapse_mid_sum'],
        'BuildingLandslide': stats_buildings_lsimpact['impact_sum'],
        'RoadLandslide': stats_roads_lsimpact['impact_sum']
    })

    # Load administrative boundaries
    nepal_admin = gpd.read_file(
        os.path.join(data_dir, "shp", "hermes_NPL_new_wgs", "hermes_NPL_new_wgs_2.shp")
    )

    # Transform to UTM 45N for better visualization
    nepal_admin = nepal_admin.to_crs("EPSG:32645")

    return {
        'impact_data': impact_data,
        'nepal_admin': nepal_admin,
        'stats_eqimpact': stats_eqimpact
    }


def create_figure_3(impact_data, nepal_admin, output_dir):
    """
    Create Figure 3: District impacts ratio with bar chart and choropleth map.

    Parameters
    ----------
    impact_data : pd.DataFrame
        DataFrame with columns: District, Event, Earthquake, BuildingLandslide, RoadLandslide
    nepal_admin : gpd.GeoDataFrame
        GeoDataFrame of Nepal administrative boundaries
    output_dir : str
        Directory to save output figures
    """
    # Define styling
    colors_dict = {
        'Earthquake': '#9e9e9e',
        'BuildingLandslide': '#7474ee',
        'RoadLandslide': '#3700ff'
    }

    labels_dict = {
        'Earthquake': 'Buildings damaged by shaking',
        'BuildingLandslide': 'Buildings damaged by landslides',
        'RoadLandslide': 'Roads damaged by landslides'
    }

    print("Creating Figure 3: District-wise impact bar plots with ratio map...")

    # Calculate maximum impacts by district
    impact_max_by_district = impact_data.groupby('District').agg(
        Earthquake=('Earthquake', 'max'),
        BuildingLandslide=('BuildingLandslide', 'max'),
        RoadLandslide=('RoadLandslide', 'max')
    ).reset_index()

    # Calculate landslide to shaking ratio
    impact_max_by_district['LandslideRatio'] = (
        (impact_max_by_district['BuildingLandslide'] + impact_max_by_district['RoadLandslide']) /
        impact_max_by_district['Earthquake'] * 100
    )

    # Sort by earthquake impacts for visualization
    impact_max_by_district = impact_max_by_district.sort_values('Earthquake', ascending=False)

    # ==========================================================================
    # Create combined figure with both panels
    # ==========================================================================
    fig3 = plt.figure(figsize=(14, 16))
    gs = fig3.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.25)

    # ==========================================================================
    # PANEL A: Bar chart
    # ==========================================================================
    ax1 = fig3.add_subplot(gs[0, :])

    # Stacked bar plot
    districts = impact_max_by_district['District'].tolist()
    bottoms = np.zeros(len(districts))

    for hazard in ['Earthquake', 'BuildingLandslide', 'RoadLandslide']:
        impact_values = impact_max_by_district.set_index('District').loc[districts][hazard].values
        ax1.bar(np.arange(len(districts)), impact_values, bottom=bottoms,
                label=labels_dict[hazard], color=colors_dict[hazard],
                edgecolor='black', linewidth=0.2)
        bottoms += impact_values

    # Add ratio line on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(districts)), impact_max_by_district['LandslideRatio'],
             color='red', linewidth=1.5, marker='o', markersize=3)

    ax1.set_xticks(np.arange(len(districts)))
    ax1.set_xticklabels(districts, rotation=90, fontsize=12)
    ax1.set_xlabel('District', fontsize=14)
    ax1.set_ylabel('Absolute Count', fontsize=14)
    ax2.set_ylabel('Landslide/Shaking Damage Ratio (%)', fontsize=14, color='red')

    # Add "A" label
    ax1.text(-0.05, 1.05, 'A', transform=ax1.transAxes, fontsize=20, va='top')

    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
    ax1.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

    # Add some margin at the top to prevent clipping
    ax1.margins(y=0.1)

    # Combined legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles1.append(Line2D([0], [0], color='red', lw=1.5, marker='o', markersize=3))
    labels1.append('Landslide/Shaking Damage Ratio')
    ax1.legend(handles1, labels1, loc='upper center', frameon=True, fontsize=11)

    # ==========================================================================
    # PANEL B: Map with geopandas and contextily
    # ==========================================================================
    ax3 = fig3.add_subplot(gs[1, :])

    # Merge ratio data with admin boundaries
    nepal_admin_with_ratios = nepal_admin.merge(
        impact_max_by_district, left_on='DISTRICT', right_on='District', how='left'
    )

    # Add basemap with contextily (need to use Web Mercator for contextily)
    try:
        import contextily as ctx
        # Convert to Web Mercator for basemap
        nepal_admin_ratios_mercator = nepal_admin_with_ratios.to_crs("EPSG:3857")

        # Plot in Web Mercator with transparency
        nepal_admin_ratios_mercator.plot(
            column='LandslideRatio',
            ax=ax3,
            cmap='rainbow',
            edgecolor='gray',
            linewidth=0.3,
            alpha=0.6,
            legend=False,
            missing_kwds={'color': 'lightgray', 'edgecolor': 'red', 'hatch': '////', 'alpha': 0.3}
        )

        ctx.add_basemap(ax3, crs=nepal_admin_ratios_mercator.crs,
                       source=ctx.providers.OpenTopoMap,
                       alpha=0.5, attribution=False, zoom=8)
        print("  ✓ Contextily basemap added")

        # Add axis labels with lat/lon conversion
        def format_lon(x, pos):
            lon = x / 20037508.34 * 180
            return f'{lon:.1f}E'

        def format_lat(y, pos):
            lat = (2 * np.arctan(np.exp(y / 20037508.34 * 180 * np.pi / 180)) - np.pi / 2) * 180 / np.pi
            return f'{lat:.1f}N'

        ax3.xaxis.set_major_formatter(ticker.FuncFormatter(format_lon))
        ax3.yaxis.set_major_formatter(ticker.FuncFormatter(format_lat))
        ax3.tick_params(axis='both', labelsize=10)

    except Exception as e:
        print(f"  ! Could not add contextily basemap: {e}")
        # Fallback: use WGS84 with simple background
        nepal_admin_ratios_wgs84 = nepal_admin_with_ratios.to_crs("EPSG:4326")
        nepal_admin_ratios_wgs84.plot(
            column='LandslideRatio',
            ax=ax3,
            cmap='rainbow',
            edgecolor='gray',
            linewidth=0.3,
            alpha=0.6,
            legend=False,
            missing_kwds={'color': 'lightgray', 'edgecolor': 'red', 'hatch': '////', 'alpha': 0.3}
        )
        ax3.set_facecolor('lightgray')
        ax3.tick_params(axis='both', labelsize=10)
        ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.1f}E'))
        ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, p: f'{y:.1f}N'))

    # Use adjustable='box-forced' for better alignment
    ax3.set_aspect('equal', adjustable='box')

    # Add colorbar with matching position to ax2
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="3%", pad=0.1)

    # Create scalar mappable for colorbar
    vmin = nepal_admin_with_ratios['LandslideRatio'].dropna().min()
    vmax = nepal_admin_with_ratios['LandslideRatio'].dropna().max()
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap='rainbow', norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('Landslide/Shaking Damage Ratio (%)', fontsize=14, color='red')
    cbar.ax.tick_params(labelsize=12)

    # Add "B" label matching style of "A"
    ax3.text(-0.05, 1.05, 'B', transform=ax3.transAxes, fontsize=20, va='top')

    # Save combined figure with more padding
    fig3.tight_layout(pad=2.0)

    png_path = os.path.join(output_dir, "Fig3_district_impacts_ratio.png")

    fig3.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"  ✓ Figure 3 saved to:")
    print(f"    - PNG: {png_path}")

    plt.show()

    return fig3


def main():
    """
    Main function to generate Figure 3.
    """
    # Configuration
    base_dir = os.path.expanduser("/mnt/CEPH_PROJECTS/Proslide/Alex/SajagN")
    data_dir = os.path.expanduser("/mnt/CEPH_PROJECTS/Proslide/Alex/SajagN/data")
    output_dir = os.path.join(base_dir, "FIGURES")

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    data = load_data(data_dir)

    # Create figure
    fig = create_figure_3(data['impact_data'], data['nepal_admin'], output_dir)

    print("\nFigure 3 generation complete!")


if __name__ == "__main__":
    main()
