"""
Figure 6: Standard Deviation of Impacts Across Scenarios

This script generates a 3x1 panel grid showing the spatial variability (standard deviation)
of impacts from earthquake shaking and landslides across Nepal districts. Each row represents
a hazard type (Earthquake, Building Landslide, Road Landslide) using a logarithmic color scale
to highlight high-variability regions.

Output files:
    - Fig6_standard_deviation.png (high-resolution PNG)
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
from utils import apply_science_style


warnings.filterwarnings('ignore')
apply_science_style()


def create_map(data, column, ax, title, cmap, transform='log', legend_title='Impacts',
              highlight_nodata=False, nodata_condition=None, vmin=None, vmax=None, add_basemap=True):
    """
    Create a map with contextily basemap and properly positioned colorbar.

    Parameters:
    -----------
    data : GeoDataFrame
        The data to plot
    column : str
        The column to use for coloring
    ax : matplotlib.axes.Axes
        The axes to plot on
    title : str
        The title for the plot
    cmap : str
        The colormap to use
    transform : str
        'log' for logarithmic scale, 'identity' for linear scale
    legend_title : str
        Title for the colorbar
    vmin, vmax : float or None
        Manual color scale limits
    add_basemap : bool
        Whether to add contextily basemap
    """
    # Convert to Web Mercator for contextily
    plot_data = data.to_crs("EPSG:3857") if add_basemap else data.copy()

    valid_data = plot_data[column].dropna()
    norm = None

    vmin_calc = vmin if vmin is not None else (valid_data.min() if len(valid_data) > 0 else 0)
    vmax_calc = vmax if vmax is not None else (valid_data.max() if len(valid_data) > 0 else 1)

    if transform == 'log':
        positive_values = valid_data[valid_data > 0]
        if len(positive_values) > 0:
            log_vmin = vmin_calc if vmin_calc > 0 else positive_values.min()
            log_vmax = vmax_calc
            if np.isclose(log_vmin, log_vmax):
                log_vmax = log_vmin * 1.1 + 0.1
            norm = colors.LogNorm(vmin=log_vmin, vmax=log_vmax)
        else:
            norm = colors.Normalize(vmin=vmin_calc, vmax=vmax_calc)
    else:
        if np.isclose(vmin_calc, vmax_calc):
            vmax_calc = vmin_calc + 1
        norm = colors.Normalize(vmin=vmin_calc, vmax=vmax_calc)

    # Plot the data with transparency
    plot_data.plot(
        column=column, ax=ax, cmap=cmap, norm=norm, legend=False,
        edgecolor='gray', linewidth=0.3, alpha=0.6,
        missing_kwds={"color": "lightgray", "edgecolor": "red", "hatch": "////", "alpha": 0.3}
    )

    # Add contextily basemap
    if add_basemap:
        try:
            import contextily as ctx
            ctx.add_basemap(ax, crs=plot_data.crs,
                           source=ctx.providers.OpenTopoMap,
                           alpha=0.5, attribution=False, zoom=8)
        except Exception as e:
            print(f"Could not add basemap: {e}")
            ax.set_facecolor('lightgray')

    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label(legend_title, fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    if transform == 'log':
        cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    elif 'Percent' in legend_title or '%' in legend_title:
        cbar.ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
    else:
        cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.2f}'))

    # Add coordinate labels
    def format_lon(x, pos):
        lon = x / 20037508.34 * 180
        return f'{lon:.1f}E'

    def format_lat(y, pos):
        lat = (2 * np.arctan(np.exp(y / 20037508.34 * 180 * np.pi / 180)) - np.pi / 2) * 180 / np.pi
        return f'{lat:.1f}N'

    if add_basemap:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_lon))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_lat))
        ax.tick_params(axis='both', labelsize=10)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_title(title, fontsize=14, pad=10)
    ax.set_aspect('equal', adjustable='box')

    return ax


def load_data(base_dir, data_dir):
    """
    Load and prepare all necessary data for figure generation.

    Parameters:
    -----------
    base_dir : str
        Base directory path
    data_dir : str
        Data directory path

    Returns:
    --------
    nepal_admin_impacts : GeoDataFrame
        GeoDataFrame with impact data merged with administrative boundaries
    impact_data : DataFrame
        DataFrame with raw impact data by district and event
    """
    stats_dir = os.path.join(data_dir, "aggregated_stats")

    # Load impact statistics
    print("Loading impact data...")
    stats_buildings_lsimpact = pd.read_csv(
        os.path.join(stats_dir, "stats_buildings_lsimpact_2024-06-24_physiog.csv")
    )
    stats_eqimpact = pd.read_csv(
        os.path.join(stats_dir, "stats_eqimpact_2024-06-24_physiog.csv")
    )
    stats_roads_lsimpact = pd.read_csv(
        os.path.join(stats_dir, "stats_roads_lsimpact_2024-06-24_physiog.csv")
    )

    # Load administrative boundaries
    print("Loading administrative boundaries...")
    nepal_admin = gpd.read_file(
        os.path.join(data_dir, "shp", "hermes_NPL_new_wgs", "hermes_NPL_new_wgs_2.shp")
    )
    nepal_admin = nepal_admin.to_crs("EPSG:32645")

    # Load asset counts
    print("Loading asset counts...")
    asset_counts_file = os.path.join(stats_dir, "district_asset_counts.csv")
    if os.path.exists(asset_counts_file):
        asset_counts = pd.read_csv(asset_counts_file)
    else:
        print("WARNING: Asset counts file not found. Creating placeholder.")
        asset_counts = pd.DataFrame({
            'District': nepal_admin['DISTRICT'].unique(),
            'BuildingCount': np.nan,
            'RoadSegmentCount': np.nan
        })

    # Combine impact data
    print("Preparing impact data...")
    impact_data = pd.DataFrame({
        'District': stats_eqimpact['DISTRICT'],
        'Event': stats_eqimpact['event'],
        'Earthquake': stats_eqimpact['collapse_mid_sum'],
        'BuildingLandslide': stats_buildings_lsimpact['impact_sum'],
        'RoadLandslide': stats_roads_lsimpact['impact_sum']
    })

    # Calculate worst-case impacts by district
    worst_case_impacts = impact_data.groupby('District').agg(
        Earthquake_worst=('Earthquake', 'max'),
        BuildingLandslide_worst=('BuildingLandslide', 'max'),
        RoadLandslide_worst=('RoadLandslide', 'max')
    ).reset_index()

    mean_impacts = impact_data.groupby('District').agg(
        Earthquake_mean=('Earthquake', 'mean'),
        BuildingLandslide_mean=('BuildingLandslide', 'mean'),
        RoadLandslide_mean=('RoadLandslide', 'mean')
    ).reset_index()

    # Merge all data
    nepal_admin_impacts = nepal_admin.merge(
        worst_case_impacts, left_on='DISTRICT', right_on='District', how='left'
    ).merge(
        mean_impacts, left_on='DISTRICT', right_on='District', how='left', suffixes=('', '_y')
    ).merge(
        asset_counts, left_on='DISTRICT', right_on='District', how='left', suffixes=('', '_z')
    )

    # Calculate proportions
    for stat in ['worst', 'mean']:
        nepal_admin_impacts[f'PropEarthquake_{stat}'] = np.where(
            nepal_admin_impacts['BuildingCount'] > 0,
            nepal_admin_impacts[f'Earthquake_{stat}'] / nepal_admin_impacts['BuildingCount'] * 100,
            np.nan
        )
        nepal_admin_impacts[f'PropBuildingLandslide_{stat}'] = np.where(
            nepal_admin_impacts['BuildingCount'] > 0,
            nepal_admin_impacts[f'BuildingLandslide_{stat}'] / nepal_admin_impacts['BuildingCount'] * 100,
            np.nan
        )
        nepal_admin_impacts[f'PropRoadLandslide_{stat}'] = np.where(
            nepal_admin_impacts['RoadSegmentCount'] > 0,
            nepal_admin_impacts[f'RoadLandslide_{stat}'] / nepal_admin_impacts['RoadSegmentCount'] * 100,
            np.nan
        )

    return nepal_admin_impacts, impact_data


def generate_figure_6(nepal_admin_impacts, impact_data, output_dir):
    """
    Generate Figure 6: Standard deviation of impacts.

    Creates a 3x1 panel grid with:
    - Row 1: Earthquake standard deviation
    - Row 2: Building Landslide standard deviation
    - Row 3: Road Landslide standard deviation

    Parameters:
    -----------
    nepal_admin_impacts : GeoDataFrame
        Impact data merged with administrative boundaries
    impact_data : DataFrame
        Raw impact data by district and event
    output_dir : str
        Directory to save output files
    """
    print("Creating Figure 6: Standard deviation of impacts...")

    # Calculate standard deviation for each district and impact type
    std_dev_data = impact_data.groupby('District').agg(
        Earthquake_std=('Earthquake', 'std'),
        BuildingLandslide_std=('BuildingLandslide', 'std'),
        RoadLandslide_std=('RoadLandslide', 'std')
    ).reset_index()

    # Merge std dev data with impacts
    nepal_admin_impacts_std = nepal_admin_impacts.merge(
        std_dev_data, left_on='DISTRICT', right_on='District', how='left', suffixes=('', '_std')
    )

    fig6, axes6 = plt.subplots(3, 1, figsize=(8, 11))

    cmap_std = 'rainbow'

    # ROW 1: Earthquake Standard Deviation
    create_map(nepal_admin_impacts_std, 'Earthquake_std', axes6[0],
              '', cmap_std, 'log', 'Standard deviation\n(number of buildings)', vmin=1, vmax=None)
    axes6[0].text(-0.12, 1.05, 'A', transform=axes6[0].transAxes,
                  fontsize=20, va='top')
    axes6[0].text(-0.25, 0.5, 'Earthquake\n(Shaking)', transform=axes6[0].transAxes,
                  fontsize=14, va='center', rotation=90)

    # ROW 2: Building Landslide Standard Deviation
    create_map(nepal_admin_impacts_std, 'BuildingLandslide_std', axes6[1],
              '', cmap_std, 'log', 'Standard deviation\n(number of buildings)', vmin=1, vmax=None)
    axes6[1].text(-0.12, 1.05, 'B', transform=axes6[1].transAxes,
                  fontsize=20, va='top')
    axes6[1].text(-0.25, 0.5, 'Building\nLandslide', transform=axes6[1].transAxes,
                  fontsize=14, va='center', rotation=90)

    # ROW 3: Road Landslide Standard Deviation
    create_map(nepal_admin_impacts_std, 'RoadLandslide_std', axes6[2],
              '', cmap_std, 'log', 'Standard deviation\n(number of road segments)', vmin=1, vmax=None)
    axes6[2].text(-0.12, 1.05, 'C', transform=axes6[2].transAxes,
                  fontsize=20, va='top')
    axes6[2].text(-0.25, 0.5, 'Road\nLandslide', transform=axes6[2].transAxes,
                  fontsize=14, va='center', rotation=90)

    fig6.tight_layout(h_pad=1.5)
    fig6.subplots_adjust(left=0.08)

    # Save with NEW filenames
    fig6.savefig(os.path.join(output_dir, "Fig6_standard_deviation.png"), dpi=300, bbox_inches='tight')
    print("✓ Figure 6 saved (standard deviation)")
    plt.close(fig6)

    return fig6


def main():
    """
    Main function to generate Figure 6.
    """
    # Set up directories
    base_dir = "/mnt/CEPH_PROJECTS/Proslide/Alex/SajagN"
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "FIGURES")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Base directory: {base_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Load data
    nepal_admin_impacts, impact_data = load_data(base_dir, data_dir)

    # Generate figure
    fig6 = generate_figure_6(nepal_admin_impacts, impact_data, output_dir)

    print("\n" + "="*80)
    print("Figure 6 generation complete!")
    print(f"Output files saved to: {output_dir}")
    print("  - Fig6_standard_deviation.png")
    print("="*80)


if __name__ == "__main__":
    main()
