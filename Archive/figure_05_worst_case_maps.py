"""
Figure 5: Worst-Case Impact Maps (Absolute and Proportional)

This script generates a 3x2 panel grid showing maximum impacts from earthquake shaking
and landslides across Nepal districts. Each row represents a hazard type (Earthquake,
Building Landslide, Road Landslide) with absolute impacts (left) and proportional impacts
(right) using a consistent color scale.

Output files:
    - Fig5_worst_case_maps.png (high-resolution PNG)
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

    return nepal_admin_impacts


def generate_figure_5(nepal_admin_impacts, output_dir):
    """
    Generate Figure 5: Worst-case impacts (absolute and proportional).

    Creates a 3x2 panel grid with:
    - Row 1: Earthquake impacts (left=absolute, right=proportional)
    - Row 2: Building Landslide impacts (left=absolute, right=proportional)
    - Row 3: Road Landslide impacts (left=absolute, right=proportional)

    Parameters:
    -----------
    nepal_admin_impacts : GeoDataFrame
        Impact data merged with administrative boundaries
    output_dir : str
        Directory to save output files
    """
    print("Creating Figure 5: Worst-case impacts (absolute and proportional)...")

    fig5, axes5 = plt.subplots(3, 2, figsize=(14, 11))

    # Use SAME color scheme and scale for all landslide panels
    # Get consistent scales across ALL impact types
    all_absolute = np.concatenate([
        nepal_admin_impacts['Earthquake_worst'].dropna().values,
        nepal_admin_impacts['BuildingLandslide_worst'].dropna().values,
        nepal_admin_impacts['RoadLandslide_worst'].dropna().values
    ])
    all_proportional = np.concatenate([
        nepal_admin_impacts['PropEarthquake_worst'].dropna().values,
        nepal_admin_impacts['PropBuildingLandslide_worst'].dropna().values,
        nepal_admin_impacts['PropRoadLandslide_worst'].dropna().values
    ])

    vmin_abs = all_absolute[all_absolute > 0].min() if np.any(all_absolute > 0) else 1
    vmax_abs = all_absolute.max()
    vmin_prop = 0
    vmax_prop = all_proportional.max()

    cmap = 'rainbow'

    # ROW 1: Earthquake (shaking)
    create_map(nepal_admin_impacts, 'Earthquake_worst', axes5[0, 0],
              '', cmap, 'log', 'Worst-case impact\n(number of buildings)', vmin=vmin_abs, vmax=vmax_abs)
    axes5[0, 0].text(-0.12, 1.05, 'A', transform=axes5[0, 0].transAxes,
                     fontsize=20, va='top')
    axes5[0, 0].text(-0.25, 0.5, 'Earthquake\n(Shaking)', transform=axes5[0, 0].transAxes,
                     fontsize=14, va='center', rotation=90)

    create_map(nepal_admin_impacts, 'PropEarthquake_worst', axes5[0, 1],
              '', cmap, 'identity', 'Worst-case impact\n(% of buildings)', vmin=vmin_prop, vmax=vmax_prop)
    axes5[0, 1].text(-0.12, 1.05, 'B', transform=axes5[0, 1].transAxes,
                     fontsize=20, va='top')

    # ROW 2: Building Landslide
    create_map(nepal_admin_impacts, 'BuildingLandslide_worst', axes5[1, 0],
              '', cmap, 'log', 'Worst-case impact\n(number of buildings)', vmin=vmin_abs, vmax=vmax_abs)
    axes5[1, 0].text(-0.12, 1.05, 'C', transform=axes5[1, 0].transAxes,
                     fontsize=20, va='top')
    axes5[1, 0].text(-0.25, 0.5, 'Building\nLandslide', transform=axes5[1, 0].transAxes,
                     fontsize=14, va='center', rotation=90)

    create_map(nepal_admin_impacts, 'PropBuildingLandslide_worst', axes5[1, 1],
              '', cmap, 'identity', 'Worst-case impact\n(% of buildings)', vmin=vmin_prop, vmax=vmax_prop)
    axes5[1, 1].text(-0.12, 1.05, 'D', transform=axes5[1, 1].transAxes,
                     fontsize=20, va='top')

    # ROW 3: Road Landslide
    create_map(nepal_admin_impacts, 'RoadLandslide_worst', axes5[2, 0],
              '', cmap, 'log', 'Worst-case impact\n(number of road segments)', vmin=vmin_abs, vmax=vmax_abs)
    axes5[2, 0].text(-0.12, 1.05, 'E', transform=axes5[2, 0].transAxes,
                     fontsize=20, va='top')
    axes5[2, 0].text(-0.25, 0.5, 'Road\nLandslide', transform=axes5[2, 0].transAxes,
                     fontsize=14, va='center', rotation=90)

    create_map(nepal_admin_impacts, 'PropRoadLandslide_worst', axes5[2, 1],
              '', cmap, 'identity', 'Worst-case impact\n(% of road segments)', vmin=vmin_prop, vmax=vmax_prop)
    axes5[2, 1].text(-0.12, 1.05, 'F', transform=axes5[2, 1].transAxes,
                     fontsize=20, va='top')

    fig5.tight_layout(h_pad=1.0, w_pad=1.5)
    fig5.subplots_adjust(left=0.08, wspace=0.35, hspace=0.35)

    # Save with NEW filenames
    fig5.savefig(os.path.join(output_dir, "Fig5_worst_case_maps.png"), dpi=300, bbox_inches='tight')
    print("✓ Figure 5 saved (worst-case absolute and proportional)")
    plt.close(fig5)

    return fig5


def main():
    """
    Main function to generate Figure 5.
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
    nepal_admin_impacts = load_data(base_dir, data_dir)

    # Generate figure
    fig5 = generate_figure_5(nepal_admin_impacts, output_dir)

    print("\n" + "="*80)
    print("Figure 5 generation complete!")
    print(f"Output files saved to: {output_dir}")
    print("  - Fig5_worst_case_maps.png")
    print("="*80)


if __name__ == "__main__":
    main()
