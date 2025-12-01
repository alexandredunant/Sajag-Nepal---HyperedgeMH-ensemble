"""
Figure 7: Remoteness Analysis - Four-Panel Visualization

This script generates Figure 7, a comprehensive 4-panel analysis showing:
- Panel A: Health posts remoteness index map
- Panel B: Correlation between District HQ and Health Posts remoteness indices
- Panel C: Percentage of roads affected by landslides vs remoteness categories
- Panel D: Total worst-case impact vs remoteness categories

The figure combines geospatial visualization with statistical analysis to understand
the relationship between district remoteness (accessibility to health facilities and
district headquarters) and multi-hazard impacts.

Output files:
- Fig7_remoteness_analysis.png (300 DPI)
- Fig7_remoteness_analysis.pdf (vector format)

Usage:
    python figure_07_remoteness_analysis.py
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.ticker as mticker
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
from utils import apply_science_style


# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
apply_science_style()

# Import from local modules
from data_loader import initialize_data, get_output_path, ensure_output_directory
from utils import apply_science_style, format_lon, format_lat


def prepare_remoteness_data(data):
    """
    Prepare remoteness data for analysis and merge with administrative boundaries.

    Parameters:
    -----------
    data : dict
        Data dictionary from initialize_data() containing remoteness and administrative data

    Returns:
    --------
    GeoDataFrame
        Administrative boundaries with remoteness indices and impact statistics
    """
    print("Preparing remoteness data...")

    # Get data components
    nepal_admin = data['admin_boundaries'].copy()
    remoteness_data = data['remoteness']['data'].copy()
    municipalities_df = data['remoteness']['municipalities'].copy()

    # Get impact statistics
    stats_eqimpact = data['impact_stats']['eqimpact']
    stats_buildings_lsimpact = data['impact_stats']['buildings_lsimpact']
    stats_roads_lsimpact = data['impact_stats']['roads_lsimpact']

    # Standardize district names
    nepal_admin['DISTRICT'] = nepal_admin['DISTRICT'].str.upper()
    municipalities_df['DISTRICT'] = municipalities_df['DISTRICT'].str.upper()
    remoteness_data['DISTRICT'] = remoteness_data['DISTRICT'].str.upper()

    # Calculate district-level remoteness indices
    print("Calculating remoteness indices by facility type...")

    # Filter by facility type and season
    health_posts = remoteness_data[
        (remoteness_data['Facility'] == 'Health posts and sub-health posts') &
        (remoteness_data['season'] == 'Normal season')
    ].copy()

    district_hq = remoteness_data[
        (remoteness_data['Facility'] == 'District Headquarters') &
        (remoteness_data['season'] == 'Normal season')
    ].copy()

    # Aggregate remoteness by district
    health_remoteness = health_posts.groupby('DISTRICT').agg(
        Health_Normal=('weighted_remoteness', 'sum')
    ).reset_index()

    dhq_remoteness = district_hq.groupby('DISTRICT').agg(
        DHQ_Normal=('weighted_remoteness', 'sum')
    ).reset_index()

    # Normalize to [0, 1]
    for col in ['Health_Normal', 'DHQ_Normal']:
        for remote_df in [health_remoteness, dhq_remoteness]:
            if col in remote_df.columns:
                max_val = remote_df[col].max()
                if max_val > 0:
                    remote_df[col] = remote_df[col] / max_val

    # Merge remoteness data
    nepal_admin = nepal_admin.merge(health_remoteness, on='DISTRICT', how='left')
    nepal_admin = nepal_admin.merge(dhq_remoteness, on='DISTRICT', how='left')

    # Calculate remoteness categories based on DHQ
    print("Creating remoteness categories...")
    nepal_admin['Remoteness_Category'] = pd.cut(
        nepal_admin['DHQ_Normal'],
        bins=[0, 0.33, 0.67, 1.0],
        labels=['Low', 'Medium', 'High']
    )

    # Aggregate impacts by district
    print("Aggregating impact statistics by district...")

    # Earthquake impacts
    eq_by_district = stats_eqimpact.groupby('DISTRICT').agg(
        EarthquakeImpact_worst=('damage_worst_case_current', 'sum')
    ).reset_index()

    # Building landslide impacts
    bldg_by_district = stats_buildings_lsimpact.groupby('DISTRICT').agg(
        BuildingImpact_worst=('damage_worst_case_current', 'sum')
    ).reset_index()

    # Road impacts
    roads_by_district = stats_roads_lsimpact.groupby('DISTRICT').agg(
        RoadImpact_worst=('damage_worst_case_current', 'sum'),
        TotalRoads=('road_segment_id', 'count'),
        RoadsLandslide_worst=('damage_worst_case_current', 'sum')
    ).reset_index()

    # Calculate proportion of roads affected by landslides
    roads_by_district['PropRoadLandslide_worst'] = (
        roads_by_district['RoadsLandslide_worst'] /
        roads_by_district['TotalRoads'].replace(0, np.nan)
    )

    # Merge all impact data
    nepal_admin['DISTRICT'] = nepal_admin['DISTRICT'].str.upper()
    nepal_admin = nepal_admin.merge(eq_by_district, on='DISTRICT', how='left')
    nepal_admin = nepal_admin.merge(bldg_by_district, on='DISTRICT', how='left')
    nepal_admin = nepal_admin.merge(roads_by_district, on='DISTRICT', how='left')

    # Calculate total worst-case impact
    nepal_admin['TotalImpact_worst'] = (
        nepal_admin['EarthquakeImpact_worst'].fillna(0) +
        nepal_admin['BuildingImpact_worst'].fillna(0) +
        nepal_admin['RoadImpact_worst'].fillna(0)
    )

    return nepal_admin


def create_figure_7(nepal_admin_risk, output_dir):
    """
    Create Figure 7: 4-panel remoteness analysis visualization.

    Parameters:
    -----------
    nepal_admin_risk : GeoDataFrame
        Administrative boundaries with remoteness and impact data
    output_dir : str
        Directory to save output files
    """
    print("\nCreating Figure 7: 4-panel remoteness visualization...")

    # Create Figure with 2x2 panels
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Convert to Web Mercator for contextily
    nepal_admin_risk_mercator = nepal_admin_risk.to_crs("EPSG:3857")

    # =============================================================================
    # PANEL A (Upper Left): Health posts remoteness map
    # =============================================================================
    ax_a = axes[0, 0]

    nepal_admin_risk_mercator.plot(
        column='Health_Normal',
        ax=ax_a,
        cmap='Blues',
        vmin=0,
        vmax=1,
        edgecolor='gray',
        linewidth=0.3,
        alpha=0.6,
        legend=False,
        missing_kwds={'color': 'lightgray', 'edgecolor': 'red', 'hatch': '////', 'alpha': 0.3}
    )

    # Add contextily basemap
    try:
        import contextily as ctx
        ctx.add_basemap(ax_a, crs=nepal_admin_risk_mercator.crs,
                       source=ctx.providers.OpenTopoMap,
                       alpha=0.5, attribution=False, zoom=8)
    except Exception as e:
        print(f"Could not add basemap to Panel A: {e}")
        ax_a.set_facecolor('lightgray')

    # Add colorbar
    divider_a = make_axes_locatable(ax_a)
    cax_a = divider_a.append_axes("right", size="3%", pad=0.1)
    norm_a = colors.Normalize(vmin=0, vmax=1)
    sm_a = plt.cm.ScalarMappable(cmap='Blues', norm=norm_a)
    sm_a.set_array([])
    cbar_a = plt.colorbar(sm_a, cax=cax_a)
    cbar_a.set_label('Remoteness Index (Health Posts)', fontsize=14)
    cbar_a.ax.tick_params(labelsize=12)

    ax_a.xaxis.set_major_formatter(mticker.FuncFormatter(format_lon))
    ax_a.yaxis.set_major_formatter(mticker.FuncFormatter(format_lat))
    ax_a.tick_params(axis='both', labelsize=12)
    ax_a.set_aspect('equal', adjustable='box')

    # Add panel label
    ax_a.text(-0.08, 1.05, 'A', transform=ax_a.transAxes, fontsize=22, va='top')

    # =============================================================================
    # PANEL B (Upper Right): Correlation of remoteness indices (scatter plot)
    # =============================================================================
    ax_b = axes[0, 1]

    valid_data = nepal_admin_risk.dropna(subset=['DHQ_Normal', 'Health_Normal'])
    ax_b.scatter(valid_data['DHQ_Normal'], valid_data['Health_Normal'],
                alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    ax_b.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1.5)
    ax_b.set_xlabel('District HQ Remoteness Index', fontsize=14)
    ax_b.set_ylabel('Health Posts Remoteness Index', fontsize=14)
    ax_b.tick_params(axis='both', labelsize=12)
    ax_b.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
    ax_b.set_aspect('equal')
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)

    # Add panel label
    ax_b.text(-0.08, 1.05, 'B', transform=ax_b.transAxes, fontsize=22, va='top')

    # =============================================================================
    # PANEL C (Lower Left): % roads affected by landslide vs remoteness categories
    # =============================================================================
    ax_c = axes[1, 0]

    if 'Remoteness_Category' in nepal_admin_risk.columns:
        # Calculate proportion of roads affected by landslides for each district
        nepal_admin_risk['PropRoadLandslide_worst_percent'] = (
            nepal_admin_risk['PropRoadLandslide_worst'] * 100
        )

        # Create box plot
        category_order = ['Low', 'Medium', 'High']
        valid_categories = nepal_admin_risk['Remoteness_Category'].dropna().unique()
        plot_order = [cat for cat in category_order if cat in valid_categories]

        sns.boxplot(x='Remoteness_Category', y='PropRoadLandslide_worst_percent',
                   data=nepal_admin_risk, ax=ax_c, order=plot_order,
                   palette=['green', 'orange', 'red'])
        ax_c.set_xlabel('Remoteness Category', fontsize=14)
        ax_c.set_ylabel('% Roads Affected by Landslides', fontsize=14)
        ax_c.tick_params(axis='both', labelsize=12)
        ax_c.grid(True, axis='y', linestyle='--', alpha=0.7, linewidth=0.5)
    else:
        ax_c.text(0.5, 0.5, 'Remoteness categories\nnot available',
                 ha='center', va='center', transform=ax_c.transAxes, fontsize=14)

    # Add panel label
    ax_c.text(-0.08, 1.05, 'C', transform=ax_c.transAxes, fontsize=22, va='top')

    # =============================================================================
    # PANEL D (Lower Right): Current worst case vs remoteness categories
    # =============================================================================
    ax_d = axes[1, 1]

    if 'Remoteness_Category' in nepal_admin_risk.columns:
        # Create box plot showing worst-case total impact vs remoteness categories
        category_order = ['Low', 'Medium', 'High']
        valid_categories = nepal_admin_risk['Remoteness_Category'].dropna().unique()
        plot_order = [cat for cat in category_order if cat in valid_categories]

        sns.boxplot(x='Remoteness_Category', y='TotalImpact_worst',
                   data=nepal_admin_risk, ax=ax_d, order=plot_order,
                   palette=['green', 'orange', 'red'])
        ax_d.set_yscale('log')
        ax_d.set_xlabel('Remoteness Category', fontsize=14)
        ax_d.set_ylabel('Total Worst-Case Impact (log scale)', fontsize=14)
        ax_d.tick_params(axis='both', labelsize=12)
        ax_d.grid(True, axis='y', linestyle='--', alpha=0.7, linewidth=0.5)

        # Format y-axis with commas
        ax_d.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    else:
        ax_d.text(0.5, 0.5, 'Remoteness categories\nnot available',
                 ha='center', va='center', transform=ax_d.transAxes, fontsize=14)

    # Add panel label
    ax_d.text(-0.08, 1.05, 'D', transform=ax_d.transAxes, fontsize=22, va='top')

    # =============================================================================
    # Save figure
    # =============================================================================
    fig.tight_layout(pad=2.0)

    png_path = os.path.join(output_dir, "Fig7_remoteness_analysis.png")

    fig.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"\nFigure 7 saved successfully:")
    print(f"  PNG: {png_path}")

    plt.show()

    return fig


def main():
    """
    Main function to generate Figure 7.
    """
    print("="*70)
    print("Figure 7: Remoteness Analysis")
    print("="*70)

    # Initialize data
    print("\nInitializing data...")
    data = initialize_data()
    ensure_output_directory(data)

    # Prepare remoteness data
    nepal_admin_risk = prepare_remoteness_data(data)

    # Create figure
    fig = create_figure_7(nepal_admin_risk, data['paths']['output_dir'])

    print("\n" + "="*70)
    print("Figure 7 generation complete!")
    print("="*70)


if __name__ == "__main__":
    main()
