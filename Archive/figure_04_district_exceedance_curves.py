"""
Figure 4: District Exceedance Curves
Creates a three-panel figure showing exceedance probability curves by physiography:
  Panel A: Earthquake/Shaking impacts
  Panel B: Building landslide impacts
  Panel C: Road landslide impacts

Each panel includes an inset map showing the physiographic regions of Nepal.

This script generates:
  - Fig4_district_exceedance_curves.png
  - Fig4_district_exceedance_curves.pdf

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
from tqdm import tqdm
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
        Dictionary containing impact_data, nepal_admin, and stats_eqimpact
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


def compute_exceedance_by_district(impact_data):
    """
    Compute exceedance probability by district and impact type.

    Parameters
    ----------
    impact_data : pd.DataFrame
        DataFrame with columns: District, Event, Earthquake, BuildingLandslide, RoadLandslide

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: District, ImpactType, Count, ExceedanceProb, Physiography
    """
    exceedance_data_by_district = []

    for district in tqdm(impact_data['District'].unique(), desc="Processing districts"):
        for impact_type in ['Earthquake', 'BuildingLandslide', 'RoadLandslide']:
            district_impacts = impact_data[impact_data['District'] == district]
            sorted_impacts = np.sort(district_impacts[impact_type].values)[::-1]
            exceedance_prob = np.arange(1, len(sorted_impacts) + 1) / len(sorted_impacts)

            for impact, prob in zip(sorted_impacts, exceedance_prob):
                exceedance_data_by_district.append({
                    'District': district,
                    'ImpactType': impact_type,
                    'Count': impact,
                    'ExceedanceProb': prob
                })

    return pd.DataFrame(exceedance_data_by_district)


def map_physiography(exceedance_data, stats_eqimpact):
    """
    Map physiography to districts in exceedance data.

    Parameters
    ----------
    exceedance_data : pd.DataFrame
        DataFrame with district exceedance data
    stats_eqimpact : pd.DataFrame
        Statistics dataframe containing physiography information

    Returns
    -------
    pd.DataFrame
        DataFrame with Physiography column added
    """
    # Create physiography mapping
    district_physiography = {
        row['DISTRICT']: row['Physiography'] for _, row in stats_eqimpact.iterrows()
    }

    # Add physiography to exceedance data
    exceedance_data['Physiography'] = exceedance_data['District'].map(district_physiography)

    # Normalize physiography naming and verify Terai
    exceedance_data['Physiography'] = exceedance_data['Physiography'].replace('Tarai', 'Terai')

    # Ensure all districts have a physiography
    if exceedance_data['Physiography'].isna().any():
        print("  ! Warning: Some districts lack physiography data. Filling with 'Unknown'.")
        exceedance_data['Physiography'] = exceedance_data['Physiography'].fillna('Unknown')

    return exceedance_data, district_physiography


def plot_exceedance_by_physiography(data, impact_type, ax, label, nepal_admin,
                                     district_physiography, add_inset=False):
    """
    Plot exceedance curves by physiography for a specific impact type.

    Parameters
    ----------
    data : pd.DataFrame
        Exceedance data with Physiography column
    impact_type : str
        One of 'Earthquake', 'BuildingLandslide', or 'RoadLandslide'
    ax : matplotlib.axes.Axes
        Axes to plot on
    label : str
        Panel label (e.g., 'A', 'B', 'C')
    nepal_admin : gpd.GeoDataFrame
        Administrative boundaries GeoDataFrame
    district_physiography : dict
        Mapping from district to physiography
    add_inset : bool
        Whether to add an inset map of physiographic regions
    """
    physiography_colors = {
        'High Mountain': '#e95aee',
        'Hill': 'red',
        'Middle Mountain': '#f5e727',
        'Siwalik': '#0fe24f',
        'Terai': '#44a2fa',
        'Unknown': 'gray'
    }

    impact_data = data[data['ImpactType'] == impact_type]

    # Plot exceedance curves for each district
    for district in impact_data['District'].unique():
        district_data = impact_data[impact_data['District'] == district]
        if not district_data.empty:
            physiography = district_data['Physiography'].iloc[0]
            ax.plot(
                district_data['Count'],
                district_data['ExceedanceProb'],
                color=physiography_colors.get(physiography, 'gray'),
                alpha=0.8,
                linewidth=0.8
            )

    # Configure axes
    ax.set_xscale('log')
    ax.set_xlim(1, 150000)
    ax.set_ylim(0, 1)

    # Set appropriate xlabel based on impact type
    if impact_type == 'Earthquake':
        xlabel_text = 'Number of Affected Buildings by Shaking'
    elif impact_type == 'BuildingLandslide':
        xlabel_text = 'Number of Affected Buildings by Landslides'
    elif impact_type == 'RoadLandslide':
        xlabel_text = 'Number of Affected Roads by Landslides'
    else:
        xlabel_text = 'Number of Impacts'

    ax.set_xlabel(xlabel_text, fontsize=14)
    ax.set_ylabel('Exceedance Probability', fontsize=14)
    ax.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.axhline(y=0.5, linestyle='--', color='gray', alpha=0.7)

    # Add panel label
    ax.text(
        -0.1, 1.1, label,
        transform=ax.transAxes, fontsize=20, va='top'
    )

    # Create legend with physiographic regions present in data
    handles = [
        Patch(color=c, label=p)
        for p, c in physiography_colors.items()
        if p in impact_data['Physiography'].unique()
    ]
    ax.legend(
        handles=handles, loc='upper right',
        title='Physiography', frameon=True,
        fancybox=True, framealpha=0.9, fontsize=11
    )

    # Add inset map if requested
    if add_inset:
        inset_ax = ax.inset_axes([0.4, 0.6, 0.5, 0.35])

        # Create physiography mapping for admin data
        nepal_admin_with_physio = nepal_admin.merge(
            pd.DataFrame(list(district_physiography.items()), columns=['DISTRICT', 'Physiography']),
            on='DISTRICT', how='left'
        )

        # Ensure 'Terai' is correctly mapped
        nepal_admin_with_physio['Physiography'] = nepal_admin_with_physio['Physiography'].replace('Tarai', 'Terai')
        nepal_admin_with_physio['Physiography'] = nepal_admin_with_physio['Physiography'].fillna('Unknown')

        # Plot physiography map
        nepal_admin_with_physio.plot(
            column='Physiography',
            ax=inset_ax,
            color=[physiography_colors.get(p, 'gray') for p in nepal_admin_with_physio['Physiography']],
            edgecolor='black',
            linewidth=0.3,
            legend=False
        )
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        inset_ax.set_aspect('equal')

    return ax


def create_figure_4(impact_data, nepal_admin, stats_eqimpact, output_dir):
    """
    Create Figure 4: District-wise exceedance probability plots.

    Parameters
    ----------
    impact_data : pd.DataFrame
        DataFrame with impact data by district
    nepal_admin : gpd.GeoDataFrame
        Administrative boundaries
    stats_eqimpact : pd.DataFrame
        Statistics dataframe with physiography information
    output_dir : str
        Directory to save output figures
    """
    print("Creating Figure 4: District-wise exceedance probability plots...")

    # Compute exceedance probabilities
    print("  Computing exceedance probabilities by district...")
    exceedance_data_by_district = compute_exceedance_by_district(impact_data)

    # Map physiography
    print("  Mapping physiography to districts...")
    exceedance_data_by_district, district_physiography = map_physiography(
        exceedance_data_by_district, stats_eqimpact
    )

    # Create figure with three subplots
    fig4 = plt.figure(figsize=(10, 10))
    gs = fig4.add_gridspec(3, 1, hspace=0.20)

    # Plot each impact type
    print("  Plotting exceedance curves...")

    plot_exceedance_by_physiography(
        exceedance_data_by_district, 'Earthquake',
        fig4.add_subplot(gs[0, :]), 'A', nepal_admin, district_physiography, add_inset=True
    )

    plot_exceedance_by_physiography(
        exceedance_data_by_district, 'BuildingLandslide',
        fig4.add_subplot(gs[1, :]), 'B', nepal_admin, district_physiography, add_inset=True
    )

    plot_exceedance_by_physiography(
        exceedance_data_by_district, 'RoadLandslide',
        fig4.add_subplot(gs[2, :]), 'C', nepal_admin, district_physiography, add_inset=True
    )

    # Save figure
    fig4.tight_layout(pad=0.4)

    png_path = os.path.join(output_dir, "Fig4_district_exceedance_curves.png")

    fig4.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"  ✓ Figure 4 saved to:")
    print(f"    - PNG: {png_path}")

    plt.show()

    return fig4


def main():
    """
    Main function to generate Figure 4.
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
    fig = create_figure_4(
        data['impact_data'],
        data['nepal_admin'],
        data['stats_eqimpact'],
        output_dir
    )

    print("\nFigure 4 generation complete!")


if __name__ == "__main__":
    main()
