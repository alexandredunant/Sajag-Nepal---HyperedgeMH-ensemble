#!/usr/bin/env python3
"""
Figure 1: Study Area Map with Administrative Boundaries

This script generates a comprehensive map of Nepal showing:
- Administrative district boundaries (77 districts)
- Ward boundaries
- DMG fault lines
- District labels
- Scale bar and north arrow
- OpenTopoMap basemap

Output files:
- Fig1_study_area_map.png (300 DPI)
- Fig1_study_area_map.pdf (publication quality)

Author: Multi-hazard Risk Analysis Team
License: MIT
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import geopandas as gpd
import contextily as ctx
from matplotlib_scalebar.scalebar import ScaleBar
import warnings


# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def setup_paths():
    """
    Set up and validate necessary directory paths.

    Returns:
    --------
    dict : Dictionary containing configured paths with keys:
        - 'base_dir': Base project directory
        - 'data_dir': Data directory containing shapefiles
        - 'output_dir': Output directory for figures
    """
    paths = {
        'base_dir': os.path.expanduser(
            "/mnt/CEPH_PROJECTS/Proslide/Alex/SajagN"
        ),
        'data_dir': os.path.expanduser(
            "/mnt/CEPH_PROJECTS/Proslide/Alex/SajagN/data"
        ),
        'output_dir': os.path.join(
            os.path.expanduser("/mnt/CEPH_PROJECTS/Proslide/Alex/SajagN"),
            "FIGURES"
        )
    }

    # Create output directory if it doesn't exist
    os.makedirs(paths['output_dir'], exist_ok=True)

    # Validate paths exist
    for path_name, path_value in paths.items():
        if path_name != 'output_dir' and not os.path.exists(path_value):
            raise FileNotFoundError(f"{path_name} not found at: {path_value}")

    return paths


def format_lon(x, pos):
    """
    Convert Web Mercator x-coordinate to longitude in degrees.

    Parameters:
    -----------
    x : float
        X-coordinate in Web Mercator projection
    pos : int
        Position parameter (required by matplotlib formatter)

    Returns:
    --------
    str
        Formatted longitude string (e.g., "85.3E")
    """
    lon = x / 20037508.34 * 180
    return f'{lon:.1f}E'


def format_lat(y, pos):
    """
    Convert Web Mercator y-coordinate to latitude in degrees.

    Parameters:
    -----------
    y : float
        Y-coordinate in Web Mercator projection
    pos : int
        Position parameter (required by matplotlib formatter)

    Returns:
    --------
    str
        Formatted latitude string (e.g., "28.5N")
    """
    lat = (2 * np.arctan(np.exp(y / 20037508.34 * 180 * np.pi / 180)) - np.pi / 2) * 180 / np.pi
    return f'{lat:.1f}N'


def load_data(paths):
    """
    Load all necessary geographic data.

    Parameters:
    -----------
    paths : dict
        Dictionary of paths from setup_paths()

    Returns:
    --------
    tuple : (districts_mercator, wards_mercator, faults_mercator)
        All GeoDataFrames converted to EPSG:3857 (Web Mercator)
    """
    print("\nLoading geographic data...")

    # Load and convert districts
    district_file = os.path.join(
        paths['data_dir'], "shp", "hermes_NPL_new_wgs", "hermes_NPL_new_wgs_2.shp"
    )
    districts = gpd.read_file(district_file)
    print(f"  ✓ Loaded {len(districts)} districts")

    # Convert to Web Mercator
    districts_mercator = districts.to_crs("EPSG:3857")
    print(f"  ✓ Converted districts to EPSG:3857 (Web Mercator)")

    # Load and convert wards if available
    ward_file = os.path.join(
        paths['data_dir'], "shp", "Ward", "NP_Ward_P21_Wgs84.shp"
    )
    wards_mercator = None
    if os.path.exists(ward_file):
        try:
            wards = gpd.read_file(ward_file)
            wards_mercator = wards.to_crs("EPSG:3857")
            print(f"  ✓ Loaded and converted {len(wards)} wards")
        except Exception as e:
            print(f"  ✗ Could not load ward boundaries: {e}")

    # Load and convert faults if available
    fault_file = os.path.join(
        paths["base_dir"], "figs", "GMT", "Nepal_Faults_DMG.shp"
    )
    faults_mercator = None
    if os.path.exists(fault_file):
        try:
            faults = gpd.read_file(fault_file)
            faults_mercator = faults.to_crs("EPSG:3857")
            print(f"  ✓ Loaded and converted {len(faults)} fault lines")
        except Exception as e:
            print(f"  ✗ Could not load fault boundaries: {e}")

    return districts_mercator, wards_mercator, faults_mercator


def create_figure(districts_mercator, wards_mercator, faults_mercator):
    """
    Create the study area map figure.

    Parameters:
    -----------
    districts_mercator : GeoDataFrame
        Districts in Web Mercator projection
    wards_mercator : GeoDataFrame or None
        Wards in Web Mercator projection (optional)
    faults_mercator : GeoDataFrame or None
        Faults in Web Mercator projection (optional)

    Returns:
    --------
    tuple : (fig, ax) - Matplotlib figure and axes objects
    """
    print("\nCreating figure...")

    # Create figure with appropriate size for publication
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Plot districts with transparent orange fill
    districts_mercator.plot(
        ax=ax,
        facecolor='orange',
        edgecolor='black',
        linewidth=2.0,
        alpha=0.5
    )
    print("  ✓ District boundaries plotted with 50% transparent orange fill")

    # Add OpenTopoMap basemap
    ctx.add_basemap(
        ax,
        crs=districts_mercator.crs,
        source=ctx.providers.OpenTopoMap,
        alpha=0.5,
        attribution=False,
        zoom=8
    )
    print("  ✓ Basemap added (OpenTopoMap)")

    # Plot ward boundaries if available
    if wards_mercator is not None:
        wards_mercator.plot(
            ax=ax,
            facecolor='none',
            edgecolor='gray',
            linewidth=0.15,
            alpha=0.5
        )
        print("  ✓ Ward boundaries plotted")

    # Plot fault lines if available - use high zorder to ensure visibility on top
    if faults_mercator is not None:
        faults_mercator.plot(
            ax=ax,
            edgecolor='red',
            linewidth=5.0,
            alpha=0.9,
            zorder=10,
            label='Faults - Nepalese Department of Mines and Geology'
        )
        print("  ✓ DMG fault lines plotted (linewidth=5.0, zorder=10)")

    # Add district labels
    for idx, row in districts_mercator.iterrows():
        dist_name = row['DISTRICT']
        centroid = row.geometry.centroid
        ax.annotate(
            dist_name.title(),
            xy=(centroid.x, centroid.y),
            fontsize=14,
            fontweight='normal',
            color='black',
            ha='center',
            va='center',
            bbox=dict(
                boxstyle='round,pad=0.2',
                facecolor='white',
                alpha=0.7,
                edgecolor='none'
            )
        )
    print(f"  ✓ Added labels for all {len(districts_mercator)} districts")

    # Set coordinate formatters
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_lat))
    ax.tick_params(axis='both', labelsize=10)

    # Set aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # Add scale bar (100 km)
    # Web Mercator uses meters, so 1 unit = 1 meter
    scalebar = ScaleBar(
        dx=1,
        units='m',
        length_fraction=0.25,
        location='lower left',
        box_alpha=0.8,
        color='black',
        font_properties={'size': 10, 'weight': 'bold'}
    )
    ax.add_artist(scalebar)
    print("  ✓ Scale bar added")

    # Add north arrow
    arrow_x = 0.95
    arrow_y = 0.95
    ax.annotate(
        'N',
        xy=(arrow_x, arrow_y),
        xytext=(arrow_x, arrow_y - 0.05),
        xycoords='axes fraction',
        fontsize=16,
        
        ha='center',
        va='bottom',
        arrowprops=dict(arrowstyle='->', lw=2, color='black')
    )
    print("  ✓ North arrow added")

    # Add legend if faults are present
    if faults_mercator is not None:
        ax.legend(
            loc='upper right',
            frameon=True,
            facecolor='white',
            framealpha=0.8,
            fontsize=14
        )

    return fig, ax


def save_figure(fig, output_dir):
    """
    Save figure in both PNG and PDF formats.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure object to save
    output_dir : str
        Directory to save figures to

    Returns:
    --------
    None
    """
    print("\nSaving figure...")

    # PNG output
    output_png = os.path.join(output_dir, "Fig1_study_area_map.png")
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"  ✓ PNG saved to: {output_png}")

    # PDF output
    output_pdf = os.path.join(output_dir, "Fig1_study_area_map.pdf")
    fig.savefig(output_pdf, bbox_inches='tight')


def main():
    """
    Main function to orchestrate the creation of Figure 1.

    This function coordinates all steps:
    1. Setup paths
    2. Load geographic data
    3. Create the figure
    4. Save to PNG and PDF
    """
    print("=" * 70)
    print("FIGURE 1: Study Area Map with Administrative Boundaries")
    print("=" * 70)

    try:
        # Setup paths
        paths = setup_paths()
        print(f"\nPaths configured:")
        print(f"  Base directory:   {paths['base_dir']}")
        print(f"  Data directory:   {paths['data_dir']}")
        print(f"  Output directory: {paths['output_dir']}")

        # Load data
        districts_mercator, wards_mercator, faults_mercator = load_data(paths)

        # Create figure
        fig, ax = create_figure(districts_mercator, wards_mercator, faults_mercator)

        # Save figure
        save_figure(fig, paths['output_dir'])

        # Display figure if in interactive environment
        try:
            plt.show()
        except:
            pass

        print("\n" + "=" * 70)
        print("✓ Figure 1 complete and saved successfully!")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"\n✗ Error generating Figure 1:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
