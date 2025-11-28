#!/usr/bin/env python3
"""
Figure S2: Peak Ground Acceleration (PGA) Maps for 30 Earthquake Scenarios

This script generates a multi-panel figure showing PGA maps for all 30
earthquake scenarios in the Robinson et al. (2018) ensemble.

Author: Generated for Sajag Nepal HyperedgeMH ensemble project
Date: 2025-11-28
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import gridspec
import rasterio
from rasterio.plot import show
import geopandas as gpd
import warnings
from utils import apply_science_style

warnings.filterwarnings('ignore')
apply_science_style()



def setup_paths():
    """
    Set up file paths for data and output.

    Returns
    -------
    dict
        Dictionary containing path configurations
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    paths = {
        'base_dir': base_dir,
        'data_dir': os.path.join(base_dir, "data"),
        'tif_dir': os.path.join(base_dir, "data", "tif", "robinson_ensemble_expanded"),
        'shp_dir': os.path.join(base_dir, "data", "shp"),
        'output_dir': "/mnt/CEPH_PROJECTS/Proslide/Alex/SajagN/Sajag-Nepal---HyperedgeMH-ensemble/FIGURES"
    }

    # Create output directory if it doesn't exist
    os.makedirs(paths['output_dir'], exist_ok=True)

    return paths


def load_scenario_files(tif_dir):
    """
    Load all PGA scenario TIF files.

    Parameters
    ----------
    tif_dir : str
        Directory containing PGA TIF files

    Returns
    -------
    list
        List of sorted TIF file paths
    """
    pattern = os.path.join(tif_dir, "Format__UTM45_*.tif")
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        raise FileNotFoundError(f"No PGA TIF files found in {tif_dir}")

    print(f"  ✓ Found {len(files)} PGA scenario files")
    return files


def extract_scenario_name(filepath):
    """
    Extract a readable scenario name from the filename.

    Parameters
    ----------
    filepath : str
        Full path to TIF file

    Returns
    -------
    str
        Formatted scenario name
    """
    filename = os.path.basename(filepath)
    # Remove prefix and extension
    name = filename.replace("Format__UTM45_", "").replace(".tif", "")
    # Replace underscores and format magnitude
    name = name.replace("_", " ").replace("pt", ".")
    # Remove IDW suffix if present
    name = name.replace(" IDW", "")
    return name


def load_nepal_boundary(shp_dir):
    """
    Load Nepal administrative boundary for overlay.

    Parameters
    ----------
    shp_dir : str
        Directory containing shapefiles

    Returns
    -------
    GeoDataFrame or None
        Nepal boundary or None if not found
    """
    try:
        # Try to find Nepal shapefile
        shp_pattern = os.path.join(shp_dir, "**", "*nepal*.shp")
        shp_files = glob.glob(shp_pattern, recursive=True)

        if not shp_files:
            shp_pattern = os.path.join(shp_dir, "**", "hermes*.shp")
            shp_files = glob.glob(shp_pattern, recursive=True)

        if shp_files:
            gdf = gpd.read_file(shp_files[0])
            # Reproject to UTM 45N if needed
            if gdf.crs != 'EPSG:32645':
                gdf = gdf.to_crs('EPSG:32645')
            print(f"  ✓ Loaded Nepal boundary from {os.path.basename(shp_files[0])}")
            return gdf
        else:
            print("  ⚠ Nepal boundary shapefile not found (will proceed without overlay)")
            return None
    except Exception as e:
        print(f"  ⚠ Could not load Nepal boundary: {e}")
        return None


def create_figure_s2(scenario_files, nepal_boundary, output_dir):
    """
    Create Figure S2 with all 30 PGA maps.

    Parameters
    ----------
    scenario_files : list
        List of PGA TIF file paths
    nepal_boundary : GeoDataFrame or None
        Nepal boundary for overlay
    output_dir : str
        Output directory path
    """
    print("\nCreating Figure S2: PGA maps for 30 earthquake scenarios...")

    n_scenarios = len(scenario_files)
    n_cols = 3
    n_rows = int(np.ceil(n_scenarios / n_cols))

    # Create figure with gridspec
    fig = plt.figure(figsize=(12, 24))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                          hspace=0.15, wspace=0.05,
                          left=0.05, right=0.97,
                          top=0.97, bottom=0.03)

    # Common colormap and normalization
    cmap = plt.cm.YlOrRd
    vmin, vmax = 0, 1.0  # PGA in g
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Plot each scenario
    for idx, filepath in enumerate(scenario_files):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        try:
            # Load raster
            with rasterio.open(filepath) as src:
                pga_data = src.read(1, masked=True)

                # Clip extreme values
                pga_data = np.clip(pga_data, 0, 1.0)

                # Plot raster
                show(pga_data, ax=ax, transform=src.transform,
                     cmap=cmap, norm=norm, interpolation='nearest')

                # Overlay Nepal boundary if available
                if nepal_boundary is not None:
                    nepal_boundary.boundary.plot(ax=ax, edgecolor='black',
                                                linewidth=0.5, alpha=0.7)

            # Extract and format title
            scenario_name = extract_scenario_name(filepath)
            ax.set_title(scenario_name, fontsize=14, pad=3)

            # Remove axis ticks and labels for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')

            # Keep axis frame
            for spine in ax.spines.values():
                spine.set_edgecolor('gray')
                spine.set_linewidth(0.5)

        except Exception as e:
            print(f"  ⚠ Error loading {os.path.basename(filepath)}: {e}")
            ax.text(0.5, 0.5, 'Error loading\nscenario',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

    # Add colorbar
    cbar_ax = fig.add_axes([0.25, 0.01, 0.5, 0.008])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', extend='max')
    cbar.set_label('Peak Ground Acceleration (g)', fontsize=16)
    cbar.ax.tick_params(labelsize=10)

    # Save figure
    png_path = os.path.join(output_dir, "FigS2_pga_ensemble_maps.png")

    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')

    plt.close(fig)

    print(f"  ✓ Figure S2 saved to:")
    print(f"    - PNG: {png_path}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("FIGURE S2: Peak Ground Acceleration (PGA) Maps for 30 Earthquake Scenarios")
    print("=" * 80)

    # Setup paths
    print("\nSetting up paths...")
    paths = setup_paths()
    print(f"  Base directory:   {paths['base_dir']}")
    print(f"  TIF directory:    {paths['tif_dir']}")
    print(f"  Output directory: {paths['output_dir']}")

    # Load scenario files
    print("\nLoading PGA scenario files...")
    scenario_files = load_scenario_files(paths['tif_dir'])

    # Load Nepal boundary for overlay
    print("\nLoading Nepal boundary...")
    nepal_boundary = load_nepal_boundary(paths['shp_dir'])

    # Create figure
    create_figure_s2(scenario_files, nepal_boundary, paths['output_dir'])

    print("\n" + "=" * 80)
    print("✓ Figure S2 generation complete!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
