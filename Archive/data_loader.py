"""
Data Loader Module for Multi-hazard Earthquake Risk Analysis

This module consolidates all data loading and initialization code for the
consolidated figures generation pipeline. It handles:
- Library imports and configuration
- Matplotlib styling
- Path setup
- CSV data loading from aggregated_stats/
- Shapefile loading
- Data preprocessing and merging

Paper: Multi-hazard scenario ensembles for estimating earthquake risk
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
import scienceplots
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# =============================================================================
# 1. IMPORT VALIDATION
# =============================================================================

def validate_imports():
    """Validate that all required libraries have been imported successfully."""
    print("All libraries imported successfully!")
    return True


# =============================================================================
# 2. MATPLOTLIB STYLE CONFIGURATION
# =============================================================================

def configure_matplotlib_style():
    """
    Configure matplotlib for scientific publication quality figures.
    Attempts to use SciencePlots style with fallback to custom settings.
    """
    try:
        plt.style.use(['science', 'no-latex', 'vibrant'])
        print("Using SciencePlots style")
    except Exception as e:
        # Custom scientific style settings if SciencePlots not installed
        print(f"SciencePlots not available ({e}), using custom scientific style")

        plt.rcParams['axes.linewidth'] = 1.0
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 11
        plt.rcParams['legend.title_fontsize'] = 11
        plt.rcParams['grid.linewidth'] = 0.5
        plt.rcParams['lines.linewidth'] = 1.5
        plt.rcParams['lines.markersize'] = 5
        plt.rcParams['figure.dpi'] = 300

    return True


# =============================================================================
# 3. PATH CONFIGURATION
# =============================================================================

def setup_paths():
    """
    Set up and create necessary directory paths.

    Returns:
    --------
    dict : Dictionary containing configured paths
    """
    paths = {}

    # Set working directory and output paths
    paths['base_dir'] = os.path.expanduser(
        "/mnt/CEPH_PROJECTS/Proslide/Alex/SajagN"
    )
    paths['data_dir'] = os.path.expanduser(
        "/mnt/CEPH_PROJECTS/Proslide/Alex/SajagN/data"
    )
    paths['output_dir'] = os.path.join(paths['base_dir'], "figs")
    paths['stats_dir'] = os.path.join(paths['data_dir'], "aggregated_stats")
    paths['shp_dir'] = os.path.join(paths['data_dir'], "shp")
    paths['remoteness_dir'] = os.path.join(paths['data_dir'], "Remotenessdata")

    # Create output directory if it doesn't exist
    os.makedirs(paths['output_dir'], exist_ok=True)

    # Print path information
    print(f"Base directory:       {paths['base_dir']}")
    print(f"Data directory:       {paths['data_dir']}")
    print(f"Stats directory:      {paths['stats_dir']}")
    print(f"Shapefile directory:  {paths['shp_dir']}")
    print(f"Remoteness directory: {paths['remoteness_dir']}")
    print(f"Output directory:     {paths['output_dir']}")

    return paths


# =============================================================================
# 4. DATA LOADING FUNCTIONS
# =============================================================================

def load_impact_statistics(stats_dir):
    """
    Load impact statistics CSV files.

    Parameters:
    -----------
    stats_dir : str
        Path to the aggregated_stats directory

    Returns:
    --------
    tuple : (stats_buildings_lsimpact, stats_eqimpact, stats_roads_lsimpact)
    """
    print("\nLoading impact statistics datasets...")

    # Building landslide impacts
    stats_buildings_lsimpact = pd.read_csv(
        os.path.join(stats_dir, "stats_buildings_lsimpact_2024-06-24_physiog.csv")
    )
    print(f"  Building landslide impacts: {len(stats_buildings_lsimpact)} records")

    # Earthquake impacts
    stats_eqimpact = pd.read_csv(
        os.path.join(stats_dir, "stats_eqimpact_2024-06-24_physiog.csv")
    )
    print(f"  Earthquake impacts: {len(stats_eqimpact)} records")

    # Road landslide impacts
    stats_roads_lsimpact = pd.read_csv(
        os.path.join(stats_dir, "stats_roads_lsimpact_2024-06-24_physiog.csv")
    )
    print(f"  Road landslide impacts: {len(stats_roads_lsimpact)} records")

    return stats_buildings_lsimpact, stats_eqimpact, stats_roads_lsimpact


def load_administrative_boundaries(shp_dir):
    """
    Load Nepal administrative boundaries shapefile and transform to UTM 45N.

    Parameters:
    -----------
    shp_dir : str
        Path to the shapefile directory

    Returns:
    --------
    GeoDataFrame : Administrative boundaries in UTM 45N (EPSG:32645)
    """
    print("\nLoading administrative boundaries...")

    # Load shapefile
    nepal_admin = gpd.read_file(
        os.path.join(shp_dir, "hermes_NPL_new_wgs", "hermes_NPL_new_wgs_2.shp")
    )

    print(f"  Original CRS: {nepal_admin.crs}")

    # Transform to UTM 45N for better visualization and area calculations
    nepal_admin = nepal_admin.to_crs("EPSG:32645")
    print(f"  Transformed to CRS: {nepal_admin.crs}")
    print(f"  Loaded {len(nepal_admin)} administrative units")

    return nepal_admin


def load_asset_counts(stats_dir, nepal_admin):
    """
    Load asset counts (buildings and roads per district).

    Parameters:
    -----------
    stats_dir : str
        Path to the aggregated_stats directory
    nepal_admin : GeoDataFrame
        Administrative boundaries (used to create placeholder if file missing)

    Returns:
    --------
    DataFrame : Asset counts by district
    """
    print("\nLoading asset counts...")

    asset_counts_file = os.path.join(stats_dir, "district_asset_counts.csv")

    if os.path.exists(asset_counts_file):
        asset_counts = pd.read_csv(asset_counts_file)
        print(f"  Loaded from: {asset_counts_file}")
        print(f"  Number of districts: {len(asset_counts)}")
        print(f"  Total buildings: {asset_counts['BuildingCount'].sum():,.0f}")
        print(f"  Total road segments: {asset_counts['RoadSegmentCount'].sum():,.0f}")
    else:
        print(f"  WARNING: Asset counts file not found at {asset_counts_file}")
        print(f"  Creating placeholder DataFrame")
        asset_counts = pd.DataFrame({
            'District': nepal_admin['DISTRICT'].unique(),
            'BuildingCount': np.nan,
            'RoadSegmentCount': np.nan
        })

    return asset_counts


def load_remoteness_data(remoteness_dir):
    """
    Load social vulnerability metrics (remoteness data).

    Parameters:
    -----------
    remoteness_dir : str
        Path to the Remotenessdata directory

    Returns:
    --------
    tuple : (remoteness_data, municipalities_data)
    """
    print("\nLoading remoteness/vulnerability data...")

    # Main remoteness data
    nepal_remoteness = pd.read_csv(
        os.path.join(remoteness_dir, "Remoteness_DFID_Data.csv")
    )
    print(f"  Remoteness data: {len(nepal_remoteness)} records")

    # Municipality-level remoteness data
    municipalities_df = pd.read_csv(
        os.path.join(remoteness_dir, "Remoteness_DFID_Municipalities.csv")
    )
    print(f"  Municipalities data: {len(municipalities_df)} records")

    return nepal_remoteness, municipalities_df


# =============================================================================
# 5. MAIN INITIALIZATION FUNCTION
# =============================================================================

def initialize_data():
    """
    Master initialization function that loads all data and returns organized
    dictionary with all datasets and paths.

    Returns:
    --------
    dict : Complete data structure containing:
        - 'paths': Dictionary of all configured paths
        - 'impact_stats': Dictionary with building, earthquake, and road impacts
        - 'admin_boundaries': GeoDataFrame of administrative units
        - 'asset_counts': DataFrame of asset counts per district
        - 'remoteness': Dictionary with remoteness and municipality data
    """
    # Validate imports
    validate_imports()

    # Configure matplotlib
    configure_matplotlib_style()

    # Setup paths
    paths = setup_paths()

    # Load all datasets
    (stats_buildings_lsimpact,
     stats_eqimpact,
     stats_roads_lsimpact) = load_impact_statistics(paths['stats_dir'])

    nepal_admin = load_administrative_boundaries(paths['shp_dir'])

    asset_counts = load_asset_counts(paths['stats_dir'], nepal_admin)

    nepal_remoteness, municipalities_df = load_remoteness_data(paths['remoteness_dir'])

    # Organize all data into a single comprehensive dictionary
    data = {
        'paths': paths,
        'impact_stats': {
            'buildings_lsimpact': stats_buildings_lsimpact,
            'eqimpact': stats_eqimpact,
            'roads_lsimpact': stats_roads_lsimpact
        },
        'admin_boundaries': nepal_admin,
        'asset_counts': asset_counts,
        'remoteness': {
            'data': nepal_remoteness,
            'municipalities': municipalities_df
        }
    }

    print("\n" + "="*70)
    print("DATA INITIALIZATION COMPLETE")
    print("="*70)

    return data


# =============================================================================
# 6. UTILITY FUNCTIONS FOR DATA ACCESS
# =============================================================================

def get_output_path(data, filename):
    """
    Generate full output path for a figure file.

    Parameters:
    -----------
    data : dict
        Data dictionary returned from initialize_data()
    filename : str
        Filename to save

    Returns:
    --------
    str : Full path for output file
    """
    return os.path.join(data['paths']['output_dir'], filename)


def ensure_output_directory(data):
    """
    Ensure output directory exists.

    Parameters:
    -----------
    data : dict
        Data dictionary returned from initialize_data()
    """
    os.makedirs(data['paths']['output_dir'], exist_ok=True)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Example usage of the data loader module.
    """
    # Load all data
    data = initialize_data()

    # Access individual datasets
    building_impacts = data['impact_stats']['buildings_lsimpact']
    eq_impacts = data['impact_stats']['eqimpact']
    admin_boundaries = data['admin_boundaries']

    # Print summary statistics
    print("\n" + "="*70)
    print("DATA SUMMARY")
    print("="*70)
    print(f"\nBuilding Landslide Impacts shape: {building_impacts.shape}")
    print(f"Earthquake Impacts shape: {eq_impacts.shape}")
    print(f"Administrative Boundaries: {len(admin_boundaries)} units")
    print(f"\nOutput directory ready at: {data['paths']['output_dir']}")
