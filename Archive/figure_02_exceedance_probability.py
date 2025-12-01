#!/usr/bin/env python3
"""
Figure 2: Exceedance Probability Plots

This script generates exceedance probability curves for three hazard types:
1. Buildings damaged by earthquake shaking
2. Buildings damaged by landslides
3. Roads damaged by landslides

The exceedance probability (or complementary cumulative distribution function)
shows the probability that a given impact value will be exceeded. This is useful
for understanding the frequency distribution of impacts across scenarios.

Output files:
- Fig2_exceedance_probability.png (300 DPI)
- Fig2_exceedance_probability.pdf (publication quality)

Author: Multi-hazard Risk Analysis Team
License: MIT
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
from utils import apply_science_style


# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
apply_science_style()


def setup_paths():
    """
    Set up and validate necessary directory paths.

    Returns:
    --------
    dict : Dictionary containing configured paths with keys:
        - 'data_dir': Data directory containing CSV files
        - 'output_dir': Output directory for figures
    """
    paths = {
        'data_dir': os.path.expanduser(
            "/mnt/CEPH_PROJECTS/Proslide/Alex/SajagN/data"
        ),
        'output_dir': os.path.join(
            os.path.expanduser("/mnt/CEPH_PROJECTS/Proslide/Alex/SajagN"),
            "FIGURES"
        )
    }

    # Validate paths exist
    if not os.path.exists(paths['data_dir']):
        raise FileNotFoundError(f"Data directory not found at: {paths['data_dir']}")

    # Create output directory if it doesn't exist
    os.makedirs(paths['output_dir'], exist_ok=True)

    return paths


def load_impact_statistics(data_dir):
    """
    Load impact statistics CSV files.

    Parameters:
    -----------
    data_dir : str
        Path to the data directory containing aggregated_stats

    Returns:
    --------
    tuple : (stats_eqimpact, stats_buildings_lsimpact, stats_roads_lsimpact)
        DataFrames containing impact statistics
    """
    print("\nLoading impact statistics...")

    stats_dir = os.path.join(data_dir, "aggregated_stats")

    # Earthquake impacts
    eq_file = os.path.join(stats_dir, "stats_eqimpact_2024-06-24_physiog.csv")
    stats_eqimpact = pd.read_csv(eq_file)
    print(f"  ✓ Earthquake impacts: {len(stats_eqimpact)} records")

    # Building landslide impacts
    buildings_ls_file = os.path.join(
        stats_dir, "stats_buildings_lsimpact_2024-06-24_physiog.csv"
    )
    stats_buildings_lsimpact = pd.read_csv(buildings_ls_file)
    print(f"  ✓ Building landslide impacts: {len(stats_buildings_lsimpact)} records")

    # Road landslide impacts
    roads_ls_file = os.path.join(
        stats_dir, "stats_roads_lsimpact_2024-06-24_physiog.csv"
    )
    stats_roads_lsimpact = pd.read_csv(roads_ls_file)
    print(f"  ✓ Road landslide impacts: {len(stats_roads_lsimpact)} records")

    return stats_eqimpact, stats_buildings_lsimpact, stats_roads_lsimpact


def calculate_exceedance(data, impact_type):
    """
    Calculate exceedance probability for a given impact type.

    Exceedance probability is the probability that a given impact value is exceeded,
    calculated as the rank divided by the total count. For a sorted array of impacts,
    the exceedance probability at position i is (i+1) / total_count.

    Parameters:
    -----------
    data : DataFrame
        Input dataframe containing impact data
    impact_type : str
        The column name of the impact type to analyze
        (e.g., 'collapse_mid_sum' for earthquake, 'impact_sum' for landslides)

    Returns:
    --------
    DataFrame
        DataFrame with columns:
        - 'Impact': Sorted impact values (descending order)
        - 'Probability': Exceedance probability [0, 1]
        - 'HazardType': The impact type label used

    Example:
    --------
    >>> df = pd.DataFrame({'collapse_mid_sum': [100, 500, 200, 1000]})
    >>> result = calculate_exceedance(df, 'collapse_mid_sum')
    >>> print(result)
           Impact  Probability     HazardType
        0    1000          0.25     Earthquake
        1     500          0.50     Earthquake
        2     200          0.75     Earthquake
        3     100          1.00     Earthquake
    """
    # Map impact column to hazard type label
    type_mapping = {
        'collapse_mid_sum': 'Earthquake',
        'impact_sum': 'BuildingLandslide',
        'road_impact_sum': 'RoadLandslide'
    }

    # Sort impacts in descending order
    sorted_impacts = np.sort(data[impact_type].values)[::-1]

    # Calculate exceedance probabilities
    exceedance_prob = np.arange(1, len(sorted_impacts) + 1) / len(sorted_impacts)

    # Determine hazard type label
    hazard_label = type_mapping.get(impact_type, impact_type)

    return pd.DataFrame({
        'Impact': sorted_impacts,
        'Probability': exceedance_prob,
        'HazardType': hazard_label
    })


def prepare_impact_data(stats_eqimpact, stats_buildings_lsimpact, stats_roads_lsimpact):
    """
    Prepare and aggregate impact data across all districts and scenarios.

    Parameters:
    -----------
    stats_eqimpact : DataFrame
        Earthquake impact statistics
    stats_buildings_lsimpact : DataFrame
        Building landslide impact statistics
    stats_roads_lsimpact : DataFrame
        Road landslide impact statistics

    Returns:
    --------
    DataFrame
        Aggregated impact data with columns:
        - 'District': District name
        - 'Event': Event/scenario identifier
        - 'Earthquake': Earthquake impact count
        - 'BuildingLandslide': Building landslide impact count
        - 'RoadLandslide': Road landslide impact count
    """
    print("\nPreparing impact data...")

    # Combine all impact types into one dataframe
    impact_data = pd.DataFrame({
        'District': stats_eqimpact['DISTRICT'],
        'Event': stats_eqimpact['event'],
        'Earthquake': stats_eqimpact['collapse_mid_sum'],
        'BuildingLandslide': stats_buildings_lsimpact['impact_sum'],
        'RoadLandslide': stats_roads_lsimpact['impact_sum']
    })

    # Aggregate impacts across all districts for each scenario
    aggregate_impacts = impact_data.groupby('Event').agg(
        Earthquake=('Earthquake', 'sum'),
        BuildingLandslide=('BuildingLandslide', 'sum'),
        RoadLandslide=('RoadLandslide', 'sum')
    ).reset_index()

    print(f"  ✓ Prepared impact data for {len(impact_data['District'].unique())} districts")
    print(f"  ✓ Total scenarios: {len(impact_data['Event'].unique())}")

    return aggregate_impacts


def define_colors_and_labels():
    """
    Define consistent colors and labels for hazard types.

    Returns:
    --------
    tuple : (colors_dict, labels_dict)
        Dictionaries mapping hazard types to colors and labels
    """
    colors_dict = {
        'Earthquake': '#9e9e9e',           # Gray
        'BuildingLandslide': '#7474ee',    # Purple
        'RoadLandslide': '#3700ff'         # Dark blue
    }

    labels_dict = {
        'Earthquake': 'Buildings damaged by shaking',
        'BuildingLandslide': 'Buildings damaged by landslides',
        'RoadLandslide': 'Roads damaged by landslides'
    }

    return colors_dict, labels_dict


def create_figure(aggregate_impacts, colors_dict, labels_dict):
    """
    Create the exceedance probability figure.

    Parameters:
    -----------
    aggregate_impacts : DataFrame
        Aggregated impact data
    colors_dict : dict
        Mapping of hazard types to colors
    labels_dict : dict
        Mapping of hazard types to labels

    Returns:
    --------
    tuple : (fig, ax) - Matplotlib figure and axes objects
    """
    print("\nCreating figure...")

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5))

    # Calculate exceedance probabilities for each hazard type
    exceedance_dfs = [
        calculate_exceedance(aggregate_impacts, col)
        for col in ['Earthquake', 'BuildingLandslide', 'RoadLandslide']
    ]
    exceedance_data = pd.concat(exceedance_dfs, ignore_index=True)

    # Plot exceedance curves
    for hazard_type in ['Earthquake', 'BuildingLandslide', 'RoadLandslide']:
        hazard_data = exceedance_data[exceedance_data['HazardType'] == hazard_type]
        ax.plot(
            hazard_data['Impact'],
            hazard_data['Probability'] * 100,
            color=colors_dict[hazard_type],
            label=labels_dict[hazard_type],
            marker='o',
            markersize=4,
            linewidth=2,
            markeredgecolor='white',
            markeredgewidth=0.5
        )
    print("  ✓ Exceedance curves plotted")

    # Set axis properties
    ax.set_xscale('log')
    ax.set_xlim(100, exceedance_data['Impact'].max() * 1.1)
    ax.set_ylim(0, 100)
    ax.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)

    # Add dashed line at 50% to show median impacts
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
    print("  ✓ Median line added")

    # Format axes
    ax.set_xlabel('Absolute Count', fontsize=14)
    ax.set_ylabel('Exceedance Probability (%)', fontsize=12)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    # Add legend
    ax.legend(
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        loc='upper right'
    )
    print("  ✓ Legend added")

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
    output_png = os.path.join(output_dir, "Fig2_exceedance_probability.png")
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"  ✓ PNG saved to: {output_png}")

    # PDF output
    output_pdf = os.path.join(output_dir, "Fig2_exceedance_probability.pdf")
    fig.savefig(output_pdf, bbox_inches='tight')


def main():
    """
    Main function to orchestrate the creation of Figure 2.

    This function coordinates all steps:
    1. Setup paths
    2. Load impact statistics
    3. Prepare aggregated impact data
    4. Create the figure
    5. Save to PNG and PDF
    """
    print("=" * 70)
    print("FIGURE 2: Exceedance Probability Plots")
    print("=" * 70)

    try:
        # Setup paths
        paths = setup_paths()
        print(f"\nPaths configured:")
        print(f"  Data directory:   {paths['data_dir']}")
        print(f"  Output directory: {paths['output_dir']}")

        # Load impact statistics
        (stats_eqimpact,
         stats_buildings_lsimpact,
         stats_roads_lsimpact) = load_impact_statistics(paths['data_dir'])

        # Prepare aggregated data
        aggregate_impacts = prepare_impact_data(
            stats_eqimpact,
            stats_buildings_lsimpact,
            stats_roads_lsimpact
        )

        # Define colors and labels
        colors_dict, labels_dict = define_colors_and_labels()

        # Create figure
        fig, ax = create_figure(aggregate_impacts, colors_dict, labels_dict)

        # Save figure
        save_figure(fig, paths['output_dir'])

        # Display figure if in interactive environment
        try:
            plt.show()
        except:
            pass

        print("\n" + "=" * 70)
        print("✓ Figure 2 complete and saved successfully!")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"\n✗ Error generating Figure 2:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
