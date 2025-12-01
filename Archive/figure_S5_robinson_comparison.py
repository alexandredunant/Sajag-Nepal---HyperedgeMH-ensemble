"""
Figure S5: Comparison of Normalized Risk Scores with Robinson et al. (2018)

This script generates Figure S5, a focused 2-panel comparison between our
Normalized Risk Scores and Robinson et al. (2018) risk scores.

The figure includes:
- Left panel (A): Horizontal bar plot comparing top 20 districts by Robinson score
  showing both Our Normalized Risk Score and Robinson Risk Score side-by-side

- Right panel (B): Scatter plot with regression line showing the correlation
  between Robinson Risk Score (x-axis) and Our Normalized Risk Score (y-axis)
  with R² value displayed and 1:1 reference line

Output files:
- FigS5_robinson_comparison.png (300 DPI)
- FigS5_robinson_comparison.pdf (vector format)

Usage:
    python figure_S5_robinson_comparison.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress
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
    dict : Dictionary containing configured paths
    """
    paths = {
        'base_dir': os.path.expanduser(
            "/mnt/CEPH_PROJECTS/Proslide/Alex/SajagN"
        ),
        'data_dir': os.path.expanduser(
            "/mnt/CEPH_PROJECTS/Proslide/Alex/SajagN/data"
        )
    }

    paths['stats_dir'] = os.path.join(paths['data_dir'], "aggregated_stats")
    paths['output_dir'] = os.path.join(paths['base_dir'], "figs")

    # Validate and create paths
    if not os.path.exists(paths['data_dir']):
        raise FileNotFoundError(f"Data directory not found at: {paths['data_dir']}")

    os.makedirs(paths['output_dir'], exist_ok=True)

    return paths


def load_robinson_and_normalized_scores(data_dir, stats_dir, output_dir):
    """
    Load Robinson scores and calculate our normalized risk scores from available data.

    Parameters:
    -----------
    data_dir : str
        Path to the data directory
    stats_dir : str
        Path to the aggregated statistics directory
    output_dir : str
        Path to the output directory

    Returns:
    --------
    pd.DataFrame
        Combined dataframe with both our and Robinson risk scores
    """
    print("Loading Robinson et al. (2018) district scores...")

    # Load Robinson scores
    robinson_file = os.path.join(stats_dir, "robinson_district_scores.csv")

    if not os.path.exists(robinson_file):
        # Try alternate location in current directory
        robinson_file = os.path.join(
            os.path.dirname(__file__), "robinson_district_scores.csv"
        )

    if not os.path.exists(robinson_file):
        # Try parent directory
        robinson_file = os.path.join(
            os.path.dirname(__file__), "..", "robinson_district_scores.csv"
        )

    if os.path.exists(robinson_file):
        robinson_data = pd.read_csv(robinson_file)
        print(f"Loaded Robinson scores from {robinson_file}")
    else:
        raise FileNotFoundError(f"Robinson scores file not found at any expected location")

    # Standardize district names in Robinson data
    if 'District' in robinson_data.columns:
        robinson_data['DISTRICT'] = robinson_data['District'].str.upper()
    elif 'district' in robinson_data.columns:
        robinson_data['DISTRICT'] = robinson_data['district'].str.upper()
    else:
        robinson_data['DISTRICT'] = robinson_data.iloc[:, 0].astype(str).str.upper()

    # Ensure Robinson_Risk_Score column exists
    if 'Robinson_Risk_Score' not in robinson_data.columns:
        if 'Total (/6)' in robinson_data.columns:
            # Normalize Total (/6) to [0,1]
            robinson_data['Robinson_Risk_Score'] = robinson_data['Total (/6)'] / 6.0
        else:
            raise ValueError("Could not find Robinson risk score column")

    # Calculate our normalized risk scores from impact statistics
    print("Loading impact statistics...")

    # Load impact data - find files with proper naming convention
    eq_impact_file = None
    bldg_ls_file = None
    roads_ls_file = None

    # Search for files
    for fname in os.listdir(stats_dir):
        if fname.startswith("stats_eqimpact") and fname.endswith(".csv"):
            eq_impact_file = os.path.join(stats_dir, fname)
        elif fname.startswith("stats_buildings_lsimpact") and fname.endswith(".csv"):
            bldg_ls_file = os.path.join(stats_dir, fname)
        elif fname.startswith("stats_roads_lsimpact") and fname.endswith(".csv"):
            roads_ls_file = os.path.join(stats_dir, fname)

    if not eq_impact_file or not bldg_ls_file or not roads_ls_file:
        raise FileNotFoundError("Could not find all required impact CSV files")

    print(f"  Loading: {os.path.basename(eq_impact_file)}")
    print(f"  Loading: {os.path.basename(bldg_ls_file)}")
    print(f"  Loading: {os.path.basename(roads_ls_file)}")

    eq_impact = pd.read_csv(eq_impact_file)
    bldg_ls = pd.read_csv(bldg_ls_file)
    roads_ls = pd.read_csv(roads_ls_file)

    # Standardize DISTRICT column and prepare aggregation
    for df in [eq_impact, bldg_ls, roads_ls]:
        if 'DISTRICT' in df.columns:
            df['DISTRICT'] = df['DISTRICT'].str.upper()
        elif 'District' in df.columns:
            df['DISTRICT'] = df['District'].str.upper()

    # Find columns that contain damage worst case data
    def get_damage_col(df, df_type='eq'):
        """Find the damage column to aggregate"""
        if df_type == 'eq':
            # For earthquake: look for collapse_low_sum (conservative estimate)
            cols = [c for c in df.columns if 'collapse' in c.lower() and 'sum' in c.lower()]
            if cols:
                return cols[0]  # Use first (low) impact as baseline
        else:
            # For landslides (building and roads): look for impact_sum
            cols = [c for c in df.columns if 'impact' in c.lower() and 'sum' in c.lower()]
            if cols:
                return cols[0]

        # Fallback: look for any sum column
        cols = [c for c in df.columns if 'sum' in c.lower()]
        if cols:
            return cols[0]
        return None

    # Get damage columns for each hazard type
    eq_col = get_damage_col(eq_impact, 'eq')
    bldg_col = get_damage_col(bldg_ls, 'ls')
    roads_col = get_damage_col(roads_ls, 'ls')

    print(f"  Using columns: EQ={eq_col}, Bldg={bldg_col}, Roads={roads_col}")

    if not eq_col or not bldg_col or not roads_col:
        raise ValueError("Could not find required damage columns")

    # Calculate worst case impacts by summing across all rows
    eq_damage = eq_impact.groupby('DISTRICT')[eq_col].sum().reset_index()
    eq_damage.columns = ['DISTRICT', 'EarthquakeImpact']

    bldg_damage = bldg_ls.groupby('DISTRICT')[bldg_col].sum().reset_index()
    bldg_damage.columns = ['DISTRICT', 'BuildingImpact']

    roads_damage = roads_ls.groupby('DISTRICT')[roads_col].sum().reset_index()
    roads_damage.columns = ['DISTRICT', 'RoadImpact']

    # Merge impact data
    merged_impacts = eq_damage.copy()
    merged_impacts = merged_impacts.merge(bldg_damage, on='DISTRICT', how='outer')
    merged_impacts = merged_impacts.merge(roads_damage, on='DISTRICT', how='outer')

    # Fill NaN with 0 and calculate total impact
    for col in ['EarthquakeImpact', 'BuildingImpact', 'RoadImpact']:
        merged_impacts[col] = merged_impacts[col].fillna(0)

    merged_impacts['S_worst'] = (
        merged_impacts['EarthquakeImpact'] +
        merged_impacts['BuildingImpact'] +
        merged_impacts['RoadImpact']
    )

    # Normalize impact score
    max_impact = merged_impacts['S_worst'].max()
    if max_impact > 0:
        merged_impacts['RiskScore_Normalized'] = merged_impacts['S_worst'] / max_impact
    else:
        merged_impacts['RiskScore_Normalized'] = 0

    print(f"  Aggregated impacts for {len(merged_impacts)} districts")

    # Merge Robinson scores
    comparison_data = merged_impacts[['DISTRICT', 'RiskScore_Normalized']].copy()
    comparison_data = comparison_data.merge(
        robinson_data[['DISTRICT', 'Robinson_Risk_Score']],
        on='DISTRICT',
        how='inner'
    )

    print(f"  Matched {len(comparison_data)} districts with Robinson data")

    return comparison_data


def create_figure_s5(comparison_data, output_dir):
    """
    Create Figure S5: 2-panel Robinson comparison figure.

    Parameters:
    -----------
    comparison_data : pd.DataFrame
        Dataframe with risk scores
    output_dir : str
        Directory to save output files
    """
    print("\nCreating Figure S5: Robinson Comparison...")

    # Filter to matched districts with complete data
    plot_data = comparison_data[
        comparison_data['Robinson_Risk_Score'].notna() &
        comparison_data['RiskScore_Normalized'].notna()
    ].copy()

    if len(plot_data) == 0:
        print("Error: No districts with complete data for comparison")
        return

    print(f"Plotting {len(plot_data)} districts with complete data")

    # Create figure with 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # =============================================================================
    # PANEL A (Left): Horizontal bar plot - Top 20 by Robinson Score
    # =============================================================================
    ax_a = axes[0]

    top_n = 20
    top_robinson = plot_data.nlargest(top_n, 'Robinson_Risk_Score')[
        ['DISTRICT', 'RiskScore_Normalized', 'Robinson_Risk_Score']
    ].copy()

    # Sort by Robinson score (ascending for horizontal bar plot)
    top_robinson = top_robinson.sort_values('Robinson_Risk_Score', ascending=True)

    x = np.arange(len(top_robinson))
    width = 0.35

    # Create horizontal bars
    bars1 = ax_a.barh(x - width/2, top_robinson['RiskScore_Normalized'], width,
                       label='Normalized Risk Score', color='#ff7f0e', alpha=0.8)
    bars2 = ax_a.barh(x + width/2, top_robinson['Robinson_Risk_Score'], width,
                       label='Robinson Risk Score', color='#1f77b4', alpha=0.8)

    ax_a.set_yticks(x)
    ax_a.set_yticklabels(top_robinson['DISTRICT'], fontsize=11)
    ax_a.set_xlabel('Risk Score', fontsize=13)
    ax_a.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax_a.grid(axis='x', alpha=0.3, linestyle='--')
    ax_a.set_xlim(0, 1.05)
    ax_a.text(-0.12, 1.03, 'A', transform=ax_a.transAxes, fontsize=18, va='top', fontweight='normal')

    # =============================================================================
    # PANEL B (Right): Scatter plot with regression line
    # =============================================================================
    ax_b = axes[1]

    # Scatter plot
    ax_b.scatter(plot_data['Robinson_Risk_Score'], plot_data['RiskScore_Normalized'],
                s=60, alpha=0.6, color='#ff7f0e', label='Districts',
                marker='o', edgecolors='black', linewidth=0.5)

    # Calculate linear regression
    slope, intercept, r_value, p_value, std_err = linregress(
        plot_data['Robinson_Risk_Score'],
        plot_data['RiskScore_Normalized']
    )

    # Create regression line
    x_line = np.array([0, 1])
    y_line = slope * x_line + intercept
    ax_b.plot(x_line, y_line, 'r-', linewidth=2, label='Linear fit')

    # Add 1:1 reference line
    ax_b.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='1:1 reference')

    # Calculate and display R² and correlation
    r_squared = r_value ** 2
    corr_coeff, _ = pearsonr(plot_data['Robinson_Risk_Score'], plot_data['RiskScore_Normalized'])

    textstr = f'R² = {r_squared:.3f}\nr = {corr_coeff:.3f}\nn = {len(plot_data)}'
    ax_b.text(0.05, 0.95, textstr,
             transform=ax_b.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=0.5))

    ax_b.set_xlabel('Robinson Risk Score', fontsize=13)
    ax_b.set_ylabel('Normalized Risk Score', fontsize=13)
    ax_b.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax_b.grid(True, alpha=0.3, linestyle='--')
    ax_b.set_xlim(0, 1.05)
    ax_b.set_ylim(0, 1.05)
    ax_b.set_aspect('equal')
    ax_b.text(-0.12, 1.03, 'B', transform=ax_b.transAxes, fontsize=18, va='top', fontweight='normal')

    # =============================================================================
    # Save figure
    # =============================================================================
    fig.tight_layout(pad=2.0)

    png_path = os.path.join(output_dir, "FigS5_robinson_comparison.png")

    fig.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"Figure S5 saved:")
    print(f"  PNG: {png_path}")

    plt.show()

    return fig


def main():
    """
    Main function to generate Figure S5.
    """
    print("="*70)
    print("Figure S5: Robinson Risk Score Comparison")
    print("="*70)

    # Setup paths
    print("\nSetting up paths...")
    paths = setup_paths()

    # Load data and prepare comparison
    print("\nLoading and preparing data...")
    comparison_data = load_robinson_and_normalized_scores(
        paths['data_dir'],
        paths['stats_dir'],
        paths['output_dir']
    )

    # Create Figure S5
    fig = create_figure_s5(comparison_data, paths['output_dir'])

    print("\n" + "="*70)
    print("Figure S5 generation complete!")
    print("="*70)


if __name__ == "__main__":
    main()
