"""
Figure 8: Multi-Hazard Risk Scores - Comprehensive 6-Panel Visualization

This script generates Figure 8, a comprehensive visualization of risk scores including:

PART 1 (2-panel maps):
- Panel A: Absolute Compounding Risk Score map
- Panel B: Normalized Risk Score map

PART 2 (4-panel comparison with Robinson et al., 2018):
- Panel A: Top 20 districts by Our Absolute Risk Score
- Panel B: Top 20 districts by Our Normalized Risk Score
- Panel C: Top 20 districts by Robinson Risk Score
- Panel D: Risk Score Correlation scatter plot

PART 3 (3-panel comparison maps):
- Panel A: Our Absolute Risk Score map
- Panel B: Our Normalized Risk Score map
- Panel C: Robinson et al. (2018) Risk Score map

The risk scores combine normalized remoteness with multi-hazard impacts using a weighted
sum approach (50% remoteness, 50% hazard impact).

Output files:
- Fig8_risk_scores.png (300 DPI) - Combined map visualization
- Fig8_risk_scores.pdf (vector format)
- Fig8_risk_scores_comparison.png (300 DPI) - Ranked comparison
- Fig8_risk_scores_comparison.pdf (vector format)
- Fig8_risk_scores_maps.png (300 DPI) - Comparative maps
- Fig8_risk_scores_maps.pdf (vector format)

Usage:
    python figure_08_risk_scores.py
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
from scipy.stats import pearsonr
import warnings
from utils import apply_science_style


# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
apply_science_style()

# Import from local modules
from data_loader import initialize_data, get_output_path, ensure_output_directory
from utils import apply_science_style, format_lon, format_lat


def prepare_risk_data(data):
    """
    Prepare risk score data by calculating normalized variables and risk scores.

    Parameters:
    -----------
    data : dict
        Data dictionary from initialize_data() containing all datasets

    Returns:
    --------
    GeoDataFrame
        Administrative boundaries with calculated risk scores
    """
    print("Preparing risk score data...")

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
    print("Calculating remoteness indices...")

    # Filter by facility type and season
    dhq = remoteness_data[
        (remoteness_data['Facility'] == 'District Headquarters') &
        (remoteness_data['season'] == 'Normal season')
    ].copy()

    health_posts = remoteness_data[
        (remoteness_data['Facility'] == 'Health posts and sub-health posts') &
        (remoteness_data['season'] == 'Normal season')
    ].copy()

    # Aggregate remoteness by district
    dhq_remoteness = dhq.groupby('DISTRICT').agg(
        DHQ_Normal=('weighted_remoteness', 'sum')
    ).reset_index()

    health_remoteness = health_posts.groupby('DISTRICT').agg(
        Health_Normal=('weighted_remoteness', 'sum')
    ).reset_index()

    # Normalize to [0, 1]
    for remote_df in [dhq_remoteness, health_remoteness]:
        for col in remote_df.columns:
            if col != 'DISTRICT':
                max_val = remote_df[col].max()
                if max_val > 0:
                    remote_df[col] = remote_df[col] / max_val

    # Merge remoteness data
    nepal_admin = nepal_admin.merge(dhq_remoteness, on='DISTRICT', how='left')
    nepal_admin = nepal_admin.merge(health_remoteness, on='DISTRICT', how='left')

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
        RoadImpact_worst=('damage_worst_case_current', 'sum')
    ).reset_index()

    # Merge all impact data
    nepal_admin['DISTRICT'] = nepal_admin['DISTRICT'].str.upper()
    nepal_admin = nepal_admin.merge(eq_by_district, on='DISTRICT', how='left')
    nepal_admin = nepal_admin.merge(bldg_by_district, on='DISTRICT', how='left')
    nepal_admin = nepal_admin.merge(roads_by_district, on='DISTRICT', how='left')

    # Calculate total worst-case impact
    nepal_admin['S_worst'] = (
        nepal_admin['EarthquakeImpact_worst'].fillna(0) +
        nepal_admin['BuildingImpact_worst'].fillna(0) +
        nepal_admin['RoadImpact_worst'].fillna(0)
    )

    # =============================================================================
    # Calculate normalized variables for risk scores
    # =============================================================================
    print("Calculating risk scores...")

    # R: Normalized remoteness index (using DHQ remoteness)
    max_remoteness = nepal_admin['DHQ_Normal'].max()
    nepal_admin['R_normalized'] = (
        nepal_admin['DHQ_Normal'] / max_remoteness
        if max_remoteness > 0 else nepal_admin['DHQ_Normal']
    )

    # S: Normalized worst-case shaking damage
    max_impact = nepal_admin['S_worst'].max()
    nepal_admin['S_normalized'] = (
        nepal_admin['S_worst'] / max_impact
        if max_impact > 0 else nepal_admin['S_worst']
    )

    # Calculate risk scores using 50/50 weighted sum
    # Absolute Compounding Risk Score: R + S (sum of normalized components)
    nepal_admin['RiskScore_Absolute'] = (
        0.5 * nepal_admin['R_normalized'] + 0.5 * nepal_admin['S_normalized']
    )

    # Normalized Risk Score: (R + S) / 2 (already normalized by definition of components)
    nepal_admin['RiskScore_Normalized'] = nepal_admin['RiskScore_Absolute'].copy()

    return nepal_admin


def load_robinson_scores(data):
    """
    Load Robinson et al. (2018) district scores and merge with our data.

    Parameters:
    -----------
    data : dict
        Data dictionary from initialize_data()

    Returns:
    --------
    GeoDataFrame
        Administrative boundaries with Robinson scores merged
    """
    print("Loading Robinson et al. (2018) district scores...")

    nepal_admin = data['admin_boundaries'].copy()
    nepal_admin['DISTRICT'] = nepal_admin['DISTRICT'].str.upper()

    # Look for Robinson scores file
    robinson_file = os.path.join(
        data['paths']['stats_dir'], "robinson_district_scores.csv"
    )

    if os.path.exists(robinson_file):
        robinson_scores = pd.read_csv(robinson_file)
        print(f"Loaded Robinson scores from {robinson_file}")

        # Standardize district names in Robinson data
        if 'DISTRICT' in robinson_scores.columns:
            robinson_scores['DISTRICT'] = robinson_scores['DISTRICT'].str.upper()
            nepal_admin = nepal_admin.merge(robinson_scores, on='DISTRICT', how='left')
        else:
            print("Warning: Robinson scores file missing DISTRICT column")
    else:
        print(f"Warning: Robinson scores file not found at {robinson_file}")
        print("Creating placeholder column for Robinson scores")
        nepal_admin['Robinson_Risk_Score'] = np.nan

    return nepal_admin


def create_figure_8_part1(nepal_admin_risk, output_dir):
    """
    Create Figure 8 Part 1: 2-panel risk score maps.

    Parameters:
    -----------
    nepal_admin_risk : GeoDataFrame
        Administrative boundaries with risk scores
    output_dir : str
        Directory to save output files
    """
    print("\nCreating Figure 8 Part 1: Multi-Hazard Risk Score Maps...")

    # Create Figure with 2 panels (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # Convert to Web Mercator for contextily
    nepal_admin_risk_mercator = nepal_admin_risk.to_crs("EPSG:3857")

    # =============================================================================
    # PANEL A (Left): Absolute Compounding Risk Score
    # =============================================================================
    ax_a = axes[0]

    Vm = 0.7  # Max value for colorbar

    nepal_admin_risk_mercator.plot(
        column='RiskScore_Absolute',
        ax=ax_a,
        cmap='rainbow',
        vmin=0,
        vmax=Vm,
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
    norm_a = colors.Normalize(vmin=0, vmax=Vm)
    sm_a = plt.cm.ScalarMappable(cmap='rainbow', norm=norm_a)
    sm_a.set_array([])
    cbar_a = plt.colorbar(sm_a, cax=cax_a)
    cbar_a.set_label('Absolute Compounding Risk Score', fontsize=14)
    cbar_a.ax.tick_params(labelsize=12)

    ax_a.xaxis.set_major_formatter(mticker.FuncFormatter(format_lon))
    ax_a.yaxis.set_major_formatter(mticker.FuncFormatter(format_lat))
    ax_a.tick_params(axis='both', labelsize=12)
    ax_a.set_aspect('equal', adjustable='box')
    ax_a.text(-0.08, 1.05, 'A', transform=ax_a.transAxes, fontsize=22, va='top')

    # =============================================================================
    # PANEL B (Right): Normalized Risk Score
    # =============================================================================
    ax_b = axes[1]

    nepal_admin_risk_mercator.plot(
        column='RiskScore_Normalized',
        ax=ax_b,
        cmap='rainbow',
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
        ctx.add_basemap(ax_b, crs=nepal_admin_risk_mercator.crs,
                       source=ctx.providers.OpenTopoMap,
                       alpha=0.5, attribution=False, zoom=8)
    except Exception as e:
        print(f"Could not add basemap to Panel B: {e}")
        ax_b.set_facecolor('lightgray')

    # Add colorbar
    divider_b = make_axes_locatable(ax_b)
    cax_b = divider_b.append_axes("right", size="3%", pad=0.1)
    norm_b = colors.Normalize(vmin=0, vmax=1)
    sm_b = plt.cm.ScalarMappable(cmap='rainbow', norm=norm_b)
    sm_b.set_array([])
    cbar_b = plt.colorbar(sm_b, cax=cax_b)
    cbar_b.set_label('Normalized Risk Score', fontsize=14)
    cbar_b.ax.tick_params(labelsize=12)

    ax_b.xaxis.set_major_formatter(mticker.FuncFormatter(format_lon))
    ax_b.yaxis.set_major_formatter(mticker.FuncFormatter(format_lat))
    ax_b.tick_params(axis='both', labelsize=12)
    ax_b.set_aspect('equal', adjustable='box')
    ax_b.text(-0.08, 1.05, 'B', transform=ax_b.transAxes, fontsize=22, va='top')

    # =============================================================================
    # Save figure
    # =============================================================================
    fig.tight_layout(pad=2.0)

    png_path = os.path.join(output_dir, "Fig8_risk_scores.png")

    fig.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"Figure 8 Part 1 saved:")
    print(f"  PNG: {png_path}")

    plt.show()

    return fig


def create_figure_8_part2_and_3(nepal_admin_comparison, output_dir):
    """
    Create Figure 8 Parts 2 and 3: Ranked comparison and comparison maps.

    This function creates both the ranked bar plot comparison and the 3-panel map comparison.

    Parameters:
    -----------
    nepal_admin_comparison : GeoDataFrame
        Administrative boundaries with all risk scores
    output_dir : str
        Directory to save output files
    """
    # =============================================================================
    # PART 2: Ranked comparison with Robinson et al.
    # =============================================================================
    print("\nCreating Figure 8 Part 2: Ranked bar plot comparison...")

    # Filter to matched districts with complete data
    plot_data = nepal_admin_comparison[
        nepal_admin_comparison['Robinson_Risk_Score'].notna() &
        nepal_admin_comparison['RiskScore_Absolute'].notna() &
        nepal_admin_comparison['RiskScore_Normalized'].notna()
    ].copy()

    if len(plot_data) == 0:
        print("Warning: No districts with complete data for comparison")
        print("Skipping Figure 8 Part 2 and Part 3")
        return

    print(f"Plotting {len(plot_data)} districts with complete data")

    # Create figure with multiple subplots
    fig_comp, axes_comp = plt.subplots(2, 2, figsize=(20, 16))

    # =============================================================================
    # PANEL A (Upper Left): Top 20 by Our Absolute Risk Score
    # =============================================================================
    ax_a = axes_comp[0, 0]

    top_n = 20
    top_abs = plot_data.nlargest(top_n, 'RiskScore_Absolute')[
        ['DISTRICT', 'RiskScore_Absolute', 'RiskScore_Normalized', 'Robinson_Risk_Score']
    ].copy()

    x = np.arange(len(top_abs))
    width = 0.25

    bars1 = ax_a.barh(x - width, top_abs['RiskScore_Absolute'], width,
                       label='Our Absolute', color='#d62728', alpha=0.8)
    bars2 = ax_a.barh(x, top_abs['RiskScore_Normalized'], width,
                       label='Our Normalized', color='#ff7f0e', alpha=0.8)
    bars3 = ax_a.barh(x + width, top_abs['Robinson_Risk_Score'], width,
                       label='Robinson', color='#1f77b4', alpha=0.8)

    ax_a.set_yticks(x)
    ax_a.set_yticklabels(top_abs['DISTRICT'], fontsize=12)
    ax_a.set_xlabel('Risk Score', fontsize=14)
    ax_a.set_title('A) Top 20 Districts by Our Absolute Risk Score',
                   fontsize=14, loc='left')
    ax_a.legend(loc='lower right', fontsize=11)
    ax_a.grid(axis='x', alpha=0.3, linestyle='--')
    ax_a.set_xlim(0, 1.0)
    ax_a.invert_yaxis()

    # =============================================================================
    # PANEL B (Upper Right): Top 20 by Our Normalized Risk Score
    # =============================================================================
    ax_b = axes_comp[0, 1]

    top_norm = plot_data.nlargest(top_n, 'RiskScore_Normalized')[
        ['DISTRICT', 'RiskScore_Absolute', 'RiskScore_Normalized', 'Robinson_Risk_Score']
    ].copy()

    x = np.arange(len(top_norm))

    bars1 = ax_b.barh(x - width, top_norm['RiskScore_Absolute'], width,
                       label='Our Absolute', color='#d62728', alpha=0.8)
    bars2 = ax_b.barh(x, top_norm['RiskScore_Normalized'], width,
                       label='Our Normalized', color='#ff7f0e', alpha=0.8)
    bars3 = ax_b.barh(x + width, top_norm['Robinson_Risk_Score'], width,
                       label='Robinson', color='#1f77b4', alpha=0.8)

    ax_b.set_yticks(x)
    ax_b.set_yticklabels(top_norm['DISTRICT'], fontsize=12)
    ax_b.set_xlabel('Risk Score', fontsize=14)
    ax_b.set_title('B) Top 20 Districts by Our Normalized Risk Score',
                   fontsize=14, loc='left')
    ax_b.legend(loc='lower right', fontsize=11)
    ax_b.grid(axis='x', alpha=0.3, linestyle='--')
    ax_b.set_xlim(0, 1.0)
    ax_b.invert_yaxis()

    # =============================================================================
    # PANEL C (Lower Left): Top 20 by Robinson Risk Score
    # =============================================================================
    ax_c = axes_comp[1, 0]

    top_rob = plot_data.nlargest(top_n, 'Robinson_Risk_Score')[
        ['DISTRICT', 'RiskScore_Absolute', 'RiskScore_Normalized', 'Robinson_Risk_Score']
    ].copy()

    x = np.arange(len(top_rob))

    bars1 = ax_c.barh(x - width, top_rob['RiskScore_Absolute'], width,
                       label='Our Absolute', color='#d62728', alpha=0.8)
    bars2 = ax_c.barh(x, top_rob['RiskScore_Normalized'], width,
                       label='Our Normalized', color='#ff7f0e', alpha=0.8)
    bars3 = ax_c.barh(x + width, top_rob['Robinson_Risk_Score'], width,
                       label='Robinson', color='#1f77b4', alpha=0.8)

    ax_c.set_yticks(x)
    ax_c.set_yticklabels(top_rob['DISTRICT'], fontsize=12)
    ax_c.set_xlabel('Risk Score', fontsize=14)
    ax_c.set_title('C) Top 20 Districts by Robinson Risk Score',
                   fontsize=14, loc='left')
    ax_c.legend(loc='lower right', fontsize=11)
    ax_c.grid(axis='x', alpha=0.3, linestyle='--')
    ax_c.set_xlim(0, 1.0)
    ax_c.invert_yaxis()

    # =============================================================================
    # PANEL D (Lower Right): Scatter plot comparison
    # =============================================================================
    ax_d = axes_comp[1, 1]

    ax_d.scatter(plot_data['Robinson_Risk_Score'], plot_data['RiskScore_Normalized'],
                s=60, alpha=0.6, color='#ff7f0e', label='Our Normalized vs Robinson',
                marker='s', edgecolors='black', linewidth=0.5)

    # Calculate and show correlations
    corr_abs, _ = pearsonr(plot_data['RiskScore_Absolute'], plot_data['Robinson_Risk_Score'])
    corr_norm, _ = pearsonr(plot_data['RiskScore_Normalized'], plot_data['Robinson_Risk_Score'])

    ax_d.text(0.05, 0.95, f'r(Absolute) = {corr_abs:.3f}\nr(Normalized) = {corr_norm:.3f}',
             transform=ax_d.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax_d.set_xlabel('Robinson Risk Score', fontsize=14)
    ax_d.set_ylabel('Our Risk Scores', fontsize=14)
    ax_d.set_title('D) Risk Score Correlation', fontsize=14, loc='left')
    ax_d.legend(loc='lower right', fontsize=11)
    ax_d.grid(True, alpha=0.3, linestyle='--')
    ax_d.set_xlim(0, 1.05)
    ax_d.set_ylim(0, 1.05)
    ax_d.set_aspect('equal')

    # Save figure
    fig_comp.tight_layout()

    png_path_comp = os.path.join(output_dir, "Fig8_risk_scores_comparison.png")
    pdf_path_comp = os.path.join(output_dir, "Fig8_risk_scores_comparison.pdf")

    fig_comp.savefig(png_path_comp, dpi=300, bbox_inches='tight')

    print(f"Figure 8 Part 2 saved:")
    print(f"  PNG: {png_path_comp}")

    plt.show()

    # =============================================================================
    # PART 3: Comparison maps
    # =============================================================================
    print("\nCreating Figure 8 Part 3: Comparison maps...")

    fig_maps, axes_maps = plt.subplots(1, 3, figsize=(24, 8))

    # Filter to matched districts only for visualization
    nepal_comparison_matched = nepal_admin_comparison[
        nepal_admin_comparison['Robinson_Risk_Score'].notna()
    ].copy()
    print(f"Visualizing {len(nepal_comparison_matched)} matched districts")

    # Convert to Web Mercator for contextily basemap
    nepal_comparison_mercator = nepal_comparison_matched.to_crs("EPSG:3857")

    # =============================================================================
    # PANEL A: Our Absolute Risk Score
    # =============================================================================
    ax_a = axes_maps[0]

    Vm_abs = 0.7

    nepal_comparison_mercator.plot(
        column='RiskScore_Absolute',
        ax=ax_a,
        cmap='rainbow',
        vmin=0,
        vmax=Vm_abs,
        edgecolor='gray',
        linewidth=0.3,
        alpha=0.7,
        legend=False
    )

    try:
        import contextily as ctx
        ctx.add_basemap(ax_a, crs=nepal_comparison_mercator.crs,
                       source=ctx.providers.OpenTopoMap,
                       alpha=0.4, attribution=False, zoom=8)
    except Exception as e:
        print(f"Could not add basemap: {e}")
        ax_a.set_facecolor('lightgray')

    divider_a = make_axes_locatable(ax_a)
    cax_a = divider_a.append_axes("right", size="3%", pad=0.1)
    norm_a = colors.Normalize(vmin=0, vmax=Vm_abs)
    sm_a = plt.cm.ScalarMappable(cmap='rainbow', norm=norm_a)
    sm_a.set_array([])
    cbar_a = plt.colorbar(sm_a, cax=cax_a)
    cbar_a.set_label('Our Absolute Risk Score', fontsize=14)
    cbar_a.ax.tick_params(labelsize=10)

    ax_a.xaxis.set_major_formatter(ticker.FuncFormatter(format_lon))
    ax_a.yaxis.set_major_formatter(ticker.FuncFormatter(format_lat))
    ax_a.tick_params(axis='both', labelsize=10)
    ax_a.set_aspect('equal', adjustable='box')
    ax_a.text(-0.08, 1.05, 'A', transform=ax_a.transAxes, fontsize=20, va='top')

    # =============================================================================
    # PANEL B: Our Normalized Risk Score
    # =============================================================================
    ax_b = axes_maps[1]

    nepal_comparison_mercator.plot(
        column='RiskScore_Normalized',
        ax=ax_b,
        cmap='rainbow',
        vmin=0,
        vmax=1,
        edgecolor='gray',
        linewidth=0.3,
        alpha=0.7,
        legend=False
    )

    try:
        ctx.add_basemap(ax_b, crs=nepal_comparison_mercator.crs,
                       source=ctx.providers.OpenTopoMap,
                       alpha=0.4, attribution=False, zoom=8)
    except Exception as e:
        ax_b.set_facecolor('lightgray')

    divider_b = make_axes_locatable(ax_b)
    cax_b = divider_b.append_axes("right", size="3%", pad=0.1)
    norm_b = colors.Normalize(vmin=0, vmax=1)
    sm_b = plt.cm.ScalarMappable(cmap='rainbow', norm=norm_b)
    sm_b.set_array([])
    cbar_b = plt.colorbar(sm_b, cax=cax_b)
    cbar_b.set_label('Our Normalized Risk Score', fontsize=14)
    cbar_b.ax.tick_params(labelsize=10)

    ax_b.xaxis.set_major_formatter(ticker.FuncFormatter(format_lon))
    ax_b.yaxis.set_major_formatter(ticker.FuncFormatter(format_lat))
    ax_b.tick_params(axis='both', labelsize=10)
    ax_b.set_aspect('equal', adjustable='box')
    ax_b.text(-0.08, 1.05, 'B', transform=ax_b.transAxes, fontsize=20, va='top')

    # =============================================================================
    # PANEL C: Robinson Risk Score
    # =============================================================================
    ax_c = axes_maps[2]

    nepal_comparison_mercator.plot(
        column='Robinson_Risk_Score',
        ax=ax_c,
        cmap='rainbow',
        vmin=0,
        vmax=1,
        edgecolor='gray',
        linewidth=0.3,
        alpha=0.7,
        legend=False
    )

    try:
        ctx.add_basemap(ax_c, crs=nepal_comparison_mercator.crs,
                       source=ctx.providers.OpenTopoMap,
                       alpha=0.4, attribution=False, zoom=8)
    except Exception as e:
        ax_c.set_facecolor('lightgray')

    divider_c = make_axes_locatable(ax_c)
    cax_c = divider_c.append_axes("right", size="3%", pad=0.1)
    norm_c = colors.Normalize(vmin=0, vmax=1)
    sm_c = plt.cm.ScalarMappable(cmap='rainbow', norm=norm_c)
    sm_c.set_array([])
    cbar_c = plt.colorbar(sm_c, cax=cax_c)
    cbar_c.set_label('Robinson et al. (2018) Risk Score', fontsize=14)
    cbar_c.ax.tick_params(labelsize=10)

    ax_c.xaxis.set_major_formatter(ticker.FuncFormatter(format_lon))
    ax_c.yaxis.set_major_formatter(ticker.FuncFormatter(format_lat))
    ax_c.tick_params(axis='both', labelsize=10)
    ax_c.set_aspect('equal', adjustable='box')
    ax_c.text(-0.08, 1.05, 'C', transform=ax_c.transAxes, fontsize=20, va='top')

    # Save figure
    fig_maps.tight_layout()

    png_path_maps = os.path.join(output_dir, "Fig8_risk_scores_maps.png")
    pdf_path_maps = os.path.join(output_dir, "Fig8_risk_scores_maps.pdf")

    fig_maps.savefig(png_path_maps, dpi=300, bbox_inches='tight')

    print(f"Figure 8 Part 3 saved:")
    print(f"  PNG: {png_path_maps}")

    plt.show()


def main():
    """
    Main function to generate Figure 8 (all parts).
    """
    print("="*70)
    print("Figure 8: Multi-Hazard Risk Scores")
    print("="*70)

    # Initialize data
    print("\nInitializing data...")
    data = initialize_data()
    ensure_output_directory(data)

    # Prepare risk score data
    nepal_admin_risk = prepare_risk_data(data)

    # Create Figure 8 Part 1: Risk score maps
    fig_part1 = create_figure_8_part1(nepal_admin_risk, data['paths']['output_dir'])

    # Attempt to load Robinson data and create Parts 2 and 3
    print("\nAttempting to load Robinson et al. (2018) data for comparison...")
    nepal_admin_comparison = load_robinson_scores(data)

    # Merge risk scores with Robinson data
    nepal_admin_comparison['DISTRICT'] = nepal_admin_comparison['DISTRICT'].str.upper()
    nepal_admin_risk['DISTRICT'] = nepal_admin_risk['DISTRICT'].str.upper()

    for col in ['RiskScore_Absolute', 'RiskScore_Normalized']:
        nepal_admin_comparison = nepal_admin_comparison.merge(
            nepal_admin_risk[['DISTRICT', col]],
            on='DISTRICT',
            how='left'
        )

    # Create Parts 2 and 3 if Robinson data is available
    if nepal_admin_comparison['Robinson_Risk_Score'].notna().sum() > 0:
        create_figure_8_part2_and_3(nepal_admin_comparison, data['paths']['output_dir'])
    else:
        print("Warning: Robinson et al. (2018) data not available - skipping Parts 2 and 3")
        print("Only Figure 8 Part 1 (risk score maps) has been generated")

    print("\n" + "="*70)
    print("Figure 8 generation complete!")
    print("="*70)


if __name__ == "__main__":
    main()
