"""
Utility functions for consolidated figures analysis.

This module contains reusable helper functions for creating maps, calculating
exceedance probabilities, formatting coordinates, and analyzing remoteness metrics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.ticker as mticker
from matplotlib import colors as mcolors
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch


def apply_science_style():
    """
    Apply SciencePlots style for publication-quality figures without LaTeX.
    """
    import scienceplots
    plt.style.use(['science', 'no-latex'])


# =============================================================================
# Coordinate Formatting Functions
# =============================================================================

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


# =============================================================================
# Map Creation Functions
# =============================================================================

def create_map(data, column, ax, title, cmap, transform='log', legend_title='Impacts',
              highlight_nodata=False, nodata_condition=None, vmin=None, vmax=None, add_basemap=True):
    """
    Create a choropleth map with contextily basemap and properly positioned colorbar.

    This function plots GeoDataFrame data with custom normalization (logarithmic or linear),
    adds a contextily basemap, and includes a formatted colorbar. Supports Web Mercator
    projection with latitude/longitude axis labels.

    Parameters:
    -----------
    data : GeoDataFrame
        The geographic data to plot
    column : str
        The column name to use for coloring the choropleth
    ax : matplotlib.axes.Axes
        The axes to plot on
    title : str
        The title for the map
    cmap : str
        The colormap to use (e.g., 'viridis', 'rainbow', 'Blues')
    transform : str, optional
        Scale transformation: 'log' for logarithmic scale, 'identity' for linear scale
        Default is 'log'
    legend_title : str, optional
        Title for the colorbar. Default is 'Impacts'
    highlight_nodata : bool, optional
        Whether to highlight missing data. Default is False
    nodata_condition : callable, optional
        A function to determine which values are considered missing data
    vmin, vmax : float or None, optional
        Manual color scale limits. If None, calculated from data
    add_basemap : bool, optional
        Whether to add OpenTopoMap basemap via contextily. Default is True

    Returns:
    --------
    matplotlib.axes.Axes
        The modified axes object with the map

    Notes:
    ------
    - Data is automatically converted to Web Mercator (EPSG:3857) if basemap is added
    - Missing values are shown in light gray with cross-hatching pattern
    - For log scale, only positive values are used for normalization
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

    # Format colorbar ticks based on data type
    if transform == 'log':
        cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    elif 'Percent' in legend_title or '%' in legend_title:
        cbar.ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
    else:
        cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.2f}'))

    # Add coordinate labels
    if add_basemap:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_lon))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_lat))
        ax.tick_params(axis='both', labelsize=10)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_title(title, fontsize=14, pad=10)
    ax.set_aspect('equal', adjustable='box')

    return ax


# =============================================================================
# Exceedance Analysis Functions
# =============================================================================

def calculate_exceedance(data, impact_type):
    """
    Calculate exceedance probability for a given impact type.

    Exceedance probability is the probability that a given impact value is exceeded,
    calculated as the rank divided by the total count. This is useful for
    understanding the frequency distribution of impacts.

    Parameters:
    -----------
    data : DataFrame
        Input dataframe containing impact data
    impact_type : str
        The column name of the impact type to analyze
        (e.g., 'Earthquake', 'BuildingLandslide', 'RoadLandslide')

    Returns:
    --------
    DataFrame
        DataFrame with columns:
        - 'Impact': Sorted impact values (descending)
        - 'Probability': Exceedance probability [0, 1]
        - 'HazardType': The impact type used

    Example:
    --------
    >>> df = pd.DataFrame({'Earthquake': [100, 500, 200, 1000]})
    >>> result = calculate_exceedance(df, 'Earthquake')
    >>> print(result)
       Impact  Probability  HazardType
    0    1000          0.25  Earthquake
    1     500          0.50  Earthquake
    2     200          0.75  Earthquake
    3     100          1.00  Earthquake
    """
    sorted_impacts = np.sort(data[impact_type].values)[::-1]
    exceedance_prob = np.arange(1, len(sorted_impacts) + 1) / len(sorted_impacts)
    return pd.DataFrame({
        'Impact': sorted_impacts,
        'Probability': exceedance_prob,
        'HazardType': impact_type
    })


def plot_exceedance_by_physiography(data, impact_type, ax, label, add_inset=False,
                                   district_physiography=None, nepal_admin=None):
    """
    Plot exceedance probability curves colored by physiography.

    Creates a log-scale plot of exceedance probabilities with each district's curve
    colored according to its physiographic region (High Mountain, Hill, Middle Mountain,
    Siwalik, or Terai). Optionally includes an inset map showing physiography.

    Parameters:
    -----------
    data : DataFrame
        Exceedance data with columns: 'District', 'ImpactType', 'Count', 'ExceedanceProb',
        'Physiography'
    impact_type : str
        The impact type to plot ('Earthquake', 'BuildingLandslide', or 'RoadLandslide')
    ax : matplotlib.axes.Axes
        The axes to plot on
    label : str
        Panel label (e.g., 'A', 'B', 'C') for multi-panel figures
    add_inset : bool, optional
        Whether to add an inset map showing physiography. Default is False
    district_physiography : dict, optional
        Mapping of district names to physiography. Required if add_inset=True
    nepal_admin : GeoDataFrame, optional
        Administrative boundaries for Nepal. Required if add_inset=True

    Returns:
    --------
    matplotlib.axes.Axes
        The modified axes object

    Notes:
    ------
    - X-axis is logarithmic scale (1 to 150,000 impacts)
    - Y-axis shows probability [0, 1]
    - Color scheme is standardized across all plots
    - Missing districts have physiography 'Unknown' (gray)
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

    # Plot exceedance curves for each district, colored by physiography
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

    # Set axis properties
    ax.set_xscale('log')
    ax.set_xlim(1, 150000)
    ax.set_ylim(0, 1)

    # Set labels based on impact type
    if impact_type == 'Earthquake':
        xlabel_text = 'Number of Affected Buildings by Shaking'
    elif impact_type == 'BuildingLandslide':
        xlabel_text = 'Number of Affected Buildings by Landslides'
    elif impact_type == 'RoadLandslide':
        xlabel_text = 'Number of Affected Roads by Landslides'
    else:
        xlabel_text = 'Number of Impacts'

    ax.set_xlabel(xlabel_text, fontsize=14, fontweight='bold')
    ax.set_ylabel('Exceedance Probability', fontsize=14, fontweight='bold')
    ax.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.axhline(y=0.5, linestyle='--', color='gray', alpha=0.7)

    # Add panel label
    ax.text(
        -0.1, 1.1, label,
        transform=ax.transAxes, fontsize=20, fontweight='bold', va='top'
    )

    # Create legend
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
    if add_inset and district_physiography is not None and nepal_admin is not None:
        inset_ax = ax.inset_axes([0.4, 0.6, 0.5, 0.35])

        nepal_admin_with_physio = nepal_admin.merge(
            pd.DataFrame(list(district_physiography.items()), columns=['DISTRICT', 'Physiography']),
            on='DISTRICT', how='left'
        )
        # Ensure 'Terai' is correctly mapped
        nepal_admin_with_physio['Physiography'] = nepal_admin_with_physio['Physiography'].replace('Tarai', 'Terai')
        nepal_admin_with_physio['Physiography'] = nepal_admin_with_physio['Physiography'].fillna('Unknown')

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


# =============================================================================
# Remoteness Analysis Functions
# =============================================================================

def calculate_facility_remoteness(data, season='Normal season'):
    """
    Calculate normalized remoteness index for facilities by district.

    Computes a remoteness index based on weighted travel time categories.
    The index is normalized to [0, 1] range where 1 represents maximum remoteness.

    Parameters:
    -----------
    data : DataFrame
        Filtered remoteness data with columns:
        - 'DISTRICT': District name (uppercase)
        - 'season': Season type ('Normal season' or 'Monsoon season')
        - 'weighted_remoteness': Pre-calculated weighted remoteness values
    season : str, optional
        The season to analyze ('Normal season' or 'Monsoon season').
        Default is 'Normal season'

    Returns:
    --------
    DataFrame
        Index (DISTRICT) to RemoteIndex mapping
        - RemoteIndex: Normalized remoteness value [0, 1]

    Notes:
    ------
    - Values close to 0 indicate low remoteness (good accessibility)
    - Values close to 1 indicate high remoteness (poor accessibility)
    - Districts with missing data will not appear in output

    Example:
    --------
    >>> remoteness = calculate_facility_remoteness(data, 'Normal season')
    >>> print(remoteness.loc['KATHMANDU'])
    RemoteIndex    0.15
    """
    season_data = data[data['season'] == season].copy()
    district_remoteness = season_data.groupby('DISTRICT').agg(
        RemoteIndex=('weighted_remoteness', 'sum')
    )
    max_remote = district_remoteness['RemoteIndex'].max()
    if max_remote > 0:
        district_remoteness['RemoteIndex'] = district_remoteness['RemoteIndex'] / max_remote
    return district_remoteness
