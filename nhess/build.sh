#!/usr/bin/env bash
# Build the NHESS submission (manuscript + supplement).
#
# The canonical sources are ../references.bib and the figures in ../FIGURES/
# (written by ../scripts/*.py). This script copies them into nhess/ under the
# file names Copernicus expects (fig01.png, ...), then compiles both documents.
# Usage: bash nhess/build.sh   (from anywhere)
set -euo pipefail
cd "$(dirname "$0")"

cp ../references.bib references.bib

# Main-text figures, in order of first citation
cp ../FIGURES/revised_v2/Fig1_study_area_map.png                figures/fig01.png
cp ../FIGURES/revised_v2/Fig2_exceedance_probability.png        figures/fig02.png
cp ../FIGURES/revised_v2/Fig3_district_impacts_ratio.png        figures/fig03.png
cp ../FIGURES/revised_v2/Fig4_district_exceedance_curves.png    figures/fig04.png
cp ../FIGURES/revised_v2/Fig5_worst_case_absolute_proportional.png figures/fig05.png
cp ../FIGURES/revised_v2/Fig6_standard_deviation.png            figures/fig06.png
cp ../FIGURES/revised_v2/Fig7_remoteness_analysis.png           figures/fig07.png
cp ../FIGURES/revised_v2/Fig8_risk_scores.png                   figures/fig08.png
cp ../FIGURES/revised_v2/Fig9_robinson_comparison.png           figures/fig09.png

# Supplement figures
cp ../FIGURES/FigS1_fragility_functions.png                     figures/figS1.png
cp ../FIGURES/FigS2_pga_ensemble_maps.png                       figures/figS2.png
cp ../FIGURES/FigS3_median_collapse_probability.png             figures/figS3.png
cp ../FIGURES/revised_v2/FigS4_remoteness_clusters.png          figures/figS4.png
cp ../FIGURES/revised_v2/FigS5_district_names.png               figures/figS5.png

latexmk -pdf -interaction=nonstopmode nhess_manuscript.tex
latexmk -pdf -interaction=nonstopmode nhess_supplement.tex
