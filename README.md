# Multi-hazard scenario ensembles for estimating earthquake risk

Code, figures and manuscript for Dunant et al., *Multi-hazard scenario ensembles for estimating
earthquake risk* (submitted to *Natural Hazards and Earth System Sciences*). The study applies the
hypergraph multi-hazard impact model of Dunant et al. (2025) to the 30-earthquake scenario ensemble of
Robinson et al. (2018) to estimate shaking and coseismic-landslide impacts on buildings and roads in Nepal.

## Repository layout

```
manuscript.tex              working manuscript (elsarticle; formerly main_updt.tex)
latexmkrc                   lets Overleaf/latexmk find files in nhess/ when building from the root
nhess/                      NHESS submission (Copernicus LaTeX package 7.16)
  nhess_manuscript.tex        main paper, document class [nhess, manuscript]{copernicus}
  nhess_supplement.tex        supplement (Figs. S1-S5, Tables S1-S2)
  references.bib              bibliography (the only copy; also used by manuscript.tex)
  figures/                    figures renamed fig01.png ... fig09.png, figS1.png ... figS5.png
  copernicus.cls/.cfg/.bst, pdfscreen*.sty
FIGURES/
  revised_v2/               Figs. 1-9, S4, S5, Tables S1-S2 (CSV), manuscript_numbers.json
  FigS1-S4_*.png            supplement figures
scripts/
  make_figures.py           Figs. 1-9, S4, S5, Tables S1-S2 and all numbers quoted in the text
  figure_S2_pga_ensemble_maps.py
  figure_S3_median_collapse.py
  utils.py
model/                      hypergraph multi-hazard impact model (ensemble simulation code)
data/                       input data (not tracked; see below)
requirements.txt
```

## Reproducing the figures

```bash
pip install -r requirements.txt
python scripts/make_figures.py                 # set SAVE_EPS=0 to skip the large EPS copies
python scripts/figure_S2_pga_ensemble_maps.py
python scripts/figure_S3_median_collapse.py
```

All paths are relative to the repository root. `make_figures.py` also writes
`FIGURES/revised_v2/manuscript_numbers.json`, which holds every number quoted in the text.
Fig. S1 (fragility functions) was produced separately and has no script in this repository.

## Building the NHESS submission

On Overleaf (GitHub sync), select `nhess/nhess_manuscript.tex` or `nhess/nhess_supplement.tex` as the main
document. The root `latexmkrc` adds `nhess/` to the TeX, BibTeX and graphics search paths, so the documents
compile whether the build runs from the project root or from `nhess/`. Locally:

```bash
cd nhess && latexmk -pdf nhess_manuscript.tex && latexmk -pdf nhess_supplement.tex
```

`copernicus.cls` needs the `fontawesome5`, `fontawesome` and `vntex` TeX packages (included in full TeX Live
and on Overleaf). `nhess_manuscript.tex` is derived from `manuscript.tex`; text changes made in one must be
copied to the other.

The figures in `nhess/figures/` are copies, under the names Copernicus expects, of the script outputs in
`FIGURES/`. After regenerating figures, refresh the copies from the repository root:

```bash
F=FIGURES; V=FIGURES/revised_v2; N=nhess/figures
cp $V/Fig1_study_area_map.png $N/fig01.png;              cp $V/Fig2_exceedance_probability.png $N/fig02.png
cp $V/Fig3_district_impacts_ratio.png $N/fig03.png;      cp $V/Fig4_district_exceedance_curves.png $N/fig04.png
cp $V/Fig5_worst_case_absolute_proportional.png $N/fig05.png; cp $V/Fig6_standard_deviation.png $N/fig06.png
cp $V/Fig7_remoteness_analysis.png $N/fig07.png;         cp $V/Fig8_risk_scores.png $N/fig08.png
cp $V/Fig9_robinson_comparison.png $N/fig09.png
cp $F/FigS1_fragility_functions.png $N/figS1.png;        cp $F/FigS2_pga_ensemble_maps.png $N/figS2.png
cp $F/FigS3_median_collapse_probability.png $N/figS3.png; cp $V/FigS4_remoteness_clusters.png $N/figS4.png
cp $V/FigS5_district_names.png $N/figS5.png
```

## Data

`data/` is not tracked (git-ignored). The figure scripts expect:

```
data/
  aggregated_stats/
    stats_eqimpact_2024-06-24_physiog.csv          district stats, buildings damaged by shaking (30 scenarios)
    stats_buildings_lsimpact_2024-06-24_physiog.csv district stats, buildings damaged by landslides
    stats_roads_lsimpact_2024-06-24_physiog.csv     district stats, road segments damaged by landslides
    district_asset_counts.csv                        buildings and 100 m road segments per district
  shp/
    hermes_NPL_new_wgs/hermes_NPL_new_wgs_2.*      district boundaries (77 districts)
    Ward/NP_Ward_P21_Wgs84.*                         ward boundaries
    faults/Nepal_Faults_DMG.*                        fault traces, Department of Mines and Geology, Nepal
    faults/HimaTibetMap.geojson                      regional fault traces (Fig. 1)
  Remotenessdata/
    Remoteness_DFID_Data.csv                         travel-time classes by municipality (Banick and Kawasoe, 2019)
    Remoteness_DFID_Municipalities.csv
  Robinson2018/robinson_district_scores.csv        district risk scores of Robinson et al. (2018)
  tif/robinson_ensemble_expanded/                  PGA rasters of the 30 scenario earthquakes (UTM 45N)
```

These ~220 MB are a subset of the project data store. The raw per-asset model outputs (~76 GB)
from which `aggregated_stats/` was derived are kept outside this repository.

## Model

`model/` contains the ensemble simulation code. The entry point is `mcpu_ensemble_impacts_full30_withCutoff.py`;
it reads from `./data/shp` and `./data/tif/robinson_ensemble_expanded` and writes to
`./data/results/robinson_ensemble/`. Its other inputs (building and road databases, slope units `su_updt.shp`,
susceptibility statistics `*_fR_stats*.csv`) are not part of the `data/` subset above. One diagnostic plot in
`mods_vulnerability_caseload_ensemble.py` still saves to a path from the original run (`/Volumes/prs/...`).
See Dunant et al. (2025) for a full description of the model.
