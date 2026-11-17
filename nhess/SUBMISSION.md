# NHESS submission checklist

Initial submission (review stage), per https://www.natural-hazards-and-earth-system-sciences.net/submission.html.
LaTeX sources and separate figure files are only requested after acceptance.

## Files to upload

| Item | File | Limit |
|---|---|---|
| Manuscript PDF (1-column, line numbers, fonts embedded, figures near first mention) | `nhess/nhess_manuscript.pdf` | 50 MB |
| Supplement PDF | `nhess/nhess_supplement.pdf` | 50 MB |

Build both from `nhess/`:

```bash
latexmk -pdf nhess_manuscript.tex && latexmk -pdf nhess_supplement.tex
```

## Text for the upload form

### Abstract (plain text)

Anticipating the likely impacts of a future large earthquake is essential for effective humanitarian contingency planning. Ideally, that planning should incorporate both the range of potential future earthquake locations and magnitudes, and earthquake-related secondary hazards – which in many continental settings includes coseismic landsliding. Here, we extend a previously-developed earthquake scenario ensemble approach to explicitly include coseismic landsliding using a hypergraph network model to quantify multi-hazard risk across the ensemble. We apply this model to a set of 30 plausible large-magnitude (Mw > 7) earthquakes across Nepal to evaluate the combined impacts of shaking and landsliding on buildings and road infrastructure. While shaking damage dominates the overall impact profile, our findings reveal that landslides substantially increase impacts in steep and remote regions where landslide impacts on buildings and roads can exceed 50 % of the number of buildings directly damaged by shaking. The results show a strong and statistically significant correlation between district remoteness and the proportion of road segments damaged by landslides (Spearman ρ = 0.81), which compounds the challenges of emergency response and of access to health care and government facilities. These findings are consistent with an urban-rural risk divide shaped by Nepal's diverse topography, in which densely populated urban and lowland districts face the largest absolute numbers of buildings damaged by shaking, while remote hill and mountain districts face the highest proportional impacts and the greatest likely challenges to post-earthquake access. This multi-hazard ensemble approach provides a more comprehensive understanding of earthquake risk than shaking-only assessments, offering valuable insights for contingency planning by humanitarian organisations and the Government of Nepal.

### Short summary (draft, 479/500 characters including spaces)

Plans for the next large earthquake in Nepal usually consider only building damage from shaking. We simulated 30 possible large earthquakes and added the landslides they would trigger. Shaking causes most of the damage, but in steep, remote hill and mountain districts landslides add almost as many impacts again and damage many roads, where help is already hardest to deliver. Our district-level results can help humanitarian agencies and the government decide where to prepare.

## Before submitting

- [x] Competing interests: none (standard Copernicus wording)
- [x] Fig. S5 caption
- [ ] Short summary above (drafted)
- [ ] Commit and push, so the GitHub URL in the code and data availability statement shows the final code and data
