# Lets the NHESS documents in nhess/ compile when the build runs from the project root
# (as on Overleaf): TeX, BibTeX and graphics lookups also search nhess/ for
# copernicus.cls/.cfg/.bst, pdfscreen*.sty, figures/ and references.bib.
ensure_path('TEXINPUTS', './nhess//');
ensure_path('BIBINPUTS', './nhess//');
ensure_path('BSTINPUTS', './nhess//');
