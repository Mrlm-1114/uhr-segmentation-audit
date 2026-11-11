# Figure sources

Figure 1: run `python gen_teaser.py` in this directory, then
`pdflatex fig_teaser.tex`. This is the current manuscript's figure generator.

Figure 2: the current source is `figure2/fig_framework.tex`. Supply authorized
source crops to `figure2/prep_assets.py`, whose positional arguments are the
external thumbnail folder, external qualitative folder, and output assets folder.
The expected filenames are documented in that script. Compile the TeX file from
its own directory. Dataset pixels are omitted from this repository.

Figure 3: run `python figures/figure3.py --output ../generated_figures` from the
repository root. The script reads `records/`, checks arithmetic and creates PDF,
SVG and PNG outputs using the manuscript's layout and plotting functions.

Figure 4: compile `fig_qual_col.tex` with privately supplied `qual_src/` assets
named `dg_rgb.png`, `dg_gt.png`, `dg_b5.png`, `dg_isdoff.png` and the corresponding
`in_` files. No images or label maps are included. TeX Live/MiKTeX with TikZ,
standalone and Helvetica packages is required for Figures 1, 2 and 4.
