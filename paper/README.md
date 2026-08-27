# Seqcore manuscript

A bioRxiv-format preprint introducing Seqcore: its array representation, the
vectorization techniques built on it, and a benchmark comparison against
Biopython and pure Python.

## Files

- `seqcore.tex` — manuscript source
- `seqcore.pdf` — compiled output
- `make_figures.py` — generates every figure **and** `table1.tex` from the
  committed benchmark JSON in `../benchmarks/results/`
- `figures/` — generated PDF (for LaTeX) and PNG (for preview)
- `table1.tex` — generated table body, `\input` by the manuscript

## Building

Figures and the table must be generated first, since the manuscript `\input`s
`table1.tex`:

```bash
python paper/make_figures.py
tectonic paper/seqcore.tex
```

`pdflatex paper/seqcore.tex` (run twice, for cross-references) works equally well.

## Reproducing the numbers

No timing in the manuscript is transcribed by hand. `make_figures.py` generates
the figures, `table1.tex`, and `numbers.tex` — a set of `\newcommand` macros
covering every value the prose quotes, **including the platform sentence**. The
manuscript references those macros, so the text cannot drift from the data.

To regenerate from scratch:

```bash
python benchmarks/benchmark_suite.py     # on each machine you want to report
python paper/make_figures.py
```

## Multiple machines

`make_figures.py` picks up the newest result per distinct machine and reports
them together. One machine supplies the headline numbers (scaling figures,
Table 1, and the values quoted in the text); the others appear in the
cross-platform panel of Figure 4b.

The headline machine is chosen by `PRIMARY_MACHINE` near the top of
`make_figures.py` — a substring of the machine label, currently `"Mac mini"`.
Change that one string to switch which machine the manuscript quotes; the
platform sentence, Table 1 and every number update together.

Machine labels come from the benchmark metadata (`hardware_model`, `cpu_model`),
so an EC2 run self-identifies as e.g. `AWS g5.2xlarge (AMD EPYC 7R32)` and a Mac
as `Mac mini (M4 Pro)`.

## GPU figure

Figure 5 is generated only if a `gpu_*.json` result from
`benchmarks/benchmark_gpu.py` is present; without one it is skipped and the rest
of the figure set still builds.

## Figure style

Figures follow Nature Portfolio conventions: Helvetica/Arial throughout,
lowercase bold 8 pt panel labels, 5–7 pt body text, thin axes, and the
NPG-inspired categorical palette (Seqcore blue `#4DBBD5`, Biopython red
`#E64B35`, pure Python teal `#00A087`, charcoal `#424242` for axes and text).
Colours are consistent across every panel. Output is vector PDF with text
retained as glyphs (`pdf.fonttype = 42`), sized to Nature's single-column
(89 mm) and double-column (183 mm) widths.
