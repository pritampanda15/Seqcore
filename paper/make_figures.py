#!/usr/bin/env python3
"""Generate the Seqcore manuscript figures and Table 1.

Everything is derived from the committed benchmark JSON under
``benchmarks/results/``; no timing is hard-coded here or in the manuscript.

Styling follows Nature Portfolio figure conventions: Helvetica/Arial
throughout, lowercase bold 8 pt panel labels, 5-7 pt body text, thin axes, and
the NPG-inspired categorical palette. Output is vector PDF (for LaTeX) plus PNG
(for quick preview).

Regenerate with::

    python paper/make_figures.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
FIGDIR = HERE / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# --- NPG-inspired categorical palette -------------------------------------
NPG_BLUE = "#4DBBD5"  # Seqcore
NPG_RED = "#E64B35"  # Biopython
NPG_TEAL = "#00A087"  # pure Python
NPG_NAVY = "#3C5488"  # schematic accent
NPG_ORANGE = "#F39B7F"  # schematic auxiliary
CHARCOAL = "#424242"  # text, axes, reference lines

SERIES = [
    ("seqcore", "Seqcore", NPG_BLUE, "o"),
    ("biopython", "Biopython", NPG_RED, "s"),
    ("python", "Pure Python", NPG_TEAL, "^"),
]

# Nature: single column 89 mm, double column 183 mm.
MM = 1 / 25.4
COL1, COL2 = 89 * MM, 183 * MM

mpl.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "pdf.fonttype": 42,  # keep text as vector glyphs, not outlines
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 6,
        "text.color": CHARCOAL,
        "axes.labelsize": 7,
        "axes.labelcolor": CHARCOAL,
        "axes.edgecolor": CHARCOAL,
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "xtick.color": CHARCOAL,
        "ytick.color": CHARCOAL,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "legend.fontsize": 6,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.3,
        "lines.linewidth": 1.1,
        "lines.markersize": 2.8,
        "lines.markeredgewidth": 0,
        # Math is rendered in the same family as body text, per Nature style.
        "mathtext.fontset": "custom",
        "mathtext.rm": "Helvetica",
        "mathtext.it": "Helvetica:italic",
        "mathtext.bf": "Helvetica:bold",
        "mathtext.default": "it",
    }
)


# Which machine supplies the headline numbers (scaling figures, Table 1 and the
# values quoted in the manuscript). The other measured machines appear in the
# cross-platform panel. Set to a substring of the desired machine label.
PRIMARY_MACHINE = "Mac mini"


def machine_label(d: dict) -> str:
    """Short human-readable name for the machine a result was measured on."""
    cpu = (d.get("cpu_model") or d.get("processor") or "unknown").replace("Apple ", "")
    hw = d.get("hardware_model") or ""
    if d.get("system") == "Darwin":
        base = hw.split(" (")[0] or "Mac"
        return f"{base} ({cpu})"
    if hw and hw[0].isalpha() and "." in hw:  # looks like an EC2 instance type
        return f"AWS {hw} ({cpu})"
    return f"{hw} ({cpu})" if hw else cpu


def load_suites() -> dict:
    """Newest suite result per distinct machine, keyed by machine label.

    Only files carrying the self-describing metadata are considered, so older
    results from before that field existed are ignored rather than mixed in.
    """
    suites = {}
    for f in sorted(glob.glob(str(ROOT / "benchmarks" / "results" / "suite_*.json"))):
        d = json.loads(Path(f).read_text())
        if not d.get("library_versions"):
            continue
        suites[machine_label(d)] = d
    if not suites:
        raise SystemExit(
            "No benchmark results with platform metadata. "
            "Run: python benchmarks/benchmark_suite.py"
        )
    return suites


def primary_suite(suites: dict) -> tuple:
    """The machine whose numbers the manuscript quotes."""
    for label, d in suites.items():
        if PRIMARY_MACHINE.lower() in label.lower():
            return label, d
    label = sorted(suites)[0]
    print(f"  (warning: PRIMARY_MACHINE {PRIMARY_MACHINE!r} not found; using {label})")
    return label, suites[label]


def newest(pattern: str) -> dict:
    """Load the most recent benchmark result matching a glob pattern."""
    files = sorted(glob.glob(str(ROOT / "benchmarks" / "results" / pattern)))
    if not files:
        raise SystemExit(
            f"No benchmark results matching {pattern}. " "Run: python benchmarks/benchmark_suite.py"
        )
    return json.loads(Path(files[-1]).read_text())


def panel_label(ax, letter: str, dx: float = -0.20, dy: float = 1.06) -> None:
    """Place a lowercase bold 8 pt panel label at a consistent offset."""
    ax.text(
        dx,
        dy,
        letter,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="left",
        color=CHARCOAL,
    )


def clean(ax, grid_axis: str | None = "both") -> None:
    """Apply the shared quiet-axis treatment."""
    if grid_axis:
        ax.grid(True, axis=grid_axis, which="major", ls=":", lw=0.35, color="#D8D8D8", zorder=0)
    ax.set_axisbelow(True)


def arr(values) -> np.ndarray:
    """Series to float array with None mapped to NaN so gaps are not drawn."""
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def save(fig, name: str) -> None:
    """Write vector PDF for LaTeX and PNG for preview."""
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  figures/{name}.pdf")


# =========================================================== Figure 1: design
def figure_design() -> None:
    """Schematic of the array layout and the wavefront recurrence."""
    fig = plt.figure(figsize=(COL2, 2.55))
    ax1 = fig.add_axes([0.015, 0.04, 0.47, 0.88])
    ax2 = fig.add_axes([0.525, 0.04, 0.47, 0.88])
    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.axis("off")

    # --- a: batched ordinal encoding, left to right -----------------------
    seqs = ["ACGT", "GGCA", "TA"]
    code = {"A": 0, "C": 1, "G": 2, "T": 3}
    cw, ch, gap = 0.92, 0.92, 1.0
    rows_y = [2.0, 0.9, -0.2]
    x_in, x_out = 0.0, 7.4

    for r, sq in enumerate(seqs):
        ax1.text(
            x_in - 0.30,
            rows_y[r] + ch / 2,
            r"$s_%d$" % (r + 1),
            ha="right",
            va="center",
            fontsize=6.5,
            color=CHARCOAL,
        )
        for c in range(4):
            filled = c < len(sq)
            ax1.add_patch(
                Rectangle(
                    (x_in + c * gap, rows_y[r]),
                    cw,
                    ch,
                    facecolor="#E8F5F9" if filled else "#F7F7F7",
                    edgecolor=NPG_BLUE if filled else "#C4C4C4",
                    lw=0.5,
                    linestyle="-" if filled else (0, (1.2, 1.2)),
                )
            )
            if filled:
                ax1.text(
                    x_in + c * gap + cw / 2,
                    rows_y[r] + ch / 2,
                    sq[c],
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    family="monospace",
                    color=CHARCOAL,
                )

    ax1.add_patch(
        FancyArrowPatch(
            (4.55, 1.36), (6.95, 1.36), arrowstyle="-|>", mutation_scale=6, color=CHARCOAL, lw=0.7
        )
    )
    ax1.text(
        5.75,
        1.62,
        "one concatenation,\none table gather",
        ha="center",
        va="bottom",
        fontsize=6,
        color=CHARCOAL,
        linespacing=1.35,
    )

    for r, sq in enumerate(seqs):
        for c in range(4):
            filled = c < len(sq)
            ax1.add_patch(
                Rectangle(
                    (x_out + c * gap, rows_y[r]),
                    cw,
                    ch,
                    facecolor=NPG_BLUE if filled else "#F7F7F7",
                    edgecolor=NPG_BLUE if filled else "#C4C4C4",
                    lw=0.5,
                    linestyle="-" if filled else (0, (1.2, 1.2)),
                )
            )
            if filled:
                ax1.text(
                    x_out + c * gap + cw / 2,
                    rows_y[r] + ch / 2,
                    str(code[sq[c]]),
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white",
                )
        ax1.text(
            x_out + 4 * gap + 0.55,
            rows_y[r] + ch / 2,
            str(len(sq)),
            ha="center",
            va="center",
            fontsize=6.5,
            color=NPG_NAVY,
        )

    ax1.text(
        x_out + 4 * gap + 0.55,
        rows_y[0] + ch + 0.22,
        r"$\ell$",
        ha="center",
        va="bottom",
        fontsize=7,
        color=NPG_NAVY,
    )
    ax1.text(
        x_out + 1.9,
        rows_y[2] - 0.42,
        r"uint8 matrix $D$   +   lengths $\ell$",
        ha="center",
        va="top",
        fontsize=6,
        color=CHARCOAL,
    )
    ax1.text(
        1.84,
        rows_y[2] - 0.42,
        "ragged batch of sequences",
        ha="center",
        va="top",
        fontsize=6,
        color=CHARCOAL,
    )

    ax1.set_xlim(-1.1, 12.6)
    ax1.set_ylim(-1.55, 3.55)

    # --- b: anti-diagonal wavefront ---------------------------------------
    n = 6

    def corner(i, j):
        return j, n - 1 - i

    for i in range(n):
        for j in range(n):
            ax2.add_patch(
                Rectangle(
                    corner(i, j),
                    0.9,
                    0.9,
                    facecolor=NPG_BLUE,
                    alpha=0.10 + 0.028 * (i + j),
                    edgecolor="white",
                    lw=0.6,
                )
            )

    for d, colour in ((4, NPG_ORANGE), (5, NPG_NAVY)):
        for i in range(n):
            j = d - i
            if 0 <= j < n:
                x, y = corner(i, j)
                ax2.add_patch(
                    Rectangle((x, y), 0.9, 0.9, facecolor="none", edgecolor=colour, lw=0.9)
                )
        i0 = max(0, d - n + 1)
        x, y = corner(i0, d - i0)
        ax2.text(
            x + 0.45,
            y + 1.04,
            r"$d$" if d == 4 else r"$d+1$",
            ha="center",
            fontsize=6.5,
            color=colour,
        )

    ti, tj = 3, 2
    tx, ty = corner(ti, tj)
    for si, sj in ((ti - 1, tj - 1), (ti - 1, tj), (ti, tj - 1)):
        sx, sy = corner(si, sj)
        ax2.add_patch(
            FancyArrowPatch(
                (sx + 0.45, sy + 0.45),
                (tx + 0.45, ty + 0.45),
                arrowstyle="-|>",
                mutation_scale=5.5,
                color=CHARCOAL,
                lw=0.65,
                shrinkA=3.5,
                shrinkB=3.5,
            )
        )
    ax2.add_patch(Rectangle((tx, ty), 0.9, 0.9, facecolor="none", edgecolor=CHARCOAL, lw=1.0))

    ax2.text(
        n + 0.55,
        4.35,
        "each cell reads only\nearlier anti-diagonals",
        fontsize=6,
        va="center",
        ha="left",
        color=CHARCOAL,
        linespacing=1.35,
    )
    ax2.text(
        n + 0.55,
        2.75,
        r"cells sharing $d=i+j$" "\nare independent",
        fontsize=6,
        va="center",
        ha="left",
        color=CHARCOAL,
        linespacing=1.35,
    )
    ax2.text(
        n + 0.55,
        1.05,
        r"$O(mn)\;\rightarrow\;O(m+n)$" "\ninterpreter steps",
        fontsize=6.5,
        va="center",
        ha="left",
        color=NPG_NAVY,
        linespacing=1.35,
    )

    ax2.set_xlim(-0.35, 12.1)
    ax2.set_ylim(-0.6, 7.0)

    # Placed on the figure so both labels sit at one height despite the
    # panels having different data ranges.
    for x, letter in ((0.015, "a"), (0.525, "b")):
        fig.text(
            x, 0.965, letter, fontsize=8, fontweight="bold", va="top", ha="left", color=CHARCOAL
        )

    save(fig, "fig1_design")


# ==================================================== Figure 2: batch scaling
def figure_batch(primary: dict) -> None:
    """Throughput scaling of the batch operations."""
    d = primary["batch"]
    ops = [
        ("gc_content", "GC content"),
        ("translate", "Translation"),
        ("reverse_complement", "Reverse complement"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.0))
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.20, top=0.86, wspace=0.30)

    for ax, (letter, (key, title)) in zip(axes, zip("abc", ops)):
        v = d[key]
        x = np.array(v["sizes"], dtype=float)
        for impl, label, colour, marker in SERIES:
            if impl not in v:
                continue
            ax.loglog(
                x, arr(v[impl]), marker=marker, color=colour, label=label, markeredgecolor=colour
            )
        ax.set_xlabel("Sequences (n)")
        ax.text(
            0.5,
            1.02,
            title,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7,
            color=CHARCOAL,
        )
        clean(ax)
        panel_label(ax, letter)

    axes[0].set_ylabel("Wall-clock time (s)")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 0.99))
    save(fig, "fig2_batch")


# ======================================================= Figure 3: alignment
def figure_alignment(primary: dict) -> None:
    """Pairwise alignment cost and cell throughput against sequence length."""
    d = primary["align"]
    x = np.array(d["lengths"], dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2, 2.15))
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.19, top=0.88, wspace=0.28)

    for impl, label, colour, marker in SERIES:
        if impl not in d:
            continue
        ax1.loglog(
            x, arr(d[impl]), marker=marker, color=colour, label=label, markeredgecolor=colour
        )

    ref = x**2 / x[-1] ** 2 * d["seqcore"][-1]
    ax1.loglog(x, ref, ls=(0, (3, 2)), lw=0.7, color=CHARCOAL, label=r"$O(L^2)$", zorder=1)
    ax1.set_xlabel(r"Sequence length $L$ (bp)")
    ax1.set_ylabel("Wall-clock time (s)")
    clean(ax1)
    ax1.legend(loc="upper left")
    panel_label(ax1, "a")

    for impl, label, colour, marker in SERIES:
        if impl not in d:
            continue
        cells = x * x
        thr = cells / arr(d[impl]) / 1e6
        ax2.semilogx(x, thr, marker=marker, color=colour, label=label, markeredgecolor=colour)
    ax2.set_xlabel(r"Sequence length $L$ (bp)")
    ax2.set_ylabel(r"Throughput ($10^6$ DP cells s$^{-1}$)")
    ax2.set_yscale("log")
    clean(ax2)
    panel_label(ax2, "b")

    save(fig, "fig3_alignment")


# ============================================ Figure 4: k-mers + summary bars
def figure_kmers_and_summary(primary: dict, suites: dict, primary_label: str) -> None:
    """k-mer dispatch behaviour and the cross-platform summary."""
    d = primary["kmer"]
    ks = np.array(d["ks"], dtype=float)
    n_seq = d.get("n_sequences", 2000)
    seq_len = primary.get("seq_length", 1000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2, 2.15))
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.19, top=0.88, wspace=0.42)

    # --- a: k-mer counting across k ---------------------------------------
    for impl, label, colour, marker in SERIES:
        series = d.get(impl)
        if not series or all(v is None for v in series):
            continue
        ax1.plot(ks, arr(series), marker=marker, color=colour, label=label, markeredgecolor=colour)

    finite = [v for impl, *_ in SERIES for v in (d.get(impl) or []) if v is not None]
    ax1.set_ylim(0, max(finite) * 1.30)

    windows = n_seq * (seq_len - ks + 1)
    over = np.nonzero(4.0**ks > windows)[0]
    if over.size:
        cross = ks[over[0]]
        ax1.axvline(cross, color=CHARCOAL, ls=(0, (3, 2)), lw=0.6)
        ax1.text(
            cross + 0.4,
            ax1.get_ylim()[1] * 0.98,
            "$4^k$ > windows:\ninteger path not used",
            fontsize=5.5,
            color=CHARCOAL,
            va="top",
            ha="left",
            linespacing=1.35,
        )

    ax1.set_xlabel(r"$k$-mer length $k$")
    ax1.set_ylabel("Wall-clock time (s)")
    ax1.set_xticks(ks)
    clean(ax1)
    ax1.legend(loc="center left", bbox_to_anchor=(0.02, 0.62))
    panel_label(ax1, "a")

    # --- b: same comparison on every machine measured ---------------------
    labels = [primary_label] + [k for k in suites if k != primary_label]
    per_machine = {lab: summary_rows(suites[lab]) for lab in labels}
    ordered = sorted(per_machine[primary_label], key=lambda r: r["speedup_vs_best"])
    op_order = [r["short"] for r in ordered]

    yy = np.arange(len(op_order))
    bar_colours = [NPG_BLUE, NPG_NAVY, NPG_TEAL]
    height = 0.8 / max(len(labels), 1)

    for m, lab in enumerate(labels):
        lookup = {r["short"]: r["speedup_vs_best"] for r in per_machine[lab]}
        vals_m = [lookup.get(name, np.nan) for name in op_order]
        offset = (m - (len(labels) - 1) / 2) * height
        ax2.barh(
            yy - offset,
            vals_m,
            height,
            linewidth=0,
            color=bar_colours[m % len(bar_colours)],
            label=lab.split(" (")[0],
        )

    ax2.axvline(1.0, color=CHARCOAL, lw=0.7, ls=(0, (2, 2)))
    ax2.set_xscale("log")
    ax2.set_xlim(0.02, 90)
    ax2.set_yticks(yy)
    ax2.set_yticklabels(op_order, fontsize=5.6)
    ax2.set_xlabel("Seqcore speedup vs fastest alternative")
    ax2.text(
        1.0,
        len(op_order) - 0.30,
        " slower  |  faster",
        fontsize=5.0,
        color=CHARCOAL,
        ha="center",
        va="bottom",
    )
    clean(ax2, grid_axis="x")
    ax2.legend(loc="lower right", fontsize=5.2)
    panel_label(ax2, "b", dx=-0.40)

    save(fig, "fig4_kmers_summary")


# ==================================================== Figure 5 (optional): GPU
def figure_gpu() -> bool:
    """GPU offload analysis, if a gpu_*.json result is present.

    Returns False (drawing nothing) when no GPU benchmark has been run, so the
    figure set still builds on a machine without a GPU.
    """
    files = sorted(glob.glob(str(ROOT / "benchmarks" / "results" / "gpu_2*.json")))
    files = [f for f in files if json.loads(Path(f).read_text()).get("kind") == "gpu-headroom"]
    if not files:
        print("  (no GPU results; skipping GPU figure)")
        return False

    d = json.loads(Path(files[-1]).read_text())
    n_max = max(r["n_sequences"] for r in d["results"])
    at_max = [r for r in d["results"] if r["n_sequences"] == n_max]

    pretty = {
        "gc_content": "GC content",
        "reverse_complement": "Rev. complement",
        "translate": "Translation",
    }

    def label(op):
        if "_k" in op:
            return "$k$-mers ($k$=%s)" % op.split("_k")[-1]
        return pretty.get(op, op)

    names = [label(r["operation"]) for r in at_max]
    kernel = np.array([r["headroom_compute"] for r in at_max])
    e2e = np.array([r["headroom_end2end"] for r in at_max])
    kern_ms = np.array([r["cupy_gpu_compute"] for r in at_max]) * 1000
    xfer_ms = np.maximum(np.array([r["cupy_gpu_end2end"] for r in at_max]) * 1000 - kern_ms, 0)
    cpu_s = np.array([r["seqcore_cpu"] for r in at_max])

    op_colours = [NPG_BLUE, NPG_RED, NPG_TEAL, NPG_NAVY]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(COL2, 2.25))
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.30, top=0.88, wspace=0.42)

    # --- a: the two speedups ----------------------------------------------
    x = np.arange(len(names))
    bw = 0.36
    ax1.bar(x - bw / 2, kernel, bw, color=NPG_BLUE, linewidth=0, label="Resident data")
    ax1.bar(x + bw / 2, e2e, bw, color=NPG_NAVY, linewidth=0, label="Per call, with transfer")
    ax1.axhline(1.0, color=CHARCOAL, lw=0.6, ls=(0, (2, 2)))
    ax1.set_yscale("log")
    for xi, v in zip(np.concatenate([x - bw / 2, x + bw / 2]), np.concatenate([kernel, e2e])):
        ax1.text(xi, v * 1.15, f"{v:.0f}×", ha="center", fontsize=5.0, color=CHARCOAL)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=25, ha="right", fontsize=5.4)
    ax1.set_ylabel("Speedup over CPU")
    ax1.set_ylim(0.8, max(kernel) * 5.5)
    clean(ax1, grid_axis="y")
    ax1.legend(loc="upper center", ncol=1, fontsize=5.2, bbox_to_anchor=(0.45, 1.03))
    panel_label(ax1, "a")

    # --- b: where the per-call time goes ----------------------------------
    ax2.barh(x, kern_ms, 0.55, color=NPG_BLUE, linewidth=0, label="Kernel")
    ax2.barh(x, xfer_ms, 0.55, left=kern_ms, color=NPG_ORANGE, linewidth=0, label="Host transfer")
    total = kern_ms + xfer_ms
    for xi, (k_, t_) in enumerate(zip(kern_ms, xfer_ms)):
        ax2.text(
            k_ + t_ + total.max() * 0.03,
            xi,
            f"{t_ / (k_ + t_) * 100:.0f}%",
            va="center",
            fontsize=5.2,
            color=CHARCOAL,
        )
    ax2.set_yticks(x)
    ax2.set_yticklabels(names, fontsize=5.4)
    ax2.set_xlabel("Time per call (ms)")
    ax2.set_xlim(0, total.max() * 1.30)
    clean(ax2, grid_axis="x")
    ax2.legend(loc="lower right", fontsize=5.2)
    panel_label(ax2, "b", dx=-0.44)

    # --- c: amortization over chained operations --------------------------
    n_calls = np.array([1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 200, 500])
    for i, r in enumerate(at_max):
        k_s = r["cupy_gpu_compute"]
        x_s = r["cupy_gpu_end2end"] - k_s
        speed = (n_calls * cpu_s[i]) / (x_s + n_calls * k_s)
        ax3.plot(n_calls, speed, color=op_colours[i % len(op_colours)], lw=1.0, label=names[i])
        ax3.axhline(
            kernel[i], color=op_colours[i % len(op_colours)], lw=0.5, ls=(0, (2, 2)), alpha=0.7
        )
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("Operations before returning to host")
    ax3.set_ylabel("Effective speedup")
    clean(ax3)
    ax3.legend(loc="lower right", fontsize=5.0)
    panel_label(ax3, "c")

    gpu_name = d.get("gpu", {}).get("name", "GPU")
    fig.text(
        0.5,
        0.045,
        "%s; batch of %s x %d bp. Dashed lines in c: resident-data limit."
        % (gpu_name, f"{n_max:,}", d["seq_length"]),
        ha="center",
        fontsize=5.4,
        color=CHARCOAL,
    )

    save(fig, "fig5_gpu")
    return True


# ================================================ shared summary and Table 1
def summary_rows(d: dict) -> list[dict]:
    """Per-operation summary shared by Figure 4b and Table 1."""
    rows = []

    n_max = d["batch"]["gc_content"]["sizes"][-1]
    cfg_batch = r"$%s \times %d$\,bp" % (f"{n_max:,}".replace(",", "{,}"), d["seq_length"])
    for key, label, short in (
        ("gc_content", "GC content", "GC content"),
        ("translate", "Translation", "Translation"),
        ("reverse_complement", "Reverse complement", "Rev. complement"),
    ):
        v = d["batch"][key]
        rows.append(
            {
                "label": label,
                "short": short,
                "config": cfg_batch,
                "seqcore": v["seqcore"][-1],
                "biopython": v["biopython"][-1],
                "python": v["python"][-1],
            }
        )

    l_max = d["align"]["lengths"][-1]
    rows.append(
        {
            "label": "Global alignment",
            "short": "Alignment",
            "config": r"$L=%d$\,bp" % l_max,
            "seqcore": d["align"]["seqcore"][-1],
            "biopython": d["align"]["biopython"][-1],
            "python": d["align"]["python"][-1],
        }
    )

    ks = d["kmer"]["ks"]
    cfg_kmer = r"$%s \times %d$\,bp" % (
        f"{d['kmer']['n_sequences']:,}".replace(",", "{,}"),
        d["seq_length"],
    )
    for k, short in ((8, "k-mers (k = 8)"), (12, "k-mers (k = 12)")):
        if k not in ks:
            continue
        i = ks.index(k)
        rows.append(
            {
                "label": r"$k$-mers ($k=%d$)" % k,
                "short": short,
                "config": cfg_kmer,
                "seqcore": d["kmer"]["seqcore"][i],
                "biopython": d["kmer"]["biopython"][i],
                "python": d["kmer"]["python"][i],
            }
        )

    for r in rows:
        alts = [v for v in (r["biopython"], r["python"]) if v is not None]
        r["best_alt"] = min(alts) if alts else None
        r["speedup_vs_best"] = (r["best_alt"] / r["seqcore"]) if alts else None
    return rows


def write_table(rows: list[dict]) -> None:
    """Emit the Table 1 body so the manuscript hard-codes no timing."""

    def fmt(v):
        return "---" if v is None else f"{v:.3f}"

    def ratio(v):
        if v is None:
            return "---"
        return (r"$%.1f\times$" if v >= 10 else r"$%.2f\times$") % v

    lines = [
        "% Generated by paper/make_figures.py -- do not edit by hand.",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Operation & Configuration & Seqcore (s) & Biopython (s) & Pure Python (s)"
        r" & Speedup \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            "%s & %s & %s & %s & %s & %s \\\\"
            % (
                r["label"],
                r["config"],
                fmt(r["seqcore"]),
                fmt(r["biopython"]),
                fmt(r["python"]),
                ratio(r["speedup_vs_best"]),
            )
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "table1.tex").write_text("\n".join(lines) + "\n")
    print("  table1.tex")


def write_numbers(rows: list[dict], d: dict, suites: dict, primary_label: str) -> None:
    """Emit \\newcommand definitions for every value quoted in the manuscript.

    The prose then references macros rather than transcribed digits, so the text
    cannot drift from the benchmark data it describes. Regenerating on new
    hardware updates the numbers and the platform sentence together.
    """
    batch, align, kmer = d["batch"], d["align"], d["kmer"]
    libs = d.get("library_versions", {})

    def texnum(x, places=1):
        return f"{x:,.{places}f}".replace(",", "{,}")

    def textesc(value: str) -> str:
        """Escape the LaTeX specials that can appear in platform strings."""
        out = str(value)
        for char, repl in (
            ("\\", r"\textbackslash{}"),
            ("&", r"\&"),
            ("%", r"\%"),
            ("$", r"\$"),
            ("#", r"\#"),
            ("_", r"\_"),
            ("{", r"\{"),
            ("}", r"\}"),
            ("~", r"\textasciitilde{}"),
            ("^", r"\textasciicircum{}"),
        ):
            out = out.replace(char, repl)
        return out

    n_max = batch["gc_content"]["sizes"][-1]
    seq_len = d["seq_length"]
    bases = n_max * seq_len

    gc = {k: batch["gc_content"][k][-1] for k in ("seqcore", "biopython", "python")}
    tr = {k: batch["translate"][k][-1] for k in ("seqcore", "biopython", "python")}
    rc = {k: batch["reverse_complement"][k][-1] for k in ("seqcore", "biopython", "python")}

    l_lo, l_hi = align["lengths"][0], align["lengths"][-1]
    a_lo, a_hi = align["seqcore"][0], align["seqcore"][-1]
    a_bio = align["biopython"][-1]

    ks = kmer["ks"]
    k8 = ks.index(8) if 8 in ks else 0
    k12 = ks.index(12) if 12 in ks else -1
    windows = kmer["n_sequences"] * (seq_len - np.array(ks) + 1)
    over = np.nonzero(4.0 ** np.array(ks, dtype=float) > windows)[0]
    cross = ks[over[0]] if over.size else ks[-1]

    cpu = textesc(d.get("cpu_model") or d.get("processor") or "unknown CPU")
    system = d.get("system", "")
    release = d.get("platform", "")
    machine = (d.get("machine") or "").replace("_", "-") or "unknown"

    # A compact, line-breakable OS label; the raw platform string is too long
    # to typeset as an unbreakable \texttt run.
    kernel = ""
    for part in release.split("-"):
        if part and part[0].isdigit():
            kernel = ".".join(part.split(".")[:2])
            break
    hw = textesc((d.get("hardware_model") or "").split(" (")[0])
    cores = d.get("cpu_count")
    core_note = f", {cores} cores" if cores else ""
    if system == "Linux":
        host = f"an {hw} instance ({cpu}{core_note})" if hw else f"a Linux host ({cpu}{core_note})"
        os_label = f"Linux {kernel}".strip() + f", {machine}"
    elif system == "Darwin":
        host = f"a {hw} ({cpu}{core_note})" if hw else f"an Apple silicon Mac ({cpu}{core_note})"
        os_label = f"macOS {kernel}".strip() + f", {machine}"
    else:
        host = f"a {system} host ({cpu}{core_note})"
        os_label = f"{system} {kernel}".strip() + f", {machine}"

    defs = {
        # platform
        "BenchHost": host,
        "BenchPlatform": textesc(os_label),
        "BenchPlatformFull": textesc(release),
        "BenchPython": d.get("python", "?"),
        "BenchNumpy": libs.get("numpy") or "?",
        "BenchBiopython": libs.get("biopython") or "?",
        # batch configuration
        "BatchN": f"{n_max:,}".replace(",", "{,}"),
        "BatchLen": str(seq_len),
        # GC content
        "GCms": texnum(gc["seqcore"] * 1000, 0),
        "GCthroughput": texnum(bases / gc["seqcore"] / 1e9, 1),
        "GCvsBio": texnum(gc["biopython"] / gc["seqcore"], 1),
        "GCvsPy": texnum(gc["python"] / gc["seqcore"], 1),
        # translation
        "TransMs": texnum(tr["seqcore"] * 1000, 0),
        "TransThroughput": texnum(bases / tr["seqcore"] / 1e6, 0),
        "TransVsBio": texnum(tr["biopython"] / tr["seqcore"], 1),
        "TransVsPy": texnum(tr["python"] / tr["seqcore"], 1),
        # reverse complement
        "RCms": texnum(rc["seqcore"] * 1000, 0),
        "RCbioms": texnum(rc["biopython"] * 1000, 0),
        "RCvsBio": texnum(rc["biopython"] / rc["seqcore"], 2),
        # alignment
        "AlignLenLo": str(l_lo),
        "AlignLenHi": str(l_hi),
        "AlignCellsLo": texnum(l_lo * l_lo / a_lo / 1e6, 1),
        "AlignCellsHi": texnum(l_hi * l_hi / a_hi / 1e6, 1),
        "AlignBioCells": texnum(l_hi * l_hi / a_bio / 1e6, 0),
        "AlignSlower": texnum(a_hi / a_bio, 0),
        # k-mers
        "KmerN": f"{kmer['n_sequences']:,}".replace(",", "{,}"),
        "KmerEightSpeedup": texnum(kmer["python"][k8] / kmer["seqcore"][k8], 1),
        "KmerTwelveSpeedup": texnum(kmer["python"][k12] / kmer["seqcore"][k12], 1),
        "KmerCross": str(cross),
    }

    # Cross-platform macros: the secondary machine, and the range of speedups
    # observed for the same operation across machines.
    others = [lab for lab in suites if lab != primary_label]
    if others:
        other_label = others[0]
        other = suites[other_label]
        defs["OtherMachine"] = textesc(other_label)
        defs["OtherCPU"] = textesc(other.get("cpu_model", "?"))
        defs["OtherCores"] = str(other.get("cpu_count", "?"))
        defs["OtherPython"] = other.get("python", "?")
        defs["OtherGCvsBio"] = texnum(
            other["batch"]["gc_content"]["biopython"][-1]
            / other["batch"]["gc_content"]["seqcore"][-1],
            1,
        )
        defs["OtherTransVsBio"] = texnum(
            other["batch"]["translate"]["biopython"][-1]
            / other["batch"]["translate"]["seqcore"][-1],
            1,
        )
        defs["OtherAlignSlower"] = texnum(
            other["align"]["seqcore"][-1] / other["align"]["biopython"][-1], 0
        )
        defs["OtherRCvsBio"] = texnum(
            other["batch"]["reverse_complement"]["biopython"][-1]
            / other["batch"]["reverse_complement"]["seqcore"][-1],
            2,
        )
        # How much faster the primary machine is, per operation.
        ratios = []
        for op in ("gc_content", "translate", "reverse_complement"):
            ratios.append(other["batch"][op]["seqcore"][-1] / batch[op]["seqcore"][-1])
        ratios.append(other["align"]["seqcore"][-1] / align["seqcore"][-1])
        defs["PrimaryFasterLo"] = texnum(min(ratios), 1)
        defs["PrimaryFasterHi"] = texnum(max(ratios), 1)
        defs["NumMachines"] = str(len(suites))

    # The Seqcore version the manuscript describes. Taken from the installed
    # package, and cross-checked against the version each benchmark recorded so
    # that results produced by different code cannot be reported as one release.
    try:
        from importlib.metadata import version as _pkg_version

        current = _pkg_version("seqcore")
    except Exception:
        current = None

    measured = {
        lab: (v.get("library_versions", {}) or {}).get("seqcore") for lab, v in suites.items()
    }
    distinct = {v for v in measured.values() if v}
    if len(distinct) > 1:
        raise SystemExit(
            "Benchmark results were produced by different Seqcore versions "
            f"({sorted(distinct)}). Re-run benchmark_suite.py on every machine "
            "with the same version before generating the manuscript."
        )
    if current and distinct and current not in distinct:
        print(
            f"  WARNING: package is {current} but benchmarks were run with "
            f"{sorted(distinct)[0]}. Re-run the suite before submitting."
        )
    defs["SeqcoreVersion"] = current or (sorted(distinct)[0] if distinct else "?")
    defs["PrimaryMachine"] = textesc(primary_label)
    defs["PrimaryCores"] = str(d.get("cpu_count", "?"))

    lines = ["% Generated by paper/make_figures.py -- do not edit by hand."]
    lines += [r"\newcommand{\%s}{%s}" % (name, value) for name, value in defs.items()]
    (HERE / "numbers.tex").write_text("\n".join(lines) + "\n")
    print("  numbers.tex")


if __name__ == "__main__":
    print("Generating manuscript figures...")
    suites = load_suites()
    primary_label, primary = primary_suite(suites)
    print(f"  primary machine: {primary_label}")
    for other in (k for k in suites if k != primary_label):
        print(f"  also measured  : {other}")

    rows = summary_rows(primary)
    figure_design()
    figure_batch(primary)
    figure_alignment(primary)
    figure_kmers_and_summary(primary, suites, primary_label)
    figure_gpu()
    write_table(rows)
    write_numbers(rows, primary, suites, primary_label)
    print(f"Done -> {FIGDIR}")
