#!/usr/bin/env python3
"""Render the Seqcore outreach explainer (~85 s, 1080p MP4).

Speed figures come from the committed benchmark JSON, not from constants here,
so the video cannot claim something the repository does not measure.

Usage:  python make_video.py [--preview] [--out FILE]
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

FPS = 30
W, H = 16, 9  # inches at dpi=120 -> 1920x1080

INK = "#F2F2F2"  # primary text on dark
DIM = "#9A9A9A"
BG = "#141821"
BLUE = "#4DBBD5"
RED = "#E64B35"
TEAL = "#00A087"
NAVY = "#3C5488"
ORANGE = "#F39B7F"
BASE_COLOR = {"A": BLUE, "C": TEAL, "G": ORANGE, "T": NAVY}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
    }
)


# ----------------------------------------------------------------- data
def find_repo(explicit: str | None = None) -> Path:
    """Locate the Seqcore checkout holding the benchmark results."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates += [
        Path.cwd(),
        *Path(__file__).resolve().parents,
        Path.home() / "Seqcore",
    ]
    for c in candidates:
        for probe in (c, c / "Seqcore"):
            if (probe / "benchmarks" / "results").is_dir():
                return probe
    raise SystemExit(
        "Could not find a Seqcore checkout with benchmarks/results/.\n"
        "Pass one explicitly:  python make_video.py --repo /path/to/Seqcore"
    )


def load_numbers(repo: Path) -> dict:
    """Pull the quoted figures out of the committed benchmark results."""
    n = {}
    files = sorted(glob.glob(str(repo / "benchmarks/results/suite_*.json")))
    suites = [json.loads(Path(f).read_text()) for f in files]
    suites = [s for s in suites if s.get("library_versions")]
    if not suites:
        raise SystemExit(
            f"No benchmark results with metadata under {repo}/benchmarks/results/.\n"
            "Run: python benchmarks/benchmark_suite.py"
        )
    mac = next((s for s in suites if s.get("system") == "Darwin"), suites[-1])

    tr = mac["batch"]["translate"]
    n["seqcore_s"] = tr["seqcore"][-1]
    n["biopython_s"] = tr["biopython"][-1]
    n["n_seqs"] = tr["sizes"][-1]
    n["speedup"] = n["biopython_s"] / n["seqcore_s"]

    al = mac["align"]
    n["align_seqcore"] = al["seqcore"][-1]
    n["align_biopython"] = al["biopython"][-1]
    n["align_slower"] = n["align_seqcore"] / n["align_biopython"]
    return n


# ------------------------------------------------------------- helpers
def ease(t: float) -> float:
    """Smoothstep, for movement that starts and stops gently."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def fade(ax, t, t_in=0.4, t_out=None, hold=None):
    """Alpha envelope: fade in, hold, fade out."""
    a = ease(t / t_in) if t < t_in else 1.0
    if hold is not None and t > hold:
        a = 1.0 - ease((t - hold) / (t_out or 0.4))
    return max(0.0, min(1.0, a))


def text(ax, x, y, s, size=34, color=INK, alpha=1.0, weight="normal", ha="center"):
    if alpha <= 0.01:
        return
    ax.text(
        x,
        y,
        s,
        color=color,
        fontsize=size,
        alpha=alpha,
        ha=ha,
        va="center",
        fontweight=weight,
        transform=ax.transAxes,
    )


def bead(ax, x, y, w, h, color, alpha=1.0, label=None, label_color="white", size=26, lw=0):
    if alpha <= 0.01:
        return
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.012",
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
            linewidth=lw,
            transform=ax.transAxes,
        )
    )
    if label:
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            color=label_color,
            fontsize=size,
            ha="center",
            va="center",
            alpha=alpha,
            transform=ax.transAxes,
            fontweight="bold",
        )


SEQ = "ACGTACGGTACAGTCCATGGACTTAGCATCGGATCAGTACCGATTGACCA"


# -------------------------------------------------------------- scenes
def scene_intro(ax, t):
    """0-12 s: four letters."""
    text(ax, 0.5, 0.78, "Inside every living thing", 40, INK, fade(t, 0.8, hold=10.5))
    text(
        ax,
        0.5,
        0.70,
        "is an instruction book.",
        40,
        INK,
        fade(t - 0.7, 0.8, hold=9.8) if t > 0.7 else 0,
    )

    if t > 3.0:
        a = fade(t - 3.0, 0.9, hold=7.5)
        text(ax, 0.5, 0.55, "It is written with four letters.", 30, DIM, a)

    letters = "ACGT"
    for i, ch in enumerate(letters):
        start = 4.6 + i * 0.45
        if t < start:
            continue
        a = fade(t - start, 0.5, hold=5.9 - i * 0.45)
        p = ease(min((t - start) / 0.5, 1))
        x = (1 - (3 * 0.116 + 0.086)) / 2 + i * 0.116
        y = 0.30 - (1 - p) * 0.03
        bead(ax, x, y, 0.086, 0.14, BASE_COLOR[ch], a, ch, "white", 54)


def scene_scale(ax, t):
    """12-26 s: the book is enormous."""
    a0 = fade(t, 0.7, hold=11.5)
    text(ax, 0.5, 0.82, "But the book is enormous.", 40, INK, a0)

    p = ease(min(t / 5.0, 1.0))
    n = int(6 + p * 170)
    total_w = 0.86
    bw = total_w / max(n, 1)
    rng = np.random.default_rng(7)
    bases = rng.choice(list("ACGT"), size=n)
    for i, ch in enumerate(bases):
        bead(
            ax,
            0.07 + i * bw,
            0.46,
            bw * 0.86,
            0.13,
            BASE_COLOR[ch],
            a0,
            ch if n <= 26 else None,
            "white",
            22,
        )

    if t > 5.6:
        a = fade(t - 5.6, 0.8, hold=5.6)
        text(ax, 0.5, 0.30, "3,000,000,000 letters in a single cell", 34, BLUE, a)
    if t > 8.4:
        a = fade(t - 8.4, 0.8, hold=3.0)
        text(ax, 0.5, 0.20, "Scientists need to ask it questions.", 27, DIM, a)


def scene_slow(ax, t):
    """26-40 s: one letter at a time."""
    a0 = fade(t, 0.6, hold=12.0)
    text(ax, 0.5, 0.86, "Computers used to read it", 36, INK, a0)
    text(ax, 0.5, 0.79, "one letter at a time.", 36, INK, a0)

    n = 34
    bw = 0.84 / n
    for i in range(n):
        bead(ax, 0.08 + i * bw, 0.50, bw * 0.84, 0.13, BASE_COLOR[SEQ[i % len(SEQ)]], a0 * 0.5)

    # A cursor that steps, deliberately slowly.
    step_time = 0.34
    idx = int(max(t - 1.6, 0) / step_time)
    if t > 1.6 and idx < n:
        x = 0.08 + idx * bw
        bead(ax, x, 0.50, bw * 0.84, 0.13, BASE_COLOR[SEQ[idx % len(SEQ)]], a0)
        ax.add_patch(
            Rectangle(
                (x - 0.006, 0.487),
                bw * 0.84 + 0.012,
                0.156,
                fill=False,
                edgecolor=INK,
                lw=2.5,
                alpha=a0,
                transform=ax.transAxes,
            )
        )
    counted = min(idx + 1, n) if t > 1.6 else 0
    if t > 1.6:
        text(ax, 0.5, 0.34, f"{counted} of {n} read", 30, DIM, a0)
    if t > 9.6:
        a = fade(t - 9.6, 0.7, hold=3.4)
        text(ax, 0.5, 0.20, "It works. It is just slow.", 32, ORANGE, a)


def scene_fast(ax, t):
    """40-56 s: numbers, all at once."""
    a0 = fade(t, 0.7, hold=13.5)
    text(ax, 0.5, 0.88, "Seqcore turns the letters into numbers", 34, INK, a0)
    text(ax, 0.5, 0.815, "and looks at all of them at once.", 34, INK, a0)

    rows, cols = 6, 26
    bw, bh = 0.031, 0.085
    x0, y0 = 0.5 - cols * bw / 2, 0.60
    code = {"A": "0", "C": "1", "G": "2", "T": "3"}
    rng = np.random.default_rng(11)
    grid = rng.choice(list("ACGT"), size=(rows, cols))

    flip = ease(min(max(t - 1.2, 0) / 2.2, 1.0))
    for r in range(rows):
        for c in range(cols):
            appear = 1.2 + (r * cols + c) * 0.0035
            if t < appear:
                continue
            ch = grid[r, c]
            lab = code[ch] if flip > 0.55 else ch
            col = BLUE if flip > 0.55 else BASE_COLOR[ch]
            bead(
                ax,
                x0 + c * bw,
                y0 - r * bh,
                bw * 0.88,
                bh * 0.84,
                col,
                a0,
                lab,
                "white" if flip > 0.55 else "white",
                17,
            )

    # One flash: the whole batch answered in a single operation.
    if 5.4 < t < 6.6:
        pulse = 1.0 - abs(t - 6.0) / 0.6
        ax.add_patch(
            Rectangle(
                (x0 - 0.012, y0 - (rows - 1) * bh - 0.012),
                cols * bw + 0.024,
                rows * bh + 0.024,
                fill=False,
                edgecolor=INK,
                lw=4,
                alpha=max(pulse, 0) * a0,
                transform=ax.transAxes,
            )
        )
    if t > 6.3:
        a = fade(t - 6.3, 0.6, hold=6.7)
        text(ax, 0.5, 0.145, "Same answer. Much less waiting.", 34, BLUE, a)


def scene_numbers(ax, t):
    """56-66 s: the real measurement."""
    a0 = fade(t, 0.7, hold=8.0)
    text(ax, 0.5, 0.86, f"Translating {NUM['n_seqs']:,} sequences", 36, INK, a0)

    grow = ease(min(max(t - 0.9, 0) / 1.5, 1.0))
    bio, sc = NUM["biopython_s"], NUM["seqcore_s"]
    full = 0.60

    for i, (lab, val, col) in enumerate([("Biopython", bio, DIM), ("Seqcore", sc, BLUE)]):
        y = 0.60 - i * 0.20
        w = full * (val / bio) * grow
        ax.add_patch(
            FancyBboxPatch(
                (0.20, y),
                max(w, 1e-4),
                0.105,
                boxstyle="round,pad=0,rounding_size=0.01",
                facecolor=col,
                edgecolor="none",
                alpha=a0,
                transform=ax.transAxes,
            )
        )
        text(ax, 0.185, y + 0.052, lab, 26, DIM, a0, ha="right")
        if grow > 0.75:
            text(
                ax,
                0.22 + max(w, 0.02),
                y + 0.052,
                f"{val:.2f} s",
                28,
                INK,
                a0 * ease((grow - 0.75) / 0.25),
                ha="left",
            )

    if t > 3.6:
        a = fade(t - 3.6, 0.7, hold=4.4)
        text(ax, 0.5, 0.22, f"{NUM['speedup']:.0f}x faster", 46, BLUE, a, "bold")


def scene_honest(ax, t):
    """66-85 s: the turn."""
    a0 = fade(t, 0.9, hold=16.5)
    text(ax, 0.5, 0.86, "But Seqcore is not the fastest", 36, INK, a0)
    text(ax, 0.5, 0.79, "at everything.", 36, INK, a0)

    if t > 2.6:
        a = fade(t - 2.6, 0.8, hold=13.5)
        text(ax, 0.5, 0.63, "Lining up two sequences, Biopython still wins.", 30, DIM, a)
        grow = ease(min((t - 3.2) / 1.4, 1.0)) if t > 3.2 else 0
        if grow > 0:
            for i, (lab, val, col) in enumerate(
                [
                    ("Biopython", NUM["align_biopython"], TEAL),
                    ("Seqcore", NUM["align_seqcore"], RED),
                ]
            ):
                y = 0.44 - i * 0.15
                w = 0.55 * (val / NUM["align_seqcore"]) * grow
                ax.add_patch(
                    FancyBboxPatch(
                        (0.24, y),
                        max(w, 1e-4),
                        0.085,
                        boxstyle="round,pad=0,rounding_size=0.01",
                        facecolor=col,
                        edgecolor="none",
                        alpha=a,
                        transform=ax.transAxes,
                    )
                )
                text(ax, 0.225, y + 0.042, lab, 24, DIM, a, ha="right")
            text(
                ax,
                0.5,
                0.20,
                f"{NUM['align_slower']:.0f}x slower",
                40,
                RED,
                fade(t - 5.2, 0.7, hold=11.0) if t > 5.2 else 0,
                "bold",
            )

    if t > 9.4:
        a = fade(t - 9.4, 0.9, hold=8.0)
        text(ax, 0.5, 0.085, "We put that in the paper, and in red.", 28, INK, a)


def scene_end(ax, t):
    """85-92 s: end card."""
    a = fade(t, 1.0, hold=4.6)
    text(ax, 0.5, 0.60, "Seqcore", 76, INK, a, "bold")
    text(ax, 0.5, 0.48, "A tool that shows you its losses", 30, DIM, a)
    text(ax, 0.5, 0.42, "is one you can trust with its wins.", 30, DIM, a)
    if t > 1.6:
        text(
            ax,
            0.5,
            0.24,
            "github.com/pritampanda15/Seqcore",
            26,
            BLUE,
            fade(t - 1.6, 0.8, hold=3.0),
        )


TIMELINE = [
    (12.0, scene_intro),
    (14.0, scene_scale),
    (14.0, scene_slow),
    (16.0, scene_fast),
    (10.0, scene_numbers),
    (19.0, scene_honest),
    (7.0, scene_end),
]
TOTAL = sum(d for d, _ in TIMELINE)


def draw(frame_idx, ax):
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(BG)

    t = frame_idx / FPS
    acc = 0.0
    for dur, fn in TIMELINE:
        if t < acc + dur:
            fn(ax, t - acc)
            break
        acc += dur
    return []


# The README clip: the stretch spanning the cut from "one letter at a time" to
# the whole batch at once, which is the argument in miniature.
GIF_START, GIF_LENGTH = 35.0, 14.0


def write_gif(source: Path, out: Path, fps: int = 15, width: int = 960) -> None:
    """Convert a segment of the rendered video into a README-sized GIF.

    Two passes: ffmpeg builds an optimal palette for the segment, then applies
    it. A single pass would quantize to a generic 256-colour palette and band
    badly on the flat background.
    """
    import subprocess
    import tempfile

    scale = f"fps={fps},scale={width}:-1:flags=lanczos"
    with tempfile.TemporaryDirectory() as tmp:
        palette = Path(tmp) / "palette.png"
        common = ["-ss", str(GIF_START), "-t", str(GIF_LENGTH), "-i", str(source)]
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                *common,
                "-vf",
                f"{scale},palettegen=max_colors=200:stats_mode=diff",
                str(palette),
                "-y",
            ],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                *common,
                "-i",
                str(palette),
                "-lavfi",
                f"{scale}[x];[x][1:v]paletteuse=dither=sierra2_4a:" "diff_mode=rectangle",
                "-loop",
                "0",
                str(out),
                "-y",
            ],
            check=True,
        )
    print(f"Wrote {out}  ({out.stat().st_size/1e6:.2f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="render 12 s only")
    ap.add_argument("--out", default="seqcore_explainer.mp4")
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--repo", default=None, help="path to the Seqcore checkout")
    ap.add_argument(
        "--gif", action="store_true", help="also write the README GIF from the rendered video"
    )
    args = ap.parse_args()

    global NUM
    NUM = load_numbers(find_repo(args.repo))

    duration = 12.0 if args.preview else TOTAL
    frames = int(duration * FPS)

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])

    print(f"Rendering {duration:.0f}s ({frames} frames) at {W*args.dpi}x{H*args.dpi}...")
    anim = animation.FuncAnimation(
        fig, draw, frames=frames, fargs=(ax,), interval=1000 / FPS, blit=False
    )
    writer = animation.FFMpegWriter(
        fps=FPS,
        bitrate=6000,
        metadata={"title": "Seqcore", "artist": "Pritam Kumar Panda"},
        extra_args=["-pix_fmt", "yuv420p", "-preset", "medium"],
    )
    out = Path(__file__).resolve().parent / args.out
    anim.save(str(out), writer=writer, dpi=args.dpi)
    print(f"Wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")

    if args.gif:
        write_gif(out, out.with_suffix(".gif"))


if __name__ == "__main__":
    main()
