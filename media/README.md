# Seqcore outreach video

A ~92-second animated explainer for a general audience.

## Files

| File | What it is |
|---|---|
| `narration.md` | Narration script with timings and a shot list |
| `make_video.py` | Renders the animation to MP4 |
| `seqcore_explainer.mp4` | 1920x1080, 30 fps, **silent** |
| `seqcore_explainer.gif` | 960x540, 14 s excerpt used in the project README |

## Rendering

From the repository root:

```bash
python media/make_video.py                # full ~92 s
python media/make_video.py --gif          # also rewrite the README GIF
python media/make_video.py --preview      # first 12 s, for quick iteration
python media/make_video.py --dpi 60       # 960x540, renders much faster
```

The GIF is a 14-second excerpt starting at 0:35, chosen because it spans the cut
from "one letter at a time" to the whole batch at once, which is the argument in
miniature. `GIF_START` and `GIF_LENGTH` near the bottom of `make_video.py` set
the segment. It is built with a two-pass ffmpeg palette; a single pass quantizes
to a generic palette and bands badly against the flat background.

The project README embeds the GIF by absolute `raw.githubusercontent.com` URL
rather than a relative path, because that README is also the PyPI description
and relative image paths do not resolve there.

Requires `ffmpeg` and matplotlib. The script finds the repository on its own;
pass `--repo /path/to/Seqcore` if you run it from somewhere unusual.

## The numbers are not hard-coded

`load_numbers()` reads `benchmarks/results/suite_*.json` and derives every figure
shown on screen: the translation times, the speedup, and the alignment
slowdown. Re-run the benchmarks, re-render, and the video updates itself. It
cannot display a number the repository does not measure.

That matters for the last third of the video, which is about not overstating
results. It would be a strange film to make with invented figures.

## Audio

The MP4 has **no audio track**. Scene timings match `narration.md`, so a
voiceover can be recorded against it directly:

- 0:00 intro, the four letters
- 0:12 the scale of the problem
- 0:26 reading one letter at a time
- 0:40 the array representation
- 0:56 the measured speedup vs Biopython
- 1:06 what Seqcore is slower at
- 1:25 end card

To mux a recorded voiceover in:

```bash
ffmpeg -i media/seqcore_explainer.mp4 -i voiceover.wav \
       -c:v copy -c:a aac -shortest seqcore_explainer_vo.mp4
```

## Pacing

The turn at 1:06 is the point of the piece. Leave a full beat of silence before
"But Seqcore is not the fastest at everything" -- the visuals hold there
deliberately.

Biopython is named on screen rather than called "an older tool". It is the
accurate comparison and the more generous one, and hedging the name would
undercut the argument the closing third is making.

## Adapting

- **Vertical cut for social:** change `W, H = 16, 9` to `9, 16` and re-check the
  text positions, which are in axis-fraction coordinates.
- **Shorter cut:** edit `TIMELINE` near the bottom of `make_video.py`; each entry
  is `(duration_seconds, scene_function)`.
- **Palette:** the constants near the top match the NPG colours used for the
  figures in `paper/`.
