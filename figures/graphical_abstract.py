#!/usr/bin/env python
"""
graphical_abstract.py
=====================
Graphical abstract for the Hydrological Processes submission: a standalone,
single-panel version of Figure 2a (pooled coherent FTD gap histogram, baseline
vs RCP8.5 future, with the 90-day convention and the censored slow tail).

Differences from Figure 2a, for legibility at Table-of-Contents size
    * no panel letter;
    * legend inside the axes, upper right, with shortened labels and an
      opaque frame -- the region above the density curves right of ~30 d is
      empty in a single panel of this aspect, so an inside placement no
      longer hits the peak, and the 90-day line passes behind it;
    * larger fonts and heavier lines.

Input and censoring are identical to fig02_continuum.py (its `load` is reused).

Usage
    python graphical_abstract.py --input slow_full_flow.parquet --outdir figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import figure_style as S
from fig02_continuum import CUTOFF, RCPS, load


def make_figure(df, outdir: str, future_rcp: str) -> None:
    base = df.loc[df["period"] == "baseline", "gap_days"].to_numpy()
    fut = df.loc[(df["period"] == "future") & (df["rcp"] == future_rcp),
                 "gap_days"].to_numpy()

    fig, ax = plt.subplots(figsize=(120 * S.MM, 80 * S.MM))

    xmax = 180
    bins = np.arange(0, xmax + 1, 5.0)
    for g, c, lab in ((base, S.C_BASELINE, "Baseline (1982-2010)"),
                      (fut, S.C_FUTURE,
                       f"Future, {S.RCP_LABELS[future_rcp]} (2051-2080)")):
        ax.hist(g, bins=bins, density=True, histtype="step",
                color=c, lw=2.0, label=lab)
    ax.axvspan(CUTOFF, xmax, color=S.C_TAIL, alpha=0.12, lw=0)
    ax.axvline(CUTOFF, color=S.OKABE_ITO["black"], lw=1.0, ls=(0, (4, 2)))
    ax.text(CUTOFF - 4, ax.get_ylim()[1] * 0.45, "90-day convention",
            rotation=90, va="center", ha="right")
    ax.text(0.985, 0.30, "censored\nslow tail", transform=ax.transAxes,
            color=S.C_TAIL, va="center", ha="right", fontweight="bold")
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Flood-to-drought transition gap (days)")
    ax.set_ylabel("Probability density")
    # Opaque, borderless frame: the legend necessarily spans the 90-day line,
    # which should pass behind the text rather than through it.
    ax.legend(loc="upper right", frameon=True, facecolor="white",
              edgecolor="none", framealpha=1.0)

    S.save(fig, Path(outdir) / "graphical_abstract")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="slow_full_flow.parquet",
                    help="per-transition parquet from slow_transition_analysis.py")
    ap.add_argument("--future-rcp", default="rcp85", choices=RCPS)
    ap.add_argument("--max-gap", type=int, default=720,
                    help="production gap bound in days; 0 for uncensored")
    ap.add_argument("--outdir", default="figures", help="output directory")
    args = ap.parse_args()

    S.set_style()
    # Scale the house 8/7 pt style up for a thumbnail-sized display.
    mpl.rcParams.update({"font.size": 10, "axes.labelsize": 11,
                         "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "legend.fontsize": 10})
    df = load(args.input, max_gap=args.max_gap or None)
    make_figure(df, args.outdir, args.future_rcp)


if __name__ == "__main__":
    main()
