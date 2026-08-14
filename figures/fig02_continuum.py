#!/usr/bin/env python
"""
fig02_continuum.py
==================
Figure 2 -- the flood-to-drought (FTD) transition-gap continuum
(manuscript Section 3.1).

Message
-------
Pooled coherent FTD gaps form a single, smooth, right-skewed population that
declines monotonically through the conventional 90-day cut-off with no secondary
mode, break, or inflection -- in the baseline and in every future scenario. The
"abrupt/slow" boundary therefore partitions a featureless region of a continuous
distribution, and a fixed 90-day window simply censors the tail of that
population, a tail that roughly doubles under warming.

Manuscript values to reproduce
    Baseline FTD : median 13 d, 90th pctl 56 d, slow share (>90 d) 5.0 %
    Future  FTD  : 90th pctl 86-91 d, slow share 9.2-10.1 % across the four RCPs

Panels
    (a) Pooled FTD gap histogram (density), baseline vs a representative future
        (RCP8.5 by default), with the 90-day convention marked and the censored
        slow tail (> 90 d) shaded. Shows the absence of any feature at 90 d.
    (b) Survival function 1 - ECDF for the baseline and all four future RCPs on a
        log y-axis. A distinct slow population would appear as a shoulder near
        90 d; instead every curve decays smoothly, and warming lifts the whole
        tail without creating a break.

Input
    slow_full_flow.parquet  (per-transition table from slow_transition_analysis.py)
    Required columns: gap_days, direction, regime, period, rcp, passes_coherence

Usage
    python fig02_continuum.py --input slow_full_flow.parquet --outdir figures
    python fig02_continuum.py --input slow_full_flow.parquet --future-rcp rcp85
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import agu_style as S

RCPS = ("rcp26", "rcp45", "rcp60", "rcp85")
CUTOFF = 90  # the conventional abrupt/slow window (days)



def load(parquet: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet)
    need = {"gap_days", "direction", "period", "passes_coherence"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"{parquet} is missing columns: {sorted(missing)}")
    df = df[(df["direction"] == "FTD") & (df["passes_coherence"])].copy()
    df["gap_days"] = df["gap_days"].astype(float)
    return df


def _stats(g: np.ndarray) -> tuple[float, float, float]:
    """median, 90th percentile, slow share (> 90 d), in that order."""
    if len(g) == 0:
        return np.nan, np.nan, np.nan
    return (float(np.median(g)),
            float(np.percentile(g, 90)),
            float(np.mean(g > CUTOFF)))


def summarise(df: pd.DataFrame) -> None:
    base = df.loc[df["period"] == "baseline", "gap_days"].to_numpy()
    med, p90, share = _stats(base)
    print(f"  baseline FTD  : median {med:.0f} d, p90 {p90:.0f} d, "
          f"slow share {share*100:.1f}%  (n={len(base):,})")
    for rcp in RCPS:
        fut = df.loc[(df["period"] == "future") & (df["rcp"] == rcp),
                     "gap_days"].to_numpy()
        med, p90, share = _stats(fut)
        print(f"  future {rcp}  : median {med:.0f} d, p90 {p90:.0f} d, "
              f"slow share {share*100:.1f}%  (n={len(fut):,})")


def survival(g: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """1 - ECDF evaluated on `grid` (fraction of gaps exceeding each value)."""
    g = np.sort(g)
    return 1.0 - np.searchsorted(g, grid, side="right") / len(g)


def make_figure(df: pd.DataFrame, outdir: str, future_rcp: str,
                survival_max: int = 210) -> None:
    base = df.loc[df["period"] == "baseline", "gap_days"].to_numpy()
    fut = df.loc[(df["period"] == "future") & (df["rcp"] == future_rcp),
                 "gap_days"].to_numpy()

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(S.W_2COL, 70 * S.MM),
        gridspec_kw=dict(width_ratios=[1.0, 1.0], wspace=0.30),
    )

    # ---- Panel (a): pooled histogram, baseline vs representative future ----
    xmax = 180
    bins = np.arange(0, xmax + 1, 5.0)
    # Gaps beyond xmax are dropped (not clipped) so the histogram does not pile
    # up a spurious bar at the right edge; the full tail is shown in panel (b).
    for g, c, lab in ((base, S.C_BASELINE, "Baseline (WY1982-2010)"),
                      (fut, S.C_FUTURE,
                       f"Future (WY2051-2080, {S.RCP_LABELS[future_rcp]})")):
        axA.hist(g, bins=bins, density=True, histtype="step",
                 color=c, lw=1.3, label=lab)
    # shade the censored slow tail (> 90 d)
    axA.axvspan(CUTOFF, xmax, color=S.C_TAIL, alpha=0.12, lw=0)
    axA.axvline(CUTOFF, color=S.OKABE_ITO["black"], lw=0.8, ls=(0, (4, 2)))
    axA.text(CUTOFF - 4, axA.get_ylim()[1] * 0.55, "90-day convention",
             rotation=90, fontsize=7, va="center", ha="right")
    axA.text(0.985, 0.32, "censored\nslow tail", transform=axA.transAxes,
             color=S.C_TAIL, fontsize=7, va="center", ha="right")
    axA.set_xlim(0, xmax)
    axA.set_xlabel("FTD transition gap (days)")
    axA.set_ylabel("Probability density")
    # Placed above the axes, not "upper right" inside it -- the baseline
    # peak sits at low x with a wide legend (long RCP label text), so an
    # inside-axes placement always ends up overlapping the peak regardless
    # of corner; outside is the only placement that is safe by construction.
    axA.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7,
              handletextpad=0.5, borderaxespad=0.3)
    S.panel_label(axA, "a")

    # ---- Panel (b): survival function, baseline + all four future RCPs -----
    grid = np.linspace(0, survival_max, survival_max + 1)
    axB.plot(grid, survival(base, grid), color=S.C_BASELINE, lw=1.4,
             label="Baseline")
    for rcp in RCPS:
        g = df.loc[(df["period"] == "future") & (df["rcp"] == rcp),
                   "gap_days"].to_numpy()
        if len(g):
            axB.plot(grid, survival(g, grid), color=S.RCP_COLORS[rcp], lw=1.0,
                     label=f"Future {S.RCP_LABELS[rcp]}")
    # y-axis floor: the decade just below the smallest positive fraction visible
    # within the capped range, so capping does not leave an empty bottom decade.
    vis = [survival(base, grid)]
    for rcp in RCPS:
        g = df.loc[(df["period"] == "future") & (df["rcp"] == rcp),
                   "gap_days"].to_numpy()
        if len(g):
            vis.append(survival(g, grid))
    allvals = np.concatenate(vis)
    pos = allvals[allvals > 0]
    ymin = pos.min() if pos.size else 1e-4
    floor = max(1e-4, 10 ** np.floor(np.log10(ymin)))
    axB.axvline(CUTOFF, color=S.OKABE_ITO["black"], lw=0.8, ls=(0, (4, 2)))
    axB.set_yscale("log")
    axB.set_xlim(0, survival_max)
    axB.set_ylim(floor, 1.0)
    axB.set_xlabel("FTD transition gap (days)")
    axB.set_ylabel("Fraction of gaps exceeding value")
    axB.legend(loc="upper right")
    S.panel_label(axB, "b")

    S.save(fig, Path(outdir) / "fig02_continuum")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="slow_full_flow.parquet",
                    help="per-transition parquet from slow_transition_analysis.py")
    ap.add_argument("--future-rcp", default="rcp85", choices=RCPS,
                    help="RCP shown against baseline in panel (a)")
    ap.add_argument("--survival-max", type=int, default=210,
                    help="x-axis limit (days) for the panel (b) survival curves; "
                         "default 210 caps the sparse far tail (use 365 for full)")
    ap.add_argument("--outdir", default="figures", help="output directory")
    args = ap.parse_args()

    S.set_style()
    df = load(args.input)
    print("Figure 2 -- FTD transition-gap continuum")
    summarise(df)
    make_figure(df, args.outdir, args.future_rcp, args.survival_max)


if __name__ == "__main__":
    main()
