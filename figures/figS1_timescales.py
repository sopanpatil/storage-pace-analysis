#!/usr/bin/env python
"""
figS1_timescales.py
===================
Figure S1 -- recession-timescale separation of HBV's two runoff-generating
stores across the 621 analysed catchments (manuscript Section 2.5 / 3.2).

Rationale
---------
The store attribution classifies each transition by which flow component -- fast
flow (Q0+Q1, from the upper zone UZ) or baseflow (Q2, from the lower zone LZ) --
paces its terminal approach to the drought threshold. That classification is
only meaningful if the two stores occupy structurally distinct recession-timescale
bands. Converting the calibrated response coefficients to e-folding timescales,

        tau = -1 / ln(1 - K),

the upper zone (interflow constant K1) and the lower zone (baseflow constant K2)
should separate cleanly. This figure documents that separation.

Manuscript values to reproduce (WY calibration, 621 catchments)
    tau_UZ  median 2.9 d  [IQR 2.8-3.9]
    tau_LZ  median 10.3 d [IQR 7.5-21.1]
    IQRs non-overlapping; LZ is the slower store in 98.2 % of catchments;
    median separation ratio tau_LZ / tau_UZ = 2.9x.

Panels
    (a) Paired distributions of tau_UZ and tau_LZ (log x-axis), with medians and
        IQRs marked, showing the non-overlapping interquartile ranges.
    (b) Per-catchment scatter tau_UZ vs tau_LZ about the 1:1 line, coloured by
        whether the lower zone is the slower store; annotates the 98.2 % fraction.

Input
    calibrated_parameters.csv  (gauge_id, ..., K1, K2, ..., used_in_analysis)

Usage
    python figS1_timescales.py --params calibrated_parameters.csv --outdir figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import agu_style as S


def tau(K: np.ndarray) -> np.ndarray:
    """e-folding recession timescale (days) for reservoir constant K in (0,1)."""
    K = np.clip(np.asarray(K, dtype=float), 1e-9, 1 - 1e-9)
    return -1.0 / np.log1p(-K)


def load(params_csv: str) -> pd.DataFrame:
    df = pd.read_csv(params_csv)
    if "used_in_analysis" in df.columns:
        df = df[df["used_in_analysis"].astype(bool)].copy()
    for col in ("K1", "K2"):
        if col not in df.columns:
            raise KeyError(f"expected column '{col}' in {params_csv}")
    df["tau_uz"] = tau(df["K1"])   # interflow, upper zone
    df["tau_lz"] = tau(df["K2"])   # baseflow, lower zone
    return df


def summarise(df: pd.DataFrame) -> None:
    def q(x):
        return np.percentile(x, [25, 50, 75])
    uz, lz = df["tau_uz"].to_numpy(), df["tau_lz"].to_numpy()
    uq, lq = q(uz), q(lz)
    lz_slower = float(np.mean(lz > uz)) * 100
    ratio = np.median(lz / uz)
    print(f"  n catchments            : {len(df)}")
    print(f"  tau_UZ median [IQR]     : {uq[1]:.1f} d  [{uq[0]:.1f}, {uq[2]:.1f}]")
    print(f"  tau_LZ median [IQR]     : {lq[1]:.1f} d  [{lq[0]:.1f}, {lq[2]:.1f}]")
    print(f"  IQRs overlap?           : {'yes' if uq[2] >= lq[0] else 'no'}")
    print(f"  LZ slower than UZ        : {lz_slower:.1f} % of catchments")
    print(f"  median separation ratio : {ratio:.1f}x")


def make_figure(df: pd.DataFrame, outdir: str) -> None:
    uz, lz = df["tau_uz"].to_numpy(), df["tau_lz"].to_numpy()

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(S.W_2COL, 68 * S.MM),
        gridspec_kw=dict(width_ratios=[1.15, 1.0], wspace=0.34),
    )

    # ---- Panel (a): paired distributions on a log axis ---------------------
    lo = min(uz.min(), lz.min())
    hi = max(uz.max(), lz.max())
    bins = np.logspace(np.log10(lo), np.log10(hi), 40)
    axA.hist(uz, bins=bins, color=S.C_FAST, alpha=0.65,
             label=r"Upper zone $\tau_{UZ}$ (fast flow)", edgecolor="none")
    axA.hist(lz, bins=bins, color=S.C_SLOW, alpha=0.55,
             label=r"Lower zone $\tau_{LZ}$ (baseflow)", edgecolor="none")
    axA.set_xscale("log")
    axA.set_xlabel("Recession e-folding timescale $\\tau$ (days)")
    axA.set_ylabel("Number of catchments")

    # medians + IQR whiskers just under the top of the panel
    axA.set_ylim(top=axA.get_ylim()[1] * 1.18)   # headroom for bars + whiskers
    ymax = axA.get_ylim()[1]
    for x, c, yy in ((uz, S.C_FAST, 0.95), (lz, S.C_SLOW, 0.87)):
        p25, p50, p75 = np.percentile(x, [25, 50, 75])
        y = yy * ymax
        axA.plot([p25, p75], [y, y], color=c, lw=1.4, solid_capstyle="butt")
        axA.plot(p50, y, marker="o", color=c, ms=3.5, mec="white", mew=0.5)
    axA.legend(loc="upper right", bbox_to_anchor=(1.0, 1.02))
    S.panel_label(axA, "a")

    # ---- Panel (b): per-catchment scatter about the 1:1 line ---------------
    slower = lz > uz
    axB.scatter(uz[slower], lz[slower], s=6, color=S.C_SLOW, alpha=0.55,
                edgecolor="none", label="LZ slower (as expected)")
    axB.scatter(uz[~slower], lz[~slower], s=8, color=S.OKABE_ITO["vermil"],
                alpha=0.85, edgecolor="none", label="UZ slower")
    lim = [0.8 * lo, 1.35 * hi]
    axB.plot(lim, lim, color=S.OKABE_ITO["black"], lw=0.7, ls="--", zorder=0)
    axB.set_xscale("log"); axB.set_yscale("log")
    axB.set_xlim(lim); axB.set_ylim(lim)
    axB.set_xlabel(r"$\tau_{UZ}$ (days)")
    axB.set_ylabel(r"$\tau_{LZ}$ (days)")
    pct = 100 * slower.mean()
    # Bottom-right, stacked above the marker legend -- top-left (the
    # original position) is where catchment points are densest (most
    # catchments sit at tau_UZ ~ 3-5 d with a wide spread of tau_LZ), so the
    # text sat directly on top of many points there. The low-tau_LZ/
    # high-tau_UZ corner is structurally near-empty (LZ is slower than UZ in
    # 98.2% of catchments), so bottom-right is clear by construction, not
    # just in this particular sample.
    axB.text(0.97, 0.30, f"LZ slower in {pct:.1f}% of catchments",
             transform=axB.transAxes, va="bottom", ha="right", fontsize=7)
    axB.legend(loc="lower right")
    S.panel_label(axB, "b")

    S.save(fig, Path(outdir) / "figS1_timescales")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", default="calibrated_parameters.csv",
                    help="calibrated_parameters.csv (with K1, K2, used_in_analysis)")
    ap.add_argument("--outdir", default="figures", help="output directory")
    args = ap.parse_args()

    S.set_style()
    df = load(args.params)
    print("Figure S1 -- recession-timescale separation")
    summarise(df)
    make_figure(df, args.outdir)


if __name__ == "__main__":
    main()
