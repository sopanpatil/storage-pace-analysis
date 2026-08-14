#!/usr/bin/env python
"""
fig04_corroboration.py
=======================
Figure 4 -- observed groundwater corroborates the lower-zone mechanism in
fractured aquifers (manuscript Section 3.3 / Methods Section 2.6).

Design
------
Section 3.3 reports three tests, applied to the same 42 borehole-monitored
catchments, and the figure follows that structure directly:

    (a) Interannual co-variation  -- Spearman rho between the observation-forced
        lower zone and the observed borehole level (seasonal cycle removed),
        by aquifer class.
    (b) Event-based co-decline    -- for catchments with at least one
        corroborable slow FTD transition, the median across-transition rank
        correlation vs. the median observed-level coherence, by aquifer class,
        point size giving the number of transitions checked.
    (c) Identifiability cross-check -- co-variation strength vs. public-water-
        supply abstraction, marker shape distinguishing catchments where the
        lower-zone recession constant K2 is identified by discharge from those
        where it sits at its calibration floor. Flags the two results the text
        singles out: catchment 34012 (floored, negligible abstraction, negative
        correlation) and the two heavily abstracted Chalk misses.

This figure intentionally does not attempt an illustrative dry-down time-series
panel (e.g. simulated LZ vs. observed level at Abbotstone through an example
transition): that needs the raw daily/monthly LZ and borehole series, which are
not part of the three summary tables this script consumes.

Manuscript values this script is expected to reproduce (from the provided
corroboration tables)
    Median interannual rho_anom : Chalk 0.43, other fractured 0.38, sandstone 0.06
    Best Chalk sites            : 0.85, 0.73, ... (0.55 = the flagship daily
                                   site, catchment 42016 / "Abbotstone")
    Corroborable transitions    : 33 total, 24 at 8 Chalk sites
    Chalk event test            : co-decline 100% over 24 dry-downs; across the
                                   8 Chalk SITES, median across-transition rho
                                   0.98 and median observed coherence 0.88.
                                   Both reductions are per-site medians of
                                   per-transition values; the per-transition
                                   values themselves are not in these tables, so
                                   the site-level median is the only figure the
                                   archive supports. Coherence separates by
                                   record type (daily 0.26/0.71 vs monthly
                                   0.72-1.00), which is why the manuscript now
                                   leads on the co-decline fraction.
    Identifiability             : catchment 34012 floored, rho = -0.33, no
                                   abstraction; catchment 37010 floored but
                                   still corroborates, rho = +0.45; catchments
                                   39007 and 34004 are Chalk misses with >90%
                                   public-water-supply abstraction

Inputs
    --corrob       borehole_corroboration_clean.csv
                   (gauge_id, tier, aquifer, BFI, n, rho_level, rho_anom,
                    rho_recession, covariation, K2, tau_LZ_cal, tau_LZ_mrc,
                    tau_GWL_mrc, pace_reliable, K2_floor)
    --summary      corroboration_summary_final.csv
                   (gauge_id, aquifer, BFI, tier, rho_anom, covariation,
                    K2_floor, n_slow_ftd, n_checked, frac_codecline, med_rho,
                    med_gwl_coh, grp)
    --run-summary  hbv_run_summary_45catchments.csv
                   (gauge_id, used, tier, aquifer, baseflow_index,
                    abs_watersupply_perc, K2, tau_LZ_d, K2_at_bound, ...)
                   Used only for the abstraction percentage in panel (c);
                   filtered to used == True (drops the three catchments,
                   39023/39027/39088, that are outside the 621-catchment
                   analysis set).

Usage
    python fig04_corroboration.py \
        --corrob borehole_corroboration_clean.csv \
        --summary corroboration_summary_final.csv \
        --run-summary hbv_run_summary_45catchments.csv \
        --outdir figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import agu_style as S

# grp labels, in the order used throughout the manuscript
GRP_ORDER = ["Chalk", "Other", "Permo-Triassic sst"]
GRP_LABELS = {
    "Chalk": "Chalk",
    "Other": "Other fractured, oolitic, limestone",
    "Permo-Triassic sst": "Permo-Triassic sandstone",
}
GRP_COLORS = {
    "Chalk": S.C_SLOW,                    # corroborates -> baseflow blue
    "Other": S.OKABE_ITO["green"],
    "Permo-Triassic sst": S.C_FUTURE,     # decoupled -> the "warning" vermilion
}

ABBOTSTONE_ID = 42016   # flagship daily Chalk site (med_gwl_coh = 0.71 matches text)
FLOORED_NEG_ID = 34012  # Chalk carrier, K2 floored, negative correlation
FLOORED_OK_ID = 37010   # Chalk carrier, K2 floored, still corroborates (rho ~0.45)
ABSTRACTED_MISSES = (39007, 34004)  # Chalk misses, >90% public-supply abstraction


def load(corrob_csv: str, summary_csv: str, run_summary_csv: str) -> pd.DataFrame:
    bc = pd.read_csv(corrob_csv)
    cs = pd.read_csv(summary_csv)
    hs = pd.read_csv(run_summary_csv)
    if "used" in hs.columns:
        hs = hs[hs["used"].astype(bool)]

    keep_cs = ["gauge_id", "n_slow_ftd", "n_checked", "frac_codecline",
               "med_rho", "med_gwl_coh", "grp"]
    df = bc.merge(cs[keep_cs], on="gauge_id", how="left")
    df = df.merge(hs[["gauge_id", "abs_watersupply_perc"]], on="gauge_id",
                  how="left")
    if df["grp"].isna().any():
        missing = df.loc[df["grp"].isna(), "gauge_id"].tolist()
        raise ValueError(f"no aquifer-group ('grp') label for gauges: {missing}")
    df["grp"] = pd.Categorical(df["grp"], categories=GRP_ORDER, ordered=True)
    return df


def summarise(df: pd.DataFrame) -> None:
    print("  --- (a) interannual co-variation ---")
    for g in GRP_ORDER:
        sub = df[df["grp"] == g]
        print(f"    {g:20s} median rho_anom = {sub['rho_anom'].median():.2f} "
              f"(n={len(sub)})")
    top3 = df[df["grp"] == "Chalk"].nlargest(3, "rho_anom")
    print("    top 3 Chalk sites  :", top3["rho_anom"].round(2).tolist(),
          " gauges", top3["gauge_id"].tolist())

    print("  --- (b) event-based co-decline (catchments with n_checked > 0) ---")
    chk = df[df["n_checked"] > 0]
    print(f"    total corroborable transitions checked: {int(chk['n_checked'].sum())}")
    for g in GRP_ORDER:
        sub = chk[chk["grp"] == g]
        if len(sub):
            # three decimals: the Chalk rank-correlation median is exactly
            # 0.975, which two-decimal float formatting renders as 0.97 while
            # the manuscript rounds it to 0.98. Printing 0.975 avoids the
            # apparent disagreement.
            print(f"    {g:20s} n_catch={len(sub)}, n_checked="
                  f"{int(sub['n_checked'].sum())}, med(med_rho)="
                  f"{sub['med_rho'].median():.3f}, "
                  f"med(med_gwl_coh)={sub['med_gwl_coh'].median():.3f}, "
                  f"mean frac_codecline={sub['frac_codecline'].mean():.2f}")

    print("  --- (c) identifiability cross-check ---")
    r = df.loc[df["gauge_id"] == FLOORED_NEG_ID].iloc[0]
    print(f"    catchment {FLOORED_NEG_ID} (floored, unabstracted): "
          f"rho_anom={r['rho_anom']:.2f}, abstraction={r['abs_watersupply_perc']:.1f}%")
    r = df.loc[df["gauge_id"] == FLOORED_OK_ID].iloc[0]
    print(f"    catchment {FLOORED_OK_ID} (floored, still corroborates): "
          f"rho_anom={r['rho_anom']:.2f}")
    for gid in ABSTRACTED_MISSES:
        r = df.loc[df["gauge_id"] == gid].iloc[0]
        print(f"    catchment {gid} (abstraction miss): rho_anom={r['rho_anom']:.2f}, "
              f"abstraction={r['abs_watersupply_perc']:.1f}%")


def make_figure(df: pd.DataFrame, outdir: str) -> None:
    fig, (axA, axB, axC) = plt.subplots(
        1, 3, figsize=(S.W_2COL, 64 * S.MM),
        gridspec_kw=dict(width_ratios=[0.92, 0.92, 1.16], wspace=0.55),
    )

    # ---- Panel (a): interannual co-variation by aquifer class -------------
    rng = np.random.default_rng(0)
    xpos = list(range(len(GRP_ORDER)))
    for xi, g in zip(xpos, GRP_ORDER):
        sub = df[df["grp"] == g]
        x = xi + rng.uniform(-0.14, 0.14, len(sub))
        axA.scatter(x, sub["rho_anom"], s=10, color=GRP_COLORS[g], alpha=0.75,
                   edgecolor="none")
        med = sub["rho_anom"].median()
        axA.plot([xi - 0.24, xi + 0.24], [med, med], color="black", lw=1.2,
                 zorder=5)
    axA.axhline(0, color="0.6", lw=0.6)
    axA.set_xticks(xpos)
    axA.set_xticklabels([GRP_LABELS[g] for g in GRP_ORDER], fontsize=6.5,
                        rotation=25, ha="right", rotation_mode="anchor")
    axA.set_xlim(xpos[0] - 0.55, xpos[-1] + 0.55)
    axA.set_ylabel(r"Interannual co-variation, $\rho_{anom}$")
    axA.set_ylim(-0.5, 1.0)
    S.panel_label(axA, "a")

    # ---- Panel (b): event-based co-decline (checked catchments only) ------
    chk = df[df["n_checked"] > 0].copy()
    # One marker per CATCHMENT, area scaled by how many slow transitions it
    # contributes, not one marker per transition. Shape carries the K2
    # identifiability split, matching panel (c) and the caption.
    for g in GRP_ORDER:
        for is_floored, marker in ((False, "o"), (True, "^")):
            sub = chk[(chk["grp"] == g) &
                      (chk["K2_floor"].astype(bool) == is_floored)]
            if not len(sub):
                continue
            axB.scatter(sub["med_gwl_coh"], sub["med_rho"],
                        s=14 + 5 * sub["n_checked"], color=GRP_COLORS[g],
                        marker=marker,
                        alpha=0.75, edgecolor="white", linewidth=0.4,
                        label=(GRP_LABELS[g].replace("\n", " ")
                               if not is_floored else None))
    axB.set_xlim(-0.05, 1.05)
    axB.set_ylim(-1.05, 1.05)
    axB.axhline(0, color="0.6", lw=0.6)
    axB.set_xlabel("Observed-level coherence (event test)")
    axB.set_ylabel("Across-transition rank correlation")
    # Placed below the axes -- the "lower left" corner sits close enough to
    # low-coherence, negative-correlation sandstone points (e.g. the pair
    # near coherence 0.75-0.85, rho -0.1 to -0.8) that a same-colour legend
    # swatch there risks being read as another data point.
    axB.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), fontsize=6,
              markerscale=0.7, handletextpad=0.3, borderaxespad=0.2)
    S.panel_label(axB, "b")

    # ---- Panel (c): identifiability cross-check ----------------------------
    floored = df["K2_floor"].astype(bool)
    for is_floored, marker, lab in ((False, "o", "K2 identified"),
                                    (True, "^", "K2 floored")):
        sub = df[floored == is_floored]
        axC.scatter(sub["abs_watersupply_perc"], sub["rho_anom"],
                   s=14, marker=marker,
                   c=[GRP_COLORS[g] for g in sub["grp"]],
                   alpha=0.8, edgecolor="0.3", linewidth=0.3, label=lab)
    axC.axhline(0, color="0.6", lw=0.6)
    axC.set_xlabel("Abstraction (%)")
    axC.set_ylabel(r"$\rho_{anom}$")
    axC.set_ylim(-0.5, 1.0)
    axC.set_xlim(-5, 108)

    # annotate the four sites the text singles out -- bare gauge IDs, kept
    # short so the panel stays legible; the interpretation is in the caption.
    for gid, dx, dy in (
        (FLOORED_NEG_ID, 10, -0.08),    # 34012: floored, no abstraction, rho<0
        (FLOORED_OK_ID, -18, 0.14),     # 37010: floored, still corroborates
        (39007, 8, 0.09),               # abstraction miss (rho < 0)
        (34004, 8, -0.13),              # abstraction miss (rho weak positive)
    ):
        r = df.loc[df["gauge_id"] == gid]
        if not r.empty:
            r = r.iloc[0]
            axC.annotate(str(gid), xy=(r["abs_watersupply_perc"], r["rho_anom"]),
                        xytext=(r["abs_watersupply_perc"] + dx,
                               r["rho_anom"] + dy),
                        fontsize=6, ha="left", va="center",
                        arrowprops=dict(arrowstyle="-", lw=0.4, color="0.4"))

    handles = [plt.Line2D([0], [0], marker="o", color="0.3", lw=0,
                          label="K2 identified", markersize=4),
               plt.Line2D([0], [0], marker="^", color="0.3", lw=0,
                          label="K2 floored", markersize=4)]
    axC.legend(handles=handles, loc="lower right", fontsize=6,
              handletextpad=0.3, borderaxespad=0.2)
    S.panel_label(axC, "c")

    S.save(fig, Path(outdir) / "fig04_corroboration")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corrob", default="derived_output/borehole_corroboration_clean.csv")
    ap.add_argument("--summary", default="derived_output/corroboration_summary_final.csv")
    ap.add_argument("--run-summary", default="derived_output/hbv_run_summary_45catchments.csv")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    S.set_style()
    df = load(args.corrob, args.summary, args.run_summary)
    print("Figure 4 -- observed-groundwater corroboration")
    summarise(df)
    make_figure(df, args.outdir)


if __name__ == "__main__":
    main()
