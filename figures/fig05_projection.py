#!/usr/bin/env python
"""
fig05_projection.py
====================
Figure 5 -- warming lengthens the flood-to-drought dry-down but not the
drought-to-flood recovery (manuscript Section 3.4).

Three populations, one figure
-----------------------------
The panels are computed on deliberately different catchment sets, so read the
denominators before comparing a number here against one in the text:

1. Panel (a), carriers. A catchment counts as a carrier if it had a baseline
   slow tail OR gained one in the future, matching freq_summary() in
   projection_analysis.py: n = 94-103 per RCP, of which 75-81% increase
   (manuscript Section 3.4). The black diamonds are a different statistic on
   a different set: the mean over ALL 621 catchments with zero-fill, which is
   the 0.055 to 0.097 the text reports.
2. Panel (b), strict carriers. `responder_table.parquet` holds the 79
   catchments with base_freq_decade > 0, the narrower set
   responder_characterisation.py screens. Its baseflow-index tertile means
   are therefore not the all-621 tertile means quoted in the text
   (0.244 / 0.016 / 0.032 for high / mid / low).
3. Panel (c), the retained study population. `forcing_deltas_rcp85.csv`
   covers all 671 CAMELS-GB v2 catchments; load() restricts it to the 621
   that pass the KGE screen, giving the +3.19 degC / +0.19 aridity / -56 mm
   medians the text reports. The bracketed spans in the manuscript are
   5th-95th percentiles across catchments, not minimum to maximum.

Panels
    (a) Per-catchment slow-transition frequency change (transitions/decade),
        FTD vs DTF, across the four RCPs, restricted to carriers -- catchments
        with a baseline slow tail in that direction, or that gain one in the
        future (box + strip); this reproduces the manuscript's carrier counts
        and % increasing (note 1).
    (b) Diffuse intensification: per-catchment frequency change at RCP8.5
        vs. baseline baseflow index, coloured by the responder/middle/
        non-responder grouping, with the population's tertile means marked
        (this is the 79-catchment strict-carrier population; note 2).
    (c) National drying context by the 2080s under RCP8.5: distributions of
        Delta-temperature, Delta-aridity and Delta-summer-rainfall across all
        catchments, medians marked.

Inputs
    --projection    projection_flow.parquet
                     (gauge_id, rcp, direction, d_freq_decade, base_freq_decade)
    --responder     responder_table.parquet
                     (gauge_id, baseflow_index, d_freq_decade, group, ...)
    --forcing       forcing_deltas_rcp85.csv
                     (gauge_id, d_tas_mean, d_aridity, d_p_summer, ...)

Usage
    python fig05_projection.py \
        --projection projection_flow.parquet \
        --responder responder_table.parquet \
        --forcing forcing_deltas_rcp85.csv \
        --outdir figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpecFromSubplotSpec

import figure_style as S

RCPS = ("rcp26", "rcp45", "rcp60", "rcp85")
GROUP_ORDER = ["non_responder", "middle", "responder"]
GROUP_COLORS = {
    "non_responder": S.OKABE_ITO["grey"],
    "middle": S.OKABE_ITO["green"],
    "responder": S.C_FUTURE,
}


def load(projection_pq: str, responder_pq: str, forcing_csv: str,
         params_csv: str, kge_min: float):
    pf = pd.read_parquet(projection_pq)
    rt = pd.read_parquet(responder_pq)
    fd = pd.read_csv(forcing_csv)
    need = {"gauge_id", "rcp", "direction", "d_freq_decade", "base_freq_decade"}
    missing = need - set(pf.columns)
    if missing:
        raise KeyError(f"{projection_pq} is missing columns: {sorted(missing)}")
    # The study population is the 621 catchments passing the KGE screen. A
    # catchment that yields no transitions at all (33020) is a genuine ZERO in
    # every per-catchment delta, not a missing value, so it is zero-filled back
    # in here rather than silently dropped -- otherwise the figure would report
    # n = 620 while the manuscript reports 621. Including it changes no reported
    # value at the quoted precision.
    retained = _retained_gauges(params_csv, kge_min)
    pf = _zero_fill(pf, retained)
    # forcing_deltas_rcp85.csv covers all 671 CAMELS-GB v2 catchments, including
    # the 50 dropped by the KGE screen. Panel (c) is the forcing context FOR
    # THIS STUDY's catchments, so restrict it to the retained set; otherwise the
    # medians drawn as dashed lines describe a population the figure never uses.
    fd = fd[fd["gauge_id"].astype(int).isin(retained)].copy()
    if len(fd) != len(retained):
        raise ValueError(f"forcing table covers {len(fd)} of "
                         f"{len(retained)} retained catchments")
    return pf, rt, fd


def _retained_gauges(params_csv: str, kge_min: float) -> set[int]:
    """The KGE-screened study population (621 catchments)."""
    cp = pd.read_csv(params_csv)
    need = {"gauge_id", "validation_kge"}
    missing = need - set(cp.columns)
    if missing:
        raise KeyError(f"{params_csv} is missing columns: {sorted(missing)}")
    return set(cp.loc[cp["validation_kge"] >= kge_min, "gauge_id"].astype(int))


def _zero_fill(pf: pd.DataFrame, retained: set[int]) -> pd.DataFrame:
    """Add explicit zero rows for retained catchments absent from the delta
    table, for every (rcp, direction) cell. A catchment with no detected
    transitions has a baseline frequency of zero and a change of zero."""
    have = set(pf["gauge_id"].astype(int))
    missing = sorted(retained - have)
    if not missing:
        return pf
    cells = pf[["rcp", "direction"]].drop_duplicates()
    rows = [dict(gauge_id=str(g), rcp=r.rcp, direction=r.direction,
                 d_freq_decade=0.0, base_freq_decade=0.0)
            for g in missing for r in cells.itertuples(index=False)]
    return pd.concat([pf, pd.DataFrame(rows)], ignore_index=True)


def carrier_mask(d: pd.DataFrame) -> pd.Series:
    """freq_summary()'s carrier definition (projection_analysis.py): a
    catchment counts if it had a baseline slow tail OR gained one in the
    future. This is the population behind the manuscript's carrier n and
    %-increasing statistics (Section 3.4) -- see module docstring note 1."""
    return (d["base_freq_decade"] > 0) | (d["d_freq_decade"] != 0)


def summarise(pf: pd.DataFrame, rt: pd.DataFrame, fd: pd.DataFrame) -> None:
    print("  --- (a) per-catchment frequency deltas ---")
    for direction in ("FTD", "DTF"):
        d = pf[pf["direction"] == direction]
        for rcp in RCPS:
            sub = d[d["rcp"] == rcp]
            carrier = sub[carrier_mask(sub)]
            pos = int((carrier["d_freq_decade"] > 0).sum())
            neg = int((carrier["d_freq_decade"] < 0).sum())
            pop_mean = sub["d_freq_decade"].mean()  # all 621, zero-filled -> Table 1
            print(f"    {direction} {rcp}: n_carrier={len(carrier)}, "
                  f"% increasing (of carriers)={100*pos/max(pos+neg,1):.1f}%, "
                  f"pop. mean delta (n=621, zero-filled)={pop_mean:.3f}")
    print("    [manuscript Section 3.4: carrier n = 94-103; 75-81% increasing; "
          "mean change 0.055 to 0.097, which is the population mean above, "
          "not a carrier mean. See docstring note 1.]")

    print("  --- (b) BFI-tertile stratification, 79-catchment strict-carrier "
          "population (= responder_table.parquet) ---")
    r = rt.copy()
    r["bfi_tertile"] = pd.qcut(r["baseflow_index"], 3, labels=["low", "mid", "high"])
    g = r.groupby("bfi_tertile", observed=True)["d_freq_decade"].agg(["mean", "median", "count"])
    print(g.to_string())
    print("    [manuscript Section 3.4 stratifies all 621 catchments instead: "
          "high mean 0.244 (median 0), mid 0.016, low 0.032. Broader "
          "population than the 79 carriers here; see docstring note 2.]")
    print("    group sizes:", r["group"].value_counts().to_dict())

    print("  --- (c) national forcing context (RCP8.5, 2080s) ---")
    print(f"    median dT   = {fd['d_tas_mean'].median():.2f} degC   "
          f"(621 retained; manuscript +3.19)")
    print(f"    median dAI  = {fd['d_aridity'].median():.2f}        "
          f"(621 retained; manuscript +0.19)")
    print(f"    median dP_s = {fd['d_p_summer'].median():.0f} mm      "
          f"(621 retained; manuscript -56)")


def make_figure(pf: pd.DataFrame, rt: pd.DataFrame, fd: pd.DataFrame,
                outdir: str) -> None:
    fig = plt.figure(figsize=(S.W_2COL, 66 * S.MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.1], wspace=0.5)
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1])

    # ---- Panel (a): FTD vs DTF frequency-delta distributions across RCP ---
    box_data, positions, colors = [], [], []
    pop_means = []   # population-wide (n=621, zero-filled) mean -> matches Table 1
    width = 0.34
    for i, rcp in enumerate(RCPS):
        for j, (direction, c) in enumerate((("FTD", S.C_SLOW), ("DTF", S.C_FAST))):
            sub = pf[(pf["direction"] == direction) & (pf["rcp"] == rcp)]
            d = sub[carrier_mask(sub)]
            box_data.append(d["d_freq_decade"].to_numpy())
            pos = i + (j - 0.5) * width * 1.15
            positions.append(pos)
            colors.append(c)
            pop_means.append((pos, sub["d_freq_decade"].mean()))
    bp = axA.boxplot(box_data, positions=positions, widths=width, patch_artist=True,
                     showfliers=False, medianprops=dict(color="black", lw=1.0),
                     whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7),
                     boxprops=dict(lw=0.6))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
        patch.set_edgecolor("0.25")
    mx, my = zip(*pop_means)
    axA.scatter(mx, my, marker="D", s=16, facecolor="black", edgecolor="white",
               linewidth=0.5, zorder=6)
    axA.axhline(0, color="0.6", lw=0.6, zorder=0)
    axA.set_xticks(range(len(RCPS)))
    axA.set_xticklabels([S.RCP_LABELS[r].replace("RCP", "") for r in RCPS])
    axA.set_xlabel("RCP")
    axA.set_ylabel(r"$\Delta$ frequency (slow transitions decade$^{-1}$)")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=S.C_SLOW, alpha=0.75,
                             edgecolor="0.25", label="FTD (carriers)"),
               plt.Rectangle((0, 0), 1, 1, facecolor=S.C_FAST, alpha=0.75,
                             edgecolor="0.25", label="DTF (carriers)"),
               plt.Line2D([0], [0], marker="D", color="none", markerfacecolor="black",
                         markeredgecolor="white", markersize=4.5,
                         label="Population mean\n(n=621, zero-filled)")]
    # Placed below the axes -- "upper left" sits inside the axes at a height
    # (~85-95% of the y-range) that the RCP4.5 FTD whisker (the tallest,
    # reaching ~1.45) passes directly through. Below rather than above so it
    # matches panel (b)'s legend: the two sit side by side, and legends on
    # opposite sides of the same row read as an inconsistency.
    S.legend_below(axA, handles=handles, fontsize=6, handlelength=1.2,
                  borderaxespad=0.3, labelspacing=0.5)
    S.panel_label(axA, "a")

    # ---- Panel (b): diffuseness -- delta vs BFI, RCP8.5, coloured by group -
    for grp in GROUP_ORDER:
        sub = rt[rt["group"] == grp]
        axB.scatter(sub["baseflow_index"], sub["d_freq_decade"],
                   s=12, color=GROUP_COLORS[grp], alpha=0.8,
                   edgecolor="none", label=grp.replace("_", " "))
    # tertile boundaries + means, computed within this carrier population
    r = rt.copy()
    r["bfi_tertile"], edges = pd.qcut(r["baseflow_index"], 3, labels=["low", "mid", "high"],
                                      retbins=True)
    for e in edges[1:-1]:
        axB.axvline(e, color="0.75", lw=0.6, ls=(0, (2, 2)), zorder=0)
    tmeans = r.groupby("bfi_tertile", observed=True)["d_freq_decade"].mean()
    xs = [(edges[i] + edges[i + 1]) / 2 for i in range(3)]
    ymax = max(r["d_freq_decade"].max(), 0.1) * 1.12
    for x, (lab, m) in zip(xs, tmeans.items()):
        axB.plot([x - (edges[1]-edges[0])*0.3, x + (edges[1]-edges[0])*0.3],
                [ymax * 0.95] * 2, color="black", lw=1.3)
        axB.text(x, ymax * 1.0, f"{m:.2f}", ha="center", va="bottom", fontsize=6.5)
    axB.set_ylim(top=ymax * 1.12)
    axB.set_xlabel("Baseline baseflow index")
    axB.set_ylabel(r"$\Delta$ frequency, RCP8.5 (decade$^{-1}$)")
    # Placed below the axes -- the non-responder/middle/responder scatter
    # fills most of the panel (including the lower-right corner, where grey
    # and green points sit right where an inside-axes legend would go), so
    # there is no corner left that is reliably clear of data.
    S.legend_below(axB, fontsize=6, markerscale=1.3, handletextpad=0.3,
                  borderaxespad=0.3)
    S.panel_label(axB, "b")

    # ---- Panel (c): national drying context, three mini-histograms --------
    gsC = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2], wspace=0.45)
    specs = [("d_tas_mean", "\u0394T (\u00b0C)", S.OKABE_ITO["vermil"]),
            ("d_aridity", "\u0394 aridity", S.OKABE_ITO["orange"]),
            ("d_p_summer", "\u0394$P_s$ (mm)", S.C_FAST)]
    axC_first = None
    fmt = {"d_tas_mean": "+{:.2f}", "d_aridity": "+{:.2f}", "d_p_summer": "{:.0f}"}
    medians = []          # (axes, median, label) -- annotated after the loop
    for k, (col, xlab, c) in enumerate(specs):
        axc = fig.add_subplot(gsC[k], sharey=axC_first)
        if axC_first is None:
            axC_first = axc
        vals = fd[col].dropna().to_numpy()
        axc.hist(vals, bins=24, color=c, alpha=0.8, edgecolor="none")
        med = float(np.median(vals))
        medians.append((axc, med, fmt[col].format(med)))
        axc.set_xlabel(xlab, fontsize=6.5)
        # Two ticks per panel: at this width three labels ran together
        # (e.g. "2.8 3.2 3.6" printing as "2.83.23.6"). The median is still
        # annotated directly on its line.
        axc.xaxis.set_major_locator(mticker.MaxNLocator(nbins=2, prune="both"))
        axc.tick_params(axis="x", labelsize=5.5, pad=1.5, length=2)
        if k == 0:
            axc.set_ylabel("Catchments")
            S.panel_label(axc, "c", x=-0.35)
        else:
            axc.tick_params(labelleft=False, left=False)
            axc.spines["left"].set_visible(False)

    # Median markers, drawn only once every histogram is in place. Two reasons
    # this happens here rather than inside the loop above:
    #   * the three panels share a y-axis, so the shared limit is only final
    #     now -- annotating as we went put each label at a different height,
    #     since each later, taller histogram pushed the limit up again;
    #   * the dashed line is drawn to the top of the data and no further, with
    #     the label in the headroom above it. A full-height axvline ran
    #     straight through the number, striking it out.
    top = axC_first.get_ylim()[1]
    for axc, med, label in medians:
        axc.plot([med, med], [0, top], color="black", lw=1.0, ls=(0, (3, 2)))
        axc.text(med, top * 1.06, label, ha="center", va="bottom", fontsize=6.3)
    axC_first.set_ylim(top=top * 1.26)

    S.save(fig, Path(outdir) / "fig05_projection")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projection", default="projection_flow.parquet")
    ap.add_argument("--responder", default="derived_output/responder_table.parquet")
    ap.add_argument("--forcing", default="forcing_deltas_rcp85.csv")
    ap.add_argument("--params", default="calibrated_parameters.csv",
                    help="calibrated parameters + validation_kge; defines the "
                         "621-catchment study population that the per-catchment "
                         "deltas are zero-filled onto")
    ap.add_argument("--kge-min", type=float, default=0.5,
                    help="validation-KGE screen (manuscript Section 2.1)")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    S.set_style()
    pf, rt, fd = load(args.projection, args.responder, args.forcing,
                      args.params, args.kge_min)
    print("Figure 5 -- projected change under warming")
    summarise(pf, rt, fd)
    make_figure(pf, rt, fd, args.outdir)


if __name__ == "__main__":
    main()
