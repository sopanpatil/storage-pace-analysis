#!/usr/bin/env python3
"""
verify_manuscript_numbers.py

Recompute every headline Results number from the regenerated (logged-flux)
transition table, using the EXACT reductions in projection_analysis.py
(population_shift) and slow_transition_analysis.py (attribution), and print each
beside the value currently in the manuscript with a MATCH / CHECK flag.

Definitions reproduced verbatim:
  * load: keep rows where passes_coherence is True.
  * population_shift: per direction/RCP/period, gap_days median, quantile(0.90),
    and share with gap_days > 90.  (projection_analysis.py)
  * mechanism: among coherent FTD, rate_limiting_store shares for slow and for
    abrupt regimes (normalised over UZ+LZ).  (slow_transition_analysis.py)

The transition table this reads is the production, 730-day-capped one
(manuscript Section 2.3). Running it against an uncensored table will report
CHECK on the counts and the baseline slow share.

Table 1 CONFIDENCE INTERVALS are NOT recomputed here -- they come from
bootstrap_conditional_gap.py. This script gives the Table 1 POINT values
(future median/p90/slowshare and d_median/d_slowshare); refresh the CIs from
your regenerated bootstrap_conditional_gap.csv.

Usage:
    python verify_manuscript_numbers.py --input slow_full_flow.parquet
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

RCPS = ["rcp26", "rcp45", "rcp60", "rcp85"]
ABRUPT_CUTOFF = 90

# ---- manuscript values, for side-by-side (edit here if the draft changes) --- #
MS = {
    "count_total": 290007,
    "count_ftd": 124600, "count_dtf": 165407,
    "ftd_base_median": 13.0, "ftd_base_p90": 56.0, "ftd_base_slow": 0.049,
    "ftd_fut_p90_range": (86, 91), "ftd_fut_slow_range": (0.092, 0.101),
    "dtf_base_median": 4.0, "dtf_base_p90": 22.0, "dtf_base_slow": 0.006,
    "slow_ftd_lz": 0.995, "abrupt_ftd_uz": 0.682, "abrupt_ftd_lz": 0.318,
    # Table 1 future point values (median / p90 / slowshare) per RCP
    "tbl_future": {"rcp26": (15, 86, 0.092), "rcp45": (15, 89, 0.096),
                   "rcp60": (16, 87, 0.092), "rcp85": (21, 91, 0.101)},
    "tbl_dslow": {"rcp26": 0.043, "rcp45": 0.047, "rcp60": 0.043, "rcp85": 0.052},
    "tbl_dmedian": {"rcp26": 2, "rcp45": 2, "rcp60": 3, "rcp85": 8},
}


def flag(recomputed, manuscript, tol):
    try:
        return "MATCH" if abs(recomputed - manuscript) <= tol else ">>> CHECK"
    except TypeError:
        return ""


def pop_shift(tr, direction):
    d = tr[tr["direction"] == direction]
    rows = []
    for rcp in RCPS:
        rec = {"rcp": rcp}
        for period in ("baseline", "future"):
            g = d[(d["rcp"] == rcp) & (d["period"] == period)]["gap_days"]
            rec[f"{period}_n"] = len(g)
            rec[f"{period}_median"] = round(g.median(), 1) if len(g) else np.nan
            rec[f"{period}_p90"] = round(g.quantile(0.9), 1) if len(g) else np.nan
            rec[f"{period}_slow"] = round((g > ABRUPT_CUTOFF).mean(), 3) if len(g) else np.nan
        rec["d_median"] = round(rec["future_median"] - rec["baseline_median"], 1)
        rec["d_slow"] = round(rec["future_slow"] - rec["baseline_slow"], 3)
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    tr = pd.read_parquet(args.input)
    tr = tr[tr["passes_coherence"]].copy()
    tr["gauge_id"] = tr["gauge_id"].astype(str)

    print("=" * 70)
    print("COUNTS")
    n = len(tr); nf = (tr["direction"] == "FTD").sum(); nd = (tr["direction"] == "DTF").sum()
    print(f"  coherent transitions total : {n:>8,}   ms {MS['count_total']:>8,}   {flag(n, MS['count_total'], 0)}")
    print(f"  coherent FTD               : {nf:>8,}   ms {MS['count_ftd']:>8,}   {flag(nf, MS['count_ftd'], 0)}")
    print(f"  coherent DTF               : {nd:>8,}   ms {MS['count_dtf']:>8,}   {flag(nd, MS['count_dtf'], 0)}")

    ftd = pop_shift(tr, "FTD"); dtf = pop_shift(tr, "DTF")

    print("\n" + "=" * 70)
    print("CONTINUUM  (baseline shared across RCPs -- check base_n identical)")
    print("  FTD baseline_n per RCP:", list(ftd["baseline_n"]))
    bm, bp, bs = ftd.iloc[0][["baseline_median", "baseline_p90", "baseline_slow"]]
    print(f"  FTD baseline median : {bm:>6}   ms {MS['ftd_base_median']}   {flag(bm, MS['ftd_base_median'], 0.5)}")
    print(f"  FTD baseline p90    : {bp:>6}   ms {MS['ftd_base_p90']}   {flag(bp, MS['ftd_base_p90'], 1.0)}")
    print(f"  FTD baseline slow%  : {bs:>6}   ms {MS['ftd_base_slow']}   {flag(bs, MS['ftd_base_slow'], 0.002)}")
    fp90 = (ftd["future_p90"].min(), ftd["future_p90"].max())
    fsl  = (ftd["future_slow"].min(), ftd["future_slow"].max())
    print(f"  FTD future p90 range: {fp90}   ms {MS['ftd_fut_p90_range']}")
    print(f"  FTD future slow rng : {fsl}   ms {MS['ftd_fut_slow_range']}")
    dm, dp, ds = dtf.iloc[0][["baseline_median", "baseline_p90", "baseline_slow"]]
    print(f"  DTF baseline median : {dm:>6}   ms {MS['dtf_base_median']}   {flag(dm, MS['dtf_base_median'], 0.5)}")
    print(f"  DTF baseline p90    : {dp:>6}   ms {MS['dtf_base_p90']}   {flag(dp, MS['dtf_base_p90'], 1.0)}")
    print(f"  DTF baseline slow%  : {ds:>6}   ms {MS['dtf_base_slow']}   {flag(ds, MS['dtf_base_slow'], 0.002)}")

    print("\n" + "=" * 70)
    print("MECHANISM  (coherent FTD, rate_limiting_store; normalised over UZ+LZ)")
    fc = tr[(tr["direction"] == "FTD")]
    for regime, key_uz, key_lz in [("slow", None, "slow_ftd_lz"), ("abrupt", "abrupt_ftd_uz", "abrupt_ftd_lz")]:
        r = fc[fc["regime"] == regime]["rate_limiting_store"]
        r = r[r.isin(["UZ", "LZ"])]
        share = r.value_counts(normalize=True)
        lz = float(share.get("LZ", 0.0)); uz = float(share.get("UZ", 0.0))
        if regime == "slow":
            print(f"  slow   LZ share : {lz:.3f}   ms {MS['slow_ftd_lz']}   {flag(lz, MS['slow_ftd_lz'], 0.002)}   (n={len(r):,})")
        else:
            print(f"  abrupt UZ share : {uz:.3f}   ms {MS['abrupt_ftd_uz']}   {flag(uz, MS['abrupt_ftd_uz'], 0.005)}")
            print(f"  abrupt LZ share : {lz:.3f}   ms {MS['abrupt_ftd_lz']}   {flag(lz, MS['abrupt_ftd_lz'], 0.005)}   (n={len(r):,})")
    # ensemble-median slow LZ share (the 0.9958 figure)
    slow = tr[(tr["direction"]=="FTD") & (tr["regime"]=="slow")].copy()
    slow["is_lz"] = slow["rate_limiting_store"] == "LZ"
    em = slow.groupby(["rcp","member"])["is_lz"].mean().median()
    print(f"  slow LZ share (ensemble median) : {em:.4f}")

    print("\n" + "=" * 70)
    print("TABLE 1  (future point values; refresh CIs from bootstrap CSV)")
    print(f"  {'RCP':<6}{'fut_med':>8}{'fut_p90':>9}{'fut_slow':>10}{'d_med':>7}   vs manuscript")
    for _, row in ftd.iterrows():
        rcp = row["rcp"]; ms = MS["tbl_future"][rcp]; msd = MS["tbl_dmedian"][rcp]
        fm = flag(row["future_median"], ms[0], 0.5)
        fs = flag(row["future_slow"], ms[2], 0.0005)
        ds = flag(row["d_slow"], MS["tbl_dslow"][rcp], 0.0005)
        print(f"  {rcp:<6}{row['future_median']:>8}{row['future_p90']:>9}{row['future_slow']:>10}{row['d_median']:>7}"
              f"   ms {ms} d_med {msd}  med {fm}  slow {fs}  d_slow {ds}")

    print("\nNote: 99.6% runoff-norm cross-check is a SEPARATE run "
          "(--attribution runoff-norm) and is not recomputable from this "
          "flow-attribution table; refresh it only if you want that robustness "
          "figure updated.")


if __name__ == "__main__":
    main()
