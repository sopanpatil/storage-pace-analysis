#!/usr/bin/env python
"""
coherence_filter_diagnostics.py
===============================
Diagnostics for what the coherence filter does to the store attribution and to
the shape of the gap distribution (manuscript Sections 2.4, 2.5, 3.1 and 3.2).
These are provenance tables for the reported shares, not inputs to any figure
or Supporting Information text.

Why this exists
---------------
The coherence ratio C is evaluated ON the rate-limiting store that the flow
attribution has already selected (Section 2.4 of the manuscript). Filter and
attribution are therefore not independent, and the headline "99.5 % of slow FTD
transitions are lower-zone limited" is a statement about the retained
population, not about all candidate slow transitions. A referee who reruns the
pipeline will compute the unfiltered share in minutes, so it is worth having to
hand. This script computes it, together with:

  (1) attributed-store composition of slow (> 90 d) FTD candidates, before and
      after the C >= 0.60 filter, and the per-store retention rates that drive
      the difference;
  (2) the lower-zone share as a function of gap band, filtered and unfiltered,
      which is the unpooled version of Figure 3a;
  (3) baseline -> future exceedance fractions at a ladder of gap thresholds.
      These show the 90-150 day band expanding under RCP8.5 while the far tail
      (> 270 d) contracts. The manuscript does not report the far-tail
      behaviour; it is recorded here because it bears on the composition of
      the conditional-gap sample (see bootstrap_conditional_gap.py).

The physical reading is in the manuscript: with a median e-folding timescale of
2.9 days the upper zone cannot trace a near-monotone path over 90+ days, so a
UZ-attributed slow candidate is close to guaranteed to fail C >= 0.60. That is
the filter behaving as designed, but it has to be stated rather than left for a
reader to find.

Input
    slow_full_flow.parquet  (per-transition table from slow_transition_analysis.py)
    Required columns: gap_days, direction, period, rcp, rate_limiting_store,
                      passes_coherence, gauge_id

Usage
    python coherence_filter_diagnostics.py --input slow_full_flow.parquet
    python coherence_filter_diagnostics.py --input slow_full_flow.parquet \
        --outdir derived_output
    python coherence_filter_diagnostics.py            # synthetic self-test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RCPS = ("rcp26", "rcp45", "rcp60", "rcp85")
CUTOFF = 90                      # conventional abrupt/slow window (days)
GAP_BANDS = [0, 10, 20, 30, 40, 50, 60, 75, 90, 120, 180, 730]
TAIL_THRESHOLDS = [90, 120, 150, 180, 210, 270, 365]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load(parquet: str, direction: str = "FTD") -> pd.DataFrame:
    df = pd.read_parquet(parquet)
    need = {"gap_days", "direction", "period", "rcp", "rate_limiting_store",
            "passes_coherence", "gauge_id"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"{parquet} is missing columns: {sorted(missing)}")
    df = df[df["direction"] == direction].copy()
    df["gap_days"] = df["gap_days"].astype(float)
    return df


def _store_share(sub: pd.DataFrame, store: str) -> float:
    """Share of `sub` attributed to `store`, normalised over UZ+LZ only.

    Rows with rate_limiting_store 'none' (no net change in either candidate
    store) are excluded from the denominator, matching the normalisation used
    for the manuscript's 68.2/31.8 and 99.5/0.5 splits.
    """
    att = sub["rate_limiting_store"]
    denom = int(((att == "UZ") | (att == "LZ")).sum())
    if denom == 0:
        return np.nan
    return float((att == store).sum()) / denom


# --------------------------------------------------------------------------
# (1) attribution before and after the filter
# --------------------------------------------------------------------------
def attribution_before_after(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, mask in (("slow (> 90 d)", df["gap_days"] > CUTOFF),
                         ("abrupt (<= 90 d)", df["gap_days"] <= CUTOFF)):
        sub = df[mask]
        for label, s in (("all candidates", sub),
                         ("retained (C >= 0.60)", sub[sub["passes_coherence"]]),
                         ("rejected", sub[~sub["passes_coherence"]])):
            rows.append(dict(
                regime=regime, population=label, n=len(s),
                uz_share=_store_share(s, "UZ"),
                lz_share=_store_share(s, "LZ"),
            ))
    return pd.DataFrame(rows)


def retention_by_store(df: pd.DataFrame) -> pd.DataFrame:
    """Per-attributed-store retention rate among slow candidates.

    This is the quantity that makes the filter and the attribution dependent:
    if retention differs by store, the retained composition is not the
    candidate composition.
    """
    slow = df[df["gap_days"] > CUTOFF]
    g = slow.groupby("rate_limiting_store")["passes_coherence"]
    out = g.agg(n_candidates="size", n_retained="sum",
                retention_rate="mean").reset_index()
    return out.sort_values("n_candidates", ascending=False)


# --------------------------------------------------------------------------
# (2) lower-zone share as a function of gap band
# --------------------------------------------------------------------------
def lz_share_by_band(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lo, hi in zip(GAP_BANDS[:-1], GAP_BANDS[1:]):
        band = df[(df["gap_days"] > lo) & (df["gap_days"] <= hi)]
        rows.append(dict(
            gap_lo=lo, gap_hi=hi,
            n_all=len(band),
            lz_share_all=_store_share(band, "LZ"),
            n_coherent=int(band["passes_coherence"].sum()),
            lz_share_coherent=_store_share(band[band["passes_coherence"]], "LZ"),
        ))
    return pd.DataFrame(rows)


def lz_crossover(tbl: pd.DataFrame, col: str) -> float:
    """Gap (days) at which the LZ share first exceeds 50 %, by linear
    interpolation on band midpoints. A compact scalar summary of the mechanistic
    changeover that Figure 3a shows as a curve; not quoted in the manuscript."""
    mid = 0.5 * (tbl["gap_lo"] + tbl["gap_hi"])
    y = tbl[col].to_numpy(dtype=float)
    ok = np.isfinite(y)
    mid, y = mid.to_numpy()[ok], y[ok]
    above = np.nonzero(y > 0.5)[0]
    if above.size == 0 or above[0] == 0:
        return float("nan")
    i = above[0]
    x0, x1, y0, y1 = mid[i - 1], mid[i], y[i - 1], y[i]
    return float(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0))


# --------------------------------------------------------------------------
# (3) tail exceedance, baseline vs each future RCP
# --------------------------------------------------------------------------
def tail_exceedance(df: pd.DataFrame) -> pd.DataFrame:
    c = df[df["passes_coherence"]]
    base = c.loc[c["period"] == "baseline", "gap_days"].to_numpy()
    futs = {r: c.loc[(c["period"] == "future") & (c["rcp"] == r),
                     "gap_days"].to_numpy() for r in RCPS}
    rows = []
    for t in TAIL_THRESHOLDS:
        row = {"threshold_d": t, "baseline": float(np.mean(base > t))}
        row.update({r: float(np.mean(v > t)) for r, v in futs.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def tail_quantiles(df: pd.DataFrame) -> pd.DataFrame:
    c = df[df["passes_coherence"]]
    rows = []
    pops = [("baseline", c.loc[c["period"] == "baseline", "gap_days"].to_numpy())]
    pops += [(r, c.loc[(c["period"] == "future") & (c["rcp"] == r),
                       "gap_days"].to_numpy()) for r in RCPS]
    for name, g in pops:
        if len(g) == 0:
            continue
        rows.append(dict(population=name, n=len(g),
                         median=float(np.median(g)),
                         p90=float(np.percentile(g, 90)),
                         p99=float(np.percentile(g, 99)),
                         maximum=float(np.max(g))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def report(df: pd.DataFrame, outdir: str | None) -> None:
    line = "=" * 70

    ab = attribution_before_after(df)
    print(line + "\n(1) ATTRIBUTION BEFORE AND AFTER THE COHERENCE FILTER")
    for regime, grp in ab.groupby("regime", sort=False):
        print(f"  {regime}")
        for _, r in grp.iterrows():
            print(f"    {r['population']:22s} n={r['n']:8,d}  "
                  f"UZ={r['uz_share']*100:6.2f}%  LZ={r['lz_share']*100:6.2f}%")

    ret = retention_by_store(df)
    print("\n  retention rate among slow (> 90 d) candidates, by attributed store")
    for _, r in ret.iterrows():
        print(f"    {r['rate_limiting_store']:6s} n={r['n_candidates']:8,d}  "
              f"retained={int(r['n_retained']):7,d}  "
              f"rate={r['retention_rate']*100:6.3f}%")
    rr = ret.set_index("rate_limiting_store")["retention_rate"]
    if {"LZ", "UZ"} <= set(rr.index) and rr["UZ"] > 0:
        print(f"    differential retention LZ/UZ = {rr['LZ']/rr['UZ']:.1f}x")

    bands = lz_share_by_band(df)
    print("\n" + line + "\n(2) LOWER-ZONE SHARE BY GAP BAND (FTD)")
    print(f"  {'band (d)':>12}  {'n all':>8} {'LZ% all':>8}   "
          f"{'n coh':>7} {'LZ% coh':>8}")
    for _, r in bands.iterrows():
        print(f"  {int(r['gap_lo']):4d}-{int(r['gap_hi']):4d}  "
              f"{int(r['n_all']):8,d} {r['lz_share_all']*100:8.1f}   "
              f"{int(r['n_coherent']):7,d} {r['lz_share_coherent']*100:8.1f}")
    print(f"  LZ share crosses 50% at "
          f"{lz_crossover(bands, 'lz_share_coherent'):.0f} d (coherent) / "
          f"{lz_crossover(bands, 'lz_share_all'):.0f} d (all candidates)")

    tail = tail_exceedance(df)
    print("\n" + line + "\n(3) TAIL EXCEEDANCE, COHERENT FTD")
    print(f"  {'gap >':>7} {'baseline':>9} " +
          " ".join(f"{r:>9}" for r in RCPS))
    for _, r in tail.iterrows():
        print(f"  {int(r['threshold_d']):7d} {r['baseline']:9.4f} " +
              " ".join(f"{r[c]:9.4f}" for c in RCPS))
    tq = tail_quantiles(df)
    print("\n  " + tq.to_string(index=False).replace("\n", "\n  "))

    if outdir:
        d = Path(outdir)
        d.mkdir(parents=True, exist_ok=True)
        ab.to_csv(d / "coherence_attribution_before_after.csv", index=False)
        ret.to_csv(d / "coherence_retention_by_store.csv", index=False)
        bands.to_csv(d / "lz_share_by_gap_band.csv", index=False)
        tail.to_csv(d / "tail_exceedance.csv", index=False)
        tq.to_csv(d / "tail_quantiles.csv", index=False)
        print(f"\n  wrote 5 CSVs to {d}/")


# --------------------------------------------------------------------------
def _self_test() -> None:
    """Synthetic check: a filter that preferentially rejects UZ-attributed slow
    transitions must shift the retained composition toward LZ."""
    rng = np.random.default_rng(0)
    n = 4000
    gap = rng.integers(5, 400, n).astype(float)
    store = np.where(rng.random(n) < 0.3, "UZ", "LZ")
    # UZ-attributed slow transitions almost never pass; LZ ones sometimes do
    p = np.where((gap > CUTOFF) & (store == "UZ"), 0.01,
                 np.where(gap > CUTOFF, 0.30, 0.50))
    df = pd.DataFrame(dict(
        gap_days=gap, direction="FTD", rate_limiting_store=store,
        passes_coherence=rng.random(n) < p, gauge_id="x",
        period=rng.choice(["baseline", "future"], n),
        rcp=rng.choice(RCPS, n),
    ))
    ab = attribution_before_after(df)
    slow = ab[ab["regime"].str.startswith("slow")].set_index("population")
    assert slow.loc["retained (C >= 0.60)", "lz_share"] > \
        slow.loc["all candidates", "lz_share"], "filter should enrich LZ"
    print("self-test OK: differential retention enriches the retained "
          "population in LZ, as in the production data")
    report(df, outdir=None)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=None,
                    help="per-transition parquet from slow_transition_analysis.py")
    ap.add_argument("--direction", default="FTD", choices=["FTD", "DTF"])
    ap.add_argument("--outdir", default="derived_output",
                    help="directory for the CSV summaries (omit with --no-write)")
    ap.add_argument("--no-write", action="store_true",
                    help="print only, do not write CSVs")
    args = ap.parse_args()

    if args.input is None:
        _self_test()
        return
    df = load(args.input, args.direction)
    report(df, outdir=None if args.no_write else args.outdir)


if __name__ == "__main__":
    main()
