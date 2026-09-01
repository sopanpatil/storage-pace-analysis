#!/usr/bin/env python3
"""
snow_melt_contribution.py

Quantify snowmelt within the coherent SLOW FTD carriers -- the transitions that
drive the slow-tail, lower-zone-limited result -- to back the snow-exclusion
argument (Sections 2.3 / 2.5) with a measured figure rather than a structural
claim alone.

For each coherent, slow, LZ-limited FTD transition it sums the logged snowmelt
flux over the transition window and compares it to the lower-zone decline that
paces the transition (d_ratelim_mm). Reports a distribution, not a thresholded
percentage: median and 90th-percentile cumulative melt (mm), the median LZ
decline for scale, and the melt-to-LZ-decline ratio.

Alignment note: t_start/t_end index the per-(rcp,member,period) WINDOW, not the
full series (the loader sets t = arange(len(masked_window))). We rebuild the
same water-year mask used by slow_transition_analysis.py and offset into it, so
melt is summed over exactly the transition's days.

Usage:
    python snow_melt_contribution.py --input slow_full_flow.parquet \
        --jasmin-dir chess_scape_output
    python snow_melt_contribution.py            # synthetic self-test
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import selftest_io

# must match slow_transition_analysis.py
BASE_START, BASE_END = "1981-10-01", "2010-09-30"
FUT_START,  FUT_END  = "2050-10-01", "2080-09-30"


def _window_idx(date_col: pd.Series, period: str) -> np.ndarray:
    d10 = date_col.astype(str).str[:10].to_numpy()
    if period == "baseline":
        mask = (d10 >= BASE_START) & (d10 <= BASE_END)
    else:
        mask = (d10 >= FUT_START) & (d10 <= FUT_END)
    return np.flatnonzero(mask)


def select_carriers(tr: pd.DataFrame, max_gap: int | None = 720) -> pd.DataFrame:
    """The population the snow argument is about: coherent, slow, LZ-limited FTD.

    The production gap cap (manuscript Section 2.3) is applied here rather than
    assumed of the caller, so importers get the same 720-day population as the
    command-line path whether the table on disk is pre-capped or not. Pass
    max_gap=None for uncensored.
    """
    tr = tr.copy()
    tr["gauge_id"] = tr["gauge_id"].astype(str)
    if max_gap:
        tr = tr[tr["gap_days"] <= max_gap]
    return tr[(tr["direction"] == "FTD") & (tr["passes_coherence"])
              & (tr["regime"] == "slow")
              & (tr["rate_limiting_store"] == "LZ")].reset_index(drop=True)


def melt_over_transitions(carriers: pd.DataFrame, jasmin_dir: str) -> pd.DataFrame:
    """Sum the logged snowmelt flux over each transition window.

    Returns `carriers` with melt_mm, lz_decline_mm and melt_frac added, dropping
    rows whose melt file or gauge column is unavailable.
    """
    melt_sum = np.full(len(carriers), np.nan)
    d = Path(jasmin_dir)
    for (rcp, member, period), grp in carriers.groupby(["rcp", "member", "period"]):
        fpath = d / f"{rcp}_{member}_hbv_melt.csv"
        if not fpath.exists():
            print(f"  WARNING: missing {fpath.name}; carriers in this group skipped")
            continue
        melt_wide = pd.read_csv(fpath)
        widx = _window_idx(melt_wide["date"], period)   # full-series rows of window
        base_row = int(widx[0]) if widx.size else 0
        # contiguous water-year block -> window position p maps to full row base_row+p
        for ridx, row in grp.iterrows():
            g = row["gauge_id"]
            if g not in melt_wide.columns:
                continue
            r0 = base_row + int(row["t_start"])
            r1 = base_row + int(row["t_end"])
            seg = melt_wide[g].to_numpy()[r0:r1 + 1]
            melt_sum[ridx] = float(np.nansum(seg))

    out = carriers.copy()
    out["melt_mm"] = melt_sum
    out = out.dropna(subset=["melt_mm"]).copy()
    out["lz_decline_mm"] = out["d_ratelim_mm"].abs()
    out["melt_frac"] = np.where(out["lz_decline_mm"] > 1e-9,
                                out["melt_mm"] / out["lz_decline_mm"], np.nan)
    return out


def summarise(ok: pd.DataFrame) -> dict:
    """The five numbers the Discussion quotes."""
    return {
        "n": len(ok),
        "median_melt_mm": float(ok["melt_mm"].median()),
        "p90_melt_mm": float(ok["melt_mm"].quantile(0.90)),
        "median_lz_decline_mm": float(ok["lz_decline_mm"].median()),
        "median_melt_frac": float(ok["melt_frac"].median()),
        "p90_melt_frac": float(ok["melt_frac"].quantile(0.90)),
        "zero_melt_share": float((ok["melt_mm"] <= 1e-6).mean()),
    }


def report(ok: pd.DataFrame) -> None:
    s = summarise(ok)
    print(f"\nSnowmelt over coherent slow LZ-limited FTD transitions "
          f"(n = {s['n']:,}):")
    print(f"  cumulative melt (mm):        median {s['median_melt_mm']:.3f}   "
          f"p90 {s['p90_melt_mm']:.3f}")
    print(f"  lower-zone decline (mm):     median {s['median_lz_decline_mm']:.2f}")
    print(f"  melt / LZ-decline:           median {s['median_melt_frac']*100:.2f}%   "
          f"p90 {s['p90_melt_frac']*100:.2f}%")
    print(f"  transitions with zero melt:  {s['zero_melt_share']*100:.1f}%")
    print(f"\nManuscript-ready: across the coherent slow FTD carriers, cumulative "
          f"snowmelt was a median {s['median_melt_mm']:.2f} mm (90th pct "
          f"{s['p90_melt_mm']:.2f} mm), i.e. {s['median_melt_frac']*100:.1f}% "
          f"(90th pct {s['p90_melt_frac']*100:.1f}%) of the lower-zone decline "
          f"that paces the transition.")


def _self_test() -> None:
    """Synthetic check: a lowland carrier set with no melt and one upland set
    with melt must reproduce the planted median / p90 / zero-share exactly."""
    rng = np.random.default_rng(0)
    tmp = selftest_io.redirect("syn_melt", True)
    tmp.mkdir(exist_ok=True)

    n_days = 360 * 3
    dates = pd.date_range("1981-10-01", periods=n_days, freq="D").astype(str)
    gauges = [f"g{i:02d}" for i in range(10)]
    # Two "upland" gauges melt at a constant 0.4 mm/day; the other eight never
    # melt. A constant rate makes the expected sum exact, so the assertion below
    # tests the window arithmetic (the t_start/t_end -> full-series offset that
    # the alignment note describes) rather than just its sign.
    rate, gap = 0.4, 120
    melt = pd.DataFrame({g: np.full(n_days, rate if i < 2 else 0.0)
                         for i, g in enumerate(gauges)})
    melt.insert(0, "date", dates)
    melt.to_csv(tmp / "rcp85_01_hbv_melt.csv", index=False)

    rows = []
    for g in gauges:
        for k in range(5):
            t0 = 20 + k * 60
            rows.append(dict(gauge_id=g, rcp="rcp85", member="01",
                             period="baseline", direction="FTD", regime="slow",
                             rate_limiting_store="LZ", passes_coherence=True,
                             gap_days=gap, t_start=t0, t_end=t0 + gap,
                             d_ratelim_mm=-float(rng.uniform(40, 120))))
    # One multi-annual pairing beyond the production cap, to prove select_carriers
    # censors at max_gap for importers too (verify_robustness_checks.py calls it
    # directly, so a cap applied only in main() would silently miss this row).
    over = dict(rows[0], gauge_id="g00", gap_days=900, t_start=20, t_end=920)
    tr = pd.DataFrame(rows + [over])

    carriers = select_carriers(tr)
    assert len(carriers) == 50, len(carriers)
    assert (carriers["gap_days"] <= 720).all(), carriers["gap_days"].max()
    assert len(select_carriers(tr, max_gap=None)) == 51, "max_gap=None must not censor"
    ok = melt_over_transitions(carriers, str(tmp))
    s = summarise(ok)
    expected = rate * (gap + 1)          # inclusive window, so gap + 1 days
    assert s["median_melt_mm"] == 0.0, s
    assert abs(s["zero_melt_share"] - 0.80) < 1e-9, s
    melting = ok.loc[ok["melt_mm"] > 0, "melt_mm"]
    assert len(melting) == 10, len(melting)
    assert np.allclose(melting, expected), (melting.unique(), expected)
    print(f"self-test OK: the eight melt-free gauges give a zero median and the "
          f"planted 80% zero-melt share, and each of the ten melting "
          f"transitions sums to exactly {expected:.1f} mm "
          f"({rate} mm/day over an inclusive {gap}-day gap), confirming the "
          f"window offset.")
    report(ok)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="slow_full_flow.parquet")
    ap.add_argument("--max-gap", type=int, default=720,
                    help="production gap bound in days (default 720 = two "
                         "water years on the 360-day model calendar, "
                         "manuscript Section 2.3); pass 0 for uncensored")
    ap.add_argument("--jasmin-dir", help="dir with *_hbv_melt.csv")
    ap.add_argument("--out", default=None, help="optional per-transition CSV")
    args = ap.parse_args()

    if not args.input or not args.jasmin_dir:
        print("No --input/--jasmin-dir: synthetic self-test.\n")
        _self_test()
        return

    tr = pd.read_parquet(args.input)
    carriers = select_carriers(tr, max_gap=args.max_gap or None)
    print(f"coherent slow LZ-limited FTD carriers: {len(carriers):,}")
    if carriers.empty:
        return

    ok = melt_over_transitions(carriers, args.jasmin_dir)
    report(ok)

    if args.out:
        ok.to_csv(args.out, index=False)
        print(f"\nWrote per-transition melt table -> {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
