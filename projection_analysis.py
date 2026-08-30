"""
projection_analysis.py

Baseline -> future change in flood-drought transition pace across the four
CHESS-SCAPE RCPs, from the transition table produced by
slow_transition_analysis.py.

This is the step that tests whether warming changes the transition regime -- the
only part of the analysis that uses the future projections rather than
characterising the present. It deliberately does NOT rely on per-catchment gap
medians (too sparse in the slow regime). It rests on:

  (1) POPULATION distribution shift  -- pooled coherent gaps, baseline vs future,
      per RCP (median / 90th / >90-day share). No per-catchment power needed.
  (2) Per-catchment slow-transition FREQUENCY change -- counts (robust), with
      zero-fill so a catchment that loses its slow transitions registers as a
      decrease. Ensemble-median of the per-member delta (consistent with the
      companion abrupt-regime pipeline, dfaa-analysis).
  (3) RCP GRADIENT -- does the change scale monotonically 2.6 -> 8.5? A monotone
      gradient is a climate signal; a non-monotone one is noise.

Significance is assessed at the POPULATION level (Wilcoxon signed-rank + sign
test across catchments, per RCP), which avoids the per-catchment
multiple-comparison problem rather than ignoring it.

Thresholds are the fixed baseline Q5/Q80 already baked into the transition
table, so future changes reflect distributional shifts, not moving definitions.

Usage:
    python projection_analysis.py --input slow_full_flow.parquet \
        --attr camels_gb_v2_hydrologic_attributes.csv
    python projection_analysis.py            # synthetic self-test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import selftest_io

try:
    from scipy.stats import wilcoxon, binomtest
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

RCPS = ["rcp26", "rcp45", "rcp60", "rcp85"]
MEMBERS = ["01", "04", "06", "15"]
BASE_WY, FUT_WY = 29, 30          # water years per window (fixed by definition)
ABRUPT_CUTOFF = 90               # days


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #
def load(parquet: str, max_gap: int | None = 720) -> pd.DataFrame:
    """Load the coherent transitions, censored at the production gap cap.

    Applied here rather than assumed of the input, so results sit on the same
    720-day production footing as the manuscript (Section 2.3) whether the
    table on disk is pre-capped or not. Pass max_gap=None for uncensored.
    """
    tr = pd.read_parquet(parquet)
    tr = tr[tr["passes_coherence"]].copy()
    if max_gap:
        tr = tr[tr["gap_days"] <= max_gap].copy()
    tr["gauge_id"] = tr["gauge_id"].astype(str)
    return tr


# --------------------------------------------------------------------------- #
# (1) Population distribution shift                                           #
# --------------------------------------------------------------------------- #
def population_shift(tr: pd.DataFrame, direction: str) -> pd.DataFrame:
    d = tr[tr["direction"] == direction]
    rows = []
    for rcp in RCPS:
        rec = {"rcp": rcp}
        for period in ("baseline", "future"):
            g = d[(d["rcp"] == rcp) & (d["period"] == period)]["gap_days"]
            rec[f"{period}_n"] = len(g)
            rec[f"{period}_median"] = round(g.median(), 1) if len(g) else np.nan
            rec[f"{period}_p90"] = round(g.quantile(0.9), 1) if len(g) else np.nan
            rec[f"{period}_slowshare"] = round((g > ABRUPT_CUTOFF).mean(), 3) if len(g) else np.nan
        rec["d_median"] = round(rec["future_median"] - rec["baseline_median"], 1)
        rec["d_slowshare"] = round(rec["future_slowshare"] - rec["baseline_slowshare"], 3)
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# (2) Per-catchment slow-frequency change (zero-filled, ensemble-median)      #
# --------------------------------------------------------------------------- #
def per_catchment_freq_delta(tr: pd.DataFrame, direction: str,
                             slow_only: bool = True) -> pd.DataFrame:
    """Return per (gauge, rcp) ensemble-median delta in slow transitions/decade."""
    d = tr[tr["direction"] == direction]
    if slow_only:
        d = d[d["regime"] == "slow"]
    gauges = sorted(tr["gauge_id"].unique())

    # counts on the observed rows
    cnt = (d.groupby(["gauge_id", "rcp", "member", "period"])
             .size().rename("n").reset_index())
    # full grid so absences become zeros
    grid = pd.MultiIndex.from_product(
        [gauges, RCPS, MEMBERS, ["baseline", "future"]],
        names=["gauge_id", "rcp", "member", "period"]).to_frame(index=False)
    cnt = grid.merge(cnt, how="left", on=["gauge_id", "rcp", "member", "period"])
    cnt["n"] = cnt["n"].fillna(0.0)
    wy = {"baseline": BASE_WY, "future": FUT_WY}
    cnt["per_decade"] = cnt["n"] / cnt["period"].map(wy) * 10.0

    # per-member delta, then ensemble median across members
    wide = cnt.pivot_table(index=["gauge_id", "rcp", "member"],
                           columns="period", values="per_decade").reset_index()
    wide["delta"] = wide["future"] - wide["baseline"]
    out = (wide.groupby(["gauge_id", "rcp"])["delta"].median()
               .rename("d_freq_decade").reset_index())
    base = (wide.groupby(["gauge_id", "rcp"])["baseline"].median()
                .rename("base_freq_decade").reset_index())
    return out.merge(base, on=["gauge_id", "rcp"])


def freq_summary(delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rcp in RCPS:
        sub = delta[delta["rcp"] == rcp]
        x = sub["d_freq_decade"].to_numpy()
        x = x[~np.isnan(x)]
        pos = int((x > 0).sum())
        neg = int((x < 0).sum())
        # catchments that actually carry a slow transition in either period
        carrier = sub[(sub["base_freq_decade"] > 0) | (sub["d_freq_decade"] != 0)]
        nz = carrier["d_freq_decade"].to_numpy()
        rec = {"rcp": rcp, "n_catch": len(x), "n_carrier": len(nz),
               "mean_delta": round(float(np.mean(x)), 3),
               "median_delta_carrier": round(float(np.median(nz)), 3) if len(nz) else np.nan,
               "pct_increasing": round(pos / max(pos + neg, 1), 3)}
        if _HAVE_SCIPY and (pos + neg) >= 10:
            xnz = x[x != 0]
            rec["wilcoxon_p"] = round(float(wilcoxon(xnz).pvalue), 4) if len(xnz) else np.nan
            rec["sign_p"] = round(float(binomtest(pos, pos + neg).pvalue), 4)
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# (3) RCP gradient                                                            #
# --------------------------------------------------------------------------- #
def rcp_gradient(freq_sum: pd.DataFrame, pop: pd.DataFrame) -> str:
    inc = freq_sum.set_index("rcp")["pct_increasing"].reindex(RCPS).to_numpy()
    mn = freq_sum.set_index("rcp")["mean_delta"].reindex(RCPS).to_numpy()
    ps = pop.set_index("rcp")["d_slowshare"].reindex(RCPS).to_numpy()

    def mono(v):
        v = v[~np.isnan(v)]
        if len(v) < 2:
            return "n/a"
        if np.all(np.diff(v) >= -1e-9):
            return "monotone-up"
        if np.all(np.diff(v) <= 1e-9):
            return "monotone-down"
        return "non-monotone"

    return (f"  %-increasing gradient (2.6->8.5): {inc.round(3).tolist()}  [{mono(inc)}]\n"
            f"  mean-delta  gradient (2.6->8.5): {mn.round(3).tolist()}  [{mono(mn)}]\n"
            f"  slow-share  gradient (2.6->8.5): {ps.round(3).tolist()}  [{mono(ps)}]")


# --------------------------------------------------------------------------- #
# BFI stratification (optional)                                               #
# --------------------------------------------------------------------------- #
def bfi_stratify(delta: pd.DataFrame, attr_src: str, rcp: str = "rcp85") -> str:
    src = Path(attr_src)
    raw = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)
    hy = raw[["gauge_id", "baseflow_index"]].drop_duplicates("gauge_id").copy()
    hy["gauge_id"] = hy["gauge_id"].astype(str)
    m = delta[delta["rcp"] == rcp].merge(hy, on="gauge_id")
    if m.empty:
        return "  (no overlap with attribute table)"
    m["bfi_tertile"] = pd.qcut(m["baseflow_index"], 3, duplicates="drop")
    g = (m.groupby("bfi_tertile", observed=True)["d_freq_decade"]
           .agg(["size", "mean", "median"]).round(3))
    return g.to_string()


# --------------------------------------------------------------------------- #
# Report                                                                      #
# --------------------------------------------------------------------------- #
def report(tr: pd.DataFrame, attr_csv: str | None) -> pd.DataFrame:
    all_delta = []
    for direction in ("FTD", "DTF"):
        print(f"\n{'='*66}\n{direction}\n{'='*66}")
        pop = population_shift(tr, direction)
        print("\n(1) Population gap distribution, baseline -> future (pooled, coherent):")
        print(pop.to_string(index=False))

        delta = per_catchment_freq_delta(tr, direction, slow_only=True)
        delta["direction"] = direction
        all_delta.append(delta)
        fs = freq_summary(delta)
        print("\n(2) Per-catchment SLOW-frequency change (transitions/decade, "
              "ensemble-median delta):")
        print(fs.to_string(index=False))

        print("\n(3) RCP gradient:")
        print(rcp_gradient(fs, pop))

        if attr_csv:
            print("\n    SLOW-frequency delta by baseline-BFI tertile (RCP8.5):")
            print(bfi_stratify(delta, attr_csv, "rcp85"))
    return pd.concat(all_delta, ignore_index=True)


# --------------------------------------------------------------------------- #
# Synthetic self-test                                                         #
# --------------------------------------------------------------------------- #
def _make_synthetic_parquet(path: str) -> None:
    """Plant an RCP-scaled increase in slow FTD frequency; DTF flat (control)."""
    rng = np.random.default_rng(0)
    gauges = [f"g{i:03d}" for i in range(80)]
    rcp_boost = {"rcp26": 0.2, "rcp45": 0.6, "rcp60": 1.0, "rcp85": 1.8}
    rows = []
    for g in gauges:
        aquifer = rng.random() < 0.3       # 30% high-storage catchments
        for rcp in RCPS:
            for m in MEMBERS:
                for period in ("baseline", "future"):
                    for direction in ("FTD", "DTF"):
                        base_rate = 3.0
                        if direction == "FTD" and period == "future" and aquifer:
                            base_rate += rcp_boost[rcp] * 4.0   # planted signal
                        n = rng.poisson(base_rate)
                        for _ in range(n):
                            slow = rng.random() < (0.4 if aquifer else 0.1)
                            gap = rng.integers(91, 400) if slow else rng.integers(3, 90)
                            store = ("LZ" if slow else "UZ")
                            rows.append(dict(
                                gauge_id=g, rcp=rcp, member=m, period=period,
                                direction=direction, gap_days=int(gap),
                                regime="slow" if slow else "abrupt",
                                rate_limiting_store=store, passes_coherence=True,
                                baseflow_index=np.nan))
    df = pd.DataFrame(rows)
    # attach a BFI proxy so stratification self-test works
    bfi = {g: float(rng.uniform(0.25, 0.92)) for g in gauges}
    df["baseflow_index"] = df["gauge_id"].map(bfi)
    df.to_parquet(path, index=False)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=str, default=None,
                    help="Transition parquet from slow_transition_analysis.py.")
    ap.add_argument("--attr", type=str, default=None,
                    help="CAMELS-GB hydrologic attributes CSV (for BFI strata).")
    ap.add_argument("--max-gap", type=int, default=720,
                    help="production gap bound in days (default 720 = two "
                         "water years on the 360-day model calendar, "
                         "manuscript Section 2.3); pass 0 for uncensored")
    ap.add_argument("--out", type=str, default="projection_flow.parquet")
    args = ap.parse_args()

    if args.input:
        tr = load(args.input, max_gap=args.max_gap or None)
        attr = args.attr
        print(f"Loaded {len(tr):,} coherent transitions from {args.input}")
    else:
        print("No --input: synthetic self-test (planted RCP-scaled FTD signal, "
              "flat DTF control).\n")
        tmp = selftest_io.redirect("synthetic_transitions.parquet", True)
        _make_synthetic_parquet(tmp)
        tr = load(tmp)
        attr = tmp  # carries a baseflow_index column
        if not _HAVE_SCIPY:
            print("(scipy unavailable: significance columns skipped)\n")

    delta = report(tr, attr)
    out = selftest_io.redirect(args.out, selftest=not args.input)
    delta.to_parquet(out, index=False)
    print(f"\nWrote per-catchment deltas -> {out.resolve()}")
    if not args.input:
        selftest_io.announce([out])


if __name__ == "__main__":
    main()
