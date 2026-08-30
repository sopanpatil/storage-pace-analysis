#!/usr/bin/env python
"""
bootstrap_paired_differences.py
===============================
Confidence intervals on DIFFERENCES between conditional-gap changes
(manuscript Section 3.4), and an ensemble-aware variant of the headline CI.

Why this exists
---------------
bootstrap_conditional_gap.py puts a CI on each change separately. The
manuscript then argued that two changes differ because their intervals do not
overlap ("RCP8.5 separates from the lower scenarios"; "+8 d FTD vs +2 d DTF").
That reasoning is not valid here: the contrasts share the same catchments and,
for the RCP comparisons, the same baseline, so the two estimates are strongly
positively correlated and non-overlap is an over-conservative and
mis-specified test. The correct quantity is a CI on the difference itself,
computed on a bootstrap that resamples the shared clusters ONCE per replicate
and evaluates both contrasts on that same resample. This script does that.

It also addresses a second issue with the headline CI. The four CHESS-SCAPE
ensemble members are pooled before resampling, so a catchment-only cluster
bootstrap treats them as four independent replicates of the same catchment and
returns an interval conditional on that specific four-member ensemble. With
--resample-members the members are resampled (with replacement, within each
resampled catchment) alongside the catchments, propagating the ensemble
component of the uncertainty. With only four members this widens the interval
substantially, which is the honest result: the manuscript already reports a
per-member spread of +1 to +16 days at RCP8.5.

Statistics are the pooled conditional-gap statistics of projection_analysis.py,
so point estimates reproduce Table 1 exactly.

Input
    slow_full_flow.parquet   (per-transition table; coherent rows only are used)
    Required columns: gap_days, direction, period, rcp, member, gauge_id,
                      passes_coherence

Usage
    python bootstrap_paired_differences.py --input slow_full_flow.parquet
    python bootstrap_paired_differences.py --input slow_full_flow.parquet \
        --resample-members --n-boot 5000 --out derived_output/paired_differences.csv
    python bootstrap_paired_differences.py            # synthetic self-test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RCPS = ["rcp26", "rcp45", "rcp60", "rcp85"]
MEMBERS = ["01", "04", "06", "15"]
ABRUPT_CUTOFF = 90
DEFAULT_NBOOT = 5000
DEFAULT_CI = 95.0


# --------------------------------------------------------------------------- #
def load(parquet: str, max_gap: int | None = 720) -> pd.DataFrame:
    """Load the coherent transitions, censored at the production gap cap.

    Applied here rather than assumed of the input, so results sit on the same
    720-day production footing as the manuscript (Section 2.3) whether the
    table on disk is pre-capped or not. Pass max_gap=None for uncensored.
    """
    tr = pd.read_parquet(parquet)
    need = {"gap_days", "direction", "period", "rcp", "member", "gauge_id"}
    missing = need - set(tr.columns)
    if missing:
        raise KeyError(f"{parquet} is missing columns: {sorted(missing)}")
    if "passes_coherence" in tr.columns:
        tr = tr[tr["passes_coherence"]].copy()
    if max_gap:
        tr = tr[tr["gap_days"] <= max_gap].copy()
    tr["gauge_id"] = tr["gauge_id"].astype(str)
    tr["member"] = tr["member"].astype(str)
    return tr


def _stat(gaps: np.ndarray, metric: str) -> float:
    if gaps.size == 0:
        return np.nan
    if metric == "median":
        return float(np.median(gaps))
    if metric == "p90":
        return float(np.quantile(gaps, 0.90))
    if metric == "slowshare":
        return float((gaps > ABRUPT_CUTOFF).mean())
    raise ValueError(metric)


# --------------------------------------------------------------------------- #
# Index the table once, so each bootstrap replicate is pure array work.
# cells[(direction, rcp, gauge, member)] -> (baseline_gaps, future_gaps)
# --------------------------------------------------------------------------- #
def build_cells(tr: pd.DataFrame) -> dict:
    cells: dict = {}
    keys = ["direction", "rcp", "gauge_id", "member"]
    for (d, r, g, m), sub in tr.groupby(keys, sort=False):
        cells[(d, r, g, m)] = (
            sub.loc[sub["period"] == "baseline", "gap_days"].to_numpy(),
            sub.loc[sub["period"] == "future", "gap_days"].to_numpy(),
        )
    return cells


def _delta(cells: dict, direction: str, rcp: str, gauges, members,
           metric: str) -> float:
    """Conditional-gap change for one (direction, rcp) on a given resample."""
    b, f = [], []
    for g in gauges:
        for m in members:
            cell = cells.get((direction, rcp, g, m))
            if cell is None:
                continue
            b.append(cell[0])
            f.append(cell[1])
    if not b:
        return np.nan
    return _stat(np.concatenate(f), metric) - _stat(np.concatenate(b), metric)


# --------------------------------------------------------------------------- #
def run(tr: pd.DataFrame, contrasts: list[tuple], metric: str, n_boot: int,
        ci: float, seed: int, resample_members: bool) -> pd.DataFrame:
    cells = build_cells(tr)
    gauges = sorted({k[2] for k in cells})
    members = sorted({k[3] for k in cells})
    rng = np.random.default_rng(seed)
    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2

    # point estimates on the full sample
    point = {}
    for c in contrasts:
        for (d, r) in c[1:]:
            if (d, r) not in point:
                point[(d, r)] = _delta(cells, d, r, gauges, members, metric)

    reps = {c[0]: np.empty(n_boot) for c in contrasts}
    single = {(d, r): np.empty(n_boot)
              for c in contrasts for (d, r) in c[1:]}

    ng, nm = len(gauges), len(members)
    for i in range(n_boot):
        gidx = rng.integers(0, ng, ng)
        gs = [gauges[j] for j in gidx]
        ms = ([members[j] for j in rng.integers(0, nm, nm)]
              if resample_members else members)
        cache = {}
        for (d, r) in single:
            cache[(d, r)] = _delta(cells, d, r, gs, ms, metric)
            single[(d, r)][i] = cache[(d, r)]
        for name, a, b in contrasts:
            reps[name][i] = cache[a] - cache[b]

    rows = []
    for name, a, b in contrasts:
        v = reps[name][np.isfinite(reps[name])]
        rows.append(dict(
            contrast=name, metric=metric,
            estimate_a=point[a], estimate_b=point[b],
            difference=point[a] - point[b],
            ci_lo=float(np.percentile(v, lo_q)),
            ci_hi=float(np.percentile(v, hi_q)),
            p_two_sided=float(2 * min((v <= 0).mean(), (v >= 0).mean())),
            n_boot=len(v),
            members_resampled=resample_members,
        ))
    per_term = []
    for (d, r), v in single.items():
        v = v[np.isfinite(v)]
        per_term.append(dict(
            direction=d, rcp=r, metric=metric, estimate=point[(d, r)],
            ci_lo=float(np.percentile(v, lo_q)),
            ci_hi=float(np.percentile(v, hi_q)),
            members_resampled=resample_members,
        ))
    return pd.DataFrame(rows), pd.DataFrame(per_term)


DEFAULT_CONTRASTS = [
    ("FTD RCP8.5 - FTD RCP2.6", ("FTD", "rcp85"), ("FTD", "rcp26")),
    ("FTD RCP8.5 - FTD RCP4.5", ("FTD", "rcp85"), ("FTD", "rcp45")),
    ("FTD RCP8.5 - FTD RCP6.0", ("FTD", "rcp85"), ("FTD", "rcp60")),
    ("FTD RCP8.5 - DTF RCP8.5", ("FTD", "rcp85"), ("DTF", "rcp85")),
]


def _self_test() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for g in [f"g{i:03d}" for i in range(60)]:
        off = rng.normal(0, 3)
        for r in RCPS:
            shift = {"rcp26": 2, "rcp45": 2, "rcp60": 3, "rcp85": 8}[r]
            for m in MEMBERS:
                for d, base in (("FTD", 13), ("DTF", 4)):
                    s = shift if d == "FTD" else 2
                    for per, mu in (("baseline", base + off),
                                    ("future", base + off + s)):
                        n = 40
                        rows.append(pd.DataFrame(dict(
                            gap_days=np.abs(rng.normal(mu, 6, n)),
                            direction=d, period=per, rcp=r, member=m,
                            gauge_id=g, passes_coherence=True)))
    tr = pd.concat(rows, ignore_index=True)
    diffs, per_term = run(tr, DEFAULT_CONTRASTS, "median", 300, DEFAULT_CI,
                          0, False)
    row = diffs[diffs["contrast"] == "FTD RCP8.5 - DTF RCP8.5"].iloc[0]
    assert row["ci_lo"] > 0, "FTD change should exceed DTF change in the fixture"
    print("self-test OK: paired bootstrap recovers a positive FTD-DTF "
          "difference with a CI excluding zero")
    _print(diffs, per_term)


def _print(diffs: pd.DataFrame, per_term: pd.DataFrame) -> None:
    print("\nPer-term changes (same resamples):")
    for _, r in per_term.iterrows():
        print(f"  {r['direction']} {r['rcp']}: {r['estimate']:+.1f} "
              f"[{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]")
    print("\nPaired differences:")
    for _, r in diffs.iterrows():
        print(f"  {r['contrast']:26s} {r['difference']:+.1f} d "
              f"[{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]  p={r['p_two_sided']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=None)
    ap.add_argument("--metric", default="median",
                    choices=["median", "p90", "slowshare"])
    ap.add_argument("--n-boot", type=int, default=DEFAULT_NBOOT)
    ap.add_argument("--ci", type=float, default=DEFAULT_CI)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resample-members", action="store_true",
                    help="also resample the four ensemble members, propagating "
                         "ensemble uncertainty into the interval")
    ap.add_argument("--max-gap", type=int, default=720,
                    help="production gap bound in days (default 720 = two "
                         "water years on the 360-day model calendar, "
                         "manuscript Section 2.3); pass 0 for uncensored")
    ap.add_argument("--out", default=None, help="CSV path for the differences")
    args = ap.parse_args()

    if args.input is None:
        _self_test()
        return

    tr = load(args.input, max_gap=args.max_gap or None)
    diffs, per_term = run(tr, DEFAULT_CONTRASTS, args.metric, args.n_boot,
                          args.ci, args.seed, args.resample_members)
    _print(diffs, per_term)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        diffs.to_csv(args.out, index=False)
        per_term.to_csv(str(args.out).replace(".csv", "_per_term.csv"),
                        index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
