"""
bootstrap_conditional_gap.py

Sampling uncertainty for the headline pace metric -- the change in the
CONDITIONAL gap (median gap given that a transition occurs) between the
baseline (WY1982-2010) and future (WY2051-2080) windows, per RCP and
direction. Companion to projection_analysis.py: it wraps the exact same
pooled-gap statistic that produces Table 1, and attaches a confidence
interval to it so the +8 d RCP8.5 figure is reported with an uncertainty
bound rather than as a bare point estimate.

Why this metric needs no frequency correction:
    Event thresholds are the fixed baseline Q5/Q80 baked into the transition
    table, so the *rate* of threshold crossing changes under warming. The
    conditional gap is measured only within realised transitions (pooled
    gap_days), so it does not inherit that change in frequency directly --
    which is why the manuscript leads on it. It is NOT fully immune to the
    fixed thresholds, however: more frequent crossing of a fixed Q80 also
    changes which transitions are realised and where their endpoints fall, so
    the composition of the conditional sample shifts too. The far-tail
    contraction under RCP8.5 reported by coherence_filter_diagnostics.py is
    the visible symptom. This script quantifies only sampling uncertainty.

See also:
    bootstrap_paired_differences.py -- CIs on DIFFERENCES between two changes
    (scenario vs scenario, FTD vs DTF). Comparing the intervals produced here
    for two different contrasts is NOT a valid test of their difference,
    because the contrasts share catchments and a baseline.

Why a CLUSTER bootstrap (not i.i.d. over transitions):
    Transitions within a catchment are not independent -- one groundwater-
    dominated catchment contributes many correlated slow gaps. An i.i.d.
    bootstrap over pooled transitions would therefore understate the CI. We
    resample the independent spatial units (catchments; gauge_id) with
    replacement, and recompute the baseline->future difference on the SAME
    resampled catchment set each iteration, so the two periods are paired
    through catchments. The four ensemble members are repeated realisations
    of each catchment; their contribution is reported separately as a
    per-member spread rather than folded into the cluster CI.

Point estimates reproduce projection_analysis.population_shift() exactly
(same pooled median / p90 / >90-day share), so Table 1 is unchanged; this
script only adds ci_lo / ci_hi and the per-member min/median/max.

Usage:
    python bootstrap_conditional_gap.py --input slow_full_flow.parquet
    python bootstrap_conditional_gap.py --input slow_full_flow.parquet \
        --direction FTD --n-boot 5000 --cluster gauge --seed 0 \
        --out bootstrap_conditional_gap.csv
    python bootstrap_conditional_gap.py            # synthetic self-test
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import selftest_io

RCPS = ["rcp26", "rcp45", "rcp60", "rcp85"]
MEMBERS = ["01", "04", "06", "15"]
ABRUPT_CUTOFF = 90            # days; boundary for the >90-day "slow" share
DEFAULT_NBOOT = 5000
DEFAULT_CI = 95.0             # central CI width (%)


# --------------------------------------------------------------------------- #
# Loading (identical filter to projection_analysis.load)                      #
# --------------------------------------------------------------------------- #
def load(parquet: str, max_gap: int | None = 720) -> pd.DataFrame:
    """Load the coherent transitions, censored at the production gap cap.

    Applied here rather than assumed of the input, so results sit on the same
    720-day production footing as the manuscript (Section 2.3) whether the
    table on disk is pre-capped or not. Pass max_gap=None for uncensored.
    """
    tr = pd.read_parquet(parquet)
    if "passes_coherence" in tr.columns:
        tr = tr[tr["passes_coherence"]].copy()
    if max_gap:
        tr = tr[tr["gap_days"] <= max_gap].copy()
    tr["gauge_id"] = tr["gauge_id"].astype(str)
    return tr


# --------------------------------------------------------------------------- #
# The three pooled statistics (must match projection_analysis)                #
# --------------------------------------------------------------------------- #
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
# Cluster bootstrap                                                           #
# --------------------------------------------------------------------------- #
def _cluster_arrays(d: pd.DataFrame, cluster_keys: list[str]) -> list[tuple]:
    """One (baseline_gaps, future_gaps) pair per cluster, as numpy arrays."""
    out = []
    for _, sub in d.groupby(cluster_keys, sort=False):
        b = sub.loc[sub["period"] == "baseline", "gap_days"].to_numpy()
        f = sub.loc[sub["period"] == "future", "gap_days"].to_numpy()
        out.append((b, f))
    return out


def bootstrap_direction(tr: pd.DataFrame, direction: str, n_boot: int,
                        cluster: str, ci: float, seed: int) -> pd.DataFrame:
    d = tr[tr["direction"] == direction]
    cluster_keys = {"gauge": ["gauge_id"],
                    "gauge_member": ["gauge_id", "member"],
                    "none": None}[cluster]
    rng = np.random.default_rng(seed)
    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
    metrics = ["median", "p90", "slowshare"]
    rows = []

    for rcp in RCPS:
        dr = d[d["rcp"] == rcp]
        base_all = dr.loc[dr["period"] == "baseline", "gap_days"].to_numpy()
        fut_all = dr.loc[dr["period"] == "future", "gap_days"].to_numpy()

        # point estimates (identical to projection_analysis.population_shift)
        point = {m: _stat(fut_all, m) - _stat(base_all, m) for m in metrics}

        # bootstrap replicates of the delta
        if cluster == "none":
            reps = {m: np.empty(n_boot) for m in metrics}
            nb, nf = base_all.size, fut_all.size
            for i in range(n_boot):
                bb = base_all[rng.integers(0, nb, nb)] if nb else base_all
                ff = fut_all[rng.integers(0, nf, nf)] if nf else fut_all
                for m in metrics:
                    reps[m][i] = _stat(ff, m) - _stat(bb, m)
        else:
            clusters = _cluster_arrays(dr, cluster_keys)
            n_cl = len(clusters)
            base_arr = [c[0] for c in clusters]
            fut_arr = [c[1] for c in clusters]
            reps = {m: np.empty(n_boot) for m in metrics}
            for i in range(n_boot):
                idx = rng.integers(0, n_cl, n_cl)
                bb = np.concatenate([base_arr[j] for j in idx]) if n_cl else base_all
                ff = np.concatenate([fut_arr[j] for j in idx]) if n_cl else fut_all
                for m in metrics:
                    reps[m][i] = _stat(ff, m) - _stat(bb, m)

        # per-member spread of the delta (does +8 d rest on one member?)
        per_member = []
        for mem in MEMBERS:
            dm = dr[dr["member"] == mem]
            bm = dm.loc[dm["period"] == "baseline", "gap_days"].to_numpy()
            fm = dm.loc[dm["period"] == "future", "gap_days"].to_numpy()
            per_member.append(_stat(fm, "median") - _stat(bm, "median"))
        per_member = np.array([x for x in per_member if not np.isnan(x)])

        for m in metrics:
            r = reps[m][~np.isnan(reps[m])]
            rec = {"rcp": rcp, "direction": direction, "metric": m,
                   "point": round(point[m], 3),
                   "ci_lo": round(float(np.percentile(r, lo_q)), 3) if r.size else np.nan,
                   "ci_hi": round(float(np.percentile(r, hi_q)), 3) if r.size else np.nan,
                   "base_n": int(base_all.size), "fut_n": int(fut_all.size)}
            if m == "median" and per_member.size:
                rec["member_min"] = round(float(per_member.min()), 3)
                rec["member_med"] = round(float(np.median(per_member)), 3)
                rec["member_max"] = round(float(per_member.max()), 3)
            rows.append(rec)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Report                                                                      #
# --------------------------------------------------------------------------- #
def report(tr: pd.DataFrame, directions, n_boot, cluster, ci, seed) -> pd.DataFrame:
    out = []
    unit = {"gauge": "catchment", "gauge_member": "catchment x member",
            "none": "transition (i.i.d.)"}[cluster]
    print(f"Cluster bootstrap: unit = {unit}; {n_boot:,} replicates; "
          f"{ci:.0f}% CI; seed {seed}")
    for direction in directions:
        res = bootstrap_direction(tr, direction, n_boot, cluster, ci, seed)
        out.append(res)
        print(f"\n{'='*72}\n{direction}: conditional gap change, baseline -> future "
              f"(days; slowshare in prob.)\n{'='*72}")
        show = res.copy()
        show["estimate [CI]"] = show.apply(
            lambda r: f"{r['point']:+.1f} [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]", axis=1)
        cols = ["rcp", "metric", "estimate [CI]", "base_n", "fut_n"]
        print(show[cols].to_string(index=False))

        med = res[res["metric"] == "median"]
        if "member_min" in med.columns:
            print("\n  per-member Δmedian spread (min / median / max across the 4 members):")
            for _, r in med.iterrows():
                print(f"    {r['rcp']}:  {r['member_min']:+.1f} / "
                      f"{r['member_med']:+.1f} / {r['member_max']:+.1f} d")
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- #
# Synthetic self-test                                                         #
# --------------------------------------------------------------------------- #
def _make_synthetic_parquet(path: str) -> None:
    """Plant an RCP-scaled FTD gap lengthening; DTF flat control. The recovered
    point deltas should be ~ +2/+2/+3/+8-ish and CIs should EXCLUDE 0 for FTD
    (esp. RCP8.5) and INCLUDE 0 for DTF."""
    rng = np.random.default_rng(0)
    gauges = [f"g{i:03d}" for i in range(120)]
    fut_shift = {"rcp26": 2.0, "rcp45": 2.0, "rcp60": 3.0, "rcp85": 8.0}
    rows = []
    for g in gauges:
        aquifer = rng.random() < 0.3
        for rcp in RCPS:
            for m in MEMBERS:
                for direction in ("FTD", "DTF"):
                    for period in ("baseline", "future"):
                        n = rng.poisson(12)
                        # baseline right-skewed gaps ~ median 13 (FTD) / 4 (DTF)
                        loc = 13.0 if direction == "FTD" else 4.0
                        gaps = rng.gamma(shape=1.6, scale=loc / 1.6, size=n)
                        if period == "future" and direction == "FTD":
                            gaps = gaps + fut_shift[rcp] * (1.4 if aquifer else 0.8)
                        for gp in gaps:
                            gp = max(1.0, gp)
                            rows.append(dict(
                                gauge_id=g, rcp=rcp, member=m, period=period,
                                direction=direction, gap_days=float(round(gp)),
                                regime="slow" if gp > ABRUPT_CUTOFF else "abrupt",
                                passes_coherence=True))
    pd.DataFrame(rows).to_parquet(path, index=False)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=str, default=None,
                    help="Transition parquet from slow_transition_analysis.py.")
    ap.add_argument("--direction", type=str, default="both",
                    choices=["FTD", "DTF", "both"])
    ap.add_argument("--n-boot", type=int, default=DEFAULT_NBOOT)
    ap.add_argument("--cluster", type=str, default="gauge",
                    choices=["gauge", "gauge_member", "none"],
                    help="Bootstrap resampling unit (default: gauge = catchment).")
    ap.add_argument("--ci", type=float, default=DEFAULT_CI)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-gap", type=int, default=720,
                    help="production gap bound in days (default 720 = two "
                         "water years on the 360-day model calendar, "
                         "manuscript Section 2.3); pass 0 for uncensored")
    ap.add_argument("--out", type=str, default="bootstrap_conditional_gap.csv")
    args = ap.parse_args()

    if args.input:
        tr = load(args.input, max_gap=args.max_gap or None)
        print(f"Loaded {len(tr):,} coherent transitions from {args.input}")
    else:
        print("No --input: synthetic self-test (planted +2/+2/+3/+8 FTD shift, "
              "flat DTF control).\n")
        tmp = selftest_io.redirect("synthetic_gaps.parquet", True)
        _make_synthetic_parquet(tmp)
        tr = load(tmp)

    directions = ["FTD", "DTF"] if args.direction == "both" else [args.direction]
    res = report(tr, directions, args.n_boot, args.cluster, args.ci, args.seed)

    out = selftest_io.redirect(args.out, selftest=not args.input)
    res.to_csv(out, index=False) if out.suffix == ".csv" else res.to_parquet(out, index=False)
    print(f"\nWrote bootstrap table -> {out.resolve()}")
    if not args.input:
        selftest_io.announce([out])


if __name__ == "__main__":
    main()
