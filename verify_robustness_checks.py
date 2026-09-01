#!/usr/bin/env python3
"""
verify_robustness_checks.py

The two robustness figures the manuscript quotes that CANNOT be recomputed from
the committed transition table, because each needs the raw HBV output in
`chess_scape_output/`:

  1. The store-attribution cross-check (Section 3.2). The headline 99.5% is the
     flow-weighted attribution. Replacing it with the range-normalised
     storage-change criterion is reported to assign 99.5% of slow FTD
     transitions to the lower zone. That criterion reads the UZ and LZ state
     series, which slow_full_flow.parquet does not carry, so it needs a second
     pass over the raw output with attribution="runoff-norm".

  2. The snowmelt statistics (Discussion). Median 0.00 mm cumulative melt over a
     slow FTD transition, 90th percentile 0.26 mm, below 1% of the lower-zone
     decline, and 88% of transitions receiving no meltwater at all. These read
     the logged melt flux from `<rcp>_<member>_hbv_melt.csv`.

Both are quoted in the manuscript as properties of the production run, which is
720-day capped (Section 2.3). Run this after any change to the pairing,
coherence or cap settings, so the two robustness numbers stay on the same
footing as the headline ones that verify_manuscript_numbers.py covers.

Usage:
    python verify_robustness_checks.py --jasmin-dir chess_scape_output \
        --params calibrated_parameters.csv --input slow_full_flow.parquet
    python verify_robustness_checks.py            # synthetic self-test

The production table IS the flow-attribution run, so by default only the
runoff-norm side is recomputed: one pipeline pass, about as long as one
production run of slow_transition_analysis.py. Add --rerun-flow to recompute
both, which also confirms the production table is reproducible from this code.
Restrict either with --rcps / --members for a smoke test.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import slow_transition_analysis as sta
import snow_melt_contribution as smc

# ---- values as reported in the manuscript (edit here if the draft changes) --- #
# All values below are from the 720-day-capped production run (JASMIN,
# 2026-08-11), which is what the manuscript now reports.
MS = {
    "runoff_norm_slow_lz": 0.995,     # Section 3.2, "in near-exact agreement"
    "flow_slow_lz": 0.995,            # Section 3.2 headline
    "melt_median_mm": 0.00,           # Discussion
    "melt_p90_mm": 0.15,
    "melt_frac_p90": 0.004,           # p90 of the per-transition melt/LZ ratio
    "melt_zero_share": 0.89,          # "89% receive no meltwater at all"
}
# Tolerances match the precision each value is quoted at, so a change that would
# alter the printed figure trips a CHECK rather than passing on a loose bound.
TOL = {
    "runoff_norm_slow_lz": 0.0005, "flow_slow_lz": 0.0005,
    "melt_median_mm": 0.005, "melt_p90_mm": 0.005,
    "melt_frac_p90": 0.0005, "melt_zero_share": 0.005,
}


def flag(got, want, tol) -> str:
    if got is None or (isinstance(got, float) and np.isnan(got)):
        return "n/a"
    return "MATCH" if abs(got - want) <= tol else ">>> CHECK"


def slow_lz_share(tr: pd.DataFrame) -> tuple[float, int]:
    """Share of coherent slow FTD transitions attributed to LZ, normalised over
    UZ+LZ -- the same reduction verify_manuscript_numbers.py uses."""
    d = tr[(tr["direction"] == "FTD") & (tr["passes_coherence"])
           & (tr["regime"] == "slow")]["rate_limiting_store"]
    d = d[d.isin(["UZ", "LZ"])]
    if not len(d):
        return float("nan"), 0
    return float((d == "LZ").mean()), len(d)


def check_attribution(jasmin_dir: str, params: str, rcps, members,
                      max_gap: int, flow_table: str | None = None) -> None:
    """Compare the two attribution criteria on the same pipeline settings.

    The flow-weighted side is read from the production table when one is given,
    since that table IS the flow-attribution run; only the runoff-norm side then
    needs a pipeline pass. Pass flow_table=None to rerun both, e.g. to confirm
    the production table is reproducible from the current code.
    """
    print("=" * 72)
    print("(1) STORE-ATTRIBUTION CROSS-CHECK  (runoff-norm vs flow-weighted)")
    print("=" * 72)
    params_by_gauge = sta.load_params(params)

    if flow_table:
        tr = pd.read_parquet(flow_table)
        got = sorted(tr["attribution"].unique()) if "attribution" in tr else []
        if got != ["flow"]:
            raise ValueError(
                f"{flow_table} was built with attribution={got}, not ['flow']; "
                f"pass --rerun-flow to recompute it instead of reading it.")
        share, n = slow_lz_share(tr)
        print(f"  {'flow':12s} slow FTD -> LZ : {share:.4f}  (n={n:,})   "
              f"ms {MS['flow_slow_lz']}   "
              f"{flag(share, MS['flow_slow_lz'], TOL['flow_slow_lz'])}   [from {flow_table}]")
        todo = [("runoff-norm", "runoff_norm_slow_lz")]
    else:
        todo = [("flow", "flow_slow_lz"), ("runoff-norm", "runoff_norm_slow_lz")]

    for attribution, key in todo:
        cfg = sta.Config(attribution=attribution, max_gap_days=max_gap)
        tr = sta.run_jasmin(jasmin_dir, cfg, rcps=tuple(rcps),
                            members=tuple(members),
                            params_by_gauge=params_by_gauge)
        share, n = slow_lz_share(tr)
        print(f"  {attribution:12s} slow FTD -> LZ : {share:.4f}  (n={n:,})   "
              f"ms {MS[key]}   {flag(share, MS[key], TOL[key])}   [recomputed]")


def check_snowmelt(input_parquet: str, jasmin_dir: str, max_gap: int = 720) -> None:
    print("\n" + "=" * 72)
    print("(2) SNOWMELT WITHIN SLOW FTD CARRIERS")
    print("=" * 72)
    tr = pd.read_parquet(input_parquet)
    carriers = smc.select_carriers(tr, max_gap=max_gap or None)
    ok = smc.melt_over_transitions(carriers, jasmin_dir)
    s = smc.summarise(ok)
    print(f"  n carriers with melt data : {s['n']:,} of {len(carriers):,}")
    # (label, key in summarise()'s dict, key in MS/TOL)
    for label, stat_key, ms_key in (
        ("median melt (mm)      ", "median_melt_mm", "melt_median_mm"),
        ("p90 melt (mm)         ", "p90_melt_mm", "melt_p90_mm"),
        ("p90 melt / LZ decline ", "p90_melt_frac", "melt_frac_p90"),
        ("zero-melt share       ", "zero_melt_share", "melt_zero_share"),
    ):
        got = s[stat_key]
        print(f"  {label}: {got:8.4f}   ms {MS[ms_key]}   "
              f"{flag(got, MS[ms_key], TOL[ms_key])}")


def _self_test() -> None:
    """Exercise both reductions on synthetic fixtures, without any real data."""
    print("No --jasmin-dir/--input: synthetic self-test.\n")

    print("(1) attribution reduction")
    tr = pd.DataFrame({
        "direction": ["FTD"] * 10 + ["DTF"] * 3,
        "passes_coherence": [True] * 13,
        "regime": ["slow"] * 8 + ["abrupt"] * 2 + ["slow"] * 3,
        "rate_limiting_store": ["LZ"] * 7 + ["UZ"] + ["UZ", "LZ"] + ["LZ"] * 3,
    })
    share, n = slow_lz_share(tr)
    assert n == 8 and abs(share - 7 / 8) < 1e-12, (share, n)
    print(f"    slow FTD -> LZ = {share:.3f} over n={n} "
          f"(DTF rows and abrupt rows correctly excluded)")

    print("\n(2) snowmelt reduction")
    smc._self_test()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jasmin-dir", help="chess_scape_output/ (raw HBV output)")
    ap.add_argument("--params", default="calibrated_parameters.csv")
    ap.add_argument("--input", default="slow_full_flow.parquet",
                    help="production transition table, for the snowmelt check")
    ap.add_argument("--rcps", default=",".join(sta.RCPS))
    ap.add_argument("--members", default=",".join(sta.MEMBERS))
    ap.add_argument("--max-gap", type=int, default=720,
                    help="production gap bound (manuscript Section 2.3)")
    ap.add_argument("--skip-attribution", action="store_true",
                    help="snowmelt check only (the attribution pass is slow)")
    ap.add_argument("--rerun-flow", action="store_true",
                    help="recompute the flow-weighted side too, instead of "
                         "reading it from --input; also confirms that the "
                         "production table is reproducible from this code")
    args = ap.parse_args()

    if not args.jasmin_dir:
        _self_test()
        return

    if not args.skip_attribution:
        check_attribution(args.jasmin_dir, args.params,
                          args.rcps.split(","), args.members.split(","),
                          args.max_gap,
                          flow_table=None if args.rerun_flow else args.input)
    check_snowmelt(args.input, args.jasmin_dir, args.max_gap)


if __name__ == "__main__":
    main()
