"""
si_sensitivity_analysis.py

Supporting Information sensitivity analysis for Table S1 and Table S2: how the
headline FTD transition-pace results (Sections 3.1 and 3.4 of the main text)
respond to three parameters of the storage-based pairing and coherence filter
(slow_transition_analysis.Config):

  - resaturation_delta  (storage re-saturation tolerance; production 0.15)
  - max_gap_days        (upper bound on transition gap; production 720)
  - coherence_min       (coherence threshold; production 0.60)

For each grid point (varying one parameter, holding the other two at their
production values), this script reruns the full transition-detection pipeline
end to end via slow_transition_analysis.run_jasmin and reports the same four
headline statistics quoted in the main text, pooled over the FTD, coherent
transitions:

  - baseline median gap, 90th percentile, slow share (> 90 d)
  - RCP8.5 future median gap, 90th percentile, slow share
  - RCP8.5 conditional median-gap change (future median - baseline median)
  - n coherent FTD transitions retained (baseline + future, RCP8.5)

Two grids are run: the pairing grid (resaturation_delta x max_gap_days,
Table S1) and the coherence grid (coherence_min, Table S2). Output is written
as one CSV per grid (SI Tables S1 and S2).

This is compute-equivalent to one full production run
(slow_transition_analysis.py --attribution flow) per grid point, so on
JASMIN restrict --rcps to rcp85 (the default) unless the full four-RCP
picture is specifically wanted; the main text's four-RCP headline numbers
(Table 1) are unaffected by these parameters at fixed production settings,
so RCP8.5 alone is sufficient to demonstrate robustness.

Usage (production, JASMIN):
    python si_sensitivity_analysis.py --jasmin-dir chess_scape_output \
        --params calibrated_parameters.csv --rcps rcp85 \
        --out-prefix si_sensitivity

Self-test (no args): runs the same two grids on the synthetic
fast/slow/snow archetypes used by slow_transition_analysis.py's own
self-test. Numbers from the self-test are NOT real results and must not be
pasted into the manuscript -- they only demonstrate the pipeline end-to-end.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import pandas as pd

import selftest_io

import slow_transition_analysis as sta

# Production (default) parameter values, per the main text.
PROD_RESAT = 0.15
PROD_MAXGAP = 720
PROD_COHMIN = 0.60

DEFAULT_RESAT_GRID = [0.10, 0.15, 0.20, 0.25]
DEFAULT_MAXGAP_GRID = [360, 540, 720, 1080]
DEFAULT_COHMIN_GRID = [0.50, 0.55, 0.60, 0.65, 0.70]

ABRUPT_CUTOFF = 90


# --------------------------------------------------------------------------- #
# Core: run one Config and summarise pooled coherent FTD gaps                 #
# --------------------------------------------------------------------------- #
def _summarise(tr: pd.DataFrame, rcp: str) -> dict:
    """Pooled coherent FTD summary stats for one Config's transition table."""
    d = tr[(tr["direction"] == "FTD") & (tr["passes_coherence"])]
    if "rcp" in d.columns:
        d = d[d["rcp"] == rcp]
    rec = {}
    for period in ("baseline", "future"):
        g = d[d["period"] == period]["gap_days"] if "period" in d.columns else pd.Series(dtype=float)
        rec[f"{period}_n"] = int(len(g))
        rec[f"{period}_median"] = round(float(g.median()), 1) if len(g) else float("nan")
        rec[f"{period}_p90"] = round(float(g.quantile(0.9)), 1) if len(g) else float("nan")
        rec[f"{period}_slowshare"] = (round(float((g > ABRUPT_CUTOFF).mean()), 3)
                                      if len(g) else float("nan"))
    rec["d_median"] = round(rec["future_median"] - rec["baseline_median"], 1)
    rec["n_coherent"] = rec["baseline_n"] + rec["future_n"]
    return rec


def _run_one(cfg: sta.Config, jasmin_dir: str | None, params_by_gauge: dict | None,
            gauges: list[str] | None, rcps: tuple[str, ...], members: tuple[str, ...],
            synthetic_df: pd.DataFrame | None, rcp_for_summary: str) -> dict:
    if jasmin_dir:
        tr = sta.run_jasmin(jasmin_dir, cfg, gauges=gauges, rcps=rcps,
                            members=members, params_by_gauge=params_by_gauge)
    else:
        tr = sta.analyse(synthetic_df, cfg)
    return _summarise(tr, rcp_for_summary)


# --------------------------------------------------------------------------- #
# Grids                                                                       #
# --------------------------------------------------------------------------- #
def pairing_grid(base_cfg: sta.Config, resat_grid: list[float], maxgap_grid: list[int],
                 **run_kwargs) -> pd.DataFrame:
    """Vary resaturation_delta and max_gap_days one at a time around production
    values (not a full cross product, to keep compute cost linear in grid size)."""
    rows = []
    for r in resat_grid:
        cfg = replace(base_cfg, resaturation_delta=r, max_gap_days=PROD_MAXGAP)
        rec = {"varied": "resaturation_delta", "value": r,
               "is_production": abs(r - PROD_RESAT) < 1e-9}
        rec.update(_run_one(cfg, **run_kwargs))
        rows.append(rec)
    for m in maxgap_grid:
        cfg = replace(base_cfg, resaturation_delta=PROD_RESAT, max_gap_days=m)
        rec = {"varied": "max_gap_days", "value": m,
               "is_production": m == PROD_MAXGAP}
        rec.update(_run_one(cfg, **run_kwargs))
        rows.append(rec)
    return pd.DataFrame(rows)


def coherence_grid(base_cfg: sta.Config, cohmin_grid: list[float],
                   **run_kwargs) -> pd.DataFrame:
    rows = []
    for c in cohmin_grid:
        cfg = replace(base_cfg, coherence_min=c)
        rec = {"varied": "coherence_min", "value": c,
               "is_production": abs(c - PROD_COHMIN) < 1e-9}
        rec.update(_run_one(cfg, **run_kwargs))
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jasmin-dir", type=str, default=None)
    ap.add_argument("--params", type=str, default=None,
                    help="Calibrated HBV params (required with --jasmin-dir, "
                         "attribution='flow').")
    ap.add_argument("--gauges", type=str, default=None)
    ap.add_argument("--rcps", type=str, default="rcp85",
                    help="Comma-separated RCPs to run through the pipeline "
                         "(default rcp85 only, for compute cost).")
    ap.add_argument("--members", type=str, default=None)
    ap.add_argument("--summary-rcp", type=str, default="rcp85",
                    help="Which RCP's future period to report against baseline.")
    ap.add_argument("--resat-grid", type=str, default=None,
                    help="Comma-separated resaturation_delta values "
                         f"(default {DEFAULT_RESAT_GRID}).")
    ap.add_argument("--maxgap-grid", type=str, default=None,
                    help="Comma-separated max_gap_days values "
                         f"(default {DEFAULT_MAXGAP_GRID}).")
    ap.add_argument("--cohmin-grid", type=str, default=None,
                    help="Comma-separated coherence_min values "
                         f"(default {DEFAULT_COHMIN_GRID}).")
    ap.add_argument("--out-prefix", type=str, default="si_sensitivity")
    args = ap.parse_args()

    resat_grid = ([float(x) for x in args.resat_grid.split(",")]
                 if args.resat_grid else DEFAULT_RESAT_GRID)
    maxgap_grid = ([int(x) for x in args.maxgap_grid.split(",")]
                  if args.maxgap_grid else DEFAULT_MAXGAP_GRID)
    cohmin_grid = ([float(x) for x in args.cohmin_grid.split(",")]
                  if args.cohmin_grid else DEFAULT_COHMIN_GRID)

    base_cfg = sta.Config(resaturation_delta=PROD_RESAT, max_gap_days=PROD_MAXGAP,
                          coherence_min=PROD_COHMIN,
                          attribution="flow" if args.jasmin_dir else "runoff-norm")

    if args.jasmin_dir:
        params_by_gauge = sta.load_params(args.params) if args.params else None
        gauges = args.gauges.split(",") if args.gauges else None
        rcps = tuple(args.rcps.split(","))
        members = tuple(args.members.split(",")) if args.members else sta.MEMBERS
        run_kwargs = dict(jasmin_dir=args.jasmin_dir, params_by_gauge=params_by_gauge,
                          gauges=gauges, rcps=rcps, members=members,
                          synthetic_df=None, rcp_for_summary=args.summary_rcp)
        print(f"Production run: dir={args.jasmin_dir}  rcps={rcps}  "
              f"members={members}\n")
    else:
        print("No --jasmin-dir: self-test on synthetic fast/slow/snow "
              "archetypes.\n"
              "*** These are SYNTHETIC numbers for pipeline/format-checking "
              "only. ***\n*** Do not paste them into the manuscript. ***\n")
        synth = sta._make_synthetic()
        run_kwargs = dict(jasmin_dir=None, params_by_gauge=None, gauges=None,
                          rcps=(), members=(), synthetic_df=synth,
                          rcp_for_summary="rcp85")

    selftest = not args.jasmin_dir
    prefix = selftest_io.redirect(args.out_prefix, selftest)

    print("Running pairing-parameter grid (resaturation_delta, max_gap_days)...")
    pdf = pairing_grid(base_cfg, resat_grid, maxgap_grid, **run_kwargs)
    print(pdf.to_string(index=False))
    pdf.to_csv(f"{prefix}_pairing.csv", index=False)

    print("\nRunning coherence-threshold grid (coherence_min)...")
    cdf = coherence_grid(base_cfg, cohmin_grid, **run_kwargs)
    print(cdf.to_string(index=False))
    cdf.to_csv(f"{prefix}_coherence.csv", index=False)

    print(f"\nWrote {prefix}_pairing.csv and {prefix}_coherence.csv")
    if selftest:
        selftest_io.announce([f"{prefix}_pairing.csv"])


if __name__ == "__main__":
    main()
