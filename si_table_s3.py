"""
si_table_s3.py

Generates Table S3 of the Supporting Information: the full ranked table of
univariate discriminators (attribute, median in responders, median in
non-responders, rank-biserial effect size, p-value, Benjamini-Hochberg
FDR-adjusted p-value) tested against the FTD responder/non-responder split,
as summarised in Section 3.4 of the main text.

This is a thin wrapper around responder_characterisation.py: it reuses that
script's load_merged / define_groups / univariate functions unchanged (so
Table S3 is guaranteed consistent with the numbers quoted in-text and with
responder_table.parquet), and adds a LaTeX longtable formatter.

Output
------
  table_s3.csv   -- full univariate table (all candidate attributes)
  table_s3.tex   -- the same table as a ready-to-inline longtable, sorted by
                    |rank-biserial effect|, with FDR < 0.05 rows bolded (none
                    are expected to qualify per the main-text null result, but
                    the formatting handles it if that changes on a rerun).

Usage (production, JASMIN, after projection_analysis.py + forcing_deltas.py):
    python si_table_s3.py --deltas projection_flow.parquet \
        --attr-dir . --params calibrated_parameters.csv \
        --forcing forcing_deltas_rcp85.csv \
        --direction FTD --rcp rcp85 --responder-thresh 0.5 \
        --out-prefix table_s3

Self-test (no args): runs on responder_characterisation.py's own synthetic
fixture (planted aridity + K2 signal) to prove the formatter end-to-end.
Numbers from the self-test are NOT real results and must not be pasted into
the manuscript -- they only demonstrate the table renders correctly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import selftest_io

import responder_characterisation as rc

# Human-readable labels for the LaTeX table (falls back to the raw column
# name if an attribute is not listed here).
LABELS = {
    "baseflow_index": "Baseflow index",
    "runoff_ratio": "Runoff ratio",
    "slope_fdc": "Slope of flow-duration curve",
    "stream_elas": "Streamflow elasticity",
    "Q95": "Q95 (low-flow index)",
    "low_q_dur": "Low-flow spell duration",
    "high_q_dur": "High-flow spell duration",
    "frac_high_perc": "Fraction high-productivity aquifer",
    "inter_high_perc": "Fraction intermediate-productivity aquifer",
    "aridity": "Aridity index (baseline)",
    "frac_snow": "Fraction precipitation as snow",
    "p_seasonality": "Precipitation seasonality",
    "p_mean": "Mean precipitation",
    "pet_mean": "Mean PET",
    "high_prec_freq": "High-precipitation frequency",
    "area": "Catchment area",
    "elev_mean": "Mean elevation",
    "dpsbar": "Mean basin slope",
    "K0": "K0 (near-surface fast-flow constant)",
    "K1": "K1 (interflow constant)",
    "K2": "K2 (baseflow constant)",
    "UZL": "UZL (upper-zone threshold)",
    "PERC": "PERC (percolation capacity)",
    "FC": "FC (field capacity)",
    "BETA": "BETA (recharge shape parameter)",
    "LP": "LP (evaporation reduction threshold)",
    "d_p_annual": "$\\Delta$ annual rainfall",
    "d_pet_annual": "$\\Delta$ annual PET",
    "d_aridity": "$\\Delta$ aridity",
    "d_p_summer": "$\\Delta$ summer rainfall",
    "d_pet_summer": "$\\Delta$ summer PET",
    "d_aridity_summer": "$\\Delta$ summer aridity",
    "d_seasonality": "$\\Delta$ precipitation seasonality",
    "d_tas_mean": "$\\Delta$ mean temperature",
}


def to_latex(uni: pd.DataFrame, caption: str, label: str) -> str:
    """Render the univariate table as a longtable. FDR < 0.05 rows are bolded
    (none expected under the main-text null result)."""
    header = (r"Attribute & Resp. & Non-resp. & "
             r"$r$ & $p$ & FDR $p$ \\")
    lines = [
        r"\begin{center}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{longtable}{p{5.2cm} r r r r r}",
        r"\caption{" + caption + r"} \label{" + label + r"} \\",
        r"\hline",
        header,
        r"& \multicolumn{2}{c}{median} & & & \\",
        r"\hline",
        r"\endfirsthead",
        r"\hline",
        header,
        r"& \multicolumn{2}{c}{median} & & & \\",
        r"\hline",
        r"\endhead",
        r"\hline",
        r"\endfoot",
    ]
    for _, row in uni.iterrows():
        name = LABELS.get(row["attribute"], row["attribute"].replace("_", r"\_"))
        vals = (f"{row['median_resp']:.3g} & {row['median_nonresp']:.3g} & "
                f"{row.get('effect_r', float('nan')):+.3f} & "
                f"{row.get('p', float('nan')):.4f} & "
                f"{row.get('fdr', float('nan')):.4f}")
        row_str = f"{name} & {vals} \\\\"
        if row.get("fdr", 1.0) < 0.05:
            row_str = r"\textbf{" + name + "} & " + vals + r" \\"
        lines.append(row_str)
    lines += [r"\hline", r"\end{longtable}", r"\end{center}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deltas", type=str, default=None)
    ap.add_argument("--attr-dir", type=str, default=None)
    ap.add_argument("--params", type=str, default=None)
    ap.add_argument("--forcing", type=str, default=None)
    ap.add_argument("--direction", type=str, default="FTD")
    ap.add_argument("--rcp", type=str, default="rcp85")
    ap.add_argument("--responder-thresh", type=float, default=0.5)
    ap.add_argument("--out-prefix", type=str, default="table_s3")
    args = ap.parse_args()

    if args.deltas:
        d = rc.load_merged(args.deltas, args.attr_dir, args.params,
                           args.direction, args.rcp, forcing=args.forcing)
        print(f"Merged {len(d)} catchments for {args.direction} {args.rcp}")
        note = ""
    else:
        print("No --deltas: self-test on responder_characterisation.py's "
              "synthetic fixture (planted aridity + K2 signal).\n"
              "*** These are SYNTHETIC numbers for format-checking only. ***\n"
              "*** Do not paste them into the manuscript. ***\n")
        tmp = selftest_io.redirect("syn_resp_s1", True); tmp.mkdir(exist_ok=True)
        dp, adir, pp = rc._make_synthetic(tmp)
        d = rc.load_merged(dp, adir, pp, "FTD", "rcp85")
        note = ("SELF-TEST OUTPUT (synthetic fixture) -- for pipeline "
                "verification only, not for submission.\n")

    g = rc.define_groups(d, args.responder_thresh)
    nR = (g["group"] == "responder").sum()
    nN = (g["group"] == "non_responder").sum()
    print(f"responder={nR}, non_responder={nN} "
          f"(responder = d_freq_decade >= {args.responder_thresh})")

    predictors = [c for c in rc.CANDIDATES if c in g.columns]
    predictors += [c for c in g.columns
                  if c.startswith("d_") and c != "d_freq_decade"
                  and c not in predictors]
    uni = rc.univariate(g, predictors)
    if uni.empty:
        print("No attribute had >=3 responders and >=3 non-responders; "
              "table is empty.")
        return

    prefix = selftest_io.redirect(args.out_prefix, selftest=not args.deltas)
    csv_path = Path(f"{prefix}.csv")
    uni.to_csv(csv_path, index=False)

    caption = (f"Full ranked univariate discriminators between responder and "
               f"non-responder carrier catchments (FTD, {args.rcp.upper().replace('RCP','RCP ')}, "
               f"responder threshold $\\Delta \\geq$ {args.responder_thresh} slow "
               f"transitions decade$^{{-1}}$), ranked by $|$rank-biserial effect$|$. "
               f"No attribute clears Benjamini--Hochberg false-discovery-rate "
               f"control at $\\alpha$ = 0.05 in the production run (Section 3.4 "
               f"of the manuscript); rows would be bolded here if that changes.")
    tex = to_latex(uni, caption, "tab:S3")
    tex_path = Path(f"{prefix}.tex")
    tex_path.write_text((f"% {note}" if note else "") + tex + "\n")

    print(f"\nWrote {len(uni)} attributes -> {csv_path.resolve()}")
    print(f"Wrote LaTeX longtable -> {tex_path.resolve()}")
    if not args.deltas:
        selftest_io.announce([csv_path, tex_path])


if __name__ == "__main__":
    main()
