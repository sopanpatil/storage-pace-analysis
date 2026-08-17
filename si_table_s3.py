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
responder_table.parquet).

Output
------
  table_s3.csv   -- full univariate table (all candidate attributes), sorted
                    by |rank-biserial effect|; this CSV is the SI Table S3
                    output.

Usage (production, JASMIN, after projection_analysis.py + forcing_deltas.py):
    python si_table_s3.py --deltas projection_flow.parquet \
        --attr-dir . --params calibrated_parameters.csv \
        --forcing forcing_deltas_rcp85.csv \
        --direction FTD --rcp rcp85 --responder-thresh 0.5 \
        --out-prefix table_s3

Self-test (no args): runs on responder_characterisation.py's own synthetic
fixture (planted aridity + K2 signal) to prove the pipeline end-to-end.
Numbers from the self-test are NOT real results and must not be pasted into
the manuscript -- they only demonstrate the table renders correctly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import selftest_io

import responder_characterisation as rc


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
    else:
        print("No --deltas: self-test on responder_characterisation.py's "
              "synthetic fixture (planted aridity + K2 signal).\n"
              "*** These are SYNTHETIC numbers for format-checking only. ***\n"
              "*** Do not paste them into the manuscript. ***\n")
        tmp = selftest_io.redirect("syn_resp_s1", True); tmp.mkdir(exist_ok=True)
        dp, adir, pp = rc._make_synthetic(tmp)
        d = rc.load_merged(dp, adir, pp, "FTD", "rcp85")

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

    print(f"\nWrote {len(uni)} attributes -> {csv_path.resolve()}")
    if not args.deltas:
        selftest_io.announce([csv_path])


if __name__ == "__main__":
    main()
