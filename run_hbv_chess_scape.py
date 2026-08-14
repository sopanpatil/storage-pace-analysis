"""
run_hbv_chess_scape.py
-----------------------
Stage 1 of the pipeline. Runs the calibrated HBV model across CHESS-SCAPE
forcing for a given RCP and ensemble member, and saves the discharge, the
internal storage states, and the response-routine generating fluxes needed by
the downstream transition analysis.

This driver uses the HBV implementation in the separate ``hbv-model``
repository (``from hbv_model.hbv import HBVModel``), which must be cloned as
a sibling directory or otherwise be importable. Requires ``hbv-model >=
v1.1.0`` (https://github.com/sopanpatil/hbv-model), the version that logs
the generating fluxes (Q0, Q1, Q2) and snowmelt directly during the run, in
addition to the storage states. Those logged fluxes are what the
store-attribution step consumes; see the output list below.

Calibrated parameters are read from a single consolidated CSV
(``calibrated_parameters.csv``, from the separate ``hbv-model`` repository),
using the 13 HBV parameter columns (TT, CFMAX, CFR, CWH, FC, LP, BETA, K0, K1,
K2, UZL, PERC, MAXBAS) and ignoring the calibration/validation metadata columns.

Output files (wide: a 'date' column plus one column per gauge id), written to
``chess_scape_output/`` and named to match the loaders in
``slow_transition_analysis.py`` (``_ALL_SUFFIX``) and
``snow_melt_contribution.py`` (``*_hbv_melt.csv``):

    States (rounded to %.4f, matching the committed convention):
        <rcp>_<ensemble>_hbv_discharge.csv   MAXBAS-routed discharge Q (mm/day)
        <rcp>_<ensemble>_hbv_sm.csv          soil moisture SM (mm)
        <rcp>_<ensemble>_hbv_uz.csv          upper-zone store UZ (mm)
        <rcp>_<ensemble>_hbv_lz.csv          lower-zone store LZ (mm)
        <rcp>_<ensemble>_hbv_sp.csv          snowpack SP (mm w.e.)

    Generating fluxes and snowmelt (high precision -- see note):
        <rcp>_<ensemble>_hbv_q0.csv          near-surface fast flow Q0 (mm/day)
        <rcp>_<ensemble>_hbv_q1.csv          interflow Q1 (mm/day)
        <rcp>_<ensemble>_hbv_q2.csv          baseflow Q2 (mm/day)
        <rcp>_<ensemble>_hbv_melt.csv        snowmelt (mm/day)

Precision note. The discharge and storage states are written at %.4f, the
convention the state files have always used. The generating fluxes (Q0, Q1, Q2)
and snowmelt are written at higher precision because the attribution step reads
them *directly* rather than reconstructing fast/baseflow from the rounded UZ/LZ
states; writing them at %.4f would discard the precision advantage that is the
whole reason for logging them. These are pre-routing generation-space fluxes
(the store attribution operates in that space, ahead of MAXBAS routing;
Methods Section 2.5), whereas the discharge file is the routed series used for
Q5/Q80 event detection.

Forcing files (same wide layout, one continuous 360-day-calendar series):
    <chess_scape_output>/<rcp>_<ensemble>_pr_catchment_means_combined.csv   (mm/day)
    <chess_scape_output>/<rcp>_<ensemble>_tas_catchment_means_combined.csv  (degC)
    <chess_scape_output>/<rcp>_<ensemble>_pet_catchment_means_combined.csv  (mm/day)

Usage
-----
python run_hbv_chess_scape.py --rcp rcp85 --ensemble 01 \
    --params-csv /path/to/hbv-model/calibrated_parameters.csv

# all four members of one RCP:
for m in 01 04 06 15; do
    python run_hbv_chess_scape.py --rcp rcp85 --ensemble $m \
        --params-csv ../hbv-model/calibrated_parameters.csv
done
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from hbv_model.hbv import HBVModel

CHESS_SCAPE_ROOT = Path("./chess_scape_output")
N_WORKERS = 8

# The 13 free HBV parameters, in the order hbv_model.hbv.HBVModel.PARAM_NAMES expects.
HBV_PARAM_COLS = ["TT", "CFMAX", "CFR", "CWH", "FC", "LP", "BETA",
                  "K0", "K1", "K2", "UZL", "PERC", "MAXBAS"]

# Output variable -> (states dict key in hbv_model.hbv.HBVModel.run(), file suffix).
# States are written at %.4f; fluxes/melt at high precision (see module docstring).
STATE_VARS = [("SM", "sm"), ("UZ", "uz"), ("LZ", "lz"), ("SP", "sp")]
FLUX_VARS = [("Q0", "q0"), ("Q1", "q1"), ("Q2", "q2"), ("melt", "melt")]

STATE_FLOAT_FMT = "%.4f"
FLUX_FLOAT_FMT = "%.8e"   # ~8 significant figures; direct-read precision


def _run_catchment(cid, params):
    """Worker: run HBV for one catchment, return routed Q plus the state and
    flux series the pipeline needs. Reads forcing from process-local globals
    set by _init_worker (avoids re-pickling the forcing per task)."""
    try:
        model = HBVModel(params)
        q_routed, states = model.run(
            _run_catchment.precip[cid], _run_catchment.temp[cid], _run_catchment.evap[cid]
        )
        out = {"Q": q_routed}
        for key, _ in STATE_VARS:
            out[key] = states[key]
        for key, _ in FLUX_VARS:
            out[key] = states[key]
        return cid, out
    except Exception as exc:
        print(f"  ERROR: HBV / {cid}: {exc}", flush=True)
        return cid, None


def _init_worker(precip, temp, evap):
    _run_catchment.precip = precip
    _run_catchment.temp = temp
    _run_catchment.evap = evap


def load_params_csv(params_csv_path):
    """Load calibrated_parameters.csv into {gauge_id (str): {param: value}},
    using only the 13 HBV parameter columns and ignoring metadata columns
    (calibration_kge / validation_kge / used_in_analysis)."""
    df = pd.read_csv(params_csv_path, dtype={"gauge_id": str})
    missing = [c for c in HBV_PARAM_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"params CSV missing HBV columns: {missing}")
    return {row["gauge_id"]: {p: row[p] for p in HBV_PARAM_COLS}
            for _, row in df.iterrows()}


def load_forcing(data_dir, rcp, ensemble):
    """Read the pr / tas / pet catchment-mean CSVs and check they share the
    same dates and gauge columns."""
    def _read(var):
        path = data_dir / f"{rcp}_{ensemble}_{var}_catchment_means_combined.csv"
        return pd.read_csv(path)

    pr_df, tas_df, pet_df = _read("pr"), _read("tas"), _read("pet")
    dates = pr_df["date"]
    for name, df in [("tas", tas_df), ("pet", pet_df)]:
        if not df["date"].equals(dates):
            raise ValueError(f"Date mismatch between pr and {name}")
        if list(df.columns) != list(pr_df.columns):
            raise ValueError(f"Column mismatch between pr and {name}")
    return pr_df, tas_df, pet_df


def _write_wide(results, catchment_ids, dates, out_path, key, float_fmt):
    """Assemble a {cid: series} result into a wide 'date' + gauge-columns CSV,
    preserving the input catchment order and skipping catchments that failed."""
    data = {"date": dates}
    for cid in catchment_ids:
        r = results.get(cid)
        if r is not None:
            data[cid] = r[key]
    df = pd.DataFrame(data)
    df.to_csv(out_path, index=False, float_format=float_fmt)
    print(f"  Written: {out_path.name}  shape={df.shape}", flush=True)
    return df


def run_hbv(rcp, ensemble, pr_df, tas_df, pet_df, params_by_catchment,
            output_dir, n_workers=N_WORKERS):
    dates = pr_df["date"]
    catchment_ids = [c for c in pr_df.columns if c != "date"]

    all_outputs = [("Q", "discharge", STATE_FLOAT_FMT)]
    all_outputs += [(k, s, STATE_FLOAT_FMT) for k, s in STATE_VARS]
    all_outputs += [(k, s, FLUX_FLOAT_FMT) for k, s in FLUX_VARS]
    out_paths = {suffix: output_dir / f"{rcp}_{ensemble}_hbv_{suffix}.csv"
                 for _, suffix, _ in all_outputs}

    if all(p.exists() for p in out_paths.values()):
        print(f"  [skip] all HBV outputs already exist for {rcp}/{ensemble}", flush=True)
        return

    print(f"\n  Model: HBV | {len(catchment_ids)} catchments", flush=True)

    precip = {cid: pr_df[cid].to_numpy() for cid in catchment_ids}
    temp = {cid: tas_df[cid].to_numpy() for cid in catchment_ids}
    evap = {cid: pet_df[cid].to_numpy() for cid in catchment_ids}

    work, skipped_params = [], []
    for cid in catchment_ids:
        params = params_by_catchment.get(cid)
        if params is None:
            skipped_params.append(cid)
        else:
            work.append((cid, params))
    if skipped_params:
        print(f"  WARNING: no parameters for {len(skipped_params)} catchments, skipping", flush=True)

    results, failed, completed = {}, [], 0
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                             initargs=(precip, temp, evap)) as pool:
        futures = {pool.submit(_run_catchment, cid, params): cid for cid, params in work}
        for future in as_completed(futures):
            cid, out = future.result()
            if out is not None:
                results[cid] = out
            else:
                failed.append(cid)
            completed += 1
            if completed % 100 == 0 or completed == len(work):
                print(f"    ...{completed}/{len(work)} done", flush=True)

    q_df = _write_wide(results, catchment_ids, dates,
                       out_paths["discharge"], "Q", STATE_FLOAT_FMT)
    sim_cols = [c for c in q_df.columns if c != "date"]
    if sim_cols:
        print(f"  Q range: {q_df[sim_cols].min().min():.3f} -- "
              f"{q_df[sim_cols].max().max():.3f} mm/day", flush=True)

    for key, suffix, fmt in all_outputs:
        if suffix == "discharge":
            continue
        _write_wide(results, catchment_ids, dates, out_paths[suffix], key, fmt)

    if failed:
        print(f"  FAILED (runtime error): {len(failed)} catchments: {failed}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run calibrated HBV (hbv-model's HBVModel) on CHESS-SCAPE forcing, "
                    "logging states, generating fluxes, and snowmelt.")
    parser.add_argument("--rcp", required=True, choices=["rcp26", "rcp45", "rcp60", "rcp85"])
    parser.add_argument("--ensemble", required=True, help="Ensemble member, e.g. 01, 04, 06, 15.")
    parser.add_argument("--params-csv", required=True,
                        help="Path to calibrated_parameters.csv (hbv-model).")
    parser.add_argument("--data-dir", type=Path, default=CHESS_SCAPE_ROOT,
                        help="Directory holding the forcing CSVs and receiving the outputs.")
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir

    print(f"{'='*60}\nRCP: {args.rcp}  |  Ensemble: {args.ensemble}\n{'='*60}")

    print("Loading calibrated parameters...", flush=True)
    params_by_catchment = load_params_csv(args.params_csv)
    print(f"  {len(params_by_catchment)} catchments with parameters", flush=True)

    print("Loading forcing data...", flush=True)
    try:
        pr_df, tas_df, pet_df = load_forcing(data_dir, args.rcp, args.ensemble)
    except FileNotFoundError as e:
        print(f"ERROR: forcing file not found: {e}")
        sys.exit(1)

    n_dates = len(pr_df["date"])
    n_catch = len([c for c in pr_df.columns if c != "date"])
    print(f"  {n_dates} timesteps, {n_catch} catchments", flush=True)

    run_hbv(args.rcp, args.ensemble, pr_df, tas_df, pet_df,
            params_by_catchment, data_dir, n_workers=args.workers)

    print(f"\nAll done for {args.rcp} / {args.ensemble}.")


if __name__ == "__main__":
    main()
