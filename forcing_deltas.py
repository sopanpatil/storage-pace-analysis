"""
forcing_deltas.py

Per-catchment baseline -> future change in climate forcing, from the CHESS-SCAPE
catchment-mean pr / pet / tas files, for feeding into responder_characterisation.py.

Rationale: static CAMELS attributes barely separate the catchments whose slow
flood-to-drought transitions intensify under warming. That points to the driver
being WHERE the forcing changes most, not what kind of catchment it is. This
computes each catchment's change in aridity, summer PET, summer rainfall,
seasonality and temperature (ensemble-median across members) so those Delta-forcing
variables can be tested as discriminators.

Input files (same wide layout as the HBV outputs: col 'date' + gauge-ID columns):
    <dir>/<rcp>_<member>_pr_catchment_means_combined.csv    (mm/day)
    <dir>/<rcp>_<member>_pet_catchment_means_combined.csv   (mm/day)
    <dir>/<rcp>_<member>_tas_catchment_means_combined.csv    (degC)

Windows and calendar match slow_transition_analysis.py (360-day, string-date
slicing; baseline WY1982-2010, future WY2051-2080).

Output: one row per gauge with baseline_/future_/d_ columns for each metric.

    python forcing_deltas.py --jasmin-dir chess_scape_output --rcp rcp85 \
        --out forcing_deltas_rcp85.csv
    python forcing_deltas.py            # synthetic self-test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import selftest_io

MEMBERS = ("01", "04", "06", "15")
BASE_START, BASE_END = "1981-10-01", "2010-09-30"
FUT_START, FUT_END = "2050-10-01", "2080-09-30"
METRICS = ["p_annual", "pet_annual", "aridity", "p_summer", "pet_summer",
           "aridity_summer", "seasonality", "tas_mean"]


def _read(d: str, rcp: str, member: str, var: str,
          gauges: list[str] | None) -> pd.DataFrame:
    usecols = (["date"] + gauges) if gauges else None
    return pd.read_csv(Path(d) / f"{rcp}_{member}_{var}_catchment_means_combined.csv",
                       usecols=usecols)


def _masks_and_month(date_col: pd.Series):
    s = date_col.astype(str)
    d10 = s.str[:10].to_numpy()
    month = s.str[5:7].astype(int).to_numpy()
    base = (d10 >= BASE_START) & (d10 <= BASE_END)
    fut = (d10 >= FUT_START) & (d10 <= FUT_END)
    return base, fut, month


def period_stats(pr, pet, tas, month, mask) -> dict:
    """Vectorised over the gauge axis. Arrays are (time, gauge)."""
    idx = np.flatnonzero(mask)
    P, E, T, M = pr[idx], pet[idx], tas[idx], month[idx]
    jja = np.isin(M, [6, 7, 8])
    djf = np.isin(M, [12, 1, 2])

    def _safe_ratio(num, den):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)

    Psum, Pmean = P.sum(0), P.mean(0)
    return {
        "p_annual": Pmean * 360.0,
        "pet_annual": E.mean(0) * 360.0,
        "aridity": _safe_ratio(E.sum(0), Psum),
        "p_summer": P[jja].mean(0) * 90.0,
        "pet_summer": E[jja].mean(0) * 90.0,
        "aridity_summer": _safe_ratio(E[jja].sum(0), P[jja].sum(0)),
        "seasonality": _safe_ratio(P[djf].mean(0) - P[jja].mean(0), Pmean),
        "tas_mean": T.mean(0),
    }


def compute(d: str, rcp: str, members=MEMBERS,
            gauges: list[str] | None = None) -> pd.DataFrame:
    base_stack = {m: [] for m in METRICS}
    fut_stack = {m: [] for m in METRICS}
    gcols = None
    for mem in members:
        pr = _read(d, rcp, mem, "pr", gauges)
        pet = _read(d, rcp, mem, "pet", gauges)
        tas = _read(d, rcp, mem, "tas", gauges)
        base, fut, month = _masks_and_month(pr["date"])
        gcols = [c for c in pr.columns if c != "date"]
        A = {"pr": pr[gcols].to_numpy(), "pet": pet[gcols].to_numpy(),
             "tas": tas[gcols].to_numpy()}
        bs = period_stats(A["pr"], A["pet"], A["tas"], month, base)
        fs = period_stats(A["pr"], A["pet"], A["tas"], month, fut)
        for m in METRICS:
            base_stack[m].append(bs[m])
            fut_stack[m].append(fs[m])
        print(f"  {rcp} member {mem}: {len(gcols)} gauges")

    out = {"gauge_id": gcols}
    for m in METRICS:
        b = np.median(np.vstack(base_stack[m]), axis=0)   # ensemble median
        f = np.median(np.vstack(fut_stack[m]), axis=0)
        out[f"baseline_{m}"] = b
        out[f"future_{m}"] = f
        out[f"d_{m}"] = f - b
    return pd.DataFrame(out)


def sanity(df: pd.DataFrame) -> str:
    """Physical sanity check -- absurd numbers here mean the files were misread."""
    def stat(col):
        s = df[col]
        return f"median={s.median():.2f} [{s.quantile(.05):.2f}, {s.quantile(.95):.2f}]"
    return ("\n  SANITY (expect GB-plausible values):\n"
            f"    baseline annual rainfall (mm): {stat('baseline_p_annual')}\n"
            f"    baseline aridity (PET/P):      {stat('baseline_aridity')}\n"
            f"    warming dTas (degC):           {stat('d_tas_mean')}\n"
            f"    d aridity:                     {stat('d_aridity')}\n"
            f"    d summer rainfall (mm/season): {stat('d_p_summer')}")


# --------------------------------------------------------------------------- #
# Synthetic self-test                                                         #
# --------------------------------------------------------------------------- #
def _make_mock(tmp: Path) -> str:
    n = 360 * 100
    y, mo, day = 1980, 12, 1
    dates = []
    for _ in range(n):
        dates.append(f"{y:04d}-{mo:02d}-{day:02d} 12:00:00")
        day += 1
        if day > 30:
            day = 1; mo += 1
        if mo > 12:
            mo = 1; y += 1
    dates = np.array(dates)
    doy = np.arange(n) % 360
    season = np.cos(2 * np.pi * doy / 360)     # +1 winter, -1 summer
    is_future = np.array([int(s[:4]) >= 2050 for s in dates])
    gauges = [f"g{i}" for i in range(6)]
    rng = np.random.default_rng(0)
    for mem in MEMBERS:
        for var in ("pr", "pet", "tas"):
            M = {"date": dates}
            for gi, g in enumerate(gauges):
                if var == "pr":
                    base = 3.0 + 1.5 * season + rng.gamma(0.4, 2, n) * (0.5 + 0.3 * season)
                    base = base - is_future * (0.4 - 0.3 * season) * (gi / 5)  # drier summers, gauge-varying
                elif var == "pet":
                    base = np.clip(1.5 - 1.2 * season, 0.1, None) + is_future * 0.4 * (gi / 5)
                else:
                    base = 9 - 7 * season + is_future * 3.0
                M[g] = np.clip(base, 0 if var != "tas" else -50, None)
            pd.DataFrame(M).to_csv(tmp / f"rcp85_{mem}_{var}_catchment_means_combined.csv",
                                   index=False)
    return str(tmp)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jasmin-dir", type=str, default=None)
    ap.add_argument("--rcp", type=str, default="rcp85")
    ap.add_argument("--gauges", type=str, default=None)
    ap.add_argument("--out", type=str, default=None,
                    help="Output CSV (default: forcing_deltas_<rcp>.csv).")
    args = ap.parse_args()

    out = selftest_io.redirect(args.out or f"forcing_deltas_{args.rcp}.csv",
                               selftest=not args.jasmin_dir)
    if args.jasmin_dir:
        gauges = args.gauges.split(",") if args.gauges else None
        df = compute(args.jasmin_dir, args.rcp, gauges=gauges)
    else:
        print("Self-test on synthetic forcing (planted gauge-varying summer drying).\n")
        tmp = selftest_io.redirect("syn_forcing", True); tmp.mkdir(exist_ok=True)
        df = compute(_make_mock(tmp), "rcp85")

    print(sanity(df))
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} catchments x {len(METRICS)} metrics -> "
          f"{Path(out).resolve()}")
    if not args.jasmin_dir:
        selftest_io.announce([out])


if __name__ == "__main__":
    main()
