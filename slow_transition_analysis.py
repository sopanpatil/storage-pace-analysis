"""
slow_transition_analysis.py

Diagnostic for NON-ABRUPT (slow-regime) flood-drought transitions from HBV
discharge + internal storage states forced by CHESS-SCAPE.

Complement to the abrupt-regime pipeline (companion dfaa-analysis repository), which
CENSORS transitions with a gap > 90 days. This tool removes that cap and asks
two questions the abrupt design cannot:

  (1) Does a COHERENT slow tail exist once storage-based coherence is demanded,
      or does the >90-day population fall apart into incoherent seasonal drift?
  (2) Which HBV store (SM / UZ / LZ / snow) is RATE-LIMITING each transition?
      Hypothesis: abrupt = SM/UZ-limited; slow tail = LZ-limited.

Key methodological pivot vs the abrupt pipeline
------------------------------------------------
The abrupt pipeline pairs on DISCHARGE-event adjacency ("no intervening event
of either type"), which structurally censors slow transitions: minor peaks
during a long drawdown either reset the precursor or leave it pending. Here,
supersession is redefined on the STORAGE TRAJECTORY -- a later flood only resets
the flood precursor if total liquid storage RE-SATURATES near its flood-state
level (mirrored for droughts). Minor peaks that do not refill storage are
tolerated, and a pair is retained only if the storage path between endpoints is
near-monotone (coherence filter). That is the defence against "this is just the
annual cycle": incoherent, wobbling paths are rejected, not counted.

INPUT SCHEMA (one long dataframe)
---------------------------------
    gauge_id : int/str
    rcp      : str    e.g. 'rcp85'
    member   : str    e.g. '01'
    period   : str    'baseline' | 'future'
    t        : int    integer day index (avoids 360-day-calendar arithmetic;
                      matches the string-date handling in the abrupt pipeline)
    Q        : float  simulated discharge (mm/day)
    SM       : float  HBV soil-moisture store (mm)
    UZ       : float  HBV upper-zone store (mm)
    LZ       : float  HBV lower-zone store (mm)
    SNOW     : float  HBV snowpack water equivalent (mm)

Thresholds Q5 / Q80 are computed on the BASELINE period and held FIXED (as in
the abrupt pipeline) so that changes reflect distribution shifts, not a moving
definition. For real ensemble runs, pass ensemble-median thresholds via
`thresholds=`.

The 90-day window is gone, but a generous 720-day (two water year) bound is
retained to exclude multi-annual pairings, which fall outside the
seasonal-to-annual regime the paper examines. That is Config.max_gap_days, and
720 is the production value used for every reported result -- do not run
uncensored and then read numbers off the result.

Run `python slow_transition_analysis.py` with no arguments to execute a
self-test on synthetic fast / slow / snow catchment archetypes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

import selftest_io


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    q_flood_pctl: float = 95.0     # Q5: flow exceeded 5% of the time
    q_drought_pctl: float = 20.0   # Q80: flow exceeded 80% of the time
    flood_min_dur: int = 1         # days
    flood_pool_gap: int = 5        # floods < this many days apart are merged
    drought_min_spell: int = 5     # days; droughts are NOT pooled
    max_gap_days: int | None = 720   # two water years; None = uncensored
    resaturation_delta: float = 0.15  # storage re-saturation tolerance (norm.)
    coherence_min: float = 0.60    # keep pairs with near-monotone storage path
    coherence_store: str = "ratelim"  # 'ratelim' (rate-limiting store) | 'total'
    min_storage_change_mm: float = 2.0  # ignore transitions with trivial total dS
    min_ratelim_change_mm: float = 1.0  # ...and trivial rate-limiting-store dS
    terminal_frac: float = 0.20    # terminal window = this fraction of the gap
    terminal_min_days: int = 14    # ...but at least this many days
    snow_driver_mm: float = 5.0    # |snow change| above this flags a snow driver
    abrupt_cutoff_days: int = 90   # boundary used ONLY for reporting/labelling
    # Terminal-control candidate set. In HBV, SM generates NO streamflow (Q comes
    # from UZ + LZ), so a discharge-threshold crossing should be attributed to the
    # runoff stores. 'flow' = HBV flow-weighted (fast Q0+Q1 from UZ vs baseflow Q2
    # from LZ), physically exact, needs calibrated K0/K1/K2/UZL; 'runoff-norm' =
    # {UZ,LZ} storage, range-normalised; 'runoff' = {UZ,LZ} raw mm; 'legacy' =
    # {SM,UZ,LZ} raw mm (SM-biased).
    attribution: str = "runoff-norm"
    validate_fluxes: bool = False  # print max|logged-reconstructed| per member


# --------------------------------------------------------------------------- #
# Event detection (mirrors the abrupt pipeline)                               #
# --------------------------------------------------------------------------- #
def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive (start, end) index runs where mask is True."""
    if not mask.any():
        return []
    idx = np.flatnonzero(mask)
    splits = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[splits + 1]))
    ends = np.concatenate((idx[splits], [idx[-1]]))
    return list(zip(starts.tolist(), ends.tolist()))


def detect_floods(Q: np.ndarray, q5: float, cfg: Config) -> list[tuple[int, int]]:
    runs = _runs(Q > q5)
    runs = [r for r in runs if (r[1] - r[0] + 1) >= cfg.flood_min_dur]
    if not runs:
        return []
    # pool floods separated by < flood_pool_gap days
    pooled = [list(runs[0])]
    for s, e in runs[1:]:
        if s - pooled[-1][1] - 1 < cfg.flood_pool_gap:
            pooled[-1][1] = e
        else:
            pooled.append([s, e])
    return [tuple(r) for r in pooled]


def detect_droughts(Q: np.ndarray, q80: float, cfg: Config) -> list[tuple[int, int]]:
    runs = _runs(Q < q80)
    return [r for r in runs if (r[1] - r[0] + 1) >= cfg.drought_min_spell]  # no pooling


# --------------------------------------------------------------------------- #
# Storage-based pairing (the slow-regime core)                                #
# --------------------------------------------------------------------------- #
def _normalise(x: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def pair_transitions(
    floods: list[tuple[int, int]],
    droughts: list[tuple[int, int]],
    total_S: np.ndarray,
    direction: str,
    cfg: Config,
) -> list[tuple[int, int]]:
    """
    One-to-one, temporally ordered pairing with STORAGE-based supersession.

    FTD: precursor = flood (high-storage state); superseded only if a later
         flood RE-SATURATES storage (normalised storage returns within
         `resaturation_delta` of the precursor's flood-state level).
    DTF: mirror -- precursor = drought (low-storage state); superseded only if a
         later drought RE-DEPLETES storage back near the precursor level.
    """
    s = _normalise(total_S)
    if direction == "FTD":
        precursors = [("P", e) for _, e in floods]     # use flood END
        followers = [("F", s0) for s0, _ in droughts]  # use drought ONSET
        superseded = lambda s_new, s0: s_new >= s0 - cfg.resaturation_delta
    elif direction == "DTF":
        precursors = [("P", e) for _, e in droughts]   # use drought END
        followers = [("F", s0) for s0, _ in floods]    # use flood ONSET
        superseded = lambda s_new, s0: s_new <= s0 + cfg.resaturation_delta
    else:
        raise ValueError(direction)

    events = sorted(precursors + followers, key=lambda z: z[1])
    pending: tuple[int, float] | None = None  # (t0, normalised storage at t0)
    pairs: list[tuple[int, int]] = []

    for kind, t in events:
        if kind == "P":
            if pending is None:
                pending = (t, s[t])
            elif superseded(s[t], pending[1]):
                pending = (t, s[t])
            # else: minor event -- keep original precursor (tolerate the wiggle)
        else:  # follower
            if pending is not None:
                t0, _ = pending
                gap = t - t0
                if gap >= 0 and (cfg.max_gap_days is None or gap <= cfg.max_gap_days):
                    pairs.append((t0, t))
                pending = None  # consumed
    return pairs


def coherence(total_S: np.ndarray, t0: int, t1: int) -> float:
    """Net change / total variation over [t0, t1]. 1 = perfectly monotone."""
    seg = total_S[t0:t1 + 1]
    if len(seg) < 2:
        return 1.0
    net = abs(seg[-1] - seg[0])
    tv = np.abs(np.diff(seg)).sum()
    return float(net / tv) if tv > 1e-9 else 1.0


# --------------------------------------------------------------------------- #
# Store attribution                                                           #
# --------------------------------------------------------------------------- #
def reconstruct_flows(UZ: np.ndarray, LZ: np.ndarray, p: dict) -> tuple[np.ndarray, np.ndarray]:
    """Exact HBV flow components from END-OF-TIMESTEP stored states + params.
    Verified machine-precise against hbv-model's hbv.py. fast (Q0+Q1) is the UZ outflow,
    slow (Q2) is the LZ baseflow. Stored UZ is post-Q0/Q1, LZ post-Q2, hence
    the 1/(1-K) back-outs."""
    K0, K1, K2, UZL = p["K0"], p["K1"], p["K2"], p["UZL"]
    UZ_b = UZ / (1.0 - K1)
    Q1 = K1 * UZ_b
    Q0 = np.where(UZ_b > UZL, K0 * (UZ_b - UZL) / (1.0 - K0), 0.0)
    Q2 = K2 * LZ / (1.0 - K2)
    return Q0 + Q1, Q2          # (fast <- UZ, slow <- LZ)


def attribute(
    df: pd.DataFrame, t0: int, t1: int, direction: str, cfg: Config,
    params: dict | None = None,
) -> dict:
    """Magnitude partition + terminal-control classification for one transition."""
    stores = ["SM", "UZ", "LZ"]
    sign = -1.0 if direction == "FTD" else 1.0  # depletion (FTD) vs recharge (DTF)

    # net change in the transition direction (positive = "did work")
    dS = {k: sign * (df[k].iloc[t1] - df[k].iloc[t0]) for k in stores}
    active = {k: v for k, v in dS.items() if v > 0}
    total_active = sum(active.values())
    mag_frac = {k: (active.get(k, 0.0) / total_active if total_active > 1e-9 else np.nan)
                for k in stores}

    gap = t1 - t0
    w = max(cfg.terminal_min_days, int(cfg.terminal_frac * gap))
    ts = max(t0, t1 - w)

    if cfg.attribution == "flow":
        # Physically exact: classify by which HBV FLOW component (fast<-UZ vs
        # baseflow<-LZ) carries the terminal change in discharge across the
        # threshold. SM has no outflow and cannot be rate-limiting by construction.
        if params is None:
            raise ValueError("attribution='flow' requires per-catchment params "
                             "(K0,K1,K2,UZL).")
        if "FAST" in df.columns and "SLOW" in df.columns:
            # logged generating fluxes saved during the model run (JASMIN path)
            qf, qs = df["FAST"].to_numpy(), df["SLOW"].to_numpy()
        else:
            # fallback: reconstruct from stored states (synthetic self-test)
            qf, qs = reconstruct_flows(df["UZ"].to_numpy(), df["LZ"].to_numpy(), params)
        dfast = sign * (qf[t1] - qf[ts])
        dslow = sign * (qs[t1] - qs[ts])
        active_t = {k: v for k, v in {"UZ": dfast, "LZ": dslow}.items() if v > 0}
        # magnitude partition also on flow, over the whole interval
        mf, ms = sign * (qf[t1] - qf[t0]), sign * (qs[t1] - qs[t0])
        act_m = {k: v for k, v in {"UZ": mf, "LZ": ms}.items() if v > 0}
        tot_m = sum(act_m.values())
        mag_frac = {"SM": np.nan,
                    "UZ": act_m.get("UZ", 0.0) / tot_m if tot_m > 1e-9 else np.nan,
                    "LZ": act_m.get("LZ", 0.0) / tot_m if tot_m > 1e-9 else np.nan}
    else:
        # storage-based terminal control (legacy / runoff / runoff-norm)
        candidates = ["SM", "UZ", "LZ"] if cfg.attribution == "legacy" else ["UZ", "LZ"]
        dS_term = {k: sign * (df[k].iloc[t1] - df[k].iloc[ts]) for k in candidates}
        if cfg.attribution == "runoff-norm":
            rng = {k: float(df[k].max() - df[k].min()) for k in candidates}
            dS_term = {k: (v / rng[k] if rng[k] > 1e-9 else 0.0)
                       for k, v in dS_term.items()}
        active_t = {k: v for k, v in dS_term.items() if v > 0}

    total_t = sum(active_t.values())
    term_frac = {k: (active_t.get(k, 0.0) / total_t if total_t > 1e-9 else np.nan)
                 for k in stores}   # SM ~ context only under runoff/flow
    rate_limiting = max(active_t, key=active_t.get) if active_t else "none"

    # snow: source for DTF (melt) / competing sink for FTD (accumulation)
    d_snow = df["SNOW"].iloc[t1] - df["SNOW"].iloc[t0]
    if direction == "DTF":
        snow_driver = (-d_snow) > cfg.snow_driver_mm   # snow decreased -> melt fed recharge
    else:
        snow_driver = d_snow > cfg.snow_driver_mm      # snow accumulated -> water locked up

    total_S = (df["SM"] + df["UZ"] + df["LZ"]).to_numpy()
    # Per-store coherence. coh_ratelim is what the filter runs on (see
    # analyse_series, cfg.coherence_store="ratelim"): a genuine slow drawdown can
    # re-wet at the surface while the rate-limiting store declines monotonically,
    # so total-storage coherence would reject it. The others are kept as
    # diagnostics and for the cfg.coherence_store="total" comparison.
    per_store_coh = {k: coherence(df[k].to_numpy(), t0, t1) for k in stores}
    coh_ratelim = per_store_coh.get(rate_limiting, np.nan)
    d_ratelim = float(dS.get(rate_limiting, np.nan))
    return {
        "gap_days": gap,
        "regime": "abrupt" if gap <= cfg.abrupt_cutoff_days else "slow",
        "d_total_storage_mm": float(sign * (total_S[t1] - total_S[t0])),
        "d_ratelim_mm": d_ratelim,
        "d_SM_mm": float(dS["SM"]),                        # antecedent context
        "attribution": cfg.attribution,
        "coherence": coherence(total_S, t0, t1),          # total storage
        "coh_SM": per_store_coh["SM"], "coh_UZ": per_store_coh["UZ"],
        "coh_LZ": per_store_coh["LZ"], "coh_ratelim": coh_ratelim,
        "mag_SM": mag_frac["SM"], "mag_UZ": mag_frac["UZ"], "mag_LZ": mag_frac["LZ"],
        "term_SM": term_frac["SM"], "term_UZ": term_frac["UZ"], "term_LZ": term_frac["LZ"],
        "rate_limiting_store": rate_limiting,
        "snow_driver": bool(snow_driver),
        "Q_start": float(df["Q"].iloc[t0]), "Q_end": float(df["Q"].iloc[t1]),
    }


# --------------------------------------------------------------------------- #
# Per-series driver                                                           #
# --------------------------------------------------------------------------- #
def analyse_series(
    df: pd.DataFrame, direction: str, cfg: Config,
    thresholds: tuple[float, float] | None = None,
    params: dict | None = None,
) -> pd.DataFrame:
    """
    df: one (gauge_id, rcp, member, period) series, sorted by t.
    thresholds: (q5, q80) if precomputed on baseline; else computed from df.
    params: per-catchment HBV params (K0,K1,K2,UZL) -- required for attribution='flow'.
    """
    df = df.sort_values("t").reset_index(drop=True)
    Q = df["Q"].to_numpy()
    if thresholds is None:
        q5 = np.percentile(Q, cfg.q_flood_pctl)
        q80 = np.percentile(Q, cfg.q_drought_pctl)
    else:
        q5, q80 = thresholds

    floods = detect_floods(Q, q5, cfg)
    droughts = detect_droughts(Q, q80, cfg)
    total_S = (df["SM"] + df["UZ"] + df["LZ"]).to_numpy()
    pairs = pair_transitions(floods, droughts, total_S, direction, cfg)

    rows = []
    for t0, t1 in pairs:
        rec = attribute(df, t0, t1, direction, cfg, params=params)
        if cfg.coherence_store == "ratelim":
            coh_ok = rec["coh_ratelim"] >= cfg.coherence_min
            change_ok = (rec["rate_limiting_store"] != "none"
                         and abs(rec["d_ratelim_mm"]) >= cfg.min_ratelim_change_mm)
        else:  # 'total' -- reproduces the earlier behaviour
            coh_ok = rec["coherence"] >= cfg.coherence_min
            change_ok = abs(rec["d_total_storage_mm"]) >= cfg.min_storage_change_mm
        rec["passes_coherence"] = bool(coh_ok and change_ok)
        for key in ("gauge_id", "rcp", "member", "period"):
            if key in df.columns:
                rec[key] = df[key].iloc[0]
        rec["direction"] = direction
        rec["t_start"], rec["t_end"] = int(t0), int(t1)
        rows.append(rec)
    return pd.DataFrame(rows)


def analyse(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Loop over all (gauge, rcp, member, period) groups and both directions.

    Baseline-derived Q5/Q80 are computed once per (gauge, rcp, member) and
    reused for the future period, matching the fixed-threshold design.
    """
    keys = [k for k in ("gauge_id", "rcp", "member") if k in df.columns]
    out = []
    for _, sub in df.groupby(keys) if keys else [((), df)]:
        base = sub[sub["period"] == "baseline"] if "period" in sub else sub
        Qb = base["Q"].to_numpy()
        thr = (np.percentile(Qb, cfg.q_flood_pctl),
               np.percentile(Qb, cfg.q_drought_pctl))
        periods = sub.groupby("period") if "period" in sub else [("all", sub)]
        for _, per in periods:
            for direction in ("FTD", "DTF"):
                out.append(analyse_series(per, direction, cfg, thresholds=thr))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# --------------------------------------------------------------------------- #
# JASMIN loader: raw HBV wide CSVs -> schema, in memory-bounded passes         #
# --------------------------------------------------------------------------- #
# Files: <dir>/<rcp>_<member>_hbv_<suffix>.csv, wide (col 'date' + gauge-ID
# columns), one continuous 360-day-calendar series 1980-12 .. 2080-11.
RCPS = ("rcp26", "rcp45", "rcp60", "rcp85")
MEMBERS = ("01", "04", "06", "15")
VAR_SUFFIX = {"Q": "discharge", "SM": "sm", "UZ": "uz", "LZ": "lz", "SNOW": "sp"}
# Logged HBV generating fluxes (pre-routing), saved DURING the model run at full
# precision: FAST = Q0+Q1 (upper zone), SLOW = Q2 (lower zone). Read directly in
# attribution="flow" instead of reconstructing from the %.4f-stored UZ/LZ states.
FLUX_SUFFIX = {"Q0": "q0", "Q1": "q1", "Q2": "q2"}
_ALL_SUFFIX = {**VAR_SUFFIX, **FLUX_SUFFIX}
# Water-year windows (string-date bounds; dates compared on their first 10 chars
# so the 360-day calendar needs no arithmetic), matching the companion
# abrupt-regime pipeline (dfaa-analysis).
BASE_START, BASE_END = "1981-10-01", "2010-09-30"   # WY1982-2010
FUT_START, FUT_END = "2050-10-01", "2080-09-30"     # WY2051-2080


def _hbv_path(d: str, rcp: str, member: str, var: str) -> Path:
    return Path(d) / f"{rcp}_{member}_hbv_{_ALL_SUFFIX[var]}.csv"


def _read_hbv(d: str, rcp: str, member: str, var: str,
              gauges: list[str] | None) -> pd.DataFrame:
    usecols = (["date"] + gauges) if gauges else None
    df = pd.read_csv(_hbv_path(d, rcp, member, var), usecols=usecols)
    return df


def _window_masks(date_col: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    d10 = date_col.astype(str).str[:10].to_numpy()
    base = (d10 >= BASE_START) & (d10 <= BASE_END)
    fut = (d10 >= FUT_START) & (d10 <= FUT_END)
    return base, fut


def _ensemble_thresholds(d: str, rcp: str, members: tuple[str, ...],
                         gauges: list[str] | None, cfg: Config) -> dict:
    """Per-gauge Q5/Q80 = median across members of each member's BASELINE
    percentiles (matches the fixed, ensemble-median threshold used in the
    companion abrupt-regime pipeline)."""
    q5_stack, q80_stack, cols = [], [], None
    for m in members:
        dis = _read_hbv(d, rcp, m, "Q", gauges)
        base, _ = _window_masks(dis["date"])
        gcols = [c for c in dis.columns if c != "date"]
        cols = gcols
        vals = dis.loc[base, gcols].to_numpy()
        q5_stack.append(np.percentile(vals, cfg.q_flood_pctl, axis=0))
        q80_stack.append(np.percentile(vals, cfg.q_drought_pctl, axis=0))
    q5 = np.median(np.vstack(q5_stack), axis=0)
    q80 = np.median(np.vstack(q80_stack), axis=0)
    return {g: (float(q5[i]), float(q80[i])) for i, g in enumerate(cols)}


def load_params(path: str, only_used: bool = True) -> dict:
    """Load calibrated HBV params keyed by gauge_id. Needs columns
    gauge_id, K0, K1, K2, UZL (extra columns ignored). If a 'used_in_analysis'
    column is present and only_used=True, restrict to those catchments (the companion pipeline's
    accepted set) so flow attribution rests only on well-calibrated parameters."""
    p = Path(path)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    need = {"gauge_id", "K0", "K1", "K2", "UZL"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"params file missing columns: {sorted(missing)}. "
                         f"Found: {list(df.columns)}")
    n_all = len(df)
    if only_used and "used_in_analysis" in df.columns:
        df = df[df["used_in_analysis"].astype(bool)]
        print(f"  params: {len(df)} of {n_all} catchments "
              f"(used_in_analysis=True)")
    df["gauge_id"] = df["gauge_id"].astype(str)
    return {r["gauge_id"]: {"K0": float(r["K0"]), "K1": float(r["K1"]),
                            "K2": float(r["K2"]), "UZL": float(r["UZL"])}
            for _, r in df.iterrows()}


def run_jasmin(d: str, cfg: Config, gauges: list[str] | None = None,
               rcps: tuple[str, ...] = RCPS,
               members: tuple[str, ...] = MEMBERS,
               params_by_gauge: dict | None = None) -> pd.DataFrame:
    """Load raw HBV outputs and extract transitions. One member's five wide
    files are held at a time; per-gauge series are built as views."""
    if cfg.attribution == "flow" and not params_by_gauge:
        raise ValueError("attribution='flow' needs --params <calibrated params>.")
    missing_params = set()
    out = []
    for rcp in rcps:
        thr = _ensemble_thresholds(d, rcp, members, gauges, cfg)
        for m in members:
            _load_vars = list(VAR_SUFFIX) + (
                list(FLUX_SUFFIX) if cfg.attribution == "flow" else [])
            wide = {v: _read_hbv(d, rcp, m, v, gauges) for v in _load_vars}
            base, fut = _window_masks(wide["Q"]["date"])
            gcols = [c for c in wide["Q"].columns if c != "date"]
            if params_by_gauge is not None:
                gcols = [g for g in gcols if g in params_by_gauge]  # -> 621 set
            arrs = {v: wide[v][gcols].to_numpy() for v in wide}  # (time, gauge)
            if cfg.validate_fluxes and cfg.attribution == "flow":
                K0 = np.array([params_by_gauge[g]["K0"] for g in gcols])
                K1 = np.array([params_by_gauge[g]["K1"] for g in gcols])
                K2 = np.array([params_by_gauge[g]["K2"] for g in gcols])
                UZLa = np.array([params_by_gauge[g]["UZL"] for g in gcols])
                UZ_b = arrs["UZ"] / (1.0 - K1)
                Q0r = np.where(UZ_b > UZLa, K0 * (UZ_b - UZLa) / (1.0 - K0), 0.0)
                Q1r = K1 * UZ_b
                Q2r = K2 * arrs["LZ"] / (1.0 - K2)
                mfast = np.nanmax(np.abs((arrs["Q0"] + arrs["Q1"]) - (Q0r + Q1r)))
                mslow = np.nanmax(np.abs(arrs["Q2"] - Q2r))
                print(f"  [validate] {rcp} {m}: max|logged-recon| "
                      f"fast={mfast:.2e} slow={mslow:.2e} mm/day "
                      f"(expect ~1e-4, bounded by %.4f states)")
            for gi, g in enumerate(gcols):
                gp = params_by_gauge.get(g) if params_by_gauge else None
                if cfg.attribution == "flow" and gp is None:
                    missing_params.add(g)
                    continue
                for period, mask in (("baseline", base), ("future", fut)):
                    idx = np.flatnonzero(mask)
                    _sd = {
                        "Q": arrs["Q"][idx, gi], "SM": arrs["SM"][idx, gi],
                        "UZ": arrs["UZ"][idx, gi], "LZ": arrs["LZ"][idx, gi],
                        "SNOW": arrs["SNOW"][idx, gi],
                    }
                    if cfg.attribution == "flow":
                        _sd["FAST"] = arrs["Q0"][idx, gi] + arrs["Q1"][idx, gi]
                        _sd["SLOW"] = arrs["Q2"][idx, gi]
                    sdf = pd.DataFrame(_sd)
                    sdf["t"] = np.arange(len(sdf))
                    sdf["gauge_id"] = g
                    sdf["rcp"] = rcp
                    sdf["member"] = m
                    sdf["period"] = period
                    for direction in ("FTD", "DTF"):
                        out.append(analyse_series(sdf, direction, cfg,
                                                  thresholds=thr[g], params=gp))
            print(f"  {rcp} member {m}: {len(gcols)} gauges processed")
    if missing_params:
        print(f"  WARNING: {len(missing_params)} gauges skipped (no calibrated "
              f"params): {sorted(missing_params)[:5]}...")
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #
def summarise(tr: pd.DataFrame) -> str:
    if tr.empty:
        return "No transitions detected."
    lines = []
    for direction in ("FTD", "DTF"):
        d = tr[tr["direction"] == direction]
        if d.empty:
            continue
        coh = d[d["passes_coherence"]]
        lines.append(f"\n=== {direction} ===")
        lines.append(f"  transitions (all / coherent):      {len(d)} / {len(coh)}")
        for label, sub in (("all pairs", d), ("coherent only", coh)):
            if sub.empty:
                continue
            slow = sub[sub["regime"] == "slow"]
            frac_slow = len(slow) / len(sub)
            lines.append(
                f"  [{label}] gap median={sub['gap_days'].median():.0f}d  "
                f"90th={sub['gap_days'].quantile(.9):.0f}d  "
                f">90d share={frac_slow:.2f}")
        if not coh.empty:
            # store attribution, coherent transitions, abrupt vs slow
            for regime in ("abrupt", "slow"):
                r = coh[coh["regime"] == regime]
                if r.empty:
                    continue
                mix = r["rate_limiting_store"].value_counts(normalize=True)
                mix_str = ", ".join(f"{k}:{v:.2f}" for k, v in mix.items())
                lines.append(f"  rate-limiting store ({regime}, coherent): {mix_str}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Synthetic self-test: HBV-lite fast / slow / snow archetypes                 #
# --------------------------------------------------------------------------- #
def _simulate(archetype: str, n_days: int, seed: int, warming: float = 0.0) -> pd.DataFrame:
    """Minimal HBV-like model: snow -> SM (ET) -> UZ (fast) -> LZ (slow) -> Q."""
    rng = np.random.default_rng(seed)
    day = np.arange(n_days)
    season = 0.5 * (1 + np.cos(2 * np.pi * (day % 360) / 360))  # 1 in winter
    temp = 8 - 7 * np.cos(2 * np.pi * (day % 360) / 360) + warming + rng.normal(0, 2, n_days)
    # precipitation: seasonal wet + stochastic storms
    P = rng.gamma(0.35, 4.0, n_days) * (0.4 + season)
    pet = np.clip(0.3 + 0.12 * np.maximum(temp, 0), 0, None) * (1 + 0.3 * warming)

    params = {
        "fast":  dict(fc=120, k1=0.35, k2=0.10, perc=3.0, snow=False),  # flashy, empties fast
        "slow":  dict(fc=200, k1=0.20, k2=0.010, perc=1.2, snow=False), # chalk: slow LZ
        "snow":  dict(fc=140, k1=0.30, k2=0.05, perc=2.0, snow=True),   # upland w/ snow
    }[archetype]

    SM = UZ = LZ = SNW = 0.0
    fc, k1, k2, perc = params["fc"], params["k1"], params["k2"], params["perc"]
    out = np.zeros((n_days, 5))
    for i in range(n_days):
        p = P[i]
        if params["snow"]:
            if temp[i] < 0:
                SNW += p; p = 0.0
            elif SNW > 0:
                melt = min(SNW, 2.5 * max(temp[i], 0))
                SNW -= melt; p += melt
        # soil moisture + ET
        SM += p
        recharge = p * (SM / fc) ** 2 if SM > 0 else 0.0
        SM -= recharge
        et = min(SM, pet[i] * min(SM / (0.6 * fc), 1.0))
        SM = max(SM - et, 0.0)
        # reservoirs
        UZ += recharge
        percol = min(UZ, perc)
        UZ -= percol; LZ += percol
        q_uz = k1 * UZ; UZ -= q_uz
        q_lz = k2 * LZ; LZ -= q_lz
        out[i] = [q_uz + q_lz, SM, UZ, LZ, SNW]
    return pd.DataFrame(out, columns=["Q", "SM", "UZ", "LZ", "SNOW"]).assign(t=day)


def _make_synthetic() -> pd.DataFrame:
    frames = []
    n = 360 * 20  # 20 water years
    for gid, arch in enumerate(["fast", "slow", "snow"]):
        for period, warm, seed in (("baseline", 0.0, gid), ("future", 3.0, gid + 100)):
            f = _simulate(arch, n, seed, warming=warm)
            f["gauge_id"] = f"{arch}_{gid}"
            f["rcp"] = "rcp85"; f["member"] = "01"; f["period"] = period
            frames.append(f)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jasmin-dir", type=str, default=None,
                    help="Directory of raw HBV wide CSVs (chess_scape_output). "
                         "Triggers the JASMIN loader.")
    ap.add_argument("--gauges", type=str, default=None,
                    help="Comma-separated gauge IDs to subset (smoke test).")
    ap.add_argument("--rcps", type=str, default=None,
                    help="Comma-separated RCPs (default all four).")
    ap.add_argument("--members", type=str, default=None,
                    help="Comma-separated members (default 01,04,06,15).")
    ap.add_argument("--input", type=str, default=None,
                    help="Parquet/CSV already in schema. Omit for self-test.")
    ap.add_argument("--out", type=str, default="slow_full_flow.parquet")
    ap.add_argument("--max-gap", type=int, default=720,
                    help="Upper bound on the transition gap, in days (default "
                         "720 = two water years, the production value; pass 0 "
                         "for an uncensored distribution pass).")
    ap.add_argument("--coherence-min", type=float, default=0.60)
    ap.add_argument("--attribution", type=str, default="runoff-norm",
                    choices=["runoff-norm", "runoff", "legacy", "flow"],
                    help="Terminal-control candidate set: runoff-norm={UZ,LZ} "
                         "range-normalised (default); runoff={UZ,LZ} raw; "
                         "legacy={SM,UZ,LZ} raw; flow=HBV flow-weighted (exact, "
                         "needs --params).")
    ap.add_argument("--validate-fluxes", action="store_true",
                    help="Print max|logged-reconstructed| flux per member "
                         "(one-time provenance check; flow mode only).")
    ap.add_argument("--params", type=str, default=None,
                    help="Calibrated HBV params (csv/parquet: gauge_id,K0,K1,K2,UZL). "
                         "Required for --attribution flow.")
    args = ap.parse_args()

    cfg = Config(max_gap_days=(args.max_gap or None),
                 coherence_min=args.coherence_min,
                 attribution=args.attribution,
                 validate_fluxes=getattr(args, "validate_fluxes", False))
    params_by_gauge = load_params(args.params) if args.params else None

    if args.jasmin_dir:
        gauges = args.gauges.split(",") if args.gauges else None
        rcps = tuple(args.rcps.split(",")) if args.rcps else RCPS
        members = tuple(args.members.split(",")) if args.members else MEMBERS
        print(f"JASMIN loader: dir={args.jasmin_dir}  gauges="
              f"{gauges or 'ALL'}  rcps={rcps}  members={members}\n")
        tr = run_jasmin(args.jasmin_dir, cfg, gauges=gauges,
                        rcps=rcps, members=members,
                        params_by_gauge=params_by_gauge)
        print("Config:", asdict(cfg))
        print(summarise(tr))
        out = Path(args.out)
        tr.to_parquet(out, index=False)
        print(f"\nWrote {len(tr):,} transitions -> {out.resolve()}")
        return

    if args.input:
        p = Path(args.input)
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        print(f"Loaded {len(df):,} rows from {p}")
    else:
        print("No --input given: running self-test on synthetic "
              "fast / slow / snow archetypes.\n")
        df = _make_synthetic()

    selftest = not args.input
    tr = analyse(df, cfg)
    print("Config:", asdict(cfg))
    print(summarise(tr))

    # per-archetype view for the self-test
    if not args.input and not tr.empty:
        print("\n--- per-archetype (coherent FTD, baseline) ---")
        sub = tr[(tr.direction == "FTD") & (tr.passes_coherence)
                 & (tr.period == "baseline")]
        for gid, g in sub.groupby("gauge_id"):
            slow = g[g.regime == "slow"]
            store = g.rate_limiting_store.value_counts().idxmax() if len(g) else "-"
            print(f"  {gid:10s} n={len(g):3d}  slow%={len(slow)/max(len(g),1):.2f}  "
                  f"median_gap={g.gap_days.median():.0f}d  dominant_store={store}")

    out = selftest_io.redirect(args.out, selftest)
    tr.to_parquet(out, index=False)
    print(f"\nWrote {len(tr):,} transitions -> {out.resolve()}")
    if selftest:
        selftest_io.announce([out])


if __name__ == "__main__":
    main()
