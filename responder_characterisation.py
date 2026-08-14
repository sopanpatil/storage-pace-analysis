"""
responder_characterisation.py

Why do a minority of high-storage catchments intensify their SLOW flood-to-drought
transitions under warming while most do not?

Compares RESPONDERS (large projected increase in slow-FTD frequency) against
NON-RESPONDERS, restricted to catchments that carry a baseline slow tail (so we
control for "has a slow tail at all" -- i.e. BFI -- and isolate "why does warming
intensify it here"). Tests static CAMELS-GB attributes and calibrated HBV
parameters as discriminators.

Inputs
------
  --deltas   projection_flow.parquet   (from projection_analysis.py:
             columns gauge_id, rcp, direction, d_freq_decade, base_freq_decade)
  --attr-dir directory holding the four camels_gb_v2_*_attributes.csv files
  --params   calibrated_parameters.csv (HBV params + calibration_kge)

Method
------
  1. Merge deltas (chosen direction/RCP) with all attributes + HBV params.
  2. Groups: carriers = base_freq_decade > 0; responder = d_freq_decade >= thresh;
     non_responder = carrier with d_freq_decade <= 0.
  3. Univariate Mann-Whitney U per attribute, rank-biserial effect size,
     Benjamini-Hochberg FDR. Ranked by |effect|.
  4. Parsimonious multivariate logistic on the top non-redundant separators
     (standardised), odds ratios -- indicative only given small n.

Run with no args for a synthetic self-test (planted aridity+K2 signal).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import selftest_io

try:
    from scipy.stats import mannwhitneyu
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

ATTR_FILES = {
    "clim": "camels_gb_v2_climatic_attributes.csv",
    "topo": "camels_gb_v2_topographic_attributes.csv",
    "hgeo": "camels_gb_v2_hydrogeology_attributes.csv",
    "hydro": "camels_gb_v2_hydrologic_attributes.csv",
}
# Candidate discriminators (present across the four tables + params)
CANDIDATES = [
    "baseflow_index", "runoff_ratio", "slope_fdc", "stream_elas", "Q95",
    "low_q_dur", "high_q_dur", "frac_high_perc", "inter_high_perc",
    "aridity", "frac_snow", "p_seasonality", "p_mean", "pet_mean",
    "high_prec_freq", "area", "elev_mean", "dpsbar",
    "K0", "K1", "K2", "UZL", "PERC", "FC", "BETA", "LP",
]


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def load_merged(deltas: str, attr_dir: str, params: str,
                direction: str, rcp: str, forcing: str | None = None) -> pd.DataFrame:
    d = pd.read_parquet(deltas)
    d = d[(d["direction"] == direction) & (d["rcp"] == rcp)].copy()
    d["gauge_id"] = d["gauge_id"].astype(str)
    d = d[["gauge_id", "d_freq_decade", "base_freq_decade"]]

    ad = Path(attr_dir)
    for key, fn in ATTR_FILES.items():
        a = pd.read_csv(ad / fn)
        a["gauge_id"] = a["gauge_id"].astype(str)
        keep = ["gauge_id"] + [c for c in a.columns
                               if c in CANDIDATES and c not in d.columns]
        d = d.merge(a[keep], on="gauge_id", how="left")

    p = pd.read_csv(params)
    p["gauge_id"] = p["gauge_id"].astype(str)
    pk = ["gauge_id"] + [c for c in p.columns
                         if c in CANDIDATES and c not in d.columns]
    d = d.merge(p[pk], on="gauge_id", how="left")

    if forcing:
        fp = Path(forcing)
        fdf = pd.read_parquet(fp) if fp.suffix == ".parquet" else pd.read_csv(fp)
        fdf["gauge_id"] = fdf["gauge_id"].astype(str)
        fcols = ["gauge_id"] + [c for c in fdf.columns
                                if c.startswith("d_") and c not in d.columns]
        d = d.merge(fdf[fcols], on="gauge_id", how="left")
    return d


def define_groups(d: pd.DataFrame, thresh: float) -> pd.DataFrame:
    carrier = d["base_freq_decade"] > 0
    d = d[carrier].copy()
    d["group"] = np.where(d["d_freq_decade"] >= thresh, "responder",
                          np.where(d["d_freq_decade"] <= 0, "non_responder", "middle"))
    return d


def univariate(d: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    R = d[d["group"] == "responder"]
    N = d[d["group"] == "non_responder"]
    rows = []
    for c in predictors:
        r, n = R[c].dropna(), N[c].dropna()
        if len(r) < 3 or len(n) < 3:
            continue
        rec = {"attribute": c, "median_resp": round(float(r.median()), 4),
               "median_nonresp": round(float(n.median()), 4)}
        if _HAVE_SCIPY:
            U, p = mannwhitneyu(r, n, alternative="two-sided")
            # rank-biserial effect size. mannwhitneyu(r, n) returns U for the
            # FIRST argument (r = responders), so positive effect_r here means
            # "responders tend to have larger values than non-responders",
            # matching the convention stated in the docstring / Text S3.
            rec["effect_r"] = round(2 * U / (len(r) * len(n)) - 1, 3)  # rank-biserial
            rec["p"] = p
        rows.append(rec)
    out = pd.DataFrame(rows)
    if not out.empty and "p" in out:
        out["fdr"] = _bh_fdr(out["p"].to_numpy())
        out["p"] = out["p"].round(4); out["fdr"] = out["fdr"].round(4)
        out["abs_effect"] = out["effect_r"].abs()
        out = out.sort_values("abs_effect", ascending=False).drop(columns="abs_effect")
    return out


def multivariate(d: pd.DataFrame, top: list[str]) -> str:
    sub = d[d["group"].isin(["responder", "non_responder"])].copy()
    sub = sub.dropna(subset=top)
    y = (sub["group"] == "responder").astype(int).to_numpy()
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        return "  (too few in a group for a stable multivariate fit)"
    X = sub[top].to_numpy(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    try:
        import statsmodels.api as sm
        Xc = sm.add_constant(X)
        res = sm.Logit(y, Xc).fit(disp=0, maxiter=200)
        lines = ["  standardised logistic (odds ratio per +1 SD):"]
        for name, coef, p in zip(["const"] + top, res.params, res.pvalues):
            if name == "const":
                continue
            lines.append(f"    {name:16s} OR={np.exp(coef):5.2f}  p={p:.3f}")
        lines.append(f"    pseudo-R2={res.prsquared:.2f}, n={len(y)}")
        return "\n".join(lines)
    except Exception:
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            return ("  (multivariate step skipped: neither statsmodels nor "
                    "scikit-learn is installed; both are in requirements.txt)")
        m = LogisticRegression(max_iter=500).fit(X, y)
        lines = ["  standardised logistic coefficients (sklearn):"]
        for name, coef in zip(top, m.coef_[0]):
            lines.append(f"    {name:16s} beta={coef:+.2f}")
        return "\n".join(lines)


def report(d: pd.DataFrame, thresh: float) -> pd.DataFrame:
    g = define_groups(d, thresh)
    nR = (g["group"] == "responder").sum()
    nN = (g["group"] == "non_responder").sum()
    nM = (g["group"] == "middle").sum()
    print(f"Groups (carriers only): responder={nR}, non_responder={nN}, "
          f"middle={nM}  (responder = d_freq_decade >= {thresh})")
    predictors = [c for c in CANDIDATES if c in g.columns]
    predictors += [c for c in g.columns          # forcing deltas, if merged
                   if c.startswith("d_") and c != "d_freq_decade"
                   and c not in predictors]
    uni = univariate(g, predictors)
    print("\nUnivariate discriminators (responder vs non_responder, "
          "ranked by |rank-biserial effect|):")
    print(uni.to_string(index=False))

    if not uni.empty and "effect_r" in uni:
        # top non-redundant predictors: take strongest few, drop obvious dupes
        top = [c for c in uni["attribute"].tolist()][:6]
        print("\nMultivariate (indicative, small-n):")
        print(multivariate(g, top))
    return g


# --------------------------------------------------------------------------- #
# Synthetic self-test                                                         #
# --------------------------------------------------------------------------- #
def _make_synthetic(tmp: Path) -> tuple[str, str, str]:
    rng = np.random.default_rng(0)
    n = 200
    gid = [str(1000 + i) for i in range(n)]
    bfi = rng.uniform(0.3, 0.9, n)
    carrier = bfi > 0.55
    aridity = rng.uniform(0.4, 1.3, n)
    K2 = rng.uniform(0.005, 0.12, n)
    frac_snow = rng.uniform(0, 0.2, n)
    # planted: responders are carriers with high aridity AND low K2
    score = (aridity - 0.8) * 2 - (K2 - 0.05) * 20 + rng.normal(0, 0.5, n)
    d_freq = np.where(carrier & (score > 0.8), rng.uniform(0.5, 2.5, n),
                      np.where(carrier, rng.uniform(-0.2, 0.2, n), 0.0))
    base_freq = np.where(carrier, rng.uniform(0.2, 3, n), 0.0)

    deltas = pd.DataFrame({"gauge_id": gid, "rcp": "rcp85", "direction": "FTD",
                           "d_freq_decade": d_freq, "base_freq_decade": base_freq})
    deltas.to_parquet(tmp / "d.parquet", index=False)

    clim = pd.DataFrame({"gauge_id": gid, "aridity": aridity,
                         "frac_snow": frac_snow, "p_seasonality": rng.uniform(-0.5, 0.5, n),
                         "p_mean": rng.uniform(1, 6, n), "pet_mean": rng.uniform(1, 3, n),
                         "high_prec_freq": rng.uniform(5, 30, n),
                         "low_prec_freq": rng.uniform(100, 300, n),
                         "high_prec_dur": rng.uniform(1, 3, n),
                         "low_prec_dur": rng.uniform(3, 10, n)})
    topo = pd.DataFrame({"gauge_id": gid, "area": rng.uniform(10, 2000, n),
                         "elev_mean": rng.uniform(50, 800, n), "dpsbar": rng.uniform(20, 200, n),
                         "gauge_lat": rng.uniform(50, 58, n)})
    hgeo = pd.DataFrame({"gauge_id": gid, "frac_high_perc": bfi * 0.8 + rng.uniform(0, .2, n),
                         "inter_high_perc": rng.uniform(0, 1, n)})
    hydro = pd.DataFrame({"gauge_id": gid, "baseflow_index": bfi,
                          "runoff_ratio": rng.uniform(0.2, 0.9, n),
                          "slope_fdc": rng.uniform(1, 5, n), "stream_elas": rng.uniform(0.5, 3, n),
                          "Q95": rng.uniform(0.01, 1, n), "low_q_dur": rng.uniform(3, 30, n),
                          "high_q_dur": rng.uniform(1, 4, n)})
    for name, df in [("climatic", clim), ("topographic", topo),
                     ("hydrogeology", hgeo), ("hydrologic", hydro)]:
        df.to_csv(tmp / f"camels_gb_v2_{name}_attributes.csv", index=False)

    params = pd.DataFrame({"gauge_id": gid, "K0": rng.uniform(.05, .5, n),
                           "K1": rng.uniform(.01, .3, n), "K2": K2,
                           "UZL": rng.uniform(0, 100, n), "PERC": rng.uniform(0, 6, n),
                           "FC": rng.uniform(50, 500, n), "BETA": rng.uniform(1, 6, n),
                           "LP": rng.uniform(.3, 1, n),
                           "used_in_analysis": True})
    params.to_csv(tmp / "params.csv", index=False)
    return str(tmp / "d.parquet"), str(tmp), str(tmp / "params.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deltas", type=str, default=None)
    ap.add_argument("--attr-dir", type=str, default=None)
    ap.add_argument("--params", type=str, default=None)
    ap.add_argument("--forcing", type=str, default=None,
                    help="forcing_deltas output (csv/parquet) to add Delta-forcing "
                         "variables as candidate discriminators.")
    ap.add_argument("--direction", type=str, default="FTD")
    ap.add_argument("--rcp", type=str, default="rcp85")
    ap.add_argument("--responder-thresh", type=float, default=0.5)
    ap.add_argument("--out", type=str, default="responder_table.parquet")
    args = ap.parse_args()

    if args.deltas:
        d = load_merged(args.deltas, args.attr_dir, args.params,
                        args.direction, args.rcp, forcing=args.forcing)
        print(f"Merged {len(d)} catchments for {args.direction} {args.rcp}")
    else:
        print("Self-test: planted responders = high-aridity + low-K2 carriers.\n")
        tmp = selftest_io.redirect("syn_resp", True); tmp.mkdir(exist_ok=True)
        dp, adir, pp = _make_synthetic(tmp)
        d = load_merged(dp, adir, pp, "FTD", "rcp85")

    g = report(d, args.responder_thresh)
    out = selftest_io.redirect(args.out, selftest=not args.deltas)
    g.to_parquet(out, index=False)
    print(f"\nWrote responder table -> {out.resolve()}")
    if not args.deltas:
        selftest_io.announce([out])


if __name__ == "__main__":
    main()
