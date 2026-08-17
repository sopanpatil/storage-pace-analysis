#!/usr/bin/env python3
"""
regenerate_and_verify_fluxes.py
================================

Re-run the (flux-logging) HBV model over the archived CHESS-SCAPE simulations,
PROVE that the re-run reproduces the existing archive (to its stored precision),
and only then write out the generating-flux series Q0, Q1, Q2 and snowmelt.

A deterministic re-run with the same params and forcing that generated the
archive, but it VERIFIES the regenerated SM/UZ/LZ/SP against the archived state
CSVs before writing, so the logged fluxes provably belong to the run behind the
Results. For a Methods-section provenance claim, that check is the point. This
is a verification/provenance tool in the same family as verify_robustness_checks.py
and verify_manuscript_numbers.py: it does not produce any headline number (the
tracked pipeline reconstructs Q0/Q1/Q2 from the archived states analytically),
it confirms that reconstruction equals a true flux-logging re-run.

Per (rcp, ensemble):
  1. Load forcing (pr, pet, tas) and per-catchment JSON params (pipeline source).
  2. Re-run the flux-logging HBVModel, full continuous series, default initial
     states -- exactly as the archive was generated.
  3. VERIFY regenerated SM, SP, UZ, LZ and routed Q against the archive.
  4. Independently check flux closure (Q0+Q1+Q2 == Qgen) and agreement with the
     Section-2.5 reconstruction applied to the *archived* states.
  5. If verification passes, write {rcp}_{ens}_hbv_{q0,q1,q2,melt}.csv at %.10g.

The archived state CSVs are stored at %.4f (see the batch writer's
float_format), so a correct re-run matches them only to ~5e-5 per value, not to
0. The default --abs-tol 2e-4 confirms "same computation" without tripping on
that storage rounding. A residual materially above ~1e-3 is the real signal that
something (params, forcing, or model version) has shifted.

Usage (JASMIN, against the archived CHESS-SCAPE output)
------------------------------------------------------
  module load jaspy
  # one member:
  python regenerate_and_verify_fluxes.py --archive-dir chess_scape_output \
      --params-root calibrated_params --model-path .  --rcps rcp26 --members 01
  # all 16:
  python regenerate_and_verify_fluxes.py --archive-dir chess_scape_output \
      --params-root calibrated_params --model-path .
  # prove the tool with no archive data (synthetic self-test):
  python regenerate_and_verify_fluxes.py --selftest --model-path .

--model-path is the directory that holds the flux-logging HBVModel, i.e. one
whose states dict includes Q0/Q1/Q2/melt. load_hbv() looks for
`rainfallrunoff/models.py` first (the unpublished multi-model toolkit, if
present) and falls back to `hbv_model/hbv.py`; the published hbv_model already
logs these fluxes, so `--model-path .` resolves against the hbv_model symlink
in this repo.
"""
from __future__ import annotations
import argparse, importlib.util, json, os, sys
import numpy as np
import pandas as pd

# --- archive naming (matches the chess_scape_output layout) ----------------- #
STATE_SUFFIX   = {"SM": "hbv_sm", "SP": "hbv_sp", "UZ": "hbv_uz",
                  "LZ": "hbv_lz", "Q": "hbv_discharge"}
FORCING_SUFFIX = {"pr": "pr_catchment_means_combined",
                  "pet": "pet_catchment_means_combined",
                  "tas": "tas_catchment_means_combined"}
FLUX_SUFFIX    = {"Q0": "hbv_q0", "Q1": "hbv_q1", "Q2": "hbv_q2", "melt": "hbv_melt"}
DATE_COL       = "date"
FLUX_FLOAT_FMT = "%.10g"    # full-precision fluxes (not the archive's %.4f)


def load_hbv(model_path):
    """Load the flux-logging HBVModel from rainfallrunoff/models.py (preferred)
    or hbv_model/hbv.py, whichever exists under model_path."""
    for rel, modname in (("rainfallrunoff/models.py", "rr_models_fluxlog"),
                         ("hbv_model/hbv.py", "hbv_fluxlog")):
        cand = os.path.join(model_path, rel)
        if os.path.isfile(cand):
            spec = importlib.util.spec_from_file_location(modname, cand)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[modname] = mod            # needed for numba cached @njit
            spec.loader.exec_module(mod)
            HBV = mod.HBVModel
            probe = HBV({"TT":0.,"CFMAX":3.,"CFR":.05,"CWH":.1,"FC":250.,"LP":.7,
                         "BETA":2.,"K0":.3,"K1":.1,"K2":.02,"UZL":20.,"PERC":2.,
                         "MAXBAS":3.})
            _, st = probe.run(np.zeros(5), np.zeros(5), np.zeros(5))
            missing = {"Q0","Q1","Q2","melt"} - set(st)
            if missing:
                sys.exit(f"{cand} is not the flux-logging model "
                         f"(states dict missing {missing}); this tool needs an "
                         f"HBVModel that returns Q0/Q1/Q2/melt in its states dict.")
            return HBV
    sys.exit(f"No rainfallrunoff/models.py or hbv_model/hbv.py under {model_path}")


def reconstruct(UZ_s, LZ_s, K0, K1, K2, UZL):
    Q2 = (K2 / (1.0 - K2)) * LZ_s
    UZ_b = UZ_s / (1.0 - K1)
    Q1 = K1 * UZ_b
    Q0 = np.where(UZ_b > UZL, K0 * (UZ_b - UZL) / (1.0 - K0), 0.0)
    return Q0, Q1, Q2


def load_params_json(params_root, cid):
    f = os.path.join(params_root, "hbv", f"{cid}_hbv_params.json")
    if not os.path.isfile(f):
        return None
    with open(f) as fh:
        return json.load(fh)["params"]


def apath(d, rcp, m, suf):  return os.path.join(d, f"{rcp}_{m}_{suf}.csv")
def read_wide(p):           return pd.read_csv(p, dtype={DATE_COL: str}).set_index(DATE_COL)


def process_member(HBVModel, archive_dir, out_dir, rcp, member, params_root,
                   abs_tol, write_fluxes, verbose, params_df=None):
    pr  = read_wide(apath(archive_dir, rcp, member, FORCING_SUFFIX["pr"]))
    pet = read_wide(apath(archive_dir, rcp, member, FORCING_SUFFIX["pet"]))
    tas = read_wide(apath(archive_dir, rcp, member, FORCING_SUFFIX["tas"]))
    arch = {}
    for v, s in STATE_SUFFIX.items():
        p = apath(archive_dir, rcp, member, s)
        arch[v] = read_wide(p) if os.path.isfile(p) else None
    index = pr.index
    gauges = [c for c in pr.columns if c != DATE_COL]
    flux_out = {k: pd.DataFrame(index=index, dtype=np.float64) for k in FLUX_SUFFIX}
    worst = {v: (0.0, 0.0, None) for v in STATE_SUFFIX}
    max_closure = max_recon = 0.0
    n_done = n_noparam = 0

    for g in gauges:
        p = load_params_json(params_root, g) if params_root else None
        if p is None and params_df is not None and g in params_df.index:
            p = {k: float(params_df.at[g, k]) for k in HBVModel.PARAM_NAMES}
        if p is None:
            n_noparam += 1
            for k in FLUX_SUFFIX: flux_out[k][g] = np.nan
            continue
        p = {k: float(p[k]) for k in HBVModel.PARAM_NAMES}
        P, T, E = (pr[g].to_numpy(float), tas[g].to_numpy(float), pet[g].to_numpy(float))
        Qr, st = HBVModel(p).run(P, T, E)

        for v in ("SM", "SP", "UZ", "LZ"):
            if arch[v] is None: continue
            a = arch[v][g].to_numpy(float); d = np.abs(st[v] - a)
            ae = float(d.max()); re = float((d/np.maximum(np.abs(a),1e-9)).max())
            if ae > worst[v][0]: worst[v] = (ae, re, g)
        if arch["Q"] is not None:
            a = arch["Q"][g].to_numpy(float); d = np.abs(Qr - a)
            ae = float(d.max()); re = float((d/np.maximum(np.abs(a),1e-9)).max())
            if ae > worst["Q"][0]: worst["Q"] = (ae, re, g)

        max_closure = max(max_closure,
                          float(np.max(np.abs(st["Q0"]+st["Q1"]+st["Q2"]-st["Qgen"]))))
        if arch["UZ"] is not None and arch["LZ"] is not None:
            Q0r, Q1r, Q2r = reconstruct(arch["UZ"][g].to_numpy(float),
                                        arch["LZ"][g].to_numpy(float),
                                        p["K0"], p["K1"], p["K2"], p["UZL"])
            max_recon = max(max_recon,
                            float(np.max(np.abs(st["Q0"]-Q0r))),
                            float(np.max(np.abs(st["Q1"]-Q1r))),
                            float(np.max(np.abs(st["Q2"]-Q2r))))
        flux_out["Q0"][g]=st["Q0"]; flux_out["Q1"][g]=st["Q1"]
        flux_out["Q2"][g]=st["Q2"]; flux_out["melt"][g]=st["melt"]
        n_done += 1

    worst_abs = max(w[0] for w in worst.values())
    # recon is limited by the archive's %.4f states, so allow ~1e-3 there
    passed = worst_abs <= abs_tol and max_closure < 1e-9 and max_recon <= 1e-3
    if verbose:
        print(f"\n[{rcp}_{member}]  gauges verified: {n_done}"
              f"{f'  (no params: {n_noparam})' if n_noparam else ''}")
        for v in ("SM","SP","UZ","LZ","Q"):
            ae,re,g = worst[v]; print(f"    {v:2s}  max|abs|={ae:.3e}  max|rel|={re:.3e}  (worst {g})")
        print(f"    flux closure |Q0+Q1+Q2-Qgen| = {max_closure:.3e}")
        print(f"    recon agree  |logged-recon|  = {max_recon:.3e}  (bounded by %.4f states)")
        print(f"    VERDICT: {'PASS' if passed else 'FAIL'}")
    if write_fluxes and passed:
        os.makedirs(out_dir, exist_ok=True)
        for k, suf in FLUX_SUFFIX.items():
            out = flux_out[k].copy(); out.insert(0, DATE_COL, index)
            out.to_csv(apath(out_dir, rcp, member, suf), index=False,
                       float_format=FLUX_FLOAT_FMT)
        if verbose: print(f"    wrote {rcp}_{member}_hbv_{{q0,q1,q2,melt}}.csv -> {out_dir}")
    elif write_fluxes and verbose:
        print(f"    NOT writing fluxes for {rcp}_{member}: verification failed.")
    return passed, worst_abs, max_recon, max_closure


def selftest(model_path):
    print("SELFTEST: fabricate a %.4f archive from the model, then verify + write.\n")
    HBV = load_hbv(model_path)
    rng = np.random.default_rng(7)
    tmp = os.path.join(model_path, "_selftest_archive"); os.makedirs(tmp, exist_ok=True)
    pr_root = os.path.join(tmp, "params"); os.makedirs(os.path.join(pr_root,"hbv"), exist_ok=True)
    rcp, member, n = "rcp85", "01", 4000
    gauges = ["1001","2001","3001","45009"]; dates=[f"D{i:05d}" for i in range(n)]
    def wide(cd): df=pd.DataFrame(cd); df.insert(0,DATE_COL,dates); return df
    cols={v:{} for v in ("pr","pet","tas","SM","SP","UZ","LZ","Q")}
    b=HBV.PARAM_BOUNDS
    for g in gauges:
        P=np.clip(rng.gamma(.6,4.,n)-.5,0,None)
        T=8+9*np.sin(np.arange(n)*2*np.pi/360)+rng.normal(0,3,n)
        E=np.clip(2.5+2*np.sin(np.arange(n)*2*np.pi/360)+rng.normal(0,.4,n),0,None)
        p={k:float(rng.uniform(lo,hi)) for k,(lo,hi) in b.items()}
        Qr,st=HBV(p).run(P,T,E)
        cols["pr"][g],cols["pet"][g],cols["tas"][g]=P,E,T
        cols["SM"][g],cols["SP"][g],cols["UZ"][g],cols["LZ"][g],cols["Q"][g]=(
            st["SM"],st["SP"],st["UZ"],st["LZ"],Qr)
        json.dump({"params":p}, open(os.path.join(pr_root,"hbv",f"{g}_hbv_params.json"),"w"))
    wide(cols["pr"]).to_csv(apath(tmp,rcp,member,FORCING_SUFFIX["pr"]),index=False)
    wide(cols["pet"]).to_csv(apath(tmp,rcp,member,FORCING_SUFFIX["pet"]),index=False)
    wide(cols["tas"]).to_csv(apath(tmp,rcp,member,FORCING_SUFFIX["tas"]),index=False)
    for v in ("SM","SP","UZ","LZ","Q"):   # write archive at %.4f, like the real batch
        wide(cols[v]).to_csv(apath(tmp,rcp,member,STATE_SUFFIX[v]),index=False,float_format="%.4f")
    passed,wabs,wrec,wclo=process_member(HBV,tmp,os.path.join(tmp,"flux_out"),
        rcp,member,pr_root,abs_tol=2e-4,write_fluxes=True,verbose=True)
    print("\nSELFTEST RESULT:", "PASS" if passed else "FAIL")
    print(f"  state/Q vs %.4f archive : {wabs:.3e}  (storage rounding, not model drift)")
    print(f"  flux closure            : {wclo:.3e}  (exact)")
    print(f"  recon vs %.4f states    : {wrec:.3e}")
    return 0 if passed else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive-dir"); ap.add_argument("--out-dir")
    ap.add_argument("--params-root", help="dir with hbv/<cid>_hbv_params.json (pipeline source)")
    ap.add_argument("--params-csv", help="fallback: calibrated_parameters.csv")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--rcps", nargs="+", default=["rcp26","rcp45","rcp60","rcp85"])
    ap.add_argument("--members", nargs="+", default=["01","04","06","15"])
    ap.add_argument("--abs-tol", type=float, default=2e-4)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: sys.exit(selftest(a.model_path))
    if not (a.archive_dir and (a.params_root or a.params_csv)):
        ap.error("--archive-dir and (--params-root or --params-csv) required unless --selftest")
    HBV = load_hbv(a.model_path)
    pdf = None
    if a.params_csv:
        pdf = pd.read_csv(a.params_csv); pdf["gauge_id"]=pdf["gauge_id"].astype(str)
        pdf = pdf.set_index("gauge_id")
    out_dir = a.out_dir or a.archive_dir
    all_pass=True; summ=[]
    for rcp in a.rcps:
        for m in a.members:
            if not os.path.isfile(apath(a.archive_dir,rcp,m,FORCING_SUFFIX["pr"])):
                print(f"[skip] no forcing for {rcp}_{m}"); continue
            ok,wabs,_,_ = process_member(HBV,a.archive_dir,out_dir,rcp,m,a.params_root,
                a.abs_tol,write_fluxes=not a.no_write,verbose=True,params_df=pdf)
            all_pass&=ok; summ.append((f"{rcp}_{m}",ok,wabs))
    print("\n"+"="*60+"\nSUMMARY")
    for name,ok,wabs in summ: print(f"  {name:10s} {'PASS' if ok else 'FAIL':4s} max|abs|={wabs:.3e}")
    print("="*60)
    if not all_pass:
        print("\nAt least one member failed verification. The logged fluxes cannot be "
              "claimed to belong to the archived run -- inspect the failing member's "
              "params/forcing before proceeding.")
        sys.exit(1)
    print("\nAll members verified against the archive. The logged Q0/Q1/Q2/melt belong "
          "to the run behind the Results.")
    sys.exit(0)


if __name__ == "__main__":
    main()
