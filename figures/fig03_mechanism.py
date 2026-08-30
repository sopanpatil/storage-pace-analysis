#!/usr/bin/env python
"""
fig03_mechanism.py
==================
Figure 3 -- slowness has one mechanism; abruptness has two
(manuscript Section 3.2).

Message
-------
Decomposing each FTD transition into fast flow (Q0+Q1, from the upper zone UZ)
and baseflow (Q2, from the lower zone LZ) and classifying it by the store that
paces its terminal approach to the drought threshold shows a systematic change
of mechanism along the pace continuum. Slow FTD transitions are 99.5 %
lower-zone (baseflow) limited; abrupt FTD transitions split 68.2 % fast-flow /
31.8 % baseflow. Slowness admits essentially one route (baseflow depletion);
abruptness admits two (fast-flow recession OR baseflow collapse from a depleted
lower zone). The figure carries this twofold content: the quantification and
the asymmetry.

Manuscript values to reproduce
    Slow FTD   : 99.5 % LZ-limited (0.5 % UZ)
    Abrupt FTD : 68.2 % UZ-limited, 31.8 % LZ-limited

Panels
    (a) Fraction of FTD transitions that are baseflow (LZ) limited as a function
        of transition gap, showing convergence on baseflow as gaps lengthen;
        the 90-day convention is marked for reference only.
    (b) The asymmetry as two stacked bars -- abrupt vs slow -- partitioned into
        fast-flow (UZ) and baseflow (LZ) limited shares.
    (c) OPTIONAL spatial panel: analysed catchments coloured by baseflow index,
        with baseline slow-tail carriers outlined, drawn over a GB coastline
        (--basemap) if one is supplied. Rendered only if BOTH a
        catchment-geometry file (--geom) and the CAMELS-GB hydrologic attribute
        table (--attr-dir) are supplied; otherwise the figure is the (a)+(b)
        core, which is already a complete, submittable figure. The coastline
        is optional independently of that -- panel (c) still renders without
        it, just without the geographic context. The mosaic is restricted to
        the catchments retained by the KGE >= 0.5 screen (--params, the
        used_in_analysis flag in calibrated_parameters.csv) so that it shows
        the 621 analysed catchments rather than all 671 in CAMELS-GB, and the
        outlines are drawn from baseline-period slow transitions only, both
        matching what the manuscript caption states.

Input
    slow_full_flow.parquet  (per-transition table)
        columns: gap_days, direction, regime, rate_limiting_store,
                 passes_coherence, gauge_id, period
    (optional) catchment geometry (GeoPackage/shapefile, or a .zip archive of
        shapefile components -- e.g. camels_gb_v2_catchment_boundaries.zip as
        delivered on JASMIN, read directly without unzipping) keyed by gauge id
    (optional) CAMELS-GB v2 hydrologic attribute table (baseflow_index)
    (optional) calibrated_parameters.csv (used_in_analysis) -- the KGE screen
    (optional) GB coastline outline (gb_outline_27700.geojson) -- the Office
        for National Statistics Open Geography Portal "Countries" boundary
        dataset, England/Scotland/Wales unioned, Northern Ireland dropped,
        reprojected to EPSG:27700, redistributed under the Open Government
        Licence v3.0. Built once (not fetched at runtime, since JASMIN has no
        general internet access) and bundled alongside this script.

Usage
    python fig03_mechanism.py --input slow_full_flow.parquet --outdir figures
    python fig03_mechanism.py --input slow_full_flow.parquet \
        --geom camels_gb_v2_catchment_boundaries.zip --geom-id ID_STRING \
        --attr-dir camels_gb_v2_hydrologic_attributes.csv \
        --basemap gb_outline_27700.geojson \
        --params calibrated_parameters.csv --outdir figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import agu_style as S

CUTOFF = 90


def load(parquet: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet)
    need = {"gap_days", "direction", "regime", "rate_limiting_store",
            "passes_coherence", "period"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"{parquet} is missing columns: {sorted(missing)}")
    df = df[(df["direction"] == "FTD") & (df["passes_coherence"])].copy()
    # keep only transitions with an identified runoff-generating store
    df = df[df["rate_limiting_store"].isin(["UZ", "LZ"])].copy()
    df["gap_days"] = df["gap_days"].astype(float)
    df["is_lz"] = (df["rate_limiting_store"] == "LZ")
    return df


def summarise(df: pd.DataFrame) -> None:
    for regime in ("abrupt", "slow"):
        g = df[df["regime"] == regime]
        if len(g):
            lz = g["is_lz"].mean() * 100
            print(f"  {regime:6s} FTD : LZ-limited {lz:5.1f}% | "
                  f"UZ-limited {100-lz:5.1f}%  (n={len(g):,})")


def make_figure(df: pd.DataFrame, outdir: str,
                geom: str | None, geom_id: str, attr_dir: str | None,
                basemap: str | None = None, params: str | None = None) -> None:

    have_map = geom is not None and attr_dir is not None
    if have_map:
        try:
            import geopandas  # noqa: F401
        except ImportError:
            print("  [panel c] geopandas not available; rendering (a)+(b) core.")
            have_map = False

    ncol = 3 if have_map else 2
    widths = [1.0, 0.7, 1.05] if have_map else [1.0, 0.75]
    fig, axes = plt.subplots(
        1, ncol, figsize=(S.W_2COL if have_map else S.W_1P5COL + 20 * S.MM,
                          64 * S.MM),
        gridspec_kw=dict(width_ratios=widths, wspace=0.42),
    )
    axA, axB = axes[0], axes[1]

    # ---- Panel (a): P(LZ-limited) vs gap ----------------------------------
    edges = np.array([0, 10, 20, 30, 45, 60, 90, 130, 180, 260, 360, 720],
                     dtype=float)
    cent = 0.5 * (edges[:-1] + edges[1:])
    frac, nper = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (df["gap_days"] >= lo) & (df["gap_days"] < hi)
        nper.append(int(m.sum()))
        frac.append(df.loc[m, "is_lz"].mean() if m.any() else np.nan)
    frac = np.array(frac)
    axA.plot(cent, frac * 100, color=S.C_SLOW, lw=1.3, marker="o", ms=3.2,
             mec="white", mew=0.4)
    axA.axvline(CUTOFF, color=S.OKABE_ITO["black"], lw=0.8, ls=(0, (4, 2)))
    axA.text(CUTOFF + 6, 8, "abrupt | slow", fontsize=7, rotation=0, va="bottom")
    axA.set_xscale("log")
    axA.set_xlim(edges[0] + 3, edges[-1])
    axA.set_ylim(0, 103)
    axA.set_xlabel("FTD transition gap (days)")
    axA.set_ylabel("Baseflow (LZ)-limited (%)")
    S.panel_label(axA, "a")

    # ---- Panel (b): asymmetry as stacked bars -----------------------------
    order = ["abrupt", "slow"]
    lz_share = np.array([df.loc[df.regime == r, "is_lz"].mean() for r in order]) * 100
    uz_share = 100 - lz_share
    x = np.arange(len(order))
    axB.bar(x, uz_share, width=0.62, color=S.C_FAST, label="Fast flow (UZ)")
    axB.bar(x, lz_share, width=0.62, bottom=uz_share, color=S.C_SLOW,
            label="Baseflow (LZ)")
    for xi, (u, l) in enumerate(zip(uz_share, lz_share)):
        # fast-flow (UZ) share -- bottom segment
        if u >= 4:
            axB.text(xi, u / 2, f"{u:.1f}", ha="center", va="center",
                     color="white", fontsize=7)
        elif u > 0:  # thin sliver: label just outside the bar
            axB.text(xi + 0.34, u / 2, f"{u:.1f}", ha="left", va="center",
                     color=S.C_FAST, fontsize=7)
        # baseflow (LZ) share -- top segment
        axB.text(xi, u + l / 2, f"{l:.1f}", ha="center", va="center",
                 color="white", fontsize=7)
    axB.set_xticks(x)
    axB.set_xticklabels(["Abrupt\n(\u2264 90 d)", "Slow\n(> 90 d)"])
    axB.set_ylim(0, 100)
    axB.set_ylabel("Share of FTD transitions (%)")
    axB.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1)
    S.panel_label(axB, "b")

    # ---- Panel (c): optional spatial map ----------------------------------
    if have_map:
        import geopandas as gpd
        axC = axes[2]
        gdf = _read_geom(geom)
        if geom_id not in gdf.columns:
            raise KeyError(f"'{geom_id}' not in geometry columns: {list(gdf.columns)}")
        bfi = _read_bfi(attr_dir)
        # gauge ids arrive as int in some files and str in others (e.g. CSV vs
        # parquet); coerce both sides to a plain digit string so the merge
        # doesn't silently drop everything on a dtype mismatch.
        gdf = gdf.copy()
        gdf["_gid"] = gdf[geom_id].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        bfi["_gid"] = bfi["gauge_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        merged = gdf.merge(bfi[["_gid", "baseflow_index"]], on="_gid", how="inner")
        # The geometry and attribute tables both cover all 671 CAMELS-GB
        # catchments; the analysis uses the 621 that pass the KGE >= 0.5
        # screen. Without this the mosaic would silently include the 50
        # excluded catchments (which the caption does not claim to show).
        retained = _read_retained(params)
        if retained is not None:
            merged = merged[merged["_gid"].isin(retained)].copy()
        if merged.empty:
            raise ValueError(
                "geometry/attribute merge produced zero rows -- check that "
                f"--geom-id ('{geom_id}') and the attribute table's gauge_id "
                "use the same catchment identifiers.")
        print(f"  [panel c] {len(merged)}/{len(gdf)} geometries retained and "
              f"matched to a baseflow index (source: {attr_dir})")

        # GB coastline, drawn first so the catchment mosaic sits on top of it.
        # CRS is reprojected to match the catchment geometry's CRS in case the
        # basemap file (e.g. gb_outline_27700.geojson) differs.
        basemap_gdf = None
        if basemap is not None:
            basemap_gdf = gpd.read_file(basemap)
            if gdf.crs is not None and basemap_gdf.crs is not None \
                    and basemap_gdf.crs != gdf.crs:
                basemap_gdf = basemap_gdf.to_crs(gdf.crs)
            basemap_gdf.plot(ax=axC, facecolor="0.94", edgecolor="0.4",
                            linewidth=0.5, zorder=0)

        # carriers: catchments with any baseline slow FTD transition. The
        # period filter is deliberate and applies to this panel only -- (a)
        # and (b) pool both periods, but the outlines are described in the
        # manuscript as the *baseline* slow tail.
        slow = df[(df["regime"] == "slow") & (df["period"] == "baseline")]
        carriers = set(slow["gauge_id"].astype(str)
                      .str.strip().str.replace(r"\.0$", "", regex=True).unique())
        # geopandas builds the colorbar internally, so pick its axes out of
        # the figure afterwards -- it has to travel with axC when the panel
        # gaps are equalised below.
        pre_axes = set(fig.axes)
        merged.plot(ax=axC, column="baseflow_index", cmap="viridis",
                   vmin=0, vmax=1, linewidth=0.15, edgecolor="0.6", zorder=1,
                   legend=True,
                   legend_kwds=dict(label="Baseflow index", shrink=0.55,
                                    fraction=0.045, pad=0.02))
        cbar_axes = [a for a in fig.axes if a not in pre_axes]
        if carriers:
            hit = merged[merged["_gid"].isin(carriers)]
            if len(hit):
                hit.boundary.plot(ax=axC, color=S.OKABE_ITO["vermil"],
                                 linewidth=0.5, zorder=2)
                bfi_hit = hit["baseflow_index"]
                bfi_rest = merged.loc[~merged["_gid"].isin(carriers),
                                      "baseflow_index"]
                print(f"  [panel c] baseline slow-tail carriers outlined: "
                      f"{len(hit)}; median baseflow index "
                      f"{bfi_hit.median():.2f} vs {bfi_rest.median():.2f} "
                      f"for the remaining {len(bfi_rest)}")
        axC.set_axis_off()
        axC.set_aspect("equal")
        # extent from the coastline (if present) rather than just the matched
        # subset, so a partial attribute match doesn't crop the map to a
        # scattered handful of catchments -- GB stays whole either way.
        extent_src = basemap_gdf if basemap_gdf is not None else gdf
        xmin, ymin, xmax, ymax = extent_src.total_bounds
        pad = 0.02 * max(xmax - xmin, ymax - ymin)
        axC.set_xlim(xmin - pad, xmax + pad)
        axC.set_ylim(ymin - pad, ymax + pad)
        S.panel_label(axC, "c", x=0.02, y=0.99)

    # (a) and (b) carry y-axis labels and tick labels that overflow their
    # gridspec cells, while (c) is an equal-aspect map that under-fills its
    # own; with a plain wspace that renders as (a)-(b) nearly touching and a
    # wide gap before (c). Space the panels by their drawn extents instead.
    groups = [[axA], [axB]] + ([[axC] + cbar_axes] if have_map else [])
    S.equalise_panel_gaps(fig, groups)

    S.save(fig, Path(outdir) / "fig03_mechanism")
    plt.close(fig)


def _read_geom(path: str):
    """Read the catchment-geometry file. Transparently handles a .zip archive
    of shapefile components (as delivered from JASMIN, e.g.
    camels_gb_v2_catchment_boundaries.zip) via GDAL's virtual zip filesystem,
    so --geom can point straight at the .zip without unzipping it first."""
    import geopandas as gpd
    p = Path(path)
    if p.suffix.lower() != ".zip":
        return gpd.read_file(path)
    try:
        return gpd.read_file(f"zip://{p}")
    except Exception as e:
        # shapefile components might sit inside a subfolder within the
        # archive; locate the .shp member explicitly and point GDAL at it.
        import zipfile
        with zipfile.ZipFile(p) as zf:
            shp_members = [n for n in zf.namelist() if n.lower().endswith(".shp")]
        if not shp_members:
            raise FileNotFoundError(f"no .shp file found inside {p}") from e
        return gpd.read_file(f"zip://{p}!{shp_members[0]}")


def _read_retained(params: str | None) -> set[str] | None:
    """Gauge ids passing the KGE >= 0.5 screen, from calibrated_parameters.csv's
    'used_in_analysis' flag (same source as fig01_studyarea.py's panel (a)
    split). Returns None -- meaning 'do not restrict' -- when the file or the
    column is unavailable, since it is a gitignored external input."""
    if params is None or not Path(params).is_file():
        print(f"  [panel c] {params or 'calibrated_parameters.csv'} not found; "
              "mapping every catchment in the geometry file (the 621/671 "
              "retained split needs it).")
        return None
    t = pd.read_csv(params)
    if "used_in_analysis" not in t.columns or "gauge_id" not in t.columns:
        print(f"  [panel c] '{params}' has no 'used_in_analysis' column; "
              "mapping every catchment in the geometry file.")
        return None
    keep = t.loc[t["used_in_analysis"].astype(bool), "gauge_id"]
    return set(keep.astype(str).str.strip().str.replace(r"\.0$", "", regex=True))


def _read_bfi(attr_src: str) -> pd.DataFrame:
    """Locate baseflow_index + gauge_id, either in a single file (csv/parquet)
    or by searching a directory of CAMELS-GB attribute tables."""
    p = Path(attr_src)
    cands = [p] if p.is_file() else (
        list(p.glob("*hydrologic*")) + list(p.glob("*hydro*"))
        + list(p.glob("*.csv")) + list(p.glob("*.parquet")))
    for f in cands:
        try:
            t = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
        except Exception:
            continue
        if "baseflow_index" in t.columns and "gauge_id" in t.columns:
            return t[["gauge_id", "baseflow_index"]].dropna()
    raise FileNotFoundError(
        f"no table with columns gauge_id + baseflow_index found at {attr_src}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="slow_full_flow.parquet")
    ap.add_argument("--geom", default=None,
                    help="catchment geometry file: shapefile (.shp), "
                         "GeoPackage (.gpkg), or a .zip of shapefile "
                         "components (read directly, no unzipping needed)")
    ap.add_argument("--geom-id", default="ID_STRING",
                    help="gauge-id column in the geometry file "
                         "(camels_gb_v2_catchment_boundaries.shp: ID_STRING)")
    ap.add_argument("--attr-dir", default=None,
                    help="CAMELS-GB hydrologic attribute table (file or dir) "
                         "with gauge_id + baseflow_index")
    ap.add_argument("--basemap", default=None,
                    help="optional GB coastline outline (GeoJSON/GeoPackage) "
                         "drawn beneath the catchment mosaic in panel (c); "
                         "see gb_outline_27700.geojson")
    ap.add_argument("--params", default="calibrated_parameters.csv",
                    help="calibrated_parameters.csv, for the "
                         "'used_in_analysis' (KGE >= 0.5) flag that restricts "
                         "panel (c) to the 621 analysed catchments")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    S.set_style()
    df = load(args.input)
    print("Figure 3 -- store-attribution asymmetry")
    summarise(df)
    make_figure(df, args.outdir, args.geom, args.geom_id, args.attr_dir,
                args.basemap, args.params)


if __name__ == "__main__":
    main()
