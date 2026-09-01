#!/usr/bin/env python
"""
fig01_studyarea.py
===================
Figure 1 -- the study catchments and the observed-groundwater corroboration
network (manuscript Sections 2.1 and 2.6).

Message
-------
Two geographic facts the rest of the paper states only as counts: where the
621 analysed catchments sit within Great Britain's climatic gradient, and
where the 42 boreholes used to corroborate the lower-zone mechanism
(Section 3.3) sit relative to the aquifers they monitor.

Manuscript values to reproduce
    Catchments : 621 of 671 retained (validation-period KGE >= 0.5); 50 excluded
    Boreholes  : 42 total -- Chalk 17, other fractured/oolitic/limestone 14,
                 Permo-Triassic sandstone 11
    Labelled   : 37010, 34012 (Section 3.3 identifiability cross-check),
                 39007, 34004 (abstraction-affected Chalk misses),
                 42016 "Abbotstone" (flagship daily Chalk site)

Panels
    (a) All 671 candidate catchments as boundary polygons (British National
        Grid), coloured by baseline mean annual rainfall for the 621 retained
        after the validation-period KGE >= 0.5 screen (Section 2.1); the 50
        excluded catchments are filled flat mid-grey, dark enough to separate
        clearly from the near-white GB land backdrop (see LAND_FILL /
        EXCLUDED_FILL), and drawn on top of the retained mosaic so that the
        nested ones stay visible (see _panel_a_geom).
    (b) The 42 corroboration catchments (Section 2.6), filled by aquifer
        class with the same palette as fig04_corroboration.py, over the
        full catchment mosaic in pale grey for context; the five catchments
        discussed individually in the text are labelled.

Both panels are drawn over a GB coastline outline (gb_outline_27700.geojson,
bundled alongside this script -- built from the ONS Open Geography Portal's
official "Countries" boundary dataset, i.e. the same country-level ONS source
used for Scotland/England/Wales assignment in the companion dfaa-analysis
repository. England, Scotland and Wales are unioned into one Great Britain
outline; Northern Ireland is dropped. This is deliberately a country-level
dissolve, not a union of many smaller units (e.g. local authority districts):
unioning finer-grained polygons leaves sliver gaps at shared edges that are
invisible at LAD scale but read as broken/fragmented coastline around
dense island groups such as Orkney and Shetland once dissolved to national
scale). The axis extent for both panels is taken from this outline (unioned
with the catchment bounds), not from the catchments alone -- Shetland has no
gauged CAMELS-GB catchment, so bounding on catchment extent alone would crop
it off the figure entirely. Matches the coastline role fig03_mechanism.py's
panel c plays there, though that figure sources its own basemap file
separately (see its docstring). If --basemap points at a missing file, or is
passed as an empty string, the outline falls back to the dissolved union of
the 671 catchment polygons themselves (see _dissolved_outline()) --
self-contained, but traces the tiled catchments' edge rather than the true
coast, and (being catchment-bounded) will not extend to Shetland.

Rendering falls back to catchment outlet points, rather than polygons, if
`geopandas` or the boundary file (--geom) is unavailable -- the same
graceful-degradation behaviour as fig03_mechanism.py's own map panel. The
point fallback needs no large external file, so the figure is never blocked
on it; --basemap still applies in that case.

Input
    --topo            camels_gb_v2_topographic_attributes.csv  (bundled)
                       (gauge_id, gauge_easting, gauge_northing)
    --climate         camels_gb_v2_climatic_attributes.csv  (bundled)
                       (gauge_id, p_mean -- mean daily rainfall, mm/day,
                        observed CAMELS-GB hydroclimatology; multiplied by
                        365.25 here for a real-calendar annual total. This is
                        independent of, and need not exactly match,
                        forcing_deltas.py's CHESS-SCAPE-derived 360-day-
                        calendar p_annual used elsewhere in the pipeline --
                        both describe the same climatic gradient.)
    --params          calibrated_parameters.csv  (external, from hbv-model;
                       gauge_id, used_in_analysis). Optional: without it,
                       panel (a) shows all 671 catchments coloured by
                       rainfall and skips the excluded/retained distinction.
    --corrob-summary  derived_output/corroboration_summary_final.csv
                       (gauge_id, grp)
    --geom            catchment boundary polygons: shapefile (.shp),
                       GeoPackage (.gpkg), or a .zip of shapefile components
                       (read directly, no unzipping needed), keyed by
                       --geom-id. Defaults to the copy of
                       camels_gb_v2_catchment_boundaries.zip bundled
                       alongside this script; this file is large and is
                       **not** committed to the repository (see .gitignore
                       and fig03_mechanism.py's docstring) -- obtain it from
                       the CAMELS-GB v2 dataset if it is not already present.
    --basemap         GB coastline outline (GeoJSON/GeoPackage), drawn
                       beneath both panels. Defaults to the
                       gb_outline_27700.geojson bundled alongside this
                       script; pass an empty string to force the
                       dissolved-catchment fallback outline instead.

Usage
    python fig01_studyarea.py --outdir figures
    python fig01_studyarea.py --params calibrated_parameters.csv --outdir figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import figure_style as S

# grp labels/colours, matched to fig04_corroboration.py so the two figures
# read as one visual system.
GRP_ORDER = ["Chalk", "Other", "Permo-Triassic sst"]
GRP_LABELS = {
    "Chalk": "Chalk",
    "Other": "Other fractured, oolitic, limestone",
    "Permo-Triassic sst": "Permo-Triassic sandstone",
}
GRP_COLORS = {
    "Chalk": S.C_SLOW,
    "Other": S.OKABE_ITO["green"],
    "Permo-Triassic sst": S.C_FUTURE,
}
GRP_MARKERS = {"Chalk": "o", "Other": "s", "Permo-Triassic sst": "^"}

# Greys for the map backdrop and for panel (a)'s excluded catchments. These are
# deliberately far apart: an earlier version filled the land backdrop at 0.94
# and the excluded catchments at 0.82, which is too small a step to read at
# print size -- the 50 excluded catchments vanished into the GB landmass. Keep
# any future adjustment to these two values well separated (and keep the land
# fill lighter than the pale end of YlGnBu, so lightly-rained retained
# catchments still stand off the background too).
LAND_FILL = "0.97"          # GB coastline polygon, both panels
EXCLUDED_FILL = "0.60"      # panel (a): catchments failing the KGE >= 0.5 screen
EXCLUDED_EDGE = "0.30"

DEFAULT_GEOM = str(Path(__file__).parent / "camels_gb_v2_catchment_boundaries.zip")
DEFAULT_BASEMAP = str(Path(__file__).parent / "gb_outline_27700.geojson")

# catchments discussed individually in Section 3.3; labelled in panel (b) as
# {gauge_id: (label, x-offset pt, y-offset pt)}. Offsets are hand-tuned
# against these catchments' actual positions (39007 and 42016 sit close
# together in the southwest; 34012/34004/37010 spread north-to-south along
# the east) so the five labels do not collide.
LABELLED = {
    42016: ("42016 (Abbotstone)", 10, -16),
    39007: ("39007", -55, 10),
    37010: ("37010", 10, 6),
    34012: ("34012", 10, 8),
    34004: ("34004", 10, -14),
}


def _norm_id(s: pd.Series) -> pd.Series:
    """Coerce a gauge-id column to a plain digit string so merges across
    files that store it as int vs. str (e.g. bundled CSV vs. an externally
    supplied calibrated_parameters.csv, or a shapefile's ID_STRING field) do
    not silently drop rows."""
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def load_catchments(topo_csv: str, climate_csv: str,
                     params_csv: str | None) -> pd.DataFrame:
    topo = pd.read_csv(topo_csv)
    clim = pd.read_csv(climate_csv)
    need_topo = {"gauge_id", "gauge_easting", "gauge_northing"}
    need_clim = {"gauge_id", "p_mean"}
    if need_topo - set(topo.columns):
        raise KeyError(f"{topo_csv} is missing columns: "
                        f"{sorted(need_topo - set(topo.columns))}")
    if need_clim - set(clim.columns):
        raise KeyError(f"{climate_csv} is missing columns: "
                        f"{sorted(need_clim - set(clim.columns))}")

    topo, clim = topo.copy(), clim.copy()
    topo["_gid"] = _norm_id(topo["gauge_id"])
    clim["_gid"] = _norm_id(clim["gauge_id"])
    df = topo[["_gid", "gauge_id", "gauge_easting", "gauge_northing"]].merge(
        clim[["_gid", "p_mean"]], on="_gid", how="inner")
    df["p_annual"] = df["p_mean"] * 365.25

    df["used_in_analysis"] = True
    if params_csv is not None and Path(params_csv).is_file():
        params = pd.read_csv(params_csv)
        if "used_in_analysis" in params.columns:
            params = params.copy()
            params["_gid"] = _norm_id(params["gauge_id"])
            flag = (params[["_gid", "used_in_analysis"]]
                    .drop_duplicates("_gid"))
            df = df.drop(columns="used_in_analysis").merge(
                flag, on="_gid", how="left")
            df["used_in_analysis"] = df["used_in_analysis"].fillna(False).astype(bool)
        else:
            print(f"  [panel a] '{params_csv}' has no 'used_in_analysis' "
                  "column; showing all catchments as retained.")
    else:
        print(f"  [panel a] {params_csv or 'calibrated_parameters.csv'} not "
              "found; showing all catchments as retained (the excluded/"
              "retained split needs it -- see the hbv-model dependency in "
              "the top-level README).")
    return df.drop(columns="_gid")


def load_boreholes(corrob_csv: str, topo_csv: str) -> pd.DataFrame:
    corrob = pd.read_csv(corrob_csv)
    topo = pd.read_csv(topo_csv)
    need = {"gauge_id", "grp"}
    if need - set(corrob.columns):
        raise KeyError(f"{corrob_csv} is missing columns: "
                        f"{sorted(need - set(corrob.columns))}")

    corrob, topo = corrob.copy(), topo.copy()
    corrob["_gid"] = _norm_id(corrob["gauge_id"])
    topo["_gid"] = _norm_id(topo["gauge_id"])
    df = corrob[["_gid", "gauge_id", "grp"]].merge(
        topo[["_gid", "gauge_easting", "gauge_northing"]], on="_gid", how="left")
    missing = df.loc[df["gauge_easting"].isna(), "gauge_id"].tolist()
    if missing:
        raise ValueError(
            f"no topographic coordinates for corroboration gauges: {missing}")
    df["grp"] = pd.Categorical(df["grp"], categories=GRP_ORDER, ordered=True)
    return df.drop(columns="_gid")


def summarise(catch: pd.DataFrame, bore: pd.DataFrame) -> None:
    n_total, n_used = len(catch), int(catch["used_in_analysis"].sum())
    print(f"  catchments: {n_used} of {n_total} retained "
          f"({n_total - n_used} excluded)")
    rain = catch.loc[catch["used_in_analysis"], "p_annual"]
    print(f"  baseline mean annual rainfall (retained catchments): "
          f"median {rain.median():.0f} mm  [{rain.min():.0f}, {rain.max():.0f}]  "
          f"(for comparison, the CHESS-SCAPE WY1982-2010 baseline quoted in "
          f"Section 3.4 is 984 mm [617, 2219]; this panel colours by the "
          f"CAMELS-GB climatology, which is a different quantity)")
    print(f"  corroboration boreholes: {len(bore)} total")
    for g in GRP_ORDER:
        print(f"    {GRP_LABELS[g]:38s} n={int((bore['grp'] == g).sum())}")
    missing_labels = set(LABELLED) - set(bore["gauge_id"])
    if missing_labels:
        print(f"  [panel b] labelled gauges not found in the corroboration "
              f"table: {sorted(missing_labels)}")


def _read_geom(path: str):
    """Read the catchment-geometry file. Transparently handles a .zip archive
    of shapefile components (as delivered from JASMIN, e.g.
    camels_gb_v2_catchment_boundaries.zip) via GDAL's virtual zip filesystem,
    so --geom can point straight at the .zip without unzipping it first.
    Identical in spirit to fig03_mechanism.py's helper of the same name."""
    import geopandas as gpd
    p = Path(path)
    if p.suffix.lower() != ".zip":
        return gpd.read_file(path)
    try:
        return gpd.read_file(f"zip://{p}")
    except Exception as e:
        import zipfile
        with zipfile.ZipFile(p) as zf:
            shp_members = [n for n in zf.namelist() if n.lower().endswith(".shp")]
        if not shp_members:
            raise FileNotFoundError(f"no .shp file found inside {p}") from e
        return gpd.read_file(f"zip://{p}!{shp_members[0]}")


def load_geometry(geom: str | None, geom_id: str):
    """Return the catchment-boundary GeoDataFrame, or None if geopandas or
    the geometry file is unavailable (the figure then falls back to
    catchment outlet points for both panels)."""
    if geom is None or not Path(geom).is_file():
        print(f"  [map] geometry file not found at '{geom}'; "
              "falling back to catchment outlet points.")
        return None
    try:
        import geopandas  # noqa: F401
    except ImportError:
        print("  [map] geopandas not available; falling back to catchment "
              "outlet points.")
        return None
    gdf = _read_geom(geom)
    if geom_id not in gdf.columns:
        raise KeyError(f"'{geom_id}' not in geometry columns: {list(gdf.columns)}")
    gdf = gdf.copy()
    gdf["_gid"] = _norm_id(gdf[geom_id])
    return gdf


def _draw_basemap(ax, basemap: str | None):
    """Plot the coastline file and return it (so callers can use its extent
    for axis limits); returns None if no basemap is configured."""
    if basemap is None:
        return None
    import geopandas as gpd
    gdf = gpd.read_file(basemap)
    gdf.plot(ax=ax, facecolor=LAND_FILL, edgecolor="0.4", linewidth=0.5, zorder=0)
    return gdf


def _dissolved_outline(gdf):
    """A GB outline derived from the union of the 671 catchment polygons,
    drawn as a fallback when the bundled gb_outline_27700.geojson (or a
    user-supplied --basemap) is unavailable. It traces the tiled catchments'
    outer edge rather than the true coast (small estuaries and other
    unmonitored coastal strips are not catchment area), so it is close to,
    but visibly rougher than, a proper coastline -- only used if --basemap
    cannot be read.
    """
    import geopandas as gpd
    return gpd.GeoDataFrame(geometry=[gdf.geometry.union_all()], crs=gdf.crs)


def _panel_a_geom(ax, gdf, catch: pd.DataFrame) -> None:
    c = catch.copy()
    c["_gid"] = _norm_id(c["gauge_id"])
    merged = gdf.merge(c[["_gid", "p_annual", "used_in_analysis"]],
                       on="_gid", how="inner")
    retained = merged[merged["used_in_analysis"]]
    excluded = merged[~merged["used_in_analysis"]]
    retained.plot(ax=ax, column="p_annual", cmap="YlGnBu",
                 linewidth=0.1, edgecolor="0.4", zorder=2,
                 legend=True,
                 legend_kwds=dict(label="Baseline mean annual rainfall (mm)",
                                  shrink=0.55, fraction=0.045, pad=0.02))
    # Excluded catchments go on TOP of the retained mosaic, not under it.
    # CAMELS-GB gauges nest: a headwater catchment is wholly contained in the
    # catchment of every gauge downstream of it. Drawing the excluded set
    # first hid 35 of the 50 completely (100% of their area overpainted by a
    # larger retained catchment) and clipped two more, so the panel showed
    # only ~13 of them and understated the exclusions badly. Painted last,
    # each one reads as a grey patch inside its coloured parent.
    if len(excluded):
        excluded.plot(ax=ax, facecolor=EXCLUDED_FILL, edgecolor=EXCLUDED_EDGE,
                     linewidth=0.15, zorder=3,
                     label=f"Excluded by screening (n={len(excluded)})")
    if len(excluded):
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=EXCLUDED_FILL,
                                 edgecolor=EXCLUDED_EDGE,
                                 label=f"Excluded by screening (n={len(excluded)})")]
        ax.legend(handles=handles, loc="upper center",
                 bbox_to_anchor=(0.5, -0.02), fontsize=6,
                 handletextpad=0.3, borderaxespad=0.3)


def _panel_a_points(ax, fig, catch: pd.DataFrame) -> None:
    used = catch[catch["used_in_analysis"]]
    excluded = catch[~catch["used_in_analysis"]]
    sc = ax.scatter(used["gauge_easting"], used["gauge_northing"],
                    c=used["p_annual"], cmap="YlGnBu", s=7,
                    edgecolor="none", zorder=2)
    if len(excluded):
        ax.scatter(excluded["gauge_easting"], excluded["gauge_northing"],
                  marker="x", s=10, linewidth=0.6, color=S.OKABE_ITO["grey"],
                  zorder=3, label=f"Excluded by screening (n={len(excluded)})")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02),
                 fontsize=6, handletextpad=0.3, borderaxespad=0.3)
    cb = fig.colorbar(sc, ax=ax, shrink=0.55, fraction=0.045, pad=0.02)
    cb.set_label("Baseline mean annual rainfall (mm)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)


def _panel_b_geom(ax, gdf, bore: pd.DataFrame) -> None:
    b = bore.copy()
    b["_gid"] = _norm_id(b["gauge_id"])
    gdf.plot(ax=ax, facecolor="0.92", edgecolor="0.75", linewidth=0.1, zorder=1)
    merged = gdf.merge(b[["_gid", "gauge_id", "grp"]], on="_gid", how="inner")
    for g in GRP_ORDER:
        sub = merged[merged["grp"] == g]
        if len(sub):
            sub.plot(ax=ax, facecolor=GRP_COLORS[g], edgecolor="white",
                     linewidth=0.4, zorder=2)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=GRP_COLORS[g],
                             edgecolor="white",
                             label=f"{GRP_LABELS[g]} (n={int((bore['grp'] == g).sum())})")
              for g in GRP_ORDER]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
             fontsize=6, handletextpad=0.3, borderaxespad=0.3)
    for gid, (label, dx, dy) in LABELLED.items():
        r = merged.loc[merged["gauge_id"] == gid]
        if r.empty:
            continue
        pt = r.iloc[0]["geometry"].representative_point()
        ax.annotate(label, xy=(pt.x, pt.y),
                   xytext=(dx, dy), textcoords="offset points",
                   fontsize=5.5, ha="left" if dx >= 0 else "right",
                   va="bottom" if dy >= 0 else "top",
                   arrowprops=dict(arrowstyle="-", lw=0.4, color="0.3"))


def _panel_b_points(ax, catch: pd.DataFrame, bore: pd.DataFrame) -> None:
    ax.scatter(catch["gauge_easting"], catch["gauge_northing"],
              s=2.5, color="0.85", edgecolor="none", zorder=1)
    for g in GRP_ORDER:
        sub = bore[bore["grp"] == g]
        ax.scatter(sub["gauge_easting"], sub["gauge_northing"],
                  marker=GRP_MARKERS[g], s=20, color=GRP_COLORS[g],
                  edgecolor="white", linewidth=0.4, zorder=3,
                  label=f"{GRP_LABELS[g]} (n={len(sub)})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=6,
             markerscale=0.9, handletextpad=0.3, borderaxespad=0.3)
    for gid, (label, dx, dy) in LABELLED.items():
        r = bore.loc[bore["gauge_id"] == gid]
        if r.empty:
            continue
        r = r.iloc[0]
        ax.annotate(label, xy=(r["gauge_easting"], r["gauge_northing"]),
                   xytext=(dx, dy), textcoords="offset points",
                   fontsize=5.5, ha="left" if dx >= 0 else "right",
                   va="bottom" if dy >= 0 else "top",
                   arrowprops=dict(arrowstyle="-", lw=0.4, color="0.3"))


def make_figure(catch: pd.DataFrame, bore: pd.DataFrame, outdir: str,
                gdf, basemap: str | None) -> None:
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(S.W_2COL, 105 * S.MM),
        gridspec_kw=dict(width_ratios=[1.0, 1.0], wspace=0.28),
    )

    if gdf is not None:
        outline_gdf = None
        if basemap is not None:
            outline_gdf = _draw_basemap(axA, basemap)
            _draw_basemap(axB, basemap)
        else:
            outline_gdf = _dissolved_outline(gdf)
            for ax in (axA, axB):
                outline_gdf.plot(ax=ax, facecolor=LAND_FILL, edgecolor="0.4",
                                 linewidth=0.5, zorder=0)
        _panel_a_geom(axA, gdf, catch)
        _panel_b_geom(axB, gdf, bore)
        # extent from the coastline/outline, not just the catchments -- some
        # of Great Britain (e.g. Shetland) has no gauged CAMELS-GB catchment,
        # so bounding on the catchments alone would crop those areas off.
        bounds = np.array([gdf.total_bounds] +
                          ([outline_gdf.total_bounds] if outline_gdf is not None else []))
        xmin, ymin = bounds[:, 0].min(), bounds[:, 1].min()
        xmax, ymax = bounds[:, 2].max(), bounds[:, 3].max()
        pad = 0.02 * max(xmax - xmin, ymax - ymin)
        xlim, ylim = (xmin - pad, xmax + pad), (ymin - pad, ymax + pad)
    else:
        if basemap is None:
            print("  [map] no --basemap and no polygon geometry available; "
                  "panels will show catchment outlet points with no "
                  "coastline context.")
        outline_gdf = _draw_basemap(axA, basemap)
        _panel_a_points(axA, fig, catch)
        _draw_basemap(axB, basemap)
        _panel_b_points(axB, catch, bore)
        # extent from the coastline (if present), not just the catchment
        # outlets -- see the identical comment in the gdf branch above.
        catch_bounds = np.array([catch["gauge_easting"].min(), catch["gauge_northing"].min(),
                                 catch["gauge_easting"].max(), catch["gauge_northing"].max()])
        bounds = np.array([catch_bounds] +
                          ([outline_gdf.total_bounds] if outline_gdf is not None else []))
        xmin, ymin = bounds[:, 0].min(), bounds[:, 1].min()
        xmax, ymax = bounds[:, 2].max(), bounds[:, 3].max()
        pad = 0.03 * max(xmax - xmin, ymax - ymin)
        xlim, ylim = (xmin - pad, xmax + pad), (ymin - pad, ymax + pad)

    for ax, letter in ((axA, "a"), (axB, "b")):
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_axis_off()
        S.panel_label(ax, letter, x=0.02, y=0.99)

    S.save(fig, Path(outdir) / "fig01_studyarea")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topo", default="camels_gb_v2_topographic_attributes.csv")
    ap.add_argument("--climate", default="camels_gb_v2_climatic_attributes.csv")
    ap.add_argument("--params", default="calibrated_parameters.csv",
                    help="calibrated_parameters.csv (external, from hbv-model); "
                         "optional -- see module docstring")
    ap.add_argument("--corrob-summary",
                    default="derived_output/corroboration_summary_final.csv")
    ap.add_argument("--geom", default=DEFAULT_GEOM,
                    help="catchment geometry file: shapefile (.shp), "
                         "GeoPackage (.gpkg), or a .zip of shapefile "
                         "components. Defaults to the copy bundled alongside "
                         "this script; pass an empty string to force the "
                         "point-based fallback.")
    ap.add_argument("--geom-id", default="ID_STRING",
                    help="gauge-id column in the geometry file "
                         "(camels_gb_v2_catchment_boundaries.shp: ID_STRING)")
    ap.add_argument("--basemap", default=DEFAULT_BASEMAP,
                    help="GB coastline outline (GeoJSON/GeoPackage) drawn "
                         "beneath both panels. Defaults to the bundled "
                         "gb_outline_27700.geojson; pass an empty string to "
                         "force the dissolved-catchment fallback outline.")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    params_csv = args.params if Path(args.params).is_file() else None
    basemap = args.basemap if args.basemap and Path(args.basemap).is_file() else None
    if args.basemap and basemap is None:
        print(f"  [map] --basemap '{args.basemap}' not found; falling back "
              "to the dissolved-catchment outline (geometry available) or "
              "no coastline context (point fallback).")

    S.set_style()
    catch = load_catchments(args.topo, args.climate, params_csv)
    bore = load_boreholes(args.corrob_summary, args.topo)
    gdf = load_geometry(args.geom or None, args.geom_id)
    print("Figure 1 -- study catchments and corroboration network")
    summarise(catch, bore)
    make_figure(catch, bore, args.outdir, gdf, basemap)


if __name__ == "__main__":
    main()
