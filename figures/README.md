# Figures

Scripts that render the manuscript and Supporting Information figures. They read
the pipeline outputs (from the repository root, or from `derived_output/` for the
committed summary tables) and write a vector PDF plus a 600-dpi PNG preview.
`agu_style.py` holds the shared AGU/Wiley styling (85–170 mm widths, 8 pt text,
Okabe–Ito palette) and is imported by the figure scripts, so it lives here
alongside them.

Run the scripts **from the repository root** so `import agu_style` and the
default input paths resolve. `fig01` and `fig04` render from a fresh clone with
no arguments (their defaults point at the bundled CAMELS-GB attribute tables and
`derived_output/`). `fig05` also renders from a fresh clone, but its
`--projection` and `--forcing` defaults name the working-directory paths a full
pipeline run writes, so point them at the committed copies under
`derived_output/` (as in the command below). Only `fig02` and `fig03` need
`slow_full_flow.parquet`, which is too large to commit and must be regenerated
first.

```bash
python figures/fig01_studyarea.py    --outdir figures   # inputs default to bundled tables + derived_output/
python figures/fig02_continuum.py    --input slow_full_flow.parquet --outdir figures
python figures/fig03_mechanism.py    --input slow_full_flow.parquet --outdir figures \
    --geom figures/camels_gb_v2_catchment_boundaries.zip \
    --attr-dir camels_gb_v2_hydrologic_attributes.csv \
    --basemap figures/gb_outline_27700.geojson   # panel (c); omit these three for the (a)+(b) core
python figures/fig04_corroboration.py --outdir figures   # inputs default to derived_output/
python figures/fig05_projection.py   --projection derived_output/projection_flow.parquet \
    --forcing derived_output/forcing_deltas_rcp85.csv --outdir figures   # --responder defaults to derived_output/
python figures/figS1_timescales.py   --params calibrated_parameters.csv --outdir figures
```

| Script | Figure | Main input(s) |
|--------|--------|---------------|
| `fig01_studyarea.py` | Fig. 1 (study catchments, corroboration network) | `camels_gb_v2_topographic_attributes.csv`, `camels_gb_v2_climatic_attributes.csv` (both bundled), `derived_output/corroboration_summary_final.csv`, `camels_gb_v2_catchment_boundaries.zip`, `gb_outline_27700.geojson` (bundled); `calibrated_parameters.csv` optional (KGE exclusion split) |
| `fig02_continuum.py` | Fig. 2 (gap continuum) | `slow_full_flow.parquet` (regenerated) |
| `fig03_mechanism.py` | Fig. 3 (store mechanism, BFI map) | `slow_full_flow.parquet` (regenerated), `camels_gb_v2_catchment_boundaries.zip`, `camels_gb_v2_hydrologic_attributes.csv` (`baseflow_index`), `gb_outline_27700.geojson` (bundled); `calibrated_parameters.csv` optional (restricts panel (c) to the 621 retained catchments) |
| `fig04_corroboration.py` | Fig. 4 (borehole corroboration) | the three corroboration tables in `derived_output/` |
| `fig05_projection.py` | Fig. 5 (projection) | `derived_output/projection_flow.parquet`, `derived_output/forcing_deltas_rcp85.csv`, `derived_output/responder_table.parquet` (all committed) |
| `figS1_timescales.py` | Fig. S1 (recession timescales) | `calibrated_parameters.csv` (from hbv-model; timescales computed from K1/K2) |

**External dependencies:**
- `fig01_studyarea.py` and `fig03_mechanism.py` both render their maps as
  catchment boundary polygons from `camels_gb_v2_catchment_boundaries.zip`
  (read via the GDAL `zip://` virtual filesystem, keyed by `ID_STRING`). This
  file is large and is **not** committed to the repository (see `.gitignore`)
  — obtain it from the CAMELS-GB v2 dataset (see the top-level README) and
  place it in `figures/` (`fig01_studyarea.py`'s `--geom` default) or pass
  `--geom` explicitly. Both scripts degrade gracefully to catchment outlet
  points (from the bundled topographic attribute table) if `geopandas` or the
  boundary file is unavailable; `fig03_mechanism.py` instead skips its map
  panel entirely, since panels (a)-(b) there are already a complete figure
  without it.
- `fig01_studyarea.py` also reads `calibrated_parameters.csv` (from hbv-model)
  to mark the 50 catchments excluded by the KGE screen; without it, the script
  still runs and shows all 671 catchments as retained.
- `fig01_studyarea.py`'s coastline backdrop, `gb_outline_27700.geojson`, *is*
  bundled (~4.3 MB): built from the ONS Open Geography Portal's official
  "Countries" boundary dataset (the same country-level ONS source used for
  Scotland/England/Wales assignment in the companion dfaa-analysis
  repository), unioning England/Scotland/Wales and dropping Northern Ireland.
  It is dissolved at country level rather than from smaller units (e.g. local
  authority districts) deliberately — unioning finer-grained polygons leaves
  sliver gaps at shared edges that read as broken coastline around dense
  island groups (Orkney, Shetland) once dissolved to national scale. Both
  panels' axis extent comes from this outline unioned with the catchment
  bounds, not the catchments alone, since Shetland has no gauged CAMELS-GB
  catchment and would otherwise be cropped off. Pass `--basemap` to use a
  different file, or `--basemap ""` to fall back to an outline derived from
  the catchment polygons themselves (which, being catchment-bounded, will not
  extend to Shetland).
