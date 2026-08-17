# storage-pace-analysis

Analysis pipeline for **"Storage controls on the pace of flood–drought
transitions across Great Britain under a warming climate"** (Patil, Dallison &
Jahanshahi, submitted to *Water Resources Research*).

The pipeline runs the HBV rainfall–runoff model across CHESS-SCAPE climate
projections for 621 CAMELS-GB v2 catchments, identifies flood-to-drought (FTD)
and drought-to-flood (DTF) transitions with a storage-based pairing and a
coherence filter, attributes each transition to the HBV store that limits its
pace, corroborates the lower-zone mechanism against observed borehole levels,
and quantifies how the pace of these transitions changes under warming.

This is the slow/storage-limited counterpart to the abrupt-regime analysis in
the companion repository
[**dfaa-analysis**](https://github.com/sopanpatil/dfaa-analysis). The HBV model
and its SCE-UA calibration live in
[**hbv-model**](https://github.com/sopanpatil/hbv-model).

## Requirements

Python 3.10+ and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

`numba` (JIT kernels in [hbv-model](https://github.com/sopanpatil/hbv-model)'s
`hbv.py`, required transitively) and `pyarrow` (parquet I/O) are required;
`geopandas` is used only for the catchment-boundary maps in
`figures/fig01_studyarea.py` and `figures/fig03_mechanism.py`. Both scripts
degrade gracefully without it — `fig01_studyarea.py` falls back to catchment
outlet points, and `fig03_mechanism.py` drops its map panel.

**`hbv-model` is a required external dependency, not on PyPI.** Clone it as a
sibling directory, then symlink its package into this repository so
`run_hbv_chess_scape.py`'s `from hbv_model.hbv import HBVModel` resolves —
cloning as a sibling alone is not enough, since Python only searches the
directory containing the script being run, not its parent or siblings:

```bash
git clone https://github.com/sopanpatil/hbv-model ../hbv-model
ln -s ../hbv-model/hbv_model hbv_model
```

Requires **`hbv-model` `v1.1.0` or later**
(`https://doi.org/10.5281/zenodo.21860981`) — the version that logs the
`Q0`/`Q1`/`Q2`/`melt` generating fluxes the store-attribution step in this
pipeline depends on. Earlier versions (e.g. `v1.0.0`, used by the companion
`dfaa-analysis` repository) do not expose these fields and will raise a
`KeyError` in `run_hbv_chess_scape.py`.

Most analysis scripts run a self-contained synthetic self-test when called with
no arguments, e.g. `python projection_analysis.py`, exercising the code without
any external data. Self-test output is written under `selftest/` (see
`selftest_io.py`) so a bare run can never overwrite a production file. The
exceptions are `verify_manuscript_numbers.py`, which requires its input, and
`run_hbv_chess_scape.py`, which needs the external `hbv-model` package.

## Data

### External datasets (not bundled)

| Dataset | Used for | Source |
|---------|----------|--------|
| CAMELS-GB v2 (Coxon et al., 2026) | catchment attributes, observed streamflow/borehole records, catchment boundaries | NERC Environmental Data Service |
| CHESS-SCAPE (Robinson et al., 2023) | daily precipitation, temperature, wind, humidity, and radiation forcing for the four RCPs × four members | Centre for Environmental Data Analysis (CEDA) |
| Derived PET | third HBV forcing input, computed from CHESS-SCAPE variables | this repository, `pet_penman_monteith.py` (Stage 0; FAO-56 Penman-Monteith, not a native CHESS-SCAPE variable) |
| Calibrated HBV parameters (`calibrated_parameters.csv`) and the HBV model itself (`HBVModel`) | per-catchment HBV parameters, KGE/`used_in_analysis` flags, and the model implementation | [hbv-model](https://github.com/sopanpatil/hbv-model) `>= v1.1.0` (`https://doi.org/10.5281/zenodo.21860981`) |
| `chess_scape_output/` (catchment-mean forcing + HBV outputs, all 16 RCP × member combinations) | full Stage 0–1 reproducibility without a JASMIN rerun | archived on Zenodo at `https://doi.org/10.5281/zenodo.21861386`, including the raster-to-catchment extraction code as provenance |

The pipeline consumes CHESS-SCAPE as per-catchment daily means, one wide CSV per
variable, member, and RCP:
`<rcp>_<member>_{pr,pet,tas}_catchment_means_combined.csv`.

### Bundled data (this repository)

Four CAMELS-GB v2 attribute tables that the code reads directly are included at
the repository root:

- `camels_gb_v2_climatic_attributes.csv`
- `camels_gb_v2_topographic_attributes.csv`
- `camels_gb_v2_hydrogeology_attributes.csv`
- `camels_gb_v2_hydrologic_attributes.csv`

These are redistributed from CAMELS-GB v2 under the Open Government Licence
v3.0; please retain the CAMELS-GB attribution when reusing them.

Also bundled, under `figures/`:

- `gb_outline_27700.geojson` (~4.3 MB) — the Great Britain coastline drawn
  beneath the maps in `figures/fig01_studyarea.py` and
  `figures/fig03_mechanism.py`. Derived from the Office for National Statistics
  Open Geography Portal "Countries" boundary dataset (England, Scotland and
  Wales unioned; Northern Ireland dropped; reprojected to EPSG:27700), and
  likewise redistributed under the Open Government Licence v3.0 — please retain
  the ONS attribution when reusing it. Note that the CAMELS-GB v2 *catchment*
  boundaries (`figures/camels_gb_v2_catchment_boundaries.zip`) are **not**
  bundled; see the external-datasets table above.

## Repository layout

```
storage-pace-analysis/
├── pet_penman_monteith.py          Stage 0: FAO-56 PET from catchment-mean CHESS-SCAPE variables
├── run_hbv_chess_scape.py          Stage 1: run HBV over CHESS-SCAPE; log states, fluxes, snowmelt
├── slow_transition_analysis.py     Stages 2–5: event detection, pairing, coherence, store attribution
├── recession_timescales.py         UZ/LZ recession timescales (Fig. S1)
├── snow_melt_contribution.py       snowmelt within slow FTD carriers
├── projection_analysis.py          Stage 6: baseline→future change in transition pace, per RCP
├── bootstrap_conditional_gap.py    conditional-gap bootstrap CIs
├── bootstrap_paired_differences.py CIs on DIFFERENCES between changes (RCP vs RCP, FTD vs DTF)
├── coherence_filter_diagnostics.py coherence-filter vs store-attribution coupling; tail exceedance
├── forcing_deltas.py               per-catchment baseline→future forcing-change metrics
├── responder_characterisation.py   what distinguishes catchments whose slow FTDs intensify
├── si_sensitivity_analysis.py      Supporting Information: pairing/coherence sensitivity tables
├── si_table_s3.py                  Supporting Information: Table S3 (candidate discriminators)
├── verify_manuscript_numbers.py    recompute and check the reported Results numbers
├── verify_robustness_checks.py     the two robustness numbers that need the raw HBV output
├── regenerate_and_verify_fluxes.py provenance check: prove the re-run reproduces the archived HBV states/fluxes
├── selftest_io.py                  keeps no-argument self-test output out of the production paths
├── camels_gb_v2_*_attributes.csv   bundled CAMELS-GB v2 attribute tables (OGL v3.0)
├── figures/                        figure scripts + shared AGU styling (see figures/README.md)
├── derived_output/                 small summary tables backing the figures (see derived_output/README.md)
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Pipeline

Each stage reads the outputs of the previous one. Directories default to
`chess_scape_output/` for the HBV inputs/outputs and the repository root for the
transition tables.

**Stage -1 — raw CHESS-SCAPE catchment extraction.** Extracts catchment-mean
daily climate variables (`pr`, `tas`, `tasmax`, `tasmin`, `sfcWind`, `hurs`,
`rsds`) from the raw CHESS-SCAPE 1 km gridded netCDFs, against the CAMELS-GB
v2 catchment boundaries, then concatenates the resulting per-month CSVs into
one continuous `<rcp>_<ensemble>_<var>_catchment_means_combined.csv` per
variable. This requires direct filesystem access to CEDA's archive and only
runs on JASMIN, so it is not part of this repository. The code (extraction,
LOTUS array-job submission, and concatenation) is archived alongside the full
`chess_scape_output/` data on Zenodo at
`https://doi.org/10.5281/zenodo.21861386`, as a record of provenance.

**Stage 0 — potential evapotranspiration.** Computes daily PET (FAO-56
Penman-Monteith, adapted for the 360-day calendar) from the catchment-mean
`tas`, `tasmax`, `tasmin`, `sfcWind`, `hurs`, and `rsds` files Stage -1
produces, giving the third HBV forcing input alongside `pr` and `tas`.

```bash
python pet_penman_monteith.py --rcp rcp85 --ensemble 01 --data-dir chess_scape_output
```

**Stage 1 — HBV simulation.** Runs the calibrated HBV across CHESS-SCAPE forcing
for one RCP and member, writing wide (date × gauge) CSVs for discharge, the
SM/UZ/LZ/SP states, the generating fluxes Q0/Q1/Q2, and snowmelt. The fluxes are
logged directly during the run at high precision, which is what lets the
attribution classify each transition exactly rather than reconstructing flows
from rounded states.

```bash
for rcp in rcp26 rcp45 rcp60 rcp85; do
  for m in 01 04 06 15; do
    python run_hbv_chess_scape.py --rcp $rcp --ensemble $m \
        --params-csv path/to/calibrated_parameters.csv
  done
done
```

**Stages 2–5 — transition extraction.** Event detection (fixed baseline Q5/Q80
thresholds), storage-based pairing of floods to droughts, the coherence filter
(C ≥ 0.60), and store attribution (fast Q0+Q1 vs baseflow Q2) all run in
`slow_transition_analysis.py`, producing the full transition table.

```bash
python slow_transition_analysis.py --jasmin-dir chess_scape_output \
    --params calibrated_parameters.csv --attribution flow \
    --out slow_full_flow.parquet
```

`--attribution flow` selects the flow-weighted attribution the paper reports;
the default `runoff-norm` is the storage-based robustness variant. The 730-day
gap bound (manuscript Section 2.3) is applied by default; pass `--max-gap 0`
only for an uncensored distribution pass, and do not read reported numbers off
the result.

**Stage 6 — projection metrics.** Baseline→future change in transition pace per
RCP: pooled gap-distribution shift, per-catchment zero-filled frequency change,
and the RCP gradient.

```bash
python projection_analysis.py --input slow_full_flow.parquet \
    --attr camels_gb_v2_hydrologic_attributes.csv --out projection_flow.parquet
```

**Supporting analyses.**

```bash
python recession_timescales.py    --params calibrated_parameters.csv
python snow_melt_contribution.py  --input slow_full_flow.parquet --jasmin-dir chess_scape_output
python bootstrap_conditional_gap.py --input slow_full_flow.parquet
python bootstrap_paired_differences.py --input slow_full_flow.parquet \
    --out derived_output/paired_differences.csv          # add --resample-members for the ensemble-aware CI
python coherence_filter_diagnostics.py --input slow_full_flow.parquet   # -> derived_output/*.csv
python forcing_deltas.py --jasmin-dir chess_scape_output --rcp rcp85   # -> forcing_deltas_rcp85.csv
python responder_characterisation.py --deltas projection_flow.parquet \
    --attr-dir . --params calibrated_parameters.csv \
    --forcing forcing_deltas_rcp85.csv --out derived_output/responder_table.parquet
python si_sensitivity_analysis.py
python si_table_s3.py --deltas projection_flow.parquet --attr-dir . \
    --params calibrated_parameters.csv --forcing forcing_deltas_rcp85.csv \
    --out-prefix derived_output/table_s3                # -> SI Table S3
```

The small summary tables consumed by the figures are committed under
`derived_output/`; when you regenerate them, write them there (as with
`responder_table.parquet` above) so the figure scripts find them by default.

**Observational corroboration.** The observation-forced HBV run and the three
borehole tests are provided as derived summary tables in `derived_output/`
(rather than as a standalone script); they feed `figures/fig04_corroboration.py`
and, for the aquifer-class panel, `figures/fig01_studyarea.py`.

## Derived data and figures

See `derived_output/README.md` for the committed summary tables and
`figures/README.md` for how to render each figure. Large intermediates
(`slow_full_flow.parquet`, `projection_flow.parquet`, the HBV output CSVs) are
regenerable and are not committed.

## Verifying the reported numbers

```bash
python verify_manuscript_numbers.py --input slow_full_flow.parquet
```

recomputes the Results figures (transition counts, baseline/future gap medians
and slow shares, per-RCP changes) from the transition table and flags any that
disagree with the values reported in the manuscript.

Two further figures cannot be checked from the transition table, because each
needs the raw HBV output: the range-normalised store-attribution cross-check
(99.6%, Section 3.2), which must re-read the UZ/LZ state series, and the
snowmelt statistics in the Discussion, which read the logged melt flux. Both are
covered by:

```bash
python verify_robustness_checks.py --jasmin-dir chess_scape_output \
    --params calibrated_parameters.csv --input slow_full_flow.parquet

# snowmelt only (the attribution pass reruns the whole pipeline twice):
python verify_robustness_checks.py --jasmin-dir chess_scape_output \
    --input slow_full_flow.parquet --skip-attribution
```

Run it after any change to the pairing, coherence or gap-cap settings, so the
robustness numbers stay on the same production footing as the headline ones.

Finally, the logged generating fluxes (Q0/Q1/Q2 and snowmelt) that underpin the
fast/slow-store attribution can be re-derived and checked against the archive:

```bash
python regenerate_and_verify_fluxes.py --archive-dir chess_scape_output \
    --params-csv calibrated_parameters.csv --model-path .

# prove the tool with no archive data (synthetic self-test):
python regenerate_and_verify_fluxes.py --selftest --model-path .
```

This re-runs the flux-logging HBV, verifies the re-run reproduces the archived
states (to the archive's storage precision) with exact flux closure and
agreement with the Section 2.5 reconstruction, and only then writes the
full-precision fluxes — establishing that the logged fluxes belong to the run
behind the Results. It is a provenance check, not a producer of headline
numbers.

## Citation

If you use this code, please cite the software via `CITATION.cff` and the
accompanying paper (see the same file for the reference).

## License

MIT (code) — see `LICENSE`. Bundled CAMELS-GB v2 attribute tables are under the
Open Government Licence v3.0.
