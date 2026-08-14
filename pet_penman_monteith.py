"""
pet_penman_monteith.py
-----------------------
Stage 0 of the pipeline. Calculates daily reference evapotranspiration
(PET / ET0) for CAMELS-GB v2 catchments using the FAO-56 Penman-Monteith
equation (Allen et al., 1998), adapted for the 360-day calendar used by
CHESS-SCAPE climate projections. Its output is the third HBV forcing input
(alongside pr and tas) that run_hbv_chess_scape.py reads.

Inputs (catchment-averaged CHESS-SCAPE time series, wide format: date x
gauge_id; the same layout as the pr/tas files run_hbv_chess_scape.py reads):
    <rcp>_<ensemble>_tas_catchment_means_combined.csv      mean daily air temp (degC)
    <rcp>_<ensemble>_tasmax_catchment_means_combined.csv   max daily air temp (degC)
    <rcp>_<ensemble>_tasmin_catchment_means_combined.csv   min daily air temp (degC)
    <rcp>_<ensemble>_sfcWind_catchment_means_combined.csv  near-surface wind (m/s)
    <rcp>_<ensemble>_hurs_catchment_means_combined.csv     relative humidity (%)
    <rcp>_<ensemble>_rsds_catchment_means_combined.csv     downwelling shortwave (W/m2)

These six variables (plus pr) are catchment-mean extractions from the raw
CHESS-SCAPE 1 km gridded netCDFs; the zonal-averaging step that produces them
is a separate, upstream preparation step, not part of this script.

Catchment metadata (from the bundled camels_gb_v2_topographic_attributes.csv):
    gauge_id, gauge_lat, elev_mean

Output:
    <rcp>_<ensemble>_pet_catchment_means_combined.csv
    Daily PET (mm/day), same wide format as the inputs.

Notes on the 360-day calendar:
    Each "year" has 12 months of 30 days, so day-of-year runs 1-360. Day-of-
    year is recomputed directly from the date string's month/day fields
    ((month-1)*30 + day) rather than via datetime/cftime parsing, since
    360-day calendar dates are not valid Gregorian dates (e.g. no day 31, and
    a real Feb 30th). The seasonal solar geometry terms (inverse relative
    Earth-Sun distance, solar declination) use a period of 360 days instead
    of 365.25, keeping the seasonal cycle aligned with the 360-day calendar.
    This is an approximation -- the FAO-56 constants were derived for a real
    year -- but it is the standard adaptation used for climate model output.

Wind height:
    CHESS-SCAPE sfcWind is nominally a near-surface (commonly ~10 m) wind
    product. FAO-56 Penman-Monteith requires wind speed at 2 m height; this
    script applies the standard FAO-56 logarithmic wind profile adjustment
    from 10 m to 2 m (--wind-height-m, default 10.0; set to 2.0 to skip the
    adjustment if your wind data is already at 2 m).

Usage
-----
python pet_penman_monteith.py --rcp rcp85 --ensemble 01 --data-dir chess_scape_output

Self-test (no arguments; synthetic single-catchment inputs):
python pet_penman_monteith.py
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

CHESS_SCAPE_ROOT = Path("./chess_scape_output")
TOPO_ATTRS_DEFAULT = "camels_gb_v2_topographic_attributes.csv"

DAYS_IN_YEAR = 360  # CHESS-SCAPE 360-day calendar (12 x 30-day months)

ALBEDO = 0.23               # FAO-56 reference crop albedo
STEFAN_BOLTZMANN = 4.903e-9  # MJ K^-4 m^-2 day^-1
SOLAR_CONSTANT = 0.0820      # MJ m^-2 min^-1
WM2_TO_MJM2DAY = 0.0864      # W/m2 -> MJ/m2/day (86400 s/day x 1e-6 MJ/J)

VARIABLES = ["tas", "tasmax", "tasmin", "sfcWind", "hurs", "rsds"]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def day_of_year_360(date_str):
    """Day-of-year (1-360) for a 360-day calendar date string, e.g.
    '1980-12-01 12:00:00' -> 331. Derived directly from month/day, since
    360-day calendar dates are not valid Gregorian dates."""
    date_part = date_str.split(" ")[0]
    _, month, day = date_part.split("-")
    month, day = int(month), int(day)
    return (month - 1) * 30 + day


def saturation_vapour_pressure(temp_c):
    """FAO-56 saturation vapour pressure (kPa) at temperature temp_c (degC)."""
    return 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))


def wind_speed_to_2m(wind_speed, measurement_height_m):
    """FAO-56 logarithmic wind profile adjustment to a standard 2 m height
    (Eq. 47, Allen et al., 1998). No-op if measurement_height_m == 2.0."""
    if measurement_height_m == 2.0:
        return wind_speed
    return wind_speed * (4.87 / np.log(67.8 * measurement_height_m - 5.42))


def calculate_pet_vectorized(tas, tasmax, tasmin, sfcwind, hurs, rsds,
                             latitude, elevation, doy_array,
                             wind_height_m=10.0, days_in_year=DAYS_IN_YEAR):
    """Vectorized FAO-56 Penman-Monteith reference ET0 (mm/day) for a single
    catchment's full time series. All climate arrays must be 1D and the same
    length as doy_array; latitude and elevation are per-catchment scalars.
    rsds is expected in W/m2 (CHESS-SCAPE convention) and converted internally."""
    tmean = tas
    rsds_mj = rsds * WM2_TO_MJM2DAY

    pressure = 101.3 * ((293 - 0.0065 * elevation) / 293) ** 5.26
    gamma = 0.000665 * pressure

    es_tmax = saturation_vapour_pressure(tasmax)
    es_tmin = saturation_vapour_pressure(tasmin)
    es = (es_tmax + es_tmin) / 2.0
    ea = es * (hurs / 100.0)

    es_tmean = saturation_vapour_pressure(tmean)
    delta = (4098 * es_tmean) / ((tmean + 237.3) ** 2)

    u2 = wind_speed_to_2m(sfcwind, wind_height_m)

    lat_rad = math.radians(latitude)
    dr = 1 + 0.033 * np.cos((2 * np.pi / days_in_year) * doy_array)
    sol_decl = 0.409 * np.sin((2 * np.pi / days_in_year) * doy_array - 1.39)
    ws = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(sol_decl), -1.0, 1.0))

    ra = (24 * 60 / math.pi) * SOLAR_CONSTANT * dr * (
        ws * math.sin(lat_rad) * np.sin(sol_decl)
        + math.cos(lat_rad) * np.cos(sol_decl) * np.sin(ws)
    )
    rso = (0.75 + 2e-5 * elevation) * ra

    rns = (1 - ALBEDO) * rsds_mj
    tmax_k = tasmax + 273.16
    tmin_k = tasmin + 273.16

    # Guard the Rs/Rso cloudiness ratio on rare zero-Rso edge cases (only
    # relevant at very high latitudes in deep winter; not expected for GB
    # catchments, but clipped defensively).
    cloudiness_ratio = np.clip(rsds_mj / np.where(rso > 0, rso, np.nan), 0.0, 1.0)

    rnl = (
        STEFAN_BOLTZMANN
        * ((tmax_k**4 + tmin_k**4) / 2.0)
        * (0.34 - 0.14 * np.sqrt(np.clip(ea, 0, None)))
        * (1.35 * cloudiness_ratio - 0.35)
    )

    rn = rns - rnl
    soil_heat_flux = 0.0  # assumed negligible at daily timestep (FAO-56)

    numerator = 0.408 * delta * (rn - soil_heat_flux) + gamma * (900 / (tmean + 273)) * u2 * (es - ea)
    denominator = delta + gamma * (1 + 0.34 * u2)

    return numerator / denominator


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def load_variable_csvs(data_dir, rcp, ensemble):
    data = {}
    for var in VARIABLES:
        path = data_dir / f"{rcp}_{ensemble}_{var}_catchment_means_combined.csv"
        data[var] = pd.read_csv(path)

    reference = data["tas"]
    dates = reference["date"]
    catchment_ids = [c for c in reference.columns if c != "date"]
    for var, df in data.items():
        if not df["date"].equals(dates):
            raise ValueError(f"Date column mismatch between tas and {var}")
        if list(df.columns) != list(reference.columns):
            raise ValueError(f"Catchment column mismatch between tas and {var}")
    return data, dates, catchment_ids


def run_pet(data_dir, rcp, ensemble, topo_attrs_path, wind_height_m, out_path):
    print("Loading topographic attributes...", flush=True)
    topo = pd.read_csv(topo_attrs_path).set_index("gauge_id")
    latitude_lookup = topo["gauge_lat"].to_dict()
    elevation_lookup = topo["elev_mean"].to_dict()

    print("Loading catchment-mean climate variable files...", flush=True)
    data, dates, catchment_ids = load_variable_csvs(data_dir, rcp, ensemble)

    missing_topo = [cid for cid in catchment_ids if int(cid) not in latitude_lookup]
    if missing_topo:
        raise ValueError(f"Catchments missing from topographic attributes: {missing_topo}")

    print(f"Computing day-of-year (360-day calendar) for {len(dates)} dates...", flush=True)
    doy_array = dates.apply(day_of_year_360).to_numpy()

    print(f"Calculating PET for {len(catchment_ids)} catchments...", flush=True)
    pet_results = {"date": dates}
    for i, cid in enumerate(catchment_ids):
        cid_int = int(cid)
        pet = calculate_pet_vectorized(
            tas=data["tas"][cid].to_numpy(),
            tasmax=data["tasmax"][cid].to_numpy(),
            tasmin=data["tasmin"][cid].to_numpy(),
            sfcwind=data["sfcWind"][cid].to_numpy(),
            hurs=data["hurs"][cid].to_numpy(),
            rsds=data["rsds"][cid].to_numpy(),
            latitude=latitude_lookup[cid_int],
            elevation=elevation_lookup[cid_int],
            doy_array=doy_array,
            wind_height_m=wind_height_m,
        )
        # PET cannot be physically negative; FAO-56 recommends clipping small
        # negative values (which can occur on cold, calm, humid winter nights).
        pet_results[cid] = np.clip(pet, 0, None)
        if (i + 1) % 100 == 0 or (i + 1) == len(catchment_ids):
            print(f"  ...{i + 1}/{len(catchment_ids)} catchments done", flush=True)

    pet_df = pd.DataFrame(pet_results)
    pet_df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\nWrote PET -> {Path(out_path).resolve()}", flush=True)
    print(f"Output shape: {pet_df.shape}", flush=True)
    print(f"PET value range: {pet_df[catchment_ids].min().min():.3f} to "
          f"{pet_df[catchment_ids].max().max():.3f} mm/day", flush=True)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Calculate daily PET (FAO-56 Penman-Monteith) from "
                    "catchment-mean CHESS-SCAPE variables.")
    ap.add_argument("--rcp", required=True, choices=["rcp26", "rcp45", "rcp60", "rcp85"])
    ap.add_argument("--ensemble", required=True, help="Ensemble member, e.g. 01, 04, 06, 15.")
    ap.add_argument("--data-dir", type=Path, default=CHESS_SCAPE_ROOT,
                    help="Directory holding the catchment-mean variable CSVs "
                         "and receiving the PET output.")
    ap.add_argument("--topo-attrs", type=str, default=TOPO_ATTRS_DEFAULT,
                    help="Path to camels_gb_v2_topographic_attributes.csv.")
    ap.add_argument("--wind-height-m", type=float, default=10.0,
                    help="Height (m) of the sfcWind measurement/product; "
                         "set to 2.0 to skip the FAO-56 height adjustment.")
    return ap.parse_args()


def _self_test():
    """Synthetic single-catchment, 30-day self-test: checks the pipeline runs
    end-to-end and PET falls in a physically plausible daily range for
    temperate mid-latitude summer conditions."""
    print("Running self-test (synthetic catchment, no external data)...")
    rng = np.random.default_rng(0)
    n = 30
    doy = np.arange(151, 151 + n)  # early June in the 360-day calendar

    tas = 15 + 3 * rng.standard_normal(n)
    tasmax = tas + 5 + rng.random(n) * 2
    tasmin = tas - 5 - rng.random(n) * 2
    sfcwind = np.clip(3 + rng.standard_normal(n), 0.2, None)
    hurs = np.clip(70 + 10 * rng.standard_normal(n), 20, 100)
    rsds = np.clip(200 + 60 * rng.standard_normal(n), 20, None)

    pet = calculate_pet_vectorized(
        tas=tas, tasmax=tasmax, tasmin=tasmin, sfcwind=sfcwind, hurs=hurs, rsds=rsds,
        latitude=52.0, elevation=100.0, doy_array=doy, wind_height_m=10.0,
    )
    pet = np.clip(pet, 0, None)

    print(f"  PET range: {pet.min():.2f} - {pet.max():.2f} mm/day "
          f"(mean {pet.mean():.2f})")
    assert np.all(pet >= 0), "PET must be non-negative after clipping"
    assert 0.5 < pet.mean() < 8.0, \
        "Mean PET outside plausible range for temperate June conditions"

    # Sanity check: day-of-year parsing for a 360-day calendar date
    assert day_of_year_360("1980-12-01 12:00:00") == 331
    assert day_of_year_360("1980-01-01 12:00:00") == 1

    # Sanity check: wind height adjustment reduces 10m wind toward 2m
    u2 = wind_speed_to_2m(np.array([5.0]), 10.0)
    assert u2[0] < 5.0, "2m-adjusted wind should be lower than 10m wind"
    assert wind_speed_to_2m(np.array([5.0]), 2.0)[0] == 5.0, "no-op at 2m"

    print("Self-test passed.")


def main():
    import sys
    if len(sys.argv) == 1:
        _self_test()
        return
    args = parse_args()
    out_path = args.data_dir / f"{args.rcp}_{args.ensemble}_pet_catchment_means_combined.csv"
    run_pet(args.data_dir, args.rcp, args.ensemble, args.topo_attrs,
            args.wind_height_m, out_path)


if __name__ == "__main__":
    main()
