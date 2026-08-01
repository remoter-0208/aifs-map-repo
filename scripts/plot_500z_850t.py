#!/usr/bin/env python3
"""
ECMWF AIFS-single (data-driven forecast) auto plotter
======================================================

Downloads fields from ECMWF's open-data feed for the AIFS-single model
and draws three weather-models.info-style charts per forecast step:

    1. 2m temperature (shaded)
    2. Mean sea level pressure (contoured) + 6-hour precipitation (shaded)
    3. 500 hPa geopotential height (contoured) + 850 hPa temperature (shaded)

Usage:
    python plot_500z_850t.py --steps 0 6 12 18 24 ... 240 \
        --domain japan --out-dir docs/output

If --date/--time are omitted, the ecmwf-opendata client automatically
resolves the most recent forecast cycle for which data is available.

Notes on the 6-hour precipitation:
    AIFS-single's "tp" (total precipitation) field, like ECMWF's other
    forecast products, is a *cumulative* accumulation from the start of
    the forecast (step 0). To get the amount that fell in the 6 hours
    ending at `step`, we also fetch `tp` at `step - 6` and difference
    the two. Values are cached across steps so consecutive 6-hourly
    steps don't trigger duplicate downloads. Step 0 has no meaningful
    precipitation and is skipped for that panel.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from ecmwf.opendata import Client

MODEL = "aifs-single"
STREAM = "oper"
TYPE = "fc"

# lon_min, lon_max, lat_min, lat_max
DOMAINS = {
    "japan": (110, 162, 20, 50),
    "japan_wide": (100, 170, 15, 55),
    "global": (-180, 180, -90, 90),
}

# ---------------------------------------------------------------------
# 850 hPa temperature (degC) fill palette (weather-models.info style)
# ---------------------------------------------------------------------
T850_LEVELS = np.arange(-15, 30, 3)
T850_COLORS = [
    "#ffffff", "#d9d9d9", "#bfbfbf", "#8fa6ff", "#4d79ff",
    "#00b4ff", "#00e0c0", "#00c000", "#ffff00", "#ffb400",
    "#ff7f00", "#ff0000", "#c00000", "#ff33cc", "#a000c0",
]
Z_INTERVAL = 60  # 500 hPa geopotential height contour spacing (m)

# ---------------------------------------------------------------------
# 2 m temperature (degC) fill palette — wider range than 850 hPa T
# ---------------------------------------------------------------------
T2M_LEVELS = np.arange(-30, 45, 5)
T2M_COLORS = [
    "#8b00ff", "#4b0082", "#0000cd", "#0080ff", "#00bfff",
    "#00e0c0", "#00c000", "#7fff00", "#ffff00", "#ffb400",
    "#ff7f00", "#ff3300", "#c00000", "#800000",
]

# ---------------------------------------------------------------------
# 6-hour precipitation (mm) fill palette
# ---------------------------------------------------------------------
PRECIP_LEVELS = [0.1, 1, 2, 4, 6, 10, 15, 20, 30, 50, 70, 100]
PRECIP_COLORS = [
    "#c6ffff", "#9fefff", "#63c6ff", "#3f9bff", "#2f6fff",
    "#2f2fff", "#8f2fff", "#ff2fff", "#ff2f8f", "#ff2f2f", "#ff8f2f",
]
MSLP_INTERVAL = 4  # hPa


def build_cmap(levels, colors):
    n = len(levels) - 1
    cmap = LinearSegmentedColormap.from_list("cmap", colors[:n], N=n)
    norm = BoundaryNorm(levels, cmap.N)
    return cmap, norm


# =======================================================================
# Fetching
# =======================================================================

def fetch_pl(step: int, date, time, workdir: Path) -> tuple[Path, "datetime"]:
    """Download gh@500 and t@850 for a single step."""
    client = Client(source="ecmwf", model=MODEL)
    target = workdir / f"aifs_pl_{step:03d}.grib2"
    request = dict(
        stream=STREAM,
        type=TYPE,
        step=step,
        param=["gh", "t"],
        levelist=[500, 850],
        target=str(target),
    )
    if date is not None:
        request["date"] = date
    if time is not None:
        request["time"] = time

    result = client.retrieve(**request)
    print(f"[step {step:03d}] retrieved pl cycle {result.datetime}", file=sys.stderr)
    return target, result.datetime


def fetch_sfc(step: int, date, time, workdir: Path, params=("2t", "msl", "tp")) -> tuple[Path, "datetime"]:
    """Download surface fields (2m temp, MSLP, total precip) for a single step."""
    client = Client(source="ecmwf", model=MODEL)
    target = workdir / f"aifs_sfc_{step:03d}.grib2"
    request = dict(
        stream=STREAM,
        type=TYPE,
        step=step,
        param=list(params),
        target=str(target),
    )
    if date is not None:
        request["date"] = date
    if time is not None:
        request["time"] = time

    result = client.retrieve(**request)
    print(f"[step {step:03d}] retrieved sfc cycle {result.datetime}", file=sys.stderr)
    return target, result.datetime


# =======================================================================
# Loading
# =======================================================================

def load_pl(grib_path: Path):
    gh500 = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": "gh", "level": 500}},
    )["gh"]
    t850 = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": "t", "level": 850}},
    )["t"]
    return gh500, t850 - 273.15  # K -> degC


def load_t2m(grib_path: Path):
    ds = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "heightAboveGround", "level": 2}
        },
    )
    return ds["t2m"] - 273.15  # K -> degC


def load_msl(grib_path: Path):
    ds = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": "msl"}},
    )
    return ds["msl"] / 100.0  # Pa -> hPa


def load_tp(grib_path: Path):
    """Cumulative total precipitation (m from forecast start) -> mm."""
    ds = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": "tp"}},
    )
    return ds["tp"]   # m -> mm


# =======================================================================
# Plotting helpers
# =======================================================================

def _new_axes(domain: str):
    lon_min, lon_max, lat_min, lat_max = DOMAINS[domain]
    fig = plt.figure(figsize=(11, 8.5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    return fig, ax


def _finish(fig, ax, cf, cbar_label, title, out_path: Path):
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.97)
    cbar = fig.colorbar(cf, ax=ax, orientation="vertical", pad=0.02, shrink=0.85)
    cbar.set_label(cbar_label, fontsize=11)
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.07, top=0.88)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}", file=sys.stderr)


def plot_t2m(t2m, init_dt, step: int, domain: str, out_path: Path):
    cmap, norm = build_cmap(T2M_LEVELS, T2M_COLORS)
    fig, ax = _new_axes(domain)

    cf = ax.contourf(
        t2m.longitude, t2m.latitude, t2m,
        levels=T2M_LEVELS, cmap=cmap, norm=norm, extend="both",
        transform=ccrs.PlateCarree(),
    )

    valid_dt = init_dt + timedelta(hours=step)
    title = (
        "AIFS-single 2m Temperature\n"
        f"Init: {init_dt:%Y-%m-%d %HUTC}    FT={step:03d}h    "
        f"Valid: {valid_dt:%Y-%m-%d %HUTC}"
    )
    _finish(fig, ax, cf, "2m Temperature (°C)", title, out_path)


def plot_mslp_precip(msl, precip6h, init_dt, step: int, domain: str, out_path: Path):
    cmap, norm = build_cmap(PRECIP_LEVELS, PRECIP_COLORS)
    fig, ax = _new_axes(domain)

    cf = ax.contourf(
        precip6h.longitude, precip6h.latitude, precip6h,
        levels=PRECIP_LEVELS, cmap=cmap, norm=norm, extend="max",
        transform=ccrs.PlateCarree(),
    )

    z_levels = np.arange(
        np.floor(msl.min().item() / MSLP_INTERVAL) * MSLP_INTERVAL,
        np.ceil(msl.max().item() / MSLP_INTERVAL) * MSLP_INTERVAL + MSLP_INTERVAL,
        MSLP_INTERVAL,
    )
    cs = ax.contour(
        msl.longitude, msl.latitude, msl,
        levels=z_levels, colors="black", linewidths=1.0,
        transform=ccrs.PlateCarree(),
    )
    ax.clabel(cs, fmt="%d", fontsize=8, inline=True)

    valid_dt = init_dt + timedelta(hours=step)
    title = (
        "AIFS-single MSLP + 6h Precipitation\n"
        f"Init: {init_dt:%Y-%m-%d %HUTC}    FT={step:03d}h    "
        f"Valid: {valid_dt:%Y-%m-%d %HUTC}"
    )
    _finish(fig, ax, cf, "6h Precipitation (mm)", title, out_path)


def plot_500z_850t(gh500, t850c, init_dt, step: int, domain: str, out_path: Path):
    cmap, norm = build_cmap(T850_LEVELS, T850_COLORS)
    fig, ax = _new_axes(domain)

    cf = ax.contourf(
        t850c.longitude, t850c.latitude, t850c,
        levels=T850_LEVELS, cmap=cmap, norm=norm, extend="both",
        transform=ccrs.PlateCarree(),
    )

    z_levels = np.arange(
        np.floor(gh500.min().item() / Z_INTERVAL) * Z_INTERVAL,
        np.ceil(gh500.max().item() / Z_INTERVAL) * Z_INTERVAL + Z_INTERVAL,
        Z_INTERVAL,
    )
    cs = ax.contour(
        gh500.longitude, gh500.latitude, gh500,
        levels=z_levels, colors="black", linewidths=1.0,
        transform=ccrs.PlateCarree(),
    )
    ax.clabel(cs, fmt="%d", fontsize=8, inline=True)

    valid_dt = init_dt + timedelta(hours=step)
    title = (
        "AIFS-single 500Z + 850T\n"
        f"Init: {init_dt:%Y-%m-%d %HUTC}    FT={step:03d}h    "
        f"Valid: {valid_dt:%Y-%m-%d %HUTC}"
    )
    _finish(fig, ax, cf, "850 hPa Temperature (°C)", title, out_path)


# =======================================================================
# Main
# =======================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--domain", choices=list(DOMAINS), default="japan")
    ap.add_argument("--out-dir", type=Path, default=Path("docs/output"))
    ap.add_argument("--date", default=None, help="e.g. 2026-07-30 (default: latest)")
    ap.add_argument("--time", type=int, default=None, help="0/6/12/18 (default: latest)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # cache of loaded tp fields (mm cumulative), keyed by step, so that
    # consecutive 6-hourly steps don't re-download the previous step's data
    tp_cache: dict[int, xr.DataArray] = {}

    def get_tp(step: int, tmp: Path, date, time) -> xr.DataArray:
        if step not in tp_cache:
            sfc_path, _ = fetch_sfc(step, date, time, tmp, params=("tp",))
            tp_cache[step] = load_tp(sfc_path)
        return tp_cache[step]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for step in args.steps:
            # --- 500Z + 850T ---
            pl_path, init_dt = fetch_pl(step, args.date, args.time, tmp)
            gh500, t850c = load_pl(pl_path)

            dated = args.out_dir / f"{init_dt:%Y%m%d%H}_500z850t_{step:03d}.png"
            latest = args.out_dir / f"latest_{step:03d}.png"  # kept for backward-compat
            plot_500z_850t(gh500, t850c, init_dt, step, args.domain, dated)
            shutil.copyfile(dated, latest)
            dated.unlink()

            # --- surface fields: 2t, msl, tp ---
            sfc_path, sfc_init_dt = fetch_sfc(step, args.date, args.time, tmp)
            t2m = load_t2m(sfc_path)
            msl = load_msl(sfc_path)
            tp_cache[step] = load_tp(sfc_path)  # reuse instead of re-downloading

            # 1) 2m temperature
            dated = args.out_dir / f"{sfc_init_dt:%Y%m%d%H}_t2m_{step:03d}.png"
            latest = args.out_dir / f"latest_t2m_{step:03d}.png"
            plot_t2m(t2m, sfc_init_dt, step, args.domain, dated)
            shutil.copyfile(dated, latest)
            dated.unlink()

            # 2) MSLP + 6h precipitation
            if step == 0:
                print("[step 000] skipping precipitation panel (no accumulation yet)", file=sys.stderr)
            else:
                prev_step = max(step - 6, 0)
                tp_now = tp_cache[step]
                tp_prev = get_tp(prev_step, tmp, args.date, args.time)
                precip6h = (tp_now - tp_prev).clip(min=0)

                dated = args.out_dir / f"{sfc_init_dt:%Y%m%d%H}_mslp_precip_{step:03d}.png"
                latest = args.out_dir / f"latest_mslp_precip_{step:03d}.png"
                plot_mslp_precip(msl, precip6h, sfc_init_dt, step, args.domain, dated)
                shutil.copyfile(dated, latest)
                dated.unlink()


if __name__ == "__main__":
    main()
