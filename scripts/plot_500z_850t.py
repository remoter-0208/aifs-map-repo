#!/usr/bin/env python3
"""
ECMWF AIFS-single (data-driven forecast) auto plotter
======================================================

Downloads 500 hPa geopotential height (gh) and 850 hPa temperature (t)
from ECMWF's open-data feed for the AIFS-single model and draws a
weather-models.info-style chart (850 hPa T shaded, 500 hPa Z contoured)
for one or more forecast steps.

Usage:
    python plot_500z_850t.py --steps 24 48 72 96 120 144 168 192 216 240 \
        --domain japan --out-dir docs/output

If --date/--time are omitted, the ecmwf-opendata client automatically
resolves the most recent forecast cycle for which data is available.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
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

# 850 hPa temperature (degC) fill palette, tuned to resemble the
# weather-models.info 850hPa T scale.
T_LEVELS = np.arange(-32, 30, 4)  # -32 .. 28 step 4
T_COLORS = [
    "#ffffff", "#d9d9d9", "#bfbfbf", "#8fa6ff", "#4d79ff",  # -32..-12
    "#00b4ff", "#00e0c0", "#00c000", "#ffff00", "#ffb400",  # -8..12
    "#ff7f00", "#ff0000", "#c00000", "#ff33cc", "#a000c0",  # 16..28
]

# 500 hPa geopotential height contour interval (metres), classic 60 m spacing
Z_INTERVAL = 60


def build_cmap():
    n = len(T_LEVELS) - 1
    colors = T_COLORS[:n]
    cmap = LinearSegmentedColormap.from_list("t850", colors, N=n)
    norm = BoundaryNorm(T_LEVELS, cmap.N)
    return cmap, norm


def fetch(step: int, date=None, time=None, workdir: Path = Path(".")) -> Path:
    """Download gh@500 and t@850 for a single step into a GRIB2 file."""
    client = Client(source="ecmwf", model=MODEL)
    target = workdir / f"aifs_{step:03d}.grib2"
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
    print(f"[step {step:03d}] retrieved cycle {result.datetime}", file=sys.stderr)
    return target, result.datetime


def load(grib_path: Path):
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


def plot(gh500, t850c, init_dt, step: int, domain: str, out_path: Path):
    lon_min, lon_max, lat_min, lat_max = DOMAINS[domain]
    cmap, norm = build_cmap()

    fig = plt.figure(figsize=(11, 8.5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    cf = ax.contourf(
        t850c.longitude, t850c.latitude, t850c,
        levels=T_LEVELS, cmap=cmap, norm=norm,
        extend="both", transform=ccrs.PlateCarree(),
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

    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    valid_dt = init_dt + __import__("datetime").timedelta(hours=step)
    ax.set_title(
        f"ECMWF AIFS-single  Init: {init_dt:%Y-%m-%d %HZ}  "
        f"FT{step:03d}h  Valid: {valid_dt:%m-%d %HZ}\n500hPa gh & 850hPa T",
        fontsize=11,
    )

    cbar = fig.colorbar(cf, ax=ax, orientation="vertical", pad=0.02, shrink=0.85)
    cbar.set_label("850hPa T (\u00b0C)")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--domain", choices=list(DOMAINS), default="japan")
    ap.add_argument("--out-dir", type=Path, default=Path("docs/output"))
    ap.add_argument("--date", default=None, help="e.g. 2026-07-30 (default: latest)")
    ap.add_argument("--time", type=int, default=None, help="0/6/12/18 (default: latest)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for step in args.steps:
            grib_path, init_dt = fetch(step, args.date, args.time, tmp)
            gh500, t850c = load(grib_path)
            dated_name = args.out_dir / f"{init_dt:%Y%m%d%H}_{step:03d}.png"
            latest_name = args.out_dir / f"latest_{step:03d}.png"
            plot(gh500, t850c, init_dt, step, args.domain, dated_name)
            # overwrite a fixed-name copy so the static page always
            # points at the newest run without needing directory listing
            import shutil
            shutil.copyfile(dated_name, latest_name)
            dated_name.unlink()  # don't accumulate history in the repo


if __name__ == "__main__":
    main()
