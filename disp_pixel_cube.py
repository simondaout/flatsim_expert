#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################################
# Author        : Simon DAOUT (CRPG-ENSG)
# Adapted from invers_disp_pixel.py: same pixel-vs-reference-pixel plotting
# idea, but reading directly from a FLATSIM CNES_DTs_geo_*.tiff cube (one
# GeoTIFF band per date, dates taken from the accompanying list_images.txt)
# instead of the raw depl_cumule BIP binary + lect.in. No basis-function
# inversion / decomposition here (see invers_disp_pixel.py / invers_temp.py
# for that) — this is the raw, per-pixel double-difference time series:
#     disp(t) = [cube(line, column, t) - cube(line, column, imref)]
#             - [cube(lineref, columnref, t) - cube(lineref, columnref, imref)]
################################################################################

"""
disp_pixel_cube.py
-------------------
Plot the LOS displacement time series at one pixel of a FLATSIM
CNES_DTs_geo_*.tiff cube, relative to a reference pixel.

Usage:
    python disp_pixel_cube.py --cube=<path> --line=<value> --column=<value> \\
        --lineref=<value> --columnref=<value> [--list_images=<path>] \\
        [--imref=<value>] [--scale=<value>] [--windowsize=<value>] \\
        [--windowrefsize=<value>] [--linear=yes/no] [--dateslim=<min,max>] \\
        [--plot_dateslim=<min,max>] [--bounds=<ymin,ymax>] [--color=<value>] \\
        [--fillstyle=<value>] [--out=<path>] [--no-display]

Example (D019_sud_2021-2025):
    python disp_pixel_cube.py \\
        --cube=data/D019_sud_2021-2025/NSBAS_TS-PKG_S1_TIBET-HIM-D019SUD-VV-2021-2025_IW123_2021-06-08_2025-12-20/CNES_DTs_geo_8rlks.tiff \\
        --line=3289 --column=2271 --lineref=3268 --columnref=2263
"""

import os
import sys
import argparse
import numpy as np

# ── headless-safe plotting (see check_results.py) ───────────────────────────
import matplotlib
if sys.platform != "darwin" and not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

from osgeo import gdal, osr

# Sentinel-1 C-band wavelength: 5.5465763 cm. |factor| = lambda/(4*pi).
# Signed (default): positive displacement = motion TOWARDS the satellite,
# same convention as the MV-LOS / lin_coeff mm(/yr) GeoTIFFs written by
# check_results.py. Pass --scale=1 to keep raw radians instead.
LOS_MM_PER_RAD = -4.4138249


def read_list_images(path):
    """
    Read a FLATSIM list_images.txt (same 6-column layout read elsewhere in
    this project, e.g. invers_disp_pixel.py / invers_temp.py):
      col0: image number (unused here)   col1: date YYYYMMDD
      col2: Doppler freq (unused)        col3: decimal date
      col4: unused                       col5: perpendicular baseline
    Returns (idates [YYYYMMDD ints], dates [decimal years]).
    """
    data = np.loadtxt(path, comments="#", usecols=(1, 3), unpack=True)
    idates = data[0].astype(int)
    dates = data[1]
    return idates, dates


def find_list_images(cube_path):
    """list_images.txt is expected next to the cube (same TS directory)."""
    cand = os.path.join(os.path.dirname(cube_path), "list_images.txt")
    return cand if os.path.isfile(cand) else None


def read_pixel_series(ds, line, col, w=0):
    """
    Read the full band stack at one pixel (or the median over a
    (2w+1)x(2w+1) window centered on it, if w > 0).
    Returns a 1-D array of length nbands, NaN where nodata.
    """
    nbands = ds.RasterCount
    if w > 0:
        arr = ds.ReadAsArray(
            xoff=max(col - w, 0), yoff=max(line - w, 0),
            xsize=2 * w + 1, ysize=2 * w + 1,
        ).astype(np.float64)  # (nbands, ny, nx)
        series = np.nanmedian(arr, axis=(1, 2))
    else:
        arr = ds.ReadAsArray(xoff=col, yoff=line, xsize=1, ysize=1).astype(np.float64)
        series = arr.reshape(nbands)

    nd = ds.GetRasterBand(1).GetNoDataValue()
    if nd is not None:
        series[series == nd] = np.nan
    return series


def pixel_to_lonlat(gt, proj_wkt, col, row):
    """
    Convert a (col, row) pixel *center* to (lon, lat) in WGS84 degrees,
    using the cube's own GDAL geotransform + projection (GeoTransform
    convention: x = gt[0] + col*gt[1] + row*gt[2], idem for y/gt[3:6]).
    If the cube has no projection info, (col,row) is assumed to already be
    geographic (lon, lat) - i.e. the geotransform's units are degrees.
    """
    xc, yc = col + 0.5, row + 0.5
    x = gt[0] + xc * gt[1] + yc * gt[2]
    y = gt[3] + xc * gt[4] + yc * gt[5]
    if not proj_wkt:
        return x, y
    src = osr.SpatialReference()
    src.ImportFromWkt(proj_wkt)
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(4326)
    for srs in (src, dst):
        try:
            srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        except AttributeError:
            pass
    if src.IsSame(dst):
        return x, y
    ct = osr.CoordinateTransformation(src, dst)
    lon, lat, _ = ct.TransformPoint(x, y)
    return lon, lat


def parse_dateslim(s, name):
    """Parse a 'Datemin,Datemax' (YYYYMMDD,YYYYMMDD) CLI value."""
    try:
        a, b = s.split(",")
        return int(a), int(b)
    except Exception:
        sys.exit(f"--{name} must be Datemin,Datemax (YYYYMMDD,YYYYMMDD), got: {s!r}")


def parse_bounds(s):
    """Parse a 'yMin,yMax' CLI value."""
    try:
        a, b = s.split(",")
        return float(a), float(b)
    except Exception:
        sys.exit(f"--bounds must be yMin,yMax, got: {s!r}")


def fit_linear(t, y):
    """
    Simple weighted-equally least-squares fit y ~ const + rate * t (same
    'reference' + 'linear' basis pair as invers_disp_pixel.py --linear=yes,
    without the rest of its inversion machinery). NaNs in y are dropped.
    Returns (const, rate, rate_sigma, n_used) - rate_sigma is NaN if there
    are not enough points to estimate it.
    """
    valid = ~np.isnan(y)
    t_v, y_v = t[valid], y[valid]
    n = len(t_v)
    if n < 2:
        return np.nan, np.nan, np.nan, n
    G = np.vstack([np.ones_like(t_v), t_v]).T
    sol, _, _, _ = np.linalg.lstsq(G, y_v, rcond=None)
    const, rate = sol
    if n > 2:
        resid = y_v - G @ sol
        dof = n - 2
        sigma2 = np.sum(resid ** 2) / dof
        cov = sigma2 * np.linalg.pinv(G.T @ G)
        rate_sigma = np.sqrt(cov[1, 1])
    else:
        rate_sigma = np.nan
    return const, rate, rate_sigma, n


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cube", required=True,
                    help="Path to a FLATSIM CNES_DTs_geo_*.tiff (or _radar_) cube")
    p.add_argument("--list_images", default=None,
                    help="Path to list_images.txt (default: next to --cube)")
    p.add_argument("--line", type=int, required=True, help="Pixel line (row, 0-based)")
    p.add_argument("--column", type=int, required=True, help="Pixel column (0-based)")
    p.add_argument("--lineref", type=int, required=True, help="Reference pixel line")
    p.add_argument("--columnref", type=int, required=True, help="Reference pixel column")
    p.add_argument("--imref", type=int, default=1,
                    help="Reference image number, 1-based (band set to 0 at "
                         "that date, for both pixels) [default: 1]")
    p.add_argument("--scale", type=float, default=LOS_MM_PER_RAD,
                    help=f"Multiply the raw (rad) cube values by this factor. "
                         f"[default: {LOS_MM_PER_RAD} -> mm, +towards satellite; "
                         f"pass --scale=1 to keep raw radians]")
    p.add_argument("--windowsize", type=int, default=0,
                    help="Median over a (2w+1)x(2w+1) window around the pixel [default: 0]")
    p.add_argument("--windowrefsize", type=int, default=None,
                    help="Same, around the reference pixel [default: --windowsize]")
    p.add_argument("--linear", default="yes",
                    help="Estimate and plot a linear rate through the "
                         "(referenced) time series, as in invers_disp_pixel.py "
                         "--linear=yes [default: yes]")
    p.add_argument("--dateslim", default=None,
                    help="Datemin,Datemax (YYYYMMDD,YYYYMMDD) - restrict which "
                         "dates are plotted AND used for the linear fit "
                         "(the reference image/pixel are unaffected) [default: all]")
    p.add_argument("--plot_dateslim", default=None,
                    help="Datemin,Datemax (YYYYMMDD,YYYYMMDD) - x-axis display "
                         "range only, does not affect the data or the fit "
                         "[default: auto, full data range]")
    p.add_argument("--bounds", default=None,
                    help="yMin,yMax for the y-axis [default: auto]")
    p.add_argument("--color", default="blue", help="Marker color [default: blue]")
    p.add_argument("--fillstyle", default="none",
                    help="Marker fill style: none,full,top,bottom,left,right [default: none]")
    p.add_argument("--out", default=None,
                    help="Output figure path [default: Disp_<line>_<column>.pdf "
                         "next to --cube, i.e. the dataset's working directory]")
    p.add_argument("--no-map", action="store_true",
                    help="Do not add the last-date localization map panel")
    p.add_argument("--map-pad", type=int, default=60,
                    help="Half-size (pixels) of the localization map crop around "
                         "the pixel/reference pixel pair [default: 60]")
    p.add_argument("--no-display", action="store_true",
                    help="Do not call plt.show() (batch/headless mode)")
    args = p.parse_args()

    wref = args.windowrefsize if args.windowrefsize is not None else args.windowsize
    imref = args.imref - 1
    unit = "mm" if args.scale != 1 else "rad"

    list_images = args.list_images or find_list_images(args.cube)
    if not list_images:
        sys.exit(f"list_images.txt not found next to {args.cube} "
                  f"- pass --list_images explicitly")
    idates, dates = read_list_images(list_images)

    ds = gdal.Open(args.cube)
    if ds is None:
        sys.exit(f"Cannot open {args.cube}")
    nbands = ds.RasterCount
    if nbands != len(dates):
        sys.exit(f"{args.cube} has {nbands} bands but {list_images} lists "
                  f"{len(dates)} dates - they must match")

    # pixel/reference-pixel positions in WGS84 lon/lat, from the cube's own
    # geotransform + projection (used to label the point on the plot)
    gt, proj_wkt = ds.GetGeoTransform(), ds.GetProjection()
    lon, lat = pixel_to_lonlat(gt, proj_wkt, args.column, args.line)
    lonref, latref = pixel_to_lonlat(gt, proj_wkt, args.columnref, args.lineref)

    print(f"Cube             : {args.cube}  ({nbands} bands, "
          f"{ds.RasterYSize}x{ds.RasterXSize})")
    print(f"Pixel            : line={args.line}, column={args.column}"
          + (f"  (median over {2*args.windowsize+1}x{2*args.windowsize+1})" if args.windowsize else "")
          + f"  ->  lon={lon:.5f}, lat={lat:.5f}")
    print(f"Reference pixel  : line={args.lineref}, column={args.columnref}"
          + (f"  (median over {2*wref+1}x{2*wref+1})" if wref else "")
          + f"  ->  lon={lonref:.5f}, lat={latref:.5f}")
    print(f"Reference image  : {idates[imref]} (band {imref + 1}/{nbands})")
    print(f"Scale            : {args.scale}  (unit: {unit})")

    # decimal-year date of the reference image, captured now (before any
    # --dateslim filtering below) so the linear fit's time origin never
    # shifts even if that date is later filtered out of the plot
    imref_date = dates[imref]

    px    = read_pixel_series(ds, args.line, args.column, w=args.windowsize)
    pxref = read_pixel_series(ds, args.lineref, args.columnref, w=wref)

    # ── localization map: last-date band, cropped around the two pixels ────
    # (captured now, before any --dateslim filtering below, since the map
    # always shows the cube's true last band regardless of --dateslim)
    map_arr = map_extent = last_idate = None
    if not args.no_map:
        last_idate = idates[-1]
        pad = max(args.map_pad, 1)
        lo_r = max(min(args.line, args.lineref) - pad, 0)
        hi_r = min(max(args.line, args.lineref) + pad, ds.RasterYSize)
        lo_c = max(min(args.column, args.columnref) - pad, 0)
        hi_c = min(max(args.column, args.columnref) + pad, ds.RasterXSize)
        last_band = ds.GetRasterBand(nbands)
        map_arr = last_band.ReadAsArray(lo_c, lo_r, hi_c - lo_c, hi_r - lo_r).astype(np.float64)
        nd = last_band.GetNoDataValue()
        if nd is not None:
            map_arr[map_arr == nd] = np.nan
        map_arr *= args.scale
        map_extent = (lo_r, hi_r, lo_c, hi_c)  # row/col offsets of the crop

    del ds

    # temporal referencing (per pixel, band imref -> 0), then spatial
    # double-difference against the reference pixel
    px    = px    - px[imref]
    pxref = pxref - pxref[imref]
    disp = (px - pxref) * args.scale

    n_nan = int(np.isnan(disp).sum())
    if n_nan:
        print(f"  WARNING: {n_nan}/{nbands} dates are NaN (nodata) at one of the two pixels.")

    # ── --dateslim: restrict which dates are plotted AND fit ────────────────
    # (applied after temporal referencing above, so the reference image/pixel
    # stay unaffected even if imref itself falls outside the requested range)
    if args.dateslim:
        dmin, dmax = parse_dateslim(args.dateslim, "dateslim")
        mask = (idates >= dmin) & (idates <= dmax)
        n_kept = int(mask.sum())
        if n_kept < 1:
            sys.exit(f"--dateslim={args.dateslim} excludes all {nbands} dates")
        print(f"Dateslim         : {dmin}-{dmax}  ({n_kept}/{nbands} dates kept)")
        idates, dates, disp = idates[mask], dates[mask], disp[mask]

    # ── linear rate (invers_disp_pixel.py --linear=yes) ─────────────────────
    do_linear = args.linear == "yes"
    if do_linear:
        t_years = dates - imref_date
        const, rate, rate_sigma, n_used = fit_linear(t_years, disp)
        if np.isnan(rate):
            print(f"  WARNING: not enough valid points ({n_used}) to fit a linear rate.")
            do_linear = False
        elif np.isnan(rate_sigma):
            print(f"Linear rate      : {rate:.2f} {unit}/yr  (n={n_used}, no uncertainty: n=2)")
        else:
            print(f"Linear rate      : {rate:.2f} +/- {rate_sigma:.2f} {unit}/yr  (n={n_used})")

    # ── plot ──────────────────────────────────────────────────────────────
    x = [mdates.date2num(datetime.strptime(str(d), "%Y%m%d")) for d in idates]

    if map_arr is not None:
        fig, (ax, ax_map) = plt.subplots(
            1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [2.3, 1]})
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m/%d"))
    ax.plot(x, disp, "o", color=args.color, fillstyle=args.fillstyle, markersize=4,
            label=f"lon={lon:.5f}°, lat={lat:.5f}°")
    if do_linear:
        x_fit = [x[0], x[-1]]
        y_fit = [const + rate * t_years[0], const + rate * t_years[-1]]
        rate_label = (f"Rate: {rate:.2f} +/- {rate_sigma:.2f} {unit}/yr" if not np.isnan(rate_sigma)
                      else f"Rate: {rate:.2f} {unit}/yr")
        ax.plot(x_fit, y_fit, "-r", label=rate_label)
    ax.axhline(0, color="grey", lw=0.7, alpha=0.6)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"LOS displacement ({unit})"
                  + (", +towards satellite" if args.scale == LOS_MM_PER_RAD else ""))
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.3)

    # ── --bounds: y-axis limits ──────────────────────────────────────────
    if args.bounds:
        ymin, ymax = parse_bounds(args.bounds)
        ax.set_ylim(ymin, ymax)

    # ── --plot_dateslim: x-axis display range only (no data filtering) ──
    if args.plot_dateslim:
        d0, d1 = parse_dateslim(args.plot_dateslim, "plot_dateslim")
        ax.set_xlim(mdates.date2num(datetime.strptime(str(d0), "%Y%m%d")),
                    mdates.date2num(datetime.strptime(str(d1), "%Y%m%d")))

    # ── localization map panel ───────────────────────────────────────────
    if map_arr is not None:
        lo_r, hi_r, lo_c, hi_c = map_extent
        vmin, vmax = np.nanpercentile(map_arr, [2, 98])
        im = ax_map.imshow(map_arr, cmap="Greys_r", vmin=vmin, vmax=vmax,
                            extent=(lo_c, hi_c, hi_r, lo_r))  # keep row axis pointing down

        # pixel window(s), if a median window was used
        if args.windowsize:
            ax_map.add_patch(plt.Rectangle(
                (args.column - args.windowsize - 0.5, args.line - args.windowsize - 0.5),
                2 * args.windowsize + 1, 2 * args.windowsize + 1,
                edgecolor="red", facecolor="none", lw=1))
        if wref:
            ax_map.add_patch(plt.Rectangle(
                (args.columnref - wref - 0.5, args.lineref - wref - 0.5),
                2 * wref + 1, 2 * wref + 1,
                edgecolor="cyan", facecolor="none", lw=1))

        ax_map.plot(args.column, args.line, "x", color="red", markersize=10,
                    markeredgewidth=2, label="pixel")
        ax_map.plot(args.columnref, args.lineref, "+", color="cyan", markersize=10,
                    markeredgewidth=2, label="reference")
        ax_map.set_title(f"Dernière date: {last_idate}\n(bande {nbands}/{nbands})",
                          fontsize="small")
        ax_map.set_xlabel("column")
        ax_map.set_ylabel("line")
        ax_map.legend(loc="best", fontsize="x-small")
        fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04, label=unit)

    fig.autofmt_xdate()
    fig.tight_layout()

    # default: save next to the cube (the dataset's working directory), not
    # wherever the script happens to be launched from
    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.cube)),
        f"Disp_{args.line}_{args.column}.pdf")
    fig.savefig(out_path)
    print(f"  -> {out_path}")

    if not args.no_display:
        plt.show()


if __name__ == "__main__":
    main()
