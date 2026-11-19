#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################################
# Author        : Simon DAOUT (CRPG-ENSG)
# Adapted from invers_disp_pixel.py: same pixel-vs-reference-pixel plotting
# idea, but reading directly from a FLATSIM CNES_DTs_geo_*.tiff cube (one
# GeoTIFF band per date, dates taken from the cube metadata, or from list_images.txt)
# instead of the raw depl_cumule BIP binary + lect.in. No basis-function
# inversion / decomposition here (see invers_disp_pixel.py / invers_temp.py
# for that) — this is the raw, per-pixel double-difference time series:
#     disp(t) = [cube(line, column, t) - cube(line, column, imref)]
#             - [cube(lineref, columnref, t) - cube(lineref, columnref, imref)]
################################################################################

"""
disp_pixel_cube.py
-------------------
Plot the LOS displacement time series at one or several pixels of a FLATSIM
CNES_DTs_geo_*.tiff cube, relative to a reference pixel.
Several pixels: give comma-separated lists to --line and --column (same length).

Usage:
    python disp_pixel_cube.py --cube=<path> --line=<values> --column=<values> \\
        --lineref=<value> --columnref=<value> [--list_images=<path>] \\
        [--imref=<value>] [--scale=<value>] [--windowsize=<value>] \\
        [--windowrefsize=<value>] [--linear=yes/no] [--dateslim=<min,max>] \\
        [--plot_dateslim=<min,max>] [--bounds=<ymin,ymax>] [--color=<value>] \\
        [--fillstyle=<value>] [--sharey] [--out=<path>] [--no-display]

Example (D019_sud_2021-2025):
    python disp_pixel_cube.py \\
        --cube=data/D019_sud_2021-2025/NSBAS_TS-PKG_S1_TIBET-HIM-D019SUD-VV-2021-2025_IW123_2021-06-08_2025-12-20/CNES_DTs_geo_8rlks.tiff \\
        --line=3289 --column=2271 --lineref=3268 --columnref=2263

Example, several pixels (one panel per pixel, all shown on the map):
    python disp_pixel_cube.py --cube=CNES_DTs_geo_8rlks.tiff --list_images=list_images.txt \\
        --line=3485,3529,3540 --column=2013,1978,1910 --lineref=355 --columnref=1955 \\
        --windowsize=2 --windowrefsize=2
"""

import os
import re
import sys
import warnings
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


def read_dates_from_cube(ds):
    """
    Read the acquisition date of each band from the cube's own metadata
    (GDAL band descriptions, e.g. "Bande tserie: 20210608", stored in the
    GeoTIFF or in its .aux.xml sidecar, which must stay next to the cube).
    Returns (idates [YYYYMMDD ints], dates [decimal years], same convention as
    the 'decimal date' column of list_images.txt: year + doy/365.1), or None
    if any band has no YYYYMMDD date in its description.
    """
    idates = []
    for b in range(1, ds.RasterCount + 1):
        m = re.search(r"(?<!\d)(\d{8})(?!\d)", ds.GetRasterBand(b).GetDescription() or "")
        if not m:
            return None
        try:
            datetime.strptime(m.group(1), "%Y%m%d")
        except ValueError:
            return None
        idates.append(int(m.group(1)))
    idates = np.array(idates, dtype=int)
    dates = np.array([
        d.year + float(d.strftime("%j")) / 365.1
        for d in (datetime.strptime(str(i), "%Y%m%d") for i in idates)])
    return idates, dates


def find_list_images(cube_path):
    """list_images.txt is expected next to the cube (same TS directory)."""
    cand = os.path.join(os.path.dirname(cube_path), "list_images.txt")
    return cand if os.path.isfile(cand) else None


def detect_zero_reference(ds, imref, step=8):
    """
    Some cubes are already referenced to image imref (that band is then 0
    everywhere, and 0 is a legitimate value there); in others (e.g. raw
    FLATSIM cubes with nodata=0) the band holds real values and 0 means
    "no data". Tell them apart on a decimated copy of the cube: True if
    band imref is (almost) never non-zero where the cube has data.
    """
    nb = ds.RasterCount
    other = nb - 1 if imref != nb - 1 else 0
    if other == imref:
        return False
    bx, by = max(ds.RasterXSize // step, 1), max(ds.RasterYSize // step, 1)
    ref = ds.GetRasterBand(imref + 1).ReadAsArray(buf_xsize=bx, buf_ysize=by).astype(np.float64)
    oth = ds.GetRasterBand(other + 1).ReadAsArray(buf_xsize=bx, buf_ysize=by).astype(np.float64)
    valid = np.isfinite(oth) & (oth != 0) & (oth != 9990) & (oth != 9999)
    if valid.sum() == 0:
        return False
    return float(np.mean(np.isfinite(ref[valid]) & (ref[valid] != 0))) < 1e-3


def read_window(ds, line, col, w=0, imref=0, zero_ref=False):
    """
    Read the band stack in the (2w+1)x(2w+1) window centered on a pixel
    (clipped at the cube edges) and mask invalid values (nodata, 9990, 9999,
    NaN) with NaN. Returns an array (nbands, ny, nx).
    zero_ref: band imref is a zero reference band (see detect_zero_reference),
    so 0 is valid there unless the whole pixel is masked.
    """
    nbands = ds.RasterCount
    x0, y0 = max(col - w, 0), max(line - w, 0)
    x1, y1 = min(col + w + 1, ds.RasterXSize), min(line + w + 1, ds.RasterYSize)
    arr = ds.ReadAsArray(xoff=x0, yoff=y0, xsize=x1 - x0, ysize=y1 - y0).astype(np.float64)
    arr = arr.reshape(nbands, y1 - y0, x1 - x0)

    bad = (arr == 9990) | (arr == 9999) | ~np.isfinite(arr)
    nd = ds.GetRasterBand(1).GetNoDataValue()
    if nd is not None and np.isfinite(nd):
        zero_nd = (arr == nd)
        if zero_ref:
            zero_nd[imref] = False          # 0 is a valid value in the reference band
        bad |= zero_nd
    arr[bad] = np.nan
    if zero_ref and nbands > 1:
        # pixels masked in every other band (e.g. lakes) are invalid in the reference band too
        others = np.arange(nbands) != imref
        arr[imref][np.all(np.isnan(arr[others]), axis=0)] = np.nan
    return arr


def series_from_window(arr, band, stat="mean"):
    """
    Reference each pixel of the window to its own value at `band` (set to 0
    there, as in invers_disp_pixel.py; a pixel NaN at `band` just drops out),
    then average the window ignoring NaN (nanmean, or nanmedian).
    Returns a 1-D array of length nbands (NaN where no valid pixel is left).
    """
    arr = arr - arr[band][None, :, :]
    flat = arr.reshape(arr.shape[0], -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slices
        return np.nanmean(flat, axis=1) if stat == "mean" else np.nanmedian(flat, axis=1)


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


def parse_int_list(s, name):
    """Parse a single value or a comma/space separated list of integers."""
    try:
        vals = [int(v) for v in str(s).replace(",", " ").split()]
        if not vals:
            raise ValueError
        return vals
    except ValueError:
        sys.exit(f"--{name} must be an integer or a comma-separated list of integers, got: {s!r}")


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
                    help="Path to list_images.txt. Optional: by default the dates are read "
                         "from the cube metadata (band descriptions); "
                         "list_images.txt next to --cube is only used as a fallback")
    p.add_argument("--line", "--lines", required=True,
                    help="Pixel line (row, 0-based); several pixels: comma-separated "
                         "list, e.g. 3485,3529,3540")
    p.add_argument("--column", "--cols", required=True,
                    help="Pixel column (0-based); several pixels: comma-separated "
                         "list with as many values as --line, e.g. 2013,1978,1910")
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
                    help="Half-size w of the (2w+1)x(2w+1) window around the pixel, averaged ignoring NaN (see --stat) [default: 0]")
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
    p.add_argument("--stat", choices=["mean", "median"], default="mean",
                    help="Statistic over the (2w+1)x(2w+1) window, NaN ignored "
                         "[default: mean]")
    p.add_argument("--cmap", default="coolwarm",
                    help="Matplotlib colormap of the localization map [default: coolwarm]")
    p.add_argument("--sharey", action="store_true",
                    help="With several pixels: same y-axis (ordinates) for all "
                         "panels, so the pixels are directly comparable "
                         "(ignored if --bounds is given)")
    p.add_argument("--color", default="blue", help="Marker color [default: blue]")
    p.add_argument("--fillstyle", default="none",
                    help="Marker fill style: none,full,top,bottom,left,right [default: none]")
    p.add_argument("--out", default=None,
                    help="Output figure path [default: Disp_<line>_<column>.pdf "
                         "(Disp_<line0>_<column0>_n<N>.pdf for N pixels) "
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

    # one or several pixels (comma-separated lists, same length)
    lines = parse_int_list(args.line, "line")
    cols = parse_int_list(args.column, "column")
    if len(lines) != len(cols):
        sys.exit(f"--line ({len(lines)} values) and --column ({len(cols)} values) "
                  f"must have the same number of values")
    npix = len(lines)

    ds = gdal.Open(args.cube)
    if ds is None:
        sys.exit(f"Cannot open {args.cube}")
    nbands = ds.RasterCount

    # dates: --list_images if given, else the cube metadata, else a
    # list_images.txt found next to the cube
    list_images = args.list_images
    if list_images:
        idates, dates = read_list_images(list_images)
        dates_src = list_images
    else:
        from_cube = read_dates_from_cube(ds)
        if from_cube is not None:
            idates, dates = from_cube
            dates_src = "cube metadata (band descriptions)"
        else:
            list_images = find_list_images(args.cube)
            if not list_images:
                sys.exit(f"No date found in the band descriptions of {args.cube} and no "
                          f"list_images.txt next to it - pass --list_images explicitly")
            idates, dates = read_list_images(list_images)
            dates_src = list_images + " (no dates in the cube metadata)"
    if nbands != len(dates):
        sys.exit(f"{args.cube} has {nbands} bands but {dates_src} gives "
                  f"{len(dates)} dates - they must match")

    # pixel/reference-pixel positions in WGS84 lon/lat, from the cube's own
    # geotransform + projection (used to label the points on the plot)
    gt, proj_wkt = ds.GetGeoTransform(), ds.GetProjection()
    lonlats = [pixel_to_lonlat(gt, proj_wkt, c, l) for l, c in zip(lines, cols)]
    lonref, latref = pixel_to_lonlat(gt, proj_wkt, args.columnref, args.lineref)

    print(f"Cube             : {args.cube}  ({nbands} bands, "
          f"{ds.RasterYSize}x{ds.RasterXSize})")
    for k, (l, c, (lon, lat)) in enumerate(zip(lines, cols, lonlats)):
        tag = f"Pixel {k}" if npix > 1 else "Pixel"
        print(f"{tag:<17}: line={l}, column={c}"
              + (f"  ({args.stat} over {2*args.windowsize+1}x{2*args.windowsize+1})" if args.windowsize else "")
              + f"  ->  lon={lon:.5f}, lat={lat:.5f}")
    print(f"Reference pixel  : line={args.lineref}, column={args.columnref}"
          + (f"  ({args.stat} over {2*wref+1}x{2*wref+1})" if wref else "")
          + f"  ->  lon={lonref:.5f}, lat={latref:.5f}")
    print(f"Reference image  : {idates[imref]} (band {imref + 1}/{nbands})")
    print(f"Dates            : {dates_src}")
    print(f"Scale            : {args.scale}  (unit: {unit})")

    # decimal-year date of the reference image, captured now (before any
    # --dateslim filtering below) so the linear fit's time origin never
    # shifts even if that date is later filtered out of the plot
    imref_date = dates[imref]

    zero_ref = detect_zero_reference(ds, imref)
    print(f"Reference band   : band {imref + 1} is "
          + ("a zero reference (0 = valid there)" if zero_ref
             else "regular data (0 = nodata, masked)"))
    warrs = [read_window(ds, l, c, w=args.windowsize, imref=imref, zero_ref=zero_ref)
             for l, c in zip(lines, cols)]
    rarr = read_window(ds, args.lineref, args.columnref, w=wref, imref=imref, zero_ref=zero_ref)
    vref = np.isfinite(rarr).sum(axis=(1, 2))   # valid window pixels of the reference, per date

    # Per pixel: the time series is set to 0 at the reference image imref. If the pixel
    # window (or the reference window) has no valid pixel at that date, fall back on the
    # closest date where both have data (the displacement is then 0 at that date instead;
    # rates are unaffected, only the constant offset changes).
    pxs, refband = [], []
    cache = {}
    for k, warr in enumerate(warrs):
        who = f"pixel {k}" if npix > 1 else "the pixel"
        vpix = np.isfinite(warr).sum(axis=(1, 2))
        ok = np.flatnonzero((vpix > 0) & (vref > 0))
        if len(ok) == 0:
            print(f"  WARNING: {who}: its window and the reference window never have data on "
                  f"the same date -> empty series. Try a larger --windowsize/--windowrefsize "
                  f"or another reference pixel.")
            pxs.append(np.full(nbands, np.nan))
            refband.append(None)
            continue
        bk = imref if (vpix[imref] > 0 and vref[imref] > 0) else int(ok[np.argmin(np.abs(ok - imref))])
        if bk != imref:
            side = "its window" if vpix[imref] == 0 else "the reference window"
            if vpix[imref] == 0 and vref[imref] == 0:
                side = "its window and the reference window"
            print(f"  NOTE: {who}: no valid pixel in {side} at the reference image "
                  f"(band {imref + 1}, {idates[imref]}) -> referenced to the closest common "
                  f"valid date instead: band {bk + 1}, {idates[bk]}.")
        if bk not in cache:
            cache[bk] = series_from_window(rarr, bk, args.stat)
        pxs.append(series_from_window(warr, bk, args.stat) - cache[bk])
        refband.append(bk)

    # ── localization map: last-date band, cropped around all the pixels and
    # the reference pixel ───────────────────────────────────────────────────
    # (captured now, before any --dateslim filtering below, since the map
    # always shows the cube's true last band regardless of --dateslim)
    map_arr = map_extent = last_idate = None
    if not args.no_map:
        last_idate = idates[-1]
        pad = max(args.map_pad, 1)
        all_l = lines + [args.lineref]
        all_c = cols + [args.columnref]
        lo_r = max(min(all_l) - pad, 0)
        hi_r = min(max(all_l) + pad, ds.RasterYSize)
        lo_c = max(min(all_c) - pad, 0)
        hi_c = min(max(all_c) + pad, ds.RasterXSize)
        last_band = ds.GetRasterBand(nbands)
        map_arr = last_band.ReadAsArray(lo_c, lo_r, hi_c - lo_c, hi_r - lo_r).astype(np.float64)
        nd = last_band.GetNoDataValue()
        if nd is not None:
            map_arr[map_arr == nd] = np.nan
        map_arr *= args.scale
        map_extent = (lo_r, hi_r, lo_c, hi_c)  # row/col offsets of the crop

    del ds

    # temporal referencing and spatial double-difference against the reference
    # pixel were done above (per window pixel, at band imref or the fallback band)
    disps = [px * args.scale for px in pxs]

    for k, disp in enumerate(disps):
        n_nan = int(np.isnan(disp).sum())
        if n_nan:
            who = f"pixel {k}" if npix > 1 else "the pixel"
            print(f"  WARNING: {n_nan}/{nbands} dates are NaN (nodata) at {who} or at the reference pixel.")

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
        idates, dates = idates[mask], dates[mask]
        disps = [d[mask] for d in disps]

    # ── linear rate (invers_disp_pixel.py --linear=yes), one per pixel ──────
    do_linear = args.linear == "yes"
    t_years = dates - imref_date
    fits = [None] * npix
    if do_linear:
        for k, disp in enumerate(disps):
            const, rate, rate_sigma, n_used = fit_linear(t_years, disp)
            tag = f"Linear rate {k}" if npix > 1 else "Linear rate"
            if np.isnan(rate):
                print(f"  WARNING: not enough valid points ({n_used}) to fit a linear rate"
                      + (f" at pixel {k}." if npix > 1 else "."))
            elif np.isnan(rate_sigma):
                print(f"{tag:<17}: {rate:.2f} {unit}/yr  (n={n_used}, no uncertainty: n=2)")
                fits[k] = (const, rate, rate_sigma)
            else:
                print(f"{tag:<17}: {rate:.2f} +/- {rate_sigma:.2f} {unit}/yr  (n={n_used})")
                fits[k] = (const, rate, rate_sigma)

    # ── plot ──────────────────────────────────────────────────────────────
    x = [mdates.date2num(datetime.strptime(str(d), "%Y%m%d")) for d in idates]

    # one time-series panel per pixel (stacked), localization map on the right
    ncols_grid = 2 if map_arr is not None else 1
    figh = 4 if npix == 1 else min(2.6 * npix + 1.4, 14)
    fig = plt.figure(figsize=(13 if map_arr is not None else 10, figh))
    gs = fig.add_gridspec(npix, ncols_grid,
                          width_ratios=[2.3, 1] if map_arr is not None else None)
    axes = []
    for k in range(npix):
        axes.append(fig.add_subplot(gs[k, 0], sharex=axes[0] if axes else None,
                                    sharey=axes[0] if (axes and args.sharey) else None))
    ax_map = fig.add_subplot(gs[:, 1]) if map_arr is not None else None

    for k, ax in enumerate(axes):
        lon, lat = lonlats[k]
        disp, fit = disps[k], fits[k]
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m/%d"))
        label = (f"[{k}] line={lines[k]}, col={cols[k]}: lon={lon:.5f}°, lat={lat:.5f}°"
                 if npix > 1 else f"lon={lon:.5f}°, lat={lat:.5f}°")
        ax.plot(x, disp, "o", color=args.color, fillstyle=args.fillstyle, markersize=4,
                label=label)
        if fit is not None:
            const, rate, rate_sigma = fit
            x_fit = [x[0], x[-1]]
            y_fit = [const + rate * t_years[0], const + rate * t_years[-1]]
            rate_label = (f"Rate: {rate:.2f} +/- {rate_sigma:.2f} {unit}/yr" if not np.isnan(rate_sigma)
                          else f"Rate: {rate:.2f} {unit}/yr")
            ax.plot(x_fit, y_fit, "-r", label=rate_label)
        ax.axhline(0, color="grey", lw=0.7, alpha=0.6)
        ax.set_ylabel((f"LOS displacement ({unit})" if npix == 1 else f"LOS disp. ({unit})")
                      + (", +towards satellite" if (args.scale == LOS_MM_PER_RAD and npix == 1) else ""))
        ax.legend(loc="best", fontsize="small")
        ax.grid(True, alpha=0.3)

        # ── --bounds: y-axis limits ──────────────────────────────────────
        if args.bounds:
            ymin, ymax = parse_bounds(args.bounds)
            ax.set_ylim(ymin, ymax)

        # ── --plot_dateslim: x-axis display range only (no data filtering)
        if args.plot_dateslim:
            d0, d1 = parse_dateslim(args.plot_dateslim, "plot_dateslim")
            ax.set_xlim(mdates.date2num(datetime.strptime(str(d0), "%Y%m%d")),
                        mdates.date2num(datetime.strptime(str(d1), "%Y%m%d")))

    axes[-1].set_xlabel("Date")
    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    # ── localization map panel ───────────────────────────────────────────
    if map_arr is not None:
        lo_r, hi_r, lo_c, hi_c = map_extent
        vmin, vmax = np.nanpercentile(map_arr, [2, 98])
        im = ax_map.imshow(map_arr, cmap=args.cmap, vmin=vmin, vmax=vmax,
                            extent=(lo_c, hi_c, hi_r, lo_r))  # keep row axis pointing down

        # pixel window(s), if a median window was used
        for l, c in zip(lines, cols):
            if args.windowsize:
                ax_map.add_patch(plt.Rectangle(
                    (c - args.windowsize - 0.5, l - args.windowsize - 0.5),
                    2 * args.windowsize + 1, 2 * args.windowsize + 1,
                    edgecolor="black", facecolor="none", lw=1))
        if wref:
            ax_map.add_patch(plt.Rectangle(
                (args.columnref - wref - 0.5, args.lineref - wref - 0.5),
                2 * wref + 1, 2 * wref + 1,
                edgecolor="lime", facecolor="none", lw=1))

        for k, (l, c) in enumerate(zip(lines, cols)):
            ax_map.plot(c, l, "x", color="black", markersize=10,
                        markeredgewidth=2, label=("pixel" if npix == 1 else "pixels") if k == 0 else None)
            if npix > 1:
                ax_map.annotate(str(k), (c, l), xytext=(4, 4), textcoords="offset points",
                                color="black", fontsize="small", fontweight="bold")
        ax_map.plot(args.columnref, args.lineref, "+", color="lime", markersize=10,
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
    default_name = (f"Disp_{lines[0]}_{cols[0]}.pdf" if npix == 1
                    else f"Disp_{lines[0]}_{cols[0]}_n{npix}.pdf")
    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.cube)), default_name)
    fig.savefig(out_path)
    print(f"  -> {out_path}")

    if not args.no_display:
        plt.show()


if __name__ == "__main__":
    main()
