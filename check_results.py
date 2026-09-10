#!/usr/bin/env python3
"""
check_results.py — Prepare inversion inputs and validation plots for a FLATSIM
                   time series directory.

Usage
-----
    # Full workflow (prepare + validate):
    python check_results.py <track_dir>
    python check_results.py data/Tienshan/D107_NORD

    # Prepare input files for invers_temp.py only, then exit:
    python check_results.py data/Tienshan/D107_NORD --prepare

    # Explicit paths:
    python check_results.py data/Tienshan/D107_NORD --aux /path/to/AUX
    python check_results.py data/Tienshan/D107_NORD/TS   # TS dir also accepted

Arguments
---------
  track_dir     Track directory containing TS/ and AUX/ subdirectories,
                OR the TS directory directly.
  --aux         AUX directory (auto-detected as sibling AUX/ if omitted).
  --save        Output directory for figures (default: <track_dir>/VALIDATION/).
  --no-display  Do not call plt.show() — useful for batch/headless runs.
  --prepare     Only prepare invers_temp.py input files, skip plots.

Directory convention
--------------------
  data/
  └── <site>/<track>/
      ├── TS/          time series (CNES_DTs_geo_*.tiff, RMSdate.txt, …)
      ├── AUX/         auxiliary data (baseline.rsc, list_iw*_sd_*.txt, …)
      └── VALIDATION/  figures written here (created automatically)

Step 0 — Prepare invers_temp.py inputs (always runs first)
----------------------------------------------------------------
  Reads AUX/baseline.rsc  → TS/list_images.txt
  Filters list_images.txt against TS/RMSdate.txt (removes missing dates,
    prints each removed date, and applies the same filter to
    list_ramp_sigma_inverted_img.txt and inaps.txt)
  TS/RMSdate.txt col 3     → TS/inrms.txt
  AUX/list_ramp_sigma_inverted_img.txt → TS/ (filtered copy)
  col 2 of ramp_sigma      → TS/inaps.txt
  Prints the ready-to-run invers_temp.py command.

Steps 1–12 — Validation plots (skipped if --prepare)
-----------------------------------------------------
  1.  SD variation along range          (list_iw*_sd_sx)
  2.  SD variation along azimuth        (list_iw*_sd_sy)
  3.  Quadratic SD along azimuth        (list_iw*_sd_qy)  ← click for date tooltip
  4.  SD tôle ondulée                   (list_iw*_sd_syy)
  5.  SD sigma                          (list_iw*_sd_sigma)
  6.  SD constant term                  (list_iw*_sd_cst)
  7.  IW phase jumps                    (list_iw12/23_merge)
  8.  Interferogram statistics summary  (unwrapping fractions, variance)
  9.  Bt histogram (ascending order) + unwrapping fraction vs Bt / season
 10.  RMS per date + RMS per ifg vs Bt
 11.  Interferogram network             (blue = kept, red = removed)
 12.  Per-image APS vs time             (aps_N.txt from invers_temp)
      Coefficient maps                  (lin_coeff, ampwt_coeff, phiwt_coeff)
      AUX PNG images                    (burst maps, SD maps, …)

Dependencies
------------
  numpy, matplotlib, gdal (osgeo)
  mplcursors  (optional — enables date tooltip on scatter plots)
    pip install mplcursors
"""

import os
import sys
import re
import argparse
import glob
import numpy as np
import matplotlib
if sys.platform != "darwin" and not os.environ.get("DISPLAY"):
    # Headless / remote Linux server (e.g. inside tmux or an SSH session
    # without X11 forwarding): force a non-interactive backend so that
    # plt.show() below can never block waiting for a display.
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import subprocess
import atexit

# ── project utils ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
import utils as U

matplotlib.rcParams.update({
    "figure.dpi"      : 120,
    "axes.titlesize"  : 10,
    "axes.labelsize"  : 9,
    "xtick.labelsize" : 8,
    "ytick.labelsize" : 8,
    "legend.fontsize" : 8,
})

IW_COLORS   = ["tab:blue", "tab:orange", "tab:green"]
IW_LABELS   = ["IW1", "IW2", "IW3"]
EPOCH_S     = datetime(2014, 1, 1).timestamp()   # same reference as check_results.sh


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: decimal year from YYYYMMDD string (bash-compatible)
# ─────────────────────────────────────────────────────────────────────────────
def _dec(s):
    """Convert date string to decimal year. Accepts both YYYYMMDD and decimal."""
    s = str(s).strip()
    try:
        # Already decimal (e.g. '2014.78564453')
        val = float(s)
        if val < 3000 and '.' in s:
            return val
        # Integer-like but not YYYYMMDD (e.g. '2014' as plain year)
        if len(s) <= 4:
            return val
        # YYYYMMDD
        return U.date_yyyymmdd_to_decimal(s)
    except ValueError:
        return U.date_yyyymmdd_to_decimal(s)


def _bt_yr(d1, d2):
    return abs(_dec(d2) - _dec(d1))


def _savefig(fig, save_dir, name, aux_dir=None):
    path = os.path.join(save_dir, name)
    fig.savefig(path, bbox_inches="tight")
    print(f"  → {path}")
    if aux_dir and os.path.isdir(aux_dir):
        import shutil
        shutil.copy2(path, os.path.join(aux_dir, name))


# ─────────────────────────────────────────────────────────────────────────────
#  1-3  SD time-series plots (sx / sy / qy / syy / sigma / cst / phs_grad)
# ─────────────────────────────────────────────────────────────────────────────
SD_GROUPS = [
    ("sd_sx",       "SD variation along range",           "SD (range)"),
    ("sd_sy",       "SD variation along azimuth",         "SD (azimuth)"),
    ("sd_qy",       "Quadratic SD along azimuth (cos)",   "SD (quad az)"),
    ("sd_syy",      "SD tôle ondulée",                    "SD (syy)"),
    ("sd_sigma",    "SD standard (sigma)",                "SD sigma"),
    ("sd_cst",      "SD constant term",                   "SD cst"),
]


def _add_date_tooltip(fig, ax, scatter_artists, date_labels):
    """
    Add mplcursors tooltip showing the date when hovering a scatter point.
    Falls back silently if mplcursors is not installed.
    scatter_artists : list of PathCollection (one per IW)
    date_labels     : list of lists of date strings (parallel to scatter_artists)
    """
    try:
        import mplcursors
        for artist, labels in zip(scatter_artists, date_labels):
            cursor = mplcursors.cursor(artist, hover=True)
            _labels = labels  # closure capture
            @cursor.connect("add")
            def on_add(sel, _lbl=_labels):
                idx = sel.index
                sel.annotation.set_text(_lbl[idx] if idx < len(_lbl) else "?")
                sel.annotation.get_bbox_patch().set(fc="lightyellow", alpha=0.9)
    except ImportError:
        pass   # mplcursors optional


def plot_sd_group(aux_dir, save_dir, display):
    for tag, title, ylabel in SD_GROUPS:
        fig, ax = plt.subplots(figsize=(14, 4))
        any_data = False
        all_artists = []
        all_labels  = []
        all_dates_str = []   # YYYYMMDD strings for x-ticks
        for iw, (col, lbl) in enumerate(zip(IW_COLORS, IW_LABELS)):
            pattern = os.path.join(aux_dir, f"list_iw{iw+1}_{tag}_inverted_img.txt")
            if not os.path.exists(pattern):
                continue
            dates, vals = U.read_sd_txt(pattern)
            if len(dates) == 0:
                continue
            dec_dates = [_dec(d) for d in dates]
            sc = ax.scatter(dec_dates, vals, color=col, label=lbl, s=12, zorder=3)
            ax.plot(dec_dates, vals, "-", color=col, linewidth=0.6, alpha=0.5)
            all_artists.append(sc)
            all_labels.append(dates)
            if len(dates) > len(all_dates_str):
                all_dates_str = dates
            any_data = True

        if not any_data:
            plt.close(fig)
            continue

        # Readable x-axis: one tick per year
        dec_all = [_dec(d) for d in all_dates_str]
        years = sorted({int(d) for d in dec_all})
        ax.set_xticks(years)
        ax.set_xticklabels(years, rotation=45, ha="right")
        # secondary: mark each image as a thin vertical line
        for d in dec_all:
            ax.axvline(d, color="grey", lw=0.2, alpha=0.3)

        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Tooltip on click (all tags) — especially useful for sd_qy
        if display:
            _add_date_tooltip(fig, ax, all_artists, all_labels)

        _savefig(fig, save_dir, f"check_{tag}.png")
        if display:
            plt.show()
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  4  IW merge (phase jumps across subswaths)
# ─────────────────────────────────────────────────────────────────────────────
def plot_iw_merge(aux_dir, save_dir, display):
    fig, ax = plt.subplots(figsize=(10, 4))
    any_data = False
    for pair, col, lbl in [("12", "tab:blue", "IW1-IW2"), ("23", "tab:orange", "IW2-IW3")]:
        fpath = os.path.join(aux_dir, f"list_iw{pair}_merge_inverted_img.txt")
        if not os.path.exists(fpath):
            continue
        dates, vals = U.read_sd_txt(fpath)
        dec_dates = [_dec(d) for d in dates]
        ax.plot(dec_dates, vals, "o", color=col, label=lbl, markersize=3)
        any_data = True
    if not any_data:
        plt.close(fig)
        return
    ax.set_title("Phase jumps across subswath limits")
    ax.set_xlabel("Date (decimal year)")
    ax.set_ylabel("Phase jump (rad)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _savefig(fig, save_dir, "check_iw_merge.png")
    if display:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  5  Interferogram stats summary (text + histogram)
# ─────────────────────────────────────────────────────────────────────────────
BT_BINS = [
    ("< 6 d",      0.0,   0.025),
    ("12 d",       0.025, 0.04),
    ("18 d",       0.04,  0.055),
    ("24 d",       0.055, 0.075),
    ("1 mo",       0.075, 0.15),
    ("2 mo",       0.15,  0.2),
    ("3 mo",       0.2,   0.4),
    ("6 mo",       0.4,   0.8),
    ("> 1 yr",     0.8,   np.inf),
]


def _count_bt_bins(ifg_list):
    counts = []
    for lbl, lo, hi in BT_BINS:
        n = sum(1 for r in ifg_list if lo < _bt_yr(r["date1"], r["date2"]) <= hi
                or (hi == np.inf and _bt_yr(r["date1"], r["date2"]) > lo))
        counts.append((lbl, n))
    return counts


def compute_ifg_stats(ts_dir, aux_dir):
    """Return a dict with all the statistics printed in check_results.sh."""
    unw_frac = U.read_list_unw_frac(os.path.join(aux_dir, "list_unw_frac.txt"))
    ramp_ra  = U.read_list_ramp_ra(os.path.join(aux_dir, "list_ramp_ra_estimated_ifg.txt"))
    rms_ifg  = U.read_rmsinterfero(os.path.join(ts_dir,  "RMSinterfero.txt"))

    try:
        ramp_merge = U.read_list_ramp_ra(os.path.join(aux_dir, "list_iw12_merge_estimated_ifg.txt"))
        n_merge = len(ramp_merge)
    except FileNotFoundError:
        n_merge = "N/A"

    all_ori    = {(r["date1"], r["date2"]) for r in unw_frac}
    after_ramp = {(r["date1"], r["date2"]) for r in ramp_ra}
    kept       = {(r["date1"], r["date2"]) for r in rms_ifg}

    removed_unw = all_ori  - after_ramp
    removed_var = after_ramp - kept

    # unwrapping fractions on kept ifgs
    frac_map = {(r["date1"], r["date2"]): r["fraction"] for r in unw_frac}
    kept_fracs = [frac_map.get(k, np.nan) for k in kept if k in frac_map]
    avg_frac = np.nanmean(kept_fracs) if kept_fracs else np.nan
    max_frac = np.nanmax(kept_fracs) if kept_fracs else np.nan
    min_frac = np.nanmin(kept_fracs) if kept_fracs else np.nan

    # variance on kept ifgs
    # Accept both the per-ifg and per-image ramp sigma files
    for _sigma_fname in ["list_ramp_sigma_estimated_ifg.txt",
                         "list_ramp_sigma_inverted_img.txt"]:
        sigma_f = os.path.join(aux_dir, _sigma_fname)
        if os.path.exists(sigma_f):
            break
    sig_map = {}
    if os.path.exists(sigma_f):
        rows = U.read_list_ramp_sigma(sigma_f)
        # per-ifg rows have date1/date2; per-image rows have yyyymmdd only
        for r in rows:
            if "date1" in r:
                sig_map[(r["date1"], r["date2"])] = r["sigma"]
    kept_sigs = [sig_map.get(k, np.nan) for k in kept if k in sig_map]
    avg_sig = np.nanmean(kept_sigs) if kept_sigs else np.nan
    max_sig = np.nanmax(kept_sigs) if kept_sigs else np.nan
    min_sig = np.nanmin(kept_sigs) if kept_sigs else np.nan

    # 1-yr ifgs removed
    one_yr_removed = sum(1 for d1, d2 in removed_unw if _bt_yr(d1, d2) > 0.8)
    one_yr_kept    = sum(1 for r in rms_ifg if _bt_yr(r["date1"], r["date2"]) > 0.8)

    # image counts
    sd_sx_f = os.path.join(aux_dir, "list_iw1_sd_sx_inverted_img.txt")
    nim_ini = 0
    if os.path.exists(sd_sx_f):
        with open(sd_sx_f) as f:
            nim_ini = sum(1 for _ in f)
    nim_inverted = len(U.read_rmsdate(os.path.join(ts_dir, "RMSdate.txt")))

    return {
        "n_merge"       : n_merge,
        "n_unw"         : len(all_ori),
        "n_removed_unw" : len(removed_unw),
        "n_removed_var" : len(removed_var),
        "n_kept"        : len(kept),
        "avg_frac"      : avg_frac,
        "max_frac"      : max_frac,
        "min_frac"      : min_frac,
        "avg_sig"       : avg_sig,
        "max_sig"       : max_sig,
        "min_sig"       : min_sig,
        "one_yr_removed": one_yr_removed,
        "one_yr_kept"   : one_yr_kept,
        "nim_ini"       : nim_ini,
        "nim_inverted"  : nim_inverted,
        "kept_ifgs"     : list(rms_ifg),
        "well_unw"      : [r for r in unw_frac if (r["date1"], r["date2"]) in after_ramp],
    }


def print_stats(stats):
    print("\n" + "=" * 55)
    print("  FLATSIM validation summary")
    print("=" * 55)
    print(f"  Number of merged ifg              : {stats['n_merge']}")
    print(f"  Number of unwrapped ifg            : {stats['n_unw']}")
    print(f"  Removed (unwrapping fraction low)  : {stats['n_removed_unw']}")
    print(f"  Removed (large variance)           : {stats['n_removed_var']}")
    print(f"  Kept in time series                : {stats['n_kept']}")
    print(f"  1-yr ifg removed                   : {stats['one_yr_removed']}")
    print(f"  1-yr ifg kept                      : {stats['one_yr_kept']}")
    print(f"  Images processed (ini)             : {stats['nim_ini']}")
    print(f"  Images in time series              : {stats['nim_inverted']}")
    print(f"  Avg unwrapping fraction (kept)     : {stats['avg_frac']:.3f}"
          f"  max={stats['max_frac']:.3f}  min={stats['min_frac']:.3f}")
    print(f"  Avg variance (kept)                : {stats['avg_sig']:.3f}"
          f"  max={stats['max_sig']:.3f}  min={stats['min_sig']:.3f}")
    print("=" * 55 + "\n")


def plot_bt_histogram(stats, save_dir, display):
    kept = stats["kept_ifgs"]
    counts = _count_bt_bins(kept)
    labels = [c[0] for c in counts]
    values = [c[1] for c in counts]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color="steelblue", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of interferograms")
    ax.set_title("Histogram of kept interferograms by temporal baseline")
    ax.bar_label(bars, padding=2, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, save_dir, "check_histo_bt.png")
    if display:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  6  Unwrapping fraction vs Bt and vs season
# ─────────────────────────────────────────────────────────────────────────────
def plot_unw_frac_vs_bt(stats, save_dir, display):
    well = stats["well_unw"]
    if not well:
        return
    bt    = [_bt_yr(r["date1"], r["date2"]) for r in well]
    frac  = [r["fraction"] for r in well]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors_bt = ["tomato" if f < 0.5 else "steelblue" for f in frac]
    ax.scatter(bt, frac, s=10, alpha=0.7, color=colors_bt)
    ax.axhline(0.5, color="red", ls="--", lw=0.8)
    ax.set_xlabel("Temporal baseline (yr)")
    ax.set_ylabel("Unwrapping fraction")
    ax.set_title("Unwrapping fraction vs temporal baseline  (red < 0.5)")
    ax.grid(True, alpha=0.3)
    _savefig(fig, save_dir, "check_unw_frac_vs_bt.png")
    if display:
        plt.show()
    plt.close(fig)

    # season plot (only for Bt > 0.5 yr)
    long_bt = [(r, _bt_yr(r["date1"], r["date2"])) for r in well
               if _bt_yr(r["date1"], r["date2"]) > 0.5]
    if not long_bt:
        return
    season = []
    fracs  = []
    for r, bt_val in long_bt:
        d2_dec = _dec(r["date2"])
        d1_dec = _dec(r["date1"])
        season.append(d2_dec % 1)
        fracs.append(r["fraction"])
        season.append(d1_dec % 1)
        fracs.append(r["fraction"])

    fig, ax = plt.subplots(figsize=(8, 4))
    colors_s = ["tomato" if f < 0.5 else "steelblue" for f in fracs]
    ax.scatter(season, fracs, s=10, alpha=0.7, color=colors_s)
    ax.axhline(0.5, color="red", ls="--", lw=0.8)
    ax.set_xlabel("Fractional year (season)")
    ax.set_ylabel("Unwrapping fraction")
    ax.set_title("Unwrapping fraction vs season (1-yr ifg only)  (red < 0.5)")
    ax.grid(True, alpha=0.3)
    _savefig(fig, save_dir, "check_unw_frac_vs_season.png")
    if display:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  7  RMS per date
# ─────────────────────────────────────────────────────────────────────────────
def plot_rms_date(ts_dir, save_dir, display):
    rmsdate = U.read_rmsdate(os.path.join(ts_dir, "RMSdate.txt"))
    if not rmsdate:
        return
    dates = [_dec(r[0]) for r in rmsdate]
    rms   = [r[1] for r in rmsdate]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, rms, "o-", color="steelblue", markersize=4, linewidth=0.8)
    ax.set_xlabel("Date (decimal year)")
    ax.set_ylabel("RMS (rad)")
    ax.set_title("Per-date RMS (image quality)")
    ax.grid(True, alpha=0.3)
    _savefig(fig, save_dir, "check_rms_date.png")
    if display:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  8  RMS per interferogram + RMS vs Bt
# ─────────────────────────────────────────────────────────────────────────────
def plot_rms_interfero(ts_dir, save_dir, display):
    rms_ifg = U.read_rmsinterfero(os.path.join(ts_dir, "RMSinterfero.txt"))
    if not rms_ifg:
        return

    bt  = [_bt_yr(r["date1"], r["date2"]) for r in rms_ifg]
    rms = [r["rms"] for r in rms_ifg]

    # flag high-RMS ifg
    high_rms = [(r, v) for r, v in zip(rms_ifg, rms) if v > 0.9]
    if high_rms:
        print(f"\n  Ifg with RMS > 0.9 (likely unwrapping errors):")
        for r, v in sorted(high_rms, key=lambda x: -x[1]):
            print(f"    {r['date1']} – {r['date2']}  RMS={v:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    # panel 1 — RMS vs temporal baseline
    sc = ax1.scatter(bt, rms, c=rms, cmap="RdYlGn_r", s=12, alpha=0.8, vmin=0, vmax=1.5)
    plt.colorbar(sc, ax=ax1, label="RMS (rad)")
    ax1.axhline(0.9, color="red", ls="--", lw=0.8, label="RMS = 0.9")
    ax1.set_xlabel("Temporal baseline (yr)")
    ax1.set_ylabel("RMS (rad)")
    ax1.set_title("Interferogram RMS vs temporal baseline")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # panel 2 — RMS per interferogram in original order
    colors = ["tomato" if v > 0.9 else "steelblue" for v in rms]
    ax2.bar(range(len(rms)), rms, color=colors, width=1.0, edgecolor="none")
    ax2.axhline(0.9, color="red", ls="--", lw=0.8, label="RMS = 0.9")
    ax2.set_xlabel("Interferogram index")
    ax2.set_ylabel("RMS (rad)")
    ax2.set_title(f"Interferogram RMS  —  {len(rms_ifg)} ifg  "
                  f"({sum(1 for v in rms if v > 0.9)} > 0.9)")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _savefig(fig, save_dir, "check_rms_ifg_vs_bt.png")
    if display:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  9  Ramp residuals
#
#  list_ramp_sigma_estimated_ifg.txt — residual variance PER IFG after ramp
#    estimation, before TS inversion. Cols: YYYYMMDD1 YYYYMMDD2 ... sigma
#
#  list_ramp_sigma_inverted_img.txt  — residual APS PER IMAGE after TS
#    inversion (closure condition). Cols: decimal_date sigma YYYYMMDD
#    These are different objects (one per ifg, one per image) and should
#    look different.
#
#  list_ramp_ra_estimated_ifg.txt — IFGs that passed the unwrapping fraction
#    threshold and had their range/azimuth ramps estimated. These are the IFGs
#    that entered the TS inversion.
# ─────────────────────────────────────────────────────────────────────────────
def plot_variance_comparison(aux_dir, ts_dir, save_dir, display):
    sigma_ifg_f = os.path.join(aux_dir, "list_ramp_sigma_estimated_ifg.txt")
    sigma_img_f = os.path.join(aux_dir, "list_ramp_sigma_inverted_img.txt")
    ra_f        = os.path.join(aux_dir, "list_ramp_ra_estimated_ifg.txt")

    rms_ifg = U.read_rmsinterfero(os.path.join(ts_dir, "RMSinterfero.txt"))
    kept    = {(r["date1"], r["date2"]) for r in rms_ifg}

    has_ifg = os.path.exists(sigma_ifg_f) and os.path.exists(ra_f)
    has_img = os.path.exists(sigma_img_f)
    if not has_ifg and not has_img:
        print("  No ramp sigma files found, skipping.")
        return

    ncols = (1 if has_ifg else 0) + (1 if has_img else 0)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]
    ax_idx = 0

    # panel A — per-IFG residual variance vs Bt (blue=kept, red=removed)
    if has_ifg:
        ax = axes[ax_idx]; ax_idx += 1
        sig_rows = U.read_list_ramp_sigma(sigma_ifg_f)
        ifg_rows = [r for r in sig_rows if "date1" in r]
        sig_map  = {(r["date1"], r["date2"]): r["sigma"] for r in ifg_rows}
        ra_rows  = U.read_list_ramp_ra(ra_f)

        bt_all, sig_all, col_all = [], [], []
        for r in ra_rows:
            key = (r["date1"], r["date2"])
            if key in sig_map:
                bt_all.append(_bt_yr(r["date1"], r["date2"]))
                sig_all.append(sig_map[key])
                col_all.append("tomato" if key not in kept else "steelblue")

        ax.scatter(bt_all, sig_all, c=col_all, s=12, alpha=0.8)
        ax.set_xlabel("Temporal baseline (yr)")
        ax.set_ylabel("Residual sigma (rad)")
        ax.set_title("Per-IFG residual variance after ramp estimation\n"
                     "(blue = kept in TS,  red = removed)")
        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([0],[0], marker='o', color='w', markerfacecolor='steelblue',
                   markersize=7, label=f"kept ({col_all.count('steelblue')})"),
            Line2D([0],[0], marker='o', color='w', markerfacecolor='tomato',
                   markersize=7, label=f"removed ({col_all.count('tomato')})"),
        ])
        ax.grid(True, alpha=0.3)

    # panel B — per-image APS after TS inversion vs time
    if has_img:
        ax = axes[ax_idx]; ax_idx += 1
        img_rows = U.read_list_ramp_sigma(sigma_img_f)
        img_rows = [r for r in img_rows if "yyyymmdd" in r]
        if img_rows:
            dates_img = [_dec(r["yyyymmdd"]) for r in img_rows]
            sigma_img = [r["sigma"] for r in img_rows]
            ax.plot(dates_img, sigma_img, "o-", color="steelblue",
                    markersize=4, linewidth=0.8)
            ax.set_xlabel("Date (decimal year)")
            ax.set_ylabel("APS (rad)")
            ax.set_title("Per-image APS after TS inversion\n"
                         "(list_ramp_sigma_inverted_img)")
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title("list_ramp_sigma_inverted_img\n(no per-image rows found)")
            ax.axis("off")

    fig.suptitle("Ramp residuals — IFG variance  vs  image APS", fontsize=11)
    fig.tight_layout()
    _savefig(fig, save_dir, "check_variance_comparison.png")
    if display:
        plt.show()
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
#  10  Interferogram network
# ─────────────────────────────────────────────────────────────────────────────
def plot_ifg_network(ts_dir, aux_dir, save_dir, display):
    baseline_f = os.path.join(aux_dir, "baseline.rsc")
    if not os.path.exists(baseline_f):
        print("  baseline.rsc not found, skipping network plot.")
        return

    # Read baseline perpendicular for each date
    bl_map = {}
    with open(baseline_f) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    bl_map[parts[0]] = float(parts[1])
                except ValueError:
                    pass

    unw_frac = U.read_list_unw_frac(os.path.join(aux_dir, "list_unw_frac.txt"))
    rms_ifg  = U.read_rmsinterfero(os.path.join(ts_dir, "RMSinterfero.txt"))

    all_ori = {(r["date1"], r["date2"]) for r in unw_frac}
    kept    = {(r["date1"], r["date2"]) for r in rms_ifg}
    removed = all_ori - kept

    # Single figure: all ifg with colour coding
    # kept = blue, removed = red
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot all image nodes
    all_dates = sorted({d for d1, d2 in all_ori for d in [d1, d2]})
    for d in all_dates:
        b = bl_map.get(d)
        if b is not None:
            ax.plot(_dec(d), b, "ko", markersize=3, zorder=5)

    # Removed interferograms in red (behind)
    for d1, d2 in removed:
        b1, b2 = bl_map.get(d1), bl_map.get(d2)
        if b1 is not None and b2 is not None:
            ax.plot([_dec(d1), _dec(d2)], [b1, b2],
                    color="tomato", lw=0.8, alpha=0.7, zorder=2)

    # Kept interferograms in blue (in front)
    for d1, d2 in kept:
        b1, b2 = bl_map.get(d1), bl_map.get(d2)
        if b1 is not None and b2 is not None:
            ax.plot([_dec(d1), _dec(d2)], [b1, b2],
                    color="steelblue", lw=0.8, alpha=0.8, zorder=3)

    # Legend proxies
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="steelblue", lw=1.5, label=f"Kept ({len(kept)})"),
        Line2D([0], [0], color="tomato",    lw=1.5, label=f"Removed ({len(removed)})"),
        Line2D([0], [0], color="black", marker="o", lw=0, markersize=4, label="Image"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    ax.set_title("Interferogram network  —  blue: kept   red: removed")
    ax.set_xlabel("Date (decimal year)")
    ax.set_ylabel("Perpendicular baseline (m)")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    _savefig(fig, save_dir, "check_ifg_network.png")
    if display:
        plt.show()
    plt.close(fig)


def _plot_network_ax(ax, ifg_list, bl_map):
    """Helper kept for check_results_inc.py compatibility."""
    dates = sorted({d for r in ifg_list for d in [r["date1"], r["date2"]]})
    for d in dates:
        b = bl_map.get(d)
        if b is not None:
            ax.plot(_dec(d), b, "ko", markersize=2)
    for r in ifg_list:
        d1, d2 = r["date1"], r["date2"]
        b1, b2 = bl_map.get(d1), bl_map.get(d2)
        if b1 is not None and b2 is not None:
            ax.plot([_dec(d1), _dec(d2)], [b1, b2], "b-", lw=0.5, alpha=0.5)


# ─────────────────────────────────────────────────────────────────────────────
#  11  APS_N.txt — per-image APS from invers_temp.py
# ─────────────────────────────────────────────────────────────────────────────
def _load_list_images(ts_dir):
    fpath = os.path.join(ts_dir, "list_images.txt")
    if not os.path.exists(fpath):
        return []
    dates = []
    with open(fpath) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                dates.append(parts[1])
    return dates


def plot_sigma_vs_time(ts_dir, save_dir, display):
    """Plot per-image APS_N.txt vs time (one curve per iteration)."""
    sigma_files = sorted(
        glob.glob(os.path.join(ts_dir, "aps_*.txt")),
        key=lambda p: int(os.path.basename(p).replace("aps_","").replace(".txt",""))
    )
    if not sigma_files:
        print("  No aps_N.txt files found, skipping.")
        return
    dates = _load_list_images(ts_dir)
    if not dates:
        print("  list_images.txt not found, cannot plot sigma vs time.")
        return
    dec_dates = [_dec(d) for d in dates]
    colors = plt.cm.Blues(np.linspace(0.4, 1.0, len(sigma_files)))
    fig, ax = plt.subplots(figsize=(11, 4))
    for fpath, col in zip(sigma_files, colors):
        sigma = np.loadtxt(fpath)
        niter = os.path.basename(fpath).replace("aps_","").replace(".txt","")
        n = min(len(dec_dates), len(sigma))
        ax.plot(dec_dates[:n], sigma[:n], "o-", color=col, markersize=4,
                linewidth=0.8, label=f"iter {niter}")
    ax.set_xlabel("Date (decimal year)")
    ax.set_ylabel("APS (rad)")
    ax.set_title("Per-image APS vs time (invers_temp iterations)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _savefig(fig, save_dir, "check_sigma_vs_time.png")
    if display:
        plt.show()
    plt.close(fig)



def plot_median_vs_time(ts_dir, save_dir, display):
    """Plot median residual on reference zone from ref_median_N.txt."""
    ref_files = sorted(
        glob.glob(os.path.join(ts_dir, "ref_median_*.txt")),
        key=lambda p: int(os.path.basename(p).replace("ref_median_","").replace(".txt",""))
    )
    if not ref_files:
        print("  No ref_median_N.txt files found, skipping.")
        return
    dates = _load_list_images(ts_dir)
    if not dates:
        return
    dec_dates = [_dec(d) for d in dates]
    colors = plt.cm.Blues(np.linspace(0.4, 1.0, len(ref_files)))
    fig, ax = plt.subplots(figsize=(11, 4))
    for fpath, col in zip(ref_files, colors):
        arr   = np.loadtxt(fpath, comments="#")
        niter = os.path.basename(fpath).replace("ref_median_","").replace(".txt","")
        med   = arr[:, 1] if arr.ndim > 1 else arr
        n     = min(len(dec_dates), len(med))
        ax.plot(dec_dates[:n], med[:n], "o-", color=col, markersize=4,
                linewidth=0.8, label=f"iter {niter}")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Date (decimal year)")
    ax.set_ylabel("Median on ref zone (rad)")
    ax.set_title("Median residual on reference zone vs time\n"
                 "(should converge to 0 — if large: use --ref_zone)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _savefig(fig, save_dir, "check_median_vs_time.png")
    if display:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  12  Coefficient maps: lin_coeff, ampwt_coeff, phiwt_coeff
# ─────────────────────────────────────────────────────────────────────────────
def _get_dims(ts_dir):
    """Return (ncol, nlign) from lect.in or any .hdr in ts_dir."""
    lect = os.path.join(ts_dir, "lect.in")
    if os.path.exists(lect):
        vals = open(lect).read().split()
        if len(vals) >= 2:
            return int(vals[0]), int(vals[1])
    for hdr in glob.glob(os.path.join(ts_dir, "*.hdr")):
        try:
            info = U.read_envi_hdr(hdr)
            return info["ncol"], info["nlign"]
        except Exception:
            pass
    return None


def _read_coeff_map(ts_dir, name, suffix="_coeff"):
    """
    Read a _coeff (or _sigcoeff, pass suffix="_sigcoeff") map produced by
    invers_temp.py. Tries GeoTIFF (.tif/.tiff) first using gdal, then ENVI .r4.
    Returns (array float32, source_path) or (None, None).
    """
    try:
        from osgeo import gdal as _gdal
        _gdal_ok = True
    except ImportError:
        _gdal_ok = False
    for ext in [".tif", ".tiff"]:
        tif = os.path.join(ts_dir, f"{name}{suffix}{ext}")
        if os.path.exists(tif):
            if _gdal_ok:
                try:
                    ds = _gdal.Open(tif)
                    if ds is not None:
                        arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
                        nd  = ds.GetRasterBand(1).GetNoDataValue()
                        if nd is not None:
                            arr[arr == nd] = np.nan
                        del ds
                        return arr, tif
                except Exception as e:
                    logger.warning(f"gdal cannot read {tif}: {e}")
            # fallback: matplotlib imread (works for single-band float tif)
            try:
                arr = plt.imread(tif).astype(np.float32)
                return arr, tif
            except Exception:
                pass
    r4 = os.path.join(ts_dir, f"{name}{suffix}.r4")
    if os.path.exists(r4):
        dims = _get_dims(ts_dir)
        if dims:
            ncol, nlign = dims
            return np.fromfile(r4, dtype=np.float32).reshape(nlign, ncol), r4
    return None, None


def plot_coeff_maps(ts_dir, save_dir, display):
    """
    Plot lin_coeff, ampwt_coeff, phiwt_coeff (and ref_coeff if present).
    """
    maps_cfg = [
        ("lin",   "Linear velocity",     "RdBu_r", "rad/yr"),
        ("ampwt", "Seasonal amplitude",  "viridis", "amplitude"),
        ("phiwt", "Seasonal phase",      "hsv",     "phase (rad)"),
        ("ref",   "Constant term",       "RdBu_r",  "rad"),
    ]
    available = [(n, t, c, u) for n, t, c, u in maps_cfg
                 if _read_coeff_map(ts_dir, n)[0] is not None]
    if not available:
        print("  No coefficient maps found, skipping.")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for ax, (name, title, cmap, unit) in zip(axes, available):
        arr, src = _read_coeff_map(ts_dir, name)
        masked = np.where(arr == 0, np.nan, arr)
        finite = masked[np.isfinite(masked)]
        if len(finite) == 0:
            ax.set_title(f"{title}\n(empty)")
            ax.axis("off")
            continue
        vmax = np.nanpercentile(np.abs(finite), 95)
        vmin = -vmax if cmap == "RdBu_r" else np.nanpercentile(finite, 2)
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad("lightgrey")
        im = ax.imshow(masked, cmap=cmap_obj, vmin=vmin, vmax=vmax,
                       aspect="auto", origin="upper")
        plt.colorbar(im, ax=ax, label=unit, shrink=0.8)
        ax.set_title(f"{title}\n{os.path.basename(src)}", fontsize=9)
        ax.axis("off")

    fig.suptitle("invers_temp.py — output maps", fontsize=11)
    fig.tight_layout()
    _savefig(fig, save_dir, "check_coeff_maps.png")
    if display:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  invers_temp.py coefficient maps (rad, rad/yr) -> LOS physical units (mm,
#  mm/yr), GeoTIFF
# ─────────────────────────────────────────────────────────────────────────────
# Sentinel-1 C-band wavelength: 5.5465763 cm. |factor| = lambda/(4*pi).
# Applied with a NEGATIVE sign for signed LOS quantities (the velocity
# itself), so that positive = motion TOWARDS the satellite (range decrease).
# Applied with the POSITIVE (unsigned) value for magnitude-only quantities
# (uncertainties, seasonal amplitude), which have no direction and must stay
# >= 0 - flipping their sign would be physically meaningless.
LOS_MM_PER_RAD = 4.4138249

# (coeff_name, suffix, type_code, unit, signed)
#   coeff_name/suffix -> source file "<coeff_name><suffix>.tif" (or .r4)
#   type_code         -> used in the output filename, e.g. 'MV', 'SIG-MV'
#   unit              -> 'mmyr' (rate) or 'mm' (magnitude), used in the name
#   signed            -> True: multiply by -LOS_MM_PER_RAD (directional)
#                        False: multiply by +LOS_MM_PER_RAD (magnitude, >=0)
LOS_MM_PRODUCTS = [
    ("lin",   "_coeff",    "MV",      "mmyr", True),
    ("lin",   "_sigcoeff", "SIG-MV",  "mmyr", False),
    ("ampwt", "_coeff",    "AMP",     "mm",   False),
    ("ampwt", "_sigcoeff", "SIG-AMP", "mm",   False),
]


def _derive_swath_years(ts_dir, track_dir):
    """
    Best-effort '<swath>[-<year0>-<year1>]' code, e.g. 'D019SUD-2021-2025',
    'A131-2020-2025', used to name the exported physical-unit GeoTIFFs.

    Tries, in order:
      1. the token right before the polarization code (VV/VH/HH/HV) in the
         dataset directory name for the swath, plus the first two 4-digit
         years found between the polarization code and '_IW' for the range,
         e.g. '..._TIBET-HIM-D019SUD-VV-2021-2025_IW123_...' -> 'D019SUD-2021-2025'
      2. the basename of the track directory for the swath (trailing
         date-range suffix and separators stripped), plus a year range found
         in that same basename if present, e.g. 'D019_sud_2021-2025' ->
         'D019SUD-2021-2025'
    """
    base = os.path.basename(os.path.normpath(ts_dir))
    m = re.search(r'-([A-Za-z0-9]+)-(?:VV|VH|HH|HV)-(.+?)(?:_IW|$)', base)
    if m:
        swath = m.group(1).upper()
        years = re.findall(r'\d{4}', m.group(2))
        if len(years) >= 2:
            return f"{swath}-{years[0]}-{years[1]}"
        return swath

    fallback = os.path.basename(os.path.normpath(track_dir))
    years = re.search(r'(\d{4})[-_](\d{4})', fallback)
    swath = re.sub(r'[_-]\d{4}.*$', '', fallback)
    swath = re.sub(r'[_-]', '', swath).upper() or "TRACK"
    if years:
        return f"{swath}-{years.group(1)}-{years.group(2)}"
    return swath


def write_los_product_mm(ts_dir, track_dir, coeff_name, type_code, unit, signed,
                          suffix="_coeff", swath_years=None):
    """
    Convert one invers_temp.py coefficient map (rad or rad/yr) to a LOS
    GeoTIFF in physical units (mm or mm/yr).

    Writes '<swath[-years]>_<type_code>-LOS_geo<looks>_<unit>.tiff' into
    ts_dir (next to the other CNES products) and returns its path, or None
    if the source map or a georeferenced sibling product isn't found.
    """
    arr, src = _read_coeff_map(ts_dir, coeff_name, suffix=suffix)
    if arr is None:
        print(f"  No {coeff_name}{suffix} found - {type_code} {unit} map not written.")
        return None

    factor = -LOS_MM_PER_RAD if signed else LOS_MM_PER_RAD
    arr_out = (arr.astype(np.float64) * factor).astype(np.float32)

    try:
        from osgeo import gdal as _gdal
    except ImportError:
        print(f"  WARNING: gdal not available - cannot write {type_code} {unit} GeoTIFF.")
        return None

    # Georeferencing: prefer the source file itself if it is already a
    # GeoTIFF (invers_temp.py wrote it), otherwise borrow it from a sibling
    # CNES product on the same grid.
    gt = proj = None
    if src and src.lower().endswith((".tif", ".tiff")):
        ds = _gdal.Open(src)
        if ds is not None:
            gt, proj = ds.GetGeoTransform(), ds.GetProjection()
            del ds
    if gt is None:
        for pattern in ("CNES_MV-LOS_geo_*.tiff", "CNES_DTs_geo_*.tiff"):
            for cand in sorted(glob.glob(os.path.join(ts_dir, pattern))):
                ds = _gdal.Open(cand)
                if ds is None:
                    continue
                if (ds.RasterXSize, ds.RasterYSize) == (arr.shape[1], arr.shape[0]):
                    gt, proj = ds.GetGeoTransform(), ds.GetProjection()
                    del ds
                    break
                del ds
            if gt is not None:
                break
    if gt is None:
        print(f"  WARNING: no matching georeferenced product found - "
              f"{type_code} {unit} GeoTIFF written without a CRS.")

    swath_years = swath_years or _derive_swath_years(ts_dir, track_dir)
    looks = U.get_looks(ts_dir)  # e.g. '_8rlks'
    out_path = os.path.join(ts_dir, f"{swath_years}_{type_code}-LOS_geo{looks}_{unit}.tiff")

    driver = _gdal.GetDriverByName("GTiff")
    nlign, ncol = arr_out.shape
    out_ds = driver.Create(out_path, ncol, nlign, 1, _gdal.GDT_Float32)
    band = out_ds.GetRasterBand(1)
    band.WriteArray(arr_out)
    band.SetNoDataValue(0)
    if gt is not None:
        out_ds.SetGeoTransform(gt)
        out_ds.SetProjection(proj)
    band.FlushCache()
    del out_ds

    note = "+towards satellite" if signed else "magnitude, >= 0"
    print(f"  -> {out_path}  ({type_code}, {unit}, {note})")
    return out_path


def write_los_mm_products(ts_dir, track_dir, swath_name=None):
    """
    Export lin_coeff / lin_sigcoeff / ampwt_coeff / ampwt_sigcoeff (rad,
    rad/yr) as LOS physical-unit GeoTIFFs (mm, mm/yr) - see LOS_MM_PRODUCTS.
    Returns the list of output paths written (None entries skipped/missing).
    """
    swath_years = swath_name or _derive_swath_years(ts_dir, track_dir)
    paths = []
    for coeff_name, suffix, type_code, unit, signed in LOS_MM_PRODUCTS:
        paths.append(write_los_product_mm(
            ts_dir, track_dir, coeff_name, type_code, unit, signed,
            suffix=suffix, swath_years=swath_years,
        ))
    return paths


# ─────────────────────────────────────────────────────────────────────────────
#  Velocity and seasonal maps comparison
# ─────────────────────────────────────────────────────────────────────────────
def _read_tif(path):
    """Read first band of a GeoTIFF → (array, nodata-masked). Returns None on failure."""
    try:
        from osgeo import gdal
        ds = gdal.Open(path)
        if ds is None:
            return None
        arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        nd  = ds.GetRasterBand(1).GetNoDataValue()
        if nd is not None:
            arr[arr == nd] = np.nan
        arr[arr == 0] = np.nan   # mask zeros (unprocessed pixels)
        return arr
    except Exception:
        return None


def _imshow_map(ax, arr, cmap, title, unit, pct=95):
    """Plot a 2-D map with symmetric colour scale and colorbar."""
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        ax.set_title(f"{title}\n(empty)"); ax.axis("off"); return
    vmax = np.nanpercentile(np.abs(finite), pct)
    vmin = -vmax if cmap in ("RdBu_r", "bwr") else np.nanpercentile(finite, 100 - pct)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("lightgrey")
    im = ax.imshow(arr, cmap=cmap_obj, vmin=vmin, vmax=vmax,
                   aspect="auto", origin="upper")
    plt.colorbar(im, ax=ax, label=unit, shrink=0.8, pad=0.02)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


# ─────────────────────────────────────────────────────────────────────────────
#  Net quality maps (from CNES_Net_geo_*.tiff)
# ─────────────────────────────────────────────────────────────────────────────
def plot_net_maps(ts_dir, save_dir, display):
    """
    Display Net quality maps extracted from CNES_Net_geo_*.tiff:
      RMSpixel  — RMS misclosure per pixel (rad)
      Net_nifg  — nb interferograms used per pixel
      Net_nimg  — nb images used per pixel
      Net_tcoh  — temporal coherence proxy
      Net_bias  — bias proxy (rad)
    """
    bands = [
        ("RMSpixel", "RMS misclosure\n(rad)",          "hot_r",   98),
        ("Net_nifg",  "Nb ifg used",                    "viridis", 98),
        ("Net_nimg",  "Nb images used",                 "viridis", 98),
        ("Net_tcoh",  "Temporal coherence\n(proxy)",   "RdYlGn",  98),
        ("Net_bias",  "Bias proxy\n(rad)",              "RdBu_r",  95),
    ]
    avail = []
    for name, title, cmap, pct in bands:
        arr = _read_tif(os.path.join(ts_dir, f"{name}.tif"))
        if arr is not None:
            avail.append((arr, title, cmap, pct))

    if not avail:
        print("  No Net maps found (run --prepare first to extract them).")
        return

    n = len(avail)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (arr, title, cmap, pct) in zip(axes, avail):
        _imshow_map(ax, arr, cmap, title, "", pct=pct)

    fig.suptitle("FLATSIM network quality maps  (CNES_Net_geo)", fontsize=11)
    fig.tight_layout()
    _savefig(fig, save_dir, "check_net_maps.png")
    if display:
        plt.show()
    plt.close(fig)


def plot_velocity_maps(ts_dir, save_dir, display):
    """
    Figure 1 — Velocity comparison
        Left  : CNES_MV-LOS_geo_*.tiff  (FLATSIM product)
        Right : lin_coeff.tif / lin_coeff.r4  (invers_temp output, if present)

    Figure 2 — Seasonal model (if invers_temp outputs exist)
        ampwt_coeff.tif  — amplitude
        phiwt_coeff.tif  — phase
        cos_coeff.tif    — cosine term
        sin_coeff.tif    — sine term
    """
    # ── Velocity maps ────────────────────────────────────────────────────────
    mv_path = next(iter(sorted(glob.glob(
        os.path.join(ts_dir, "CNES_MV-LOS_geo_*.tiff")))), None)
    lin_arr, lin_src = _read_coeff_map(ts_dir, "lin")

    mv_arr = _read_tif(mv_path) if mv_path else None

    if mv_arr is None and lin_arr is None:
        print("  No velocity maps found, skipping.")
        return

    panels = []
    if mv_arr is not None:
        panels.append((mv_arr, "FLATSIM velocity\n(CNES_MV-LOS)", "RdBu_r", "rad/yr"))
    if lin_arr is not None:
        lin_masked = np.where(lin_arr == 0, np.nan, lin_arr)
        panels.append((lin_masked, "Iterated velocity\n(lin_coeff)", "RdBu_r", "rad/yr"))

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]
    for ax, (arr, title, cmap, unit) in zip(axes, panels):
        _imshow_map(ax, arr, cmap, title, unit)

    # difference panel if both exist
    if mv_arr is not None and lin_arr is not None:
        lin_masked = np.where(lin_arr == 0, np.nan, lin_arr)
        # normalise lin to same scale as mv (approximate: both in rad or mm)
        # just show side by side — user can judge the scaling
        fig.suptitle("Velocity maps — FLATSIM product vs iterated inversion",
                     fontsize=11)
    else:
        fig.suptitle("Velocity map", fontsize=11)

    fig.tight_layout()
    _savefig(fig, save_dir, "check_velocity_maps.png")
    if display:
        plt.show()
    plt.close(fig)

    # ── Seasonal maps ────────────────────────────────────────────────────────
    seas_cfg = [
        ("ampwt", "Seasonal amplitude", "viridis",  "amplitude"),
        ("phiwt", "Seasonal phase",     "hsv",       "phase (rad)"),
        ("cos",   "Cosine term",        "RdBu_r",    "rad"),
        ("sin",   "Sine term",          "RdBu_r",    "rad"),
    ]
    seas_avail = []
    for name, title, cmap, unit in seas_cfg:
        arr, src = _read_coeff_map(ts_dir, name)
        if arr is not None:
            seas_avail.append((np.where(arr == 0, np.nan, arr), title, cmap, unit))

    if not seas_avail:
        return

    fig2, axes2 = plt.subplots(1, len(seas_avail), figsize=(6 * len(seas_avail), 5))
    if len(seas_avail) == 1:
        axes2 = [axes2]
    for ax, (arr, title, cmap, unit) in zip(axes2, seas_avail):
        _imshow_map(ax, arr, cmap, title, unit)

    fig2.suptitle("Seasonal model (invers_temp output)", fontsize=11)
    fig2.tight_layout()
    _savefig(fig2, save_dir, "check_seasonal_maps.png")
    if display:
        plt.show()
    plt.close(fig2)



# ─────────────────────────────────────────────────────────────────────────────
#  Burst images (display PNG images from AUX dir)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
#  Burst time-latitude images
# ─────────────────────────────────────────────────────────────────────────────
def plot_burst_images(aux_dir, save_dir, display):
    """Display plot_time_lat_iw*.png — one separate figure per subswath."""
    matches = sorted(glob.glob(os.path.join(aux_dir, "plot_time_lat_iw*.png")))
    if not matches:
        print("  plot_time_lat_iw*.png not found, skipping.")
        return
    for fpath in matches:
        stem = os.path.basename(fpath).replace(".png", "")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(plt.imread(fpath))
        ax.axis("off")
        fig.suptitle(stem, fontsize=10)
        fig.tight_layout()
        _savefig(fig, save_dir, f"check_{stem}.png")
        if display:
            plt.show()
        plt.close(fig)


def display_aux_images(aux_dir, save_dir, display):
    patterns = [
        ("plot_time_lat_iw*.png",           "Burst time-latitude distribution"),
        ("list_iw*_sd_cst_inverted_img.png","SD constant per image"),
        ("list_iw*_sd_phs_grad_inverted_img.png", "Phase gradient per image"),
        ("list_iw*_sd_sigma_inverted_img.png",    "SD sigma per image"),
        ("list_iw_merge_inverted_img.png",        "IW merge per image"),
        ("list_ramp_az_ra_sigma_inverted_img.png","Ramp az/ra sigma"),
        ("list_ramp_sigma_estimated_ifg.png",     "Ramp sigma per ifg"),
        ("list_unw_frac.png",                     "Unwrapping fraction"),
    ]
    for pat, title in patterns:
        matches = sorted(glob.glob(os.path.join(aux_dir, pat)))
        if not matches:
            continue
        n = len(matches)
        is_burst = "time_lat" in pat   # burst maps → 3 rows
        if is_burst and n > 1:
            fig, axes = plt.subplots(n, 1, figsize=(10, 5 * n))
        else:
            fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        axes = np.atleast_1d(axes).flatten()
        for ax, fpath in zip(axes, matches):
            img = plt.imread(fpath)
            ax.imshow(img)
            ax.set_title(os.path.basename(fpath), fontsize=7)
            ax.axis("off")
        fig.suptitle(title)
        stem = pat.replace("*", "iw").replace(".png", "")
        _savefig(fig, save_dir, f"check_img_{stem}.png")
        if display:
            plt.show()
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def _find_validation_dir(parent, swath_years):
    """
    Reuse an existing VALIDATION (or VALIDATION_*) directory under parent if
    one is already there — so repeated runs don't fragment figures across
    differently-named folders. Otherwise default to 'VALIDATION_<swath_years>'
    for a first run, e.g. 'VALIDATION_D019SUD-2021-2025'.
    """
    plain = os.path.join(parent, "VALIDATION")
    if os.path.isdir(plain):
        return plain
    existing = sorted(d for d in glob.glob(os.path.join(parent, "VALIDATION_*"))
                       if os.path.isdir(d))
    if existing:
        return existing[0]
    return os.path.join(parent, f"VALIDATION_{swath_years}")


def _resolve_dirs(track_dir_or_ts, aux_override=None, save_override=None):
    """
    Resolve TS, AUX and VALIDATION directories from a flexible input path.

    Accepted inputs for track_dir_or_ts:
      - Track directory  (e.g. data/Tienshan/D107_NORD)
        → TS  = <track>/TS
        → AUX = <track>/AUX
        → OUT = <track>/VALIDATION_<swath-years> (see _find_validation_dir)
      - TS directory explicitly (e.g. data/Tienshan/D107_NORD/TS)
        → AUX auto-detected (replace TS→AUX in parent, or find_aux_dir)
        → OUT = <parent>/VALIDATION_<swath-years>

    --aux and --save always override the auto-detection.
    """
    p = os.path.abspath(track_dir_or_ts)

    if os.path.basename(p) == "TS" or not os.path.isdir(os.path.join(p, "TS")):
        # Input is already the TS directory
        ts_dir  = p
        parent  = os.path.dirname(p)
    else:
        # Input is the track directory — TS and AUX are subdirectories
        ts_dir  = os.path.join(p, "TS")
        parent  = p

    # AUX: explicit override > sibling AUX/ > find_aux_dir heuristic
    if aux_override:
        aux_dir = os.path.abspath(aux_override)
    else:
        sibling_aux = os.path.join(parent, "AUX")
        if os.path.isdir(sibling_aux):
            aux_dir = sibling_aux
        else:
            aux_dir = U.find_aux_dir(ts_dir)

    # VALIDATION output dir
    save_dir = (os.path.abspath(save_override) if save_override
                else _find_validation_dir(parent, _derive_swath_years(ts_dir, parent)))

    return ts_dir, aux_dir, save_dir



# ─────────────────────────────────────────────────────────────────────────────
#  0  Prepare input files for invers_temp.py
# ─────────────────────────────────────────────────────────────────────────────
def prepare_inversion(ts_dir, aux_dir):
    """
    Build list_images.txt, inrms.txt and inaps.txt in ts_dir,
    ready for invers_temp.py.

    Equivalent awk steps:
      awk '{print 0,$1,0,$5,0,$2}' baseline.rsc > list_images.txt
      awk 'NR==FNR{a[$2];next} !($2 in a){print $2}' RMSdate.txt list_images.txt
      awk '{print $3}' RMSdate.txt > inrms.txt
      cp ../AUX/list_ramp_sigma_inverted_img.txt .
      awk '{print $2}' list_ramp_sigma_inverted_img.txt > inaps.txt
    """
    import shutil

    baseline_rsc  = os.path.join(aux_dir, "baseline.rsc")
    rmsdate_path  = os.path.join(ts_dir,  "RMSdate.txt")
    list_img_path = os.path.join(ts_dir,  "list_images.txt")
    inrms_path    = os.path.join(ts_dir,  "inrms.txt")
    ramp_src      = os.path.join(aux_dir, "list_ramp_sigma_inverted_img.txt")
    ramp_dst      = os.path.join(ts_dir,  "list_ramp_sigma_inverted_img.txt")
    inaps_path    = os.path.join(ts_dir,  "inaps.txt")

    # ── check required inputs ────────────────────────────────────────────────
    missing = [(p, lbl) for p, lbl in [(baseline_rsc, "AUX/baseline.rsc"),
                                        (rmsdate_path,  "TS/RMSdate.txt")]
               if not os.path.exists(p)]
    if missing:
        print("  WARNING: skipping inversion prep — missing files:")
        for _, lbl in missing:
            print(f"    {lbl}")
        return

    # ── step 1: baseline.rsc → list_images.txt ──────────────────────────────
    # baseline.rsc cols: YYYYMMDD  bperp  col3  col4  date_decimal  ...
    # output: 0  YYYYMMDD  0  date_decimal  0  bperp
    rows = []
    with open(baseline_rsc) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                int(parts[0]); float(parts[1]); float(parts[4])
                rows.append((parts[0], parts[4], parts[1]))
            except ValueError:
                continue
    if not rows:
        print("  WARNING: baseline.rsc has no valid rows, skipping.")
        return
    with open(list_img_path, "w") as f:
        for yyyymmdd, dec, bperp in rows:
            f.write(f"0 {yyyymmdd} 0 {dec} 0 {bperp}\n")
    print(f"  list_images.txt : {len(rows)} images")

    # ── step 2: find dates absent from RMSdate.txt ───────────────────────────
    rms_dates = set()
    with open(rmsdate_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                rms_dates.add(parts[1])

    kept, removed = [], []
    with open(list_img_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                (kept if parts[1] in rms_dates else removed).append(line)
    removed_dates = {l.split()[1] for l in removed}

    with open(list_img_path, "w") as f:
        f.writelines(kept)

    if removed_dates:
        print(f"  list_images.txt : {len(removed_dates)} date(s) removed (absent from RMSdate.txt):")
        for d in sorted(removed_dates):
            print(f"    - {d}")
    print(f"  list_images.txt : {len(kept)} images kept")

    # ── step 3: RMSdate.txt col 3 → inrms.txt ───────────────────────────────
    rms_vals = []
    with open(rmsdate_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                rms_vals.append(parts[2])
    with open(inrms_path, "w") as f:
        f.write("\n".join(rms_vals) + "\n")
    print(f"  inrms.txt       : {len(rms_vals)} values")

    # ── step 4: copy + filter list_ramp_sigma_inverted_img.txt ───────────────
    # format: decimal_date  sigma  YYYYMMDD — filter out removed_dates (col 2)
    if not os.path.exists(ramp_src):
        print("  WARNING: list_ramp_sigma_inverted_img.txt not found in AUX, skipping inaps.txt")
        return

    ramp_kept, ramp_removed = [], []
    with open(ramp_src) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                (ramp_removed if parts[2] in removed_dates else ramp_kept).append(line)

    with open(ramp_dst, "w") as f:
        f.writelines(ramp_kept)

    if ramp_removed:
        print(f"  list_ramp_sigma : {len(ramp_removed)} line(s) removed (same dates as above)")
    print(f"  list_ramp_sigma : {len(ramp_kept)} lines kept")

    # ── step 5: sigma col (col 1) → inaps.txt ────────────────────────────────
    aps_vals = []
    with open(ramp_dst) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                aps_vals.append(parts[1])
    with open(inaps_path, "w") as f:
        f.write("\n".join(aps_vals) + "\n")
    print(f"  inaps.txt       : {len(aps_vals)} values")

    # ── step 6: extract RMSpixel and Net bands from CNES_Net_geo_*.tiff ──────
    #
    #  Band descriptions (FLATSIM docs: products/Net.html):
    #    band 1 — RMS misclosure per pixel (rad)            → RMSpixel
    #    band 2 — nb of interferograms used per pixel       → Net_nifg
    #    band 3 — nb of images used per pixel               → Net_nimg
    #    band 4 — temporal coherence proxy (from triplets)  → Net_tcoh
    #    band 5 — bias proxy (rad)                          → Net_bias
    net_files = sorted(glob.glob(os.path.join(ts_dir, "CNES_Net_geo_*.tiff")))
    if net_files:
        try:
            from osgeo import gdal as _gdal
            net_path = net_files[0]
            ds = _gdal.Open(net_path)
            n_bands = ds.RasterCount
            band_defs = {
                1: ("RMSpixel",   "RMS misclosure (rad)"),
                2: ("Net_nifg",   "Nb interferograms used"),
                3: ("Net_nimg",   "Nb images used"),
                4: ("Net_tcoh",   "Temporal coherence proxy"),
                5: ("Net_bias",   "Bias proxy (rad)"),
            }
            driver = _gdal.GetDriverByName("GTiff")
            gt     = ds.GetGeoTransform()
            proj   = ds.GetProjection()
            ncols  = ds.RasterXSize
            nlines = ds.RasterYSize
            for b, (name, desc) in band_defs.items():
                if b > n_bands:
                    continue
                arr = ds.GetRasterBand(b).ReadAsArray().astype(np.float32)
                nd  = ds.GetRasterBand(b).GetNoDataValue()
                if nd is not None:
                    arr[arr == nd] = np.nan
                out_path = os.path.join(ts_dir, f"{name}.tif")
                out_ds = driver.Create(out_path, ncols, nlines, 1, _gdal.GDT_Float32)
                out_ds.GetRasterBand(1).WriteArray(arr)
                out_ds.SetGeoTransform(gt)
                out_ds.SetProjection(proj)
                out_ds.GetRasterBand(1).FlushCache()
                del out_ds
                print(f"  {name}.tif extracted  ({desc})")
            del ds
        except Exception as e:
            print(f"  WARNING: could not extract Net bands: {e}")
    else:
        print("  WARNING: CNES_Net_geo_*.tiff not found — RMSpixel not extracted")

    # ── step 7: print ready-to-run command ──────────────────────────────────
    cube = next((os.path.basename(p)
                 for p in sorted(glob.glob(os.path.join(ts_dir, "CNES_DTs_geo_*.tiff")))),
                "CNES_DTs_geo_8rlks.tiff")
    print(f"""
  Run from {ts_dir}:
    python invers_temp.py \\
        --cube={cube} \\
        --list_images=list_images.txt \\
        --rms=inrms.txt \\
        --aps=inaps.txt \\
        --linear=yes --seasonal=yes --niter=2 --plot=no
""")

# -----------------------------------------------------------------------------
#  Console logging - always mirror everything printed (including the
#  invers_temp.py subprocess output) to VALIDATION/check_results.log, in
#  addition to the terminal. Uses the real `tee` so subprocess output
#  (inherited file descriptors) is captured too, not just this process's
#  own print() calls.
# -----------------------------------------------------------------------------
def _start_logging(save_dir, log_name="check_results.log"):
    log_path = os.path.join(save_dir, log_name)
    try:
        tee_proc = subprocess.Popen(["tee", log_path], stdin=subprocess.PIPE, bufsize=0)
    except (FileNotFoundError, OSError):
        print(f"  (warning: 'tee' not available - {log_name} will not be written)")
        return None
    os.dup2(tee_proc.stdin.fileno(), 1)
    os.dup2(tee_proc.stdin.fileno(), 2)

    def _stop():
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            tee_proc.stdin.close()
            tee_proc.wait(timeout=5)
        except Exception:
            pass

    atexit.register(_stop)
    print(f"Log            : {log_path}")
    return log_path


def main():
    parser = argparse.ArgumentParser(
        description="FLATSIM TS validation — Python equivalent of check_results.sh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Pass the track directory (TS/ and AUX/ are found automatically):
  python check_results.py data/Tienshan/D107_NORD

  # Or pass the TS directory explicitly:
  python check_results.py data/Tienshan/D107_NORD/TS

  # Override AUX or output directory:
  python check_results.py data/Tienshan/D107_NORD --aux /other/path/AUX
""")
    parser.add_argument("track_dir",
                        help="Track directory (containing TS/ and AUX/) "
                             "or TS directory directly")
    parser.add_argument("--aux",    help="AUX directory (auto-detected if omitted)")
    parser.add_argument("--save",   help="Output directory (default: <track>/VALIDATION)")
    parser.add_argument("--no-display", action="store_true",
                        help="Do not call plt.show() (batch/headless mode)")
    parser.add_argument("--prepare", action="store_true",
                        help="Prepare input files for invers_temp.py "
                             "(list_images.txt, inrms.txt, inaps.txt) and exit")
    parser.add_argument("--no-inversion", action="store_true",
                        help="Skip automatic invers_temp.py run")
    parser.add_argument("--invers-args", default="",
                        help="Extra arguments passed to invers_temp.py "
                             "(e.g. '--niter=3 --ref_zone=100,400,200,800)')")
    parser.add_argument("--swath-name", default=None,
                        help="Short '<swath>[-<years>]' code used to name the "
                             "exported mm/mm-yr GeoTIFFs, e.g. 'D019SUD-2021-2025' "
                             "(auto-detected from the dataset directory name if omitted)")
    args = parser.parse_args()

    ts_dir, aux_dir, save_dir = _resolve_dirs(args.track_dir, args.aux, args.save)
    display = not args.no_display

    os.makedirs(save_dir, exist_ok=True)
    _start_logging(save_dir)

    print(f"TS  dir    : {ts_dir}")
    print(f"AUX dir    : {aux_dir}")
    print(f"Output dir : {save_dir}")

    # ── Step 0: prepare inversion inputs ────────────────────────────────────
    print("\n[0] Preparing invers_temp.py input files …")
    prepare_inversion(ts_dir, aux_dir)
    if args.prepare:
        print("--prepare: done. Exiting before validation plots.")
        return

    # ── Auto-run invers_temp.py (unless --no-inversion) ─────────────────────
    if not args.no_inversion:
        import subprocess, sys as _sys
        invers_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "invers_temp.py")
        if os.path.exists(invers_script):
            cmd = ([_sys.executable, invers_script, args.track_dir]
                   + (args.invers_args.split() if args.invers_args else []))
            print(f"\n[inversion] Running: {chr(32).join(cmd)}")
            subprocess.run(cmd, check=False)
        else:
            print(f"  invers_temp.py not found at {invers_script} — skipping.")

    # ── Run all checks ──────────────────────────────────────────────────────
    print("\n[1/15] Burst time-latitude images …")
    plot_burst_images(aux_dir, save_dir, display)

    print("\n[2/15] SD time-series plots …")
    plot_sd_group(aux_dir, save_dir, display)

    print("[3/15] IW merge plots …")
    plot_iw_merge(aux_dir, save_dir, display)

    print("[4/15] Interferogram statistics …")
    stats = compute_ifg_stats(ts_dir, aux_dir)
    print_stats(stats)

    print("[5/15] Bt histogram …")
    plot_bt_histogram(stats, save_dir, display)

    print("[6/15] Unwrapping fraction vs Bt / season …")
    plot_unw_frac_vs_bt(stats, save_dir, display)

    print("[7/15] RMS per date …")
    plot_rms_date(ts_dir, save_dir, display)

    print("[8/15] RMS per interferogram …")
    plot_rms_interfero(ts_dir, save_dir, display)

    print("[9/15] Variance comparison …")
    plot_variance_comparison(aux_dir, ts_dir, save_dir, display)

    print("[10/15] Interferogram network …")
    plot_ifg_network(ts_dir, aux_dir, save_dir, display)

    print("[11/15] Per-image APS (aps_N.txt) …")
    plot_sigma_vs_time(ts_dir, save_dir, display)

    print("[12/15] Median residual on reference zone …")
    plot_median_vs_time(ts_dir, save_dir, display)

    print("[13/15] Net quality maps (RMSpixel, nifg, nimg, tcoh, bias) …")
    plot_net_maps(ts_dir, save_dir, display)

    print("[14/15] Coefficient maps (lin, ampwt, phiwt) …")
    plot_coeff_maps(ts_dir, save_dir, display)
    write_los_mm_products(ts_dir, args.track_dir, swath_name=args.swath_name)

    print("[15/15] Velocity and seasonal maps …")
    plot_velocity_maps(ts_dir, save_dir, display)

    # print("[13/13] AUX PNG images …")
    # display_aux_images(aux_dir, save_dir, display)

    print("\nAll done.")


if __name__ == "__main__":
    main()