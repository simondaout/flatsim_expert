#!/usr/bin/env python3
"""
check_results_inc.py — Compare a new FLATSIM TS run against a previous one.

Usage
-----
    # Pass track directories (TS/ and AUX/ found automatically):
    python check_results_inc.py <new_track_dir> <prev_track_dir>
    python check_results_inc.py data/Tienshan/D107_NORD data/Tienshan/D107_NORD/INCREMENT

    # Or pass TS directories directly:
    python check_results_inc.py data/.../TS data/.../INCREMENT/TS

    # Explicit AUX override:
    python check_results_inc.py data/.../D107_NORD data/.../INCREMENT \
        --aux /path/to/new/AUX --prev-aux /path/to/prev/AUX

Arguments
---------
  track_dir       New track or TS directory.
  prev_track_dir  Previous track or TS directory.
  --aux           AUX dir for new run (auto-detected as sibling AUX/ if omitted).
  --prev-aux      AUX dir for previous run (auto-detected if omitted).
  --save          Output directory (default: <new_track>/VALIDATION/).
  --no-display    Headless/batch mode — do not call plt.show().

Directory convention
--------------------
  data/
  └── <site>/<track>/
      ├── TS/           new time series
      ├── AUX/          new auxiliary data
      ├── VALIDATION/   figures written here
      └── INCREMENT/
          ├── TS/       previous time series
          └── AUX/      previous auxiliary data (if different)

Comparison plots (new = blue, previous = red, offset-aligned on common dates)
------------------------------------------------------------------------------
  1.  SD variation along range          (list_iw*_sd_sx)
  2.  SD variation along azimuth        (list_iw*_sd_sy)
  3.  Quadratic SD along azimuth        (list_iw*_sd_qy)
  4.  SD tôle ondulée                   (list_iw*_sd_syy)
  5.  SD sigma                          (list_iw*_sd_sigma)
  6.  SD constant term                  (list_iw*_sd_cst)
  7.  IW phase jumps                    (list_iw12/23_merge)
  8.  Ramp in range / azimuth / sigma   (list_ramp_*)
  9.  Bt histogram side-by-side
 10.  Interferogram statistics for both runs (printed to console)
 11.  RMS per date
 12.  RMS per ifg vs Bt
 13.  Per-image uncertainty σ vs time   (sigma_N.txt, all iterations)
      Coefficient maps for new run      (lin_coeff, ampwt_coeff, phiwt_coeff)
      AUX PNG images for both runs
"""

import os
import sys
import argparse
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import utils as U
from check_results import (
    plot_bt_histogram,
    plot_unw_frac_vs_bt,
    plot_rms_date,
    plot_rms_interfero,
    compute_ifg_stats,
    print_stats,
    _dec,
    _bt_yr,
    _savefig,
    BT_BINS,
    _count_bt_bins,
    plot_sigma_vs_time,
    plot_coeff_maps,
    _load_list_images,
    display_aux_images,
    _resolve_dirs,
)

matplotlib.rcParams.update({
    "figure.dpi"      : 120,
    "axes.titlesize"  : 10,
    "axes.labelsize"  : 9,
    "xtick.labelsize" : 8,
    "ytick.labelsize" : 8,
    "legend.fontsize" : 8,
})

IW_COLORS = ["tab:blue", "tab:orange", "tab:green"]
IW_LABELS = ["IW1", "IW2", "IW3"]


# ─────────────────────────────────────────────────────────────────────────────
#  Align offset between old and new SD series on common dates
# ─────────────────────────────────────────────────────────────────────────────
def _align_series(new_dates, new_vals, old_dates, old_vals):
    """
    Compute the mean offset between old and new on common dates and return
    offset-corrected old series, ready to overlay on the new.

    Returns: old_dates_aligned, old_vals_aligned (with mean offset removed)
    """
    new_map = dict(zip(new_dates, new_vals))
    old_map = dict(zip(old_dates, old_vals))
    common  = set(new_map) & set(old_map)
    if not common:
        print("    (no common dates for alignment)")
        return old_dates, old_vals

    offsets = [new_map[d] - old_map[d] for d in common]
    mean_off = np.mean(offsets)
    old_vals_aligned = np.array(old_vals) + mean_off
    return old_dates, old_vals_aligned


# ─────────────────────────────────────────────────────────────────────────────
#  SD comparison plots (with offset alignment)
# ─────────────────────────────────────────────────────────────────────────────
SD_GROUPS = [
    ("sd_cst",  "SD constant term (offset-aligned)",   "SD cst"),
    ("sd_sx",   "SD variation along range",            "SD (range)"),
    ("sd_sy",   "SD variation along azimuth",          "SD (azimuth)"),
    ("sd_qy",   "Quadratic SD along azimuth (cos)",    "SD (quad az)"),
    ("sd_syy",  "SD tôle ondulée",                     "SD (syy)"),
    ("sd_sigma","SD standard (sigma)",                 "SD sigma"),
]


def plot_sd_comparison(aux_dir, prev_aux_dir, save_dir, display):
    for tag, title, ylabel in SD_GROUPS:
        fig, ax = plt.subplots(figsize=(12, 5))
        any_data = False

        for iw, (col, lbl) in enumerate(zip(IW_COLORS, IW_LABELS)):
            # New run
            fnew = os.path.join(aux_dir, f"list_iw{iw+1}_{tag}_inverted_img.txt")
            # Previous run
            fold = os.path.join(prev_aux_dir, f"list_iw{iw+1}_{tag}_inverted_img.txt")

            if not os.path.exists(fnew):
                continue
            ndates, nvals = U.read_sd_txt(fnew)
            dec_new = [_dec(d) for d in ndates]
            ax.plot(dec_new, nvals, "o", color=col, label=f"{lbl} (new)",
                    markersize=4)
            any_data = True

            if os.path.exists(fold):
                odates, ovals = U.read_sd_txt(fold)
                _, ovals_al   = _align_series(ndates, nvals, odates, ovals)
                dec_old = [_dec(d) for d in odates]
                ax.plot(dec_old, ovals_al, "x", color=col,
                        label=f"{lbl} (prev, aligned)", markersize=4,
                        alpha=0.6, linestyle="--")

        if not any_data:
            plt.close(fig)
            continue

        ax.set_title(title)
        ax.set_xlabel("Date (decimal year)")
        ax.set_ylabel(ylabel)
        ax.legend(ncol=2, fontsize=7)
        ax.grid(True, alpha=0.3)
        _savefig(fig, save_dir, f"inc_{tag}.png")
        if display:
            plt.show()
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  IW merge (phase jumps) comparison
# ─────────────────────────────────────────────────────────────────────────────
def plot_iw_merge_comparison(aux_dir, prev_aux_dir, save_dir, display):
    fig, ax = plt.subplots(figsize=(12, 5))
    any_data = False

    for pair, col, lbl in [("12", "tab:blue", "IW1-IW2"), ("23", "tab:orange", "IW2-IW3")]:
        fnew = os.path.join(aux_dir,      f"list_iw{pair}_merge_inverted_img.txt")
        fold = os.path.join(prev_aux_dir, f"list_iw{pair}_merge_inverted_img.txt")

        if not os.path.exists(fnew):
            continue
        ndates, nvals = U.read_sd_txt(fnew)
        ax.plot([_dec(d) for d in ndates], nvals, "o",
                color=col, label=f"{lbl} (new)", markersize=4)
        any_data = True

        if os.path.exists(fold):
            odates, ovals = U.read_sd_txt(fold)
            _, ovals_al   = _align_series(ndates, nvals, odates, ovals)
            ax.plot([_dec(d) for d in odates], ovals_al, "x",
                    color=col, label=f"{lbl} (prev, aligned)",
                    markersize=4, alpha=0.6, linestyle="--")

    if not any_data:
        plt.close(fig)
        return

    ax.set_title("Phase jumps across subswath limits — new vs previous")
    ax.set_xlabel("Date (decimal year)")
    ax.set_ylabel("Phase jump (rad)")
    ax.legend(ncol=2, fontsize=7)
    ax.grid(True, alpha=0.3)
    _savefig(fig, save_dir, "inc_iw_merge.png")
    if display:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Ramp comparisons (range / azimuth / sigma)
# ─────────────────────────────────────────────────────────────────────────────
RAMP_FILES = [
    ("list_ramp_ra_inverted_img.txt",    "Ramps in range",          "ramp_ra"),
    ("list_ramp_az_inverted_img.txt",    "Ramps in azimuth",        "ramp_az"),
    ("list_ramp_sigma_inverted_img.txt", "Ramp sigma per image",    "ramp_sigma"),
]


def plot_ramp_comparison(aux_dir, prev_aux_dir, save_dir, display):
    for fname, title, stem in RAMP_FILES:
        fnew = os.path.join(aux_dir,      fname)
        fold = os.path.join(prev_aux_dir, fname)
        if not os.path.exists(fnew):
            continue

        ndates, nvals = U.read_sd_txt(fnew)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot([_dec(d) for d in ndates], nvals, "o-",
                color="steelblue", label="new", markersize=4)

        if os.path.exists(fold):
            odates, ovals = U.read_sd_txt(fold)
            _, ovals_al   = _align_series(ndates, nvals, odates, ovals)
            ax.plot([_dec(d) for d in odates], ovals_al, "x--",
                    color="tomato", label="previous (aligned)",
                    markersize=4, alpha=0.8)

        ax.set_title(f"{title} — new vs previous")
        ax.set_xlabel("Date (decimal year)")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        _savefig(fig, save_dir, f"inc_{stem}.png")
        if display:
            plt.show()
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  RMS per date comparison
# ─────────────────────────────────────────────────────────────────────────────
def plot_rms_date_comparison(ts_dir, prev_ts_dir, save_dir, display):
    def _load(d):
        f = os.path.join(d, "RMSdate.txt")
        if not os.path.exists(f):
            return [], []
        rows = U.read_rmsdate(f)
        return [_dec(r[0]) for r in rows], [r[1] for r in rows]

    nd, nv = _load(ts_dir)
    od, ov = _load(prev_ts_dir)

    fig, ax = plt.subplots(figsize=(12, 4))
    if nd:
        ax.plot(nd, nv, "o-", color="steelblue", markersize=4, label="new")
    if od:
        ax.plot(od, ov, "x--", color="tomato",   markersize=4, label="previous", alpha=0.8)
    ax.set_xlabel("Date (decimal year)")
    ax.set_ylabel("RMS (rad)")
    ax.set_title("Per-date RMS — new vs previous")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _savefig(fig, save_dir, "inc_rms_date.png")
    if display:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  RMS per interferogram vs Bt comparison
# ─────────────────────────────────────────────────────────────────────────────
def plot_rms_ifg_comparison(ts_dir, prev_ts_dir, save_dir, display):
    def _load(d):
        f = os.path.join(d, "RMSinterfero.txt")
        if not os.path.exists(f):
            return [], []
        rows = U.read_rmsinterfero(f)
        bt   = [_bt_yr(r["date1"], r["date2"]) for r in rows]
        rms  = [r["rms"] for r in rows]
        return bt, rms

    nbt, nrms = _load(ts_dir)
    obt, orms = _load(prev_ts_dir)

    fig, ax = plt.subplots(figsize=(9, 5))
    if obt:
        ax.scatter(obt, orms, s=10, alpha=0.5, color="tomato", label="previous")
    if nbt:
        ax.scatter(nbt, nrms, s=10, alpha=0.7, color="steelblue", label="new")
    ax.axhline(0.9, color="red", ls="--", lw=0.8, label="RMS = 0.9")
    ax.set_xlabel("Temporal baseline (yr)")
    ax.set_ylabel("RMS (rad)")
    ax.set_title("Interferogram RMS vs Bt — new vs previous")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _savefig(fig, save_dir, "inc_rms_ifg_vs_bt.png")
    if display:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Bt histogram comparison
# ─────────────────────────────────────────────────────────────────────────────
def plot_bt_histogram_comparison(ts_dir, prev_ts_dir, save_dir, display):
    def _load_kept(d):
        f = os.path.join(d, "RMSinterfero.txt")
        if not os.path.exists(f):
            return []
        return U.read_rmsinterfero(f)

    new_kept  = _load_kept(ts_dir)
    prev_kept = _load_kept(prev_ts_dir)

    new_counts  = _count_bt_bins(new_kept)
    prev_counts = _count_bt_bins(prev_kept)

    labels = [c[0] for c in new_counts]
    x      = np.arange(len(labels))
    w      = 0.38

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w/2, [c[1] for c in prev_counts], w, label="previous", color="tomato",   alpha=0.8)
    ax.bar(x + w/2, [c[1] for c in new_counts],  w, label="new",      color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of interferograms")
    ax.set_title("Kept interferograms by Bt — new vs previous")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, save_dir, "inc_histo_bt.png")
    if display:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Sigma comparison (per-image uncertainty across iterations)
# ─────────────────────────────────────────────────────────────────────────────
def plot_sigma_comparison(ts_dir, prev_ts_dir, save_dir, display):
    """Overlay sigma_N.txt curves for new (blue) and previous (red) runs."""
    import glob as _glob
    fig, ax = plt.subplots(figsize=(12, 4))
    any_data = False

    for run_dir, base_col, run_lbl in [
        (prev_ts_dir, "tomato",    "previous"),
        (ts_dir,      "steelblue", "new"),
    ]:
        sigma_files = sorted(
            _glob.glob(os.path.join(run_dir, "sigma_*.txt")),
            key=lambda p: int(os.path.basename(p).replace("sigma_","").replace(".txt",""))
        )
        dates = _load_list_images(run_dir)
        if not sigma_files or not dates:
            continue
        dec_dates = [_dec(d) for d in dates]
        alphas = np.linspace(0.5, 1.0, len(sigma_files))
        for fpath, alpha in zip(sigma_files, alphas):
            sigma = np.loadtxt(fpath)
            niter = os.path.basename(fpath).replace("sigma_","").replace(".txt","")
            n = min(len(dec_dates), len(sigma))
            ax.plot(dec_dates[:n], sigma[:n], "o-", color=base_col,
                    markersize=3, linewidth=0.8, alpha=alpha,
                    label=f"{run_lbl} iter {niter}")
        any_data = True

    if not any_data:
        plt.close(fig)
        return
    ax.set_xlabel("Date (decimal year)")
    ax.set_ylabel("sigma (per-image uncertainty)")
    ax.set_title("Per-image uncertainty vs time — new vs previous")
    ax.legend(ncol=2, fontsize=7)
    ax.grid(True, alpha=0.3)
    _savefig(fig, save_dir, "inc_sigma_vs_time.png")
    if display:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Common ifg / date counts
# ─────────────────────────────────────────────────────────────────────────────
def print_common_stats(ts_dir, prev_ts_dir, aux_dir, prev_aux_dir):
    # Common dates (before unwrapping)
    sd_new  = os.path.join(aux_dir,      "list_iw1_sd_cst_inverted_img.txt")
    sd_prev = os.path.join(prev_aux_dir, "list_iw1_sd_cst_inverted_img.txt")
    if os.path.exists(sd_new) and os.path.exists(sd_prev):
        dn, _ = U.read_sd_txt(sd_new)
        dp, _ = U.read_sd_txt(sd_prev)
        common_dates = set(dn) & set(dp)
        print(f"  Common dates (before unwrapping) : {len(common_dates)}")

    # Common ifg (in TS)
    rms_new  = U.read_rmsinterfero(os.path.join(ts_dir,      "RMSinterfero.txt"))
    rms_prev = U.read_rmsinterfero(os.path.join(prev_ts_dir, "RMSinterfero.txt"))
    new_set  = {(r["date1"], r["date2"]) for r in rms_new}
    prev_set = {(r["date1"], r["date2"]) for r in rms_prev}
    print(f"  Common interferograms (in TS)    : {len(new_set & prev_set)}")
    print(f"  New-only interferograms          : {len(new_set - prev_set)}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="FLATSIM incremental TS comparison — Python equivalent of check_results_inc.sh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Pass the two track directories (TS/ and AUX/ found automatically):
  python check_results_inc.py data/Tienshan/D107_NORD data/Tienshan/D107_NORD/INCREMENT

  # Or pass TS directories explicitly:
  python check_results_inc.py data/.../TS data/.../INCREMENT/TS
""")
    parser.add_argument("track_dir",      help="New track or TS directory")
    parser.add_argument("prev_track_dir", help="Previous track or TS directory")
    parser.add_argument("--aux",          help="AUX dir for new run (auto-detected if omitted)")
    parser.add_argument("--prev-aux",     help="AUX dir for previous run (auto-detected if omitted)")
    parser.add_argument("--save",         help="Output dir (default: <new_track>/VALIDATION)")
    parser.add_argument("--no-display",   action="store_true", help="Headless/batch mode")
    args = parser.parse_args()

    ts_dir,      aux_dir,      save_dir = _resolve_dirs(args.track_dir,      args.aux,      args.save)
    prev_ts_dir, prev_aux_dir, _        = _resolve_dirs(args.prev_track_dir, args.prev_aux, None)
    display = not args.no_display

    os.makedirs(save_dir, exist_ok=True)

    print(f"New  TS  dir : {ts_dir}")
    print(f"Prev TS  dir : {prev_ts_dir}")
    print(f"New  AUX dir : {aux_dir}")
    print(f"Prev AUX dir : {prev_aux_dir}")
    print(f"Output dir   : {save_dir}")

    print("\n[stats] Common dates / ifg:")
    print_common_stats(ts_dir, prev_ts_dir, aux_dir, prev_aux_dir)

    print("\n[1/9] SD comparison plots …")
    plot_sd_comparison(aux_dir, prev_aux_dir, save_dir, display)

    print("[2/9] IW merge comparison …")
    plot_iw_merge_comparison(aux_dir, prev_aux_dir, save_dir, display)

    print("[3/9] Ramp comparison …")
    plot_ramp_comparison(aux_dir, prev_aux_dir, save_dir, display)

    print("[4/9] Bt histogram comparison …")
    plot_bt_histogram_comparison(ts_dir, prev_ts_dir, save_dir, display)

    print("[5/9] Interferogram statistics …")
    print("\n  NEW RUN:")
    print_stats(compute_ifg_stats(ts_dir, aux_dir))
    print("  PREVIOUS RUN:")
    print_stats(compute_ifg_stats(prev_ts_dir, prev_aux_dir))

    print("[6/9] RMS per date comparison …")
    plot_rms_date_comparison(ts_dir, prev_ts_dir, save_dir, display)

    print("[7/9] RMS per ifg vs Bt comparison …")
    plot_rms_ifg_comparison(ts_dir, prev_ts_dir, save_dir, display)

    print("[8/10] Sigma (per-image uncertainty) comparison …")
    plot_sigma_comparison(ts_dir, prev_ts_dir, save_dir, display)

    print("[9/10] Coefficient maps (lin, ampwt, phiwt) …")
    plot_coeff_maps(ts_dir, save_dir, display)

    print("[10/10] AUX burst images …")
    display_aux_images(aux_dir,      save_dir, display)
    display_aux_images(prev_aux_dir, save_dir, display)

    print("\nAll done.")


if __name__ == "__main__":
    main()
