"""
utils.py — FLATSIM data reading utilities

Covers:
  - ENVI flat binary (.r4 / BIP cube)
  - FLATSIM text files: RMSdate.txt, RMSinterfero.txt, images_retenues,
    list_unw_frac.txt, list_ramp_*.txt, list_iw*_sd_*.txt, baseline.rsc
  - Path discovery helpers (TS dir → AUX dir)
"""

import os
import re
import glob
import numpy as np
from datetime import datetime, timedelta


# --------------------------------------------------------------------------- #
#  Path helpers
# --------------------------------------------------------------------------- #

def find_aux_dir(ts_dir):
    """
    Given the TS directory path, locate the corresponding AUX (DAUX) directory.
    Convention: replace '_TS' with '_DAUX' in the directory name and look one
    level up (same parent).

    Returns the AUX directory path, or raises FileNotFoundError.
    """
    ts_dir  = os.path.abspath(ts_dir)
    parent  = os.path.dirname(ts_dir)
    ts_name = os.path.basename(ts_dir)
    aux_name = ts_name.replace("_TS", "_DAUX")
    # also handle NSBAS_TS-PKG → NSBAS_DAUX-PKG naming
    aux_name = aux_name.replace("NSBAS_TS-PKG", "NSBAS_DAUX-PKG")

    candidate = os.path.join(parent, aux_name)
    if os.path.isdir(candidate):
        return candidate

    # fallback: search parent for a DAUX directory containing the track name
    track = _extract_trackname(ts_dir)
    for d in os.listdir(parent):
        if "DAUX" in d and track in d:
            return os.path.join(parent, d)

    raise FileNotFoundError(
        f"Cannot find AUX directory for '{ts_dir}'. Expected '{candidate}'."
    )


def _extract_trackname(path):
    """Extract track name from path (3rd '-'-delimited segment of basename)."""
    parts = os.path.basename(path).split("-")
    return parts[2] if len(parts) > 2 else "UNKNOWN"


def get_looks(ts_dir):
    """Return the looks suffix string, e.g. '_8rlks', from MV-LOS tiff."""
    matches = glob.glob(os.path.join(ts_dir, "CNES_MV-LOS_radar_*.tiff"))
    if not matches:
        return "_8rlks"
    fname  = os.path.basename(matches[0])
    parts  = fname.replace(".tiff", "").split("_")
    return "_" + parts[3]   # e.g. '8rlks' → '_8rlks'


# --------------------------------------------------------------------------- #
#  ENVI flat binary helpers
# --------------------------------------------------------------------------- #

def read_envi_hdr(hdr_path):
    info = {}
    with open(hdr_path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith(";"):
                k, v = line.split("=", 1)
                info[k.strip().lower()] = v.strip()
    return {
        "ncol"  : int(info["samples"]),
        "nlign" : int(info["lines"]),
        "ndat"  : int(info.get("bands", 1)),
    }


def read_r4(path, ncol, nlign):
    """Read single-band float32 ENVI image → (nlign, ncol) array."""
    return np.fromfile(path, dtype=np.float32).reshape(nlign, ncol)


def write_r4(path, arr):
    arr.astype(np.float32).tofile(path)


def read_bip_cube(path, ncol, nlign, ndat):
    """Read BIP float32 cube → (nlign, ncol, ndat) array."""
    return np.fromfile(path, dtype=np.float32).reshape(nlign, ncol, ndat)


# --------------------------------------------------------------------------- #
#  Text file readers
# --------------------------------------------------------------------------- #

def read_rmsdate(path):
    """
    Read RMSdate.txt.
    Format: idx  YYYYMMDD  rms_value
    Returns: list of (YYYYMMDD_str, rms).
    """
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                rows.append((parts[1], float(parts[2])))
    return rows


def read_rmsinterfero(path):
    """
    Read RMSinterfero.txt.
    Format: idx  YYYYMMDD1  YYYYMMDD2  rms_value
    Returns list of dicts with keys: date1, date2, rms.
    """
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                rows.append({
                    "date1": parts[1],
                    "date2": parts[2],
                    "rms"  : float(parts[3]),
                })
    return rows


def read_list_unw_frac(path):
    """
    Read list_unw_frac.txt.
    Format: YYYYMMDD1  YYYYMMDD2  fraction
    Returns list of dicts with keys: date1, date2, fraction.
    """
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                rows.append({
                    "date1"   : parts[0],
                    "date2"   : parts[1],
                    "fraction": float(parts[2]),
                })
    return rows


def read_list_ramp_sigma(path):
    """
    Read list_ramp_sigma_inverted_img.txt or list_ramp_sigma_estimated_ifg.txt.

    Two known layouts:
      - 3 columns: decimal_date  sigma  YYYYMMDD   (per-image, inverted)
      - 4+ columns: date1  date2  ...  sigma        (per-ifg, estimated)
    Returns list of dicts. Per-image rows have keys: date, yyyymmdd, sigma.
    Per-ifg rows have keys: date1, date2, sigma.
    """
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3:
                rows.append({
                    "date"    : parts[0],   # decimal
                    "yyyymmdd": parts[2],   # YYYYMMDD
                    "sigma"   : float(parts[1]),
                })
            elif len(parts) >= 4:
                rows.append({
                    "date1": parts[0],
                    "date2": parts[1],
                    "sigma": float(parts[3]),
                })
    return rows


def read_list_ramp_ra(path):
    """
    Read list_ramp_ra_estimated_ifg.txt.
    Format: YYYYMMDD1  YYYYMMDD2  ramp_ra  ...  (9 columns)
    Returns list of dicts with keys: date1, date2.
    """
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                rows.append({"date1": parts[0], "date2": parts[1]})
    return rows


def read_sd_txt(path):
    """
    Generic reader for list_iw*_sd_*.txt files.
    Format: decimal_date  value  err  idx  YYYYMMDD
    col 0 = decimal date, col 1 = value, col 4 = YYYYMMDD (when present).
    Returns: dates (list of YYYYMMDD strings), values (np.array).
    """
    dates, values = [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                # prefer YYYYMMDD (col 4) over decimal date (col 0)
                date = parts[4] if len(parts) >= 5 else parts[0]
                dates.append(date)
                values.append(float(parts[1]))
    return dates, np.array(values)


def read_baseline_rsc(path):
    """
    Read baseline.rsc.
    Each line: date  perp_baseline  [other_cols]
    Returns dict: {YYYYMMDD_str: perp_baseline_float}
    """
    bl = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5:
                # format from flatsim2mp.sh: 0 Im 0 date alevel base
                # but baseline.rsc may have different format
                bl[parts[0]] = float(parts[1])
    return bl


def read_images_retenues(path):
    """
    Read images_retenues (written by flatsim2mp.sh from baseline.rsc).
    Columns: 0  Im(YYYYMMDD)  0  date_decimal  alevel  baseline
    Returns list of dicts.
    """
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 6:
                rows.append({
                    "date_str"    : parts[1],
                    "date_decimal": float(parts[3]),
                    "baseline"    : float(parts[5]),
                })
    return rows


def read_list_merge(path):
    """Generic reader for list_iw*_merge_*.txt (date  value)."""
    return read_sd_txt(path)


# --------------------------------------------------------------------------- #
#  Date utilities
# --------------------------------------------------------------------------- #

def date_yyyymmdd_to_decimal(s):
    """Convert 'YYYYMMDD' string to decimal year (Fortran-compatible formula)."""
    idate = int(s)
    ww = idate // 10000
    y  = (idate - ww * 10000) // 100
    z  = idate - (ww * 10000 + y * 100)
    return ww + ((y - 1) * 30.5 + z) / 365.0


def date_str_to_datetime(s):
    return datetime.strptime(s, "%Y%m%d")


def bt_years(d1_str, d2_str):
    """Temporal baseline in decimal years between two YYYYMMDD strings."""
    return abs(date_yyyymmdd_to_decimal(d2_str) - date_yyyymmdd_to_decimal(d1_str))


def classify_bt(bt):
    """Return a label for a temporal baseline in years (same bins as check_results.sh)."""
    if bt > 0.8:   return "1yr"
    if bt > 0.4:   return "6mo"
    if bt > 0.2:   return "3mo"
    if bt > 0.15:  return "2mo"
    if bt > 0.075: return "1mo"
    if bt > 0.055: return "24d"
    if bt > 0.04:  return "18d"
    if bt > 0.025: return "12d"
    return "6d"


# --------------------------------------------------------------------------- #
#  Interferogram network helper
# --------------------------------------------------------------------------- #

def build_ifg_sets(ts_dir, aux_dir):
    """
    Build the three sets of interferograms used in check_results:
      - all_ifg_ori  : from list_unw_frac.txt  (all initially unwrapped)
      - ifg_after_ramp: from list_ramp_ra_estimated_ifg.txt (passed unw threshold)
      - ifg_kept     : from RMSinterfero.txt   (used in time series)
    Each set is a set of (date1, date2) tuples.
    """
    unw_frac = read_list_unw_frac(os.path.join(aux_dir, "list_unw_frac.txt"))
    ramp_ra  = read_list_ramp_ra(os.path.join(aux_dir, "list_ramp_ra_estimated_ifg.txt"))
    rms_ifg  = read_rmsinterfero(os.path.join(ts_dir, "RMSinterfero.txt"))

    all_ori      = {(r["date1"], r["date2"]) for r in unw_frac}
    after_ramp   = {(r["date1"], r["date2"]) for r in ramp_ra}
    kept         = {(r["date1"], r["date2"]) for r in rms_ifg}

    return all_ori, after_ramp, kept
