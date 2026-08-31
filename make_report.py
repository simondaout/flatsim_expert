#!/usr/bin/env python3
"""
make_report.py — build a single-page HTML report for a FLATSIM check_results.py run.

Usage
-----
    python make_report.py data/Turquie/A131

    # with the validation-summary stats (recommended): first save the console
    # output of check_results.py to a log file next to the track, e.g.
    #
    #   python check_results.py data/Turquie/A131/TS | tee data/Turquie/A131/check_results.log
    #
    # then:
    python make_report.py data/Turquie/A131

The script looks for VALIDATION/*.png inside <track_dir> and writes
<track_dir>/VALIDATION/report.html next to them, with plain relative
<img src="..."> links (no base64 embedding) — so the VALIDATION/ folder
(figures + report.html) is a self-contained bundle you can zip and send,
no upload anywhere required.

Options
-------
    --val-dir DIR   override the validation figures directory
                    (default: <track_dir>/VALIDATION)
    --log FILE      explicit path to a saved check_results.py console log,
                    used to fill in the summary stats / flagged interferograms.
                    If omitted, the script looks for *.log directly inside
                    <track_dir>; if none is found the report is generated
                    without the stats overview (figures only).
    --out FILE      override the output path (default: <val-dir>/report.html)
"""
import argparse
import glob
import os
import re

# Set this to the project's GitHub URL to show a "relancer sur GitHub" link
# in the report sidebar (left empty = no link shown).
REPO_URL = "https://github.com/simondaout/flatsim_expert"

# ---------------------------------------------------------------------------
# Figure catalogue: filename -> (pipeline step, section key, caption)
# Mirrors the table in README.md. Missing files are simply skipped, so this
# works whether check_results.py was run to completion or with --prepare /
# --no-inversion (fewer figures).
# ---------------------------------------------------------------------------
FIGURES = [
    ("check_plot_time_lat_iw1.png", "01/15", "burst", "Burst time–latitude — IW1"),
    ("check_plot_time_lat_iw2.png", "01/15", "burst", "Burst time–latitude — IW2"),
    ("check_plot_time_lat_iw3.png", "01/15", "burst", "Burst time–latitude — IW3"),

    ("check_sd_sx.png", "02/15", "sd", "Systematic deviation along range"),
    ("check_sd_sy.png", "02/15", "sd", "Systematic deviation along azimuth"),
    ("check_sd_qy.png", "02/15", "sd", "Quadratic SD along azimuth"),
    ("check_sd_syy.png", "02/15", "sd", "SD — tôle ondulée (azimuth corrugation)"),
    ("check_sd_sigma.png", "02/15", "sd", "SD — sigma"),
    ("check_sd_cst.png", "02/15", "sd", "SD — constant term"),

    ("check_iw_merge.png", "03/15", "merge", "Phase jumps, IW1–IW2 and IW2–IW3"),

    ("check_histo_bt.png", "05/15", "network", "Kept interferograms by temporal baseline"),
    ("check_unw_frac_vs_bt.png", "06/15", "network", "Unwrapping fraction vs. temporal baseline"),
    ("check_unw_frac_vs_season.png", "06/15", "network", "Unwrapping fraction vs. season"),
    ("check_rms_date.png", "07/15", "network", "RMS per date"),
    ("check_rms_ifg_vs_bt.png", "08/15", "network", "Interferogram RMS vs. Bt"),
    ("check_variance_comparison.png", "09/15", "network", "Per-interferogram variance vs. per-image APS"),
    ("check_ifg_network.png", "10/15", "network", "Interferogram network (kept vs. removed)"),

    ("check_sigma_vs_time.png", "11/15", "inversion", "Per-image APS vs. time"),
    ("check_median_vs_time.png", "12/15", "inversion", "Median residual, reference zone, vs. time"),

    ("check_net_maps.png", "13/15", "maps", "Network quality — RMSpixel, nifg, nimg, tcoh, bias"),
    ("check_coeff_maps.png", "14/15", "maps", "Inversion coefficients — lin, ampwt, phiwt, ref"),
    ("check_velocity_maps.png", "15/15", "maps", "Velocity — FLATSIM MV-LOS vs. lin_coeff vs. RMSpixel"),
    ("check_seasonal_maps.png", "15/15", "maps", "Seasonal model — amplitude, phase, cos, sin"),
]

SECTIONS = [
    ("burst", "Burst coverage", "STEP 01/15"),
    ("sd", "SD stability", "STEP 02/15"),
    ("merge", "IW merge", "STEP 03/15"),
    ("network", "Network health", "STEPS 05–10/15"),
    ("inversion", "Inversion diagnostics", "STEPS 11–12/15"),
    ("maps", "Output maps", "STEPS 13–15/15"),
]


def fmt_date(d):
    return f"{d[0:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 else d


# ---------------------------------------------------------------------------
# Log parsing (optional) — pulls the validation summary block, the flagged
# high-RMS interferograms, and the total run duration out of a saved
# check_results.py console log. Nothing here is required for the report to
# build; sections are just omitted if the log is absent or a piece isn't found.
# ---------------------------------------------------------------------------
def parse_log(text):
    stats = {}

    # the banner delimiter is a long run of '=' (the data lines only ever contain
    # single '=' inside "max=" / "min=", so requiring 10+ avoids matching those)
    m = re.search(r"FLATSIM validation summary\s*={10,}\s*(.*?)\s*={10,}", text, re.S)
    if m:
        block = m.group(1)
        for line in block.splitlines():
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key, val = key.strip(), val.strip()
            if not key or not val:
                continue
            mm = re.match(r"([\d.]+)\s+max=([\d.]+)\s+min=([\d.]+)", val)
            if mm:
                stats[key] = {"mean": mm.group(1), "max": mm.group(2), "min": mm.group(3)}
            else:
                stats[key] = val

    flagged = []
    m = re.search(
        r"Ifg with RMS\s*>\s*([\d.]+)\s*\(likely unwrapping errors\):\s*(.*?)(?:\n\s*→|\n\s*\[)",
        text, re.S,
    )
    if m:
        stats["_rms_threshold"] = m.group(1)
        for line in m.group(2).splitlines():
            mm = re.match(r"\s*(\d{8})\s*[–-]\s*(\d{8})\s+RMS=([\d.]+)", line)
            if mm:
                flagged.append((mm.group(1), mm.group(2), float(mm.group(3))))
    flagged.sort(key=lambda r: -r[2])

    m = re.search(r"Done in ([\d.]+)s", text)
    if m:
        secs = float(m.group(1))
        h, rem = divmod(int(secs), 3600)
        mn, sc = divmod(rem, 60)
        stats["_duration"] = f"{h}h{mn:02d}min ({secs:.1f}s)" if h else f"{mn}min{sc:02d}s ({secs:.1f}s)"

    return stats, flagged


def find_dataset_dir(track_dir):
    """
    Best-effort: the NSBAS_TS-PKG_* run directory that sits next to VALIDATION/.
    A track directory can also hold a NSBAS_DAUX-PKG_* delivery (auxiliary
    products) — TS-PKG is preferred since that's the one check_results.py
    actually processes; DAUX-PKG is only used as a fallback.
    """
    names = [n for n in sorted(os.listdir(track_dir))
             if n.startswith("NSBAS_") and os.path.isdir(os.path.join(track_dir, n))]
    for n in names:
        if "TS-PKG" in n:
            return n
    return names[0] if names else None


def parse_dataset_name(name):
    """
    Turn a dataset directory name such as
      NSBAS_TS-PKG_S1_TURQUIE-A131-VV-2-2020-2025_IW123_2020-05-03_2025-08-29
    into a readable sentence: satellite, polarization, subswaths, date range.
    Best-effort — any piece that doesn't match is simply left out.
    """
    parts = {}

    # underscores/hyphens are word characters in Python's regex, so a plain
    # \b...\b fails at the boundary of e.g. "_S1_" — match the separators explicitly
    m = re.search(r"(?:^|[_-])S1[AB]?(?:[_-]|$)", name)
    if m:
        parts["sat"] = "Sentinel-1"

    m = re.search(r"-(VV|VH|HH|HV)-", name)
    if m:
        parts["pol"] = m.group(1)

    m = re.search(r"(?:^|[_-])IW(\d{1,3})(?:[_-]|$)", name)
    if m:
        parts["iw"] = "-".join(f"IW{d}" for d in m.group(1))

    dates = re.findall(r"\d{4}-\d{2}-\d{2}", name)
    if dates:
        parts["date_start"] = dates[0]
        parts["date_end"] = dates[-1]

    return parts


def build_subtitle(dataset_dir, track_dir):
    if not dataset_dir:
        return f"FLATSIM validation — {track_dir}"
    p = parse_dataset_name(dataset_dir)
    bits = []
    if p.get("sat"):
        bits.append(p["sat"])
    if p.get("pol"):
        bits.append(f'polarisation {p["pol"]}')
    if p.get("iw"):
        bits.append(f'sous-fauchées {p["iw"]}')
    if p.get("date_start") and p.get("date_end"):
        bits.append(f'{p["date_start"]} &rarr; {p["date_end"]}')
    return " &middot; ".join(bits) if bits else dataset_dir


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------
def figure_card(fname, step, caption):
    return f'''<figure class="fig-card">
          <div class="fig-eyebrow"><span class="step">STEP {step}</span></div>
          <img src="{fname}" alt="{caption}" loading="lazy" onclick="openLightbox(this)">
          <figcaption>{caption}</figcaption>
        </figure>'''


def build_overview(stats, flagged, title, subtitle, meta_line, path_line):
    if not stats:
        return f'''<header class="report-head" id="overview">
      <div class="eyebrow">FLATSIM &middot; validation Sentinel-1 &middot; check_results.py</div>
      <h1 class="title">{title}</h1>
      <div class="subtitle">{subtitle}</div>
      {f'<div class="path-line">{path_line}</div>' if path_line else ''}
      <p class="no-stats">Pas de log fourni &mdash; relance avec <code class="mono">--log</code>
      (ou un fichier <code class="mono">*.log</code> dans le dossier de la track) pour inclure
      le résumé de validation et les interférogrammes flagués.</p>
    </header>
    <section></section>'''

    def tile(label, value, sub="", flag=False):
        cls = "tile flag" if flag else "tile"
        subhtml = f'<div class="sub">{sub}</div>' if sub else ""
        flaghtml = '<div class="flag-note">&lt; seuil 0.5</div>' if flag else ""
        return f'''<div class="{cls}"><div class="label">{label}</div>
          <div class="value num">{value}</div>{subhtml}{flaghtml}</div>'''

    unwrapped = stats.get("Number of unwrapped ifg", "—")
    kept = stats.get("Kept in time series", "—")
    rm_low = stats.get("Removed (unwrapping fraction low)", "0")
    rm_var = stats.get("Removed (large variance)", "0")
    oneyr_kept = stats.get("1-yr ifg kept", "—")
    oneyr_rm = stats.get("1-yr ifg removed", "0")
    unw = stats.get("Avg unwrapping fraction (kept)")
    var = stats.get("Avg variance (kept)")
    images_ts = stats.get("Images in time series", "—")
    images_ini = stats.get("Images processed (ini)", "—")

    # "Number of merged ifg" (list_iw12_merge_estimated_ifg.txt) is deliberately
    # not shown: it spans the full Sentinel-1 archive for the track (since 2014),
    # not this product's processing window, so it isn't comparable to the other
    # counts here — unwrapping quality is already covered by the tiles below.
    tiles = [tile("Ifg déroulés", unwrapped)]
    tiles.append(tile("Gardés en série", kept))
    if isinstance(unw, dict):
        low = float(unw["mean"]) < 0.5
        tiles.append(tile("Fraction déroul. moy.", unw["mean"],
                           f'max {unw["max"]} &middot; min {unw["min"]}', flag=low))
    if isinstance(var, dict):
        tiles.append(tile("Variance moy.", var["mean"], f'max {var["max"]} &middot; min {var["min"]}'))
    tiles.append(tile("Ifg 1 an gardés", oneyr_kept, f"{oneyr_rm} retirés"))

    funnel = (
        f'<div class="funnel"><b class="num">{unwrapped}</b> déroulés <span class="arrow">&rarr;</span> '
        f'<span class="minus num">&minus;{rm_low}</span> frac. faible <span class="arrow">&rarr;</span> '
        f'<span class="minus num">&minus;{rm_var}</span> variance forte <span class="arrow">&rarr;</span> '
        f'<span class="final num">{kept} gardés</span></div>'
    )

    callout = ""
    if flagged:
        rows = "\n".join(
            f'<tr><td class="mono">{fmt_date(a)}</td><td class="mono">{fmt_date(b)}</td>'
            f'<td class="mono num">{rms:.3f}</td></tr>'
            for a, b, rms in flagged
        )
        thr = stats.get("_rms_threshold", "0.9")
        callout = f'''<div class="callout">
        <h3>{len(flagged)} interférogrammes flagués &mdash; RMS &gt; {thr}</h3>
        <table class="flag-table">
          <thead><tr><th>Date 1</th><th>Date 2</th><th>RMS</th></tr></thead>
          <tbody>
{rows}
          </tbody>
        </table>
      </div>'''

    duration = stats.get("_duration")
    extra_meta = f'<span><b>Durée inversion</b> {duration}</span>' if duration else ""
    images_meta = f'<span><b>Images</b> {images_ts} / {images_ini}</span>' if images_ts != "—" else ""

    header = f'''<header class="report-head" id="overview">
      <div class="eyebrow">FLATSIM &middot; validation Sentinel-1 &middot; check_results.py</div>
      <h1 class="title">{title}</h1>
      <div class="subtitle">{subtitle}</div>
      <div class="meta-row">{meta_line}{extra_meta}{images_meta}</div>
      {f'<div class="path-line">{path_line}</div>' if path_line else ''}
    </header>'''

    body = f'''<section>
      <div class="tiles">
        {''.join(tiles)}
      </div>
      {funnel}
      {callout}
    </section>'''

    return header + "\n" + body


TEMPLATE = r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root{
    --paper:#F2F5F6; --surface:#FFFFFF; --surface-2:#E8EDEF;
    --ink:#0F1A20; --ink-2:#4A5A62; --ink-3:#7C8B92;
    --line:#D6DFE3; --line-strong:#C1CDD2;
    --accent:#0E7C86; --accent-ink:#FFFFFF; --accent-soft:#E3F2F1;
    --warm:#B5651D;
    --good:#2F7D53;
    --warn:#96650A; --warn-bg:#FBF0D9; --warn-line:#E7CE9A;
    --shadow:0 1px 2px rgba(15,26,32,.05), 0 8px 24px -12px rgba(15,26,32,.18);
    --radius:12px;
    --mono:'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
    --sans:'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }
  @media (prefers-color-scheme: dark){
    :root:not([data-theme="light"]){
      --paper:#0A1216; --surface:#111B21; --surface-2:#16232A;
      --ink:#E7EEF1; --ink-2:#A3B4BB; --ink-3:#71838B;
      --line:#223037; --line-strong:#2C3D45;
      --accent:#3FC4CD; --accent-ink:#052326; --accent-soft:#132A2C;
      --warm:#E3924E;
      --good:#57BC8B;
      --warn:#E4B657; --warn-bg:#26200D; --warn-line:#4A3E17;
      --shadow:0 1px 2px rgba(0,0,0,.4), 0 14px 32px -14px rgba(0,0,0,.65);
    }
  }
  :root[data-theme="dark"]{
    --paper:#0A1216; --surface:#111B21; --surface-2:#16232A;
    --ink:#E7EEF1; --ink-2:#A3B4BB; --ink-3:#71838B;
    --line:#223037; --line-strong:#2C3D45;
    --accent:#3FC4CD; --accent-ink:#052326; --accent-soft:#132A2C;
    --warm:#E3924E;
    --good:#57BC8B;
    --warn:#E4B657; --warn-bg:#26200D; --warn-line:#4A3E17;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 14px 32px -14px rgba(0,0,0,.65);
  }

  *{box-sizing:border-box;}
  html{-webkit-text-size-adjust:100%;}
  body{
    margin:0; background:var(--paper); color:var(--ink);
    font-family:var(--sans); font-size:15px; line-height:1.55;
    -webkit-font-smoothing:antialiased;
  }
  h1,h2,h3{font-family:var(--sans); font-weight:600; text-wrap:balance; margin:0;}
  a{color:var(--accent);}
  .mono{font-family:var(--mono);}
  code.mono{background:var(--surface-2); padding:1px 5px; border-radius:4px; font-size:.9em;}
  .num{font-variant-numeric:tabular-nums;}

  .shell{display:grid; grid-template-columns:252px 1fr; max-width:1280px; margin:0 auto;}
  .sidebar{
    position:sticky; top:0; align-self:start; height:100vh;
    padding:28px 16px; border-right:1px solid var(--line);
    display:flex; flex-direction:column; gap:22px;
  }
  .brand{display:flex; align-items:center; gap:10px; padding:0 8px;}
  .brand-mark{
    width:30px; height:30px; border-radius:8px; background:var(--accent);
    color:var(--accent-ink); display:flex; align-items:center; justify-content:center;
    font-family:var(--mono); font-weight:600; font-size:12px; flex:none;
  }
  .brand-text{line-height:1.2;}
  .brand-text .name{font-weight:600; font-size:14px;}
  .brand-text .sub{font-size:11px; color:var(--ink-3); font-family:var(--mono);}

  nav.toc{display:flex; flex-direction:column; gap:2px;}
  nav.toc a{
    display:flex; align-items:baseline; gap:9px; padding:8px 8px;
    border-radius:8px; text-decoration:none; color:var(--ink-2);
    font-size:13.5px; font-weight:500; border-left:2px solid transparent;
  }
  nav.toc a .n{font-family:var(--mono); font-size:10.5px; color:var(--ink-3); width:14px; flex:none;}
  nav.toc a:hover{background:var(--surface-2); color:var(--ink);}
  nav.toc a.active{background:var(--accent-soft); color:var(--accent); border-left-color:var(--accent);}
  nav.toc a.active .n{color:var(--accent);}

  .sidebar-foot{margin-top:auto; padding:0 8px; font-size:11px; color:var(--ink-3); font-family:var(--mono);}
  .sidebar-foot a{color:var(--accent); text-decoration:none; font-weight:500;}
  .sidebar-foot a:hover{text-decoration:underline;}

  main{padding:36px 40px 90px; min-width:0;}

  header.report-head{margin-bottom:30px;}
  .eyebrow{
    font-family:var(--mono); font-size:11.5px; letter-spacing:.06em; text-transform:uppercase;
    color:var(--accent); font-weight:600; margin-bottom:10px;
  }
  h1.title{font-size:30px; letter-spacing:-.01em;}
  .subtitle{color:var(--ink-2); font-size:15px; margin-top:6px;}
  .meta-row{
    display:flex; flex-wrap:wrap; gap:8px 22px; margin-top:16px;
    font-family:var(--mono); font-size:12px; color:var(--ink-3);
  }
  .meta-row b{color:var(--ink-2); font-weight:500;}
  .path-line{
    margin-top:12px; font-family:var(--mono); font-size:11.5px; color:var(--ink-3);
    background:var(--surface-2); border:1px solid var(--line); border-radius:8px;
    padding:8px 12px; overflow-x:auto; white-space:pre;
  }
  .no-stats{font-size:13px; color:var(--ink-3); margin-top:16px;}

  section{scroll-margin-top:24px; padding-top:8px; margin-bottom:52px;}
  section > .section-head{display:flex; align-items:baseline; gap:12px; margin-bottom:16px; padding-bottom:10px; border-bottom:1px solid var(--line);}
  section > .section-head h2{font-size:18px;}
  section > .section-head .count{font-family:var(--mono); font-size:11.5px; color:var(--ink-3);}

  .tiles{display:grid; grid-template-columns:repeat(auto-fit, minmax(148px,1fr)); gap:12px; margin-bottom:18px;}
  .tile{
    background:var(--surface); border:1px solid var(--line); border-radius:var(--radius);
    padding:14px 16px; box-shadow:var(--shadow);
  }
  .tile .label{font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:var(--ink-3); font-weight:600;}
  .tile .value{font-family:var(--mono); font-size:24px; font-weight:600; margin-top:6px; font-variant-numeric:tabular-nums;}
  .tile .sub{font-size:11.5px; color:var(--ink-3); margin-top:3px; font-family:var(--mono);}
  .tile.flag{border-color:var(--warn-line); background:var(--warn-bg);}
  .tile.flag .value{color:var(--warn);}
  .tile.flag .flag-note{display:inline-block; margin-top:6px; font-size:11px; font-weight:600; color:var(--warn); font-family:var(--sans);}

  .funnel{
    display:flex; flex-wrap:wrap; align-items:center; gap:8px;
    font-family:var(--mono); font-size:12.5px; color:var(--ink-2);
    background:var(--surface); border:1px solid var(--line); border-radius:var(--radius);
    padding:14px 16px; margin-bottom:14px;
  }
  .funnel b{color:var(--ink); font-weight:600;}
  .funnel .arrow{color:var(--ink-3);}
  .funnel .minus{color:var(--warn);}
  .funnel .final{color:var(--good); font-weight:600;}

  .callout{
    background:var(--surface); border:1px solid var(--warn-line); border-left:3px solid var(--warn);
    border-radius:var(--radius); padding:16px 18px; box-shadow:var(--shadow);
  }
  .callout h3{font-size:14.5px; color:var(--ink);}
  table.flag-table{width:100%; border-collapse:collapse; margin-top:12px; font-size:13px;}
  table.flag-table th{
    text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.03em;
    color:var(--ink-3); font-weight:600; padding:6px 10px; border-bottom:1px solid var(--line);
  }
  table.flag-table td{padding:7px 10px; border-bottom:1px solid var(--line);}
  table.flag-table tr:last-child td{border-bottom:none;}
  table.flag-table td.num{color:var(--warm); font-weight:600;}

  .fig-grid{display:grid; grid-template-columns:repeat(auto-fit, minmax(300px,1fr)); gap:18px;}
  .fig-grid.wide{grid-template-columns:repeat(auto-fit, minmax(420px,1fr));}
  .fig-card{
    background:var(--surface); border:1px solid var(--line); border-radius:var(--radius);
    padding:14px; box-shadow:var(--shadow); margin:0;
  }
  .fig-eyebrow{margin-bottom:8px;}
  .fig-eyebrow .step{font-family:var(--mono); font-size:10.5px; color:var(--ink-3); letter-spacing:.03em;}
  .fig-card img{
    width:100%; display:block; border-radius:7px; background:#fff;
    border:1px solid var(--line); cursor:zoom-in;
  }
  .fig-card figcaption{margin-top:10px; font-size:13px; color:var(--ink-2);}

  #lightbox{
    position:fixed; inset:0; background:rgba(5,10,13,.86); display:none;
    align-items:center; justify-content:center; z-index:50; padding:28px; cursor:zoom-out;
  }
  #lightbox.open{display:flex;}
  #lightbox img{max-width:95vw; max-height:92vh; border-radius:8px; background:#fff; box-shadow:0 20px 60px rgba(0,0,0,.5);}

  @media (max-width: 860px){
    .shell{grid-template-columns:1fr;}
    .sidebar{
      position:sticky; top:0; height:auto; flex-direction:row; align-items:center;
      overflow-x:auto; padding:12px 14px; gap:14px; background:var(--paper);
      border-right:none; border-bottom:1px solid var(--line); z-index:10;
    }
    .brand{flex:none;}
    nav.toc{flex-direction:row; flex:none;}
    nav.toc a{white-space:nowrap;}
    .sidebar-foot{display:none;}
    main{padding:24px 18px 70px;}
    h1.title{font-size:24px;}
  }
  @media (prefers-reduced-motion: reduce){ *{scroll-behavior:auto !important;} }
</style>
</head>
<body>

<div id="lightbox" onclick="closeLightbox()"><img id="lightbox-img" src="" alt=""></div>

<div class="shell">
  <aside class="sidebar">
    <div class="brand">
      <div class="brand-mark">FS</div>
      <div class="brand-text">
        <div class="name">FLATSIM</div>
        <div class="sub">validation report</div>
      </div>
    </div>
    <nav class="toc" id="toc">
      <a href="#overview"><span class="n">00</span>Overview</a>
__NAV_LINKS__
    </nav>
__SIDEBAR_FOOT__
  </aside>

  <main>
__OVERVIEW__

__SECTIONS__
  </main>
</div>

<script>
  function openLightbox(img){
    var lb = document.getElementById('lightbox');
    var lbImg = document.getElementById('lightbox-img');
    lbImg.src = img.src;
    lbImg.alt = img.alt;
    lb.classList.add('open');
  }
  function closeLightbox(){
    document.getElementById('lightbox').classList.remove('open');
    document.getElementById('lightbox-img').src = '';
  }
  document.addEventListener('keydown', function(e){ if(e.key === 'Escape') closeLightbox(); });

  (function(){
    var links = Array.prototype.slice.call(document.querySelectorAll('#toc a'));
    var sections = links.map(function(a){ return document.querySelector(a.getAttribute('href')); });
    var io = new IntersectionObserver(function(entries){
      entries.forEach(function(entry){
        var idx = sections.indexOf(entry.target);
        if(idx === -1) return;
        if(entry.isIntersecting){
          links.forEach(function(l){ l.classList.remove('active'); });
          links[idx].classList.add('active');
        }
      });
    }, { rootMargin: '-15% 0px -70% 0px', threshold: 0 });
    sections.forEach(function(s){ if(s) io.observe(s); });
  })();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("track_dir", help="track directory, e.g. data/Turquie/A131")
    ap.add_argument("--val-dir", default=None, help="validation figures dir (default: <track_dir>/VALIDATION)")
    ap.add_argument("--log", default=None, help="saved check_results.py console log (for the stats overview)")
    ap.add_argument("--out", default=None, help="output HTML path (default: <val-dir>/report.html)")
    args = ap.parse_args()

    track_dir = os.path.normpath(args.track_dir)
    val_dir = os.path.normpath(args.val_dir) if args.val_dir else os.path.join(track_dir, "VALIDATION")
    out_path = args.out or os.path.join(val_dir, "report.html")

    if not os.path.isdir(val_dir):
        raise SystemExit(f"Dossier introuvable: {val_dir}")

    # --- stats / log -------------------------------------------------------
    log_path = args.log
    if not log_path:
        candidates = glob.glob(os.path.join(track_dir, "*.log")) + glob.glob(os.path.join(val_dir, "*.log"))
        log_path = candidates[0] if candidates else None

    stats, flagged = {}, []
    if log_path and os.path.isfile(log_path):
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            stats, flagged = parse_log(f.read())

    # --- identity ------------------------------------------------------------
    parts = [p for p in track_dir.split(os.sep) if p]
    site, track = (parts[-2], parts[-1]) if len(parts) >= 2 else (track_dir, "")
    title = f"{site} — Track {track}" if track else site
    dataset_dir = find_dataset_dir(track_dir)
    subtitle = build_subtitle(dataset_dir, track_dir)
    meta_line = f'<span><b>Dataset</b> {dataset_dir}</span>' if dataset_dir else ""
    path_line = os.path.abspath(os.path.join(track_dir, dataset_dir)) if dataset_dir else ""

    overview_html = build_overview(stats, flagged, title, subtitle, meta_line, path_line)

    # --- figures ---------------------------------------------------------
    present = {key: [] for key, _, _ in SECTIONS}
    for fname, step, section, caption in FIGURES:
        if os.path.isfile(os.path.join(val_dir, fname)):
            present[section].append((fname, step, caption))

    nav_links = []
    sections_html = []
    for key, label, count_label in SECTIONS:
        figs = present[key]
        if not figs:
            continue
        idx = f"{len(nav_links) + 1:02d}"
        nav_links.append(f'      <a href="#{key}"><span class="n">{idx}</span>{label}</a>')
        wide = " wide" if len(figs) <= 2 else ""
        cards = "\n".join(figure_card(f, s, c) for f, s, c in figs)
        sections_html.append(f'''    <section id="{key}">
      <div class="section-head"><h2>{label}</h2><span class="count">{count_label} &middot; {len(figs)} figure{'s' if len(figs) > 1 else ''}</span></div>
      <div class="fig-grid{wide}">
{cards}
      </div>
    </section>''')

    if REPO_URL:
        sidebar_foot = f'<div class="sidebar-foot"><a href="{REPO_URL}" target="_blank" rel="noopener">Relancer sur GitHub &rarr;</a></div>'
    else:
        sidebar_foot = ""

    total_figs = sum(len(v) for v in present.values())
    html = (
        TEMPLATE
        .replace("__TITLE__", title)
        .replace("__NAV_LINKS__", "\n".join(nav_links))
        .replace("__OVERVIEW__", overview_html)
        .replace("__SECTIONS__", "\n\n".join(sections_html))
        .replace("__SIDEBAR_FOOT__", sidebar_foot)
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"report.html écrit : {out_path}")
    print(f"  figures trouvées : {total_figs} / {len(FIGURES)}")
    print(f"  stats            : {'depuis ' + log_path if stats else 'aucune (pas de log trouvé)'}")
    print(f"  interférogrammes flagués : {len(flagged)}")


if __name__ == "__main__":
    main()
