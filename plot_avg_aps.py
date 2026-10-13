#!/usr/bin/env python3
"""
plot_avg_aps.py — Two figures from invers_temp.py outputs:

    Figure 1 — APS std vs time, one curve per iteration
    Figure 2 — Median on reference zone vs time, one curve per iteration

Usage
-----
    python plot_avg_aps.py <track_dir> [--save DIR] [--no-display]

Reads from TS/: aps_N.txt, ref_median_N.txt, list_images.txt
"""

import os, sys, glob, argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def _resolve_ts_dir(path):
    p = os.path.abspath(path)
    if os.path.basename(p) == 'TS' or not os.path.isdir(os.path.join(p, 'TS')):
        return p
    return os.path.join(p, 'TS')


def _dec(s):
    try:
        x = datetime.strptime(str(s).strip(), '%Y%m%d')
        return float(x.strftime('%Y')) + float(x.strftime('%j')) / 365.25
    except ValueError:
        return float(str(s).strip())


def load_dates(ts_dir):
    for fname in ['list_images.txt', 'images_retenues']:
        path = os.path.join(ts_dir, fname)
        if os.path.exists(path):
            dates = []
            with open(path) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        dates.append(_dec(parts[1]))
            return np.array(dates)
    return None


def load_iter(ts_dir, pattern):
    result = []
    for f in glob.glob(os.path.join(ts_dir, pattern)):
        try:
            n   = int(os.path.basename(f).replace('.txt','').split('_')[-1])
            arr = np.loadtxt(f, comments='#', dtype=np.float64)
            result.append((n, arr))
        except Exception:
            pass
    return sorted(result, key=lambda x: x[0])


def plot_iter(dates, iter_list, col, ylabel, title, fname,
              save_dir, display, hline=None):
    """Plot one curve per iteration, Blues colormap."""
    if not iter_list:
        return
    fig, ax = plt.subplots(figsize=(13, 4))
    colors  = plt.cm.Blues(np.linspace(0.35, 1.0, max(len(iter_list), 2)))
    for (n, arr), color in zip(iter_list, colors):
        v  = arr if arr.ndim == 1 else arr[:, col]
        nd = min(len(dates), len(v))
        ax.plot(dates[:nd], v[:nd], 'o-', color=color,
                markersize=4, linewidth=0.8, label=f'iter {n}')
    if hline is not None:
        ax.axhline(hline, color='black', lw=0.8, ls='--')
    ax.set_xlabel('Date (decimal year)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, fname)
    fig.savefig(path, bbox_inches='tight', dpi=120)
    print(f'  → {path}')
    if display:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot APS and median evolution')
    parser.add_argument('track_dir')
    parser.add_argument('--save', default=None)
    parser.add_argument('--no-display', action='store_true')
    args = parser.parse_args()

    ts_dir   = _resolve_ts_dir(args.track_dir)
    save_dir = os.path.abspath(args.save) if args.save else ts_dir
    display  = not args.no_display
    os.makedirs(save_dir, exist_ok=True)

    dates    = load_dates(ts_dir)
    aps_list = load_iter(ts_dir, 'aps_*.txt')
    ref_list = load_iter(ts_dir, 'ref_median_*.txt')

    if dates is None:
        print('ERROR: list_images.txt not found.'); sys.exit(1)
    if not aps_list and not ref_list:
        print('No aps_N.txt or ref_median_N.txt found. Run invers_temp.py first.')
        sys.exit(1)

    # summary
    print(f'\nAPS files   : {[f"aps_{n}.txt" for n,_ in aps_list]}')
    print(f'Median files: {[f"ref_median_{n}.txt" for n,_ in ref_list]}')
    for n, arr in aps_list:
        v = arr.flatten()
        print(f'  aps_{n}  mean={v.mean():.3f}  max={v.max():.3f}  min={v.min():.3f} rad')
    for n, arr in ref_list:
        if arr.ndim > 1:
            med = arr[:, 1]
            print(f'  ref_median_{n}  mean={med.mean():.4f}  '
                  f'max={med.max():.4f}  min={med.min():.4f} rad')

    # Figure 1 — APS std
    plot_iter(dates, aps_list, col=0,
              ylabel='APS std (rad)',
              title='Per-date APS std vs time  (decreasing = converging)',
              fname='plot_aps_vs_time.png',
              save_dir=save_dir, display=display)

    # Figure 2 — Median on reference zone
    plot_iter(dates, ref_list, col=1,
              ylabel='Median on ref zone (rad)',
              title='Median residual on reference zone vs time\n'
                    '(should converge to 0 — if large: use --ref_zone)',
              fname='plot_median_vs_time.png',
              save_dir=save_dir, display=display,
              hline=0.0)


if __name__ == '__main__':
    main()
