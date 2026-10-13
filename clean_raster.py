#!/usr/bin/env python3
# -*- coding: utf-8 -*-
############################################
#
# PyGdalSAR: An InSAR post-processing package
# written in Python-Gdal
#
############################################
# Author        : Simon DAOUT (Oxford)
############################################

"""
clean_raster.py
-------------
Clean a raster file given an other r4 file (mask) and a threshold on this mask

Usage: clean_raster.py --infile=<path> --outfile=<path> [--mask=<path>] [--threshold=<value>]
       [--perc=<value>] [--crop=<values>] [--buff=<value>] [--lectfile=<path>] [--scale=<value>]
       [--scale_mask=<value>] [--ramp=<lin/quad/cub/4/no>] [--ref=<jstart,jend,istart,iend>]
       [--removeNAN=<yes/no>] [--cst=<value>] [--absolute=<yes/no>] [--reverse=<yes/no>]
       [--filter=<HP/LP>] [--fwindsize=<value>]

(argparse-based: --infile <path> and --infile=<path> are both accepted)
"""

import argparse
import os
import sys
import textwrap

import numpy as np
import scipy.optimize as opt
import scipy.linalg as lst
from scipy import ndimage

from matplotlib import pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def parse_args():
    p = argparse.ArgumentParser(
        description="Clean a raster file given an other r4 file (mask) and a "
                     "threshold on this mask",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--infile", required=True, help="r4 file to clean")
    p.add_argument("--outfile", required=True, help="output file")
    p.add_argument("--mask", default=None, help="r4 file used as mask")
    p.add_argument("--threshold", default=None,
                    help="threshold value on mask file (keep pixel with mask > "
                         "threshold) [default: 1e10]")
    p.add_argument("--reverse", default=None,
                    help="yes/no - if yes mask values above mask threshold "
                         "[default: no]")
    p.add_argument("--absolute", default=None,
                    help="yes/no - if yes, take the absolute values of the mask "
                         "[default: no]")
    p.add_argument("--scale", default=None, help="scale data [default: 1]")
    p.add_argument("--scale_mask", default=None, help="scale the mask [default: 1]")
    p.add_argument("--lectfile", default=None,
                    help="path of the lect.in file [default: lect.in]")
    p.add_argument("--crop", default=None,
                    help="crop option with smoothing of boundaries "
                         "[default: 0,ncols,0,nlines]")
    p.add_argument("--buff", default=None,
                    help="number of pixels for crop smoothing [default: 50]")
    p.add_argument("--perc", default=None,
                    help="percentile of hidden LOS pixel for the estimation and "
                         "clean outliers [default: 100]")
    p.add_argument("--ramp", default=None,
                    help="lin/quad/cub/4/no - correct the map from ramp in range "
                         "and azimuth [default: no]")
    p.add_argument("--ref", default=None,
                    help="jstart,jend,istart,iend - set to zero displacements "
                         "from jstart to jend")
    p.add_argument("--removeNAN", default=None, help="yes/no - replace NaN by 0")
    p.add_argument("--cst", default=None, help="add constant to map")
    p.add_argument("--filter", default=None,
                    help="HP/LP - apply a high pass (HP) or low pass (LP) "
                         "gaussian filter to the image")
    p.add_argument("--fwindsize", default=None,
                    help="filter window size [default: 16]")
    return p.parse_args()


def main():
    print()
    print()
    print('Author: Simon DAOUT')
    print()
    print()

    args = parse_args()

    infile = args.infile
    outfile = args.outfile
    maskf = 'no' if args.mask is None else args.mask
    seuil = 1e10 if args.threshold is None else float(args.threshold)
    lecfile = "lect.in" if args.lectfile is None else args.lectfile
    scale = 1. if args.scale is None else float(args.scale)
    scale_mask = 1. if args.scale_mask is None else float(args.scale_mask)
    setzero = 'no' if args.removeNAN is None else args.removeNAN
    perc = 100 if args.perc is None else float(args.perc)
    shift = 0 if args.cst is None else float(args.cst)

    if (args.ramp is None) or (args.ramp == 'no'):
        ramp = 'no'
    elif args.ramp == 'lin':
        ramp = 'lin'
    elif args.ramp == 'quad':
        ramp = 'quad'
    elif args.ramp == 'cub':
        ramp = 'cub'
    elif args.ramp == '4':
        ramp = '4'
    elif args.ramp == 'yes':
        print('ramp==yes is depricated. Use lin/quad/cub/4/no. Set ramp to lin')
        ramp = 'lin'
    else:
        print('ramp argument not recognized. Exit!')
        sys.exit()

    reverse = (args.reverse == 'yes')
    absolute = (args.absolute == 'yes')
    fwindsize = 16 if args.fwindsize is None else int(args.fwindsize)

    ds_extension = os.path.splitext(infile)[1]
    if ds_extension in (".tif", ".tiff", ".grd"):
        from osgeo import gdal
        sformat = 'GRID'
        ds = gdal.Open(infile, gdal.GA_ReadOnly)
        band = ds.GetRasterBand(1)
        # Attributes
        print("> Driver:   ", ds.GetDriver().ShortName)
        print("> Size:     ", ds.RasterXSize, 'x', ds.RasterYSize, 'x', ds.RasterCount)
        print("> Datatype: ", gdal.GetDataTypeName(band.DataType))
        nlines, ncols = ds.RasterYSize, ds.RasterXSize
        gt = ds.GetGeoTransform()
        proj = ds.GetProjectionRef()
        driver = gdal.GetDriverByName('GTiff')
        m = band.ReadAsArray(0, 0, ds.RasterXSize, ds.RasterYSize,
                              ds.RasterXSize, ds.RasterYSize)
    else:
        sformat = 'R4'
        # read lect.in
        ncols, nlines = list(map(int, open(lecfile).readline().split(None, 2)[0:2]))
        fid = open(infile, 'r')
        m = np.fromfile(fid, dtype=np.float32).reshape((nlines, ncols))
        # m[m==0.0] = float('NaN')

    if args.ref is None:
        lin_start, lin_end, col_start, col_end = None, None, None, None
    else:
        ref = list(map(int, args.ref.replace(',', ' ').split()))
        try:
            lin_start, lin_end, col_start, col_end = ref[0], ref[1], ref[2], ref[3]
        except Exception:
            lin_start, lin_end = ref[0], ref[1]
            col_start, col_end = 0, ncols

    # mask
    if maskf != 'no':
        ds_extension = os.path.splitext(maskf)[1]
        if ds_extension in (".tif", ".tiff", ".grd"):
            from osgeo import gdal
            ds = gdal.Open(maskf, gdal.GA_ReadOnly)
            band = ds.GetRasterBand(1)
            # Attributes
            print("> Driver:   ", ds.GetDriver().ShortName)
            print("> Size:     ", ds.RasterXSize, 'x', ds.RasterYSize, 'x', ds.RasterCount)
            print("> Datatype: ", gdal.GetDataTypeName(band.DataType))
            mask = band.ReadAsArray(0, 0, ds.RasterXSize, ds.RasterYSize,
                                     ds.RasterXSize, ds.RasterYSize)
        else:
            fid2 = open(maskf, 'r')
            mask = np.fromfile(fid2, dtype=np.float32)[:nlines * ncols].reshape((nlines, ncols))
            mask = mask * scale_mask
            mask[np.isnan(mask)] = 0
    else:
        mask = np.full((nlines, ncols), np.nan, dtype=np.float32)

    mf = np.copy(m)
    # apply mask: keep pixel with mask > threshold (--reverse=yes flips it,
    # keeping mask < threshold instead)
    if absolute:
        mask = np.abs(mask)
    if reverse:
        kk = np.nonzero(np.logical_or(mask == 0, mask > seuil))
    else:
        kk = np.nonzero(np.logical_or(mask == 0, mask < seuil))
    mf[kk] = float('NaN')

    # apply scale and manual cst
    mf = scale * (mf - shift)

    if ramp != 'no':
        # clean for ramp
        maxlos, minlos = np.nanpercentile(m, 99.5), np.nanpercentile(m, .5)
        print('Clean outliers for ramp estimation outside:', maxlos, minlos)
        kk = np.nonzero(np.logical_or(m < minlos, m > maxlos))
        m_ramp = np.copy(mf)
        m_ramp[kk] = float('NaN')

    # normalize pixel coordinates to O(1) range before fitting the ramp
    # polynomial: fitting cub/4 directly in raw pixel-index units (up to a
    # few thousand) is numerically unstable (x**3 ~ 1e11) and can produce
    # wildly diverging coefficients; this rescaling does not change the
    # fitted surface itself (ordinary least squares is invariant to an
    # affine rescaling of the predictors, applied consistently on both the
    # fit and the full-image reconstruction below), it only conditions it
    rg_scale = float(max(ncols, 1))
    az_scale = float(max(nlines, 1))

    if ramp == 'lin':
        index = np.nonzero(~np.isnan(m_ramp))
        temp = np.array(index).T
        mi = mf[index].flatten()  # fit against the current working array (post mask/scale/ramp-order), not the frozen raw m
        az = temp[:, 0]
        rg = temp[:, 1]

        G = np.zeros((len(mi), 3))
        G[:, 0] = rg / rg_scale
        G[:, 1] = az / az_scale
        G[:, 2] = 1

        x0 = lst.lstsq(G, mi)[0]
        _func = lambda x: np.sum(((np.dot(G, x) - mi)) ** 2)
        _fprime = lambda x: 2 * np.dot(G.T, (np.dot(G, x) - mi))
        pars = opt.fmin_slsqp(_func, x0, fprime=_fprime, iter=2000, full_output=True, iprint=0)[0]

        a = pars[0]
        b = pars[1]
        c = pars[2]
        print('Remove ramp %f x  + %f y + %f' % (a, b, c))

        # built as float32: at full-image size this matrix is the single
        # largest allocation in the script, and float64 here was wasted
        # precision (coordinates are already normalized to O(1) above)
        G = np.zeros((len(mask.flatten()), 3), dtype=np.float32)
        for i in range(nlines):
            G[i * ncols:(i + 1) * ncols, 0] = np.arange((ncols)) / rg_scale
            G[i * ncols:(i + 1) * ncols, 1] = i / az_scale
        G[:, 2] = 1

        # apply ramp correction
        temp = np.dot(G, pars.astype(np.float32))
        ramp = temp.reshape(nlines, ncols)
        mf = mf - ramp
        del temp

    elif ramp == 'quad':
        index = np.nonzero(~np.isnan(m_ramp))
        temp = np.array(index).T
        mi = mf[index].flatten()  # fit against the current working array (post mask/scale/ramp-order), not the frozen raw m
        az = temp[:, 0]
        rg = temp[:, 1]

        G = np.zeros((len(mi), 4))
        G[:, 0] = (rg / rg_scale) ** 2
        G[:, 1] = rg / rg_scale
        G[:, 2] = az / az_scale
        G[:, 3] = 1

        x0 = lst.lstsq(G, mi)[0]
        _func = lambda x: np.sum(((np.dot(G, x) - mi)) ** 2)
        _fprime = lambda x: 2 * np.dot(G.T, (np.dot(G, x) - mi))
        pars = opt.fmin_slsqp(_func, x0, fprime=_fprime, iter=2000, full_output=True, iprint=0)[0]
        a = pars[0]
        b = pars[1]
        c = pars[2]
        d = pars[3]
        print('Remove ramp %f x**2 %f x  + %f y + %f' % (a, b, c, d))

        G = np.zeros((len(mask.flatten()), 4), dtype=np.float32)
        for i in range(nlines):
            G[i * ncols:(i + 1) * ncols, 0] = (np.arange((ncols)) / rg_scale) ** 2
            G[i * ncols:(i + 1) * ncols, 1] = np.arange((ncols)) / rg_scale
            G[i * ncols:(i + 1) * ncols, 2] = i / az_scale
        G[:, 3] = 1
        # apply ramp correction
        temp = np.dot(G, pars.astype(np.float32))
        ramp = temp.reshape(nlines, ncols)
        mf = mf - ramp
        del temp

    elif ramp == 'cub':
        index = np.nonzero(~np.isnan(m_ramp))
        temp = np.array(index).T
        mi = mf[index].flatten()  # fit against the current working array (post mask/scale/ramp-order), not the frozen raw m
        az = temp[:, 0]
        rg = temp[:, 1]

        G = np.zeros((len(mi), 7))
        G[:, 0] = (rg / rg_scale) ** 3
        G[:, 1] = (rg / rg_scale) ** 2
        G[:, 2] = rg / rg_scale
        G[:, 3] = (az / az_scale) ** 3
        G[:, 4] = (az / az_scale) ** 2
        G[:, 5] = az / az_scale
        G[:, 6] = 1

        x0 = lst.lstsq(G, mi)[0]
        _func = lambda x: np.sum(((np.dot(G, x) - mi)) ** 2)
        _fprime = lambda x: 2 * np.dot(G.T, (np.dot(G, x) - mi))
        pars = opt.fmin_slsqp(_func, x0, fprime=_fprime, iter=2000, full_output=True, iprint=0)[0]
        a = pars[0]
        b = pars[1]
        c = pars[2]
        d = pars[3]
        e = pars[4]
        f = pars[5]
        g = pars[6]
        print('Remove ramp %f x**3 + %f x**2 + %f x  + %f y**3 + %f y**2 + %f y + %f'
              % (a, b, c, d, e, f, g))

        G = np.zeros((len(mask.flatten()), 7), dtype=np.float32)
        for i in range(nlines):
            G[i * ncols:(i + 1) * ncols, 0] = (np.arange((ncols)) / rg_scale) ** 3
            G[i * ncols:(i + 1) * ncols, 1] = (np.arange((ncols)) / rg_scale) ** 2
            G[i * ncols:(i + 1) * ncols, 2] = np.arange((ncols)) / rg_scale
            G[i * ncols:(i + 1) * ncols, 3] = (i / az_scale) ** 3
            G[i * ncols:(i + 1) * ncols, 4] = (i / az_scale) ** 2
            G[i * ncols:(i + 1) * ncols, 5] = i / az_scale
        G[:, 6] = 1
        # apply ramp correction
        temp = np.dot(G, pars.astype(np.float32))
        ramp = temp.reshape(nlines, ncols)
        mf = mf - ramp
        del temp

    elif ramp == '4':
        index = np.nonzero(~np.isnan(m_ramp))
        temp = np.array(index).T
        mi = mf[index].flatten()  # fit against the current working array (post mask/scale/ramp-order), not the frozen raw m
        az = temp[:, 0]
        rg = temp[:, 1]

        G = np.zeros((len(mi), 9))
        G[:, 0] = (rg / rg_scale) ** 3
        G[:, 1] = (rg / rg_scale) ** 2
        G[:, 2] = rg / rg_scale
        G[:, 3] = (az / az_scale) ** 3
        G[:, 4] = (az / az_scale) ** 2
        G[:, 5] = ((az / az_scale) * (rg / rg_scale)) ** 2
        G[:, 6] = (az / az_scale) * (rg / rg_scale)
        G[:, 7] = az / az_scale
        G[:, 8] = 1

        x0 = lst.lstsq(G, mi)[0]
        _func = lambda x: np.sum(((np.dot(G, x) - mi)) ** 2)
        _fprime = lambda x: 2 * np.dot(G.T, (np.dot(G, x) - mi))
        pars = opt.fmin_slsqp(_func, x0, fprime=_fprime, iter=2000, full_output=True, iprint=0)[0]
        a = pars[0]
        b = pars[1]
        c = pars[2]
        d = pars[3]
        e = pars[4]
        f = pars[5]
        g = pars[6]
        h = pars[7]
        i_ = pars[8]
        print('Remove ramp %f x**3 + %f x**2 + %f x  + %f y**3 + %f y**2 + %f xy**2 + %f xy + %f y + %f'
              % (a, b, c, d, e, f, g, h, i_))

        G = np.zeros((len(mask.flatten()), 9), dtype=np.float32)
        for i in range(nlines):
            G[i * ncols:(i + 1) * ncols, 0] = (np.arange((ncols)) / rg_scale) ** 3
            G[i * ncols:(i + 1) * ncols, 1] = (np.arange((ncols)) / rg_scale) ** 2
            G[i * ncols:(i + 1) * ncols, 2] = np.arange((ncols)) / rg_scale
            G[i * ncols:(i + 1) * ncols, 3] = (i / az_scale) ** 3
            G[i * ncols:(i + 1) * ncols, 4] = (i / az_scale) ** 2
            G[i * ncols:(i + 1) * ncols, 5] = ((i / az_scale) * (np.arange((ncols)) / rg_scale)) ** 2
            G[i * ncols:(i + 1) * ncols, 6] = (i / az_scale) * (np.arange((ncols)) / rg_scale)
            G[i * ncols:(i + 1) * ncols, 7] = i / az_scale
        G[:, 8] = 1
        # apply ramp correction
        temp = np.dot(G, pars.astype(np.float32))
        ramp = temp.reshape(nlines, ncols)
        mf = mf - ramp
        del temp

    else:
        ramp = np.zeros(np.shape(m))

    # apply high pass / low pass filter (after the ramp: the ramp removes the
    # deterministic polynomial-shaped regional trend, and the filter then
    # additionally suppresses any remaining power at wavelengths > cutoff
    # that the polynomial did not capture, on the ramp-corrected residual)
    if args.filter == 'HP':
        # make array with nans replaced by zeros and filter it
        m_filter_vals = np.copy(mf)
        m_filter_vals[np.isnan(mf)] = 0.
        m_lp_vals = ndimage.gaussian_filter(m_filter_vals, fwindsize)
        # make same size array full of ones, but set to zero where there is a nan in mf
        m_filter_ones = 0 * np.copy(mf) + 1
        m_filter_ones[np.isnan(mf)] = 0.
        m_lp_ones = ndimage.gaussian_filter(m_filter_ones, fwindsize)
        # find the ratio to make coefficients sum to one near nan values
        m_lp = m_lp_vals / m_lp_ones
        # subtract to obtain hp filter
        mf = mf - m_lp

    elif args.filter == 'LP':
        m_filter = np.copy(mf)
        index = np.isnan(mf)
        m_filter[index] = 0.
        mf = ndimage.gaussian_filter(m_filter, fwindsize)
        mf[index] = float('nan')

    if args.ref is not None:
        cst = np.nanmean(mf[lin_start:lin_end, col_start:col_end])
        mf = mf - cst
        ramp = ramp - cst

    # crop
    if args.crop is None:
        crop = [0, ncols, 0, nlines]
    else:
        crop = list(map(float, args.crop.replace(',', ' ').split()))
    ibeg, iend, jbeg, jend = int(crop[0]), int(crop[1]), int(crop[2]), int(crop[3])
    if iend > ncols:
        iend = ncols
    if jend > nlines:
        jend = nlines

    if iend - ibeg < ncols or jend - jbeg < nlines:
        buf = 50 if args.buff is None else int(args.buff)
        ibeg1, iend1 = 0, ibeg
        ibeg2, iend2 = iend, ncols
        jbeg1, jend1 = 0, jbeg
        jbeg2, jend2 = jend, nlines

        # top
        if jend1 > 0:
            for j in range(buf):
                for i in range(ibeg, iend):
                    mf[jend1 + j, i] = mf[jend1 + j, i] * (float(j + 1) / buf)

        # bottom
        if jbeg2 < nlines:
            for j in range(buf):
                for i in range(ibeg, iend):
                    mf[jbeg2 - (buf - j), i] = mf[jbeg2 - (buf - j), i] - mf[jbeg2 - (buf - j), i] * (float(j + 1) / buf)

        # left
        if iend1 > 0:
            for j in range(buf):
                for i in range(jbeg, jend):
                    mf[i, iend1 + (buf - (j + 1))] = mf[i, iend1 + (buf - (j + 1))] - mf[i, iend1 + (buf - (j + 1))] * (float(j + 1) / buf)

        # right
        if ibeg2 < ncols:
            for j in range(buf):
                for i in range(jbeg, jend):
                    mf[i, ibeg2 - (buf - j)] = mf[i, ibeg2 - (buf - j)] - mf[i, ibeg2 - (buf - j)] * (float(j + 1) / buf)

        mf[jbeg1:jend1, :] = float('NaN')
        mf[jbeg2:jend2, :] = float('NaN')
        mf[:, ibeg1:iend1] = float('NaN')
        mf[:, ibeg2:iend2] = float('NaN')

    # clean based on perc
    maxlos, minlos = np.nanpercentile(mf, perc), np.nanpercentile(mf, (100 - perc))
    print('Clean outliers outside {}-{} with perc:{}:'.format(maxlos, minlos, perc))
    kk = np.nonzero(np.logical_or(mf < minlos, mf > maxlos))
    # mf[kk] = float('NaN')
    mf[kk] = 0

    # Plot
    vmax = np.nanpercentile(mf, 98)
    vmin = np.nanpercentile(mf, 2)

    if setzero == 'yes':
        # replace "NaN" to 0
        mf[np.isnan(mf)] = 0.
        # mf[np.isnan(m)] = np.nan

    # save output
    if sformat == 'R4':
        fid3 = open(outfile, 'wb')
        mf.flatten().astype('float32').tofile(fid3)
        fid3.close()
    else:
        from osgeo import gdal
        dst_ds = driver.Create(outfile, ncols, nlines, 1, gdal.GDT_Float32)
        dst_band2 = dst_ds.GetRasterBand(1)
        dst_band2.WriteArray(mf, 0, 0)
        dst_ds.SetGeoTransform(gt)
        dst_ds.SetProjection(proj)
        dst_band2.FlushCache()
        del dst_ds

    try:
        from matplotlib.colors import LinearSegmentedColormap
        cm_locs = os.environ["PYGDALSAR"] + '/contrib/python/colormaps/'
        cmap = LinearSegmentedColormap.from_list('roma', np.loadtxt(cm_locs + "roma.txt"))
        cmap = cmap.reversed()
    except Exception:
        cmap = cm.rainbow

    fig = plt.figure(0, figsize=(12, 6))
    ax = fig.add_subplot(2, 2, 1)
    cax = ax.imshow(m, cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title('\n'.join(textwrap.wrap(os.path.basename(infile), 30)), fontsize=9)
    plt.setp(ax.get_xticklabels(), visible=False)
    divider = make_axes_locatable(ax)
    c = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax, cax=c)

    ax = fig.add_subplot(2, 2, 2)
    cax = ax.imshow(mask, cmap, vmin=0, vmax=np.nanpercentile(mask, 99), interpolation='nearest')
    ax.set_title('MASK')
    plt.setp(ax.get_xticklabels(), visible=False)
    divider = make_axes_locatable(ax)
    c = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax, cax=c)

    ax = fig.add_subplot(2, 2, 3)
    cax = ax.imshow(ramp, cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title('DATA - RAMP')
    plt.setp(ax.get_xticklabels(), visible=False)
    divider = make_axes_locatable(ax)
    c = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax, cax=c)

    ax = fig.add_subplot(2, 2, 4)
    cax = ax.imshow(mf, cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_title('\n'.join(textwrap.wrap(os.path.basename(outfile), 30)), fontsize=9)
    divider = make_axes_locatable(ax)
    c = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax, cax=c)

    fig.tight_layout()

    try:
        fig.savefig('{}_clean.pdf'.format(infile), format='PDF', dpi=180)
    except Exception:
        pass

    plt.show()


if __name__ == "__main__":
    main()
