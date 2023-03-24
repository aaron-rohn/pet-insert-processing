import os, datetime
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

doi_bins = np.array([5,10,15,20])
lyso_attn_length = 12
doi_quantiles = 1 - np.exp(-doi_bins / lyso_attn_length)

CrystalFitDtype = np.dtype([
    ('crystal', int), ('peak',float), ('FWHM',float),
    ('5mm',float), ('10mm',float), ('15mm',float)])

def photopeak(x, peak, fwhm, amp, slope, intercept):
    # model the photopeak as a FWHM-parametrized gaussian plus a linear offset
    y = amp * np.exp(-4 * np.log(2) * np.power(x-peak,2) / np.power(fwhm,2))
    return y + (x-peak)*slope + intercept

def fit_photopeak(energy = None, n = None, bins = None, approx = False):
    if energy is not None:
        n,bins = np.histogram(energy, bins = 100,
                range = np.quantile(energy, [0, 0.95]))
        bins = bins[:-1]

    if approx:
        # just return the max location and approximate fwhm,
        # rather than actually fitting the data
        idx = np.argmax(n)
        thr = n < (n[idx]/2)
        upper = np.argmax(thr[idx:]) + idx
        lower = idx - np.argmax(thr[idx::-1])
        return bins[idx], bins[upper] - bins[lower]

    # fit the data with a gaussian and linear offset

    idx = np.argmax((bins*2) * n**2)
    ppeak_energy = round(bins[idx])
    subset = bins > (ppeak_energy * 0.7)

    xl, xh = ppeak_energy*0.8, ppeak_energy*1.2
    xl, xh = np.searchsorted(bins, [xl,xh])
    xl, xh = np.clip([xl, xh], 0, len(n)-1)
    intercept = (n[xh] + n[xl]) / 2
    slope = (n[xh] - n[xl]) / (xh - xl)

    pinit = (ppeak_energy,
             ppeak_energy * 0.2,
             n[idx],
             slope,
             intercept)
    
    bounds = ((ppeak_energy/2, ppeak_energy*2),
              (ppeak_energy * 0.05, ppeak_energy * 0.5),
              (n[idx] * 0.1, n[idx] * 2),
              (-np.inf, np.inf),
              (-np.inf, np.inf))

    try:
        popt, *_ = optimize.curve_fit(photopeak,
                                      bins[subset], n[subset],
                                      p0 = pinit, bounds = list(zip(*bounds)))
    except RuntimeError:
        popt = pinit

    return popt

def get_doi(doi):
    n,bins = np.histogram(doi, bins = 100,
            range = np.quantile(doi, [0, 0.95]))
    bins = bins[:-1]
    n, bins = n[::-1], bins[::-1]

    n = np.cumsum(n).astype(float)
    n = n / n[-1] * doi_quantiles[-1]
    thresholds = np.interp(doi_quantiles[:-1], n, bins)
    return thresholds.round(1)

def summarize_crystal(data, crystal, approx = False):
    e = data['E']
    peak, fwhm, *_ = fit_photopeak(e, approx = approx)
    fwhm = abs(fwhm)

    data_windowed = data[((peak-fwhm) < e) & (e < (peak+fwhm))]

    try:
        thresholds = get_doi(data_windowed['D'])
    except IndexError:
        print(f'Error measuring DOI thresholds: {peak}, {fwhm}')
        thresholds = np.array([2200,2000,1800])

    return np.concatenate([[crystal, peak, fwhm], thresholds])

def calculate_lut_statistics(lut, data):
    idx = np.ravel_multi_index((data['Y'],data['X']), lut.shape) # 1d event position
    idx = lut.flat[idx] # crystal ID for each event, 0-361

    # sort data by crystal IDs
    order = np.argsort(idx)
    d_sort = data[order]
    c_sort = idx[order]

    # get the index corresponding to each crystal
    # note that the last bin (361) corresponds to invalid events
    crystals, locs = np.unique(c_sort, return_index = True)

    # iterate over each subset of data for all but the last crystal ID
    res = np.array([summarize_crystal(d_sort[s:e], x)
        for x,s,e in zip(crystals, locs[:-1], locs[1:])])

    # return a structured array with fitting results for each crystal
    return np.core.records.fromarrays(res.T, CrystalFitDtype)
