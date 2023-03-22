import os
import numpy as np
import concurrent.futures
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

def fit_photopeak(energy = None, n = None, bins = None):
    if energy is not None:
        n,bins = np.histogram(energy, bins = 100,
                range = np.quantile(energy, [0, 0.95]))
        bins = bins[:-1]

    """
    idx = np.argmax(n)
    thr = n < (n[idx]/2)
    upper = np.argmax(thr[idx:]) + idx
    lower = idx - np.argmax(thr[idx::-1])
    return bins[idx], bins[upper] - bins[lower]
    """

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

def summarize_crystal(data, crystal):
    e = data['E']
    peak, fwhm, *_ = fit_photopeak(e)
    fwhm = abs(fwhm)

    g = data[((peak-fwhm) < e) & (e < (peak+fwhm))]
    try:
        thresholds = get_doi(g['DOI'])
    except IndexError:
        print(f'Error measuring DOI thresholds: {peak}, {fwhm}')
        thresholds = np.array([2200,2000,1800])

    return np.concatenate([[crystal, peak,fwhm], thresholds])

def calculate_lut_statistics(lut, data):
    idx = np.ravel_multi_index((data['Y'],data['X']), lut.shape)
    value = lut.flat[idx]
    order = np.argsort(value)

    data[:] = data[order]
    value[:] = value[order]
    locs, *_ = np.nonzero(np.diff(value, prepend = -1))
    res = np.array([summarize_crystal(data[s:e],value[s])
        for s,e in zip(locs[:-1], locs[1:])])
    return np.core.records.fromarrays(res.T, CrystalFitDtype)
