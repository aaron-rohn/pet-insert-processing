import os
import numpy as np
from scipy import optimize
from collections import defaultdict
import matplotlib.pyplot as plt

import petmr

nbins = 100

doi_bins = np.linspace(0, 20, petmr.ndoi+1)
lyso_attn_length = 12
doi_quantiles = 1 - np.exp(-doi_bins / lyso_attn_length)

def photopeak(x, peak, fwhm, amp, slope, intercept):
    # model the photopeak as a FWHM-parametrized gaussian plus a linear offset
    y = amp * np.exp(-4 * np.log(2) * np.power(x-peak,2) / np.power(fwhm,2))
    return y + (x-peak)*slope + intercept

def hist_and_update(values, q0 = 0, q1 = 0.97, n = None, bins = None):
    """ histogram samples and add them to an existing histogram
    defined by arguments n and bins.
    """

    newbins = np.linspace(*np.quantile(values, [q0,q1]), nbins + 1)

    if bins is None:
        bins = newbins
    elif bins[0] < 0:
        bins[:] = newbins

    if n is None:
        n = np.zeros(nbins)

    cts, _ = np.histogram(values, bins = bins)
    n += cts

    return n, bins

def fit_photopeak(energy, **hist):
    n, bins = hist_and_update(energy, 0.01, 0.97, **hist)

    # fit the data with a gaussian and linear offset

    b = bins[:-1]
    ppeak_idx = np.argmax(b**2 * n)
    ppeak_energy = round(b[ppeak_idx])
    subset = b > (ppeak_energy * 0.5)

    xl, xh = ppeak_energy*0.8, ppeak_energy*1.2
    xl, xh = np.searchsorted(b, [xl,xh])
    xl, xh = np.clip([xl, xh], 0, len(n)-1)
    intercept = (n[xh] + n[xl]) / 2
    slope = (n[xh] - n[xl]) / (xh - xl)

    pinit = (ppeak_energy,
             ppeak_energy * 0.2,
             n[ppeak_idx],
             slope,
             intercept)
    
    bounds = ((ppeak_energy/2, ppeak_energy*2),             # photopeak position
              (ppeak_energy * 0.05, ppeak_energy * 0.5),    # fwhm
              (n[ppeak_idx] * 0.1, n[ppeak_idx] * 2),       # amplitude
              (-np.inf, np.inf),
              (-np.inf, np.inf))

    try:
        popt, *_ = optimize.curve_fit(photopeak,
                                      b[subset], n[subset],
                                      p0 = pinit, bounds = list(zip(*bounds)))
    except RuntimeError:
        popt = pinit

    return popt

def get_doi(doi, **hist):
    n, bins = hist_and_update(doi, 0, 1, **hist)

    pdf = np.cumsum(n[::-1]).astype(float)
    pdf = pdf / pdf[-1] * doi_quantiles[-1]

    return np.interp(doi_quantiles, pdf, bins[-2::-1])

def summarize_crystal(data, crystal, hist = None):
    # get DOI thresholds based on non-masked data
    thresholds = get_doi(data['D'])
    thresholds = np.round(thresholds)

    ppeaks = []
    fwhms  = []

    # fit photopeak for each DOI bin
    for h,l in zip(thresholds[:-1], thresholds[1:]):
        mask = (l < data['D']) & (data['D'] < h)
        if len(mask) > 0:
            peak, fwhm, *_ = fit_photopeak(data[mask]['E'])
        else:
            peak, fwhm = -1, -1

        ppeaks.append(round(peak))
        fwhms.append(round(fwhm))

    return crystal, thresholds[1:], ppeaks, fwhms

def calculate_lut_statistics(lut, data, hist = None):
    if hist is None: hist = defaultdict(lambda: None)

    idx = np.ravel_multi_index((data['Y'],data['X']), lut.shape) # 1d event position
    idx = lut.flat[idx] # crystal ID for each event, 0-361

    # sort data by crystal IDs
    order = np.argsort(idx)
    d_sort = data[order]
    c_sort = idx[order]

    # get the index corresponding to each crystal
    # note that the last bin (361) corresponds to invalid events
    crystals, locs = np.unique(c_sort, return_index = True)

    return [summarize_crystal(d_sort[s:e], x)
            for x,s,e in zip(crystals, locs[:-1], locs[1:])]
