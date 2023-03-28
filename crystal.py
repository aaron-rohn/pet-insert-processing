import os
import numpy as np
from scipy import optimize
from collections import defaultdict
import matplotlib.pyplot as plt

nbins = 100

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
    n, bins = hist_and_update(energy, 0, 0.97, **hist)

    # fit the data with a gaussian and linear offset

    b = bins[:-1]
    ppeak_idx = np.argmax((b*2) * n**2)
    ppeak_energy = round(b[ppeak_idx])
    subset = b > (ppeak_energy * 0.7)

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
    try:
        n, bins = hist_and_update(doi, 0, 0.97, **hist)
    except IndexError:
        # no energy windowed counts were present for this crystal
        return np.array([2200, 2000, 1800])

    pdf = np.cumsum(n[::-1]).astype(float)
    pdf = pdf / pdf[-1] * doi_quantiles[-1]
    return np.interp(doi_quantiles[:-1], pdf, bins[-2::-1])

def summarize_crystal(data, crystal, hist = None):
    if hist is None:
        ehist = dhist = {}
    else:
        ehist = {'n': hist[0,:-1], 'bins': hist[1]}
        dhist = {'n': hist[2,:-1], 'bins': hist[3]}

    peak, fwhm, *_ = fit_photopeak(data['E'], **ehist)
    mask = ((peak - fwhm/2) < data['E']) & (data['E'] < (peak + fwhm/2))
    thresholds = get_doi(data[mask]['D'], **dhist)
    return np.concatenate([[crystal, peak, fwhm], thresholds])

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

    # iterate over each subset of data for all but the last crystal ID
    res = np.array([summarize_crystal(d_sort[s:e], x, hist[x])
        for x,s,e in zip(crystals, locs[:-1], locs[1:])])

    # return a structured array with fitting results for each crystal
    return np.core.records.fromarrays(res.T, CrystalFitDtype)
