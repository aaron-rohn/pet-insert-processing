import os
import datetime
import numpy as np
from scipy import optimize
from collections import defaultdict
import matplotlib.pyplot as plt
import parallel_sort as psort
import concurrent.futures

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
    try:
        n, bins = hist_and_update(energy, 0.01, 0.97, **hist)
    except IndexError:
        # `energy` probably had length 0
        return -1, -1

    b = bins[:-1]

    # estimate the photopeak position and correct subset of data
    ppeak_idx = np.argmax(b**2 * n)
    ppeak_energy = round(b[ppeak_idx])
    subset = b > (ppeak_energy * 0.5)

    # estimate the parameters of the linear offset
    xl, xh = ppeak_energy*0.8, ppeak_energy*1.2
    xl, xh = np.searchsorted(b, [xl,xh])
    xl, xh = np.clip([xl, xh], 0, len(n)-1)
    intercept = (n[xh] + n[xl]) / 2
    slope = (n[xh] - n[xl]) / (xh - xl)

    # initial parameter estimates
    pinit = (ppeak_energy,       # photopeak position
             ppeak_energy * 0.2, # photopeak FWHM
             n[ppeak_idx],       # photopeak amplitude
             slope,              # slope of the linear offset
             intercept)          # linear offset at the photopeak
    
    # parameter bounds for fitting
    bounds = ((ppeak_energy/2, ppeak_energy*2),
              (ppeak_energy * 0.05, ppeak_energy * 0.5),
              (n[ppeak_idx] * 0.1, n[ppeak_idx] * 2),
              (-np.inf, np.inf),
              (-np.inf, np.inf))

    try:
        # fit the data with a gaussian and linear offset
        popt, *_ = optimize.curve_fit(photopeak,
                                      b[subset], n[subset],
                                      p0 = pinit, bounds = list(zip(*bounds)))
    except (RuntimeError, ValueError):
        popt = pinit

    # return photopeak and fwhm
    return round(popt[0]), round(popt[1])

def get_doi(doi, **hist):
    n, bins = hist_and_update(doi, 0, 1, **hist)

    # The PDF ranges from 0 to about 0.8, corresponding to the
    # probability of photon interaction in the crystal

    # the histogram n is reversed since larger DOI values
    # correspond to the start of the crystal.
    pdf = np.cumsum(n[::-1]).astype(float)
    pdf = pdf / pdf[-1] * doi_quantiles[-1]

    # the DOI bin values are reversed to match the counts (as above)
    # and are truncated to the same length (removing the last value)
    return np.interp(doi_quantiles, pdf, bins[-2::-1])

def summarize_crystal(data, crystal, hist = None):
    data.sort(order = 'D')

    try:
        # get DOI thresholds based on non-masked data
        thresholds = get_doi(data['D'])
    except IndexError:
        invalid = np.full(petmr.ndoi, -1)
        return crystal, invalid, invalid, invalid

    # thresholds are in decreasing order
    thresholds = np.round(thresholds)
    idx = np.searchsorted(data['D'], thresholds)

    fits = [fit_photopeak(data['E'][l:g]) for g,l in zip(idx[:-1], idx[1:])]
    ppeaks, fwhms = zip(*fits)

    # The first threshold corresponds to the front of the crystal
    return crystal, thresholds[1:], ppeaks, fwhms

def calculate_lut_statistics(lut, data, workers = None):
    idx = np.ravel_multi_index((data['Y'],data['X']), lut.shape) # 1d event position
    idx = lut.flat[idx]

    order = psort.argsort(idx)
    idx = idx[order]
    data = data[order]

    # get the index corresponding to each crystal, 0-361
    # note that the last bin (361) corresponds to invalid events
    locs = np.searchsorted(idx, np.arange(petmr.ncrystals_total + 1))

    futs = []
    with concurrent.futures.ProcessPoolExecutor(workers or os.cpu_count()) as ex:
        for i, (l,g) in enumerate(zip(locs[:-1], locs[1:])):
            if i != petmr.ncrystals_total:
                f = ex.submit(summarize_crystal, data[l:g], i)
                futs.append(f)

    return [f.result() for f in futs]
