import os
import pandas as pd
import numpy as np
import concurrent.futures
from scipy import optimize
import matplotlib.pyplot as plt

doi_bins = np.array([5,10,15,20])
lyso_attn_length = 12
doi_quantiles = 1 - np.exp(-doi_bins / lyso_attn_length)

def photopeak(x, peak, fwhm, amp, slope, intercept):
    # model the photopeak as a FWHM-parametrized gaussian plus a linear offset
    y = amp * np.exp(-4 * np.log(2) * ((x-peak) ** 2) / (fwhm**2))
    return y + (x-peak)*slope + intercept

def fit_photopeak(energy = None, n = None, bins = None):
    if energy is not None:
        n,bins = np.histogram(energy, bins = 100,
                range = np.quantile(energy, [0, 0.95]))
        bins = bins[:-1]

    ppeak_idx = np.argmax((bins*2) * n**2)
    ppeak_energy = round(bins[ppeak_idx])
    subset = bins > (ppeak_energy * 0.7)

    xl, xh = ppeak_energy*0.8, ppeak_energy*1.2
    xl, xh = np.searchsorted(bins, [xl,xh])
    xl, xh = np.clip([xl, xh], 0, len(n)-1)
    intercept = (n[xh] + n[xl]) / 2
    slope = (n[xh] - n[xl]) / (xh - xl)

    pinit = (ppeak_energy,
             ppeak_energy * 0.2,
             n[ppeak_idx],
             slope,
             intercept)
    
    bounds = ((ppeak_energy/2, ppeak_energy*2),
              (ppeak_energy * 0.05, ppeak_energy * 0.5),
              (n[ppeak_idx] * 0.1, n[ppeak_idx] * 2),
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

def summarize_crystal(grp, name = None):
    if name is None: name = grp.name
    e = grp['es']
    peak, fwhm, *_ = fit_photopeak(e)
    fwhm = abs(fwhm)

    g = grp[((peak-fwhm/2) < e) & (e < (peak+fwhm/2))]
    try:
        thresholds = get_doi(g['doi'])
    except IndexError:
        print(f'Error measuring DOI thresholds: {peak}, {fwhm}')
        thresholds = np.array([2200,2000,1800])

    vals = np.concatenate([[peak,fwhm], thresholds])
    return pd.DataFrame(vals[None,:],
                        columns = ['peak','FWHM','5mm','10mm','15mm'])

def calculate_lut_statistics(lut, data):
    lut_df = pd.DataFrame({
        'x'  : np.tile(np.arange(lut.shape[1]), lut.shape[0]),
        'y'  : np.repeat(np.arange(lut.shape[0]), lut.shape[1]),
        'lut': lut.flat})

    lm_df = pd.DataFrame(data, columns = ['es', 'doi', 'x', 'y'])
    lm_df = lm_df.merge(lut_df, on = ['x', 'y'])
    lm_df = lm_df.drop(columns = ['x', 'y'])
    lm_df = lm_df.groupby('lut')

    with concurrent.futures.ThreadPoolExecutor(os.cpu_count()) as ex:
        fut = [ex.submit(summarize_crystal, g, n) for n, g in lm_df]
        ret = [f.result() for f in fut]
    return pd.concat(ret, ignore_index = True)
    #return lm_df.apply(summarize_crystal)
