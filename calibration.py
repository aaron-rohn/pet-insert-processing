import os, json, tempfile
import numpy as np
import cv2 as cv
import tkinter as tk
import tkinter.filedialog
from scipy import ndimage

import petmr, crystal, pyelastix

n_doi_bins = 4096
max_events = int(1e9)

CoincidenceDType = np.dtype([
    ('blkb', np.uint8), ('blka', np.uint8),
    ('prompt', np.uint8), ('tdiff', np.int8),
    ('eaF', np.uint16), ('eaR', np.uint16),
    ('ebF', np.uint16), ('ebR', np.uint16),
    ('xa', np.uint16), ('ya', np.uint16),
    ('xb', np.uint16), ('yb', np.uint16),
    ('abstime', np.uint16)])

VisualizationDType = np.dtype([
    ('E', np.uint16), ('DOI', np.uint16),
    ('X', np.uint16), ('Y', np.uint16)])

class CoincidenceFileHandle:
    def __init__(self, data, nperiods = 10, naverage = 10):
        self.data = np.memmap(data, CoincidenceDType)
        nev = self.data.shape[0]

        idx = np.linspace(0, nev-1, nperiods*naverage + 1, dtype = np.ulonglong)
        times = self.data['abstime'][idx].astype(float)

        rollover = np.diff(times)
        rollover = np.where(rollover < 0)[0]
        for i in rollover:
            times[i+1:] += 2**16

        ev_rate = np.diff(idx) / np.diff(times)

        self.event_rate = ev_rate.reshape(-1,naverage).mean(1)
        self.times = times[:-1].reshape(-1,naverage).mean(1)
        self.idx = idx[:nperiods*naverage:naverage]
        self.fpos = self.idx * CoincidenceDType.itemsize

        duration = (times[-1] - times[0]) / 10
        print(f'{nev} events over {duration}: {nev / duration} cps')

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i == self.idx.size:
            raise StopIteration
        
        i = self.i

        s, si = self.fpos[i], self.idx[i]
        e, ei = (-1,-1) if i == self.idx.size-1 else (self.fpos[i+1], self.idx[i+1])
        d = self.data[si:ei]

        self.i += 1
        return s, e, d

    def events_per_period(self, max_events = None):
        n = int(np.diff(self.idx).mean())
        return n if max_events is None else min(n, int(max_events))

def load_block_coincidence_data(data, blk):
    idxa = np.nonzero(data['blka'] == blk)[0]
    idxb = np.nonzero(data['blkb'] == blk)[0]

    rowa = data[idxa]
    rowb = data[idxb]

    tf = tempfile.NamedTemporaryFile()
    arr = np.memmap(tf.name, VisualizationDType, mode = 'w+',
            shape = (len(idxa) + len(idxb),))

    arr['E'] = np.concatenate(
            [rowa['eaF'] + rowa['eaR'],
             rowb['ebF'] + rowb['ebR']])

    tmp = np.concatenate([rowa['eaF'], rowb['ebF']]).astype(float)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        arr['DOI'] = tmp * n_doi_bins / arr['E']

    arr['X'] = np.concatenate([rowa['xa'], rowb['xb']])
    arr['Y'] = np.concatenate([rowa['ya'], rowb['yb']])

    return arr

def load_block_singles_data(data, blk):
    # data returned from the C library is not structured
    # columns are block, e_front, e_rear, x, y

    idx = np.nonzero(data[:,0] == blk)[0]
    subset = data[idx]

    tf = tempfile.NamedTemporaryFile()
    arr = np.memmap(tf.name, VisualizationDType,
            mode = 'w+', shape = (len(idx),))

    # Energy sum -> eF + eR
    arr['E'] = subset[:,1] + subset[:,2]

    # DOI -> eF / eSUM
    tmp = subset[:,1].astype(float)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        arr['DOI'] = tmp * n_doi_bins / arr['E']

    arr['X'] = subset[:,3]
    arr['Y'] = subset[:,4]

    return arr

def apply_energy_threshold(data, cfg, blk):
    blk_ppeak = cfg[str(blk)]['photopeak']
    blk_res = cfg[str(blk)]['FWHM']
    lld, uld = blk_ppeak - blk_res, blk_ppeak + blk_res

    es = data['E']
    return np.where((lld < es) & (es < uld))[0]

def flood_preprocess(fld):
    fld = fld.astype(float)
    fld = ndimage.gaussian_laplace(fld, 1.5)
    fld /= fld.min()
    fld[fld < 0] = 0
    with np.errstate(invalid='ignore'):
        fld /= ndimage.gaussian_filter(fld, 20)
    np.nan_to_num(fld, False, 0, 0, 0)
    return fld

def make_cfg_subdir(cfg_dir, subdir):
    cfg = os.path.join(cfg_dir, str(subdir))
    fld = os.path.join(cfg, 'flood')
    lut = os.path.join(cfg, 'lut')

    for d in [cfg, fld, lut]:
        try:
            os.mkdir(d)
        except FileExistsError: pass

    return cfg

def create_cfg_vals(data, lut, blk, cfg, energy_hist = None):
    """
    Config json format:

    - block
        - photopeak
        - FWHM
        - crystal
            - energy
                - photopeak
                - FWHM 
            - DOI
                - thresholds
    """

    blk_vals = cfg[blk] = {}
    xtal_vals = blk_vals['crystal'] = {}

    if energy_hist is None:
        es = data['E']
        rng = np.quantile(es, [0.01, 0.99])
        nbins = int(round((rng[1] - rng[0]) / 10))
        n, bins = np.histogram(es, bins = nbins, range = rng)
    else:
        n, bins = energy_hist

    peak, fwhm, *_ = crystal.fit_photopeak(n = n, bins = bins[:-1])

    blk_vals['photopeak'] = peak
    blk_vals['FWHM'] = fwhm 

    pks = crystal.calculate_lut_statistics(lut, data)
    for lut, row in pks.iterrows():
        this_xtal = xtal_vals[lut] = {}
        this_xtal['energy'] = {'photopeak': row['peak'], 'FWHM': row['FWHM']}
        this_xtal['DOI'] = row[['5mm','10mm','15mm']].tolist()

def register(src, dst, nres = 4, niter = 500, spacing = 32):
    pars = pyelastix.get_default_params()
    pars.NumberOfResolution = nres
    pars.MaximumNumberOfIterations = niter
    pars.FinalGridSpacingInPhysicalUnits = spacing

    _, (xf, yf) = pyelastix.register(src, dst, pars, verbose = 0)
    x = np.tile(np.arange(512, dtype = np.float32), 512).reshape(512,512)
    return x + xf, x.T + yf

def remap(lut, x, y):
    return cv.remap(lut, x, y, cv.INTER_NEAREST,
            borderMode = cv.BORDER_CONSTANT, borderValue = int(lut.max()))


def create_scaled_calibration(subset):
    lut_dim = [512,512]
    calib_dims = (petmr.nblocks, petmr.ncrystals_total)

    luts = np.ones([petmr.nblocks] + lut_dim, dtype = np.intc) * petmr.ncrystals_total
    ppeak = np.ones(calib_dims, np.double) * -1;
    doi = np.ones(calib_dims + (petmr.ndoi,), np.double) * -1;

    for blk in range(64):
        blk_data = load_block_coincidence_data(subset, blk)

        # get the energy histogram and block photopeak

        es = blk_data['E']
        rng = np.quantile(es, [0.01, 0.99])
        nbins = int(round((rng[1] - rng[0]) / 10))
        n, bins = np.histogram(es, bins = nbins, range = rng)
        peak, fwhm, *_ = crystal.fit_photopeak(n = n, bins = bins[:-1])
        ppeak[blk] = peak

        # get the energy windowed flood

        idx, *_ = np.nonzero(((peak-fwhm) < es) & (es < (peak+fwhm)))
        x, y = blk_data['X'][idx], blk_data['Y'][idx]
        fld, *_ = np.histogram2d(y, x, bins = 512, range = [[0,511],[0,511]])
        fld = flood_preprocess(fld)

        # load the reference flood and LUT, and generate the warped LUT

        ref_fld = np.fromfile(f'{cfgdir}/flood/block{blk}.raw', np.intc).reshape(512,512)
        ref_fld = flood_preprocess(ref_fld)

        fields = register(ref_fld, fld, 2, 250, 32)

        ref_lut = np.fromfile(f'{cfgdir}/lut/block{blk}.lut', np.intc).reshape(512,512)
        luts[blk] = remap(ref_lut, *fields)

        # get energy and DOI calibration values

        stats_df = crystal.calculate_lut_statistics(luts[blk], blk_data)
        stats_df = stats_df.filter(items = range(petmr.ncrystals_total), axis = 0)
        ppeak[blk][stats_df.index] = stats_df['peak'].to_numpy()
        doi[blk][stats_df.index] = stats_df[['5mm', '10mm', '15mm']].to_numpy()

    return luts, ppeak, doi
