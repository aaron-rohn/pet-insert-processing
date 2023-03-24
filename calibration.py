import os, json, tempfile, datetime
import numpy as np
import cv2 as cv
import tkinter as tk
import tkinter.filedialog
import concurrent.futures
from scipy import ndimage
from numpy.lib import recfunctions as rfn

import petmr, crystal, pyelastix

n_doi_bins = 4096
max_events = int(1e9)

SingleDType = np.dtype([
    ('block', np.uint16),
    ('eF', np.uint16), ('eR', np.uint16),
    ('x', np.uint16), ('y', np.uint16)])

CoincidenceDType = np.dtype([
    ('blkb', np.uint8), ('blka', np.uint8),
    ('prompt', np.uint8), ('tdiff', np.int8),
    ('eaF', np.uint16), ('eaR', np.uint16),
    ('ebF', np.uint16), ('ebR', np.uint16),
    ('xa', np.uint16), ('ya', np.uint16),
    ('xb', np.uint16), ('yb', np.uint16),
    ('abstime', np.uint16)])

VisualizationDType = np.dtype([
    ('E', np.uint16), ('D', np.uint16),
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

def load_all_blocks(data_in, start = 0, end = max_events):
    data = np.memmap(data_in, np.uint16).reshape(-1,11)
    subset = data[start:end]

    order = np.argsort(subset[:,0]) # can be replaced with cython pargsort
    d = subset[order]

    uq, idx = np.unique(d[:,0], return_index = True)
    idx = np.append(idx, [d.shape[0]-1])
    ablks, bblks = uq >> 8, uq & 0xFF

    def get_idx(b, blks, idx):
        i = np.nonzero(blks == b)[0]
        if len(i) == 0: return np.array([], int)
        return np.concatenate([np.arange(s,e) for s,e in zip(idx[i], idx[i+1])])

    def extract_data(din, dout, idx, out_row, in_col):
        slc = slice(out_row, out_row + idx.size, None)
        if idx.size > 0:
            dout['E'][slc] = d[idx,in_col+0] + d[idx,in_col+1]
            dout['X'][slc] = d[idx,in_col+4]
            dout['Y'][slc] = d[idx,in_col+5]
            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                dout['D'][slc] = d[idx,in_col].astype(float) * n_doi_bins / dout['E'][slc]
    
    def load(blk):
        idxa = get_idx(blk, ablks, idx)
        idxb = get_idx(blk, bblks, idx)
        na, nb = idxa.size, idxb.size

        tf = tempfile.NamedTemporaryFile()
        arr = np.memmap(tf.name, VisualizationDType, mode = 'w+', shape = (na+nb,))

        extract_data(d, arr, idxa, 0, 2)
        extract_data(d, arr, idxb, na, 4)
        return arr

    with concurrent.futures.ThreadPoolExecutor(os.cpu_count()) as ex:
        futs = [ex.submit(load, blk) for blk in range(petmr.nblocks)]

    return {blk: f.result() for blk,f in enumerate(futs)}

def copy_to_vis(din, dout, ab, offset):
    subset = dout[offset:(offset + din.size)]
    ef, er, x, y = f'e{ab}F', f'e{ab}R', f'x{ab}', f'y{ab}'
    subset['E'] = din[ef] + din[er]
    subset['X'] = din[x]
    subset['Y'] = din[y]
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        subset['D'] = din[ef].astype(float) * n_doi_bins / subset['E']

def load_block_coincidence_data(data, blk):
    idxa = np.nonzero(data['blka'] == blk)[0]
    idxb = np.nonzero(data['blkb'] == blk)[0]
    tf = tempfile.NamedTemporaryFile()
    arr = np.memmap(tf.name, VisualizationDType, mode = 'w+',
            shape = (idxa.size + idxb.size,))
    copy_to_vis(data[idxa], arr, 'a', 0)
    copy_to_vis(data[idxb], arr, 'b', idxa.size)
    return arr

def load_block_singles_data(data, blk):
    idx = np.nonzero(data['block'] == blk)[0]
    tf = tempfile.NamedTemporaryFile()
    arr = np.memmap(tf.name, VisualizationDType,
            mode = 'w+', shape = (idx.size,))
    copy_to_vis(data[idx], arr, '', 0)
    return arr

def flood_preprocess(fld):
    fld = fld.astype(float)
    fld = ndimage.gaussian_laplace(fld, 1.5)
    fld /= fld.min()
    fld[fld < 0] = 0
    with np.errstate(invalid='ignore'):
        fld /= ndimage.gaussian_filter(fld, 20)
    np.nan_to_num(fld, False, 0, 0, 0)
    return fld

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
    for row in pks:
        this_xtal = xtal_vals[int(row['crystal'])] = {}
        this_xtal['energy'] = {'photopeak': row['peak'], 'FWHM': row['FWHM']}
        this_xtal['DOI'] = row[['5mm','10mm','15mm']].tolist()

def register(src, dst, nres = 4, niter = 500, spacing = 32, type = 'BSPLINE'):
    pars = pyelastix.get_default_params(type = type)
    pars.NumberOfResolution = nres
    pars.MaximumNumberOfIterations = niter
    pars.FinalGridSpacingInPhysicalUnits = spacing

    _, (xf, yf) = pyelastix.register(
            np.ascontiguousarray(src),
            np.ascontiguousarray(dst),
            pars, verbose = 0)

    nr, nc = src.shape
    x = np.tile(np.arange(nc, dtype = np.float32), nr).reshape(src.shape)
    return x + xf, x.T + yf

def remap(lut, x, y):
    return cv.remap(lut, x, y, cv.INTER_NEAREST,
            borderMode = cv.BORDER_CONSTANT, borderValue = int(lut.max()))

def create_scaled_calibration(data, cfgdir, sync = None):
    luts    = np.ones((petmr.nblocks, 512, 512), np.intc) * petmr.ncrystals_total
    ppeak   = np.ones((petmr.nblocks, petmr.ncrystals_total), np.double) * -1;
    doi     = np.ones((petmr.nblocks, petmr.ncrystals_total, petmr.ndoi), np.double) * -1;

    with concurrent.futures.ThreadPoolExecutor(16) as ex:
        futs = [ex.submit(scale_single_block, data, blk, cfgdir, sync)
                for blk in range(petmr.nblocks)]

    for blk, f in enumerate(futs):
        luts[blk], ppeak[blk], doi[blk] = f.result()

    return luts, ppeak, doi

def scale_single_block(data, blk, cfgdir, sync = None):
    blk_data = load_block_coincidence_data(data, blk)
    ppeak = np.ones(petmr.ncrystals_total, np.double) * -1
    doi = np.ones((petmr.ncrystals_total, petmr.ndoi), np.double) * -1

    # get the energy histogram and block photopeak

    es = blk_data['E']
    rng = np.quantile(es, [0.01, 0.99])
    nbins = int(round((rng[1] - rng[0]) / 10))
    n, bins = np.histogram(es, bins = nbins, range = rng)
    peak, fwhm, *_ = crystal.fit_photopeak(n = n, bins = bins[:-1])
    ppeak[:] = peak # block photopeak is default

    # get the energy windowed flood

    idx = np.nonzero(((peak-fwhm) < es) & (es < (peak+fwhm)))[0]
    x, y = blk_data['X'][idx], blk_data['Y'][idx]
    fld, *_ = np.histogram2d(y, x, bins = 512, range = [[0,511],[0,511]])

    # load the reference flood and LUT, and generate the warped LUT

    ref_fld = np.fromfile(f'{cfgdir}/flood/block{blk}.raw', np.intc).reshape(512,512)
    xf, yf = register(ref_fld, fld, nres = 2, niter = 250, type = 'AFFINE')
    ref_lut = np.fromfile(f'{cfgdir}/lut/block{blk}.lut', np.intc).reshape(512,512)
    lut = remap(ref_lut, xf, yf)

    # get energy and DOI calibration values

    stats = crystal.calculate_lut_statistics(lut, blk_data)
    ppeak[stats['crystal']] = stats['peak']
    doi[stats['crystal']] = rfn.structured_to_unstructured(
            stats[['5mm', '10mm', '15mm']])

    if sync is not None:
        block, lk, q = sync
        with lk:
            block += 1
            q.put(float(block / petmr.nblocks * 100))

    return lut, ppeak, doi
