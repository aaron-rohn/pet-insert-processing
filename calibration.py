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

        # sample events at regular intervals (e.g. every 100e6 events)
        idx = np.linspace(0, nev-1, nperiods*naverage + 1, dtype = np.ulonglong)

        # get the absolute time (in 0.1s intervals) at each sampled event
        times = self.data['abstime'][idx].astype(float)

        # correct for instances where the 16 bit counter rolls over (every ~2hrs)
        rollover = np.diff(times)
        rollover = np.nonzero(rollover < 0)[0]
        for i in rollover:
            times[i+1:] += 2**16

        # event rate is the number of events over the elapsed time between samples
        ev_rate = np.diff(idx) / np.diff(times)

        # average consecutive samples to reduce noise
        self.event_rate = ev_rate.reshape(-1,naverage).mean(1)
        self.times = times[:-1].reshape(-1,naverage).mean(1)

        # calculate file positions corresponding to each time point
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
    """ group the input data by block to hopefully accelerate data loading """

    data = np.memmap(data_in, np.uint16).reshape(-1,11)[start:end]

    # the first column represents the coincidence block pairs
    # sort events according to the block pair where they occurred
    order = np.argsort(data[:,0]) # can be replaced with cython pargsort

    # get a view of the same data in a structured format, and sort
    ds = np.memmap(data_in, CoincidenceDType)[start:end][order]
    blocks = data[:,0][order]

    # determine the indices corresponding to each unique block pair
    uq, idx = np.unique(blocks, return_index = True)
    idx = np.append(idx, [blocks.size-1])
    ablks, bblks = uq >> 8, uq & 0xFF

    def get_idx(b, blks, idx):
        """ identify all indices corresponding to a particular block """
        i = np.nonzero(blks == b)[0]
        if len(i) == 0: return np.array([], int)
        return np.concatenate([np.arange(s,e) for s,e in zip(idx[i], idx[i+1])])

    def load(blk):
        idxa = get_idx(blk, ablks, idx)
        idxb = get_idx(blk, bblks, idx)
        na, nb = idxa.size, idxb.size

        tf = tempfile.NamedTemporaryFile()
        arr = np.memmap(tf.name, VisualizationDType,
                mode = 'w+', shape = (na+nb,))

        copy_to_vis(ds[idxa], arr[:na], 'a')
        copy_to_vis(ds[idxb], arr[na:], 'b')
        return arr

    with concurrent.futures.ThreadPoolExecutor(os.cpu_count()) as ex:
        futs = [ex.submit(load, blk) for blk in range(petmr.nblocks)]

    return {blk: f.result() for blk,f in enumerate(futs)}

def copy_to_vis(din, dout, ab = ''):
    # copy data from CoincidenceDType/SingleDType to VisualizationDType
    ef, er, x, y = f'e{ab}F', f'e{ab}R', f'x{ab}', f'y{ab}'
    dout['E'] = din[ef] + din[er]
    dout['X'] = din[x]
    dout['Y'] = din[y]
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        dout['D'] = din[ef].astype(float) * n_doi_bins / dout['E']

def load_block_coincidence_data(data, blk):
    idxa = np.nonzero(data['blka'] == blk)[0]
    idxb = np.nonzero(data['blkb'] == blk)[0]
    tf = tempfile.NamedTemporaryFile()
    arr = np.memmap(tf.name, VisualizationDType, mode = 'w+',
            shape = (idxa.size + idxb.size,))
    copy_to_vis(data[idxa], arr[:idxa.size], 'a')
    copy_to_vis(data[idxb], arr[idxa.size:], 'b')
    return arr

def load_block_singles_data(data, blk):
    idx = np.nonzero(data['block'] == blk)[0]
    tf = tempfile.NamedTemporaryFile()
    arr = np.memmap(tf.name, VisualizationDType,
            mode = 'w+', shape = (idx.size,))
    copy_to_vis(data[idx], arr)
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
    peak, fwhm, *_ = crystal.fit_photopeak(data['E'])
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

def create_scaled_calibration(data, cfgdir, hist, sync = None):
    luts    = np.ones((petmr.nblocks, 512, 512), np.intc) * petmr.ncrystals_total
    ppeak   = np.ones((petmr.nblocks, petmr.ncrystals_total), np.double) * -1;
    doi     = np.ones((petmr.nblocks, petmr.ncrystals_total, petmr.ndoi), np.double) * -1;

    with concurrent.futures.ThreadPoolExecutor(os.cpu_count()) as ex:
        futs = [ex.submit(scale_single_block,
            data, blk, cfgdir, hist[blk], sync) for blk in range(petmr.nblocks)]

    for blk, f in enumerate(futs):
        luts[blk], ppeak[blk], doi[blk] = f.result()

    return luts, ppeak, doi

def scale_single_block(data, blk, cfgdir, hist, sync = None):
    d = load_block_coincidence_data(data, blk)
    ppeak = np.ones(petmr.ncrystals_total, np.double) * -1
    doi = np.ones((petmr.ncrystals_total, petmr.ndoi), np.double) * -1

    # get the energy histogram, photopeak, and energy windowed flood

    peak, fwhm, *_ = crystal.fit_photopeak(d['E'])
    windowed = d[((peak-fwhm) < d['E']) & (d['E'] < (peak+fwhm))]
    fld, *_ = np.histogram2d(windowed['Y'], windowed['X'],
            bins = 512, range = [[0,511],[0,511]])

    ppeak[:] = peak # block photopeak is default

    # load the reference flood and LUT, and generate the warped LUT

    ref_fld = np.fromfile(f'{cfgdir}/flood/block{blk}.raw', np.intc).reshape(512,512)
    ref_lut = np.fromfile(f'{cfgdir}/lut/block{blk}.lut', np.intc).reshape(512,512)

    xf, yf = register(ref_fld, fld, nres = 2, niter = 250, type = 'AFFINE')
    lut = remap(ref_lut, xf, yf)

    # get energy and DOI calibration values

    stats = crystal.calculate_lut_statistics(lut, d, hist)
    ppeak[stats['crystal']] = stats['peak']
    doi[stats['crystal']] = rfn.structured_to_unstructured(
            stats[['5mm', '10mm', '15mm']])

    if sync is not None:
        block, lk, q = sync
        with lk:
            block += 1
            q.put(float(block / petmr.nblocks * 100))

    return lut, ppeak, doi
