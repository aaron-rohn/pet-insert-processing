import os, json, tempfile, datetime
import base64
import zlib
import io
import numpy as np
import cv2 as cv
import tkinter as tk
import tkinter.filedialog
import concurrent.futures
from PIL import Image
from scipy import ndimage
from numpy.lib import recfunctions as rfn
import parallel_sort as psort

import petmr, crystal, pyelastix

import matplotlib.pyplot as plt

n_doi_bins = 4096
max_events = int(1e9)

SingleDType = np.dtype([
    ('block', np.uint16),
    ('eF', np.uint16), ('eR', np.uint16),
    ('x', np.uint16), ('y', np.uint16)])

CoincidenceDType = np.dtype([
    ('blkb', np.uint8), ('blka', np.uint8),
    ('tdiff', np.int8), ('prompt', np.uint8),
    ('eaF', np.uint16), ('eaR', np.uint16),
    ('ebF', np.uint16), ('ebR', np.uint16),
    ('xa', np.uint16), ('ya', np.uint16),
    ('xb', np.uint16), ('yb', np.uint16),
    ('abstime', np.uint16)])

VisualizationDType = np.dtype([
    ('E', np.uint16), ('D', np.uint16),
    ('X', np.uint16), ('Y', np.uint16)])

ListmodeDType = np.dtype([
    ('ra', np.uint8), ('xa', np.uint16),
    ('rb', np.uint8), ('xb', np.uint16),
    ('eb', np.uint8), ('ea', np.uint8),
    ('db', np.uint8), ('da', np.uint8),
    ('t',  np.uint16), ('td', np.int8),
    ('prompt', bool)])

class ListmodeLoader:
    bytes_per_ev = 8
    def __init__(self, fname, periods = None, counts = None):
        self.nev = os.path.getsize(fname) / ListmodeLoader.bytes_per_ev
        if counts is None:
            if periods is None:
                raise ValueError('Counts or periods must be specified')
            counts = int(self.nev / periods)

        self.fname = fname
        self.counts = counts

    def __len__(self):
        return int(self.nev / self.counts)

    def __iter__(self):
        self.ctr = int(0)
        return self

    def __next__(self):
        if self.ctr >= self.nev:
            raise StopIteration

        end = int(min(self.nev, self.ctr + self.counts))
        d = petmr.listmode_to_arr(self.fname, self.ctr, end)
        self.ctr = end

        return rfn.unstructured_to_structured(
                d, ListmodeDType)

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

def img_encode(img, encode = True, jpeg = False):
    """ encode and decode LUT and flood images into strings
    for insertion into the configuration json file. Apply
    jpeg compression for floods and zlib compression for LUTs

    account for some variation in the json files, like zlib
    compression of the flood images and uint16 vs int32 encoding
    of the LUTs
    """

    if encode:
        # encode -> produce ascii string from image
        if jpeg:
            # for floods, apply jpeg compression
            buf = io.BytesIO()
            img = (img / img.max() * 255.0).astype(np.uint8)
            Image.fromarray(img).save(buf, 'JPEG', quality = 10)
            img = buf.getvalue()
        else:
            img = img.astype(np.uint16)

        img = zlib.compress(img, level = zlib.Z_BEST_COMPRESSION)
        return base64.b64encode(img).decode('ascii')

    else:
        # decode -> produce image from ascii string
        i = base64.b64decode(bytes(img, 'ascii'))
        try:
            i = zlib.decompress(i)
        except zlib.error: # data was not compressed
            pass

        if jpeg:
            # decode jpeg bytes to image
            return np.asarray(Image.open(io.BytesIO(i)).convert('L')).astype(np.intc)
        else:
            dt = np.intc if len(i) > (512*512*2) else np.uint16 # handle different LUT types
            return np.frombuffer(i, dt).astype(np.intc).reshape(512,512)

def cfg_img_display(cfg_file, block):
    with open(cfg_file, 'r') as fd:
        cfg = json.load(fd)

    lut = img_encode(cfg[str(block)]['LUT'], encode = False)
    fld = img_encode(cfg[str(block)]['FLD'], encode = False, jpeg = True)

    plt.imshow(lut)
    plt.show()

    plt.imshow(fld)
    plt.show()

def flood_preprocess(fld):
    fld = fld.astype(float)

    # remove the bright spot observed in the center of some floods
    width,middle = int(15/2),int(fld.shape[0]/2)
    window = slice(middle-width,middle+width)
    center = fld[window,window]
    thr,val = np.quantile(center, [0.96, 0.8])
    center[center > thr] = val

    fld = ndimage.gaussian_laplace(fld, 1.5)
    fld /= fld.min()
    fld[fld < 0] = 0
    with np.errstate(invalid='ignore'):
        fld /= ndimage.gaussian_filter(fld, 20)
    np.nan_to_num(fld, False, 0, 0, 0)
    return fld

def lut_edges(lut: np.array) -> np.ma.array:
    yd = np.diff(lut, axis = 0, prepend = lut.max()) != 0
    xd = np.diff(lut, axis = 1, prepend = lut.max()) != 0
    overlay = np.logical_or(xd, yd)
    overlay = ndimage.binary_dilation(overlay, np.ones((3,3)))
    return np.ma.array(overlay, mask = (overlay == 0))

def create_cfg_vals(data, lut, blk, cfg, energy_hist = None):
    """
    Config json format:

    - block
        - photopeak
        - FWHM
        - crystal
            - photopeak
            - FWHM
            - DOI
    """

    blk_vals = cfg[blk]
    xtal_vals = blk_vals['crystal'] = {}

    peak, fwhm, *_ = crystal.fit_photopeak(data['E'])
    blk_vals['photopeak'] = peak
    blk_vals['FWHM'] = fwhm 

    pks = crystal.calculate_lut_statistics(lut, data)

    for xtal, doi_thr, ppeaks, fwhms in pks:
        this_xtal = xtal_vals[int(xtal)] = {}
        this_xtal['photopeak'] = list(ppeaks)
        this_xtal['FWHM'] = list(fwhms)
        this_xtal['DOI'] = list(doi_thr)

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

def create_scaled_calibration(data, cfg, sync = None):
    luts    = np.ones((petmr.nblocks, 512, 512), np.intc) * petmr.ncrystals_total
    ppeak   = np.ones((petmr.nblocks, petmr.ncrystals_total, petmr.ndoi), np.double) * -1;
    doi     = np.ones((petmr.nblocks, petmr.ncrystals_total, petmr.ndoi), np.double) * -1;

    with concurrent.futures.ThreadPoolExecutor(4) as ex:
        futs = [ex.submit(scale_single_block,
            data, blk, cfg, sync) for blk in range(petmr.nblocks)]

    for blk, f in enumerate(futs):
        luts[blk], ppeak[blk], doi[blk] = f.result()

    return luts, ppeak, doi

def scale_single_block(data, blk, cfg, sync = None):
    d = load_block_coincidence_data(data, blk)

    ppeak = np.ones((petmr.ncrystals_total, petmr.ndoi), np.double) * -1
    doi   = np.ones((petmr.ncrystals_total, petmr.ndoi), np.double) * -1

    # get the energy histogram, photopeak, and energy windowed flood

    peak, fwhm, *_ = crystal.fit_photopeak(d['E']) # block photopeak
    ppeak[:] = peak # block photopeak is default

    lld, uld = peak-fwhm, peak+fwhm
    mask = (lld < d['E']) & (d['E'] < uld)
    windowed = d[mask]

    fld, *_ = np.histogram2d(windowed['Y'], windowed['X'],
            bins = 512, range = [[0,511],[0,511]])
    fld = flood_preprocess(fld)

    # load the reference flood and LUT, and generate the warped LUT

    ref_fld = img_encode(cfg[str(blk)]['FLD'], encode = False, jpeg = True)
    ref_lut = img_encode(cfg[str(blk)]['LUT'], encode = False)

    # reference flood is already preprocessed and scaled to 255.0
    ref_fld = ref_fld.astype(float) / ref_fld.max() * fld.max()
    ref_fld = ref_fld.astype(fld.dtype)

    try:
        xf, yf = register(ref_fld, fld, nres = 2, niter = 250, type = 'AFFINE')
        lut = remap(ref_lut, xf, yf)
    except RuntimeError as e:
        print(f'block {blk}: error registering flood to reference ({e})')
        lut = ref_lut

    edges = lut_edges(lut)

    # get energy and DOI calibration values

    pks = crystal.calculate_lut_statistics(lut, d, workers = 4)

    for xtal, doi_thr, ppeaks, fwhms in pks:
        ppeak[xtal] = ppeaks
        doi[xtal] = doi_thr

    if sync is not None:
        block, lk, status_queue, floods_queue = sync
        with lk:
            block += 1
            status_queue.put(float(block / petmr.nblocks * 100))
            floods_queue.put((fld,edges))

    return lut, ppeak, doi
