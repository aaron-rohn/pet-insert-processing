import os, petmr, json, pyelastix, crystal, tempfile
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from data_loader import coincidence_cols, n_doi_bins, max_events
from scipy import ndimage

def open_coincidence_file(f, nperiods = 10, naverage = 10):
    if isinstance(f, str):
        sz = os.path.getsize(f)
        nevents = int((sz/2) // coincidence_cols)
        data = np.memmap(f, np.uint16, shape = (nevents, coincidence_cols))
    else:
        data = f
        nevents = data.shape[0]

    ev_per_period = int(nevents / nperiods)
    times_idx = np.linspace(0, nevents-1, nperiods*naverage + 1, dtype = np.ulonglong)
    times = data[times_idx,10].astype(float)

    rollover = np.diff(times)
    rollover = np.where(rollover < 0)[0]
    for i in rollover:
        times[i+1:] += 2**16

    ev_rate = np.diff(times_idx) / np.diff(times)

    times = times[:-1].reshape(-1,naverage).mean(1)
    ev_rate = ev_rate.reshape(-1,naverage).mean(1)

    fpos = np.cumsum(times_idx)[:nperiods*naverage:naverage]
    fpos *= (2 * coincidence_cols)

    return data, ev_per_period, ev_rate, times, fpos

def load_block_coincidence_data(data, blka, blkb, blk):
    idxa = np.where(blka == blk)[0]
    idxb = np.where(blkb == blk)[0]

    rowa = data[idxa,:]
    rowb = data[idxb,:]

    tf = tempfile.NamedTemporaryFile()

    shape = (len(idxa) + len(idxb), 4)
    arr = np.memmap(tf.name, np.uint16, mode = 'w+', shape = shape)

    # Energy sum
    arr[:,0] = np.concatenate(
            [rowa[:,2] + rowa[:,3], rowb[:,4] + rowb[:,5]])

    # DOI
    tmp = np.concatenate([rowa[:,2], rowb[:,4]]).astype(float)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        tmp *= (n_doi_bins / arr[:,0])
    arr[:,1] = tmp

    arr[:,2] = np.concatenate([rowa[:,6], rowb[:,8]]) # X
    arr[:,3] = np.concatenate([rowa[:,7], rowb[:,9]]) # Y

    return arr

def load_block_singles_data(d, blocks, blk):
    tf = tempfile.NamedTemporaryFile()
    idx = np.where(blocks == blk)[0]
    arr = np.memmap(tf.name, np.uint16,
            mode = 'w+', shape = (len(idx), 4))

    # Energy sum -> eF + eR
    arr[:,0] = d[1][idx] + d[2][idx]

    # DOI -> eF / eSUM
    tmp = d[1][idx].astype(float)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        tmp *= (n_doi_bins / arr[:,0])
    arr[:,1] = tmp

    # X, Y
    arr[:,2] = d[3][idx]
    arr[:,3] = d[4][idx]

    return arr

def apply_energy_threshold(data, cfg, blk):
    blk_ppeak = cfg[str(blk)]['photopeak']
    blk_res = cfg[str(blk)]['FWHM']
    lld, uld = blk_ppeak - blk_res, blk_ppeak + blk_res

    es = data[:,0]
    return np.where((lld < es) & (es < uld))[0]

def load_img(cfg_dir, blk, is_flood = False):
    if is_flood:
        subdir = 'flood'
        fname = f'block{blk}.raw'
    else:
        subdir = 'lut'
        fname = f'block{blk}.lut'

    fname = os.path.join(cfg_dir, subdir, fname)
    return np.fromfile(fname, np.intc).reshape(512,512)

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

def create_cfg_vals(data, lut, blk, cfg):
    blk_vals = cfg[blk] = {}
    xtal_vals = blk_vals['crystal'] = {}

    es = data[:,0]
    rng = np.quantile(es, [0.01, 0.99])
    nbins = int(round((rng[1] - rng[0]) / 10))
    n, bins = np.histogram(es, bins = nbins, range = rng)
    peak, fwhm, *_ = crystal.fit_photopeak(n = n, bins = bins[:-1])

    blk_vals['photopeak'] = peak
    blk_vals['FWHM'] = fwhm 

    pks = crystal.calculate_lut_statistics(lut, data)
    for (lut,_), row in pks.iterrows():
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

def create_scaled_calibration(coincidence_file, cfg_dir,
                              status_queue, data_queue, terminate):

    data, ev_per_period, ev_rate, *_ = open_coincidence_file(coincidence_file)
    default_cfg = os.path.join(cfg_dir, 'default')
    status_queue.put((0, (None,None)))

    for i, rt in enumerate(ev_rate):
        new_cfg = make_cfg_subdir(cfg_dir, round(rt))

        nev = min([ev_per_period, max_events])
        start = ev_per_period * i
        end = start + nev

        data_subset = data[start:end]
        blocks = data_subset[:,0]
        blka, blkb = blocks >> 8, blocks & 0xFF
        unique_blocks = np.unique(np.concatenate([blka, blkb])).tolist()

        with open(os.path.join(default_cfg, 'config.json')) as f:
            cfg = json.load(f)

        cfg_new_vals = {}

        for ub in unique_blocks:
            status_queue.put((
                (i * 64 + ub) / (len(ev_rate) * 64) * 100,
                (i, ub)))

            if terminate.is_set():
                data_queue.put(None)
                return

            arr = load_block_coincidence_data(data_subset, blka, blkb, ub)
            idx = apply_energy_threshold(arr, cfg, ub)

            # Create the flood histogram

            x, y = arr[idx,2], arr[idx,3]
            fld, *_ = np.histogram2d(y, x, bins = 512,
                    range = [[0,511],[0,511]])

            fld.astype(np.intc).tofile(
                    os.path.join(new_cfg, 'flood', f'block{ub}.raw'))

            fld = flood_preprocess(fld)

            # Load the reference flood and perform the registration

            ref_fld = load_img(default_cfg, ub, is_flood = True)
            ref_fld = flood_preprocess(ref_fld)

            xf, yf = register(ref_fld, fld, 4, 500, 32)

            ref_lut = load_img(default_cfg, ub, is_flood = False)
            deformed_lut = cv.remap(ref_lut, xf, yf, cv.INTER_NEAREST,
                    borderMode = cv.BORDER_CONSTANT, borderValue = int(ref_lut.max()))

            deformed_lut.astype(np.intc).tofile(
                    os.path.join(new_cfg, 'lut', f'block{ub}.lut'))

            # Calculate energy and DOI bins for each crystal
            create_cfg_vals(arr, deformed_lut, ub, cfg_new_vals)

        # save the config file
        with open(os.path.join(new_cfg, 'config.json'), 'w') as f:
            json.dump(cfg_new_vals, f)

    data_queue.put(None)
