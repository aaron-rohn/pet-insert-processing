import os, petmr, json, pyelastix, crystal
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from data_loader import coincidence_cols, n_doi_bins, max_events

def create_scaled_calibration(coincidence_file, cfg_dir):
    sz = os.path.getsize(coincidence_file)
    nevents = int((sz/2) // coincidence_cols)
    data = np.memmap(coincidence_file, np.uint16,
                     shape = (nevents, coincidence_cols))

    nperiods = 50
    ev_per_period = int(nevents / nperiods)
    times = data[::ev_per_period,10].astype(float)

    rollover = np.diff(times)
    rollover = np.where(rollover < 0)[0]
    for i in rollover:
        times[i+1:] += 2**16

    ev_rate = ev_per_period / np.diff(times)

    current_cfg = os.path.join(cfg_dir, 'default')
    pars = pyelastix.get_default_params()
    pars.NumberOfResolution = 2
    pars.MaximumNumberOfIterations = 200

    for i, rt in reversed(list(enumerate(ev_rate))):
        # Create the new config dirs for the current event rate
        new_cfg = os.path.join(cfg_dir, f'{round(rt)}')
        new_fld = os.path.join(new_cfg, 'flood')
        new_lut = os.path.join(new_cfg, 'lut')
        for d in [new_cfg, new_fld, new_lut]:
            try:
                os.mkdir(d)
            except FileExistsError: pass

        nev = min([ev_per_period, max_events])
        start = ev_per_period * i
        end = start + nev

        print(f'Period {i}')

        data_subset = data[start:end]
        blocks = data_subset[:,0]
        blka, blkb = blocks >> 8, blocks & 0xFF
        unique_blocks = np.unique(np.concatenate([blka, blkb])).tolist()

        with open(os.path.join(current_cfg, 'config.json')) as f:
            cfg = json.load(f)

        cfg_new_vals = {}

        for ub in unique_blocks:
            print(f'Block {ub}')

            # Load the data

            idxa = np.where(blka == ub)[0]
            idxb = np.where(blkb == ub)[0]

            rowa = data_subset[idxa,:]
            rowb = data_subset[idxb,:]

            arr = np.zeros((len(idxa) + len(idxb), 4), np.uint16)

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

            # Energy threshold the data

            blk_ppeak = cfg[str(ub)]['photopeak']
            blk_res = cfg[str(ub)]['FWHM']
            eth = blk_ppeak - blk_res, blk_ppeak + blk_res

            es = arr[:,0]
            idx = np.where((eth[0] < es) & (es < eth[1]))[0]

            # Create the flood histogram

            x, y = arr[idx,2], arr[idx,3]
            fld, *_ = np.histogram2d(y, x, bins = 512,
                    range = [[0,511],[0,511]])
            fld_max = fld.max()
            fld = fld.astype(float) / fld_max

            # Load the reference flood and perform the registration

            ref_fld_fname = os.path.join(current_cfg, 'flood', f'block{ub}.raw')
            ref_fld = np.fromfile(ref_fld_fname, np.intc).reshape(512,512)
            ref_fld = ref_fld.astype(float) / ref_fld.max()

            deformed_flood, field = pyelastix.register(ref_fld, fld, pars, verbose = 0)
            deformed_flood = (deformed_flood * fld_max).astype(np.intc)
            deformed_flood.tofile(os.path.join(new_fld, f'block{ub}.raw'))

            xfield, yfield = field
            x = np.tile(np.arange(512, dtype = np.float32), 512).reshape(512,512)
            y = x.T + yfield
            x = x   + xfield
            field = (x, y)

            # Apply the warp to the reference LUT and save it

            ref_lut_fname = os.path.join(current_cfg, 'lut', f'block{ub}.lut')
            ref_lut = np.fromfile(ref_lut_fname, np.intc).reshape(512,512)

            deformed_lut = cv.remap(ref_lut, field[0], field[1], cv.INTER_NEAREST,
                    borderMode = cv.BORDER_CONSTANT, borderValue = int(ref_lut.max()))

            deformed_lut.tofile(os.path.join(new_lut, f'block{ub}.lut'))

            # Calculate energy and DOI bins for each crystal

            cfg_new_vals[ub] = {}
            blk_vals = cfg_new_vals[ub]
            blk_vals['crystal'] = {}
            xtal_vals = blk_vals['crystal']

            rng = np.quantile(es, [0.01, 0.99])
            nbins = int(round((rng[1] - rng[0]) / 10))
            n,bins = np.histogram(es, bins = nbins, range = rng)
            peak, fwhm, *_ = crystal.fit_photopeak(n = n, bins = bins[:-1])

            blk_vals['photopeak'] = peak
            blk_vals['FWHM'] = fwhm 

            pks = crystal.calculate_lut_statistics(deformed_lut, arr)
            for (lut,_), row in pks.iterrows():
                this_xtal = {}
                xtal_vals[lut] = this_xtal
                this_xtal['energy'] = {'photopeak': row['peak'], 'FWHM': row['FWHM']}
                this_xtal['DOI'] = row[['5mm','10mm','15mm']].tolist()

        # save the config file
        new_cfg_file = os.path.join(new_cfg, 'config.json')
        with open(new_cfg_file, 'w') as f:
            json.dump(cfg_new_vals, f)

        current_cfg = new_cfg
