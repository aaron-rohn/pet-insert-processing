import os, json
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter.ttk import Separator
from multiprocessing import Pool

from flood import nearest_peak
from figures import ThresholdHist, FloodHist
from data_loader import DataLoaderPopup, coincidence_filetypes
from entry_fields import NumericEntry

pd.options.mode.chained_assignment = None

class WrappingLabel(tk.Label):
    def reconfig(self, *args, **kwds):
        width = self.winfo_width()
        self.config(wraplength = width)

    def __init__(self, master = None, **kwargs):
        super().__init__(master, **kwargs)
        self.bind('<Configure>', self.reconfig)

class FileSelector:
    def __init__(self, root, update_data_cb):
        self.root = root
        self.sort_coin = tk.IntVar()

        """ callbacks to get the reference data from the caller,
        or to set the reference data held by the caller """
        self.update_data_cb = update_data_cb 

        self.frame = tk.Frame(self.root)

        self.load_button = tk.Button(self.frame, text = "Select files", command = self.load)
        self.coincidences = tk.Checkbutton(self.frame, text = "Sort Coincidences", variable = self.sort_coin)

    def pack(self, **kwds):
        self.frame.pack(**kwds)

        self.load_button.pack(side = tk.TOP, padx = 5, pady = 10)
        self.coincidences.pack(side = tk.BOTTOM, padx = 5, pady = 10)

    def loading_error(self, err):
        tk.messagebox.showerror(message = f'{err}')

    def update_data_cb_wrapper(self, d):
        """ Wrap the process of updating the data set in the app 
        to catch any potential errors that happened when retreiving
        the data
        """
        if isinstance(d, Exception):
            self.loading_error(d)
        else:
            self.update_data_cb(d)

    def load(self):
        """ Start the data loading process, and show a popup if
        the input is invalid
        """
        try:
            DataLoaderPopup(self.root, self, self.update_data_cb_wrapper)
        except (ValueError, RuntimeError) as err:
            self.loading_error(err)

class ScrolledListbox:
    def __init__(self, root, title = None):
        self.root = root
        self.frame = tk.Frame(self.root)
        self.active_var = tk.Variable()
        self.title = tk.Label(self.frame, text = title) if title else None
        self.active = tk.Listbox(self.frame, listvariable = self.active_var, exportselection = False)
        self.scroll = tk.Scrollbar(self.frame, orient = tk.VERTICAL, command = self.active.yview)
        self.active.config(yscrollcommand = self.scroll.set)

    def pack(self, **kwds):
        self.frame.pack(**kwds)

        if self.title is not None:
            self.title.pack(side = tk.TOP, pady = 5)
        self.active.pack(fill = tk.X, expand = True, side = tk.LEFT)
        self.scroll.pack(fill = tk.Y, side = tk.RIGHT)

    def get_active(self, all_blocks = False):
        all_items = self.active_var.get()
        if all_blocks: return all_items

        sel = self.active.curselection()
        return all_items[sel[0]] if len(sel) == 1 else None

    def set_active(self, position = None):
        self.active.select_clear(0, tk.END)
        if position is not None:
            self.active.selection_set(position)
            self.active.activate(position)

    def set(self, blks):
        return self.active_var.set(blks)

    def bind(self, new_block_cb):
        self.active.bind('<<ListboxSelect>>', new_block_cb)

class Plots:
    def __init__(self, root, data, get_block, set_block):
        self.d    = None
        self.data = data
        self.get_block = get_block
        self.set_block = set_block

        self.save_xtal_photopeak = tk.IntVar()
        self.transform_flood = tk.IntVar()

        self.button_frame = tk.Frame(root)

        self.register_button = tk.Button(self.button_frame,
                text = "Register peaks",
                command = lambda: self.flood.register())

        self.transform_flood_cb = tk.Checkbutton(self.button_frame,
                text = "Transform flood",
                variable = self.transform_flood,
                command = self.flood_cb)

        self.select_dir_button = tk.Button(self.button_frame,
                text = "Select Directory",
                command = lambda: self.check_output_dir(True))

        self.store_flood_button = tk.Button(self.button_frame,
                text = "Store Flood",
                command = self.store_flood_cb)

        self.store_lut_button = tk.Button(self.button_frame,
                text = "Store LUT",
                command = self.store_lut_cb)

        self.save_xtal_cb = tk.Checkbutton(self.button_frame,
                text = "Store crystal photopeak",
                variable = self.save_xtal_photopeak)

        self.button_frame.pack(pady = 10);
        self.register_button.pack(side = tk.LEFT, padx = 5)
        self.transform_flood_cb.pack(side = tk.LEFT, padx = 5)

        Separator(self.button_frame, orient = tk.VERTICAL).pack(
                side = tk.LEFT, fill = tk.Y, padx = 20, pady = 5)

        self.select_dir_button.pack(side = tk.LEFT, padx = 5)
        self.store_flood_button.pack(side = tk.LEFT, padx = 5)
        self.store_lut_button.pack(side = tk.LEFT, padx = 5)
        self.save_xtal_cb.pack(side = tk.LEFT, padx = 5)

        self.frame = tk.Frame(root)
        self.frame.pack(padx = 5, pady = 5)
        self.frame.columnconfigure(0, weight = 1)
        self.frame.columnconfigure(1, weight = 1)
        self.frame.columnconfigure(2, weight = 1)

        self.flood  = FloodHist(self.frame, column = 0, row = 0)
        self.energy = ThresholdHist(self.frame, is_energy = True, column = 1, row = 0)
        self.doi    = ThresholdHist(self.frame, is_energy = False, column = 2, row = 0)

        # when one plot is updated, update other plots accordingly
        # callbacks are triggered after a threshold drag is finished
        self.energy.callback.append(self.flood_cb)
        self.energy.callback.append(self.doi_cb)
        self.doi.callback.append(self.flood_cb)
        self.doi.callback.append(self.energy_cb)

        self.output_dir = None

    def plots_update(self, *args):
        """ Update all plots when new data is available """
        blk = self.get_block()
        all_data = self.data()

        if all_data is not None:
            self.d = all_data.get_group(blk)
            self.energy.update(self.d['es'], retain = False)
            self.doi_cb(retain = False)
            self.flood_cb()
        else:
            self.d = None

    def flood_cb(self):
        """ Update the flood according to energy and DOI thresholds """
        eth = self.energy.thresholds()
        dth = self.doi.thresholds()

        if self.d is not None:
            data_subset = self.d.query('({} < es < {}) & ({} < doi < {})'.format(*eth, *dth))
            self.flood.update(data_subset, smoothing = 1.5, warp = self.transform_flood.get())

    def doi_cb(self, retain = True):
        """ Update the DOI according to the energy thresholds """
        eth = self.energy.thresholds()
        if self.d is not None:
            data_subset = self.d.query('{} < es < {}'.format(*eth))
            self.doi.update(data_subset['doi'], retain)

    def energy_cb(self, retain = True):
        """ Update the energy according to the DOI thresholds """
        dth = self.doi.thresholds()
        if self.d is not None:
            data_subset = self.d.query('{} < doi < {}'.format(*dth))
            self.energy.update(data_subset['es'], retain)

    def check_output_dir(self, reset = False):
        if reset: self.output_dir = None
        self.output_dir = self.output_dir or tk.filedialog.askdirectory(
                title = "Configuration data directory",
                initialdir = '/')

        return self.output_dir

    def store_flood_cb(self):
        output_dir = self.check_output_dir()
        if not output_dir or self.flood.f is None:
            return

        blk = self.get_block()
        flood_fname = os.path.join(output_dir, f'block{blk}.raw')
        self.flood.img.astype(np.intc).tofile(flood_fname)

    def store_lut_cb(self):
        output_dir = self.check_output_dir()
        if not output_dir or self.flood.f is None:
            return

        blk = self.get_block()

        """ store the LUT for this block to the specified directory """
        lut_fname = os.path.join(output_dir, f'block{blk}.lut')
        lut = nearest_peak((self.flood.img_size,)*2,
                self.flood.pts.reshape(-1,2))
        lut = self.flood.f.warp_lut(lut)
        lut.astype(np.intc).tofile(lut_fname)

        """ update json file with photopeak position for this block """
        config_file = os.path.join(output_dir, 'conig.json')

        try:
            with open(config_file, 'r') as f:
                cfg = json.load(f)
        except FileNotFoundError: cfg = {}

        if blk not in cfg: cfg[blk] = {}
        cfg[blk]['block'] = round(self.energy.peak)

        """ if requested, add the photopeak position for each LUT value """
        if self.save_xtal_photopeak.get():
            # create df grouped by LUT number

            lut_df = pd.DataFrame({
                'x'  : np.tile(np.arange(lut.shape[1]), lut.shape[0]),
                'y'  : np.repeat(np.arange(lut.shape[0]), lut.shape[1]),
                'lut': lut.flat
            })

            self.d.loc[:,['x','y']] *= 511.0
            self.d = self.d.astype({'x': int, 'y': int})
            self.d = self.d.merge(lut_df, on = ['x', 'y'])
            self.d = self.d.groupby(['lut'])

            # find the photopeak for each crystal
            def get_photopeak(grp):
                n,bins = np.histogram(grp['es'].values, bins = 100)
                return round(bins[np.argmax(bins[:-1] * n**2)])

            pks = self.d.apply(get_photopeak)

            # record the crystal photopeak
            for lut,pk in pks.iteritems():
                cfg[blk][str(lut)] = pk

        # save the config file
        with open(config_file, 'w') as f:
            json.dump(cfg, f)

        """ increment the active block and update the UI """

        all_blks = self.get_block(all_blocks = True)
        try: 
            self.set_block(all_blks.index(blk) + 1)
            self.plots_update()
        except KeyError: pass
