import os, json
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter.ttk import Separator
import matplotlib.pyplot as plt

from flood import nearest_peak
from figures import ThresholdHist, FloodHist

from data_loader import (
        SinglesLoader,
        CoincidenceLoader,
        CoincidenceSorter,
        validate_singles)

pd.options.mode.chained_assignment = None

class WrappingLabel(tk.Label):
    def reconfig(self, *args, **kwds):
        width = self.winfo_width()
        self.config(wraplength = width)

    def __init__(self, master = None, **kwargs):
        super().__init__(master, **kwargs)
        self.bind('<Configure>', self.reconfig)

class FileSelector(tk.Frame):
    def __init__(self, root, callback):
        super().__init__(root)
        self.callback = callback 

        self.load_sgls_button = tk.Button(self, text = "Load Singles",
                command = lambda: self.loader_wrapper(SinglesLoader))

        self.load_coin_button = tk.Button(self, text = "Load Coincidences",
                command = lambda: self.loader_wrapper(CoincidenceLoader))

        self.sort_coin_button = tk.Button(self, text = "Sort Coincidences",
                command = lambda: self.loader_wrapper(CoincidenceSorter))

        self.validate_button = tk.Button(self, text = "Validate files", command = validate_singles)

    def pack(self, **kwds):
        super().pack(**kwds)
        args = {'side': tk.TOP, 'padx': 5, 'pady': 5, 'fill': tk.X, 'expand': True}
        self.load_sgls_button.pack(**args)
        self.load_coin_button.pack(**args)
        self.sort_coin_button.pack(**args)
        self.validate_button.pack(**args)

    def cfg_buttons(self, state):
        self.load_sgls_button.config(state = state) 
        self.load_coin_button.config(state = state) 
        self.sort_coin_button.config(state = state) 
        self.validate_button.config(state = state)

    def callback_wrapper(self, response):
        self.cfg_buttons(tk.NORMAL)
        if isinstance(response, Exception):
            tk.messagebox.showerror(message = f'{response}')
        else:
            self.callback(response)

    def loader_wrapper(self, loader):
        self.cfg_buttons(tk.DISABLED)
        try:
            self.loader = loader(self.callback_wrapper)
        except Exception as e:
            self.callback_wrapper(e)

class ScrolledListbox(tk.Frame):
    def __init__(self, root, title = None):
        super().__init__(root)
        self.active_var = tk.Variable()

        self.title_text = title
        self.title = tk.Label(self, text = self.title_text) if self.title_text else None
        self.active = tk.Listbox(self, listvariable = self.active_var, exportselection = False)
        self.scroll = tk.Scrollbar(self, orient = tk.VERTICAL, command = self.active.yview)
        self.active.config(yscrollcommand = self.scroll.set)

    def pack(self, **kwds):
        super().pack(**kwds)

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
        self.title.config(text = f'{self.title_text} ({len(blks)})')
        return self.active_var.set(blks)

    def bind(self, new_block_cb):
        self.active.bind('<<ListboxSelect>>', new_block_cb)

class Plots:
    def __init__(self, root, data, get_block, set_block):
        self.d = None
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
        self.d = self.data(blk)

        self.energy.update(self.d[:,0], retain = False)
        self.doi_cb(retain = False)
        self.flood_cb()

    def flood_cb(self):
        """ Update the flood according to energy and DOI thresholds """
        eth = self.energy.thresholds()
        dth = self.doi.thresholds()

        es = self.d[:,0]
        doi = self.d[:,1]
        idx = np.where((eth[0] < es) & (es < eth[1]) &
                       (dth[0] < doi) & (doi < dth[1]))[0]

        self.flood.update(self.d[idx,2], self.d[idx,3],
                smoothing = 1.5, warp = self.transform_flood.get())

    def doi_cb(self, retain = True):
        """ Update the DOI according to the energy thresholds """
        eth = self.energy.thresholds()
        es = self.d[:,0]
        idx = np.where((eth[0] < es) & (es < eth[1]))[0]
        self.doi.update(self.d[idx,1], retain)

    def energy_cb(self, retain = True):
        """ Update the energy according to the DOI thresholds """
        dth = self.doi.thresholds()
        doi = self.d[:,1]
        idx = np.where((dth[0] < doi) & (doi < dth[1]))[0]
        self.energy.update(self.d[idx,0], retain)

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
        print(f'Store LUT for block {blk}')

        """ store the LUT for this block to the specified directory """
        lut_fname = os.path.join(output_dir, f'block{blk}.lut')
        lut = nearest_peak((self.flood.img_size,)*2,
                self.flood.pts.reshape(-1,2))
        lut = self.flood.f.warp_lut(lut)
        lut.astype(np.intc).tofile(lut_fname)

        """ update json file with photopeak position for this block """
        config_file = os.path.join(output_dir, 'config.json')

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

            lm_df = pd.DataFrame(self.d, columns = ['es', 'doi', 'x', 'y'])
            lm_df = lm_df.merge(lut_df, on = ['x', 'y'])
            lm_df = lm_df.groupby(['lut'])

            # find the photopeak for each crystal
            def get_photopeak(grp):
                n,bins = np.histogram(grp['es'].values, bins = 100,
                        range = np.quantile(grp['es'].values, [0, 0.95]))
                return round(bins[np.argmax(bins[:-1] * n**2)])

            pks = lm_df.apply(get_photopeak)

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
