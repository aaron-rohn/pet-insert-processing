import pandas as pd
import numpy as np
import tkinter as tk

import petmr
from figures import ThresholdHist, Flood

class App():
    def open_file(self):
        try:
            fname = tk.filedialog.askopenfilenames()
            if not fname: raise Exception('No file selected')
            self.file_indicator.config(text = fname)
        except Exception as e:
            self.file_indicator.config(text = str(e))
            return

        cols = ['block', 'time', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        d = []

        for f in fname:
            vals = petmr.read(f, False)
            vals = pd.concat([pd.Series(v) for v in vals], axis = 1)
            d.append(vals)

        d = pd.concat(d, ignore_index = True)
        d.columns = cols

        d = d.assign(e1 = d.loc[:,'A':'D'].sum(axis=1),
                     e2 = d.loc[:,'E':'H'].sum(axis=1))

        d = d.query('e1 > 0 & e2 > 0')

        d = d.assign(e_sum = d['e1'] + d['e2'],
                     x1 = (d['A'] + d['B']) / d['e1'],
                     y1 = (d['B'] + d['C']) / d['e1'],
                     x2 = (d['E'] + d['F']) / d['e2'],
                     y2 = (d['F'] + d['G']) / d['e2'])

        d = d.assign(doi = d['e1'] / d['e_sum'])

        blocks = np.sort(np.unique(d['block']))
        self.active_blocks.set([str(b) for b in blocks])
        self.d = d

    def plots_update(self, ev):
        blk = self.active_blocks_ind.get(tk.ANCHOR)
        self.d_blk = self.d.query('block == {}'.format(blk))
        self.doi.update(self.d_blk['doi'])
        self.energy.update(self.d_blk['e_sum'])
        self.flood.update(self.d_blk)

    def __init__(self):
        self.root = tk.Tk()

        self.file_frame = tk.Frame(self.root)
        self.file_button = tk.Button(self.file_frame, text = "Select file", command = self.open_file)
        self.file_indicator = tk.Label(self.file_frame, bg = 'white', text = '', anchor = 'w', relief = tk.SUNKEN, borderwidth = 1, height = 2) 
        self.file_frame.pack(fill = tk.X, expand = True, padx = 10, pady = 10)
        self.file_button.pack(side = tk.LEFT, padx = 10, pady = 10)
        self.file_indicator.pack(side = tk.LEFT, fill = tk.X, expand = True, padx = 10, pady = 10)

        self.blk_ind_frame = tk.Frame(self.root)
        self.active_blocks = tk.StringVar()
        self.active_blocks_ind = tk.Listbox(self.blk_ind_frame, listvariable = self.active_blocks)
        self.active_blocks_ind.bind('<<ListboxSelect>>', self.plots_update)
        self.active_blocks_scr = tk.Scrollbar(self.blk_ind_frame, orient = tk.VERTICAL, command = self.active_blocks_ind.yview)
        self.active_blocks_ind.config(yscrollcommand = self.active_blocks_scr.set)
        self.blk_ind_frame.pack(fill = tk.X, expand = True, padx = 10, pady = 10)
        self.active_blocks_ind.pack(fill = tk.X, expand = True, side = tk.LEFT)
        self.active_blocks_scr.pack(fill = tk.Y, side = tk.RIGHT)

        self.plt_frame = tk.Frame(self.root)
        self.plt_frame.pack(side = tk.TOP, fill = tk.BOTH, expand = True, padx = 10, pady = 10)
        self.plt_frame.columnconfigure(0, weight = 1)
        self.plt_frame.columnconfigure(1, weight = 1)
        self.plt_frame.columnconfigure(2, weight = 1)
        self.flood = Flood(self.plt_frame, column = 0, row = 0, sticky = 'EW')
        self.energy = ThresholdHist(self.plt_frame, column = 1, row = 0, sticky = 'EW')
        self.doi = ThresholdHist(self.plt_frame, column = 2, row = 0, sticky = 'EW')

        flood_cb = lambda: self.flood.update(self.d_blk.query('({} < e_sum < {}) & ({} < doi < {})'.format(*self.energy.thresholds(), *self.doi.thresholds())))
        energy_cb = lambda: self.doi.update(self.d_blk.query('{} < e_sum < {}'.format(*self.energy.thresholds()))['doi'], retain = True)
        doi_cb = lambda: self.energy.update(self.d_blk.query('{} < doi < {}'.format(*self.doi.thresholds()))['e_sum'], retain = True)
        self.energy.callback.append(flood_cb)
        self.doi.callback.append(flood_cb)
        self.energy.callback.append(energy_cb)
        self.doi.callback.append(doi_cb)

app = App()
app.root.mainloop()
