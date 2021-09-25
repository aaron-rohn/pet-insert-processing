import pandas as pd
import numpy as np
import tkinter as tk

import petmr
from figures import ThresholdHist, FloodHist

class FileSelector():
    def __init__(self, root, on_click):
        self.root = root
        self.frame = tk.Frame(self.root)

        self.button = tk.Button(self.frame, text = "Select file", command = on_click)
        self.label = tk.Label(self.frame, bg = 'white', text = '', anchor = 'w', relief = tk.SUNKEN, borderwidth = 1, height = 2) 

        self.frame.pack(fill = tk.X, expand = True, padx = 10, pady = 10)
        self.button.pack(side = tk.LEFT, padx = 10, pady = 10)
        self.label.pack(side = tk.LEFT, fill = tk.X, expand = True, padx = 10, pady = 10)

class ScrolledListbox():
    def __init__(self, root, on_click):
        self.root = root
        self.frame = tk.Frame(self.root)

        self.active_var = tk.StringVar()
        self.active = tk.Listbox(self.frame, listvariable = self.active_var)
        self.active.bind('<<ListboxSelect>>', on_click)

        self.scroll = tk.Scrollbar(self.frame, orient = tk.VERTICAL, command = self.active.yview)
        self.active.config(yscrollcommand = self.scroll.set)

        self.frame.pack(fill = tk.X, expand = True, padx = 10, pady = 10)
        self.active.pack(fill = tk.X, expand = True, side = tk.LEFT)
        self.scroll.pack(fill = tk.Y, side = tk.RIGHT)

    def get(self): return self.active.get(tk.ANCHOR)
    def set(self, blks): return self.active_var.set(blks)

class App():
    def open_file(self):
        try:
            fname = tk.filedialog.askopenfilenames()
            if not fname: raise Exception('No file selected')
            self.file.label.config(text = fname)
        except Exception as e:
            self.file.label.config(text = str(e))
            return

        cols = ['block', 'time', 'A1', 'B1', 'C1', 'D1', 'A2', 'B2', 'C2', 'D2']
        d = []

        for f in fname:
            vals = petmr.read(f, False)
            vals = pd.concat([pd.Series(v) for v in vals], axis = 1)
            vals.columns = cols
            d.append(vals)

        d = (pd.concat(d, ignore_index = True)
               .assign(e1  = lambda df: df.loc[:,'A1':'D1'].sum(axis=1),
                       e2  = lambda df: df.loc[:,'A2':'D2'].sum(axis=1),
                       es  = lambda df: df['e1'] + df['e2'],
                       x1  = lambda df: (df['A1'] + df['B1']) / df['e1'],
                       y1  = lambda df: (df['A1'] + df['D1']) / df['e1'],
                       x2  = lambda df: (df['A2'] + df['B2']) / df['e2'],
                       y2  = lambda df: (df['A2'] + df['D2']) / df['e2'],
                       y   = lambda df: (df['y1'] + df['y2']) / 2.0,
                       doi = lambda df: df['e1'] / df['es']))

        blocks = np.sort(np.unique(d['block']))
        self.block.set(list(blocks))
        self.d = d

    def plots_update(self, ev):
        self.d_blk = self.d.query('block == {}'.format(self.block.get()))
        self.energy.update(self.d_blk['es'], retain = False)
        self.doi_cb(retain = False)
        self.flood_cb()

    def flood_cb(self):
        eth = self.energy.thresholds()
        dth = self.doi.thresholds()
        data_subset = self.d_blk.query('({} < es < {}) & ({} < doi < {})'.format(*eth, *dth))
        self.flood.update(data_subset)

    def doi_cb(self, retain = True):
        eth = self.energy.thresholds()
        data_subset = self.d_blk.query('{} < es < {}'.format(*eth))
        self.doi.update(data_subset['doi'], retain)

    def energy_cb(self, retain = True):
        dth = self.doi.thresholds()
        data_subset = self.d_blk.query('{} < doi < {}'.format(*dth))
        self.energy.update(data_subset['es'], retain)

    def __init__(self):
        self.root = tk.Tk()
        self.file = FileSelector(self.root, self.open_file)
        self.block = ScrolledListbox(self.root, self.plots_update)

        self.plt_frame = tk.Frame(self.root)
        self.plt_frame.pack(side = tk.TOP, fill = tk.BOTH, expand = True, padx = 10, pady = 10)
        self.plt_frame.columnconfigure(0, weight = 1)
        self.plt_frame.columnconfigure(1, weight = 1)
        self.plt_frame.columnconfigure(2, weight = 1)
        self.flood = FloodHist(self.plt_frame, column = 0, row = 0, sticky = 'EW')
        self.energy = ThresholdHist(self.plt_frame, is_energy = True, column = 1, row = 0, sticky = 'EW')
        self.doi = ThresholdHist(self.plt_frame, is_energy = False, column = 2, row = 0, sticky = 'EW')

        self.energy.callback.append(self.flood_cb)
        self.doi.callback.append(self.flood_cb)
        self.energy.callback.append(self.doi_cb)
        self.doi.callback.append(self.energy_cb)

app = App()
app.root.mainloop()
