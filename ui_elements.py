import pandas as pd
import tkinter as tk
from figures import ThresholdHist, FloodHist
from petmr import singles

class FileSelector():
    def __init__(self, root):
        self.root = root
        self.frame = tk.Frame(self.root)
        self.button = tk.Button(self.frame, text = "Select file")
        self.label = tk.Label(self.frame, bg = 'white', text = '', anchor = 'w', relief = tk.SUNKEN, borderwidth = 1, height = 2) 

        self.sort_coin = tk.IntVar()
        self.coincidences = tk.Checkbutton(self.frame, text = "Sort Coincidences", variable = self.sort_coin)

    def pack(self):
        self.frame.pack(fill = tk.X, expand = True, padx = 10, pady = 10)
        self.button.pack(side = tk.LEFT, padx = 10, pady = 10)
        self.label.pack(side = tk.LEFT, fill = tk.X, expand = True, padx = 10, pady = 10)
        self.coincidences.pack(side = tk.LEFT, padx = 10, pady = 10)

    def load(self):
        """ Load listmode data from one or more files and calculate necessary columns """
        fname = tk.filedialog.askopenfilenames()
        self.label.config(text = fname or "No file selected")

        c = ['block', 'time', 'A1', 'B1', 'C1', 'D1', 'A2', 'B2', 'C2', 'D2']
        d = [pd.DataFrame(dict(zip(c, singles(f)))) for f in fname]
        d = (pd.concat(d, ignore_index = True)
               .assign(e1  = lambda df: df.loc[:,'A1':'D1'].sum(axis=1),
                       e2  = lambda df: df.loc[:,'A2':'D2'].sum(axis=1),
                       es  = lambda df: df['e1'] + df['e2'],
                       doi = lambda df: df['e1'] / df['es'],
                       x1  = lambda df: (df['A1'] + df['B1']) / df['e1'],
                       y1  = lambda df: (df['A1'] + df['D1']) / df['e1'],
                       x2  = lambda df: (df['A2'] + df['B2']) / df['e2'],
                       y2  = lambda df: (df['A2'] + df['D2']) / df['e2'],
                       y   = lambda df: (df['y1'] + df['y2']) / 2.0)
               .groupby('block'))

        return d

    def set(self, text): self.label.config(text = text)
    def bind(self, cb): self.button.config(command = cb)

class ScrolledListbox():
    def __init__(self, root):
        self.root = root
        self.frame = tk.Frame(self.root)
        self.active_var = tk.StringVar()
        self.active = tk.Listbox(self.frame, listvariable = self.active_var)
        self.scroll = tk.Scrollbar(self.frame, orient = tk.VERTICAL, command = self.active.yview)
        self.active.config(yscrollcommand = self.scroll.set)

    def pack(self):
        self.frame.pack(fill = tk.X, expand = True, padx = 10, pady = 10)
        self.active.pack(fill = tk.X, expand = True, side = tk.LEFT)
        self.scroll.pack(fill = tk.Y, side = tk.RIGHT)

    def get(self): return self.active.get(tk.ANCHOR)
    def set(self, blks): return self.active_var.set(blks)
    def bind(self, cb): self.active.bind('<<ListboxSelect>>', cb)

class Plots():
    def __init__(self, root, data, block):
        self.data = data    # callback to get the current data frame
        self.block = block  # callback to get the selected block

        self.frame = tk.Frame(root)
        self.frame.pack(side = tk.TOP, fill = tk.BOTH, expand = True, padx = 10, pady = 10)
        self.frame.columnconfigure(0, weight = 1)
        self.frame.columnconfigure(1, weight = 1)
        self.frame.columnconfigure(2, weight = 1)

        self.flood  = FloodHist(self.frame, column = 0, row = 0, sticky = 'EW')
        self.energy = ThresholdHist(self.frame, is_energy = True, column = 1, row = 0, sticky = 'EW')
        self.doi    = ThresholdHist(self.frame, is_energy = False, column = 2, row = 0, sticky = 'EW')

        # when one plot is updated, update other plots accordingly
        # callbacks are triggered after a threshold drag is finished
        self.energy.callback.append(self.flood_cb)
        self.energy.callback.append(self.doi_cb)
        self.doi.callback.append(self.flood_cb)
        self.doi.callback.append(self.energy_cb)

    def plots_update(self, ev):
        """ Update all plots when new data is available """
        self.d = self.data().get_group(self.block())

        self.energy.update(self.d['es'], retain = False)
        self.doi_cb(retain = False)
        self.flood_cb()

    def flood_cb(self):
        """ Update the flood according to energy and DOI thresholds """
        eth = self.energy.thresholds()
        dth = self.doi.thresholds()
        data_subset = self.d.query('({} < es < {}) & ({} < doi < {})'.format(*eth, *dth))
        self.flood.update(data_subset)

    def doi_cb(self, retain = True):
        """ Update the DOI according to the energy thresholds """
        eth = self.energy.thresholds()
        data_subset = self.d.query('{} < es < {}'.format(*eth))
        self.doi.update(data_subset['doi'], retain)

    def energy_cb(self, retain = True):
        """ Update the energy according to the DOI thresholds """
        dth = self.doi.thresholds()
        data_subset = self.d.query('{} < doi < {}'.format(*dth))
        self.energy.update(data_subset['es'], retain)
