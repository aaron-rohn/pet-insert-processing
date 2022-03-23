import tkinter as tk
from tkinter import ttk
from ui_elements import FileSelector, ScrolledListbox, Plots
from sinogram_elements import SinogramDisplay

import pandas as pd
import numpy as np

class App:
    def collect_data(self, d):
        self.d = d
        self.block.set(list(self.d.keys()))

        #self.d = self.d.groupby('block')
        #self.block.set(list(self.d.groups.keys()))

    """ Allow member elements to query the current data or selected block """
    def return_data(self, block): return self.d[block]
    def return_block(self, *args, **kwds): return self.block.get_active(*args, **kwds)
    def set_block(self, *args, **kwds): return self.block.set_active(*args, **kwds)

    def __init__(self, root):
        self.d = None
        self.d_idx = None

        self.base = ttk.Notebook(root)
        listmode_frame = tk.Frame(self.base)
        sinogram_frame = tk.Frame(self.base)
        self.base.add(listmode_frame, text = "Listmode Processing")
        self.base.add(sinogram_frame, text = "Sinogram Processing")
        self.base.pack(fill = tk.BOTH, expand = True)

        lm_top_frame = tk.Frame(listmode_frame)
        lm_top_frame.pack(fill = tk.X, expand = True)

        self.file = FileSelector(lm_top_frame, self.collect_data)
        self.block = ScrolledListbox(lm_top_frame, "Active Blocks")

        self.file.pack(side = tk.LEFT, padx = 30, pady = 20)
        self.block.pack(side = tk.LEFT, fill = tk.BOTH, expand = True, padx = 30, pady = 20)

        self.plots = Plots(listmode_frame, self.return_data, self.return_block, self.set_block)
        self.block.bind(self.plots.plots_update)

        self.sino = SinogramDisplay(sinogram_frame)
        self.sino.pack()

root = tk.Tk()
app = App(root)
root.mainloop()
