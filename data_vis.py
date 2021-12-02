import pandas as pd
import tkinter as tk
from tkinter import ttk
from ui_elements import FileSelector, ScrolledListbox, Plots

class App():
    def collect_data(self, d):
        """ Sorted coincidence data is provides as a tuple with
        the original unmodified data, which is itself a 2-tuple.
        Each item is a list of 6 numpy arrays with values for either
        half of the coincidence:
         ([blka, ea1, ea2, xa, ya, tdiff], [blkb, eb1, eb2, xb, yb, tdiff])
        The second item of the tuple is a dataframe with the necessary
        columns for plotting. A and B coincidence df's are concatenated.

        Other data, either singles or coincidences loaded from disk,
        will be a dataframe with the necessary columns for plotting
        """
        if isinstance(d, tuple):
            self.original, self.d = d
            self.d = self.d.groupby('block')
        else:
            self.original = None
            self.d = d.groupby('block')

        self.block.set(list(self.d.groups.keys()))

    """ Allow member elements to query the current data or selected block """
    def return_data(self, original = False): return self.original if original else self.d
    def return_block(self, *args, **kwds): return self.block.get_active(*args, **kwds)
    def set_block(self, *args, **kwds): return self.block.set_active(*args, **kwds)

    def __init__(self, root):
        self.original = None
        self.d = None

        self.base = ttk.Notebook(root)
        self.listmode_frame = tk.Frame(self.base)
        self.sinogram_frame = tk.Frame(self.base)
        self.base.add(self.listmode_frame, text = "Listmode Processing")
        self.base.add(self.sinogram_frame, text = "Sinogram Processing")
        self.base.pack()

        self.file = FileSelector(self.listmode_frame, self.return_data, self.collect_data)
        self.block = ScrolledListbox(self.listmode_frame)
        self.file.pack()
        self.block.pack()
        self.plots = Plots(self.listmode_frame, self.return_data, self.return_block, self.set_block)
        self.block.bind(self.plots.plots_update)

root = tk.Tk()
app = App(root)
root.mainloop()
