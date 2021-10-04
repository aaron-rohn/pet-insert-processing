import pandas as pd
import tkinter as tk
from ui_elements import FileSelector, ScrolledListbox, Plots

class App():
    def collect_data(self, d):
        if isinstance(d, list):
            self.d_original = d
            self.d = pd.concat(d, ignore_index = True)
        else:
            self.d_original = None
            self.d = d

        self.d = self.d.groupby('block')
        self.block.set(list(self.d.groups.keys()))

    def get_data(self, original = False): return self.d_original if original else self.d
    def get_block(self): return self.block.get()

    def __init__(self, root):
        self.file = FileSelector(root, self.collect_data)
        self.block = ScrolledListbox(root)
        self.file.pack()
        self.block.pack()
        self.plots = Plots(root, self.get_data, self.get_block)
        self.block.bind(self.plots.plots_update)

root = tk.Tk()
app = App(root)
root.mainloop()
