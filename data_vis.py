import pandas as pd
import tkinter as tk
from ui_elements import FileSelector, ScrolledListbox, Plots

class App():
    def collect_data(self, d):
        if isinstance(d, tuple):
            self.original, self.d = d
        else:
            self.original = None
            self.d = d

        self.d = self.d.groupby('block')
        self.block.set(list(self.d.groups.keys()))

    def return_data(self, original = False): return self.original if original else self.d
    def return_block(self): return self.block.get()

    def __init__(self, root):
        self.file = FileSelector(root, self.return_data, self.collect_data)
        self.block = ScrolledListbox(root)
        self.file.pack()
        self.block.pack()
        self.plots = Plots(root, self.return_data, self.return_block)
        self.block.bind(self.plots.plots_update)

root = tk.Tk()
app = App(root)
root.mainloop()
