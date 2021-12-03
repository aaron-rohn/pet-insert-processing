import tkinter as tk
from tkinter import ttk
from ui_elements import FileSelector, ScrolledListbox, Plots
from sinogram_elements import SinogramDisplay

class App:
    def collect_data(self, d):
        self.d = d.groupby('block')
        self.block.set(list(self.d.groups.keys()))

    """ Allow member elements to query the current data or selected block """
    def return_data(self): return self.d
    def return_block(self, *args, **kwds): return self.block.get_active(*args, **kwds)
    def set_block(self, *args, **kwds): return self.block.set_active(*args, **kwds)

    def __init__(self, root):
        self.d = None

        self.base = ttk.Notebook(root)
        self.listmode_frame = tk.Frame(self.base)
        self.sinogram_frame = tk.Frame(self.base)
        self.base.add(self.listmode_frame, text = "Listmode Processing")
        self.base.add(self.sinogram_frame, text = "Sinogram Processing")
        self.base.pack()

        self.file = FileSelector(self.listmode_frame, self.collect_data)
        self.block = ScrolledListbox(self.listmode_frame)
        self.file.pack()
        self.block.pack()
        self.plots = Plots(self.listmode_frame, self.return_data, self.return_block, self.set_block)
        self.block.bind(self.plots.plots_update)

        self.sino = SinogramDisplay(self.sinogram_frame)
        self.sino.pack()

root = tk.Tk()
app = App(root)
root.mainloop()
