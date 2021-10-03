import tkinter as tk
from ui_elements import FileSelector, ScrolledListbox, Plots

class App():
    def collect_data(self, d):
        self.d = d
        self.block.set(list(d.groups.keys()))

    def get_data(self): return self.d
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
