import pandas as pd
import tkinter as tk
from petmr import read
from ui_elements import FileSelector, ScrolledListbox, Plots

class App():
    def open_file(self):
        """ Load listmode data from one or more files and calculate necessary columns """
        fname = tk.filedialog.askopenfilenames()
        self.file.set(fname or "No file selected")

        c = ['block', 'time', 'A1', 'B1', 'C1', 'D1', 'A2', 'B2', 'C2', 'D2']
        self.d = [pd.DataFrame(dict(zip(c, read(f, False)))) for f in fname]
        self.d = (pd.concat(self.d, ignore_index = True)
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

        blocks = self.d.groups.keys()
        self.block.set(list(blocks))

    def get_data(self): return self.d
    def get_block(self): return self.block.get()

    def __init__(self, root):
        self.root = root
        self.file = FileSelector(self.root)
        self.block = ScrolledListbox(self.root)
        self.file.pack()
        self.block.pack()

        self.plots = Plots(self.root, self.get_data, self.get_block)

        self.file.bind(self.open_file)
        self.block.bind(self.plots.plots_update)

root = tk.Tk()
app = App(root)
root.mainloop()
