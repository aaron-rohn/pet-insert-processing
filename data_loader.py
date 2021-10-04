import os
import threading
import queue
import pandas as pd
import tkinter as tk
from tkinter.ttk import Progressbar
from petmr import singles, coincidences

singles_filetypes = [("Singles",".SGL")]
coincidence_filetypes = [("Coincidences",".COIN")]
col_names = ['block', 'time', 'A1', 'B1', 'C1', 'D1', 'A2', 'B2', 'C2', 'D2']

class DataLoaderPopup:
    def __init__(self, root, fileselector, callback):
        self.fileselector = fileselector
        self.callback = callback 
        self.terminate = threading.Event()
        self.data_queue = queue.Queue()
        self.stat_queue = queue.Queue()
        self.output_file = None

        self.input_files = list(tk.filedialog.askopenfilenames(
            title = "Load listmode data",
            initialdir = "/",
            filetypes = singles_filetypes + coincidence_filetypes))

        if not self.input_files: raise ValueError("No input file specified")
        _, ext = os.path.splitext(self.input_files[0])
        self.allow_store = False

        if self.fileselector.sort_coin.get():
            if ext != ".SGL": raise ValueError("Input files must be singles " 
                                               "when sorting coincidences")
            self.output_file = tk.filedialog.asksaveasfilename(
                    title = "Output file, or none",
                    initialdir = os.path.expanduser('~'),
                    filetypes = coincidence_filetypes) or None
            self.bg = threading.Thread(target = self.coincidences)
            self.allow_store = True

        else:
            if ext == ".SGL":
                self.bg = threading.Thread(target = self.load_singles)
            elif ext == ".COIN":
                raise ValueError("Not implemented")
            else:
                raise ValueError("Unsupported file type")

        self.popup = tk.Toplevel(root)
        self.popup.title('Progress')
        self.popup.attributes('-type', 'dialog')
        self.popup.protocol("WM_DELETE_WINDOW", self.on_close)
        self.progbar = Progressbar(self.popup, length = 500)
        self.progbar.pack(fill = tk.X, expand = True, padx = 10, pady = 10)
        self.counts_label = tk.Label(self.popup, text = 'Counts: 0')
        self.counts_label.pack(pady = 10)

        self.fileselector.label.config(text = self.input_files)
        self.fileselector.load_button.config(state = tk.DISABLED)
        self.fileselector.store_button.config(state = tk.DISABLED)
        self.bg.start()
        self.check()

    def check(self):
        while not self.stat_queue.empty():
            perc, counts = self.stat_queue.get()
            self.counts_label.config(text = 'Counts: ' + f'{counts:,}')
            self.progbar['value'] = perc

        if self.bg.is_alive():
            self.popup.after(1000, self.check)
        else:
            if not self.data_queue.empty():
                d = self.data_queue.get()
                self.callback(d)

            self.popup.destroy()
            self.fileselector.store_button.config(
                    state = tk.NORMAL if self.allow_store else tk.DISABLED)
            self.fileselector.load_button.config(state = tk.NORMAL)

    def on_close(self):
        self.terminate.set()
        self.bg.join()
        self.check()

    def coincidences(self):
        d = coincidences(self.terminate, self.stat_queue, self.input_files, self.output_file)
        d = [pd.DataFrame(dict(zip(col_names,di))) for di in d]
        d = [self.add_cols(di) for di in d]
        self.data_queue.put(d)

    def load_singles(self):
        d = singles(self.input_files)
        d = pd.DataFrame(dict(zip(col_names, d)))
        d = self.add_cols(d)
        self.data_queue.put(d)

    def add_cols(self, d):
        return d.assign(e1  = lambda df: df.loc[:,'A1':'D1'].sum(axis=1),
                        e2  = lambda df: df.loc[:,'A2':'D2'].sum(axis=1),
                        es  = lambda df: df['e1'] + df['e2'],
                        doi = lambda df: df['e1'] / df['es'],
                        x1  = lambda df: (df['A1'] + df['B1']) / df['e1'],
                        y1  = lambda df: (df['A1'] + df['D1']) / df['e1'],
                        x2  = lambda df: (df['A2'] + df['B2']) / df['e2'],
                        y2  = lambda df: (df['A2'] + df['D2']) / df['e2'],
                        y   = lambda df: (df['y1'] + df['y2']) / 2.0)
