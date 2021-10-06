import os
import threading
import queue
import petmr
import pandas as pd
import tkinter as tk
import numpy as np
from tkinter.ttk import Progressbar

singles_filetypes = [("Singles",".SGL")]
coincidence_filetypes = [("Coincidences",".COIN")]

class DataLoaderPopup:
    def __init__(self, root, fileselector, update_data_cb):
        self.fileselector = fileselector
        self.update_data_cb = update_data_cb 
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

        if ext == '.COIN':
            # Just load the coincidences, regardless of coincidence sorting checkbox
            self.bg = threading.Thread(target = self.load_coincidences)

        elif ext == '.SGL':
            if self.fileselector.sort_coin.get():
                # Sort singles to coincidences
                self.output_file = tk.filedialog.asksaveasfilename(
                        title = "Output file, or none",
                        initialdir = os.path.expanduser('~'),
                        filetypes = coincidence_filetypes) or None
                self.bg = threading.Thread(target = self.sort_coincidences)
                self.allow_store = not bool(self.output_file)
            else:
                # Just load the singles
                self.bg = threading.Thread(target = self.load_singles)

        else:
            # Got an invalid filetype
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
            self.counts_label.config(text = f'Counts: {counts:,}')
            self.progbar['value'] = perc

        if self.bg.is_alive():
            self.popup.after(1000, self.check)
        else:
            if not self.data_queue.empty():
                d = self.data_queue.get()
                self.update_data_cb(d)
            else:
                raise RuntimeError("Data loading failed")

            self.popup.destroy()
            self.fileselector.store_button.config(
                    state = tk.NORMAL if self.allow_store else tk.DISABLED)
            self.fileselector.load_button.config(state = tk.NORMAL)

    def on_close(self):
        self.terminate.set()
        self.bg.join()
        self.check()

    def load_singles(self):
        names = ['block', 'time', 'A1', 'B1', 'C1', 'D1', 'A2', 'B2', 'C2', 'D2']
        d = petmr.singles(self.input_files)
        d = pd.DataFrame(dict(zip(names, d)))
        d = d.assign(e1  = lambda df: df.loc[:,'A1':'D1'].sum(axis=1),
                     e2  = lambda df: df.loc[:,'A2':'D2'].sum(axis=1),
                     es  = lambda df: df['e1'] + df['e2'],
                     doi = lambda df: df['e1'] / df['es'],
                     x1  = lambda df: (df['A1'] + df['B1']) / df['e1'],
                     y1  = lambda df: (df['A1'] + df['D1']) / df['e1'],
                     x2  = lambda df: (df['A2'] + df['B2']) / df['e2'],
                     y2  = lambda df: (df['A2'] + df['D2']) / df['e2'],
                     x   = lambda df: df['x1'],
                     y   = lambda df: (df['y1'] + df['y2']) / 2.0)
        self.data_queue.put(d)

    def sort_coincidences(self):
        ab = petmr.coincidences(self.terminate, self.stat_queue, self.input_files, self.output_file)
        d = self.prepare_coincidences(ab)
        self.data_queue.put((ab, d))

    def load_coincidences(self):
        d = [petmr.load(f) for f in self.input_files];
        d = [self.prepare_coincidences(di) for di in d]
        d = pd.concat(d, ignore_index = True)
        self.data_queue.put(d)

    def prepare_coincidences(self, ab):
        names = ['block', 'e1', 'e2', 'x', 'y', 'tdiff']
        d = [pd.DataFrame(np.column_stack(a), columns = names) for a in ab]
        d = pd.concat(d, ignore_index = True)
        d = d.assign(x   = lambda df: df['x'] / 511,
                     y   = lambda df: df['y'] / 511,
                     es  = lambda df: df['e1'] + df['e2'],
                     doi = lambda df: df['e1'] / df['es'])
        return d
