import threading
import queue
import pandas as pd
import tkinter as tk
from tkinter.ttk import Progressbar
from petmr import singles, coincidences

class DataLoader(threading.Thread):
    def __init__(self, fname, coin, terminate, stat_queue, data_queue):
        super().__init__()
        self.fname = fname
        self.coin = coin
        self.terminate = terminate
        self.stat_queue = stat_queue
        self.data_queue = data_queue

    def run(self):
        """ Load listmode data from one or more files and calculate necessary columns """
        c = ['block', 'time', 'A1', 'B1', 'C1', 'D1', 'A2', 'B2', 'C2', 'D2']

        if self.coin:
            d = coincidences(self.terminate, self.stat_queue, self.fname)
            d = [pd.DataFrame(dict(zip(c,di))) for di in d]
            d = pd.concat(d, ignore_index = True)
        else:
            d = singles(self.fname)
            d = pd.DataFrame(dict(zip(c, d)))

        d = (d.assign(e1  = lambda df: df.loc[:,'A1':'D1'].sum(axis=1),
                       e2  = lambda df: df.loc[:,'A2':'D2'].sum(axis=1),
                       es  = lambda df: df['e1'] + df['e2'],
                       doi = lambda df: df['e1'] / df['es'],
                       x1  = lambda df: (df['A1'] + df['B1']) / df['e1'],
                       y1  = lambda df: (df['A1'] + df['D1']) / df['e1'],
                       x2  = lambda df: (df['A2'] + df['B2']) / df['e2'],
                       y2  = lambda df: (df['A2'] + df['D2']) / df['e2'],
                       y   = lambda df: (df['y1'] + df['y2']) / 2.0)
               .groupby('block'))

        self.data_queue.put(d)

class DataLoaderPopup:
    def __init__(self, root, button, fname, coin, callback):
        self.popup = tk.Toplevel(root)
        self.popup.title('Read data')
        self.popup.attributes('-type', 'dialog')
        self.popup.protocol("WM_DELETE_WINDOW", self.on_close)
        self.button = button
        self.button.config(state = tk.DISABLED)

        self.progbar = Progressbar(
                self.popup,
                orient = tk.HORIZONTAL,
                length = 500,
                mode = 'determinate')
        self.progbar.pack(fill = tk.X, expand = True, padx = 10, pady = 10)
        self.counts_label = tk.Label(self.popup, text = 'Counts: 0')
        self.counts_label.pack(pady = 10)

        self.callback = callback 
        self.terminate = threading.Event()
        self.data_queue = queue.Queue()
        self.stat_queue = queue.Queue()
        self.bg = DataLoader(fname, coin, self.terminate, self.stat_queue, self.data_queue)
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
            self.button.config(state = tk.NORMAL)

    def on_close(self):
        self.terminate.set()
        self.bg.join()
        self.check()
