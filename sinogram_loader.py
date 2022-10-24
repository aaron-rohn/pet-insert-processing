import os, threading, queue, petmr
import tkinter as tk
from tkinter.ttk import Progressbar
import numpy as np
import matplotlib.pyplot as plt

from data_loader import (
        coincidence_cols,
        scaling_nevents,
        scaling_factor,
        read_times,
        coincidence_filetypes
        )

class SinogramLoaderPopup:
    def __init__(self, root, callback, target,
                 fname, cfgdir, *args):

        nperiods = 100
        scaling, times, fpos = read_times(fname, nperiods)
        self.callback = callback 
        self.target = target

        self.popup = tk.Toplevel(root)
        self.popup.title('Progress')
        self.popup.attributes('-type', 'dialog')
        self.popup.protocol("WM_DELETE_WINDOW", self.on_close)
        self.progbar = Progressbar(self.popup, length = 500)
        self.progbar.pack(fill = tk.X, expand = True, padx = 10, pady = 10)

        self.terminate = threading.Event()
        self.data_queue = queue.Queue()
        self.stat_queue = queue.Queue()

        self.bg = threading.Thread(target = self.handle_listmode, 
                args = list(args) + [fname, cfgdir,
                                     scaling, fpos,
                                     self.terminate,
                                     self.stat_queue,
                                     self.data_queue])

        self.bg.start()
        self.check()

    def on_close(self):
        self.terminate.set()
        self.bg.join()
        self.check()

    def check(self, interval = 100):
        while not self.stat_queue.empty():
            perc = self.stat_queue.get()
            self.progbar['value'] = perc

        if self.bg.is_alive():
            self.popup.after(interval, self.check)
        else:
            if self.callback is not None:
                if not self.data_queue.empty():
                    self.callback(self.data_queue.get())
                else:
                    self.callback(RuntimeError("Sinogram sorting failed"))
            self.popup.destroy()

    def handle_listmode(self, *args):
        try:
            self.target(*args)
        except RuntimeError as e:
            self.data_queue.put(e)
