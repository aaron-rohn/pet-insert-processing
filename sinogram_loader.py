import threading
import tkinter as tk
from tkinter.ttk import Progressbar
import petmr
import queue

class SinogramLoaderPopup:
    def __init__(self, root, callback, fname, lut_dir):
        self.fname = fname
        self.lut_dir = lut_dir
        self.callback = callback 

        self.data_queue = queue.Queue()
        self.stat_queue = queue.Queue()
        self.terminate = threading.Event()

        self.popup = tk.Toplevel(root)
        self.popup.title('Progress')
        self.popup.attributes('-type', 'dialog')
        self.popup.protocol("WM_DELETE_WINDOW", self.on_close)
        self.progbar = Progressbar(self.popup, length = 500)
        self.progbar.pack(fill = tk.X, expand = True, padx = 10, pady = 10)

        self.bg = threading.Thread(
                target = petmr.sort_sinogram,
                args = [self.fname,
                        self.lut_dir,
                        self.terminate,
                        self.stat_queue,
                        self.data_queue])

        self.bg.start()
        self.check()

    def on_close(self):
        self.terminate.set()
        self.bg.join()
        self.check()

    def check(self, interval = 1000):
        while not self.stat_queue.empty():
            perc = self.stat_queue.get()
            self.progbar['value'] = perc

        if self.bg.is_alive():
            self.popup.after(interval, self.check)
        else:
            if not self.data_queue.empty():
                self.callback(self.data_queue.get())
            else:
                self.callback(RuntimeError("Sinogram sorting failed"))
            self.popup.destroy()
