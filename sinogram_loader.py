import threading, queue
import tkinter as tk
from tkinter.ttk import Progressbar

class SinogramLoaderPopup:
    def __init__(self, root, callback, target, *args):
        self.target = target
        self.callback = callback
        self.status = queue.Queue()
        self.terminate = threading.Event()

        self.popup = tk.Toplevel(root)
        self.popup.title('Progress')
        self.popup.attributes('-type', 'dialog')
        self.popup.protocol("WM_DELETE_WINDOW", self.terminate.set)
        self.progbar = Progressbar(self.popup, length = 500)
        self.progbar.pack(fill = tk.X, expand = True, padx = 10, pady = 10)

        self.thr = threading.Thread(target = self.wrapper, args = args)
        self.thr.start()
        self.check()

    def check(self, interval = 100):
        while not self.status.empty():
            self.progbar['value'] = self.status.get()

        if self.thr.is_alive():
            self.progbar.after(interval, self.check)
        else:
            self.popup.destroy()
            if self.callback is not None:
                self.callback(self.result)

    def wrapper(self, *args):
        self.result = self.target(*args, self.terminate, self.status)
