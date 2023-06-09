import os, threading, queue, tempfile
import concurrent.futures
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable
from tkinter.ttk import Progressbar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure, SubplotParams
from numpy.lib import recfunctions as rfn

import petmr

from calibration import (
        CoincidenceFileHandle,
        SingleDType,
        CoincidenceDType,
        load_block_singles_data,
        load_block_coincidence_data,
        max_events)

from filedialog import (askopenfilename,
                        askopenfilenames,
                        asksaveasfilename)

singles_filetypes = [("Singles",".SGL")]
coincidence_filetypes = [("Coincidences",".COIN")]
listmode_filetypes = [("Listmode data",".lm")]
sinogram_filetypes = [("Sinogram file", ".raw")]

def validate_singles():
    fnames = askopenfilenames(title = "Validate singles files",
                              filetypes = singles_filetypes)

    if not fnames: return

    with concurrent.futures.ThreadPoolExecutor(len(fnames)) as executor:
        futures = [executor.submit(petmr.validate_singles_file, f) for f in fnames]
        valid = [fut.result() for fut in futures]

    messages = []
    for f, v in zip(fnames, valid):
        base = os.path.basename(f) + ': '
        if v is True:
            messages.append(base + 'valid')
        else:
            if not v[0]: messages.append(base + 'no reset')
            if not v[1]: messages.append(base + 'm0 no timetags')
            if not v[2]: messages.append(base + 'm1 no timetags')
            if not v[3]: messages.append(base + 'm2 no timetags')
            if not v[4]: messages.append(base + 'm3 no timetags')

    tk.messagebox.showinfo(message = '\n'.join(messages))

class ProgressPopup(tk.Toplevel):
    def __init__(self, callback = None, fmt = 'Counts: {:,}'):
        super().__init__()
        self.status     = queue.Queue()
        self.data       = queue.Queue()
        self.terminate  = threading.Event()

        self.callback = callback
        self.fmt = fmt

        self.title('Progress')
        self.attributes('-type', 'dialog')
        self.protocol("WM_DELETE_WINDOW", self.terminate.set)
        self.progbar = Progressbar(self, length = 500)
        self.counts_label = tk.Label(self, text = 'Initializing...')
        self.progbar.pack(fill = tk.X, expand = True, padx = 10, pady = 10)
        self.counts_label.pack(pady = 10)

        self.update()

    def update(self, interval = 100):
        while not self.status.empty():
            vals = self.status.get()
            if isinstance(vals, Iterable):
                perc, val = vals
                self.counts_label.config(text = self.fmt.format(val))
            else: perc = vals
            
            self.progbar['value'] = float(perc)

        if self.data.empty():
            self.after(interval, self.update)
        else:
            self.destroy()
            if self.callback is not None:
                self.callback(self.data.get())

class SinglesLoader(ProgressPopup):
    def __init__(self, callback, flood_type):
        self.flood_type = flood_type
        self.callback = callback
        self.input_file = askopenfilename(
                title = "Load singles listmode data",
                filetypes = singles_filetypes)

        if self.input_file:
            super().__init__(self.callback)
            threading.Thread(target = self.load_singles).start()
        else:
            callback(ValueError("No file specified"))

    def load_singles(self):
        try:
            d = petmr.singles(self.input_file, self.flood_type, max_events,
                    self.terminate, self.status)
        except Exception as e:
            self.data.put(e)
            return

        d = rfn.unstructured_to_structured(d, SingleDType, copy = False)
        unique_blocks = np.unique(d['block'])
        with concurrent.futures.ThreadPoolExecutor(os.cpu_count()) as ex:
            fut = {ub: ex.submit(load_block_singles_data, d, ub) for ub in unique_blocks}
            self.data.put({ub: f.result() for ub, f in fut.items()})

class CoincidenceSorter(ProgressPopup):
    def __init__(self, callback):
        self.callback = lambda f: CoincidenceLoader(callback, f)
        self.input_files = askopenfilenames(
                title = "Select singles listmode data to sort",
                filetypes = singles_filetypes)

        if self.input_files:
            self.output_file = asksaveasfilename(
                    title = "Output file, or none",
                    filetypes = coincidence_filetypes)

            super().__init__(self.callback)
            threading.Thread(target = self.sort_coincidences).start()
        else:
            callback(ValueError("No files specified"))

    def sort_coincidences(self):
        f = self.output_file or tempfile.NamedTemporaryFile()
        fname = self.output_file or f.name
        nev = 0 if self.output_file else max_events

        try:
            petmr.coincidences(self.input_files, fname, nev,
                    self.status, self.terminate)
            self.data.put(f)
        except Exception as e:
            self.data.put(e)

class CoincidenceLoader(tk.Toplevel):
    def __init__(self, callback, f = None):
        if isinstance(f, Exception):
            callback(f)
            return

        if f is None:
            f = askopenfilename(
                    title = "Load coincidence listmode data",
                    filetypes = coincidence_filetypes)

        if not f:
            callback(ValueError("No file specified"))
            return

        self.f = f
        self.callback = callback 

        try:
            self.data = np.memmap(f, CoincidenceDType)
        except Exception as e:
            callback(e)
            return

        # create the UI window

        super().__init__()
        self.attributes('-type', 'dialog')
        self.title('Coincidence time distribution')
        self.protocol("WM_DELETE_WINDOW", lambda: [self.callback(), self.destroy()])

        self.fig = Figure()
        self.plt = self.fig.add_subplot()
        self.canvas = FigureCanvasTkAgg(self.fig, master = self)

        button_frame = tk.Frame(self)
        self.load_button = tk.Button(button_frame, text = 'Load Selected',
                command = lambda: self.load_start(self.load))
        self.save_button = tk.Button(button_frame, text = 'Save Selected',
                command = lambda: self.load_start(self.save))

        self.canvas.get_tk_widget().pack(padx = 10, pady = 10)
        button_frame.pack()
        self.load_button.pack(padx = 10, pady = 10, side = tk.LEFT)
        self.save_button.pack(padx = 10, pady = 10, side = tk.LEFT)

        self.canvas.mpl_connect('button_press_event', self.drag_start)
        self.canvas.mpl_connect('button_release_event', self.drag_stop)
        self.pp = None
        self.connection = None
        self.active_line = None
        self.block_files = {}
        self.draw_hist()

    def draw_hist(self):
        cf = CoincidenceFileHandle(self.f, 500, 1)
        self.ev_rate = cf.event_rate
        self.times = cf.times / 10
        self.idx = cf.idx

        self.lims = (self.times[0], self.times[-1])
        self.init_lines(self.lims)

        self.plt.plot(self.times, self.ev_rate)
        self.plt.set_xlabel('Time (seconds)')
        self.plt.set_ylabel('Coincidence Rate (CPS)')
        self.plt.grid()
        self.canvas.draw()

    """ Methods for moving the cursors to select a time span """

    def init_lines(self, lims):
        self.lines = [self.plt.axvline(x,linewidth=3,color='r') for x in lims]

    def drag_start(self, ev):
        if ev.xdata is not None:
            self.connection = self.canvas.mpl_connect('motion_notify_event', self.cursor_set)

            # set the nearest line as active
            xpos = np.array([l.get_xdata()[0] for l in self.lines])
            idx = np.argmin(np.abs(ev.xdata - xpos))
            self.active_line = self.lines[idx]

    def drag_stop(self, ev):
        if self.connection is not None:
            self.canvas.mpl_disconnect(self.connection)
            self.connection = None
            self.active_line = None

    def cursor_set(self, ev):
        if ev.xdata is not None and self.active_line is not None:
            if (ev.xdata > self.lims[0]) & (ev.xdata < self.lims[1]):
                self.active_line.set_xdata([ev.xdata]*2)
                self.canvas.draw()

    """ Methods for loading or saving a subset of the coincidence data """

    def load_start(self, target):
        self.pp = ProgressPopup(fmt = 'Block: {}')

        self.bg = threading.Thread(target = target)
        self.load_button.config(state = tk.DISABLED)
        self.save_button.config(state = tk.DISABLED)
        self.bg.start()
        self.load_check()

    def load_check(self):
        if self.bg.is_alive():
            self.after(100, self.load_check)
        else:
            self.load_button.config(state = tk.NORMAL)
            self.save_button.config(state = tk.NORMAL)
            self.callback(self.block_files)
            self.pp.data.put(None)
            self.pp = None

    def subset(self, nev = max_events):
        start_end = np.sort([l.get_xdata()[0] for l in self.lines])
        times = np.round(start_end)

        start_end = np.searchsorted(self.times, start_end)
        start, end = self.idx[start_end]

        if nev:
            end = min(end, int(start + nev))

        return start, end, *times

    def load(self):
        start, end, t0, t1 = self.subset()
        subset = self.data[start:end]
        print(f'Load {t0}s to {t1}s: {round((end-start) / 1e6)}M events')

        lk = threading.Lock()
        n = 0

        def launch(ub):
            nonlocal n
            d = load_block_coincidence_data(subset, ub)
            with lk:
                self.pp.status.put((n / petmr.nblocks * 100, n))
                n += 1
            return d

        with concurrent.futures.ThreadPoolExecutor(os.cpu_count()) as ex:
            futs = {ub: ex.submit(launch, ub) for ub in range(petmr.nblocks)}
            self.block_files = {ub: f.result() for ub, f in futs.items()}

    def save(self):
        start, end, t0, t1 = self.subset(None)
        subset = self.data[start:end]

        nev = subset.shape[0]
        idx = int(np.clip(np.log10(nev) / 3, 1, 4))
        char = ['K', 'M', 'B', 'T'][idx-1]
        print(f'Save {round(nev/(1e3 ** idx), 1)}{char} events from {t0}s to {t1}s')

        newfile = asksaveasfilename(title = "New coincidence file",
                                    filetypes = coincidence_filetypes)

        if newfile is None: return

        arr = np.memmap(newfile, subset.dtype,
                mode = 'w+', shape = subset.shape)
        arr[:] = subset[:]

