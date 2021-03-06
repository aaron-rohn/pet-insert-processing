import os, threading, queue, traceback, tempfile, petmr
import concurrent.futures
import pandas as pd
import tkinter as tk
import numpy as np
from tkinter.ttk import Progressbar

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure, SubplotParams

singles_filetypes = [("Singles",".SGL")]
coincidence_filetypes = [("Coincidences",".COIN")]

n_doi_bins = 4096
max_events = int(1e9)
coincidence_cols = 11

class ProgressPopup(tk.Toplevel):
    def __init__(self, stat_queue, data_queue, terminate, callback):
        super().__init__()
        self.stat_queue = stat_queue
        self.data_queue = data_queue
        self.terminate = terminate
        self.callback = callback

        self.title('Progress')
        self.attributes('-type', 'dialog')
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.progbar = Progressbar(self, length = 500)
        self.counts_label = tk.Label(self, text = '')

        self.progbar.pack(fill = tk.X, expand = True, padx = 10, pady = 10)
        self.counts_label.pack(pady = 10)

    def on_close(self):
        self.terminate.set()

    def update(self, interval = 100):
        while not self.stat_queue.empty():
            perc, counts = self.stat_queue.get()
            self.counts_label.config(text = f'Counts: {counts:,}')
            self.progbar['value'] = perc

        if self.data_queue.empty():
            self.after(interval, self.update)
        else:
            self.destroy()
            self.callback(self.data_queue.get())

class SinglesLoader:
    def __init__(self, callback):
        self.terminate = threading.Event()
        self.data_queue = queue.Queue()
        self.stat_queue = queue.Queue()

        self.input_file = tk.filedialog.askopenfilename(
            title = "Load singles listmode data",
            initialdir = "/",
            filetypes = singles_filetypes)

        if not self.input_file: raise ValueError("No file specified")
        
        self.bg = threading.Thread(target = self.load_singles)
        self.bg.start()

        self.popup = ProgressPopup(self.stat_queue,
                                   self.data_queue,
                                   self.terminate,
                                   callback)
        self.popup.update()

    def load_singles(self):
        args = [self.terminate, self.stat_queue, self.input_file, max_events]
        d = petmr.singles(*args)

        blocks = d[0]
        unique_blocks = np.unique(blocks).tolist()
        block_files = {}

        for ub in unique_blocks:
            tf = tempfile.NamedTemporaryFile()
            idx = np.where(blocks == ub)[0]
            arr = np.memmap(tf.name, np.uint16,
                    mode = 'w+', shape = (len(idx), 4))

            # Energy sum -> eF + eR
            arr[:,0] = d[1][idx] + d[2][idx]

            # DOI -> eF / eSUM
            tmp = d[1][idx].astype(float)
            tmp *= (n_doi_bins / arr[:,0])
            arr[:,1] = tmp

            # X, Y
            arr[:,2] = d[3][idx]
            arr[:,3] = d[4][idx]

            block_files[ub] = arr

        self.data_queue.put(block_files)

class CoincidenceLoader:
    def __init__(self, callback):
        self.input_file = tk.filedialog.askopenfilename(
                title = "Load coincidence listmode data",
                initialdir = "/",
                filetypes = coincidence_filetypes)

        if not self.input_file: raise ValueError("No file specified")

        sz = os.path.getsize(self.input_file)
        nrow = int((sz/2) // coincidence_cols)
        #data = np.memmap(self.input_file, np.uint16).reshape((-1,coincidence_cols))
        data = np.memmap(self.input_file, np.uint16, shape = (nrow, coincidence_cols))
        self.prof = CoincidenceProfilePlot(data, callback)

class CoincidenceSorter:
    def __init__(self, outside_callback):
        self.terminate = threading.Event()
        self.data_queue = queue.Queue()
        self.stat_queue = queue.Queue()

        # this is the callback given to the CoincidenceProfilePlot
        self.outside_callback = outside_callback

        self.input_files = []
        while True:
            self.input_files += list(tk.filedialog.askopenfilenames(
                title = "Select singles listmode data to sort",
                initialdir = "/",
                filetypes = singles_filetypes + coincidence_filetypes))
            if not tk.messagebox.askyesno(message = "Select additional files?"):
                break

        if not self.input_files: raise ValueError("No files specified")

        self.output_file = tk.filedialog.asksaveasfilename(
                title = "Output file, or none",
                initialdir = os.path.dirname(self.input_files[0]),
                filetypes = coincidence_filetypes) or None
        
        self.bg = threading.Thread(target = self.sort_coincidences)
        self.bg.start()

        self.popup = ProgressPopup(self.stat_queue,
                                   self.data_queue,
                                   self.terminate,
                                   self.callback)
        self.popup.update()

    def sort_coincidences(self):
        """ Load coincicdence data by sorting the events in one or more singles files. If
        multiple files are provided, they must be time-aligned from a single acquisition
        """
        
        args = [self.terminate, self.stat_queue, self.input_files]

        if self.output_file is not None:
            args += [self.output_file]
            petmr.coincidences(*args)
            self.data_queue.put(self.output_file)
        else:
            tf = tempfile.NamedTemporaryFile()
            args += [tf.name, max_events]
            petmr.coincidences(*args)
            self.data_queue.put(tf)

    def callback(self, data_file):
        """ this is called from the context of the ProgressPopup once
        data is put on the data queue """

        if isinstance(data_file, str):
            sz = os.path.getsize(data_file)
        else:
            sz = os.fstat(data_file.fileno()).st_size

        nrow = int((sz/2) // coincidence_cols)
        #data = np.memmap(data_file, np.uint16).reshape((-1,coincidence_cols))
        data = np.memmap(data_file, np.uint16, shape = (nrow, coincidence_cols))
        self.prof = CoincidenceProfilePlot(data, self.outside_callback)

class CoincidenceProfilePlot(tk.Toplevel):
    def __init__(self, data, callback):
        super().__init__()
        self.attributes('-type', 'dialog')
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.callback = callback 
        self.data = data

        self.fig = Figure()
        self.plt = self.fig.add_subplot()
        self.canvas = FigureCanvasTkAgg(self.fig, master = self)
        self.load_button = tk.Button(self, text = 'Load Selected',
                command = self.load_start)

        self.canvas.get_tk_widget().pack(padx = 10, pady = 10)
        self.load_button.pack(pady = 10)

        self.canvas.mpl_connect('button_press_event', self.drag_start)
        self.canvas.mpl_connect('button_release_event', self.drag_stop)

        self.connection = None
        self.active_line = None
        self.draw_hist()
        self.set_title()

    def on_close(self):
        self.callback({})
        self.destroy()

    def set_title(self, status = None):
        title = 'Coincidence time distribution'
        if status is not None:
            title += (' - ' + status)
        self.title(title)

    def draw_hist(self):
        t = self.data[:,10]
        nevents = t.shape[0]
        time_span = t[-1] - t[0]
        time_span_sec = time_span / 10

        if time_span_sec < 10:
            sec_to_avg = 0.1
        elif time_span_sec < 100:
            sec_to_avg = 1.0
        else:
            sec_to_avg = 10.0

        self.ev_per_period = int(nevents / time_span_sec * sec_to_avg)

        self.times = self.data[::self.ev_per_period,10]
        self.lims = (self.times[0], self.times[-1])
        self.init_lines(self.lims)

        self.ev_rate = self.ev_per_period / np.diff(self.times)

        k_samples = 6
        kernel = np.ones(k_samples) / k_samples

        padded = np.concatenate((np.repeat(self.ev_rate[0],  k_samples/2),
                                 self.ev_rate,
                                 np.repeat(self.ev_rate[-1], k_samples/2)))

        self.ev_rate = np.convolve(padded, kernel, mode = 'valid')
        self.plt.plot(self.times, self.ev_rate)
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

    """ Methods for loading a subset of the coincidence data """

    def load_start(self):
        self.bg = threading.Thread(target = self.load, daemon = True)
        self.load_button.config(state = tk.DISABLED)
        self.bg.start()
        self.set_title('subset data')
        self.load_check()

    def load_check(self):
        if self.bg.is_alive():
            self.after(100, self.load_check)
        else:
            self.set_title()
            self.load_button.config(state = tk.NORMAL)
            self.callback(self.block_files)

    def load(self):
        start, end = sorted([l.get_xdata()[0] for l in self.lines])
        print(f'Load from {round(start/10)}s to {round(end/10)}s')

        start, end = np.searchsorted(self.times, [start,end]) * self.ev_per_period
        data_subset = self.data[start:end,:]
        nev = data_subset.shape[0]

        if nev > max_events:
            data_subset = data_subset[0:max_events,:]

        print(f'Load {data_subset.shape[0] / 1e6}M events')

        blocks = data_subset[:,0]
        blka = blocks >> 8
        blkb = blocks & 0xFF
        unique_blocks = np.unique(np.concatenate([blka,blkb])).tolist()
        self.block_files = {}

        for ub in unique_blocks:
            print(f'Block {ub}')
            idxa = np.where(blka == ub)[0]
            idxb = np.where(blkb == ub)[0]

            rowa = data_subset[idxa,:]
            rowb = data_subset[idxb,:]

            tf = tempfile.NamedTemporaryFile()

            shape = (len(idxa) + len(idxb), 4)
            arr = np.memmap(tf.name, np.uint16,
                    mode = 'w+', shape = shape)

            # Energy sum
            arr[:,0] = np.concatenate(
                    [rowa[:,2] + rowa[:,3], rowb[:,4] + rowb[:,5]])

            # DOI
            tmp = np.concatenate([rowa[:,2], rowb[:,4]]).astype(float)
            tmp *= (n_doi_bins / arr[:,0])
            arr[:,1] = tmp

            # X, Y
            arr[:,2] = np.concatenate([rowa[:,6], rowb[:,8]])
            arr[:,3] = np.concatenate([rowa[:,7], rowb[:,9]])

            self.block_files[ub] = arr

def validate_singles():
    fnames = list(tk.filedialog.askopenfilenames(
            title = "Validate singles files",
            initialdir = "/",
            filetypes = singles_filetypes))

    if not fnames: return

    with concurrent.futures.ThreadPoolExecutor() as executor:
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
