import os, threading, queue, traceback, tempfile
import concurrent.futures
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from tkinter.ttk import Progressbar
from collections.abc import Iterable
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure, SubplotParams

import petmr

from calibration import (load_block_singles_data,
                         load_block_coincidence_data,
                         CoincidenceFileHandle,
                         n_doi_bins,
                         max_events,
                         coincidence_cols)

singles_filetypes = [("Singles",".SGL")]
coincidence_filetypes = [("Coincidences",".COIN")]
listmode_filetypes = [("Listmode data",".lm")]

class ProgressPopup(tk.Toplevel):
    def __init__(self, stat_queue, data_queue, terminate, callback, fmt = 'Counts: {:,}'):
        super().__init__()
        self.stat_queue = stat_queue
        self.data_queue = data_queue
        self.terminate = terminate
        self.callback = callback
        self.fmt = fmt

        self.title('Progress')
        self.attributes('-type', 'dialog')
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.progbar = Progressbar(self, length = 500)
        self.counts_label = tk.Label(self, text = '')

        self.progbar.pack(fill = tk.X, expand = True, padx = 10, pady = 10)
        self.counts_label.pack(pady = 10)
        self.update()

    def on_close(self):
        self.terminate.set()

    def update(self, interval = 100):
        while not self.stat_queue.empty():
            perc, val = self.stat_queue.get()

            self.counts_label.config(text =
                self.fmt.format(*val) if isinstance(val, Iterable) else
                self.fmt.format(val))

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

    def load_singles(self):
        args = [self.terminate, self.stat_queue, self.input_file, max_events]
        d = petmr.singles(*args)

        blocks = d[0]
        unique_blocks = np.unique(blocks).tolist()
        block_files = {}

        for ub in unique_blocks:
            block_files[ub] = load_block_singles_data(
                    d, blocks, ub)

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
        data = np.memmap(self.input_file, np.uint16, shape = (nrow, coincidence_cols))
        self.prof = CoincidenceProfilePlot(data, callback)

class CoincidenceSorter:
    def __init__(self, outside_callback):
        self.terminate = threading.Event()
        self.data_queue = queue.Queue()
        self.stat_queue = queue.Queue()

        # this is the callback given to the CoincidenceProfilePlot
        self.outside_callback = outside_callback

        self.input_files = list(tk.filedialog.askopenfilenames(
            title = "Select singles listmode data to sort",
            initialdir = "/", filetypes = singles_filetypes))

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
        cf = CoincidenceFileHandle(self.data, 500, 1)
        self.ev_rate = cf.event_rate
        self.times = cf.times
        self.idx = cf.idx

        self.lims = (self.times[0], self.times[-1])
        self.init_lines(self.lims)

        self.plt.plot(self.times, self.ev_rate)
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
        self.bg = threading.Thread(target = target, daemon = True)
        self.load_button.config(state = tk.DISABLED)
        self.save_button.config(state = tk.DISABLED)
        self.bg.start()
        self.set_title('subset data')
        self.load_check()

    def load_check(self):
        if self.bg.is_alive():
            self.after(100, self.load_check)
        else:
            self.set_title()
            self.load_button.config(state = tk.NORMAL)
            self.save_button.config(state = tk.NORMAL)
            self.callback(self.block_files, self.current_ev_rate)

    def load(self):
        start_end = np.sort([l.get_xdata()[0] for l in self.lines])
        print('Load from {}s to {}s'.format(*np.round(start_end / 10)))

        start_end = np.searchsorted(self.times, start_end)
        self.current_ev_rate = self.ev_rate[start_end[0]]
        start, end = self.idx[start_end]

        if end - start > max_events:
            end = int(start + max_events)

        data_subset = self.data[start:end]

        print(f'Load {data_subset.shape[0] / 1e6}M events')

        blocks = data_subset[:,0]
        blka = blocks >> 8
        blkb = blocks & 0xFF
        unique_blocks = np.unique(np.concatenate([blka,blkb])).tolist()
        self.block_files = {}

        for ub in unique_blocks:
            print(f'Block {ub}', end = '\r')
            self.block_files[ub] = load_block_coincidence_data(
                    data_subset, blka, blkb, ub)
        print('')

    def save(self):
        self.block_files = {}
        self.current_ev_rate = 0

        start_end = np.sort([l.get_xdata()[0] for l in self.lines])
        start_end = np.searchsorted(self.times, start_end)
        start, end = self.idx[start_end]
        data_subset = self.data[start:end,:]
        nev = data_subset.shape[0]

        idx = int(np.clip(np.log10(nev) / 3, 1, 4))
        char = ['K', 'M', 'B', 'T'][idx-1]
        print(f'Save {round(nev/(1e3 ** idx), 1)}{char} events from {round(start/10)}s to {round(end/10)}s')

        newfile = tk.filedialog.asksaveasfilename(
                title = "New coincidence file",
                initialdir = os.path.abspath(os.path.sep),
                filetypes = coincidence_filetypes)

        if not newfile: return

        arr = np.memmap(newfile, np.uint16,
                mode = 'w+', shape = data_subset.shape)
        arr[:,:] = data_subset[:,:]

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
