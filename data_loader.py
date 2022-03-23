import os, threading, queue, traceback, tempfile, petmr
import pandas as pd
import tkinter as tk
import numpy as np
from tkinter.ttk import Progressbar

singles_filetypes = [("Singles",".SGL")]
coincidence_filetypes = [("Coincidences",".COIN")]

n_doi_bins = 4096
max_events = int(500e6)

class DataLoaderPopup:
    def __init__(self, root, fileselector, update_data_cb):
        self.fileselector = fileselector
        self.update_data_cb = update_data_cb 
        self.terminate = threading.Event()
        self.data_queue = queue.Queue()
        self.stat_queue = queue.Queue()
        self.output_file = None

        self.input_files = []

        while True:
            self.input_files += list(tk.filedialog.askopenfilenames(
                title = "Load listmode data",
                initialdir = "/",
                filetypes = singles_filetypes + coincidence_filetypes))
            if not tk.messagebox.askyesno(message = "Select additional files?"):
                break

        if not self.input_files: raise ValueError("No input file specified")
        _, ext = os.path.splitext(self.input_files[0])

        if ext == '.COIN':
            # Just load the coincidences, regardless of coincidence sorting checkbox
            self.bg = threading.Thread(target = self.load_coincidences)

        elif ext == '.SGL':
            if self.fileselector.sort_coin.get():
                # Sort singles to coincidences
                self.output_file = tk.filedialog.asksaveasfilename(
                        title = "Output file, or none",
                        initialdir = os.path.dirname(self.input_files[0]),
                        filetypes = coincidence_filetypes) or None
                self.bg = threading.Thread(target = self.sort_coincidences)

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
        self.counts_label = tk.Label(self.popup, text = '')
        self.counts_label.pack(pady = 10)

        self.fileselector.load_button.config(state = tk.DISABLED)
        self.bg.start()
        self.check()

    def check(self, interval = 1000):
        """ Check (in the UI thread) if the data loader thread has
        completed. When implemented, update the status information
        in the popup.
        Once the thread is completed, retrieve the result from the
        data queue, return it to the caller via the provided callback,
        update UI elements to reflect the exit state, and finish
        """

        while not self.stat_queue.empty():
            perc, counts, *label = self.stat_queue.get()
            label = label[0] if label else 'Counts'
            self.counts_label.config(text = f'{label}: {counts:,}')
            self.progbar['value'] = perc

        if self.bg.is_alive():
            self.popup.after(interval, self.check)

        else:
            if not self.data_queue.empty():
                self.update_data_cb(self.data_queue.get())
            else:
                self.update_data_cb(RuntimeError("Data loading failed"))

            self.popup.destroy()
            self.fileselector.load_button.config(state = tk.NORMAL)

    def on_close(self):
        """ If the popup is asked to close before the background thread has completed,
        ask the background thread to exit gracefully, then check for any output data
        before returning
        """
        self.terminate.set()
        self.bg.join()
        self.check()

    def load_singles(self):
        """ Load singles data from one .SGL file and
        group events by block
        """

        try:
            d = petmr.singles(self.input_files[0], max_events)
        except Exception as ex:
            print(traceback.format_exc())
            self.data_queue.put(ex)
            return

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

    def sort_coincidences(self):
        """ Load coincicdence data by sorting the events in one or more singles files. If
        multiple files are provided, they must be time-aligned from a single acquisition
        """
        
        args = [self.terminate, self.stat_queue, self.input_files]

        try:
            if self.output_file is not None:
                args += [0, self.output_file]
                petmr.coincidences(*args)
                self.data_queue.put({})
            else:
                tf = tempfile.NamedTemporaryFile()
                args += [max_events, tf.name]
                petmr.coincidences(*args)
                self.load_coincidences(tf.name)

        except Exception as ex:
            print(traceback.format_exc())
            self.data_queue.put(ex)

    def load_coincidences(self, coincidence_file = None):
        """ Load coincidence data from a file on disk, and
        group events by block to visualize flood, energy, and
        doi histograms.
        """

        if coincidence_file is None:
            coincidence_file = self.input_files[0]

        d = np.memmap(coincidence_file, np.uint16).reshape((-1,10))

        blocks = d[:,0]
        blka = blocks >> 8
        blkb = blocks & 0xFF
        unique_blocks = np.unique(np.concatenate([blka,blkb])).tolist()
        block_files = {}

        for ub in unique_blocks:
            idxa = np.where(blka == ub)[0]
            idxb = np.where(blkb == ub)[0]

            rowa = d[idxa,:]
            rowb = d[idxb,:]

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

            block_files[ub] = arr

        self.data_queue.put(block_files)
