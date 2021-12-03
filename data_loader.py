import os
import gc
import threading
import queue
import petmr
import pandas as pd
import tkinter as tk
import numpy as np
from tkinter.ttk import Progressbar

singles_filetypes = [("Singles",".SGL")]
coincidence_filetypes = [("Coincidences",".COIN")]

def coincidence_to_df(lst_of_arrays):
    names = ['block', 'e1', 'e2', 'x', 'y', 'tdiff']
    return pd.DataFrame(np.column_stack(lst_of_arrays), columns = names)

def prepare_coincidences(d):
    """ Add the columns necessary for plotting to the coincidence dataframe """
    d['x'] /= 511
    d['y'] /= 511
    d.insert(len(d.columns), 'es',  d['e1'] + d['e2'])
    d.insert(len(d.columns), 'doi', d['e1'] / d['es'])
    del d['e1']
    del d['e2']
    return d

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

        self.fileselector.label.config(text = self.input_files)
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
        """ Load singles data from one or more .SGL files. If multiple files are provided,
        the data will be concatenated into a single dataframe
        """
        names = ['block', 'time', 'A1', 'B1', 'C1', 'D1', 'A2', 'B2', 'C2', 'D2']

        d = []
        for i,f in enumerate(self.input_files):
            d.append(petmr.singles(f))
            perc = float(i + 1) / len(self.input_files) * 100
            self.stat_queue.put((perc, i + 1, "Completed"))
        
        d = [pd.DataFrame(dict(zip(names, di))) for di in d]
        d = pd.concat(d, ignore_index = True)
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
        """ Load coincicdence data by sorting the events in one or more singles files. If
        multiple files are provided, they must be time-aligned from a single acquisition
        """
        # (a,b) -> [a_df, b_df] -> ab_df
        d = petmr.coincidences(
                self.terminate,
                self.stat_queue,
                self.input_files,
                self.output_file)

        d = [coincidence_to_df(c) for c in d]
        d = pd.concat(d, ignore_index = True)
        d = prepare_coincidences(d)
        self.data_queue.put(d)

    def load_coincidences(self):
        """ Load coincidence data from a file on disk. If multiple file are provided,
        the dataframes will be concatenated
        """

        d = petmr.load(self.input_files[0])
        d = [coincidence_to_df(di) for di in d]
        d = pd.concat(d, ignore_index = True)
        d = prepare_coincidences(d)
        #d.info(memory_usage = 'deep')
        self.data_queue.put(d)

