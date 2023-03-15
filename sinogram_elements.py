import os, glob, re, json, threading
import tkinter as tk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure, SubplotParams
import matplotlib.pyplot as plt
import concurrent.futures

from scipy import ndimage
import petmr
from data_loader import coincidence_filetypes, listmode_filetypes, sinogram_filetypes
from sinogram_loader import SinogramLoaderPopup
from filedialog import (check_config_dir,
                        askopenfilename,
                        askopenfilenames,
                        asksaveasfilename,
                        askformatfilenames)

class SinogramDisplay(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.sino_data = None
        self.ldr = None
        self.button_frame = tk.Frame(self)

        self.save_lm   = tk.Button(self.button_frame, text = "Save Listmode", command = self.save_listmode)
        self.load_lm = tk.Button(self.button_frame, text = "Load Listmode", command = self.load_listmode)
        self.save_sino = tk.Button(self.button_frame, text = "Save Sinogram", command = self.save_sinogram)
        self.load_sino = tk.Button(self.button_frame, text = "Load Sinogram", command = self.load_sinogram)
        self.multiply_button = tk.Button(self.button_frame, text = "Multiply", command = lambda: self.operation(np.multiply))
        self.subtract_button = tk.Button(self.button_frame, text = "Subtract", command = lambda: self.operation(np.subtract))

        self.energy_window_var = tk.DoubleVar(value = 0.2)
        self.max_doi_var = tk.IntVar(value = petmr.ndoi)
        self.sort_prompts_var = tk.BooleanVar(value = True)
        self.sort_delays_var = tk.BooleanVar(value = False)

        self.cb_frame = tk.Frame(self)
        self.energy_window_menu = tk.OptionMenu(self.cb_frame, self.energy_window_var,
                0.2, 0.4, 0.6, 0.8, 1.0, -1.0)
        self.max_doi_menu = tk.OptionMenu(self.cb_frame, self.max_doi_var, *np.arange(0,petmr.ndoi+1))
        self.sort_prompts_cb = tk.Checkbutton(self.cb_frame, text = "Prompts", variable = self.sort_prompts_var)
        self.sort_delays_cb = tk.Checkbutton(self.cb_frame, text = "Delays", variable = self.sort_delays_var)

        self.plot_frame = tk.Frame(self)
        self.plot_frame.rowconfigure(0, weight = 1)
        self.plot_frame.columnconfigure(0, weight = 1)
        self.plot_frame.columnconfigure(1, weight = 1)

        # Plot for counts in each pair of system planes

        self.plane_counts_fig = Figure(subplotpars = SubplotParams(0,0,1,1))
        self.plane_counts_plt = self.plane_counts_fig.add_subplot(frame_on = False)
        self.plane_counts_canvas = FigureCanvasTkAgg(
                self.plane_counts_fig, master = self.plot_frame)
        self.plane_counts_canvas.get_tk_widget().config(bd = 3, relief = tk.GROOVE)
        self.plane_counts_canvas.draw()
        self.plane_counts_canvas.get_tk_widget().grid(column = 0, row = 0, sticky = 'NSEW', padx = 5, pady = 5)

        self.plane_counts_canvas.mpl_connect('button_press_event', self.click)

        # Plot for a selected sinogram

        self.sinogram_fig = Figure(subplotpars = SubplotParams(0,0,1,1))
        self.sinogram_plt = self.sinogram_fig.add_subplot()
        self.sinogram_canvas = FigureCanvasTkAgg(
                self.sinogram_fig, master = self.plot_frame)
        self.sinogram_canvas.get_tk_widget().config(bd = 3, relief = tk.GROOVE)
        self.sinogram_canvas.draw()
        self.sinogram_canvas.get_tk_widget().grid(column = 1, row = 0, sticky = 'NSEW', padx = 5, pady = 5)

    def pack(self):
        self.button_frame.pack(pady = 30)

        self.save_lm.pack(side = tk.LEFT, padx = 5)
        self.load_lm.pack(side = tk.LEFT, padx = 5)
        self.save_sino.pack(side = tk.LEFT, padx = 5)
        self.load_sino.pack(side = tk.LEFT, padx = 5)
        self.multiply_button.pack(side = tk.LEFT, padx = 5)
        self.subtract_button.pack(side = tk.LEFT, padx = 5)

        self.cb_frame.pack(pady = 5)

        tk.Label(self.cb_frame, text = "Energy window: ").pack(side = tk.LEFT)
        self.energy_window_menu.pack(side = tk.LEFT, padx = 5)

        tk.Label(self.cb_frame, text = "Max DOI: ").pack(side = tk.LEFT)
        self.max_doi_menu.pack(side = tk.LEFT, padx = 5)

        self.sort_prompts_cb.pack(side = tk.LEFT, padx = 5)
        self.sort_delays_cb.pack(side = tk.LEFT, padx = 5)

        self.plot_frame.pack(fill = tk.BOTH, expand = True, padx = 5, pady = 5)

    def count_map_draw(self):
        self.plane_counts_plt.clear()
        self.sinogram_plt.clear()
        if self.sino_data is not None:
            self.plane_counts_plt.imshow(
                    self.sino_data.sum((2,3)),
                    aspect = 'auto')
            self.plane_counts_plt.invert_yaxis()
            self.plane_counts_canvas.draw()

    def click(self, ev):
        self.sinogram_plt.clear()
        h = int(np.floor(ev.xdata))
        v = int(np.ceil(ev.ydata))
        if self.sino_data is not None:
            print(f'row: {v} col: {h}')
            d = self.sino_data[v,h]
            self.sinogram_plt.imshow(d, aspect = 'auto',
                                     vmin = 0, vmax = np.quantile(d, 0.999))
            self.sinogram_plt.invert_yaxis()
            self.sinogram_canvas.draw()

    @staticmethod
    def remap_sinogram(s: np.ndarray):
        nr = s.shape[0]
        nt = s.shape[2]
        nt2 = int(nt/2)
        npos = s.shape[3]

        s0 = np.copy(s)
        sout = np.zeros((nr,nr,nt2,npos), s.dtype)

        for i in range(nr):
            for j in range(nr):
                if i <= j:
                    s[i,j] = np.roll(s[i,j], nt2, 0)
                    s[i,j] += np.fliplr(s0[j,i])

                    sout[i,j]  = s[i,j,0:nt2]
                    if i != j:
                        sout[j,i] += np.fliplr(s[i,j,nt2:nt])

        return sout

    def load_luts(self, calib_dir):
        lut_dim = [512,512]
        lut_arr = np.ones([petmr.nblocks] + lut_dim, dtype = np.intc) * petmr.ncrystals_total
        luts = glob.glob(os.path.join(calib_dir, 'lut', '*'))

        for fname in luts:
            # Lut fils are named .../blockXX.lut
            lut_idx = re.findall(r'\d+', os.path.basename(fname))
            lut_idx = int(lut_idx[0])
            lut_arr[lut_idx] = np.fromfile(fname, np.intc).reshape(lut_dim)

        return np.ascontiguousarray(lut_arr)

    def load_json_cfg(self, cfgdir):
        dims = (petmr.nblocks, petmr.ncrystals_total)
        ppeak = np.ones(dims, np.double) * -1;
        doi = np.ones(dims + (petmr.ndoi,), np.double) * -1;

        cfg_file = glob.glob(os.path.join(cfgdir,'*.json'))[0]
        with open(cfg_file, 'r') as f:
            cfg = json.load(f)

        for blk, bval in cfg.items():
            # use block photopeak as the default value
            ppeak[int(blk),:] = bval['photopeak']

            for xtal, xval in bval['crystal'].items():
                if int(xtal) >= petmr.ncrystals_total:
                    continue

                # apply crystal specific thresholds where available
                ppeak[int(blk), int(xtal)] = xval['energy']['photopeak']
                doi[int(blk), int(xtal),:] = xval['DOI']

        return ppeak, doi

    def save_listmode(self):
        coin_fnames = askopenfilenames(title = "Select 1+ coincidence files",
                                      filetypes = coincidence_filetypes)
        if not coin_fnames: return

        cfgdir = check_config_dir()
        if not cfgdir: return

        lm_fnames = askformatfilenames(coin_fnames,
                filetypes = listmode_filetypes)
        if not lm_fnames: return

        lut = self.load_luts(cfgdir)
        ppeak, doi = self.load_json_cfg(cfgdir)
        ewindow = self.energy_window_var.get()

        if len(coin_fnames) > 1:
            self.save_listmode_multi(coin_fnames, lm_fnames, ewindow, lut, ppeak, doi)
        else:
            SinogramLoaderPopup(
                    self, None, petmr.save_listmode,
                    lm_fnames[0], coin_fnames[0], ewindow, lut, ppeak, doi)

    def load_listmode(self):
        lm_fnames = askopenfilenames(title = "Select 1+ listmode file",
                                 filetypes = listmode_filetypes)
        if not lm_fnames: return

        prompts = self.sort_prompts_var.get()
        delays  = self.sort_delays_var.get()
        max_doi = self.max_doi_var.get()

        if len(lm_fnames) > 1:
            sino_fnames = askformatfilenames(lm_fnames,
                    filetypes = sinogram_filetypes)

            if not sino_fnames: return

            self.load_listmode_multi(lm_fnames, sino_fnames, prompts, delays, max_doi)

        else:
            def callback(sinogram):
                self.sino_data = self.remap_sinogram(sinogram)
                self.count_map_draw()

            SinogramLoaderPopup(
                    self, callback, petmr.load_listmode,
                    lm_fnames[0], prompts, delays, max_doi)

    def save_listmode_multi(self, coin_fnames, lm_fnames, *args):
        def launch():
            with concurrent.futures.ThreadPoolExecutor(4) as ex:
                for coin,lm in zip(coin_fnames, lm_fnames):
                    ex.submit(petmr.save_listmode, lm, coin, *args)
            print(f'Finished saving {len(coin_fnames)} listmode files')

        threading.Thread(target = launch).start()

    def load_listmode_multi(self, lm_fnames, sino_fnames, *args):
        def remap_and_save(fname, fut):
            s = self.remap_sinogram(fut.result())
            petmr.save_sinogram(fname, s)

        def launch():
            with concurrent.futures.ThreadPoolExecutor(4) as ex:
                for lm, sino in zip(lm_fnames, sino_fnames):
                    fut = ex.submit(petmr.load_listmode, lm, *args)
                    fut.add_done_callback(lambda f: remap_and_save(sino,f))
            print(f'Finished creating sinograms for {len(lm_fnames)} files')

        threading.Thread(target = launch).start()

    def load_sinogram(self):
        fname = askopenfilename(title = "Select sinogram file",
                                filetypes = [("Sinogram file", ".raw")])

        if fname:
            self.sino_data = petmr.load_sinogram(fname)
            self.count_map_draw()

    def save_sinogram(self):
        if self.sino_data is None: return

        fname = asksaveasfilename(title = "Save sinogram file",
                                  filetypes = [("Sinogram file", ".raw")])

        if not fname: return

        petmr.save_sinogram(fname, self.sino_data)

    def operation(self, op):
        if self.sino_data is None:
            return

        other_fname = askopenfilename(title = "Select sinogram for operation",
                                      filetypes = [("Sinogram file", ".raw")])

        if not other_fname:
            return

        other = petmr.load_sinogram(other_fname)
        op(self.sino_data, other, out = self.sino_data)
        np.nan_to_num(self.sino_data, copy = False,
                      nan = 1, posinf = 1, neginf = 1)
        self.count_map_draw()
