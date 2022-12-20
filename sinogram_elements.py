import os, glob, re, json
import tkinter as tk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure, SubplotParams
import matplotlib.pyplot as plt

from scipy import ndimage
import petmr, calibration
from data_loader import coincidence_filetypes, listmode_filetypes
from sinogram_loader import SinogramLoaderPopup

class SinogramDisplay:
    def __init__(self, root):
        self.sino_data = None
        self.ldr = None
        self.root = root
        self.button_frame = tk.Frame(self.root)

        self.save_lm   = tk.Button(self.button_frame, text = "Save Listmode", command = self.save_listmode)
        self.load_lm = tk.Button(self.button_frame, text = "Load Listmode", command = self.load_listmode)
        self.save_sino = tk.Button(self.button_frame, text = "Save Sinogram", command = self.save_sinogram)
        self.load_sino = tk.Button(self.button_frame, text = "Load Sinogram", command = self.load_sinogram)
        self.create_norm_button = tk.Button(self.button_frame, text = "Create Norm", command = self.create_norm)
        self.multiply_button = tk.Button(self.button_frame, text = "Multiply", command = lambda: self.operation(np.multiply))
        self.subtract_button = tk.Button(self.button_frame, text = "Subtract", command = lambda: self.operation(np.subtract))

        self.sort_prompts_var = tk.IntVar()
        self.sort_delays_var = tk.IntVar()
        self.cb_frame = tk.Frame(self.root)
        self.sort_prompts_cb = tk.Checkbutton(self.cb_frame, text = "Prompts", variable = self.sort_prompts_var)
        self.sort_delays_cb = tk.Checkbutton(self.cb_frame, text = "Delays", variable = self.sort_delays_var)

        self.plot_frame = tk.Frame(self.root)
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
        self.create_norm_button.pack(side = tk.LEFT, padx = 5)
        self.multiply_button.pack(side = tk.LEFT, padx = 5)
        self.subtract_button.pack(side = tk.LEFT, padx = 5)

        self.cb_frame.pack(pady = 5)
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

    def remap_sinogram(self, s):
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

    def find_calib_dirs(self, cfgdir, coincidence_file):
        """ For a coincidence file, identify the count-rate configuration directory
        that is the closest match for each included time period. Return a list of the
        calibration directories and corresponding file offsets
        """

        # find the count rates with existing calibrations in the cfg dir
        dirs = glob.glob(os.path.join(cfgdir, '*'))
        dirs  = np.array([d for d in dirs if os.path.basename(d).isnumeric()])

        if len(dirs) == 0:
            print('No event rate directories found, using default')
            return [os.path.join(cfgdir, 'default')], np.array([0], np.ulonglong)

        rates = np.array([int(os.path.basename(d)) for d in dirs])
        order = np.argsort(rates)

        dirs = dirs[order]
        rates = rates[order]

        cf = calibration.CoincidenceFileHandle(coincidence_file)

        # find the nearest corresponding count rate in the calibration data set
        nearest = lambda vals, v: np.abs(vals - v).argmin()
        calib_idx = np.array([nearest(rates, rt) for rt in cf.event_rate])
        dirs = dirs[calib_idx]

        d, f = [], []
        for di, fi in zip(dirs, cf.file_position()):
            if di not in d:
                d.append(di)
                f.append(fi)

        return d, np.array(f)

    def load_luts(self, calib_dirs):
        lut_dim = [512,512]
        lut_arr = np.ones([petmr.nblocks, len(calib_dirs)] + lut_dim, dtype = np.intc) * petmr.ncrystals_total

        for i, d in enumerate(calib_dirs):
            luts = glob.glob(os.path.join(d, 'lut', '*'))

            for fname in luts:
                # Lut fils are named .../blockXX.lut
                lut_idx = re.findall(r'\d+', os.path.basename(fname))
                lut_idx = int(lut_idx[0])
                lut_arr[lut_idx, i] = np.fromfile(fname, np.intc).reshape(lut_dim)

        return np.ascontiguousarray(lut_arr)

    def load_json_cfg(self, calib_dirs):
        dims = (petmr.nblocks, len(calib_dirs), petmr.ncrystals_total)
        ppeak = np.ones(dims, np.double) * -1;
        doi = np.ones(dims + (petmr.ndoi,), np.double) * -1;

        for i, d in enumerate(calib_dirs):
            cfg_file = glob.glob(os.path.join(d,'*.json'))[0]
            with open(cfg_file, 'r') as f:
                cfg = json.load(f)

            for blk, bval in cfg.items():
                ppeak[int(blk),i,:] = bval['photopeak']

                for xtal, xval in bval['crystal'].items():
                    if int(xtal) >= petmr.ncrystals_total:
                        continue

                    ppeak[int(blk), i, int(xtal)] = xval['energy']['photopeak']
                    doi[int(blk), i, int(xtal),:] = xval['DOI']

        return ppeak, doi

    def save_listmode(self):
        coin_fname = tk.filedialog.askopenfilename(
                title = "Select coincidence file",
                initialdir = '/',
                filetypes = coincidence_filetypes)
        if not coin_fname: return
        base = os.path.dirname(coin_fname)
        parent = os.path.dirname(base)

        cfgdir = tk.filedialog.askdirectory(
                title = "Select configuration directory",
                initialdir = parent)
        if not cfgdir: return

        lmfname = tk.filedialog.asksaveasfilename(
                title = "Save listmode file",
                initialdir = parent,
                filetypes = listmode_filetypes)
        if not lmfname: return

        calib_dirs, fpos = self.find_calib_dirs(cfgdir, coin_fname)
        lut = self.load_luts(calib_dirs)
        ppeak, doi = self.load_json_cfg(calib_dirs)

        self.ldr = SinogramLoaderPopup(
                self.root, None, petmr.save_listmode,
                lmfname, coin_fname, cfgdir,
                lut, fpos, ppeak, doi)

    def load_listmode(self):
        fname = tk.filedialog.askopenfilename(
                title = "Select listmode file",
                initialdir = '/',
                filetypes = listmode_filetypes)

        if not fname: return

        def sorting_callback(result):
            if isinstance(result, RuntimeError):
                self.sino_data = None
                tk.messagebox.showerror(message =
                        f'{type(result).__name__}: {" ".join(result.args)}')
            else:
                result = self.remap_sinogram(result)
                self.sino_data = result

            self.count_map_draw()
            self.ldr = None

        max_doi = 0
        self.ldr = SinogramLoaderPopup(
                self.root, sorting_callback, petmr.sort_sinogram,
                fname, self.sort_prompts_var.get(), self.sort_delays_var.get(), max_doi)

    def load_sinogram(self):
        fname = tk.filedialog.askopenfilename(
                title = "Select sinogram file",
                initialdir = '/',
                filetypes = [("Sinogram file", ".raw")])

        if fname:
            self.sino_data = petmr.load_sinogram(fname)
            self.count_map_draw()

    def save_sinogram(self):
        if self.sino_data is None: return

        fname = tk.filedialog.asksaveasfilename(
                title = "Save sinogram file",
                initialdir = '/',
                filetypes = [("Sinogram file", ".raw")])

        if not fname: return

        petmr.save_sinogram(fname, self.sino_data)

    def create_norm(self):
        if self.sino_data is None:
            return

        # average over angular dimension
        proj = self.sino_data.mean((0,1,2))

        np.divide(proj[None,None,None,:], self.sino_data,
                  out = self.sino_data)
        np.nan_to_num(self.sino_data, copy = False,
                      nan = 1, posinf = 1, neginf = 1)
        self.count_map_draw()

    def operation(self, op):
        if self.sino_data is None:
            return

        other_fname = tk.filedialog.askopenfilename(
                title = "Select sinogram for operation",
                initialdir = '/',
                filetypes = [("Sinogram file", ".raw")])

        if not other_fname:
            return

        other = petmr.load_sinogram(other_fname)
        op(self.sino_data, other, out = self.sino_data)
        np.nan_to_num(self.sino_data, copy = False,
                      nan = 1, posinf = 1, neginf = 1)
        self.count_map_draw()
