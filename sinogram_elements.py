import os, glob, re
import cv2 as cv
import tkinter as tk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure, SubplotParams
import matplotlib.pyplot as plt

from scipy import ndimage
import petmr
from data_loader import coincidence_filetypes, listmode_filetypes, read_times
from sinogram_loader import SinogramLoaderPopup

class SinogramDisplay:
    def __init__(self, root):
        self.sino_data = None
        self.ldr = None
        self.root = root
        self.button_frame = tk.Frame(self.root)

        self.load_coin = tk.Button(self.button_frame, text = "Load Listmode", command = self.sort_sinogram)
        self.load_sino = tk.Button(self.button_frame, text = "Load Sinogram", command = self.load_sinogram)
        self.save_sino = tk.Button(self.button_frame, text = "Save Sinogram", command = self.save_sinogram)
        self.save_lm   = tk.Button(self.button_frame, text = "Save Listmode", command = self.save_listmode)
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
        self.load_coin.pack(side = tk.LEFT, padx = 5)
        self.load_sino.pack(side = tk.LEFT, padx = 5)
        self.save_sino.pack(side = tk.LEFT, padx = 5)
        self.save_lm.pack(side = tk.LEFT, padx = 5)
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

    def sort_sinogram(self):
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

        self.ldr = SinogramLoaderPopup(
                self.root, sorting_callback, petmr.sort_sinogram,
                fname, self.sort_prompts_var.get(), self.sort_delays_var.get())

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

        scaling, times, fpos = read_times(coin_fname, 100)

        # Load and rescale the LUTs
        luts = sorted(glob.glob(f'{cfgdir}/*.lut'))
        lut_arr = np.ones((64, len(scaling), 512, 512), dtype = np.intc) * (19*19)
        for fname in luts:
            lut_idx = re.findall(r'\d+', os.path.basename(fname))
            lut_idx = int(lut_idx[0])
            lut = np.fromfile(fname, np.intc).reshape((512,512))
            for i, scale in enumerate(scaling):
                lut_scale = cv.resize(lut, None, fx = scale, fy = scale,
                                      interpolation = cv.INTER_NEAREST)
                nr, nc = [int(rc/2) for rc in lut_scale.shape]
                lut_arr[lut_idx,i] = lut_scale[nr-256:nr+256, nc-256:nc+256]

        print(lut_arr.shape)

        self.ldr = SinogramLoaderPopup(
                self.root, None, petmr.save_listmode,
                lmfname, coin_fname, cfgdir, np.ascontiguousarray(lut_arr), fpos)

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
