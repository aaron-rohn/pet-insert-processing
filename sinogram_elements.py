import os
import tkinter as tk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure, SubplotParams

from scipy import ndimage
import petmr
from data_loader import coincidence_filetypes
from sinogram_loader import SinogramLoaderPopup

class SinogramDisplay:
    def __init__(self, root):
        self.sino_data = None
        self.ldr = None
        self.root = root
        self.button_frame = tk.Frame(self.root)

        self.load_coin = tk.Button(self.button_frame, text = "Load Coincidences", command = self.sort_sinogram)
        self.load_sino = tk.Button(self.button_frame, text = "Load Sinogram", command = self.load_sinogram)
        self.save_sino = tk.Button(self.button_frame, text = "Save Sinogram", command = self.save_sinogram)
        self.save_lm   = tk.Button(self.button_frame, text = "Save Listmode", command = self.save_listmode)
        self.create_norm_button = tk.Button(self.button_frame, text = "Create Norm", command = self.create_norm)
        self.apply_norm_button = tk.Button(self.button_frame, text = "Apply Norm", command = self.apply_norm)

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
        self.apply_norm_button.pack(side = tk.LEFT, padx = 5)
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
        v = int(np.floor(ev.ydata))
        if self.sino_data is not None:
            print(f'row: {v} col: {h}')
            self.sinogram_plt.imshow(self.sino_data[v,h,:,:],
                                     aspect = 'auto')
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
                title = "Select coincidence file",
                initialdir = '/',
                filetypes = coincidence_filetypes)

        if not fname: return

        base = os.path.dirname(fname)
        cfgdir = tk.filedialog.askdirectory(
                title = "Select configuration directory",
                initialdir = base)

        if not cfgdir: return

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

        self.ldr = SinogramLoaderPopup(self.root,
                sorting_callback, fname, cfgdir)

    def load_sinogram(self):
        fname = tk.filedialog.askopenfilename(
                title = "Select sinogram file",
                initialdir = '/')

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

        """
        # Michelogram ordering parameters for interfile header
        nring = self.sino_data.shape[0]
        span_len = [nring] + list(np.repeat(np.arange(nring-1, 0, -1), 2))
        pos_rd = np.arange(1, nring)
        rd = np.zeros(pos_rd.size*2 + 1, int)
        rd[1::2] = pos_rd
        rd[2::2] = -pos_rd
        rd = list(rd)
        """

    def save_listmode(self):
        fname = tk.filedialog.askopenfilename(
                title = "Select coincidence file",
                initialdir = '/',
                filetypes = coincidence_filetypes)
        if not fname: return
        base = os.path.dirname(fname)

        cfgdir = tk.filedialog.askdirectory(
                title = "Select configuration directory",
                initialdir = base)
        if not cfgdir: return

        lmfname = tk.filedialog.asksaveasfilename(
                title = "Save listmode file",
                initialdir = base,
                filetypes = [("Listmode file", ".lm")])
        if not lmfname: return
        
        petmr.save_listmode(fname, lmfname, cfgdir)

    def create_norm(self):
        fnames = tk.filedialog.askopenfilenames(
                title = "Select sinogram file",
                initialdir = '/',
                filetypes = [("Sinogram file", ".raw")])

        if not fnames: return

        # sum the provided sinograms
        sinogram = None
        for fn in fnames:
            s = petmr.load_sinogram(fn)

            if sinogram is None:
                sinogram = s
            else:
                sinogram += s

        # average over angular dimension
        proj = sinogram.mean(2)
        sinogram = proj[:,:,None,:] / sinogram
        self.sino_data = np.nan_to_num(sinogram,
                nan = 1, posinf = 1, neginf = 1)
        self.count_map_draw()

    def apply_norm(self):
        fname = tk.filedialog.askopenfilename(
                title = "Select sinogram file",
                initialdir = '/',
                filetypes = [("Sinogram file", ".raw")])

        if not fname: return

        base = os.path.dirname(fname)
        norm = tk.filedialog.askopenfilename(
                title = "Select normalization sinogram",
                initialdir = base,
                filetypes = [("Sinogram file", ".raw")])

        if not norm: return

        # load sinogram and norm
        self.sino_data = petmr.load_sinogram(fname)
        norm_sino      = petmr.load_sinogram(norm)

        # divide sinogram by norm, suppress invalid values
        self.sino_data *= norm_sino
        self.sino_data = np.nan_to_num(self.sino_data,
                nan = 1, posinf = 1, neginf = 1)

        self.count_map_draw()
