import os
import tkinter as tk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure, SubplotParams

import petmr
from data_loader import coincidence_filetypes

class SinogramDisplay:
    def __init__(self, root):
        self.sino_data = None
        self.root = root
        self.button_frame = tk.Frame(self.root)

        self.load_coin = tk.Button(self.button_frame, text = "Load Coincidences", command = self.sort_sinogram)
        self.load_sino = tk.Button(self.button_frame, text = "Load Sinogram", command = self.load_sinogram)
        self.save_sino = tk.Button(self.button_frame, text = "Save Sinogram", command = self.save_sinogram)

        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.columnconfigure(0, weight = 1)
        self.plot_frame.columnconfigure(1, weight = 1)

        # Plot for counts in each pair of system planes

        self.plane_counts_fig = Figure(subplotpars = SubplotParams(0,0,1,1))
        self.plane_counts_plt = self.plane_counts_fig.add_subplot()
        self.plane_counts_canvas = FigureCanvasTkAgg(
                self.plane_counts_fig, master = self.plot_frame)
        self.plane_counts_canvas.draw()
        self.plane_counts_canvas.get_tk_widget().grid(column = 0, row = 0, sticky = 'NSEW')

        self.plane_counts_canvas.mpl_connect('button_press_event', self.click)

        # Plot for a selected sinogram

        self.sinogram_fig = Figure(subplotpars = SubplotParams(0,0,1,1))
        self.sinogram_plt = self.sinogram_fig.add_subplot()
        self.sinogram_canvas = FigureCanvasTkAgg(
                self.sinogram_fig, master = self.plot_frame)
        self.sinogram_canvas.draw()
        self.sinogram_canvas.get_tk_widget().grid(column = 1, row = 0, sticky = 'NSEW')

    def pack(self):
        self.button_frame.pack(pady = 10)
        self.load_coin.pack(side = tk.LEFT, padx = 5)
        self.load_sino.pack(side = tk.LEFT, padx = 5)
        self.save_sino.pack(side = tk.LEFT, padx = 5)

        self.plot_frame.pack()

    def count_map_draw(self):
        self.plane_counts_plt.clear()
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
            self.sinogram_plt.imshow(self.sino_data[v,h,:,:],
                                     aspect = 'auto')
            self.sinogram_plt.invert_yaxis()
            self.sinogram_canvas.draw()

    def sort_sinogram(self):
        fname = tk.filedialog.askopenfilename(
                title = "Select coincidence file",
                initialdir = os.path.expanduser('~'),
                filetypes = coincidence_filetypes)

        base = os.path.dirname(fname)
        cfgdir = tk.filedialog.askdirectory(
                title = "Select configuration directory",
                initialdir = base)

        if fname and cfgdir:
            self.sino_data = petmr.sort_sinogram(fname, cfgdir)
            self.count_map_draw()

    def load_sinogram(self):
        fname = tk.filedialog.askopenfilename(
                title = "Select sinogram file",
                initialdir = os.path.expanduser('~'))

        if fname:
            self.sino_data = petmr.load_sinogram(fname)
            self.count_map_draw()

    def save_sinogram(self):
        if self.sino_data is not None:
            fname = tk.filedialog.asksaveasfilename(
                    title = "Save sinogram file",
                    initialdir = os.path.expanduser('~'),
                    filetypes = [("Sinogram file", ".raw")])
            petmr.save_sinogram(fname, self.sino_data)
