import os
import json
import copy
import matplotlib
import threading
import numpy as np
import tkinter as tk

from scipy import ndimage
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure, SubplotParams

from flood import Flood, PerspectiveTransformDialog
from data_loader import ProgressPopup, coincidence_filetypes
import crystal
from calibration import create_cfg_vals, lut_edges, img_encode
from filedialog import check_config, asksaveasfilename

def try_open(fname: str) -> dict:
    try:
        with open(fname, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.decoder.JSONDecodeError): return {}

def get_marker_size(widget):
    min_window_size = min(widget.winfo_height(), widget.winfo_width())
    return (1 if min_window_size < 300 else
            2 if min_window_size < 500 else
            3 if min_window_size < 600 else 4)

class MPLFigure(FigureCanvasTkAgg):
    def __init__(self, root, show_axes = True, **args):
        margins = (0.1, 0.1, 0.95, 0.95) if show_axes else (0,0,1,1)
        self.fig = Figure(subplotpars = SubplotParams(*margins))
        self.plot = self.fig.add_subplot(frame_on = show_axes)

        super().__init__(self.fig, master = root)
        self.widget = self.get_tk_widget()
        self.widget.config(bd = 3, relief = tk.GROOVE)
        self.widget.grid(**args)
        self.draw()

        self.cmap = copy.copy(matplotlib.colormaps['Pastel1'])
        self.cmap.set_bad(alpha = 0)

class FloodHist(MPLFigure):
    def __init__(self, root, **args):
        self.img_size = 512
        self.npts = 19
        self.img = np.zeros((self.img_size,self.img_size))
        self.f = None
        self.selection = None
        self.pts = None
        self.pts_active = None
        self.overlay = None
        self.draw_voronoi = False
        self.draw_points = True

        super().__init__(root, show_axes = False, **args)
        self.mpl_connect('button_press_event', self.click)

    def click(self, ev):
        if self.pts is None: return

        loc = np.array([ev.xdata, ev.ydata])
        self.pts_active = np.zeros((self.npts, self.npts), dtype = bool)

        if self.selection is None:
            dst = np.linalg.norm(self.pts - loc, axis = 2)
            self.selection = np.unravel_index(np.argmin(dst), dst.shape)
            self.pts_active[self.selection[0],:] = True
            self.pts_active[:,self.selection[1]] = True
        else:
            self.pts[self.selection[0],:,1] = loc[1]
            self.pts[:,self.selection[1],0] = loc[0]
            self.pts.sort(0)
            self.pts.sort(1)
            self.selection = None

        self.redraw()

    def redraw(self):
        """
        Coordinate layout:
        
        'Animal' side
    x0y1         x1y1
        #########
        #D     A#
        #   0   #
        #C     B#
        #########
    x0y0         x1y0
        #########
        #       #
        #   1   #
        #       #
        #########
        #########
        #       #
        #   2   #
        #       #
        #########
        #########
        #       #
        #   4   #
        #       #
        #########
        'Cabinet' side

        View from rear SiPM array, looking inward
        System is inserted through MRI cabinet in rear

        """

        self.plot.clear()
        self.plot.imshow(self.f.fld, aspect = 'auto')

        if self.overlay is not None:
            self.plot.imshow(self.overlay, aspect = 'auto', cmap = self.cmap)

        if self.draw_voronoi:
            vor = Voronoi(self.pts.reshape(-1,2))
            voronoi_plot_2d(vor, ax = self.plot,
                    show_vertices = False,
                    show_points = False,
                    line_colors = 'grey',
                    line_alpha = 0.5)

        if self.draw_points:
            active = self.pts[self.pts_active].T
            inactive = self.pts[~self.pts_active].T
            ms = get_marker_size(self.widget)
            self.plot.plot(*active, '.b', *inactive, '.r', ms = ms)

        # Invert Y axis to display A channel in Top right
        self.plot.invert_yaxis()
        self.plot.set_xlim(0,self.img_size-1)
        self.plot.set_ylim(0,self.img_size-1)
        self.draw()
    
    def register(self):
        """ Apply a deformable 2D registration to the current point set
        based on the loaded flood.
        """
        if self.pts is not None:
            self.pts = self.pts.reshape(self.npts*self.npts, 2)
            self.pts = self.f.register_peaks(self.pts)
            self.pts = self.pts.reshape(self.npts, self.npts, 2)
            self.redraw()

    def update(self, x, y, warp = None, overlay = None,
               draw_voronoi = False, draw_points = True,
               preprocess = True):

        self.draw_voronoi = draw_voronoi
        self.draw_points = draw_points
        self.overlay = overlay

        # First coord -> rows -> y
        # Second coord -> cols -> x
        self.img, *_ = np.histogram2d(y, x, bins = self.img_size,
                range = [[0,self.img_size-1],[0,self.img_size-1]])

        self.f = Flood(self.img, warp, preprocess)

        if draw_voronoi or draw_points:
            self.pts = self.f.estimate_peaks()
        else:
            self.pts = None

        if self.pts is not None:
            self.pts = self.pts.T.reshape(self.npts, self.npts, 2)

        self.pts_active = np.zeros((self.npts, self.npts), dtype = bool)
        self.redraw()

class ThresholdHist(MPLFigure):
    def __init__(self, root, is_energy, **args):
        super().__init__(root, **args)
        self.init_lines()
        self.is_energy = is_energy
        self.e_window = 0.2
        self.peak = 0
        self.connection = None
        self.active_line = None
        self.callback = []

        self.mpl_connect('button_press_event', self.drag_start)
        self.mpl_connect('button_release_event', self.drag_stop)

    def init_lines(self, lims = (0,1)):
        self.lines = [self.plot.axvline(x,linewidth=3,color='r') for x in lims]

    def update(self, data, retain = False):
        last_rng = self.thresholds()
        self.plot.clear()
        rng = np.quantile(data, [0, 0.97 if self.is_energy else 1])
        nbins = int(round((rng[1] - rng[0]) / 10))
        n,bins,_ = self.plot.hist(data, bins = nbins, range = rng)

        self.counts = n
        self.bins = bins

        # search for the photopeak
        self.peak = (
                bins[np.argmax(bins[:-1] * n**2)] if self.is_energy
                else bins[np.argmax(n)])

        if retain:
            rng = last_rng
        elif self.is_energy:
            # Energy histogram - take % window around peak
            rng = [(1-self.e_window)*self.peak, (1+self.e_window)*self.peak]
        else:
            # DOI histogram
            rng = np.quantile(data, [0.20, 0.99])

        self.init_lines(rng)
        self.plot.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
        self.draw()

    def thresholds(self):
        return list(np.sort([l.get_xdata()[0] for l in self.lines]))

    def drag_start(self, ev):
        if ev.xdata is not None:
            self.connection = self.mpl_connect('motion_notify_event', self.cursor_set)

            # set the nearest line as active
            xpos = np.array([l.get_xdata()[0] for l in self.lines])
            idx = np.argmin(np.abs(ev.xdata - xpos))
            self.active_line = self.lines[idx]

    def drag_stop(self, ev):
        if self.connection is not None:
            self.mpl_disconnect(self.connection)
            self.connection = None
            self.active_line = None
            [cb() for cb in self.callback]

    def cursor_set(self, ev):
        if ev.xdata is not None and self.active_line is not None:
            self.active_line.set_xdata([ev.xdata]*2)
            self.draw()

class Plots(tk.Frame):
    def __init__(self, root, get_data, get_block, incr_block, **args):
        super().__init__(root)
        self.pack(**args)
        self.get_data = get_data 
        self.get_block = get_block 
        self.incr_block = incr_block

        self.d = None
        self.warp = None

        # Flood operation buttons

        self.button_frame = tk.Frame(self)

        self.select_cfg_button = tk.Button(self.button_frame,
                text = "Select Config File", command = lambda: check_config(True))

        self.store_lut_button = tk.Button(self.button_frame,
                text = "Store Configuration", command = self.store_lut_cb)

        self.register_button = tk.Button(self.button_frame, text = "Register Peaks")

        self.transform_button = tk.Button(self.button_frame,
                text = "Perspective Transform", command = self.perspective_transform)

        self.save_flood_button = tk.Button(self.button_frame,
                text = "Save Flood", command = self.save_flood)

        self.preprocess_flood = tk.BooleanVar(value = True)
        self.preprocess_flood_cb = tk.Checkbutton(self.button_frame,
                text = "Preprocess Flood", variable = self.preprocess_flood,
                command = self.flood_cb)

        self.show_points = tk.BooleanVar(value = True)
        self.show_points_cb = tk.Checkbutton(self.button_frame,
                text = "Overlay Points", variable = self.show_points,
                command = self.flood_cb)

        self.show_voronoi = tk.BooleanVar(value = False)
        self.show_voronoi_cb = tk.Checkbutton(self.button_frame,
                text = "Overlay Voronoi", variable = self.show_voronoi,
                command = self.flood_cb)

        self.show_lut = tk.BooleanVar(value = False)
        self.show_lut_cb = tk.Checkbutton(self.button_frame,
                text = "Overlay LUT", variable = self.show_lut,
                command = self.flood_cb)

        self.button_frame.pack(pady = 10);
        self.select_cfg_button.pack(side = tk.LEFT, padx = 5)
        self.store_lut_button.pack(side = tk.LEFT, padx = 5)
        self.register_button.pack(side = tk.LEFT, padx = 5)
        self.transform_button.pack(side = tk.LEFT, padx = 5)
        self.save_flood_button.pack(side = tk.LEFT, padx = 5)

        self.preprocess_flood_cb.pack(side = tk.LEFT, padx = 5)
        self.show_points_cb.pack(side = tk.LEFT, padx = 5)
        self.show_voronoi_cb.pack(side = tk.LEFT, padx = 5)
        self.show_lut_cb.pack(side = tk.LEFT, padx = 5)

        # Flood, energy and DOI plots

        frm = tk.Frame(self)
        frm.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)
        args = {'row': 0, 'columnspan': 1, 'sticky': tk.NSEW}
        self.flood  = FloodHist(frm, column = 0, **args)
        self.energy = ThresholdHist(frm, is_energy = True, column = 1, **args)
        self.doi    = ThresholdHist(frm, is_energy = False, column = 2, **args)
        frm.columnconfigure((0,1,2), weight = 1, uniform = 'column')
        frm.rowconfigure(0, weight = 1, uniform = 'row')

        self.register_button.config(command = self.flood.register)

        # when one plot is updated, update other plots accordingly
        # callbacks are triggered after a threshold drag is finished
        self.energy.callback.append(self.flood_cb)
        self.energy.callback.append(self.doi_cb)
        self.doi.callback.append(self.flood_cb)
        self.doi.callback.append(self.energy_cb)

    def save_flood(self):
        fname = asksaveasfilename(
                title = "Save flood to file",
                filetypes = [("Raw image", ".raw")])

        if self.flood.f is not None and fname is not None:
            # image is 512x512 float64
            self.flood.f.fld.tofile(fname)

    def perspective_transform(self):
        def callback(mat):
            self.warp = mat
            self.flood_cb()
        PerspectiveTransformDialog(self, self.flood.f.fld, callback)

    def plots_update(self, *args):
        """ Update all plots when new data is available """
        self.warp = None
        self.d = self.get_data()

        self.energy.update(self.d['E'], retain = False)
        self.doi_cb(retain = False)
        self.flood_cb()

    def create_lut_borders(self, cfg_file):
        with open(cfg_file, 'r') as fd:
            cfg = json.load(fd)

        if (blk := str(self.get_block())) in cfg:
            lut = img_encode(cfg[blk]['LUT'], False)
            return lut_edges(lut)
        else:
            return None

    def flood_cb(self):
        """ Update the flood according to energy and DOI thresholds """
        if self.d is None: return

        lut = None
        if self.show_lut.get() and (cfg_file := check_config()):
            lut = self.create_lut_borders(cfg_file)

        eth = self.energy.thresholds()
        dth = self.doi.thresholds()

        es = self.d['E']
        doi = self.d['D']

        idx = (eth[0] < es) & (es < eth[1]) & (dth[0] < doi) & (doi < dth[1])
        windowed = self.d[idx]
        self.flood.update(windowed['X'], windowed['Y'],
                          warp = self.warp, overlay = lut,
                          draw_voronoi = self.show_voronoi.get(),
                          draw_points = self.show_points.get(),
                          preprocess = self.preprocess_flood.get())

    def doi_cb(self, retain = True):
        """ Update the DOI according to the energy thresholds """
        eth = self.energy.thresholds()
        es = self.d['E']
        idx = np.nonzero((eth[0] < es) & (es < eth[1]))[0]
        self.doi.update(self.d['D'][idx], retain)

    def energy_cb(self, retain = True):
        """ Update the energy according to the DOI thresholds """
        dth = self.doi.thresholds()
        doi = self.d['D']
        idx = np.nonzero((dth[0] < doi) & (doi < dth[1]))[0]
        self.energy.update(self.d['E'][idx], retain)

    def store_lut_cb(self):
        if (config_file := check_config()) is None or self.flood.f is None:
            return

        # calculate the LUT based on the peak positions
        peak_pos = self.flood.pts.reshape(-1,2)
        lut = Flood.nearest_peak((self.flood.img_size,)*2, peak_pos)

        # invert any perspective transform, if necessary
        if self.warp is not None:
            lut = Flood.warp_lut(lut, self.warp)
            self.warp = None

        self.store_lut_button.config(state = tk.DISABLED)

        # create energy and DOI calibrations
        cfg = try_open(config_file)
        blk = self.get_block()
        blk_vals = cfg[blk] = {}
        blk_vals['LUT'] = img_encode(lut)
        blk_vals['FLD'] = img_encode(self.flood.f.fld, jpeg = True)

        t = threading.Thread(target = create_cfg_vals,
                args = [self.d, lut, blk, cfg])
        t.start()

        def finish():
            self.store_lut_button.config(state = tk.NORMAL)

            with open(config_file, 'w') as f:
                json.dump(cfg, f)

            try:
                self.incr_block()
            except tk.TclError:
                tk.messagebox.showinfo('Completed', 'Calibration complete for last block')

        def check():
            self.after(100, check) if t.is_alive() else finish()

        check()
