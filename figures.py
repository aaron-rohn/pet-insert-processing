import os, glob, json, copy, matplotlib, threading, queue
import numpy as np
import tkinter as tk
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy import ndimage
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure, SubplotParams

from flood import Flood, PerspectiveTransformDialog
from data_loader import ProgressPopup, coincidence_filetypes
import crystal, calibration

class MPLFigure:
    def __init__(self, root, show_axes = True, **pack_args):
        if show_axes:
            self.fig = Figure()
        else:
            self.fig = Figure(subplotpars = SubplotParams(0,0,1,1))

        self.plot = self.fig.add_subplot(frame_on = show_axes)
        self.canvas = FigureCanvasTkAgg(self.fig, master = root)
        self.canvas.get_tk_widget().config(bd = 3, relief = tk.GROOVE)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(**pack_args)

class FloodHist(MPLFigure):
    def __init__(self, root, **pack_args):
        super().__init__(root, show_axes = False, **pack_args)

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

        self.canvas.mpl_connect('button_press_event', self.click)

        self.cmap = copy.copy(matplotlib.colormaps['Pastel1'])
        self.cmap.set_bad(alpha = 0)

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

        if self.pts is not None:
            active = self.pts[self.pts_active].T
            inactive = self.pts[~self.pts_active].T

            if self.draw_voronoi:
                vor = Voronoi(self.pts.reshape(-1,2))
                voronoi_plot_2d(vor, ax = self.plot, 
                        show_vertices = False, 
                        show_points = False, 
                        line_colors = 'grey',
                        line_alpha = 0.5)

            if self.draw_points:
                self.plot.plot(*active, '.b', *inactive, '.r', ms = 4)

        # Invert Y axis to display A channel in Top right
        self.plot.invert_yaxis()
        self.plot.set_xlim(0,self.img_size-1)
        self.plot.set_ylim(0,self.img_size-1)
        self.canvas.draw()
    
    def register(self):
        """ Apply a deformable 2D registration to the current point set
        based on the loaded flood.
        """
        if self.pts is None:
            return

        self.pts = self.pts.reshape(self.npts*self.npts, 2)
        self.pts = self.f.register_peaks(self.pts)
        self.pts = self.pts.reshape(self.npts, self.npts, 2)
        self.redraw()

    def update(self, x, y, warp = None, overlay = None,
               draw_voronoi = False, draw_points = True):

        # First coord -> rows -> y
        # Second coord -> cols -> x
        self.img, *_ = np.histogram2d(y, x, bins = self.img_size,
                range = [[0,self.img_size-1],[0,self.img_size-1]])

        self.f = Flood(self.img, warp)

        try:
            self.pts = self.f.estimate_peaks()
            self.pts = self.pts.T.reshape(self.npts, self.npts, 2)
        except RuntimeError as e:
            # Failed to find sufficient peaks
            print(repr(e))
            self.pts = None

        self.pts_active = np.zeros((self.npts, self.npts), dtype = bool)
        self.draw_voronoi = draw_voronoi
        self.draw_points = draw_points
        self.overlay = overlay
        self.redraw()

class ThresholdHist(MPLFigure):
    def __init__(self, root, is_energy, **pack_args):
        super().__init__(root, **pack_args)
        self.init_lines()
        self.is_energy = is_energy
        self.e_window = 0.2
        self.peak = 0
        self.connection = None
        self.active_line = None
        self.callback = []

        self.canvas.mpl_connect('button_press_event', self.drag_start)
        self.canvas.mpl_connect('button_release_event', self.drag_stop)

    def init_lines(self, lims = (0,1)):
        self.lines = [self.plot.axvline(x,linewidth=3,color='r') for x in lims]

    def update(self, data, retain = False):
        last_rng = self.thresholds()
        self.plot.clear()
        rng = np.quantile(data, [0.01, 0.99])
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
        self.canvas.draw()

    def thresholds(self):
        return list(np.sort([l.get_xdata()[0] for l in self.lines]))

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
            [cb() for cb in self.callback]

    def cursor_set(self, ev):
        if ev.xdata is not None and self.active_line is not None:
            self.active_line.set_xdata([ev.xdata]*2)
            self.canvas.draw()

class Plots(tk.Frame):
    def __init__(self, root, return_data, return_block, set_block, **pack_args):
        super().__init__(root)
        self.pack(**pack_args)
        self.return_data = return_data
        self.return_block = return_block
        self.set_block = set_block

        self.d = None
        self.output_dir = None
        self.warp = None

        # Flood operation buttons

        self.button_frame = tk.Frame(self)

        self.select_dir_button = tk.Button(self.button_frame,
                text = "Select Directory", command = lambda: self.check_output_dir(True))

        self.store_lut_button = tk.Button(self.button_frame,
                text = "Store Configuration", command = self.store_lut_cb)

        #self.create_scaled_config = tk.Button(self.button_frame, text = "Scale Configuration", command = self.scale_config)

        self.register_button = tk.Button(self.button_frame, text = "Register Peaks")

        self.transform_button = tk.Button(self.button_frame,
                text = "Perspective Transform", command = self.perspective_transform)

        self.show_points = tk.IntVar(value = 1)
        self.show_points_cb = tk.Checkbutton(self.button_frame,
                text = "Overlay Points", variable = self.show_points)

        self.show_voronoi = tk.IntVar()
        self.show_voronoi_cb = tk.Checkbutton(self.button_frame,
                text = "Overlay Voronoi", variable = self.show_voronoi)

        self.show_lut = tk.IntVar()
        self.show_lut_cb = tk.Checkbutton(self.button_frame,
                text = "Overlay LUT", variable = self.show_lut)

        self.button_frame.pack(pady = 10);
        self.select_dir_button.pack(side = tk.LEFT, padx = 5)
        self.store_lut_button.pack(side = tk.LEFT, padx = 5)
        #self.create_scaled_config.pack(side = tk.LEFT, padx = 5)
        self.register_button.pack(side = tk.LEFT, padx = 5)
        self.transform_button.pack(side = tk.LEFT, padx = 5)
        self.show_points_cb.pack(side = tk.LEFT, padx = 5)
        self.show_voronoi_cb.pack(side = tk.LEFT, padx = 5)
        self.show_lut_cb.pack(side = tk.LEFT, padx = 5)

        # Flood, energy and DOI plots

        args = {'fill': tk.BOTH, 'expand': True, 'side': tk.LEFT}
        self.flood  = FloodHist(self, **args)
        self.energy = ThresholdHist(self, is_energy = True, **args)
        self.doi    = ThresholdHist(self, is_energy = False, **args)
        self.register_button.config(command = self.flood.register)

        # when one plot is updated, update other plots accordingly
        # callbacks are triggered after a threshold drag is finished
        self.energy.callback.append(self.flood_cb)
        self.energy.callback.append(self.doi_cb)
        self.doi.callback.append(self.flood_cb)
        self.doi.callback.append(self.energy_cb)

    def perspective_transform(self):
        def callback(mat):
            self.warp = mat
            self.flood_cb()
        PerspectiveTransformDialog(self, self.flood.f.fld, callback)

    def plots_update(self, *args):
        """ Update all plots when new data is available """
        self.warp = None
        blk = self.return_block()
        self.d, self.ev_rate = self.return_data(blk)

        self.energy.update(self.d[:,0], retain = False)
        self.doi_cb(retain = False)
        self.flood_cb()

    def create_lut_borders(self):
        blk = self.return_block()

        dirs = glob.glob(os.path.join(self.output_dir, '*'))
        dirs  = np.array([d for d in dirs if os.path.basename(d).isnumeric()])
        rates = np.array([int(os.path.basename(d)) for d in dirs])
        order = np.argsort(rates)

        dirs = dirs[order]
        rates = rates[order]
        idx = np.abs(rates - self.ev_rate).argmin()

        lut_fname = os.path.join(dirs[idx], 'lut', f'block{blk}.lut')
        lut = np.fromfile(lut_fname, np.intc).reshape((512,512))

        yd = np.diff(lut, axis = 0, prepend = lut.max()) != 0
        xd = np.diff(lut, axis = 1, prepend = lut.max()) != 0

        overlay = np.logical_or(xd, yd)
        overlay = ndimage.binary_dilation(overlay, np.ones((3,3)))
        return np.ma.array(overlay, mask = (overlay == 0))

    def flood_cb(self):
        """ Update the flood according to energy and DOI thresholds """
        if self.d is None: return

        lut = None
        if self.show_lut.get() and self.output_dir:
            lut = self.create_lut_borders()

        eth = self.energy.thresholds()
        dth = self.doi.thresholds()

        es = self.d[:,0]
        doi = self.d[:,1]
        idx = np.where((eth[0] < es) & (es < eth[1]) &
                       (dth[0] < doi) & (doi < dth[1]))[0]

        x, y = self.d[idx,2], self.d[idx,3]
        self.flood.update(x, y,
                          warp = self.warp, overlay = lut,
                          draw_voronoi = self.show_voronoi.get(),
                          draw_points = self.show_points.get())

    def doi_cb(self, retain = True):
        """ Update the DOI according to the energy thresholds """
        eth = self.energy.thresholds()
        es = self.d[:,0]
        idx = np.where((eth[0] < es) & (es < eth[1]))[0]
        self.doi.update(self.d[idx,1], retain)

    def energy_cb(self, retain = True):
        """ Update the energy according to the DOI thresholds """
        dth = self.doi.thresholds()
        doi = self.d[:,1]
        idx = np.where((dth[0] < doi) & (doi < dth[1]))[0]
        self.energy.update(self.d[idx,0], retain)

    def check_output_dir(self, reset = False):
        if reset: self.output_dir = None
        self.output_dir = self.output_dir or tk.filedialog.askdirectory(
                title = "Configuration data directory",
                initialdir = '/')

        default_dir = os.path.join(self.output_dir, 'default')
        lut_dir = os.path.join(default_dir, 'lut')
        fld_dir = os.path.join(default_dir, 'flood')

        for d in [default_dir, lut_dir, fld_dir]:
            try:
                os.mkdir(d)
            except FileExistsError: pass

        return self.output_dir

    def store_lut_cb(self):
        output_dir = self.check_output_dir()
        if not output_dir or self.flood.f is None:
            return

        blk = self.return_block()
        print(f'Store calibration data for block {blk}...', end = ' ', flush = True)

        # store the LUT for this block to the specified directory
        lut = Flood.nearest_peak((self.flood.img_size,)*2,
                self.flood.pts.reshape(-1,2))

        if self.warp is not None:
            lut = Flood.warp_lut(lut, self.warp)
            self.warp = None

        lut_fname = os.path.join(output_dir, 'default', 'lut', f'block{blk}.lut')
        lut.astype(np.intc).tofile(lut_fname)
        flood_fname = os.path.join(output_dir, 'default', 'flood', f'block{blk}.raw')
        self.flood.img.astype(np.intc).tofile(flood_fname)

        # update json file with photopeak position for this block
        config_file = os.path.join(output_dir, 'default', 'config.json')

        try:
            with open(config_file, 'r') as f:
                cfg = json.load(f)
        except FileNotFoundError: cfg = {}

        calibration.create_cfg_vals(self.d, lut, blk, cfg,
                                    (self.energy.counts, self.energy.bins))

        # save the config file
        with open(config_file, 'w') as f:
            json.dump(cfg, f)

        print('Done')

        # increment the active block and update the UI
        all_blks = self.return_block(all_blocks = True)
        try: 
            self.set_block(all_blks.index(blk) + 1)
            self.plots_update()
        except KeyError: pass

    def scale_config(self):
        input_file = tk.filedialog.askopenfilename(
                title = "Load coincidence listmode data",
                initialdir = "/",
                filetypes = coincidence_filetypes)

        cfgdir = self.check_output_dir()

        stat_queue = queue.Queue()
        data_queue = queue.Queue()
        terminate = threading.Event()
        thr = threading.Thread(target = calibration.create_scaled_calibration,
                               args = [input_file, cfgdir, stat_queue, data_queue, terminate])
        thr.start()

        ProgressPopup(stat_queue, data_queue, terminate,
                      callback = lambda *args: None,
                      fmt = 'Period: {} Block: {}')
