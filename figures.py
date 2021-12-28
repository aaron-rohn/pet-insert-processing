import numpy as np
import tkinter as tk
import flood as fld
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure, SubplotParams
from scipy.spatial import Voronoi, voronoi_plot_2d

class FloodHist:
    def __init__(self, frame, **kwargs):
        self.img_size = 512
        self.npts = 19

        self.fig = Figure(subplotpars = SubplotParams(0,0,1,1))
        self.plot = self.fig.add_subplot(frame_on = False)
        self.canvas = FigureCanvasTkAgg(self.fig, master = frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(**kwargs)

        self.img = np.zeros((self.img_size,self.img_size))
        self.f = None
        self.selection = None
        self.pts = None
        self.pts_active = None

        self.canvas.mpl_connect('button_press_event', self.click)

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
            self.pts = np.flip(self.pts, (0,1))
            self.selection = None

        self.redraw()

    def redraw(self):
        """
        Coordinate layout:
        
        'Animal' side
        #########
        #D     A#
        #   0   #
        #C     B#
        #########
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

        if self.pts is not None:
            active = self.pts[self.pts_active].T
            inactive = self.pts[~self.pts_active].T

            vor = Voronoi(self.pts.reshape(-1,2))
            """
            voronoi_plot_2d(vor, ax = self.plot, 
                    show_vertices = False, 
                    show_points = False, 
                    line_colors = 'grey',
                    line_alpha = 0.5)
            """

            self.plot.plot(*active, '.b', *inactive, '.r', ms = 1)

        # Invert Y axis to display A channel in Top right
        self.plot.invert_yaxis()
        self.plot.set_xlim(0,self.img_size-1)
        self.plot.set_ylim(0,self.img_size-1)
        self.canvas.draw()

    def update(self, data, smoothing, warp):
        # Image rows (1st coordinate) are the Y value ((A+D)/e)
        # Image cols (2nd coordinate) are the X value ((A+B)/e)
        self.img,*_ = np.histogram2d(data['y'], data['x'],
                                     bins = self.img_size, range = [[0,1],[0,1]])
        self.f = fld.Flood(self.img, smoothing, warp)

        try:
            self.pts = self.f.estimate_peaks()
            self.pts = self.pts.T.reshape(self.npts, self.npts, 2)
        except RuntimeError as e:
            print(repr(e))
            self.pts = None

        self.pts_active = np.zeros((self.npts, self.npts), dtype = bool)
        self.redraw()

class ThresholdHist:
    def update(self, data, retain = False):
        last_rng = self.thresholds()
        self.plot.clear()
        rng = np.quantile(data, [0.01, 0.99])
        n,bins,_ = self.plot.hist(data, bins = 512, range = rng)

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
            # DOI histogram - take 50% central counts
            rng = np.quantile(data, [0.1, 0.9])

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

    def init_lines(self, lims = (0,1)):
        self.lines = [self.plot.axvline(x,linewidth=3,color='r') for x in lims]

    def __init__(self, frame, is_energy, **kwargs):
        self.is_energy = is_energy
        self.e_window = 0.2
        self.peak = 0

        self.fig = Figure()
        self.plot = self.fig.add_subplot()
        self.init_lines()
        self.canvas = FigureCanvasTkAgg(self.fig, master = frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(**kwargs)

        self.canvas.mpl_connect('button_press_event', self.drag_start)
        self.canvas.mpl_connect('button_release_event', self.drag_stop)

        self.connection = None
        self.active_line = None
        self.callback = []
