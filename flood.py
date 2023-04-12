import cv2 as cv
import numpy as np
from scipy import ndimage, signal
from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree
import pyelastix
import matplotlib.pyplot as plt

from calibration import flood_preprocess

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure, SubplotParams
from matplotlib.patches import Circle

class Flood:

    @staticmethod
    def nearest_peak(shape, pks, distance_upper_bound = 30):
        tree = KDTree(pks)
        x,y = [np.arange(l) for l in shape]
        grid = np.array(np.meshgrid(y,x)).reshape(2, np.prod(shape))
        _,nearest = tree.query(grid.T, workers = -1,
                               distance_upper_bound = distance_upper_bound)
        return nearest.reshape(shape)

    @staticmethod
    def warp_lut(lut, warp):
        return cv.warpPerspective(lut, warp, lut.shape,
                flags = cv.WARP_INVERSE_MAP | cv.INTER_NEAREST,
                borderMode = cv.BORDER_CONSTANT, borderValue = int(lut.max()))

    def __init__(self, flood, warp = None, preprocess = True):
        self.fld = flood.astype(float)

        if warp is not None:
            self.fld = cv.warpPerspective(self.fld, warp, self.fld.shape)

        if preprocess:
            self.fld = flood_preprocess(self.fld)

    def register_peaks(self, pks):
        """ Register a starting set of peaks to the preprocessed flood. The registration
        is a 2D deformable registration using pyelastix. The peaks should generally be
        aligned with the rows and columns of the flood. The registration can compensate for
        local deformations in the flood.
        The output peaks are actually an estimate, since the deformation map is not really
        invertable. The most accurate method is to apply the deformation map to the LUT.
        """
        pks_in = pks.astype(int)
        pks_map = np.zeros(self.fld.shape)
        for pk in pks_in:
            pks_map[pk[1], pk[0]] = 1
        pks_blur = ndimage.gaussian_filter(pks_map, 1)

        pars = pyelastix.get_default_params()
        pars.NumberOfResolution = 2
        pars.MaximumNumberOfIterations = 200

        _, (xf, yf) = pyelastix.register(pks_blur, self.fld, pars, verbose = 0)
        pks_out = np.zeros(pks.shape)

        # Note that this is just an estimate
        # It assumes that the displacement field is smooth
        for pk, pk_out in zip(pks_in, pks_out):
            pk_out[0] = pk[0] - xf[pk[1], pk[0]]
            pk_out[1] = pk[1] - yf[pk[1], pk[0]]

        return pks_out

    def find_1d_peaks(self, axis = 0):
        s = np.sum(self.fld, axis)
        cog = np.average(np.arange(len(s)), weights = s)

        distance = 10
        n = 19
        n_side = 9

        # Find up to 19 peaks satisfying a large min. distance between peaks
        main_pks,_ = signal.find_peaks(s, distance = distance)
        main_order = s[main_pks].argsort()
        main_pks = main_pks[main_order[::-1]][:n]

        # Pick a center peak based on the flood COG, then take up to 9 peaks on L and R
        center_pk_idx = main_pks[np.argmin(np.abs(main_pks - cog))]
        lpk = list(main_pks[main_pks < center_pk_idx])[:n_side]
        rpk = list(main_pks[main_pks > center_pk_idx])[:n_side]

        # If 9 peaks were not found on the L or R, decrease the min. distance and repeat
        while len(lpk) < n_side or len(rpk) < n_side:
            distance -= 1

            # failed to find sufficient peaks
            if distance == 0: return None

            # Find peaks satisfying the new min, distance, and remove original peaks
            other_pks,_ = signal.find_peaks(s, distance = distance)
            other_pks = list(set(other_pks) - set(main_pks))

            # If no new peaks are found, just continue to reduce the min distance again
            if len(other_pks) == 0: continue

            # Sort newly found peaks by height
            other_order = s[other_pks].argsort()
            other_pks = np.array(other_pks)[other_order[::-1]]

            # Add to L or R peaks from newly found peaks, as needed
            other_lpk = other_pks[other_pks < center_pk_idx]
            other_rpk = other_pks[other_pks > center_pk_idx]
            lpk += list(other_lpk[:(n_side-len(lpk))])
            rpk += list(other_rpk[:(n_side-len(rpk))])

        pks = lpk + [center_pk_idx] + rpk
        pks.sort()
        return pks

    def estimate_peaks(self):
        rows = self.find_1d_peaks(1)
        cols = self.find_1d_peaks(0)
        pks = np.array(np.meshgrid(cols,rows)).reshape(2, len(rows)*len(cols))
        return pks

class PerspectiveTransformDialog(tk.Toplevel):
    def __init__(self, root, flood, callback):
        super().__init__(root)
        self.title('Perspective Transform')
        self.attributes('-type', 'dialog')

        self.fig = Figure(subplotpars = SubplotParams(0,0,1,1))
        self.plot = self.fig.add_subplot(frame_on = False)
        self.canvas = FigureCanvasTkAgg(self.fig, master = self)
        self.canvas.get_tk_widget().config(bd = 3, relief = tk.GROOVE)
        self.canvas.draw()

        self.button_frame = tk.Frame(self)
        self.preview_button = tk.Button(self, text = 'Preview', command = self.preview)
        self.revert_button = tk.Button(self, text = 'Revert', command = self.revert)
        self.apply_button = tk.Button(self, text = 'Apply', command = self.apply)
        self.close_button = tk.Button(self, text = 'Close', command = self.destroy)

        self.canvas.get_tk_widget().pack(side = tk.TOP, fill = tk.BOTH, expand = True)
        self.button_frame.pack()
        self.preview_button.pack(side = tk.LEFT)
        self.revert_button.pack(side = tk.LEFT)
        self.apply_button.pack(side = tk.LEFT)
        self.close_button.pack(side = tk.LEFT)

        self.canvas.mpl_connect('button_press_event', self.drag_start)
        self.canvas.mpl_connect('button_release_event', self.drag_stop)

        x0 = 20
        x1 = 511 - x0
        self.default_points = np.array(
                [[x0,x0],[x0,x1],[x1,x1],[x1,x0]])
        self.points = [Circle(pt, 4, zorder = 10) for pt in self.default_points]
        self.flood = flood
        self.revert()
        self.active_point = None
        self.mat = None
        self.callback = callback

    def get_points(self):
        return np.array([pt.center for pt in self.points])

    def drag_start(self, ev):
        self.connection = self.canvas.mpl_connect('motion_notify_event', self.cursor_set)
        pt_centers = self.get_points()
        pt = np.array([ev.xdata, ev.ydata])
        dst = np.linalg.norm(pt_centers - pt, axis = 1)
        self.active_point = self.points[np.argmin(dst)]

    def drag_stop(self, ev):
        self.canvas.mpl_disconnect(self.connection)
        self.active_point = None

    def cursor_set(self, ev):
        if self.active_point is not None:
            self.active_point.set(center = [ev.xdata,ev.ydata])
            self.canvas.draw()

    def apply(self):
        self.preview()
        self.callback(self.mat)
        self.destroy()

    def preview(self):
        pts = self.get_points()
        xmin, ymin  = pts.min(0)
        xmax, ymax  = pts.max(0)
        target = np.array([[xmin,ymin], [xmin,ymax], [xmax,ymax], [xmax,ymin]])

        self.mat = cv.getPerspectiveTransform(
                pts.astype(np.float32), target.astype(np.float32))
        warped_flood = cv.warpPerspective(self.flood, self.mat, self.flood.shape)

        self.plot.clear()
        self.plot.imshow(warped_flood, aspect = 'auto', zorder = 0)
        self.plot.invert_yaxis()
        self.plot.set_xlim(0,511)
        self.plot.set_ylim(0,511)
        self.canvas.draw()

        self.preview_button.config(state = tk.DISABLED)
        self.revert_button.config(state = tk.NORMAL)

    def revert(self):
        self.plot.clear()
        self.plot.imshow(self.flood, aspect = 'auto', zorder = 0)
        [self.plot.add_patch(pt) for pt in self.points]
        self.plot.invert_yaxis()
        self.plot.set_xlim(0,511)
        self.plot.set_ylim(0,511)
        self.canvas.draw()

        self.preview_button.config(state = tk.NORMAL)
        self.revert_button.config(state = tk.DISABLED)

