import cv2 as cv
import numpy as np
from scipy import ndimage, signal
from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree
import pyelastix
import matplotlib.pyplot as plt

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure, SubplotParams
from matplotlib.patches import Circle

def nearest_peak(shape, pks):
    tree = KDTree(pks)
    x,y = [np.arange(l) for l in shape]
    grid = np.array(np.meshgrid(y,x)).reshape(2, np.prod(shape))
    _,nearest = tree.query(grid.T, workers = -1, distance_upper_bound = 30)
    nearest = nearest.reshape(shape)
    return nearest

def sort_polar(points):
    disp = points - points.mean(0)
    acos = np.arccos(disp[:,0] / np.linalg.norm(disp, axis = 1))
    acos = np.where(disp[:,1] > 0, acos, np.pi*2 - acos)
    return points[np.argsort(acos)]

class Flood:
    def __init__(self, flood, warp = None):
        self.fld = np.copy(flood).astype(np.float64)

        self.field = None # 2d nonrigid deformation field
        self.warp = warp # perspective transform matrix
        if self.warp is not None:
            self.fld = cv.warpPerspective(self.fld, self.warp, self.fld.shape)

        # remove the background
        blur = ndimage.gaussian_filter(self.fld, 5)
        e0 = self.edges(blur, 1)
        e1 = self.edges(blur, 0)
        mask = np.ones(self.fld.shape, dtype = bool)
        mask[e0[0]:e0[1], e1[0]:e1[1]] = False
        self.fld[mask] = 0

        # log filter and normalize
        self.fld = ndimage.gaussian_laplace(self.fld, 1.5)
        self.fld /= np.min(self.fld)
        self.fld[self.fld < 0] = 0

        # remove low frequency components
        with np.errstate(invalid='ignore'):
            self.fld /= ndimage.gaussian_filter(self.fld, 20)
        self.fld = np.nan_to_num(self.fld, 0)

        # remove outliers
        q = np.quantile(self.fld, 0.95)
        filt = ndimage.median_filter(self.fld, 5)
        self.fld = np.where(self.fld < q, self.fld, filt)

    def warp_lut(self, lut):
        if self.warp is None:
            return lut
        else:
            return cv.warpPerspective(lut, self.warp, lut.shape,
                    flags = cv.WARP_INVERSE_MAP | cv.INTER_NEAREST,
                    borderMode = cv.BORDER_CONSTANT, borderValue = int(lut.max()))

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
        deformed, field = pyelastix.register(pks_blur, self.fld, pars, verbose = 0)

        xfield, yfield = field
        x = np.tile(np.arange(512, dtype = np.float32), 512).reshape(512,512)
        y = x.T + yfield
        x = x   + xfield

        self.field = (x, y)

        pks_out = np.zeros(pks.shape)

        # Note that this is just an estimate
        # It assumes that the displacement field is smooth
        for pk, pk_out in zip(pks_in, pks_out):
            pk_out[0] = pk[0] - xfield[pk[1], pk[0]]
            pk_out[1] = pk[1] - yfield[pk[1], pk[0]]

        return pks_out

    def register_lut(self, lut):
        """ Apply the previously calculated deformation map the a LUT. If the
        peaks used to calculate the LUT were already registered to the flood,
        this method need not be called.
        """
        if self.field is None:
            return lut
        else:
            return cv.remap(lut, self.field[0], self.field[1], cv.INTER_NEAREST,
                    borderMode = cv.BORDER_CONSTANT, borderValue = int(lut.max()))

    def edges(self, f, axis = 0, qtl = [0.02, 0.98]):
        p = np.sum(f, axis)
        p_csum = np.cumsum(p)
        p_csum_nan = np.copy(p_csum)
        p_csum_nan[p == 0] = np.NaN

        qtl_vals = np.nanquantile(p_csum_nan, qtl)
        ledge, redge = np.interp(qtl_vals, p_csum, np.arange(len(p_csum)))
        return [int(ledge), int(redge)]

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

            if distance == 0: raise RuntimeError('Failed to find sufficient peaks')

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


        """
        plt.plot(s)
        plt.axvline(center_pk_idx, color = 'red')
        [plt.axvline(pk, color = 'blue') for pk in lpk+rpk]
        plt.show()
        """

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

