import cv2 as cv
import numpy as np
from scipy import ndimage, signal
from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree
import matplotlib.pyplot as plt

def nearest_peak(shape, pks):
    tree = KDTree(pks)
    x,y = [np.arange(l) for l in shape]
    grid = np.array(np.meshgrid(y,x)).reshape(2, np.prod(shape))
    _,nearest = tree.query(grid.T, workers = -1, distance_upper_bound = 30)
    nearest = nearest.reshape(shape)
    return nearest

class Flood:
    def __init__(self, f, ksize = 1.0, warp = False):
        self.fld = np.copy(f).astype(np.float64)

        # warp flood to rectangle
        self.warped = warp
        self.transformation_matrix = None
        if warp:
            try:
                self.fld, self.transformation_matrix = self.apply_warp(self.fld)
            except RuntimeError as e:
                print(f'Failed to warp flood: {e}')
                self.warped = False

        # remove the background
        blur = ndimage.gaussian_filter(self.fld, 5)
        e0 = self.edges(blur, 1, 20)
        e1 = self.edges(blur, 0, 20)
        mask = np.ones(self.fld.shape, dtype = bool)
        mask[e0[0]:e0[1], e1[0]:e1[1]] = False
        self.fld[mask] = 0

        # log filter and normalize
        self.fld = ndimage.gaussian_laplace(self.fld, ksize)
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

    def apply_warp(self, f):
        gaussian_filter_sigma = 10
        binary_mask_threshold = 10
        contour_threshold = 50

        blur = ndimage.gaussian_filter(f, gaussian_filter_sigma)
        blur = cv.convertScaleAbs(blur, alpha = 255.0/blur.max())

        mask = (blur > binary_mask_threshold).astype(np.uint8)
        ctor, hier = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        ctor_approx = cv.approxPolyDP(ctor[0], contour_threshold, True)

        if ctor_approx.shape[0] != 4:
            raise RuntimeError('Could not find approximate bounding rectangle')

        cmin, rmin = ctor_approx.squeeze().min(0)
        cmax, rmax = ctor_approx.squeeze().max(0)
        ctor_target = np.array([[cmin,rmin], [cmin,rmax], [cmax,rmax], [cmax,rmin]])

        mat = cv.getPerspectiveTransform(
                ctor_approx.astype(np.float32), ctor_target.astype(np.float32))

        f_out = cv.warpPerspective(f, mat, f.shape)

        return f_out, mat


    def edges(self, f, axis = 0, threshold = 10):
        p = np.sum(f, axis)
        thresh = np.max(p) / threshold
        ledge = np.argmax(p > thresh)
        redge = len(p) - np.argmax(p[::-1] > thresh)
        return np.array([ledge, redge])

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
