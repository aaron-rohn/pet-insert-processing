import glob
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

class Flood():
    def __init__(self, f, ksize = 1):
        if isinstance(f, str):
            self.fld = np.fromfile(f, 'int32').reshape((512,512))
        else:
            self.fld = np.copy(f)

        self.fld = self.fld.astype('double')
        self.blur = ndimage.gaussian_filter(self.fld, 5)

        e0 = self.edges(1, 10)
        e1 = self.edges(0, 10)
        mask = np.ones(self.fld.shape, dtype = bool)
        mask[e0[0]:e0[1], e1[0]:e1[1]] = False
        self.fld[mask] = 0

        self.fld = self.log_filter(ksize)

        with np.errstate(invalid='ignore'):
            self.fld /= ndimage.gaussian_filter(self.fld, 20)

        self.fld = np.nan_to_num(self.fld, 0)
        self.correct_outliers()

    def log_filter(self, ksize = 1):
        f = ndimage.gaussian_laplace(self.fld, ksize)
        f = f / np.min(f)
        f[f < 0] = 0
        return f

    def edges(self, axis = 0, threshold = 10):
        p = np.sum(self.blur, axis)
        thresh = np.max(p) / threshold
        ledge = np.argmax(p > thresh)
        redge = len(p) - np.argmax(p[::-1] > thresh)
        return np.array([ledge, redge])

    def correct_outliers(self):
        cts, vals = np.histogram(self.fld)
        cts, vals = cts[::-1], vals[::-1]
        for c,v in zip(cts,vals):
            if c > 500:
                filt = ndimage.median_filter(self.fld, 5)
                self.fld = np.where(self.fld < v, self.fld, filt)
                break

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
        rows = self.find_1d_peaks(1)[::-1]
        cols = self.find_1d_peaks(0)
        pks = np.array(np.meshgrid(cols,rows)).reshape(2, len(rows)*len(cols))
        return pks


if __name__ == "__main__":
    floods = glob.glob("/home/aaron/Downloads/block_*.raw")
    for f in floods:
        print(f)
        fld = Flood(f)
        fld.estimate_peaks()
