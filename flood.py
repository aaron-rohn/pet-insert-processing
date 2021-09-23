import glob
import numpy as np
from scipy import ndimage, signal
import matplotlib.pyplot as plt

class Flood():
    def __init__(self, f, ksize = 1):
        if isinstance(f, str):
            self.fld = np.fromfile(f, 'int32').reshape((512,512))
        else:
            self.fld = f

        self.fld = self.fld.astype('double')
        self.fld = self.log_filter(ksize)

        ta = self.edges(1, 10)
        ax = self.edges(0, 10)
        mask = np.zeros(self.fld.shape)
        mask[ta[0]-10:ta[1]+10,ax[0]-10:ax[1]+10] = 1
        self.fld = self.fld * mask / ndimage.gaussian_filter(self.fld, 20)
        self.fld = np.nan_to_num(self.fld, 0)

        self.correct_outliers()

    def sample(self, n):
        flat = self.fld.flatten() / np.sum(self.fld)
        sample_idx = np.random.choice(flat.size, size = int(n), p = flat)
        idx = np.unravel_index(sample_idx, self.fld.shape)
        return np.array(idx)

    def log_filter(self, ksize = 1):
        f = ndimage.gaussian_laplace(self.fld, ksize)
        f = f / np.min(f)
        f[f < 0] = 0
        return f

    def edges(self, axis = 0, threshold = 10):
        p = np.sum(self.fld, axis)
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

        main_pks,_ = signal.find_peaks(s, distance = distance)
        main_order = s[main_pks].argsort()
        main_pks = main_pks[main_order[::-1]][:n]

        center_pk_idx = main_pks[np.argmin(np.abs(main_pks - cog))]
        lpk = list(main_pks[main_pks < center_pk_idx])[:n_side]
        rpk = list(main_pks[main_pks > center_pk_idx])[:n_side]

        while len(lpk) < n_side or len(rpk) < n_side:
            distance -= 1

            other_pks,_ = signal.find_peaks(s, distance = distance)
            other_pks = list(set(other_pks) - set(main_pks))
            if len(other_pks) == 0: continue

            other_order = s[other_pks].argsort()
            other_pks = np.array(other_pks)[other_order[::-1]]

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

        #print(distance)

        return pks

    def estimate_peaks(self):
        ta = self.find_1d_peaks(1)
        ax = self.find_1d_peaks(0)
        pks = np.array(np.meshgrid(ta, ax)).reshape(2,len(ta)*len(ax))

        """
        plt.imshow(self.fld.T)
        plt.scatter(*pks, s = 4, color = 'red')
        plt.show()
        """

        return pks

if __name__ == "__main__":
    floods = glob.glob("/home/aaron/Downloads/block_*.raw")
    for f in floods:
        print(f)
        fld = Flood(f)
        pks = fld.estimate_peaks()
