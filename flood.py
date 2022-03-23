import cv2 as cv
import numpy as np
from scipy import ndimage, signal
from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree
import pyelastix
import matplotlib.pyplot as plt

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
    def __init__(self, f, ksize = 1.5, warp = False):
        self.fld = np.copy(f).astype(np.float64)

        self.field = None # 2d nonrigid deformation field

        # warp flood to rectangle
        self.transformation_matrix = None
        self.warped = False
        if warp:
            try:
                self.fld, self.transformation_matrix = self.apply_warp(self.fld)
                self.warped = True
            except RuntimeError as e:
                print(f'Failed to warp flood: {e}')

        # remove the background
        blur = ndimage.gaussian_filter(self.fld, 5)
        e0 = self.edges(blur, 1)
        e1 = self.edges(blur, 0)
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
        """ Warping the flood tries to apply a perspective transform to
        make the flood nearly rectangular. This is used to compensate for
        gain differences between the readout channels, and should be the
        first step in processing. The inverse warp should be applied to the
        LUT before saving it.
        """
        gaussian_filter_sigma = 10
        binary_mask_threshold = 10
        contour_threshold = 50

        blur = ndimage.gaussian_filter(f, gaussian_filter_sigma)
        blur = cv.convertScaleAbs(blur, alpha = 255.0/blur.max())

        mask = (blur > binary_mask_threshold).astype(np.uint8)
        ctor, hier = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for _ in range(10):
            ctor_approx = cv.approxPolyDP(ctor[0], contour_threshold, True)
            nvertex = ctor_approx.shape[0]

            if nvertex == 4:
                break
            elif nvertex > 4:
                contour_threshold += 5
            elif nvertex < 4:
                contour_threshold -= 5

            print(f'contour threshold: {contour_threshold}')

        else:
            raise RuntimeError('Could not find approximate bounding rectangle')

        ctor_approx = ctor_approx.squeeze()
        cmin, rmin  = ctor_approx.min(0)
        cmax, rmax  = ctor_approx.max(0)
        ctor_target = np.array([[cmin,rmin], [cmin,rmax], [cmax,rmax], [cmax,rmin]])

        ctor_approx = sort_polar(ctor_approx)
        ctor_target = sort_polar(ctor_target)

        idx = np.argwhere((ctor_target == [cmin,rmin]).all(1)).flatten()[0]
        ctor_approx = np.roll(ctor_approx, idx)
        ctor_target = np.roll(ctor_target, idx)

        mat = cv.getPerspectiveTransform(
                ctor_approx.astype(np.float32), ctor_target.astype(np.float32))

        f_out = cv.warpPerspective(f, mat, f.shape)
        return f_out, mat

    def warp_lut(self, lut):
        """ If the flood was warped before processing, apply the inverse warp
        to the LUT before saving.
        """
        if self.transformation_matrix is None:
            return lut
        else:
            return cv.warpPerspective(lut, self.transformation_matrix, lut.shape,
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
