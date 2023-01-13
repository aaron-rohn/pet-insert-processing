import numpy as np
import data_loader

class Coincidences:
    def __init__(self, fname):
        self.arr = np.memmap(fname, np.uint16, 'r')
        self.arr = self.arr.reshape((-1, data_loader.coincidence_cols))
        self.subset = None

    def data(self):
        return self.arr if self.subset is None else self.subset

    def apply_subset(self, start = None, end = None, mask = None, reset = False):
        if reset:
            self.subset = None
            return

        d = self.data()
        if mask is not None:
            self.subset = d[mask,:]
        else:
            self.subset = d[int(start):int(end),:]

    def blka(self): return self.data()[:,0] >> 8
    def blkb(self): return self.data()[:,0] & 0xFF
    def prompt(self): return self.data()[:,1] >> 8
    def tdiff(self): return (self.data()[:,1] & 0xFF).astype(np.int8)

    def e_aF(self): return self.data()[:,2]
    def e_aR(self): return self.data()[:,3]
    def e_bF(self): return self.data()[:,4]
    def e_bR(self): return self.data()[:,5]

    def e_a(self): return self.e_aF() + self.e_aR()
    def e_b(self): return self.e_bF() + self.e_bR()

    def doi_a(self):
        d = self.data()
        return d[:,2] / (d[:,2] + d[:,3])

    def doi_b(self):
        d = self.data()
        return d[:,4] / (d[:,4] + d[:,5])

    def x_a(self): return self.data()[:,6]
    def y_a(self): return self.data()[:,7]
    def x_b(self): return self.data()[:,8]
    def y_b(self): return self.data()[:,9]

    def abstime(self): return self.data()[:,10]
