import os, ctypes
from ctypes import c_int, c_uint, c_ulonglong

class ListmodeEvent(ctypes.Structure):
    _fields_ = [
            ('ring_a',      c_uint, 7),
            ('crystal_a',   c_uint, 9),

            ('ring_b',      c_uint, 7),
            ('crystal_b',   c_uint, 9),

            ('energy_b',    c_uint, 6),
            ('energy_a',    c_uint, 6),
            ('doi_b',       c_uint, 2),
            ('doi_a',       c_uint, 2),

            ('abstime',     c_uint, 10),
            ('tdiff',       c_int,  5),
            ('prompt',      c_uint, 1)]

class Listmode(ctypes.Structure):
    _fields_ = [('nevents', c_ulonglong),
                ('events', ctypes.POINTER(ListmodeEvent))]

    def __init__(self, fname):
        self.nevents = os.path.getsize(fname) // ctypes.sizeof(Listmode)
        ev = (Listmode * self.nevents)()
        with open(fname, 'rb') as f:
            f.readinto(ev)

        self.events = ctypes.cast(ev, ctypes.POINTER(ListmodeEvent))
