import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure, SubplotParams

class Flood():
    def update(self, data):
        self.plot.clear()
        self.plot.hist2d(data['x1'], (data['y1'] + data['y2']) / 2.0, bins = 256, range = [[0,1],[0,1]])
        self.canvas.draw()

    def __init__(self, frame, **kwargs):
        self.fig = Figure(subplotpars = SubplotParams(0,0,1,1))
        self.plot = self.fig.add_subplot(frame_on = False)
        self.canvas = FigureCanvasTkAgg(self.fig, master = frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(**kwargs)

class ThresholdHist():
    def update(self, data, retain = False):
        last_rng = self.thresholds()
        self.plot.clear()
        rng = np.quantile(data, [0.01, 0.99])
        self.plot.hist(data, bins = 512, range = rng)
        self.init_lines(last_rng if retain else rng)
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

    def __init__(self, frame, **kwargs):
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
