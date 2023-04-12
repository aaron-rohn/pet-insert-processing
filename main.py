#!/usr/bin/python3.11

import math
import tkinter as tk
from tkinter import ttk
from figures import Plots
from sinogram_elements import SinogramDisplay
from data_loader import (
        SinglesLoader,
        CoincidenceLoader,
        CoincidenceSorter,
        validate_singles)

def fmt(num):
    n = math.log10(num) // 3
    suffix = ['', 'K', 'M'][int(n)]
    return f'{round(num / 10**(n*3), 1)}{suffix}'

class FileSelector(tk.Frame):
    FloodTypes = {'FRONT': 0, 'REAR': 1, 'BOTH': 2}

    def __init__(self, root, callback):
        super().__init__(root)
        self.callback = callback 

        self.load_sgls_button = tk.Button(self, text = "Load Singles",
                command = lambda: self.loader(SinglesLoader,
                    FileSelector.FloodTypes[self.flood_type.get()]))

        self.load_coin_button = tk.Button(self, text = "Load Coincidences",
                command = lambda: self.loader(CoincidenceLoader))

        self.sort_coin_button = tk.Button(self, text = "Sort Coincidences",
                command = lambda: self.loader(CoincidenceSorter))

        self.validate_button = tk.Button(self, text = "Validate singles", command = validate_singles)

        self.flood_type = tk.StringVar(self, 'BOTH')
        self.flood_type_menu = ttk.Combobox(self, textvariable = self.flood_type, state = 'readonly')
        self.flood_type_menu['values'] = list(FileSelector.FloodTypes.keys())

    def pack(self, **kwds):
        super().pack(**kwds)
        self.load_sgls_button.pack(**kwds)
        self.load_coin_button.pack(**kwds)
        self.sort_coin_button.pack(**kwds)
        self.validate_button.pack(**kwds)
        tk.Label(self, text = 'Flood Type:').pack(**kwds)
        self.flood_type_menu.pack(**kwds)

    def cfg_buttons(self, state):
        self.load_sgls_button.config(state = state) 
        self.load_coin_button.config(state = state) 
        self.sort_coin_button.config(state = state) 
        self.validate_button.config(state = state)

    def loader(self, ldr, *args):
        self.cfg_buttons(tk.DISABLED)

        def cb(*cbargs):
            self.cfg_buttons(tk.NORMAL)
            self.callback(*cbargs)

        ldr(cb, *args)

class BlockDisplay(tk.Frame):
    def __init__(self, root, title):
        super().__init__(root)
        self.title_text = title
        self.title = tk.Label(self, text = f'{self.title_text}:')

        self.active = tk.StringVar(root, "")
        self.menu = ttk.Combobox(root, textvariable = self.active, state = 'readonly')

    def pack(self, **kwds):
        super().pack(**kwds)
        self.title.pack(**kwds)
        self.menu.pack(**kwds)

    def incr(self):
        self.menu.current(newindex = self.menu.current() + 1)

    def set(self, *items):
        self.title.config(text = f'{self.title_text} ({len(items)}):')
        self.menu['values'] = items

    def get(self):
        return int(self.active.get().split()[0])

class ProcessingUI(ttk.Notebook):
    def set_data(self, d = None):
        if isinstance(d, Exception):
            tk.messagebox.showerror(message = f'{d}')
        elif isinstance(d, dict):
            self.d = d
            self.block.set(*[f'{a}  -  {fmt(b.shape[0])}' for a,b in d.items()])

    def get_data(self):
        return self.d[self.block.get()]

    def __init__(self, root):
        super().__init__(root)

        listmode_frame = tk.Frame(self)
        self.sino = SinogramDisplay(self)

        self.add(listmode_frame, text = "Listmode Processing")
        self.add(self.sino, text = "Sinogram Processing")
        self.pack(fill = tk.BOTH, expand = True)

        lm_top_frame = tk.Frame(listmode_frame)
        lm_top_frame.pack(side = tk.TOP)

        self.file = FileSelector(lm_top_frame, self.set_data)
        self.block = BlockDisplay(lm_top_frame, "Blocks")

        self.file.pack(side = tk.LEFT, padx = 5, pady = 5, fill = tk.X, expand = False)
        self.block.pack(side = tk.LEFT, padx = 5, pady = 5, fill = tk.X, expand = False)

        ttk.Separator(listmode_frame).pack(padx = 30, pady = 0, fill = tk.X)

        self.plots = Plots(listmode_frame, self.get_data, self.block.get, self.block.incr,
                           padx = 5, pady = 5, fill = tk.BOTH, expand = True)

        self.block.active.trace_add('write', self.plots.plots_update)

        self.sino.pack()

if __name__ == "__main__":
    root = tk.Tk(className = 'Processing')
    root.title('PET data processing')
    ProcessingUI(root)
    root.mainloop()
