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

class WrappingLabel(tk.Label):
    def reconfig(self, *args, **kwds):
        width = self.winfo_width()
        self.config(wraplength = width)

    def __init__(self, master = None, **kwargs):
        super().__init__(master, **kwargs)
        self.bind('<Configure>', self.reconfig)

class FileSelector(tk.Frame):
    def __init__(self, root, callback):
        super().__init__(root)
        self.callback = callback 

        self.load_sgls_button = tk.Button(self, text = "Load Singles",
                command = lambda: self.loader_wrapper(SinglesLoader))

        self.load_coin_button = tk.Button(self, text = "Load Coincidences",
                command = lambda: self.loader_wrapper(CoincidenceLoader))

        self.sort_coin_button = tk.Button(self, text = "Sort Coincidences",
                command = lambda: self.loader_wrapper(CoincidenceSorter))

        self.validate_button = tk.Button(self, text = "Validate files", command = validate_singles)

    def pack(self, **kwds):
        super().pack(**kwds)
        args = {'side': tk.TOP, 'padx': 5, 'pady': 5, 'fill': tk.X, 'expand': True}
        self.load_sgls_button.pack(**args)
        self.load_coin_button.pack(**args)
        self.sort_coin_button.pack(**args)
        self.validate_button.pack(**args)

    def cfg_buttons(self, state):
        self.load_sgls_button.config(state = state) 
        self.load_coin_button.config(state = state) 
        self.sort_coin_button.config(state = state) 
        self.validate_button.config(state = state)

    def callback_wrapper(self, response, *args, **kwds):
        self.cfg_buttons(tk.NORMAL)
        if isinstance(response, Exception):
            tk.messagebox.showerror(message = f'{response}')
        else:
            self.callback(response, *args, **kwds)

    def loader_wrapper(self, loader):
        self.cfg_buttons(tk.DISABLED)
        try:
            self.loader = loader(self.callback_wrapper)
        except Exception as e:
            self.callback_wrapper(e)

class ScrolledListbox(tk.Frame):
    def __init__(self, root, title = None):
        super().__init__(root)
        self.active_var = tk.Variable()

        self.title_text = title
        self.title = tk.Label(self, text = self.title_text) if self.title_text else None
        self.active = tk.Listbox(self, listvariable = self.active_var, exportselection = False)
        self.scroll = tk.Scrollbar(self, orient = tk.VERTICAL, command = self.active.yview)
        self.active.config(yscrollcommand = self.scroll.set)

    def pack(self, **kwds):
        super().pack(**kwds)

        if self.title is not None:
            self.title.pack(side = tk.TOP, pady = 5)

        self.active.pack(fill = tk.X, expand = True, side = tk.LEFT)
        self.scroll.pack(fill = tk.Y, side = tk.RIGHT)

    def get_active(self, all_blocks = False):
        all_items = self.active_var.get()
        all_items = [int(i.split()[0]) for i in all_items]
        if all_blocks: return all_items

        sel = self.active.curselection()
        return all_items[sel[0]] if len(sel) == 1 else None

    def set_active(self, position = None):
        self.active.select_clear(0, tk.END)
        if position is not None:
            self.active.selection_set(position)
            self.active.activate(position)

    def set(self, blks):
        self.title.config(text = f'{self.title_text} ({len(blks)})')
        return self.active_var.set(blks)

    def bind(self, new_block_cb):
        self.active.bind('<<ListboxSelect>>', new_block_cb)

class App(ttk.Notebook):
    def collect_data(self, d, ev_rate = 0):
        def fmt(num):
            n = math.log10(num) // 3
            suffix = ['', 'K', 'M'][int(n)]
            return f'{round(num / 10**(n*3), 1)}{suffix}'

        self.ev_rate = ev_rate
        self.d = d
        self.block.set([f'{a}  -  {fmt(b.shape[0])}' for a,b in d.items()])

    def return_data(self, block):
        return self.d[block], self.ev_rate
    
    def return_block(self, all_blocks = False):
        return self.block.get_active(all_blocks)

    def set_block(self, *args, **kwds):
        return self.block.set_active(*args, **kwds)

    def __init__(self, root):
        super().__init__(root)
        root.title('PET data processing')

        listmode_frame = tk.Frame(self)
        sinogram_frame = tk.Frame(self)
        self.add(listmode_frame, text = "Listmode Processing")
        self.add(sinogram_frame, text = "Sinogram Processing")
        self.pack(fill = tk.BOTH, expand = True)

        lm_top_frame = tk.Frame(listmode_frame)
        lm_top_frame.pack(side = tk.TOP, fill = tk.X, expand = False)

        self.file = FileSelector(lm_top_frame, self.collect_data)
        self.block = ScrolledListbox(lm_top_frame, "Active Blocks")

        self.file.pack(side = tk.LEFT, padx = 30, pady = 20, fill = tk.X, expand = False)
        self.block.pack(side = tk.LEFT, fill = tk.X, expand = True, padx = 30, pady = 20)

        self.plots = Plots(listmode_frame, self.return_data, self.return_block, self.set_block,
                           padx = 5, pady = 5, fill = tk.BOTH, expand = True)
        self.block.bind(self.plots.plots_update)

        self.sino = SinogramDisplay(sinogram_frame)
        self.sino.pack()

if __name__ == "__main__":
    root = tk.Tk(className = 'PET data processing')
    app = App(root)
    root.mainloop()
