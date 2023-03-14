import os
import tkinter as tk
import tkinter.filedialog

cfg_dir = None
last_dir = '/mnt/acq'

def dialog(fn, *args, **kwds):
    global last_dir
    result = fn(*args, **kwds, initialdir = last_dir)

    if not result:
        return None

    if isinstance(result, tuple):
        result = list(result)
        last_dir = os.path.dirname(result[0])
    elif os.path.isdir(result):
        last_dir = result
    else:
        # either existing or new file
        last_dir = os.path.dirname(result)

    return result

askopenfilename   = lambda *args, **kwds: dialog(tk.filedialog.askopenfilename, *args, **kwds)
askopenfilenames  = lambda *args, **kwds: dialog(tk.filedialog.askopenfilenames, *args, **kwds)
asksaveasfilename = lambda *args, **kwds: dialog(tk.filedialog.asksaveasfilename, *args, **kwds)
askdirectory      = lambda *args, **kwds: dialog(tk.filedialog.askdirectory, *args, **kwds)

def askformatfilenames(files, **kwds):
    fmt = dialog(tk.filedialog.asksaveasfilename,
            title = "Format filenames: {i}->index, {f}->filename", **kwds)

    if not fmt: return None

    names = [os.path.basename(f) for f in files]
    names = [os.path.splitext(n) for n in names]
    names, _ = zip(*names)

    try:
        return [fmt.format(i = idx, f = fname) for idx,fname in enumerate(names)]
    except Exception as e:
        print(f'Error formatting file names: {e}')
        return None

def check_config_dir(reset = False):
    global cfg_dir
    if reset: cfg_dir = None

    if cfg_dir is None:
        new_dir = askdirectory(title = "Configuration data directory")

        if not new_dir:
            return None

        cfg_dir = new_dir

    lut_dir = os.path.join(cfg_dir, 'lut')
    fld_dir = os.path.join(cfg_dir, 'flood')

    for d in [cfg_dir, lut_dir, fld_dir]:
        try:
            os.mkdir(d)
        except FileExistsError: pass

    return cfg_dir
