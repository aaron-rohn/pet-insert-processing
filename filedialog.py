import os
import pathlib
import tkinter as tk
import tkinter.filedialog

cfg = None
last_dir = '/mnt/acq'

def dialog(fn, *args, **kwds):
    global last_dir

    if 'initialdir' not in kwds:
        kwds['initialdir'] = last_dir

    result = fn(*args, **kwds)

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

def check_config(reset = False):
    global cfg
    if reset: cfg = None

    if cfg is None:
        new = asksaveasfilename(title = "Select JSON configuration file",
                filetypes = [('JSON files', '.json')])

        if not new:
            return None

        cfg = new

    pathlib.Path(cfg).touch()
    return cfg
