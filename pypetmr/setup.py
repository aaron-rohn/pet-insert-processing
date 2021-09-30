import glob
import numpy as np
from distutils.core import setup, Extension

petmr_ext = Extension('petmr', glob.glob('src/*.cpp'), language = 'c++')
setup(name = 'petmr', 
      ext_modules = [petmr_ext],
      include_dirs = ['./include', np.get_include()])
