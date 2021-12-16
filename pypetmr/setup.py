import glob
import numpy as np
from setuptools import setup, Extension

petmr_ext = Extension('petmr', glob.glob('src/*.cpp'), language = 'c++',
        define_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')])
setup(name = 'petmr', 
      ext_modules = [petmr_ext],
      include_dirs = ['./include', np.get_include()])
