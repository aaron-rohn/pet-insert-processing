import glob
import numpy as np
from setuptools import setup, Extension

petmr_ext = Extension('petmr', glob.glob('src/*.cpp'), language = 'c++',
                      define_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
                      extra_compile_args = ['-std=c++20', '-Ofast', '-march=native'],
                      include_dirs = ['./include', '../libsingles', np.get_include()],
                      library_dirs = ['../libsingles'],
                      libraries = ['singles'])

setup(name = 'petmr', ext_modules = [petmr_ext])
