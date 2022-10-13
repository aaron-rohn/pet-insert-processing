import glob
import numpy as np
from setuptools import setup, Extension

petmr_ext = Extension('petmr', glob.glob('src/*.cpp'), language = 'c++',
                      define_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
                      extra_compile_args = ['-std=c++20', '-Wno-deprecated-enum-enum-conversion'],
                      libraries = ['opencv_core', 'opencv_imgproc'],
                      include_dirs = ['./include', np.get_include(), '/usr/include/opencv4'])

setup(name = 'petmr', ext_modules = [petmr_ext])
