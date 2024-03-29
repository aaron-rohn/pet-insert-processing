import os
import ycm_core
import numpy as np

flags = [
    '-std=c++20', '-xc++',
    '-I', '/usr/include/python3.10',
    '-I', os.path.dirname(__file__) + "/include",
    '-I', np.get_include()]

SOURCE_EXTENSIONS = [ '.cpp', '.cxx', '.cc', '.c', ]

def FlagsForFile( filename, **kwargs ):
    return {'flags': flags, 'do_cache': True}
