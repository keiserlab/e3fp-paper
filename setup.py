from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name="E3FP paper code",
    ext_modules=cythonize('e3fp_paper/crossvalidation/kernels.pyx'),
    include_dirs=[numpy.get_include()]
)
