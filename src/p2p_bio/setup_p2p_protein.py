'''
python setup_p2p_protein.py build_ext -i

'''
from setuptools import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np

NAME = "protein for BFEchange of PPI"
VERSION = "0.1"
DESCR = "A wrapper of PPIprotein"
URL = "http://weilab.math.msu.edu"
REQUIRES = ['numpy', 'cython']

AUTHOR = "Jiahui Chen"
EMAIL = "jiahuic@uark.edu"

LICENSE = "Apache 2.0"

SRC_DIR = "p2p_protein"
PACKAGES = [SRC_DIR]

compile_args = ['-O2', '-std=c++11']
sources_list = [f'{SRC_DIR}/prot.pyx', 
                f'{SRC_DIR}/lib/Atom.cpp', 
                f'{SRC_DIR}/lib/Protein.cpp']
ext_1 = Extension(SRC_DIR + ".prot",
                  sources_list,
                  extra_compile_args = compile_args,
                  libraries=[],
                  include_dirs=[np.get_include()],
                  language="c++")

setup(install_requires=REQUIRES,
      packages=PACKAGES,
      zip_safe=False,
      name=NAME,
      version=VERSION,
      description=DESCR,
      author=AUTHOR,
      author_email=EMAIL,
      url=URL,
      license=LICENSE,
      cmdclass={"build_ext": build_ext},
      ext_modules=[ext_1]
      )