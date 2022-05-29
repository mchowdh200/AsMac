from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Distutils import build_ext

## TODO
# this will install AsMac_model/AsMac_utility, etc. to the current env
# To keep them under some main namespace, we should create another directory
# and put the relevant files there in the future
ext_modules=[Extension(
    '_softnw', ['_softnw.pyx'],
    libraries=['m'],
    extra_compile_args=['-ffast-math'])]

setup(
    name = 'AsMac',
    packages=find_packages(exclude=['build', 'CppAlign', 'misc']),
    zip_safe=False,
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules)
