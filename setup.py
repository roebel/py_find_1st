#! /usr/bin/env python
from __future__ import print_function
from ez_setup import use_setuptools
use_setuptools()

from setuptools import setup
from distutils.extension import Extension
from distutils.command.build_ext import build_ext
from distutils.command.sdist import sdist 

import numpy as np
import os
import sys
import re
import subprocess
import errno

def compiler_is_clang(comp) :
    print("check for clang compiler ...", end=' ')
    try:
        cc_output = subprocess.check_output(comp+['--version'],
                                            stderr = subprocess.STDOUT, shell=False)
    except OSError as ex:
        print("compiler test call failed with error {0:d} msg: {1}".format(ex.errno, ex.strerror))
        print("no")
        return False

    ret = re.search(b'clang', cc_output) is not None
    if ret :
        print("yes")
    else:
        print("no")
    return ret


class build_ext_subclass( build_ext ):
    def build_extensions(self):
        #c = self.compiler.compiler_type
        #print "compiler attr", self.compiler.__dict__
        #print "compiler", self.compiler.compiler
        #print "compiler is",c
        if compiler_is_clang(self.compiler.compiler):
            for e in self.extensions:
                e.extra_compile_args.append('-stdlib=libstdc++')
                e.extra_compile_args.append('-Wno-unused-function')
            for e in self.extensions:
                e.extra_link_args.append('-stdlib=libstdc++')
        build_ext.build_extensions(self)

        
        
find_1st_ext = Extension("find_1st", ["utils_find_1st/find_1st.cpp"],
                          include_dirs=[np.get_include()],
                           language="c++"   )


ext_modules=[find_1st_ext]


# get _pysndfile version number
for line in open("utils_find_1st/__init__.py") :
    if "version" in line:
        version = re.split('[()]', line)[1].replace(',','.')
        break

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def update_long_descr():
    README_path     = os.path.join(os.path.dirname(__file__), 'README.md')
    LONG_DESCR_path = os.path.join(os.path.dirname(__file__), 'LONG_DESCR')
    if ((not os.path.exists(LONG_DESCR_path))
                          or os.path.getmtime(README_path) > os.path.getmtime(LONG_DESCR_path)):
        try :
            subprocess.check_call(["pandoc", "-f", "markdown", '-t', 'rst', '-o', LONG_DESCR_path, README_path], shell=False)
        except (OSError, subprocess.CalledProcessError) :
            print("setup.py::error:: pandoc command failed. Cannot update LONG_DESCR.txt from modified README.txt")
    return open(LONG_DESCR_path).read()

def read_long_descr():
    LONG_DESCR_path = os.path.join(os.path.dirname(__file__), 'LONG_DESCR')
    return open(LONG_DESCR_path).read()

class sdist_subclass(sdist) :
    def run(self):
        # Make sure the compiled Cython files in the distribution are up-to-date
        update_long_descr()
        sdist.run(self)


setup( name = "py_find_1st",
       version = version,
       packages = ['utils_find_1st'],
       ext_package = 'utils_find_1st',
       ext_modules = ext_modules,
       author = "A. Roebel",
       description = "Numpy extension module for efficient search of first array index that compares true",
       cmdclass = {'build_ext': build_ext_subclass, "sdist": sdist_subclass },
       author_email = "axel.dot.roebel@ircam.dot.fr",
        long_description = read_long_descr(),
        license = "GPL",
        url = "http://forge.ircam.fr/p/py_find_1st",
        keywords = "numpy,extension,find",
        classifiers = [
            "Topic :: Software Development :: Libraries :: Python Modules",
            "Programming Language :: Python :: 3",
            "Development Status :: 5 - Production/Stable",
            "License :: OSI Approved :: GNU General Public License (GPL)",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: POSIX :: Linux",
            "Operating System :: Microsoft :: Windows",
        ],
       
       # don't install as zip file because those cannot be analyzed by pyinstaller
       zip_safe = False,
    )
