#! /usr/bin/env python
from __future__ import print_function
from ez_setup import use_setuptools
use_setuptools()

from setuptools import setup
from distutils.extension import Extension
from distutils.command.build_ext import build_ext
import numpy as np
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
                     include_dirs=[np.get_include()])


ext_modules=[find_1st_ext]
    
setup( name = "as_pysrc_utils_find_1st",
       version = "1.0",
       packages = ['utils_find_1st'],
       ext_package = 'utils_find_1st',
       ext_modules = ext_modules,
       author = "A. Roebel",
       author_email = "axel.roebel@ircam.fr",
       description = "Extension modules used in as_pysrc.utils for efficient search of first array index that compares true",
       license = "Copyright IRCAM",
       keywords = "",
       cmdclass = {'build_ext': build_ext_subclass }, 
       # don't install as zip file because those cannot be analyzed by pyinstaller
       zip_safe = False,
    )
