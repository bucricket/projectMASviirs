#!/usr/bin/env python

from __future__ import print_function
import subprocess
import os
import distutils

# set project base directory structure
base = os.getcwd()
    
try:
    from setuptools import setup
    setup_kwargs = {'entry_points': {'console_scripts':['processviirs=processviirs.processVIIRS:main']}}
except ImportError:
    from distutils.core import setup
    setup_kwargs = {'scripts': ['bin/processviirs']}
    
from processviirs import __version__

#=====build DMS binaries===============================
# get Anaconda root location
p = subprocess.Popen(["conda", "info", "--root"],stdout=subprocess.PIPE)
out = p.communicate()
condaPath = out[0][:-1]
    
prefix  = os.environ.get('PREFIX')
processDi = os.path.abspath(os.path.join(prefix,os.pardir))
processDir = os.path.join(processDi,'work')
libEnv = os.path.join(prefix,'lib')
binEnv = os.path.join(prefix,'bin')
libDir = os.path.join(processDir,'source','lib')
binDir = os.path.join(processDir,'source','bin')
distutils.dir_util.copy_tree(binDir,binEnv)

#===compile all binaries======================

print ("installing apply_water_mask...")
mkPath = os.path.join(processDir,'source','apply_water_mask')
os.chdir(mkPath)
subprocess.call(["scons","-Q","--prefix=%s" % prefix,"install"])
subprocess.call(["scons","-c"])

print ("installing atmos_corr...")
mkPath = os.path.join(processDir,'source','atmos_corr')
os.chdir(mkPath)
subprocess.call(["scons","-Q","--prefix=%s" % prefix,"install"])
subprocess.call(["scons","-c"])

print ("installing gen_lookup...")
mkPath = os.path.join(processDir,'source','gen_lookup')
os.chdir(mkPath)
subprocess.call(["scons","-Q","--prefix=%s" % prefix,"install"])
subprocess.call(["scons","-c"])

print ("installing grid_CM...")
mkPath = os.path.join(processDir,'source','grid_CM')
os.chdir(mkPath)
subprocess.call(["scons","-Q","--prefix=%s" % prefix,"install"])
subprocess.call(["scons","-c"])

print ("installing grid_I5...")
mkPath = os.path.join(processDir,'source','grid_I5')
os.chdir(mkPath)
subprocess.call(["scons","-Q","--prefix=%s" % prefix,"install"])
subprocess.call(["scons","-c"])

print ("installing merge_day_night...")
mkPath = os.path.join(processDir,'source','merge_day_night')
os.chdir(mkPath)
subprocess.call(["scons","-Q","--prefix=%s" % prefix,"install"])
subprocess.call(["scons","-c"])

print ("installing predict_dtrad...")
mkPath = os.path.join(processDir,'source','predict_dtrad')
os.chdir(mkPath)
subprocess.call(["scons","-Q","--prefix=%s" % prefix,"install"])
subprocess.call(["scons","-c"])
os.chdir(base)

#=============setup the python scripts============================



setup(
    name="projectmasviirs",
    version=__version__,
    description="prepare lai data for input to ALEXI",
    author="Mitchell Schull",
    author_email="mitch.schull@noaa.gov",
    url="https://github.com/bucricket/projectMASviirs.git",
#    py_modules=['processviirs'],
    packages= ['processviirs'],
    platforms='Posix; MacOS X; Windows',
    license='BSD 3-Clause',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        # Uses dictionary comprehensions ==> 2.7 only
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: GIS',
    ],  
    **setup_kwargs
)

