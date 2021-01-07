"""
Setup of exploratory grasping codebase.
"""

import os
from setuptools import setup, Extension
import numpy as np

# load README.md as long_description
long_description = ''
if os.path.exists('README.md'):
    with open('README.md', 'r') as f:
        long_description = f.read()

setup_requirements = [
    'Cython',
    'numpy'
]

requirements = [
    'pyyaml',                   # Reading yaml config files
    'tqdm',                     # For pretty progress bars
    'pandas',                   # For visualization of results
    'seaborn',                  # For visualization of results
    'matplotlib',               # For visualization of results
    'trimesh'                   # For viewing stable poses
]

libraries = []
if os.name == 'posix':
    libraries.append('m')

extra_compile_args = ["-O3", "-ffast-math", "-march=native"]
extra_link_args = []

extensions = [
    Extension("UCRL.evi.evi",
              ["UCRL/evi/evi.pyx"],
              include_dirs=[np.get_include()],
              libraries=libraries,
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
    Extension("UCRL.evi._max_proba",
              ["UCRL/evi/_max_proba.pyx"],
              include_dirs=[np.get_include()],
              libraries=libraries,
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
    Extension("UCRL.evi._utils",
              ["UCRL/evi/_utils.pyx"],
              include_dirs=[np.get_include()],
              libraries=libraries,
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
    Extension("UCRL.evi.free_evi",
              ["UCRL/evi/free_evi.pyx"],
              include_dirs=[np.get_include()],
              libraries=libraries,
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
    Extension("UCRL.cython.ExtendedValueIteration",
              ["UCRL/cython/ExtendedValueIteration.pyx"],
              include_dirs=[np.get_include()],
              libraries=libraries,
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
    Extension("UCRL.cython.max_proba",
              ["UCRL/cython/max_proba.pyx"],
              include_dirs=[np.get_include()],
              libraries=libraries,
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
    Extension("UCRL.evi.prova",
              ["UCRL/evi/prova.pyx"],
              include_dirs=[np.get_include()],
              libraries=libraries,
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
    Extension("UCRL.evi._free_utils",
              ["UCRL/evi/_free_utils.pyx"],
              include_dirs=[np.get_include()],
              libraries=libraries,
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
    Extension("UCRL.evi.scevi",
              ["UCRL/evi/scevi.pyx"],
              include_dirs=[np.get_include()],
              libraries=libraries,
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args)
]

setup(
    name='grasp_exploration',
    version='0.0.1',
    description='Exploratory grasping project code',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Grasp Exploration Authors',
    author_email='graspexplorationauthors@gmail.com',
    license='MIT',
    url='',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        'Natural Language :: English',
        'Topic :: Scientific/Engineering'
    ],
    packages=['grasp_exploration'],
    setup_requires=setup_requirements,
    install_requires=requirements,
    ext_modules=extensions
)
