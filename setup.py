# -*- coding: utf-8 -*-

import os.path
from codecs import open

from setuptools import setup

# find the location of this file
this_directory = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get the current version number from inside the module
with open(os.path.join(this_directory, 'pyrasa', 'version.py')) as version_file:
    exec(version_file.read())

with open("requirements.txt") as requirements_file:
    install_requires = requirements_file.read().splitlines()

setup(
    name='PyRASA',
    version=__version__,
    packages=['pyrasa'],
    url='https://github.com/schmidtfa/pyrasa',
    license='BSD (2 clause)',
    author='Fabian Schmidt & Thomas Hartmann',
    author_email='schmidtfa91@gmail.com',
    description='Spectral parametrization based on IRASA',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=install_requires,
    tests_require = ['pytest'],
    extras_require = {
        'irasa_mne' : ['mne']
    },
    classifiers=[
        'Development Status :: 1 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
    ],
    keywords=['spectral parametrization', 'oscillations', 'power spectra', '1/f',],
)