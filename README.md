## PyRASA

[![License](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![codecov](https://codecov.io/github/schmidtfa/pyrasa/graph/badge.svg?token=4AL4NJUQU6)](https://codecov.io/github/schmidtfa/pyrasa)


Pyrasa is a repository that is build around the IRASA algorithm (Wen & Liu, 2016) to parametrize power and coherence spectra.
WARNING - This repository is under heavy development and core functionality may change on a daily basis...

Wen, H., & Liu, Z. (2016). Separating fractal and oscillatory components in the power spectrum of neurophysiological signal. Brain topography, 29, 13-26.

### Documentation
Documentation for PyRASA will soon be available [here].


### Installation
To install the latest stable version of PyRASA, you can soon use pip:

``` $ pip install pyrasa ```

or conda:

``` $ conda install pyrasa ```

### Dependencies
The minimum required dependencies to run PyRASA are:

[numpy](https://github.com/numpy/numpy)
[scipy](https://github.com/scipy/scipy)
[pandas](https://github.com/pandas-dev/pandas)

For full functionality, some functions require:

[mne](https://github.com/mne-tools/mne-python)

TODO: We should maybe test which versions of numpy and pandas work and force this more explicitely

### How to contribute
Please take a look at the [CONTRIBUTING.md](CONTRIBUTING.md) file for more information.


### Reference

If you are using the IRASA algorithm it probably makes sense to cite the smart people who came up with it:

Wen, H., & Liu, Z. (2016). Separating fractal and oscillatory components in the power spectrum of neurophysiological signal. Brain topography, 29, 13-26.

If you are using PyRASA it would be nice, if you could additionally cite us:

Schmidt F., Hartmann T., & Weisz, N. (2049). PyRASA - Spectral parameterization in python based on IRASA. EVERY JOURNAL JUST NOT FRONTIERS