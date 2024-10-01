PyRASA - Spectral parameterization in python based on IRASA
===========================================================

.. image:: https://www.repostatus.org/badges/latest/wip.svg
   :target: https://www.repostatus.org/#wip
   :alt: Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.

.. image:: https://img.shields.io/badge/License-BSD_2--Clause-orange.svg
   :target: https://opensource.org/licenses/BSD-2-Clause
   :alt: License

.. image:: http://www.mypy-lang.org/static/mypy_badge.svg
   :target: http://mypy-lang.org/
   :alt: Checked with mypy

.. image:: https://coveralls.io/repos/github/schmidtfa/pyrasa/badge.svg?branch=main
   :target: https://coveralls.io/github/schmidtfa/pyrasa?branch=main
   :alt: Coverage Status


PyRASA is a Python library designed to separate and parametrize aperiodic (fractal) and periodic (oscillatory) components in time series data based on the IRASA algorithm (Wen & Liu, 2016).

Features
--------

- **Aperiodic and Periodic Decomposition:** Utilize the IRASA algorithm to decompose power spectra into aperiodic and periodic components, enabling better interpretation of neurophysiological signals.
- **Time Resolved Spectral Parametrization:** Perform time resolved spectral parametrization, allowing you to track changes in spectral components over time.
- **Support for Raw and Epoched MNE Objects:** PyRASA provides functions designed for both continuous (Raw) and event-related (Epochs) data, making it versatile for various types of EEG/MEG analyses.
- **Consistent Ontology:** PyRASA uses the same jargon to label parameters as specparam, the most commonly used tool to parametrize power spectra, to allow users to easily switch between tools depending on their needs, while keeping the labeling of features consistent.
- **Custom Aperiodic Fit Models:** In addition to the built-in "fixed" and "knee" models for aperiodic fitting, users can specify their custom aperiodic fit functions, offering flexibility in how aperiodic components are modeled.

Documentation
-------------

Documentation for PyRASA, including detailed descriptions of functions, parameters, and examples, will soon be available `here`_.

.. _here: https://github.com/schmidtfa/pyrasa

Installation
------------

To install the latest stable version of PyRASA, you can soon use pip::

   $ pip install pyrasa

or conda::

   $ conda install pyrasa

Dependencies
------------

PyRASA has the following dependencies:

- **Core Dependencies:**
  - `numpy <https://github.com/numpy/numpy>`_
  - `scipy <https://github.com/scipy/scipy>`_
  - `pandas <https://github.com/pandas-dev/pandas>`_

- **Optional Dependencies for Full Functionality:**
  - `mne <https://github.com/mne-tools/mne-python>`_: Required for directly working with EEG/MEG data in `Raw` or `Epochs` formats.

Example Usage
-------------

Decompose spectra into periodic and aperiodic components::

   from pyrasa.irasa import irasa

   irasa_out = irasa(sig, 
                     fs=fs, 
                     band=(.1, 200), 
                     psd_kwargs={'nperseg': duration*fs, 
                                 'noverlap': duration*fs*overlap
                                },
                     hset_info=(1, 2, 0.05))

.. image:: https://raw.githubusercontent.com/schmidtfa/pyrasa/main/simulations/example_knee.png
   :alt: Example knee image


Extract periodic parameters::

   irasa_out.get_peaks()

+-----------+-----+--------+--------+
|  ch_name  |  cf |   bw   |   pw   |
+===========+=====+========+========+
|     0     | 9.5 | 1.4426 | 0.4178 |
+-----------+-----+--------+--------+

Extract aperiodic parameters::

   irasa_out.fit_aperiodic_model(fit_func='knee').aperiodic_params

+-----------+--------+--------------+--------------+-----------+--------------------+--------+-----------+
|  Offset   |  Knee  | Exponent_1   | Exponent_2   | fit_type  | Knee Frequency (Hz) |   tau  |  ch_name |
+===========+========+==============+==============+===========+====================+========+===========+
| 1.737e-16 | 60.94  | 0.0396       | 1.4727       | knee      | 14.131             | 0.0113 |     0     |
+-----------+--------+--------------+--------------+-----------+--------------------+--------+-----------+

And the goodness of fit::

   irasa_out.fit_aperiodic_model(fit_func='knee').gof

+------------+------------+------------+------------+-----------+-----------+
|     mse    | r_squared  |     BIC    |     AIC    | fit_type  |  ch_name  |
+============+============+============+============+===========+===========+
|  0.000051  | 0.999751   | -3931.840  | -3947.806  | knee      |     0     |
+------------+------------+------------+------------+-----------+-----------+

How to Contribute
-----------------

Contributions to PyRASA are welcome! Whether it's raising issues, improving documentation, fixing bugs, or adding new features, your help is appreciated. Please refer to the `CONTRIBUTING.md <CONTRIBUTING.md>`_ file for more information on how to get involved.

Reference
---------

If you are using IRASA, please cite the smart people who came up with the algorithm:

Wen, H., & Liu, Z. (2016). Separating fractal and oscillatory components in the power spectrum of neurophysiological signal. *Brain Topography*, 29, 13-26. https://doi.org/10.1007/s10548-015-0448-0

If you are using PyRASA, it would be nice if you could additionally cite us (whenever the paper is finally ready):

Schmidt F., Hartmann T., & Weisz, N. (2049). PyRASA - Spectral parameterization in python based on IRASA. *SOME JOURNAL THAT LIKES US*
