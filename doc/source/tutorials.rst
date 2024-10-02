:orphan:
=========
Tutorials
=========

This section contains a number of tutorials, to get you started with PyRASA.

Introductory
------------

******************
1. Getting Started
******************

This notebook gets you familiar with the IRASA algorithm and shows you the basic functionality
of PyRASA.

  :doc:`Getting Started <../examples/basic_functionality>`

******************************
2. Improving your IRASA models
******************************

This notebook shows you how to improve your IRASA model fits.

  :doc:`Improve your IRASA <../examples/improving_irasa_models>`

****************************
3. Pitfalls when using IRASA
****************************

This notebook outlines common pitfalls when fitting IRASA models.

  :doc:`Pitfalls <../../examples/irasa_pitfalls>`


*********************
3. hset Optimization
*********************

IRASA comes only with a single hyperparameter - the set of up-/downsampling factors.
Here we introduce a method to optimize this hset to get the most out of your model.

  :doc:`Optimization <../../examples/hset_optimization>`

************
4. IRASA MNE
************

Are you analysing M/EEG data using MNE Python? You might be happy to hear that you can directly
apply IRASA to your raw or epoched data objects. Open the notebook to see how its done.

  :doc:`IRASA in MNE <../../examples/irasa_mne>`


***********************
4. Time Frequency IRASA
***********************

Did you know that IRASA can be used in the timefrequency domain for a time resolved spectral parametrization?
Open this notebook to see how its done.  

  :doc:`Time-Frequency IRASA <../../examples/irasa_sprint>`



Advanced
--------

*********************************
1. Custom Aperiodic Fit Functions
*********************************

PyRASA allows you to define your own functions to model aperiodic activity.
This notebook shows you how its done.

  :doc:`Custom Aperiodic models <../../examples/custom_fit_functions>`
