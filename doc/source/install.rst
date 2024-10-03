
==================
Installation Guide
==================
This section provides detailed information about installing PyRASA. 
Most of PyRASA's functionality is available with the basic requirements, 
but PyRASA also has an optional dependency to integrate PyRASA in your workflow, when using MNE Python.
This guide will cover both basic and fully-fledged PyRASA installs and several installation methods.

Stable
PyRASA can be installed either using pip or conda-forge.

Using pip
---------
.. code:: bash
    
    pip install pyrasa

Using conda-forge
-----------------
.. code:: bash
    
    conda install -c conda-forge pyrasa


Development
-----------

If you want to install the latest development version of PyRASA, use the following command:

.. code:: bash
    
    pip install git+https://github.com/schmidtfa/pyrasa


Dependencies
------------


Required dependencies
=====================
The required dependencies for installing PyRASA are:

.. code::

 numpy>=1.26, <3 
 pandas>=2.1, <3 
 scipy>=1.12
 attrs 
 
and

.. code::

    python>=3.11


Optional dependencies
=====================

Optionally you can combine PyRASA with MNE Python to better integrate spectral parametrization in your
M/EEG analysis workflow.

.. code::

    mne







