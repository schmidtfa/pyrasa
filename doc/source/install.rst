
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

 numpy >= 1.26
 pandas >= 2.1 
 scipy >= 1.12
 attrs 
 
and

.. code::

    python >= 3.11


Optional dependencies
=====================

Optionally you can combine PyRASA with MNE Python to better integrate spectral parametrization in your
M/EEG analysis workflow. If you already have an MNE Python installation running you can try to install PyRASA in the respective environment.
If you don't already have MNE Python in your environment you can install PyRASA including its optional dependency 
`mne <https://mne.tools/stable/index.html>` using either pip or conda-forge. 

Using pip
---------
.. code:: bash
    
    pip install "pyrasa[mne]"

Using conda-forge
-----------------
.. code:: bash
    
    conda install -c conda-forge pyrasa mne


For a more detailed instruction on how to configure your MNE Python installation please refer to the `mne installation guide <https://mne.tools/stable/install/manual_install.html#manual-install>`.






