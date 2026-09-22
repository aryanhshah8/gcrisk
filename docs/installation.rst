Installation
============

Requirements
------------

* Python >= 3.10
* NumPy >= 1.24, SciPy >= 1.10, pandas >= 2.0, matplotlib >= 3.7

Install
-------

.. code-block:: bash

   git clone https://github.com/aryanhshah8/gcrisk.git
   cd gcrisk
   pip install -e ".[dev]"
   python scripts/download_data.py

TOPAS (optional)
----------------

The proton therapy benchmark requires OpenTOPAS-RBE on PATH.
The checked-in reference CSV ``data/topas/proton_water_v79_reference.csv``
can be used without running TOPAS (``--skip-run`` flag).
