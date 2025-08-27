Welcome to python-tide's documentation!
====================================

.. image:: ../logo_corrai.svg
   :width: 200px
   :align: center

**Corrai** is a Python library for efficient model exploration and sampling.
It provides tools to:

- Define parameters and generate samples
- Perform sensitivity and uncertainty analysis
- Explore parameter spaces through single-objective and multi-objective optimization
- Build surrogate models with state-of-the-art machine learning techniques to reduce computation time

Corrai is built on top of popular libraries such as **scikit-learn**, **pandas**, **SALib**, and **pymoo**.
The source code for python-tide is available on `GitHub <https://github.com/BuildingEnergySimulationTools/corrai>`_

Installation
------------

Corrai requires Python 3.10 or later.
The recommended way to install corrai is via pip:

.. code-block:: bash

   pip install corrai

This will install python-tide and all its dependencies.

Quick Example
------------

.. code-block:: python

   import pandas as pd
   import numpy as np
   from tide.plumbing import Plumber

   # Create sample data
   data = pd.DataFrame({
       "temp__°C__zone1": [20, 21, np.nan, 23],
       "humid__%HR__zone1": [50, 55, 60, np.nan]
   }, index=pd.date_range("2023", freq="h", periods=4))

   # Define pipeline
   pipe_dict = {
       "pre_processing": {"°C": [["ReplaceThreshold", {"upper": 25}]]},
       "common": [["Interpolate", ["linear"]]]
   }

   # Create plumber and process data
   plumber = Plumber(data, pipe_dict)
   corrected = plumber.get_corrected_data()

   # Analyze gaps
   gaps = plumber.get_gaps_description()

   # Plot data
   fig = plumber.plot(plot_gaps=True)
   fig.show()

Dependencies
------------

- "numpy>=1.22.4, <2.0"
- "pandas>=2.1.0, <3.0"
- "scipy>=1.9.1, <2.0"
- "scikit-learn>=1.2.2, <2.0"
- "pymoo>=0.6.0.1"
- "salib>=1.4.7"
- "fmpy>=0.3.6"
- "matplotlib>=3.5.1"
- "plotly>=5.3.1"
- "fastprogress>=1.0.3"

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 