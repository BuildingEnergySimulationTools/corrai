Welcome to Corrai's documentation!
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
-------------

.. code-block:: python

    import pandas as pd

    from corrai.base.parameter import Parameter
    from corrai.sensitivity import SobolSanalysis, MorrisSanalysis
    from corrai.base.model import Ishigami

    SIMULATION_OPTIONS = {
        "start": "2009-01-01 00:00:00",
        "end": "2009-01-01 05:00:00",
        "timestep": "h",
    }

    PARAMETER_LIST = [
        Parameter("par_x1", (-3.14159265359, 3.14159265359), model_property="x1"),
        Parameter("par_x2", (-3.14159265359, 3.14159265359), model_property="x2"),
        Parameter("par_x3", (-3.14159265359, 3.14159265359), model_property="x3"),
    ]

    # Configure a Sobol sensitivity analysis
    sobol = SobolSanalysis(
        parameters=PARAMETER_LIST,
        model=Ishigami(),
        simulation_options=SIMULATION_OPTIONS,
    )

    # Draw sample, and run simulations
    sobol.add_sample(15**2, simulate=True, n_cpu=1, calc_second_order=True)

    # Corrai works for models that returns time series
    # Ishigami model here will return the same value for the given parameters
    # from START to END at 1h timestep
    sobol.analyze('res', method="mean")["mean_res"]

    # Default aggregation method is mean value of the timeseries
    sobol.plot_bar('res')

    # Display 2nd order matrix for parameters interaction
    sobol.plot_s2_matrix('res')

    # Display mean output values of the sample as hist
    sobol.sampler.sample.plot_hist('res')

    # Compute dynamic sensitivity analisys at plot
    # Obviously, in this example indexes value do not vary
    sobol.plot_dynamic_metric('res', sensitivity_metric="ST", freq="h")



.. toctree::
   :maxdepth: 2
   :caption: User Guide

   api_reference/index

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