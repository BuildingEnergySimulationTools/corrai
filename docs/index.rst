Welcome to Corrai's documentation!
====================================

.. image:: ../logo_corrai.svg
   :width: 200px
   :align: center

**Corrai** is a Python library for scientific exploration of complex systems.
It provides a unified framework for **model definition**, **parameterization**, **sampling**, **optimization**, **sensitivity analysis**, and **surrogate modeling**.

While originally motivated by building energy research, Corrai is **domain-independent** and can be applied to any problem requiring model calibration, uncertainty quantification, or reduced-order modeling.

Main Features
-------------

- **Sampling methods**

  * Generate experimental designs using built-in samplers: Sobol, Latin Hypercube, FAST, Morris, random, or custom samplers (``Sample`` and ``Sampler``).
  * Easily connect samples with model parameters to prepare sensitivity or optimization studies.

- **Sensitivity & uncertainty analysis**

  * Built-in analyzers for variance-based (Sobol), screening (Morris), and FAST methods (``sensitivity.py``). See `SAlib <https://salib.readthedocs.io/en/latest/>`_.
  * Quantify the influence of each parameter on model outputs.

- **Optimization and calibration**

  * Multi-objective parameter identification and model calibration (``optimize.py``).
  * Integrated with evolutionary and gradient-based optimizers (`pymoo <https://pymoo.org/>`_).

- **Surrogate modeling**

  * Train ML-based surrogates (linear, polynomial, SVR, random forest, MLP, …).
  * Grid-search hyperparameter tuning (`scikit-learn <https://scikit-learn.org/stable/>`_).
  * Evaluate accuracy with statistical metrics (``nmbe``, ``cv_rmse``).

- **Visualization support**

  * Plotting utilities to inspect results, sensitivity indices, surrogate accuracy, etc.

- **Model abstraction**

  * Define analytical, external simulator, numerical, or FMU-driven models (classes ``Model`` and ``ModelicaFmuModel``).
  * Associate parameters with model properties such as domain, initial values, continuity (class ``Parameter``).

Installation
------------

Corrai requires Python 3.10 or above.
The recommended way to install corrai is via pip:

.. code-block:: bash

   pip install corrai

This will install python-tide and all its dependencies.

Quick Example
-------------

.. code-block:: python

    from corrai.base.parameter import Parameter
    from corrai.sensitivity import SobolSanalysis
    from corrai.base.model import Ishigami

    PARAMETER_LIST = [
        Parameter("par_x1", (-3.14159265359, 3.14159265359), model_property="x1"),
        Parameter("par_x2", (-3.14159265359, 3.14159265359), model_property="x2"),
        Parameter("par_x3", (-3.14159265359, 3.14159265359), model_property="x3"),
    ]

    # Configure a Sobol sensitivity analysis
    sobol = SobolSanalysis(
        parameters=PARAMETER_LIST,
        model=Ishigami(),
    )

    # Draw sample, and run simulations
    sobol.add_sample(2**10, simulate=True, n_cpu=1, calc_second_order=True)

    # Corrai works for models that returns either time series or series
    # depending on their static or dynamic nature
    # Ishigami is a static model (see is_dynamic attribute)
    sobol.analyze("res")

    # Default aggregation method is mean value of the timeseries
    sobol.plot_bar("res", sensitivity_metric="ST")

    # Display 2nd order matrix for parameters interaction
    sobol.plot_s2_matrix("res")

    # Display mean output values of the sample as hist
    sobol.sampler.sample.plot_hist("res")

    # Display parallel coordinate plot
    sobol.plot_pcp(["res"])


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