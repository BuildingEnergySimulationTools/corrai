<p align="center">
  <img src="https://raw.githubusercontent.com/BuildingEnergySimulationTools/corrai/main/logo_corrai.svg" alt="CorrAI" width="200"/>
</p>


[![PyPI](https://img.shields.io/pypi/v/corrai?label=pypi%20package)](https://pypi.org/project/corrai/)
[![Static Badge](https://img.shields.io/badge/python-3.10_%7C_3.12-blue)](https://pypi.org/project/corrai/)
[![codecov](https://codecov.io/gh/BuildingEnergySimulationTools/corrai/branch/main/graph/badge.svg?token=F51O9CXI61)](https://codecov.io/gh/BuildingEnergySimulationTools/corrai)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Documentation Status](https://readthedocs.org/projects/corrai/badge/?version=latest)](https://corrai.readthedocs.io/en/latest/?badge=latest)

---

# Corrai: A Framework for Modeling, Sampling, Optimization and Surrogates

**Corrai** is a Python library for scientific exploration of complex systems.  
It provides a unified framework for **model definition**, **parameterization**, **sampling**, **optimization**, **sensitivity analysis**, and **surrogate modeling**.  

While originally motivated by building energy research, Corrai is **domain-independent** and can be applied to any problem requiring model calibration, uncertainty quantification, or reduced-order modeling.

---

## Main Features

- **Sampling methods**
  - Generate experimental designs using built-in samplers: Sobol, Latin Hypercube, FAST, Morris, random, or custom samplers (`Sample` and `Sampler`). 
  - Easily connect samples with model parameters to prepare sensitivity or optimization studies.


- **Sensitivity & uncertainty analysis**
  - Built-in analyzers for variance-based (Sobol), screening (Morris), and FAST methods (sensitivity.py).([SAlib](https://salib.readthedocs.io/en/latest/)).  
  - Quantify the influence of each parameter on model outputs.  
  

- **Optimization and calibration**  
  - Single and multi-objective parameter identification and model calibration (`optimize.py`).  
  - Integrated with evolutionary and gradient-based optimizers ([pymoo](https://pymoo.org/)).  


- **Surrogate modeling**  
  - Train ML-based surrogates (linear, polynomial, SVR, random forest, MLP, …).
  - Grid-search hyperparameter tuning ([sklearn](https://scikit-learn.org/stable/)).
  - Evaluate accuracy with statistical metrics (`nmbe`, `cv_rmse`).   


- **Visualization support**  
  - Plotting utilities to inspect results, sensitivity indices, surrogate accuracy, etc.


- **Model abstraction**  
  - Define analytical, external simulator, numerical, or FMU-driven models (classes `Model` and `ModelicaFmuModel`).  
  - Associate parameters with model properties such as domain, initial values, continuity (class `Parameter`)

---
## Getting started

### Installation 
Corrai requires Python 3.10 or above.
The recommended way to install corrai is via pip:

```
pip install corrai
```

This will install python-tide and all its dependencies.


### Quick example

```python
    import pandas as pd

from corrai.base.parameter import Parameter
from corrai.sensitivity import SobolSanalysis, MorrisSanalysis
from corrai.base.model import IshigamiDynamic

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
    model=IshigamiDynamic(),
    simulation_options=SIMULATION_OPTIONS,
)

# Draw sample, and run simulations
sobol.add_sample(15 ** 2, simulate=True, n_cpu=1, calc_second_order=True)

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
```

### Sponsors
<table style="border-collapse: collapse;">
<tr style="border: 1px solid transparent;">
<td width="150" >
<img src="https://upload.wikimedia.org/wikipedia/commons/b/b7/Flag_of_Europe.svg" alt="eu_flag" width="150"/>
</td>
<td>
The development of this library has been supported by METABUILDING LABS Project, which
has received funding from the European Union’s Horizon 2020 Research and Innovation
Programme under Grant Agreement No. 953193. The sole responsibility for the content of
this library lies entirely with the author’s view. The European Commission is not
responsible for any use that may be made of the information it contains. 
</td>
</tr>
</table>

