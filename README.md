<p align="center">
  <img src="https://raw.githubusercontent.com/BuildingEnergySimulationTools/corrai/main/logo_corrai.svg" alt="CorrAI" width="200"/>
</p>

[![PyPI](https://img.shields.io/pypi/v/corrai?label=pypi%20package)](https://pypi.org/project/corrai/)
[![Static Badge](https://img.shields.io/badge/python-3.10_%7C_3.12-blue)](https://pypi.org/project/corrai/)
[![codecov](https://codecov.io/gh/BuildingEnergySimulationTools/corrai/branch/main/graph/badge.svg?token=F51O9CXI61)](https://codecov.io/gh/BuildingEnergySimulationTools/corrai)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Measured Data Exploration for Physical and Mathematical Models

This Python library is designed to handle measured data from test benches or Building
Energy Management Systems (BEMS).
It offers physical model calibration frameworks and original AI methods.

## Features

The library includes the following features:

- **Data cleaning:** Based on [Pandas](https://pandas.pydata.org/), it
  uses [Scikit-learn](https://scikit-learn.org/stable/) framework
  to simplify data cleaning process through the creation of pipelines for time series.
- **Data plotting:** Generates plots of measured data, visualizes gaps and cleaning
  methods effects.
- **Physical model calibration:** Provides base class to define calibration problem,
  uses [Pymoo](https://pymoo.org/) optimization methods for parameters identification.
- **Building usage modeling:** Generates time series of occupancy-related usage
  (Domestic Hot water consumption, grey water use...).
- **AI tools for HVAC FDD:** Includes artificial intelligence tools for
  Heating Ventilation and Air Conditioning (HVAC) systems fault detection and
  diagnostics (FDD).

## Getting started

The source code is currently hosted on GitHub
at: https://github.com/BuildingEnergySimulationTools/corrai

Tutorials are available in the dedicated folder.

Released version are available at the Python Package Index (PyPI):

```
# PyPI
pip install corrai
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



