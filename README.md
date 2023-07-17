<p align="center">
  <img src="https://raw.githubusercontent.com/BuildingEnergySimulationTools/corrai/main/logo_corrai.svg" alt="CorrAI" width="200"/>
</p>

[![Static Badge](https://img.shields.io/badge/version-0.1.0-orange)](https://pypi.org/project/corrai/)
[![Static Badge](https://img.shields.io/badge/python-3.8_%7C_3.11-blue)](https://pypi.org/project/corrai/)
[![codecov](https://codecov.io/gh/BuildingEnergySimulationTools/corrai/branch/main/graph/badge.svg?token=F51O9CXI61)](https://codecov.io/gh/BuildingEnergySimulationTools/corrai)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)


## Measured Data Exploration for Physical and Mathematical Models
This Python library is designed to handle measured data from test benches or Building Energy Management Systems (BEMS). 
It offers physical model calibration frameworks and original AI methods.

## Features
The library includes the following features:
- **Data cleaning:** Based on [Pandas](https://pandas.pydata.org/), it uses [Scikit-learn](https://scikit-learn.org/stable/) framework
to simplify data cleaning process through the creation of pipelines for time series.
- **Data plotting:** Generates plots of measured data, visualizes gaps and cleaning methods effects.
- **Physical model calibration:** Provides base class to define calibration problem, uses [Pymoo](https://pymoo.org/) optimization methods for parameters identification.
- **Building usage modeling:** Generates time series of occupancy-related usage 
(Domestic Hot water consumption, grey water use...).
- **AI tools for HVAC FDD:** Includes artificial intelligence tools for 
Heating Ventilation and Air Conditioning (HVAC) systems fault detection and diagnostics (FDD).

## Getting started
The source code is currently hosted on GitHub at: https://github.com/BuildingEnergySimulationTools/corrai

Tutorials are available in the dedicated folder. 

Released version are available at the Python Package Index (PyPI):
```
# PyPI
pip install corrai
```