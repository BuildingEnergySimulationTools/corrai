<p align="center">
  <img src="https://github.com/BuildingEnergySimulationTools/corrai/blob/main/logo_corrai.svg" alt="CorrAI" width="200"/>
</p>

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
- **AI tools for HVAC FDD:** The library includes artificial intelligence tools for 
Heating Ventilation and Air Conditioning (HVAC) systems fault detection and diagnostics (FDD).
