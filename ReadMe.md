## 📚 Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Installation](#installation)  
4. [Quickstart](#quickstart)  
   - [Import and Instantiate](#1-import-and-instantiate)  
   - [Run Simulations](#2-run-simulations)  
5. [Repository Structure](#repository-structure)  
6. [Assumptions & Configuration](#assumptions--configuration)  
7. [Validation](#validation)  
8. [References](#references)
9. [Citation](#citation)  
10. [License](#license)

# Overview

`EVforecaster` uses the 2002 - 2023 UK National Travel Survey (UK-NTS) to generate domestic EV charging demand curves. The UK-NTS contains trip level data, such as: trip start time, trip end time, trip distance, trip location, etc. Although the majority of trips in the UK-NTS are likely to have been carried out by combustion engine vehicles (CEVs), this simulation-based approach assumes that electric vehicles (EVs) carried out the trips. Taking the trips as given, `EVforecaster` tries to model where, at what time and for how long EVs would charge to be able to successfully undertake the trips. Once this is modelled, the information is aggregated to plot weekly demand curves at 5-minute granularity. The simulation algorithms requires the following information:

* Each trip location is randomly allocated a charger and a charging rate (kW/hour) based on a distribution.
* Each individual is allocated a car, which is modelled as a battery size and efficiency due. Based on a distribution.
* An individual's charging decision is modelled using the probabilistic function developed in Pareschi et al. (2020).
* Most simulation parameters (such as those mentioned above) are centrally defined in `config.py`.

It is capable of creating annual weekly demand curves that aggregate all the weeks of the year, weekly demand curves for specific seasons (defined by week ranges) or weekly demand curves for each week of the year.

Additionally, it is able to compare weekly demand curves for the largest domestic UK-based EV charging pilot, the 2017 Electric Chargepoint Analysis (ECA) (DfT, 2018). It plots the mean simulated demand curve (and standard errors) alongside the ECA and a histrogram of $R^2$ values below.

Various experiments can be ran to test model performance, as compared to the ECA, for different parameter configurations. Some are available in `showcase.ipynb`

# Installation

* From GitHub


* From PyPi

TBC


# References

- Pareschi, G., et al. (2020). *Are travel surveys a good basis for EV models?* Applied Energy, 275.
- Department for Transport (2018). Electric Chargepoint Analysis 2017: Domestics. Statistical release. UK Government. https://www.gov.uk/government/statistics/electric-vehicle-chargepoint-analysis-2017