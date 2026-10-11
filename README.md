# Predicting Power Consumption of Commerical Buildings in the U.S.

UC Berkeley, MIDS 207 Final Project

## Team

* Jiayi Hou (jhou12@berkeley.edu)
* Tom Kennedy (tken2556@berkeley.edu)
* Alex Mwamsindo (ammwamsindo@berkeley.edu)
* Yera Smagulov (ysmagulov@berkeley.edu)

## Overview


## Data

* US EIA, 2018 CBECS Survey Data, www.eia.gov, 2023. Retrieved from https://www.eia.gov/consumption/commercial/data/2018/index.php?view=microdata on October 11, 2026.

## Planned Approach

* Baselines: linear regression
* Models: gradient boosting and FNN
* Evaluation: MAE and RMSE, plus peak power error

## Repo Structure

```
data/         raw and processed data (not tracked)
notebooks/    exploration and modeling notebooks
src/          preprocessing, features, and model code
results/      figures and metrics
```

## Getting Started

```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
```

## References

* US EIA, 2018 CBECS Survey Data, www.eia.gov, 2023. Retrieved from https://www.eia.gov/consumption/commercial/data/2018/index.php?view=microdata on October 11, 2026.
* Robinson, C., Dilkina, B., Hubbs, J., Zhang, W., Guhathakurta, S., Brown, M. A., & Pendyala, R. M. (2017). Machine learning approaches for estimating commercial building energy consumption. Applied Energy, 208, 889–904. https://doi.org/10.1016/j.apenergy.2017.09.060