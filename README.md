# datasci207_final_project
# Predicting Power Consumption of Generative AI Workloads

UC Berkeley, MIDS 207 Final Project

## Team

* Jiayi Hou (jhou12@berkeley.edu)
* Tom Kennedy (tken2556@berkeley.edu)
* Alex Mwamsindo (ammwamsindo@berkeley.edu)
* Yera Smagulov (ysmagulov@berkeley.edu)

## Overview

Can we predict power draw for generative AI jobs from signals we can observe before or during a run, such as training vs. inference, LLM vs. image generation, and node count? Data center power and cooling get sized before the job mix is known, so accurate forecasts for modern GPU workloads matter for infrastructure planning.

## Data

* **NLR Generative AI Workload Power Profiles** (H100 GPU data center, Vercellino et al., 2026): https://data.nlr.gov/submissions/312
* **Energy Efficient AI System Performance** (Kaggle, 7,167 records): https://www.kaggle.com/datasets/colabsss/energy-efficient-ai-system-performance-dataset

## Planned Approach

* Baselines: linear regression and regularized regression
* Models: gradient boosting (XGBoost) and a neural network
* Evaluation: MAE and RMSE, plus peak power error, with held out job types and configurations to test generalization

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

* Vercellino et al. (2026). Measurement of generative AI workload power profiles for whole facility data center infrastructure planning. https://arxiv.org/abs/2604.07345
* Lin et al. (2020). An artificial neural network approach to power consumption model construction for servers in cloud data centers. IEEE TSUSC.
* Chuang et al. (2023). ECDX: Energy consumption prediction model based on distance correlation and XGBoost for edge data center. Information Sciences.
