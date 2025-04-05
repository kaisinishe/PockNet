# PockNet Project

[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/YourGithubName/PockNet#license)

## Overview

PockNet is a machine learning framework designed for binding site prediction using various models, including TabNet and XGBoost. The project leverages PyTorch Lightning for training and evaluation, and includes configurations for hyperparameter tuning and logging.

______________________________________________________________________

## Preprocessing

The preprocessing pipeline for this project is adapted from [p2rank](https://github.com/rdk/p2rank), a tool for protein-ligand binding site prediction. Special thanks to the p2rank team for their contributions to the preprocessing methodology.

______________________________________________________________________

## Installation

To set up the project on your local machine, follow these steps:

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or higher
- pip (Python package manager)
- [conda](https://docs.conda.io/en/latest/miniconda.html)

### Using Conda

1. Clone the repository:

   ```bash
   git clone https://github.com/YourGithubName/PockNet.git
   cd PockNet
   ```

2. Create a conda environment and install dependencies:

   ```bash
   conda env create -f environment.yaml -n pocknet
   conda activate pocknet
   ```

3. Install PyTorch according to your system and CUDA version:
   Follow the instructions at [PyTorch.org](https://pytorch.org/get-started/locally/).

### Verifying Installation

To verify the installation, run the following command to check if the tests pass:

```bash
pytest tests/
```

______________________________________________________________________

## TabNet Model

The TabNet model is implemented in `src/models/tabnet_binding_site_module.py`. It uses the TabNet architecture for binding site prediction and includes the following features:

- **Metrics**: Mean Squared Error (MSE), R2 Score, and Intersection over Union (IoU).
- **Loss Function**: Combines MSE, IoU loss, and a custom loss term.
- **Configurations**: Defined in `configs/model/tabnet_binding_site.yaml`.

### Key Hyperparameters

- `n_d`: Width of the decision prediction layer.
- `n_a`: Width of the attention embedding for each mask.
- `n_steps`: Number of steps in the architecture.
- `gamma`: Coefficient for feature reuse in masks.

### Training

To train the TabNet model, use the following command:

```bash
python src/train.py experiment=tabnet_chen11_bu48
```
______________________________________________________________________

## XGBoost Model

The XGBoost model is implemented in `src/models/xgboost_binding_site_module.py`. It wraps the XGBoost classifier in a PyTorch Lightning module and includes:

- **Metrics**: MSE, R2 Score, and IoU.
- **Custom Training Loop**: Fits the XGBoost model at the end of each epoch.
- **Configurations**: Defined in `configs/model/xgboost_binding_site.yaml`.

### Key Hyperparameters

- `learning_rate`: Step size shrinkage used in updates.
- `max_depth`: Maximum depth of a tree.
- `n_estimators`: Number of boosting rounds.
- `subsample`: Fraction of samples used for training each tree.
- `colsample_bytree`: Fraction of features used for training each tree.

### Training

To train the XGBoost model, use the following command:

```bash
python src/train.py experiment=xgboost_binding_site
```

---


### Results XGBoost and TabNet

The results of the XGBoost and TabNet compared with the p2rank evaluated on the test set bu48 are shown below. The results are based on the MSE, R2 Score, and IoU metrics.

| Model   | MSE   | R2    | IoU   |
|---------|-------|-------|-------|
| TabNet  | 0.096 | 0.341 | 0.24 |
| XGBoost | 0.042 | 0.361 | 0.23 |
| p2rank  | NaN | 0.324 | 0.21 |______________________________________________________________________

## Logging and Monitoring

The project supports logging with WandB, TensorBoard, and other tools. Configure the logger in the `configs/logger/` directory.
