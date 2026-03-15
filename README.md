# CT-DiffNet
**CT-DiffNet** is a deep learning framework for predicting atomistic migration barriers in complex concentrated solid solutions (e.g., Ni-based superalloys).

This repository contains the core models, training scripts, and baseline comparisons to reproduce our binary-to-ternary extrapolation tasks (e.g., training on Ni-Al and Ni-Re to predict Ni-Al-Re).

## 📂 Repository Structure

- `ablation_studies/`: Scripts for architectural ablation tests.
- `data/`: Main data for training and evaluation.
- `models/`: Core neural network and baseline architectures (`CNNTransformer3D`).

## 🛠️ Installation

Create a virtual environment and install the required packages:

```bash
conda create -n ctdiffnet python=3.9
conda activate ctdiffnet
pip install -r requirements.txt
```

## 🚀 Quick Start

1. Train and Evaluate CT-DiffNet:

Bash

python models/CT-diffNet*.py

2. Run XGBoost Baseline:
   
Bash

python models/XGB*.py

## 📜 Citation

If you use this code, please cite our paper:

in submission
