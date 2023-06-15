# SC-GNN-Explain

## Setup

To run the Jupyter notebook, a Python environment is necessary. My recommendation is using [MiniConda](https://docs.conda.io/en/latest/miniconda.html) to create a new environment. The following commands create a new environment called `sc-gnn-explain` and install all necessary packages and is tested on Ubuntu 22.04. LTS. When running Windows, the Windows Subsystem for Linux (WSL2) is recommended.

```bash
conda create -y -n sc-gnn-explain python=3.10.8
conda activate sc-gnn-explain
pip install -r requirements.txt
```

When using a GPU, PyTorch needs to be installed with CUDA support. The following commands installs PyTorch 1.13.0 with CUDA 11.7 support.

```bash
conda create -y -n sc-gnn-explain-gpu python=3.10.8
conda activate sc-gnn-explain-gpu
conda install -y -c conda-forge cudnn=8.4.1.50 cudatoolkit=11.7.0
pip install -r requirements-gpu.txt
```

## Overview

The repository consists mainly of the Jupyter notebook `explainability.ipynb`. Everything else is used to run the notebook. The `src`-directory contains a few snippets of the code used in my Master thesis for convenience. Note that some parts are deleted due to dependency issues so only the parts used in the notebook are guaranteed to work. The `data`-directory contains the intestine dataset in a memory-efficient format. The `results`-directory is used to save the results of the notebook and also contains the results of the run done by me to be able to analyze them without having to rerun the notebook.
