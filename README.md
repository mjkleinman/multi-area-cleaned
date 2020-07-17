# Recurrent neural network models of multi-area computation underlying decision-making

This Python package trains and analyzes multi-area RNNs. It trains RNNs by expanding on the pycog repository (https://github.com/frsong/pycog).

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)

# Overview
We train and analyze multi-area RNNs.

# Documenation

# System Requirements
The code mainly depends on the Python scientific stack.

```
numpy
Theano
matplotlib
```

# Installation Guide:
In a virtual environment, using Python 2.7.17, install the dependencies

```
pip install -r requirements.txt
```
# examples



```
python do.py models/2020-04-10_cb_simple_3areas.py train
```

To run the hyperparameter sweeps to get mutual information and dpca variance:
```
python sims/get_mutualinfo_vals.sh
python sims/get_dpca_vals.sh
```
To generate the figures, run:

```
sims/Revision (Finalized).ipynb
```



