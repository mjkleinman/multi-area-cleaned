# Recurrent neural network models of multi-area computation underlying decision-making

This Python package trains and analyzes multi-area RNNs. It trains RNNs by expanding on the pycog repository (https://github.com/frsong/pycog). This was written using Python 2.7.17.

- [Overview](#overview)
- [Installation Guide](#installation-guide)
- [Examples](#examples)

# Overview
We train and analyze multi-area RNNs.


# Installation Guide:
In a virtual environment, using Python 2.7.17, install the dependencies

1. Create a virtual environment
```
$ virtualenv --system-site-packages -p python2.7 your-virtual-env-name
$ source your-virtual-env-name/bin/activate
```

When you are finished working in the virtual environment, run:
```
$ deactivate
```

2. Install the dependencies
```
pip install -r requirements.txt
```

# Examples
To generate the figures, run the Jupyter notebook:

```
sims/Revision (Finalized).ipynb
sims/Revision [exemplar].ipynb
dynamics.ipynb
```

These `sims/Revision (Finalized).ipynb` used saved values for the hyperparameter sweeps. The mutual information and dpca variance values are generated using:

```
python sims/get_mutualinfo_vals.sh
python sims/get_dpca_vals.sh
```

The null/potent values are generated using `python null_potent_dpca.py`.

A network is trained by running
```
python do.py models/2020-04-10_cb_simple_3areas.py train
```

Multiple networks are trained by running:
```
python sims/three_rnn_train.py
```

Please send any questions about the code to michael.kleinman@ucla.edu


