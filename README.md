# Recurrent neural network models of multi-area computation underlying decision-making

This Python package trains and analyzes multi-area RNNs. It trains RNNs by expanding on the pycog repository (https://github.com/frsong/pycog). This was written using Python 2.7.17.

- [Overview](#overview)
- [Installation Guide](#installation-guide)
- [Examples](#examples)

# Overview
We train and analyze multi-area RNNs.


# Installation Guide:

1. Create a virtual environment. We recommend using virtualenvrapper (https://virtualenvwrapper.readthedocs.io/en/latest/)
```

$ pip install virtualenvwrapper
$ mkvirtualenv -p python2.7 your-virtual-env-name
```
2. Add `multi-area-cleaned` to path

```
$ add2path \path\to\multi-area-cleaned
```

3. Install the dependencies
```
pip install -r requirements.txt
```

When you are finished working in the virtual environment, run:
```
$ deactivate
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

The null/potent values are generated using `python sims/null_potent_dpca.py`.

A network is trained by running
```
python examples/do.py examples/models/2020-04-10_cb_simple_3areas.py train
```

Multiple networks are trained by running: `python sims/three_rnn_train.py` with the parameter configuration defined in the dictionary `cfg_mk.py`

Please send any questions about the code to michael.kleinman@ucla.edu


