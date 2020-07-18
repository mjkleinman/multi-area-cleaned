# Recurrent neural network models of multi-area computation underlying decision-making

This Python package trains and analyzes multi-area RNNs. It trains RNNs by expanding on the pycog repository (https://github.com/frsong/pycog). Our package was written using Python 2.7.17.

- [Installation Guide](#installation-guide)
- [Examples](#examples)

# Installation Guide:
This should take only a couple minutes on a local computer.

1. Create a virtual environment. We recommend using virtualenvrapper (https://virtualenvwrapper.readthedocs.io/en/latest/)

```
$ pip install virtualenvwrapper
$ source /usr/local/bin/virtualenvwrapper.sh
$ mkvirtualenv -p python2.7 your-virtual-env-name
```

2. Add `multi-area-cleaned` to path

```
$ add2virtualenv /path/to/multi-area-cleaned
$ add2virtualenv /path/to/multi-area-cleaned/pycog
```

3. Install the dependencies
```
pip install -r requirements.txt
```
This step installs everything from jupyter notebook to matplotlib for plotting. Note these will be done inside your virtual environment and will not disrupt your base python installation.

When you are finished working in the virtual environment, run:
```
$ deactivate
```

# Examples
To generate the figures for the paper, run the following Jupyter notebooks. For `Revision_main.ipynb` and `Revision_exemplar.ipynb`, most of the cells rely on saved data and can be run immediately. Generating the PSTHs and PC plots takes a bit of time. `Dynamics_polished` should run in about 5 minutes. You may get occassional matplotlib warnings, but these can be generally be ignored.

```
sims/Revision_main.ipynb
sims/Revision_exemplar.ipynb
sims/dynamics_polished.ipynb
```

These `sims/Revision_main.ipynb` uses saved values for the hyperparameter sweeps. The mutual information and dpca variance values are generated using the following files. To run these scripts, update the paths (`path` and `rnn_datapath`) in `cfg_mk.py`. Note that these scripts take a few hours to run.

```
./sims/get_mutualinfo_vals.sh
./sims/get_dpca_vals.sh
```

The null/potent values are generated using `python sims/null_potent_dpca.py`.

A network is trained by running:
```
python examples/do.py examples/models/2020-04-10_cb_simple_3areas.py train
```

The different RNN modelfiles are in `examples/models/` and the trained models are in `saved_rnns_server_apr/data/`

Please send any questions about the code to michael.kleinman@ucla.edu


