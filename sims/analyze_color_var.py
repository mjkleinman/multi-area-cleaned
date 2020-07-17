import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
from sklearn.model_selection import train_test_split
from mutual_information import nnDecode
import sys
import subprocess
import numpy as np
import pdb
from pycog.trialRNN import PSTH
from cfg_mk import cfg_mk
from dPCA import dPCA
# function to call unix lines


import argparse


p = argparse.ArgumentParser()
p.add_argument('--suffix', type=str, default=cfg_mk['suffix'])
p.add_argument('--num_units', type=int, default=cfg_mk['num_units'])
p.add_argument('--rnn_areas', type=int, default=cfg_mk['rnn_areas'])
p.add_argument('--num_seeds', type=int, default=cfg_mk['num_seeds'])
p.add_argument('--use_dale', default=cfg_mk['use_dale'])
p.add_argument('--modelpath', type=str, default=cfg_mk['modelpath'])
p.add_argument('--linear_mut_info', default=cfg_mk['linear_mut_info'])
p.add_argument('--rnn_datapath', type=str, default=cfg_mk['rnn_datapath'])
# p.add_argument('--dpca_with_time', default=cfg_mk['use_dale'])
a = p.parse_args()

cfg_mk['suffix'] = a.suffix
cfg_mk['num_units'] = a.num_units
cfg_mk['rnn_areas'] = a.rnn_areas
cfg_mk['num_seeds'] = a.num_seeds
cfg_mk['use_dale'] = a.use_dale
cfg_mk['modelpath'] = a.modelpath
cfg_mk['linear_mut_info'] = a.linear_mut_info
cfg_mk['rnn_datapath'] = a.rnn_datapath


# function to call unix lines
def call(s):
    rv = subprocess.call(s.split())
    if rv != 0:
        sys.stdout.flush()
        print("Something went wrong (return code {}).".format(rv)
              + " We're probably out of memory.")
        sys.exit(1)


datapath = cfg_mk['rnn_datapath']
if datapath == '/Users/michael/Desktop/tibi_backup/tibi/saved_rnns/three_rnns/':  # for the earlier rnn
    os.chdir(datapath)
else:
    os.chdir(datapath + cfg_mk['modelpath'])

# # modelpath
modelpath = cfg_mk['path'] + 'examples/models/cb_analyze_fixed-cb.py'


plt.close('all')
if cfg_mk['num_seeds'] > -1:
    all_files = [cfg_mk['modelpath'] + cfg_mk['suffix'] + '_seed=' + str(seed) + '.pkl' for seed in range(cfg_mk['num_seeds'])]
else:
    all_files = [cfg_mk['modelpath'] + cfg_mk['suffix'] + '.pkl']


cvar_store = np.zeros((len(all_files), ))
dvar_store = np.zeros((len(all_files), ))
xvar_store = np.zeros((len(all_files), ))

for mm in range(len(all_files)):
    this_file = all_files[mm]  # '2020-01-06_cb_3areas_ff0p1_fb0p05_lambdaw=1.pkl'

    vin = 0.10**2
    var_in = np.array(((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, vin, 0), (0, 0, 0, vin)))  # 0.20**2
    var_rec = (0.05) ** 2  # 0.05 ** 2

    nt = 10  # 25
    psth = PSTH(this_file, modelpath, rnnparams={'var_in': var_in, 'var_rec': var_rec}, num_trials=nt, seed=1, threshold=0.6)
    filename = this_file[:-4]  # to remove the .pkl

    psth.sort = ['dirs', 'scols']
    psth.set_align(align='cb')
    psth.gen_psth()
    call('rm -f *_copy.pkl')
    # bounds = [-300, 100]  # used to be -500, 500

# TODO: Clean this up so not hardcoded
    N = cfg_mk['num_units']
    Ninh_per_layer = (N // cfg_mk['rnn_areas']) // 5
    Nexc_per_layer = Ninh_per_layer * 4
    if cfg_mk['use_dale']:
        if cfg_mk['rnn_areas'] == 1:
            idx3 = np.arange(300)
        elif cfg_mk['rnn_areas'] == 2:
            idx3 = np.hstack((np.arange(120, 240), np.arange(270, 300)))
        elif cfg_mk['rnn_areas'] == 3:  # add in num_units clause here
            idx3 = np.hstack((np.arange(Nexc_per_layer * 2, Nexc_per_layer * 3), np.arange(N - Ninh_per_layer, N)))
        elif cfg_mk['rnn_areas'] == 4:
            idx3 = np.hstack((np.arange(180, 240), np.arange(285, 300)))
    else:  # only analyzed 3 area
        idx3 = np.arange(200, 300)

    idx1 = np.hstack((np.arange(0, 80), np.arange(240, 260)))
    num_cond1 = 2
    num_cond2 = len(psth.psths) // num_cond1
    X = np.zeros((len(idx3), 210, num_cond1, num_cond2))
    for i in range(num_cond1):
        for j in range(num_cond2):
            X[:, :, i, j] = psth.psths[i * num_cond1 + j]['psth'][idx3, 90:300]

    dpca = dPCA.dPCA(labels='tdc', join={'td': ['d', 'td'], 'tc': ['c', 'tc'], 'tdc': ['dc', 'tdc']}, n_components=1)
    Z = dpca.fit_transform(X)
    cvar, dvar, xvar = dpca.calculate_variance(X)

    # TODO: orthogonalize the variance
    cvar_store[mm] = cvar
    dvar_store[mm] = dvar
    xvar_store[mm] = xvar

data_save_path = cfg_mk['path'] + "sims/revision/scratch_data_dpca"
os.chdir(data_save_path)
print(cvar_store)
np.save(cfg_mk['modelpath'] + cfg_mk['suffix'] + '_dpca_color' + '.npy', cvar_store)
np.save(cfg_mk['modelpath'] + cfg_mk['suffix'] + '_dpca_direction' + '.npy', dvar_store)
np.save(cfg_mk['modelpath'] + cfg_mk['suffix'] + '_dpca_context' + '.npy', xvar_store)
