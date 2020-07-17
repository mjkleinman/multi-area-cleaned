import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
from sklearn.model_selection import train_test_split
# from mutual_information import nnDecode
import sys
import subprocess
import numpy as np
import pdb
from pycog.trial_chandr import PSTH
from cfg_mk import cfg_mk
# function to call unix lines


def call(s):
    rv = subprocess.call(s.split())
    if rv != 0:
        sys.stdout.flush()
        print("Something went wrong (return code {}).".format(rv)
              + " We're probably out of memory.")
        sys.exit(1)


def getDim(rates_test):
    cov = np.cov(rates_test.T)
    eigs = np.linalg.eig(cov)[0]
    dim = np.sum(eigs) ** 2 / np.sum(eigs ** 2)
    return dim


datapath = cfg_mk['rnn_datapath']
os.chdir(datapath + cfg_mk['modelpath'])
# os.chdir(datapath)

# # modelpath
modelpath = cfg_mk['path'] + 'examples/models/cb_analyze_fixed-cb.py'


plt.close('all')
if cfg_mk['num_seeds'] > -1:
    all_files = [cfg_mk['modelpath'] + cfg_mk['suffix'] + '_seed=' + str(seed) + '.pkl' for seed in range(cfg_mk['num_seeds'])]
else:
    all_files = [cfg_mk['modelpath'] + cfg_mk['suffix'] + '.pkl']

dims_store = np.zeros((len(all_files), cfg_mk['rnn_areas']))
# dim1_store = np.zeros((len(all_files), ))
# dim2_store = np.zeros((len(all_files), ))
# dim3_store = np.zeros((len(all_files), ))
# dim4_store = np.zeros((len(all_files), ))

for mm in range(len(all_files)):
    this_file = all_files[mm]  # '2020-01-06_cb_3areas_ff0p1_fb0p05_lambdaw=1.pkl'

    vin = 0.10**2
    var_in = np.array(((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, vin, 0), (0, 0, 0, vin)))  # 0.20**2
    var_rec = (0.05) ** 2  # 0.05 ** 2

    nt = 10
    psth = PSTH(this_file, modelpath, rnnparams={'var_in': var_in, 'var_rec': var_rec}, num_trials=nt, seed=1, threshold=0.6)
    filename = this_file[:-4]  # to remove the .pkl

    psth.sort = ['scols', 'dirs']
    psth.set_align(align='mv')
    call('rm -f *_copy.pkl')
    bounds = [-300, 100]  # used to be -500, 500

    time_mv = psth.rts + [int(trial['info']['epochs']['check'][0]) for trial in psth.trials]
    time_mv = time_mv // psth.dt
    time_mv = time_mv.astype(int)
    rstarts = time_mv + bounds[0] // psth.dt
    rends = time_mv + bounds[1] // psth.dt

    direction_store = psth.dirs
    color_store = psth.scols
    rates = np.array([np.mean(psth.trials[i]['r'][:, rstarts[i]:rends[i]], axis=1) for i in range(nt * 28)])
    # idx_nans = np.array([np.isnan(rate) for rate in rates])
    # idx_keep = np.array([not np.all(bool_across_trials) for bool_across_trials in idx_nans.T])
    # print(sum(idx_keep))
    # pdb.set_trace()

    idx_keep = ~np.isnan(rates).any(axis=1)
    rates = rates[idx_keep]
    print(rates.shape)
    # rates = rates[:, idx_keep]

# TODO: Clean this up so not hardcoded
    if cfg_mk['use_dale']:
        if cfg_mk['rnn_areas'] == 1:
            idx1 = np.arange(300)
        elif cfg_mk['rnn_areas'] == 2:
            idx1 = np.hstack((np.arange(0, 120), np.arange(240, 270)))
            idx2 = np.hstack((np.arange(120, 240), np.arange(270, 300)))
        elif cfg_mk['rnn_areas'] == 3:  # add in num_units clause here
            idx1 = np.hstack((np.arange(0, 80), np.arange(240, 260)))
            idx2 = np.hstack((np.arange(80, 160), np.arange(260, 280)))
            idx3 = np.hstack((np.arange(160, 240), np.arange(280, 300)))
        elif cfg_mk['rnn_areas'] == 4:
            idx1 = np.hstack((np.arange(0, 60), np.arange(240, 255)))
            idx2 = np.hstack((np.arange(60, 120), np.arange(255, 270)))
            idx3 = np.hstack((np.arange(120, 180), np.arange(270, 285)))
            idx4 = np.hstack((np.arange(180, 240), np.arange(285, 300)))
    else:  # only analyzed 3 area
        idx3 = np.arange(200, 300)

    # todo: clean up code so it works for variable areas
    if cfg_mk['rnn_areas'] == 1:
        dims_store[mm, 0] = getDim(rates[:, idx1])
    elif cfg_mk['rnn_areas'] == 2:
        dims_store[mm, 0] = getDim(rates[:, idx1])
        dims_store[mm, 1] = getDim(rates[:, idx2])
    elif cfg_mk['rnn_areas'] == 3:
        dims_store[mm, 0] = getDim(rates[:, idx1])
        dims_store[mm, 1] = getDim(rates[:, idx2])
        dims_store[mm, 2] = getDim(rates[:, idx3])
    elif cfg_mk['rnn_areas'] == 4:
        dims_store[mm, 0] = getDim(rates[:, idx1])
        dims_store[mm, 1] = getDim(rates[:, idx2])
        dims_store[mm, 2] = getDim(rates[:, idx3])
        dims_store[mm, 3] = getDim(rates[:, idx4])


data_save_path = cfg_mk['path'] + "sims/revision/scratch_data"
os.chdir(data_save_path)
np.save(cfg_mk['modelpath'] + cfg_mk['suffix'] + '_dims' + '.npy', dims_store)
