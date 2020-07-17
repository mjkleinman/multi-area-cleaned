# here I am going to try and see if there's an advantage of a minimal representation
# performance as a function of checkerboard noise?

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import sys
import subprocess
import numpy as np
import os
import cPickle as pkl
import matplotlib.pyplot as plt
import pdb
from pycog.trialRNN import PSTH
from cfg_mk import cfg_mk
import argparse


p = argparse.ArgumentParser()
p.add_argument('--modelpath', type=str, default=cfg_mk['modelpath'])
a = p.parse_args()
cfg_mk['modelpath'] = a.modelpath


def coh(coh):
    return 2 * coh / 225 - 1


# function to call unix lines
def call(s):
    rv = subprocess.call(s.split())
    if rv != 0:
        sys.stdout.flush()
        print("Something went wrong (return code {}).".format(rv)
              + " We're probably out of memory.")
        sys.exit(1)


datapath = cfg_mk['rnn_datapath']
os.chdir(datapath + cfg_mk['modelpath'])

# # modelpath
modelpath = cfg_mk['path'] + 'examples/models/cb_analyze_fixed-cb.py'

if cfg_mk['num_seeds'] > -1:
    all_files = [cfg_mk['modelpath'] + cfg_mk['suffix'] + '_seed=' + str(seed) + '.pkl' for seed in range(cfg_mk['num_seeds'])]
else:
    all_files = [cfg_mk['modelpath'] + cfg_mk['suffix'] + '.pkl']

num_files = len(all_files)
choose_low_store = np.zeros((num_files, ))
choose_med_store = np.zeros((num_files,))
choose_high_store = np.zeros((num_files, ))

u_conds = []
for (a, this_file) in enumerate(all_files):
    print 'On RNN {}, number {} of {}'.format(this_file, a, len(all_files))

    # run trials to get PSTHs
    vrec1 = 0.05 ** 2
    vrec2 = 0.075**2
    vrec3 = 0.1**2
    vin1 = 0.1 ** 2
    var_in1 = np.array(((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, vin1, 0), (0, 0, 0, vin1)))  # 0.20**2
    var_in2 = np.array(((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, vin1, 0), (0, 0, 0, vin1)))  # 0.20**2
    var_in3 = np.array(((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, vin1, 0), (0, 0, 0, vin1)))  # 0.20**2
    filename = this_file[:-4]
    num_trials = 25
    psth1 = PSTH(this_file, modelpath, rnnparams={'var_in': var_in1, 'var_rec': vrec1}, num_trials=num_trials, seed=1, threshold=0.6)
    psth2 = PSTH(this_file, modelpath, rnnparams={'var_in': var_in1, 'var_rec': vrec2}, num_trials=num_trials, seed=1, threshold=0.6)
    psth3 = PSTH(this_file, modelpath, rnnparams={'var_in': var_in1, 'var_rec': vrec3}, num_trials=num_trials, seed=1, threshold=0.6)

    correct_choices_low = np.array([trial['info']['choice'] for trial in psth1.trials])
    correct_choices_med = np.array([trial['info']['choice'] for trial in psth2.trials])
    correct_choices_high = np.array([trial['info']['choice'] for trial in psth3.trials])

    choose_low = np.sum(psth1.choices == correct_choices_low) / len(correct_choices_low)
    choose_med = np.sum(psth2.choices == correct_choices_med) / len(correct_choices_med)
    choose_high = np.sum(psth3.choices == correct_choices_high) / len(correct_choices_high)

    choose_low_store[a] = choose_low
    choose_med_store[a] = choose_med
    choose_high_store[a] = choose_high

print('low:' + str(np.mean(choose_low_store)))
print('med:' + str(np.mean(choose_med_store)))
print('high:' + str(np.mean(choose_high_store)))

data_save_path = cfg_mk['path'] + "sims/revision/scratch_data"
os.chdir(data_save_path)
np.save(cfg_mk['modelpath'] + cfg_mk['suffix'] + '_uconds' + '.npy', u_conds)
np.save(cfg_mk['modelpath'] + cfg_mk['suffix'] + '_0p05noise' + '.npy', choose_low_store)
np.save(cfg_mk['modelpath'] + cfg_mk['suffix'] + '_0p075noise' + '.npy', choose_med_store)
np.save(cfg_mk['modelpath'] + cfg_mk['suffix'] + '_0p1noise' + '.npy', choose_high_store)


call('rm -f *_copy.pkl')
