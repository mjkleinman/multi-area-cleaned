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
from pycog.trial_chandr import PSTH
from cfg_mk import cfg_mk
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
modelpath = '/Users/michael/Documents/tibi/examples/models/cb_analyze_fixed-cb.py'


plt.close('all')
if cfg_mk['num_seeds'] > -1:
    all_files = [cfg_mk['modelpath'] + cfg_mk['suffix'] + '_seed=' + str(seed) + '.pkl' for seed in range(cfg_mk['num_seeds'])]
else:
    all_files = [cfg_mk['modelpath'] + cfg_mk['suffix'] + '.pkl']

num_areas = cfg_mk['rnn_areas']
mi_store = np.zeros((len(all_files), num_areas))
mi_store_color = np.zeros((len(all_files), num_areas))
mi_store_context = np.zeros((len(all_files), num_areas))

acc_store = np.zeros((len(all_files), num_areas))
acc_store_color = np.zeros((len(all_files), num_areas))
acc_store_context = np.zeros((len(all_files), num_areas))

if num_areas == 1:
    idx1 = np.arange(0, 300)
elif num_areas == 2:
    idx1 = np.hstack((np.arange(0, 120), np.arange(240, 270)))
    idx2 = np.hstack((np.arange(120, 240), np.arange(270, 300)))
elif num_areas == 3:
    idx1 = np.hstack((np.arange(0, 80), np.arange(240, 260)))
    idx2 = np.hstack((np.arange(80, 160), np.arange(260, 280)))
    idx3 = np.hstack((np.arange(160, 240), np.arange(280, 300)))
elif num_areas == 4:
    idx1 = np.hstack((np.arange(0, 60), np.arange(240, 255)))
    idx2 = np.hstack((np.arange(60, 120), np.arange(255, 270)))
    idx3 = np.hstack((np.arange(120, 180), np.arange(270, 285)))
    idx4 = np.hstack((np.arange(180, 240), np.arange(285, 300)))

for mm in range(len(all_files)):
    for layer in range(num_areas):
        this_file = all_files[mm]

        vin = 0.10**2
        var_in = np.array(((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, vin, 0), (0, 0, 0, vin)))
        var_rec = (0.1) ** 2  # 0.1 #0.05 ** 2

        nt = 100
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
        context_store = psth.lrs
        rates = np.array([np.mean(psth.trials[i]['r'][:, rstarts[i]:rends[i]], axis=1) for i in range(nt * 28)])
        idx_keep = ~np.isnan(rates).any(axis=1)
        rates = rates[idx_keep]

        if layer == 0:
            idx = idx1
        elif layer == 1:
            idx = idx2
        elif layer == 2:
            idx = idx3
        elif layer == 3:
            idx = idx4

        rates_test = rates[:, idx]
        idx1d = np.where(direction_store[idx_keep] == 1)
        idx2d = np.where(direction_store[idx_keep] == -1)
        idx1c = np.where(color_store[idx_keep] == 1)
        idx2c = np.where(color_store[idx_keep] == -1)
        idx1x = np.where(context_store[idx_keep] == 1)
        idx2x = np.where(context_store[idx_keep] == -1)
        layer_store = rates_test

        # direction information
        num_s1 = len(idx1d[0])
        num_s2 = len(idx2d[0])
        print (num_s1)
        y = np.zeros((num_s1 + num_s2))
        y[:num_s1] = 1
        y[num_s1:] = 0
        s1 = layer_store[idx1d, :].squeeze(axis=0)
        s2 = layer_store[idx2d, :].squeeze(axis=0)
        x = np.concatenate((s1, s2), axis=0)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.75, random_state=42)
        mi_store[mm, layer], acc_store[mm, layer] = nnDecode(X_train, y_train, X_test, y_test, cfg_mk['linear_mut_info'])

        # color information
        num_s1_c = len(idx1c[0])
        num_s2_c = len(idx2c[0])
        y = np.zeros((num_s1_c + num_s2_c))
        y[:num_s1_c] = 1
        y[num_s1_c:] = 0
        s1 = layer_store[idx1c, :].squeeze(axis=0)
        s2 = layer_store[idx2c, :].squeeze(axis=0)
        x = np.concatenate((s1, s2), axis=0)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.75, random_state=42)
        mi_store_color[mm, layer], acc_store_color[mm, layer] = nnDecode(X_train, y_train, X_test, y_test, cfg_mk['linear_mut_info'])

        # context information
        num_s1_c = len(idx1c[0])
        num_s2_c = len(idx2c[0])
        y = np.zeros((num_s1_c + num_s2_c))
        y[:num_s1_c] = 1
        y[num_s1_c:] = 0
        s1 = layer_store[idx1x, :].squeeze(axis=0)
        s2 = layer_store[idx2x, :].squeeze(axis=0)
        x = np.concatenate((s1, s2), axis=0)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.75, random_state=42)
        mi_store_context[mm, layer], acc_store_context[mm, layer] = nnDecode(X_train, y_train, X_test, y_test, cfg_mk['linear_mut_info'])


print ('context:' + str(1 - (mi_store_context * 1.44)))
print ('color:' + str(1 - (mi_store_color * 1.44)))
print ('direction:' + str(1 - (mi_store * 1.44)))

data_save_path = "/Users/michael/Documents/GitHub/multi-area/sims/revision/exemplar_new/scratch_data_mi"
os.chdir(data_save_path)
# add in linear
suffix_linear = ''
if cfg_mk['linear_mut_info']:
    suffix_linear = '_dropout_linear'

np.save(cfg_mk['modelpath'] + cfg_mk['suffix'] + '_color' + suffix_linear + '_layers.npy', 1 - (mi_store_color * 1.44))
np.save(cfg_mk['modelpath'] + cfg_mk['suffix'] + '_direction' + suffix_linear + '_layers.npy', 1 - (mi_store * 1.44))
np.save(cfg_mk['modelpath'] + cfg_mk['suffix'] + '_context' + suffix_linear + '_layers.npy', 1 - (mi_store_context * 1.44))
np.save(cfg_mk['modelpath'] + cfg_mk['suffix'] + '_acc_color' + suffix_linear + '_layers.npy', acc_store_color)
np.save(cfg_mk['modelpath'] + cfg_mk['suffix'] + '_acc_direction' + suffix_linear + '_layers.npy', acc_store)
np.save(cfg_mk['modelpath'] + cfg_mk['suffix'] + '_acc_context' + suffix_linear + '_layers.npy', acc_store_context)
