# here I am going to try and see if there's an advantage of a minimal representation
# performance as a function of checkerboard noise?

from __future__ import division
import numpy as np
import os
import pdb
import sys
import subprocess
import numpy as np
import pdb
from pycog.trial_chandr import PSTH
from cfg_mk import cfg_mk
import argparse


p = argparse.ArgumentParser()
p.add_argument('--rnn_areas', type=int, default=cfg_mk['rnn_areas'])
p.add_argument('--area_noise', type=int, default=cfg_mk['area_noise'])
p.add_argument('--diff_area_noise', default=cfg_mk['diff_area_noise'])
p.add_argument('--modelpath', type=str, default=cfg_mk['modelpath'])
a = p.parse_args()

cfg_mk['rnn_areas'] = a.rnn_areas
cfg_mk['area_noise'] = a.area_noise
cfg_mk['diff_area_noise'] = a.diff_area_noise
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
modelpath = '/Users/michael/Documents/tibi/examples/models/cb_analyze_fixed-cb.py'

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
    vrec1 = 0.1 ** 2
    vrec2 = 0.15 ** 2
    vrec3 = 0.2 ** 2
    vin1 = 0.1 ** 2
    var_in1 = np.array(((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, vin1, 0), (0, 0, 0, vin1)))  # 0.20**2
    var_in2 = np.array(((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, vin1, 0), (0, 0, 0, vin1)))  # 0.20**2
    var_in3 = np.array(((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, vin1, 0), (0, 0, 0, vin1)))  # 0.20**2
    # var_rec = 0.05**2
    filename = this_file[:-4]
    num_trials = 25
    psth1 = PSTH(this_file, modelpath, rnnparams={'var_in': var_in1, 'var_rec': vrec1, 'diff_area_noise': cfg_mk['diff_area_noise'], 'rnn_areas': cfg_mk['rnn_areas'], 'area_noise': cfg_mk['area_noise']}, num_trials=num_trials, seed=1, threshold=0.6)
    psth2 = PSTH(this_file, modelpath, rnnparams={'var_in': var_in1, 'var_rec': vrec2, 'diff_area_noise': cfg_mk['diff_area_noise'], 'rnn_areas': cfg_mk['rnn_areas'], 'area_noise': cfg_mk['area_noise']}, num_trials=num_trials, seed=1, threshold=0.6)
    psth3 = PSTH(this_file, modelpath, rnnparams={'var_in': var_in1, 'var_rec': vrec3, 'diff_area_noise': cfg_mk['diff_area_noise'], 'rnn_areas': cfg_mk['rnn_areas'], 'area_noise': cfg_mk['area_noise']}, num_trials=num_trials, seed=1, threshold=0.6)
    # conds = psth1.conds
    # conds_cohs = 2 * (conds / 225) - 1
    # correct_choices = np.array([trial['info']['choice'] for trial in psth1.trials])
    # ev_cohs = np.abs(conds_cohs) * correct_choices
    # u_conds = np.unique(conds_cohs)
    # choose_low = np.zeros_like(u_conds).astype(float)
    # choose_med = np.zeros_like(u_conds).astype(float)
    # choose_high = np.zeros_like(u_conds).astype(float)

    # for i, cond in enumerate(u_conds):

    #     m2 = np.where(np.abs(ev_cohs - cond) < 0.001)[0]
    #     choose_low[i] = np.sum(psth1.choices[m2] == 1)
    #     choose_med[i] = np.sum(psth2.choices[m2] == 1)
    #     choose_high[i] = np.sum(psth3.choices[m2] == 1)  # this is only getting reaches to the right

    #     choose_low[i] /= (len(m2))
    #     choose_med[i] /= (len(m2))
    #     choose_high[i] /= (len(m2))   # this is only getting reaches to the right

    # choose_low_store[a, :] = choose_low
    # choose_med_store[a, :] = choose_med
    # choose_high_store[a, :] = choose_high
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

# f = plt.figure()
# ax = f.gca()

# y1 = np.mean(choose_low_store, axis=0)
# y2 = np.mean(choose_med_store, axis=0)
# y3 = np.mean(choose_high_store, axis=0)

# ax.plot(u_conds, y1, marker='.', markersize=20, label='low noise', color='black')
# ax.plot(u_conds, y2, marker='.', markersize=20, label='medium noise')
# ax.plot(u_conds, y3, marker='.', markersize=20, label='high noise')

# yerr_low = np.std(choose_low_store, axis=0).T / np.sqrt(8)
# yerr_med = np.std(choose_med_store, axis=0).T / np.sqrt(8)
# yerr_high = np.std(choose_high_store, axis=0).T / np.sqrt(8)

# ax.fill_between(u_conds, y1 - yerr_low, y1 + yerr_low, alpha=0.5, color='black')
# ax.fill_between(u_conds, y2 - yerr_med, y2 + yerr_med, alpha=0.5)
# ax.fill_between(u_conds, y3 - yerr_high, y3 + yerr_high, alpha=0.5)


# ax.legend()
# ax.axvline(0, linestyle='--', color='black')
# ax.set_ylim((-0.05, 1.05))
# ax.set_xlim((-1.0, 1.0))
# ax.set_yticks((0, 0.5, 1))
# ax.set_xticks((-1, -0.5, 0, 0.5, 1))
# ax.set_xlabel('Directional evidence')
# ax.set_ylabel('Proportion of reaches to right')
# ax.set_title('Psychometric function for RNN checkerboard task')

# savepath = '/Users/michael/Documents/tibi/examples/work/figsrevision/'
# area = str(cfg_mk['area'])  # todo: have this come from a config file
# plt.savefig(savepath + 'perturb_psych_3area_rec_a' + area + '.pdf')

# area = str(cfg_mk['area'])  # todo: have this come from a config file
# os.chdir("/Users/michael/Documents/tibi/sims/revision/scratch_data")
# np.save('u_conds_three_a' + area + '.npy', u_conds)
# np.save('low_noise_three_a' + area + '.npy', choose_low_store)
# np.save('med_noise_three_a' + area + '.npy', choose_med_store)
# np.save('high_noise_three_a' + area + '.npy', choose_high_store)

data_save_path = "/Users/michael/Documents/GitHub/multi-area/sims/revision/scratch_data"
os.chdir(data_save_path)
area = str(cfg_mk['area_noise'])  # todo: have this come from a config file
np.save(cfg_mk['modelpath'] + cfg_mk['suffix'] + '_a' + area + '_uconds' + '.npy', u_conds)
np.save(cfg_mk['modelpath'] + cfg_mk['suffix'] + '_a' + area + '_0p1.npy', choose_low_store)
np.save(cfg_mk['modelpath'] + cfg_mk['suffix'] + '_a' + area + '_0p15.npy', choose_med_store)
np.save(cfg_mk['modelpath'] + cfg_mk['suffix'] + '_a' + area + '_0p2.npy', choose_high_store)

call('rm -f *_copy.pkl')
