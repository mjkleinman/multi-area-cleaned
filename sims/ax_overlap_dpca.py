import matplotlib.pyplot as plt
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import pdb
import sys
import subprocess
import numpy as np
import pdb
from pycog.trial_chandr import PSTH
from cfg_mk import cfg_mk
from dPCA import dPCA
import re


# os.chdir("/Users/michael/Desktop/tibi_backup/tibi/saved_rnns/three_rnns")
os.chdir(cfg_mk['rnn_datapath'])

# get all files in this directory
onlyfiles = [f for f in listdir('.') if isfile(join('.', f))]
regex = re.compile('2018-08-29_cb_3areas_ff0p\d_fb0p\d*_seed=\d.pkl')
# regex = re.compile('2018-08-29_cb_3areas_ff0p1_fb0p05_seed=\d.pkl')
all_files = [string for string in onlyfiles if re.match(regex, string)]

modelpath = cfg_mk['path'] + 'examples/models/cb_analyze_fixed-cb.py'
num_files = len(all_files)
num_axes = 2
overlap12_store = np.zeros((5, 8, num_axes, 1))


def getOverlap(dpca_axes, matrix):
    # computes overlap between axes and feedforward connectivity matrix
    overlap_store = np.zeros((num_axes, 1))
    for rank in range(1):
        n = rank  # 4 ## NEED TO COMPUTE n
        u, s, vt = np.linalg.svd(matrix)
        v = vt.T
        ps = v[:, :1]
        color_proj = ps.reshape(80,).dot(dpca_axes[:, 0])
        dir_proj = ps.reshape(80,).dot(dpca_axes[:, 1])
        layer2c = np.array([color_proj, dir_proj])
        overlap_store[:, rank] = layer2c

    return overlap_store


params = ['0p1', '0p2', '0p3', '0p5', '1']


# for (file_id, this_file) in enumerate(all_files):
for p, param in enumerate(params):
    for seed in range(8):
        basepath = '2020-04-10_cb_simple_3areas_ff=' + param

        vin = 0.10**2
        var_in = np.array(((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, vin, 0), (0, 0, 0, vin)))
        var_rec = (0.05) ** 2
        nt = 10
        psth = PSTH(basepath + '/' + basepath + '_seed=' + str(seed) + '.pkl', modelpath, rnnparams={'var_in': var_in, 'var_rec': var_rec}, num_trials=nt, seed=1, threshold=0.6)

        # get the feedforward matrices
        Win = psth.rnn.Win
        Wout = psth.rnn.Wout
        Wrec = psth.rnn.Wrec
        W12 = Wrec[80:160, 0:80]  # To, From
        W21 = Wrec[0:80, 80:160]
        W23 = Wrec[160:240, 80:160]
        W32 = Wrec[80:160, 160:240]

        idx1 = np.hstack((np.arange(80)))
        idx2 = np.hstack((np.arange(80, 160)))
        idx3 = np.hstack((np.arange(160, 240)))

        # prepare psths for dpca
        psth.sort = ['dirs', 'scols']
        psth.gen_psth()
        num_cond1 = 2
        num_cond2 = len(psth.psths) // num_cond1

        X1 = np.zeros((len(idx1), 210, num_cond1, num_cond2))
        X2 = np.zeros((len(idx1), 210, num_cond1, num_cond2))
        X3 = np.zeros((len(idx1), 210, num_cond1, num_cond2))
        for i in range(num_cond1):
            for j in range(num_cond2):
                X1[:, :, i, j] = psth.psths[i * num_cond2 + j]['psth'][idx1, 90:300]
        dpca1 = dPCA.dPCA(labels='tdc', join={'td': ['d', 'td'], 'tc': ['c', 'tc'], 'tdc': ['dc', 'tdc']}, n_components=1)
        Z1 = dpca1.fit_transform(X1)
        dpca_axes1 = np.array([dpca1.P['tc'], dpca1.P['td']]).reshape(2, 80).T
        overlap12_store[p, seed, :, :] = getOverlap(dpca_axes1, W12)

overlap12_store = overlap12_store.reshape(5, 8, num_axes)

savepath = cfg_mk['path'] + 'sims/revision/exemplar_new/scratch_data_dpca/'
np.save(savepath + 'ax_overlap_dpca.npy', overlap12_store)
# np.save(savepath + 'null_potent_dpca_l2.npy', overlap23_store)
# np.save(savepath + 'null_potent_dpca_fb_l2.npy', overlap21_store)
# np.save(savepath + 'null_potent_dpca_fb_l3.npy', overlap32_store)
