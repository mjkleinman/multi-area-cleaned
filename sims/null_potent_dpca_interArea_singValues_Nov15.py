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
from pycog.trialRNN import PSTH
from cfg_mk import cfg_mk
from dPCA import dPCA
import re


# os.chdir("/Users/michael/Desktop/tibi_backup/tibi/saved_rnns/three_rnns")
os.chdir(cfg_mk['rnn_datapath'])
modelpath = cfg_mk['path'] + 'examples/models/cb_analyze_fixed-cb.py'
NUM_UNITS = 100
params = ['0p1'] #, '0p2', '0p3', '0p5', '1']
NUM_PARAMS = len(params)

overlap12_store = np.zeros((NUM_PARAMS, 8, 3, NUM_UNITS))
overlap23_store = np.zeros((NUM_PARAMS, 8, 3, NUM_UNITS))
# overlap33_store = np.zeros((NUM_PARAMS, 8, 3, NUM_UNITS))
sing12_store = np.zeros((NUM_PARAMS, 8, NUM_UNITS))
sing23_store = np.zeros((NUM_PARAMS, 8, NUM_UNITS))
# sing33_store = np.zeros((NUM_PARAMS, 8, NUM_UNITS))
area1_idx = list(range(0,80)) + list(range(240,260))
area2_idx = list(range(80,160)) + list(range(260,280))
area3_idx = list(range(160,240)) + list(range(280,300))


def getOverlap(dpca_axes, matrix):
    # computes overlap between axes and feedforward connectivity matrix
    overlap_store = np.zeros((3, NUM_UNITS))
    for rank in range(NUM_UNITS):
        n = rank  # 4 ## NEED TO COMPUTE n
        # q, r = np.linalg.qr(dpca_axes)
        u, s, vt = np.linalg.svd(matrix)
        v = vt.T
        ns = v[:, n:]
        ps = v[:, :n]
        Pn = ns.dot(np.linalg.inv(ns.T.dot(ns)).dot(ns.T))
        Pp = ps.dot(np.linalg.inv(ps.T.dot(ps)).dot(ps.T))
        projn2c = np.linalg.norm(Pn.dot(dpca_axes[:, 0])) ** 2
        projp2c = np.linalg.norm(Pp.dot(dpca_axes[:, 0])) ** 2
        projn2d = np.linalg.norm(Pn.dot(dpca_axes[:, 1])) ** 2
        projp2d = np.linalg.norm(Pp.dot(dpca_axes[:, 1])) ** 2

        # do analyes for a random vector
        rand_size = 100
        rv2 = np.zeros(rand_size, )
        rv_n = np.zeros(rand_size, )
        rv_p = np.zeros(rand_size, )
        projn2rv = np.zeros(rand_size, )
        projp2rv = np.zeros(rand_size, )
        for ii in range(rand_size):
            rv = np.random.randn(NUM_UNITS,)
            rv = rv / np.linalg.norm(rv)
            projn2rv[ii] = np.linalg.norm(Pn.dot(rv)) ** 2
            projp2rv[ii] = np.linalg.norm(Pp.dot(rv)) ** 2
            rv2[ii] = projn2rv[ii] + projp2rv[ii]
            rv_n[ii] = projn2rv[ii] / rv2[ii]
            rv_p[ii] = projp2rv[ii] / rv2[ii]
        mean_projn2rv = np.mean(rv_n)
        mean_projp2rv = np.mean(rv_p)

        c2 = projn2c + projp2c
        d2 = projn2d + projp2d

        layer1c = np.array([projn2c / c2, mean_projn2rv, projn2d / d2])
        layer2c = np.array([projp2c / c2, mean_projp2rv, projp2d / d2])

        overlap_store[:, rank] = layer2c

    return overlap_store, s

# for (file_id, this_file) in enumerate(all_files):
for p, param in enumerate(params):
    for seed in range(8):
        basepath = '2020-04-10_cb_simple_3areas_ff=' + param

        vin = 0.10**2
        var_in = np.array(((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, vin, 0), (0, 0, 0, vin)))
        var_rec = (0.05) ** 2
        nt = 10
        # changed seed from 1 to seed
        psth = PSTH(basepath + '/' + basepath + '_seed=' + str(seed) + '.pkl', modelpath, rnnparams={'var_in': var_in, 'var_rec': var_rec}, num_trials=nt, seed=1, threshold=0.6)

        # get the feedforward matrices
        Win = psth.rnn.Win
        Wout = psth.rnn.Wout
        Wrec = psth.rnn.Wrec
        W12 = Wrec[np.ix_(area2_idx, area1_idx)]  # To, From
        W23 = Wrec[np.ix_(area3_idx, area2_idx)]
        # W33 = Wrec[np.ix_(area3_idx, area3_idx)]

        # idx1 = np.hstack((np.arange(80)))
        # idx2 = np.hstack((np.arange(80, 160)))
        # idx3 = np.hstack((np.arange(160, 240)))

        # prepare psths for dpca
        psth.sort = ['dirs', 'scols']
        psth.gen_psth()
        num_cond1 = 2
        num_cond2 = len(psth.psths) // num_cond1

        X1 = np.zeros((len(area1_idx), 210, num_cond1, num_cond2))
        X2 = np.zeros((len(area2_idx), 210, num_cond1, num_cond2))
        # X3 = np.zeros((len(area3_idx), 210, num_cond1, num_cond2))
        for i in range(num_cond1):
            for j in range(num_cond2):
                X1[:, :, i, j] = psth.psths[i * num_cond2 + j]['psth'][area1_idx, 90:300]
                X2[:, :, i, j] = psth.psths[i * num_cond2 + j]['psth'][area2_idx, 90:300]
                # X3[:, :, i, j] = psth.psths[i * num_cond2 + j]['psth'][area3_idx, 90:300]
        dpca1 = dPCA.dPCA(labels='tdc', join={'td': ['d', 'td'], 'tc': ['c', 'tc'], 'tdc': ['dc', 'tdc']}, n_components=1)
        dpca2 = dPCA.dPCA(labels='tdc', join={'td': ['d', 'td'], 'tc': ['c', 'tc'], 'tdc': ['dc', 'tdc']}, n_components=1)
        # dpca3 = dPCA.dPCA(labels='tdc', join={'td': ['d', 'td'], 'tc': ['c', 'tc'], 'tdc': ['dc', 'tdc']}, n_components=1)
        Z1 = dpca1.fit_transform(X1)
        Z2 = dpca2.fit_transform(X2)
        # Z3 = dpca3.fit_transform(X3)
        dpca_axes1 = np.array([dpca1.P['tc'], dpca1.P['td']]).reshape(2, NUM_UNITS).T
        dpca_axes2 = np.array([dpca2.P['tc'], dpca2.P['td']]).reshape(2, NUM_UNITS).T
        # dpca_axes3 = np.array([dpca3.P['tc'], dpca3.P['td']]).reshape(2, NUM_UNITS).T
        overlap12_store[p, seed, :, :], sing12_store[p, seed, :] = getOverlap(dpca_axes1, W12)
        overlap23_store[p, seed, :, :], sing23_store[p, seed, :] = getOverlap(dpca_axes2, W23)
        # overlap33_store[p, seed, :, :], sing33_store[p, seed, :] = getOverlap(dpca_axes3, W33)

savepath = cfg_mk['path'] + 'logs/dpca_inter_area_inhIncl_nov15/'
np.save(savepath + 'null_potent_dpca_a12.npy', overlap12_store)
np.save(savepath + 'null_potent_dpca_a23.npy', overlap23_store)
# np.save(savepath + 'null_potent_dpca_a33.npy', overlap33_store)
np.save(savepath + 'sing_values_12.npy', sing12_store)
np.save(savepath + 'sing_values_23.npy', sing23_store)
# np.save(savepath + 'sing_values_33.npy', sing33_store)