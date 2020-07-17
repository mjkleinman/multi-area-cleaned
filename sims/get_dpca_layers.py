import numpy as np
from dPCA import dPCA
from pycog.trial_chandr import PSTH
import os

idx1 = np.hstack((np.arange(80), np.arange(240, 260)))
idx2 = np.hstack((np.arange(80, 160), np.arange(260, 280)))
idx3 = np.hstack((np.arange(160, 240), np.arange(280, 300)))

# rnnpath
# rnnbase = '/Users/michael/Desktop/tibi_backup/tibi/saved_rnns/three_rnns/'
rnnbase = cfg_mk['path'] + 'saved_rnns_server_apr/data/2020-04-10_cb_simple_3areas/'

# modelpath
modelpath = cfg_mk['path'] + 'examples/models/cb_analyze_fixed-cb.py'

cvar = np.zeros((8, 3))
dvar = np.zeros((8, 3))
xvar = np.zeros((8, 3))
for file_id in range(8):
    # filename_pkl = '2018-08-29_cb_3areas_ff0p1_fb0p05_seed=' + str(file_id) + '.pkl'
    filename_pkl = '2020-04-10_cb_simple_3areas_seed=' + str(file_id) + '.pkl'
    vin = 0.1 ** 2  # 0.2**2
    var_in = np.array(((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, vin, 0), (0, 0, 0, vin)))  # 0.20**2
    var_rec = 0.05 ** 2  # 0.1**2 #0.05
    psth = PSTH(rnnbase + filename_pkl, modelpath, rnnparams={'var_in': var_in, 'var_rec': var_rec}, num_trials=10, seed=1, threshold=0.6)
    psth.sort = ['dirs', 'scols']
    psth.gen_psth()
    psths = psth.psths
    num_cond1 = 2
    num_cond2 = len(psths) // num_cond1
    X1 = np.zeros((100, 210, num_cond1, num_cond2))
    X2 = np.zeros((100, 210, num_cond1, num_cond2))
    X3 = np.zeros((100, 210, num_cond1, num_cond2))

    for i in range(num_cond1):
        for j in range(num_cond2):
            X1[:, :, i, j] = psths[i * num_cond2 + j]['psth'][idx1, 90:300]
            X2[:, :, i, j] = psths[i * num_cond2 + j]['psth'][idx2, 90:300]
            X3[:, :, i, j] = psths[i * num_cond2 + j]['psth'][idx3, 90:300]

    dpca1 = dPCA.dPCA(labels='tdc', join={'td': ['d', 'td'], 'tc': ['c', 'tc'], 'tdc': ['dc', 'tdc']}, n_components=1)
    dpca2 = dPCA.dPCA(labels='tdc', join={'td': ['d', 'td'], 'tc': ['c', 'tc'], 'tdc': ['dc', 'tdc']}, n_components=1)
    dpca3 = dPCA.dPCA(labels='tdc', join={'td': ['d', 'td'], 'tc': ['c', 'tc'], 'tdc': ['dc', 'tdc']}, n_components=1)
    Z1 = dpca1.fit_transform(X1)
    Z2 = dpca2.fit_transform(X2)
    Z3 = dpca3.fit_transform(X3)
    cvar[file_id, 0], dvar[file_id, 0], xvar[file_id, 0] = dpca1.calculate_variance(X1)
    cvar[file_id, 1], dvar[file_id, 1], xvar[file_id, 1] = dpca2.calculate_variance(X2)
    cvar[file_id, 2], dvar[file_id, 2], xvar[file_id, 2] = dpca3.calculate_variance(X3)

# save data
# if rnnbase == '/Users/michael/Documents/GitHub/multi-area/saved_rnns_server_apr/data/2020-04-10_cb_simple_3areas/':
#     data_save_path =
data_save_path = cfg_mk['path'] + "sims/revision/scratch_data_dpca"
os.chdir(data_save_path)
np.save('dpca_layers_color.npy', cvar)
np.save('dpca_layers_direction.npy', dvar)
np.save('dpca_layers_context.npy', xvar)
