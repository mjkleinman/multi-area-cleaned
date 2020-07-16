# script to train many single RNNs
import pdb
import sys
import subprocess
import numpy as np
import os
from cfg_mk import cfg_mk
import pdb
# number of RNNs to train
num_runs = cfg_mk['num_seeds']
gpu_id = 1

# function to call unix lines


def call(s):
    rv = subprocess.call(s.split())
    if rv != 0:
        sys.stdout.flush()
        print("Something went wrong (return code {}).".format(rv)
              + " We're probably out of memory.")
        sys.exit(1)


# cd into the testing directory
path = cfg_mk['path']
os.chdir(path + 'testing')
modelpath = cfg_mk['modelpath']
suffix = cfg_mk['suffix']

# Run
# seeds = np.arange(num_runs)
seeds = np.arange(6, 8)
ffs = ['0p1']
fbs = ['0p05']
# lambdars = ['1e-1', '1']
# lambdaws = ['1e-2', '1e-1']
# lambdaws = ['1']
# lrs = ['1e-5', '1e-4', '2.5e-4', '5e-4', '7.5e-4']  # -6 --> -4#5e-5 too
lrs = ['5e-5']

for lr in lrs:
    for seed in seeds:
        for ff in ffs:
            for fb in fbs:
                # suffix_seed = suffix + '_lr=' + str(lr) + '_seed=' + str(seed)
                suffix_seed = suffix + '_seed=' + str(seed)
                call("python train_cb.py {} -s {} -g {} -lr {} -suffix {}".format(modelpath, seed, gpu_id, lr, suffix_seed))
