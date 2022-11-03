import numpy as np
from pycog.trialRNN import PSTH
from tqdm import tqdm

dirpath = '/Users/michael/Documents/GitHub/multi-area-cleaned/'
modelpath = dirpath + 'examples/models/cb_analyze_fixed-cb.py'
path_3areas_dale = 'saved_rnns_server_apr/data/2020-04-10_cb_simple_3areas/'
path_3areas_nodale = 'saved_rnns_server_apr/data/2020-04-10_cb_simple_3areas_nodale_ff=0p1/'
path_3areas_ffi = 'saved_rnns_server_apr/data/2020-04-10_cb_simple_3areas_correctdale_ffi=0p1/'
filename_3areas_dale = '2020-04-10_cb_simple_3areas_seed={}.pkl'
filename_3areas_nodale = '2020-04-10_cb_simple_3areas_nodale_ff=0p1_seed={}.pkl'
filename_3areas_ffi = '2020-04-10_cb_simple_3areas_correctdale_ffi=0p1_seed={}.pkl'

# Perturbation to output weights
vin = 0.10**2
var_in = np.array(((0, 0, 0, 0), (0,0,0,0), (0,0,vin,0), (0,0,0,vin)))
var_rec = 0.05**2
nt = 20
n_monte = 10
num_seeds = 8
eps = 0.0001

# seed = 7
# noise_scale = 0.1
# Todo: Move the Before and after to individual function
# Clean this up
for noise_scale in tqdm([0.1, 0.2, 0.3, 0.5, 1]):
    dale_no_perturb_perf, dale_perturb_perf = [], []
    nodale_no_perturb_perf, nodale_perturb_perf = [], []
    ffi_no_perturb_perf, ffi_perturb_perf = [], []
    for seed in range(num_seeds):
        # Dale
        print('Dale')
        rnnbase = dirpath + path_3areas_dale
        filename_pkl = filename_3areas_dale.format(seed)
        psth_dale = PSTH(rnnbase + filename_pkl, modelpath, rnnparams={'var_in': var_in, 'var_rec': var_rec},
                         num_trials=nt, seed=1, threshold=0.6)
        for offset in range(n_monte):
            np.random.seed(seed=seed + offset)
            Wout_temp_original = np.copy(psth_dale.rnn.Wout[:, 160:240])
            Wout_temp = Wout_temp_original + np.random.normal(size=Wout_temp_original.shape) * noise_scale * np.max(Wout_temp_original)
            Wout = np.zeros_like(psth_dale.rnn.Wout)
            Wout_idx_nonzero = np.where(Wout_temp_original > eps, Wout_temp, Wout_temp_original)
            Wout[:, 160:240] = Wout_idx_nonzero # Wout_temp
            psth_dale_new = PSTH(rnnbase + filename_pkl, modelpath, rnnparams={'var_in': var_in, 'var_rec': var_rec},
                                 num_trials=nt, seed=1, threshold=0.6, Wout=Wout)
            print("Accuracy Before Perturbation: {:.2f}".format(psth_dale.eval_performance()))
            print("Accuracy After Perturbation: {:.2f}".format(psth_dale_new.eval_performance()))
            dale_no_perturb_perf.append(psth_dale.eval_performance())
            dale_perturb_perf.append(psth_dale_new.eval_performance())

        # No Dale
        print('No Dale')
        rnnbase = dirpath + path_3areas_nodale
        filename_pkl = filename_3areas_nodale.format(seed)
        psth_nodale = PSTH(rnnbase + filename_pkl, modelpath, rnnparams={'var_in': var_in, 'var_rec': var_rec},
                           num_trials=nt, seed=1, threshold=0.6)
        for offset in range(n_monte):
            np.random.seed(seed=seed + offset)
            Wout_temp_original = np.copy(psth_nodale.rnn.Wout[:, 200:300])
            Wout_temp = Wout_temp_original + np.random.normal(size=Wout_temp_original.shape) * noise_scale * np.max(Wout_temp_original)
            Wout = np.zeros_like(psth_nodale.rnn.Wout)
            Wout_idx_nonzero = np.where(Wout_temp_original > eps, Wout_temp, Wout_temp_original)
            Wout[:, 200:300] = Wout_idx_nonzero # Wout_temp
            psth_nodale_new = PSTH(rnnbase + filename_pkl, modelpath, rnnparams={'var_in': var_in, 'var_rec': var_rec},
                                   num_trials=nt, seed=1, threshold=0.6, Wout=Wout)
            print("Accuracy Before Perturbation: {:.2f}".format(psth_nodale.eval_performance()))
            print("Accuracy After Perturbation: {:.2f}".format(psth_nodale_new.eval_performance()))
            nodale_no_perturb_perf.append(psth_nodale.eval_performance())
            nodale_perturb_perf.append(psth_nodale_new.eval_performance())

        # Feedforward
        print('Feedforward Inhibition')
        rnnbase = dirpath + path_3areas_ffi
        filename_pkl = filename_3areas_ffi.format(seed)
        psth_ffi = PSTH(rnnbase + filename_pkl, modelpath, rnnparams={'var_in': var_in, 'var_rec': var_rec},
                        num_trials=nt, seed=1, threshold=0.6)
        for offset in range(n_monte):
            np.random.seed(seed=seed + offset)
            Wout_temp_original = np.copy(psth_ffi.rnn.Wout[:, 160:240])
            Wout_temp = Wout_temp_original + np.random.normal(size=Wout_temp_original.shape) * noise_scale * np.max(Wout_temp_original)
            Wout = np.zeros_like(psth_ffi.rnn.Wout)
            Wout_idx_nonzero = np.where(Wout_temp_original > eps, Wout_temp, Wout_temp_original)
            Wout[:, 160:240] = Wout_idx_nonzero # Wout_temp
            psth_ffi_new = PSTH(rnnbase + filename_pkl, modelpath, rnnparams={'var_in': var_in, 'var_rec': var_rec},
                                num_trials=nt, seed=1, threshold=0.6, Wout=Wout)
            print("Accuracy Before Perturbation: {:.2f}".format(psth_ffi.eval_performance()))
            print("Accuracy After Perturbation: {:.2f}".format(psth_ffi_new.eval_performance()))
            ffi_no_perturb_perf.append(psth_ffi.eval_performance())
            ffi_perturb_perf.append(psth_ffi_new.eval_performance())

    # Print and output and save
    print(sum(dale_no_perturb_perf)/len(dale_no_perturb_perf))
    print(sum(dale_perturb_perf)/len(dale_perturb_perf))
    print(sum(nodale_no_perturb_perf)/len(nodale_no_perturb_perf))
    print(sum(nodale_perturb_perf)/len(nodale_perturb_perf))
    print(sum(ffi_no_perturb_perf)/len(ffi_no_perturb_perf))
    print(sum(ffi_perturb_perf)/len(ffi_perturb_perf))

    # fname = dirpath + 'logs/computational_advantage_noise{}_nt{}_nmonte{}.npz'.format(noise_scale, nt, n_monte)
    # fname = dirpath + 'logs/computational_advantage_relativeNoise{}_nt{}_nmonte{}.npz'.format(noise_scale, nt, n_monte)
    fname = dirpath + 'logs/computational_advantage_relativeNoiseNonZero{}_nt{}_nmonte{}.npz'.format(noise_scale, nt, n_monte)
    np.savez(fname,
             dale_no_perturb_perf=np.array(dale_no_perturb_perf), dale_perturb_perf=np.array(dale_perturb_perf),
             nodale_no_perturb_perf=np.array(nodale_no_perturb_perf), nodale_perturb_perf=np.array(nodale_perturb_perf),
             ffi_no_perturb_perf=np.array(ffi_no_perturb_perf), ffi_perturb_perf = np.array(ffi_perturb_perf))
