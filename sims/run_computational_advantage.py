import numpy as np
from pycog.trialRNN import Trial
from tqdm import tqdm

dirpath = '/Users/michael/Documents/GitHub/multi-area-cleaned/'
modelpath = dirpath + 'examples/models/cb_analyze_fixed-cb.py'
path_1area_dale = 'saved_rnns_server_apr/data/2020-04-10_cb_simple_1area/'
path_2areas_dale = 'saved_rnns_server_apr/data/2020-04-10_cb_simple_2areas/'
path_3areas_dale = 'saved_rnns_server_apr/data/2020-04-10_cb_simple_3areas/'
path_3areas_nodale = 'saved_rnns_server_apr/data/2020-04-10_cb_simple_3areas_nodale_ff=0p1/'
path_3areas_ffi = 'saved_rnns_server_apr/data/2020-04-10_cb_simple_3areas_correctdale_ffi=0p1/'
filename_1area_dale = '2020-04-10_cb_simple_1area_seed={}.pkl'
filename_2areas_dale = '2020-04-10_cb_simple_2areas_seed={}.pkl'
filename_3areas_dale = '2020-04-10_cb_simple_3areas_seed={}.pkl'
filename_3areas_nodale = '2020-04-10_cb_simple_3areas_nodale_ff=0p1_seed={}.pkl'
filename_3areas_ffi = '2020-04-10_cb_simple_3areas_correctdale_ffi=0p1_seed={}.pkl'

dale_1area_out_indices = np.arange(0, 240)
dale_2area_out_indices = np.arange(120, 240)
dale_3area_out_indices = np.arange(160, 240)
nodale_out_indices = np.arange(200, 300)

# Perturbation to output weights
vin = 0.10**2
var_in = np.array(((0, 0, 0, 0), (0,0,0,0), (0,0,vin,0), (0,0,0,vin)))
var_rec = 0.05**2
nt = 5 # 20 # 20
n_monte = 5 #10
num_seeds = 8
eps = 0.0001

def evaluate_perturbation_effect(filepath, filename, seed, indices_out, noise_scale):
    performance_perturb, performance_no_perturb = [],[]
    rnnbase = dirpath + filepath
    filename_pkl = filename.format(seed)
    psth_no_perturb = Trial(rnnbase + filename_pkl, modelpath, rnnparams={'var_in': var_in, 'var_rec': var_rec},
                     num_trials=nt, seed=1, threshold=0.6)
    for offset in range(n_monte):
        np.random.seed(seed=seed + offset)
        Wout_temp_original = np.copy(psth_no_perturb.rnn.Wout[:, indices_out])
        Wout_temp = Wout_temp_original + np.random.normal(size=Wout_temp_original.shape) * noise_scale * 0.25 # np.max(Wout_temp_original)
        # print (np.max(Wout_temp_original))
        Wout = np.zeros_like(psth_no_perturb.rnn.Wout)
        Wout_idx_nonzero = np.where(np.abs(Wout_temp_original) > eps, Wout_temp, Wout_temp_original)
        Wout[:, indices_out] = Wout_idx_nonzero  # Wout_temp
        psth_perturb = Trial(rnnbase + filename_pkl, modelpath, rnnparams={'var_in': var_in, 'var_rec': var_rec},
                             num_trials=nt, seed=1, threshold=0.6, Wout=Wout)
        print("Accuracy Before Perturbation: {:.2f}".format(psth_no_perturb.eval_performance()))
        print("Accuracy After Perturbation: {:.2f}".format(psth_perturb.eval_performance()))
        performance_no_perturb.append(psth_no_perturb.eval_performance())
        performance_perturb.append(psth_perturb.eval_performance())

    return performance_perturb, performance_no_perturb

# seed = 7
# noise_scale = 0.1
# Todo: Move the Before and after to individual function
# Clean this up
for noise_scale in tqdm([0.1, 0.2, 0.3, 0.5, 1]):
    dale_no_perturb_perf, dale_perturb_perf = [], []
    nodale_no_perturb_perf, nodale_perturb_perf = [], []
    ffi_no_perturb_perf, ffi_perturb_perf = [], []
    dale_1a_no_perturb_perf, dale_1a_perturb_perf = [], []
    dale_2a_no_perturb_perf, dale_2a_perturb_perf = [], []
    for seed in range(num_seeds):
        # Dale
        print('Dale')
        perf_perturb_list, perf_no_perturb_list = evaluate_perturbation_effect(filepath=path_3areas_dale,
                                                                               filename=filename_3areas_dale,
                                                                               seed=seed, indices_out=dale_3area_out_indices,
                                                                               noise_scale=noise_scale)
        dale_no_perturb_perf = dale_no_perturb_perf + perf_no_perturb_list
        dale_perturb_perf = dale_perturb_perf + perf_perturb_list

        # No Dale
        print('No Dale')
        perf_perturb_list, perf_no_perturb_list = evaluate_perturbation_effect(filepath=path_3areas_nodale,
                                                                               filename=filename_3areas_nodale,
                                                                               seed=seed, indices_out=nodale_out_indices,
                                                                               noise_scale=noise_scale)
        nodale_no_perturb_perf = nodale_no_perturb_perf + perf_no_perturb_list
        nodale_perturb_perf = nodale_perturb_perf + perf_perturb_list

        # Feedforward
        print('Feedforward Inhibition')
        perf_perturb_list, perf_no_perturb_list = evaluate_perturbation_effect(filepath=path_3areas_ffi,
                                                                               filename=filename_3areas_ffi,
                                                                               seed=seed,
                                                                               indices_out=dale_3area_out_indices,
                                                                               noise_scale=noise_scale)
        ffi_no_perturb_perf = ffi_no_perturb_perf + perf_no_perturb_list
        ffi_perturb_perf = ffi_perturb_perf + perf_perturb_list

        # Two area
        print('Dale 2 Area')
        perf_perturb_list, perf_no_perturb_list = evaluate_perturbation_effect(filepath=path_2areas_dale,
                                                                               filename=filename_2areas_dale,
                                                                               seed=seed,
                                                                               indices_out=dale_2area_out_indices,
                                                                               noise_scale=noise_scale)
        dale_2a_no_perturb_perf = dale_2a_no_perturb_perf + perf_no_perturb_list
        dale_2a_perturb_perf = dale_2a_perturb_perf + perf_perturb_list

        # One Area
        print('Dale 1 Area')
        perf_perturb_list, perf_no_perturb_list = evaluate_perturbation_effect(filepath=path_1area_dale,
                                                                               filename=filename_1area_dale,
                                                                               seed=seed,
                                                                               indices_out=dale_1area_out_indices,
                                                                               noise_scale=noise_scale)
        dale_1a_no_perturb_perf = dale_1a_no_perturb_perf + perf_no_perturb_list
        dale_1a_perturb_perf = dale_1a_perturb_perf + perf_perturb_list

    # Print and output and save
    print(sum(dale_no_perturb_perf)/len(dale_no_perturb_perf))
    print(sum(dale_perturb_perf)/len(dale_perturb_perf))
    print(sum(nodale_no_perturb_perf)/len(nodale_no_perturb_perf))
    print(sum(nodale_perturb_perf)/len(nodale_perturb_perf))
    print(sum(ffi_no_perturb_perf)/len(ffi_no_perturb_perf))
    print(sum(ffi_perturb_perf)/len(ffi_perturb_perf))
    print(sum(dale_1a_no_perturb_perf)/len(dale_1a_no_perturb_perf))
    print(sum(dale_1a_perturb_perf)/len(dale_1a_perturb_perf))
    print(sum(dale_2a_no_perturb_perf)/len(dale_2a_no_perturb_perf))
    print(sum(dale_2a_perturb_perf)/len(dale_2a_perturb_perf))

    # fname = dirpath + 'logs/computational_advantage_noise{}_nt{}_nmonte{}.npz'.format(noise_scale, nt, n_monte)
    # fname = dirpath + 'logs/computational_advantage_relativeNoise{}_nt{}_nmonte{}.npz'.format(noise_scale, nt, n_monte)
    # fname = dirpath + 'logs/nov3/computational_advantage_relativeNoiseNonZero{}_nt{}_nmonte{}.npz'.format(noise_scale, nt, n_monte)
    fname = dirpath + 'logs/nov3/computational_advantage_0p25NoiseNonZero{}_nt{}_nmonte{}.npz'.format(noise_scale, nt, n_monte)
    np.savez(fname,
             dale_no_perturb_perf=np.array(dale_no_perturb_perf), dale_perturb_perf=np.array(dale_perturb_perf),
             nodale_no_perturb_perf=np.array(nodale_no_perturb_perf), nodale_perturb_perf=np.array(nodale_perturb_perf),
             ffi_no_perturb_perf=np.array(ffi_no_perturb_perf), ffi_perturb_perf = np.array(ffi_perturb_perf),
             dale_1a_no_perturb_perf=np.array(dale_1a_no_perturb_perf), dale_1a_perturb_perf=np.array(dale_1a_perturb_perf),
             dale_2a_no_perturb_perf=np.array(dale_2a_no_perturb_perf), dale_2a_perturb_perf=np.array(dale_2a_perturb_perf),
             )
