from __future__ import division
import numpy as np
from pycog import tasktools
import pdb

learning_rate = 5e-5
max_gradient_norm = 0.1

# =======================
# Task-related parameters
# =======================

# Conditions
conds = [11, 45, 67, 78, 90, 101, 108, 117, 124, 135, 147, 158, 180, 214]
left_rights = [-1, 1]  # it can either be -1 (left is red) or 1 (left is green)
nconditions = len(conds) * len(left_rights)

# CB drawn time is Uniform from (600, 1000)ms
cb_drawn_bounds = [600, 1000]
# cb_drawn_bounds = [1100, 1500] ##### FOR DEBUGGING PURPOSES!

# Catch probability -- this causes the network to stay at zero when the input is zero.
pcatch = 0.1

# =================
# Network structure
# =================
Nin = 4  # number of inputs; #left target, #right target, red coh, green coh
N = 300  # number of units
Nout = 2  # number of outputs, two racers, one for each choice

tau = 50
tau_in = 50
dt = 10
rectify_inputs = False

train_brec = True
train_bout = False
baseline_in = 0

# noise
vin = 0.10**2
var_in = np.array(((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, vin, 0), (0, 0, 0, vin)))  # 0.20**2
#var_in = 0.20**2
var_rec = 0.05**2

# L2 reg
# L2 reg
lambda2_in = 1  # 0#1e-3
lambda2_rec = 1  # 1e-3
lambda2_out = 1  # 1e-3
lambda2_r = 0  # 1.9e-3
lambda_Omega = 2

# hidden activation
hidden_activation = 'rectify'
#hidden_activation = 'tanh'


# Other important parameters for timing.  Define them up here rather than later in the functions where they could be obscured.
T_catch = 2000  # length of a catch trial.
start_delay_mean = 200  # in ms, mean start of trial
start_delay_var = 50  # in ms, variance of start time.
post_delay_set = 300  # in ms, how much time after trial to turn off all inputs.
# check_time = 1500 # in ms, how much time a validation trial is given after post-checkerboard.
check_time_min = 1500
check_time_max = 1500
no_reward_time = 200  # in ms, how long after the check_board is on do we not care what the network does.

# E/I - ei an N vectors of 1's and -1's, EXC indices of +1's, INH indices of -1's
ei, EXC, INH = tasktools.generate_ei(N)

Nexc = len(EXC)
Ninh = len(INH)

EXC_SENSORY = EXC[:Nexc // 3]
INH_SENSORY = INH[:Ninh // 3]
EXC_PREMOTOR = EXC[Nexc // 3:2 * Nexc // 3]
INH_PREMOTOR = INH[Ninh // 3:2 * Ninh // 3]
EXC_MOTOR = EXC[2 * Nexc // 3:]
INH_MOTOR = INH[2 * Ninh // 3:]

Cin = np.zeros((N, Nin))
Cin[EXC_SENSORY + INH_SENSORY, :] = 1

rng = np.random.RandomState(1000)

ff_prop = 0.1  # feed forward proportion connectivity
fb_prop = 0.05  # feed back connectivity
ff_inh_prop = 0.1  # feed forward proportion to inhibitory units

Crec = np.zeros((N, N))
for i in EXC_SENSORY:
    # no projections from PFC to M1.
    Crec[i, EXC_SENSORY] = 1
    Crec[i, i] = 0
    Crec[i, EXC_PREMOTOR] = 1 * (rng.uniform(size=len(EXC_PREMOTOR)) < fb_prop)  # these are back projections
    Crec[EXC_PREMOTOR, i] = 1 * (rng.uniform(size=len(EXC_PREMOTOR)) < ff_prop)  # these are forward projections.
    Crec[INH_PREMOTOR, i] = 1 * (rng.uniform(size=len(INH_PREMOTOR)) < ff_inh_prop)  # these are forward projections.
    Crec[i, INH_SENSORY] = np.sum(Crec[i, EXC_SENSORY]) / len(INH_SENSORY)
for i in EXC_PREMOTOR:
    Crec[i, EXC_PREMOTOR] = 1
    Crec[i, i] = 0
    Crec[i, EXC_MOTOR] = 1 * (rng.uniform(size=len(EXC_MOTOR)) < fb_prop)  # these are back projections
    Crec[EXC_MOTOR, i] = 1 * (rng.uniform(size=len(EXC_MOTOR)) < ff_prop)  # these are forward projections
    Crec[INH_MOTOR, i] = 1 * (rng.uniform(size=len(INH_MOTOR)) < ff_inh_prop)  # these are forward projections.
    Crec[i, INH_PREMOTOR] = np.sum(Crec[i, EXC_PREMOTOR]) / len(INH_PREMOTOR)
for i in EXC_MOTOR:
    Crec[i, EXC_MOTOR] = 1
    Crec[i, i] = 0
    Crec[i, INH_MOTOR] = np.sum(Crec[i, EXC_MOTOR]) / len(INH_MOTOR)
for i in INH_SENSORY:
    Crec[i, EXC_SENSORY] = 1
    Crec[i, INH_SENSORY] = np.sum(Crec[i, EXC_SENSORY]) / (len(INH_SENSORY) - 1)
    Crec[i, i] = 0
for i in INH_PREMOTOR:
    Crec[i, EXC_PREMOTOR] = 1
    Crec[i, INH_PREMOTOR] = np.sum(Crec[i, EXC_PREMOTOR]) / (len(INH_PREMOTOR) - 1)
    Crec[i, i] = 0
for i in INH_MOTOR:
    Crec[i, EXC_MOTOR] = 1
    Crec[i, INH_MOTOR] = np.sum(Crec[i, EXC_MOTOR]) / (len(INH_MOTOR) - 1)
    Crec[i, i] = 0

Crec /= np.linalg.norm(Crec, axis=1)[:, np.newaxis]

# This seems reasonable; we'll keep it for now
# 2016/08/23 -- no this constrains output units; Wout = Cout_mask_plastic * Wout + Cout_mask_fixed

# output readout, from N to Nout
Cout = np.zeros((Nout, N))
# output excitatory units initialized to 1.  inhibitory stays at zero.
Cout[:, EXC_MOTOR] = 1


def coh(cond):
    return 2 * (cond / 225) - 1


def generate_trial(rng, dt, params):

    catch_trial = False

    if params['name'] in ['gradient', 'test']:

        if params.get('catch', rng.rand() < pcatch):
            catch_trial = True
        else:
            cond = params.get('conds', rng.choice(conds))
            left_right = params.get('left_rights', rng.choice(left_rights))

    elif params['name'] == 'validation':
        b = params['minibatch_index'] % (nconditions + 1)
        if b == 0:
            catch_trial = True
        else:
            k0, k1 = tasktools.unravel_index(b - 1, (len(conds), len(left_rights)))
            cond = conds[k0]
            left_right = left_rights[k1]
    else:
        raise ValueError("Unknown trial type.")

    # ======
    # Epochs
    # ======

    if catch_trial:
        start_delay = int(np.abs(rng.normal(start_delay_mean, start_delay_var)))
        post_delay = int(post_delay_set)

        # On some catch trials, we will turn on the targets

        epochs = {
            'pre_targets': (0, start_delay),
            'targets': (start_delay, T_catch - post_delay),
            'post_targets': (T_catch - post_delay, T_catch),
            'T': T_catch
        }

    else:
        check_time = int(rng.uniform(low=check_time_min, high=check_time_max) // dt * dt)
        start_delay = int(np.abs(rng.normal(start_delay_mean, start_delay_var)) // dt * dt)
        check_drawn = int(rng.uniform(low=cb_drawn_bounds[0], high=cb_drawn_bounds[1]) // dt * dt)
        no_reward = int(no_reward_time)
        stim_on = int(check_time)  # time after checkerboard, chosen somewhat arbitrarily here, can't make this a function of rt
        post_delay = int(post_delay_set)

        T = start_delay + check_drawn + stim_on + post_delay
        T = int((T // dt) * dt)

        assert (T - post_delay == start_delay + check_drawn + stim_on), 'Some rounding problems in the timing'

        epochs = {
            'pre_targets': (0, start_delay),
            'targets': (start_delay, T - post_delay),
            'check': (start_delay + check_drawn, T - post_delay),
            'decision': (start_delay + check_drawn + no_reward, T - post_delay),
            'post_targets': (T - post_delay, T),
            'T': T
        }

    # ==========
    # Trial info
    # ==========

    t, e = tasktools.get_epochs_idx(dt, epochs)  # Time, task epochs in discrete time
    trial = {'t': t, 'epochs': epochs}           # Trial

    if catch_trial:
        trial['info'] = {'start_delay': start_delay, 'post_delay': post_delay, 'catch': True, 'dt': dt, 'epochs': epochs}
    else:
        if left_right == -1 and cond > 112 or left_right == 1 and cond < 112:
            correct_choice = -1
        else:
            correct_choice = 1

        # Trial info
        trial['info'] = {'cond': cond, 'left_right': left_right, 'choice': correct_choice,
                         'start_delay': start_delay, 'check_drawn': check_drawn, 'stim_on': stim_on,
                         'post_delay': post_delay, 'catch': False, 'dt': dt, 'epochs': epochs}
    # ======
    # Inputs
    # ======

    X = np.zeros((len(t), Nin))

    if not catch_trial:
        X[e['targets'], 0] = left_right  # this is the identity of the left target: -1 is red, +1 is green
        X[e['targets'], 1] = -left_right  # this is the identity of the right target

        X[e['check'], 2] = coh(cond)  # coherence of red
        X[e['check'], 3] = coh(225 - cond)  # coherence of green

        # Assertions to make sure everything is okay.
        assert np.any(X[e['targets'], 0] + X[e['targets'], 1] < 1e-12), 'The left and right target colors are not consistent.'
        assert np.any(X[e['check'], 2] + X[e['check'], 3] < 1e-12), 'The coherences are not consistent.'

    else:
        # with probability 0.5, show just the targets.
        show_targets = rng.uniform(0, 1)
        if show_targets > 0.5:

            left_right = params.get('left_rights', rng.choice(left_rights))

            X[e['targets'], 0] = left_right  # index 0 vs 1 is left vs right, value -1 vs 1 is red vs green
            X[e['targets'], 1] = -left_right

    trial['inputs'] = X

    # ======
    # Output
    # ======

    if params.get('target_output', False):

        Y = np.zeros((len(t), Nout))  # Output matrix
        M = np.zeros_like(Y)         # Mask matrix

        # Y[:,0] is left racer
        # Y[:,1] is right racer

        # Hold values
        hi = 1
        lo = 0

        if catch_trial:
            M[:] = 1
        else:
            # During the check on and decision formation period, you want to output correctly.
            # correct_choice = 1 is a right reach
            if correct_choice == 1:
                Y[e['decision'], 1] = hi
                Y[e['decision'], 0] = lo
            elif correct_choice == -1:
                Y[e['decision'], 0] = hi
                Y[e['decision'], 1] = lo

            # We care about the network epoch at all times except the no_reward period.
            M[e['pre_targets'] + e['targets'] + e['decision'] + e['post_targets'], :] = 1

        # Outputs and mask
        trial['outputs'] = Y
        trial['mask'] = M

    return trial



# Performance measure
#performance = tasktools.performance_cb_simple_racers
#
# Termination criterion
#TARGET_PERFORMANCE = 85
# def terminate(pcorrect_history):
#    return np.mean(pcorrect_history[-3:]) > TARGET_PERFORMANCE
#
# Performance measure
performance = tasktools.performance_cb_simple_racers_cond_thresh

# Termination criterion
TARGET_PERFORMANCE = 65  # 60 #70 #80 #85 #90


def terminate(pcorrect_history):
    return np.mean(pcorrect_history[-1:]) > TARGET_PERFORMANCE


# Validation dataset
n_validation = 100 * (nconditions + 1)
