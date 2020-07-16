"""
Some commonly used functions for defining a task.

"""
from __future__ import division

import numpy as np
import pdb
from sklearn.metrics import r2_score

#-----------------------------------------------------------------------------------------
# Define E/I populations
#-----------------------------------------------------------------------------------------

def generate_ei(N, pE=0.8):
    """
    E/I signature.

    Parameters
    ----------

    N : int
        Number of recurrent units.

    pE : float, optional
         Fraction of units that are excitatory. Default is the usual value for cortex.

    """
    assert 0 <= pE <= 1

    Nexc = int(pE*N)
    Ninh = N - Nexc

    idx = range(N)
    EXC = idx[:Nexc]
    INH = idx[Nexc:]

    ei       = np.ones(N, dtype=int)
    ei[INH] *= -1

    return ei, EXC, INH

#-----------------------------------------------------------------------------------------
# Functions for defining task epochs
#-----------------------------------------------------------------------------------------

def get_idx(t, interval):
    start, end = interval

    return list(np.where((start < t) & (t <= end))[0])

def get_epochs_idx(dt, epochs):
    t = np.linspace(dt, epochs['T'], int(epochs['T']/dt))
    #assert t[1] - t[0] == dt, "[ tasktools.get_epochs_idx ] dt doesn't fit into T."

    return t, {k: get_idx(t, v) for k, v in epochs.items() if k != 'T'}

#-----------------------------------------------------------------------------------------
# Functions for generating epoch durations that are multiples of the time step
#-----------------------------------------------------------------------------------------

def uniform(rng, dt, xmin, xmax):
    return (rng.uniform(xmin, xmax)//dt)*dt

def truncated_exponential(rng, dt, mean, xmin=0, xmax=np.inf):
    while True:
        x = rng.exponential(mean)
        if xmin <= x < xmax:
            return (x//dt)*dt

def truncated_normal(rng, dt, mean, sigma, xmin=-np.inf, xmax=np.inf):
    while True:
        x = rng.normal(mean, sigma)
        if xmin <= x < xmax:
            return (x//dt)*dt

#-----------------------------------------------------------------------------------------
# Functions for generating orientation tuning curves
#-----------------------------------------------------------------------------------------

def deg2rad(s):
    return s*np.pi/180

def vonMises(s, spref, g=1, kappa=5, b=0, convert=True):
    arg = s - spref
    if convert:
        arg = deg2rad(arg)

    return g*np.exp(kappa*(np.cos(arg)-1)) + b

#-----------------------------------------------------------------------------------------
# Convert batch index to condition
#-----------------------------------------------------------------------------------------

def unravel_index(b, dims):
    return np.unravel_index(b, dims, order='F')

#-----------------------------------------------------------------------------------------
# Functions for generating connection matrices
#-----------------------------------------------------------------------------------------

def generate_Crec(ei, p_exc=1, p_inh=1, rng=None, seed=1, allow_self=False):
    if rng is None:
        rng = np.random.RandomState(seed)

    N    = len(ei)
    exc, = np.where(ei > 0)
    inh, = np.where(ei < 0)

    C = np.zeros((N, N))
    for i in exc:
        C[i,exc] = 1*(rng.uniform(size=len(exc)) < p_exc)
        if not allow_self:
            C[i,i] = 0
        C[i,inh]  = 1*(rng.uniform(size=len(inh)) < p_inh)
        C[i,inh] *= np.sum(C[i,exc])/np.sum(C[i,inh])
    for i in inh:
        C[i,exc] = 1*(rng.uniform(size=len(exc)) < p_exc)
        C[i,inh] = 1*(rng.uniform(size=len(inh)) < p_inh)
        if not allow_self:
            C[i,i] = 0
        C[i,inh] *= np.sum(C[i,exc])/np.sum(C[i,inh])
    C /= np.linalg.norm(C, axis=1)[:,np.newaxis]

    return C

#-----------------------------------------------------------------------------------------
# Callbacks
#-----------------------------------------------------------------------------------------

def correct_2afc_bias(trials, z, rmin=0.45, rmax=0.55):
    """
    Use to correct bias in the psychometric curve.

    """
    ends    = [len(trial['t'])-1 for trial in trials]
    choices = [np.argmax(z[ends[i],i]) for i, end in enumerate(ends)]

    r = choices.count(0)/choices.count(1)
    x = max(min(1/(1 + r), rmax), rmin)
    print(r, [x, 1-x])
    #return None
    return [x, 1-x]

#-----------------------------------------------------------------------------------------
# Performance measure
#-----------------------------------------------------------------------------------------

def performance_2afc(trials, z):
    ends    = [len(trial['t'])-1 for trial in trials]
    choices = [np.argmax(z[ends[i],i]) for i, end in enumerate(ends)]
    correct = [choice == trial['info']['choice']
               for choice, trial in zip(choices, trials) if trial['info']]
    return 100*sum(correct)/len(correct)

def performance_2afc_min_condition(trials, z):
    ends    = [len(trial['t'])-1 for trial in trials]
    choices = [np.argmax(z[ends[i],i]) for i, end in enumerate(ends)]

    correct = {}
    for choice, trial in zip(choices, trials):
        if not trial['info']:
            continue

        cond = tuple(trial['info'].values())
        correct.setdefault(cond, []).append(choice == trial['info']['choice'])
    correct = [sum(c)/len(c) for c in correct.values()]

    return 100*min(correct)

def performance_cb(trials, z):
    avg_vel = np.mean(z, axis=0)
    endPoints = [np.cumsum(z[:,i])[-1] for i in np.arange(len(trials))]
    num_correct = 0
    num_trials = 0

    for (i, trial) in enumerate(trials):
        if not trial['info']:
            continue
        
        choice = trial['info']['choice']
        if choice:
            inWindow = endPoints[i] / 100 * trials[i]['info']['dt'] > 80 and endPoints[i] / 100 * trials[i]['info']['dt'] < 120
            num_correct += inWindow
        else:
            inWindow = endPoints[i] / 100 * trials[i]['info']['dt'] < -80 and endPoints[i] / 100 * trials[i]['info']['dt'] > -120
            num_correct += inWindow

        num_trials += 1

    return 100*num_correct / num_trials

def performance_cb_simple(trials, z):
    
    post_delay = trials[0]['info']['post_delay']
    dt = trials[0]['info']['dt']

    ends    = [len(trial['t'])-1 for trial in trials]
    # The 50 here checks 50 ms before the post_delay period.
    choices = [z[ends[i] - (50 + post_delay) // dt - 1, i][0] for i, end in enumerate(ends)]
    
    num_correct = float(0)
    num_trials = 0

    for (i, trial) in enumerate(trials):
        if trial['info']['catch']:
            continue
    
        choice = trial['info']['choice']
        
        #if np.abs(choices[i] - choice) < 0.3:
        #    num_correct += 1
        if np.sign(choices[i]) == choice:
			num_correct += 1


        num_trials += 1 
        
    return 100 * num_correct / num_trials

def performance_cb_simple_threshold(trials, z):
    threshold = 0.5
    post_delay = trials[0]['info']['post_delay']
    dt = trials[0]['info']['dt']

    ends    = [len(trial['t'])-1 for trial in trials]
    # The 50 here checks 50 ms before the post_delay period.
    choices = [z[ends[i] - (50 + post_delay) // dt - 1, i][0] for i, end in enumerate(ends)]
    
    num_correct = float(0)
    num_trials = 0

    for (i, trial) in enumerate(trials):
        if trial['info']['catch']:
            continue
    
        choice = trial['info']['choice']
        
        #if np.abs(choices[i] - choice) < 0.3:
        #    num_correct += 1
        if np.sign(choices[i]) == choice & np.abs(choices[i]) > threshold:
			num_correct += 1


        num_trials += 1 
        
    return 100 * num_correct / num_trials
 
def performance_cb_simple_racers(trials, z):
    
    post_delay = trials[0]['info']['post_delay']
    dt = trials[0]['info']['dt']

    ends    = [len(trial['t'])-1 for trial in trials]
    # The 50 here checks 50 ms before the post_delay period.
    choices = [np.argmax(z[ends[i] - (50 + post_delay) // dt - 1, i]) for i, end in enumerate(ends)]
    choices = np.array(choices) * 2 - 1
    correct_choices = [trial['info']['choice'] for trial in trials if not trial['info']['catch']]
    correct = [choice == trial['info']['choice'] for choice, trial in zip(choices, trials) if not trial['info']['catch']]

    return 100 * sum(correct) / len(correct)

def performance_cb_simple_racers_cond(trials, z):
    
    post_delay = trials[0]['info']['post_delay']
    dt = trials[0]['info']['dt']

    ends    = [len(trial['t'])-1 for trial in trials]
    # The 50 here checks 50 ms before the post_delay period.
    choices = [np.argmax(z[ends[i] - (50 + post_delay) // dt - 1, i]) for i, end in enumerate(ends)]
    choices = np.array(choices) * 2 - 1
    correct_choices = [trial['info']['choice'] for trial in trials if not trial['info']['catch']]
    correct = [choice == trial['info']['choice'] for choice, trial in zip(choices, trials) if not trial['info']['catch']]

    num_left = np.sum(np.array(correct_choices) == -1)
    num_right = np.sum(np.array(correct_choices) == 1)
    correct_left = [choice == trial['info']['choice'] and trial['info']['choice'] == -1 for choice, trial in zip(choices, trials) if not trial['info']['catch']]
    correct_right = [choice == trial['info']['choice'] and trial['info']['choice'] == 1 for choice, trial in zip(choices, trials) if not trial['info']['catch']]

    return np.min((100 * np.sum(correct_left) / num_left, 100 * np.sum(correct_right) / num_right))

def performance_cb_simple_racers_cond_thresh(trials, z):
   
    thresh = 0.6
    post_delay = trials[0]['info']['post_delay']
    dt = trials[0]['info']['dt']

    ends    = [len(trial['t'])-1 for trial in trials]
    # The 50 here checks 50 ms before the post_delay period.
    values  = [z[ends[i] - (50 + post_delay) // dt - 1, i, np.argmax(z[ends[i] - (50 + post_delay) // dt - 1, i])] for i, end in enumerate(ends)]
    choices = [np.argmax(z[ends[i] - (50 + post_delay) // dt - 1, i]) for i, end in enumerate(ends)]
    choices = np.array(choices) * 2 - 1
    correct_choices = [trial['info']['choice'] for trial in trials if not trial['info']['catch']]
    correct = [choice == trial['info']['choice'] for choice, trial in zip(choices, trials) if not trial['info']['catch']]

    num_left = np.sum(np.array(correct_choices) == -1)
    num_right = np.sum(np.array(correct_choices) == 1)
    correct_left = [choice == trial['info']['choice'] and trial['info']['choice'] == -1 and value > thresh for choice, trial, value in zip(choices, trials, values) if not trial['info']['catch']]
    correct_right = [choice == trial['info']['choice'] and trial['info']['choice'] == 1 and value > thresh for choice, trial, value in zip(choices, trials, values) if not trial['info']['catch']]

    return np.min((100 * np.sum(correct_left) / num_left, 100 * np.sum(correct_right) / num_right))
  
def get_targets(cond):
    x = np.cos(cond * np.pi / 180)
    y = np.sin(cond * np.pi / 180)

    return (x,y)

def get_outputs(cond, movement_time, dt):
    targ_x, targ_y = get_targets(cond)
    out_x = np.zeros((movement_time))
    out_y = np.zeros((movement_time))

    # generate now the velocity profile -- this is a normal distribution centered around reach_time // 2 
    rt = 150
    reach_time = 500
    peak_vel = reach_time // 2
    vel_var = (reach_time // 7)**2 # 3 stds on either side
    t = np.arange(reach_time)
    vel_profile = 1/np.sqrt(2*np.pi*vel_var) * np.exp(-(t - peak_vel)** 2 / (2*vel_var))
    pos_profile = np.cumsum(vel_profile)
    
    # normalize vel_profile now
    vel_profile = vel_profile * 1 / np.max(vel_profile)

    # this is the reach part of the trace
    out_x[rt:rt+reach_time] = targ_x * pos_profile
    out_y[rt:rt+reach_time] = targ_y * pos_profile

    out_x[rt+reach_time:] = targ_x
    out_y[rt+reach_time:] = targ_y
                                                                        
    # this is the velocity part of the trace
    vout_x = np.zeros((movement_time))
    vout_y = np.zeros((movement_time))
    vout_x[rt:rt+reach_time] = np.cos(cond * np.pi / 180) * vel_profile
    vout_y[rt:rt+reach_time] = np.sin(cond * np.pi / 180) * vel_profile

    return (vout_x[::dt], vout_y[::dt], out_x[::dt], out_y[::dt])
    
def performance_cora_r2(trials, z):
    
    # z.shape is (Time, Trials, dims)

    dt = trials[0]['info']['dt']
    ytrue = []
    ydec = []

    for i in np.arange(len(trials)):
		# iterate trial by trial

        if 'cond' in trials[i]['info']:
            # Trial length
            T_i = trials[i]['info']['epochs']['T']
    
    		# decoded positions and velocities
            z_dec = z[:T_i // dt, i, :]
    
            # true positions and velocities.
            Y = np.zeros((T_i // dt, 4))
            cond = trials[i]['info']['cond']
            e_check = trials[i]['info']['epochs']['check']
            post_targets = trials[i]['info']['epochs']['post_targets']
            true_vel_x, true_vel_y, true_pos_x, true_pos_y = get_outputs(cond, e_check[1] - e_check[0], dt)
            mv_start = e_check[0] // dt
            mv_end = e_check[1] // dt
            post_start = post_targets[0] // dt
            post_end = post_targets[1] // dt
            Y[mv_start:mv_end, 0] = true_vel_x
            Y[mv_start:mv_end, 1] = true_vel_y
            Y[mv_start:mv_end, 2] = true_pos_x
            Y[mv_start:mv_end, 3] = true_pos_y
            Y[post_start:post_end, 2] = true_pos_x[-1]
            Y[post_start:post_end, 3] = true_pos_y[-1]
    
            ytrue.append(Y)
            ydec.append(z_dec)

    ytrues = np.concatenate(ytrue, axis=0)
    ydecs = np.concatenate(ydec, axis=0)
    r2s_vel = r2_score(ytrues[:,0:2], ydecs[:, 0:2], multioutput='uniform_average')
    r2s_pos = r2_score(ytrues[:,2:], ydecs[:, 2:], multioutput='uniform_average')
			   
    min_r2 = np.min((r2s_vel, r2s_pos))
    return min_r2

def performance_cora_r2_back(trials, z):
    
    # z.shape is (Time, Trials, dims)
    dt = trials[0]['info']['dt']
    ytrue = []
    ydec = []

    for i in np.arange(len(trials)):
		# iterate trial by trial

        if 'cond' in trials[i]['info']:
            # Trial length
            T_i = trials[i]['info']['epochs']['T']
    
    		# decoded positions and velocities
            z_dec = z[:T_i // dt, i, :]
    
            # true positions and velocities.
            Y = np.zeros((T_i // dt, 4))
            cond = trials[i]['info']['cond']
            e_check = trials[i]['info']['epochs']['out']
            post_targets = trials[i]['info']['epochs']['post_targets']
            back = trials[i]['info']['epochs']['back']
            true_vel_x, true_vel_y, true_pos_x, true_pos_y = get_outputs(cond, back[1] - back[0], dt)
            mv_start = e_check[0] // dt
            mv_end = e_check[1] // dt
            post_start = post_targets[0] // dt
            post_end = post_targets[1] // dt
            back_start = back[0] // dt
            back_end = back[1] // dt
            Y[mv_start:mv_end, 0] = true_vel_x
            Y[mv_start:mv_end, 1] = true_vel_y
            Y[mv_start:mv_end, 2] = true_pos_x
            Y[mv_start:mv_end, 3] = true_pos_y
            Y[back_start:back_end, 0] = -true_vel_x
            Y[back_start:back_end, 1] = -true_vel_y
            Y[back_start:back_end, 2] = np.flip(true_pos_x, axis=0)
            Y[back_start:back_end, 3] = np.flip(true_pos_y, axis=0)
    
            ytrue.append(Y)
            ydec.append(z_dec)

    ytrues = np.concatenate(ytrue, axis=0)
    ydecs = np.concatenate(ydec, axis=0)
    r2s_vel = r2_score(ytrues[:,0:2], ydecs[:, 0:2], multioutput='uniform_average')
    r2s_pos = r2_score(ytrues[:,2:], ydecs[:, 2:], multioutput='uniform_average')

    min_r2 = np.min((r2s_vel, r2s_pos))
    return min_r2
