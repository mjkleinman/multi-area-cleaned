"""
Recurrent neural network for testing networks outside of Theano.

"""
from __future__ import absolute_import
from __future__ import division

import numpy as np
import sys
import imp
import pdb
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import matplotlib as mpl
mpl.use('Agg') # need this when running on server w/o x
import matplotlib.pyplot as plt
#plt.ioff() # need this when running on server w/o x
import matplotlib.cm as cm
import itertools
from functools import reduce

# For optimization of fixed points
import theano
import theano.tensor as T
from scipy.optimize import minimize

# Factor analysis
from sklearn.decomposition import FactorAnalysis

# coeff of determination
from sklearn.metrics import r2_score

from .rnn   import RNN
from .      import tasktools

#import autograd.numpy as autonp

#from pymanopt.manifolds import Stiefel
#from pymanopt import Problem
#from pymanopt.solvers import SteepestDescent


THIS = 'pycog.trial'

# ==========================================
# Calculates coherence given num red squares
# ==========================================
def coh_r(cb_cond):
    return 2*(cb_cond / 225) - 1

def inv_coh_r(coh):
    # ROUNDS TO THE NEAREST COH!  This is a useful tool.
    cohs        = np.array([11, 45, 67, 78, 90, 101, 108, 117, 124, 135, 147, 158, 180, 214])
    this_coh    = 0.5 * (coh + 1) * 225

    diffs       = np.abs(cohs - this_coh)

    idx_close   = np.where(diffs == min(diffs))[0][0]

    return cohs[idx_close]

# ===========================================
# Returns the reach direction given an input.
# ===========================================

def which_dir(rnn_in=np.array((0,0,0,0))):

    if rnn_in[0] > rnn_in[1]: # then the left target is green
        d = -1 if rnn_in[3] > rnn_in[2] else 1
    else: # then the left target is red
        d = 1 if rnn_in[3] > rnn_in[2] else -1

    return 0 if rnn[2] == rnn[3] else d

# ========================
# Helper function ismember
# ========================
def ismember(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return [bind.get(itm, None) for itm in a]

class Trial(object):

    dtype = np.float32

    def __init__(self, rnnfile, modelfile, num_trials=100, seed=1, target_output=False, rnnparams={}, threshold=None, Wrec=None):
        """
        Initialize the RNN from a saved training file.

        Parameters
        ----------

        rnnfile:  str, required
                   Path to the RNN

        modelfile: str, required
                   Path to the modelfile

        num_trials: int, optional
                    Number of trials to run for each condition

        seed: int, optional
                    The random seed.

        target_output: bool, optional
                    Whether to store target outputs in the RNN

        rnnparams: dict, optional
                    Parameters to override

        """

        self.rnn        = RNN(rnnfile, rnnparams=rnnparams, verbose=False)
        if Wrec is not None:
            self.rnn.Wrec = Wrec
        self.m          = imp.load_source('model', modelfile)
        self.ntrials    = num_trials * self.m.nconditions
        self.rng        = np.random.RandomState(seed)
        self.dt         = self.rnn.p['dt']
        self.target_output = target_output

        # Build the trial information
        self.build_trials(threshold=threshold)

    # =================================================================================
    # Generates many RNN trials - pre-requisite to psychometric, RT curves, psths, etc.
    # =================================================================================
    def run_trials(self):

        w           = len(str(self.ntrials))
        trials      = []
        conds       = []
        backspaces  = 0

        for i in range(self.ntrials):

            # Get the condition and left-right orientation
            b           = i % self.m.nconditions
            k1, k2      = tasktools.unravel_index(b, (len(self.m.conds), len(self.m.left_rights)))
            cond        = self.m.conds[k1]
            left_right  = self.m.left_rights[k2]

            # Generate a trial
            trial_func = self.m.generate_trial
            trial_args = {
                'name': 'test',
                'cond': cond,
                'left_right': left_right,
                'catch': False
            }

            info = self.rnn.run(inputs=(trial_func, trial_args), rng=self.rng)

            # Display important information
            s = ("\r Trial {:>{}}/{}: left_right: {}, cond: {}".format(i+1, w, self.ntrials, info['left_right'], info['cond']))
            sys.stdout.write(backspaces*'\b' + s)
            sys.stdout.flush()
            backspaces = len(s)

            dt = self.rnn.t[1] - self.rnn.t[0]
            #pdb.set_trace()
            trial = {
                    't': self.rnn.t[::],
                    'u': self.rnn.u[:,::],
                    'r': self.rnn.r[:,::],
                    'x': self.rnn.x[:,::],
                    'z': self.rnn.z[:,::],
                    'info': info
                }

            if self.target_output:
                trial['out'] = self.rnn.out[:,::]

            trials.append(trial)
            conds.append(cond)

        self.trials = np.array(trials)
        self.conds  = np.array([trial['info']['cond'] for trial in trials])
        self.cohs   = self.conds

    # ===========================
    # Adds reaction time to trials
    # ============================
    def add_rt_to_trials(self, threshold=0.25):
        choices, rts = self.get_choices(threshold=threshold)
        self.choices, self.rts = (choices, rts)

        # Print the number of nan RTs
        print '\nThe proportion of NaN RTs is {}'.format(np.sum(np.isnan(rts)) / len(rts))


    # ========================================================
    # Add checkerboard ons to trials, as well as trial lengths
    # ========================================================
    def add_landmark_times_to_trials(self):
        self.cbs    = np.array([trial['info']['epochs']['check'][0] for trial in self.trials]).astype(int)
        self.Ts     = np.array([trial['info']['epochs']['post_targets'][1] for trial in self.trials]).astype(int)

    # ===========================================================================
    # Adds actual reach direction of reach to trials, field 'dir'; -1 vs l is 1-r
    # ===========================================================================
    def add_dir_to_trials(self):

        # -1 is to the left, 1 is to the right
        self.dirs   = self.choices

        # legacy
        for i in np.arange(self.ntrials):
            self.trials[i]['dir'] = self.dirs[i]

    # =============================================================================
    # Adds actual color direction of reach to trials, field 'scol' (selected color)
    # =============================================================================
    def add_scol_to_trials(self):

        # -1 if reach to left, +1 if reach to right
        dirs    = self.dirs

        # -1 if left is red, +1 if left is green.
        lrs     = np.array([trial['info']['left_right'] for trial in self.trials]).astype(int)

        scols = []

        for i in np.arange(self.ntrials):
            if self.dirs[i] == -1:
                if lrs[i] == -1:
                    scols.append(-1)
                else:
                    scols.append(+1)
            else:
                if lrs[i] == -1:
                    scols.append(+1)
                else:
                    scols.append(-1)

        self.scols = np.array(scols)

    # =========================================================
    # Adds prompted direction of reach to trials, field 'pdirs'
    # =========================================================
    def add_pdir_to_trials(self):
        # -1 if correct was reach to left, +1 if reach to right
        self.pdirs  = np.array([trial['info']['choice'] for trial in self.trials])

    # =========================================================
    # Adds which color is on the left or right, field 'lrs'
    # =========================================================
    def add_lrs_to_trials(self):
        # -1 if left is red, +1 if left is green
        self.lrs = np.array([trial['info']['left_right'] for trial in self.trials])

    # ===============================================================
    # Adds prompted color direction of reach to trials, field 'pscol'
    # ===============================================================
    def add_pscol_to_trials(self):

        # -1 if prompted left, +1 if prompted green
        pdirs   = [trial['info']['choice'] for trial in self.trials]

        # -1 if left is red, +1 if left is green.
        lrs     = np.array([trial['info']['left_right'] for trial in self.trials]).astype(int)

        # let -1 be red, +1 be green
        pscols  = np.array(lrs + (lrs != pdirs).astype(int)) #-1 and +1 for lrs == dirs, 0 if lrs=-1 and he reached to right (i.e., green), and +2 if lrs=+1 and he reached left (i.e., red)
        pscols[pscols == 0] = +1
        pscols[pscols == 2] = -1

        self.pscols = pscols

    # ============================
    # Adds success field to trials
    # ============================
    def add_success_to_trials(self, threshold=0):

        correct_choices = np.array([trial['info']['choice'] for trial in self.trials])
        self.successes = np.zeros(self.ntrials)

        for i in np.arange(self.ntrials):
            self.trials[i]['success'] = correct_choices[i] == self.choices[i]
            self.successes = correct_choices[i] == self.choices[i]

    # ========================
    # Build the list of trials
    # ========================
    def build_trials(self, threshold=None):
        self.run_trials()
        self.add_rt_to_trials(threshold=threshold) # this function also adds self.choices
        self.add_dir_to_trials()
        self.add_success_to_trials(threshold=threshold)
        self.add_scol_to_trials()
        self.add_pdir_to_trials()
        self.add_pscol_to_trials()
        self.add_lrs_to_trials()
        self.add_landmark_times_to_trials()

    # =========================
    # Filter the list of trials
    # =========================
    def filter_trials(self):
        rts = np.array([trial['rt'] for trial in self.trials])
        suc = np.array([trial['success'] for trial in self.trials])
        mask            = suc & ~np.isnan(rts)

        self.trials     = self.trials[mask]

        # Task properties
        self.conds      = self.conds[mask]
        self.cohs       = self.conds # this is a duplicate field, for psths
        self.dirs       = self.dirs[mask]
        self.successes  = self.successes[mask]
        self.scols      = self.scols[mask]
        self.pdirs      = self.pdirs[mask]
        self.pscols     = self.pscols[mask]
        self.lrs        = self.lrs[mask]

        # Task analogs
        self.rts        = self.rts[mask]
        self.cbs        = self.cbs[mask]
        self.Ts         = self.Ts[mask]
        self.ntrials    = len(self.trials)

    # =================================
    # Check the performance of the RNNs
    # =================================

    def eval_performance(self):

        performance = tasktools.performance_cb_simple_racers

        dt = self.dt
        zz = [np.array(trial['z'][0,:]) for trial in self.trials]
        lens = [len(z) for z in zz]
        max_len = np.max(lens)

        z = np.zeros((max_len, self.ntrials, 2))

        for i in range(len(zz)):
            z[:lens[i], i, :] = self.trials[i]['z'].T

        return performance(self.trials, z)

    # ======================================================
    # Plots trials per condition - used to check RNN output.
    # ======================================================
    def plot_trials_cond(self, cond=11, nplot='all', f=None, savepath=None, filename=None):

        dt          = self.dt
        conds       = self.conds
        idx_cond    = np.where(np.array(conds) == cond)[0]

        if nplot == 'all':
            f_iter = range(len(idx_cond))
        else:
            f_iter = range(min(nplot, len(idx_cond)))

        if f is None:
            f = plt.figure()
        ax = f.gca()

        if len(self.trials[0]['z']) == 2:
            # plot two DVs
            for i in f_iter:
                z1 = self.trials[idx_cond[i]]['z'][0,:]
                z2 = self.trials[idx_cond[i]]['z'][1,:]
                ax.plot(np.arange(len(z1)) * dt, z1, color='blue')
                ax.plot(np.arange(len(z2)) * dt, z2, color='orange')
                ax.axvline(self.trials[idx_cond[i]]['info']['check_drawn'], color='m')
        else:
            # plot one DV
            for i in f_iter:
                z1 = self.trials[idx_cond[i]]['z'][0,:]
                if self.pdirs[idx_cond[i]] == -1:
                    ax.plot(np.arange(len(z1)) * dt, z1, color='blue')
                else:
                    ax.plot(np.arange(len(z1)) * dt, z1, color='orange')
                ax.axvline(self.trials[idx_cond[i]]['info']['check_drawn'], color='m')

        if filename is None:
            filename = 'output'

        if savepath is not None:
            f.savefig(savepath + filename + '_c={}.pdf'.format(cond))

        return f

    # ==================================
    # Plots trials across all conditions
    # ==================================
    def plot_all_trials(self, savepath=None, filename=None):

        uconds = np.unique(self.conds)

        for i in uconds:
              self.plot_trials_cond(cond=i, savepath=savepath, filename=filename)


    # ===========
    # Get choices
    # ===========

    def get_choices(self, threshold=False):

        choices = []
        rts     = []

        for (i, trial) in enumerate(self.trials):
            if len(trial['z'][:,0]) > 1:
                #choice, rt = self.get_choice_twoDVs(i, threshold=threshold)
                choice, rt = self.get_choice_oldTwoDV_deprecated(i, threshold)
            else:
                choice, rt = self.get_choice_oneDV(i, threshold=threshold)

            choices.append(choice)
            rts.append(rt)

        return np.array(choices), np.array(rts)

    def get_choice_oldTwoDV_deprecated(self, idx=0, threshold=False):

        trial = self.trials[idx]
        check_on = (trial['info']['start_delay'] + trial['info']['check_drawn']) // self.dt

        if not threshold:
            return 2*np.argmax(trial['z'][:,-1]) - 1, np.nan

        over_threshold_0, = np.where(np.abs(trial['z'][0]) > threshold)
        over_threshold_1, = np.where(np.abs(trial['z'][1]) > threshold)
        rt_dt_0 = -1
        rt_dt_1 = -1

        # reaction time can't be negative, so remove negative RTs
        if len(over_threshold_1) > 0:
            candidate_rts_1,    = np.where(over_threshold_1 > check_on)
            if len(candidate_rts_1) > 0:
                rt_dt_1         = over_threshold_1[candidate_rts_1[0]]

        # reaction time can't be negative, so remove negative RTs
        if len(over_threshold_0) > 0:
            candidate_rts_0,    = np.where(over_threshold_0 > check_on)
            if len(candidate_rts_0) > 0:
                rt_dt_0         = over_threshold_0[candidate_rts_0[0]]

        # Reaction time
        #w0, = np.where(trial['z'][0] > threshold)
        #w1, = np.where(trial['z'][1] > threshold)
        w0 = (rt_dt_0 - check_on) * self.dt
        w1 = (rt_dt_1 - check_on) * self.dt

        if w0 < 0 and w1 < 0:
            #return np.nan, np.nan  # returning nan for choice may break other fns
            return 2*np.argmax(trial['z'][:, -1]) - 1, np.nan
            #return np.nan, np.nan

        if w1 < 0:
            return -1, w0
        if w0 < 0:
            return 1, w1
        if w0 < w1:
            return -1, w0

        return 1, w1


    def get_choice_twoDVs(self, idx=0, threshold=False):
        # this is a differential threshold

        trial = self.trials[idx]
        check_on = (trial['info']['start_delay'] + trial['info']['check_drawn']) // self.dt

        # this remains the same; if no threshold, return the biggest DV at the end of the trial.  It's been corrected to incorporate post_delay
        if not threshold:
            return 2*np.argmax(trial['z'][:,-1 - trial['info']['post_delay'] // trial['info']['dt']]) - 1, np.nan

        # new version relying on dv diffs
        dv_diff = trial['z'][1] - trial['z'][0]
        over_threshold, = np.where(np.abs(dv_diff) > threshold)

        rt_dt = -1

        # reaction time can't be negative, so remove negative RTs
        if len(over_threshold) > 0:
            candidate_rts,  = np.where(over_threshold > check_on)
            if len(candidate_rts) > 0:
                rt_dt               = over_threshold[candidate_rts[0]]

        # it did not cross threshold
        if rt_dt < 0:
            #return whatever was more deliberated at end of trial
            return 2*np.argmax(trial['z'][:, - trial['info']['post_delay'] // trial['info']['dt']]) - 1, np.nan

        # else it did cross threshold
        else:
            rt = (rt_dt - check_on) * self.dt
            choice = 2*np.argmax(trial['z'][:, rt_dt]) - 1
            return choice, rt

    def get_choice_oneDV(self, idx=0, threshold=False):

        trial = self.trials[idx]

        check_on = (trial['info']['start_delay'] + trial['info']['check_drawn']) // self.dt

        # this remains the same; if no threshold, return the biggest DV at the end of the trial.  It's been corrected to incorporate post_delay
        if not threshold:
            return np.sign(np.squeeze(trial['z'][:,-1 - trial['info']['post_delay'] // trial['info']['dt']])), np.nan

        # new version relying on dv diffs
        dv_diff = np.squeeze(trial['z'][:])
        over_threshold, = np.where(np.abs(dv_diff) > threshold)

        rt_dt = -1

        # reaction time can't be negative, so remove negative RTs
        if len(over_threshold) > 0:
            candidate_rts,  = np.where(over_threshold > check_on)
            if len(candidate_rts) > 0:
                rt_dt               = over_threshold[candidate_rts[0]]

        # it did not cross threshold
        if rt_dt < 0:
            #return whatever was more deliberated at end of trial
            return np.sign(np.squeeze(trial['z'][:, - trial['info']['post_delay'] // trial['info']['dt']])), np.nan

        # else it did cross threshold
        else:
            rt = (rt_dt - check_on) * self.dt
            choice = np.sign(np.squeeze(trial['z'][:, rt_dt]))
            return choice, rt

    # ========================
    # Plots psychometric curve
    # ========================
    def psychometric(self, savepath=None, filename=None):

        conds = self.conds
        correct_choices = np.array([trial['info']['choice'] for trial in self.trials])
        left_rights     = np.array([trial['info']['left_right'] for trial in self.trials])

        #choices, rts   = self.get_choices(threshold=threshold)
        choices, rts    = (self.choices, self.rts)
        trial_outcome   = choices == correct_choices
        choose_red      = choices == left_rights

        u_conds             = np.unique(conds)
        success_rates       = np.zeros_like(u_conds).astype(float)
        choose_red_rates    = np.zeros_like(u_conds).astype(float)

        for i,cond in enumerate(u_conds):
            trialMask           = np.where(conds == cond)[0]
            success_rates[i]    = np.sum(correct_choices[trialMask] == choices[trialMask]).astype(float) / len(trialMask)
            choose_red_rates[i] = np.sum(choose_red[trialMask]) / len(trialMask)

        f   = plt.figure()
        ax  = f.gca()

        ax.plot(2*(u_conds / 225) - 1, choose_red_rates, marker='.', markersize=20)
        ax.axvline(0, linestyle='--')
        ax.set_ylim((-0.05, 1.05))
        ax.set_xlim((-1.0, 1.0))
        ax.set_xlabel('Checkerboard coherence')
        ax.set_ylabel('Proportion of reaches to red target')
        ax.set_title('Psychometric function for RNN checkerboard task')

        if filename is None:
            filename = 'psychometric'

        if savepath is not None:
            f.savefig(savepath + filename + '.pdf')

        return (2*(u_conds / 225) - 1, choose_red_rates, success_rates)
        #return f

    # ===========================
    # Plots reaction time vs cond
    # ===========================

    def reaction_time(self, savepath=None, filename=None):

        conds   = self.conds
        rts     = self.rts

        u_conds     = np.unique(conds)
        rts_cond    = np.zeros_like(u_conds).astype(float)

        for i,cond in enumerate(u_conds):
            trialMask   = np.where(conds == cond)[0]
            rts_cond[i] = np.nanmean(rts[trialMask])

        f   = plt.figure()
        ax  = f.gca()
        ax.plot(2*(u_conds / 225) - 1, rts_cond, marker='.', markersize=20)
        ax.set_xlim((-1.0, 1.0))
        ax.set_xlabel('Checkerboard coherence')
        ax.set_ylabel('Reaction time')
        ax.set_title('Reaction time for RNN checkerboard task')

        if filename is None:
            filename = 'reaction_time'

        if savepath is not None:
            f.savefig(savepath + filename + '.pdf')

        return rts_cond
        #return f

    # ===============================
    # Plots reaction time std vs cond
    # ===============================
    def reaction_time_std(self, savepath=None):

        conds   = self.conds
        rts     = self.rts

        u_conds = np.unique(conds)
        rts_std = np.zeros_like(u_conds).astype(float)

        for i,cond in enumerate(u_conds):
            trialMask = np.where(conds == cond)[0]
            rts_std[i] = np.sqrt(np.nanvar(rts[trialMask]))

        f   = plt.figure()
        ax  = f.gca()

        ax.plot(2*(u_conds / 225) - 1, rts_std, marker='.', markersize=20)
        ax.set_xlim((-1.0, 1.0))
        ax.set_xlabel('Checkerboard coherence')
        ax.set_ylabel('Reaction time $\sigma$')
        ax.set_title('Standard Deviation for RNN checkerboard task')

        if savepath is not None:
            f.savefig(savepath + 'reaction_time_std.pdf')

        return f


    # =================================
    # Reaction time histograms per cond
    # =================================
    def reaction_time_hist(self, savepath=None):

        conds   = self.conds
        rts     = self.rts

        u_conds = np.unique(conds)
        rts_cond = np.zeros_like(u_conds).astype(float)

        for i,cond in enumerate(u_conds):
            trialMask   = np.where(conds == cond)[0]
            rts_cond    = rts[trialMask]

            f   = plt.figure()
            ax  = f.gca()

            # Plot all RTs that are not NaN, which will cause plt.hist() to throw up.
            ax.hist(rts_cond[~np.isnan(rts_cond)])
            ax.set_xlabel('Reaction time [ms]')
            ax.set_ylabel('Counts')
            ax.set_title('Reaction time histogram for {} red squares'.format(cond))

            if savepath is not None:
                f.savefig(savepath + 'rt_hist_c={}.pdf'.format(cond))


    # ===============
    # RNN's structure
    # ===============
    def plot_structure(self, savepath=None, filename=None):

        if filename is None:
            filename = 'structure'

        f = self.rnn.plot_structure(sortby=None)  # f is in the figure class of pycog/figtools;
        if savepath is not None:
            f.savefig(savepath + filename + '.pdf')


    # =================
    # Behavioral input
    # =================
    def plot_inputs_pub(self, i=0, savepath=None, display=True):
        # i is the trial number

        # Now plot these trial inputs
        f   = plt.figure()
        ax  = f.gca()

        # separation in plotting the inputs along the y-axis
        sepY        = 2

        # offset so ticks don't collide
        tickOffset  = 0.1

        T   = self.trials[i]['info']['epochs']['T'] // self.dt
        Nin = len(self.trials[i]['u'])
        uu  = self.trials[i]['u']
        uu  = uu[[2,1,0]]

        base, targ_on   = np.array(self.trials[i]['info']['epochs']['pre_targets']) // self.dt
        go, targ_off    = np.array(self.trials[i]['info']['epochs']['check']) // self.dt

        for j in np.arange(Nin):
            # Targ X highest
            ax.plot(np.arange(0, T), tickOffset + j*sepY + uu[j][:-1], color='gray', linewidth=2)

        # Remove borders
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        #ax.spines["bottom"].set_visible(False)

        # Remove ticks on top and right
        ax.tick_params(axis='x', bottom='on', top='off', labelbottom='on', direction='out')
        ax.tick_params(axis='y', left='off', right='off', labelleft='off')

        # set the limits so you can see the bottom and top traces
        ax.set_ylim([-0.11, sepY * (Nin - 1) + 0.11])

        # Set labels
        ax.text(-50, 0.5, 'Go cue', verticalalignment='center')
        ax.text(-50, 2.5, 'Target\n$y$-position', verticalalignment='center')
        ax.text(-50, 3.5, 'Target\n$x$-position', verticalalignment='center')

        # Plot the ticks for go-cue and targets on
        plt.xticks((base, targ_on, go, targ_off), ("Center hold", "Target on", "Go cue", "Target off"))
        plt.yticks(())

        if savepath is not None:
            f.savefig(savepath + 'inputs.pdf')

        if not display:
            f.clf()

    # =================
    # Behavioral output
    # =================
    def plot_outputs_pub(self, i=0, savepath=None, display=True, line_color='gray', f=None, return_handles=False):
        # i is the trial number

        # Now plot these trial inputs
        if f is None:
            f   = plt.figure()
        ax  = f.gca()

        # this one will be a bit more straightforward

        # offset so ticks don't collide

        T       = self.trials[i]['info']['epochs']['T'] // self.dt
        Nout    = len(self.trials[i]['z'])
        zz      = self.trials[i]['z']

        base, targ_on   = np.array(self.trials[i]['info']['epochs']['pre_targets']) // self.dt
        go, targ_off    = np.array(self.trials[i]['info']['epochs']['check']) // self.dt

        handles = []

        for j in np.arange(Nout):
            # Targ X highest
            handle, = ax.plot(np.arange(0, T), zz[j][:-1] - 2 * (j < 2), color=line_color, linewidth=2)
            handles.append(handle)

        # Remove borders
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        #ax.spines["bottom"].set_visible(False)

        # Remove ticks on top and right
        ax.tick_params(axis='x', bottom='on', top='off', labelbottom='on', direction='out')
        ax.tick_params(axis='y', left='off', right='off', labelleft='off')

        # Plot the ticks for go-cue and targets on
        plt.xticks((base, targ_on, go, targ_off), ("Center hold", "Target on", "Go cue", "Target off"))

        ax.text(0, np.mean(zz[0]) * 5, '$x$-position', verticalalignment='center')
        ax.text(0, np.mean(zz[1]) * 5, '$y$-position', verticalalignment='center')
        ax.text(0, np.mean(zz[0]) * 5 - 2, '$x$-velocity', verticalalignment='center')
        ax.text(0, np.mean(zz[1]) * 5 - 2, '$y$-velocity', verticalalignment='center')

        if savepath is not None:
            f.savefig(savepath + 'outputs.pdf')
        if not display:
            f.clear()

        if return_handles:
            return (f, handles)
        else:
            return f

    def plot_inputs_switch_pub(self, i=0, savepath=None, filename='inputs.pdf', display=True):
        # i is the trial number

        # Now plot these trial inputs
        f   = plt.figure()
        ax  = f.gca()

        # separation in plotting the inputs along the y-axis
        sepY        = 2

        # offset so ticks don't collide
        tickOffset  = 0.1

        T   = self.trials[i]['info']['epochs']['T'] // self.dt
        Nin = len(self.trials[i]['u'])
        uu  = self.trials[i]['u']
        uu  = uu[[2,1,0]]

        base, targ_on   = np.array(self.trials[i]['info']['epochs']['pre_targets']) // self.dt
        switch, _ = np.array(self.trials[i]['info']['epochs']['switch']) // self.dt
        go, targ_off    = np.array(self.trials[i]['info']['epochs']['check']) // self.dt

        for j in np.arange(Nin):
            # Targ X highest
            ax.plot(np.arange(0, T), tickOffset + j*sepY + uu[j][:-1], color='gray', linewidth=2)

        # Remove borders
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        #ax.spines["bottom"].set_visible(False)

        # Remove ticks on top and right
        ax.tick_params(axis='x', bottom='on', top='off', labelbottom='on', direction='out')
        ax.tick_params(axis='y', left='off', right='off', labelleft='off')

        # set the limits so you can see the bottom and top traces
        ax.set_ylim([-0.11, sepY * (Nin) + 0.11])

        # Set labels
        ax.text(-50, 0.5, 'Go cue', verticalalignment='center')
        ax.text(-50, 2.5, 'Target\n$y$-position', verticalalignment='center')
        ax.text(-50, 4, 'Target\n$x$-position', verticalalignment='center')

        # Plot the ticks for go-cue and targets on
        plt.xticks((base, targ_on, switch, go, targ_off), ("Center hold", "Target\non",  "Switch", "Go cue", "Target\noff"))
        plt.yticks(())

        if savepath is not None:
            f.savefig(savepath + filename)

        if not display:
            f.clf()

    # =================
    # Behavioral output
    # =================
    def plot_outputs_switch_pub(self, i=0, savepath=None, filename='outputs.pdf', display=True, f=None):
        # i is the trial number

        # Now plot these trial inputs
        if f is None:
            f   = plt.figure()
        ax  = f.gca()

        # this one will be a bit more straightforward

        # offset so ticks don't collide

        T       = self.trials[i]['info']['epochs']['T'] // self.dt
        Nout    = len(self.trials[i]['z'])
        zz      = self.trials[i]['z']

        base, targ_on   = np.array(self.trials[i]['info']['epochs']['pre_targets']) // self.dt
        switch, _ = np.array(self.trials[i]['info']['epochs']['switch']) // self.dt
        go, targ_off    = np.array(self.trials[i]['info']['epochs']['check']) // self.dt

        for j in np.arange(Nout):
            # Targ X highest
            ax.plot(np.arange(0, T), zz[j][:-1] - 2 * (j < 2), color='gray', linewidth=2)

        # Remove borders
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        #ax.spines["bottom"].set_visible(False)

        # Remove ticks on top and right
        ax.tick_params(axis='x', bottom='on', top='off', labelbottom='on', direction='out')
        ax.tick_params(axis='y', left='off', right='off', labelleft='off')

        # Plot the ticks for go-cue and targets on
        plt.xticks((base, targ_on, switch, go, targ_off), ("Center hold", "Target\non", "Switch", "Go cue", "Target\noff"))

        ax.text(0, np.mean(zz[0]) * 5, '$x$-position', verticalalignment='center')
        ax.text(0, np.mean(zz[1]) * 5, '$y$-position', verticalalignment='center')
        ax.text(0, np.mean(zz[0]) * 5 - 2, '$x$-velocity', verticalalignment='center')
        ax.text(0, np.mean(zz[1]) * 5 - 2, '$y$-velocity', verticalalignment='center')

        if savepath is not None:
            f.savefig(savepath + filename)
        if not display:
            f.clear()

        return f
    # =================
    # Behavioral output
    # =================
    def plot_outputs_switch_pos(self, i=0, savepath=None, filename='outputs.pdf', display=True, f=None, cond_color=np.array([0,0,0])):
        # i is the trial number

        # Now plot these trial inputs
        if f is None:
            f   = plt.figure()
        ax  = f.gca()

        # this one will be a bit more straightforward

        # offset so ticks don't collide

        T       = self.trials[i]['info']['epochs']['T'] // self.dt
        Nout    = len(self.trials[i]['z'])
        zz      = self.trials[i]['z']

        ax.plot(zz[2,:], zz[3,:], color=cond_color)
#       ax.plot(np.cumsum(zz[0,:]) / 20, np.cumsum(zz[1,:])/ 20, color='r')

        if savepath is not None:
            f.savefig(savepath + filename)
        if not display:
            f.clear()

        return f

    # ========
    # calc R2s
    # ========

    def calc_r2s_all(self):
        # calculates r2s across all trials

        # sort trials

        ytrue = [trial['out'].T for trial in self.trials]
        ydec = [trial['z'].T for trial in self.trials]

        ytrues = np.concatenate(ytrue, axis=0)
        ydecs = np.concatenate(ydec, axis=0)

        r2s_vel = r2_score(ytrues[:,0:2], ydecs[:, 0:2], multioutput='uniform_average')
        r2s_pos = r2_score(ytrues[:,2:], ydecs[:, 2:], multioutput='uniform_average')

        return (r2s_pos, r2s_vel)



# =============================
# Returns color given coherence
# =============================

def plot_linestyle(*args):
    # take pairs (sort, value) and returns the linestyle
    linestyle = '-'

    for sortable, value in args:
        if sortable == 'dirs':
            linestyle = '--' if value == 1 else '-'

    return linestyle


def plot_color(*args):
    # takes pairs (sort, value) and returns the color.

    cond_color = (0, 0, 0, 1)       # cm returns (red, green, blue, alpha)

    for sortable, value in args:
        if sortable == 'cohs':
            cond_color  = cm.RdYlGn(1 - (1 + coh_r(value)) / 2)
        elif sortable == 'rts':
            cond_color  = cm.Blues(value / 400)
        elif sortable == 'conds':
            cond_color  = cm.RdYlGn(value / 360)

    return cond_color

class PSTH(Trial):

    # ==========
    # Initialize
    # ==========

    def __init__(self, rnnfile, modelfile, num_trials=100, seed=1, target_output=False, rnnparams={}, threshold=None, sort=['dirs', 'cohs'], align='cb', Wrec=None):

        # self.sort is a LIST of what to sort by.  If you wanted to sort only by 'coh', then set sort = ['coh'].  This is important to allow sorting by multiple features.
        self.sort       = sort
        self.align      = align
        self.psths      = None
        super(PSTH, self).__init__(rnnfile, modelfile, num_trials, seed, target_output, rnnparams, threshold, Wrec)
        #if Wrec is not None:
        #    pdb.set_trace()
        #    self.rnn.Wrec = Wrec

        # We only operate on successful trials
        #self.filter_trials()
        self.define_align_times()

    # =====================================================================
    # Set the align time for each PSTH.  These are hard-coded and how we handle the alignment internally.
    # =====================================================================
    def define_align_times(self):

        if self.align == 'cb':
            start_time  = 500 // self.dt
            align_time  = 2000 // self.dt
            stop_time   = 4000 // self.dt
        elif self.align == 'mv':
            start_time  = 1500 // self.dt
            align_time  = 3500 // self.dt
            stop_time   = 4500 // self.dt
        elif self.align == 'end':
            start_time  = (2 * self.longest_trial() - 3000) // self.dt
            align_time  = 2 * self.longest_trial() // self.dt
            stop_time   = align_time
        elif self.align == 'start':
            start_time  = 0
            align_time  = 0
            stop_time   = 3000 // self.dt
        else:
            print 'No valid align time.  Aligning to start.'
            start_time  = 0
            align_time  = 0
            stop_time   = 4000 // self.dt

        self.start_time = start_time
        self.align_time = align_time
        self.stop_time  = stop_time

    # =====================================================================
    # Set a new alignment for the PSTH
    # =====================================================================
    def set_align(self, align='cb'):
        self.align = align
        self.define_align_times()

    # =========================================================================
    # Calculates where to insert the single trial trajectory in the PSTH matrix
    # =========================================================================
    def calc_align_idxs(self):

        check_onsets    = self.cbs // self.dt
        trial_ends      = self.Ts // self.dt
        check_plus_rts  = (self.cbs + self.rts) // self.dt

        # make check_plus_rts an int array.
        nan_locs = np.where(np.isnan(check_plus_rts))[0]
        check_plus_rts = check_plus_rts.astype(int)
        check_plus_rts[nan_locs] = 0

        if self.align == 'cb':
            start_idxs      = self.align_time - check_onsets
            stop_idxs       = start_idxs + trial_ends + 1

        elif self.align == 'mv':
            start_idxs      = self.align_time - check_plus_rts
            stop_idxs       = start_idxs + trial_ends + 1

        elif self.align == 'end':
            start_idxs      = self.align_time - trial_ends - 1
            stop_idxs       = self.align_time * np.ones_like(trial_ends)

        elif self.align == 'start':
            start_idxs      = self.align_time * np.ones_like(trial_ends)
            stop_idxs       = self.align_time + trial_ends + 1

        else:
            print 'Not a valid align time, aligning to the start'
            start_idxs      = self.align_time * np.ones_like(trial_ends)
            stop_idxs       = self.align_time + trial_ends + 1

        return start_idxs, stop_idxs

    # ================================================
    # Sort trials according to a useful categorization
    # ================================================
    def sort_trials(self, threshold=1, rt_bin=25):
        # threshold is the number of trials needed in a condition to actually append it.
        # The function sorts by self.sort.  self.sort can be a list of how to sort, and will sort in that order.
        #   - Note, the newest version of sort_trials only sorts by giving you the indices of the desired condition.

        dt          = self.dt
        conds       = []
        uconds      = []

        # Get a list of conditions that we need to sort by.
        for i in np.arange(len(self.sort)):
            # sorting is a bit unique for rt
            if self.sort[i] == 'rts':
                min_rt  = np.floor(np.nanmin(self.rts) / rt_bin) * rt_bin # rounded to rt_bin
                max_rt  = np.ceil(np.nanmax(self.rts) / rt_bin) * rt_bin # rounded to rt_bin
                conds   += [np.floor(self.rts / rt_bin) * rt_bin] # put everything into its correct bin
                uconds  += [np.arange(min_rt, max_rt, rt_bin)]

            else:
                conds       += [self.__getattribute__(self.sort[i])]
                uconds      += [np.unique(conds[i])]

        zip_conds       = np.array(zip(*conds))
        sorted_trials   = []
        # iterates over all things to sort by.
        for cond in itertools.product(*uconds):
            one_cond            = {}
            one_cond['sort']    = self.sort
            one_cond['cond']    = cond

            one_cond['idxs']    = np.where([np.all(zip_cond == cond) for zip_cond in zip_conds])[0]

            if len(one_cond['idxs']) > threshold:
                sorted_trials.append(one_cond)
            else:
                print '{} condition {} did not have at least {} trials, so it was discarded.'.format(str(self.sort), str(cond), str(threshold))

        # sorted_trials is a list, where each entry of the list is a dict containing the identifier 'cond' and 'trials'.
        #pdb.set_trace()
        return np.array(sorted_trials)


    # ========================================================
    # Get longest trial from the trials struct - used for PSTH
    # ========================================================
    def longest_trial(self):
        return np.max([len(trial['z'][0,:]) for trial in self.trials]) * self.dt

    # =================
    # Calculates a PSTH
    # =================
    def calc_psth(self, sub_idxs=None, field='r'):

        if sub_idxs is None:
            sub_idxs = np.arange(len(self.trials))

        dt          = self.dt
        max_length  = self.longest_trial()
        num_neurons = self.trials[0][field].shape[0]
        ntrials     = len(sub_idxs)

        p_mtx = np.empty((ntrials, num_neurons, 2*max_length))
        p_mtx[:] = np.nan       # initialize to nans

        # insert trials into each psth
        start_idxs, stop_idxs   = self.calc_align_idxs()

        # subsample trials
        trials      = self.trials[sub_idxs]
        start_idxs  = start_idxs[sub_idxs]
        stop_idxs   = stop_idxs[sub_idxs]

        for i in range(ntrials):
            p_mtx[i,:,start_idxs[i]:stop_idxs[i]] = trials[i][field]

        # calculate the psth
        psth = np.nanmean(p_mtx, axis=0)

        # replace nans with 0's.
        nan_mask = np.all(np.isnan(psth), axis=0) # check if all rows along a column are NaNs.
        psth[:, nan_mask] = 0

        # return the psth
        return psth[:, self.start_time:self.stop_time]

    # ===================
    # Generates the PSTHs
    # ===================

    def gen_psth(self, field='r', threshold=1, rt_bin=25):
        # A few things need to happen here.
        # First we need to sort the trials.
        # Then we need to calculate the psths for each collection of trials
        # Then we'll store this as self.psth.

        # sort the trials
        sorted_trials = self.sort_trials(threshold=threshold, rt_bin=rt_bin)

        # now calculate the PSTHs for each condition
        psth_collection = []
        for i in np.arange(len(sorted_trials)):

            one_psth            = {}
            one_psth['sort']    = sorted_trials[i]['sort']
            one_psth['cond']    = sorted_trials[i]['cond']
            one_psth['psth']    = self.calc_psth(sub_idxs=sorted_trials[i]['idxs'], field=field)
            one_psth['x_psth']  = self.calc_psth(sub_idxs=sorted_trials[i]['idxs'], field='x')
            one_psth['u_psth']  = self.calc_psth(sub_idxs=sorted_trials[i]['idxs'], field='u')


            psth_collection.append(one_psth)

        self.psths = np.array(psth_collection)

    # =======================
    # Plots a PSTH collection
    # =======================

    def plot_psth(self, N=None, savepath=None):
        # with N you can control how many PSTHs are plotted.
        if N is None:
            N = self.rnn.p['N']

        # Iterate over the number of neurons
        for i in np.arange(N):
            print 'Plotting PSTH for neuron {}'.format(i)

            f   = plt.figure()
            ax  = f.gca()

            # Plot each PSTH.
            for j in np.arange(len(self.psths)):

                cond_color  = plot_color(*zip(self.psths[j]['sort'], self.psths[j]['cond']))
                cond_ls     = plot_linestyle(*zip(self.psths[j]['sort'], self.psths[j]['cond']))

                ax.plot(np.arange(self.start_time, self.stop_time) * self.dt, self.psths[j]['psth'][i,:], color=cond_color, linestyle=cond_ls)
                ax.axvline(self.align_time * self.dt, color='b')
                ax.set_title('Neuron {}, sortables={}, align={}'.format(i, self.sort, self.align))

                if savepath is not None:
                    f.savefig(savepath + 'neuron={}_sortables={}_align={}.pdf'.format(i, self.sort, self.align))

    def plot_psth_pub(self, N=None, savepath=None, filename=None, display=True):

        # This will plot PSTHs aligned to go_cue
        prior_align = None
        prior_sort  = None
        start_plot_psth = 70 #MK 09-23-19 hack so the analyze_fixed_cb starts at the beginning on the trial
        end_psth = 50
        #if self.align != 'cb':

        #   print 'Re-aligning the data to cb'
        #   prior_align = self.align
        #   prior_sort  = self.sort

        #   self.sort   = ['conds']
        #   self.set_align(align='cb')
        #   self.gen_psth()

        base, targ_on   = np.array(self.trials[0]['info']['epochs']['pre_targets']) // self.dt
        go, targ_off    = np.array(self.trials[0]['info']['epochs']['check']) // self.dt
        delay           = (go - targ_on) * self.dt

        # with N you can control how many PSTHs are plotted.
        if N is None:
            N = self.rnn.p['N']

        if not isinstance(N, list):
            N = np.arange(N)

        # Iterate over the number of neurons
        for i in N:
            print 'Plotting PSTH for neuron {}'.format(i)

            f   = plt.figure()
            ax  = f.gca()

            # Plot each PSTH.
            for j in np.arange(len(self.psths)):

                cond_color  = plot_color(*zip(self.psths[j]['sort'], self.psths[j]['cond']))
                cond_ls     = plot_linestyle(*zip(self.psths[j]['sort'], self.psths[j]['cond']))
                #pdb.set_trace()
                ax.plot(np.arange(self.start_time + start_plot_psth, self.stop_time-end_psth) * self.dt, self.psths[j]['psth'][i,start_plot_psth:-end_psth], color=cond_color, linestyle=cond_ls)
                #ax.axvline(self.align_time * self.dt, linestyle='--')
                #ax.axvline(self.align_time * self.dt - delay, linestyle='--')

                # Remove borders
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax.spines["bottom"].set_visible(False)

                # Remove ticks on top and right
                ax.get_xaxis().tick_bottom()
                ax.get_yaxis().tick_left()

                # Plot the ticks for go-cue and targets on
                #plt.xticks((self.align_time*self.dt - delay, self.align_time*self.dt), ("Targets on", "Checkerboard on"))
                # in chand's task, we have multiple delay periods, so it only makes sense to show checkerboard on.  In cora's task there were also multiple delay periods but for results we averaged to the same delay period.
                # I'm plotting a tick for 0 because this xticks function throws an error if there's only one tick -_-
                plt.xticks((0, self.align_time*self.dt), ("0", "Checkerboard on"))
                plt.yticks(())

            # Add line for scalebar
            xmin, xmax  = ax.get_xlim()
            ymin, ymax  = ax.get_ylim()
            yrange      = ymax - ymin
            ax.plot((xmax-400, xmax-100), (ymin + 0.01*yrange, ymin + 0.01*yrange), linewidth=2)

            # Add text for scalebar
            ax.text(xmax-250, ymin + 0.05*yrange, '300 ms', horizontalalignment='center')

            if filename is None:
                filename = ''

            if savepath is not None:
                f.savefig(savepath + filename + '_sortables={}_align={}_neuron={}.pdf'.format(self.sort, self.align, i))

            if not display:
                f.clear()

        # Re-align data if necessary
        if prior_align is not None:
            self.sort   = ['conds']
            self.set_align(align=prior_align)
            self.gen_psth()

    def calc_r2s(self):
        # calculates r2s per condition

        # sort trials
        sorted_trials = self.sort_trials()
        r2s_pos = []
        r2s_vel = []

        for i in np.arange(len(sorted_trials)):
            ytrue = [self.trials[idx]['out'].T for idx in sorted_trials[i]['idxs']]
            ydec = [self.trials[idx]['z'].T for idx in sorted_trials[i]['idxs']]

            ytrues = np.concatenate(ytrue, axis=0)
            ydecs = np.concatenate(ydec, axis=0)

            r2s_pos.append(r2_score(ytrues[:,0:2], ydecs[:, 0:2]))
            r2s_vel.append(r2_score(ytrues[:,2:], ydecs[:, 2:]))

        return (r2s_pos, r2s_vel)

    # =============
    # Extract PSTHs
    # =============

    # Extracts a PSTH given a filter.  This code was used so much I decided to make it a function
    def extract_psths(self, filters=None, field='r'):
#pdb.set_trace()
        if field == 'r':
            psths = np.array([psth['psth'] for psth in self.psths])
        elif field == 'x':
            psths = np.array([psth['x_psth'] for psth in self.psths])
        elif field == 'u':
            psths = np.array([psth['u_psth'] for psth in self.psths])
        else:
            assert False, 'Invalid field.'

        #all_conds  = np.array([zip(psth['sort'], psth['cond']) for psth in self.psths])
        all_conds   = [zip(psth['sort'], psth['cond']) for psth in self.psths]
        idxs        = find_conds(all_conds, filters)

        return psths[idxs]

    # ============================
    # Choice probability functions
    # ============================
    def calc_pds(self, bounds=[-500, 200], align='mv', reset_align=True, sort=None):
        # Computes the directional and coherence preferred direction, for the purposes of later calculating ROC curves.
        # align is where the psth's are aligned to;
        # bounds are where you take the mean firing rate over with respect to align, i.e.,
        #     align + bounds[0] to align + bounds[1]
        # reset align resets the align back to prior_align at the end.

        # if sort is None, go by what the class sort has already done.

        regen_psth = False

        # Now, there ought to be two PSTHs in these scenarios, i.e., they are either sorted by direction, prompted direction, or if it was red or green.
        if sort is not None:
            self.sort = sort
            regen_psth = True

        prior_align = self.align

        if prior_align != align:
            self.set_align(align=align)
            regen_psth = True

        if regen_psth:
            self.gen_psth()

        if self.sort == ['cohs']:
            # we need to reduce to two psths for red and green
            filters = [('cohs', 'r')]
            psth0 = self.extract_psths(filters=[('cohs', 'r')])
            psth1 = self.extract_psths(filters=[('cohs', 'g')])
        else:
            psth0 = self.psths[0]
            psth1 = self.psths[1]

        # This assert removed b/c we have to manually extract the coh PSTHs that average over r and g
        # assert len(self.psths) == 2, 'The sort did not return only two types of PSTH.'

        # Initialize appropriate variables to calculate pds
        N = len(self.psths[0]['psth'][:,0]) # the number of neurons
        pds = np.ones(N) * np.NaN

        # We're always going to compare polar opposites. 0 is left or red, 1 is right or green
        # JON: check that 0 is indeed left or red; 1 is right or green.  When verified, comment below this line.

        # Iterate over all the neurons
        for i in np.arange(N):

            tstart  = self.align_time + bounds[0]
            tstop   = self.align_time + bounds[1]

            # Let's look at the average in the desired range.
            rates_class1    = np.mean(psth1['psth'][i, tstart:tstop])
            rates_class0    = np.mean(psth0['psth'][i, tstart:tstop])
            rates_diff      = rates_class1 - rates_class0

            # assign -- give the base value, as later it can be processed.
            pds[i] = rates_diff

        if prior_align != align  and reset_align:
            self.set_align(align=prior_align)

        return pds

    # Targeted dimensionality reduction
    def tdr(self, idx1=np.hstack((np.arange(80), np.arange(240, 260))), idx2=np.hstack((np.arange(80, 160), np.arange(260, 280))), idx3=np.hstack((np.arange(160, 240), np.arange(280, 300)))):

        ntrials = self.ntrials
        nunits = 300
        ntime = 260
        rates = np.zeros((nunits, ntrials, ntime))
        for i in range(ntrials):
            rates[:, i, :] = self.trials[i]['r'][:, :ntime]
        zrates = rates

        # compute the units zscore across trial and time
        m = np.mean(zrates, axis=(1, 2))
        std = np.std(zrates, axis=(1, 2))
        zrates2 = (zrates.reshape(nunits, ntrials * ntime) - m.reshape(nunits, 1)).reshape(nunits, ntrials, ntime)
        zrates3 = (zrates2.reshape(nunits, ntrials * ntime) / std.reshape(nunits, 1)).reshape(nunits, ntrials, ntime)


        ncoef = 4
        F = np.zeros((ncoef, ntrials))
        F[0, :] = self.choices
        F[1, :] = self.scols
        F[2, :] = self.lrs
        F[3, :] = 1

        A = np.linalg.inv(F.dot(F.T)).dot(F)
        betas = np.tensordot(A, np.swapaxes(rates, 0, 1), axes=([1], [0]))

        maxind1 = {}
        maxind2 = {}
        maxind3 = {}
        for j in range(ncoef):
            maxnorm1 = 0
            maxnorm2 = 0
            maxnorm3 = 0
            for i in range(ntime):
                temp1 = np.linalg.norm(betas[j, idx1, i])
                temp2 = np.linalg.norm(betas[j, idx2, i])
                temp3 = np.linalg.norm(betas[j, idx3, i])
                if temp1 > maxnorm1:
                    maxnorm1 = temp1
                    maxind1[j] = i
                if temp2 > maxnorm2:
                    maxnorm2 = temp2
                    maxind2[j] = i
                if temp3 > maxnorm3:
                    maxnorm3 = temp3
            maxind3[j] = i
        return betas, maxind1, maxind2, maxind3

    # ============================
    # Choice probability functions
    # ============================

    def rate_dist(self, rates, min_rate=0, max_rate=100, bins=100):
        hist, edges = np.histogram(rates, bins=bins, range=(min_rate, max_rate))
        return (edges, hist)

    def choice_prob_single(self, rates, bins=100):
        # rates is a list of two rates, rates0 being one condition and rates1 being the other
        min_rate    = min(min(rates[0]), min(rates[1]))
        max_rate    = max(max(rates[0]), max(rates[1]))

        edges, rate_dist0   = self.rate_dist(rates[0], min_rate=min_rate, max_rate=max_rate, bins=bins)
        _, rate_dist1       = self.rate_dist(rates[1], min_rate=min_rate, max_rate=max_rate, bins=bins)

        # these are the pmf distributions of the rates in the two conditions.
        rate_dist0 = rate_dist0 / len(rates[0])
        rate_dist1 = rate_dist1 / len(rates[1])

        # these are the cdf of the rates in the two conditions
        cdist0 = np.cumsum(rate_dist0)
        cdist1 = np.cumsum(rate_dist1)

        # insert 0's in case there's a large instance of 0 firing rates
        cdist0 = np.insert(cdist0, 0, 0)
        cdist1 = np.insert(cdist1, 0, 0)

        # false and true positive rates
        false_pos_rate  = np.flipud(1 - cdist0)
        true_pos_rate   = np.flipud(1 - cdist1)

        choice_prob = np.trapz(true_pos_rate, false_pos_rate)

        return choice_prob

    def choice_prob(self, min_rate=0, bounds=[-500,500]):
        # This function calculates the choice probability for all neurons.
        # The choice probability category is determined by the sort.
        # min_rate is the minimum rate that the max of the rates must exceed.
        # bounds is the amount of time around align to calculate the mean firing rates.

        ### NOTE EVERYTHING SHOULD BE ALIGNED TO MV, that's why the rates are compared between 200:300

        N = len(self.trials[0]['r'][:,0])

        choice_probs = np.ones(N) * np.nan

        time_mv = self.rts + [int(trial['info']['epochs']['check'][0]) for trial in self.trials]
        time_mv = time_mv // self.dt
        time_mv = time_mv.astype(int)
        rstarts = time_mv + bounds[0] // self.dt
        rends   = time_mv + bounds[1] // self.dt

        # Extract single trial rate vectors to compare, rate0 and rate1; sorted_trials uses self.sort, that's why we don't need if statements here for what sort.
        sorted_trials = self.sort_trials()
        trials0     = self.trials[sorted_trials[0]['idxs']]
        trials1     = self.trials[sorted_trials[1]['idxs']]

        # make rates
        rates0      = np.array([np.mean(trial['r'][:, rstarts[idx]:rends[idx]], axis=1) for idx, trial in zip(sorted_trials[0]['idxs'], trials0)]).T
        rates1      = np.array([np.mean(trial['r'][:, rstarts[idx]:rends[idx]], axis=1) for idx, trial in zip(sorted_trials[1]['idxs'], trials1)]).T

        # remove nans
        idx_nans0 = np.array([np.isnan(rate) for rate in rates0])
        idx_keep0 = np.array([not np.all(bool_across_trials) for bool_across_trials in idx_nans0.T])
        idx_nans1 = np.array([np.isnan(rate) for rate in rates1])
        idx_keep1 = np.array([not np.all(bool_across_trials) for bool_across_trials in idx_nans1.T])

        rates0  = rates0[:, idx_keep0]
        rates1  = rates1[:, idx_keep1]
        # Iterate over the neurons
        for i in np.arange(N):
            if np.max(np.concatenate((rates0[i,:], rates1[i,:]))) > min_rate:
                if np.mean(rates1[i,:]) > np.mean(rates0[i,:]):
                    choice_probs[i] = self.choice_prob_single([rates0[i,:], rates1[i,:]])
                else:
                    choice_probs[i] = self.choice_prob_single([rates1[i,:], rates0[i,:]])
                 # noisy integration of ROC led to cp < 0.5, in which case we should have done it in the flipped order.
                if choice_probs[i] < 0.5:
                    choice_probs[i] = 1-choice_probs[i]
        return choice_probs


    def choice_prob_bugged(self, min_rate=0, bounds=[-500,200]):
        # This function calculates the choice probability for all neurons.
        # The choice probability category is determined by the sort.
        # min_rate is the minimum rate that the max of the rates must exceed.
        # bounds is the amount of time around align to calculate the mean firing rates.


        ### NOTE EVERYTHING SHOULD BE ALIGNED TO MV, that's why the rates are compared between 200:300

        N = len(self.trials[0]['r'][:,0])

        choice_probs = np.ones(N) * np.nan

        # Extract single trial rates
        if self.sort == ['cohs']:
            rates0  = self.extract_psths(filters=[('cohs', 'r')]) #red
            rates1  = self.extract_psths(filters=[('cohs', 'g')]) #green
            # these rates are conds x neurons x time, so this way is fundamentally flawed

            # Average rates across conditions.
            rates0  = np.mean(rates0, axis=0)
            rates1  = np.mean(rates1, axis=0)
        elif self.sort == ['dirs']:
            rates0  = self.psths[0]['psth'] #left -- but check this to be sure
            rates1  = self.psths[1]['psth'] #right

        elif self.sort == ['pdirs']:
            rates0  = self.psths[0]['psth'] #left -- but check this to be sure
            rates1  = self.psths[1]['psth'] #right
        elif self.sort == ['scols']:
            rates0  = self.psths[0]['psth'] #red
            rates1  = self.psths[1]['psth'] #green
        else:
            assert False, 'There was not a valid sort for PSTHs'

        # Iterate over the neurons
        for i in np.arange(N):
            if np.max(np.concatenate((rates0[i,:], rates1[i,:]))) > min_rate:
#               if np.mean(rates1[i,:]) > np.mean(rates0[i,:]):
#                   choice_probs[i] = self.choice_prob_single([rates0[i,:], rates1[i,:]])
#               else:
#                   choice_probs[i] = self.choice_prob_single([rates1[i,:], rates0[i,:]])
## JCK for getting rid of low fr.
                if np.mean(rates1[i,:]) > np.mean(rates0[i,:]):
                    choice_probs[i] = self.choice_prob_single([rates0[i,200:300], rates1[i,200:300]])
                else:
                    choice_probs[i] = self.choice_prob_single([rates1[i,200:300], rates0[i,200:300]])
                #if choice_probs[i] > 0.8:
                #   pdb.set_trace()
        return choice_probs

    def scatter_choice_pub(self, f=None, cond_color=None, xlim=None, ylim=None, savepath=None, filename=None, regen_psth=True):

        prior_align = self.align
        prior_sort  = self.sort

        # First, get the selected color choice porbabilities.
        self.sort = ['scols']
        self.set_align(align='mv')
        self.gen_psth()
        choice_probs_col    = self.choice_prob()

        # Second, get the selected direction choice probabilities.
        self.sort = ['dirs']
        self.set_align(align='mv')
        self.gen_psth()
        choice_probs_dir    = self.choice_prob()

        # Now scatter plot
        if f is None:
            f = plt.figure()
        ax = f.gca()

        if cond_color is None:
            cond_color = [0,0,0]

        ax.plot(choice_probs_dir, choice_probs_col, color=cond_color, linestyle='', marker='.')

        if xlim is None:
            xlim = [0.5, 1]

        if ylim is None:
            ylim = [0.5, 1]

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # remove top and right axes
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # remove top and right ticks
        ax.tick_params(axis='x', bottom='on', top='off', labelbottom='on')
        ax.tick_params(axis='y', left='on', right='off', labelleft='on')

        if filename is None:
            filename = 'scatter.pdf'

        # save it down
        if savepath is not None:
            f.savefig(savepath + filename.format(self.dims + 1, self.sort, self.align))

        # regenerate the psth
        if regen_psth:
            self.sort = prior_sort
            self.set_align(align=prior_align)
            self.gen_psth()

        return f

# ============
# PCA function
# ============

def pca(data):  # the dimensions are rows, the observations are columns
    # Performs PCA on what I'm used to (i.e., a transposed data matrix; I don't know why those statisticians changed things.)

    means           = np.mean(data, axis=1)
    data_centered   = (data.T - means).T
    evecs, evals, _ = np.linalg.svd(np.cov(data_centered))
    scores          = np.dot(evecs.T, data_centered)

    return evecs, evals, scores, means

# ========================
# Factor analysis function
# ========================

def fa(data, D=2): # the dimensions are rows, the observations are columns; D is the dimensionality
    # Performs FA on dataset.
    facAn   = FactorAnalysis(n_components=D)
    facAn.fit(data.T)

    # facAn.components_ is W
    # diag(facAn.noise_variance_) are the uniqueness
    # mu is the mean of the data
    W   = facAn.components_.T
    Psi = np.diag(facAn.noise_variance_)
    mu  = np.mean(data, axis=1)

    # orthonormalize the components
    U,S,V   = np.linalg.svd(W)
    S       = np.diag(S)

    # project
    data_centered   = (data.T - mu).T
    data_cov        = W.dot(W.T) + Psi
    data_cov_inv    = np.linalg.inv(data_cov)
    projector       = S.dot(V.T).dot(W.T).dot(data_cov_inv)
    scores          = projector.dot(data_centered)

    return U, data_cov, scores, projector
# Helper functions

def norms(x):
    # gets the norm of each vector in the matrix x
    xn = np.zeros(x.shape[1])
    for i in range(len(xn)):
        xn[i] = np.linalg.norm(x[:,i])

    return xn

def relu(x):
    return x * (x > 0)

def d_relu(x):
    return (x > 0)

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - np.tanh(x)**2

def tf_rnn_rdot_norm(rnn, x, inputs=np.array((0,0,0,0))):
    return 0.5 * T.dot(tf_rnn_rdot(rnn, x, inputs), tf_rnn_rdot(rnn, x, inputs))

def tf_rnn_rdot(rnn, x, inputs=np.array((0,0,0,0))):

    if rnn.p['hidden_activation'] == 'tanh':
        act = tanh
    elif rnn.p['hidden_activation'] == 'rectify':
        act = relu
    else:
        assert False, 'Invalid activation function'

    tau     = rnn.p['tau']
    dt      = rnn.p['dt']
    alpha   = dt / tau

    # No noise here.
    xn = x + alpha*(-x + T.dot(rnn.Wrec, act(x)) + rnn.brec + T.dot(rnn.Win, inputs))

    # calculate rdot
    rn  = act(xn)
    r   = act(x)

    return (rn - r) / dt

def tf_rnn_xdot(rnn, x, inputs=np.array((0,0,0,0))):

    if rnn.p['hidden_activation'] == 'tanh':
        act = tanh
    elif rnn.p['hidden_activation'] == 'rectify':
        act = relu
    else:
        assert False, 'Invalid activation function'

    return (-x + T.dot(rnn.Wrec, act(x)) + rnn.brec + T.dot(rnn.Win, inputs)) / rnn.p['tau']

def tf_rnn_xdot_norm(rnn, x, inputs=np.array((0,0,0,0))):
    return 0.5 * T.dot(tf_rnn_xdot(rnn, x, inputs), tf_rnn_xdot(rnn, x, inputs))

def find_conds(all_conds, filters):
    # assumes all_conds is a list of lists of tuples.
    # e.g., all_conds = [[('dirs', -1), ('cohs', 11)], [('dirs', -1), ('cohs', 45)], [('dirs', -1), ('cohs', 67)], [('dirs', -1), ('cohs', 78)], [('dirs', -1), ('cohs', 90)], [('dirs', -1), ('cohs', 101)], [('dirs', -1), ('cohs', 117)], [('dirs', -1), ('cohs', 124)], [('dirs', -1), ('cohs', 135)], [('dirs', -1), ('cohs', 147)], [('dirs', -1), ('cohs', 158)], [('dirs', -1), ('cohs', 180)], [('dirs', -1), ('cohs', 214)], [('dirs', 1), ('cohs', 11)], [('dirs', 1), ('cohs', 45)], [('dirs', 1), ('cohs', 67)], [('dirs', 1), ('cohs', 78)], [('dirs', 1), ('cohs', 90)], [('dirs', 1), ('cohs', 101)], [('dirs', 1), ('cohs', 124)], [('dirs', 1), ('cohs', 135)], [('dirs', 1), ('cohs', 147)], [('dirs', 1), ('cohs', 158)], [('dirs', 1), ('cohs', 180)], [('dirs', 1), ('cohs', 214)]]
    # as obtained via 'all_conds = np.array([zip(psth['sort'], psth['cond']) for psth in self.psths])'
    # filter is a list of tuples to sort by, e.g., filter = [('dirs', -1), 'cohs', 'r'].  Note, 'cohs' is handled specially, so that 'r' looks for 'coh' greater than 225/2, and vice versa for green if you so desire.
    # this returns the indices of all_conds that pass the filter.

    if filters is None:
        return np.arange(len(all_conds))
    else:
        idxs    = []
        cohs    = np.array([11, 45, 67, 78, 90, 101, 108, 117, 124, 135, 147, 158, 180, 214])
        r_cohs  = cohs[np.where(cohs > 225//2)[0]]
        g_cohs  = cohs[np.where(cohs < 225//2)[0]]

        sort_names  = np.array([f[0] for f in filters])
        where_coh   = np.where(sort_names == 'cohs')[0]

        # iterate in reversed order so I can just remove these elements from the list after I'm done.
        for i in sorted(where_coh, reverse=True):
            # check if this is a string, if so, then we add to the filters
            if type(filters[i][1]) is str:
                add_filters = zip(itertools.repeat('cohs'), r_cohs) if filters[i][1] == 'r' else zip(itertools.repeat('cohs'), g_cohs)
                filters     = filters + add_filters
                del filters[i]

        for f in filters:
            # Do logical statement over each condition.  This is a messy list comprehension.  It works.
            acceptable_conds    = [np.any(np.all(np.array(cond) == f, axis=1)) for cond in all_conds]
            these_idxs          = np.where(acceptable_conds)[0]

            idxs.append(these_idxs)

        # we do a union over the cohs and an intersection over the rest
        sort_names  = np.array([f[0] for f in filters])
        where_coh   = np.where(sort_names == 'cohs')

        if where_coh:
            idxs        = np.array(idxs)
            coh_idxs    = np.hstack(idxs[where_coh])
            idxs        = np.delete(idxs, where_coh)

            # now cast back to list... this is probalby not as clean as it could be
            idxs        = idxs.tolist()
            idxs.append(coh_idxs)

        # Return the intersection
        return reduce(np.intersect1d, idxs)



class Dynamics(PSTH):

    # ==========
    # Initialize
    # ==========

    def __init__(self, rnnfile, modelfile, num_trials=100, seed=1, target_output=False, rnnparams={}, threshold=None, sort=['dirs', 'cohs'], align='cb', dims=np.array((0,1)), partition_pca=None):

        # Init as a PSTH and generate them
        super(Dynamics, self).__init__(rnnfile, modelfile, num_trials, seed, target_output, rnnparams, threshold, sort, align)
        self.gen_psth()

        # Calculate the matrix to perform PCA on
        data    = np.hstack([one_psth['psth'] for one_psth in self.psths])
        data_x  = np.hstack([one_psth['x_psth'] for one_psth in self.psths])
        data_rd = np.hstack([np.diff(one_psth['psth'], axis=1) for one_psth in self.psths])

        # Assign all class variables, including the PCs.  This is automatically performed.
        self.PCs, self.evals, _, self.means = pca(data)
        self.x_PCs, _, _, self.x_means      = pca(data_x)
        self.dims                           = np.array(dims)

        # Assign the factors
        self.FAs, self.cov, self.scores, self.FA_projector = fa(data)

        # partitioned PCA to do some analyses
        #  assumes the user passed in a list of idxs for the partitions as 'partition_pca'
        if partition_pca is not None:
            num_partitions = len(partition_pca)
            self.partitions = partition_pca
            self.partition_pca = []
            for i in np.arange(num_partitions):
                tPCs, _, _, tmeans = pca(data[partition_pca[i], :])
                txPCs, _, _, txmeans = pca(data_x[partition_pca[i], :])
                pc_dict = {'PCs': tPCs,
                            'means': tmeans,
                            'x_PCs': txPCs,
                            'x_means': txmeans}

                self.partition_pca.append(pc_dict)

       # assign the PC rate dimensions
        self.rPCs, self.rEvals, _, self.rMeans = pca(data_rd)

        n = 100     # hack while i fix isoWr code
        rPCEvals = []
        for i in np.arange(n):
            rPCEvals.append(self.rPCs[:,i].T.dot(self.PCs).dot(np.diag(self.evals)).dot(self.PCs.T).dot(self.rPCs[:,i]))
        self.rPCEvals = rPCEvals

        # compile necessary theano functions
        self.compile_theano_costs()

        # Threshold value of the objective function to arrive at a fixed point.  Modify this to change how lenient you are in finding fixed points.
        self.fp_threshold                   = 1e-8
        self.fpx_threshold                  = 1e-9

        # How the space is to be partitioned for dynamics flow fields -- this assigns self.partition_points
        self.partition_space()
        self.partition_deltas()

        # Define the activation function
        if self.rnn.p['hidden_activation'] == 'tanh':
            self.act = tanh
            self.invAct = np.arctanh
        elif self.rnn.p['hidden_activation'] == 'rectify':
            self.act = relu
            self.invAct = relu      # can't figure it out if it's less than 0, so just set it to 0.
        else:
            assert False, 'Invalid activation function'

    def steifel_cost(self, X):
        data        = np.hstack([one_psth['psth'] for one_psth in self.psths])
        data_dots   = np.hstack([np.diff(one_psth['psth'], axis=1) for one_psth in self.psths])

        Wr_minus_I = autonp.dot(self.Wr - autonp.eye(self.Wr.shape[0]))
        adjacent_states = autonp.dot(Wr_minus_I, data_dots)
        data_cov = autonp.cov(data.T)
        log_barrier = autonp.log(autonp.matmul(autonp.matmul(X[:,0].T, data_cov), X[:,0]))
        log_barrier += autonp.log(autonp.matmul(autonp.matmul(X[:,1].T, data_cov), X[:,1]))

        return 10*np.linalg.norm(np.matmul(Wr, ortho_space), ord='fro') + log_barrier


    # ==============================
    # Plots the principal components
    # ==============================

    def plot_pcs(self, f=None, savepath=None, window=None, filters=None, cond_ls=None, cond_color=None):

        # This plots psths, and if you specify 'filters', you can plot a subset of the psths.
        # filter is an array of tuples to sort by, e.g., [('dirs', -1), ('cohs', 'red')].
        # note for cohs, it will accept red and green and filter based off the cb coherence.

        if f is None:
            f = plt.figure()
        ax = f.gca()

        if window is None:
            window = np.arange(self.psths[0]['psth'].shape[1])

        all_conds   = np.array([zip(psth['sort'], psth['cond']) for psth in self.psths])
        idxs        = find_conds(all_conds, filters)

        # Plot each PSTH.
        for j in idxs:

            # Project the data
            pc_scores   = np.dot(self.PCs[:, self.dims].T, (self.psths[j]['psth'].T - self.means).T)

            # Plot colors and line-styles
            c_color     = plot_color(*zip(self.psths[j]['sort'], self.psths[j]['cond'])) if cond_color is None else cond_color
            c_ls        = plot_linestyle(*zip(self.psths[j]['sort'], self.psths[j]['cond'])) if cond_ls is None else cond_ls

            # Plot the PCs
            ax.plot(pc_scores[0,window], pc_scores[1,window], color=c_color, linestyle=c_ls)
            ax.set_title('PCs {}, sortables={}, align={}'.format(self.dims + 1, self.sort, self.align))

            ax.plot(pc_scores[0, window[-1]], pc_scores[1, window[-1]], marker='o', markerfacecolor='y')

            # save it down
            if savepath is not None:
                f.savefig(savepath + 'PCs={}_sortables={}_align={}.pdf'.format(self.dims + 1, self.sort, self.align))

        return f

    def plot3_pcs_area(self, f=None, savepath=None, window=None, filters=None, cond_ls=None, cond_color=None, partition_idx=0, is_x=False, alpha=1):

        # This plots psths, and if you specify 'filters', you can plot a subset of the psths.
        # filter is an array of tuples to sort by, e.g., [('dirs', -1), ('cohs', 'red')].
        # note for cohs, it will accept red and green and filter based off the cb coherence.

        if f is None:
            f = plt.figure()
        ax = f.gca(projection="3d")

        if window is None:
            window = np.arange(self.psths[0]['psth'].shape[1])

        all_conds   = np.array([zip(psth['sort'], psth['cond']) for psth in self.psths])
        idxs        = find_conds(all_conds, filters)

        # grab out the partition PCs, etc.
        PCs = self.partition_pca[partition_idx]['PCs']
        means = self.partition_pca[partition_idx]['means']

        if is_x:
            PCs = self.partition_pca[partition_idx]['x_PCs']
            means = self.partition_pca[partition_idx]['x_means']

        # Plot each PSTH.
        for j in idxs:

            # Project the data
            pc_scores   = np.dot(PCs[:, self.dims].T, (self.psths[j]['psth'][self.partitions[partition_idx]].T - means).T)

            if is_x:
                pc_scores   = np.dot(PCs[:, self.dims].T, (self.psths[j]['x_psth'][self.partitions[partition_idx]].T - means).T)


            # Plot colors and line-styles
            c_color     = plot_color(*zip(self.psths[j]['sort'], self.psths[j]['cond'])) if cond_color is None else cond_color
            c_ls        = plot_linestyle(*zip(self.psths[j]['sort'], self.psths[j]['cond'])) if cond_ls is None else cond_ls

            # Plot the PCs
            ax.plot3D(pc_scores[0,window], pc_scores[1,window], pc_scores[2, window], color=c_color, linestyle=c_ls, alpha=alpha)
            #ax.set_title('PCs {}, sortables={}, align={}'.format(self.dims + 1, self.sort, self.align))

            #ax.plot(pc_scores[0, window[-1]], pc_scores[1, window[-1]], marker='o', markerfacecolor='y')

            # save it down
            if savepath is not None:
                f.savefig(savepath + 'PCs={}_sortables={}_align={}.pdf'.format(self.dims + 1, self.sort, self.align))

        return f

    def plot3_pcs_axis(self, axis, start_loc, scale=1, f=None, is_x=True, partition_idx=0, arrow_color='black'):

        if f is None:
            print('Warning! We are making a new figure. Did you mean to not pass in a figure handle?')
            f = plt.figure()
        ax = f.gca(projection="3d")

       # grab out the partition PCs, etc.
        PCs = self.partition_pca[partition_idx]['PCs']
        means = self.partition_pca[partition_idx]['means']

        if is_x:
            PCs = self.partition_pca[partition_idx]['x_PCs']
            means = self.partition_pca[partition_idx]['x_means']

        if start_loc == 'baseline':
            start_loc = np.dot(PCs[:, self.dims].T, (self.psths[0]['x_psth'][self.partitions[partition_idx],0].T - means).T)
        elif start_loc == 'targets1':
            start_loc = np.dot(PCs[:, self.dims].T, (self.psths[0]['x_psth'][self.partitions[partition_idx],150].T - means).T)
        elif start_loc == 'targets2':
            start_loc = np.dot(PCs[:, self.dims].T, (self.psths[8]['x_psth'][self.partitions[partition_idx],150].T - means).T)
        elif start_loc == 'zeros':
            start_loc = np.zeros(2)
        else:
            start_loc = np.zeros(2)
            print('YOU DID NOT SPECIFY A VALID START LOC, SETTING TO ZERO')

        # you should not mean subtract, as the pc_input is a delta
        pc_axis = np.dot(PCs.T, axis)

        x = np.array((start_loc[0] -scale*pc_axis[0], start_loc[0], start_loc[0] + scale*pc_axis[0]))
        y = np.array((start_loc[1] -scale*pc_axis[1], start_loc[1], start_loc[1] + scale*pc_axis[1]))
        z = np.array((start_loc[2] -scale*pc_axis[2], start_loc[2], start_loc[2] + scale*pc_axis[2]))

        # Plot the PCs, from a starting location
        ax.plot3D(x, y, z, color=arrow_color)
        #ax.plot((start_loc[0], start_loc[0] + scale*pc_input[0]), (start_loc[1], start_loc[1] + scale*pc_input[1]), color='k')
        #ax.set_title('PCs {}, sortables={}, align={}'.format(self.dims + 1, self.sort, self.align))

        return f


    def plot_pcs_axis(self, axis, start_loc, scale=1, f=None, is_x=True, partition_idx=0, arrow_color='black'):

        if f is None:
            print('Warning! We are making a new figure. Did you mean to not pass in a figure handle?')
            f = plt.figure()

        ax = f.gca()

       # grab out the partition PCs, etc.
        PCs = self.partition_pca[partition_idx]['PCs']
        means = self.partition_pca[partition_idx]['means']

        if is_x:
            PCs = self.partition_pca[partition_idx]['x_PCs']
            means = self.partition_pca[partition_idx]['x_means']

        if start_loc == 'baseline':
            start_loc = np.dot(PCs[:, self.dims].T, (self.psths[0]['x_psth'][self.partitions[partition_idx],0].T - means).T)
        elif start_loc == 'targets1':
            start_loc = np.dot(PCs[:, self.dims].T, (self.psths[27]['x_psth'][self.partitions[partition_idx],150].T - means).T)
        elif start_loc == 'targets2':
            start_loc = np.dot(PCs[:, self.dims].T, (self.psths[12]['x_psth'][self.partitions[partition_idx],150].T - means).T)
        elif start_loc == 'zeros':
            start_loc = np.zeros(2)
        elif start_loc == 'Jloc':
            start_loc = np.array((-17.5, -5))
        elif start_loc == 'Jloc2':
            start_loc = np.array((-6, -4))


        # you should not mean subtract, as the pc_input is a delta
        pc_axis = np.dot(PCs.T, axis)

        # Plot the PCs, from a starting location
        ax.arrow(start_loc[0], start_loc[1], scale*pc_axis[0], scale*pc_axis[1], color=arrow_color, head_width=0.5, head_length=1)
        #ax.plot((start_loc[0], start_loc[0] + scale*pc_input[0]), (start_loc[1], start_loc[1] + scale*pc_input[1]), color='k')
        ax.set_title('PCs {}, sortables={}, align={}'.format(self.dims + 1, self.sort, self.align))

        return f


    def plot_pcs_area(self, f=None, savepath=None, window=None, filters=None, cond_ls=None, cond_color=None, partition_idx=0, is_x=False, alpha=1):

        # This plots psths, and if you specify 'filters', you can plot a subset of the psths.
        # filter is an array of tuples to sort by, e.g., [('dirs', -1), ('cohs', 'red')].
        # note for cohs, it will accept red and green and filter based off the cb coherence.

        if f is None:
            f = plt.figure()
        ax = f.gca()

        if window is None:
            window = np.arange(self.psths[0]['psth'].shape[1])

        all_conds   = np.array([zip(psth['sort'], psth['cond']) for psth in self.psths])
        idxs        = find_conds(all_conds, filters)

        # grab out the partition PCs, etc.
        PCs = self.partition_pca[partition_idx]['PCs']
        means = self.partition_pca[partition_idx]['means']

        if is_x:
            PCs = self.partition_pca[partition_idx]['x_PCs']
            means = self.partition_pca[partition_idx]['x_means']

        var_capt = 0

        # Plot each PSTH.
        for j in idxs:

            # Project the data
            pc_scores   = np.dot(PCs[:, self.dims].T, (self.psths[j]['psth'][self.partitions[partition_idx]].T - means).T)
            psth_var = np.var((self.psths[j]['psth'][self.partitions[partition_idx]].T - means).T, axis=1)

            if is_x:
                pc_scores   = np.dot(PCs[:, self.dims].T, (self.psths[j]['x_psth'][self.partitions[partition_idx]].T - means).T)
                psth_var = np.var((self.psths[j]['x_psth'][self.partitions[partition_idx]].T - means).T, axis=1)

            pc_var = np.var(pc_scores[self.dims,:], axis=1)

            # Plot colors and line-styles
            c_color     = plot_color(*zip(self.psths[j]['sort'], self.psths[j]['cond'])) if cond_color is None else cond_color
            c_ls        = plot_linestyle(*zip(self.psths[j]['sort'], self.psths[j]['cond'])) if cond_ls is None else cond_ls

            # Plot the PCs
            ax.plot(pc_scores[0,window], pc_scores[1,window], color=c_color, linestyle=c_ls, alpha=alpha)
            var_capt += np.sum(pc_var) / np.sum(psth_var)

        var_capt /= len(idxs)
        ax.set_title('PCs {}, sortables={}, align={}, var_capt={}'.format(self.dims + 1, self.sort, self.align, var_capt))

        #ax.plot(pc_scores[0, window[-1]], pc_scores[1, window[-1]], marker='o', markerfacecolor='y')

        # save it down
        if savepath is not None:
            f.savefig(savepath + 'PCs={}_sortables={}_align={}.pdf'.format(self.dims + 1, self.sort, self.align))

        return f

    def plot_input_pcs_area(self, inputs, f=None, savepath=None, window=None, filters=None, cond_ls=None, cond_color=None, partition_idx=0, is_x=True, start_loc='baseline', scale=1, arrow_color=np.array([0.5,0.5,0.5])):

        # This plots psths, and if you specify 'filters', you can plot a subset of the psths.
        # filter is an array of tuples to sort by, e.g., [('dirs', -1), ('cohs', 'red')].
        # note for cohs, it will accept red and green and filter based off the cb coherence.

        if f is None:
            f = plt.figure()
        ax = f.gca()

        if window is None:
            window = np.arange(self.psths[0]['psth'].shape[1])

        all_conds   = np.array([zip(psth['sort'], psth['cond']) for psth in self.psths])
        idxs        = find_conds(all_conds, filters)

        # grab out the partition PCs, etc.
        PCs = self.partition_pca[partition_idx]['PCs']
        means = self.partition_pca[partition_idx]['means']

        if is_x:
            PCs = self.partition_pca[partition_idx]['x_PCs']
            means = self.partition_pca[partition_idx]['x_means']

		# it doesn't matter if you do it on the partition or not.
        to_project = np.dot(self.rnn.Win[self.partitions[partition_idx],:], inputs)

		# you should not mean subtract, as the pc_input is a delta
        pc_input = np.dot(PCs.T, to_project)

        if start_loc == 'baseline':
            start_loc = np.dot(PCs[:, self.dims].T, (self.psths[0]['x_psth'][self.partitions[partition_idx],0].T - means).T)
        elif start_loc == 'targets1':
            start_loc = np.dot(PCs[:, self.dims].T, (self.psths[22]['x_psth'][self.partitions[partition_idx],150].T - means).T)
        elif start_loc == 'targets2':
            start_loc = np.dot(PCs[:, self.dims].T, (self.psths[12]['x_psth'][self.partitions[partition_idx],150].T - means).T)

		# Plot the PCs, from a starting location
        #ax.arrow(start_loc[0], start_loc[1], scale*pc_input[0], scale*pc_input[1], color=arrow_color, head_width=0.5, head_length=1)
        ax.plot((start_loc[0], start_loc[0] + scale*pc_input[0]), (start_loc[1], start_loc[1] + scale*pc_input[1]), color=arrow_color)
        ax.set_title('PCs {}, sortables={}, align={}'.format(self.dims + 1, self.sort, self.align))

		#ax.plot(pc_scores[0, window[-1]], pc_scores[1, window[-1]], marker='o', markerfacecolor='y')

		# save it down
        if savepath is not None:
            f.savefig(savepath + 'input_PCs={}_sortables={}_align={}.pdf'.format(self.dims + 1, self.sort, self.align))

        return f


    # ======================================
    # Plots the principal components vs time
    # ======================================

    def plot_pcs_vs_time(self, savepath=None):

        for i in self.dims:
            f   = plt.figure()
            ax  = f.gca()

            for j in np.arange(len(self.psths)):

                # Project the data
                pc_scores   = np.dot(self.PCs[:, i].T, (self.psths[j]['psth'].T - self.means).T)

                # Plot colors and line-styles
                cond_color  = plot_color(*zip(self.psths[j]['sort'], self.psths[j]['cond']))
                cond_ls     = plot_linestyle(*zip(self.psths[j]['sort'], self.psths[j]['cond']))

                # Plot the PCs across time
                ax.plot(np.arange(self.start_time, self.stop_time) * self.dt, pc_scores, color=cond_color, linestyle=cond_ls)
                ax.axvline(self.align_time * self.dt, color='b')
                ax.set_title('PCs {}, sortables={}, align={}'.format(i + 1, self.sort, self.align))

                # save it down
                if savepath is not None:
                    f.savefig(savepath + 'PCs={}_sortables={}_align={}.pdf'.format(i + 1, self.sort, self.align))

        return f

    # ================
    # Plot the factors
    # ================

    def plot_factors(self, f=None, savepath=None, window=None, filters=None, cond_ls=None, cond_color=None):

        if f is None:
            f = plt.figure()
        ax = f.gca()

        if window is None:
            window = np.arange(self.psths[0]['psth'].shape[1])

        all_conds   = np.array([zip(psth['sort'], psth['cond']) for psth in self.psths])
        idxs        = find_conds(all_conds, filters)

        # Plot each PSTH.
        for j in idxs:
            # Project the data
            fa_scores   = np.dot(self.FA_projector, (self.psths[j]['psth'].T - self.means).T)

            # Plot colors and line-styles
            c_color     = plot_color(*zip(self.psths[j]['sort'], self.psths[j]['cond'])) if cond_color is None else cond_color
            c_ls        = plot_linestyle(*zip(self.psths[j]['sort'], self.psths[j]['cond'])) if cond_ls is None else cond_ls

            # Plot the PCs
            ax.plot(fa_scores[0,window], fa_scores[1,window], color=c_color, linestyle=c_ls)
            ax.set_title('PCs {}, sortables={}, align={}'.format(self.dims + 1, self.sort, self.align))

            ax.plot(fa_scores[0, window[-1]], fa_scores[1, window[-1]], marker='o', markerfacecolor='y')

            # save it down
            if savepath is not None:
                f.savefig(savepath + 'FAs={}_sortables={}_align={}.pdf'.format(self.dims + 1, self.sort, self.align))

        return f

    # ======================================
    # Plots the factors vs time
    # ======================================

    def plot_factors_vs_time(self, savepath=None):

        for i in self.dims:
            f   = plt.figure()
            ax  = f.gca()

            for j in np.arange(len(self.psths)):

                # Project the data
                fa_scores   = np.dot(self.FA_projector[i,:], (self.psths[j]['psth'].T - self.means).T)

                # Plot colors and line-styles
                cond_color  = plot_color(*zip(self.psths[j]['sort'], self.psths[j]['cond']))
                cond_ls     = plot_linestyle(*zip(self.psths[j]['sort'], self.psths[j]['cond']))

                # Plot the PCs across time
                ax.plot(np.arange(self.start_time, self.stop_time) * self.dt, fa_scores, color=cond_color, linestyle=cond_ls)
                ax.axvline(self.align_time * self.dt, color='b')
                ax.set_title('PCs {}, sortables={}, align={}'.format(i + 1, self.sort, self.align))

                # save it down
                if savepath is not None:
                    f.savefig(savepath + 'FAs={}_sortables={}_align={}.pdf'.format(i + 1, self.sort, self.align))

        return f

    # ====================================================
    # Plot the eigenvalues, both individual and cumulative
    # ====================================================

    def plot_evals(self, savepath=None):

        # calculations
        var_per_dim = self.evals / sum(self.evals)
        accum_var   = np.cumsum(var_per_dim)

        dim         = np.where(accum_var > 0.9)[0][0]

        # First plot solo evals
        f   = plt.figure()
        ax  = f.gca()

        ax.plot(var_per_dim, linewidth=0, marker='.')
        ax.set_title('Variance per dimension; dimensionality={} captures {} variance'.format(dim + 1, accum_var[dim]))

        # save it down
        if savepath is not None:
            f.savefig(savepath + 'evals_solo.pdf')

        # Next plot cumulative evals
        f   = plt.figure()
        ax  = f.gca()

        ax.plot(accum_var, linewidth=0, marker='.')
        ax.set_title('Accumulated variance; dimensionality={} captures {} variance'.format(dim + 1, accum_var[dim]))

        # save it down
        if savepath is not None:
            f.savefig(savepath + 'evals_cum.pdf')

    # =================================
    # Helper functions for RNN dynamics
    # =================================

    def rnn_xdot(self, x, inputs=np.array((0,0,0,0))):
        return (-x + np.dot(self.rnn.Wrec, self.act(x)) + self.rnn.brec + np.dot(self.rnn.Win, inputs)) / self.rnn.p['tau']

    def rnn_xdot_area(self, x, inputs=np.array((0,0,0,0)), area=0):
        idx = self.partitions[area]
        return (-x + np.dot(self.rnn.Wrec[idx,idx], self.act(x)) + self.rnn.brec[idx] + np.dot(self.rnn.Win[idx,:], inputs)) / self.rnn.p['tau']

    def rnn_xdot_norm(self, x, inputs=np.array((0,0,0,0))):
        return 0.5 * np.dot(self.rnn_xdot(x, inputs), self.rnn_xdot(x, inputs))

    def rnn_rdot(self, x, inputs=np.array((0,0,0,0))):

        tau     = self.rnn.p['tau']
        dt      = self.rnn.p['dt']
        alpha   = dt / tau

        # Calculate \dot{r}
        xn = x + dt * self.rnn_xdot(x, inputs)

        rn  = self.act(xn)
        r   = self.act(x)

        return (rn - r) / dt

    def rnn_rdot_norm(self, x, inputs=np.array((0,0,0,0))):
        return 0.5 * np.dot(self.rnn_rdot(x, inputs), self.rnn_rdot(x, inputs))

    def t_rnn_xdot(self, x, inputs=np.array((0,0,0,0))):
        return (-x + T.dot(self.rnn.Wrec, self.act(x)) + self.rnn.brec + T.dot(self.rnn.Win, inputs)) / self.rnn.p['tau']

    def t_rnn_xdot_norm(self, x, inputs=np.array((0,0,0,0))):
        return 0.5 * T.dot(self.t_rnn_xdot(x, inputs), self.t_rnn_xdot(x, inputs))

    def t_rnn_rdot(self, x, inputs=np.array((0,0,0,0))):

        tau     = self.rnn.p['tau']
        dt      = self.rnn.p['dt']
        alpha   = dt / tau

        # No noise here.
        xn = x + alpha*(-x + T.dot(self.rnn.Wrec, self.act(x)) + rnn.brec + T.dot(self.rnn.Win, inputs))

        xn2 = x + dt * self.t_rnn_xdot(x, inputs)

        # calculate rdot
        rn  = self.act(xn)
        r   = self.act(x)

        return (rn - r) / dt

    def t_rnn_rdot_norm(self, x, inputs=np.array((0,0,0,0))):
        return 0.5 * T.dot(self.t_rnn_rdot(x, inputs), self.t_rnn_rdot(x, inputs))

    def get_fixed_point(self, x, inputs, type='x'):

        if type == 'r':
            res = minimize(self.compute_cost_r, x, args=(inputs),
                    method='Newton-CG',
                    jac=self.compute_jac_r,
                    hess=self.compute_hess_r,
                    options={'disp': False}
                    )
        elif type == 'x':
            res = minimize(self.compute_cost_x, x, args=(inputs),
                    method='Newton-CG',
                    jac=self.compute_jac_x,
                    hess=self.compute_hess_x,
                    bounds=(self.fp_threshold, None),
                    options={'disp': False}
                    )
        else:
            assert False, 'Not a valid type to look for fixed points.'

        # res.x is the fixed point, res.fun is the function value, res.success is whether it was successful or not
        return res.x, res.fun, res.success

    # Compile theano functions
    def compile_theano_costs(self):
        tx  = T.dvector('x')
        tin = T.dvector('in')
        tfr = tf_rnn_rdot_norm(self.rnn, tx, tin)
        tfx = tf_rnn_xdot_norm(self.rnn, tx, tin)

        jac_fr = T.grad(tfr, tx)
        jac_fx = T.grad(tfx, tx)

        Hr, updates_r   = theano.scan(lambda i, jac_fr, tx: T.grad(jac_fr[i], tx), sequences=T.arange(jac_fr.shape[0]), non_sequences=[jac_fr, tx])
        f               = theano.function([tx, tin], Hr, updates=updates_r)

        self.compute_cost_r = theano.function([tx, tin], tfr)
        self.compute_jac_r  = theano.function([tx, tin], jac_fr)
        self.compute_hess_r = theano.function([tx, tin], Hr, updates=updates_r)

        Hx, updates_x   = theano.scan(lambda i, jac_fx, tx: T.grad(jac_fx[i], tx), sequences=T.arange(jac_fx.shape[0]), non_sequences=[jac_fx, tx])
        f               = theano.function([tx, tin], Hx, updates=updates_x)

        self.compute_cost_x = theano.function([tx, tin], tfx)
        self.compute_jac_x  = theano.function([tx, tin], jac_fx)
        self.compute_hess_x = theano.function([tx, tin], Hx, updates=updates_x)

    # Returns (fixed_points, values, stable)
    def sample_fixed_points(self, inputs=np.array((0,0,0,0)), sample=1000, type='x'):

        # Get the dts so we know how to sample adequately
        dt      = self.rnn.p['dt']
        sample  = sample // dt

        # Length of each psth.
        p_lens  = [p['x_psth'].shape[1] for p in self.psths]

        # Fixed points
        r_fixs  = []    # location of the fixed point
        vals    = []    # value at that location
        stable  = []    # is the fixed point stable?

        print('in routine to calc fixed points with {} samples'.format(sample))

        for i in np.arange(len(p_lens)):

            for j in np.arange(p_lens[i] // sample):

                print('i={} of {}'.format(i, len(p_lens)))
                print('j={} of {}'.format(j, p_lens[i]//sample))

                if type == 'r':
                    x0 = self.psths[i]['psth'][:, j * sample]
                elif type == 'x':
                    x0 = self.psths[i]['x_psth'][:, j * sample]
                else:
                    assert False, 'Not a valid type to find fixed point by.'

                # Find a fixed points
                r_fix, val, success = self.get_fixed_point(x0, inputs, type=type)

                if success:
                    r_fixs.append(r_fix)
                    vals.append(val)

                    # Find whether the fixed point is stable or not.
                    if type == 'r':
                        hessf = self.compute_hess_r(r_fix, inputs)
                    elif type == 'x':
                        hessf = self.compute_hess_x(r_fix, inputs)
                    else:
                        assert False, 'Not a valid type to find fixed point by.'

                    evals = np.linalg.eigvals(hessf)
                    stable.append(np.all(evals >= 0))

        return np.array(r_fixs), np.array(vals), np.array(stable)

    # Plot fixed points (accepts a figure handle)
    def plot_fixed_points(self, inputs=np.array((0,0,0,0)), sample=1000, f=None, savepath=None, type='x', color=None, projector='PCs'):

        if f is None:
            f = self.plot_pcs()

        ax  = f.gca()

        r_fixs, vals, stable    = self.sample_fixed_points(inputs=inputs, sample=sample, type=type)
        idx_thresholds          = np.where(vals < np.max((self.fp_threshold, self.fpx_threshold)))[0]

        if len(idx_thresholds) == 0:
            print 'Warning: no fixed points were acceptable under your threshold; the minimum value of the objective that was successful was: {}.  We will plot this minimum.'.format(np.min(vals))
            idx_thresholds      = np.where(vals == np.min(vals))[0]

        for i in idx_thresholds:
            if color is None:
                fp_col  = 'r' if stable[i] else 'b'
            else:
                fp_col  = color

            fp_marker = '.'
            fp_markersize = 20
            if self.fpx_threshold > self.fp_threshold:
                if vals[i] < self.fpx_threshold and vals[i] > self.fp_threshold:
                    fp_marker = 'x'
                    fp_markersize = 10
            # get the projector
            P = getattr(self, projector)
            P = P[:, self.dims]
            means = self.means if projector is 'PCs' or 'rPCs' else means

            if type == 'r':
                fp_proj = np.dot(P.T, (r_fixs[i] - means).T)
            elif type == 'x':
                fp_proj = np.dot(P.T, (self.act(r_fixs[i]) - self.means).T)

            ax.plot(fp_proj[0], fp_proj[1], linewidth=0, marker=fp_marker, color=fp_col, markersize=fp_markersize, alpha=0.5)

        # save it down
        if savepath is not None:
            f.savefig(savepath + 'fixed-points_inputs={}.pdf'.format(inputs))

        return f

    # partitions the space to sample dynamics
    def partition_space(self, x_sample=10, y_sample=10, xp_min=0, xp_max=0, yp_min=0, yp_max=0):

        # we should be sampling x, not r, because later on, we are sampling these points from the x-space.
        data        = np.hstack([one_psth['x_psth'] for one_psth in self.psths])
        scores      = np.dot(self.x_PCs[:, self.dims].T, (data.T - self.x_means).T)

        x_bounds    = (np.floor(min(scores[0,:])) - xp_min, np.ceil(max(scores[0,:])) + xp_max)
        y_bounds    = (np.floor(min(scores[1,:])) - yp_min, np.ceil(max(scores[1,:])) + yp_max)

        x_pts       = np.linspace(x_bounds[0], x_bounds[1], x_sample)
        y_pts       = np.linspace(y_bounds[0], y_bounds[1], y_sample)

        self.partition_points   = (x_pts, y_pts)

        return (x_pts, y_pts)

    # partition for deltas to sample dynamics -- this assigns self.delta_samples
    def partition_deltas(self, x_sample=15, y_sample=15, xmin=-5, xmax=5, ymin=-5, ymax=5):

        x_pts   = np.linspace(xmin, xmax, x_sample)
        y_pts   = np.linspace(ymin, ymax, y_sample)

        self.delta_samples  = (x_pts, y_pts)
        return (x_pts, y_pts)

    # samples dynamics at the given x0, legacy version

    def sample_local_dynamics_area_sub(self, x0=None, inputs=np.array((0,0,0,0)), projector='PCs', partition_idx=0):


        # This should be on the entire dimensional data, since the RNN should update according to its equations
        #r   = self.act(x0)
        #rdot = self.rnn_rdot(x0, inputs)

        r = x0
        rdot = self.rnn_xdot_area(x0, inputs, area=partition_idx)

        # get the projector
        P = self.partition_pca[partition_idx]['x_PCs']#getattr(self.partition_pca[partition_idx], projector)
        means = self.partition_pca[partition_idx]['x_means'] if projector is 'PCs' or 'rPCs' else np.zeros_like(self.means)

        # Now project this back into the lower dimensional space
        # NOTE WE DON'T SUBTRACT THE MEANS AS THEY CANCEL OUT FOR X_{k+1} - X_k
        s       = np.dot(P.T, r - means)
        sdot    = np.dot(P.T, rdot)

        return s, sdot



    # samples dynamics at the given x0
    def sample_local_dynamics(self, x0=None, inputs=np.array((0,0,0,0)), projector='PCs'):

        # Let's first get what r is, the projected value of r will be returned
        r   = self.act(x0)

        # Now calculate derivative at this point.
        rdot = self.rnn_rdot(x0, inputs)

        # get the projector
        P = getattr(self, projector)
        means = self.means if projector is 'PCs' or 'rPCs' else np.zeros_like(self.means)

        # Now project this back into the lower dimensional space
        # NOTE WE DON'T SUBTRACT THE MEANS AS THEY CANCEL OUT FOR X_{k+1} - X_k
        s       = np.dot(P.T, r - means)
        sdot    = np.dot(P.T, rdot)

        return s, sdot

    def sample_local_dynamics_area(self, x0=None, inputs=np.array((0,0,0,0)), projector='PCs', partition_idx=0):


        # This should be on the entire dimensional data, since the RNN should update according to its equations
        #r   = self.act(x0)
        #rdot = self.rnn_rdot(x0, inputs)

        r = x0
        rdot = self.rnn_xdot(x0, inputs)

        # get the projector
        P = self.partition_pca[partition_idx]['x_PCs']#getattr(self.partition_pca[partition_idx], projector)
        means = self.partition_pca[partition_idx]['x_means'] if projector is 'PCs' or 'rPCs' else np.zeros_like(self.means)

        # Now project this back into the lower dimensional space
        # NOTE WE DON'T SUBTRACT THE MEANS AS THEY CANCEL OUT FOR X_{k+1} - X_k
        s       = np.dot(P.T, r[self.partitions[partition_idx]] - means)
        sdot    = np.dot(P.T, rdot[self.partitions[partition_idx]])

        return s, sdot

    def plot_local_dynamics_delta_area_x_temporal(self, inputs=np.array((0,0,0,0)), times=[150], conds=[9], f=None, scale=1, force_alignment=None, t_sample=None, coh=None, reset_align=True, filters=None, partition_idx=0, xlim=None, ylim=None):

        # coh is for sampling get_higher_dims, it is the red coherence (NOT CB).
        x_deltas, y_deltas  = self.delta_samples

        # store values in the following array.
        ss      = np.zeros((2, len(x_deltas), len(y_deltas)))
        sdots   = np.zeros((2, len(x_deltas), len(y_deltas)))

   		# manually set sampling value to be x at the PC location
        PCs = self.partition_pca[partition_idx]['x_PCs']
        means = self.partition_pca[partition_idx]['x_means']

		# Now plot them.
        if f is None:
            f = plt.figure()
        ax  = f.gca()


        for t in times:
            for x in conds:

                if np.all(inputs == np.array((0,0,0,0))): # then we're at start_loc == baseline:
                    x0 = self.psths[x]['x_psth'][:,t]
                    #print('start loc is baseline')
                elif np.all(inputs[:2] == np.array((1,-1))): # then we're at start_loc = targets1:
                    x0 = self.psths[x]['x_psth'][:,t]
                    #print('start loc is targets1')
                elif np.all(inputs[:2] == np.array((-1,1))): # then we're at start_loc = targets2
                    x0 = self.psths[x]['x_psth'][:,t]
                    #print('start loc is targets2')
                else:
                    print('NO START LOC. x0 not assigned, will error out.')

                #if t == 150:
                #    pdb.set_trace()
                # Let's try to move around in this space now.
                for i in np.arange(len(x_deltas)):
                    for j in np.arange(len(y_deltas)):

                        # Get the high-dimensional point from which we're sampling dynamics
                        x0_s                = x0 + np.dot(self.x_PCs[:,self.dims],  np.hstack((x_deltas[i], y_deltas[j]))) # no need to add self.x_means since this is a delta
                        this_s, this_sdot   = self.sample_local_dynamics_area(x0=x0_s, inputs=inputs, partition_idx=partition_idx)
                        sdots[:, i, j]      = this_sdot[self.dims]
                        ss[:, i, j]         = this_s[self.dims]
                        #pdb.set_trace()

                if xlim is not None:
                    ax.set_xlim(xlim)
                if ylim is not None:
                    ax.set_ylim(ylim)

                for i in np.arange(len(x_deltas)):
                    for j in np.arange(len(y_deltas)):
                        arrow_color = np.array([0,0,0])
                        #ax.arrow(ss[0, i, j], ss[1, i, j], scale*sdots[0,i,j], scale*sdots[1,i,j], color=arrow_color, head_width=0.5, head_length=1)
                        ax.plot([ss[0, i, j], ss[0,i,j] + scale*sdots[0,i,j]], [ss[1,i,j], ss[1,i,j] + scale*sdots[1,i,j]], color=arrow_color, linewidth=2)
                        #ax.plot(ss[0,i,j] + scale*sdots[0,i,j], ss[1,i,j] + scale*sdots[1,i,j], linewidth=0, marker='o', markersize=10)

        return f


    def plot_local_dynamics_delta_area_x(self, inputs=np.array((0,0,0,0)), f=None, scale=1, force_alignment=None, t_sample=None, coh=None, reset_align=True, filters=None, partition_idx=0, xlim=None, ylim=None):

        # coh is for sampling get_higher_dims, it is the red coherence (NOT CB).
        x_deltas, y_deltas  = self.delta_samples

        # store values in the following array.
        ss      = np.zeros((2, len(x_deltas), len(y_deltas)))
        sdots   = np.zeros((2, len(x_deltas), len(y_deltas)))

   		# manually set sampling value to be x at the PC location
        PCs = self.partition_pca[partition_idx]['x_PCs']
        means = self.partition_pca[partition_idx]['x_means']

        if np.all(inputs == np.array((0,0,0,0))): # then we're at start_loc == baseline:
            x0 = self.psths[0]['x_psth'][:,80]
            print('start loc is baseline')
        elif np.all(inputs[:2] == np.array((1,-1))): # then we're at start_loc = targets1:
            x0 = self.psths[25]['x_psth'][:,150]
            print('start loc is targets1')
        elif np.all(inputs[:2] == np.array((-1,1))): # then we're at start_loc = targets2
            x0 = self.psths[9]['x_psth'][:,200]
            print('start loc is targets2')
        else:
            print('NO START LOC. x0 not assigned, will error out.')

        # Let's try to move around in this space now.
        for i in np.arange(len(x_deltas)):
            for j in np.arange(len(y_deltas)):

                # Get the high-dimensional point from which we're sampling dynamics
                x0_s                = x0 + np.dot(self.x_PCs[:,self.dims],  np.hstack((x_deltas[i], y_deltas[j]))) # no need to add self.x_means since this is a delta
                this_s, this_sdot   = self.sample_local_dynamics_area(x0=x0_s, inputs=inputs, partition_idx=partition_idx)
                sdots[:, i, j]      = this_sdot[self.dims]
                ss[:, i, j]         = this_s[self.dims]

                #pdb.set_trace()
		# Now plot them.
        if f is None:
            f = plt.figure()
        ax  = f.gca()


        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        for i in np.arange(len(x_deltas)):
            for j in np.arange(len(y_deltas)):
                #ax.arrow(ss[0, i, j], ss[1, i, j], scale*sdots[0,i,j], scale*sdots[1,i,j], color=np.array([0.5, 0.5, 0.5]), head_width=0.5, head_length=1)
                ax.plot([ss[0, i, j], ss[0,i,j] + scale*sdots[0,i,j]], [ss[1,i,j], ss[1,i,j] + scale*sdots[1,i,j]], color='k', linewidth=2)
                #ax.plot(ss[0,i,j] + scale*sdots[0,i,j], ss[1,i,j] + scale*sdots[1,i,j], linewidth=0, marker='o', markersize=10)

        return f

    # Gets the higher dimensions to plot; or returns the value of the PSTH at the desired time point
    def get_higher_dims(self, inputs=np.array((0,0,0,0)), num_higher_dims=5, return_psth=False, field='r', force_alignment=None, t_sample=0, reset_align=True, filters=None):
        # Returns the higher dims vector given an input.
        # The organization is as follows:
        #   - Unless force alignment is given ('start', 'cb', or 'end'), we're going to align based off of the inputs.
        #     - If inputs is all zeros, align to trial start and grab the first 100ms.
        #     - If the targets are on but the checkerboard is off, align to cb_on and take the 100ms preceding cb_on
        #     - If all inputs are on, align to trial end and grab the last 100ms.
        #
        #   - If return_psth is True, then it returns the PSTH at this time instead of the higher_dims
        #   - If t_sample is not None, it's a bin (in dt, not in ms) that is sampled in the absolute psth.  It is ONLY used when you want to return_psth.  Otherwise, for get_higher_dims returning the PCs this isn't even used.
        #       NOTE: if the alignment is end, then t_post can't be greater than 0 or else it'll sample out of bounds.  We have thus made t_post subtract from the last element, so note its special usage in the case of align='end'
        # - reset_align set to false means the alignment will not be reset at the end.  this helps reduce the number of psths calculated during the movie, which can otherwise be beastly (two psth calcs for each frame... what a nightmare).

        prior_align = self.align
        prior_sort  = self.sort

        start_dim   = np.max(self.dims) + 1
        higher_dims = np.zeros((num_higher_dims))

        # Okay, so if force_alignment is none, then we're going to base it on the inputs
        align   = force_alignment

        if align is None:
            if np.sum(np.abs(inputs)) == 0:
                align   = 'start'
            elif (inputs[0] != 0 or inputs[1] != 0) and inputs[2] == 0 and inputs[3] == 0:
                align   = 'cb'
            else:
                align   = 'end'

        # This here is an align to 0.
        if align == 'start':

            # then the input is all zeros.
            if prior_align != align:
                self.set_align(align=align)

            # if reset_align is true, we don't want to have to go through the trouble of regenerating psths; please note that self.psths is assigned by self.gen_psth(), NOT self.calc_psth()
            if reset_align:
                if filters is None:
                    this_psth   = self.calc_psth(field=field)
                else:
                    # If filters is not None, then we need to filter by a condition and thus need to re-gen the PSTHs
                    if prior_align != align:
                        self.gen_psth()
                    psths       = self.extract_psths(filters=filters, field=field)
                    this_psth   = np.mean(psths, axis=0)
            else:
                if prior_align != align:
                    self.gen_psth(field=field)

                psths       = self.extract_psths(filters=filters, field=field)
                ### JCK 2017-10-25 -- HOW WAS THIS NOT AN ERROR BEFORE?!
                #this_psth  = np.mean(self.psths, axis=0)       # average across all conditions
                this_psth = np.mean(psths, axis=0)

            these_pcs   = np.dot(self.x_PCs.T, (this_psth.T - self.x_means).T)
            higher_dims = np.mean(these_pcs[start_dim:start_dim+num_higher_dims, :100//self.dt], axis=1)

            if t_sample is None:
                t_sample = 0

            this_psth   = this_psth[:, t_sample]

            if reset_align:
                # then we didn't gen_psths and we can safely reset the align.  This saves us the headache of having to reset the alignment and calculate another psth
                self.set_align(align=prior_align)

        elif align == 'cb':
            # then the targets are on but the checkerboard is not on.

            # set the alignments if necessary.  here we have to gen_psth() as we need the sort correct.
            if prior_align != align or prior_sort != ['conds']:
                self.sort  = ['conds']
                self.set_align(align=align)
                self.gen_psth()

            if filters is None:
                # Collapse across target input configurations.
                # Now assign target classifications; target=-1 means left is red (1,-1) and target=1 means right is red (-1,1)

                if inputs[0] == -1: # left target is red
                    p0      = self.extract_psths(filters=[('dirs', -1), ('cohs', 'r')], field='r')
                    p1      = self.extract_psths(filters=[('dirs', +1), ('cohs', 'g')], field='r')
                else: # left target is green
                    p0      = self.extract_psths(filters=[('dirs', -1), ('cohs', 'g')], field='r')
                    p1      = self.extract_psths(filters=[('dirs', +1), ('cohs', 'r')], field='r')

                psths   = np.vstack((p0, p1))

            else:
                psths = self.extract_psths(filters=filters, field='r')

            this_psth   = np.mean(psths, axis=0)        # average across all conditions
            these_pcs   = np.dot(self.x_PCs.T, (this_psth.T - self.x_means).T)
            higher_dims = np.mean(these_pcs[start_dim:start_dim+num_higher_dims, (self.align_time - self.start_time) - 100 // self.dt:(self.align_time - self.start_time)], axis=1)

            if t_sample is None:
                t_sample = self.align_time - self.start_time

            this_psth   = this_psth[:, t_sample]

        elif align == 'end':

            # We'll want to find the input condition in the sorted_trials
            #   The inputs tell us which direction it was and coherence.
            if filters is None:
                this_dir    = +1 if (inputs[0] == 1 and inputs[2] > inputs[3]) or (inputs[0] == -1 and inputs[3] > inputs[2]) else -1
                this_cb     = inv_coh_r(inputs[2])
                filters = [('dirs', this_dir), ('cohs', this_cb)]

            # You could do a calc psth on just the subset, but this leads to some obfuscation in other tasks.  So we're going to work with gen_psth to be safe.  Sorry.  Refer to the initial git commit if you want to change this.  The code is there.
            if prior_align != align or prior_sort != ['dirs', 'cohs']:
                self.sort  = ['dirs', 'cohs']
                self.set_align(align=align)
                self.gen_psth()

            psths       = self.extract_psths(filters=filters, field='r')
            this_psth   = np.mean(psths, axis=0)

            these_pcs   = np.dot(self.PCs.T, (this_psth.T - self.means).T)
            higher_dims = np.mean(these_pcs[start_dim:start_dim+num_higher_dims, -100 // self.dt:], axis=1)

            if t_sample is None:
                t_sample = -1

            this_psth   = this_psth[:, t_sample]

        # Reset if necessary
        if (self.sort != prior_sort or self.align != prior_align):
            if reset_align:
                print 'Resetting the PSTH as get_higher_dims modified some of the values temporarily.'
                self.sort = prior_sort
                self.set_align(align=prior_align)
                self.gen_psth()


        if return_psth:
            return this_psth
        else:
            return higher_dims

    # plot dynamics movie frames
    def plot_dynamics_movie_frames(self, framerate=100, savepath='/tmp/', xlim=[-10,10], ylim=[-10,10], filters=None, scale=1, Ts=None, cond_color=None, cond_ls=None, history=None):

        # - generates the frames for making a movie.  It stores them temporarily in savepath.
        # - 1/framerate (in ms) is how much spacing there is between each frame.
        # - we will be plotting the activity of the PSTH, for a given condition.
        # - currently there is no ability to subsample the video -- it's much easier to make the whole video and then in post processing to take the frames that you want.
        # - Ts = [lower, upper] are the time bounds of the movie, expressed in ms.
        # - all right then... here we go.

        # Align to cb if the dataset wasn't.
        prior_align = None
        prior_sort  = None

        if self.align != 'cb':

            print 'Re-aligning the data to checkerboard'
            prior_align = self.align
            prior_sort  = self.sort

            self.sort   = ['dirs', 'cohs']
            self.set_align(align='cb')

        psths = self.psths

        # converts frame no. to absolute time and vice versa
        frame_to_time   = (1000 / framerate) / self.dt
        time_to_frame   = self.dt / (1000 / framerate)

        # Find appropriate temporal landmarks
        if Ts is None:
            T               = (self.stop_time - self.start_time) * self.dt
            T_start, T_end  = (0, T // self.dt)
        else:
            T_start, T_end  = (Ts[0] // self.dt, Ts[1] // self.dt)

        f_start, f_end  = (T_start * time_to_frame, T_end * time_to_frame)
        frames          = np.arange(int(f_start), int(f_end))

        # subidxs to filter -- yes, I know plot_pcs does this automatically but we still need it for the inputs.
        all_conds   = np.array([zip(psth['sort'], psth['cond']) for psth in psths])
        idxs        = find_conds(all_conds, filters)

        if len(idxs) > 1:
            # there are multiple indices.  we want the one where the input has the greatest coh.  we can do the following because we know how the psth is sorted, i.e., by self.sort = ['dirs', 'cohs'] and by self.align = cb
            cohs        = [coh_r(psths[idx]['cond'][1]) for idx in idxs]
            abs_cohs    = np.abs(cohs)

            which_idx   = np.where(abs_cohs - max(abs_cohs) == 0)[0][0]
            idx_input   = idxs[which_idx]
        else:
            idx_input   = idxs[0]

        # Now iterate over the frames.

        for i in frames:

            f   = plt.figure()

            # In the current implementation, u_psth is an average, and so the inputs turning on are not explicitly what was seen.  My proposed way to currently account for this is to round the input along the target dimension, or else input psths with exact timing (i.e., targets always come on at the same time).

            stop_frame  = (np.ceil(i * self.dt) // self.dt).astype(int)
            # Plot the PCs
            history     = 0 if history is None else i - history
            t_start     = int(max(0, history) * frame_to_time)
            t_stop      = int(i * frame_to_time)
            f           = self.plot_pcs(f=f, window=np.arange(t_start, t_stop+1), filters=filters, cond_color=cond_color, cond_ls=cond_ls)

            # Determine the input
            inputs      = psths[idx_input]['u_psth'][:, t_stop]
            # moves the rounding threshold up to 0.7
            inputs[0]   = np.round(inputs[0] * 0.7 / 0.5)
            inputs[1]   = np.round(inputs[1] * 0.7 / 0.5)

            # Now, generate the plot_local_dynamics
            f           = self.plot_local_dynamics_delta(f=f, inputs=inputs, scale=scale, reset_align=False, t_sample=t_stop, force_alignment='cb', filters=filters) # setting reset_align=False means to not regenerate the PSTHs every time in get_higher_dims

            # set the axes
            ax          = f.gca()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            # and save
            i_str   = str(i)
            f.savefig(savepath + 'frame_' + i_str.zfill(5) + '.pdf')

            # Kill this figure now after having saved it.
            f.clf()
            plt.close(f)

        # Re-align data if necessary
        if prior_align is not None:
            self.sort   = ['dirs', 'cohs']
            self.set_align(align=prior_align)

    # plot dynamics movie frames
    def plot_dynamics_movie_frames_cora(self, framerate=100, savepath='/tmp/', xlim=[-2,2], ylim=[-2,2], filters=None, scale=1, Ts=None, cond_color=None, cond_ls=None, history=None):

        # - generates the frames for making a movie.  It stores them temporarily in savepath.
        # - 1/framerate (in ms) is how much spacing there is between each frame.
        # - we will be plotting the activity of the PSTH, for a given condition.
        # - currently there is no ability to subsample the video -- it's much easier to make the whole video and then in post processing to take the frames that you want.
        # - Ts = [lower, upper] are the time bounds of the movie, expressed in ms.
        # - all right then... here we go.

        # Align to cb if the dataset wasn't.
        prior_align = None
        prior_sort  = None

        if self.align != 'start':

            print 'Re-aligning the data to start'
            prior_align = self.align
            prior_sort  = self.sort

            self.sort   = ['conds']
            self.set_align(align='start')
            self.gen_psth()

        psths = self.psths

        # converts frame no. to absolute time and vice versa
        frame_to_time   = (1000 / framerate) / self.dt
        time_to_frame   = self.dt / (1000 / framerate)

        # Find appropriate temporal landmarks
        if Ts is None:
            T               = (self.stop_time - self.start_time) * self.dt
            T_start, T_end  = (0, T // self.dt)
        else:
            T_start, T_end  = (Ts[0] // self.dt, Ts[1] // self.dt)

        f_start, f_end  = (T_start * time_to_frame, T_end * time_to_frame)
        frames          = np.arange(int(f_start), int(f_end))

        # subidxs to filter -- yes, I know plot_pcs does this automatically but we still need it for the inputs.
        all_conds   = np.array([zip(psth['sort'], psth['cond']) for psth in psths])
        idxs        = find_conds(all_conds, filters)

        if len(idxs) > 1:
            # there are multiple indices.  we want the one where the input has the greatest coh.  we can do the following because we know how the psth is sorted, i.e., by self.sort = ['dirs', 'cohs'] and by self.align = cb
            cohs        = [coh_r(psths[idx]['cond'][1]) for idx in idxs]
            abs_cohs    = np.abs(cohs)

            which_idx   = np.where(abs_cohs - max(abs_cohs) == 0)[0][0]
            idx_input   = idxs[which_idx]
        else:
            idx_input   = idxs[0]

        # Now iterate over the frames.

        for i in frames:

            f   = plt.figure()

            # In the current implementation, u_psth is an average, and so the inputs turning on are not explicitly what was seen.  My proposed way to currently account for this is to round the input along the target dimension, or else input psths with exact timing (i.e., targets always come on at the same time).

            stop_frame  = (np.ceil(i * self.dt) // self.dt).astype(int)
            # Plot the PCs
            history     = 0 if history is None else i - history
            t_start     = int(max(0, history) * frame_to_time)
            t_stop      = int(i * frame_to_time)
            f           = self.plot_pcs(f=f, window=np.arange(t_start, t_stop+1), filters=filters, cond_color=cond_color, cond_ls=cond_ls)

            # Determine the input
            inputs      = psths[idx_input]['u_psth'][:, t_stop]
            # moves the rounding threshold up to 0.7
            inputs[0]   = np.round(inputs[0] * 0.7 / 0.5)
            inputs[1]   = np.round(inputs[1] * 0.7 / 0.5)

            # Now, generate the plot_local_dynamics
            f           = self.plot_local_dynamics_delta(f=f, inputs=inputs, scale=scale, reset_align=False, t_sample=t_stop, force_alignment='cb', filters=filters) # setting reset_align=False means to not regenerate the PSTHs every time in get_higher_dims

            # set the axes
            ax          = f.gca()
            #ax.set_xlim(xlim)
            #ax.set_ylim(ylim)

            # and save
            i_str   = str(i)
            f.savefig(savepath + 'frame_' + i_str.zfill(5) + '.pdf')

            # Kill this figure now after having saved it.
            f.clf()
            plt.close(f)

        # Re-align data if necessary
        if prior_align is not None:
            self.sort   = ['conds']
            self.set_align(align=prior_align)
            self.gen_psth()

    def plot_dynamics_movie_frames_cora_pub(self, framerate=100, savepath='/tmp/', xlim=[-2,2], ylim=[-2,2], filters=None, scale=1, Ts=None, cond_color=None, cond_ls=None, history=None, projector='PCs'):

        # - generates the frames for making a movie.  It stores them temporarily in savepath.
        # - 1/framerate (in ms) is how much spacing there is between each frame.
        # - we will be plotting the activity of the PSTH, for a given condition.
        # - currently there is no ability to subsample the video -- it's much easier to make the whole video and then in post processing to take the frames that you want.
        # - Ts = [lower, upper] are the time bounds of the movie, expressed in ms.
        # - all right then... here we go.

        # Align to cb if the dataset wasn't.
        prior_align = None
        prior_sort  = None

        if self.align != 'start':

            print 'Re-aligning the data to start'
            prior_align = self.align
            prior_sort  = self.sort

            self.sort   = ['conds']
            self.set_align(align='start')
            self.gen_psth()

        psths = self.psths

        # converts frame no. to absolute time and vice versa
        frame_to_time   = (1000 / framerate) / self.dt
        time_to_frame   = self.dt / (1000 / framerate)

        # Find appropriate temporal landmarks
        if Ts is None:
            T               = (self.stop_time - self.start_time) * self.dt
            T_start, T_end  = (0, T // self.dt)
        else:
            T_start, T_end  = (Ts[0] // self.dt, Ts[1] // self.dt)

        f_start, f_end  = (T_start * time_to_frame, T_end * time_to_frame)
        frames          = np.arange(int(f_start), int(f_end))

        # subidxs to filter -- yes, I know plot_pcs does this automatically but we still need it for the inputs.
        all_conds   = np.array([zip(psth['sort'], psth['cond']) for psth in psths])
        idxs        = find_conds(all_conds, filters)

        if len(idxs) > 1:
            # there are multiple indices.  we want the one where the input has the greatest coh.  we can do the following because we know how the psth is sorted, i.e., by self.sort = ['dirs', 'cohs'] and by self.align = cb
            cohs        = [coh_r(psths[idx]['cond'][1]) for idx in idxs]
            abs_cohs    = np.abs(cohs)

            which_idx   = np.where(abs_cohs - max(abs_cohs) == 0)[0][0]
            idx_input   = idxs[which_idx]
        else:
            idx_input   = idxs[0]

        # Now iterate over the frames.

        prior_input = np.array((-100,-100,-100))

        for i in frames:

            f   = plt.figure()

            # In the current implementation, u_psth is an average, and so the inputs turning on are not explicitly what was seen.  My proposed way to currently account for this is to round the input along the target dimension, or else input psths with exact timing (i.e., targets always come on at the same time).

            stop_frame  = (np.ceil(i * self.dt) // self.dt).astype(int)
            # Plot the PCs
            history     = 0 if history is None else i - history
            t_start     = int(max(0, history) * frame_to_time)
            t_stop      = int(i * frame_to_time)
            #f          = self.plot_pcs(f=f, window=np.arange(t_start, t_stop+1), filters=filters, cond_color=cond_color, cond_ls=cond_ls)
            f           = self.plot_neural_traj2d_pub(f=f, window=np.arange(t_start, t_stop+1), filters=filters, cond_color=cond_color, cond_ls=cond_ls, cond_lw=3, projector=projector)

            # Determine the input
            inputs      = psths[idx_input]['u_psth'][:, t_stop]
            # moves the rounding threshold up to 0.7
            #inputs[0]  = np.round(inputs[0] * 0.7 / 0.5)
            #inputs[1]  = np.round(inputs[1] * 0.7 / 0.5)

#           if np.sum(np.abs(prior_input - inputs)):
#               pdb.set_trace()
#               prior_input = inputs

            # Now, generate the plot_local_dynamics
            f           = self.plot_local_dynamics_delta_pub(f=f, inputs=inputs, scale=scale, reset_align=False, t_sample=t_stop, force_alignment='start', filters=filters, projector=projector) # setting reset_align=False means to not regenerate the PSTHs every time in get_higher_dims

            # And FPs
            f           = self.plot_fixed_points(f=f, inputs=inputs, type='x', projector=projector)

            # set the axes
            ax          = f.gca()
            #ax.set_xlim(xlim)
            #ax.set_ylim(ylim)

            # title
            if inputs[2] == 0:
                title = 'Movement'
            elif inputs[0] == 0 and inputs[1] == 0:
                title = 'Baseline'
            else:
                title = 'Preparatory'
            ax.set_title(title)

            # and save
            i_str   = str(i)
            f.savefig(savepath + 'frame' + i_str.zfill(6) + '.png', dpi=400)

            # Kill this figure now after having saved it.
            f.clf()
            plt.close(f)

        # Re-align data if necessary
        if prior_align is not None:
            self.sort   = ['conds']
            self.set_align(align=prior_align)
            self.gen_psth()


    # publication quality neural trajectories for delay and no-delay
#   def plot_neural_traj2d_pub(self, window=None, f=None, filters=None, cond_color=None, cond_ls=None, cond_lw=None, savepath=None, filename=None, display=True, alphaval=1, windowEnd=0, targ_on_label=False, go_cue_label=False, pc_label=False, ax_loc=None, xlim=None, ylim=None, projector='PCs'):
#
#       # Align to cb if the dataset wasn't.
#       prior_align = None
#       prior_sort  = None
#
#       # This was the if statement for running Cora stuff; for Chand let's align to cb though if using the right analyze script, these should be equivalent (i.e., cb should always come on at the same time and there should be no start variability FOR THE PURPOSES OF MAKING THESE PC PLOTS)
#       #if self.align != 'start':
#       if self.align != 'cb':
#
#           print 'Re-aligning the data to cb'
#           prior_align = self.align
#           prior_sort  = self.sort
#
#           self.sort   = ['conds']
#           self.set_align(align='start')
#           self.gen_psth()
#
#       if f is None:
#           f = plt.figure()
#       ax = f.gca()
#
#       base, targ_on   = np.array(self.trials[0]['info']['epochs']['pre_targets']) // self.dt
#       go, targ_off    = np.array(self.trials[0]['info']['epochs']['check']) // self.dt
#
#       if window is None:
#           # don't go all the way to the end, similar to cora; maybe let's go 500ms from the end (note depending ond delay, etc. this will change).
#           window = np.arange(self.psths[0]['psth'].shape[1] - windowEnd)
#
#       all_conds   = np.array([zip(psth['sort'], psth['cond']) for psth in self.psths])
#       idxs        = find_conds(all_conds, filters)
#
#       # Plot each traj.
#       for j in idxs:
#
#           # get the projector
#           P = getattr(self, projector)
#           P = P[:, self.dims]
#           means = self.means if projector is 'PCs' or 'rPCs' else means
#
#           # Project the data
#           pc_scores   = np.dot(P.T, (self.psths[j]['psth'].T - means).T)
#
#           # Plot colors and line-styles
#           c_color     = plot_color(*zip(self.psths[j]['sort'], self.psths[j]['cond'])) if cond_color is None else cond_color
#           c_ls        = plot_linestyle(*zip(self.psths[j]['sort'], self.psths[j]['cond'])) if cond_ls is None else cond_ls
#           c_lw        = 1 if cond_lw is None else cond_lw
#
#           # Plot the PCs
#           ax.plot(pc_scores[0,window], pc_scores[1,window], color=c_color, linestyle=c_ls, alpha=alphaval, linewidth=c_lw)
#           ### now plot important time markers
#           # targ on
#           if targ_on_label:
#               ax.plot(pc_scores[0, targ_on], pc_scores[1, targ_on], color=(0.4,0.7,1), marker='.', markersize=20)
#           # go-cue
#           if go_cue_label:
#               ax.plot(pc_scores[0, go], pc_scores[1, go], color=(0.4,0.7,0.4), marker='.', markersize=20)
#
#       if xlim is not None:
#           ax.set_xlim(xlim)
#       if ylim is not None:
#           ax.set_ylim(ylim)
#
#       # Add line for scalebar
#       xmin, xmax  = ax.get_xlim()
#       ymin, ymax  = ax.get_ylim()
#       xrange      = xmax - xmin
#       yrange      = ymax - ymin
#
#       # make the plot pretty
#       ax.spines["top"].set_visible(False)
#       ax.spines["right"].set_visible(False)
#       ax.spines["left"].set_visible(False)
#       ax.spines["bottom"].set_visible(False)
#
#       ## Remove ticks on top and right
#       #ax.get_xaxis().tick_bottom()
#       #ax.get_yaxis().tick_left()
#       ax.tick_params(axis='x', bottom='off', top='off', labelbottom='off')
#       ax.tick_params(axis='y', left='off', right='off', labelleft='off')
#
#       ## Add PC labels
#
#       ## Plot the ticks for go-cue and targets on
#       #plt.xticks((self.align_time*self.dt - delay, self.align_time*self.dt), ("Target on", "Go cue"))
#       #plt.yticks(())
#       # PC_1 label
#
#       if ax_loc is None:
#           x_loc = xmin
#           y_loc = ymin
#
#       if pc_label:
#           ax.plot([x_loc + 0.01*xrange, x_loc+0.11*xrange], [y_loc+0.01*yrange, y_loc+0.01*yrange], linewidth=3, color='k')
#           ax.text(x_loc+0.13*xrange, y_loc + 0.01*yrange, "PC$_{}$".format(self.dims[0] + 1), verticalalignment='center', fontsize=12)
#           # PC_2 label
#           ax.plot([x_loc + 0.01*yrange, x_loc + 0.01*yrange], [y_loc + 0.01*yrange, y_loc + 0.11*yrange], linewidth=3, color='k')
#           ax.text(x_loc+0.01*xrange, y_loc + 0.13*yrange, "PC$_{}$".format(self.dims[1] + 1), horizontalalignment='center', fontsize=12)
#
#       ## Labels
#       if targ_on_label:
#           ax.plot(xmax - 0.2 * xrange, ymin + 0.15*yrange, marker='.', markersize=20, color=(0.4, 0.7, 1))
#           ax.text(xmax - 0.175*xrange, ymin + 0.15*yrange, "Target onset", verticalalignment='center', fontsize=12, color=(0.4, 0.7, 1))
#
#       if go_cue_label:
#           ax.plot(xmax - 0.2 * xrange, ymin + 0.075*yrange, marker='.', markersize=20, color=(0.4, 0.7, 0.4))
#           ax.text(xmax - 0.175*xrange, ymin + 0.075*yrange, "Go cue", verticalalignment='center', fontsize=12, color=(0.4, 0.7, 0.4))
#
#       if filename is None:
#           filename = 'traj.pdf'
#
#       # save it down
#       if savepath is not None:
#           f.savefig(savepath + filename.format(self.dims + 1, self.sort, self.align))
#       # Re-align data if necessary
#       if prior_align is not None:
#           self.sort   = ['conds']
#           self.set_align(align=prior_align)
#           self.gen_psth()
#
#       if not display:
#           f.clear()
#
#       return f

    def plot_neural_traj3d_partition_pub(self, window=None, f=None, filters=None, cond_color=None, cond_ls=None, cond_lw=None, savepath=None, filename=None, display=True, alphaval=1, windowStart=0, windowEnd=0, targ_on_label=False, go_cue_label=False, pc_label=False, ax_loc=None, xlim=None, ylim=None, projector='PCs', close=True, betas=None):

        from mpl_toolkits.mplot3d import Axes3D
        # Align to cb if the dataset wasn't.
        prior_align = None
        prior_sort  = None

        # This was the if statement for running Cora stuff; for Chand let's align to cb though if using the right analyze script, these should be equivalent (i.e., cb should always come on at the same time and there should be no start variability FOR THE PURPOSES OF MAKING THESE PC PLOTS)
        #if self.align != 'start':
        if self.align != 'cb':

            print 'Re-aligning the data to cb'
            prior_align = self.align
            prior_sort  = self.sort

            self.sort   = ['conds']
            self.set_align(align='start')
            self.gen_psth()

        base, targ_on   = np.array(self.trials[0]['info']['epochs']['pre_targets']) // self.dt
        go, targ_off    = np.array(self.trials[0]['info']['epochs']['check']) // self.dt

        if window is None:
            # don't go all the way to the end, similar to cora; maybe let's go 500ms from the end (note depending ond delay, etc. this will change).
            window = np.arange(windowStart, self.psths[0]['psth'].shape[1] - windowEnd)

        all_conds   = np.array([zip(psth['sort'], psth['cond']) for psth in self.psths])
        idxs        = find_conds(all_conds, filters)

        for pidx, partition in enumerate(self.partitions):
            f = plt.figure()
            ax = f.add_subplot(111, projection='3d')

            pca_dict = self.partition_pca[pidx]
            P = pca_dict['PCs']
            self.dims = np.array([0, 1, 2])
            P = P[:, self.dims]
            means = pca_dict['means']

            # Plot each traj.
            for j in idxs:

                # Project the data
                pc_scores   = np.dot(P.T, (self.psths[j]['psth'][partition, :].T - means).T)
                # Plot colors and line-styles
                c_color     = plot_color(*zip(self.psths[j]['sort'], self.psths[j]['cond'])) if cond_color is None else cond_color
                c_ls        = plot_linestyle(*zip(self.psths[j]['sort'], self.psths[j]['cond'])) if cond_ls is None else cond_ls
                c_lw        = 1 if cond_lw is None else cond_lw

                # Plot the PCs
                # pdb.set_trace()
                ax.plot(pc_scores[0,window], pc_scores[1,window], pc_scores[2,window], color=c_color, linestyle=c_ls, alpha=alphaval, linewidth=c_lw)
                ### now plot important time markers
                # targ on
                if targ_on_label:
                    ax.plot(pc_scores[0, targ_on], pc_scores[1, targ_on], color=(0.4,0.7,1), marker='.', markersize=20)
                # go-cue
                if go_cue_label:
                    ax.plot(pc_scores[0, go], pc_scores[1, go], color=(0.4,0.7,0.4), marker='.', markersize=20)

                # plot the inputs
                inp_scores = np.dot(P.T, self.rnn.Win.dot(self.psths[j]['u_psth'][:, :])[partition, :])
                ax.plot(inp_scores[0,window], inp_scores[1,window], inp_scores[2, window], color='black', linestyle=c_ls, alpha=alphaval, linewidth=c_lw)

            # TODO clean up code:
            tdr_axes = P.T.dot(betas)
            # pdb.set_trace()
            x2 = tdr_axes[0, 0]
            y2 = tdr_axes[1, 0]
            z2 = tdr_axes[2, 0]
            x1 = -x2
            y1 = -y2
            z1 = -z2

            ax.plot([x1, x2], [y1, y2], [z1, z2], marker = 'o', label='context')
            x2 = tdr_axes[0, 1]
            y2 = tdr_axes[1, 1]
            z2 = tdr_axes[2, 1]
            x1 = -x2
            y1 = -y2
            z1 = -z2
            ax.plot([x1, x2], [y1, y2], [z1, z2], marker = 'o', label='color coherence')
            x2 = tdr_axes[0, 2]
            y2 = tdr_axes[1, 2]
            z2 = tdr_axes[2, 2]
            x1 = -x2
            y1 = -y2
            z1 = -z2
            ax.plot([x1, x2], [y1, y2], [z1, z2], marker = 'o', label='direction')

            ax.legend()
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)

            # Add line for scalebar
            xmin, xmax  = ax.get_xlim()
            ymin, ymax  = ax.get_ylim()
            xrange      = xmax - xmin
            yrange      = ymax - ymin

            # make the plot pretty
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)

            ## Remove ticks on top and right
            #ax.get_xaxis().tick_bottom()
            #ax.get_yaxis().tick_left()
            # ax.tick_params(axis='x', bottom='off', top='off', labelbottom='off')
            # ax.tick_params(axis='y', left='off', right='off', labelleft='off')

            ## Add PC labels

            ## Plot the ticks for go-cue and targets on
            #plt.xticks((self.align_time*self.dt - delay, self.align_time*self.dt), ("Target on", "Go cue"))
            #plt.yticks(())
            # PC_1 label

            if ax_loc is None:
                x_loc = xmin
                y_loc = ymin

            if pc_label:
                ax.plot([x_loc + 0.01*xrange, x_loc+0.11*xrange], [y_loc+0.01*yrange, y_loc+0.01*yrange], linewidth=3, color='k')
                ax.text(x_loc+0.13*xrange, y_loc + 0.01*yrange, "PC$_{}$".format(self.dims[0] + 1), verticalalignment='center', fontsize=12)
                # PC_2 label
                ax.plot([x_loc + 0.01*yrange, x_loc + 0.01*yrange], [y_loc + 0.01*yrange, y_loc + 0.11*yrange], linewidth=3, color='k')
                ax.text(x_loc+0.01*xrange, y_loc + 0.13*yrange, "PC$_{}$".format(self.dims[1] + 1), horizontalalignment='center', fontsize=12)

            ## Labels
            if targ_on_label:
                ax.plot(xmax - 0.2 * xrange, ymin + 0.15*yrange, marker='.', markersize=20, color=(0.4, 0.7, 1))
                ax.text(xmax - 0.175*xrange, ymin + 0.15*yrange, "Target onset", verticalalignment='center', fontsize=12, color=(0.4, 0.7, 1))

            if go_cue_label:
                ax.plot(xmax - 0.2 * xrange, ymin + 0.075*yrange, marker='.', markersize=20, color=(0.4, 0.7, 0.4))
                ax.text(xmax - 0.175*xrange, ymin + 0.075*yrange, "Go cue", verticalalignment='center', fontsize=12, color=(0.4, 0.7, 0.4))

            if filename is None:
                filename = 'traj.pdf'

            # save it down
            # pdb.set_trace()
            # savepath = '/home2/michael/work/projects/information/results/'
            if savepath is not None:
                f.savefig(savepath + 'partition={}_'.format(pidx) + filename.format(self.dims + 1, self.sort, self.align))

                for ii in range(0,360,20):
                    ax.view_init(elev=10., azim=ii)
                    f.savefig(savepath + 'movie/partition={}_'.format(pidx) +  'movie%d.pdf' % ii)

            # Re-align data if necessary
            if prior_align is not None:
                self.sort   = ['conds']
                self.set_align(align=prior_align)
                self.gen_psth()

            plt.close('all')

        return f

    def plot_neural_traj2d_partition_pub(self, window=None, f=None, filters=None, cond_color=None, cond_ls=None, cond_lw=None, savepath=None, filename=None, display=True, alphaval=1, windowStart=0, windowEnd=0, targ_on_label=False, go_cue_label=False, pc_label=False, ax_loc=None, xlim=None, ylim=None, projector='PCs', close=True, betas=None):

        # Align to cb if the dataset wasn't.
        prior_align = None
        prior_sort  = None

        # This was the if statement for running Cora stuff; for Chand let's align to cb though if using the right analyze script, these should be equivalent (i.e., cb should always come on at the same time and there should be no start variability FOR THE PURPOSES OF MAKING THESE PC PLOTS)
        #if self.align != 'start':
        if self.align != 'cb':

            print 'Re-aligning the data to cb'
            prior_align = self.align
            prior_sort  = self.sort

            self.sort   = ['conds']
            self.set_align(align='start')
            self.gen_psth()

        base, targ_on   = np.array(self.trials[0]['info']['epochs']['pre_targets']) // self.dt
        go, targ_off    = np.array(self.trials[0]['info']['epochs']['check']) // self.dt

        if window is None:
            # don't go all the way to the end, similar to cora; maybe let's go 500ms from the end (note depending ond delay, etc. this will change).
            window = np.arange(windowStart, self.psths[0]['psth'].shape[1] - windowEnd)

        all_conds   = np.array([zip(psth['sort'], psth['cond']) for psth in self.psths])
        idxs        = find_conds(all_conds, filters)

        for pidx, partition in enumerate(self.partitions):
            f = plt.figure()
            ax = f.gca()

            pca_dict = self.partition_pca[pidx]
            P = pca_dict['PCs']
            P = P[:, self.dims]
            means = pca_dict['means']

            # Plot each traj.
            for j in idxs:

                # Project the data
                pc_scores   = np.dot(P.T, (self.psths[j]['psth'][partition, :].T - means).T)
                # Plot colors and line-styles
                c_color     = plot_color(*zip(self.psths[j]['sort'], self.psths[j]['cond'])) if cond_color is None else cond_color
                c_ls        = plot_linestyle(*zip(self.psths[j]['sort'], self.psths[j]['cond'])) if cond_ls is None else cond_ls
                c_lw        = 1 if cond_lw is None else cond_lw

                # Plot the PCs
                # pdb.set_trace()
                ax.plot(pc_scores[0,window], pc_scores[1,window], color=c_color, linestyle=c_ls, alpha=alphaval, linewidth=c_lw)
                ### now plot important time markers
                # targ on
                if targ_on_label:
                    ax.plot(pc_scores[0, targ_on], pc_scores[1, targ_on], color=(0.4,0.7,1), marker='.', markersize=20)
                # go-cue
                if go_cue_label:
                    ax.plot(pc_scores[0, go], pc_scores[1, go], color=(0.4,0.7,0.4), marker='.', markersize=20)

                # plot the Inputs
                inp_scores = np.dot(P.T, self.rnn.Win.dot(self.psths[j]['u_psth'][:, :])[partition, :])
                ax.plot(inp_scores[0,window], inp_scores[1,window], color='black', linestyle=c_ls, alpha=alphaval, linewidth=c_lw)

            # TODO clean up code:
            if betas is not None:
                tdr_axes = P.T.dot(betas)
                x2 = tdr_axes[0, 0]
                y2 = tdr_axes[1, 0]
                x1 = -x2
                y1 = -y2
                ax.plot([x1, x2], [y1, y2], marker = 'o', label='color')
                x2 = tdr_axes[0, 1]
                y2 = tdr_axes[1, 1]
                x1 = -x2
                y1 = -y2
                ax.plot([x1, x2], [y1,y2], marker = 'o', label='direction')
                x2 = tdr_axes[0, 2]
                y2 = tdr_axes[1, 2]
                x1 = -x2
                y1 = -y2
                ax.plot([x1, x2], [y1,y2], marker = 'o', label='context')

            ax.legend()
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)

            # Add line for scalebar
            xmin, xmax  = ax.get_xlim()
            ymin, ymax  = ax.get_ylim()
            xrange      = xmax - xmin
            yrange      = ymax - ymin

            # make the plot pretty
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)

            ## Remove ticks on top and right
            #ax.get_xaxis().tick_bottom()
            #ax.get_yaxis().tick_left()
            # ax.tick_params(axis='x', bottom='off', top='off', labelbottom='off')
            # ax.tick_params(axis='y', left='off', right='off', labelleft='off')

            ## Add PC labels

            ## Plot the ticks for go-cue and targets on
            #plt.xticks((self.align_time*self.dt - delay, self.align_time*self.dt), ("Target on", "Go cue"))
            #plt.yticks(())
            # PC_1 label

            if ax_loc is None:
                x_loc = xmin
                y_loc = ymin

            if pc_label:
                ax.plot([x_loc + 0.01*xrange, x_loc+0.11*xrange], [y_loc+0.01*yrange, y_loc+0.01*yrange], linewidth=3, color='k')
                ax.text(x_loc+0.13*xrange, y_loc + 0.01*yrange, "PC$_{}$".format(self.dims[0] + 1), verticalalignment='center', fontsize=12)
                # PC_2 label
                ax.plot([x_loc + 0.01*yrange, x_loc + 0.01*yrange], [y_loc + 0.01*yrange, y_loc + 0.11*yrange], linewidth=3, color='k')
                ax.text(x_loc+0.01*xrange, y_loc + 0.13*yrange, "PC$_{}$".format(self.dims[1] + 1), horizontalalignment='center', fontsize=12)

            ## Labels
            if targ_on_label:
                ax.plot(xmax - 0.2 * xrange, ymin + 0.15*yrange, marker='.', markersize=20, color=(0.4, 0.7, 1))
                ax.text(xmax - 0.175*xrange, ymin + 0.15*yrange, "Target onset", verticalalignment='center', fontsize=12, color=(0.4, 0.7, 1))

            if go_cue_label:
                ax.plot(xmax - 0.2 * xrange, ymin + 0.075*yrange, marker='.', markersize=20, color=(0.4, 0.7, 0.4))
                ax.text(xmax - 0.175*xrange, ymin + 0.075*yrange, "Go cue", verticalalignment='center', fontsize=12, color=(0.4, 0.7, 0.4))

            if filename is None:
                filename = 'traj.pdf'

            # save it down
            # pdb.set_trace()
            # savepath = '/home2/michael/work/projects/information/results/'
            if savepath is not None:
                f.savefig(savepath + 'partition={}_'.format(pidx) + filename.format(self.dims + 1, self.sort, self.align))
            # Re-align data if necessary
            if prior_align is not None:
                self.sort   = ['conds']
                self.set_align(align=prior_align)
                self.gen_psth()

            plt.close('all')

        return f


    # Plots the local dynamics from deltas
    def plot_local_dynamics_delta_pub(self, inputs=np.array((0,0,0,0)), f=None, scale=1, force_alignment=None, t_sample=None, coh=None, reset_align=True, filters=None, xlim=None, ylim=None, projector='PCs'):

        # coh is for sampling get_higher_dims, it is the red coherence (NOT CB).
        x_deltas, y_deltas  = self.delta_samples

        # store values in the following array.
        ss      = np.zeros((2, len(x_deltas), len(y_deltas)))
        sdots   = np.zeros((2, len(x_deltas), len(y_deltas)))
        sdot_norms = np.zeros((len(x_deltas), len(y_deltas)))

        # Create the input to the RNN from which we sample dynamics
        higher_dim_inputs   = inputs.astype('float64')  # needed since coh is fractional

        # Allow the user to specify the coherences if desired for the input.
        if coh is not None:
            higher_dim_inputs[2]    = +coh
            higher_dim_inputs[3]    = -coh

        # Call to sample initial point
        x0      = self.get_higher_dims(inputs=higher_dim_inputs, return_psth=True, field='x', force_alignment=force_alignment, t_sample=t_sample, reset_align=reset_align, filters=filters)

        # Get all the derivatives at the sample points.
        for i in np.arange(len(x_deltas)):
            for j in np.arange(len(y_deltas)):

                # Get the high-dimensional point from which we're sampling dynamics
                if projector is 'PCs' or 'rPCs':
                    x0_s                = x0 + np.dot(self.x_PCs[:,self.dims],  np.hstack((x_deltas[i], y_deltas[j]))) # no need to add self.x_means since this is a delta

                if projector is 'isoWr':
                    r0          = self.act(x0)
                    rx_deltas   = self.act(x_deltas)
                    ry_deltas   = self.act(y_deltas)
                    r0_s        = r0 + np.dot(self.isoWr[:, self.dims], np.hstack((rx_deltas[i], ry_deltas[j])))
                    # Clip
                    idx_below   = np.where(r0_s < -1)
                    idx_above   = np.where(r0_s > 1)
                    r0_s[idx_below] = -1+1e-10
                    r0_s[idx_above] = 1-1e-10

                    x0_s        = self.invAct(r0_s)

                this_s, this_sdot   = self.sample_local_dynamics(x0=x0_s, inputs=inputs, projector=projector)
                sdots[:, i, j]      = this_sdot[self.dims]
                sdot_norms[i,j]     = np.linalg.norm(this_sdot[self.dims])

#               if np.isnan(sdot_norms[i,j]):
#                   pdb.set_trace()
                ss[:, i, j]         = this_s[self.dims]

                # This is debugging code.
                #if i == 11 and j == 7:
                #   x0_save = x0_s

        # This is debugging code.
        #idx_x, idx_y   = np.where(sdot_norms == np.min(sdot_norms))
        #x0_fp, _, _    = self.get_fixed_point(x0, inputs)
        #s_fp, s_sdot   = self.sample_local_dynamics(x0=x0_fp, inputs=inputs, projector=projector)
        #s_sdot_norm        = np.linalg.norm(s_sdot[self.dims])


        #r0_save        = self.rnn_rdot(x0_save, inputs)
        #r0_fp          = self.rnn_rdot(x0_fp, inputs)
        #_, sd_save         = self.sample_local_dynamics(x0=x0_save, inputs=inputs, projector=projector)
        #sd_fp          = s_sdot

#       pdb.set_trace()

        # Now plot them.
        if f is None:
            f = plt.figure()
        ax  = f.gca()

        round_f = 1
        rx_min  = np.min(ss[0]) // round_f * round_f
        rx_max  = np.max(ss[0] + round_f) // round_f * round_f
        ry_min  = np.min(ss[1]) // round_f * round_f
        ry_max  = np.max(ss[1] + round_f) // round_f * round_f

        ax.set_xlim([rx_min, rx_max])
        ax.set_ylim([ry_min, ry_max])

        sdots   = self.rnn.p['dt'] * sdots * scale # since dt = 10ms; this is actually how far the trajectory will go

        # calculate magnitudes for arrow scaling
        # first dim is 2 since these are two d arrows, the rest we reshape over
        smags   = sdots.reshape((2, np.prod(sdots.shape[1:])))
        smags   = np.sqrt(smags[0,:]**2 + smags[1,:]**2)
        smax    = np.max(smags)
        smax    = 0.1 # this normalizes across all portraits

        # For debugging
        #ax.plot(s_fp[self.dims[0]], s_fp[self.dims[1]], marker='.', color='g')

        for i in np.arange(len(x_deltas)):
            for j in np.arange(len(y_deltas)):
                this_mag = np.sqrt(sdots[0, i, j]**2 + sdots[1, i, j]**2)
                ax.arrow(ss[0, i, j], ss[1, i, j], sdots[0, i, j], sdots[1, i, j], color=np.array([0.5, 0.5, 0.5]), head_width=0.05 * this_mag/smax, head_length=0.1 * this_mag/smax)

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        return f


    # publication quality neural trajectories for delay and no-delay
    def plot_neural_traj2d_pub(self, window=None, f=None, filters=None, cond_color=None, cond_ls=None, cond_lw=None, savepath=None, filename=None, display=True, alphaval=1, windowStart=0, windowEnd=0, targ_on_label=False, go_cue_label=False, pc_label=False, ax_loc=None, xlim=None, ylim=None, projector='PCs'):

        # Align to cb if the dataset wasn't.
        prior_align = None
        prior_sort  = None

        # This was the if statement for running Cora stuff; for Chand let's align to cb though if using the right analyze script, these should be equivalent (i.e., cb should always come on at the same time and there should be no start variability FOR THE PURPOSES OF MAKING THESE PC PLOTS)
        if self.align != 'start':
        #if self.align != 'cb':
            print 'Re-aligning the data to start'
            prior_align = self.align
            prior_sort  = self.sort

            self.sort   = ['conds']
            self.set_align(align='start')
            self.gen_psth()

        if f is None:
            f = plt.figure()
        ax = f.gca()

        base, targ_on   = np.array(self.trials[0]['info']['epochs']['pre_targets']) // self.dt
        go, targ_off    = np.array(self.trials[0]['info']['epochs']['check']) // self.dt

        if window is None:
            # don't go all the way to the end, similar to cora; maybe let's go 500ms from the end (note depending ond delay, etc. this will change).
            window = np.arange(windowStart, self.psths[0]['psth'].shape[1] - windowEnd)

        all_conds   = np.array([zip(psth['sort'], psth['cond']) for psth in self.psths])
        idxs        = find_conds(all_conds, filters)

        # Plot each traj.
        for j in idxs:

            # get the projector
            P = getattr(self, projector)
            P = P[:, self.dims]
            means = self.means if projector is 'PCs' or 'rPCs' else means
            # Project the data
            pc_scores   = np.dot(P.T, (self.psths[j]['psth'].T - means).T)

            # Plot colors and line-styles
            c_color     = plot_color(*zip(self.psths[j]['sort'], self.psths[j]['cond'])) if cond_color is None else cond_color
            c_ls        = plot_linestyle(*zip(self.psths[j]['sort'], self.psths[j]['cond'])) if cond_ls is None else cond_ls
            c_lw        = 1 if cond_lw is None else cond_lw

            # Plot the PCs
            ax.plot(pc_scores[0,window], pc_scores[1,window], color=c_color, linestyle=c_ls, alpha=alphaval, linewidth=c_lw)
            ### now plot important time markers
            # targ on
            if targ_on_label:
                ax.plot(pc_scores[0, targ_on], pc_scores[1, targ_on], color=(0.4,0.7,1), marker='.', markersize=20)
            # go-cue
            if go_cue_label:
                ax.plot(pc_scores[0, go], pc_scores[1, go], color=(0.4,0.7,0.4), marker='.', markersize=20)

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        # Add line for scalebar
        xmin, xmax  = ax.get_xlim()
        ymin, ymax  = ax.get_ylim()
        xrange      = xmax - xmin
        yrange      = ymax - ymin

        # make the plot pretty
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        ## Remove ticks on top and right
        #ax.get_xaxis().tick_bottom()
        #ax.get_yaxis().tick_left()
        ax.tick_params(axis='x', bottom='off', top='off', labelbottom='off')
        ax.tick_params(axis='y', left='off', right='off', labelleft='off')

        ## Add PC labels

        ## Plot the ticks for go-cue and targets on
        #plt.xticks((self.align_time*self.dt - delay, self.align_time*self.dt), ("Target on", "Go cue"))
        #plt.yticks(())
        # PC_1 label

        if ax_loc is None:
            x_loc = xmin
            y_loc = ymin

        if pc_label:
            ax.plot([x_loc + 0.01*xrange, x_loc+0.11*xrange], [y_loc+0.01*yrange, y_loc+0.01*yrange], linewidth=3, color='k')
            ax.text(x_loc+0.13*xrange, y_loc + 0.01*yrange, "PC$_{}$".format(self.dims[0] + 1), verticalalignment='center', fontsize=12)
            # PC_2 label
            ax.plot([x_loc + 0.01*yrange, x_loc + 0.01*yrange], [y_loc + 0.01*yrange, y_loc + 0.11*yrange], linewidth=3, color='k')
            ax.text(x_loc+0.01*xrange, y_loc + 0.13*yrange, "PC$_{}$".format(self.dims[1] + 1), horizontalalignment='center', fontsize=12)

        ## Labels
        if targ_on_label:
            ax.plot(xmax - 0.2 * xrange, ymin + 0.15*yrange, marker='.', markersize=20, color=(0.4, 0.7, 1))
            ax.text(xmax - 0.175*xrange, ymin + 0.15*yrange, "Target onset", verticalalignment='center', fontsize=12, color=(0.4, 0.7, 1))

        if go_cue_label:
            ax.plot(xmax - 0.2 * xrange, ymin + 0.075*yrange, marker='.', markersize=20, color=(0.4, 0.7, 0.4))
            ax.text(xmax - 0.175*xrange, ymin + 0.075*yrange, "Go cue", verticalalignment='center', fontsize=12, color=(0.4, 0.7, 0.4))

        if filename is None:
            filename = 'traj.pdf'

        # save it down
        savepath = '/home2/michael/work/projects/information/results'
        if savepath is not None:
            f.savefig(savepath + filename.format(self.dims + 1, self.sort, self.align))
        # Re-align data if necessary
        if prior_align is not None:
            self.sort   = ['conds']
            self.set_align(align=prior_align)
            self.gen_psth()

        if not display:
            f.clear()

        return f

    # Plots the local dynamics from deltas
    def plot_local_dynamics_delta_pub(self, inputs=np.array((0,0,0,0)), f=None, scale=1, force_alignment=None, t_sample=None, coh=None, reset_align=True, filters=None, xlim=None, ylim=None, projector='PCs'):

        # coh is for sampling get_higher_dims, it is the red coherence (NOT CB).
        x_deltas, y_deltas  = self.delta_samples

        # store values in the following array.
        ss      = np.zeros((2, len(x_deltas), len(y_deltas)))
        sdots   = np.zeros((2, len(x_deltas), len(y_deltas)))
        sdot_norms = np.zeros((len(x_deltas), len(y_deltas)))

        # Create the input to the RNN from which we sample dynamics
        higher_dim_inputs   = inputs.astype('float64')  # needed since coh is fractional

        # Allow the user to specify the coherences if desired for the input.
        if coh is not None:
            higher_dim_inputs[2]    = +coh
            higher_dim_inputs[3]    = -coh

        # Call to sample initial point
        x0      = self.get_higher_dims(inputs=higher_dim_inputs, return_psth=True, field='x', force_alignment=force_alignment, t_sample=t_sample, reset_align=reset_align, filters=filters)

        # Get all the derivatives at the sample points.
        for i in np.arange(len(x_deltas)):
            for j in np.arange(len(y_deltas)):

                # Get the high-dimensional point from which we're sampling dynamics
                if projector is 'PCs' or 'rPCs':
                    x0_s                = x0 + np.dot(self.x_PCs[:,self.dims],  np.hstack((x_deltas[i], y_deltas[j]))) # no need to add self.x_means since this is a delta

                if projector is 'isoWr':
                    r0          = self.act(x0)
                    rx_deltas   = self.act(x_deltas)
                    ry_deltas   = self.act(y_deltas)
                    r0_s        = r0 + np.dot(self.isoWr[:, self.dims], np.hstack((rx_deltas[i], ry_deltas[j])))
                    # Clip
                    idx_below   = np.where(r0_s < -1)
                    idx_above   = np.where(r0_s > 1)
                    r0_s[idx_below] = -1+1e-10
                    r0_s[idx_above] = 1-1e-10

                    x0_s        = self.invAct(r0_s)

                this_s, this_sdot   = self.sample_local_dynamics(x0=x0_s, inputs=inputs, projector=projector)
                sdots[:, i, j]      = this_sdot[self.dims]
                sdot_norms[i,j]     = np.linalg.norm(this_sdot[self.dims])

#               if np.isnan(sdot_norms[i,j]):
#                   pdb.set_trace()
                ss[:, i, j]         = this_s[self.dims]

                # This is debugging code.
                #if i == 11 and j == 7:
                #   x0_save = x0_s

        # This is debugging code.
        #idx_x, idx_y   = np.where(sdot_norms == np.min(sdot_norms))
        #x0_fp, _, _    = self.get_fixed_point(x0, inputs)
        #s_fp, s_sdot   = self.sample_local_dynamics(x0=x0_fp, inputs=inputs, projector=projector)
        #s_sdot_norm        = np.linalg.norm(s_sdot[self.dims])


        #r0_save        = self.rnn_rdot(x0_save, inputs)
        #r0_fp          = self.rnn_rdot(x0_fp, inputs)
        #_, sd_save         = self.sample_local_dynamics(x0=x0_save, inputs=inputs, projector=projector)
        #sd_fp          = s_sdot

#       pdb.set_trace()

        # Now plot them.
        if f is None:
            f = plt.figure()
        ax  = f.gca()

        round_f = 1
        rx_min  = np.min(ss[0]) // round_f * round_f
        rx_max  = np.max(ss[0] + round_f) // round_f * round_f
        ry_min  = np.min(ss[1]) // round_f * round_f
        ry_max  = np.max(ss[1] + round_f) // round_f * round_f

        ax.set_xlim([rx_min, rx_max])
        ax.set_ylim([ry_min, ry_max])

        sdots   = self.rnn.p['dt'] * sdots * scale # since dt = 10ms; this is actually how far the trajectory will go

        # calculate magnitudes for arrow scaling
        # first dim is 2 since these are two d arrows, the rest we reshape over
        smags   = sdots.reshape((2, np.prod(sdots.shape[1:])))
        smags   = np.sqrt(smags[0,:]**2 + smags[1,:]**2)
        smax    = np.max(smags)
        smax    = 0.1 # this normalizes across all portraits

        # For debugging
        #ax.plot(s_fp[self.dims[0]], s_fp[self.dims[1]], marker='.', color='g')

        for i in np.arange(len(x_deltas)):
            for j in np.arange(len(y_deltas)):
                this_mag = np.sqrt(sdots[0, i, j]**2 + sdots[1, i, j]**2)
                ax.arrow(ss[0, i, j], ss[1, i, j], sdots[0, i, j], sdots[1, i, j], color=np.array([0.5, 0.5, 0.5]), head_width=0.05 * this_mag/smax, head_length=0.1 * this_mag/smax)

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        return f

    # eigenvalue plot
    def plot_evals_pub(self, savepath=None, filename='evals.pdf', display=True, attr='evals'):

        evals = getattr(self, attr)

        # calculations
        var_per_dim = evals / sum(evals)
        accum_var   = np.cumsum(var_per_dim)

        # First plot solo evals
        f   = plt.figure()
        ax  = f.gca()

        ax.plot(var_per_dim, linewidth=0, marker='x', color=np.array([0.3, 0.3, 0.3]), markersize=7)
        ax.plot(accum_var, linewidth=0, marker='.', color=np.array([0,0,0]), markersize=7)

        # make the plot pretty
        num_neurons = self.psths[0]['psth'].shape[0]

        # set which axes are invisible
        ax.spines["top"].set_visible(False)

        # axis limits
        ax.set_ylim([0, 1.1])
        ax.set_xlim([0 - 0.025 * num_neurons, 1.025 * num_neurons])

        # change axis colors
        ax.spines["right"].set_color('#4d4d4d')

        # axis tick properties
        ax.tick_params(axis='x', bottom='on', top='off', labelbottom='on', direction='out')
        ax.tick_params(axis='y', left='on', right='on', labelleft='on', labelright='on', direction='out')

        # axis tick locations
        ax.xaxis.set_ticks(np.arange(0, num_neurons + 1, num_neurons // 5))
        ax.yaxis.set_ticks(np.arange(0, 1.1, 1/5))

        # save it down
        if savepath is not None:
            f.savefig(savepath + filename)

        if not display:
            f.clear()

        return f


    def plot_something_vs_neurons(self, savepath=None, filename='blank.pdf', display=True, attr='WrSingVals'):

        data = getattr(self, attr)

        f   = plt.figure()
        ax  = f.gca()

        ax.plot(data, linewidth=0, marker='x', color=np.array([0.3, 0.3, 0.3]), markersize=7)

        # make the plot pretty
        num_neurons = self.psths[0]['psth'].shape[0]

        # set which axes are invisible
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # axis limits
        ax.set_ylim([0, 1.1 * np.max(data)])
        ax.set_xlim([0 - 0.025 * num_neurons, 1.025 * num_neurons])

        # change axis colors
        ax.spines["right"].set_color('#4d4d4d')

        # axis tick properties
        ax.tick_params(axis='x', bottom='on', top='off', labelbottom='on', direction='out')
        ax.tick_params(axis='y', left='on', right='off', labelleft='on', labelright='off', direction='out')

        # axis tick locations
        ax.xaxis.set_ticks(np.arange(0, num_neurons + 1, num_neurons // 5))
        ax.yaxis.set_ticks(np.arange(0, np.max(data)*1.01, np.max(data)))

        # save it down
        if savepath is not None:
            f.savefig(savepath + filename)

        if not display:
            f.clear()

        return f


    # PCs vs time plot
    def plot_pcs_vs_time_pub(self, dim=0, savepath=None, filename='PC_time.pdf', display=True):

        f   = plt.figure()
        ax  = f.gca()

        base, targ_on   = np.array(self.trials[0]['info']['epochs']['pre_targets']) // self.dt
        go, targ_off    = np.array(self.trials[0]['info']['epochs']['check']) // self.dt
        delay           = (go - targ_on) * self.dt

        for j in np.arange(len(self.psths)):

            # Project the data
            pc_scores   = np.dot(self.PCs[:, dim].T, (self.psths[j]['psth'].T - self.means).T)

            # Plot colors and line-styles
            cond_color  = plot_color(*zip(self.psths[j]['sort'], self.psths[j]['cond']))
            cond_ls     = plot_linestyle(*zip(self.psths[j]['sort'], self.psths[j]['cond']))

            # Plot the PCs across time
            ax.plot(np.arange(self.start_time, self.stop_time) * self.dt, pc_scores, color=cond_color, linestyle=cond_ls)
            #ax.axvline(self.align_time * self.dt, color='b')
            #ax.set_title('PCs {}, sortables={}, align={}'.format(i + 1, self.sort, self.align))

        # Remove borders
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        # Remove ticks on top and right
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        # Plot the ticks for go-cue and targets on
        plt.xticks((self.align_time*self.dt - delay, self.align_time*self.dt), ("Target on", "Go cue"))
        plt.yticks(())

        # save it down
        if savepath is not None:
            f.savefig(savepath + filename)

        if not display:
            f.clear()

        return f




