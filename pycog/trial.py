"""
Recurrent neural network for testing networks outside of Theano.

"""
from __future__ import absolute_import
from __future__ import division

import numpy as np
import sys
import imp
import pdb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
from functools import reduce

# For optimization of fixed points
import theano
import theano.tensor as T
from scipy.optimize import minimize

from .rnn 	import RNN
from .		import tasktools

THIS = 'pycog.trial'

# ==========================================
# Calculates coherence given num red squares
# ==========================================
def coh_r(cb_cond):
	return 2*(cb_cond / 225) - 1

def inv_coh_r(coh):
	# ROUNDS TO THE NEAREST COH!  This is a useful tool.
	cohs 		= np.array([11, 45, 67, 78, 90, 101, 108, 117, 124, 135, 147, 158, 180, 214])
	this_coh	= 0.5 * (coh + 1) * 225

	diffs 		= np.abs(cohs - this_coh)

	idx_close	= np.where(diffs == min(diffs))[0][0]

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

	def __init__(self, rnnfile, modelfile, num_trials=100, seed=1, rnnparams={}):
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

		rnnparams: dict, optional
					Parameters to override

		"""
		
		self.rnn 		= RNN(rnnfile, rnnparams=rnnparams, verbose=False) 
		self.m 			= imp.load_source('model', modelfile)
		self.ntrials	= num_trials * self.m.nconditions
		self.rng 		= np.random.RandomState(seed)
		self.dt 		= self.rnn.p['dt']

		# Build the trial information
		self.build_trials()

	# =================================================================================
	# Generates many RNN trials - pre-requisite to psychometric, RT curves, psths, etc.
	# =================================================================================
	def run_trials(self):

		w			= len(str(self.ntrials))
		trials 		= []
		conds 		= []
		backspaces 	= 0

		for i in range(self.ntrials):

			# Get the condition and left-right orientation
			b 			= i % self.m.nconditions
			k1, k2 		= tasktools.unravel_index(b, (len(self.m.conds), len(self.m.left_rights)))
			cond 		= self.m.conds[k1]
			left_right 	= self.m.left_rights[k2]

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
			trial = {
					't': self.rnn.t[::],
					'u': self.rnn.u[:,::],
					'r': self.rnn.r[:,::],
					'x': self.rnn.x[:,::],
					'z': self.rnn.z[:,::],
					'info': info
				}

			trials.append(trial)
			conds.append(cond)

		self.trials = np.array(trials)
		self.conds 	= np.array([trial['info']['cond'] for trial in trials])
		self.cohs	= self.conds

	# ===========================
	# Adds reaction time to trials 
	# ============================
	def add_rt_to_trials(self, threshold=0.25):

		dt = self.dt
		self.rts = np.zeros(self.ntrials)

		for i in np.arange(self.ntrials):
			
			choice = self.trials[i]['info']['choice']
			check_on = (self.trials[i]['info']['start_delay'] + self.trials[i]['info']['check_drawn']) // dt
			
			if choice == 1:
				rt 			= np.where(self.trials[i]['z'][0,:] > choice - threshold)[0]
				valid_rts 	= np.where(rt > check_on)[0]
				rt 			= rt[valid_rts]
			else:
				rt 			= np.where(self.trials[i]['z'][0,:] < choice + threshold)[0]
				valid_rts 	= np.where(rt > check_on)[0]
				rt 			= rt[valid_rts]
			
			if np.any(rt):
				self.trials[i]['rt'] 	= (rt[0] - check_on) * dt
				self.rts[i] 			= (rt[0] - check_on) * dt
			else:
				self.trials[i]['rt'] 	= np.nan
				self.rts[i] 			= np.nan 

	# ========================================================
	# Add checkerboard ons to trials, as well as trial lengths
	# ========================================================
	def add_landmark_times_to_trials(self):
		self.cbs	= np.array([trial['info']['epochs']['check'][0] for trial in self.trials]).astype(int)
		self.Ts		= np.array([trial['info']['epochs']['post_targets'][1] for trial in self.trials]).astype(int)

	# ===========================================================================
	# Adds actual reach direction of reach to trials, field 'dir'; -1 vs l is 1-r
	# ===========================================================================
	def add_dir_to_trials(self):
		
		# -1 is to the left, 1 is to the right
		dirs 		= np.array([np.mean(trial['z'][0,:]) > 0 for trial in self.trials]).astype(int)
		dirs 		= 2 * dirs - 1
		self.dirs 	= dirs

		# legacy
		for i in np.arange(self.ntrials):
			self.trials[i]['dir'] = dirs[i]

	# =============================================================================
	# Adds actual color direction of reach to trials, field 'scol' (selected color)
	# =============================================================================
	def add_scol_to_trials(self):
		
		# -1 if reach to left, +1 if reach to right
		dirs	= np.array([np.mean(trial['z'][0,:]) > 0 for trial in self.trials]).astype(int)
		dirs 	= 2 * dirs - 1

		# -1 if left is red, +1 if left is green.
		lrs		= np.array([trial['info']['left_right'] for trial in self.trials]).astype(int)

		# let -1 be red, +1 be green
		scols 	= np.array(lrs + (lrs != dirs).astype(int)) #-1 and +1 for lrs == dirs, 0 if lrs=-1 and he reached to right (i.e., green), and +2 if lrs=+1 and he reached left (i.e., red)
		scols[scols == 0] = +1
		scols[scols == 2] = -1

		self.scols = scols

	# =========================================================
	# Adds prompted direction of reach to trials, field 'pdirs'
	# =========================================================
	def add_pdir_to_trials(self):
		# -1 if correct was reach to left, +1 if reach to right
		self.pdirs 	= np.array([trial['info']['choice'] for trial in self.trials])

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
		pdirs 	= [trial['info']['choice'] for trial in self.trials]

		# -1 if left is red, +1 if left is green.
		lrs		= np.array([trial['info']['left_right'] for trial in self.trials]).astype(int)

		# let -1 be red, +1 be green
		pscols 	= np.array(lrs + (lrs != pdirs).astype(int)) #-1 and +1 for lrs == dirs, 0 if lrs=-1 and he reached to right (i.e., green), and +2 if lrs=+1 and he reached left (i.e., red)
		pscols[pscols == 0] = +1
		pscols[pscols == 2] = -1

		self.pscols = pscols

	# ============================
	# Adds success field to trials 
	# ============================
	def add_success_to_trials(self, threshold=0):
		
		correct_choices = np.array([trial['info']['choice'] for trial in self.trials])
		if not threshold:
			decisions = np.array(np.sign([trial['z'][0, -1 - trial['info']['post_delay'] // trial['info']['dt']] for trial in self.trials])).astype(int)
			trial_outcome = decisions == correct_choices
		else:
			decisions = np.array([trial['z'][0, -1 - trial['info']['post_delay'] // trial['info']['dt']] for trial in self.trials])
			trial_outcome = np.abs(decisions - correct_choices) < threshold

		self.successes = np.zeros(self.ntrials)

		for i in np.arange(self.ntrials):
			self.trials[i]['success'] = trial_outcome[i]
			self.successes

	# ========================
	# Build the list of trials
	# ========================
	def build_trials(self):
		self.run_trials()
		self.add_rt_to_trials()
		self.add_dir_to_trials()
		self.add_success_to_trials()
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
		mask 			= suc & ~np.isnan(rts)

		self.trials 	= self.trials[mask]

		# Task properties
		self.conds 		= self.conds[mask]
		self.cohs 		= self.conds # this is a duplicate field, for psths
		self.dirs 		= self.dirs[mask]
		self.successes	= self.successes[mask]
		self.scols 		= self.scols[mask]
		self.pdirs		= self.pdirs[mask]
		self.pscols		= self.pscols[mask]
		self.lrs 		= self.lrs[mask]
		
		# Task analogs
		self.rts 		= self.rts[mask]
		self.cbs 		= self.cbs[mask]
		self.Ts 		= self.Ts[mask]
		self.ntrials 	= len(self.trials)
	
	# =================================
	# Check the performance of the RNNs
	# =================================

	def eval_performance(self):

		performance = tasktools.performance_cb_simple

		dt = self.dt
		zz = [np.array(trial['z'][0,:]) for trial in self.trials]
		lens = [len(z) for z in zz]
		max_len = np.max(lens)

		z = np.zeros((max_len, self.ntrials, 1))

		for i in range(len(zz)):
			z[:lens[i], i, 0] = zz[i]
			
		return performance(self.trials, z)

	# ======================================================
	# Plots trials per condition - used to check RNN output.
	# ======================================================
	def plot_trials_cond(self, cond=11, nplot='all', f=None, savepath=None):
		
		dt 			= self.dt
		conds 		= self.conds
		idx_cond 	= np.where(np.array(conds) == cond)[0]
		
		if nplot == 'all':
			f_iter = range(len(idx_cond))
		else:
			f_iter = range(min(nplot, len(idx_cond)))

		if f is None:
			f = plt.figure()
		ax = f.gca()

		for i in f_iter:
			z = self.trials[idx_cond[i]]['z'][0,:]
			ax.plot(np.arange(len(z)) * dt, z)
			ax.axvline(self.trials[idx_cond[i]]['info']['check_drawn'], color='m')	

		if savepath is not None:
			f.savefig(savepath + 'output_c={}.pdf'.format(cond))

		return f

	# ==================================
	# Plots trials across all conditions
	# ==================================
	def plot_all_trials(self, savepath=None):
	
		uconds = np.unique(self.conds)

		for i in uconds:
			  self.plot_trials_cond(cond=i, savepath=savepath)
		      

	# ========================
	# Plots psychometric curve
	# ========================
	def psychometric(self, threshold=0, savepath=None):
		
		conds = self.conds
		correct_choices = np.array([trial['info']['choice'] for trial in self.trials])
		left_rights 	= np.array([trial['info']['left_right'] for trial in self.trials])
		
		if not threshold:
			decisions 		= np.array(np.sign([trial['z'][0, -(1 + trial['info']['post_delay']) // trial['info']['dt']] for trial in self.trials])).astype(int)
			trial_outcome 	= decisions == correct_choices
			choose_red 		= decisions == left_rights
		else:
			decisions 		= np.array([trial['z'][0, -(1 + trial['info']['post_delay']) // trial['info']['dt']] for trial in self.trials])
			trial_outcome 	= np.abs(decisions - correct_choices) < threshold
			choose_red 		= np.abs(decisions - left_rights) < threshold
		
		u_conds 			= np.unique(conds)
		success_rates 		= np.zeros_like(u_conds).astype(float)
		choose_red_rates 	= np.zeros_like(u_conds).astype(float)
		
		for i,cond in enumerate(u_conds):
			trialMask 			= np.where(conds == cond)[0]
			success_rates[i] 	= np.sum(correct_choices[trialMask] == decisions[trialMask]).astype(float) / len(trialMask)
			choose_red_rates[i] = np.sum(choose_red[trialMask]) / len(trialMask)
			
		f 	= plt.figure()
		ax 	= f.gca()

		ax.plot(2*(u_conds / 225) - 1, choose_red_rates, marker='.', markersize=20)
		ax.set_ylim((-0.05, 1.05))
		ax.set_xlim((-1.0, 1.0))
		ax.set_xlabel('Checkerboard coherence')
		ax.set_ylabel('Proportion of reaches to red target')
		ax.set_title('Psychometric function for RNN checkerboard task')

		if savepath is not None:
			f.savefig(savepath + 'psychometric.pdf')

		#return (2*(u_conds / 225) - 1, choose_red_rates, success_rates)
		return f

	# ===========================
	# Plots reaction time vs cond 
	# ===========================

	def reaction_time(self, savepath=None):
		
		conds 	= self.conds
		rts  	= self.rts
						
		u_conds 	= np.unique(conds)
		rts_cond 	= np.zeros_like(u_conds).astype(float)
		
		for i,cond in enumerate(u_conds):
			trialMask 	= np.where(conds == cond)[0]
			rts_cond[i] = np.nanmean(rts[trialMask])
			
		f 	= plt.figure()
		ax 	= f.gca()
		ax.plot(2*(u_conds / 225) - 1, rts_cond, marker='.', markersize=20)
		ax.set_xlim((-1.0, 1.0))
		ax.set_xlabel('Checkerboard coherence')
		ax.set_ylabel('Reaction time')
		ax.set_title('Reaction time for RNN checkerboard task')
			
		if savepath is not None:
			f.savefig(savepath + 'reaction_time.pdf')
		return f	

	# ===============================
	# Plots reaction time std vs cond 
	# ===============================
	def reaction_time_std(self, savepath=None):
		
		conds 	= self.conds
		rts 	= self.rts
			
		u_conds = np.unique(conds)
		rts_std = np.zeros_like(u_conds).astype(float)
		
		for i,cond in enumerate(u_conds):
			trialMask = np.where(conds == cond)[0]
			rts_std[i] = np.sqrt(np.nanvar(rts[trialMask]))
			
		f 	= plt.figure()
		ax 	= f.gca()

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
		
		conds 	= self.conds
		rts  	= self.rts
						
		u_conds = np.unique(conds)
		rts_cond = np.zeros_like(u_conds).astype(float)
		
		for i,cond in enumerate(u_conds):
			trialMask 	= np.where(conds == cond)[0]
			rts_cond 	= rts[trialMask]

			f 	= plt.figure()
			ax 	= f.gca()

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
	def plot_structure(self, savepath=None):
			
		f = self.rnn.plot_structure(sortby=None)  # f is in the figure class of pycog/figtools;
		if savepath is not None:
			f.save(savepath + 'structure')

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

	cond_color = (0, 0, 0, 1)		# cm returns (red, green, blue, alpha)

	for sortable, value in args:
	  	if sortable == 'cohs':
			cond_color 	= cm.RdYlGn(1 - (1 + coh_r(value)) / 2)
		elif sortable == 'rts':
			cond_color 	= cm.Blues(value / 1200)

	return cond_color

class PSTH(Trial):

	# ==========
	# Initialize
	# ==========

	def __init__(self, rnnfile, modelfile, num_trials=100, seed=1, rnnparams={}, sort=['dirs', 'cohs'], align='cb'):

		# self.sort is a LIST of what to sort by.  If you wanted to sort only by 'coh', then set sort = ['coh'].  This is important to allow sorting by multiple features.
		self.sort 		= sort
		self.align 		= align
		self.psths 		= None
		super(PSTH, self).__init__(rnnfile, modelfile, num_trials, seed, rnnparams) 

		# We only operate on successful trials
		self.filter_trials()
		self.define_align_times()

	# =====================================================================
	# Set the align time for each PSTH.  These are hard-coded and how we handle the alignment internally.
	# =====================================================================
	def define_align_times(self):

		if self.align == 'cb':
		  	start_time	= 500 // self.dt
			align_time 	= 1500 // self.dt
			stop_time 	= 3500 // self.dt
		elif self.align == 'mv':
			start_time 	= 1500 // self.dt
			align_time 	= 3500 // self.dt
			stop_time 	= 4500 // self.dt
		elif self.align == 'end':
			start_time 	= (2 * self.longest_trial() - 3000) // self.dt
			align_time 	= 2 * self.longest_trial() // self.dt
			stop_time	= align_time
		elif self.align == 'start':
			start_time	= 0
		 	align_time 	= 0
			stop_time 	= 3000 // self.dt
		else:
		  	print 'No valid align time.  Aligning to start.'
			start_time	= 0
		 	align_time 	= 0
			stop_time 	= 3000 // self.dt

		self.start_time	= start_time
		self.align_time = align_time
		self.stop_time 	= stop_time

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

		check_onsets	= self.cbs // self.dt
		trial_ends 		= self.Ts // self.dt
		check_plus_rts	= (self.cbs + self.rts) // self.dt

		if self.align == 'cb':
			start_idxs		= self.align_time - check_onsets
			stop_idxs 		= start_idxs + trial_ends + 1	
		
		elif self.align == 'mv':
			start_idxs 		= self.align_time - check_plus_rts
			stop_idxs 		= start_idxs + trial_ends + 1

		elif self.align == 'end':
			start_idxs 		= self.align_time - trial_ends - 1
			stop_idxs 		= self.align_time * np.ones_like(trial_ends)

		elif self.align == 'start':
			start_idxs 		= self.align_time * np.ones_like(trial_ends)
			stop_idxs 		= self.align_time + trial_ends + 1

		else: 
		  	print 'Not a valid align time, aligning to the start'
			start_idxs 		= self.align_time * np.ones_like(trial_ends)
			stop_idxs 		= self.align_time + trial_ends + 1

		return start_idxs, stop_idxs

	# ================================================
	# Sort trials according to a useful categorization
	# ================================================
	def sort_trials(self, threshold=1, rt_bin=200):
		# threshold is the number of trials needed in a condition to actually append it.
		# The function sorts by self.sort.  self.sort can be a list of how to sort, and will sort in that order.
		#   - Note, the newest version of sort_trials only sorts by giving you the indices of the desired condition.

		dt 			= self.dt
		conds		= []
		uconds 		= []

		# Get a list of conditions that we need to sort by.
		for i in np.arange(len(self.sort)):
			# sorting is a bit unique for rt
			if self.sort[i] == 'rts':
				min_rt	= np.floor(np.min(self.rts) / rt_bin) * rt_bin # rounded to rt_bin
				max_rt 	= np.ceil(np.max(self.rts) / rt_bin) * rt_bin # rounded to rt_bin 
				conds 	+= [np.floor(self.rts / rt_bin) * rt_bin] # put everything into its correct bin
				uconds	+= [np.arange(min_rt, max_rt, rt_bin)]		

			else:
				conds 		+= [self.__getattribute__(self.sort[i])]
				uconds 		+= [np.unique(conds[i])]

		zip_conds 		= np.array(zip(*conds))
		sorted_trials 	= []
		# iterates over all things to sort by.
		for cond in itertools.product(*uconds):
			one_cond 			= {}
			one_cond['sort'] 	= self.sort
			one_cond['cond'] 	= cond

			one_cond['idxs']	= np.where([np.all(zip_cond == cond) for zip_cond in zip_conds])[0]

			if len(one_cond['idxs']) > threshold:
				sorted_trials.append(one_cond)
			else:
			  	print '{} condition {} did not have at least {} trials, so it was discarded.'.format(str(self.sort), str(cond), str(threshold))

		# sorted_trials is a list, where each entry of the list is a dict containing the identifier 'cond' and 'trials'.
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

		dt 			= self.dt
		max_length 	= self.longest_trial()
		num_neurons	= self.trials[0][field].shape[0]
		ntrials		= len(sub_idxs)
		
		p_mtx = np.empty((ntrials, num_neurons, 2*max_length)) 
		p_mtx[:] = np.nan		# initialize to nans
	
		# insert trials into each psth
		start_idxs, stop_idxs	= self.calc_align_idxs()

		# subsample trials
		trials 		= self.trials[sub_idxs]
		start_idxs 	= start_idxs[sub_idxs]
		stop_idxs 	= stop_idxs[sub_idxs]

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

	def gen_psth(self, field='r', threshold=1, rt_bin=200):
		# A few things need to happen here.
		# First we need to sort the trials.
		# Then we need to calculate the psths for each collection of trials
		# Then we'll store this as self.psth.


		# sort the trials
		sorted_trials = self.sort_trials(threshold=threshold, rt_bin=rt_bin)	

		# now calculate the PSTHs for each condition
		psth_collection = [] 
		for i in np.arange(len(sorted_trials)):

			one_psth 			= {}
			one_psth['sort'] 	= sorted_trials[i]['sort']
			one_psth['cond'] 	= sorted_trials[i]['cond']
			one_psth['psth'] 	= self.calc_psth(sub_idxs=sorted_trials[i]['idxs'], field=field)
			one_psth['x_psth']	= self.calc_psth(sub_idxs=sorted_trials[i]['idxs'], field='x')
			one_psth['u_psth']	= self.calc_psth(sub_idxs=sorted_trials[i]['idxs'], field='u')


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

			f 	= plt.figure()
			ax	= f.gca()
			
			# Plot each PSTH.
			for j in np.arange(len(self.psths)):

				cond_color 	= plot_color(*zip(self.psths[j]['sort'], self.psths[j]['cond']))
				cond_ls		= plot_linestyle(*zip(self.psths[j]['sort'], self.psths[j]['cond']))

				ax.plot(np.arange(self.start_time, self.stop_time) * self.dt, self.psths[j]['psth'][i,:], color=cond_color, linestyle=cond_ls)
				ax.axvline(self.align_time * self.dt, color='b')
				ax.set_title('Neuron {}, sortables={}, align={}'.format(i, self.sort, self.align))

				if savepath is not None:
					f.savefig(savepath + 'neuron={}_sortables={}_align={}.pdf'.format(i, self.sort, self.align)) 

# ============
# PCA function
# ============

def pca(data):  # the dimensions are rows, the observations are columns
	# Performs PCA on what I'm used to (i.e., a transposed data matrix; I don't know why those statisticians changed things.)

	means 			= np.mean(data, axis=1)
	data_centered 	= (data.T - means).T
	evecs, evals, _	= np.linalg.svd(np.cov(data_centered))
	scores 			= np.dot(evecs.T, data_centered)
	
	return evecs, evals, scores, means


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

def tf_rnn_rdot_norm(rnn, x, inputs=np.array((0,0,0,0))):
	return 0.5 * T.dot(tf_rnn_rdot(rnn, x, inputs), tf_rnn_rdot(rnn, x, inputs))

def tf_rnn_rdot(rnn, x, inputs=np.array((0,0,0,0))):
	
	tau 	= rnn.p['tau']
	dt 		= rnn.p['dt']
	alpha 	= dt / tau
	
	# No noise here.
	xn = x + alpha*(-x + T.dot(rnn.Wrec, relu(x)) + rnn.brec + T.dot(rnn.Win, inputs))

	# calculate rdot
	rn	= relu(xn)
	r 	= relu(x)
	
	return (rn - r) / dt

def tf_rnn_xdot(rnn, x, inputs=np.array((0,0,0,0))):
	return (-x + T.dot(rnn.Wrec, relu(x)) + rnn.brec + T.dot(rnn.Win, inputs)) / rnn.p['tau']

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
		idxs	= []
		cohs 	= np.array([11, 45, 67, 78, 90, 101, 108, 117, 124, 135, 147, 158, 180, 214])
		r_cohs	= cohs[np.where(cohs > 225//2)[0]]
		g_cohs	= cohs[np.where(cohs < 225//2)[0]]

		sort_names	= np.array([f[0] for f in filters])
		where_coh	= np.where(sort_names == 'cohs')
		
		# iterate in reversed order so I can just remove these elements from the list after I'm done.
		for i in sorted(where_coh, reverse=True):
		  	# check if this is a string, if so, then we add to the filters
			if type(filters[i][1]) is str:
				add_filters = zip(itertools.repeat('cohs'), r_cohs) if filters[i][1] == 'r' else zip(itertools.repeat('cohs'), g_cohs) 
				filters 	= filters + add_filters
				del filters[i]

		for f in filters:
			# Do logical statement over each condition.  This is a messy list comprehension.  It works.
			acceptable_conds	= [np.any(np.all(np.array(cond) == f, axis=1)) for cond in all_conds]
			these_idxs 			= np.where(acceptable_conds)[0]

			idxs.append(these_idxs)

		# we do a union over the cohs and an intersection over the rest
		sort_names	= np.array([f[0] for f in filters])
		where_coh	= np.where(sort_names == 'cohs')
	
		if where_coh:
			idxs 		= np.array(idxs)
		  	coh_idxs	= np.hstack(idxs[where_coh])
			idxs 		= np.delete(idxs, where_coh)

			# now cast back to list... this is probalby not as clean as it could be
			idxs 		= idxs.tolist()
			idxs.append(coh_idxs)

		# Return the intersection
		return reduce(np.intersect1d, idxs)

class Dynamics(PSTH):


	# ==========
	# Initialize
	# ==========

	def __init__(self, rnnfile, modelfile, num_trials=100, seed=1, rnnparams={}, sort=['dirs', 'cohs'], align='cb', dims=np.array((0,1))):

		# Init as a PSTH and generate them
		super(Dynamics, self).__init__(rnnfile, modelfile, num_trials, seed, rnnparams, sort, align) 
		self.gen_psth()

		# Calculate the matrix to perform PCA on 
		data 	= np.hstack([one_psth['psth'] for one_psth in self.psths])
		data_x 	= np.hstack([one_psth['x_psth'] for one_psth in self.psths])

		# Assign all class variables, including the PCs.  This is automatically performed.
		self.PCs, self.evals, _, self.means = pca(data)
		self.x_PCs, _, _, self.x_means 		= pca(data_x)
		self.dims 							= np.array(dims)


		# compile necessary theano functions
		self.compile_theano_costs()

		# Threshold value of the objective function to arrive at a fixed point.  Modify this to change how lenient you are in finding fixed points.
		self.fp_threshold 					= 1e-8

		# How the space is to be partitioned for dynamics flow fields -- this assigns self.partition_points
		self.partition_space()
		self.partition_deltas()

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

		all_conds 	= np.array([zip(psth['sort'], psth['cond']) for psth in self.psths])
		idxs 		= find_conds(all_conds, filters)

		# Plot each PSTH.
		for j in idxs:

			# Project the data
			pc_scores	= np.dot(self.PCs[:, self.dims].T, (self.psths[j]['psth'].T - self.means).T)

			# Plot colors and line-styles
			cond_color 	= plot_color(*zip(self.psths[j]['sort'], self.psths[j]['cond'])) if cond_color is None else cond_color
			cond_ls		= plot_linestyle(*zip(self.psths[j]['sort'], self.psths[j]['cond'])) if cond_ls is None else cond_ls

			# Plot the PCs
			ax.plot(pc_scores[0,window], pc_scores[1,window], color=cond_color, linestyle=cond_ls)
			ax.set_title('PCs {}, sortables={}, align={}'.format(self.dims + 1, self.sort, self.align)) 

			#ax.plot(pc_scores[0, window[-1]], pc_scores[1, window[-1]], marker='o', markerfacecolor='y')

			# save it down
			if savepath is not None:
				f.savefig(savepath + 'PCs={}_sortables={}_align={}.pdf'.format(self.dims + 1, self.sort, self.align)) 

		return f

	# ======================================
	# Plots the principal components vs time
	# ======================================

	def plot_pcs_vs_time(self, savepath=None):

		for i in self.dims:
			f 	= plt.figure()
			ax 	= f.gca()

			for j in np.arange(len(self.psths)):

				# Project the data
				pc_scores	= np.dot(self.PCs[:, i].T, (self.psths[j]['psth'].T - self.means).T)

				# Plot colors and line-styles
				cond_color 	= plot_color(*zip(self.psths[j]['sort'], self.psths[j]['cond']))
				cond_ls		= plot_linestyle(*zip(self.psths[j]['sort'], self.psths[j]['cond']))

				# Plot the PCs across time
				ax.plot(np.arange(self.start_time, self.stop_time) * self.dt, pc_scores, color=cond_color, linestyle=cond_ls)
				ax.axvline(self.align_time * self.dt, color='b')
				ax.set_title('PCs {}, sortables={}, align={}'.format(i + 1, self.sort, self.align)) 

				# save it down
				if savepath is not None:
					f.savefig(savepath + 'PCs={}_sortables={}_align={}.pdf'.format(i + 1, self.sort, self.align)) 

		return f


	# ====================================================
	# Plot the eigenvalues, both individual and cumulative
	# ====================================================

	def plot_evals(self, savepath=None):
		
		# calculations
		var_per_dim	= self.evals / sum(self.evals)
		accum_var	= np.cumsum(var_per_dim)

		dim 		= np.where(accum_var > 0.9)[0][0]

		# First plot solo evals
		f 	= plt.figure()
		ax	= f.gca()

		ax.plot(var_per_dim, linewidth=0, marker='.')
		ax.set_title('Variance per dimension; dimensionality={} captures {} variance'.format(dim + 1, accum_var[dim]))

		# save it down
		if savepath is not None:
			f.savefig(savepath + 'evals_solo.pdf')

		# Next plot cumulative evals
		f 	= plt.figure()
		ax	= f.gca()

		ax.plot(accum_var, linewidth=0, marker='.')
		ax.set_title('Accumulated variance; dimensionality={} captures {} variance'.format(dim + 1, accum_var[dim]))

		# save it down
		if savepath is not None:
			f.savefig(savepath + 'evals_cum.pdf')

	# =================================
	# Helper functions for RNN dynamics
	# =================================

	def rnn_xdot(self, x, inputs=np.array((0,0,0,0))):
		return (-x + np.dot(self.rnn.Wrec, relu(x)) + self.rnn.brec + np.dot(self.rnn.Win, inputs)) / self.rnn.p['tau']

	def rnn_xdot_norm(self, x, inputs=np.array((0,0,0,0))):
		return 0.5 * np.dot(self.rnn_xdot(x, inputs), self.rnn_xdot(x, inputs))

	def rnn_rdot(self, x, inputs=np.array((0,0,0,0))):
		
		tau 	= self.rnn.p['tau']
		dt 		= self.rnn.p['dt']
		alpha 	= dt / tau
		
		# Calculate \dot{r}
		xn = x + dt * self.rnn_xdot(x, inputs)

		rn 	= relu(xn)
		r 	= relu(x)
		
		return (rn - r) / dt

	def rnn_rdot_norm(self, x, inputs=np.array((0,0,0,0))):
		return 0.5 * np.dot(self.rnn_rdot(x, inputs), self.rnn_rdot(x, inputs))
		
	def t_rnn_xdot(self, x, inputs=np.array((0,0,0,0))):
		return (-x + T.dot(self.rnn.Wrec, relu(x)) + self.rnn.brec + T.dot(self.rnn.Win, inputs)) / self.rnn.p['tau']

	def t_rnn_xdot_norm(self, x, inputs=np.array((0,0,0,0))):
		return 0.5 * T.dot(self.t_rnn_xdot(x, inputs), self.t_rnn_xdot(x, inputs))

	def t_rnn_rdot(self, x, inputs=np.array((0,0,0,0))):
		
		tau 	= self.rnn.p['tau']
		dt 		= self.rnn.p['dt']
		alpha 	= dt / tau
		
		# No noise here.
		xn = x + alpha*(-x + T.dot(self.rnn.Wrec, relu(x)) + rnn.brec + T.dot(self.rnn.Win, inputs))

		xn2 = x + dt * self.t_rnn_xdot(x, inputs)

		# calculate rdot
		rn	= relu(xn)
		r 	= relu(x)
		
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
					options={'disp': False}
					)
		else:
		  	assert False, 'Not a valid type to look for fixed points.'

		# res.x is the fixed point, res.fun is the function value, res.success is whether it was successful or not
		return res.x, res.fun, res.success

	# Compile theano functions
	def compile_theano_costs(self):
		tx 	= T.dvector('x')
		tin = T.dvector('in')
		tfr	= tf_rnn_rdot_norm(self.rnn, tx, tin)
		tfx	= tf_rnn_xdot_norm(self.rnn, tx, tin)

		jac_fr = T.grad(tfr, tx)
		jac_fx = T.grad(tfx, tx)

		Hr, updates_r	= theano.scan(lambda i, jac_fr, tx: T.grad(jac_fr[i], tx), sequences=T.arange(jac_fr.shape[0]), non_sequences=[jac_fr, tx])
		f 				= theano.function([tx, tin], Hr, updates=updates_r)

		self.compute_cost_r	= theano.function([tx, tin], tfr)
		self.compute_jac_r	= theano.function([tx, tin], jac_fr)
		self.compute_hess_r	= theano.function([tx, tin], Hr, updates=updates_r)

		Hx, updates_x 	= theano.scan(lambda i, jac_fx, tx: T.grad(jac_fx[i], tx), sequences=T.arange(jac_fx.shape[0]), non_sequences=[jac_fx, tx])
		f 				= theano.function([tx, tin], Hx, updates=updates_x)

		self.compute_cost_x	= theano.function([tx, tin], tfx)
		self.compute_jac_x	= theano.function([tx, tin], jac_fx)
		self.compute_hess_x	= theano.function([tx, tin], Hx, updates=updates_x)


	# Returns (fixed_points, values, stable)
	def sample_fixed_points(self, inputs=np.array((0,0,0,0)), sample=1000, type='x'):
		
		# Get the dts so we know how to sample adequately
		dt 		= self.rnn.p['dt']
		sample 	= sample // dt
		
		# Length of each psth.
		p_lens 	= [p['x_psth'].shape[1] for p in self.psths]
		
		# Fixed points
		r_fixs	= []	# location of the fixed point
		vals 	= []	# value at that location
		stable	= []	# is the fixed point stable?
		
		for i in np.arange(len(p_lens)):
			
			for j in np.arange(p_lens[i] // sample):
				
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
	def plot_fixed_points(self, inputs=np.array((0,0,0,0)), sample=1000, f=None, savepath=None, type='x', color=None):
		
		if f is None:
		  	f = self.plot_pcs()

		ax 	= f.gca() 

		r_fixs, vals, stable	= self.sample_fixed_points(inputs=inputs, sample=sample)
		idx_thresholds			= np.where(vals < self.fp_threshold)[0]

		if len(idx_thresholds) == 0:
		  	print 'Warning: no fixed points were acceptable under your threshold; the minimum value of the objective that was successful was: {}.  We will plot this minimum.'.format(np.min(vals))
			idx_thresholds 		= np.where(vals == np.min(vals))[0]

		for i in idx_thresholds:
		  	if color is None:
				fp_col	= 'b' if stable[i] else 'r'
			else:
			  	fp_col 	= color

			if type == 'r':
			  	fp_proj = np.dot(self.PCs[:, self.dims].T, (r_fixs[i] - self.means).T)
			elif type == 'x':
				fp_proj	= np.dot(self.PCs[:, self.dims].T, (relu(r_fixs[i]) - self.means).T)

			ax.plot(fp_proj[0], fp_proj[1], linewidth=0, marker='.', color=fp_col, markersize=10)

		# save it down
		if savepath is not None:
			f.savefig(savepath + 'fixed-points_inputs={}.pdf'.format(inputs))

		return f

	# partitions the space to sample dynamics
	def partition_space(self, x_sample=10, y_sample=10, xp_min=0, xp_max=0, yp_min=0, yp_max=0):
		
		# we should be sampling x, not r, because later on, we are sampling these points from the x-space.
		data		= np.hstack([one_psth['x_psth'] for one_psth in self.psths])
		scores 		= np.dot(self.x_PCs[:, self.dims].T, (data.T - self.x_means).T)

		x_bounds 	= (np.floor(min(scores[0,:])) - xp_min, np.ceil(max(scores[0,:])) + xp_max)
		y_bounds 	= (np.floor(min(scores[1,:])) - yp_min, np.ceil(max(scores[1,:])) + yp_max)

		x_pts 		= np.linspace(x_bounds[0], x_bounds[1], x_sample)
		y_pts 		= np.linspace(y_bounds[0], y_bounds[1], y_sample)

		self.partition_points	= (x_pts, y_pts)
		
		return (x_pts, y_pts)

	# partition for deltas to sample dynamics
	def partition_deltas(self, x_sample=15, y_sample=15, xmin=-5, xmax=5, ymin=-5, ymax=5):
		
		x_pts	= np.linspace(xmin, xmax, x_sample)
		y_pts 	= np.linspace(ymin, ymax, y_sample)

		self.delta_samples	= (x_pts, y_pts)

		return (x_pts, y_pts)

	def sample_local_dynamics(self, x0=None, x=None, y=None, inputs=np.array((0,0,0,0)), higher_dims=None):

		if higher_dims is None:
			s0 = np.array((x, y))
		else:
			s0 = np.hstack((x, y, higher_dims))
			
		# Now project the point back into higher-dimensional space.
		# (100 x 2) matrix times (2 x 1) vector
		if x0 is None:
			x0 = np.dot(self.x_PCs[:, np.arange(len(s0))], s0) + self.x_means
		
		# Let's first get what r is, the projected value of r will be returned
		r 	= relu(x0)
		
		# Now calculate derivative at this point.	
		rdot = self.rnn_rdot(x0, inputs)

		# Now project this back into the lower dimensional space
		# NOTE WE DON'T SUBTRACT THE MEANS AS THEY CANCEL OUT FOR X_{k+1} - X_k
		s 		= np.dot(self.PCs.T, r - self.means)
		sdot 	= np.dot(self.PCs.T, rdot)

		return s, sdot

	def plot_local_dynamics(self, inputs=np.array((0,0,0,0)), higher_dims=None, num_higher_dims=5, scale=1, f=None):

		if higher_dims is None:
		  	higher_dims	= self.get_higher_dims(inputs=inputs, num_higher_dims=num_higher_dims)

		# get the sample points
		x_pts, y_pts 	= self.partition_points
		
		# store values in the following array.
		xx 		= np.zeros((2, len(x_pts), len(y_pts)))
		ss 		= np.zeros((2, len(x_pts), len(y_pts)))
		sdots 	= np.zeros((2, len(x_pts), len(y_pts)))
		
		# Get all the derivatives at the sample points.
		for i in np.arange(len(x_pts)):
			for j in np.arange(len(y_pts)):
				this_s, this_sdot 	= self.sample_local_dynamics(x=x_pts[i], y=y_pts[j], inputs=inputs, higher_dims=higher_dims)
				sdots[:, i, j] 		= this_sdot[self.dims]
				ss[:, i, j] 		= this_s[self.dims]
				xx[:, i, j] 		= np.array((x_pts[i], y_pts[j]))
				
		# Now plot them.
		if f is None:
			f = plt.figure()
		ax 	= f.gca()	

		round_f	= 5
		rx_min	= np.min(ss[0]) // round_f * round_f
		rx_max	= np.max(ss[0] + round_f) // round_f * round_f
		ry_min	= np.min(ss[1]) // round_f * round_f
		ry_max 	= np.max(ss[1] + round_f) // round_f * round_f

		ax.set_xlim([rx_min, rx_max])
		ax.set_ylim([ry_min, ry_max])

		sdots	= sdots * scale 

		for i in np.arange(len(x_pts)):
			for j in np.arange(len(y_pts)):
				ax.arrow(ss[0, i, j], ss[1, i, j], sdots[0, i, j], sdots[1, i, j], color=np.array([0.5, 0.5, 0.5]), head_width=0.2, head_length=0.4)

		return f

	# Plots the local dynamics from deltas 
	def plot_local_dynamics_delta(self, inputs=np.array((0,0,0,0)), f=None, scale=1, force_alignment=None, t_sample=None, coh=None, reset_align=True, filters=None):
		# coh is for sampling get_higher_dims, it is the red coherence (NOT CB).

		x_deltas, y_deltas	= self.delta_samples

		# store values in the following array.
		ss 		= np.zeros((2, len(x_deltas), len(y_deltas)))
		sdots 	= np.zeros((2, len(x_deltas), len(y_deltas)))

		# Get the initial point.
		# Note, we may want to sample at a given coherence without mucking the inputs, so we allow the passing of a variable coh

		higher_dim_inputs	= inputs.astype('float64') 	# needed since coh is fractional
		if coh is not None:
		  	higher_dim_inputs[2] 	= +coh
			higher_dim_inputs[3] 	= -coh

		x0 		= self.get_higher_dims(inputs=higher_dim_inputs, return_psth=True, field='x', force_alignment=force_alignment, t_sample=t_sample, reset_align=reset_align, filters=filters)

		# Get all the derivatives at the sample points.
		for i in np.arange(len(x_deltas)):
			for j in np.arange(len(y_deltas)):
			  	x0_s				= x0 + np.dot(self.x_PCs[:,self.dims],  np.hstack((x_deltas[i], y_deltas[j]))) # no need to add self.x_means since this is a delta
				this_s, this_sdot 	= self.sample_local_dynamics(x0=x0_s, inputs=inputs)
				sdots[:, i, j] 		= this_sdot[self.dims]
				ss[:, i, j] 		= this_s[self.dims]
				
		# Now plot them.
		if f is None:
			f = plt.figure()
		ax 	= f.gca()

		round_f	= 1
		rx_min	= np.min(ss[0]) // round_f * round_f
		rx_max	= np.max(ss[0] + round_f) // round_f * round_f
		ry_min	= np.min(ss[1]) // round_f * round_f
		ry_max 	= np.max(ss[1] + round_f) // round_f * round_f

		ax.set_xlim([rx_min, rx_max])
		ax.set_ylim([ry_min, ry_max])

		sdots	= self.rnn.p['dt'] * sdots * scale # since dt = 10ms; this is actually how far the trajectory will go

		for i in np.arange(len(x_deltas)):
			for j in np.arange(len(y_deltas)):
				ax.arrow(ss[0, i, j], ss[1, i, j], sdots[0, i, j], sdots[1, i, j], color=np.array([0.5, 0.5, 0.5]), head_width=0.1, head_length=0.2)

		# FOR DEBUGGING

		#x0 		= self.get_higher_dims(inputs=higher_dim_inputs, return_psth=True, field='r', force_alignment=force_alignment, t_sample=t_sample, reset_align=reset_align)
		#px0 = np.dot(self.PCs[:, self.dims].T, x0 - self.means)
		#ax.plot(px0[0], px0[1], marker='x', markersize=20)

		return f

	# Extracts a PSTH given a filter.  This code was used so much I decided to make it a function
	def extract_psths(self, filters=None, field='r'):

		if field == 'r':
			psths = np.array([psth['psth'] for psth in self.psths])
		elif field == 'x':
			psths = np.array([psth['x_psth'] for psth in self.psths])
		elif field == 'u':
			psths = np.array([psth['u_psth'] for psth in self.psths])
		else:
		  	assert False, 'Invalid field.'

		all_conds 	= np.array([zip(psth['sort'], psth['cond']) for psth in self.psths])
		idxs 		= find_conds(all_conds, filters)

		return psths[idxs]

	# Gets the higher dimensions to plot; or returns the value of the PSTH at the desired time point
	def get_higher_dims(self, inputs=np.array((0,0,0,0)), num_higher_dims=5, return_psth=False, field='r', force_alignment=None, t_sample=0, reset_align=True, filters=None):
		# Returns the higher dims vector given an input.
		# The organization is as follows: 
		#   - Unless force alignment is given ('start', 'cb', or 'end'), we're going to align based off of the inputs.
		#	  - If inputs is all zeros, align to trial start and grab the first 100ms.
		# 	  - If the targets are on but the checkerboard is off, align to cb_on and take the 100ms preceding cb_on
		# 	  - If all inputs are on, align to trial end and grab the last 100ms.
		#  
		# 	- If return_psth is True, then it returns the PSTH at this time instead of the higher_dims
		# 	- If t_sample is not None, it's a bin (in dt, not in ms) that is sampled in the absolute psth.  It is ONLY used when you want to return_psth.  Otherwise, for get_higher_dims returning the PCs this isn't even used.
		#		NOTE: if the alignment is end, then t_post can't be greater than 0 or else it'll sample out of bounds.  We have thus made t_post subtract from the last element, so note its special usage in the case of align='end'
		# - reset_align set to false means the alignment will not be reset at the end.  this helps reduce the number of psths calculated during the movie, which can otherwise be beastly (two psth calcs for each frame... what a nightmare).

		prior_align	= self.align
		prior_sort 	= self.sort

		start_dim	= np.max(self.dims) + 1
		higher_dims = np.zeros((num_higher_dims))
		
		# Okay, so if force_alignment is none, then we're going to base it on the inputs
		align 	= force_alignment

		if align is None:
			if np.sum(np.abs(inputs)) == 0:
			  	align	= 'start'
			elif (inputs[0] != 0 or inputs[1] != 0) and inputs[2] == 0 and inputs[3] == 0:
				align 	= 'cb'
			else: 
			  	align 	= 'end'

		# This here is an align to 0.
		if align == 'start':

		  	# then the input is all zeros.
			if prior_align != align:
				self.set_align(align=align)

			# if reset_align is true, we don't want to have to go through the trouble of regenerating psths; please note that self.psths is assigned by self.gen_psth(), NOT self.calc_psth()
			if reset_align:
			  	if filters is None:
					this_psth 	= self.calc_psth(field=field) 
				else:
				  	# If filters is not None, then we need to filter by a condition and thus need to re-gen the PSTHs
					if prior_align != align:
						self.gen_psths()
					psths 		= self.extract_psths(filters=filters, field='r')
					this_psth 	= np.mean(psths[idxs], axis=0)
			else:
			  	if prior_align != align: 
					self.gen_psth(field=field)

				psths 		= self.extract_psths(filters=filters, field='r')
				this_psth	= np.mean(self.psths[idxs], axis=0)		# average across all conditions

			these_pcs	= np.dot(self.PCs.T, (this_psth.T - self.means).T)
			higher_dims	= np.mean(these_pcs[start_dim:start_dim+num_higher_dims, :100//self.dt], axis=1)

			if t_sample is None:
			  	t_sample = 0

			this_psth 	= this_psth[:, t_sample]

			if reset_align:
			  	# then we didn't gen_psths and we can safely reset the align.  This saves us the headache of having to reset the alignment and calculate another psth
				self.set_align(align=prior_align)

		elif align == 'cb':
			# then the targets are on but the checkerboard is not on.

			# set the alignments if necessary.  here we have to gen_psth() as we need the sort correct.
			if prior_align != align or prior_sort != ['dirs', 'cohs']:
				self.sort  = ['dirs', 'cohs']
			  	self.set_align(align=align)
				self.gen_psth()

			if filters is None:
				# Collapse across target input configurations.
				# Now assign target classifications; target=-1 means left is red (1,-1) and target=1 means right is red (-1,1)

				if inputs[0] == -1: # left target is red
					p0 		= self.extract_psths(filters=[('dirs', -1), ('cohs', 'r')], field='r')
					p1 		= self.extract_psths(filters=[('dirs', +1), ('cohs', 'g')], field='r')
				else: # left target is green
					p0 		= self.extract_psths(filters=[('dirs', -1), ('cohs', 'g')], field='r')
					p1 		= self.extract_psths(filters=[('dirs', +1), ('cohs', 'r')], field='r')

				psths 	= np.vstack((p0, p1))

			else:
			  	psths = self.extract_psths(filters=filters, field='r')

			this_psth	= np.mean(psths, axis=0)		# average across all conditions
			these_pcs	= np.dot(self.PCs.T, (this_psth.T - self.means).T)
			higher_dims	= np.mean(these_pcs[start_dim:start_dim+num_higher_dims, (self.align_time - self.start_time) - 100 // self.dt:(self.align_time - self.start_time)], axis=1)

			if t_sample is None:
			  	t_sample = self.align_time - self.start_time

			this_psth	= this_psth[:, t_sample]

		elif align == 'end':

			# We'll want to find the input condition in the sorted_trials
			# 	The inputs tell us which direction it was and coherence.
			if filters is None:
				this_dir 	= +1 if (inputs[0] == 1 and inputs[2] > inputs[3]) or (inputs[0] == -1 and inputs[3] > inputs[2]) else -1
				this_cb		= inv_coh_r(inputs[2])
			  	filters	= [('dirs', this_dir), ('cohs', this_cb)]

			# You could do a calc psth on just the subset, but this leads to some obfuscation in other tasks.  So we're going to work with gen_psth to be safe.  Sorry.  Refer to the initial git commit if you want to change this.  The code is there.
			if prior_align != align or prior_sort != ['dirs', 'cohs']:
				self.sort  = ['dirs', 'cohs']
			  	self.set_align(align=align)
				self.gen_psth()

			psths		= self.extract_psths(filters=filters, field='r')
			this_psth	= np.mean(psths, axis=0)

			these_pcs	= np.dot(self.PCs.T, (this_psth.T - self.means).T)
			higher_dims	= np.mean(these_pcs[start_dim:start_dim+num_higher_dims, -100 // self.dt:], axis=1)

			if t_sample is None:
			  	t_sample = -1

			this_psth 	= this_psth[:, t_sample]

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
		prior_sort	= None

		if self.align != 'cb':

		  	print 'Re-aligning the data to checkerboard'
		  	prior_align = self.align
			prior_sort	= self.sort

			self.sort 	= ['dirs', 'cohs']
			self.set_align(align='cb')

		psths = self.psths

		# converts frame no. to absolute time and vice versa
		frame_to_time	= (1000 / framerate) / self.dt
		time_to_frame	= self.dt / (1000 / framerate)

		# Find appropriate temporal landmarks
		if Ts is None:
			T 				= (self.stop_time - self.start_time) * self.dt
			T_start, T_end 	= (0, T // self.dt)
		else:			
			T_start, T_end 	= (Ts[0] // self.dt, Ts[1] // self.dt)

		f_start, f_end	= (T_start * time_to_frame, T_end * time_to_frame)
		frames 			= np.arange(int(f_start), int(f_end))

		# subidxs to filter -- yes, I know plot_pcs does this automatically but we still need it for the inputs.
		all_conds 	= np.array([zip(psth['sort'], psth['cond']) for psth in psths])
		idxs 		= find_conds(all_conds, filters)

		if len(idxs) > 1:
		  	# there are multiple indices.  we want the one where the input has the greatest coh.  we can do the following because we know how the psth is sorted, i.e., by self.sort = ['dirs', 'cohs'] and by self.align = cb
			cohs		= [coh_r(psths[idx]['cond'][1]) for idx in idxs]
			abs_cohs	= np.abs(cohs)

			which_idx	= np.where(abs_cohs - max(abs_cohs) == 0)[0][0]
			idx_input	= idxs[which_idx]
		else:
		  	idx_input 	= idxs[0]

		# Now iterate over the frames.

		for i in frames:

			f 	= plt.figure()

			# In the current implementation, u_psth is an average, and so the inputs turning on are not explicitly what was seen.  My proposed way to currently account for this is to round the input along the target dimension, or else input psths with exact timing (i.e., targets always come on at the same time).

			stop_frame	= (np.ceil(i * self.dt) // self.dt).astype(int)
			# Plot the PCs
			history 	= 0 if history is None else i - history
			t_start 	= int(max(0, history) * frame_to_time)
			t_stop 		= int(i * frame_to_time)
			f 			= self.plot_pcs(f=f, window=np.arange(t_start, t_stop+1), filters=filters, cond_color=cond_color, cond_ls=cond_ls)

			# Determine the input
			inputs 		= psths[idx_input]['u_psth'][:, t_stop]
			# moves the rounding threshold up to 0.7
			inputs[0]	= np.round(inputs[0] * 0.7 / 0.5)
			inputs[1]	= np.round(inputs[1] * 0.7 / 0.5)

			# Now, generate the plot_local_dynamics
			f 			= self.plot_local_dynamics_delta(f=f, inputs=inputs, scale=scale, reset_align=False, t_sample=t_stop, force_alignment='cb', filters=filters) # setting reset_align=False means to not regenerate the PSTHs every time in get_higher_dims

			# set the axes
			ax			= f.gca()
			ax.set_xlim(xlim)
			ax.set_ylim(ylim)

			# and save
			i_str 	= str(i)
			f.savefig(savepath + 'frame_' + i_str.zfill(5) + '.pdf')

			# Kill this figure now after having saved it.
			f.clf()
			plt.close(f)

		# Re-align data if necessary
		if prior_align is not None:
		  	self.sort	= ['dirs', 'cohs']
		  	self.set_align(align=prior_align)

	def hi():

		pass

