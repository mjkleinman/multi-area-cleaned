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

from .trial import Trial
from .		import tasktools

THIS = 'pycog.psth'

class PSTH(object):

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


