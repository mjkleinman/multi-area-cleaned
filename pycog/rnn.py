"""
Recurrent neural network for testing networks outside of Theano.

"""
from __future__ import absolute_import
from __future__ import division

import cPickle as pickle
import os
import shutil
import sys
import time
from   collections import OrderedDict

import pdb

import numpy as np

from .utils import print_settings
from .euler import euler, euler_no_Win
from cfg_mk import cfg_mk

THIS = 'pycog.rnn'

#=========================================================================================
# Activation functions
#=========================================================================================

def rectify(x):
    return x*(x > 0)

def rectify_power(x, n=2):
    return x**n*(x > 0)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def rtanh(x):
    return np.tanh(rectify(x))

def softmax(x):
    """
    Softmax function.

    x : 2D numpy.ndarray
        The outputs must be in the second dimension.

    """
    e = np.exp(x)
    return e/np.sum(e, axis=1, keepdims=True)

activation_functions = {
    'linear':        lambda x: x,
    'rectify':       rectify,
    'rectify_power': rectify_power,
    'sigmoid':       sigmoid,
    'tanh':          np.tanh,
    'rtanh':         rtanh,
    'softmax':       softmax
}

#=========================================================================================

class RNN(object):
    """
    Recurrent neural network.

    """
    defaults = {
        'threshold': 1e-4,
        'sigma0':    0
        }
    ou_defaults = {
        'N':                 100,
        'Nin':               0,
        'Nout':              0,
        'hidden_activation': 'linear',
        'output_activation': 'linear',
        'baseline_in':       0,
        'rectify_inputs':    False,
        'var_in':            0.01**2,
        'var_rec':           0.15**2,
        'dt':                0.5,
        'tau':               100,
        'mode':              'batch'
        }
    dtype = np.float32

    #/////////////////////////////////////////////////////////////////////////////////////

    @staticmethod
    def fill(p_, varlist, all=['Win', 'Wrec', 'Wout', 'brec', 'bout', 'x0']):
        """
        Fill `p_` with `None`s if some of the parameters are not trained.

        """
        p = list(p_)
        for var in all:
            if var not in varlist:
                p.insert(all.index(var), None)

        return p

    @staticmethod
    def spectral_radius(A):
        """
        Compute the spectral radius of a matrix.

        """
        return np.max(abs(np.linalg.eigvals(A)))

    @staticmethod
    def clip_weights(name, W, threshold):
        """
        Set small weights to zero.

        """
        W[np.where(abs(W) < threshold)] = 0

    #/////////////////////////////////////////////////////////////////////////////////////

    def __init__(self, savefile=None, rnnparams={}, verbose=True):
        """
        Initialize the RNN from a saved training file.

        Parameters
        ----------

        savefile:  str, optional
                   File name for trained network. If `None`, create a default network.

        rnnparams: dict, optional
                   These parameters will override those in savefile in `RNN.defaults`.

        verbose: bool, optional
                 If `True`, report information about the RNN.

        """
        self.verbose = verbose

        if savefile is not None:
            # Check that file exists
            if not os.path.isfile(savefile):
                print("[ {}.RNN ] File {} doesn't exist.".format(THIS, savefile))
                sys.exit(1)

            # Ensure we have a readable file
            base, ext = os.path.splitext(savefile)
            savefile_copy = base + '_copy' + ext
            while True:
                shutil.copyfile(savefile, savefile_copy)
                try:
                    with file(savefile_copy, 'rb') as f:
                        save = pickle.load(f)
                    break
                except EOFError:
                    wait = 5
                    print("[ {}.RNN ] Got an EOFError, trying again in {} seconds."
                          .format(THIS, wait))
                    time.sleep(wait)

            # Parameters
            self.p = save['params']

            # Get history
            self.costs_history = save['costs_history']
            self.Omega_history = save['Omega_history']

            # Get best info
            best        = save['best']
            best_i      = best['iter']
            best_error  = (best['other_costs'][0] if len(best['other_costs']) > 0
                           else np.inf)

            best_params = best['params']
            self.Win, self.Wrec, self.Wout, self.brec, self.bout, self.x0 = best_params
			# FOR DEBUGGING - MAY HAVE ALSO BEEN USED FOR CORA JCK 2018-05-01
            #best_params = save['current']
            #self.Win, self.Wrec, self.Wout, self.brec, self.x0 = best_params
            #self.bout = best['params'][4]

            print("[ {}.RNN ] {} updates, best error = {:.8f}, spectral radius = {:.8f}"
                  .format(THIS, best_i-1, best_error, RNN.spectral_radius(self.Wrec)))

            #-----------------------------------------------------------------------------
            # Fill in parameters
            #-----------------------------------------------------------------------------

            # Default
            for k in RNN.defaults:
                self.p.setdefault(k, RNN.defaults[k])

            # Override
            #pdb.set_trace()
            for k in rnnparams:
                self.p[k] = rnnparams[k]

            # Threshold weights
            threshold = self.p['threshold']
            if threshold > 0:
                if self.Win is not None:
                    RNN.clip_weights('Win', self.Win, threshold)
                RNN.clip_weights('Wrec', self.Wrec, threshold)
                RNN.clip_weights('Wout', self.Wout, threshold)
        else:
            # Parameters
            self.p = self.ou_defaults.copy()

            # Set history
            self.costs_history = None
            self.Omega_history = None

            #-----------------------------------------------------------------------------
            # Fill in parameters
            #-----------------------------------------------------------------------------

            # Default
            for k in RNN.defaults:
                self.p.setdefault(k, RNN.defaults[k])

            # Override
            for k in rnnparams:
                self.p[k] = rnnparams[k]

            # Get best info
            self.Win  = None
            self.Wrec = np.zeros((self.p['N'], self.p['N']), dtype=RNN.dtype)
            self.Wout = None
            self.brec = np.zeros(self.p['N'], dtype=RNN.dtype)
            self.bout = None
            self.x0   = 0.1*np.ones(self.p['N'], dtype=RNN.dtype)
            print("[ {}.RNN ] No savefile provided,"
                  " created independent Ornstein-Uhlenbeck processes.".format(THIS))

    #/////////////////////////////////////////////////////////////////////////////////////

    def run(self, T=None, inputs=None, rng=None, seed=1234):
        """
        Run the network.

        Parameters
        ----------

        T : float, optional
            Duration for which to run the network. If `None`, `inputs` must not be
            `None` so that the network can be run for the trial duration.

        inputs : (generate_trial, params), optional

        rng : numpy.random.RandomState
              Random number generator. If `None`, one will be created using seed.

        seed : int, optional
               Seed for the random number generator.

        """
        if self.verbose:
            config = OrderedDict()

            config['dt']        = '{} ms'.format(self.p['dt'])
            config['threshold'] = self.p['threshold']

            print_settings(config)

        # Random number generator
        if rng is None:
            rng = np.random.RandomState(seed)

        #---------------------------------------------------------------------------------
        # Setup
        #---------------------------------------------------------------------------------

        N           = self.p['N']
        Nin         = self.p['Nin']
        Nout        = self.p['Nout']
        baseline_in = self.p['baseline_in']
        var_in      = self.p['var_in']
        var_rec     = self.p['var_rec']
        dt          = self.p['dt']
        tau         = self.p['tau']
        tau_in      = self.p.get('tau_in', self.p['tau'])
        sigma0      = self.p['sigma0']
        mode        = self.p['mode']

        # Check dt
        # JCK -- 2016-12-14, commented this out because it ruins the carriage return.

#        if np.any(dt > tau/10):
#            print("[ {}.RNN.run ] Warning: dt seems a bit large.".format(THIS))

        # Float
        dtype = self.Wrec[0,0]

        #---------------------------------------------------------------------------------
        # External input
        #---------------------------------------------------------------------------------

        if inputs is None:
            if T is None:
                raise RuntimeError("[ {}.RNN.run ] Cannot determine the trial duration."
                                   .format(THIS))

            self.t = np.linspace(0, T, int(T/dt)+1).astype(dtype)
            if self.Win is not None:
                u = np.zeros((len(self.t), Nin), dtype=dtype)
            info = None
        else:
            generate_trial, params = inputs

            trial  = generate_trial(rng, dt, params)
            info   = trial['info']
            self.t = np.concatenate(([0], trial['t']))

            u = np.zeros((len(self.t), trial['inputs'].shape[1]), dtype=dtype)
            u[1:,:] = trial['inputs']

#            info['epochs'] = trial['epochs']

        Nt = len(self.t)

        #---------------------------------------------------------------------------------
        # Variables to record
        #---------------------------------------------------------------------------------

        if self.Win is not None:
            self.u = np.zeros((Nt, Nin), dtype=dtype)
        else:
            self.u = None
        self.r = np.zeros((Nt, N), dtype=dtype)
        self.z = np.zeros((Nt, Nout), dtype=dtype)

        # JCK -- for network analysis 2017-01-27
        self.x = np.zeros((Nt, N), dtype=dtype)

		# JCK -- for output error reconstruction
        if 'target_output' in params and params['target_output'] is True:
			# There's an initial condition, so it's required that we append an output at the beginning corresponding to that initial condition
            Nout = trial['outputs'].shape[1]
            self.out = np.concatenate((np.zeros((Nout, 1)), trial['outputs'].T), axis=1)

        #---------------------------------------------------------------------------------
        # Activation functions
        #---------------------------------------------------------------------------------

        f_hidden = activation_functions[self.p['hidden_activation']]
        f_output = activation_functions[self.p['output_activation']]

        #---------------------------------------------------------------------------------
        # Integrate
        #---------------------------------------------------------------------------------

        # Time step
        alpha = dt/tau

        # Input noise

        if self.Win is not None:
            var_in = 2*tau_in/dt*var_in
            if np.isscalar(var_in) or var_in.ndim == 1:
                if np.any(var_in > 0):
                    noise_in = np.sqrt(var_in)*rng.normal(size=(Nt, Nin))
                    pdb.set_trace()
                else:
                    noise_in = np.zeros((Nt, Nin))
            else:
                noise_in = rng.multivariate_normal(np.zeros(Nin), var_in, Nt)
                if cfg_mk['remove_check_noise_before_cue']: #mk 2022-10-20 added this to config
                    noise_in[:80, :] = 0 # mk 2020-05-03. Note, networks were trained with noise to checkerboard during this period, but this was helpful for visualizations, esp. for dpca IIRC.
            noise_in = np.asarray(noise_in, dtype=dtype)

        # Recurrent noise
        # MK modify to only add noise to paricular units 2020-03-24
        # TODO: put this in some config file

        if 'diff_area_noise' in self.p:
            cfg_mk['diff_area_noise'] = self.p['diff_area_noise']
        if 'rnn_areas' in self.p:
            cfg_mk['rnn_areas'] = self.p['rnn_areas']
        if 'area_noise' in self.p:
            cfg_mk['area_noise'] = self.p['area_noise']

        diff_area_noise = cfg_mk['diff_area_noise']
        num_areas = cfg_mk['rnn_areas']
        area_noise = cfg_mk['area_noise'] # here 1 refers to area 1

        noise_rec = np.zeros((Nt, N))
        var_rec = 2/dt*var_rec

        exc_units = N//num_areas * 4 // 5
        inh_units = N//num_areas // 5

        if np.isscalar(var_rec) or var_rec.ndim == 1:
            if np.any(var_rec > 0):
                if (diff_area_noise):
                    noise_rec[:, exc_units * (area_noise - 1): exc_units*area_noise] = np.sqrt(var_rec)*rng.normal(size=(Nt, exc_units))
                    noise_rec[:, exc_units * num_areas + inh_units * (area_noise - 1): exc_units * num_areas + inh_units *area_noise] = np.sqrt(var_rec)*rng.normal(size=(Nt, inh_units))
                else:
                    noise_rec = np.sqrt(var_rec)*rng.normal(size=(Nt, N))
            else:
                noise_rec = np.zeros((Nt, N))
        else:
            noise_rec = rng.multivariate_normal(np.zeros(N), var_rec, Nt)
        noise_rec = np.asarray(np.sqrt(tau)*noise_rec, dtype=dtype)

        # Inputs
        if self.Win is not None:
            self.u = baseline_in + u + noise_in
            if self.p['rectify_inputs']:
                self.u = rectify(self.u)
        #self.u[:Nt,0] += np.sqrt(1e-10)*rng.normal(size=(Nt))

        # Initial conditions
        if hasattr(self, 'x_last'):
            if self.verbose:
                print("[ {}.RNN.run ] Continuing from previous run.".format(THIS))
            x_t = self.x_last.copy()
        else:
            x_t = self.x0.copy()
            if sigma0 > 0:
                x_t += sigma0*rng.normal(size=N)

        #pdb.set_trace()
#	x_t = np.array([-0.08144971, -0.17056126, -0.25683784, -0.04681791,  0.07596147,
#        0.04971324,  0.00821347,  0.11330351,  0.06657394, -0.08502018,
#       -0.01189554, -0.03193191, -0.06827619,  0.10935645, -0.05368976,
#        0.00148713, -0.02327032,  0.10122895, -0.02474604,  0.00526586,
#       -0.02944342,  0.08853634,  0.1010365 ,  0.04035783,  0.03544187,
#       -0.07043597,  0.01546125, -0.1412554 , -0.08804635, -0.11562942,
#       -0.10791912, -0.17746144,  0.06539391, -0.06688865,  0.0681729 ,
#       -0.17895376, -0.05990028,  0.21777974, -0.02909925, -0.05030206,
#        0.01863475,  0.12384405, -0.03512619,  0.1933993 , -0.03599336,
#       -0.02572995, -0.0039354 , -0.0086568 ,  0.07137503, -0.06441156,
#       -0.08006469,  0.00735269, -0.01259285,  0.03554103, -0.08556031,
#       -0.12842674,  0.02562003,  0.17551309,  0.11067529, -0.07405744,
#       -0.00205803, -0.09055711, -0.00443059,  0.0293319 ,  0.10702029,
#        0.10144905, -0.02773811,  0.06598996, -0.2483846 ,  0.26916387,
#       -0.14618942, -0.04570135, -0.03610583,  0.02571536, -0.00566434,
#       -0.16034591, -0.08943598,  0.08288947,  0.0303634 ,  0.05932577,
#       -0.08285363,  0.10615822,  0.06774107,  0.17477351,  0.01134081,
#        0.00612781,  0.06252678,  0.19452012,  0.07755495, -0.07339723,
#        0.00154659, -0.03806943,  0.08694982,  0.05089893, -0.2474439 ,
#       -0.08687257, -0.0287397 ,  0.03783343, -0.06448404,  0.09663936,
#       -0.03288912,  0.05549354, -0.08636012, -0.00272781,  0.0487055 ,
#       -0.15010765, -0.2473322 ,  0.13905905,  0.02180042, -0.06523814,
#        0.10753094, -0.12097017, -0.08513412, -0.00598089,  0.02748669,
#       -0.10151708, -0.12666719,  0.0095583 ,  0.13036959,  0.11900753,
#        0.01323749,  0.08784583,  0.05322224,  0.0708562 ,  0.11241318,
#        0.09178509, -0.09720641, -0.12062301, -0.08701727,  0.17485963,
#        0.01010327, -0.05911377, -0.11005395, -0.15177225,  0.04379432,
#       -0.08225442, -0.04543346,  0.07972101, -0.0332962 ,  0.01409011,
#       -0.03877443,  0.06006263,  0.02462311,  0.06107618, -0.208375  ,
#       -0.07532265, -0.15646659, -0.14516687, -0.01653849,  0.02282259,
#       -0.17458015, -0.02669471,  0.00889386,  0.04014809,  0.02712117,
#        0.09935868, -0.01443477, -0.01415407,  0.11976947, -0.0865208 ,
#       -0.05266353, -0.0605637 , -0.11946402, -0.11132394,  0.15545962,
#       -0.17110284,  0.04429939, -0.0296939 , -0.07453794,  0.02374605,
#        0.06662996,  0.06214068,  0.08210861,  0.02881965,  0.13788028,
#       -0.24280633, -0.2365477 , -0.0103526 ,  0.01145411, -0.00225056,
#        0.04792098,  0.13616617, -0.09829238, -0.11814927, -0.14260152,
#       -0.13924781,  0.01858485, -0.03570364, -0.08597969, -0.07341484,
#       -0.16681118,  0.27848943, -0.21106395,  0.12359748, -0.22558398,
#       -0.00031071, -0.00953683, -0.03128487,  0.05367166, -0.17968838])
        x_t = np.zeros_like(x_t);
        #pdb.set_trace()
        r_t = f_hidden(x_t)

        # Record initial conditions
        self.r[0] = r_t
        self.x[0] = x_t # JCK 2017-01-27
        #pdb.set_trace()
        # Integrate
        if np.isscalar(alpha):
            alpha = alpha*np.ones(N, dtype=dtype)
        if self.Win is not None:
            euler(alpha, x_t, r_t, self.Win, self.Wrec, self.brec, self.bout,
                  self.u, noise_rec, f_hidden, self.r, self.x) # JCK 2017-01-27
        else:
            euler_no_Win(alpha, x_t, r_t, self.Wrec, self.brec, self.bout,
                         noise_rec, f_hidden, self.r, self.x) # JCK 2017-01-27
        if self.Wout is not None:
            self.z = f_output(self.r.dot(self.Wout.T) + self.bout)
        else:
            self.z = self.r

        # Transpose so first dimension is units
        if self.u is not None:
            self.u = self.u.T
        self.r = self.r.T
        self.z = self.z.T
        self.x = self.x.T # JCK 2017-01-27
        #pdb.set_trace()
        # In continuous mode start from here the next time
        if mode == 'continuous':
            self.x_last = x_t

        #---------------------------------------------------------------------------------

        return info

    #/////////////////////////////////////////////////////////////////////////////////////

    def plot_costs(self):
        """
        Plot the evolution of the cost functions.

        """
        try:
            from .figtools import Figure
        except ImportError:
            print("[ {}.RNN.plot_costs ] Couldn't import pycog.figtools.".format(THIS))
            return None

        fig = Figure(w=6.5, h=3, axislabelsize=7.5, labelpadx=5, labelpady=7.5,
                     thickness=0.6, ticksize=3, ticklabelsize=6.5, ticklabelpad=2)

        w = 0.38
        h = 0.76
        plots = {'L': fig.add([0.1,  0.17, w, h]),
                 'R': fig.add([0.57, 0.17, w, h])}

        #---------------------------------------------------------------------------------
        # Loss
        #---------------------------------------------------------------------------------

        plot = plots['L']

        all = []

        # Loss
        ntrials = [int(costs[0]) for costs in self.costs_history]
        ntrials = np.asarray(ntrials, dtype=int)//int(ntrials[1]-ntrials[0])
        cost    = [costs[1][0] for costs in self.costs_history]
        plot.plot(ntrials, cost, color='0.2', lw=1, label='Cost')
        all.append(cost)

        # Regularizer
        ntrials = [int(costs[0]) for costs in self.Omega_history]
        ntrials = np.asarray(ntrials, dtype=int)//int(ntrials[1]-ntrials[0])
        cost = [costs[1] for costs in self.Omega_history]
        plot.plot(ntrials, cost, color=Figure.colors('red'), lw=1, label='Reg. term')
        all.append(cost)

        plot.xlim(0, ntrials[-1])
        plot.lim('y', all, lower=0)

        plot.xlabel(r'Number of trials ($\times 10^4$)')
        plot.ylabel('Objective function terms')

        # Legend
        props = {'prop': {'size': 7}}
        plot.legend(bbox_to_anchor=(1, 1), **props)

        #---------------------------------------------------------------------------------
        # Error
        #---------------------------------------------------------------------------------

        plot = plots['R']

        ntrials = [int(costs[0]) for costs in self.costs_history]
        ntrials = np.asarray(ntrials, dtype=int)//int(ntrials[1]-ntrials[0])
        cost = [costs[1][1] for costs in self.costs_history]

        plot.plot(ntrials, cost, color='0.2', lw=1)

        plot.xlim(ntrials[0], ntrials[-1])
        plot.lim('y', cost, lower=0)

        plot.ylabel('Error')

        #---------------------------------------------------------------------------------

        return fig

    #/////////////////////////////////////////////////////////////////////////////////////

    @staticmethod
    def plot_connection_matrix(plot, W, smap_exc=None, smap_inh=None, labelsize=5):
        """
        Plot a connection matrix, using separate colors for excitatory and inhibitory
        units.

        plot : pycog.figtools.Subplot
        W    : 2D numpy.ndarray
        smap_exc, smap_inh : matplotlib.cm.ScalarMappable

        """
        try:
            from .figtools import Figure, gradient, mpl
        except ImportError:
            print("[ {}.RNN.plot_costs ] Couldn't import pycog.figtools.".format(THIS))
            sys.exit(1)

        #---------------------------------------------------------------------------------
        # Format axes
        #---------------------------------------------------------------------------------

        plot.set_thickness(0.2)
        plot.set_tick_params(0, labelsize, 0)

        plot.xaxis.tick_top()
        plot.yaxis.tick_right()

        plot.xticks()
        plot.yticks()

        #---------------------------------------------------------------------------------
        # Display connection matrix
        #---------------------------------------------------------------------------------

        white = 'w'
        blue  = Figure.colors('strongblue')
        red   = Figure.colors('strongred')

        # Create colormaps if necessary
        if smap_exc is None:
            exc      = np.ravel(W[np.where(W > 0)])
            cmap_exc = gradient(white, blue)
            norm_exc = mpl.colors.Normalize(vmin=0, vmax=np.max(exc))
            smap_exc = mpl.cm.ScalarMappable(norm_exc, cmap_exc)
        if smap_inh is None:
            inh = -np.ravel(W[np.where(W < 0)])
            if len(inh) > 0:
                cmap_inh = gradient(white, red)
                norm_inh = mpl.colors.Normalize(vmin=0, vmax=np.max(inh))
                smap_inh = mpl.cm.ScalarMappable(norm_inh, cmap_inh)

        if W.ndim == 1:
            W = W[:,np.newaxis]

        im = np.ones(W.shape + (3,))
        for i in xrange(W.shape[0]):
            for j in xrange(W.shape[1]):
                if W[i,j] > 0:
                    im[i,j] = smap_exc.to_rgba(W[i,j])[:3]
                elif W[i,j] < 0:
                    im[i,j] = smap_inh.to_rgba(-W[i,j])[:3]

        plot.imshow(im, interpolation='nearest', aspect='auto')
        # pdb.set_trace()
        # plot.imshow(W, interpolation='nearest', aspect='auto')


    #/////////////////////////////////////////////////////////////////////////////////////

    def plot_structure(self, sortby=None):
        """
        Create a summary figure for the network's structure.

        """
        try:
            from .figtools import Figure, gradient, mpl
        except ImportError:
            print("[ {}.RNN.plot_costs ] Couldn't import pycog.figtools.".format(THIS))
            sys.exit(1)

        #---------------------------------------------------------------------------------
        # Sort the units in a particular way?
        #---------------------------------------------------------------------------------

        if sortby is None:
            order = range(self.p['N'])
        else:
            order = np.loadtxt(sortby, dtype=int)
            print("[ {}.RNN.plot_structure ] Sorting units according to {}"
                  .format(THIS, sortby))

        #---------------------------------------------------------------------------------
        # Figure
        #---------------------------------------------------------------------------------

        w   = 6.5
        h   = 4.5
        r   = w/h
        fig = Figure(w=w, h=h, axislabelsize=6.5, labelpadx=5, labelpady=5.5,
                     thickness=0.6, ticksize=3, ticklabelsize=6.5, ticklabelpad=2)

        h_cbar = 0.15

        w_rec = 0.52
        h_rec = r*w_rec

        w_in  = 0.05
        h_in  = h_rec

        w_out = w_rec
        h_out = r*w_in

        hspace = 0.02
        vspace = r*hspace

        x0 = 0.07
        y0 = 0.14

        plots = {
            'Win':       fig.add([x0+w_rec+hspace, y0,               w_in,  h_in],  None),
            'Wrec':      fig.add([x0,              y0,               w_rec, h_rec], None),
            'Wout':      fig.add([x0,              y0-vspace-h_out,  w_out, h_out], None),
            'brec':      fig.add([x0-0.025-hspace, y0,               0.025, h_rec], None),
            'bout':      fig.add([x0-0.025-hspace, y0-vspace-h_out,  0.025, h_out], None),
            'Win_dist':  fig.add([0.78,            0.73,             0.2,   0.21]),
            'Wrec_dist': fig.add([0.78,            0.41,             0.2,   0.21]),
            'Wout_dist': fig.add([0.78,            0.11,             0.2,   0.21]),
            'cbar_exc':  fig.add([0.08+0.52+0.02+0.05+0.03, 0.51, 0.02, h_cbar],
                                 'none'),
            'cbar_inh':  fig.add([0.08+0.52+0.02+0.05+0.03, 0.51-h_cbar, 0.02, h_cbar],
                                 'none')
            }

        #---------------------------------------------------------------------------------
        # Group labels
        #---------------------------------------------------------------------------------

        groups = self.p['structure'].get('groups')
        if groups is not None:
            for group, (idx, color) in groups.items():
                color = Figure.colors(color)
                lw    = 2

                #-------------------------------------------------------------------------
                # Wrec
                #-------------------------------------------------------------------------

                plot = plots['Wrec']

                units = (np.array([idx[0], idx[-1]]) + 0.5)/self.p['N']
                plot.plot(units, 1.04*np.ones(2), lw=lw, color=color,
                          transform=plot.transAxes)

                if not group.startswith('_'):
                    name = group + ' '
                else:
                    name = ''
                name += '({})'.format(len(idx))
                plot.text(np.mean(units), 1.06, name,
                          ha='center', va='bottom',
                          fontsize=6, color=color, transform=plot.transAxes)

                #-------------------------------------------------------------------------
                # Win
                #-------------------------------------------------------------------------

                plot = plots['Win']

                if self.Win is not None:
                    plot.plot(1.3*np.ones(2), 1-units, lw=lw, color=color,
                              transform=plot.transAxes)

        #---------------------------------------------------------------------------------
        # Determine range of weights
        #---------------------------------------------------------------------------------

        exc = []
        inh = []

        if self.Win is not None:
            W = self.Win
            exc.append( np.ravel(W[np.where(W > 0)]))
            inh.append(-np.ravel(W[np.where(W < 0)]))

        W = self.Wrec
        exc.append( np.ravel(W[np.where(W > 0)]))
        inh.append(-np.ravel(W[np.where(W < 0)]))

        W = self.Wout
        exc.append( np.ravel(W[np.where(W > 0)]))
        inh.append(-np.ravel(W[np.where(W < 0)]))

        exc = np.concatenate(exc)
        inh = np.concatenate(inh)

        K   = len(exc)//10
        exc = np.sort(exc)[K:-K]

        K   = len(inh)//10
        inh = np.sort(inh)[K:-K]

        #---------------------------------------------------------------------------------
        # Create color map for weights
        #---------------------------------------------------------------------------------

        white = 'w'
        blue  = Figure.colors('strongblue')
        red   = Figure.colors('strongred')

        cmap_exc = gradient(white, blue)
        # norm_exc = mpl.colors.Normalize(vmin=0, vmax=np.round(exc[-1], 1), clip=True)
        norm_exc = mpl.colors.Normalize(vmin=0, vmax=exc[-1], clip=True)

        smap_exc = mpl.cm.ScalarMappable(norm_exc, cmap_exc)

        cmap_inh = gradient(white, red)
        # norm_inh = mpl.colors.Normalize(vmin=0, vmax=np.round(inh[-1], 1), clip=True)
        norm_inh = mpl.colors.Normalize(vmin=0, vmax=inh[-1], clip=True)

        smap_inh = mpl.cm.ScalarMappable(norm_inh, cmap_inh)

        cmap_inh_r = gradient(red, white)
        norm_inh_r = mpl.colors.Normalize(vmin=-np.round(inh[-1], 1), vmax=0, clip=True)
        smap_inh_r = mpl.cm.ScalarMappable(norm_inh_r, cmap_inh_r)

        w = 0.25

        plot = plots['cbar_exc']
        cbar = mpl.colorbar.ColorbarBase(plot.ax, cmap=cmap_exc, norm=norm_exc,
                                         orientation='vertical')
        cbar.outline.set_linewidth(w)
        plot.yaxis.set_tick_params(width=0.25, size=2, labelsize=6, pad=0)
        cbar.set_ticks(smap_exc.get_clim())
        cbar.set_ticklabels(smap_exc.get_clim())

        plot = plots['cbar_inh']
        cbar = mpl.colorbar.ColorbarBase(plot.ax, cmap=cmap_inh_r, norm=norm_inh_r,
                                         orientation='vertical')
        cbar.outline.set_linewidth(w)
        plot.yaxis.set_tick_params(width=0.25, size=2, labelsize=6, pad=0)
        cbar.set_ticks(smap_inh_r.get_clim()[:1])
        cbar.set_ticklabels(smap_inh_r.get_clim()[:1])

        #---------------------------------------------------------------------------------
        # Win
        #---------------------------------------------------------------------------------

        if self.Win is not None:
            Win = self.Win
        else:
            Win = np.zeros((self.p['N'], 1))

        plot = plots['Win']

        RNN.plot_connection_matrix(plot, self.Win[order,:], smap_exc, smap_inh)

        # Input labels
        inputs = self.p['structure'].get('inputs')
        if inputs is not None:
            plot.xticks(np.arange(len(inputs)))
            if len(inputs) <= 3:
                fontsize = 5
            else:
                fontsize = 3.5
            plot.xticklabels(inputs, rotation='vertical', fontsize=fontsize)

        #---------------------------------------------------------------------------------
        # Wrec
        #---------------------------------------------------------------------------------

        plot = plots['Wrec']

        Wrec = self.Wrec[order,:][:,order]
        RNN.plot_connection_matrix(plot, Wrec, smap_exc, smap_inh)

        #---------------------------------------------------------------------------------
        # Wout
        #---------------------------------------------------------------------------------

        plot = plots['Wout']

        RNN.plot_connection_matrix(plot, self.Wout[:,order], smap_exc, smap_inh)

        # Output labels
        outputs = self.p['structure'].get('outputs')
        if outputs is not None:
            plot.yticks(np.arange(len(outputs)))
            plot.yticklabels(outputs, fontsize=6)

        #---------------------------------------------------------------------------------
        # Biases
        #---------------------------------------------------------------------------------

        plot = plots['brec']
        RNN.plot_connection_matrix(plot, self.brec[order], smap_exc, smap_inh)
        plot.xticks([0])
        plot.xticklabels(['Bias'])

        plot = plots['bout']
        RNN.plot_connection_matrix(plot, self.bout, smap_exc, smap_inh)

        #---------------------------------------------------------------------------------
        # Distribution of weights in Win
        #---------------------------------------------------------------------------------

        if self.Win is not None:
            plot = plots['Win_dist']

            # Label
            plot.text_upper_right('Win', #r'$\boldsymbol{W}_\text{\textbf{in}}$',
                                  fontsize=7, dy=-0.02)

            W       = self.Win
            Wexc    = W[np.where(W > 0)]
            pdf_exc = plot.hist(Wexc, color=Figure.colors('blue'))

            plot.lim('x', Wexc, lower=0)
            plot.lim('y', pdf_exc, lower=0)

            # x-ticks
            xmin, xmax = plot.get_xlim()
            step = max(1, int(2*xmax/3))
            plot.xticks(np.arange(0, xmax, 0.5)[::step])

            # y-ticks
            ymin, ymax = plot.get_ylim()
            step = max(1, int(ymax/3))
            plot.yticks(np.arange(0, ymax)[::step])

        #---------------------------------------------------------------------------------
        # Distribution of weights in Wrec
        #---------------------------------------------------------------------------------

        plot = plots['Wrec_dist']

        # Label
        plot.text_upper_right('W_rec', #r'$\boldsymbol{W}_\text{\textbf{rec}}$',
                              fontsize=7, dy=-0.02)

        W = self.Wrec

        # Excitatory weights
        Wexc    = W[np.where(W > 0)]
        pdf_exc = plot.hist(Wexc, color=Figure.colors('blue'))

        # Inhibitory weights
        Winh    = W[np.where(W < 0)]
        pdf_inh = plot.hist(Winh, color=Figure.colors('red'))

        plot.lim('x', [min(Winh), max(Wexc)])
        plot.lim('y', np.concatenate((pdf_exc, pdf_inh)), lower=0)

        # x-ticks
        xmin, xmax = plot.get_xlim()
        xneg = np.arange(0, -xmin, 0.5)[1:]
        xpos = np.arange(0, xmax, 0.5)[1:]
        plot.xticks(np.concatenate((-xneg, [0], xpos)))

        # y-ticks
        ymin, ymax = plot.get_ylim()
        step = max(1, int(ymax/3))
        plot.yticks(np.arange(0, ymax)[::step])

        # Summary
        W    = self.Wrec
        Wexc =  W[np.where(W > 0)]
        Winh = -W[np.where(W < 0)]

        N = self.p['N']
        #s = '$p$' + ' = {:.2f}\%'.format((len(Wexc) + len(Winh))/N**2*100)

        # Separate connection densities for E and I units.
        if self.p['ei'] is not None:
            #plot.text(0.97, 0.97-0.15, s, color='k',
            #          fontsize=6, ha='right', va='top', transform=plot.transAxes)

            Ne = len(np.where(self.p['ei'] > 0)[0])
            Ni = len(np.where(self.p['ei'] < 0)[0])

            #s = r'$p_\mathrm{E}$' + ' = {:.2f}\%'.format(len(Wexc)/(N*Ne)*100)
            #plot.text(0.97, 0.97-0.15-0.125, s, color=Figure.colors('blue'),
            #          fontsize=6, ha='right', va='top', transform=plot.transAxes)

            #s = r'$p_\mathrm{I}$' + ' = {:.2f}\%'.format(len(Winh)/(N*Ni)*100)
            #plot.text(0.97, 0.97-0.15-0.24, s, color=Figure.colors('red'),
            #          fontsize=6, ha='right', va='top', transform=plot.transAxes)
        else:
            pass
            #plot.text(0.97, 0.97-0.15, s, color='k',
            #          fontsize=6, ha='right', va='top', transform=plot.transAxes)

            #s = r'$p_+$' + ' = {:.2f}\%'.format(len(Wexc)/(N**2)*100)
            #plot.text(0.97, 0.97-0.15-0.125, s, color=Figure.colors('blue'),
            #          fontsize=6, ha='right', va='top', transform=plot.transAxes)

            #s = r'$p_-$' + ' = {:.2f}\%'.format(len(Winh)/(N**2)*100)
            #plot.text(0.97, 0.97-0.15-0.24, s, color=Figure.colors('red'),
            #          fontsize=6, ha='right', va='top', transform=plot.transAxes)

        #---------------------------------------------------------------------------------
        # Distribution of weights in Wout
        #---------------------------------------------------------------------------------

        plot = plots['Wout_dist']

        # Label
        plot.text_upper_right('W_out', #r'$\boldsymbol{W}_\text{\textbf{out}}$',
                              fontsize=7, dy=-0.02)

        W = self.Wout

        Wexc    = W[np.where(W > 0)]
        pdf_exc = plot.hist(Wexc, color=Figure.colors('blue'))

        # In case we're only reading out from excitatory units.
        Winh = W[np.where(W < 0)]
        if len(Winh) > 0:
            pdf_inh = plot.hist(Winh, color=Figure.colors('red'))
            plot.lim('x', [min(Winh), max(Wexc)])

            # x-ticks
            xmin, xmax = plot.get_xlim()
            step = max(1, int(-5*xmin/3))
            xneg = np.arange(0, -xmin, 0.2)[::step]
            step = max(1, int(5*xmax/3))
            xpos = np.arange(0, xmax, 0.2)[::step]
            plot.xticks(np.concatenate((-xneg[1:], xpos)))
        else:
            plot.lim('x', Wexc, lower=0)

            # x-ticks
            xmin, xmax = plot.get_xlim()
            step = max(1, int(5*xmax/3))
            plot.xticks(np.arange(0, xmax, 0.2)[::step])

        # y-ticks
        ymin, ymax = plot.lim('y', np.concatenate((pdf_exc, pdf_inh)), lower=0)
        step = max(1, int(ymax/3))
        plot.yticks(np.arange(0, ymax)[::step])

        #---------------------------------------------------------------------------------
        # Axis labels
        #---------------------------------------------------------------------------------

        plots['Wout_dist'].xlabel('$W$')

        #---------------------------------------------------------------------------------

        return fig
