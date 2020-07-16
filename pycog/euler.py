from __future__ import division

import numpy as np
import pdb
#import matplotlib.pyplot as plt

def euler(alpha, x_t, r_t, Win, Wrec, brec, bout, u, noise_rec, f_hidden, r, x):

  # r.shape[0] is time, btw.
  #u[0,2] = 1
  # noise_rec = np.zeros_like(noise_rec)

  for i in xrange(1, r.shape[0]):

    # x_{t+1} - x_t = dt/tau (-x_t + Wrec*f(x_t) + brec + Win*u_t + noise)
    x_t += alpha * (-x_t            # Leak
                    + Wrec.dot(r_t)  # Recurrent input
                    + brec          # Bias
                    + Win.dot(u[i])  # Input
                    + noise_rec[i])  # Recurrent noise
    r_t = f_hidden(x_t)
   # norms = (np.linalg.norm(-x_t), np.linalg.norm(Wrec.dot(r_t)), np.linalg.norm(brec), np.linalg.norm(Win.dot(u[i])), np.linalg.norm(noise_rec[i]))
   # pdb.set_trace()
    r[i] = r_t
    x[i] = x_t

#   x_t = ((1 - alpha)*x_tm1
#                       + alpha*(T.dot(r_tm1, WrecT)        # Recurrent
#                                + brec                     # Bias
#                                + T.dot(u_t[:,:Nin], WinT) # Input
#                                + u_t[:,Nin:])             # Recurrent noise
#                       )


def euler_no_Win(alpha, x_t, r_t, Wrec, brec, bout, noise_rec, f_hidden, r, x):
  for i in xrange(1, r.shape[0]):
    x_t += alpha * (-x_t            # Leak
                    + Wrec.dot(r_t)  # Recurrent input
                    + brec          # Bias
                    + noise_rec[i])  # Recurrent noise
    r_t = f_hidden(x_t)

    r[i] = r_t
    x[i] = x_t


def relu(x):
  return x * (x > 0)


#            def rnn(u_t, x_tm1, r_tm1, WinT, WrecT):
#                x_t = ((1 - alpha)*x_tm1
#                       + alpha*(T.dot(r_tm1, WrecT)        # Recurrent
#                                + brec                     # Bias
#                                + T.dot(u_t[:,:Nin], WinT) # Input
#                                + u_t[:,Nin:])             # Recurrent noise
#                       )
#                r_t = f_hidden(x_t)
#
#                return [x_t, r_t]
#
