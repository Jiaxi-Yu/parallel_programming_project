import matplotlib 
matplotlib.use('agg')
import sys
import numpy as np
from numpy import log, exp, pi,append,cos,sin,sqrt
import matplotlib.pyplot as plt
import time
from mpi4py import MPI
import corner
import emcee
import h5py

# initialise the processes
comm=MPI.COMM_WORLD
rank=comm.rank
size=comm.size

# given x, the maximum number of children processes
x = np.linspace(0., 101., 80000000)
maxproc = 30

# the reduced chi2 calculation
def chi2(par):  
    # get the model parameters  
    a, b=par
    model = a*x+b
    error = 1.
    
    # Spawn maxproc children processes which 
    # exists in the file 'chi2-child.py'
    comm = MPI.COMM_SELF.Spawn(sys.executable,
                           args=['chi2-child.py'],
                           maxprocs=maxproc)
    
    # broadcast the parameters
    lenx = np.array(len(x),'i')
    sigma_high  = np.array(1.,'d')
    comm.Bcast([lenx, MPI.INT], root=MPI.ROOT)
    comm.Bcast([sigma_high, MPI.DOUBLE], root=MPI.ROOT)

    # receive the reduced sum over all the children processes
    noise = np.zeros(lenx,'d')
    comm.Reduce(None,[noise,MPI.DOUBLE],op=MPI.SUM, root=MPI.ROOT)

    # construct the 'observed' y_mean
    y_mean = 2*x+3+noise/maxproc

    comm.Disconnect()

    return ((y_mean-model)**2/error**2).mean()

# prior for a in [1,3], b in [2,5]
def lnprior(par):
    a,b = par
    if 1. <= a <= 3. and 2. <= b <= 5.:
        return 0.0
    return -np.inf

# log-likelihood obtained from chi2
def lnprob(par):
    lp = lnprior(par)
    if not np.isfinite(lp):
        return -np.inf
    return lp - 0.5 * chi2(par)

# test the parent-child structure 
print('rank {}, chi2 {:.3f}'.format(rank,lnprob([1+2./size*rank,2+3./size*rank])))
