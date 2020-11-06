import matplotlib 
matplotlib.use('agg')
import sys
import numpy as np
from numpy import log, exp, pi,append,cos,sin,sqrt
import matplotlib.pyplot as plt
import time
import corner
import emcee
import h5py
from multiprocessing import Pool 
from mpi4py import MPI

# initialise the processes
comm=MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# given x, the maximum number of children processes
# and the uniform random arrays
x = np.linspace(0., 101., 80000000)
maxproc = 30
uniform_randoms = [np.random.RandomState(seed=1000*i).rand(len(x)) for i in range(maxproc)] 

# generate Gaussian noise
def noises(uniform): 
    sigma_high  = 1.
    half = int(len(x)/2)
    # transform the input uniform array to a Gaussian random array
    noise = append(sigma_high*sqrt(-2*log(uniform[:half]))*cos(2*pi*uniform[half:]),\
        sigma_high*sqrt(-2*log(uniform[:half]))*sin(2*pi*uniform[half:]))
    return noise

# chi2 calculation
def chi2(par):   
    # get the model parameters
    a, b=par
    model = a*x+b
    error = 1.0
    # the noises function takes every element in zip() 
    # and return to the shared memory a list
    # the number of thread it uses is maxproc
    with Pool(processes = maxproc) as p:
        noise_list = p.starmap(noises,zip(uniform_randoms))

    # generate the 'observed' data y_mean    
    y_mean = 2*x+3+np.mean(noise_list,axis=0)

    return ((y_mean-model)**2/error**2).sum()

# prior for a in [1,3], b in [2,5]
def lnprior(par):
    H0, Om = par
    if 1 <= H0 <= 3 and 2 <= Om <= 5:
        return 0.0
    return -np.inf

# log-likelihood obtained from chi2
def lnprob(par):
    lp = lnprior(par)
    if not np.isfinite(lp):
        return -np.inf
    return lp - 0.5 * chi2(par)

# test the hybrid structure 
# and compare with the parent-child structure
print('rank {}, chi2 {:.3f}'.format(rank,lnprob([1+2./size*rank,2+3./size*rank])))
