import matplotlib 
matplotlib.use('agg')
import sys
import numpy as np
from numpy import log, exp, pi,append,cos,sin,sqrt
import matplotlib.pyplot as plt
import time
from schwimmbad import MPIPool 
import corner
import emcee

# define the worker function calculated in the MPIPool
def worker(param):
    lenx,sigma_high,seed = param
    uniform = np.random.RandomState(seed=seed+1).rand(lenx).astype('float32') 
    half = int(lenx/2)
    # transform a uniform random distribution to a Gaussian distribution
    noise = append(sigma_high*sqrt(-2*log(uniform[:half]))*cos(2*pi*uniform[half:]),\
            sigma_high*sqrt(-2*log(uniform[:half]))*sin(2*pi*uniform[half:]))
    return noise

# given x, number of random seeds, data length and Gaussian dispersion parameter
x = np.linspace(0., 101., 100)
nseed = 5
lenx = len(x)
sigma_high  = 1.

# chi2 calculation
def chi2(par):
    # introduce a MPIPool
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        # the worker function takes every element in tasks and return to the master process a list
        tasks = list(zip([lenx]*nseed,[sigma_high]*nseed,[seed for seed in range(nseed)]))
        results = pool.map(worker, tasks)

    # the model, error and the 'observed' data y_mean
    model = par[0]*x+par[1]
    error = 1
    y_mean = 2*x+3+np.mean(results,axis=0)
    return ((y_mean-model)**2/error**2).sum()

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

# initialise the sampler
name = 'emcee-MPIPool'
ndim, nwalkers = 2, 50
ini = np.array([1.5, 2.5])
ini = [ini + 1e-3 * np.random.randn(ndim) \
        for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, \
        lnprob)
# start the sampling serially
sampler.run_mcmc(ini, 1000,progress=True)

# plot the results
samples = sampler.chain[:, 150:, :].reshape((-1, ndim))
fig = corner.corner(samples, labels=[r'$a$', \
    r'$b$'])
fig.show()
plt.savefig('{}.png'.format(name))
plt.close()
# print the 1-sigma range of parameters
a,b = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), \
        zip(*np.percentile(samples, [16, 50, 84], axis=0)))
print('$a = {0:.4f}_{{-{1:.4f}}}^{{+{2:.4f}}}$' \
        ''.format(a[0], a[1], a[2]))
print('$b = {0:.4f}_{{-{1:.4f}}}^' \
        '{{+{2:.4f}}}$'.format(b[0], b[1], b[2]))
