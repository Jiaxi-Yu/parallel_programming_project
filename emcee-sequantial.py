import matplotlib 
matplotlib.use('agg')
import sys
import numpy as np
from numpy import log, exp, pi,append,cos,sin,sqrt
import matplotlib.pyplot as plt
import time
import corner
import emcee

# given x, number of random seeds, the uniform random arrays
# data length and Gaussian dispersion parameter
x     = np.linspace(0., 101., 100000)#80000000)
nseed = 5
uniform_randoms = [np.random.RandomState(seed=rank+1).rand(len(x)).astype('float32') for rank in range(nseed)]
sigma_high  = 1.

# chi2 calculation
def chi2(par):
    # get the model parameters
    a, b=par
    model = a*x+b
    error = 1.

    # initialise 
    noise = np.zeros((len(x),nseed))
    half = int(len(x)/2) 

    # transform a uniform random distribution to a Gaussian distribution
    for i,uniform in enumerate(uniform_randoms):
        y[:,i] += \
         append(sigma_high*sqrt(-2*log(uniform[:half]))*cos(2*pi*uniform[half:]),\
                sigma_high*sqrt(-2*log(uniform[:half]))*sin(2*pi*uniform[half:]))

    # generate the 'observed' y using the noise
    y_mean = 2.*x+3.+np.mean(noise,axis=-1)
    
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
name = 'emcee-serial'
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
