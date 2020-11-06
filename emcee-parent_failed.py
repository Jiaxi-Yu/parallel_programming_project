import matplotlib 
matplotlib.use('agg')
import sys
import numpy as np
from numpy import log, exp, pi,append,cos,sin,sqrt
import matplotlib.pyplot as plt
import time
import corner
import emcee
from mpi4py import MPI

# initialise the processes
comm=MPI.COMM_WORLD
rank=comm.rank
size=comm.size

# given x, the maximum number of children processes
x = np.linspace(0., 101., 80000000)
maxproc = 30

# chi2 calculation
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

    return ((y_mean-model)**2/error**2).sum()

# prior for a in [1,3], b in [2,5]
def lnprior(par):
    a,b = par
    if 1 <= a <= 3 and 2 <= b <= 5:
        return 0.0
    return -np.inf

# log-likelihood obtained from chi2
def lnprob(par):
    lp = lnprior(par)
    if not np.isfinite(lp):
        return -np.inf
    return lp - 0.5 * chi2(par)

# initialise the sampler
name = 'emcee-parentl'
ndim, nwalkers = 2, 50
ini = np.array([1.5, 2.])
ini = [ini + 1e-3 * np.random.randn(ndim) \
        for i in range(nwalkers)]
# save the calculated results
backend = emcee.backends.HDFBackend(name)
backend.reset(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, \
        lnprob, backend=backend)
# start the sampling serially
sampler.run_mcmc(ini, 500, progress=True)

# plot the results
samples = sampler.chain[:, 150:, :].reshape((-1, ndim))
fig = corner.corner(samples, labels=[r'$a$', \
    r'$b$'])
fig.show()
plt.savefig('posterior_interactive.png')
plt.close()
# print the 1-sigma range of parameters
a,b = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), \
        zip(*np.percentile(samples, [16, 50, 84], axis=0)))
print('$a = {0:.4f}_{{-{1:.4f}}}^{{+{2:.4f}}}$' \
        ''.format(a[0], a[1], a[2]))
print('$b = {0:.4f}_{{-{1:.4f}}}^' \
        '{{+{2:.4f}}}$'.format(b[0], b[1], b[2]))
"""