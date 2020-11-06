import matplotlib 
matplotlib.use('agg')
import sys
import numpy as np
from numpy import log, exp, pi,append,cos,sin,sqrt
import matplotlib.pyplot as plt
import time
import pymultinest
from getdist import plots, MCSamples, loadMCSamples
import getdist

# given x, number of random seeds, the uniform random arrays
# data length and Gaussian dispersion parameter
x     = np.linspace(0., 101., 100)
nseed = 5
uniform_randoms = [np.random.RandomState(seed=rank+1).rand(len(x)).astype('float32') for rank in range(nseed)]
sigma_high  = 1.

# chi2 calculation
def chi2(a,b):
    # get the model parameters
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

# priors
def prior(cube, ndim, nparams):
    cube[0] = 2*cube[0]+1  # uniform between [1,3]
    cube[1] = 3*cube[1]+2  # uniform between [2,5]

# log-likelihood
def loglike(cube, ndim, nparams):
    a,b = cube[0],cube[1]
    return -0.5*chi2(a,b)

# parameter names, number of dimensions 
# and the root path of the output
parameters = ["a","b"]
npar = len(parameters)
fileroot = 'output/sequential_'

# run MultiNest sampler
pymultinest.run(loglike, prior, npar, outputfiles_basename=fileroot, \
                resume = False, verbose = True)

if rank ==0:
    # save the parameter names
    f=open(fileroot+'.paramnames', 'w')
    for param in parameters:
        f.write(param+'\n')
    f.close()

    # get the analyse results
    a = pymultinest.Analyzer(outputfiles_basename=fileroot, n_params = npar)

    # get the statistics and plot them with gedist
    a.get_stats()

    # the posterior error plot
    sample = loadMCSamples(fileroot)
    plt.rcParams['text.usetex'] = False
    g = plots.get_single_plotter(width_inch=4, ratio=1)
    g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add=0.4
    g.settings.title_limit_fontsize = 14
    g = plots.get_subplot_plotter()
    g.triangle_plot(sample,['a', 'b'], filled=True,title_limit=1)
    g.export(fileroot+'posterior.png')

    # print the best fit parameters and 1-sigma confidence interval
    print('Statistics from getdist ...')
    stats = sample.getMargeStats()
    best = np.zeros(npar)
    lower = np.zeros(npar)
    upper = np.zeros(npar)
    mean = np.zeros(npar)
    sigma = np.zeros(npar)
    for i in range(npar):
        par = stats.parWithName(parameters[i])
        mean[i] = par.mean
        sigma[i] = par.err
        lower[i] = par.limits[0].lower
        upper[i] = par.limits[0].upper
        best[i] = (lower[i] + upper[i]) * 0.5
        print('{0:s}: {1:.5f} + {2:.6f} - {3:.6f}, or {4:.5f} +- {5:.6f}'.format( \
            parameters[i], best[i], upper[i]-best[i], best[i]-lower[i], mean[i], \
            sigma[i]))
