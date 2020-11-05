import sys
import numpy as np
from numpy import log, exp, pi,append,cos,sin,sqrt
from mpi4py import MPI
import time 

# initialise the processes
comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()

# receive information from parent processes
lenx = np.array(0.,'i')
sigma_high  = np.array(0.,'d')
ini  = np.array(0.,'d')
comm.Bcast([lenx, MPI.INT], root=0)
comm.Bcast([sigma_high, MPI.DOUBLE], root=0)

# generate the uniform random array 
# and transform it to a Gaussian random array
uniform = np.random.RandomState(seed=rank+1).rand(lenx)
half = int(lenx/2)
noise = append(sigma_high*sqrt(-2*log(uniform[:half]))*cos(2*pi*uniform[half:]),\
        sigma_high*sqrt(-2*log(uniform[:half]))*sin(2*pi*uniform[half:]))

# sum the Gaussian random array for all the children processes
# and end the communication
comm.Reduce([noise,MPI.DOUBLE],None,op=MPI.SUM, root=0)
comm.Disconnect()

