import h5py
import numpy as np
from mpi4py import MPI
import time
comm=MPI.COMM_WORLD
rank=comm.rank
size=comm.size

def data_read(filename,quantity):

    ini = time.time()
    f=h5py.File(filename,"r")

    cata_len=f["halo"]["Vpeak"].shape[0]
    mpi_cata_len=int(cata_len)//size
    if quantity == 'coordinate':
        if rank!=size-1:
            # for the first rank-1 processes, 
            # they can read array with length equals mpi_cata_len
            CoordinatesX=f["halo"]["X"][rank*mpi_cata_len:(rank+1)*mpi_cata_len]
            CoordinatesY=f["halo"]["Y"][rank*mpi_cata_len:(rank+1)*mpi_cata_len]
            CoordinatesZ=f["halo"]["Z"][rank*mpi_cata_len:(rank+1)*mpi_cata_len]
        elif rank==size-1:
            # for the last processes, 
            # they should read all the rest lines
            if cata_len%2 ==0:
                CoordinatesX=f["halo"]["X"][rank*mpi_cata_len:]
                CoordinatesY=f["halo"]["Y"][rank*mpi_cata_len:]
                CoordinatesZ=f["halo"]["Z"][rank*mpi_cata_len:]
            else:
                # if the total catalogue length is odd
                # truncate the last one 
                CoordinatesX=f["halo"]["X"][rank*mpi_cata_len:-1]
                CoordinatesY=f["halo"]["Y"][rank*mpi_cata_len:-1]
                CoordinatesZ=f["halo"]["Z"][rank*mpi_cata_len:-1]
        # form a 3-column coordinate array
        quantity = np.vstack((CoordinatesX,CoordinatesY,CoordinatesZ)).T
    else:
        if rank!=size-1:
            VZ          =f["halo"]["VZ"][rank*mpi_cata_len:(rank+1)*mpi_cata_len]
            Vpeak       =f["halo"]["Vpeak"][rank*mpi_cata_len:(rank+1)*mpi_cata_len]
        elif rank==size-1:
            if cata_len%2 ==0:
                VZ          =f["halo"]["VZ"][rank*mpi_cata_len:]
                Vpeak       =f["halo"]["Vpeak"][rank*mpi_cata_len:]
            else:
                VZ          =f["halo"]["VZ"][rank*mpi_cata_len:-1]
                Vpeak       =f["halo"]["Vpeak"][rank*mpi_cata_len:-1]
        quantity = np.vstack((VZ,Vpeak)).T
    f.close()
    # record the time of catalogue reading for each process 
    print('rank {}: {:.4f}s'.format(rank,time.time()-ini))
    return quantity

filename="parallel/UNIT4LRG.hdf5"

# gather the separately read catalogue for later use
# cannot use allreduce due to memory overflow
data_list = comm.gather(data_read(filename,'coordinate'), root=0)
if rank ==0:
    data = data_list[0]
    for i in range(size-1):
        data = np.vstack((data,data_list[i+1]))
