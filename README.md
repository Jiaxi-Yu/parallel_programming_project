# parallel_programming_project

This project aims at improving the existing python code for the SubHalo Abundance Matching in another repository. The file reading and the fast generation of multiple Gaussian random arrays are bottlenecks of the present code. So I'd like to use mpi4py to improve the performance. Unfortunately, non of the MPI method is compatible with the Monte-Carlo Sampler Pymultinest and emcee. So except for the hdf5 reading, the other methods are not updated in the existing SHAM code.
