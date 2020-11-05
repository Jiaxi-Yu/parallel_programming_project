#!/bin/bash

#SBATCH -p p5
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH -J seq
#SBATCH --exclusive

module load openmpi

valgrind --tool=callgrind /home/astro/jiayu/Desktop/HAM-MCMC/mtsham-MPI/MTSHAM --prior-min [0,50,500] --prior-max [2.0,140,3000] --best-fit /home/astro/jiayu/Desktop/HAM-MCMC/squential_p5.dat --output /home/astro/jiayu/Desktop/HAM-MCMC/MCMCout/multinest_ -i /home/astro/jiayu/Desktop/HAM-MCMC/catalog/UNIT_hlist_0.58760.dat --cov-matrix /home/astro/jiayu/Desktop/HAM-MCMC/catalog/nersc_mps_LRG_v7_2/covR-LRG_SGC-5_25-quad.dat --fit-ref /home/astro/jiayu/Desktop/HAM-MCMC/catalog/nersc_mps_LRG_v7_2/mps_linear_LRG_SGC.dat --fit-column [4,5] --redshift 0.7018 -n 62600
