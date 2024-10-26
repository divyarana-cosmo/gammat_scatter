# Need to check the responsivity and the RMS for SNR calculation.
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from subprocess import  call

lmstelbins = 9.0 + np.arange(7)*0.25

for ll in range(len(lmstelbins) -1):
    call("mpirun -np 10 python mpi_simulate_aroundsources.py --config config  --logmstelmin %2.2f --logmstelmax %2.2f >> %d_logs.dat 2>&1 &" % (lmstelbins[ll], lmstelbins[ll+1], ll), shell=1)



