import os
import numpy as np
import sys

logMstelmin = 9.5 + 0.1*np.arange(25)
logMstelmax = 9.5 + 0.1*np.arange(1,26)

for ss,(logMmin, logMmax) in enumerate(zip(logMstelmin, logMstelmax)):
    if logMmin >= 11.5 and logMmin<9.5:
        continue
    os.system('mpirun -np 32 python mcmc_nfw_com.py --config config.ini --seed %d --logMmin %2.2f --logMmax %2.2f --Rpmin 3.0'%(ss, logMmin, logMmax))
    os.system('mpirun -np 32 python mcmc_gnfw_com.py --config config.ini --seed %d --logMmin %2.2f --logMmax %2.2f --Rpmin 3.0'%(ss, logMmin, logMmax))
    print("done", logMmin, logMmax)

#rpmin = float(sys.argv[1])
#os.system('mpirun -np 32 python mcmc_nfw_com.py --config config.ini --seed 0 --logMmin 9.50 --logMmax 9.60 --Rpmin %2.2f'%rpmin)
#os.system('mpirun -np 32 python mcmc_nfw_com.py --config config.ini --seed 10 --logMmin 10.50 --logMmax 10.60 --Rpmin %2.2f'%rpmin)
#os.system('mpirun -np 32 python mcmc_nfw_com.py --config config.ini --seed 19 --logMmin 11.40 --logMmax 11.50 --Rpmin %2.2f'%rpmin)


#logMstelmin = 9.5 + 0.1*np.arange(2)
#logMstelmax = 9.5 + 0.1*np.arange(1,3)


#
    #if logMmin != 11.2:
    #    continue
    #os.system('python make_precompute.py --config config.ini --seed %d --logMmin %2.2f --logMmax %2.2f'%(ss, logMmin, logMmax))
 

  #os.system('mpirun -np 64 python mcmc_nfw_com_fixed_conc.py  --config config.ini --seed %d --logMmin %2.2f --logMmax %2.2f'%(ss, logMmin, logMmax))
##!/bin/bash
#
#source ~/.bashrc
#myenvsetup
##python make_precompute.py --config config.ini --seed 0 --logMmin 9.5 --logMmax 10.0
##python make_precompute.py --config config.ini --seed 1 --logMmin 10.0 --logMmax 10.5
##python make_precompute.py --config config.ini --seed 2 --logMmin 10.5 --logMmax 11.0
#python make_precompute.py --config config.ini --seed 3 --logMmin 10.5 --logMmax 10.6
#
##mpirun -np 64 python mcmc_nfw_com.py --config config.ini --seed 0 --logMmin 9.5 --logMmax 10.0
##mpirun -np 64 python mcmc_nfw_com.py --config config.ini --seed 1 --logMmin 10.0 --logMmax 10.5
##mpirun -np 64 python mcmc_nfw_com.py --config config.ini --seed 2 --logMmin 10.5 --logMmax 11.0
#mpirun -np 64 python mcmc_nfw_com.py --config config.ini --seed 3 --logMmin 10.5 --logMmax 10.6
##mpirun -np 64 python mcmc_nfw_com.py --config config.ini --seed 3 --logMmin 10.9 --logMmax 11.0
#
##for ((i=0;i<10;i++))
##do
##    mpirun -np 64 python mcmc_nfw_com.py --config config.ini --seed $i
##done    
#
