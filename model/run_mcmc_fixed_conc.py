import os
import numpy as np

logMstelmin = 9.5 + 0.1*np.arange(25)
logMstelmax = 9.5 + 0.1*np.arange(1,26)


for ss,(logMmin, logMmax) in enumerate(zip(logMstelmin, logMstelmax)):
    if logMmin >= 11.5 and logMmin<9.5:
        continue
    os.system('mpirun -np 64 python mcmc_nfw_com_fixed_conc.py  --config config.ini --seed %d --logMmin %2.2f --logMmax %2.2f'%(ss, logMmin, logMmax))
    print("done", logMmin, logMmax)



