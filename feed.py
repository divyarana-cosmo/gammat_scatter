from subprocess import  call
import numpy as np


nbins = 3
binedgs = 10.0 + 0.5*np.arange(nbins+1)

#call("python simulate_aroundlens.py --config config_full  --no_shape_noise True --logmstelmin 11.6 --logmstelmax 14.0 --test_case True --seed 345 >> test_out.dat_noshp 2>&1 &", shell=1)
#call("python simulate_aroundlens.py --config config_full  --logmstelmin 11.6 --logmstelmax 14.0 --test_case True --seed 456 --no_shear True >> test_out.dat 2>&1 &", shell=1)
#call("python simulate_aroundlens.py --config config_full  --rot90 True --logmstelmin 11.6 --logmstelmax 14.0 --test_case True  --seed 456 --no_shear True >> test_out.dat_90 2>&1 &", shell=1)



for ss in np.arange(20):
    call("python simulate_aroundlens.py --config config_full  --logmstelmin 11.6 --logmstelmax 14.0 --test_case True --seed %d >> test_out.dat_%d 2>&1 &"%(ss,ss), shell=1)
    #call("python simulate_aroundlens.py --config config_full --logmstelmin 11.6 --logmstelmax 14.0 --test_case True --use_shear True --seed %s >> test_out.dat_%d 2>&1 &"%(ss,ss), shell=1)
#    call("python simulate_aroundlens.py --config config_full --rot90 True --logmstelmin 11.6 --logmstelmax 14.0 --test_case True --use_shear True --seed %s >> test_out.dat_90_%s 2>&1 &"%(ss,ss), shell=1)
#
#for bb in range(nbins):
#    call("python simulate_aroundlens.py --config config --two_percent True --no_shape_noise True --logmstelmin %s --logmstelmax %s >> out.dat_noshp_%d 2>&1 &" % (binedgs[bb], binedgs[bb+1], bb), shell=1)
#    call("python simulate_aroundlens.py --config config --two_percent True --logmstelmin %s --logmstelmax %s >> out.dat_%d 2>&1 &" % (binedgs[bb], binedgs[bb+1], bb), shell=1)
#    call("python simulate_aroundlens.py --config config --two_percent True --rot90 True --logmstelmin %s --logmstelmax %s >> out.dat_90_%d 2>&1 &" % (binedgs[bb], binedgs[bb+1], bb), shell=1)
#
#    call("python simulate_aroundlens.py --config config_full --no_shape_noise True --logmstelmin %s --logmstelmax %s >> full_out.dat_noshp_%d 2>&1 &" % (binedgs[bb], binedgs[bb+1], bb), shell=1)
#    call("python simulate_aroundlens.py --config config_full --logmstelmin %s --logmstelmax %s >> full_out.dat_%d 2>&1 &" % (binedgs[bb], binedgs[bb+1], bb), shell=1)
#    call("python simulate_aroundlens.py --config config_full --rot90 True --logmstelmin %s --logmstelmax %s >> full_out.dat_90_%d 2>&1 &" % (binedgs[bb], binedgs[bb+1], bb), shell=1)
#
#
#    
#    print("feeding pipiline: ", binedgs[bb], binedgs[bb+1])


#call("python simulate_aroundlens.py --config config_full --no_shape_noise True --logmstelmin 11.6 --logmstelmax 14.0 --test_case True --use_shear True>> test_out.dat_noshp 2>&1 &", shell=1)
#call("python simulate_aroundlens.py --config config_full --logmstelmin 11.6 --logmstelmax 14.0 --test_case True --use_shear True >> test_out.dat 2>&1 &", shell=1)
#call("python simulate_aroundlens.py --config config_full --rot90 True --logmstelmin 11.6 --logmstelmax 14.0 --test_case True --use_shear True>> test_out.dat_90 2>&1 &", shell=1)
#
#call("python simulate_aroundlens.py --config config --two_percent True --no_shape_noise True --logmstelmin 11.6 --logmstelmax 14.0 --test_case True >> test_out.dat_noshp 2>&1 &", shell=1)
#call("python simulate_aroundlens.py --config config --two_percent True --logmstelmin 11.6 --logmstelmax 14.0 --test_case True>> test_out.dat 2>&1 &", shell=1)
#call("python simulate_aroundlens.py --config config --two_percent True --rot90 True --logmstelmin 11.6 --logmstelmax 14.0 --test_case True >> test_out.dat_90 2>&1 &", shell=1)



print("completed feeding")

#call("python simulate_aroundlens.py --config config --ten_percent True --no_shape_noise True --logmstelmin %s --logmstelmax %s >> out.dat_noshp_%d 2>&1 &" % (binedgs[bb], binedgs[bb+1], bb), shell=1)
#call("python simulate_aroundlens.py --config config --ten_percent True --logmstelmin %s --logmstelmax %s >> out.dat_%d 2>&1 &" % (binedgs[bb], binedgs[bb+1], bb), shell=1)
#call("python simulate_aroundlens.py --config config --ten_percent True --rot90 True --logmstelmin %s --logmstelmax %s >> out.dat_90_%d 2>&1 &" % (binedgs[bb], binedgs[bb+1], bb), shell=1)
 




