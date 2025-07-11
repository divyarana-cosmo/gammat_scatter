from subprocess import  call
import numpy as np


#nbins = 6
#binedgs = 10.0 + 0.3*np.arange(nbins+1)
#
#
#for bb in range(nbins):
#    call("mpirun -np 4 python mcmc.py ../debug_z_0.1_0.5_two_percent/simed_sources.dat_lmstelmin_%2.2f_lmstelmax_%2.2f_with_shape_noise_w_jacks_pairs 0.007 >> out.dat_shp_%d 2>&1 &" % (binedgs[bb], binedgs[bb+1], bb), shell=1)
#    call("mpirun -np 4 python mcmc.py ../debug_z_0.1_0.5_two_percent/simed_sources.dat_lmstelmin_%2.2f_lmstelmax_%2.2f_with_shape_noise_with_90_rotation_w_jacks_pairs 0.007 >> out.dat_shp_90_%d 2>&1 &" % (binedgs[bb], binedgs[bb+1], bb), shell=1)
#    
#    print("feeding pipiline: ", binedgs[bb], binedgs[bb+1])


from glob import glob
#flist = glob('../debug_full_z_0.1_0.5/simed_sources.dat_lmstelmin_11.60_lmstelmax_14.00_with_shape_noise_*_seed_*_pairs')
#flist = glob('../debug_full_z_0.1_0.5/simed_sources.dat_lmstelmin_11.60_lmstelmax_14.00_with_shape_noise_*_seed_*_pairs')
#flist = glob('/net/dobbe/data2/debug_full_z_0.1_0.5/simed_sources.dat_lmstelmin_11.60_lmstelmax_14.00_with_shape_noise_*_seed_*_pairs')
#flist = glob('/net/dobbe/data2/debug_full_z_0.1_0.5/simed_sources.dat_lmstelmin_11.60_lmstelmax_14.00_with_shape_noise_test_case_seed_*_w_jacks')
flist = glob('/net/dobbe/data2/debug_full_z_0.1_0.5/simed_sources.dat_lmstelmin_11.60_lmstelmax_14.00_with_shape_noise_test_case_seed_*_w_jacks_pairs')
flist = np.sort(flist)
print(flist)
#exit()
for cnt,fil in enumerate(flist[:1]):
    print(cnt,fil)
    call("mpirun -np 10 python mcmc_gnfw_free_conc.py %s 0.007 0.25 0"%(fil), shell=1)
    #call("mpirun -np 4 python mcmc_gnfw.py %s 0.007 0.3 1 >> test_out.dat_shp_0.3_%d 2>&1 &"%(fil,cnt), shell=1)
    #call("mpirun -np 4 python mcmc_gnfw.py %s 0.007 0.3 0 >> test_out.dat_shp_0.3_r90_%d 2>&1 &"%(fil,cnt), shell=1)

    #call("mpirun -np 4 python mcmc_stack.py %s 0.01 0.4 1 >> test_out.dat_stackedshp_0.05_%d 2>&1 &"%(fil,cnt), shell=1)
    #call("mpirun -np 4 python mcmc_stack.py %s 0.01 0.4 0 >> test_out.dat_stackedshp_0.05_r90_%d 2>&1 &"%(fil,cnt), shell=1)

    #call("mpirun -np 4 python mcmc_free_conc.py %s 0.05 0.25 1 >> test_out.dat_shp_0.05_%d_free_conc 2>&1 &"%(fil,cnt), shell=1)
    #call("mpirun -np 4 python mcmc_free_conc.py %s 0.05 0.25 0 >> test_out.dat_shp_0.05_r90_%d_free_conc 2>&1 &"%(fil,cnt), shell=1)



    #call("mpirun -np 10 python mcmc_free_conc.py ../debug_full_z_0.1_0.5/%s 0.007 >> test_out.dat_shp_%d 2>&1 &"%(fil,cnt), shell=1)
    #call("mpirun -np 10 python mcmc.py ../debug_full_z_0.1_0.5/%s 0.007 >> test_out.dat_shp_%d 2>&1 &"%(fil,cnt), shell=1)


#call("mpirun -np 8 python mcmc.py ../debug_full_z_0.1_0.5/simed_sources.dat_lmstelmin_11.60_lmstelmax_14.00_with_shape_noise_test_case_w_jacks_pairs 0.007 >> test_out.dat_shp 2>&1 &", shell=1)
#call("mpirun -np 8 python mcmc.py ../debug_full_z_0.1_0.5/simed_sources.dat_lmstelmin_11.60_lmstelmax_14.00_with_shape_noise_with_90_rotation_test_case_w_jacks_pairs 0.007 >> test_out.dat_90 2>&1 &", shell=1)
#call("mpirun -np 8 python mcmc.py ../debug_full_z_0.1_0.5/simed_sources.dat_lmstelmin_11.60_lmstelmax_14.00_no_shape_noise_test_case_w_jacks_pairs 0.007 >> test_out.dat_noshp 2>&1 &", shell=1)

#call("mpirun -np 8 python mcmc.py ../debug_full_z_0.1_0.5/simed_sources.dat_lmstelmin_11.60_lmstelmax_14.00_with_shape_noise_test_case_w_jacks_pairs_using_shear 0.007 >> test_out.dat_shp 2>&1 &", shell=1)
#call("mpirun -np 8 python mcmc.py ../debug_full_z_0.1_0.5/simed_sources.dat_lmstelmin_11.60_lmstelmax_14.00_with_shape_noise_with_90_rotation_test_case_w_jacks_pairs_using_shear 0.007 >> test_out.dat_90 2>&1 &", shell=1)
#call("mpirun -np 8 python mcmc.py ../debug_full_z_0.1_0.5/simed_sources.dat_lmstelmin_11.60_lmstelmax_14.00_no_shape_noise_test_case_w_jacks_pairs_using_shear 0.007 >> test_out.dat_noshp 2>&1 &", shell=1)




#call("python simulate_aroundlens.py --config config_full --no_shape_noise True --logmstelmin 11.6 --logmstelmax 14.0 --test_case True >> test_out.dat_noshp 2>&1 &", shell=1)
#call("python simulate_aroundlens.py --config config_full --logmstelmin 11.6 --logmstelmax 14.0 --test_case True>> test_out.dat 2>&1 &", shell=1)
#call("python simulate_aroundlens.py --config config_full --rot90 True --logmstelmin 11.6 --logmstelmax 14.0 --test_case True >> test_out.dat_90 2>&1 &", shell=1)
#
#call("python simulate_aroundlens.py --config config --two_percent True --no_shape_noise True --logmstelmin 11.6 --logmstelmax 14.0 --test_case True >> test_out.dat_noshp 2>&1 &", shell=1)
#call("python simulate_aroundlens.py --config config --two_percent True --logmstelmin 11.6 --logmstelmax 14.0 --test_case True>> test_out.dat 2>&1 &", shell=1)
#call("python simulate_aroundlens.py --config config --two_percent True --rot90 True --logmstelmin 11.6 --logmstelmax 14.0 --test_case True >> test_out.dat_90 2>&1 &", shell=1)



#call("python simulate_aroundlens.py --config config_full  --no_shape_noise True --logmstelmin 11.6 --logmstelmax 14.0 --test_case True >> test_out.dat_noshp 2>&1 &", shell=1)
#call("python simulate_aroundlens.py --config config_full  --logmstelmin 11.6 --logmstelmax 14.0 --test_case True>> test_out.dat 2>&1 &", shell=1)
#call("python simulate_aroundlens.py --config config_full  --rot90 True --logmstelmin 11.6 --logmstelmax 14.0 --test_case True >> test_out.dat_90 2>&1 &", shell=1)

print("completed feeding")

#call("python simulate_aroundlens.py --config config --ten_percent True --no_shape_noise True --logmstelmin %s --logmstelmax %s >> out.dat_noshp_%d 2>&1 &" % (binedgs[bb], binedgs[bb+1], bb), shell=1)
#call("python simulate_aroundlens.py --config config --ten_percent True --logmstelmin %s --logmstelmax %s >> out.dat_%d 2>&1 &" % (binedgs[bb], binedgs[bb+1], bb), shell=1)
#call("python simulate_aroundlens.py --config config --ten_percent True --rot90 True --logmstelmin %s --logmstelmax %s >> out.dat_90_%d 2>&1 &" % (binedgs[bb], binedgs[bb+1], bb), shell=1)
 




