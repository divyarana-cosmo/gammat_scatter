#!/bin/bash
source ~/.bashrc
myenvsetup

python simulate_aroundlens.py --config config_full  --logmstelmin 11.6 --logmstelmax 14.0  --test_case True>> out.dat 2>&1 &
python simulate_aroundlens.py --config config_full  --logmstelmin 11.6 --logmstelmax 14.0 --no_shear True>> out.dat1 2>&1 &


#python simulate_aroundlens.py --config config  --logmstelmin 11.6 --logmstelmax 14.0 --test_case True --ten_percent  True
#python simulate_aroundlens.py --config config  --logmstelmin 11.6 --logmstelmax 14.0 --rot90 True --test_case True --ten_percent  True
#
#python simulate_aroundlens.py --config config  --logmstelmin 11.6 --logmstelmax 14.0 --test_case True --no_shape_noise True --ten_percent  True





#
#python simulate_aroundlens.py --config config  --logmstelmin 11.0 --logmstelmax 11.6 >> out.dat_4 2>&1 &
#python simulate_aroundlens.py --config config  --logmstelmin 11.0 --logmstelmax 11.6 --rot90 True >> out.dat_5 2>&1 &

#python simulate_aroundlens.py --config config  --logmstelmin 10.4 --logmstelmax 11.0 >> out.dat_6 2>&1 &
#python simulate_aroundlens.py --config config  --logmstelmin 10.4 --logmstelmax 11.0 --rot90 True >> out.dat_7 2>&1 &



#python simulate_aroundlens.py --config config  --logmstelmin 10.0 --logmstelmax 10.3
#python simulate_aroundlens.py --config config  --logmstelmin 10.3 --logmstelmax 10.6
#python simulate_aroundlens.py --config config  --logmstelmin 10.6 --logmstelmax 10.9
#python simulate_aroundlens.py --config config  --logmstelmin 10.9 --logmstelmax 11.3
#python simulate_aroundlens.py --config config  --logmstelmin 11.3 --logmstelmax 11.6
#python simulate_aroundlens.py --config config  --logmstelmin 10.0 --logmstelmax 10.3 --rot90 True 
#python simulate_aroundlens.py --config config  --logmstelmin 10.3 --logmstelmax 10.6 --rot90 True
#python simulate_aroundlens.py --config config  --logmstelmin 10.6 --logmstelmax 10.9 --rot90 True
#python simulate_aroundlens.py --config config  --logmstelmin 10.9 --logmstelmax 11.3 --rot90 True
#python simulate_aroundlens.py --config config  --logmstelmin 11.3 --logmstelmax 11.6 --rot90 True




#python simulate_aroundlens_no_jack.py --config config  --logmstelmin 11.3 --logmstelmax 11.6


#python simulate_aroundlens.py --config config  --logmstelmin 11.3 --logmstelmax 11.6 --test_case True --no_shear True
#python simulate_aroundlens.py --config config  --logmstelmin 11.6 --logmstelmax 14.0 --test_case True
#python simulate_aroundlens.py --config config  --logmstelmin 11.6 --logmstelmax 14.0 --test_case True --rot90 True
#python simulate_aroundlens_no_jack.py --config config  --logmstelmin 11.6 --logmstelmax 14.0 --no_shape_noise True
#python simulate_aroundlens.py --config config  --logmstelmin 11.6 --logmstelmax 14.0 
#python simulate_aroundsources.py --config config  --logmstelmin 11.6 --logmstelmax 14.0
#python simulate_aroundlens.py --config config  --logmstelmin 11.3 --logmstelmax 11.6
#python simulate_aroundlens.py --config config  --logmstelmin 11.0 --logmstelmax 11.3


#
#
#mpirun -np 10 python mpi_distort.py --config config  --logmstelmin 11.6 --logmstelmax 14
#mpirun -np 10 python mpi_distort.py --config config  --logmstelmin 11.3 --logmstelmax 11.6
#mpirun -np 10 python mpi_distort.py --config config  --logmstelmin 11.0 --logmstelmax 11.3

#python distort.py --config config  --logmstelmin 11.6 --logmstelmax 14
#mpirun -np 10 python mpi_distort.py --config config  --logmstelmin 11.0 --logmstelmax 11.3
#mpirun -np 10 python mpi_distort.py --config config  --logmstelmin 11.3 --logmstelmax 11.6
#mpirun -np 10 python mpi_distort.py --config config  --logmstelmin 11.6 --logmstelmax 14
#mpirun -np 10 python mpi_distort.py --config config  --logmstelmin 11.6 --logmstelmax 14
#python distort.py --config config  --logmstelmin 11.3 --logmstelmax 11.6
#python distort.py --config config  --logmstelmin 11.6 --logmstelmax 14




#mpirun -np 10 python distort.py --config config  --logmstelmin 11.6 --logmstelmax 14
#conda activate /data2/.conda/env/myenv
#mpirun -np 10 python distort.py --config config  --logmstelmin 11.0 --logmstelmax 11.3 
#mpirun -np 10 python distort.py --config config  --logmstelmin 11.3 --logmstelmax 11.6 
#mpirun -np 10 python distort.py --config config  --logmstelmin 11.6 --logmstelmax 14 
#mpirun -np 15 python distort.py --config config  --logmstelmin 11.6 --logmstelmax 14   
#mpirun -np 10 python distort.py --config config  --logmstelmin 11.6 --logmstelmax 14   --no_shape_noise True
#mpirun -np 10 python distort.py --config config  --logmstelmin 11.6 --logmstelmax 14   --no_shear True

#mpirun -np 10 python distort.py --config config  --logmstelmin 11.0 --logmstelmax 11.3  --rot90 True
#mpirun -np 10 python distort.py --config config  --logmstelmin 11.3 --logmstelmax 11.6  --rot90 True
#mpirun -np 10 python distort.py --config config  --logmstelmin 11.6 --logmstelmax 14  --rot90 True
#mpirun -np 10 python distort.py --config config  --logmstelmin 11.6 --logmstelmax 14 




#sort of everthing above 11.6

#mpirun -np 10 python distort.py --config config  --logmstelmin 11.0 --logmstelmax 11.3
#mpirun -np 10 python distort.py --config config  --logmstelmin 11.3 --logmstelmax 11.6
#mpirun -np 10 python distort.py --config config  --logmstelmin 11.6 --logmstelmax 14 #sort of everthing above 11.6
#
#mpirun -np 10 python distort.py --config config --ideal_case True --logmstelmin 11.0 --logmstelmax 11.3
#mpirun -np 10 python distort.py --config config --ideal_case True --logmstelmin 11.3 --logmstelmax 11.6
#mpirun -np 10 python distort.py --config config --ideal_case True --logmstelmin 11.6 --logmstelmax 14


