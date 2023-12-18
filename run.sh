#!/bin/bash
#conda activate /data2/.conda/env/myenv
mpirun -np 10 python distort.py --config config  --logmstelmin 11.0 --logmstelmax 11.3 
mpirun -np 10 python distort.py --config config  --logmstelmin 11.3 --logmstelmax 11.6 
mpirun -np 10 python distort.py --config config  --logmstelmin 11.6 --logmstelmax 14   

mpirun -np 10 python distort.py --config config  --logmstelmin 12.0 --logmstelmax 11.3  --rot90 True
mpirun -np 10 python distort.py --config config  --logmstelmin 11.3 --logmstelmax 11.6  --rot90 True
mpirun -np 10 python distort.py --config config  --logmstelmin 11.6 --logmstelmax 14  --rot90 True




#sort of everthing above 11.6

#mpirun -np 10 python distort.py --config config  --logmstelmin 11.0 --logmstelmax 11.3
#mpirun -np 10 python distort.py --config config  --logmstelmin 11.3 --logmstelmax 11.6
#mpirun -np 10 python distort.py --config config  --logmstelmin 11.6 --logmstelmax 14 #sort of everthing above 11.6
#
#mpirun -np 10 python distort.py --config config --ideal_case True --logmstelmin 11.0 --logmstelmax 11.3
#mpirun -np 10 python distort.py --config config --ideal_case True --logmstelmin 11.3 --logmstelmax 11.6
#mpirun -np 10 python distort.py --config config --ideal_case True --logmstelmin 11.6 --logmstelmax 14


