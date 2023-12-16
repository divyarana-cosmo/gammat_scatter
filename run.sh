#!/bin/zsh
mpirun -np 6 python distort.py --config config  --logmstelmin 11.0 --logmstelmax 11.3
mpirun -np 6 python distort.py --config config  --logmstelmin 11.3 --logmstelmax 11.6
mpirun -np 6 python distort.py --config config  --logmstelmin 11.6 --logmstelmax 14 #sort of everthing above 11.6

#mpirun -np 6 python distort.py --config config --ideal_case True --logmstelmin 11.0 --logmstelmax 11.3
#mpirun -np 6 python distort.py --config config --ideal_case True --logmstelmin 11.3 --logmstelmax 11.6
#mpirun -np 6 python distort.py --config config --ideal_case True --logmstelmin 11.6 --logmstelmax 14


