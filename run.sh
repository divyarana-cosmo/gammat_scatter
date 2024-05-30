#!/bin/bash
source ~/.bashrc
myenvsetup

python simulate_aroundsources.py --config config  --logmstelmin 9.5 --logmstelmax 10
python simulate_aroundsources.py --config config  --logmstelmin 10 --logmstelmax 10.5
python simulate_aroundsources.py --config config  --logmstelmin 10.5 --logmstelmax 11


