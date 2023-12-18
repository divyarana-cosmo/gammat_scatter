#!/bin/bash
#conda activate /data2/.conda/env/myenv
python get_plot.py --config config  --logmstelmin 11.0 --logmstelmax 11.3  --Rmin 0.01 --Rmax 0.8 --Rbins 6 --Njacks 20 
python get_plot.py --config config  --logmstelmin 11.3 --logmstelmax 11.6  --Rmin 0.01 --Rmax 0.8 --Rbins 6 --Njacks 20
python get_plot.py --config config  --logmstelmin 11.6 --logmstelmax 14    --Rmin 0.01 --Rmax 0.8 --Rbins 6 --Njacks 20


python get_plot.py --config config  --logmstelmin 11.0 --logmstelmax 11.3  --rot90 True --Rmin 0.01 --Rmax 0.8 --Rbins 6 --Njacks 20
python get_plot.py --config config  --logmstelmin 11.3 --logmstelmax 11.6  --rot90 True --Rmin 0.01 --Rmax 0.8 --Rbins 6 --Njacks 20
python get_plot.py --config config  --logmstelmin 11.6 --logmstelmax 14  --rot90 True --Rmin 0.01 --Rmax 0.8 --Rbins 6 --Njacks 20



