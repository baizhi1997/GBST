################
This package can be used on both Windows and Linux(ubuntu).
For windows, the required lib is xgboost.dll, while for linux, the program will search for libxgboost.so.

Usage: 
cd python-package
python setup.py install

This will install survboost as a python library.

For examples using sklearn, see survivaltree_sklearn_example.py 

version 1.01: added default metric(average AUC at each timestep) in pre-compiled library.
