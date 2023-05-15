import os

os.system("rm -rf build")
os.system("mkdir build")

os.system("/d/achanhon/miniconda3/bin/python -u hack_train.py")
os.system("/d/achanhon/miniconda3/bin/python -u hack_test.py")