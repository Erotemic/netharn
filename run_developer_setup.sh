#!/bin/bash
# Install dependency packages
pip install -r requirements.txt

# Install netharn in developer mode
pip install -e .

#python setup.py clean
#GCC_VERSION=6
#CC=gcc-$GCC_VERSION CXX=g++-$GCC_VERSION CUDAHOSTCXX=g++-$GCC_VERSION python setup.py develop

# Compile C extensions to improve runtime
#python setup.py build_ext --inplace

#cd ~/code/netharn/netharn/models/faster_rcnn/roi_align
#./make.sh
#cd ~/code/netharn/netharn/models/faster_rcnn/roi_pooling
#python build.py
#cd ~/code/netharn/netharn/models/faster_rcnn/roi_crop
#./make.sh
