#!/bin/bash
# Install dependency packages
pip install -r requirements.txt

# Install irharn in developer mode
pip install -e .

# Compile C extensions to improve runtime
python setup.py build_ext --inplace


cd ~/code/netharn/netharn/models/faster_rcnn/roi_align
./make.sh
cd ~/code/netharn/netharn/models/faster_rcnn/roi_pooling
python build.py
cd ~/code/netharn/netharn/models/faster_rcnn/roi_crop
./make.sh
