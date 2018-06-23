#!/bin/bash
# Install dependency packages
pip install -r requirements.txt

# Install irharn in developer mode
pip install -e .

# Compile C extensions to improve runtime
python setup.py build_ext --inplace
