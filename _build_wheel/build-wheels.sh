#!/bin/bash
set -e -x

# Install a system package required by our library
yum install -y atlas-devel

# Compile wheels for Python 3.6+
PYTHON_BINS=(
    #/opt/python/cp35*/bin 
    /opt/python/cp36*/bin
    #/opt/python/cp37*/bin
    #/opt/python/cp38*/bin
)
echo "${PYTHON_BINS[@]}"
PYBIN=${PYTHON_BINS[0]}

for PYBIN in ${PYTHON_BINS[@]}; do
    if [ -d $PYBIN ]; then
        "${PYBIN}/pip" install opencv_python
        "${PYBIN}/pip" install Cython
        "${PYBIN}/pip" install pytest
        "${PYBIN}/pip" install -e git+https://github.com/aleju/imgaug.git@master#egg=imgaug
        "${PYBIN}/pip" install -r /io/requirements.txt
        "${PYBIN}/pip" wheel /io/ -w wheelhouse/
    fi
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    auditwheel repair "$whl" -w /io/wheelhouse/
done

# Install packages and test
for PYBIN in /opt/python/*/bin/; do
    if [ -d $PYBIN ]; then
        "${PYBIN}/pip" install netharn --no-index -f /io/wheelhouse
        (cd "$HOME"; "${PYBIN}/python" -m pytest netharn)

        # FIXME: BREAKS BECAUSE OF TORCH 
        #    from torch._C import *
        #ImportError: /opt/python/cp35-cp35m/lib/python3.5/site-packages/torch/_C.cpython-35m-x86_64-linux-gnu.so: ELF file OS ABI invalid
        (cd "$HOME"; "${PYBIN}/python" -m xdoctest netharn all)
    fi
done
