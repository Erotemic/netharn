echo '''
We need to build extensions for manylinux
This can be done with docker and Centos5

https://github.com/pypa/manylinux

https://github.com/pypa/python-manylinux-demo/blob/master/travis/build-wheels.sh


Install Docker:
     sudo apt install apt-transport-https ca-certificates curl software-properties-common
     curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
     sudo apt-key fingerprint 0EBFCD88
     IS_1604=$(python -c "import ubelt as ub; print('16.04' in ub.cmd('lsb_release -a')['out'])")
     IS_1804=$(python -c "import ubelt as ub; print('18.04' in ub.cmd('lsb_release -a')['out'])")
     if [ "$IS_1604" == "True" ]; then
         sudo add-apt-repository \
            "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
            $(lsb_release -cs) \
            stable"
     fi
     if [ "$IS_1804" == "True" ]; then
         # Hack: docker-ce doesnt have an 18.04 package yet.
         sudo add-apt-repository \
             "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
             artful stable"
     fi

sudo apt update
sudo apt install -y docker-ce


# Add self to docker group
sudo groupadd docker
sudo usermod -aG docker $USER
# NEED TO LOGOUT / LOGIN to revaluate groups
su - $USER  # or we can do this

docker run hello-world


# NVIDIA-Docker
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu16.04/amd64/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update

# Install nvidia-docker2 and reload the Docker daemon configuration
sudo apt install -y nvidia-docker2
sudo pkill -SIGHUP dockerd
''' > /dev/null

REPO_DPATH=$HOME/code/netharn
DOCKER_IMAGE=quay.io/pypa/manylinux1_x86_64
PRE_CMD=""
# Interactive test
docker run -it --rm -v $REPO_DPATH:/io $DOCKER_IMAGE $PRE_CMD bash

# INSIDE DOCKER:

PYBIN=/opt/python/cp36-cp36m/bin/
$PYBIN/python -m pip install -r requirements.txt
$PYBIN/python setup.py bdist_wheel --py-limited-api=cp36


        "${PYBIN}/pip" install opencv_python
        "${PYBIN}/pip" install Cython
        "${PYBIN}/pip" install pytest
        "${PYBIN}/pip" install -e git+https://github.com/aleju/imgaug.git@master#egg=imgaug
        "${PYBIN}/pip" install torch
        "${PYBIN}/pip" install -r /io/requirements.txt


##################3


# Assume we are in the repo path
#REPO_DPATH=`pwd`

DOCKER_IMAGE=quay.io/pypa/manylinux1_i686
PRE_CMD=linux32

docker pull quay.io/pypa/manylinux1_x86_64
docker run --rm -v `pwd`:/io $DOCKER_IMAGE /io/travis/build-wheels.sh


#######
__heredoc__='''

https://wheel.readthedocs.io/en/stable/

pip install wheel

# Build a directory of wheels for pyramid and all its dependencies
pip wheel --wheel-dir=./tmp/wheelhouse netharn

# Install from cached wheels
pip install --no-index --find-links=/tmp/wheelhouse pyramid

# Install from cached wheels remotely
pip install --no-index --find-links=https://wheelhouse.example.com/ pyramid



python setup.py bdist_wheel --py-limited-api=cp36
'''
