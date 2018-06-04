echo '''
We need to build extensions for manylinux
This can be done with docker and Centos5

https://github.com/pypa/manylinux


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

# Interactive test
REPO_DPATH=$HOME/code/netharn

# Assume we are in the repo path
REPO_DPATH=`pwd`

docker pull quay.io/pypa/manylinux1_x86_64
docker run --rm -v `pwd`:/io $DOCKER_IMAGE /io/travis/build-wheels.sh


# Interactive test
docker run -it --rm -v $REPO_DPATH:/io quay.io/pypa/manylinux1_x86_64 bash
