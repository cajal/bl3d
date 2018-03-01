FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

LABEL maintainer="Erick Cobos <ecobos@bcm.edu>"

WORKDIR /src

# Install some optimization libraries
RUN apt-get update && \
    apt-get install -y libopenblas-dev libatlas-base-dev libeigen3-dev && \
    export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1   

# Install Python 3
RUN apt-get update && \
    apt-get install -y python3-dev python3-pip && \
    pip3 install numpy scipy matplotlib ipython

# Install pytorch 
#RUN pip3 install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp35-cp35m-linux_x86_64.whl && \
#    pip3 install torchvision

# Install pytorch (from source)
RUN pip3 install numpy pyyaml mkl setuptools cmake cffi typing && \
    apt-get install -y git && \
    pip3 install git+https://github.com/pytorch/pytorch && \
    pip3 install torchvision

# Install datajoint
RUN pip3 install git+https://github.com/datajoint/datajoint-python.git

# Install bl3d
ADD . /src/bl3d
RUN pip3 install -e /src/bl3d

# Clean apt lists
RUN rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/bin/bash"]
