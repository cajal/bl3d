FROM nvidia/cuda:9.2-cudnn7-devel

LABEL maintainer="Erick Cobos <ecobos@bcm.edu>"

WORKDIR /src

# Install some optimization libraries
RUN apt-get update && \
    apt-get install -y libopenblas-dev libatlas-base-dev libeigen3-dev && \
    export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1   

# Install Python 3
RUN apt-get update && \
    apt-get install -y python3-dev python3-pip python3-tk && \
    pip3 install numpy scipy matplotlib ipython jupyterlab

# Install pytorch 
RUN pip3 install http://download.pytorch.org/whl/cu92/torch-0.4.1-cp35-cp35m-linux_x86_64.whl && \
    pip3 install torchvision

# Install pytorch (from source, takes ~1h)
#RUN pip3 install numpy pyyaml mkl setuptools cmake cffi typing && \
#    apt-get install -y git && \
#    pip3 install git+https://github.com/pytorch/pytorch && \
#    pip3 install torchvision

# Install datajoint
RUN apt-get install -y libssl-dev && pip3 install datajoint

# Install bl3d
ADD . /src/bl3d
RUN pip3 install -e /src/bl3d

# Clean apt lists
RUN rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/bin/bash"]
