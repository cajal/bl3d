FROM atlab/pytorch

RUN pip install git+https://github.com/datajoint/datajoint-python.git

RUN apt-get update && apt-get install -y libx11-6 && \
    pip install matplotlib

ADD . /src/bl3d
RUN pip install -e /src/bl3d

WORKDIR /notebooks
