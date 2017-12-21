FROM atlab/pytorch:0.3.0

RUN apt-get update && apt-get install -y libx11-6 && \
    pip install matplotlib

RUN pip install git+https://github.com/datajoint/datajoint-python.git

ADD . /src/bl3d
RUN pip install -e /src/bl3d
