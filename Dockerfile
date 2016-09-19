# docker-debian-cuda - Debian 9 with CUDA Toolkit

FROM gw000/keras:1.0.8-gpu
MAINTAINER gw0 [http://gw.tnode.com/] <gw.2016@tnode.com>

# install py2-tf-cpu/gpu (Python 2, TensorFlow, CPU/GPU)
# (already installed in upstream image)

# install py2-th-cpu (Python 2, Theano, CPU/GPU)
RUN "sh" "-c" "echo nameserver 8.8.8.8 >> /etc/resolv.conf"

ARG THEANO_VERSION=0.8.2
ENV THEANO_FLAGS='device=cpu,floatX=float32'
RUN pip --no-cache-dir install git+https://github.com/Theano/Theano.git@rel-${THEANO_VERSION}

# install py3-tf-cpu/gpu (Python 3, TensorFlow, CPU/GPU)
RUN apt update -qq \
    && apt-get upgrad \
 && apt-get install --no-install-recommends -y \
    # install python 3
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-virtualenv \
    pkg-config \
    # requirements for numpy
    libopenblas-base \
    python3-numpy \
    python3-scipy \
    # requirements for keras
    python3-h5py \
    python3-yaml \
    python3-pydot \
    # requirements for matplotlib
    python3-matplotlib \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

ARG TENSORFLOW_VERSION=0.9.0
ARG TENSORFLOW_DEVICE=gpu
RUN pip3 --no-cache-dir install https://storage.googleapis.com/tensorflow/linux/${TENSORFLOW_DEVICE}/tensorflow-${TENSORFLOW_VERSION}-cp35-cp35m-linux_x86_64.whl

ARG KERAS_VERSION=1.0.8
ENV KERAS_BACKEND=tensorflow
RUN pip3 --no-cache-dir install git+https://github.com/fchollet/keras.git@${KERAS_VERSION}

# install py3-th-cpu/gpu (Python 3, Theano, CPU/GPU)
ARG THEANO_VERSION=0.8.2
ENV THEANO_FLAGS='device=cpu,floatX=float32'
RUN pip3 --no-cache-dir install git+https://github.com/Theano/Theano.git@rel-${THEANO_VERSION}

# install jupyter notebook and ipython (Python 2 and 3)
RUN pip --no-cache-dir install \
    ipython \
    ipykernel \
    jupyter \
    sklearn \
    pudb \
 && python -m ipykernel.kernelspec \
 && pip3 --no-cache-dir install \
    ipython \
    ipykernel \
 && python3 -m ipykernel.kernelspec

# install system tools
RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y \
    less \
    procps \
    vim-tiny \
    vim \
    python-tk\
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*




# configure console
RUN echo 'alias ll="ls --color=auto -lA"' >> /root/.bashrc \
 && echo '"\e[5~": history-search-backward' >> /root/.inputrc \
 && echo '"\e[6~": history-search-forward' >> /root/.inputrc

# dump package lists
RUN dpkg-query -l > /dpkg-query-l.txt \
 && pip2 freeze > /pip2-freeze.txt \
 && pip3 freeze > /pip3-freeze.txt

# for jupyter
EXPOSE 8888
# for tensorboard
EXPOSE 6006

# S3
ENV S3_LOCATION /s3data
RUN apt-get update && apt-get install -y wget automake autotools-dev g++ git libcurl4-gnutls-dev libfuse-dev libssl-dev libxml2-dev make pkg-config && git clone https://github.com/s3fs-fuse/s3fs-fuse && wget https://github.com/Yelp/dumb-init/releases/download/v1.0.1/dumb-init_1.0.1_amd64.deb && dpkg -i dumb-init_*.deb && rm dumb-init_*.deb
WORKDIR s3fs-fuse
RUN ./autogen.sh && ./configure --prefix=/usr --with-openssl && make && make install
WORKDIR /
ADD run.sh /run.sh
RUN chmod 755 /run.sh && rm -rf /s3fs-fuse && mkdir /s3data
CMD dumb-init /run.sh
VOLUME /s3data

WORKDIR /srv/
CMD /bin/bash -c 'jupyter notebook --no-browser --ip=* "$@"'