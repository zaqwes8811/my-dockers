# https://elitedatascience.com/keras-tutorial-deep-learning-in-python
# docker build -t ml_keras_theano:latest .

# Stopped...
# https://hub.docker.com/r/eipdev/alpine-jupyter-notebook/~/dockerfile/
#FROM alpine:latest
# https://stackoverflow.com/questions/43840365/does-alpine-apk-have-an-ubuntu-apt-no-install-recommends-equivalent
# RUN apk update && apk add wget && rm -rf /var/cache/apk/*

# now Ubuntu
#FROM ubuntu:16.04  # нужна cuda

FROM nvidia/cuda:7.5-devel
MAINTAINER Igor Lugansky <igor.a.lugansky@gmail.com>

#apt-get install -y python-dev

RUN apt-get update && apt-get install -y \
	build-essential \
    git-core wget \
    nano \
    python-dev \
	libjemalloc-dev \
	libtbb-dev \
 	&& rm -rf /var/lib/apt/lists/* 

# http://pythonwise.blogspot.com/2015/04/docker-miniconda-perfect-match.html
# wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
# задает вопросы
#RUN conda update -y conda

#conda install nose sphinx pydot-ng
# conda install numpy scipy mkl theano pygpu
# /root/miniconda2/bin/pip install pycuda scikit-cuda
# /root/miniconda2/bin/pip install cython

# https://stackoverflow.com/questions/42177658/how-to-switch-backend-with-keras-from-tensorflow-to-theano
# mkdir .keras
# touch .keras.json
# 
#{
#    "epsilon": 1e-07,
#    "floatx": "float32",
#    "image_data_format": "channels_last",
#    "backend": "theano"
#}

# can't install /root/miniconda2/bin/pip install --upgrade matplotlib jupyterlab ipywidgets
 /root/miniconda2/bin/conda install matplotlib

 jupyterlab ipywidgets



# jupiter

## launch
## here will be jypiter
#COPY main.sh /sbin
#RUN chmod +x /sbin/main.sh
#CMD ["/sbin/main.sh"]

/root/miniconda2/etc/jupyter lab --ip=0.0.0.0 --port=8888 --allow-root

WORKDIR /mnt/notebook
#ENTRYPOINT ["/sbin/tini", "--"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]