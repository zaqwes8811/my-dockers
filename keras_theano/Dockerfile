# 0. install nvidia-docker2

# 1. ...

# https://elitedatascience.com/keras-tutorial-deep-learning-in-python
# docker build -t ml_keras_theano:latest .

# sudo docker start ml_keras_theano_0
# sudo docker exec -it ml_keras_theano_0 bash

# Stopped...
# https://hub.docker.com/r/eipdev/alpine-jupyter-notebook/~/dockerfile/
#FROM alpine:latest
# https://stackoverflow.com/questions/43840365/does-alpine-apk-have-an-ubuntu-apt-no-install-recommends-equivalent
# RUN apk update && apk add wget && rm -rf /var/cache/apk/*

# now Ubuntu
#FROM ubuntu:16.04  # нужна cuda
#FROM nvidia/cuda:7.5-devel
# need cuDNN + runtime(врядли буду кодить для куды) + tf а не теано
# FIXME: не отработана работа с runtime образами
#FROM nvidia/cuda:7.5-cudnn5-runtime
#FROM nvidia/cuda:8.0-cudnn5-runtime
# но потом почему-то cudnn 7 доставляется
FROM nvidia/cuda:7.5-cudnn5-devel  

LABEL authors="Igor Lugansky <igor.a.lugansky@gmail.com>, Golovchanskaia Iuiia <golju@bk.ru>"

RUN apt-get update && apt-get install -y \
	build-essential \
    git-core wget \
    nano \
    curl \
    python-dev \
	libjemalloc-dev \
	libtbb-dev \
 	&& rm -rf /var/lib/apt/lists/* 

# http://pythonwise.blogspot.com/2015/04/docker-miniconda-perfect-match.html
# wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
# задает вопросы
RUN curl -LO https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh \
	&& bash Miniconda2-latest-Linux-x86_64.sh -p /root/miniconda2 -b \
	&& rm Miniconda2-latest-Linux-x86_64.sh

# conda clean????
# can't install /root/miniconda2/bin/pip install --upgrade matplotlib jupyterlab ipywidgets
# tipsy -> tisy
RUN /root/miniconda2/bin/pip install cython \
	&& /root/miniconda2/bin/conda update -y conda \
	&& /root/miniconda2/bin/conda install tensorflow-gpu numpy mkl scipy nose sphinx pydot-ng \
	matplotlib jupyterlab ipywidgets keras \
	&& /root/miniconda2/bin/conda clean -tisy

# pycuda - no if runtime then cuda.h not found
RUN /root/miniconda2/bin/pip install scikit-cuda \
	&& /root/miniconda2/bin/conda update -y conda \
	&& /root/miniconda2/bin/conda install  theano pygpu pycuda  \
	&& /root/miniconda2/bin/conda clean -tisy

# tf
# https://github.com/tensorflow/tensorflow/issues/12388
# FIXME: как проверить что на gpu Запускаем?

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

# jupiter

# launch
# here will be jypiter
COPY main.sh /sbin
RUN chmod +x /sbin/main.sh
CMD ["/sbin/main.sh"]

#/root/miniconda2/bin/jupyter-lab --ip=0.0.0.0 --port=8888 --allow-root

#WORKDIR /mnt/notebook
#ENTRYPOINT ["/sbin/tini", "--"]
#CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]