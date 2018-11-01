# 0. install nvidia-docker2

# 1. ...

# https://elitedatascience.com/keras-tutorial-deep-learning-in-python
# sudo docker build -t ml_keras_tf_theano:latest .

# sudo docker start ml_keras_tf_theano_0
# sudo docker exec -it ml_keras_tf_theano_0 bash

# Stopped...
# https://hub.docker.com/r/eipdev/alpine-jupyter-notebook/~/dockerfile/
#FROM alpine:latest
# https://stackoverflow.com/questions/43840365/does-alpine-apk-have-an-ubuntu-apt-no-install-recommends-equivalent
# RUN apk update && apk add wget && rm -rf /var/cache/apk/*

# now Ubuntu
# https://blog.ubuntu.com/2018/07/09/minimal-ubuntu-released
# https://www.zdnet.com/article/minimal-ubuntu-for-containers-and-clouds/
# 18 now minimal
# https://github.com/tianon/docker-brew-ubuntu-core/tree/c7e9f7353aa24d1c35f501e06382aed1b540e85f
#FROM ubuntu:16.04  # нужна cuda
# Понадобилась cuda runtime
#FROM ubuntu:18.04
#FROM nvidia/cuda:7.5-devel

# FROM nvidia/cuda:7.5-cudnn5-runtime

#FROM nvidia/cuda:8.0-cudnn5-runtime
# но потом почему-то cudnn 7 доставляется
# не хочется тащить все
# FROM nvidia/cuda:7.5-cudnn5-devel

#FROM nvidia/cuda:9.0-cudnn7-runtime
#FROM nvidia/cuda:8.0-cudnn5-runtime
# Theano then
FROM nvidia/cuda:7.5-cudnn5-devel


LABEL authors="Igor Lugansky <igor.a.lugansky@gmail.com>, Golovchanskaia Iuiia <golju@bk.ru>"

RUN apt-get update && apt-get install -y \
	build-essential \
    git-core wget \
    nano \
    curl \
    python-dev \
 	&& rm -rf /var/lib/apt/lists/* 

# http://pythonwise.blogspot.com/2015/04/docker-miniconda-perfect-match.html
# wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
# задает вопросы
RUN curl -LO https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh \
	&& bash Miniconda2-latest-Linux-x86_64.sh -p /root/miniconda2 -b \
	&& rm Miniconda2-latest-Linux-x86_64.sh

# Split
RUN /root/miniconda2/bin/pip install cython \
	&& /root/miniconda2/bin/conda update -y conda \
	&& /root/miniconda2/bin/conda install scipy nose sphinx pydot-ng \
	matplotlib jupyterlab ipywidgets mkl numpy \
	&& /root/miniconda2/bin/conda clean -tisy



#  - no if runtime then cuda.h not found
# https://codeyarns.com/2015/07/31/pip-install-error-with-pycuda/
# 
RUN /root/miniconda2/bin/conda install  theano pygpu \
	&& /root/miniconda2/bin/conda clean -tisy \
	&& /root/miniconda2/bin/pip install scikit-cuda pycuda


# # conda clean????
# # can't install /root/miniconda2/bin/pip install --upgrade matplotlib jupyterlab ipywidgets
# # tipsy -> tisy
# # keras поставить tf
# RUN /root/miniconda2/bin/conda install tensorflow-gpu==1.10.0 \
# 	&& /root/miniconda2/bin/conda clean -tisy

# CPU only
RUN /root/miniconda2/bin/conda install tensorflow==1.10.0 \
 	&& /root/miniconda2/bin/conda clean -tisy

RUN /root/miniconda2/bin/conda install keras=2.0.2 \
	&& /root/miniconda2/bin/conda clean -tisy

RUN wget https://cmake.org/files/v3.10/cmake-3.10.0.tar.gz \
	&& tar -xvf cmake-3.10.0.tar.gz \
	&& cd cmake-3.10.0 && ./configure && make -j4 && make install && cd ..

#################################

#theano:
#https://stackoverflow.com/questions/21608025/how-to-set-up-theano-config
# https://www.asozykin.ru/deep_learning/2017/03/11/How-to-use-gpu-with-theano.html
# https://github.com/Theano/Theano/wiki/Converting-to-the-new-gpu-back-end%28gpuarray%29
# http://queirozf.com/entries/setup-keras-theano-backend-and-gpu-on-ubuntu-16-04

# GpuArrayException: GPU is too old for CUDA version
# THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32 /root/miniconda2/bin/python2.7 TheanoGPU.py
# DEVICE=cuda0 python -c "import pygpu;pygpu.test()"
# https://stackoverflow.com/questions/47597499/gpuarrayexception-gpu-is-too-old-for-cuda-version

# Настраиваем CNMeM

# libgpuarray
# http://deeplearning.net/software/libgpuarray/installation.html

###################################
#
# tf
# From sources:
# https://gist.github.com/jorgemf/d2f3d85fadeb6e9d88ab00a06fbca0a2
#
# https://anaconda.org/anaconda/tensorflow-gpu/files
# /root/miniconda2/bin/conda install tensorflow-gpu==1.0.1
# need cuDNN + runtime(врядли буду кодить для куды) + tf а не теано
# FIXME: не отработана работа с runtime образами
# Compotibility:
# https://stackoverflow.com/questions/50622525/which-tensorflow-and-cuda-version-combinations-are-compatible

# !!! https://www.tensorflow.org/install/source
# Troubles: https://github.com/tensorflow/tensorflow/issues/18906
# https://github.com/tensorflow/tensorflow/issues/12388
# FIXME: как проверить что на gpu Запускаем?
# check:
# https://stackoverflow.com/questions/38009682/how-to-tell-if-tensorflow-is-using-gpu-acceleration-from-inside-python-shell
#
#

# CPU
# I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
# 2018-11-01 09:14:16.068667: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
# Device mapping: no known devices.

# если ставить tf - начинает качать новье  
# нет смысла ставить днн если ставить тензор, даже брать базовый образ как nvidia смысла нет
# там ставиться не совсем то, там нет cudatoolkit/nvcc and headers
#totalMemory: 963.31MiB freeMemory: 624.25MiB
#2018-11-01 08:47:38.489691: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1455] Ignoring visible gpu device (device: 0, name: GeForce GTX 560, pci bus id: 0000:01:00.0, compute capability: 2.1) 
# with Cuda compute capability 2.1. The minimum required Cuda capability is 3.0.
# https://stackoverflow.com/questions/38542763/how-can-i-make-tensorflow-run-on-a-gpu-with-capability-2-0 - NO WAY
# CPU only

###################################

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
#/root/miniconda2/bin/jupyter-notebook --ip=0.0.0.0 --port=8888 --allow-root

#WORKDIR /mnt/notebook
#ENTRYPOINT ["/sbin/tini", "--"]
#CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Troubles:
# https://github.com/Theano/Theano/issues/6533
# http://deeplearning.net/software/theano/troubleshooting.html
# https://github.com/ChicagoBoothAnalytics/Software/blob/master/AWS/EMR/.config/.theanorc
# ~/.theanorc 
#[blas]
#ldflags = -lf77blas -latlas -lgfortran #put your flags here
#ldflags = -lf77blas -latlas -lgfortran #put your flags here
#ldflags = -lmkl -lguide -lpthread -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lguide -liomp5 -lmkl_mc -lpthread


# DANGER:
# https://github.com/tensorflow/tensorflow/issues/16898
