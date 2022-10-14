# FROM ubuntu:20.04
# this docker file is for getting GPU
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
RUN apt-get update
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y apt-utils
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y gnupg2
RUN apt-get install -y wget
RUN apt-get install -y git
RUN apt-get install -y python3.8
RUN apt-get install -y curl
RUN apt-get install -y python3-distutils
RUN apt-get install -y python3-apt
RUN apt-get update
RUN apt-get install -y python3-dev
RUN apt-get install -y python3-pip
RUN apt-get install -y patchelf

COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt

RUN pip3 install -U ipython
WORKDIR /home/azureuser/Desktop