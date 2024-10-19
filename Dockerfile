# FROM python:3.12.3
FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04

RUN useradd -m -u 1000 user
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt

# get python
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install software-properties-common -y

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3-pip python3.12 python-is-python3 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip install torch --index-url https://download.pytorch.org/whl/cu124 --upgrade --force-reinstall
USER user
COPY --link --chown=1000 ./ /code

CMD [ "python", "./DataReader.py"]