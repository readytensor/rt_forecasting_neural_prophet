FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04 as builder


RUN apt-get -y update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# install python and pip and add symbolic link to python3
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.9 python3-pip

# Ensure Python 3.9 is the default python and pip versions
RUN ln -sf /usr/bin/python3.9 /usr/bin/python \
    && ln -sf /usr/bin/python3.9 /usr/bin/python3 \
    && ln -sf /usr/bin/pip3 /usr/bin/pip


COPY ./requirements.txt /opt/
RUN python3.9 -m pip install --upgrade pip \
    && python3.9 -m pip install --no-cache-dir -r /opt/requirements.txt


COPY src ./opt/src
COPY ./entry_point.sh /opt/
RUN chmod +x /opt/entry_point.sh

WORKDIR /opt/src

ENV MPLCONFIGDIR=/tmp/matplotlib
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/app:${PATH}"

RUN mkdir -p /opt/src/lightning_logs && chmod -R 777 /opt/src/lightning_logs
RUN chmod -R 777 /opt/src/


# set non-root user
USER 1000
# set entrypoint
ENTRYPOINT ["/opt/entry_point.sh"]