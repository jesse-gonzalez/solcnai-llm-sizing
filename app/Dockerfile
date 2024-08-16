# Description: Dockerfile for the Solcnai app
FROM python:3.8-slim

RUN groupadd --gid 1000 nutanix \
    && useradd --uid 1000 --gid 1000 -ms /bin/bash nutanix

RUN pip3 install --no-cache-dir --upgrade \
    pip \
    virtualenv

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

USER nutanix
WORKDIR /home/nutanix

COPY . app
COPY models/ /app/models/

ENV VIRTUAL_ENV=/home/nutanix/venv
RUN virtualenv ${VIRTUAL_ENV}
RUN . ${VIRTUAL_ENV}/bin/activate && pip install -r app/requirements.txt

EXPOSE 8501

COPY run.sh /home/nutanix
ENTRYPOINT ["./run.sh"]