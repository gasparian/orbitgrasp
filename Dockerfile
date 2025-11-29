FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV HTTP_PROXY=${HTTP_PROXY} HTTPS_PROXY=${HTTPS_PROXY} NO_PROXY=${NO_PROXY}
ENV http_proxy=${HTTP_PROXY} https_proxy=${HTTPS_PROXY} no_proxy=${NO_PROXY}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        bzip2 \
        ca-certificates \
        git \
        libglib2.0-0 \
        libgl1 && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

WORKDIR /app

COPY conda_environment.yaml .

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda env create -f conda_environment.yaml && \
    conda clean -afy

ENV PATH=/opt/conda/envs/orbitgrasp/bin:$PATH

COPY . .

RUN useradd -m orbit && \
    chown -R orbit /app
USER orbit

ENV ORBITGRASP_CONFIG=/app/scripts/single_config.yaml

EXPOSE 8000

CMD ["uvicorn", "scripts.orbitgrasp_server:app", "--host", "0.0.0.0", "--port", "8000"]
