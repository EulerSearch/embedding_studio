FROM nvcr.io/nvidia/tritonserver:24.04-pyt-python-py3

# Install required apt packages
ENV DEBIAN_FRONTEND noninteractive
# Install base utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add the Deadsnakes PPA for newer Python versions
RUN add-apt-repository ppa:deadsnakes/ppa

# Install Python 3.10 and its utilities
RUN apt-get update && apt-get install -y \
    && apt-get install -y python3-pip \
    && rm -rf /var/lib/apt/lists/*;

ENV PATH="/workspace/install/bin:${PATH}"

# Upgrade pip to the latest version
RUN python3 -m pip install --upgrade pip

# Set /tmp as the current working directory.
WORKDIR /tmp

# Install Poetry in this Docker stage.
RUN pip install poetry

# Copy the pyproject.toml and poetry.lock files to the /tmp directory.
COPY ./pyproject.toml ./poetry.lock* /tmp/

# Generate the requirements.txt file.
RUN poetry export --with ml -f requirements.txt --output requirements.txt --without-hashes

# Set the working directory
WORKDIR /embedding_studio

# Copy the requirements.txt file to the /embedding_studio directory.
RUN cp -r /tmp/requirements.txt /embedding_studio/requirements.txt

# Install the package dependencies in the requirements file.
RUN pip3 install --no-cache-dir --upgrade  --ignore-installed blinker -r requirements.txt --default-timeout=100

# Copy the application directory inside the /code directory.
COPY . /embedding_studio

# Expose ports
# 8000: HTTP service (Triton)
# 8001: GRPC service (Triton)
# 8002: Metrics service (Triton)
EXPOSE 8000
EXPOSE 8001
EXPOSE 8002

# Set the environment variable for the model repository
ENV MODEL_REPOSITORY='/models'
ENV CUDA_VISIBLE_DEVICES=0

RUN mkdir /models
RUN touch /triton.log

# Copy the start-up script to the workspace
COPY ./embedding_studio/workers/inference/start_service.sh /embedding_studio

# Start script that will manage the initiation of Dramatiq/Periodiq and then start Triton
CMD ["/bin/bash", "./start_service.sh"]
