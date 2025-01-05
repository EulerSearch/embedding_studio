# Use nvidia cuda as base image
FROM nvidia/cuda:11.7.1-base-ubuntu20.04

# Install required apt packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*; \
    apt-get update && \
    apt-get install -yq software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -yq python3.10 python3.10-distutils && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    apt install -yq python3.10-venv python3.10-dev python3-pip; \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10;

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
RUN pip3 install --no-cache-dir --upgrade -r requirements.txt --default-timeout=100

# Copy the application directory inside the /code directory.
COPY . /embedding_studio

# Set the command to run the uvicorn server.
#CMD ["dramatiq", "embedding_studio.workers.fine_tuning.mocked_worker", "--queues", "fine_tuning_mocked_worker", "--processes", "1", "--threads", "1"]

CMD ["dramatiq", "embedding_studio.workers.fine_tuning.worker", "--queues", "fine_tuning_mocked_worker", "--processes", "1", "--threads", "1"]
