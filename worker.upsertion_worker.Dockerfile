FROM python:3.10

# Set the working directory
WORKDIR /tmp
# Installing dependencies:
#  - Otherwise we'll get "ImportError: libGL.so.1: cannot open shared object file: No such file or directory"
RUN apt-get -y update &&  \
    apt-get install ffmpeg libsm6 libxext6 -y;

# Install Poetry in this Docker stage.
RUN pip install poetry

# Copy the pyproject.toml and poetry.lock files to the /tmp directory.
COPY ./pyproject.toml ./poetry.lock* /tmp/

# Generate the requirements.txt file.
RUN poetry export --without dev --with ml -f requirements.txt --output requirements.txt --without-hashes

ENV PATH="/workspace/install/bin:${PATH}"

# Set the working directory
WORKDIR /embedding_studio

# Copy the requirements.txt file to the /embedding_studio directory.
RUN cp -r /tmp/requirements.txt /embedding_studio/requirements.txt

# Install the package dependencies in the requirements file.
RUN pip install --no-cache-dir --upgrade -r requirements.txt --default-timeout=100

# Copy the application directory inside the /code directory.
COPY . /embedding_studio

# Set the command to run the uvicorn server.
CMD ["dramatiq", "embedding_studio.workers.upsertion.worker", "-Q", "upsertion_worker", "deletion_worker", "reindex_worker",  "reindex_subworker", "--processes", "4", "--threads", "8"]
