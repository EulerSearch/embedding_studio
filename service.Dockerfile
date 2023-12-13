# This is the first stage, it is named requirements-stage.
FROM python:3.9 as requirements-stage

# Set /tmp as the current working directory.
WORKDIR /tmp

# Install Poetry in this Docker stage.
RUN pip install poetry

# Copy the pyproject.toml and poetry.lock files to the /tmp directory.
COPY ./pyproject.toml ./poetry.lock* /tmp/

# Generate the requirements.txt file.
RUN poetry export --without ml,dev -f requirements.txt --output requirements.txt --without-hashes

# This is the final stage, anything here will be preserved in the final container image.
FROM python:3.9

# Set the working directory
WORKDIR /embedding_studio

# Copy the requirements.txt file to the /embedding_studio directory.
COPY --from=requirements-stage /tmp/requirements.txt /embedding_studio/requirements.txt

# Install the package dependencies in the requirements file.
RUN pip install --no-cache-dir --upgrade -r /embedding_studio/requirements.txt

# Copy the application directory inside the /code directory.
COPY . /embedding_studio

# Open port 5000 for the uvicorn server.
EXPOSE 5000

# Set the command to run the uvicorn server.
CMD ["uvicorn", "embedding_studio.main:app", "--host", "0.0.0.0", "--port", "5000", "--log-config", "embedding_studio/log_config.yaml"]
