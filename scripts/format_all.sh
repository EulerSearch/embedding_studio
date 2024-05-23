#!/bin/sh -e
set -x

# Define a list of directories to process as a space-separated string
DIRECTORIES="embedding_studio examples plugins"

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place $DIRECTORIES --exclude=__init__.py
black $DIRECTORIES
isort $DIRECTORIES
