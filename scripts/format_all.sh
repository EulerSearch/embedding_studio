#!/bin/sh -e
set -x

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place embedding_studio --exclude=__init__.py
black embedding_studio
isort embedding_studio
