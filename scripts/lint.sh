#!/usr/bin/env bash

set -x

#mypy embedding_studio
black embedding_studio --check
isort --check-only embedding_studio
flake8
