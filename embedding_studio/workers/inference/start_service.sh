#!/bin/bash

# Start Dramatiq + Periodiq to handle initial model download and setup in the background
dramatiq embedding_studio.workers.inference.worker --processes 1 --threads 1 &
DRAMATIQ_PID=$!

# Function to check if Dramatiq is still running
check_dramatiq_alive() {
    if ! kill -0 $DRAMATIQ_PID 2>/dev/null; then
        echo "Dramatiq process has terminated unexpectedly. Exiting."
        exit 1
    fi
}

# Wait for the initial model setup to complete
# Using $MODEL_REPOSITORY environment variable to check the initialization flag
while [ ! -f "${MODEL_REPOSITORY}/initialization_complete.flag" ]; do
  echo "Waiting for initial model setup to complete in ${MODEL_REPOSITORY}..."
  check_dramatiq_alive
  sleep 10
done

echo "Model setup complete, starting Triton Server..."

# Start Triton Server
tritonserver --model-repository=$MODEL_REPOSITORY --model-control-mode=poll --repository-poll-secs=15 --log-verbose=2
tail -f README.md
