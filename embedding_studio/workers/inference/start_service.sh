#!/bin/bash

# Start Dramatiq + Periodiq to handle initial model download and setup in the background
DRAMATIQ_LOG="dramatiq.log"
dramatiq embedding_studio.workers.inference.worker --queues model_deployment_worker model_deletion_worker --processes 1 --threads 1 > $DRAMATIQ_LOG 2>&1 &
DRAMATIQ_PID=$!

# Function to check if Dramatiq is still running
check_dramatiq_alive() {
    if ! kill -0 $DRAMATIQ_PID 2>/dev/null; then
        echo "Dramatiq process has terminated unexpectedly. Check logs in $DRAMATIQ_LOG."
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
TRITON_LOG="triton.log"
tritonserver --model-repository=$MODEL_REPOSITORY --model-control-mode=poll --repository-poll-secs=15 --log-verbose=2 --http-address=0.0.0.0 > $TRITON_LOG 2>&1 &

# Monitor both logs
echo "Monitoring logs from both Dramatiq and Triton Server..."
tail -f $DRAMATIQ_LOG $TRITON_LOG
