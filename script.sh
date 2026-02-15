#!/bin/bash
set -e
#
#setting uv path

export PATH="$HOME/.local/bin:$PATH"

MODEL_DIR="$HOME/Llama-3.1-8B-Instruct"
cd "$HOME/ReLax"

# Switch to sharding branch
git switch interleaved
git pull origin interleaved


# Check if TPU library is accessible
# TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=1 \
# bash -c 'until python -c "import jax; print(jax.devices(\"tpu\"))"; do sleep 2; done'

# if [ $? -ne 0 ]; then
#   echo "Warning: TPU devices not detected. installing jax lib for TPU"
#   uv pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# fi

JAX_LOG_COMPILES=1 uv run python inference.py --model_path "$MODEL_DIR"
