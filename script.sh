#!/bin/bash
set -e
#
#setting uv path
export PATH="$HOME/.local/bin:$PATH"

MODEL_DIR="$HOME/weights/Llama-3.2-3B-Instruct"
cd "$HOME/ReLax"

# Switch to sharding branch
git checkout sharding
git pull origin sharding

# if [ ! -d ".venv" ]; then
#   echo "Creating virtual environment..."
#   uv venv .venv --python=3.12
# fi
#

source .venv/bin/activate
uv pip install -e '.[dev]'
echo "ReLax initialized successfully"

# Check if TPU library is accessible
# TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=1 \
# bash -c 'until python -c "import jax; print(jax.devices(\"tpu\"))"; do sleep 2; done'

# if [ $? -ne 0 ]; then
#   echo "Warning: TPU devices not detected. installing jax lib for TPU"
#   uv pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# fi

python inference.py --model_path "$MODEL_DIR"
