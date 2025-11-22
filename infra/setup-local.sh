#!/usr/bin/env bash
set -e

# Detect OS and install appropriate backend
OS=$(uname)
if [[ "$OS" == "Darwin" ]]; then
    echo "macOS detected — installing CPU backend"
    uv sync --extra cpu
else
    # you could add GPU detection here
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
        uv sync --extra cu128
    else
        uv sync --extra cpu
    fi
fi

# Pull the data
echo "Pulling workout data from public repo..."
chmod +x ./data/get_wrkout_data.sh
./data/get_wrkout_data.sh