#!/usr/bin/env bash
set -e

# Create folders if not exists
mkdir -p data/eval
mkdir -p data/processed
mkdir -p data/raw
mkdir -p data/vector_db

# Clone or pull latest raw data
if [ -d "data/raw/.git" ]; then
    cd data/raw && git pull
else
    git clone https://github.com/wrkout/exercises.json.git data/raw
fi
