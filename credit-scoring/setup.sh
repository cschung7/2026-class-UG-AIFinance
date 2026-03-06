#!/bin/bash
# Credit Scoring Project Setup
# Run: bash setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Credit Scoring Project Setup ==="

# 1. Create virtual environment
if [ ! -d "venv" ]; then
    echo "[1/3] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[1/3] Virtual environment already exists."
fi

# 2. Install dependencies
echo "[2/3] Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

# 3. Download sample data
if [ ! -f "data/credit_data.csv" ]; then
    echo "[3/3] Generating sample credit data..."
    python3 src/generate_data.py
else
    echo "[3/3] Data already exists."
fi

echo ""
echo "=== Setup Complete ==="
echo "To start: source venv/bin/activate && jupyter notebook notebooks/"
