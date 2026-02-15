#!/bin/bash

echo "Creating conda environment..."
conda env create -f conda_linux.yml

echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate aptheta

if [ -f requirements_linux.txt ]; then
    echo "Installing pip dependencies..."
    pip install -r requirements_linux.txt
fi

echo "Done."
