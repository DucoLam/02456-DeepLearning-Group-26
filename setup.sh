#!/usr/bin/env bash

set -u  # no unset vars; don't use -e to keep menu usable

clear
echo "--------------------------------------------"
echo "   MVTec DINO Optuna Study Setup (Linux)"
echo "--------------------------------------------"
echo

# Step 0: Check for python
if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
else
    echo "ERROR: python3 or python not found in PATH. Please install Python 3."
    exit 1
fi

# Step 1: Create / reuse virtual environment
VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment."
        exit 1
    fi
    echo "Virtual environment created."
else
    echo "Virtual environment $VENV_DIR already exists. Reusing it."
fi

# Activate venv
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment."
    exit 1
fi

PYTHON_BIN=python

echo
read -r -p "Do you want to install Python dependencies from requirements.txt? (Y/n): " INSTALL_REQ
INSTALL_REQ=${INSTALL_REQ:-Y}
if [[ "$INSTALL_REQ" != "Y" && "$INSTALL_REQ" != "y" ]]; then
    :
else
    # Step 2: Install requirements
    if [ ! -f "requirements.txt" ]; then
        echo "ERROR: requirements.txt not found in this directory."
        exit 1
    fi

    echo
    echo "Installing Python dependencies from requirements.txt ..."

    # Try upgrading pip (best-effort)
    $PYTHON_BIN -m pip install --upgrade pip || {
        echo "WARNING: Failed to upgrade pip, continuing with existing version."
    }

    # Install the requirements
    if ! $PYTHON_BIN -m pip install -r requirements.txt; then
        echo "ERROR: Failed to install requirements."
        exit 1
    fi

    echo "Requirements installed."
    echo
fi

# Step 3: Detect GPU and choose default device
DEFAULT_DEVICE="cpu"
GPU_AVAILABLE="cpu"

if $PYTHON_BIN -c "import torch" >/dev/null 2>&1; then
    GPU_AVAILABLE=$($PYTHON_BIN -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")
else
    echo "torch is not importable, defaulting to CPU."
fi

if [ "$GPU_AVAILABLE" = "cuda" ]; then
    DEFAULT_DEVICE="cuda"
    echo "GPU detected. Default device will be: cuda"
else
    DEFAULT_DEVICE="cpu"
    echo "No GPU detected or CUDA not available. Default device will be: cpu"
fi
echo

# Step 4: Ask about dataset installation / download
read -r -p "Do you want to download and install the MVTec AD dataset now? (y/N): " INSTALL_DATASET
INSTALL_DATASET=${INSTALL_DATASET:-N}

if [[ "$INSTALL_DATASET" = "Y" || "$INSTALL_DATASET" = "y" ]]; then
    echo
    echo "Creating Dataset directory if missing..."
    mkdir -p "Dataset"

    echo
    echo "Downloading MVTec AD dataset (large file)..."
    curl -L -o "Dataset/mvtec_ad.tar.xz" "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"
    if [ $? -ne 0 ]; then
        echo "ERROR: Dataset download failed."
        echo "You may need to download it manually."
    else
        echo "Download complete."
        echo

        if ! command -v tar >/dev/null 2>&1; then
            echo "WARNING: 'tar' command is not available."
            echo "Dataset archive stored at:"
            echo "   Dataset/mvtec_ad.tar.xz"
            echo "Please extract it manually."
        else
            echo "Extracting dataset (this may take a few minutes)..."
            if ! tar -xf "Dataset/mvtec_ad.tar.xz" -C "Dataset"; then
                echo "ERROR: Extraction failed."
                echo "Dataset archive remains here:"
                echo "   Dataset/mvtec_ad.tar.xz"
                echo "Please extract it manually."
            else
                echo "Extraction complete."
                rm -f "Dataset/mvtec_ad.tar.xz"
                echo "Dataset installed successfully in:"
                echo "   Dataset/"
                echo
            fi
        fi
    fi
fi

echo
# Step 5: Ask for dataset root
MVTEC_ROOT="./Dataset"
read -r -p "Enter path to your MVTec Dataset root (default: ./Dataset): " MVTEC_ROOT_INPUT
if [ -n "$MVTEC_ROOT_INPUT" ]; then
    MVTEC_ROOT="$MVTEC_ROOT_INPUT"
fi

if [ ! -d "$MVTEC_ROOT" ]; then
    echo "WARNING: Directory \"$MVTEC_ROOT\" does not exist."
    echo "         You can still run, but the script may fail if the dataset is missing."
fi

echo "Using dataset root: $MVTEC_ROOT"
echo
done