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

# =========================
# Main menu loop
# =========================
while true; do
    echo "Main menu:"
    echo "  1) Show all parameter options (--help)"
    echo "  2) Run script with default parameters"
    echo "  3) Run script with custom parameters (interactive)"
    echo "  4) Exit to shell"
    read -r -p "Choose (1/2/3/4): " CHOICE

    case "$CHOICE" in
        1)
            echo
            echo "Parameter options for run_mvtec_optuna_study.py:"
            $PYTHON_BIN run_mvtec_optuna_study.py --help
            echo
            ;;
        2)
            echo
            echo "Running with default parameters:"
            echo "  mvtec-root = $MVTEC_ROOT"
            echo "  device     = $DEFAULT_DEVICE"
            echo
            $PYTHON_BIN run_mvtec_optuna_study.py --mvtec-root "$MVTEC_ROOT" --device "$DEFAULT_DEVICE"
            echo
            ;;
        3)
            echo
            echo "Custom run configuration"

            # Categories
            CATEGORIES=""
            read -r -p "Categories (comma-separated, default: all MVTec categories): " CATEGORIES
            if [ -z "$CATEGORIES" ]; then
                CATEGORIES="bottle,cable,capsule,carpet,grid,hazelnut,leather,metal_nut,pill,screw,tile,toothbrush,transistor,wood,zipper"
            fi

            # Batch size
            BATCH_SIZE=""
            read -r -p "Batch size (default: 500): " BATCH_SIZE
            if [ -z "$BATCH_SIZE" ]; then
                BATCH_SIZE=500
            fi

            # n_trials
            N_TRIALS=""
            read -r -p "Number of Optuna trials (default: 50): " N_TRIALS
            if [ -z "$N_TRIALS" ]; then
                N_TRIALS=50
            fi

            # Percentile range
            P_MIN=""
            read -r -p "Percentile min (default: 0.0): " P_MIN
            if [ -z "$P_MIN" ]; then
                P_MIN=0.0
            fi

            P_MAX=""
            read -r -p "Percentile max (default: 1.0): " P_MAX
            if [ -z "$P_MAX" ]; then
                P_MAX=1.0
            fi

            # Threshold range
            T_MIN=""
            read -r -p "Threshold min (default: 0.4): " T_MIN
            if [ -z "$T_MIN" ]; then
                T_MIN=0.4
            fi

            T_MAX=""
            read -r -p "Threshold max (default: 1.0): " T_MAX
            if [ -z "$T_MAX" ]; then
                T_MAX=1.0
            fi

            # Device
            echo "Detected default device: $DEFAULT_DEVICE"
            DEVICE=""
            read -r -p "Device to use (cpu/cuda, default: $DEFAULT_DEVICE): " DEVICE
            if [ -z "$DEVICE" ]; then
                DEVICE="$DEFAULT_DEVICE"
            fi

            # Optional: study name
            STUDY_NAME=""
            read -r -p "Optuna study name (optional, press Enter to skip): " STUDY_NAME

            # Optional: storage
            STORAGE=""
            read -r -p "Optuna storage URL (optional, e.g., sqlite:///study.db, Enter to skip): " STORAGE

            # Optional: load_if_exists
            LOAD_IF_EXISTS_FLAG=""
            if [ -n "$STORAGE" ] && [ -n "$STUDY_NAME" ]; then
                LOAD_EXISTING=""
                read -r -p "If study exists in storage, load it? (y/N): " LOAD_EXISTING
                if [[ "$LOAD_EXISTING" = "y" || "$LOAD_EXISTING" = "Y" ]]; then
                    LOAD_IF_EXISTS_FLAG="--load-if-exists"
                fi
            fi

            echo
            echo "Running run_mvtec_optuna_study.py with custom parameters:"
            echo "  mvtec-root     = $MVTEC_ROOT"
            echo "  categories     = $CATEGORIES"
            echo "  batch-size     = $BATCH_SIZE"
            echo "  n-trials       = $N_TRIALS"
            echo "  percentile-min = $P_MIN"
            echo "  percentile-max = $P_MAX"
            echo "  threshold-min  = $T_MIN"
            echo "  threshold-max  = $T_MAX"
            echo "  device         = $DEVICE"
            if [ -n "$STUDY_NAME" ]; then
                echo "  study-name     = $STUDY_NAME"
            fi
            if [ -n "$STORAGE" ]; then
                echo "  storage        = $STORAGE"
            fi
            echo

            CMD=( "$PYTHON_BIN" run_mvtec_optuna_study.py
                  --mvtec-root "$MVTEC_ROOT"
                  --categories "$CATEGORIES"
                  --batch-size "$BATCH_SIZE"
                  --n-trials "$N_TRIALS"
                  --percentile-min "$P_MIN"
                  --percentile-max "$P_MAX"
                  --threshold-min "$T_MIN"
                  --threshold-max "$T_MAX"
                  --device "$DEVICE" )

            if [ -n "$STUDY_NAME" ]; then
                CMD+=( --study-name "$STUDY_NAME" )
            fi
            if [ -n "$STORAGE" ]; then
                CMD+=( --storage "$STORAGE" )
            fi
            if [ -n "$LOAD_IF_EXISTS_FLAG" ]; then
                CMD+=( "$LOAD_IF_EXISTS_FLAG" )
            fi

            "${CMD[@]}"
            echo
            ;;
        4)
            echo "Exiting to shell."
            break
            ;;
        *)
            echo "Invalid option. Please try again."
            echo
            ;;
    esac
done
