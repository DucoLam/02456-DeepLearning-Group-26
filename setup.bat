@echo off
setlocal ENABLEDELAYEDEXPANSION

cls
echo --------------------------------------------
echo    MVTec DINO Optuna Study Setup (Windows)
echo --------------------------------------------
echo.

REM Step 0: Check for python
where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: python not found in PATH. Please install Python 3 and add it to PATH.
    goto :EOF
)

set PYTHON_BIN=python

REM Step 1: Create / reuse virtual environment
set VENV_DIR=.venv

if not exist "%VENV_DIR%" (
    echo Creating virtual environment in %VENV_DIR% ...
    "%PYTHON_BIN%" -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        goto :EOF
    )
    echo Virtual environment created.
) else (
    echo Virtual environment %VENV_DIR% already exists. Reusing it.
)

REM Activate venv
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment.
    goto :EOF
)

set PYTHON_BIN=python

echo.
set INSTALL_REQ=Y
set /p INSTALL_REQ=Do you want to install Python dependencies from requirements.txt? (Y/n): 
if /I "%INSTALL_REQ%"=="N" goto SKIP_REQUIREMENTS
if /I "%INSTALL_REQ%"=="n" goto SKIP_REQUIREMENTS

REM Step 2: Install requirements
if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found in this directory.
    goto :EOF
)

echo.
echo Installing Python dependencies from requirements.txt ...

REM Try upgrading pip (best-effort)
%PYTHON_BIN% -m pip install --upgrade pip
if errorlevel 1 (
    echo WARNING: Failed to upgrade pip, continuing with existing version.
)

REM Install the requirements
%PYTHON_BIN% -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements.
    goto :EOF
)

echo Requirements installed.
echo.

:SKIP_REQUIREMENTS

REM Step 3: Detect GPU and choose default device
set DEFAULT_DEVICE=cpu
set GPU_AVAILABLE=cpu

%PYTHON_BIN% -c "import torch" >nul 2>nul
if errorlevel 1 (
    echo torch is not importable, defaulting to CPU.
) else (
    for /f "usebackq delims=" %%D in (`%PYTHON_BIN% -c "import torch, sys; sys.stdout.write('cuda' if torch.cuda.is_available() else 'cpu')"` ) do (
        set GPU_AVAILABLE=%%D
    )
)

if /I "!GPU_AVAILABLE!"=="cuda" (
    set DEFAULT_DEVICE=cuda
    echo GPU detected. Default device will be: cuda
) else (
    set DEFAULT_DEVICE=cpu
    echo No GPU detected or CUDA not available. Default device will be: cpu
)
echo.

REM Step 4: Ask about dataset installation / download
set INSTALL_DATASET=N
set /p INSTALL_DATASET=Do you want to download and install the MVTec AD dataset now? (y/N): 

if /I "%INSTALL_DATASET%"=="Y" goto DOWNLOAD_DATASET
goto AFTER_DATASET

:DOWNLOAD_DATASET
echo.
echo Creating Dataset directory if missing...
if not exist "Dataset" (
    mkdir "Dataset"
)

echo.
echo Downloading MVTec AD dataset (large file)...
curl -L -o "Dataset\mvtec_ad.tar.xz" "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"

if errorlevel 1 (
    echo ERROR: Dataset download failed.
    echo You may need to download it manually.
    goto AFTER_DATASET
)

echo Download complete.
echo.

where tar >nul 2>nul
if errorlevel 1 (
    echo WARNING: 'tar' command is not available.
    echo Dataset archive stored at:
    echo    Dataset\mvtec_ad.tar.xz
    echo Please extract it manually.
    goto AFTER_DATASET
)

echo Extracting dataset (this may take a few minutes)...
tar -xf "Dataset\mvtec_ad.tar.xz" -C "Dataset"
if errorlevel 1 (
    echo ERROR: Extraction failed.
    echo Dataset archive remains here:
    echo    Dataset\mvtec_ad.tar.xz
    echo Please extract it manually.
    goto AFTER_DATASET
)

echo Extraction complete.
del "Dataset\mvtec_ad.tar.xz"
echo Dataset installed successfully in:
echo    Dataset\
echo.

:AFTER_DATASET
echo.
REM Step 5: Ask for dataset root
set MVTEC_ROOT=./Dataset
set /p MVTEC_ROOT=Enter path to your MVTec Dataset root (default: ./Dataset): 
if "%MVTEC_ROOT%"=="" set MVTEC_ROOT=./Dataset

if not exist "%MVTEC_ROOT%" (
    echo WARNING: Directory "%MVTEC_ROOT%" does not exist.
    echo          You can still run, but the script may fail if the dataset is missing.
)

echo Using dataset root: %MVTEC_ROOT%
echo.
endlocal