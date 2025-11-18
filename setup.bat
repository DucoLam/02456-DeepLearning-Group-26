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

REM Main menu loop
:MAIN_MENU
echo Main menu:
echo   1) Show all parameter options (--help)
echo   2) Run script with default parameters
echo   3) Run script with custom parameters (interactive)
echo   4) Exit to shell
set /p CHOICE=Choose (1/2/3/4): 

if "%CHOICE%"=="1" goto SHOW_HELP
if "%CHOICE%"=="2" goto RUN_DEFAULT
if "%CHOICE%"=="3" goto RUN_CUSTOM
if "%CHOICE%"=="4" goto EXIT_SCRIPT

echo Invalid option. Please try again.
echo.
goto MAIN_MENU

:SHOW_HELP
echo.
echo Parameter options for run_mvtec_optuna_study.py:
%PYTHON_BIN% run_mvtec_optuna_study.py --help
echo.
goto MAIN_MENU

:RUN_DEFAULT
echo.
echo Running with default parameters:
echo   mvtec-root = %MVTEC_ROOT%
echo   device     = %DEFAULT_DEVICE%
echo.

%PYTHON_BIN% run_mvtec_optuna_study.py --mvtec-root "%MVTEC_ROOT%" --device "%DEFAULT_DEVICE%"
echo.
goto MAIN_MENU

:RUN_CUSTOM
echo.
echo Custom run configuration

REM Categories
set CATEGORIES=
set /p CATEGORIES=Categories (comma-separated, default: all MVTec categories): 
if "%CATEGORIES%"=="" set CATEGORIES=bottle,cable,capsule,carpet,grid,hazelnut,leather,metal_nut,pill,screw,tile,toothbrush,transistor,wood,zipper

REM Batch size
set BATCH_SIZE=
set /p BATCH_SIZE=Batch size (default: 500): 
if "%BATCH_SIZE%"=="" set BATCH_SIZE=500

REM n_trials
set N_TRIALS=
set /p N_TRIALS=Number of Optuna trials (default: 50): 
if "%N_TRIALS%"=="" set N_TRIALS=50

REM Percentile range
set P_MIN=
set /p P_MIN=Percentile min (default: 0.0): 
if "%P_MIN%"=="" set P_MIN=0.0

set P_MAX=
set /p P_MAX=Percentile max (default: 1.0): 
if "%P_MAX%"=="" set P_MAX=1.0

REM Threshold range
set T_MIN=
set /p T_MIN=Threshold min (default: 0.4): 
if "%T_MIN%"=="" set T_MIN=0.4

set T_MAX=
set /p T_MAX=Threshold max (default: 1.0): 
if "%T_MAX%"=="" set T_MAX=1.0

REM Device
echo Detected default device: %DEFAULT_DEVICE%
set DEVICE=
set /p DEVICE=Device to use (cpu/cuda, default: %DEFAULT_DEVICE%): 
if "%DEVICE%"=="" set DEVICE=%DEFAULT_DEVICE%

REM Optional: study name
set STUDY_NAME=
set /p STUDY_NAME=Optuna study name (optional, press Enter to skip): 

REM Optional: storage
set STORAGE=
set /p STORAGE=Optuna storage URL (optional, e.g., sqlite:///study.db, Enter to skip): 

REM Optional: load_if_exists
set LOAD_IF_EXISTS_FLAG=

if "%STORAGE%"=="" goto SKIP_LOAD_QUESTION
if "%STUDY_NAME%"=="" goto SKIP_LOAD_QUESTION

set LOAD_EXISTING=
set /p LOAD_EXISTING=If study exists in storage, load it? (y/N): 
if /I "%LOAD_EXISTING%"=="y" set LOAD_IF_EXISTS_FLAG=--load-if-exists

:SKIP_LOAD_QUESTION

echo.
echo Running run_mvtec_optuna_study.py with custom parameters:
echo   mvtec-root     = %MVTEC_ROOT%
echo   categories     = %CATEGORIES%
echo   batch-size     = %BATCH_SIZE%
echo   n-trials       = %N_TRIALS%
echo   percentile-min = %P_MIN%
echo   percentile-max = %P_MAX%
echo   threshold-min  = %T_MIN%
echo   threshold-max  = %T_MAX%
echo   device         = %DEVICE%
if not "%STUDY_NAME%"=="" echo   study-name     = %STUDY_NAME%
if not "%STORAGE%"=="" echo   storage        = %STORAGE%
echo.

set CMD=%PYTHON_BIN% run_mvtec_optuna_study.py --mvtec-root "%MVTEC_ROOT%" --categories "%CATEGORIES%" --batch-size %BATCH_SIZE% --n-trials %N_TRIALS% --percentile-min %P_MIN% --percentile-max %P_MAX% --threshold-min %T_MIN% --threshold-max %T_MAX% --device %DEVICE%

if not "%STUDY_NAME%"=="" set CMD=%CMD% --study-name "%STUDY_NAME%"
if not "%STORAGE%"=="" set CMD=%CMD% --storage "%STORAGE%"
if not "%LOAD_IF_EXISTS_FLAG%"=="" set CMD=%CMD% %LOAD_IF_EXISTS_FLAG%

%CMD%
echo.
goto MAIN_MENU

:EXIT_SCRIPT
echo Exiting to shell.
endlocal
goto :EOF
