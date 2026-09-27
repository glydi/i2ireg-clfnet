@echo off
REM Windows setup: creates .venv and installs all packages (nothing global).
REM   setup.bat        auto: CUDA build of PyTorch if an NVIDIA GPU is found, else CPU
REM   setup.bat cpu    force CPU-only PyTorch
setlocal
cd /d "%~dp0"
where py >nul 2>nul && (set "PYLAUNCH=py -3") || (set "PYLAUNCH=python")
%PYLAUNCH% -c "import sys; assert sys.version_info >= (3,10)" 2>nul || (echo Python 3.10+ required: https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^) & pause & exit /b 1)

if not exist ".venv\Scripts\python.exe" (
    echo ^>^> creating virtual environment .venv
    %PYLAUNCH% -m venv .venv || (pause & exit /b 1)
)
set "VPY=.venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip wheel

set "TORCH_IDX=https://download.pytorch.org/whl/cpu"
if /i not "%1"=="cpu" (
    where nvidia-smi >nul 2>nul && set "TORCH_IDX=https://download.pytorch.org/whl/cu124"
)
echo ^>^> installing PyTorch from %TORCH_IDX%
"%VPY%" -m pip install torch torchvision --index-url %TORCH_IDX% || (pause & exit /b 1)
"%VPY%" -m pip install -r requirements.txt || (pause & exit /b 1)
"%VPY%" -c "import torch; print('>> ready. torch', torch.__version__, '| device =', 'cuda' if torch.cuda.is_available() else 'cpu')"

set "SEVENZ="
where 7z >nul 2>nul && set "SEVENZ=1"
if exist "%ProgramFiles%\7-Zip\7z.exe" set "SEVENZ=1"
if not defined SEVENZ echo ^>^> NOTE: install 7-Zip (https://www.7-zip.org) - needed once to unpack the dataset .rar
echo ^>^> next: double-click start.bat
if "%~1"=="" if not defined FROM_START pause
