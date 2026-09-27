@echo off
REM Windows one-click start: double-click start.bat (or run: start.bat ui / quick / full / folder PATH)
REM Needs Python 3.10+ from python.org ("Add python.exe to PATH" ticked) and 7-Zip for the dataset .rar.
setlocal
cd /d "%~dp0"

".venv\Scripts\python.exe" -c "import torch, gradio" 2>nul
if errorlevel 1 (
    echo ^>^> first run: installing packages ^(a few minutes^)...
    set "FROM_START=1"
    call setup.bat
)

".venv\Scripts\python.exe" run.py %*
pause
