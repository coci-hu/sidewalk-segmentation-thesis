@echo off
setlocal
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0..\.."
if defined SIDEWALK_PYTHON goto launch
if exist ".venv\Scripts\python.exe" set "SIDEWALK_PYTHON=%CD%\.venv\Scripts\python.exe"
if defined SIDEWALK_PYTHON goto launch
if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "SIDEWALK_PYTHON=%CONDA_PREFIX%\python.exe"
if defined SIDEWALK_PYTHON goto launch
if exist "E:\anaconda\envs\blind_road\python.exe" set "SIDEWALK_PYTHON=E:\anaconda\envs\blind_road\python.exe"
if not defined SIDEWALK_PYTHON set "SIDEWALK_PYTHON=python"
:launch
"%SIDEWALK_PYTHON%" "%~dp0app.py"
if errorlevel 1 (
    echo Launch failed. See applications/sidewalk_desktop/README.md for setup.
    pause
)
