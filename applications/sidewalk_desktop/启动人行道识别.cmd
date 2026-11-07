@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if not exist "E:\anaconda\envs\blind_road\pythonw.exe" (
    echo Python environment not found. Use the verified blind_road environment.
    pause
    exit /b 1
)
start "" "E:\anaconda\envs\blind_road\pythonw.exe" "%~dp0app.py"
