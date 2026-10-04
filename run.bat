@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo CoverSwap is not set up yet. Ask for help installing it.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" main.py
if errorlevel 1 pause
