@echo off
REM Build CoverSwap.exe and split models + exe into git-friendly 1 MB parts.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Create the venv first: py -3.12 -m venv .venv
    exit /b 1
)

echo [1/4] Installing PyInstaller...
".venv\Scripts\python.exe" -m pip install pyinstaller -q
if errorlevel 1 exit /b 1

echo [2/4] Building CoverSwap.exe (one-file, may take several minutes)...
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean CoverSwap.spec
if errorlevel 1 exit /b 1

if not exist "dist\CoverSwap.exe" (
    echo Build failed: dist\CoverSwap.exe not found
    exit /b 1
)

echo [3/4] Splitting weights/ into parts/...
".venv\Scripts\python.exe" chunker.py split
if errorlevel 1 exit /b 1

echo [4/4] Splitting dist\CoverSwap.exe into local runableparts/ (not committed)...
".venv\Scripts\python.exe" chunker.py split --src dist\CoverSwap.exe --parts runableparts
if errorlevel 1 exit /b 1

echo.
echo Done.
echo   parts/         model weights as 1 MB chunks
echo   runableparts/  CoverSwap.exe as 1 MB chunks
echo.
echo To run from parts:
echo   python chunker.py merge
echo   python chunker.py merge --parts runableparts
echo   dist\CoverSwap.exe
