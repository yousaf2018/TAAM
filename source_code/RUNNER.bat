@echo off
setlocal
pushd "%~dp0"

TITLE SAM3 Tracker - Launcher

:: 1. Define Paths
set "VENV_DIR=%~dp0sam3_tracker_venv"
set "PY=%VENV_DIR%\Scripts\python.exe"
set "QT_BIN=%VENV_DIR%\Lib\site-packages\PyQt6\Qt6\bin"

:: 2. Verification
if not exist "%PY%" (
    echo [ERROR] Virtual Environment not found. Run INSTALLER.bat first.
    pause
    exit
)

:: 3. THE NUCLEAR PATH FIX
:: This isolates the app from other software (Anaconda, OBS, etc.) 
:: that causes the PyQt6 DLL conflict.
set "PATH=%VENV_DIR%\Scripts;%QT_BIN%;%VENV_DIR%\Lib\site-packages\PyQt6;C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem"

echo [*] Starting SAM3 Tracker...
"%PY%" main.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application crashed. 
    echo If you see a DLL error, try running INSTALLER.bat again.
    pause
)

popd