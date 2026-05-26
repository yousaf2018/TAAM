@echo off
setlocal EnableDelayedExpansion
pushd "%~dp0"

TITLE SAM3 Tracker - Professional Installer

echo ============================================================
echo        SAM3 TRACKER - INSTALLATION AND REPAIR
echo ============================================================

:: 1. Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10 and add it to PATH.
    pause
    exit
)

:: 2. Install System Dependencies (Fixes PyQt6 DLL errors)
echo [*] Checking for Microsoft Visual C++ Redistributable...
winget install --id Microsoft.VCRedist.2015+.x64 --silent --accept-source-agreements --accept-package-agreements >nul 2>&1

:: 3. Create Virtual Environment
set "VENV_DIR=%~dp0sam3_tracker_venv"
set "PY=%VENV_DIR%\Scripts\python.exe"

if not exist "%VENV_DIR%" (
    echo [*] Creating Virtual Environment...
    python -m venv sam3_tracker_venv
)

:: 4. Install Core Packages
echo [*] Upgrading pip...
"%PY%" -m pip install --upgrade pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple

echo [*] Installing PyTorch (CUDA 12.1 Support)...
"%PY%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo [*] Installing requirements-win.txt...
if exist "requirements-win.txt" "%PY%" -m pip install -r "requirements-win.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple

echo [*] Installing local SAM3 modules...
if exist "sam3" (
    pushd sam3
    "..\%PY%" -m pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
    popd
)

:: 5. THE PYQT6 DLL FIX
echo [*] Applying PyQt6 DLL Compatibility Patch...
"%PY%" -m pip uninstall -y PyQt6 PyQt6-Qt6 PyQt6-sip
"%PY%" -m pip install PyQt6==6.6.1 PyQt6-Qt6==6.6.1 PyQt6-sip --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple

:: 6. Hugging Face Login
echo.
echo ------------------------------------------------------------
echo HUGGING FACE LOGIN
echo ------------------------------------------------------------
set /p "HF_TOKEN=Paste your token (or press Enter to skip): "
if not "!HF_TOKEN!"=="" (
    set "TEMP_HF_TOKEN=!HF_TOKEN!"
    "%PY%" -c "import os; from huggingface_hub import login; login(token=os.environ.get('TEMP_HF_TOKEN'))"
)

echo.
echo ============================================================
echo INSTALLATION COMPLETE! 
echo Use 'RUNNER.bat' to start the application.
echo ============================================================
pause