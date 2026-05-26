import os
import sys
import subprocess
import platform

# --- CONFIGURATION ---
ENV_NAME = "sam3_tracker_venv"
PYTHON_REQUIRED = (3, 9)
ROOT_REQUIREMENTS = "requirements.txt"
SAM3_DIR = "sam3"
SAM3_REQUIREMENTS = "requirements.txt"

# Taiwan fast PyPI mirror
PIP_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"

# --- HELPERS ---
def print_step(msg):
    print("\n" + "="*60)
    print(f"[*] {msg}")
    print("="*60 + "\n")

def run(cmd, error="Command failed"):
    """Run a shell command safely, exits on failure."""
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError:
        print(f"\n[ERROR] {error}")
        sys.exit(1)

def check_python():
    print_step("Checking Python version...")
    if sys.version_info < PYTHON_REQUIRED:
        print(f"[ERROR] Python {PYTHON_REQUIRED[0]}.{PYTHON_REQUIRED[1]}+ required.")
        sys.exit(1)
    print(f"Python version OK: {platform.python_version()}")

def create_venv():
    print_step(f"Creating virtual environment: {ENV_NAME}")
    run(f'python -m venv "{ENV_NAME}"', "Failed to create virtual environment.")

def get_pip():
    """Return absolute path to pip inside virtual environment."""
    if platform.system().lower() == "windows":
        pip_rel = os.path.join(ENV_NAME, "Scripts", "pip")
    else:
        pip_rel = os.path.join(ENV_NAME, "bin", "pip")
    return os.path.abspath(pip_rel)

def install_root_requirements(pip):
    """Install main requirements.txt if exists."""
    if os.path.exists(ROOT_REQUIREMENTS):
        print_step("Installing root requirements...")
        run(
            f'"{pip}" install -r "{ROOT_REQUIREMENTS}" '
            f'--prefer-binary --no-cache-dir --timeout 100 -i {PIP_MIRROR}',
            "Failed installing root requirements"
        )
    else:
        print("No root requirements.txt found — skipping")

def install_sam3(pip):
    """Install SAM3 requirements and package."""
    print_step("Installing SAM3 modules (cd into folder)...")

    if not os.path.isdir(SAM3_DIR):
        print(f"[ERROR] '{SAM3_DIR}' folder not found.")
        sys.exit(1)

    sam3_req_path = os.path.join(SAM3_DIR, SAM3_REQUIREMENTS)

    # Install SAM3 requirements and package using shell navigation
    cmd = (
        f'cd "{SAM3_DIR}" && '
        f'"{pip}" install -e . -i {PIP_MIRROR} && '
        f'cd ..'
    )

    run(cmd, "Failed installing SAM3 modules")

def finish():
    print_step("INSTALLATION COMPLETE")

    if platform.system().lower() == "windows":
        activate = f"{ENV_NAME}\\Scripts\\activate"
    else:
        activate = f"source {ENV_NAME}/bin/activate"

    print("To start using the environment:\n")
    print(f"1️⃣ Activate it:\n   {activate}")
    print("2️⃣ Run your SAM3 app:\n   python main.py")

# --- MAIN ---
def main():
    print(f"Detected OS: {platform.system()}")
    check_python()
    create_venv()

    pip = get_pip()

    print_step("Upgrading pip tools...")
    run(f'"{pip}" install --upgrade pip setuptools wheel -i {PIP_MIRROR}',
        "Failed upgrading pip")

    install_root_requirements(pip)
    install_sam3(pip)
    finish()

if __name__ == "__main__":
    main()