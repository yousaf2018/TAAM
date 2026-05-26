import os
import sys
import subprocess
import platform
import argparse

# --- CONFIGURATION ---
ENV_NAME = "sam3_tracker_venv"
PYTHON_REQUIRED = (3, 9)
ROOT_REQUIREMENTS = "requirements-win.txt"
SAM3_DIR = "sam3"
PIP_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"

TORCH_CUDA_URLS = [
    "https://download.pytorch.org/whl/cu121",
    "https://download.pytorch.org/whl/cu118"
]

def print_step(msg):
    print("\n" + "="*70)
    print(f"[*] {msg}")
    print("="*70 + "\n")

def run(cmd, error="Command failed", allow_fail=False, env=None):
    try:
        current_env = os.environ.copy()
        if env:
            current_env.update(env)
        subprocess.check_call(cmd, shell=True, env=current_env)
        return True
    except subprocess.CalledProcessError:
        if not allow_fail:
            print(f"\n[ERROR] {error}")
            sys.exit(1)
        return False

def check_python():
    print_step("Checking Python version...")
    print(f"Detected Python: {platform.python_version()}")
    if sys.version_info < PYTHON_REQUIRED:
        print(f"[ERROR] Python {PYTHON_REQUIRED[0]}.{PYTHON_REQUIRED[1]}+ required.")
        sys.exit(1)

def create_venv():
    print_step(f"Creating virtual environment: {ENV_NAME}")
    if not os.path.exists(ENV_NAME):
        run(f'python -m venv "{ENV_NAME}"')

def get_python():
    if platform.system().lower() == "windows":
        return os.path.abspath(os.path.join(ENV_NAME, "Scripts", "python.exe"))
    return os.path.abspath(os.path.join(ENV_NAME, "bin", "python"))

def install_torch(python):
    print_step("Installing PyTorch (CUDA Detection)...")
    has_gpu = run("nvidia-smi", allow_fail=True)
    if has_gpu:
        for url in TORCH_CUDA_URLS:
            if run(f'"{python}" -m pip install torch torchvision torchaudio --index-url {url}', allow_fail=True):
                code = "import torch; print(torch.cuda.is_available())"
                res = subprocess.run(f'"{python}" -c "{code}"', shell=True, capture_output=True, text=True)
                if "True" in res.stdout:
                    print("[SUCCESS] PyTorch with CUDA support installed.")
                    return
    run(f'"{python}" -m pip install torch torchvision torchaudio -i {PIP_MIRROR}')

def install_root_requirements(python):
    if os.path.exists(ROOT_REQUIREMENTS):
        print_step("Installing base requirements...")
        run(f'"{python}" -m pip install -r "{ROOT_REQUIREMENTS}" -i {PIP_MIRROR}')

def install_sam3(python):
    print_step("Installing SAM3 local modules...")
    if os.path.isdir(SAM3_DIR):
        run(f'cd "{SAM3_DIR}" && "{python}" -m pip install -e . -i {PIP_MIRROR}')

def force_upgrade_hf(python):
    print_step("Finalizing Hugging Face library...")
    run(f'"{python}" -m pip install --upgrade --no-cache-dir "huggingface_hub>=0.23.0" -i {PIP_MIRROR}')

def huggingface_login(python, token=None):
    print_step("Hugging Face Login...")
    active_token = token or os.environ.get("HF_TOKEN")
    if active_token:
        login_env = {"TEMP_HF_TOKEN": active_token}
        code = "import os; from huggingface_hub import login; t=os.environ.get('TEMP_HF_TOKEN'); login(token=t, add_to_git_credential=True)"
        if run(f'"{python}" -c "{code}"', "Login failed", env=login_env, allow_fail=True):
            print("[SUCCESS] Logged in successfully.")
            return
    
    print("[INFO] No valid token. Prompting interactive login...")
    try:
        subprocess.run(f'"{python}" -m huggingface_hub.login', shell=True, check=True)
    except subprocess.CalledProcessError:
        print("[WARNING] Login skipped.")

# --- NEW: DETAILED LOGGING FOR FUTURE USE ---
def print_finish_report():
    print("\n" + "#"*70)
    print(" " * 20 + "INSTALLATION COMPLETE")
    print("#"*70)
    
    abs_path = os.path.abspath(os.getcwd())
    venv_path = os.path.join(abs_path, ENV_NAME)
    
    print(f"\n[Project Location]: {abs_path}")
    print(f"[Virtual Env]:    {venv_path}")
    
    print("\n" + "-"*30 + " HOW TO USE THIS ENV " + "-"*30)
    
    # Check if user is in PowerShell or CMD
    is_powershell = "PSModulePath" in os.environ
    
    if is_powershell:
        print("\n>>> FOR POWERSHELL USERS:")
        print(f"    Step 1 (Activate):   .\\{ENV_NAME}\\Scripts\\Activate.ps1")
        print(f"    Step 2 (Run App):    python main.py")
    else:
        print("\n>>> FOR CMD USERS:")
        print(f"    Step 1 (Activate):   {ENV_NAME}\\Scripts\\activate.bat")
        print(f"    Step 2 (Run App):    python main.py")

    print("\n" + "-"*30 + " USEFUL COMMANDS " + "-"*30)
    print(f"Check GPU:       .\\{ENV_NAME}\\Scripts\\python -c \"import torch; print(f'CUDA: {{torch.cuda.is_available()}}')\"")
    print(f"Update Modules:  .\\{ENV_NAME}\\Scripts\\pip install -U -r {ROOT_REQUIREMENTS}")
    
    print("\n" + "#"*70 + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, help="Hugging Face Token")
    args = parser.parse_args()

    check_python()
    create_venv()
    python = get_python()

    run(f'"{python}" -m pip install --upgrade pip setuptools wheel -i {PIP_MIRROR}')
    
    install_torch(python)
    install_root_requirements(python)
    install_sam3(python)
    force_upgrade_hf(python)
    huggingface_login(python, args.hf_token)

    print_finish_report()

if __name__ == "__main__":
    main()