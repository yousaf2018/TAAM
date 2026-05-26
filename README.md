
# TAAM: Track Any Aquatic Model

**TAAM** is a high-performance, professional-grade desktop application designed for the automated tracking and behavioral analysis of aquatic animals (Zebrafish, Medaka, Daphnia, etc.) in laboratory environments.

TAAM bridges the gap between large AI foundation models and real-time edge AI, combining:

🧠 **SAM 3 (Teacher Model)** — few-shot learning & automatic dataset generation  
⚡ **YOLO (Student Model)** — ultra-fast inference (100+ FPS) for real-time tracking

![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![AI](https://img.shields.io/badge/AI-SAM3%20%2B%20YOLO-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

<p align="center">
  <img src="https://github.com/yousaf2018/TAAM-Track-Any-Aquatic-Model/blob/main/source_code/assets/Logo.png" alt="TAAM Logo" width="200">
</p>

![TAAM GUI Snapshot](https://github.com/yousaf2018/TAAM-Track-Any-Aquatic-Model/blob/main/source_code/assets/TAAM-GUI.png)

---


## Acknowledgements

This application was developed in the **[Laboratory of Professor Chung-Der Hsiao](https://cdhsiao.weebly.com/pi-cv.html)** in collaboration with **Chung Yuan Christian University, Taiwan 🇹🇼**. Special credit and sincere gratitude are extended to **Professor Hsiao**, who shared his extensive research experience in biology and multiple domains, providing invaluable guidance and supervision throughout the development of this application.

<p align="center">
  <a href="https://www.cycu.edu.tw/">
    <img src="https://raw.githubusercontent.com/yousaf2018/EthoGrid/main/images/cycu.jpg" alt="Chung Yuan Christian University Logo" width="250">
  </a>
</p>

# 🌟 Key Features

- **Fully Automated AI Pipeline:** Train YOLO from just a few annotation clicks
- **Teacher–Student Architecture:** SAM 3 generates accurate datasets → YOLO learns fast tracking
- **YOLO-Only Mode:** Use existing SAM3 outputs to train YOLO directly
- **Batch Video Processing:** Process dozens or hundreds of videos automatically
- **Scientific Data Export:** High-precision CSVs with centroids, area, frame IDs, polygons, and image paths
- **VRAM Optimization:** Video chunking + CPU offload for stable 4K processing
- **Professional UI:** Dark-mode interface, side-by-side logging, project event monitoring

---

# 🚀 TAAM Workflow

## 1️⃣ PRE-PROCESSING
Split large lab videos into manageable segments for stable GPU processing.

## 2️⃣ ANNOTATION
Draw bounding boxes on a few frames. TAAM records them as few-shot prompts.

## 3️⃣ AI TRAINING PIPELINE

### 🧠 SAM 3 Stage
- Propagates masks across videos
- Tracks objects frame-by-frame
- Exports scientific CSV measurements
- Extracts frames to **sampling pool**: `Datasets/model_name/sampling_pool/`

> **Note:** Images are named to preserve video traceability: `OriginalVideoName_frame_000123.jpg`

### 🎯 YOLO Stage
- Cleans old train/val/test splits (but keeps `sampling_pool` intact)  
- Rebuilds dataset splits from sampling pool  
- Skips deleted SAM outputs  
- Handles first-time YOLO runs even if train/val/test folders do not exist  
- Supports Detection or Segmentation training automatically

## 4️⃣ DEPLOYMENT
Use trained models for high-speed tracking on new videos.

---

# 🗂️ Dataset Structure

```
Datasets/
└── model_name/
    ├── sampling_pool/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── data.yaml
```

---

# 🛠️ Installation

## 1. 📥 Initial Setup (Common for All)

Before choosing an installation method, you must clone the repository and enter the correct folder.

Open your terminal (PowerShell on Windows, Terminal on Linux) and run:

```bash
git clone https://github.com/yousaf2018/TAAM-Track-Any-Aquatic-Model.git
cd TAAM-Track-Any-Aquatic-Model/source_code
```

---

## 2. 🪟 Windows Installation

### Method A: Automated (Recommended)

We provide two script files that handle all the complex logic, including finding the right CUDA version and fixing PyQt6 errors.

**Step 1: Run INSTALLER.bat**  
Double-click it. It will ask for your Hugging Face token and set up the environment.

**Step 2: Run RUNNER.bat**  
Always use this to start the app. It prevents "DLL Load Failed" errors.

---

### Method B: Manual Installation (Step-by-Step)

If the `.bat` files fail, follow these steps exactly in PowerShell

---

### 1. Create the Environment
```powershell
python -m venv sam3_tracker_venv
.\sam3_tracker_venv\Scripts\Activate.ps1
```

---

### 2. Install PyTorch with CUDA (Compatibility Loop)

If you have an NVIDIA GPU, try these versions one by one until one works.

#### Try CUDA 12.1 (Latest)
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print('CUDA 12.1 Working:', torch.cuda.is_available())"
```

If result is `False`, try CUDA 11.8:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print('CUDA 11.8 Working:', torch.cuda.is_available())"
```

---

### 3. Install Dependencies
```powershell
pip install -r requirements-win.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e ./sam3
```

---

### 4. Hugging Face Login (Manual Token Method)
Replace `YOUR_TOKEN_HERE` with your actual token:

```powershell
python -c "from huggingface_hub import login; login(token='YOUR_TOKEN_HERE')"
```

---

### 5. Fix PyQt6 DLL Issues
If you see `ImportError: DLL load failed`, run:

```powershell
pip uninstall -y PyQt6 PyQt6-Qt6 PyQt6-sip
pip install PyQt6==6.6.1 PyQt6-Qt6==6.6.1 --no-cache-dir
```

---

## 3. 🐧 Linux Installation

Open terminal in the `source_code` directory:

---

### 1. Create and Activate Environment
```bash
python3 -m venv sam3_tracker_venv
source sam3_tracker_venv/bin/activate
```

---

### 2. Install PyTorch
For Linux with NVIDIA GPU:
```bash
pip install torch torchvision torchaudio
```

---

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e ./sam3
```

---

### 4. Hugging Face Login
```bash
huggingface-cli login
```

Or use token directly:

```bash
python3 -c "from huggingface_hub import login; login(token='YOUR_TOKEN_HERE')"
```

---

## 🛠️ Summary of Commands for Quick Copy

| Task | Windows (PowerShell) | Linux (Terminal) |
|------|----------------------|------------------|
| Clone | git clone https://github.com/yousaf2018/TAAM-Track-Any-Aquatic-Model.git | Same |
| Venv | python -m venv sam3_tracker_venv | python3 -m venv sam3_tracker_venv |
| Activate | .\sam3_tracker_venv\Scripts\Activate.ps1 | source sam3_tracker_venv/bin/activate |
| Torch | Use cu121 or cu118 URLs | pip install torch |
| Requirements | pip install -r requirements-win.txt | pip install -r requirements.txt |
| Local Mod | pip install -e ./sam3 | Same |
| Launch | python main.py | python3 main.py |

---

## ❓ Troubleshooting

### DLL Load Failed (Windows)
This is caused by conflicting Qt installations (Anaconda, OBS, drivers).

**Solution:** Use RUNNER.bat. If running manually:

```powershell
$env:PATH = "C:\Windows\system32;C:\Windows;D:\path\to\your\project\sam3_tracker_venv\Scripts"
python main.py
```

---

### Hugging Face Login Hangs
Use non-interactive login:

```bash
python -c "from huggingface_hub import login; login(token='your_token_here')"
```

---

### CUDA returns False
Install latest NVIDIA drivers from nvidia.com and restart your system.

****
# 📄 License
MIT License

---

# 👨‍🔬 Author
Mahmood Yousaf  
PhD Researcher — Biomedical Engineering  
Chung Yuan Christian University, Taiwan
