# Promptist Reproduction Project

This repository contains a reproduction of Microsoft's **Promptist**, optimized for running locally on Windows with an NVIDIA RTX 4050 (6GB VRAM).

## 1. Project Overview & Credits
**Original Repository**: [microsoft/LMOps](https://github.com/microsoft/LMOps/tree/main/promptist)  
**Task**: We isolated the `promptist` component and modernized the inference pipeline to run offline on consumer hardware.

**Key Achievements**:
-   **Offline Inference**: All model weights are loaded from local checkpoints, not downloaded at runtime.
-   **Low-VRAM Optimization**: Implemented 8-bit quantization (`bitsandbytes`) and aggressive VRAM cleanup to fit both Promptist (GPT-2) and Stable Diffusion (v1.5) on a single 6GB GPU.
-   **Modern Stack**: Ported from PyTorch 1.x to PyTorch 2.6.0+cu124.

---

## 2. Directory Structure
```
PROMPTIST_reproduction/
├── checkpoints/                 # [EXCLUDED FROM GIT] Local model weights
│   ├── promptist_sft/           # microsoft/Promptist
│   └── clip/                    # openai/clip-vit-large-patch14
├── LMOps/                       # Original source code (sparse checkout)
├── batch_results_v2/            # Generated comparison images
├── batch_results/               # Intermediate batch results (JSON prompts)
├── .venv/                       # [EXCLUDED FROM GIT] Python Virtual Environment
├── PROJECT_REPORT.md            # Comprehensive Markdown Report
├── Promptist_Report.pdf         # PDF Export of the Report
├── Promptist_Report.html        # HTML Export of the Report
├── Swaraj_README.md             # This file
├── reproduce_results_v2.py      # Script: Basic Inference Verification
├── generate_report_images.py    # Script: End-to-End Image Pipeline
├── batch_test.py                # Script: Batch Optimization & Generation (v1)
├── save_batch_prompts.py        # Script: Extract Prompts to JSON
├── batch_test_v2.py             # Script: Batch Generation from JSON
├── export_report.py             # Script: Convert Report to PDF/HTML
└── setup_env.ps1                # Script: Environment Installation
```

---

## 3. Important Scripts we wrote
We developed several custom scripts to enable this reproduction:

### 3.1 Inference (`reproduce_results_v2.py`)
Verifies that the Promptist model loads correctly using 8-bit quantization and can generate optimized text.

### 3.2 End-to-End Generation (`generate_report_images.py`)
Runs the full pipeline:
1.  Loads Promptist (8-bit) -> Optimizes a prompt.
2.  **Clears VRAM** (Unloads model, runs Garbage Collection).
3.  Loads Stable Diffusion (float16) -> Generates verified images (`baseline.png`, `optimized.png`).

### 3.3 Batch Testing (`batch_test_v2.py`)
Generates images for a batch of 7 prompts defined in `batch_results/prompts.json`. This demonstrates robustness across different domains (Fantasy, Cyberpunk, etc.).

---

## 4. Setup & Installation
### Prerequisites
-   Windows 10/11
-   NVIDIA GPU (CUDA 11.8+ capable)
-   Python 3.11

### Installation
1.  **Environment**: We utilize a local virtual environment.
    ```powershell
    # Install dependencies (Torch 2.6, BitsAndBytes 0.49)
    ./setup_env.ps1
    ```
2.  **Patches**:
    -   **TRLX**: Modified `LMOps/promptist/trlx/setup.cfg` to remove the hard dependency on `deepspeed` (incompatible with Windows build tools).
    -   **Diffusers**: Patched `dynamic_modules_utils.py` to remove deprecated `huggingface_hub` imports.

---

## 5. Visual Reports
A full project report with images and diagrams is available in:
-   `PROJECT_REPORT.md` (Markdown source)
-   `Promptist_Report.pdf` (PDF export)
-   `Promptist_Report.html` (Web view)
