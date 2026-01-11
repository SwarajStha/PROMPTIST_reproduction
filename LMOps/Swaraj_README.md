# Project Setup & Workarounds Log

**Repository**: [microsoft/LMOps](https://github.com/microsoft/LMOps)  
**Focus Directory**: `promptist`

## 1. Repository Management
- **Sparse Checkout**: To avoid issues with long filenames in unrelated directories (`RRM-evaluation`), we utilized git sparse-checkout to purely target the `promptist` directory.
- **Location**: All relevant code is located in `LMOps/promptist`.

## 2. Environment Setup (Windows + RTX 4050)
The original `README.md` requirements were outdated (PyTorch 1.x) and incompatible with modern hardware (RTX 4050 requires CUDA 11.8+).

### System Configuration
- **Operating System**: Windows
- **GPU**: NVIDIA RTX 4050 (6GB VRAM)
- **Python Version**: **3.11.0** (Selected over system default 3.14 for compatibility)

### Dependencies
We created a custom `setup_env.ps1` script to install the following:
1.  **PyTorch 2.6.0+cu124**: Upgraded from 1.12 to support Ada Lovelace architecture.
2.  **BitsAndBytes (Windows)**: Installed `bitsandbytes` (0.49.1) to enable 8-bit quantization on Windows.
3.  **Transformers**: Upgraded to a recent version compatible with PyTorch 2.x and `bitsandbytes`.

### Workarounds & Patches
- **TRLX Installation**:
    - **Issue**: `trlx` requires `deepspeed`, which is difficult to build on Windows.
    - **Fix**: We modified `LMOps/promptist/trlx/setup.cfg` to remove `deepspeed>=0.7.3` from `install_requires`.
    - **Result**: `trlx` installed successfully, enabling usage of the library components that do not rely on DeepSpeed (primarily inference).

## 3. Usage
- **Virtual Environment**: Located in `.venv`.
- **Activation**: `.venv\Scripts\Activate.ps1`
