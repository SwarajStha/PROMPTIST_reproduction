# Create virtual environment using Python 3.11
py -3.11 -m venv .venv

# Install PyTorch with CUDA 12.4
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install bitsandbytes (Windows compatible)
.venv\Scripts\python.exe -m pip install bitsandbytes

# Install other dependencies
# Removing specific versions to allow compatibility with PyTorch 2.x
.venv\Scripts\python.exe -m pip install transformers accelerate ftfy regex tqdm scipy pytorch-lightning

# Install local packages in editable mode
# Note: trlx might fail due to deepspeed on Windows, we'll attempt it
.venv\Scripts\python.exe -m pip install -e ./LMOps/promptist/diffusers
.venv\Scripts\python.exe -m pip install -e ./LMOps/promptist/trlx
