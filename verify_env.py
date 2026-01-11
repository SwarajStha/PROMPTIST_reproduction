import sys
import torch

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

try:
    import bitsandbytes as bnb
    print(f"BitsAndBytes Version: {bnb.__version__}")
    print("BitsAndBytes imported successfully.")
except ImportError as e:
    print(f"BitsAndBytes import failed: {e}")

try:
    import transformers
    print(f"Transformers Version: {transformers.__version__}")
except ImportError as e:
    print(f"Transformers import failed: {e}")

try:
    import diffusers
    print(f"Diffusers Version: {diffusers.__version__}")
except ImportError as e:
    print(f"Diffusers import failed: {e}")

try:
    import trlx
    print(f"TRLX Version: {trlx.__version__}")
except ImportError as e:
    print(f"TRLX import failed: {e}")
    # deepspeed failure might be evident here

