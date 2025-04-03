import torch

def get_torch_version():
    """Get PyTorch version as a tuple of integers"""
    version_str = torch.__version__.split('+')[0]
    return tuple(map(int, version_str.split('.')))

def set_memory_allocation(gpu_fraction):
    """Set GPU memory allocation based on PyTorch version"""
    if gpu_fraction < 1.0:
        try:
            torch.cuda.set_per_process_memory_fraction(gpu_fraction)
            print(f"GPU memory fraction set to {gpu_fraction}")
        except Exception as e:
            print(f"Could not set GPU memory fraction: {e}")
            print(f"Current torch version: {torch.__version__}")

def load_state_dict(model, state_dict):
    """Handle loading state dict across different PyTorch versions"""
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        # Try to handle incompatible keys between versions
        if "unexpected key" in str(e) or "missing key" in str(e):
            # Filter out problematic keys or use strict=False
            model.load_state_dict(state_dict, strict=False)
            print("Warning: Model loaded with strict=False due to incompatible keys")
        else:
            raise e