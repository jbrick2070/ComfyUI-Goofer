"""
ComfyUI-Goofer auto-installer.

ComfyUI-Manager executes this file when the node pack is first loaded.
It installs base requirements and, on Linux with NVIDIA GPUs, also
installs torch-tensorrt from the correct CUDA-versioned wheel index
so the TensorRT upscaler path activates instead of falling back to
plain Real-ESRGAN.

Author: Jeffrey A. Brick
"""

import subprocess
import sys
import platform
import os
import shutil


def _pip(*args):
    """Run pip in the current Python environment."""
    subprocess.check_call([sys.executable, "-m", "pip", *args])


def _get_cuda_version():
    """Detect CUDA toolkit version from nvcc. Returns e.g. '128' or None."""
    nvcc = shutil.which("nvcc")
    if not nvcc:
        # Check common CUDA install paths
        for path in ["/usr/local/cuda/bin/nvcc", "/usr/bin/nvcc"]:
            if os.path.isfile(path):
                nvcc = path
                break
    if not nvcc:
        return None
    try:
        out = subprocess.check_output([nvcc, "--version"], text=True)
        # Parse "release 12.8" -> "128"
        for line in out.splitlines():
            if "release" in line.lower():
                parts = line.split("release")[-1].strip().split(",")[0].strip()
                major, minor = parts.split(".")[:2]
                return f"{major}{minor}"
    except Exception:
        pass
    return None


def install():
    print("[ComfyUI-Goofer] Installing base requirements...")
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    _pip("install", "-r", req_file)

    # On Linux with NVIDIA GPU, install TensorRT with correct CUDA index
    if platform.system() == "Linux":
        cuda_ver = _get_cuda_version()
        if cuda_ver:
            index_url = f"https://download.pytorch.org/whl/cu{cuda_ver}"
            print(f"[ComfyUI-Goofer] Linux detected, CUDA {cuda_ver}")
            print(f"[ComfyUI-Goofer] Installing torch-tensorrt from {index_url}")
            try:
                _pip(
                    "install", "torch-tensorrt", "torchaudio",
                    "--extra-index-url", index_url,
                )
                print("[ComfyUI-Goofer] TensorRT installed — NVIDIA upscaler will activate.")
            except subprocess.CalledProcessError as e:
                print(f"[ComfyUI-Goofer] WARNING: torch-tensorrt install failed: {e}")
                print("[ComfyUI-Goofer] Upscaler will fall back to Real-ESRGAN.")
        else:
            print("[ComfyUI-Goofer] No CUDA toolkit found (nvcc missing).")
            print("[ComfyUI-Goofer] Skipping torch-tensorrt. Upscaler will use Real-ESRGAN.")
    else:
        print("[ComfyUI-Goofer] Windows detected — using nvvfx RTX VSR (no torch-tensorrt needed).")

    print("[ComfyUI-Goofer] Install complete.")


if __name__ == "__main__":
    install()
