#!/bin/bash
# ComfyUI-Goofer + UCLA + DMM — complete RunPod installer
# Usage: bash runpod-install.sh
# Installs all three projects with proper CUDA-version-aware TensorRT setup

set -e

COMFYUI_ROOT="/workspace/runpod-slim/ComfyUI"
CUSTOM_NODES="$COMFYUI_ROOT/custom_nodes"
VENV="$COMFYUI_ROOT/.venv-cu128/bin/python"

echo "=========================================="
echo "ComfyUI Multi-Project Installer (RunPod)"
echo "=========================================="

# Check if ComfyUI exists
if [ ! -d "$COMFYUI_ROOT" ]; then
    echo "ERROR: ComfyUI not found at $COMFYUI_ROOT"
    echo "This script assumes ComfyUI is already installed."
    exit 1
fi

echo "[1/5] Cloning projects into custom_nodes..."
cd "$CUSTOM_NODES"

for repo in ComfyUI-Goofer ComfyUI-UCLA-News-Video comfyui-data-media-machine; do
    if [ -d "$repo" ]; then
        echo "  ✓ $repo already exists, pulling latest..."
        cd "$repo"
        git pull
        cd ..
    else
        echo "  → Cloning $repo..."
        git clone "https://github.com/jbrick2070/$repo.git"
    fi
done

echo ""
echo "[2/5] Installing base requirements..."
"$VENV" -m pip install -r "$CUSTOM_NODES/ComfyUI-Goofer/requirements.txt" -q

echo ""
echo "[3/5] Installing UCLA requirements..."
"$VENV" -m pip install -r "$CUSTOM_NODES/ComfyUI-UCLA-News-Video/requirements.txt" -q 2>/dev/null || echo "  (no requirements.txt, using base)"

echo ""
echo "[4/5] Installing DMM requirements..."
"$VENV" -m pip install -r "$CUSTOM_NODES/comfyui-data-media-machine/requirements.txt" -q 2>/dev/null || echo "  (no requirements.txt, using base)"

echo ""
echo "[5/5] Installing TensorRT for NVIDIA acceleration..."

# Detect CUDA version
CUDA_VERSION=$($VENV -c "import torch; print(''.join(torch.__version__.split('+')[1].split('.')[:2]))" 2>/dev/null || echo "")

if [ -z "$CUDA_VERSION" ]; then
    # Try nvcc
    CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+' | sed 's/\.//g' | head -c 3)
fi

if [ "$CUDA_VERSION" = "130" ] || [ "$CUDA_VERSION" = "129" ]; then
    echo "  → CUDA 13.x detected: installing torch-tensorrt..."
    "$VENV" -m pip install torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu130 -q
    echo "  ✓ TensorRT installed (NVIDIA tensor core optimization enabled)"
elif [ "$CUDA_VERSION" = "128" ]; then
    echo "  → CUDA 12.8 detected: TensorRT requires CUDA 13+"
    echo "  → Falling back to Real-ESRGAN (still NVIDIA GPU-accelerated)"
else
    echo "  → CUDA version unknown, skipping TensorRT"
fi

echo ""
echo "=========================================="
echo "✓ Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Restart ComfyUI:"
echo "     pkill -f comfyui || true"
echo "     cd $COMFYUI_ROOT && python main.py"
echo ""
echo "  2. Load a workflow:"
echo "     - Goofer: custom_nodes/ComfyUI-Goofer/example_workflows/goofer_blackwell_linux.json"
echo "     - UCLA: custom_nodes/ComfyUI-UCLA-News-Video/workflows/"
echo ""
echo "  3. Check logs for upscaler:"
echo "     - 'NVIDIA TensorRT ESRGAN' = TensorRT active (CUDA 13+)"
echo "     - 'Real-ESRGAN' = Fallback (still NVIDIA GPU-accelerated)"
echo ""
