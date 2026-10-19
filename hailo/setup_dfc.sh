#!/usr/bin/env bash
# Install the Hailo Dataflow Compiler 5.4.0 (Hailo-10H) + Hailo Model Zoo in a WSL venv.
# No sudo here: pygraphviz (needs libgraphviz-dev) is skipped - it is only used by the
# graph visualizer - and torch is the CPU build (DFC uses it for model conversion only).
# GPU: the DFC pins TensorFlow 2.19, which ships no kernels for RTX 50-series (Blackwell,
# sm_120) GPUs. With TF's CUDA wheels installed it detects the GPU and then fails every op
# (CUDA_ERROR_INVALID_HANDLE), so they are deliberately NOT installed: optimization and the
# emulator run on the CPU (fine for optimization levels 0-1; fine-tuning levels are slow).
set -euo pipefail
H=$HOME/hailo
WHL=hailo_dataflow_compiler-5.4.0-py3-none-linux_x86_64.whl
mkdir -p "$H"
# the wheel comes from the Hailo Developer Zone (free account): put it next to this script or in $H
[ -f "$H/$WHL" ] || cp "$(dirname "$0")/$WHL" "$H/"
cd "$H"
[ -d venv ] || python3 -m venv venv
. venv/bin/activate
python -m pip install -q --upgrade pip uv
export UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=10 UV_CONCURRENT_DOWNLOADS=8

# retry N cmd...: flaky network; fails the script (set -e) once N attempts are used up
retry() { local n=$1; shift; for i in $(seq "$n"); do "$@" && return 0; echo "attempt $i/$n failed: $*" >&2; sleep 10; done; return 1; }

echo "== torch (CPU build)"
retry 4 uv pip install "torch==2.9.1" "torchvision==0.24.1" --index-url https://download.pytorch.org/whl/cpu

echo "== DFC dependencies (without pygraphviz / torch)"
unzip -p "$WHL" "*.dist-info/METADATA" | sed -n 's/^Requires-Dist: //p' \
  | grep -viE '^(pygraphviz|torch|torchvision)([^a-z]|$)' > dfc-reqs.txt
retry 4 uv pip install -r dfc-reqs.txt

echo "== DFC wheel"
uv pip install --no-deps "$H/$WHL"

echo "== Hailo Model Zoo"
[ -d hailo_model_zoo ] || retry 3 git clone --depth 1 https://github.com/hailo-ai/hailo_model_zoo.git
uv pip install --no-deps -e hailo_model_zoo
# its setup.py requirements (hailomz imports all of them, e.g. nuscenes for 3D models), with
# headless OpenCV for WSL; its pillow<=9.3.0 pin is left out (the DFC takes any Pillow)
retry 3 uv pip install "numba==0.59.0" imageio "matplotlib==3.5.2" opencv-python-headless scikit-learn \
  termcolor tqdm pycocotools lap "motmetrics==1.2.5" omegaconf "detection-tools==0.3" "scikit-image==0.20.0" \
  nuscenes-devkit "pyquaternion==0.9.9" "Shapely>=2.0.0"
# nuscenes-devkit pulls newer pyparsing/matplotlib/numpy/scipy. Restore the DFC's exact pins: its
# model-script grammar breaks on pyparsing 3.x ("Parsing failed at: >!<" on the second line).
uv pip install "pyparsing==2.4.7" "matplotlib==3.5.2" "numpy==1.26.4" "scipy==1.12.0"

echo "== check"
python - <<'EOF'
import importlib, sys
bad = 0
for m in ("hailo_sdk_client", "hailo_model_zoo", "tensorflow", "pycocotools", "cv2", "nuscenes", "matplotlib.pyplot"):
    try:
        mod = importlib.import_module(m)
        print("OK  ", m, getattr(mod, "__version__", ""))
    except Exception as e:
        bad += 1
        print("FAIL", m, type(e).__name__, e)
import pyparsing
if pyparsing.__version__ != "2.4.7":
    bad += 1
    print("FAIL pyparsing", pyparsing.__version__, "(the DFC model-script parser needs 2.4.7)")
from hailo_sdk_client import ClientRunner
r = ClientRunner(hw_arch="hailo10h")
print("ClientRunner OK")
sys.exit(1 if bad else 0)
EOF
hailo --version || true
echo SETUP_DONE
