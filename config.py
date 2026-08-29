"""
config.py

Central path configuration. Every module imports paths from here instead of hardcoding
absolute paths -- carried forward from Week 2/3/4 review feedback (hardcoded '/home/claude/
...' paths broke single-command reproducibility on any other machine). PROJECT_ROOT is
computed relative to THIS file's own location, so this project runs correctly no matter
where it is cloned or extracted.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SRC_DIR = PROJECT_ROOT / "src"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
RESULTS_DIR = PROJECT_ROOT / "results"

MNIST_CACHE_PATH = DATA_DIR / "mnist.npz"

BASELINE_MODEL_PATH = MODELS_DIR / "baseline_fp32.pt"
PRUNED_MODEL_PATH = MODELS_DIR / "pruned.pt"
PRUNED_FINETUNED_MODEL_PATH = MODELS_DIR / "pruned_finetuned.pt"
QUANTIZED_MODEL_PATH = MODELS_DIR / "quantized_int8.pt"
COMBINED_MODEL_PATH = MODELS_DIR / "pruned_quantized_combined.pt"

BASELINE_ONNX_PATH = MODELS_DIR / "baseline_fp32.onnx"
COMBINED_ONNX_PATH = MODELS_DIR / "pruned_quantized_combined.onnx"

SEED = 42

for d in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
