"""
config.py — ConViT-X Configuration (Windows Inference)
Works on both Windows and Linux automatically.
No dataset paths needed — only inference settings matter here.
"""

import os
import torch
import platform

# ─── Base paths (relative to this file — works on any OS) ────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ─── Disease Labels (must match exactly what was used during training) ─────────
DISEASES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
NUM_CLASSES = len(DISEASES)

# ─── Model architecture (must match training settings exactly) ────────────────
IMAGE_SIZE       = 224
CNN_BACKBONE     = "densenet121"
VIT_MODEL        = "deit_small_patch16_224"
EMBED_DIM        = 384     # DeiT-Small embedding dimension
CNN_OUT_CHANNELS = 1024    # DenseNet-121 final feature channels
DROPOUT_RATE     = 0.3

# ─── X-Ray validation thresholds ─────────────────────────────────────────────
THRESHOLD           = 0.5   # sigmoid threshold for binary prediction display
XRAY_SAT_THRESHOLD  = 25    # max avg HSV saturation (X-rays are near-grayscale)
XRAY_GRAYNESS_RATIO = 0.85  # min fraction of pixels that must be near-gray

# ─── Device (auto-detected — no manual setting needed) ───────────────────────
#
# On Windows WITH  an NVIDIA GPU  → uses CUDA automatically
# On Windows WITHOUT a GPU        → uses CPU automatically  (slower but works)
# On Linux training server        → restricts to GPU 0 to avoid NCCL/MIG error
#
if platform.system() == "Linux":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count() if DEVICE == "cuda" else 0
