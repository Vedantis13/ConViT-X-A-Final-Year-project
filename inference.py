"""
inference.py — ConViT-X Inference Pipeline (Windows)
Handles:
  - Chest X-ray image validation
  - Preprocessing
  - Model prediction
  - XAI heatmap generation (Grad-CAM++ + Attention Rollout)
  - Base64 encoding for API response
"""

import io
import base64
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
from PIL import Image

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config
from model import ConViTX
from explainability import GradCAMPlusPlus, AttentionRollout, fuse_cams, overlay_heatmap


# ─── Preprocessing transform (inference — no augmentation) ───────────────────

_TRANSFORM = A.Compose([
    A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# ─── X-Ray Validation ─────────────────────────────────────────────────────────

def is_chest_xray(img_pil):
    """
    Heuristic validation that the uploaded image is a chest X-ray.

    Checks:
      1. Image dimensions (not too small or huge)
      2. Low colour saturation  (X-rays are grayscale)
      3. High grayness ratio    (R ≈ G ≈ B for most pixels)
      4. High dynamic range     (X-rays have strong contrast)

    Returns:
        (True,  "OK")               if it looks like an X-ray
        (False, "reason string")    if it does not
    """
    if img_pil is None:
        return False, "No image provided."

    w, h = img_pil.size
    if w < 64 or h < 64:
        return False, "Image is too small (minimum 64×64 pixels)."
    if w > 8192 or h > 8192:
        return False, "Image is too large (maximum 8192×8192 pixels)."

    img_rgb = np.array(img_pil.convert("RGB"), dtype=np.uint8)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Saturation check — X-rays have very low saturation
    avg_sat = float(img_hsv[:, :, 1].mean())
    if avg_sat > config.XRAY_SAT_THRESHOLD:
        return False, (
            f"Image looks like a colour photograph (saturation={avg_sat:.1f}). "
            "Please upload a grayscale chest X-ray."
        )

    # Grayness check — most pixels should have R ≈ G ≈ B
    r = img_rgb[:, :, 0].astype(int)
    g = img_rgb[:, :, 1].astype(int)
    b = img_rgb[:, :, 2].astype(int)
    gray_ratio = ((np.abs(r-g) < 20) & (np.abs(r-b) < 20) & (np.abs(g-b) < 20)).mean()
    if gray_ratio < config.XRAY_GRAYNESS_RATIO:
        return False, (
            f"Image does not appear to be an X-ray "
            f"(only {gray_ratio*100:.1f}% near-gray pixels). "
            "Please upload a chest X-ray."
        )

    # Dynamic range check — X-rays have strong contrast
    gray = np.array(img_pil.convert("L"), dtype=np.float32)
    dynamic_range = float(np.percentile(gray, 95) - np.percentile(gray, 5))
    if dynamic_range < 30:
        return False, "Image has very low contrast. Please upload a clear chest X-ray."

    return True, "OK"


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess(img_pil):
    """
    Convert PIL image to model input tensor + resized original numpy array.

    Returns:
        tensor   : [1, 3, 224, 224] float32
        orig_np  : [224, 224, 3]    uint8  RGB  (for heatmap overlay)
    """
    img_rgb  = np.array(img_pil.convert("RGB"), dtype=np.uint8)
    orig_np  = cv2.resize(img_rgb, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    tensor   = _TRANSFORM(image=img_rgb)["image"].unsqueeze(0)  # [1,3,H,W]
    return tensor, orig_np


# ─── Inference Engine ─────────────────────────────────────────────────────────

class ConViTXInference:
    """
    Loads the trained ConViT-X model and exposes a single predict() method.

    Usage:
        engine = ConViTXInference("checkpoints/convitx_best.pth")
        result = engine.predict(pil_image)

    The engine is designed to be created once and reused for every request.
    """

    def __init__(self, checkpoint_path, device=None):
        self.device = torch.device(device or config.DEVICE)
        print(f"  Loading model on {str(self.device).upper()} ...")
        self.model        = self._load(checkpoint_path)
        self.grad_cam     = GradCAMPlusPlus(self.model)
        self.attn_rollout = AttentionRollout(self.model)
        print(f"  Model ready.")

    def _load(self, path):
        """
        Load checkpoint weights into ConViTX architecture.
        Handles:
          - checkpoints saved with DataParallel (module. prefix)
          - checkpoints wrapped in a dict with 'model_state' key
          - both old and new torch.load API (weights_only argument)
        """
        model = ConViTX(
            num_classes=config.NUM_CLASSES,
            pretrained=False,       # weights come from checkpoint, not internet
            dropout_rate=config.DROPOUT_RATE,
        )

        # Load checkpoint — try weights_only=False first for newer PyTorch
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            # Older PyTorch does not have weights_only parameter
            ckpt = torch.load(path, map_location=self.device)

        # Extract state dict regardless of how checkpoint was saved
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            state = ckpt["model_state"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt  # raw state dict

        # Remove DataParallel 'module.' prefix if present
        state = {k.replace("module.", ""): v for k, v in state.items()}

        model.load_state_dict(state, strict=False)
        model.to(self.device)
        model.eval()
        return model

    def predict(self, img_pil):
        """
        Run full inference pipeline on a PIL image.

        Args:
            img_pil: PIL.Image (any mode — converted to RGB internally)

        Returns dict with keys:
            predictions  : {disease_name: probability_0_to_100, ...}
            top_findings : [(disease_name, probability), ...] sorted desc
            xai_combined : base64 PNG — fused Grad-CAM++ + Attention Rollout
            xai_gradcam  : base64 PNG — Grad-CAM++ only
            xai_rollout  : base64 PNG — Attention Rollout only
            original_img : base64 PNG — original X-ray resized to 224x224
            top_disease  : name of highest-confidence disease
        """
        tensor, orig_np = preprocess(img_pil)
        tensor = tensor.to(self.device)

        # ── Predictions ───────────────────────────────────────────────────────
        with torch.no_grad():
            logits = self.model(tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()  # [14]

        predictions  = {d: round(float(probs[i]) * 100, 2) for i, d in enumerate(config.DISEASES)}
        top_findings = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        top_idx      = int(np.argmax(probs))

        # ── XAI heatmaps ──────────────────────────────────────────────────────
        # Grad-CAM++ for highest-confidence disease
        inp_grad    = tensor.clone().requires_grad_(True)
        gradcam_map = self.grad_cam.generate(inp_grad, class_idx=top_idx)

        # Attention Rollout (global context)
        rollout_map = self.attn_rollout.generate(tensor)

        # Fused (60% Grad-CAM + 40% Rollout)
        fused_map = fuse_cams(gradcam_map, rollout_map)

        return {
            "predictions":  predictions,
            "top_findings": top_findings,
            "top_disease":  config.DISEASES[top_idx],
            "xai_combined": _to_b64(overlay_heatmap(orig_np, fused_map)),
            "xai_gradcam":  _to_b64(overlay_heatmap(orig_np, gradcam_map)),
            "xai_rollout":  _to_b64(overlay_heatmap(orig_np, rollout_map)),
            "original_img": _to_b64(orig_np),
        }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _to_b64(img_np):
    """Convert RGB numpy array to base64-encoded PNG string."""
    pil = Image.fromarray(img_np.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def pil_to_base64(img_pil, fmt="PNG"):
    """Convert a PIL image to base64 string."""
    buf = io.BytesIO()
    img_pil.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
