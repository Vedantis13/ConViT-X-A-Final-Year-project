"""
explainability.py — XAI for ConViT-X (Windows Inference)
Implements:
  - Grad-CAM++       : CNN spatial explanation (DenseNet-121 last dense block)
  - Attention Rollout: ViT global attention flow (DeiT-Small all 12 layers)
  - Fuse             : Weighted blend of both maps
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F

import config


# ─── Grad-CAM++ ───────────────────────────────────────────────────────────────

class GradCAMPlusPlus:
    """
    Grad-CAM++ hooks onto DenseNet-121's last dense block (denseblock4).
    Produces a class-discriminative heatmap showing which regions
    of the X-ray drove the prediction for a specific disease.
    """

    def __init__(self, model):
        # Unwrap DataParallel if present (Linux training artifact)
        self.model        = model.module if hasattr(model, "module") else model
        self.feature_maps = None
        self.gradients    = None
        self._register_hooks()

    def _register_hooks(self):
        target = self.model.cnn_features.denseblock4

        def fwd(module, inp, out):
            self.feature_maps = out.detach()

        def bwd(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target.register_forward_hook(fwd)
        target.register_full_backward_hook(bwd)

    def generate(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM++ heatmap for a given class.

        Args:
            input_tensor : [1, 3, H, W] tensor with requires_grad=True
            class_idx    : disease index (0-13); None = use top prediction

        Returns:
            numpy array [H, W] normalized to [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.sigmoid(logits).squeeze().argmax().item()

        self.model.zero_grad()
        logits[0, class_idx].backward(retain_graph=True)

        grads = self.gradients    # [1, C, h, w]
        fmaps = self.feature_maps  # [1, C, h, w]

        # Grad-CAM++ weight formula
        alpha_num = grads ** 2
        alpha_den = 2 * grads ** 2 + \
                    (fmaps * grads ** 3).sum(dim=(2, 3), keepdim=True) + 1e-8
        alpha   = alpha_num / alpha_den
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)

        cam = F.relu((weights * fmaps).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam,
            size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
            mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        mn, mx = cam.min(), cam.max()
        if mx - mn > 1e-8:
            cam = (cam - mn) / (mx - mn)
        return cam


# ─── Attention Rollout ────────────────────────────────────────────────────────

class AttentionRollout:
    """
    Attention Rollout propagates attention through all 12 DeiT-Small layers.
    Shows which image patches the model globally attended to — complementary
    to Grad-CAM++ which shows local CNN activations.
    """

    def __init__(self, model, discard_ratio=0.9, head_fusion="mean"):
        self.model         = model.module if hasattr(model, "module") else model
        self.discard_ratio = discard_ratio
        self.head_fusion   = head_fusion
        self.attn_weights  = []
        self._register_hooks()

    def _register_hooks(self):
        for block in self.model.transformer_blocks:
            block.attn.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        """Recompute attention weights from Q, K inside each transformer block."""
        with torch.no_grad():
            x        = input[0]
            B, N, C  = x.shape
            nh       = module.num_heads
            hd       = C // nh
            qkv = module.qkv(x).reshape(B, N, 3, nh, hd).permute(2, 0, 3, 1, 4)
            q, k, _  = qkv.unbind(0)
            attn     = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
            attn     = attn.softmax(dim=-1)
            self.attn_weights.append(attn.detach().cpu())

    def generate(self, input_tensor):
        """
        Generate attention rollout map.

        Args:
            input_tensor: [1, 3, H, W]

        Returns:
            numpy array [H, W] normalized to [0, 1]
        """
        self.attn_weights = []
        self.model.eval()

        with torch.no_grad():
            self.model(input_tensor)

        result = torch.eye(self.attn_weights[0].shape[-1])  # [50, 50]

        for attn in self.attn_weights:
            attn = attn[0]  # [heads, 50, 50]

            if self.head_fusion == "mean":
                af = attn.mean(0)
            elif self.head_fusion == "max":
                af = attn.max(0).values
            else:
                af = attn.min(0).values

            # Discard low-attention tokens
            flat   = af.view(-1)
            n_keep = int(flat.shape[0] * (1 - self.discard_ratio))
            if n_keep > 0:
                thresh = flat.topk(n_keep).values[-1]
                af = torch.where(af >= thresh, af, torch.zeros_like(af))

            I      = torch.eye(af.shape[-1])
            a      = (af + I) / 2
            a      = a / (a.sum(dim=-1, keepdim=True) + 1e-8)
            result = torch.matmul(a, result)

        mask = result[0, 1:].reshape(7, 7).numpy()  # [7, 7] patch grid
        mask = cv2.resize(mask, (config.IMAGE_SIZE, config.IMAGE_SIZE))

        mn, mx = mask.min(), mask.max()
        if mx - mn > 1e-8:
            mask = (mask - mn) / (mx - mn)
        return mask


# ─── Heatmap utilities ────────────────────────────────────────────────────────

def overlay_heatmap(original_rgb, cam, alpha=0.45):
    """
    Blend a heatmap over an original RGB image.

    Args:
        original_rgb : [H, W, 3] uint8 numpy array (RGB)
        cam          : [H, W] float [0, 1]
        alpha        : heatmap opacity

    Returns:
        [H, W, 3] uint8 numpy array (RGB)
    """
    heatmap_bgr = cv2.applyColorMap(
        (cam * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap_rgb + (1 - alpha) * original_rgb).astype(np.uint8)
    return overlay


def fuse_cams(gradcam, rollout, w_gradcam=0.6, w_rollout=0.4):
    """
    Weighted blend of Grad-CAM++ and Attention Rollout maps.
    Default: 60% Grad-CAM++ (local, more precise) + 40% Rollout (global context).
    """
    fused = w_gradcam * gradcam + w_rollout * rollout
    mn, mx = fused.min(), fused.max()
    if mx - mn > 1e-8:
        fused = (fused - mn) / (mx - mn)
    return fused
