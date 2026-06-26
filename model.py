"""
model.py — ConViT-X Architecture
Used for INFERENCE only on Windows.
pretrained=False by default so no internet connection is needed.
The trained weights are loaded from convitx_best.pth instead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm

import config


# ─── Label Correlation Module (GAT-based) ─────────────────────────────────────

class GraphAttentionLayer(nn.Module):
    """Single Graph Attention layer for disease label correlation."""

    def __init__(self, in_features, out_features, num_nodes, dropout=0.1):
        super().__init__()
        self.W       = nn.Linear(in_features, out_features, bias=False)
        self.a_src   = nn.Linear(out_features, 1, bias=False)
        self.a_dst   = nn.Linear(out_features, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leaky   = nn.LeakyReLU(0.2)

    def forward(self, h, adj):
        Wh    = self.W(h)
        e_src = self.a_src(Wh)
        e_dst = self.a_dst(Wh)
        e     = self.leaky(e_src + e_dst.transpose(-2, -1))
        zero  = -9e15 * torch.ones_like(e)
        attn  = torch.where(adj > 0, e, zero)
        attn  = F.softmax(attn, dim=-1)
        attn  = self.dropout(attn)
        return F.elu(torch.matmul(attn, Wh)), attn


class LabelCorrelationModule(nn.Module):
    """
    GAT-based Label Correlation Module.
    Refines raw logits using learned disease co-occurrence relationships.
    """

    def __init__(self, num_classes=14, hidden_dim=64, cooc_matrix=None):
        super().__init__()
        self.num_classes = num_classes

        if cooc_matrix is not None:
            adj_init = torch.tensor(cooc_matrix, dtype=torch.float32)
        else:
            adj_init = torch.eye(num_classes) * 0.5 + 0.05

        self.adj_raw    = nn.Parameter(adj_init)
        self.node_embed = nn.Embedding(num_classes, hidden_dim)
        self.gat        = GraphAttentionLayer(hidden_dim, hidden_dim, num_classes, dropout=0.1)
        self.proj       = nn.Linear(hidden_dim, 1)
        self.alpha      = nn.Parameter(torch.tensor(0.3))

    def forward(self, logits):
        B, C    = logits.shape
        adj     = torch.sigmoid(self.adj_raw)
        node_ids = torch.arange(C, device=logits.device)
        h       = self.node_embed(node_ids)
        h_out, _ = self.gat(h, adj)
        bias    = self.proj(h_out).squeeze(-1)
        confidence  = torch.sigmoid(logits)
        refinement  = confidence * bias.unsqueeze(0)
        return logits + torch.clamp(self.alpha, 0, 1) * refinement


# ─── ConViT-X Main Model ──────────────────────────────────────────────────────

class ConViTX(nn.Module):
    """
    ConViT-X: Explainable CNN-ViT model with Label Correlation.

    Architecture:
        Input (224x224x3)
          -> DenseNet-121 CNN       [B, 1024, 7, 7]  local features
          -> 1x1 Conv projection    [B, 384,  7, 7]
          -> DeiT-Small Transformer [B, 50, 384]     global context
          -> CLS token              [B, 384]
          -> Linear classifier      [B, 14]           raw logits
          -> Label Correlation GAT  [B, 14]           refined logits
    """

    def __init__(self, num_classes=14, pretrained=False,
                 dropout_rate=0.3, cooc_matrix=None):
        super().__init__()
        self.num_classes = num_classes

        # ── DenseNet-121 backbone ─────────────────────────────────────────────
        # pretrained=False: weights come from checkpoint, not ImageNet download
        densenet = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.cnn_features = densenet.features  # [B, 1024, 7, 7] for 224x224 input

        # ── Feature projection: CNN dim -> ViT dim ────────────────────────────
        self.feature_proj = nn.Sequential(
            nn.Conv2d(config.CNN_OUT_CHANNELS, config.EMBED_DIM, kernel_size=1, bias=False),
            nn.BatchNorm2d(config.EMBED_DIM),
            nn.GELU()
        )

        # ── DeiT-Small transformer ────────────────────────────────────────────
        # pretrained=False: weights come from checkpoint, not timm download
        deit = timm.create_model(config.VIT_MODEL, pretrained=pretrained, num_classes=0)
        self.transformer_blocks = deit.blocks
        self.transformer_norm   = deit.norm

        # Learnable CLS token + positional embeddings (49 patches + 1 CLS = 50)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.EMBED_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, 50, config.EMBED_DIM))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── Label Correlation Module ──────────────────────────────────────────
        self.lcm = LabelCorrelationModule(
            num_classes=num_classes,
            hidden_dim=64,
            cooc_matrix=cooc_matrix
        )

        # ── Classification head ───────────────────────────────────────────────
        self.pre_logits = nn.Sequential(
            nn.LayerNorm(config.EMBED_DIM),
            nn.Dropout(dropout_rate),
        )
        self.classifier = nn.Linear(config.EMBED_DIM, num_classes)

        # Hooks list (populated by explainability.py)
        self._hooks = []

    def forward(self, x):
        B = x.shape[0]

        # 1. CNN local features
        cnn_out = self.cnn_features(x)                    # [B, 1024, 7, 7]

        # 2. Project to transformer dim
        proj = self.feature_proj(cnn_out)                 # [B, 384, 7, 7]

        # 3. Flatten to patch tokens
        tokens = proj.flatten(2).transpose(1, 2)          # [B, 49, 384]

        # 4. Prepend CLS token
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)          # [B, 50, 384]

        # 5. Add positional embeddings
        tokens = tokens + self.pos_embed

        # 6. Transformer blocks
        for block in self.transformer_blocks:
            tokens = block(tokens)
        tokens = self.transformer_norm(tokens)

        # 7. CLS token -> classification
        cls_out = self.pre_logits(tokens[:, 0])
        logits  = self.classifier(cls_out)                # [B, 14]

        # 8. Label correlation refinement
        logits = self.lcm(logits)

        return logits
