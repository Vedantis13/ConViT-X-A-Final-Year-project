# ConViT-X-A-Final-Year-project

# ConViT-X: Explainable CNN–ViT for Multi-Label Chest X-Ray Diagnosis

> **DenseNet-121 + DeiT-Small + Label Correlation Module + Grad-CAM++ + Attention Rollout**

---

## System Requirements
- 2× NVIDIA H200 NVL (CUDA 13.1, Driver 590.48)
- Python 3.10+
- PyTorch 2.1+

---

## Project Structure
```
convitx/
├── config.py           # All hyperparameters and paths
├── dataset.py          # NIH ChestX-ray14 loader + augmentation
├── model.py            # ConViT-X architecture
├── losses.py           # Weighted BCE + Focal Loss
├── metrics.py          # AUROC, F1, precision, recall
├── explainability.py   # Grad-CAM++ and Attention Rollout
├── train.py            # Training pipeline + graph generation
├── inference.py        # Inference engine + X-ray validation
├── app.py              # Flask REST API backend
├── frontend/
│   └── index.html      # Web UI (upload/camera, XAI viewer)
├── checkpoints/        # Saved model weights (created at training)
└── outputs/
    └── graphs/         # Training graphs (created at training)
```

---

## Dataset Setup

Download from Kaggle:  
https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized

Expected layout:
```
/data/nih_chestxray14/
    images/                   (112,120 × 224×224 PNG files)
    Data_Entry_2017.csv
    train_val_list.txt         (official train split)
    test_list.txt              (official test split)
```

Set `CHESTXRAY_DIR` environment variable or edit `config.py`:
```bash
export CHESTXRAY_DIR=/data/nih_chestxray14
```

---

## Installation

```bash
cd convitx
pip install -r requirements.txt --break-system-packages
```

---

## Training

```bash
python train.py --data-dir /data/nih_chestxray14 --epochs 60 --batch-size 64
```

Training will:
- Auto-detect 2 H200 GPUs (DataParallel)
- Use mixed precision (AMP / FP16)
- Apply weighted sampling + Focal + BCE loss
- Save best checkpoint to `checkpoints/convitx_best.pth`
- Generate all graphs in `outputs/graphs/`

### Generated Graphs
| File | Description |
|---|---|
| `loss_curve.png` | Train/Val loss over epochs |
| `auroc_curve.png` | Mean AUROC over epochs with 0.90 target line |
| `lr_schedule.png` | Cosine warmup learning rate |
| `per_class_auroc_test.png` | Per-disease AUROC bar chart |
| `roc_curves_test.png` | 14 individual ROC curves |
| `label_cooccurrence.png` | Disease co-occurrence heatmap |
| `class_distribution.png` | Dataset class imbalance |

---

## Running the Web Application

```bash
# Start Flask backend
python app.py --port 5000

# Open in browser
http://localhost:5000
```

The frontend serves at the root path (`/`). Upload an image or use camera.

---

## Model Architecture

```
Input (224×224 RGB chest X-ray)
    │
    ▼ DenseNet-121
    Local feature map [B, 1024, 7, 7]
    │
    ▼ 1×1 Conv projection
    Patch tokens [B, 49, 384]  + CLS token
    │
    ▼ DeiT-Small (12 transformer blocks)
    CLS global representation [B, 384]
    │
    ▼ LayerNorm + Dropout(0.3)
    ▼ Linear(384 → 14)
    Raw logits [B, 14]
    │
    ▼ Label Correlation Module (GAT)
    Refined logits [B, 14]
    │
    ▼ Sigmoid → probabilities [B, 14]
```

### Label Correlation Module
- Learnable disease co-occurrence matrix (initialized from training data statistics)
- Graph Attention Network refines logits using pairwise disease relationships
- Residual connection: `output = logits + α × GAT_refinement(logits)`

---

## XAI Explanation Methods

| Method | Component | What it shows |
|---|---|---|
| **Grad-CAM++** | DenseNet-121 (last dense block) | Local regions driving prediction |
| **Attention Rollout** | DeiT-Small (all 12 blocks) | Global context the model attended to |
| **Fused** | Both combined (60% GradCAM + 40% Rollout) | Combined spatial explanation |

---

## Target Performance (NIH ChestX-ray14)

| Disease | Target AUROC |
|---|---|
| Atelectasis, Cardiomegaly, Effusion, etc. | > 0.80 each |
| **Mean AUROC** | **> 0.90** |

---

## X-Ray Validation

The system automatically rejects non-X-ray images by checking:
1. **Saturation**: Average HSV saturation < 25 (X-rays are grayscale)
2. **Grayness ratio**: ≥ 85% of pixels must be near-gray (R≈G≈B within ±20)
3. **Dynamic range**: Percentile range (p95−p5) > 30 (X-rays have high contrast)

---

## API Reference

### `POST /api/predict`
Upload chest X-ray for diagnosis.

**Request:** `multipart/form-data` with `image` file  
**OR** JSON `{ "image_b64": "<base64>" }`

**Response:**
```json
{
  "success": true,
  "predictions": { "Atelectasis": 72.4, "Cardiomegaly": 12.1, ... },
  "top_findings": [["Atelectasis", 72.4], ...],
  "xai_combined": "<base64 PNG>",
  "xai_gradcam":  "<base64 PNG>",
  "xai_rollout":  "<base64 PNG>",
  "original_img": "<base64 PNG>",
  "top_disease": "Atelectasis"
}
```

### `GET /api/health`
Model readiness and device status.

---

## Citation

Based on the ConViT-X project by Shedage, Shinde, Pawar, Shinde (2025).  
Department of AI & Data Science, VKBIET Baramati.
