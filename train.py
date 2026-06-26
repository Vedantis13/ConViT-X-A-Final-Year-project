"""
train.py — ConViT-X Training Pipeline
Multi-GPU training on 2×NVIDIA H200 NVL with:
  - Mixed precision (AMP)
  - Cosine LR schedule with warmup
  - Class-weighted sampling
  - Checkpoint saving
  - Comprehensive graph generation
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

import config
from dataset import build_loaders, compute_label_cooccurrence
from model import build_model
from losses import build_criterion
from metrics import compute_metrics, format_metrics


# ─── LR Schedule: Cosine with Warmup ─────────────────────────────────────────

def cosine_lr_schedule(optimizer, epoch, total_epochs, warmup_epochs,
                        base_lr, min_lr):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        t = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * t))

    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ─── Graph Generation ─────────────────────────────────────────────────────────

def save_training_graphs(history, graph_dir):
    """Generate and save all training diagnostic graphs."""
    graph_dir = Path(graph_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    plt.style.use("dark_background")
    BLUE   = "#00d4ff"
    GREEN  = "#00ff88"
    ORANGE = "#ff8c42"
    RED    = "#ff3366"
    WHITE  = "#e8eaed"

    # ── 1. Loss curves ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0a0e1a")
    ax.set_facecolor("#0d1117")
    ax.plot(epochs, history["train_loss"], color=BLUE,  lw=2, label="Train Loss")
    ax.plot(epochs, history["val_loss"],   color=GREEN, lw=2, label="Val Loss",   linestyle="--")
    ax.set_xlabel("Epoch", color=WHITE, fontsize=12)
    ax.set_ylabel("Loss",  color=WHITE, fontsize=12)
    ax.set_title("Training & Validation Loss", color=WHITE, fontsize=14, pad=15)
    ax.tick_params(colors=WHITE)
    ax.spines[:].set_color("#2d3139")
    ax.legend(facecolor="#1a1f2e", edgecolor="#2d3139", labelcolor=WHITE)
    ax.grid(alpha=0.15, color="#4a5568")
    plt.tight_layout()
    plt.savefig(graph_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 2. AUROC curve ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0a0e1a")
    ax.set_facecolor("#0d1117")
    ax.plot(epochs, history["mean_auroc"], color=ORANGE, lw=2.5, label="Mean AUROC")
    ax.axhline(0.90, color=RED, lw=1, linestyle=":", label="Target (0.90)")
    ax.fill_between(epochs, history["mean_auroc"], alpha=0.15, color=ORANGE)
    ax.set_xlabel("Epoch", color=WHITE, fontsize=12)
    ax.set_ylabel("AUROC", color=WHITE, fontsize=12)
    ax.set_title("Mean AUROC over Training", color=WHITE, fontsize=14, pad=15)
    ax.tick_params(colors=WHITE)
    ax.spines[:].set_color("#2d3139")
    ax.legend(facecolor="#1a1f2e", edgecolor="#2d3139", labelcolor=WHITE)
    ax.grid(alpha=0.15, color="#4a5568")
    ax.set_ylim(0.5, 1.0)
    plt.tight_layout()
    plt.savefig(graph_dir / "auroc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 3. LR schedule ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0a0e1a")
    ax.set_facecolor("#0d1117")
    ax.plot(epochs, history["lr"], color="#b39ddb", lw=2)
    ax.set_xlabel("Epoch", color=WHITE, fontsize=12)
    ax.set_ylabel("Learning Rate", color=WHITE, fontsize=12)
    ax.set_title("Learning Rate Schedule", color=WHITE, fontsize=14, pad=15)
    ax.tick_params(colors=WHITE)
    ax.spines[:].set_color("#2d3139")
    ax.grid(alpha=0.15, color="#4a5568")
    plt.tight_layout()
    plt.savefig(graph_dir / "lr_schedule.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved training graphs → {graph_dir}")


def save_per_class_auroc_graph(summary, graph_dir, split="Test"):
    """Bar chart of per-class AUROC."""
    graph_dir = Path(graph_dir)
    aurocs = [summary.get(f"{d}_auroc", 0) for d in config.DISEASES]

    colors = ["#00ff88" if a >= 0.90 else "#00d4ff" if a >= 0.80 else "#ff3366"
              for a in aurocs]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0a0e1a")
    ax.set_facecolor("#0d1117")

    bars = ax.barh(config.DISEASES, aurocs, color=colors, edgecolor="#1a1f2e", linewidth=0.5)
    ax.axvline(0.90, color="#ff8c42", lw=1.5, linestyle="--", label="Target 0.90")
    ax.axvline(summary.get("mean_auroc", 0), color="#ffffff", lw=1.5,
               linestyle="-", alpha=0.7, label=f"Mean {summary.get('mean_auroc', 0):.4f}")

    # Value labels
    for bar, val in zip(bars, aurocs):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left",
                color="#e8eaed", fontsize=8.5)

    ax.set_xlim(0.5, 1.02)
    ax.set_xlabel("AUROC", color="#e8eaed", fontsize=12)
    ax.set_title(f"Per-Class AUROC — {split} Set", color="#e8eaed", fontsize=14, pad=15)
    ax.tick_params(colors="#e8eaed")
    ax.spines[:].set_color("#2d3139")
    ax.legend(facecolor="#1a1f2e", edgecolor="#2d3139", labelcolor="#e8eaed")
    ax.grid(axis="x", alpha=0.15, color="#4a5568")

    plt.tight_layout()
    plt.savefig(graph_dir / f"per_class_auroc_{split.lower()}.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved per-class AUROC → {graph_dir}")


def save_label_cooccurrence_heatmap(cooc_matrix, graph_dir):
    """Visualize disease co-occurrence matrix."""
    graph_dir = Path(graph_dir)
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor("#0a0e1a")
    ax.set_facecolor("#0d1117")

    cmap = sns.color_palette("rocket_r", as_cmap=True)
    sns.heatmap(
        cooc_matrix, annot=True, fmt=".2f", cmap=cmap,
        xticklabels=config.DISEASES, yticklabels=config.DISEASES,
        ax=ax, linewidths=0.3, linecolor="#1a1f2e",
        annot_kws={"size": 7, "color": "#e8eaed"},
        cbar_kws={"shrink": 0.7}
    )
    ax.set_title("Disease Label Co-occurrence Matrix P(j|i)",
                 color="#e8eaed", fontsize=13, pad=15)
    ax.tick_params(colors="#e8eaed", labelsize=8)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(graph_dir / "label_cooccurrence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved co-occurrence heatmap → {graph_dir}")


def save_class_distribution(csv_path, graph_dir):
    """Bar chart showing class imbalance."""
    from dataset import ChestXray14Dataset
    graph_dir = Path(graph_dir)
    df = ChestXray14Dataset.get_label_frame(csv_path)
    counts = {d: int(df[d].sum()) for d in config.DISEASES}
    diseases = list(counts.keys())
    values   = list(counts.values())

    colors = plt.cm.cool(np.linspace(0.2, 0.9, len(diseases)))
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0a0e1a")
    ax.set_facecolor("#0d1117")

    bars = ax.bar(diseases, values, color=colors, edgecolor="#1a1f2e")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 100, str(val),
                ha="center", va="bottom", color="#e8eaed", fontsize=8)

    ax.set_xlabel("Disease", color="#e8eaed", fontsize=12)
    ax.set_ylabel("Count",   color="#e8eaed", fontsize=12)
    ax.set_title("NIH ChestX-ray14 Class Distribution", color="#e8eaed",
                 fontsize=14, pad=15)
    ax.tick_params(colors="#e8eaed", labelsize=8)
    plt.xticks(rotation=45, ha="right")
    ax.spines[:].set_color("#2d3139")
    ax.grid(axis="y", alpha=0.15, color="#4a5568")
    plt.tight_layout()
    plt.savefig(graph_dir / "class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved class distribution → {graph_dir}")


def save_roc_curves(all_labels, all_probs, graph_dir, split="Test"):
    """Per-class ROC curves on one figure."""
    from sklearn.metrics import roc_curve, auc
    graph_dir = Path(graph_dir)

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.patch.set_facecolor("#0a0e1a")
    axes = axes.flatten()

    palette = plt.cm.cool(np.linspace(0, 1, config.NUM_CLASSES))

    for i, (disease, color) in enumerate(zip(config.DISEASES, palette)):
        ax = axes[i]
        ax.set_facecolor("#0d1117")
        y_true = all_labels[:, i]
        y_prob  = all_probs[:, i]
        if len(np.unique(y_true)) < 2:
            ax.text(0.5, 0.5, "No positive\nsamples", ha="center",
                    va="center", color="#aaa", transform=ax.transAxes, fontsize=8)
        else:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=1.8)
            ax.fill_between(fpr, tpr, alpha=0.1, color=color)
            ax.plot([0, 1], [0, 1], ":", color="#555", lw=1)
            ax.text(0.55, 0.08, f"AUC={roc_auc:.3f}", color=color, fontsize=9,
                    transform=ax.transAxes)
        ax.set_title(disease, color="#e8eaed", fontsize=9)
        ax.tick_params(colors="#e8eaed", labelsize=7)
        ax.spines[:].set_color("#2d3139")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Hide extra subplots
    for j in range(len(config.DISEASES), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"ROC Curves — {split} Set", color="#e8eaed", fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(graph_dir / f"roc_curves_{split.lower()}.png",
                dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved ROC curves → {graph_dir}")


# ─── Training Loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    num_batches = len(loader)

    pbar = tqdm(loader, desc=f"  Train Ep {epoch:03d}", leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}")

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=config.MIXED_PRECISION):
            logits = model(images)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs  = []
    all_labels = []

    for images, labels in tqdm(loader, desc="  Eval", leave=False,
                                bar_format="{l_bar}{bar:30}{r_bar}"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=config.MIXED_PRECISION):
            logits = model(images)
            loss   = criterion(logits, labels)

        total_loss += loss.item()
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_probs  = np.concatenate(all_probs,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    summary    = compute_metrics(all_labels, all_probs)

    return total_loss / len(loader), summary, all_labels, all_probs


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    print("\n" + "═" * 60)
    print("  ConViT-X Training Pipeline")
    print("═" * 60)

    device = torch.device(config.DEVICE)
    print(f"\n  Device: {config.DEVICE}  |  GPUs: {config.NUM_GPUS}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[1/5] Building data loaders...")
    cooc_matrix = None
    if os.path.exists(config.CSV_PATH):
        cooc_matrix = compute_label_cooccurrence(config.CSV_PATH)
        save_label_cooccurrence_heatmap(cooc_matrix, config.GRAPH_DIR)
        save_class_distribution(config.CSV_PATH, config.GRAPH_DIR)

    train_loader, val_loader, test_loader, pos_weight = build_loaders()

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[2/5] Building ConViT-X model...")
    model = build_model(cooc_matrix=cooc_matrix, device=device)

    # ── Loss & Optimizer ──────────────────────────────────────────────────────
    print("\n[3/5] Setting up loss & optimizer...")
    criterion = build_criterion(pos_weight=pos_weight)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    scaler = GradScaler()

    # ── Training ──────────────────────────────────────────────────────────────
    print("\n[4/5] Training...")
    history = {
        "train_loss": [], "val_loss": [],
        "mean_auroc": [], "lr": [],
    }

    best_auroc     = 0.0
    patience_count = 0
    best_ckpt_path = os.path.join(config.CHECKPOINT_DIR, "convitx_best.pth")

    for epoch in range(1, config.EPOCHS + 1):
        t0 = time.time()

        lr = cosine_lr_schedule(
            optimizer, epoch - 1, config.EPOCHS,
            config.WARMUP_EPOCHS, config.LR, config.MIN_LR
        )

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        val_loss, val_summary, _, _ = evaluate(model, val_loader, criterion, device)

        mean_auroc = val_summary["mean_auroc"]
        elapsed    = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["mean_auroc"].append(mean_auroc)
        history["lr"].append(lr)

        print(
            f"  Ep {epoch:03d}/{config.EPOCHS} | "
            f"loss {train_loss:.4f} → {val_loss:.4f} | "
            f"AUROC {mean_auroc:.4f} | "
            f"lr {lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        # Save checkpoint
        ckpt = {
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "mean_auroc":  mean_auroc,
        }
        torch.save(ckpt, os.path.join(config.CHECKPOINT_DIR, "convitx_last.pth"))

        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            patience_count = 0
            torch.save(ckpt, best_ckpt_path)
            print(f"    ★  New best AUROC: {best_auroc:.4f}  → saved checkpoint")
        else:
            patience_count += 1
            if patience_count >= config.EARLY_STOP_PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (patience {config.EARLY_STOP_PATIENCE})")
                break

        # Save graphs every 10 epochs
        if epoch % 10 == 0 or epoch == config.EPOCHS:
            save_training_graphs(history, config.GRAPH_DIR)

    # ── Final Test Evaluation ─────────────────────────────────────────────────
    print("\n[5/5] Evaluating on test set (best checkpoint)...")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    _, test_summary, test_labels, test_probs = evaluate(
        model, test_loader, criterion, device
    )

    print("\n" + format_metrics(test_summary))
    print(f"\n  ★  Final Mean AUROC: {test_summary['mean_auroc']:.4f}")

    # Save final graphs
    save_training_graphs(history, config.GRAPH_DIR)
    save_per_class_auroc_graph(test_summary, config.GRAPH_DIR, split="Test")
    save_roc_curves(test_labels, test_probs, config.GRAPH_DIR, split="Test")

    # Save history + metrics
    with open(os.path.join(config.OUTPUT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(config.OUTPUT_DIR, "test_metrics.json"), "w") as f:
        json.dump(test_summary, f, indent=2)

    print(f"\n  All outputs saved → {config.OUTPUT_DIR}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",  default=config.DATA_DIR)
    parser.add_argument("--epochs",    type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    config.DATA_DIR    = args.data_dir
    config.EPOCHS      = args.epochs
    config.BATCH_SIZE  = args.batch_size
    config.IMAGE_DIR   = os.path.join(config.DATA_DIR, "images")
    config.CSV_PATH    = os.path.join(config.DATA_DIR, "Data_Entry_2017.csv")
    config.TRAIN_LIST  = os.path.join(config.DATA_DIR, "train_val_list.txt")
    config.TEST_LIST   = os.path.join(config.DATA_DIR, "test_list.txt")

    main(args)
