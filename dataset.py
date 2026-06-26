"""
dataset.py — NIH ChestX-ray14 Dataset Loader
Handles loading, splitting, augmentation, and class-weight computation.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config


# ─── Albumentations Augmentation Pipelines ────────────────────────────────────

def get_train_transforms():
    return A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.HorizontalFlip(p=config.AUGMENTATION["horizontal_flip"]),
        A.Affine(
            translate_percent=config.AUGMENTATION["shift_limit"],
            scale=(1 - config.AUGMENTATION["scale_limit"], 1 + config.AUGMENTATION["scale_limit"]),
            rotate=(-config.AUGMENTATION["rotation_limit"], config.AUGMENTATION["rotation_limit"]),
            p=0.6
        ),
        A.RandomBrightnessContrast(
            brightness_limit=config.AUGMENTATION["brightness_limit"],
            contrast_limit=config.AUGMENTATION["contrast_limit"],
            p=0.5
        ),
        A.GaussNoise(p=config.AUGMENTATION["gaussian_noise_p"]),
        A.ElasticTransform(alpha=1, sigma=50, p=config.AUGMENTATION["elastic_transform_p"]),
        A.CLAHE(clip_limit=2.0, p=0.3),           # Enhance X-ray contrast
        A.GridDistortion(p=0.1),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 16),
            hole_width_range=(8, 16),
            fill=0, p=config.AUGMENTATION["random_erasing_p"]
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms():
    return A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ─── Dataset Class ─────────────────────────────────────────────────────────────

class ChestXray14Dataset(Dataset):
    """
    NIH ChestX-ray14 Multi-Label Dataset.

    Expected layout:
        <data_dir>/
            images/          (112,120 PNG files)
            Data_Entry_2017.csv
            train_val_list.txt
            test_list.txt
    """

    def __init__(self, image_dir, df, transform=None):
        self.image_dir = Path(image_dir)
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_dir / row["Image Index"]

        # Load image — convert to RGB (X-rays stored as grayscale PNG)
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img, dtype=np.uint8)

        if self.transform:
            augmented = self.transform(image=img_np)
            img_tensor = augmented["image"]
        else:
            img_tensor = T.ToTensor()(img)

        # Multi-label target
        labels = torch.tensor(
            [row[d] for d in config.DISEASES], dtype=torch.float32
        )
        return img_tensor, labels

    @staticmethod
    def get_label_frame(csv_path, list_file=None):
        """Parse the NIH CSV into a DataFrame with binary columns per disease."""
        df = pd.read_csv(csv_path)

        # Optionally filter to official train/test split
        if list_file and os.path.exists(list_file):
            with open(list_file) as f:
                names = set(line.strip() for line in f)
            df = df[df["Image Index"].isin(names)]

        # Expand "Finding Labels" into binary columns
        for disease in config.DISEASES:
            df[disease] = df["Finding Labels"].apply(
                lambda x: 1 if disease in x.split("|") else 0
            ).astype(np.float32)

        return df


# ─── Data Loader Factory ──────────────────────────────────────────────────────

def build_loaders(data_dir=None, csv_path=None, train_list=None, test_list=None,
                  batch_size=None, num_workers=None):
    """Build train / val / test DataLoaders with class-balanced sampling."""
    data_dir    = data_dir    or config.DATA_DIR
    csv_path    = csv_path    or config.CSV_PATH
    train_list  = train_list  or config.TRAIN_LIST
    test_list   = test_list   or config.TEST_LIST
    batch_size  = batch_size  or config.BATCH_SIZE
    num_workers = num_workers or config.NUM_WORKERS
    image_dir   = os.path.join(data_dir, "images")

    # ── Load official splits ──────────────────────────────────────────────────
    full_df = ChestXray14Dataset.get_label_frame(csv_path)

    if os.path.exists(train_list) and os.path.exists(test_list):
        with open(train_list) as f:
            train_names = set(l.strip() for l in f)
        with open(test_list) as f:
            test_names = set(l.strip() for l in f)
        train_val_df = full_df[full_df["Image Index"].isin(train_names)]
        test_df      = full_df[full_df["Image Index"].isin(test_names)]
    else:
        # Fallback: random 80/10/10 split
        shuffled = full_df.sample(frac=1, random_state=42)
        n = len(shuffled)
        train_val_df = shuffled.iloc[:int(0.9 * n)]
        test_df      = shuffled.iloc[int(0.9 * n):]

    # Val split from train_val
    val_n    = int(len(train_val_df) * config.VAL_SPLIT)
    val_df   = train_val_df.sample(n=val_n, random_state=42)
    train_df = train_val_df.drop(val_df.index)

    print(f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = ChestXray14Dataset(image_dir, train_df, get_train_transforms())
    val_ds   = ChestXray14Dataset(image_dir, val_df,   get_val_transforms())
    test_ds  = ChestXray14Dataset(image_dir, test_df,  get_val_transforms())

    # ── Weighted sampler for class imbalance ──────────────────────────────────
    labels_arr = train_df[config.DISEASES].values  # [N, 14]
    # Weight each sample by the rarest positive label it contains
    class_pos  = labels_arr.sum(axis=0)
    class_neg  = len(labels_arr) - class_pos
    class_wt   = class_neg / (class_pos + 1e-6)   # [14]

    sample_weights = []
    for row in labels_arr:
        pos_classes = np.where(row == 1)[0]
        if len(pos_classes) > 0:
            w = class_wt[pos_classes].max()
        else:
            w = 1.0
        sample_weights.append(w)
    sample_weights = torch.tensor(sample_weights, dtype=torch.float)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # Compute per-class BCE pos_weight for loss
    pos_weight = torch.tensor(class_wt, dtype=torch.float32)

    return train_loader, val_loader, test_loader, pos_weight


def compute_label_cooccurrence(csv_path):
    """
    Compute normalized label co-occurrence matrix for LCM initialization.
    Returns a [14, 14] numpy array.
    """
    df = ChestXray14Dataset.get_label_frame(csv_path)
    labels = df[config.DISEASES].values.astype(np.float32)
    # Joint probability P(i, j)
    cooc = (labels.T @ labels)  # [14, 14]
    diag = cooc.diagonal()      # counts per class
    # Normalize: P(j | i)
    denom = diag[:, None] + 1e-8
    cooc_norm = cooc / denom
    return cooc_norm
