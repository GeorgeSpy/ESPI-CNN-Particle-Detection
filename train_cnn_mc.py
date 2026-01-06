from __future__ import annotations
import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

import torchvision
from torchvision import models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# -------------
# Dataset class
# -------------
class ESPICNNDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int = 256, augment: str = "none"):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: str) -> torch.Tensor:
        if path.lower().endswith('.npy'):
            arr = np.load(path)
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            arr = arr.astype(np.float32)
            if arr.max() > 1.0:
                arr /= arr.max()
        else:
            with Image.open(path) as im:
                im = im.convert('I;16')
                arr = np.array(im, dtype=np.uint16).astype(np.float32) / 65535.0
        arr = torch.from_numpy(arr).unsqueeze(0)
        return arr

    def _resize(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[-2:] == (self.img_size, self.img_size):
            return tensor
        tensor = F.interpolate(tensor.unsqueeze(0), size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        return tensor.squeeze(0)

    def _augment(self, img: torch.Tensor) -> torch.Tensor:
        if self.augment == 'none':
            return img
        if random.random() < 0.5:
            img = torch.flip(img, dims=[2])
        if random.random() < 0.3:
            img = torch.flip(img, dims=[1])
        if random.random() < 0.5:
            angle = random.uniform(-8, 8)
            img = torchvision.transforms.functional.rotate(img, angle, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, fill=0.0)
        if random.random() < 0.5:
            img = torchvision.transforms.functional.adjust_brightness(img, 1.0 + random.uniform(-0.1, 0.1))
            img = torchvision.transforms.functional.adjust_contrast(img, 1.0 + random.uniform(-0.1, 0.1))
        return torch.clamp(img, 0.0, 1.0)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row['path']
        label = int(row['label'])
        tensor = self._load_image(path)
        tensor = self._resize(tensor)
        tensor = self._augment(tensor)
        return tensor, label

# -------------
# Models
# -------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128 * 32 * 32, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class ResNet18Gray(nn.Module):
    def __init__(self, num_classes: int = 6, freeze_until: Optional[str] = None):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        w = self.backbone.conv1.weight.data
        w_mean = w.mean(dim=1, keepdim=True)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.backbone.conv1.weight.copy_(w_mean)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_classes)
        if freeze_until:
            self._freeze_until(freeze_until)

    def _freeze_until(self, stage: str):
        freeze = True
        for name, param in self.backbone.named_parameters():
            if name.startswith('fc'):
                freeze = False
            elif stage == 'conv1' and name.startswith('layer1.'):
                freeze = False
            elif stage == 'layer1' and name.startswith('layer2.'):
                freeze = False
            elif stage == 'layer2' and name.startswith('layer3.'):
                freeze = False
            elif stage == 'layer3' and name.startswith('layer4.'):
                freeze = False
            param.requires_grad = not freeze

    def forward(self, x):
        return self.backbone(x)

# -------------
# Utilities
# -------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class TrainConfig:
    labels_csv: str
    run_dir: str
    model: str
    img_size: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    augment: str
    sampler: str
    loss_weights: str
    label_smoothing: float
    focal_gamma: float
    freeze_until: Optional[str]
    seed: int
    device: str
    lobo_band: Optional[Tuple[float, float]]
    lodo_holdout: Optional[str]
    lobo_per_class_pct: Optional[float]
    lobo_multi_json: Optional[str]

# class weights
def make_class_weights(counts: np.ndarray, mode: str) -> np.ndarray:
    counts = counts.astype(np.float32)
    if mode == 'none':
        w = np.ones_like(counts, dtype=np.float32)
    elif mode == 'inverse':
        w = 1.0 / np.clip(counts, 1.0, None)
    elif mode == 'sqrt_inv':
        w = 1.0 / np.sqrt(np.clip(counts, 1.0, None))
    elif mode == 'effective':
        beta = 0.999
        eff = (1.0 - np.power(beta, np.clip(counts, 1.0, None))) / (1.0 - beta)
        w = 1.0 / np.clip(eff, 1e-6, None)
    else:
        raise ValueError(f"Unknown class weight mode: {mode}")
    if w.mean() > 0:
        w /= w.mean()
    return w.astype(np.float32)

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        ce = F.nll_loss(log_probs, targets, reduction='none')
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal = (1 - pt).clamp(min=0) ** self.gamma
        if self.alpha is not None:
            focal = focal * self.alpha[targets]
        return (focal * ce).mean()

# -------------
# Split helpers
# -------------
def stratified_split(df: pd.DataFrame, seed: int, val_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_idx, val_idx = [], []
    for label in sorted(df['label'].unique()):
        idx = df.index[df['label'] == label].tolist()
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_ratio)))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return df.loc[train_idx].reset_index(drop=True), df.loc[val_idx].reset_index(drop=True)


def stratified_split_full(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df_train, df_temp = stratified_split(df, seed=seed, val_ratio=0.30)
    df_val, df_test = stratified_split(df_temp, seed=seed+1, val_ratio=0.5)
    return df_train, df_val, df_test


def lobo_per_class(df: pd.DataFrame, pct: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_test_list = []
    df_train_list = []
    df_val_list = []
    rng = np.random.RandomState(seed)
    for label in sorted(df['label'].unique()):
        df_label = df[df['label'] == label].copy()
        if df_label.empty:
            continue
        df_label = df_label.sample(frac=1.0, random_state=rng.randint(0,1<<32)).reset_index(drop=True)
        n_test = max(1, int(math.ceil(len(df_label) * pct)))
        n_val = max(1, int(math.ceil(len(df_label) * pct / 2)))
        df_test_list.append(df_label.iloc[:n_test])
        df_val_list.append(df_label.iloc[n_test:n_test+n_val])
        df_train_list.append(df_label.iloc[n_test+n_val:])
    df_test = pd.concat(df_test_list).reset_index(drop=True)
    df_val = pd.concat(df_val_list).reset_index(drop=True)
    df_train = pd.concat(df_train_list).reset_index(drop=True)
    return df_train, df_val, df_test


def lobo_multi_json(df: pd.DataFrame, cfg_json: str, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with open(cfg_json, 'r', encoding='utf-8-sig') as fh:
        gaps = json.load(fh)
    df = df.copy()
    df['freq_hz'] = df['freq_hz'].astype(float)
    mask_test = np.zeros(len(df), dtype=bool)
    for label_str, ranges in gaps.items():
        label = int(label_str)
        for lo, hi in ranges:
            mask = (df['label'] == label) & (df['freq_hz'] >= lo) & (df['freq_hz'] <= hi)
            mask_test |= mask.to_numpy()
    df_test = df[mask_test].reset_index(drop=True)
    df_trainval = df[~mask_test].reset_index(drop=True)
    if df_test.empty:
        raise ValueError('No samples fell into the requested LOBO JSON ranges; test set empty.')
    df_train, df_val = stratified_split(df_trainval, seed=seed, val_ratio=0.2)
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)

# -------------
# Metrics
# -------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]):
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = rep['accuracy']
    macro_f1 = rep['macro avg']['f1-score']
    weighted_f1 = rep['weighted avg']['f1-score']
    return acc, macro_f1, weighted_f1, rep, cm

# -------------
# Main training
# -------------
def make_loaders(df_train, df_val, df_test, img_size, batch_size, augment, device, sampler_mode, class_weight_mode, num_classes):
    ds_train = ESPICNNDataset(df_train, img_size=img_size, augment=augment)
    ds_val = ESPICNNDataset(df_val, img_size=img_size, augment='none')
    ds_test = ESPICNNDataset(df_test, img_size=img_size, augment='none')
    class_counts = df_train['label'].value_counts().reindex(range(num_classes), fill_value=0).values
    class_weights_np = make_class_weights(class_counts, mode=class_weight_mode)
    pin_memory = device.startswith('cuda')
    if sampler_mode == 'weighted':
        sample_weights = class_weights_np[df_train['label'].values]
        sampler = WeightedRandomSampler(torch.from_numpy(sample_weights).double(), num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(ds_train, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=pin_memory)
    else:
        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader, torch.tensor(class_weights_np, dtype=torch.float32)


def build_model(name: str, num_classes: int, freeze_until: Optional[str]):
    if name == 'simple':
        return SimpleCNN(num_classes)
    elif name == 'resnet18':
        return ResNet18Gray(num_classes=num_classes, freeze_until=freeze_until)
    else:
        raise ValueError(f'Unknown model: {name}')


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: str, title: str):
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=labels, yticklabels=labels, ylabel='True label', xlabel='Predicted label', title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i,j], 'd'), ha='center', va='center', color='white' if cm[i,j] > thresh else 'black')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train_and_eval(cfg: TrainConfig):
    os.makedirs(cfg.run_dir, exist_ok=True)
    set_seed(cfg.seed)
    df = pd.read_csv(cfg.labels_csv)
    if 'path' not in df.columns or 'label' not in df.columns:
        raise ValueError('CSV must contain columns: path, label')
    df['path'] = df['path'].apply(os.path.normpath)
    class_names = [
        'mode_(1,1)H', 'mode_(1,1)T', 'mode_(1,2)', 'mode_(2,1)', 'mode_higher', 'other_unknown'
    ]

    if cfg.lobo_multi_json is not None:
        df_train, df_val, df_test = lobo_multi_json(df, cfg.lobo_multi_json, cfg.seed)
        split_info = {'type': 'LOBO_multi', 'gaps_json': cfg.lobo_multi_json}
    elif cfg.lobo_per_class_pct is not None:
        df_train, df_val, df_test = lobo_per_class(df, cfg.lobo_per_class_pct, cfg.seed)
        split_info = {'type': 'LOBO_per_class_pct', 'pct': cfg.lobo_per_class_pct}
    elif cfg.lobo_band is not None:
        lo, hi = cfg.lobo_band
        df_test = df[(df['freq_hz'] >= lo) & (df['freq_hz'] <= hi)].copy()
        df_trainval = df[~((df['freq_hz'] >= lo) & (df['freq_hz'] <= hi))].copy()
        df_train, df_val = stratified_split(df_trainval, seed=cfg.seed, val_ratio=0.2)
        split_info = {'type': 'LOBO_band', 'band': (lo, hi)}
    elif cfg.lodo_holdout is not None:
        holdout = cfg.lodo_holdout
        df_test = df[df['dataset_id'] == holdout].copy()
        df_trainval = df[df['dataset_id'] != holdout].copy()
        df_train, df_val = stratified_split(df_trainval, seed=cfg.seed, val_ratio=0.2)
        split_info = {'type': 'LODO', 'holdout': holdout}
    else:
        df_train, df_val, df_test = stratified_split_full(df, seed=cfg.seed)
        split_info = {'type': 'stratified_70_15_15'}

    split_meta = {
        'train': len(df_train),
        'val': len(df_val),
        'test': len(df_test),
'test_labels': df_test['label'].value_counts().sort_index().to_dict(),
        'train_labels': df_train['label'].value_counts().sort_index().to_dict(),
        'val_labels': df_val['label'].value_counts().sort_index().to_dict(),
        'split_kind': split_info,
    }
    with open(os.path.join(cfg.run_dir, 'split.json'), 'w') as f:
        json.dump(split_meta, f, indent=2)

    train_loader, val_loader, test_loader, class_weights = make_loaders(
        df_train, df_val, df_test,
        img_size=cfg.img_size,
        batch_size=cfg.batch_size,
        augment=cfg.augment,
        device=cfg.device,
        sampler_mode=cfg.sampler,
        class_weight_mode=cfg.loss_weights,
        num_classes=len(class_names)
    )
    class_weights = class_weights.to(cfg.device)
    if cfg.loss_weights == 'none':
        class_weights_for_loss = None
    elif cfg.sampler == 'weighted':
        print('[info] Weighted sampler active; disabling class-weighted loss to avoid double correction.')
        class_weights_for_loss = None
    else:
        class_weights_for_loss = class_weights

    model = build_model(cfg.model, num_classes=len(class_names), freeze_until=cfg.freeze_until)
    model.to(cfg.device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(10, cfg.epochs))

    if cfg.focal_gamma > 0:
        alpha = None
        if class_weights_for_loss is not None:
            alpha = (class_weights_for_loss / class_weights_for_loss.sum()).detach()
        criterion = FocalLoss(gamma=cfg.focal_gamma, alpha=alpha)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights_for_loss, label_smoothing=cfg.label_smoothing)
    criterion = criterion.to(cfg.device)

    scaler = GradScaler(enabled=cfg.device.startswith('cuda'))

    best_val_macro = -1.0
    best_epoch = -1
    patience_left = 10
    history = []

    def run_epoch(loader, train: bool):
        model.train(train)
        running_loss = 0.0
        y_true, y_pred = [], []
        for imgs, labels in loader:
            imgs = imgs.to(cfg.device)
            labels = labels.to(cfg.device)
            with autocast('cuda', enabled=cfg.device.startswith('cuda')):
                logits = model(imgs)
                loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            y_true.append(labels.detach().cpu().numpy())
            y_pred.append(preds.detach().cpu().numpy())
        loss = running_loss / len(loader.dataset)
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        acc, macro_f1, weighted_f1, rep, cm = compute_metrics(y_true, y_pred, labels=list(range(len(class_names))))
        return loss, acc, macro_f1, weighted_f1, rep, cm

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc, train_macro, train_weighted, _, _ = run_epoch(train_loader, train=True)
        val_loss, val_acc, val_macro, val_weighted, _, _ = run_epoch(val_loader, train=False)
        scheduler.step()
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_macro_f1': train_macro,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_macro_f1': val_macro,
        })
        print(f'EP {epoch:03d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% val_macro={val_macro*100:.2f}%')
        if val_macro > best_val_macro:
            best_val_macro = val_macro
            best_epoch = epoch
            patience_left = 10
            torch.save({'model_state': model.state_dict()}, os.path.join(cfg.run_dir, 'best.pt'))
        else:
            patience_left -= 1
        if patience_left <= 0:
            print(f'Early stopping at epoch {epoch} (best epoch {best_epoch})')
            break

    # load best model
    ckpt = torch.load(os.path.join(cfg.run_dir, 'best.pt'), map_location=cfg.device)
    model.load_state_dict(ckpt['model_state'])

    train_loss, train_acc, train_macro, train_weighted, train_rep, train_cm = run_epoch(train_loader, train=False)
    val_loss, val_acc, val_macro, val_weighted, val_rep, val_cm = run_epoch(val_loader, train=False)
    test_loss, test_acc, test_macro, test_weighted, test_rep, test_cm = run_epoch(test_loader, train=False)

    with open(os.path.join(cfg.run_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'best_epoch': best_epoch,
            'val_best_macro_f1': best_val_macro,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_macro_f1': train_macro,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_macro_f1': val_macro,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_macro_f1': test_macro,
            'test_weighted_f1': test_weighted,
            'class_names': class_names,
        }, f, indent=2)

    for split, rep, cm in [
        ('train', train_rep, train_cm),
        ('val', val_rep, val_cm),
        ('test', test_rep, test_cm),
    ]:
        with open(os.path.join(cfg.run_dir, f'{split}_classification_report.json'), 'w') as f:
            json.dump(rep, f, indent=2)
        plot_confusion_matrix(cm, class_names, os.path.join(cfg.run_dir, f'cm_{split}.png'), f'Confusion Matrix ({split})')

    with open(os.path.join(cfg.run_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)


# -------------
# Argument parser
# -------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--labels_csv', required=True)
    p.add_argument('--run_dir', required=True)
    p.add_argument('--model', choices=['simple','resnet18'], default='resnet18')
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--augment', choices=['none','light','strong'], default='strong')
    p.add_argument('--sampler', choices=['weighted','plain'], default='weighted')
    p.add_argument('--loss_weights', choices=['inverse','sqrt_inv','effective','none'], default='none')
    p.add_argument('--label_smoothing', type=float, default=0.0)
    p.add_argument('--focal_gamma', type=float, default=0.0)
    p.add_argument('--freeze_until', type=str, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    # LOBO/LODO options
    p.add_argument('--lobo_band', type=float, nargs=2, default=None)
    p.add_argument('--lodo_holdout', type=str, default=None)
    p.add_argument('--lobo_per_class_pct', type=float, default=None)
    p.add_argument('--lobo_multi_json', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(
        labels_csv=args.labels_csv,
        run_dir=args.run_dir,
        model=args.model,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        augment=args.augment,
        sampler=args.sampler,
        loss_weights=args.loss_weights,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
        freeze_until=args.freeze_until,
        seed=args.seed,
        device=args.device,
        lobo_band=tuple(args.lobo_band) if args.lobo_band is not None else None,
        lodo_holdout=args.lodo_holdout,
        lobo_per_class_pct=args.lobo_per_class_pct,
        lobo_multi_json=args.lobo_multi_json
    )
    train_and_eval(cfg)

if __name__ == '__main__':
    main()
