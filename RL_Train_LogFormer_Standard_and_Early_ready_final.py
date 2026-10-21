#!/usr/bin/env python3
"""
LogFormer-Standard and LogFormer-Early
======================================

Goal
----
Provide a fair comparison while preserving the LogFormer architecture and
training recipe.

Variants
--------
1) standard
   Full-sequence TRAIN -> full-sequence DEV checkpoint selection ->
   full-sequence TEST.

2) early
   Full-sequence TRAIN is preserved.
   25/50/75% prefixes are added to TRAIN:
      * normal full sequence: prefixes are NORMAL
      * anomalous full sequence:
          - if true event-level first_anomaly_step metadata exists, prefix labels
            are derived from it
          - otherwise NO anomalous prefixes are fabricated; only the original
            full anomalous sequence is kept
   DEV remains full-sequence, so checkpoint selection is identical in form.
   TEST is sequential prefix inference; first anomaly prediction is the alert.

3) compare
   Does NOT train. Loads saved JSON files and prints a comparison table.

Expected preprocessed files
---------------------------
The script uses the NPZ format produced by the user's LogFormer preprocessing:
    <dataset>_training_block_w<window>.npz
    <dataset>_validation_block_w<window>.npz
    <dataset>_testing_block_w<window>.npz

Required keys:
    x, y

Optional key:
    first_anomaly_step
        1-based event index of the first TRUE anomalous event.
        Use <=0 / NaN when unavailable.

No RL is used. LogFormer-Early does not learn a stopping policy.
"""

import argparse
import json
import math
import os
import random
import re
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# ORIGINAL LOGFORMER TRAINING DEFAULTS
# =============================================================================

EMBEDDING_DIM = 768
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-5
NUM_LAYERS = 1
ADAPTER_SIZE = 64
NHEAD = 8
DROPOUT = 0.1
FEEDFORWARD_DIM = 4 * EMBEDDING_DIM
SEED = 123
GRAD_CLIP = 0.5

# Paper cost setting requested by the user.
FP_UNIT_COST = 10.0
FN_UNIT_COST = 20.0
DELAY_UNIT_COST = 5.0

# LogFormer-Early training prefixes.
PREFIX_FRACTIONS = (0.25, 0.50, 0.75)


# =============================================================================
# ORIGINAL LOGFORMER DATA GENERATOR BEHAVIOR
# =============================================================================

class DataGenerator(Dataset):
    """
    Equivalent to the released LogFormer DataGenerator:
    every sequence is truncated/padded to [window_size, 768].
    """

    def __init__(self, x, y, window_size: int):
        self.x = x
        self.y = y
        self.window_size = int(window_size)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = np.asarray(self.x[index], dtype=np.float32)
        y = np.asarray(self.y[index], dtype=np.float32)

        if x.ndim != 2 or x.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"Invalid sequence shape at index {index}: {x.shape}; "
                f"expected [T, {EMBEDDING_DIM}]."
            )

        num_tokens = x.shape[0]
        use_tokens = min(int(num_tokens), self.window_size)

        padded = np.zeros(
            (self.window_size, EMBEDDING_DIM),
            dtype=np.float32,
        )
        padded[:use_tokens] = x[:use_tokens]

        return padded, y


# =============================================================================
# ORIGINAL LOGFORMER ARCHITECTURE
# =============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Adapter(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        z = self.gelu(self.linear1(x))
        z = self.linear2(z)
        return x + z


class TransformerEncoderLayerWithAdapter(nn.Module):
    """
    Included for architectural fidelity with released model.py.
    In the in-domain Standard/Early experiments, mode='classifier' uses
    PyTorch's original TransformerEncoderLayer, exactly like LogFormer.
    """

    def __init__(
        self,
        d_model,
        nhead,
        adapter_size=64,
        dim_feedforward=3072,
        dropout=0.1,
        layer_norm_eps=1e-5,
        batch_first=True,
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.adapter1 = Adapter(
            d_model,
            d_model,
            hidden_dim=adapter_size,
        )
        self.norm1 = nn.LayerNorm(
            d_model,
            eps=layer_norm_eps,
        )

        self.linear1 = nn.Linear(
            d_model,
            dim_feedforward,
        )
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(
            dim_feedforward,
            d_model,
        )
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.adapter2 = Adapter(
            d_model,
            d_model,
            hidden_dim=adapter_size,
        )
        self.norm2 = nn.LayerNorm(
            d_model,
            eps=layer_norm_eps,
        )

    def forward(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None,
    ):
        src2 = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        src2 = self.dropout1(src2)
        src2 = self.adapter1(src2)
        src = self.norm1(src + src2)

        src2 = self.linear2(
            self.dropout(
                self.activation(
                    self.linear1(src)
                )
            )
        )
        src2 = self.dropout2(src2)
        src2 = self.adapter2(src2)
        src = self.norm2(src + src2)
        return src

    def activate_adapter(self):
        for layer in (
            self.adapter1,
            self.adapter2,
            self.norm1,
            self.norm2,
        ):
            for param in layer.parameters():
                param.requires_grad = True


class Model(nn.Module):
    """
    Released LogFormer shape:
        input  [B, W, 768]
        encoder output [B, W, 768]
        flatten [B, W*768]
        fc1 -> [B, 2]
    """

    def __init__(
        self,
        mode="classifier",
        num_layers=NUM_LAYERS,
        adapter_size=ADAPTER_SIZE,
        dim=EMBEDDING_DIM,
        window_size=20,
        nhead=NHEAD,
        dim_feedforward=FEEDFORWARD_DIM,
        dropout=DROPOUT,
    ):
        super().__init__()

        if mode == "adapter":
            encoder_layer = TransformerEncoderLayerWithAdapter(
                dim,
                nhead,
                adapter_size=adapter_size,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                dim,
                nhead,
                dim_feedforward,
                dropout,
                batch_first=True,
            )

        self.trans_encder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )
        self.pos_encoder1 = PositionalEncoding(
            d_model=EMBEDDING_DIM,
        )
        self.fc1 = nn.Linear(
            dim * window_size,
            2,
        )

    def forward(self, x):
        batch_size, _, _ = x.size()
        x = self.pos_encoder1(x)
        x = self.trans_encder(x)
        x = x.contiguous().view(batch_size, -1)
        x = self.fc1(x)
        return x


# =============================================================================
# REPRODUCIBILITY / IO
# =============================================================================

def seed_everything(seed=SEED):
    warnings.filterwarnings("ignore")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_file(
    preprocessed_dir: Path,
    dataset: str,
    split: str,
    window_size: int,
) -> Path:
    return (
        preprocessed_dir
        / f"{dataset}_{split}_block_w{window_size}.npz"
    )


def infer_window_size(
    preprocessed_dir: Path,
    dataset: str,
) -> int:
    """
    Do not silently change the user's sequence construction.
    Infer the already-preprocessed window size when exactly one exists.
    """
    matches = sorted(
        preprocessed_dir.glob(
            f"{dataset}_training_block_w*.npz"
        )
    )

    if not matches:
        # Official released train_transformer.py default.
        return 20

    sizes = []
    for path in matches:
        m = re.search(r"_block_w(\d+)\.npz$", path.name)
        if m:
            sizes.append(int(m.group(1)))

    sizes = sorted(set(sizes))

    if len(sizes) == 1:
        return sizes[0]

    if not sizes:
        return 20

    raise ValueError(
        f"Multiple preprocessed window sizes found for {dataset}: "
        f"{sizes}. Pass --window-size explicitly."
    )


def load_npz_split(
    preprocessed_dir: Path,
    dataset: str,
    split: str,
    window_size: int,
):
    path = split_file(
        preprocessed_dir,
        dataset,
        split,
        window_size,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing preprocessed split: {path}\n"
            "Run your LogFormer preprocessing first, or point "
            "--preprocessed-dir to the existing NPZ directory."
        )

    data = np.load(path, allow_pickle=True)

    if "x" not in data or "y" not in data:
        raise ValueError(
            f"{path} must contain x and y."
        )

    x = data["x"]
    y = data["y"]

    first_anomaly_step = None
    for key in (
        "first_anomaly_step",
        "first_anomaly_idx",
        "anomaly_onset",
    ):
        if key in data:
            first_anomaly_step = np.asarray(
                data[key]
            )
            break

    data.close()

    if len(x) != len(y):
        raise ValueError(
            f"{path}: x/y length mismatch."
        )

    if (
        first_anomaly_step is not None
        and len(first_anomaly_step) != len(y)
    ):
        raise ValueError(
            f"{path}: onset metadata length mismatch."
        )

    return x, y, first_anomaly_step


def effective_sequence(seq, window_size: int):
    """
    Recover the actually observed part of a sequence.

    If preprocessing already padded with all-zero rows, trailing zero rows
    are removed. The result is then capped at LogFormer's window_size.
    """
    arr = np.asarray(seq, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2-D sequence, got {arr.shape}."
        )

    if arr.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Expected embedding dim {EMBEDDING_DIM}, "
            f"got {arr.shape[1]}."
        )

    arr = arr[:window_size]

    if len(arr) == 0:
        return arr

    nonzero = np.any(
        np.abs(arr) > 0,
        axis=1,
    )

    if np.any(nonzero):
        last = int(
            np.where(nonzero)[0][-1]
        ) + 1
        arr = arr[:last]

    return arr


def labels_to_binary(y):
    y = np.asarray(y)

    if y.ndim != 2 or y.shape[1] != 2:
        raise ValueError(
            f"Expected one-hot labels [N,2], got {y.shape}."
        )

    return np.argmax(y, axis=1).astype(int)


# =============================================================================
# LOGFORMER-EARLY TRAINING AUGMENTATION
# =============================================================================

def _clean_onset(value) -> Optional[int]:
    if value is None:
        return None

    try:
        if np.isnan(value):
            return None
    except TypeError:
        pass

    try:
        value = int(value)
    except (TypeError, ValueError):
        return None

    if value <= 0:
        return None

    return value


def build_early_training_set(
    x_train,
    y_train,
    onset_steps,
    window_size: int,
):
    """
    Preserve ALL full original sequences, then add valid 25/50/75% prefixes.

    Normal sequence:
        prefix -> normal

    Anomalous sequence:
        if true first_anomaly_step exists:
            prefix_end < onset -> normal
            prefix_end >= onset -> anomaly
        otherwise:
            do not fabricate prefix labels; add no anomaly prefixes.
    """
    y_binary = labels_to_binary(y_train)

    x_out = []
    y_out = []

    full_count = 0
    normal_prefix_count = 0
    anomaly_prefix_count = 0
    anomaly_sequences_without_onset = 0

    for i in range(len(y_train)):
        seq = effective_sequence(
            x_train[i],
            window_size,
        )

        if len(seq) == 0:
            continue

        full_label = np.asarray(
            y_train[i],
            dtype=np.float32,
        )

        # Always preserve the original full sequence.
        x_out.append(seq.copy())
        y_out.append(full_label.copy())
        full_count += 1

        is_anomaly = int(y_binary[i]) == 1

        onset = None
        if onset_steps is not None:
            onset = _clean_onset(
                onset_steps[i]
            )

        if is_anomaly and onset is None:
            anomaly_sequences_without_onset += 1
            continue

        T = len(seq)

        prefix_lengths = []
        for fraction in PREFIX_FRACTIONS:
            prefix_len = int(
                math.ceil(T * fraction)
            )
            prefix_len = max(
                1,
                min(prefix_len, T),
            )

            # Do not duplicate the full sequence.
            if prefix_len >= T:
                continue

            if prefix_len not in prefix_lengths:
                prefix_lengths.append(prefix_len)

        for prefix_len in prefix_lengths:
            prefix = seq[:prefix_len].copy()

            if not is_anomaly:
                prefix_label = np.asarray(
                    [1.0, 0.0],
                    dtype=np.float32,
                )
                normal_prefix_count += 1

            else:
                # onset is 1-based.
                if prefix_len < onset:
                    prefix_label = np.asarray(
                        [1.0, 0.0],
                        dtype=np.float32,
                    )
                    normal_prefix_count += 1
                else:
                    prefix_label = np.asarray(
                        [0.0, 1.0],
                        dtype=np.float32,
                    )
                    anomaly_prefix_count += 1

            x_out.append(prefix)
            y_out.append(prefix_label)

    x_obj = np.empty(
        len(x_out),
        dtype=object,
    )

    for i, seq in enumerate(x_out):
        x_obj[i] = np.asarray(
            seq,
            dtype=np.float32,
        )

    y_arr = np.asarray(
        y_out,
        dtype=np.float32,
    )

    summary = {
        "original_full_sequences_kept": full_count,
        "normal_prefixes_added": normal_prefix_count,
        "anomalous_prefixes_added": anomaly_prefix_count,
        "anomalous_sequences_without_true_onset": (
            anomaly_sequences_without_onset
        ),
        "total_early_training_samples": len(y_arr),
    }

    return x_obj, y_arr, summary


# =============================================================================
# MODEL TRAINING / FULL-SEQUENCE EVALUATION
# =============================================================================

def make_loader(
    x,
    y,
    window_size: int,
    shuffle: bool,
):
    return DataLoader(
        DataGenerator(
            x,
            y,
            window_size,
        ),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )


def build_model(
    window_size: int,
    device,
):
    model = Model(
        mode="classifier",
        num_layers=NUM_LAYERS,
        adapter_size=ADAPTER_SIZE,
        dim=EMBEDDING_DIM,
        window_size=window_size,
        nhead=NHEAD,
        dim_feedforward=FEEDFORWARD_DIM,
        dropout=DROPOUT,
    ).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    return model


def evaluate_full_sequence(
    model,
    x,
    y,
    window_size: int,
    device,
):
    loader = make_loader(
        x,
        y,
        window_size,
        shuffle=False,
    )

    model.eval()

    logits_all = []
    true_all = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(
                device
            ).to(torch.float32)

            logits = model(batch_x)

            logits_all.append(
                logits.cpu()
            )
            true_all.append(
                batch_y.cpu()
            )

    logits = torch.cat(
        logits_all,
        dim=0,
    )

    true_onehot = torch.cat(
        true_all,
        dim=0,
    ).numpy()

    y_true = np.argmax(
        true_onehot,
        axis=1,
    ).astype(int)

    y_pred = torch.argmax(
        logits,
        dim=1,
    ).numpy().astype(int)

    y_score = torch.softmax(
        logits,
        dim=1,
    )[:, 1].numpy()

    return y_true, y_pred, y_score


def train_variant(
    variant: str,
    dataset: str,
    x_train,
    y_train,
    x_dev,
    y_dev,
    window_size: int,
    checkpoint_path: Path,
    early_summary=None,
):
    """
    Same model, optimizer, LR, epochs, batch size, scheduler, loss, dimensions
    for Standard and Early. Only x_train/y_train differ.
    """
    seed_everything(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = build_model(
        window_size,
        device,
    )

    train_loader = make_loader(
        x_train,
        y_train,
        window_size,
        shuffle=True,
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0,
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
    )

    criterion = nn.BCEWithLogitsLoss()

    best_dev_f1 = -1.0
    best_epoch = -1

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n"
        + "=" * 88
    )
    print(
        f"TRAINING LOGFORMER-{variant.upper()} | DATASET={dataset}"
    )
    print(
        "=" * 88
    )
    print(
        f"window_size={window_size} | batch_size={BATCH_SIZE} | "
        f"epochs={EPOCHS} | lr={LEARNING_RATE}"
    )
    print(
        f"num_layers={NUM_LAYERS} | nhead={NHEAD} | "
        f"ffn={FEEDFORWARD_DIM} | dropout={DROPOUT}"
    )
    print(
        "optimizer=Adam(weight_decay=0) | scheduler=OneCycleLR | "
        "loss=BCEWithLogitsLoss | decision=argmax"
    )

    if early_summary is not None:
        print(
            "Early training augmentation:",
            json.dumps(
                early_summary,
                indent=2,
            ),
        )

    for epoch in range(EPOCHS):
        model.train()
        losses = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(
                device
            ).to(torch.float32)

            batch_y = batch_y.to(
                device
            ).to(torch.float32)

            optimizer.zero_grad()

            logits = model(batch_x)
            loss = criterion(
                logits,
                batch_y,
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            optimizer.step()
            scheduler.step()

            losses.append(
                float(loss.item())
            )

        dev_true, dev_pred, _ = (
            evaluate_full_sequence(
                model,
                x_dev,
                y_dev,
                window_size,
                device,
            )
        )

        dev_f1 = f1_score(
            dev_true,
            dev_pred,
            zero_division=0,
        )

        mean_loss = float(
            np.mean(losses)
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS}: "
            f"loss={mean_loss:.6f} | "
            f"DEV F1={dev_f1:.6f}"
        )

        if dev_f1 > best_dev_f1:
            best_dev_f1 = float(
                dev_f1
            )
            best_epoch = int(epoch)

            state = (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            )

            torch.save(
                {
                    "net": state,
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "dev_f1": best_dev_f1,
                    "dataset": dataset,
                    "variant": variant,
                    "window_size": window_size,
                },
                checkpoint_path,
            )

    if not checkpoint_path.exists():
        raise RuntimeError(
            "Best checkpoint was not saved."
        )

    print(
        f"Best DEV checkpoint: epoch={best_epoch + 1}, "
        f"F1={best_dev_f1:.6f}"
    )
    print(
        f"Saved: {checkpoint_path}"
    )

    return best_dev_f1, best_epoch


def load_best_model(
    checkpoint_path: Path,
    window_size: int,
    device,
):
    model = build_model(
        window_size,
        device,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    state = checkpoint["net"]

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(
            state,
            strict=True,
        )
    else:
        model.load_state_dict(
            state,
            strict=True,
        )

    model = model.to(device)
    model.eval()

    return model, checkpoint


# =============================================================================
# SEQUENTIAL PREFIX INFERENCE
# =============================================================================

def pad_one_prefix(
    prefix,
    window_size: int,
):
    arr = np.asarray(
        prefix,
        dtype=np.float32,
    )

    out = np.zeros(
        (
            1,
            window_size,
            EMBEDDING_DIM,
        ),
        dtype=np.float32,
    )

    use_len = min(
        len(arr),
        window_size,
    )

    out[
        0,
        :use_len,
        :,
    ] = arr[:use_len]

    return torch.from_numpy(out)


def evaluate_early_sequential(
    model,
    x_test,
    y_test,
    window_size: int,
    device,
):
    """
    Sequential first-alert inference using batched prefixes.

    For each test sequence evaluate prefix 1, 1:2, ..., 1:T with the
    unchanged LogFormer classifier. Prefixes are batched only for speed;
    the first prefix whose argmax is anomaly remains the alert time.

    This is NOT RL and does not learn a stopping policy.
    """
    y_true = labels_to_binary(y_test)

    y_pred = np.zeros(len(y_true), dtype=int)
    y_score = np.zeros(len(y_true), dtype=float)
    first_alert_step = np.full(len(y_true), np.nan, dtype=float)
    seq_lengths = np.zeros(len(y_true), dtype=int)

    model.eval()

    with torch.no_grad():
        for i in range(len(y_true)):
            seq = effective_sequence(x_test[i], window_size)
            T = len(seq)
            seq_lengths[i] = T

            if T == 0:
                continue

            first_alert = None
            max_seen_anomaly_prob = 0.0

            # Batch successive prefixes to reduce HDFS runtime.
            for chunk_start in range(1, T + 1, BATCH_SIZE):
                chunk_end = min(T, chunk_start + BATCH_SIZE - 1)
                steps = list(range(chunk_start, chunk_end + 1))

                batch = np.zeros(
                    (len(steps), window_size, EMBEDDING_DIM),
                    dtype=np.float32,
                )

                for j, step in enumerate(steps):
                    batch[j, :step] = seq[:step]

                batch_x = torch.from_numpy(batch).to(device)
                logits = model(batch_x)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(logits, dim=1).cpu().numpy().astype(int)

                anomaly_positions = np.where(preds == 1)[0]

                if len(anomaly_positions) > 0:
                    first_local = int(anomaly_positions[0])
                    first_alert = int(steps[first_local])
                    max_seen_anomaly_prob = max(
                        max_seen_anomaly_prob,
                        float(np.max(probs[: first_local + 1])),
                    )
                    break

                max_seen_anomaly_prob = max(
                    max_seen_anomaly_prob,
                    float(np.max(probs)),
                )

            if first_alert is not None:
                y_pred[i] = 1
                first_alert_step[i] = first_alert

            y_score[i] = max_seen_anomaly_prob

    return (
        y_true,
        y_pred,
        y_score,
        first_alert_step,
        seq_lengths,
    )


# =============================================================================
# METRICS
# =============================================================================

def _safe_auc(y_true, y_score):
    try:
        auroc = float(
            roc_auc_score(
                y_true,
                y_score,
            )
        )
    except ValueError:
        auroc = float("nan")

    try:
        auprc = float(
            average_precision_score(
                y_true,
                y_score,
            )
        )
    except ValueError:
        auprc = float("nan")

    return auroc, auprc


def classification_counts(
    y_true,
    y_pred,
):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )
    tn, fp, fn, tp = cm.ravel()
    return (
        int(tn),
        int(fp),
        int(fn),
        int(tp),
        cm,
    )


def calculate_standard_metrics(
    y_true,
    y_pred,
    y_score,
    x_test,
    window_size: int,
):
    tn, fp, fn, tp, cm = (
        classification_counts(
            y_true,
            y_pred,
        )
    )

    accuracy = float(
        accuracy_score(
            y_true,
            y_pred,
        )
    )

    precision = float(
        precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    recall = float(
        recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    f1 = float(
        f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    mcc = float(
        matthews_corrcoef(
            y_true,
            y_pred,
        )
    )

    auroc, auprc = _safe_auc(
        y_true,
        y_score,
    )

    seq_lengths = np.asarray(
        [
            len(
                effective_sequence(
                    seq,
                    window_size,
                )
            )
            for seq in x_test
        ],
        dtype=float,
    )

    true_positive_mask = (
        (y_true == 1)
        & (y_pred == 1)
    )

    detected_steps = seq_lengths[
        true_positive_mask
    ]

    detected_anomalies = int(
        np.sum(true_positive_mask)
    )

    total_anomalies = int(
        np.sum(y_true == 1)
    )

    detection_coverage = (
        detected_anomalies
        / total_anomalies
        if total_anomalies > 0
        else 0.0
    )

    avg_detection_step = (
        float(
            np.mean(
                detected_steps
            )
        )
        if detected_anomalies > 0
        else float("nan")
    )

    # Required Standard convention: waits for complete sequence.
    avg_detection_ratio = (
        1.0
        if detected_anomalies > 0
        else float("nan")
    )

    median_detection_ratio = (
        1.0
        if detected_anomalies > 0
        else float("nan")
    )

    edr25 = 0.0
    edr50 = 0.0
    edr75 = 0.0

    alert_rate = float(
        np.mean(
            y_pred == 1
        )
    )

    fp_cost = float(
        fp * FP_UNIT_COST
    )

    fn_cost = float(
        fn * FN_UNIT_COST
    )

    # sum(1.0 for every correctly detected anomaly) * 5
    delay_cost = float(
        detected_anomalies
        * DELAY_UNIT_COST
    )

    total_cost = (
        fp_cost
        + fn_cost
        + delay_cost
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc,
        "AUROC": auroc,
        "AUPRC": auprc,
        "Detection Coverage": float(detection_coverage),
        "Avg Detection Step": avg_detection_step,
        "Avg Detection Ratio": avg_detection_ratio,
        "Median Detection Ratio": median_detection_ratio,
        "EDR@25": edr25,
        "EDR@50": edr50,
        "EDR@75": edr75,
        "Alert Rate": alert_rate,
        "False-positive Cost": fp_cost,
        "False-negative Cost": fn_cost,
        "Delay Cost": delay_cost,
        "Total Cost": float(total_cost),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Confusion Matrix": cm.tolist(),
    }


def calculate_early_metrics(
    y_true,
    y_pred,
    y_score,
    first_alert_step,
    seq_lengths,
):
    tn, fp, fn, tp, cm = (
        classification_counts(
            y_true,
            y_pred,
        )
    )

    accuracy = float(
        accuracy_score(
            y_true,
            y_pred,
        )
    )

    precision = float(
        precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    recall = float(
        recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    f1 = float(
        f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    mcc = float(
        matthews_corrcoef(
            y_true,
            y_pred,
        )
    )

    auroc, auprc = _safe_auc(
        y_true,
        y_score,
    )

    true_positive_mask = (
        (y_true == 1)
        & (y_pred == 1)
        & np.isfinite(
            first_alert_step
        )
    )

    detected_steps = (
        first_alert_step[
            true_positive_mask
        ]
    )

    detected_lengths = (
        seq_lengths[
            true_positive_mask
        ].astype(float)
    )

    detected_ratios = (
        detected_steps
        / detected_lengths
        if len(detected_steps) > 0
        else np.asarray(
            [],
            dtype=float,
        )
    )

    detected_anomalies = int(
        len(detected_ratios)
    )

    total_anomalies = int(
        np.sum(y_true == 1)
    )

    detection_coverage = (
        detected_anomalies
        / total_anomalies
        if total_anomalies > 0
        else 0.0
    )

    avg_detection_step = (
        float(
            np.mean(
                detected_steps
            )
        )
        if detected_anomalies > 0
        else float("nan")
    )

    avg_detection_ratio = (
        float(
            np.mean(
                detected_ratios
            )
        )
        if detected_anomalies > 0
        else float("nan")
    )

    median_detection_ratio = (
        float(
            np.median(
                detected_ratios
            )
        )
        if detected_anomalies > 0
        else float("nan")
    )

    # EDR is conditional on correctly detected anomalies.
    # Detection Coverage separately captures missed anomalies.
    def edr(threshold):
        if detected_anomalies == 0:
            return 0.0

        return float(
            np.mean(
                detected_ratios
                <= threshold
            )
        )

    edr25 = edr(0.25)
    edr50 = edr(0.50)
    edr75 = edr(0.75)

    alert_rate = float(
        np.mean(
            y_pred == 1
        )
    )

    fp_cost = float(
        fp * FP_UNIT_COST
    )

    fn_cost = float(
        fn * FN_UNIT_COST
    )

    delay_cost = float(
        np.sum(
            detected_ratios
        )
        * DELAY_UNIT_COST
    )

    total_cost = (
        fp_cost
        + fn_cost
        + delay_cost
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc,
        "AUROC": auroc,
        "AUPRC": auprc,
        "Detection Coverage": float(detection_coverage),
        "Avg Detection Step": avg_detection_step,
        "Avg Detection Ratio": avg_detection_ratio,
        "Median Detection Ratio": median_detection_ratio,
        "EDR@25": edr25,
        "EDR@50": edr50,
        "EDR@75": edr75,
        "Alert Rate": alert_rate,
        "False-positive Cost": fp_cost,
        "False-negative Cost": fn_cost,
        "Delay Cost": delay_cost,
        "Total Cost": float(total_cost),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Confusion Matrix": cm.tolist(),
    }


# =============================================================================
# RESULT SERIALIZATION / TABLE
# =============================================================================

TABLE_METRICS = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "MCC",
    "AUROC",
    "AUPRC",
    "Detection Coverage",
    "Avg Detection Ratio",
    "EDR@25",
    "EDR@50",
    "EDR@75",
    "False-positive Cost",
    "False-negative Cost",
    "Delay Cost",
    "Total Cost",
]


def json_safe(value):
    if isinstance(value, dict):
        return {
            key: json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, list):
        return [
            json_safe(val)
            for val in value
        ]

    if isinstance(
        value,
        (np.integer,),
    ):
        return int(value)

    if isinstance(
        value,
        (np.floating,),
    ):
        value = float(value)

    if isinstance(value, float):
        if math.isnan(value):
            return None

    return value


def save_result(
    path: Path,
    dataset: str,
    variant: str,
    window_size: int,
    metrics: Dict,
    checkpoint_path: Path,
    dev_f1: float,
    best_epoch: int,
    training_summary=None,
):
    payload = {
        "dataset": dataset,
        "variant": variant,
        "window_size": int(window_size),
        "architecture": {
            "embedding_dim": EMBEDDING_DIM,
            "num_layers": NUM_LAYERS,
            "nhead": NHEAD,
            "feedforward_dim": FEEDFORWARD_DIM,
            "dropout": DROPOUT,
            "output_classes": 2,
        },
        "training": {
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "weight_decay": 0,
            "scheduler": "OneCycleLR",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "loss": "BCEWithLogitsLoss",
            "gradient_clip": GRAD_CLIP,
            "decision_rule": "argmax(logits)",
            "checkpoint_selection": "DEV F1",
            "best_dev_f1": float(dev_f1),
            "best_epoch_1based": int(best_epoch + 1),
        },
        "costs": {
            "fp_unit_cost": FP_UNIT_COST,
            "fn_unit_cost": FN_UNIT_COST,
            "delay_unit_cost": DELAY_UNIT_COST,
        },
        "checkpoint": str(checkpoint_path),
        "metrics": metrics,
    }

    if training_summary is not None:
        payload[
            "early_training_augmentation"
        ] = training_summary

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            json_safe(payload),
            f,
            indent=2,
        )


def format_number(value):
    if value is None:
        return "N/A"

    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if math.isnan(value):
        return "N/A"

    return f"{value:.6f}"


def print_comparison_table(
    standard_metrics,
    early_metrics,
):
    width = 92

    print(
        "\n"
        + "=" * width
    )
    print(
        f"{'Metric':28s} | "
        f"{'LogFormer-Standard':>27s} | "
        f"{'LogFormer-Early':>27s}"
    )
    print(
        "-" * width
    )

    for metric in TABLE_METRICS:
        std = format_number(
            standard_metrics.get(
                metric
            )
        )
        early = format_number(
            early_metrics.get(
                metric
            )
        )

        print(
            f"{metric:28s} | "
            f"{std:>27s} | "
            f"{early:>27s}"
        )

    print(
        "=" * width
    )


def result_paths(
    results_dir: Path,
    dataset: str,
):
    standard = (
        results_dir
        / f"{dataset}_LogFormer_Standard.json"
    )

    early = (
        results_dir
        / f"{dataset}_LogFormer_Early.json"
    )

    return standard, early


# =============================================================================
# VARIANTS
# =============================================================================

def run_standard(
    dataset,
    preprocessed_dir,
    results_dir,
    checkpoints_dir,
    window_size,
):
    (
        x_train,
        y_train,
        _,
    ) = load_npz_split(
        preprocessed_dir,
        dataset,
        "training",
        window_size,
    )

    (
        x_dev,
        y_dev,
        _,
    ) = load_npz_split(
        preprocessed_dir,
        dataset,
        "validation",
        window_size,
    )

    (
        x_test,
        y_test,
        _,
    ) = load_npz_split(
        preprocessed_dir,
        dataset,
        "testing",
        window_size,
    )

    checkpoint_path = (
        checkpoints_dir
        / f"{dataset}_LogFormer_Standard_best.pt"
    )

    dev_f1, best_epoch = train_variant(
        variant="standard",
        dataset=dataset,
        x_train=x_train,
        y_train=y_train,
        x_dev=x_dev,
        y_dev=y_dev,
        window_size=window_size,
        checkpoint_path=checkpoint_path,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, _ = load_best_model(
        checkpoint_path,
        window_size,
        device,
    )

    (
        y_true,
        y_pred,
        y_score,
    ) = evaluate_full_sequence(
        model,
        x_test,
        y_test,
        window_size,
        device,
    )

    metrics = calculate_standard_metrics(
        y_true,
        y_pred,
        y_score,
        x_test,
        window_size,
    )

    standard_path, _ = result_paths(
        results_dir,
        dataset,
    )

    save_result(
        standard_path,
        dataset,
        "standard",
        window_size,
        metrics,
        checkpoint_path,
        dev_f1,
        best_epoch,
    )

    print(
        f"\nSaved Standard result: {standard_path}"
    )
    print(
        json.dumps(
            json_safe(metrics),
            indent=2,
        )
    )


def run_early(
    dataset,
    preprocessed_dir,
    results_dir,
    checkpoints_dir,
    window_size,
):
    (
        x_train,
        y_train,
        onset_steps,
    ) = load_npz_split(
        preprocessed_dir,
        dataset,
        "training",
        window_size,
    )

    (
        x_dev,
        y_dev,
        _,
    ) = load_npz_split(
        preprocessed_dir,
        dataset,
        "validation",
        window_size,
    )

    (
        x_test,
        y_test,
        _,
    ) = load_npz_split(
        preprocessed_dir,
        dataset,
        "testing",
        window_size,
    )

    (
        x_early_train,
        y_early_train,
        early_summary,
    ) = build_early_training_set(
        x_train,
        y_train,
        onset_steps,
        window_size,
    )

    if onset_steps is None:
        print(
            "WARNING: no first_anomaly_step metadata was found. "
            "Normal prefixes will still be added, but anomalous prefix labels "
            "will not be fabricated. Run preprocess_LogFormer_Standard_Early.py "
            "first to preserve event-level HDFS onset metadata."
        )

    checkpoint_path = (
        checkpoints_dir
        / f"{dataset}_LogFormer_Early_best.pt"
    )

    dev_f1, best_epoch = train_variant(
        variant="early",
        dataset=dataset,
        x_train=x_early_train,
        y_train=y_early_train,
        x_dev=x_dev,
        y_dev=y_dev,
        window_size=window_size,
        checkpoint_path=checkpoint_path,
        early_summary=early_summary,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, _ = load_best_model(
        checkpoint_path,
        window_size,
        device,
    )

    (
        y_true,
        y_pred,
        y_score,
        first_alert_step,
        seq_lengths,
    ) = evaluate_early_sequential(
        model,
        x_test,
        y_test,
        window_size,
        device,
    )

    metrics = calculate_early_metrics(
        y_true,
        y_pred,
        y_score,
        first_alert_step,
        seq_lengths,
    )

    _, early_path = result_paths(
        results_dir,
        dataset,
    )

    save_result(
        early_path,
        dataset,
        "early",
        window_size,
        metrics,
        checkpoint_path,
        dev_f1,
        best_epoch,
        training_summary=early_summary,
    )

    print(
        f"\nSaved Early result: {early_path}"
    )
    print(
        json.dumps(
            json_safe(metrics),
            indent=2,
        )
    )


def run_compare(
    dataset,
    results_dir,
):
    standard_path, early_path = (
        result_paths(
            results_dir,
            dataset,
        )
    )

    if not standard_path.exists():
        raise FileNotFoundError(
            f"Missing Standard JSON: {standard_path}\n"
            "Run --variant standard first."
        )

    if not early_path.exists():
        raise FileNotFoundError(
            f"Missing Early JSON: {early_path}\n"
            "Run --variant early first."
        )

    with open(
        standard_path,
        "r",
        encoding="utf-8",
    ) as f:
        standard = json.load(f)

    with open(
        early_path,
        "r",
        encoding="utf-8",
    ) as f:
        early = json.load(f)

    print_comparison_table(
        standard["metrics"],
        early["metrics"],
    )


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run LogFormer-Standard, LogFormer-Early, "
            "or compare saved results."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name, e.g. HDFS, BGL, TH_1G, SP_150MB_ratio.",
    )

    parser.add_argument(
        "--variant",
        required=True,
        choices=(
            "standard",
            "early",
            "compare",
        ),
    )

    parser.add_argument(
        "--preprocessed-dir",
        default="preprocess/preprocessed_data",
    )

    parser.add_argument(
        "--results-dir",
        default="result_logformer_standard_early",
    )

    parser.add_argument(
        "--checkpoints-dir",
        default="checkpoints_logformer_standard_early",
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help=(
            "Use the already-preprocessed sequence/window size. "
            "If omitted, infer it from the training NPZ; "
            "if no NPZ exists, official LogFormer default 20 is used."
        ),
    )

    args = parser.parse_args()

    dataset = str(
        args.dataset
    )

    preprocessed_dir = Path(
        args.preprocessed_dir
    )

    results_dir = Path(
        args.results_dir
    )

    checkpoints_dir = Path(
        args.checkpoints_dir
    )

    if args.variant == "compare":
        # Strict requirement: compare never trains.
        run_compare(
            dataset,
            results_dir,
        )
        return

    window_size = (
        int(args.window_size)
        if args.window_size is not None
        else infer_window_size(
            preprocessed_dir,
            dataset,
        )
    )

    print(
        f"Dataset={dataset} | Variant={args.variant} | "
        f"window_size={window_size}"
    )

    if args.variant == "standard":
        run_standard(
            dataset,
            preprocessed_dir,
            results_dir,
            checkpoints_dir,
            window_size,
        )

    elif args.variant == "early":
        run_early(
            dataset,
            preprocessed_dir,
            results_dir,
            checkpoints_dir,
            window_size,
        )


if __name__ == "__main__":
    main()
