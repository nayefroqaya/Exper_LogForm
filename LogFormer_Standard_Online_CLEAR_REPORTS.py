#!/usr/bin/env python3
"""LogFormer original Standard + causal bounded-window Online evaluation.

Protocol approved for RLADLS comparison:
- Train ONCE using only original full-sequence training data, no prefix augmentation.
- Select best checkpoint using original full-sequence validation F1.
- Standard evaluates the complete sequence (up to original model window).
- Online emits irrevocable decisions for each incoming event from ONLY the
  most recent fixed-length window; the model and its weights are unchanged.
- Both evaluations use the SAME checkpoint and no test labels at inference.
- Report separate sequence-level aggregates and window-event decision metrics.
"""

import argparse
import csv
import gzip
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
# REVISED EVALUATION (no prefix training; no whole-prefix reevaluation)
# =============================================================================

def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def binary_metrics(y_true, y_pred):
    """Return exact decisions-based counts, avoiding reconstructed estimates."""
    t = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(y_pred, dtype=np.int8)
    if t.shape != p.shape:
        raise ValueError(f"Mismatched truth/prediction shapes {t.shape} vs {p.shape}")
    if t.size == 0:
        raise ValueError("Cannot score an empty test collection")
    if not np.isin(t, [0, 1]).all() or not np.isin(p, [0, 1]).all():
        raise ValueError("Labels must be binary (0/1)")
    tn, fp, fn, tp = [int(x) for x in confusion_matrix(t, p, labels=[0, 1]).ravel()]
    prec = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Precision": prec, "Recall": recall,
        "F1": safe_div(2 * prec * recall, prec + recall),
        "FPR": safe_div(fp, fp + tn),
        "Accuracy": safe_div(tp + tn, len(t)),
        "Num decisions": int(len(t)),
        "Confusion Matrix": [[tn, fp], [fn, tp]],
    }


def binary_metrics_from_counts(tp, tn, fp, fn):
    tp, tn, fp, fn = map(int, (tp, tn, fp, fn))
    prec, rec = safe_div(tp, tp + fp), safe_div(tp, tp + fn)
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Precision": prec, "Recall": rec,
        "F1": safe_div(2 * prec * rec, prec + rec),
        "FPR": safe_div(fp, fp + tn),
        "Accuracy": safe_div(tp + tn, tp + tn + fp + fn),
        "Num decisions": int(tp + tn + fp + fn),
        "Confusion Matrix": [[tn, fp], [fn, tp]],
    }


def add_sequence_costs_and_timing(stats, truths, preds, first_steps, lengths):
    """Sequence-level evaluation summary; event outcomes are reported separately.

    EDR@q has an UNCONDITIONAL denominator: all anomalous test sequences.
    AvgRatio conditions on detected anomalies; full-sequence detection is at T.
    This is an evaluation aggregate of immutable decisions, NOT a model output.
    """
    truths = np.asarray(truths, dtype=int)
    preds = np.asarray(preds, dtype=int)
    lengths = np.asarray(lengths, dtype=int)
    good = (truths == 1) & (preds == 1)
    detections = np.asarray(first_steps, dtype=float)[good]
    horizons = lengths[good].astype(float)
    valid = np.isfinite(detections) & (horizons > 0)
    if not valid.all():
        raise RuntimeError("A correctly detected sequence has an invalid first-alert step")
    ratios = detections / horizons if len(detections) else np.asarray([], dtype=float)
    if np.any(ratios <= 0) or np.any(ratios > 1):
        raise RuntimeError("First-alert ratio is outside (0,1]")
    n_anomaly = int(np.sum(truths == 1))
    stats.update({
        "Detection Coverage": safe_div(int(np.sum(good)), n_anomaly),
        "Avg Detection Step": float(np.mean(detections)) if len(detections) else None,
        "Avg Detection Ratio": float(np.mean(ratios)) if len(ratios) else None,
        "EDR@25": safe_div(int(np.sum(ratios <= .25)), n_anomaly),
        "EDR@50": safe_div(int(np.sum(ratios <= .50)), n_anomaly),
        "EDR@75": safe_div(int(np.sum(ratios <= .75)), n_anomaly),
        "False-positive Cost": float(stats["FP"] * FP_UNIT_COST),
        "False-negative Cost": float(stats["FN"] * FN_UNIT_COST),
        "Delay Cost": float(np.sum(ratios) * DELAY_UNIT_COST),
        "Total Cost": float(stats["FP"] * FP_UNIT_COST
                            + stats["FN"] * FN_UNIT_COST
                            + np.sum(ratios) * DELAY_UNIT_COST),
        "Total anomalous sequences": n_anomaly,
    })
    return stats


def observed_sequence(seq, window_size, observed_length=None):
    """Return observed original events, without exposing any future event at t.

    If explicit sequence_length is available it disambiguates genuine zero-vector
    events from padding. Old NPZ files may only have trailing-zero heuristics.
    """
    arr = np.asarray(seq, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"Invalid embedding sequence shape: {arr.shape}")
    if observed_length is not None:
        n = int(observed_length)
        if n < 0 or n > len(arr):
            raise ValueError(f"Invalid sequence_length={n} for {len(arr)} rows")
        return arr[:n]
    # Legacy NPZ fallback. Re-preprocessing is recommended to remove ambiguity.
    if len(arr) and np.any(np.any(arr != 0, axis=1)):
        last = int(np.where(np.any(arr != 0, axis=1))[0][-1]) + 1
        return arr[:last]
    return arr[:0] if len(arr) else arr


def load_split_with_metadata(preprocessed_dir, dataset, split, window_size):
    """Read original x/y and optional trusted truth metadata for scoring only."""
    x, y, onset = load_npz_split(preprocessed_dir, dataset, split, window_size)
    with np.load(split_file(preprocessed_dir, dataset, split, window_size), allow_pickle=True) as f:
        extras = {}
        for key in ("event_labels", "sequence_length", "sequence_id"):
            extras[key] = f[key] if key in f.files else None
    for key, values in extras.items():
        if values is not None and len(values) != len(y):
            raise ValueError(f"{split}: {key} and y disagree in length")
    extras["first_anomaly_step"] = onset
    return x, y, extras


def get_event_labels(extras, index, seq_length):
    vals = extras.get("event_labels")
    if vals is None:
        return None
    out = np.asarray(vals[index], dtype=int)[:seq_length]
    if len(out) != seq_length or not np.isin(out, [0, 1]).all():
        raise ValueError(f"Invalid event_labels for sequence {index}: {out.shape}")
    return out


def one_event_window_tensor(seq, end_step, window_size):
    """Construct exactly one fixed-memory causal window ending at end_step.

    Tensor order matches the original LogFormer DataGenerator: left-aligned
    observed events, followed by right-side zero padding for short windows.
    For end_step>W, the oldest events have been discarded completely.
    """
    start = max(0, end_step - window_size)
    window = seq[start:end_step]
    tensor = np.zeros((window_size, EMBEDDING_DIM), dtype=np.float32)
    tensor[:len(window)] = window
    return tensor


def write_records(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def eval_standard(model, x_test, y_test, extras, window_size, device, out_dir, dataset):
    true, pred, score = evaluate_full_sequence(model, x_test, y_test, window_size, device)
    result = binary_metrics(true, pred)
    lengths = []
    rows = []
    for i in range(len(true)):
        explicit = extras["sequence_length"]
        n = int(explicit[i]) if explicit is not None else None
        seq = observed_sequence(x_test[i], window_size, n)
        model_seen = min(len(seq), window_size)
        lengths.append(model_seen)
        sid = str(extras["sequence_id"][i]) if extras["sequence_id"] is not None else str(i)
        rows.append({
            "sequence_id": sid,
            "true_label": int(true[i]),
            "predicted_label": int(pred[i]),
            "anomaly_score": float(score[i]),
            "observed_sequence_length": int(len(seq)),
            "model_seen_length": model_seen,
            "first_alert_step": model_seen if pred[i] == 1 else "",
        })
    # Standard protocol issues one sequence prediction after seeing its model input.
    steps = [lengths[i] if pred[i] == 1 else float("nan") for i in range(len(pred))]
    result = add_sequence_costs_and_timing(result, true, pred, steps, lengths)
    result["Evaluation Unit"] = "Full-sequence (up to original LogFormer model window)"
    result["Sequences longer than model window"] = sum(len(observed_sequence(
        x_test[i], window_size,
        int(extras["sequence_length"][i]) if extras["sequence_length"] is not None else None,
    )) > window_size for i in range(len(true)))
    write_records(out_dir / f"{dataset}_LogFormer_Standard_predictions.csv", rows, list(rows[0]))
    return result


def eval_online(model, x_test, y_test, extras, window_size, device,
                out_dir, dataset, online_batch_size=32, min_events=1):
    """Strictly causal fixed-history-window event scoring.

    Every t yields a new irrevocable decision. The model does not see t+1,
    past emitted decisions are not changed, and no ground-truth is accessed
    by the scorer. Chunking groups independent causal inputs for throughput;
    outputs are written in original event order. Choose batch_size=1 for
    step-by-step physical streaming (slower; same decision semantics).

    The output is a WINDOW anomaly decision emitted at event t, not proof that
    the t-th event itself is anomalous. Corresponding ground truth is therefore
    'any anomaly in the most recent W true events' (when labels exist).
    """
    online_batch_size = int(online_batch_size)
    if online_batch_size < 1 or min_events < 1:
        raise ValueError("online_batch_size and min_events must be positive")
    true_seq = labels_to_binary(y_test)
    pred_seq = np.zeros(len(true_seq), dtype=np.int8)
    first_alerts = np.full(len(true_seq), np.nan)
    lengths = np.zeros(len(true_seq), dtype=int)
    pre_onset_alerts = 0
    post_onset_detected = 0
    known_onsets = 0
    event_counts = [0, 0, 0, 0]  # TN, FP, FN, TP vs causal window truth
    event_label_info = 0
    rows = []
    out_dir.mkdir(parents=True, exist_ok=True)
    event_path = out_dir / f"{dataset}_LogFormer_Online_event_decisions.csv.gz"
    cols = ["sequence_id", "event_index", "context_start", "context_length",
            "anomaly_score", "predicted_window_label", "true_event_label",
            "true_window_label", "is_first_alert"]
    start_clock = time.perf_counter()
    model.eval()
    with gzip.open(event_path, "wt", encoding="utf-8", newline="") as f, torch.no_grad():
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for i, seq_raw in enumerate(x_test):
            explicit = extras["sequence_length"]
            n = int(explicit[i]) if explicit is not None else None
            seq = observed_sequence(seq_raw, window_size, n)
            T = len(seq)
            lengths[i] = T
            if T == 0:
                raise ValueError(f"Empty sequence at test index {i}")
            labels = get_event_labels(extras, i, T)
            if labels is not None:
                event_label_info += 1
                if int(np.any(labels == 1)) != int(true_seq[i]):
                    raise ValueError(f"Sequence label and event labels disagree at index {i}")
            sid = str(extras["sequence_id"][i]) if extras["sequence_id"] is not None else str(i)
            onset_values = extras.get("first_anomaly_step")
            onset = int(onset_values[i]) if onset_values is not None else -1
            if onset > 0:
                known_onsets += 1
                if onset > T:
                    raise ValueError(f"Anomaly onset {onset} exceeds observed length {T} for {sid}")
            max_score, alert_events = 0.0, 0
            first, first_after_onset = None, None
            # Each input contains at most W observed events. We never feed
            # full growing prefixes to the classifier at inference time.
            for chunk_start in range(min_events, T + 1, online_batch_size):
                chunk_end = min(T + 1, chunk_start + online_batch_size)
                steps = list(range(chunk_start, chunk_end))
                batch = np.stack([one_event_window_tensor(seq, t, window_size) for t in steps])
                logits = model(torch.from_numpy(batch).to(device))
                prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                decisions = torch.argmax(logits, dim=1).cpu().numpy().astype(int)
                for j, t in enumerate(steps):
                    decision, score = int(decisions[j]), float(prob[j])
                    max_score = max(max_score, score)
                    if decision == 1:
                        alert_events += 1
                        if first is None:
                            first = t
                        if onset > 0 and t >= onset and first_after_onset is None:
                            first_after_onset = t
                    start_idx = max(0, t - window_size)
                    true_event = int(labels[t - 1]) if labels is not None else ""
                    true_window = int(np.any(labels[start_idx:t] == 1)) if labels is not None else ""
                    if labels is not None:
                        # One window decision emitted now is judged against the
                        # true recent-window state; labels are NEVER model inputs.
                        if true_window == 0 and decision == 0: event_counts[0] += 1
                        elif true_window == 0 and decision == 1: event_counts[1] += 1
                        elif true_window == 1 and decision == 0: event_counts[2] += 1
                        else: event_counts[3] += 1
                    writer.writerow({
                        "sequence_id": sid, "event_index": t,
                        "context_start": start_idx + 1,
                        "context_length": t - start_idx,
                        "anomaly_score": score,
                        "predicted_window_label": decision,
                        "true_event_label": true_event,
                        "true_window_label": true_window,
                        "is_first_alert": int(first == t),
                    })
            pred_seq[i] = int(first is not None)
            first_alerts[i] = float(first) if first is not None else np.nan
            if onset > 0 and first is not None and first < onset:
                pre_onset_alerts += 1
            if onset > 0 and first_after_onset is not None:
                post_onset_detected += 1
            rows.append({
                "sequence_id": sid, "true_label": int(true_seq[i]),
                "predicted_label_any_alert": int(pred_seq[i]),
                "observed_sequence_length": T,
                "first_alert_step": "" if first is None else first,
                "first_post_onset_alert_step": "" if first_after_onset is None else first_after_onset,
                "true_first_anomaly_step": "" if onset <= 0 else onset,
                "alert_count": alert_events,
                "max_anomaly_score": max_score,
            })
            if (i + 1) % 100 == 0 or i + 1 == len(x_test):
                elapsed = time.perf_counter() - start_clock
                print(f"Online scored {i + 1:,}/{len(x_test):,} sequences; elapsed {elapsed:.1f}s", flush=True)
    write_records(out_dir / f"{dataset}_LogFormer_Online_predictions.csv", rows, list(rows[0]))
    seq_metrics = add_sequence_costs_and_timing(
        binary_metrics(true_seq, pred_seq), true_seq, pred_seq,
        first_alerts, lengths)
    seq_metrics["Evaluation Unit"] = "Sequence-level summary: any irrevocable online alert in the sequence"
    seq_metrics["Pre-onset first alerts (sequence count)"] = pre_onset_alerts
    seq_metrics["Post-onset detected sequences"] = post_onset_detected
    seq_metrics["Sequences with known anomaly onset"] = known_onsets
    seq_metrics["Avg alert count per sequence"] = safe_div(sum(int(r["alert_count"]) for r in rows), len(rows))
    # Important: this is different from sequence aggregation, and is the
    # authentic quality measurement for immutable bounded-window decisions.
    event_metrics = binary_metrics_from_counts(
        event_counts[3], event_counts[0], event_counts[1], event_counts[2]) if event_label_info == len(x_test) else None
    if event_metrics is not None:
        event_metrics["Evaluation Unit"] = "Per emitted window decision; true label = any anomaly in recent W events"
        event_metrics["False-positive Cost"] = float(event_metrics["FP"] * FP_UNIT_COST)
        event_metrics["False-negative Cost"] = float(event_metrics["FN"] * FN_UNIT_COST)
    return {
        "metrics": seq_metrics,
        "online_window_decision_metrics": event_metrics,
        "event_ground_truth_available": event_label_info == len(x_test),
        "event_decisions_path": str(event_path),
        "online_batch_size": online_batch_size,
        "online_min_events": int(min_events),
        "online_seconds": float(time.perf_counter() - start_clock),
    }


def print_metric_report(title, metrics):
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)
    for key in ("Precision", "Recall", "F1", "FPR", "TP", "TN", "FP", "FN",
                "Avg Detection Ratio", "EDR@25", "EDR@50", "EDR@75",
                "False-positive Cost", "False-negative Cost", "Delay Cost", "Total Cost"):
        if key in metrics:
            val = metrics[key]
            print(f"{key:28s}: {val:.6f}" if isinstance(val, float) else f"{key:28s}: {val}")
    print("=" * 76)


def report_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)


def require_node_blockid_audit(preprocessed_dir: Path, dataset: str, window_size: int):
    """Refuse to train/evaluate from legacy or unaudited split NPZ files.

    The preprocessing script is the source of truth.  It audits original
    Node_block_id values before generating the split-specific NPZ archives.
    """
    audit_path = preprocessed_dir / f"{dataset}_original_id_overlap_audit.json"
    if not audit_path.is_file():
        raise RuntimeError(
            f"MISSING MANDATORY Node_block_id audit: {audit_path}. "
            "First run LogFormer_Preprocess_Standard_Online_NODE_BLOCKID_REQUIRED.py."
        )
    with audit_path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    if (report.get("dataset") != dataset or report.get("result") != "PASS"
            or report.get("id_column") != "Node_block_id"):
        raise RuntimeError(
            f"Node_block_id audit for {dataset} is not PASS: {audit_path}. "
            f"Result={report.get('result')}, id_column={report.get('id_column')}"
        )
    for split in ("training", "validation", "testing"):
        npz_path = split_file(preprocessed_dir, dataset, split, window_size)
        if not npz_path.is_file():
            raise FileNotFoundError(f"Preprocessed split not found: {npz_path}")
        with np.load(npz_path, allow_pickle=False) as data:
            if ("original_id_audit_status" not in data
                    or "original_id_audit_column" not in data):
                raise RuntimeError(
                    f"LEGACY/UNAUDITED NPZ: {npz_path}. "
                    "Regenerate all splits with the mandatory Node_block_id preprocessor."
                )
            status = str(data["original_id_audit_status"].item())
            column = str(data["original_id_audit_column"].item())
            if status != "PASS" or column != "Node_block_id":
                raise RuntimeError(
                    f"Invalid NPZ BlockId audit for {npz_path}: {status}/{column}"
                )
    print(f"PASS: mandatory Node_block_id audit and all three NPZ files verified for {dataset}.")


def run_experiment(args):
    preprocessed_dir = Path(args.preprocessed_dir)
    results_dir = Path(args.results_dir)
    checkpoints_dir = Path(args.checkpoints_dir)
    dataset = args.dataset
    window_size = args.window_size if args.window_size else infer_window_size(preprocessed_dir, dataset)
    if not (1 <= window_size <= 5000):
        raise ValueError("window_size must be between 1 and 5000")

    # Prevent bypassing leakage protection by training directly on old NPZs.
    require_node_blockid_audit(preprocessed_dir, dataset, window_size)

    checkpoint_path = checkpoints_dir / f"{dataset}_LogFormer_original_train_best.pt"
    if args.mode == "train":
        x_train, y_train, _ = load_npz_split(preprocessed_dir, dataset, "training", window_size)
        x_val, y_val, _ = load_npz_split(preprocessed_dir, dataset, "validation", window_size)
        t0 = time.perf_counter()
        dev_f1, epoch = train_variant(
            variant="original", dataset=dataset,
            x_train=x_train, y_train=y_train,
            x_dev=x_val, y_dev=y_val,
            window_size=window_size, checkpoint_path=checkpoint_path)
        train_seconds = time.perf_counter() - t0
        del x_train, y_train, x_val, y_val
    else:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint missing: {checkpoint_path}. First run --mode train")
        dev_f1, epoch, train_seconds = None, None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_best_model(checkpoint_path, window_size, device)
    if dev_f1 is None:
        dev_f1, epoch = float(ckpt["dev_f1"]), int(ckpt["epoch"])
    x_test, y_test, extras = load_split_with_metadata(preprocessed_dir, dataset, "testing", window_size)

    test_lengths = extras.get("sequence_length")
    if test_lengths is None:
        test_lengths = np.asarray([len(observed_sequence(q, window_size)) for q in x_test])
    longer = int(np.sum(np.asarray(test_lengths) > window_size))
    if longer:
        warning = (f"{longer} test sequences exceed the model window W={window_size}. "
                   "Original Standard reads only the first W, while Online can "
                   "read later events. Sequence-level comparison is NOT a "
                   "matched-horizon experiment. Segment into W-sized sequences "
                   "or use pad_truncate and regenerate NPZ.")
        if args.variant == "both" and not args.allow_long_sequences:
            raise ValueError(warning + " Use --allow-long-sequences only after reviewing the protocol.")
        print("WARNING:", warning)

    results_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    common = {
        "dataset": dataset,
        "window_size": int(window_size),
        "checkpoint": str(checkpoint_path),
        "same_checkpoint_for_standard_and_online": True,
        "training": {
            "training_procedure": "Original LogFormer, no prefix augmentation",
            "best_val_f1": float(dev_f1),
            "best_epoch_1based": int(epoch) + 1,
            "training_seconds_this_run": train_seconds,
            "seed": int(SEED),
        },
        "cost_weights": {"FP": FP_UNIT_COST, "FN": FN_UNIT_COST, "delay": DELAY_UNIT_COST},
    }
    if args.variant in ("standard", "both"):
        t0 = time.perf_counter()
        std = eval_standard(model, x_test, y_test, extras, window_size, device, results_dir, dataset)
        reports["standard"] = std
        report_json(results_dir / f"{dataset}_LogFormer_Standard.json", {
            **common, "mode": "standard", "metrics": std,
            "inference_seconds": time.perf_counter() - t0})
        print_metric_report("LogFormer STANDARD — sequence-level (full-sequence detection)", std)
    if args.variant in ("online", "both"):
        online = eval_online(model, x_test, y_test, extras, window_size, device,
                             results_dir, dataset, args.online_batch_size, args.online_min_events)
        reports["online"] = online
        report_json(results_dir / f"{dataset}_LogFormer_Online.json", {
            **common, "mode": "online", **online,
            "methodology_note": (
                "Model uses only the last W observed events and emits irrevocable "
                "window-level decisions. Sequence-level any-alert scores are offline "
                "summaries, NOT retracted or recomputed inference decisions. "
                "For event-level performance use online_window_decision_metrics. "
                "No online labels are used to adapt the model.")})
        print_metric_report("LogFormer ONLINE — sequence-level (first-alert aggregation)", online["metrics"])
        if args.show_window_metrics:
            if online["online_window_decision_metrics"] is not None:
                print_metric_report("LogFormer ONLINE DIAGNOSTIC ONLY — bounded-window decisions (not Standard)", online["online_window_decision_metrics"])
            else:
                print("Online bounded-window diagnostics unavailable: regenerate NPZ with revised preprocessing.")
    if len(reports) == 2:
        # Match PLELog Standard vs Online reporting exactly; display-only change.
        metric_names = [
            ("Precision", "Precision"), ("Recall", "Recall"),
            ("F1", "F1"), ("FPR", "FPR"),
            ("Avg Detection Ratio", "AvgRatio"),
            ("EDR@25", "EDR@25"), ("EDR@50", "EDR@50"),
            ("EDR@75", "EDR@75"),
            ("False-positive Cost", "FP Cost"),
            ("False-negative Cost", "FN Cost"),
            ("Delay Cost", "Delay Cost"),
            ("Total Cost", "Total Cost"),
        ]
        standard_metrics = reports["standard"]
        online_metrics = reports["online"]["metrics"]
        def fmt_metric(value):
            return "N/A" if value is None else f"{float(value):.6f}"
        lines = [
            "=" * 76,
            f"LogFormer {dataset} - ORIGINAL TRAINING, STANDARD vs ONLINE",
            "=" * 76,
            f"{'Metric':<23} {'Standard':>23} {'Online':>23}",
            "-" * 76,
        ]
        for key, label in metric_names:
            lines.append(
                f"{label:<23} {fmt_metric(standard_metrics.get(key)):>23} "
                f"{fmt_metric(online_metrics.get(key)):>23}"
            )
        lines.append("=" * 76)
        table = "\n".join(lines)
        print("\n" + table)
        (results_dir / f"{dataset}_LogFormer_Standard_vs_Online.txt").write_text(
            table + "\n", encoding="utf-8"
        )
    if len(reports) == 2:
        from LogFormer_Prediction_Audit import audit_prediction_outputs
        reports["prediction_audit"] = audit_prediction_outputs(dataset, results_dir)
    print(f"\nResults saved under: {results_dir.resolve()}")
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variant", choices=("standard", "online", "both"), default="both")
    parser.add_argument("--mode", choices=("train", "eval"), default="train",
                        help="train=original training once; eval=reuse saved checkpoint")
    parser.add_argument("--preprocessed-dir", default="preprocess/preprocessed_data")
    parser.add_argument("--results-dir", default="results_logformer_standard_online")
    parser.add_argument("--checkpoints-dir", default="checkpoints_logformer_standard_online")
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--allow-long-sequences", action="store_true",
                        help="Permit unequal Standard/Online horizons for >W streams; not recommended for paper comparisons")
    parser.add_argument("--online-batch-size", type=int, default=32,
                        help="Independent causal windows per inference batch; use 1 for literal streaming")
    parser.add_argument("--online-min-events", type=int, default=1,
                        help="Earliest event index with a model decision; default 1 uses right padding")
    parser.add_argument("--show-window-metrics", action="store_true",
                        help="Also print separate Online window-level diagnostics (never Standard)")
    args = parser.parse_args()
    if args.online_batch_size < 1 or args.online_min_events < 1:
        parser.error("--online-batch-size and --online-min-events must be >= 1")
    run_experiment(args)


if __name__ == "__main__":
    main()
