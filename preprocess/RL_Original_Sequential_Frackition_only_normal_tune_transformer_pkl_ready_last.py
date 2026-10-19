#!/usr/bin/env python3
"""
Unified LogFormer target tuning.

Behavior:

1) IN-DOMAIN
   setting: in_domain

   No transfer/adaptation stage is needed.
   This script exits cleanly with a message.
   The final in-domain result is produced by
   Fraktion_normaandanomal_train_transformer_pkl_ready_last.py.

2) CROSS-DATASET
   setting: cross_dataset

   Supports:
       - 1, 2, or 3 source datasets
       - exactly one target dataset

   Pipeline:
       load combined-source checkpoint
           ->
       sample target_train_fraction from BOTH:
           normal target train blocks
           anomaly target train blocks
           ->
       adapter tuning
           ->
       target validation selects best checkpoint
           ->
       target test exactly once

Example:
    source_dataset_names:
      - BGL
      - HDFS

    target_dataset_name: TH_1G

    target_normal_fraction: 0.20

means:
    20% of target normal train blocks
    +
    0% of target anomalous train blocks
"""

import argparse
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm

from dataloader import DataGenerator
from model import Model


# ============================================================
# Configuration
# ============================================================

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_setting(cfg) -> str:
    setting = str(cfg.get("setting", "in_domain")).strip().lower()

    if setting not in {"in_domain", "cross_dataset"}:
        raise ValueError(
            "setting must be either 'in_domain' or 'cross_dataset'."
        )

    return setting


def get_source_names(cfg) -> List[str]:
    names = cfg.get("source_dataset_names")

    if not isinstance(names, list):
        raise ValueError(
            "source_dataset_names must be a YAML list."
        )

    names = [str(x) for x in names]

    if not (1 <= len(names) <= 3):
        raise ValueError(
            "cross_dataset mode supports 1, 2, or 3 source datasets."
        )

    if len(set(names)) != len(names):
        raise ValueError("Duplicate source datasets are not allowed.")

    return names


def get_target_name(cfg) -> str:
    target = cfg.get("target_dataset_name")

    if target is None:
        raise ValueError(
            "target_dataset_name is required in cross_dataset mode."
        )

    return str(target)


def get_preprocessed_dir(cfg):
    return str(
        cfg.get(
            "preprocessed_dir",
            cfg.get("output_dir", "preprocess/preprocessed_data"),
        )
    )


# ============================================================
# Data helpers
# ============================================================

def load_split(
    preprocessed_dir: str,
    dataset_name: str,
    split_name: str,
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    path = os.path.join(
        preprocessed_dir,
        f"{dataset_name}_{split_name}_block_w{window_size}.npz",
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing preprocessed file: {path}"
        )

    data = np.load(path, allow_pickle=True)

    x = data["x"]
    y = data["y"]

    del data

    if len(x) == 0:
        raise ValueError(
            f"{dataset_name} {split_name} NPZ is empty."
        )

    return x, y


def print_distribution(
    name: str,
    y,
):
    labels = np.argmax(y, axis=1)

    print(
        f"{name}: "
        f"total={len(y)}, "
        f"normal={int(np.sum(labels == 0))}, "
        f"anomaly={int(np.sum(labels == 1))}"
    )


def select_normal_target_fraction(
    x,
    y,
    fraction: float,
    seed: int,
):
    """
    Select ONLY normal target-domain training sequences.

    Label encoding:
        [1, 0] = normal
        [0, 1] = anomaly

    fraction = 0.20 means:
        20% of NORMAL target training blocks
        0% of ANOMALOUS target training blocks

    Each NPZ item is one complete block/sequence, so block
    boundaries are preserved.
    """
    if not (0 < fraction <= 1):
        raise ValueError(
            "target_normal_fraction must be > 0 and <= 1."
        )

    labels = np.argmax(y, axis=1)

    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]

    if len(normal_idx) == 0:
        raise ValueError(
            "Target training data contain no normal blocks."
        )

    print("\nFull target training distribution:")
    print(f"  total:   {len(y)}")
    print(f"  normal:  {len(normal_idx)}")
    print(f"  anomaly: {len(anomaly_idx)}")

    rng = np.random.default_rng(seed)

    n_normal = max(
        1,
        int(round(len(normal_idx) * fraction)),
    )

    n_normal = min(
        n_normal,
        len(normal_idx),
    )

    selected = rng.choice(
        normal_idx,
        size=n_normal,
        replace=False,
    )

    rng.shuffle(selected)

    x_selected = x[selected]
    y_selected = y[selected]

    # Safety check: adaptation data must contain ZERO anomalies.
    selected_labels = np.argmax(y_selected, axis=1)

    if np.any(selected_labels != 0):
        raise RuntimeError(
            "Internal error: anomaly found in normal-only target sample."
        )

    print("\nSelected NORMAL-ONLY target adaptation set:")
    print(
        f"  target_normal_fraction: {fraction:.4f} "
        f"({fraction * 100:.2f}%)"
    )
    print(f"  selected normal blocks: {len(y_selected)}")
    print("  selected anomaly blocks: 0")

    print_distribution(
        "  selected target train",
        y_selected,
    )

    return x_selected, y_selected


# ============================================================
# Model helpers
# ============================================================

def normalize_state_dict_keys(state_dict):
    clean = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]

        clean[key] = value

    return clean


def get_model_state(model):
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()

    return model.state_dict()


def evaluate(
    model,
    loader,
    device,
):
    model.eval()

    pred_all = []
    true_all = []
    score_all = []

    start = time.time()

    with torch.no_grad():
        for x, y in tqdm(
            loader,
            desc="Evaluation",
            leave=False,
        ):
            x = x.to(device).to(torch.float32)
            y = y.to(device).to(torch.float32)

            out = model(x)

            pred_all.append(out.cpu())
            true_all.append(y.cpu())
            score_all.append(torch.softmax(out, dim=1)[:, 1].cpu())

    if not pred_all:
        raise ValueError("Evaluation loader is empty.")

    prediction_time = time.time() - start

    pred = torch.cat(pred_all, dim=0).numpy()
    true = torch.cat(true_all, dim=0).numpy()
    anomaly_scores = torch.cat(score_all, dim=0).numpy()

    pred_labels = np.argmax(pred, axis=1)
    true_labels = np.argmax(true, axis=1)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            pred_labels,
            average="binary",
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        true_labels,
        pred_labels,
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
        "prediction_time": prediction_time,
        "y_true": true_labels,
        "y_pred": pred_labels,
        "y_score": anomaly_scores,
    }



def get_effective_sequence_lengths(x_sequences, window_size):
    """Return effective sequence lengths after the model window-size cap."""
    lengths = []
    for seq in x_sequences:
        seq_arr = np.asarray(seq)
        if seq_arr.ndim == 0:
            seq_len = 1
        else:
            seq_len = int(seq_arr.shape[0])
        lengths.append(min(seq_len, int(window_size)))
    return np.asarray(lengths, dtype=float)


def build_full_sequence_report(
    dataset_name,
    result,
    seq_lengths,
    fp_unit_cost=10.0,
    fn_unit_cost=20.0,
    delay_unit_cost=5.0,
):
    """
    Report the ORIGINAL LogFormer full-sequence result.

    Early-detection values here are only the explicit full-sequence convention:
    a correctly detected anomaly is assigned to the sequence end.
    """
    y_true = np.asarray(result["y_true"], dtype=int)
    y_pred = np.asarray(result["y_pred"], dtype=int)
    seq_lengths = np.asarray(seq_lengths, dtype=float)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Normal (0)", "Anomaly (1)"],
        digits=4,
        zero_division=0,
    )

    total_anomalies = int(tp + fn)
    detected_anomalies = int(tp)
    coverage = detected_anomalies / total_anomalies if total_anomalies else 0.0

    detected_mask = (y_true == 1) & (y_pred == 1)
    avg_step = float(np.mean(seq_lengths[detected_mask])) if detected_anomalies else None
    avg_ratio = 1.0 if detected_anomalies else None

    fp_cost = float(fp * fp_unit_cost)
    fn_cost = float(fn * fn_unit_cost)
    delay_cost = float(tp * delay_unit_cost)
    total_cost = fp_cost + fn_cost + delay_cost

    lines = []
    lines.append(f"[ORIGINAL LOGFORMER - FULL-SEQUENCE]  # {dataset_name}")
    lines.append("")
    lines.append("[Classification Metrics - Anomaly Class]")
    lines.append("Positive class        : 1 = anomaly")
    lines.append(f"Precision             : {result['precision']:.4f}")
    lines.append(f"Recall / TPR          : {result['recall']:.4f}")
    lines.append(f"F1-score              : {result['f1']:.4f}")
    lines.append("")
    lines.append("[Classification Report - Class 0 and Class 1]")
    lines.append("Class 0               : Normal")
    lines.append("Class 1               : Anomaly")
    lines.append("")
    lines.append(report.rstrip())
    lines.append("")
    lines.append("[Confusion Matrix]")
    lines.append("Labels: 0=normal, 1=anomaly")
    lines.append(str(cm))
    lines.append(f"TP={tp} TN={tn} FP={fp} FN={fn}")
    lines.append("")
    lines.append("[Early Detection Metrics - Full-Sequence Convention]")
    lines.append("Decision point         : end of complete sequence")
    lines.append(f"Total anomalies       : {total_anomalies}")
    lines.append(f"Detected anomalies    : {detected_anomalies}")
    lines.append(f"Detection coverage    : {coverage:.4f}")
    lines.append(f"Avg detection step    : {'N/A' if avg_step is None else f'{avg_step:.4f}'}")
    lines.append(f"Avg detection ratio   : {'N/A' if avg_ratio is None else f'{avg_ratio:.4f}'}")
    lines.append("EDR@25                : 0.0000")
    lines.append("EDR@50                : 0.0000")
    lines.append("EDR@75                : 0.0000")
    lines.append("")
    lines.append("[Cost-Sensitive Metrics]")
    lines.append(f"False-positive cost   : {fp_cost:.4f}")
    lines.append(f"False-negative cost   : {fn_cost:.4f}")
    lines.append(f"Delay cost            : {delay_cost:.4f}")
    lines.append(f"Total cost            : {total_cost:.4f}")
    return "\n".join(lines)


def evaluate_sequential_prefixes(
    model,
    x_sequences,
    y,
    device,
    window_size,
    embedding_dim,
    min_prefix=1,
    stride=1,
):
    """
    Sequentialize the ORIGINAL trained LogFormer at inference time only.

    For each sequence, evaluate progressively longer prefixes. Every prefix is
    zero-padded to the original fixed LogFormer window_size, so model weights and
    architecture are unchanged. The first prefix predicted as anomaly is the
    alert/detection step. If no prefix is anomalous, the final prediction is normal.
    """
    model.eval()
    true_labels = np.argmax(y, axis=1).astype(int)
    pred_labels = np.zeros(len(true_labels), dtype=int)
    detection_steps = np.full(len(true_labels), np.nan, dtype=float)
    max_anomaly_scores = np.zeros(len(true_labels), dtype=float)
    seq_lengths = get_effective_sequence_lengths(x_sequences, window_size)

    min_prefix = max(1, int(min_prefix))
    stride = max(1, int(stride))
    start_time = time.time()

    with torch.no_grad():
        for i, seq in enumerate(tqdm(x_sequences, desc="Sequential prefix evaluation", leave=False)):
            seq_arr = np.asarray(seq, dtype=np.float32)
            if seq_arr.ndim != 2:
                raise ValueError(f"Invalid sequence shape at test index {i}: {seq_arr.shape}")
            if seq_arr.shape[1] != int(embedding_dim):
                raise ValueError(
                    f"Embedding dimension mismatch at test index {i}: "
                    f"got {seq_arr.shape[1]}, expected {embedding_dim}."
                )

            T = min(int(seq_arr.shape[0]), int(window_size))
            if T <= 0:
                continue

            first = min(min_prefix, T)
            steps = list(range(first, T + 1, stride))
            if not steps or steps[-1] != T:
                steps.append(T)

            # Evaluate all prefixes of this sequence as one batch.
            prefix_batch = np.zeros(
                (len(steps), int(window_size), int(embedding_dim)),
                dtype=np.float32,
            )
            for j, t in enumerate(steps):
                prefix_batch[j, :t, :] = seq_arr[:t, :]

            logits = model(torch.from_numpy(prefix_batch).to(device))
            prefix_probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            max_anomaly_scores[i] = float(np.max(prefix_probs))
            prefix_preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            hit = np.where(prefix_preds == 1)[0]

            if len(hit) > 0:
                first_hit = int(hit[0])
                pred_labels[i] = 1
                detection_steps[i] = float(steps[first_hit])
            else:
                pred_labels[i] = 0

    prediction_time = time.time() - start_time
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        pred_labels,
        average="binary",
        zero_division=0,
    )
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
        "prediction_time": prediction_time,
        "y_true": true_labels,
        "y_pred": pred_labels,
        "detection_steps": detection_steps,
        "seq_lengths": seq_lengths,
        "y_score": max_anomaly_scores,
    }


def build_sequential_report(
    dataset_name,
    result,
    fp_unit_cost=10.0,
    fn_unit_cost=20.0,
    delay_unit_cost=5.0,
):
    """Build real early-detection metrics from prefix-by-prefix predictions."""
    y_true = np.asarray(result["y_true"], dtype=int)
    y_pred = np.asarray(result["y_pred"], dtype=int)
    detection_steps = np.asarray(result["detection_steps"], dtype=float)
    seq_lengths = np.asarray(result["seq_lengths"], dtype=float)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Normal (0)", "Anomaly (1)"],
        digits=4,
        zero_division=0,
    )

    total_anomalies = int(np.sum(y_true == 1))
    tp_mask = (y_true == 1) & (y_pred == 1) & np.isfinite(detection_steps)
    detected_anomalies = int(np.sum(tp_mask))
    coverage = detected_anomalies / total_anomalies if total_anomalies else 0.0

    if detected_anomalies:
        tp_steps = detection_steps[tp_mask]
        tp_lengths = seq_lengths[tp_mask]
        ratios = tp_steps / np.maximum(tp_lengths, 1.0)
        avg_step = float(np.mean(tp_steps))
        avg_ratio = float(np.mean(ratios))
    else:
        ratios = np.asarray([], dtype=float)
        avg_step = None
        avg_ratio = None

    # EDR@k = fraction of ALL anomalous sequences detected by that fraction.
    edr25 = float(np.sum(ratios <= 0.25) / total_anomalies) if total_anomalies else 0.0
    edr50 = float(np.sum(ratios <= 0.50) / total_anomalies) if total_anomalies else 0.0
    edr75 = float(np.sum(ratios <= 0.75) / total_anomalies) if total_anomalies else 0.0

    fp_cost = float(fp * fp_unit_cost)
    fn_cost = float(fn * fn_unit_cost)
    delay_cost = float(np.sum(ratios) * delay_unit_cost)
    total_cost = fp_cost + fn_cost + delay_cost

    lines = []
    lines.append(f"[SEQUENTIAL LOGFORMER - PREFIX EVALUATION]  # {dataset_name}")
    lines.append("")
    lines.append("[Classification Metrics - Anomaly Class]")
    lines.append("Positive class        : 1 = anomaly")
    lines.append(f"Precision             : {result['precision']:.4f}")
    lines.append(f"Recall / TPR          : {result['recall']:.4f}")
    lines.append(f"F1-score              : {result['f1']:.4f}")
    lines.append("")
    lines.append("[Classification Report - Class 0 and Class 1]")
    lines.append("Class 0               : Normal")
    lines.append("Class 1               : Anomaly")
    lines.append("")
    lines.append(report.rstrip())
    lines.append("")
    lines.append("[Confusion Matrix]")
    lines.append("Labels: 0=normal, 1=anomaly")
    lines.append(str(cm))
    lines.append(f"TP={tp} TN={tn} FP={fp} FN={fn}")
    lines.append("")
    lines.append("[Early Detection Metrics - Measured from Prefix Predictions]")
    lines.append(f"Total anomalies       : {total_anomalies}")
    lines.append(f"Detected anomalies    : {detected_anomalies}")
    lines.append(f"Detection coverage    : {coverage:.4f}")
    lines.append(f"Avg detection step    : {'N/A' if avg_step is None else f'{avg_step:.4f}'}")
    lines.append(f"Avg detection ratio   : {'N/A' if avg_ratio is None else f'{avg_ratio:.4f}'}")
    lines.append(f"EDR@25                : {edr25:.4f}")
    lines.append(f"EDR@50                : {edr50:.4f}")
    lines.append(f"EDR@75                : {edr75:.4f}")
    lines.append("")
    lines.append("[Cost-Sensitive Metrics]")
    lines.append(f"False-positive cost   : {fp_cost:.4f}")
    lines.append(f"False-negative cost   : {fn_cost:.4f}")
    lines.append(f"Delay cost            : {delay_cost:.4f}")
    lines.append(f"Total cost            : {total_cost:.4f}")
    lines.append(f"Sequential inference time (s): {result['prediction_time']:.4f}")
    return "\n".join(lines)


def _safe_div(num, den):
    return float(num / den) if den else 0.0


def compute_comparison_metrics(
    result,
    seq_lengths,
    mode,
    fp_unit_cost=10.0,
    fn_unit_cost=20.0,
    delay_unit_cost=5.0,
):
    """Compute classification, early-detection and cost metrics for one mode."""
    y_true = np.asarray(result["y_true"], dtype=int)
    y_pred = np.asarray(result["y_pred"], dtype=int)
    y_score = np.asarray(result.get("y_score", y_pred), dtype=float)
    seq_lengths = np.asarray(seq_lengths, dtype=float)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    specificity = _safe_div(tn, tn + fp)
    fpr = _safe_div(fp, fp + tn)
    fnr = _safe_div(fn, fn + tp)

    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Precision": float(precision),
        "Recall": float(recall),
        "Specificity": float(specificity),
        "F1": float(f1),
        "F2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "FPR": float(fpr),
        "FNR": float(fnr),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }

    if len(np.unique(y_true)) == 2:
        metrics["AUROC"] = float(roc_auc_score(y_true, y_score))
        metrics["AUPRC"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["AUROC"] = np.nan
        metrics["AUPRC"] = np.nan

    total_anomalies = int(np.sum(y_true == 1))
    tp_mask = (y_true == 1) & (y_pred == 1)
    detected_anomalies = int(np.sum(tp_mask))
    coverage = _safe_div(detected_anomalies, total_anomalies)

    if mode == "standard":
        detection_steps = seq_lengths[tp_mask]
        ratios = np.ones(detected_anomalies, dtype=float)
        edr25 = edr50 = edr75 = 0.0
    elif mode == "sequential":
        all_steps = np.asarray(result["detection_steps"], dtype=float)
        valid_tp_mask = tp_mask & np.isfinite(all_steps)
        detection_steps = all_steps[valid_tp_mask]
        detected_anomalies = int(np.sum(valid_tp_mask))
        coverage = _safe_div(detected_anomalies, total_anomalies)
        ratios = detection_steps / np.maximum(seq_lengths[valid_tp_mask], 1.0)
        edr25 = _safe_div(np.sum(ratios <= 0.25), total_anomalies)
        edr50 = _safe_div(np.sum(ratios <= 0.50), total_anomalies)
        edr75 = _safe_div(np.sum(ratios <= 0.75), total_anomalies)
    else:
        raise ValueError("mode must be 'standard' or 'sequential'.")

    avg_step = float(np.mean(detection_steps)) if len(detection_steps) else np.nan
    avg_ratio = float(np.mean(ratios)) if len(ratios) else np.nan
    median_ratio = float(np.median(ratios)) if len(ratios) else np.nan
    alert_rate = _safe_div(np.sum(y_pred == 1), len(y_pred))

    fp_cost = float(fp * fp_unit_cost)
    fn_cost = float(fn * fn_unit_cost)
    delay_cost = float(np.sum(ratios) * delay_unit_cost)
    total_cost = fp_cost + fn_cost + delay_cost
    avg_cost = _safe_div(total_cost, len(y_true))

    metrics.update({
        "Detection Coverage": float(coverage),
        "Avg Detection Step": avg_step,
        "Avg Detection Ratio": avg_ratio,
        "Median Detection Ratio": median_ratio,
        "EDR@25": float(edr25),
        "EDR@50": float(edr50),
        "EDR@75": float(edr75),
        "Alert Rate": float(alert_rate),
        "False-positive Cost": fp_cost,
        "False-negative Cost": fn_cost,
        "Delay Cost": delay_cost,
        "Total Cost": total_cost,
        "Avg Cost / Sequence": float(avg_cost),
    })
    return metrics


def build_standard_vs_sequential_table(
    dataset_name,
    standard_result,
    sequential_result,
    seq_lengths,
    fp_unit_cost=10.0,
    fn_unit_cost=20.0,
    delay_unit_cost=5.0,
):
    standard = compute_comparison_metrics(
        standard_result,
        seq_lengths,
        mode="standard",
        fp_unit_cost=fp_unit_cost,
        fn_unit_cost=fn_unit_cost,
        delay_unit_cost=delay_unit_cost,
    )
    sequential = compute_comparison_metrics(
        sequential_result,
        seq_lengths,
        mode="sequential",
        fp_unit_cost=fp_unit_cost,
        fn_unit_cost=fn_unit_cost,
        delay_unit_cost=delay_unit_cost,
    )

    metric_order = [
        "Accuracy",
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "F1",
        "F2",
        "FPR",
        "FNR",
        "MCC",
        "AUROC",
        "AUPRC",
        "Detection Coverage",
        "Avg Detection Step",
        "Avg Detection Ratio",
        "Median Detection Ratio",
        "EDR@25",
        "EDR@50",
        "EDR@75",
        "Alert Rate",
        "False-positive Cost",
        "False-negative Cost",
        "Delay Cost",
        "Total Cost",
        "Avg Cost / Sequence",
    ]

    width = 92
    lines = []
    lines.append("=" * width)
    lines.append(f"LOGFORMER STANDARD VS SEQUENTIAL COMPARISON  # {dataset_name}")
    lines.append("=" * width)
    lines.append(f"{'Metric':<29} | {'Standard':>20} | {'Sequential':>20}")
    lines.append("-" * width)

    for name in metric_order:
        s = standard[name]
        q = sequential[name]
        s_text = "N/A" if not np.isfinite(s) else f"{s:.6f}"
        q_text = "N/A" if not np.isfinite(q) else f"{q:.6f}"
        lines.append(f"{name:<29} | {s_text:>20} | {q_text:>20}")

    lines.append("=" * width)
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="preprocess/RL_Sequential_config_only_normal_cross_dataset_last.yml",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    setting = get_setting(cfg)

    # ========================================================
    # IN-DOMAIN:
    # no target adaptation stage
    # ========================================================

    if setting == "in_domain":
        dataset_name = str(
            cfg.get(
                "in_domain_dataset_name",
                "<missing>",
            )
        )

        print("\n============================================================")
        print("IN-DOMAIN MODE")
        print("============================================================")
        print("Dataset:", dataset_name)
        print(
            "No tune/transfer stage is required for in-domain LogFormer."
        )
        print(
            "Use Fraktion_normaandanomal_train_transformer_pkl_ready_last.py for "
            "train -> validation -> final test."
        )
        print("Nothing was changed.")
        print("============================================================")

        return

    # ========================================================
    # CROSS-DATASET
    # ========================================================

    source_names = get_source_names(cfg)
    target_name = get_target_name(cfg)

    if target_name in source_names:
        raise ValueError(
            "Target dataset cannot also be a source dataset."
        )

    source_tag = "_".join(source_names)

    preprocessed_dir = get_preprocessed_dir(cfg)

    window_size = int(cfg.get("window_size", 120))
    seed = int(cfg.get("seed", 123))

    embedding_dim = int(cfg.get("embedding_dim", 768))
    num_layers = int(cfg.get("num_layers", 1))
    adapter_size = int(cfg.get("adapter_size", 64))
    nhead = int(cfg.get("nhead", 8))
    dropout = float(cfg.get("dropout", 0.1))
    ff_multiplier = int(cfg.get("feedforward_multiplier", 4))

    source_mode = str(cfg.get("source_mode", "classifier"))
    source_lr = float(cfg.get("source_lr", 1e-5))

    tune_mode = str(cfg.get("tune_mode", "adapter"))
    batch_size = int(cfg.get("target_batch_size", 64))
    epochs = int(cfg.get("target_epochs", 20))
    lr = float(cfg.get("target_lr", 1e-5))

    # RL-paper comparison metric costs. These do not change training.
    fp_unit_cost = float(cfg.get("fp_unit_cost", 10.0))
    fn_unit_cost = float(cfg.get("fn_unit_cost", 20.0))
    delay_unit_cost = float(cfg.get("delay_unit_cost", 5.0))

    # Sequential-prefix evaluation settings. Evaluation only; training unchanged.
    sequential_min_prefix = int(cfg.get("sequential_min_prefix", 1))
    sequential_stride = int(cfg.get("sequential_stride", 1))

    # Preferred key for this normal-only version:
    #     target_normal_fraction: 0.20
    #
    # Backward-compatible fallback:
    # if the existing YAML still contains target_train_fraction: 0.20,
    # the numeric value is reused, but sampling remains NORMAL ONLY.
    target_normal_fraction = float(
        cfg.get(
            "target_normal_fraction",
            cfg.get("target_train_fraction", 0.20),
        )
    )

    sampling_seed = int(
        cfg.get(
            "target_sampling_seed",
            seed,
        )
    )

    reinitialize_classifier = bool(
        cfg.get(
            "reinitialize_classifier",
            True,
        )
    )

    result_dir = str(cfg.get("result_dir", "result"))
    checkpoint_dir = str(cfg.get("checkpoint_dir", "checkpoints"))

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ========================================================
    # Dynamic combined-source checkpoint name
    # Must match train_transformer.
    # ========================================================

    source_suffix = (
        f"{source_tag}_"
        f"{source_mode}_"
        f"{num_layers}_"
        f"{adapter_size}_"
        f"{source_lr}"
    )

    default_source_checkpoint = os.path.join(
        checkpoint_dir,
        f"train_{source_suffix}-best.pt",
    )

    source_checkpoint = str(
        cfg.get(
            "source_best_checkpoint",
            default_source_checkpoint,
        )
    )

    target_suffix = (
        f"{target_name}_"
        f"from_{source_tag}_"
        f"{tune_mode}_"
        f"normalonly_{target_normal_fraction}_"
        f"{num_layers}_"
        f"{adapter_size}_"
        f"{lr}_"
        f"{epochs}"
    )

    result_file = os.path.join(
        result_dir,
        f"tune_{target_suffix}.txt",
    )

    default_best_path = os.path.join(
        checkpoint_dir,
        f"tune_{target_suffix}-best.pt",
    )

    default_latest_path = os.path.join(
        checkpoint_dir,
        f"tune_{target_suffix}-latest.pt",
    )

    best_path = str(
        cfg.get(
            "target_best_checkpoint",
            default_best_path,
        )
    )

    latest_path = str(
        cfg.get(
            "target_latest_checkpoint",
            default_latest_path,
        )
    )

    Path(best_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    Path(latest_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Reproducibility
    # ========================================================

    warnings.filterwarnings("ignore")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # ========================================================
    # Load TARGET TRAIN + VALIDATION only.
    #
    # IMPORTANT:
    # The target TEST split is deliberately NOT loaded here.
    # It remains untouched until training is complete and the
    # best checkpoint has been selected using validation F1.
    # ========================================================

    x_train_full, y_train_full = load_split(
        preprocessed_dir,
        target_name,
        "training",
        window_size,
    )

    x_val, y_val = load_split(
        preprocessed_dir,
        target_name,
        "validation",
        window_size,
    )

    x_train, y_train = (
        select_normal_target_fraction(
            x_train_full,
            y_train_full,
            target_normal_fraction,
            sampling_seed,
        )
    )

    del x_train_full, y_train_full

    print("\n============================================================")
    print("NORMAL-ONLY CROSS-DATASET TARGET ADAPTATION")
    print("============================================================")
    print("Source datasets:", source_names)
    print("Source tag:", source_tag)
    print("Target:", target_name)
    print("Source checkpoint:", source_checkpoint)
    print(
        "Target normal fraction:",
        f"{target_normal_fraction * 100:.2f}%",
    )
    print(
        "Target adaptation classes: NORMAL ONLY"
    )
    print(
        "Target anomaly adaptation samples: 0"
    )
    print_distribution(
        "Target validation",
        y_val,
    )
    print(
        "Target test: NOT LOADED / NOT TOUCHED YET"
    )
    print("============================================================")

    # ========================================================
    # DataLoaders
    # ========================================================

    train_loader = torch.utils.data.DataLoader(
        DataGenerator(
            x_train,
            y_train,
            window_size,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        DataGenerator(
            x_val,
            y_val,
            window_size,
        ),
        batch_size=batch_size,
        shuffle=False,
    )


    # ========================================================
    # Build target adapter model
    # ========================================================

    model = Model(
        mode="adapter",
        num_layers=num_layers,
        adapter_size=adapter_size,
        dim=embedding_dim,
        window_size=window_size,
        nhead=nhead,
        dim_feedforward=(
            ff_multiplier * embedding_dim
        ),
        dropout=dropout,
    )

    # ========================================================
    # Load pretrained source checkpoint
    # ========================================================

    if not os.path.exists(source_checkpoint):
        raise FileNotFoundError(
            f"Source checkpoint not found: {source_checkpoint}\n"
            "Run Fraktion_normaandanomal_train_transformer_pkl_ready_last.py first."
        )

    source_ckpt = torch.load(
        source_checkpoint,
        map_location="cpu",
    )

    source_state = normalize_state_dict_keys(
        source_ckpt["net"]
    )

    if reinitialize_classifier:
        # Preserve the behavior of your earlier tuning code:
        # target classifier starts fresh.
        source_state.pop(
            "fc1.weight",
            None,
        )
        source_state.pop(
            "fc1.bias",
            None,
        )

        print(
            "Target classifier head is reinitialized."
        )

    load_result = model.load_state_dict(
        source_state,
        strict=False,
    )

    print("Source checkpoint load result:")
    print(load_result)

    # ========================================================
    # Choose trainable parameters
    # ========================================================

    if tune_mode == "adapter":
        model.train_adapter()

    elif tune_mode == "classifier":
        model.train_classifier()

    elif tune_mode == "tuning":
        for param in model.parameters():
            param.requires_grad = True

    else:
        raise ValueError(
            "tune_mode must be adapter, classifier, or tuning."
        )

    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print(
            "Using",
            torch.cuda.device_count(),
            "GPUs",
        )

        model = nn.DataParallel(model)

    trainable_params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    if not trainable_params:
        raise ValueError(
            "No trainable parameters. "
            "Check Model.train_adapter() / tune_mode."
        )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_count = sum(
        p.numel()
        for p in trainable_params
    )

    print(
        "Trainable parameters:",
        f"{trainable_count}/{total_params}",
        f"({100.0 * trainable_count / total_params:.2f}%)",
    )

    optimizer = optim.Adam(
        trainable_params,
        lr=lr,
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
    )

    criterion = nn.BCEWithLogitsLoss()

    # ========================================================
    # Target tuning
    # ========================================================

    best_val_f1 = -1.0
    total_start = time.time()

    with open(
        result_file,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "NORMAL-ONLY CROSS-DATASET LOGFORMER TUNING\n"
        )
        f.write(
            f"sources={source_names}\n"
        )
        f.write(
            f"source_tag={source_tag}\n"
        )
        f.write(
            f"target={target_name}\n"
        )
        f.write(
            f"target_normal_fraction={target_normal_fraction}\n"
        )
        f.write(
            "target_adaptation_classes=normal_only\n"
        )
        f.write(
            "target_anomaly_adaptation_samples=0\n"
        )
        f.write(
            f"window_size={window_size}\n"
        )
        f.write(
            f"tune_mode={tune_mode}\n"
        )
        f.write(
            f"epochs={epochs}\n"
        )
        f.write(
            f"lr={lr}\n"
        )
        f.write(
            f"selected_target_train={len(y_train)}\n"
        )
        f.write(
            f"source_checkpoint={source_checkpoint}\n\n"
        )

    for epoch in range(epochs):
        model.train()

        losses = []

        for x, y in tqdm(
            train_loader,
            desc=f"Target epoch {epoch + 1}/{epochs}",
        ):
            x = x.to(device).to(torch.float32)
            y = y.to(device).to(torch.float32)

            optimizer.zero_grad()

            out = model(x)

            loss = criterion(
                out,
                y,
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                trainable_params,
                0.5,
            )

            optimizer.step()
            scheduler.step()

            losses.append(loss.item())

        train_loss = float(
            np.mean(losses)
        )

        # Target validation controls checkpoint selection.
        val_result = evaluate(
            model,
            val_loader,
            device,
        )

        print(
            f"Epoch {epoch + 1}: "
            f"loss={train_loss:.6f}, "
            f"Val P={val_result['precision']:.4f}, "
            f"Val R={val_result['recall']:.4f}, "
            f"Val F1={val_result['f1']:.4f}"
        )

        with open(
            result_file,
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                f"Epoch {epoch + 1}: "
                f"loss={train_loss:.6f}, "
                f"val_precision={val_result['precision']:.6f}, "
                f"val_recall={val_result['recall']:.6f}, "
                f"val_f1={val_result['f1']:.6f}\n"
            )

        checkpoint = {
            "net": get_model_state(model),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_f1": val_result["f1"],
            "setting": setting,
            "source_datasets": source_names,
            "source_tag": source_tag,
            "target_dataset": target_name,
            "target_normal_fraction": target_normal_fraction,
            "target_adaptation_classes": "normal_only",
            "target_anomaly_adaptation_samples": 0,
            "window_size": window_size,
        }

        if val_result["f1"] > best_val_f1:
            best_val_f1 = val_result["f1"]

            torch.save(
                checkpoint,
                best_path,
            )

            print(
                "Saved best target checkpoint:",
                best_path,
            )

        torch.save(
            checkpoint,
            latest_path,
        )

    # ========================================================
    # Final target test ONCE
    #
    # The TEST split is loaded only now, after all target
    # adaptation epochs are complete and the best checkpoint
    # has already been selected using validation F1.
    # ========================================================

    best_checkpoint = torch.load(
        best_path,
        map_location=device,
    )

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(
            best_checkpoint["net"]
        )
    else:
        model.load_state_dict(
            best_checkpoint["net"]
        )

    # Load target TEST only for the final evaluation.
    x_test, y_test = load_split(
        preprocessed_dir,
        target_name,
        "testing",
        window_size,
    )

    test_loader = torch.utils.data.DataLoader(
        DataGenerator(
            x_test,
            y_test,
            window_size,
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    test_result = evaluate(
        model,
        test_loader,
        device,
    )

    test_seq_lengths = get_effective_sequence_lengths(
        x_test,
        window_size,
    )

    full_report = build_full_sequence_report(
        dataset_name=target_name,
        result=test_result,
        seq_lengths=test_seq_lengths,
        fp_unit_cost=fp_unit_cost,
        fn_unit_cost=fn_unit_cost,
        delay_unit_cost=delay_unit_cost,
    )

    sequential_result = evaluate_sequential_prefixes(
        model=model,
        x_sequences=x_test,
        y=y_test,
        device=device,
        window_size=window_size,
        embedding_dim=embedding_dim,
        min_prefix=sequential_min_prefix,
        stride=sequential_stride,
    )
    sequential_report = build_sequential_report(
        dataset_name=target_name,
        result=sequential_result,
        fp_unit_cost=fp_unit_cost,
        fn_unit_cost=fn_unit_cost,
        delay_unit_cost=delay_unit_cost,
    )

    comparison_table = build_standard_vs_sequential_table(
        dataset_name=target_name,
        standard_result=test_result,
        sequential_result=sequential_result,
        seq_lengths=test_seq_lengths,
        fp_unit_cost=fp_unit_cost,
        fn_unit_cost=fn_unit_cost,
        delay_unit_cost=delay_unit_cost,
    )

    total_time = time.time() - total_start

    print("\n============================================================")
    print("FINAL TARGET TEST")
    print("============================================================")
    print("Sources:", source_names)
    print("Target:", target_name)
    print(
        "Target adaptation:",
        f"{target_normal_fraction * 100:.2f}% NORMAL ONLY",
    )
    print(
        "Target anomaly samples used for adaptation: 0"
    )
    print(
        "Number of testing data:",
        len(y_test),
    )
    print(
        "Best target checkpoint:",
        best_path,
    )
    print("============================================================")
    print()
    print(comparison_table)

    comparison_metrics_file = os.path.join(
        result_dir,
        f"RL_Sequential_{target_name}_original_vs_sequential.txt",
    )
    with open(comparison_metrics_file, "w", encoding="utf-8") as rf:
        rf.write(comparison_table)
        rf.write("\n")
    print(f"Saved original-vs-sequential metrics to: {comparison_metrics_file}")

    with open(
        result_file,
        "a",
        encoding="utf-8",
    ) as f:
        f.write("\nFINAL TARGET TEST\n")
        f.write(
            f"number_testing_data={len(y_test)}\n"
        )
        f.write(
            f"precision={test_result['precision']}\n"
        )
        f.write(
            f"recall={test_result['recall']}\n"
        )
        f.write(
            f"f1={test_result['f1']}\n"
        )
        f.write("confusion_matrix=\n")
        f.write(
            f"{test_result['confusion_matrix']}\n"
        )
        f.write(
            f"prediction_time={test_result['prediction_time']}\n"
        )
        f.write(
            f"total_time={total_time}\n"
        )
        f.write(
            f"best_checkpoint={best_path}\n"
        )
        f.write("\n")
        f.write(comparison_table)
        f.write("\n")


if __name__ == "__main__":
    main()