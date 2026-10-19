#!/usr/bin/env python3
"""
Unified LogFormer training.

Supports BOTH:

1) IN-DOMAIN
   setting: in_domain
   in_domain_dataset_name: BGL

   Training:
       BGL train
   Model selection:
       BGL validation
   Final evaluation:
       BGL test

2) CROSS-DATASET
   setting: cross_dataset
   source_dataset_names:
     - BGL
     - HDFS
   target_dataset_name: TH_1G

   Training:
       combine ALL selected source training NPZs
   Model selection:
       macro mean validation F1 across selected source datasets
   Final source checks:
       each source test once

   IMPORTANT:
       target data are NOT used by this script.
       target adaptation is performed later by tune_transformer.
"""

import argparse
import os
import random
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

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


def get_in_domain_name(cfg) -> str:
    name = cfg.get("in_domain_dataset_name")

    if name is None:
        raise ValueError(
            "in_domain_dataset_name is required for in_domain mode."
        )

    return str(name)


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

    target = cfg.get("target_dataset_name")

    if target is not None and str(target) in names:
        raise ValueError(
            "Target dataset cannot also be a source dataset."
        )

    return names


def get_training_datasets(cfg) -> Tuple[str, List[str]]:
    setting = get_setting(cfg)

    if setting == "in_domain":
        return setting, [get_in_domain_name(cfg)]

    return setting, get_source_names(cfg)


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


def to_object_sequences(
    x,
    embedding_dim: int,
):
    """
    Convert fixed-length or variable-length x into an object array.

    Each item remains one complete sequence/block.
    """
    out = np.empty(len(x), dtype=object)

    for i in range(len(x)):
        seq = np.asarray(
            x[i],
            dtype=np.float32,
        )

        if seq.ndim != 2:
            raise ValueError(
                f"Invalid sequence shape at index {i}: {seq.shape}"
            )

        if seq.shape[1] != embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch at index {i}: "
                f"got {seq.shape[1]}, expected {embedding_dim}."
            )

        out[i] = seq

    return out


def print_distribution(
    dataset_name: str,
    split_name: str,
    y,
):
    labels = np.argmax(y, axis=1)

    print(
        f"{dataset_name} {split_name}: "
        f"total={len(y)}, "
        f"normal={int(np.sum(labels == 0))}, "
        f"anomaly={int(np.sum(labels == 1))}"
    )


# ============================================================
# Evaluation helpers
# ============================================================

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

    setting, training_dataset_names = get_training_datasets(cfg)

    dataset_tag = "_".join(training_dataset_names)

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

    batch_size = int(cfg.get("source_batch_size", 64))
    epochs = int(cfg.get("source_epochs", 5))
    lr = float(cfg.get("source_lr", 1e-5))

    # RL-paper comparison metric costs. These do not change training.
    fp_unit_cost = float(cfg.get("fp_unit_cost", 10.0))
    fn_unit_cost = float(cfg.get("fn_unit_cost", 20.0))
    delay_unit_cost = float(cfg.get("delay_unit_cost", 5.0))

    # Sequential-prefix evaluation settings. Evaluation only; training unchanged.
    sequential_min_prefix = int(cfg.get("sequential_min_prefix", 1))
    sequential_stride = int(cfg.get("sequential_stride", 1))

    result_dir = str(cfg.get("result_dir", "result"))
    checkpoint_dir = str(cfg.get("checkpoint_dir", "checkpoints"))

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    suffix = (
        f"{dataset_tag}_"
        f"{source_mode}_"
        f"{num_layers}_"
        f"{adapter_size}_"
        f"{lr}"
    )

    result_file = os.path.join(
        result_dir,
        f"train_{suffix}.txt",
    )

    default_best_path = os.path.join(
        checkpoint_dir,
        f"train_{suffix}-best.pt",
    )

    default_latest_path = os.path.join(
        checkpoint_dir,
        f"train_{suffix}-latest.pt",
    )

    best_path = str(
        cfg.get(
            "source_best_checkpoint",
            default_best_path,
        )
    )

    latest_path = str(
        cfg.get(
            "source_latest_checkpoint",
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
    # Load train / validation / test
    # ========================================================

    train_x_parts = []
    train_y_parts = []

    validation_sets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    testing_sets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    print("\n============================================================")
    print("UNIFIED LOGFORMER TRAINING")
    print("============================================================")
    print("Setting:", setting)

    if setting == "in_domain":
        print("In-domain dataset:", training_dataset_names[0])
    else:
        print("Source datasets:", training_dataset_names)
        print("Target data used here: NONE")

    print("Training tag:", dataset_tag)
    print("Window size:", window_size)
    print("============================================================")

    for dataset_name in training_dataset_names:
        x_train, y_train = load_split(
            preprocessed_dir,
            dataset_name,
            "training",
            window_size,
        )

        x_val, y_val = load_split(
            preprocessed_dir,
            dataset_name,
            "validation",
            window_size,
        )

        x_test, y_test = load_split(
            preprocessed_dir,
            dataset_name,
            "testing",
            window_size,
        )

        print_distribution(
            dataset_name,
            "train",
            y_train,
        )
        print_distribution(
            dataset_name,
            "validation",
            y_val,
        )
        print_distribution(
            dataset_name,
            "test",
            y_test,
        )

        train_x_parts.append(
            to_object_sequences(
                x_train,
                embedding_dim,
            )
        )

        train_y_parts.append(
            np.asarray(
                y_train,
                dtype=np.float32,
            )
        )

        validation_sets[dataset_name] = (
            to_object_sequences(
                x_val,
                embedding_dim,
            ),
            np.asarray(
                y_val,
                dtype=np.float32,
            ),
        )

        testing_sets[dataset_name] = (
            to_object_sequences(
                x_test,
                embedding_dim,
            ),
            np.asarray(
                y_test,
                dtype=np.float32,
            ),
        )

    # Works for one dataset or many.
    x_train_combined = np.concatenate(
        train_x_parts,
        axis=0,
    )

    y_train_combined = np.concatenate(
        train_y_parts,
        axis=0,
    )

    rng = np.random.default_rng(seed)

    permutation = rng.permutation(
        len(y_train_combined)
    )

    x_train_combined = (
        x_train_combined[permutation]
    )

    y_train_combined = (
        y_train_combined[permutation]
    )

    print_distribution(
        dataset_tag,
        "combined-train",
        y_train_combined,
    )

    # ========================================================
    # DataLoaders
    # ========================================================

    train_loader = torch.utils.data.DataLoader(
        DataGenerator(
            x_train_combined,
            y_train_combined,
            window_size,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    validation_loaders = {}

    for name, (x_val, y_val) in validation_sets.items():
        validation_loaders[name] = torch.utils.data.DataLoader(
            DataGenerator(
                x_val,
                y_val,
                window_size,
            ),
            batch_size=batch_size,
            shuffle=False,
        )

    testing_loaders = {}

    for name, (x_test, y_test) in testing_sets.items():
        testing_loaders[name] = torch.utils.data.DataLoader(
            DataGenerator(
                x_test,
                y_test,
                window_size,
            ),
            batch_size=batch_size,
            shuffle=False,
        )

    # ========================================================
    # Model
    # ========================================================

    model = Model(
        mode=source_mode,
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

    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print(
            "Using",
            torch.cuda.device_count(),
            "GPUs",
        )

        model = nn.DataParallel(model)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=0,
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
    )

    criterion = nn.BCEWithLogitsLoss()

    # ========================================================
    # Train
    # ========================================================

    best_macro_val_f1 = -1.0
    total_start = time.time()

    with open(
        result_file,
        "w",
        encoding="utf-8",
    ) as f:
        f.write("UNIFIED LOGFORMER TRAINING\n")
        f.write(f"setting={setting}\n")
        f.write(
            f"training_datasets={training_dataset_names}\n"
        )
        f.write(f"dataset_tag={dataset_tag}\n")
        f.write(f"window_size={window_size}\n")
        f.write(f"epochs={epochs}\n")
        f.write(f"lr={lr}\n")
        f.write(f"batch_size={batch_size}\n")
        f.write(
            f"combined_training_samples="
            f"{len(y_train_combined)}\n\n"
        )

    for epoch in range(epochs):
        model.train()

        losses = []

        for x, y in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
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
                model.parameters(),
                0.5,
            )

            optimizer.step()
            scheduler.step()

            losses.append(loss.item())

        train_loss = float(
            np.mean(losses)
        )

        # ----------------------------------------------------
        # Validate each selected source/domain independently.
        #
        # In-domain:
        #   this is simply that dataset's validation F1.
        #
        # Multi-source:
        #   macro mean prevents a very large source validation
        #   set from dominating checkpoint selection.
        # ----------------------------------------------------

        val_f1s = []
        val_results = {}

        print(
            f"\nEpoch {epoch + 1} validation:"
        )

        for dataset_name in training_dataset_names:
            result = evaluate(
                model,
                validation_loaders[dataset_name],
                device,
            )

            val_results[dataset_name] = result
            val_f1s.append(result["f1"])

            print(
                f"  {dataset_name}: "
                f"P={result['precision']:.4f}, "
                f"R={result['recall']:.4f}, "
                f"F1={result['f1']:.4f}"
            )

        macro_val_f1 = float(
            np.mean(val_f1s)
        )

        print(
            f"Epoch {epoch + 1}: "
            f"loss={train_loss:.6f}, "
            f"macro_val_f1={macro_val_f1:.4f}"
        )

        with open(
            result_file,
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                f"Epoch {epoch + 1}: "
                f"loss={train_loss:.6f}, "
                f"macro_val_f1={macro_val_f1:.6f}\n"
            )

            for dataset_name in training_dataset_names:
                r = val_results[dataset_name]

                f.write(
                    f"  {dataset_name}: "
                    f"P={r['precision']:.6f}, "
                    f"R={r['recall']:.6f}, "
                    f"F1={r['f1']:.6f}\n"
                )

        checkpoint = {
            "net": get_model_state(model),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "macro_val_f1": macro_val_f1,
            "setting": setting,
            "training_datasets": training_dataset_names,
            "dataset_tag": dataset_tag,
            "window_size": window_size,
        }

        if macro_val_f1 > best_macro_val_f1:
            best_macro_val_f1 = macro_val_f1

            torch.save(
                checkpoint,
                best_path,
            )

            print(
                "Saved best checkpoint:",
                best_path,
            )

        torch.save(
            checkpoint,
            latest_path,
        )

    # ========================================================
    # Final test(s) once
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

    print("\n============================================================")

    if setting == "in_domain":
        print("FINAL IN-DOMAIN TEST")
    else:
        print("FINAL SOURCE TESTS")

    print("============================================================")

    test_f1s = []

    with open(
        result_file,
        "a",
        encoding="utf-8",
    ) as f:
        if setting == "in_domain":
            f.write("\nFINAL IN-DOMAIN TEST\n")
        else:
            f.write("\nFINAL SOURCE TESTS\n")

        for dataset_name in training_dataset_names:
            result = evaluate(
                model,
                testing_loaders[dataset_name],
                device,
            )

            test_f1s.append(result["f1"])

            test_seq_lengths = get_effective_sequence_lengths(
                testing_sets[dataset_name][0],
                window_size,
            )
            full_report = build_full_sequence_report(
                dataset_name=dataset_name,
                result=result,
                seq_lengths=test_seq_lengths,
                fp_unit_cost=fp_unit_cost,
                fn_unit_cost=fn_unit_cost,
                delay_unit_cost=delay_unit_cost,
            )

            x_test_seq, y_test_seq = testing_sets[dataset_name]
            sequential_result = evaluate_sequential_prefixes(
                model=model,
                x_sequences=x_test_seq,
                y=y_test_seq,
                device=device,
                window_size=window_size,
                embedding_dim=embedding_dim,
                min_prefix=sequential_min_prefix,
                stride=sequential_stride,
            )
            sequential_report = build_sequential_report(
                dataset_name=dataset_name,
                result=sequential_result,
                fp_unit_cost=fp_unit_cost,
                fn_unit_cost=fn_unit_cost,
                delay_unit_cost=delay_unit_cost,
            )

            comparison_table = build_standard_vs_sequential_table(
                dataset_name=dataset_name,
                standard_result=result,
                sequential_result=sequential_result,
                seq_lengths=test_seq_lengths,
                fp_unit_cost=fp_unit_cost,
                fn_unit_cost=fn_unit_cost,
                delay_unit_cost=delay_unit_cost,
            )

            print()
            print(comparison_table)

            comparison_metrics_file = os.path.join(
                result_dir,
                f"RL_Sequential_{dataset_name}_original_vs_sequential.txt",
            )
            with open(comparison_metrics_file, "w", encoding="utf-8") as rf:
                rf.write(comparison_table)
                rf.write("\n")
            print(f"Saved original-vs-sequential metrics to: {comparison_metrics_file}")

            f.write("\n")
            f.write(comparison_table)
            f.write("\n")

        macro_test_f1 = float(
            np.mean(test_f1s)
        )

        total_time = time.time() - total_start

        if len(training_dataset_names) > 1:
            print(
                "Macro source test F1:",
                f"{macro_test_f1:.4f}",
            )

        print(
            "Best checkpoint:",
            best_path,
        )

        f.write(
            f"macro_test_f1={macro_test_f1}\n"
        )
        f.write(
            f"total_time={total_time}\n"
        )
        f.write(
            f"best_checkpoint={best_path}\n"
        )

    print("============================================================")


if __name__ == "__main__":
    main()
